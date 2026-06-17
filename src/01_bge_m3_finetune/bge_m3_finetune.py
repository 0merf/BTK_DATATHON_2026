"""
Mentör geri bildirim metinlerini (`mentor_feedback_text`) sayısal verilere entegre etmek amacıyla 
BAAI/bge-m3 modeli ile gerçekleştirilen fine-tuning işlemi. 
Bu model, çok dilli metin analizi ile mentör notlarından anlamsal (semantic) vektörler üretmekte 
ve Boyutluluğun Laneti'ni (Curse of Dimensionality) önlemek adına bu vektörleri 1 boyutlu bir tahmin skoruna 
(ft_bgem3) indirgeyerek ana tahminleyici modellere beslemektedir. Sızıntıyı (leakage) önlemek için 
5-Fold OOF kullanılmıştır.

  - gradient_checkpointing
  - batch=4, grad_accum=8 (effective=32)
  - fp16, max_len=96
  - Freeze embedding + first 6 encoder layers
"""
import warnings; warnings.filterwarnings("ignore")
import os, gc, json, time
import numpy as np, pandas as pd, torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
import datasets

# ─── Config ──────────────────────────────────────────────────
SEED          = 42
N_FOLDS       = 5
TARGET        = "career_success_score"
ID_COL        = "student_id"
TEXT_COL      = "mentor_feedback_text"
MODEL_NAME    = "BAAI/bge-m3"
MAX_LEN       = 96
BATCH_SIZE    = 4
GRAD_ACCUM    = 8        # effective batch = 32
EPOCHS        = 4
LR            = 2e-5
WARMUP_RATIO  = 0.1
FREEZE_LAYERS = 6        # freeze embedding + first N encoder layers

def clip(p): return np.clip(p, 0, 100)

print("=" * 60)
print("  BGE-M3 FINE-TUNING — Son Şans Stratejisi")
print("=" * 60)
print(f"GPU : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Card: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
print(f"Config: batch={BATCH_SIZE}, accum={GRAD_ACCUM}, max_len={MAX_LEN}, "
      f"epochs={EPOCHS}, freeze_layers={FREEZE_LAYERS}")
print()

# ─── Data ────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(os.path.join(BASE, "train.csv"))
test  = pd.read_csv(os.path.join(BASE, "test_x.csv"))
y     = train[TARGET].values.astype("float32")
y01   = (y / 100.0).astype("float32")           # scale to [0,1]
yr    = train["application_year"].values
p_te  = test["application_year"].value_counts(normalize=True)

def wmse(o):
    per = {Y: mean_squared_error(y[yr==Y], o[yr==Y]) for Y in np.unique(yr)}
    return sum(p_te.get(Y, 0) * per[Y] for Y in per)

txt_tr = train[TEXT_COL].fillna("").tolist()
txt_te = test[TEXT_COL].fillna("").tolist()

print("Tokenizer yükleniyor...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Vocab size: {tok.vocab_size}")

def tokenize(texts):
    return tok(texts, truncation=True, max_length=MAX_LEN, padding=False)

def make_ds(texts, labels=None):
    enc = tokenize(texts)
    d = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    if labels is not None:
        d["labels"] = list(map(float, labels))
    return datasets.Dataset.from_dict(d)

# ─── Fine-tune 5-Fold ───────────────────────────────────────
print("\n>>> Test dataset hazırlanıyor...")
ds_test = make_ds(txt_te)
ft_oof  = np.zeros(len(train))
ft_tp   = np.zeros(len(test))
kf      = KFold(N_FOLDS, shuffle=True, random_state=SEED)
t0      = time.time()

for fold, (a, b) in enumerate(kf.split(train), 1):
    fold_t0 = time.time()
    print(f"\n{'='*50}")
    print(f"  FOLD {fold}/{N_FOLDS}  (train={len(a)}, val={len(b)})")
    print(f"{'='*50}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, problem_type="regression"
    )

    # Gradient checkpointing
    model.gradient_checkpointing_enable()

    # Freeze embedding + first N encoder layers
    frozen, trainable_count = 0, 0
    for name, param in model.named_parameters():
        freeze = False
        if "embeddings" in name:
            freeze = True
        elif "encoder.layer." in name:
            try:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_num < FREEZE_LAYERS:
                    freeze = True
            except (ValueError, IndexError):
                pass
        if freeze:
            param.requires_grad = False
            frozen += param.numel()
        else:
            trainable_count += param.numel()

    total_p = frozen + trainable_count
    print(f"Params: {trainable_count/1e6:.1f}M trainable / {total_p/1e6:.1f}M total "
          f"({100*trainable_count/total_p:.0f}%)")

    out_dir = os.path.join(BASE, f"bge_ft_fold{fold}")
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=50,
        report_to=[],
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        seed=SEED,
        dataloader_num_workers=0,
    )

    tr_ds = make_ds([txt_tr[i] for i in a], y01[a])
    va_ds = make_ds([txt_tr[i] for i in b])

    trainer = Trainer(
        model=model, args=args, train_dataset=tr_ds,
        data_collator=DataCollatorWithPadding(tok)
    )

    print("Eğitim başlıyor...")
    trainer.train()

    # Predict
    print("Val + test tahmini...")
    val_pred = trainer.predict(va_ds).predictions.ravel() * 100.0
    ft_oof[b] = clip(val_pred)

    test_pred = trainer.predict(ds_test).predictions.ravel() * 100.0
    ft_tp += clip(test_pred) / N_FOLDS

    fold_time = time.time() - fold_t0
    print(f"Fold {fold} tamamlandı ({fold_time/60:.1f} dk)")
    print(f"  Val metin-tek MSE (bu fold): {mean_squared_error(y[b], ft_oof[b]):.3f}")

    # Cleanup
    del model, trainer, tr_ds, va_ds
    gc.collect()
    torch.cuda.empty_cache()

total_time = time.time() - t0
ft_wmse = wmse(ft_oof)
print(f"\n{'='*60}")
print(f"  FINE-TUNING TAMAMLANDI ({total_time/60:.1f} dk)")
print(f"{'='*60}")
print(f"BGE-M3  ft metin-tek wOOF = {ft_wmse:.3f}")
print(f"Önceki  dbmdz ft wOOF     = 156.269")
print(f"Frozen  bert wOOF         = ~169.6")

# Save signal
sig_path = os.path.join(BASE, "ft_bge_m3_signal.npz")
np.savez(sig_path, ft_oof=ft_oof, ft_tp=ft_tp)
print(f"Sinyal kaydedildi: {sig_path}")

# ─── ENSEMBLE ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ENSEMBLE — BGE-M3 ft sinyali ile")
print(f"{'='*60}")

import lightgbm as lgb_lib
from catboost import CatBoostRegressor

CATEGORICAL = ["department","university_tier","target_role","hobby","preferred_social_media_platform"]
TECH = ["coding_score","problem_solving_score","data_structures_score","sql_score",
        "machine_learning_score","backend_score","frontend_score","cloud_score","devops_score"]
SOFT = ["communication_score","teamwork_score","leadership_score","presentation_score"]
INTERVIEW = ["technical_interview_score","hr_interview_score"]
PROFILE = ["project_quality_score","portfolio_score","linkedin_profile_score","cv_quality_score"]

def safe_div(a, b):
    b = np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, np.asarray(a, float) / b, np.nan)

def add_fe(df):
    df = df.copy()
    df["tech_mean"]=df[TECH].mean(1); df["tech_max"]=df[TECH].max(1)
    df["tech_min"]=df[TECH].min(1);   df["tech_std"]=df[TECH].std(1)
    df["soft_mean"]=df[SOFT].mean(1);  df["soft_std"]=df[SOFT].std(1)
    df["interview_mean"]=df[INTERVIEW].mean(1)
    df["interview_gap"]=df["technical_interview_score"]-df["hr_interview_score"]
    df["profile_mean"]=df[PROFILE].mean(1)
    df["all_score_mean"]=df[TECH+SOFT+INTERVIEW+PROFILE].mean(1)
    df["interview_conv"]=safe_div(df["interviews_attended"], df["applications_sent"])
    df["hackathon_winrate"]=safe_div(df["hackathon_awards"], df["hackathon_count"])
    df["avg_internship_len"]=safe_div(df["internship_duration_months"], df["internship_count"])
    df["github_total_stars"]=df["github_avg_stars"]*df["github_repo_count"]
    df["total_real_projects"]=df["real_client_project_count"]+df["freelance_project_count"]
    df["total_activity"]=df["hackathon_count"]+df["bootcamp_count"]+df["certification_count"]+df["github_repo_count"]
    df["years_since_grad"]=df["application_year"]-df["graduation_year"]
    df["age_at_grad"]=df["age"]-df["years_since_grad"]
    df["cgpa_x_attendance"]=df["cgpa"]*df["attendance_rate"]
    df["cgpa_minus_failed"]=df["cgpa"]-0.1*df["failed_courses_count"]
    return df

def prepare(train_df, test_df):
    train_df, test_df = add_fe(train_df), add_fe(test_df)
    nc = [c for c in train_df.columns if c != TARGET and train_df[c].isna().any()
          and not c.endswith("_isna")]
    for c in nc:
        train_df[f"{c}_isna"] = train_df[c].isna().astype(int)
        test_df[f"{c}_isna"]  = test_df[c].isna().astype(int)
    tmap = {f"Tier {i}": i for i in range(1, 5)}
    train_df["university_tier_ord"] = train_df["university_tier"].map(tmap)
    test_df["university_tier_ord"]  = test_df["university_tier"].map(tmap)
    for c in CATEGORICAL:
        cats = pd.Index(pd.concat([train_df[c], test_df[c]]).dropna().unique())
        train_df[c] = pd.Categorical(train_df[c], categories=cats)
        test_df[c]  = pd.Categorical(test_df[c],  categories=cats)
    feats = [c for c in train_df.columns if c not in (ID_COL, TARGET, TEXT_COL)]
    return train_df, test_df, feats

LGB_PARAMS = dict(
    objective="regression", metric="l2", learning_rate=0.03, num_leaves=31,
    min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.5, n_estimators=5000, random_state=SEED,
    n_jobs=-1, verbose=-1
)

tr, te, feats = prepare(train.copy(), test.copy())

# Load text signals (char, word, bert = frozen BGE-M3 Ridge)
sig = np.load(os.path.join(BASE, "text_signals_stage2.npy"), allow_pickle=True).item()
z   = np.load(os.path.join(BASE, "ft_bge_m3_signal.npz"))   # NEW BGE-M3 ft

for k in ["char", "word", "bert"]:
    tr[k] = sig[k][0]
    te[k] = sig[k][1]
tr["ft"] = z["ft_oof"]
te["ft"] = z["ft_tp"]
feats = feats + ["char", "word", "bert", "ft"]

X, Xt = tr[feats], te[feats]
kf2 = KFold(N_FOLDS, shuffle=True, random_state=SEED)
cat_idx = [feats.index(c) for c in CATEGORICAL]
Xc, Xtc = X.copy(), Xt.copy()
for c in CATEGORICAL:
    Xc[c]  = Xc[c].astype(str).fillna("NA")
    Xtc[c] = Xtc[c].astype(str).fillna("NA")

ol = np.zeros(len(tr)); tl = np.zeros(len(te))
oc = np.zeros(len(tr)); tc = np.zeros(len(te))

for fold_i, (a, b) in enumerate(kf2.split(X), 1):
    print(f"\nEnsemble Fold {fold_i}...")

    # LightGBM
    m = lgb_lib.LGBMRegressor(**LGB_PARAMS)
    m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])],
          eval_metric="l2", categorical_feature=CATEGORICAL,
          callbacks=[lgb_lib.early_stopping(100, verbose=False),
                     lgb_lib.log_evaluation(0)])
    ol[b] = clip(m.predict(X.iloc[b]))
    tl   += clip(m.predict(Xt)) / N_FOLDS

    # CatBoost (CPU — küçük veride daha hızlı)
    cm = CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        loss_function="RMSE", od_type="Iter", od_wait=100,
        random_seed=SEED, verbose=0
    )
    cm.fit(Xc.iloc[a], y[a], eval_set=(Xc.iloc[b], y[b]),
           cat_features=cat_idx, use_best_model=True)
    oc[b] = clip(cm.predict(Xc.iloc[b]))
    tc   += clip(cm.predict(Xtc)) / N_FOLDS

print(f"\nLGBM wOOF = {wmse(ol):.3f}")
print(f"CAT  wOOF = {wmse(oc):.3f}")

print("\n--- Blend Sonuçları ---")
best_score, best_wl = 999, 0.4
for wl in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]:
    score = wmse(wl * ol + (1 - wl) * oc)
    marker = ""
    if score < best_score:
        best_score = score
        best_wl = wl
        marker = " ← EN İYİ"
    print(f"  lgb={wl:.2f} / cat={1-wl:.2f}:  wOOF = {score:.3f}"
          f"   (önceki en iyi: 84.917){marker}")

# Write submission with best blend
tp = best_wl * tl + (1 - best_wl) * tc
sub_path = os.path.join(BASE, "submission_bge_m3_ft.csv")
pd.DataFrame({ID_COL: te[ID_COL], TARGET: tp}).to_csv(sub_path, index=False)
print(f"\nSubmission yazıldı: {sub_path}")

# ─── Final Summary ──────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SONUÇ KARŞILAŞTIRMASI")
print(f"{'='*60}")
print(f"  BGE-M3 ft  metin-tek wOOF : {ft_wmse:.3f}")
print(f"  dbmdz  ft  metin-tek wOOF : 156.269")
print(f"  frozen bert metin-tek wOOF: ~169.6")
print(f"")
print(f"  BGE-M3 ft  ensemble wOOF  : {best_score:.3f}  (blend {best_wl}/{1-best_wl})")
print(f"  Önceki en iyi ensemble    : 84.917  (0.4/0.6)")
print(f"  Önceki en iyi public      : 84.00")
print(f"")
if best_score < 84.917:
    print(f"  ✅ İYİLEŞME: {84.917 - best_score:.3f} puan kazanç!")
    print(f"  → submission_bge_m3_ft.csv'yi Kaggle'a submit et!")
else:
    print(f"  ❌ İyileşme yok ({best_score:.3f} vs 84.917)")
    print(f"  → Yine de public'te farklı çıkabilir, denemeye değer.")

print(f"\nToplam süre: {(time.time()-t0)/60:.1f} dakika")
print("BİTTİ! 🏁")
