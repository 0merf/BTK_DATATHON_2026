"""
CatBoost Native Text parametresi ve Huber Loss kullanılarak geliştirilmiş deney modeli.
Gürültüyü minimize etmek için 140 üretilmiş özellik yerine doğrudan korelasyonu en yüksek 
42 "Altın Sinyal" özelliğe odaklanılmıştır. Metin verileri dışarıdan vektör haline getirilmek yerine 
doğrudan CatBoost'un dahili metin işleme motoruna beslenmiştir.
Aykırı değerlerin (outlier) modele etkisini sınırlamak amacıyla asimetrik dirence sahip 
Huber Loss fonksiyonu test edilmiştir.
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy import stats
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import json

SEED = 42
N_FOLDS = 5
TARGET = "career_success_score"
ID_COL = "student_id"
TEXT_COL = "mentor_feedback_text"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Loading Data...")
train_raw = pd.read_csv(os.path.join(BASE, "train.csv"))
test_raw  = pd.read_csv(os.path.join(BASE, "test_x.csv"))

# Iterative Pseudo-Labeling (Source: mega_upgrade 82.656)
sub_old = pd.read_csv(os.path.join(BASE, "submission_finetuned.csv"))
sub_new = pd.read_csv(os.path.join(BASE, "best_model_mega_upgrade", "submission_mega_upgrade.csv"))
diff = np.abs(sub_new[TARGET] - sub_old[TARGET])
THRESHOLD = 2.0
confident_idx = np.where(diff <= THRESHOLD)[0]
y_pseudo_all = sub_new[TARGET].values.astype("float32")
print(f"Iterative PL: Using mega_upgrade | Confident samples: {len(confident_idx)}")

y_full = train_raw[TARGET].values.astype("float32")
yr     = train_raw["application_year"].values
p_te   = test_raw["application_year"].value_counts(normalize=True)

def wmse(o):
    per = {Y: mean_squared_error(y_full[yr==Y], o[yr==Y]) for Y in np.unique(yr)}
    return sum(p_te.get(Y, 0) * per[Y] for Y in per)
def clip(p): return np.clip(p, 0, 100)

# Temporal Reweighting
tr_year_counts = train_raw["application_year"].value_counts(normalize=True)
te_year_counts = test_raw["application_year"].value_counts(normalize=True)
year_weights = {}
for yr_val in tr_year_counts.index:
    year_weights[yr_val] = te_year_counts.get(yr_val, 0.01) / tr_year_counts[yr_val]
mean_w = np.mean(list(year_weights.values()))
year_weights = {k: v / mean_w for k, v in year_weights.items()}
temporal_w = np.array([year_weights[y] for y in train_raw["application_year"].values], dtype="float32")

# Golden Feature Engineering (No bloat, only highest purity signals)
CATEGORICAL = ["department","university_tier","target_role"]
TECH = ["coding_score","problem_solving_score","data_structures_score","sql_score",
        "machine_learning_score","backend_score","frontend_score","cloud_score","devops_score"]
SOFT = ["communication_score","teamwork_score","leadership_score","presentation_score"]
PROFILE = ["project_quality_score","portfolio_score","linkedin_profile_score","cv_quality_score"]

def safe_div(a, b):
    b = np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"): return np.where(b > 0, np.asarray(a, float) / b, np.nan)

def add_fe(df):
    df = df.copy()
    df["tech_mean"]=df[TECH].mean(1)
    df["soft_mean"]=df[SOFT].mean(1)
    df["profile_mean"]=df[PROFILE].mean(1)
    df["all_score_mean"]=df[TECH+SOFT+["technical_interview_score","hr_interview_score"]+PROFILE].mean(1)
    df["interview_conv"]=safe_div(df["interviews_attended"], df["applications_sent"])
    df["total_activity"]=df["hackathon_count"]+df["bootcamp_count"]+df["certification_count"]+df["github_repo_count"]
    df["years_since_grad"]=df["application_year"]-df["graduation_year"]
    df["proj_x_tech"] = df["project_quality_score"] * df["tech_mean"]
    df["cgpa_x_tech"] = df["cgpa"] * df["tech_mean"]
    return df

def target_encode_oof(train_df, test_df, col, target_col, n_folds=5, smoothing=20, seed=42):
    global_mean = train_df[target_col].mean()
    train_encoded = np.full(len(train_df), np.nan)
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[tr_idx]
        s = fold_train.groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (s["count"] * s["mean"] + smoothing * global_mean) / (s["count"] + smoothing)
        train_encoded[val_idx] = train_df.iloc[val_idx][col].map(smoothed).values
    s_all = train_df.groupby(col)[target_col].agg(["mean", "count"])
    smoothed_all = (s_all["count"] * s_all["mean"] + smoothing * global_mean) / (s_all["count"] + smoothing)
    test_encoded = test_df[col].map(smoothed_all).fillna(global_mean).values
    train_encoded = np.where(np.isnan(train_encoded), global_mean, train_encoded)
    return train_encoded, test_encoded

tr = add_fe(train_raw)
te = add_fe(test_raw)

# Target Encode Only the Most Important Categories
for col in CATEGORICAL + ["hobby"]:
    tr[f"te_{col}"], te[f"te_{col}"] = target_encode_oof(tr, te, col, TARGET)

# NLP Extra Signals
sig = np.load(os.path.join(BASE, "text_signals_stage2.npy"), allow_pickle=True).item()
z_bgem3 = np.load(os.path.join(BASE, "ft_bge_m3_signal.npz"))
tr["ft_char"] = sig["char"][0]; te["ft_char"] = sig["char"][1]
tr["ft_bgem3"] = z_bgem3["ft_oof"]; te["ft_bgem3"] = z_bgem3["ft_tp"]

# Fill NA
for c in tr.columns:
    if c not in [TARGET, TEXT_COL] and tr[c].dtype in [np.float64, np.float32, np.int64, np.int32]:
        tr[c] = tr[c].fillna(tr[c].median())
        te[c] = te[c].fillna(tr[c].median())

# Select final features
drop_cols = [ID_COL, TARGET, "hobby", "preferred_social_media_platform", "university_tier"]
feats = [c for c in tr.columns if c not in drop_cols]
print(f"Golden Features (Clean Signal): {len(feats)}")

# CatBoost expects string for categoricals
for c in CATEGORICAL: 
    if c in feats:
        tr[c] = tr[c].astype(str); te[c] = te[c].astype(str)

# XGBoost expects numeric
Xxgb = tr[feats].copy(); Xtxgb = te[feats].copy()
for c in CATEGORICAL:
    if c in feats:
        combined = pd.concat([Xxgb[c], Xtxgb[c]]).astype(str)
        mapping = {v: i for i, v in enumerate(combined.unique())}
        Xxgb[c] = Xxgb[c].map(mapping).astype(int); Xtxgb[c] = Xtxgb[c].map(mapping).astype(int)
if TEXT_COL in Xxgb.columns:
    Xxgb = Xxgb.drop(columns=[TEXT_COL])
    Xtxgb = Xtxgb.drop(columns=[TEXT_COL])

# Params
CAT_PARAMS = {
    "iterations": 5000, "learning_rate": 0.03, "depth": 6,
    "loss_function": "Huber:delta=1.5",  # The Grandmaster Secret: Robust to outliers
    "od_type": "Iter", "od_wait": 100, "verbose": 0, "text_features": [TEXT_COL],
    "task_type": "GPU"
}
XGB_PARAMS = {
    "learning_rate": 0.03, "max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.7, 
    "reg_alpha": 1.0, "reg_lambda": 3.0, "n_estimators": 5000, "tree_method": "hist", "device": "cuda", "verbosity": 0,
    "objective": "reg:pseudohubererror"
}

SEEDS = [42, 123, 2024, 7, 999]
final_cat = np.zeros(len(te))
final_xgb = np.zeros(len(te))
oof_cat_all = np.zeros(len(tr))
oof_xgb_all = np.zeros(len(tr))

cat_idx = [feats.index(c) for c in CATEGORICAL if c in feats]
text_idx = feats.index(TEXT_COL)

print("\nTraining The Native Text Fusion Model...")
PSEUDO_WEIGHT = 0.35

for seed in SEEDS:
    print(f"\n--- SEED {seed} ---")
    kf = KFold(N_FOLDS, shuffle=True, random_state=seed)
    
    oof_cat = np.zeros(len(tr)); test_cat = np.zeros(len(te))
    oof_xgb = np.zeros(len(tr)); test_xgb = np.zeros(len(te))
    
    for fold, (a, b) in enumerate(kf.split(tr), 1):
        # Pseudo Labels Prep
        X_test_conf = te[feats].iloc[confident_idx]
        Xxgb_test_conf = Xtxgb.iloc[confident_idx]
        y_test_conf = y_pseudo_all[confident_idx]
        
        y_comb = np.concatenate([y_full[a], y_test_conf])
        w_comb = np.concatenate([temporal_w[a], np.full(len(y_test_conf), PSEUDO_WEIGHT, dtype="float32")])
        
        # 1. CatBoost with NATIVE TEXT
        X_comb = pd.concat([tr[feats].iloc[a], X_test_conf], ignore_index=True)
        cm = CatBoostRegressor(**{**CAT_PARAMS, "random_seed": seed})
        cm.fit(X_comb, y_comb, cat_features=cat_idx, sample_weight=w_comb,
               eval_set=(tr[feats].iloc[b], y_full[b]), use_best_model=True)
        oof_cat[b] = clip(cm.predict(tr[feats].iloc[b]))
        test_cat += clip(cm.predict(te[feats])) / N_FOLDS
        
        # 2. XGBoost with Robust Loss (No Raw Text)
        Xxgb_comb = pd.concat([Xxgb.iloc[a], Xxgb_test_conf], ignore_index=True)
        xm = xgb.XGBRegressor(**{**XGB_PARAMS, "random_state": seed}, early_stopping_rounds=100)
        xm.fit(Xxgb_comb, y_comb, sample_weight=w_comb, eval_set=[(Xxgb.iloc[b], y_full[b])], verbose=False)
        oof_xgb[b] = clip(xm.predict(Xxgb.iloc[b]))
        test_xgb += clip(xm.predict(Xtxgb)) / N_FOLDS
        
        print(f"Fold {fold} OK", end=" | ")
        
    print(f"\nSeed {seed} OOF: CAT={wmse(oof_cat):.4f} XGB={wmse(oof_xgb):.4f}")
    final_cat += test_cat / len(SEEDS)
    final_xgb += test_xgb / len(SEEDS)
    oof_cat_all += oof_cat / len(SEEDS)
    oof_xgb_all += oof_xgb / len(SEEDS)

# ===== Grandmaster Scipy Optimize Blending =====
print("\nOptimizing Ensemble Weights on OOF...")
def blend_objective(weights):
    w_cat, w_xgb = weights
    pred = w_cat * oof_cat_all + w_xgb * oof_xgb_all
    return wmse(clip(pred))

res = minimize(blend_objective, [0.5, 0.5], bounds=[(0,1), (0,1)], constraints={'type':'eq','fun':lambda w: sum(w)-1})
best_w = res.x
print(f"Optimal Weights: CatBoost (Text) = {best_w[0]:.3f}, XGBoost = {best_w[1]:.3f}")

final_oof = clip(best_w[0] * oof_cat_all + best_w[1] * oof_xgb_all)
final_test = clip(best_w[0] * final_cat + best_w[1] * final_xgb)
print(f"FINAL BLEND wOOF: {wmse(final_oof):.4f}")

sub_path = os.path.join(BASE, "best_model_grandmaster", "submission_grandmaster.csv")
pd.DataFrame({ID_COL: te[ID_COL], TARGET: final_test}).to_csv(sub_path, index=False)

print(f"\n{'='*60}")
print(f"GRANDMASTER SUBMISSION SAVED: {sub_path}")
print(f"{'='*60}")
