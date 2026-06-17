"""
Kaggle Public Leaderboard'da en iyi performansı (82.656) gösteren ana Stacking mimarisi.
Bu kod bloğunda LightGBM, CatBoost, XGBoost ve PyTorch MLP (Yapay Sinir Ağı) algoritmaları, 
Level-2 Stacking yapısında Scipy Optimize kullanılarak optimum ağırlıklarla birleştirilmiştir.
Zaman kaymasını (Temporal Drift) modellemek için Covariate Shift optimizasyonu uygulanmış, 
eğitim verisi test yıllarına göre (2024-2026) yeniden ağırlıklandırılmıştır.
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import lightgbm as lgb_lib
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SEED = 42
N_FOLDS = 5
TARGET = "career_success_score"
ID_COL = "student_id"
TEXT_COL = "mentor_feedback_text"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_raw = pd.read_csv(os.path.join(BASE, "train.csv"))
test_raw  = pd.read_csv(os.path.join(BASE, "test_x.csv"))

# Smart PL: confident pseudo labels
sub_old = pd.read_csv(os.path.join(BASE, "submission_finetuned.csv"))
sub_new = pd.read_csv(os.path.join(BASE, "best_model_stacking", "submission_stacking.csv"))
diff = np.abs(sub_new[TARGET] - sub_old[TARGET])
THRESHOLD = 2.0
confident_idx = np.where(diff <= THRESHOLD)[0]
y_pseudo_all = sub_new[TARGET].values.astype("float32")
print(f"Confident PL samples: {len(confident_idx)}/{len(test_raw)}")

y_full = train_raw[TARGET].values.astype("float32")
yr     = train_raw["application_year"].values
p_te   = test_raw["application_year"].value_counts(normalize=True)

def wmse(o):
    per = {Y: mean_squared_error(y_full[yr==Y], o[yr==Y]) for Y in np.unique(yr)}
    return sum(p_te.get(Y, 0) * per[Y] for Y in per)
def clip(p): return np.clip(p, 0, 100)

# ───────── Temporal Reweighting ─────────
# Weight training samples to match test year distribution
tr_year_counts = train_raw["application_year"].value_counts(normalize=True)
te_year_counts = test_raw["application_year"].value_counts(normalize=True)
year_weights = {}
for yr_val in tr_year_counts.index:
    year_weights[yr_val] = te_year_counts.get(yr_val, 0.01) / tr_year_counts[yr_val]
# Normalize so mean weight = 1
mean_w = np.mean(list(year_weights.values()))
year_weights = {k: v / mean_w for k, v in year_weights.items()}
temporal_w = np.array([year_weights[y] for y in train_raw["application_year"].values], dtype="float32")
print("Temporal weights by year:")
for y_val in sorted(year_weights): print(f"  {y_val}: {year_weights[y_val]:.3f}")

# ───────── Feature Engineering ─────────
CATEGORICAL = ["department","university_tier","target_role","hobby","preferred_social_media_platform"]
TECH = ["coding_score","problem_solving_score","data_structures_score","sql_score",
        "machine_learning_score","backend_score","frontend_score","cloud_score","devops_score"]
SOFT = ["communication_score","teamwork_score","leadership_score","presentation_score"]
INTERVIEW = ["technical_interview_score","hr_interview_score"]
PROFILE = ["project_quality_score","portfolio_score","linkedin_profile_score","cv_quality_score"]

def safe_div(a, b):
    b = np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"): return np.where(b > 0, np.asarray(a, float) / b, np.nan)

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
    # NEW: Interaction features
    df["proj_x_tech"] = df["project_quality_score"] * df["tech_mean"]
    df["interview_x_profile"] = df["interview_mean"] * df["profile_mean"]
    df["real_proj_x_coding"] = df["real_client_project_count"] * df["coding_score"]
    df["github_x_opensource"] = df["github_repo_count"] * df["open_source_contribution_count"]
    df["soft_x_interview"] = df["soft_mean"] * df["interview_mean"]
    df["cgpa_x_tech"] = df["cgpa"] * df["tech_mean"]
    return df

# ───────── Target Encoding (OOF-safe) ─────────
def target_encode_oof(train_df, test_df, col, target_col, n_folds=5, smoothing=20, seed=42):
    """Out-of-fold target encoding to prevent leakage."""
    global_mean = train_df[target_col].mean()
    train_encoded = np.full(len(train_df), np.nan)
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[tr_idx]
        stats = fold_train.groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        train_encoded[val_idx] = train_df.iloc[val_idx][col].map(smoothed).values
    # For test: use all train data
    stats_all = train_df.groupby(col)[target_col].agg(["mean", "count"])
    smoothed_all = (stats_all["count"] * stats_all["mean"] + smoothing * global_mean) / (stats_all["count"] + smoothing)
    test_encoded = test_df[col].map(smoothed_all).fillna(global_mean).values
    # Fill any remaining NaN in train with global mean
    train_encoded = np.where(np.isnan(train_encoded), global_mean, train_encoded)
    return train_encoded, test_encoded

# ───────── GroupBy Aggregation Features ─────────
def add_groupby_features(train_df, test_df, cat_cols, num_cols, aggs=["mean", "std"]):
    """Add relative position features: value - group_mean."""
    new_feats = []
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    for cat in cat_cols:
        for num in num_cols:
            for agg in aggs:
                feat_name = f"{num}__{cat}__{agg}"
                group_stat = combined.groupby(cat)[num].transform(agg)
                combined[feat_name] = group_stat
                if agg == "mean":
                    diff_name = f"{num}__{cat}__diff"
                    combined[diff_name] = combined[num] - combined[feat_name]
                    new_feats.append(diff_name)
                new_feats.append(feat_name)
    train_out = combined.iloc[:len(train_df)].copy()
    test_out = combined.iloc[len(train_df):].copy().reset_index(drop=True)
    return train_out, test_out, new_feats

# ───────── Prepare Data ─────────
print("Feature Engineering...")
tr = add_fe(train_raw.copy())
te = add_fe(test_raw.copy())

# Target Encoding
TE_COLS = ["department", "target_role", "university_tier", "hobby", "preferred_social_media_platform"]
te_features = []
for col in TE_COLS:
    feat_name = f"te_{col}"
    tr[feat_name], te[feat_name] = target_encode_oof(tr, te, col, TARGET, smoothing=20)
    te_features.append(feat_name)
    print(f"  Target Encoded: {col}")

# GroupBy Aggregations
GROUPBY_CATS = ["department", "target_role", "university_tier"]
GROUPBY_NUMS = ["project_quality_score", "coding_score", "tech_mean", "communication_score", "all_score_mean"]
tr, te, groupby_feats = add_groupby_features(tr, te, GROUPBY_CATS, GROUPBY_NUMS)
print(f"  GroupBy features added: {len(groupby_feats)}")

# Missing indicators
nc = [c for c in tr.columns if c != TARGET and tr[c].isna().any() and not c.endswith("_isna")]
for c in nc:
    tr[f"{c}_isna"] = tr[c].isna().astype(int)
    te[f"{c}_isna"] = te[c].isna().astype(int)
tmap = {f"Tier {i}": i for i in range(1, 5)}
tr["university_tier_ord"] = tr["university_tier"].map(tmap)
te["university_tier_ord"] = te["university_tier"].map(tmap)

for c in CATEGORICAL:
    cats = pd.Index(pd.concat([tr[c], te[c]]).dropna().unique())
    tr[c] = pd.Categorical(tr[c], categories=cats)
    te[c] = pd.Categorical(te[c],  categories=cats)

# NLP signals
sig = np.load(os.path.join(BASE, "text_signals_stage2.npy"), allow_pickle=True).item()
z_bert  = np.load(os.path.join(BASE, "ft_transformer_signal.npz"))
z_bgem3 = np.load(os.path.join(BASE, "ft_bge_m3_signal.npz"))
for k in ["char", "word", "bert"]:
    tr[k] = sig[k][0]; te[k] = sig[k][1]
tr["ft_bert"] = z_bert["ft_oof"]; te["ft_bert"] = z_bert["ft_tp"]
tr["ft_bgem3"] = z_bgem3["ft_oof"]; te["ft_bgem3"] = z_bgem3["ft_tp"]

drop_cols = [ID_COL, TARGET, TEXT_COL]
feats = [c for c in tr.columns if c not in drop_cols]
num_feats = [c for c in feats if c not in CATEGORICAL]
print(f"Total features: {len(feats)}")

# ───────── Load best params ─────────
with open(os.path.join(BASE, "best_model_optuna", "best_params.json"), "r") as f:
    opt_params = json.load(f)
with open(os.path.join(BASE, "best_model_mlp", "best_params_mlp.json"), "r") as f:
    mlp_params = json.load(f)

LGB_BASE = opt_params["lgbm_params"].copy()
LGB_BASE.update({"objective": "regression", "metric": "l2", "n_estimators": 5000,
                  "n_jobs": -1, "verbose": -1, "subsample_freq": 1})
CAT_BASE = opt_params["catboost_params"].copy()
CAT_BASE.update({"iterations": 5000, "loss_function": "RMSE", "od_type": "Iter",
                  "od_wait": 100, "verbose": 0})
XGB_PARAMS = {"learning_rate": 0.03, "max_depth": 6, "subsample": 0.7,
              "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 3.0,
              "n_estimators": 5000, "tree_method": "hist", "random_state": SEED,
              "verbosity": 0, "early_stopping_rounds": 100}

# ───────── MLP Setup ─────────
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocessor = ColumnTransformer([("num", num_pipeline, num_feats), ("cat", cat_pipeline, CATEGORICAL)])
X_tr_np = preprocessor.fit_transform(tr[feats]).astype("float32")
X_te_np = preprocessor.transform(te[feats]).astype("float32")
INPUT_DIM = X_tr_np.shape[1]

class MLP(nn.Module):
    def __init__(self, h1, h2, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h2, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def train_nn(model, X_t, y_t, w_t, X_v, y_v, lr, wd, epochs=25):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss(reduction='none')
    ds_t = TensorDataset(torch.tensor(X_t), torch.tensor(y_t), torch.tensor(w_t))
    ds_v = TensorDataset(torch.tensor(X_v), torch.tensor(y_v))
    dl_t = DataLoader(ds_t, batch_size=256, shuffle=True)
    dl_v = DataLoader(ds_v, batch_size=256, shuffle=False)
    best_vl, best_p = float('inf'), None
    for ep in range(epochs):
        model.train()
        for bx, by, bw in dl_t:
            bx, by, bw = bx.to(device), by.to(device), bw.to(device)
            opt.zero_grad(); p = model(bx); loss = (crit(p, by) * bw).mean(); loss.backward(); opt.step()
        model.eval(); vp = []
        with torch.no_grad():
            for bx, by in dl_v: vp.append(model(bx.to(device)).cpu().numpy())
        vp = np.concatenate(vp); vl = mean_squared_error(y_v, vp)
        if vl < best_vl: best_vl = vl; best_p = vp.copy()
    return best_p

def predict_nn(model, X_np):
    model.eval()
    dl = DataLoader(TensorDataset(torch.tensor(X_np)), batch_size=256, shuffle=False)
    preds = []
    with torch.no_grad():
        for bx in dl: preds.append(model(bx[0].to(device)).cpu().numpy())
    return np.concatenate(preds)

# ───────── Multi-Seed Stacking ─────────
SEEDS = [42, 123, 2024, 7, 999]
X, Xt = tr[feats], te[feats]
cat_idx = [feats.index(c) for c in CATEGORICAL]
Xc, Xtc = X.copy(), Xt.copy()
for c in CATEGORICAL:
    Xc[c] = Xc[c].astype(str).fillna("NA")
    Xtc[c] = Xtc[c].astype(str).fillna("NA")

# For XGBoost: encode categoricals as integers
Xxgb = X.copy(); Xtxgb = Xt.copy()
for c in CATEGORICAL:
    combined_cats = pd.concat([Xxgb[c], Xtxgb[c]]).astype(str).fillna("NA")
    cat_map = {v: i for i, v in enumerate(combined_cats.unique())}
    Xxgb[c] = Xxgb[c].astype(str).fillna("NA").map(cat_map).astype(int)
    Xtxgb[c] = Xtxgb[c].astype(str).fillna("NA").map(cat_map).astype(int)

PSEUDO_WEIGHT = 0.35
final_test_preds = np.zeros(len(te))

for seed_i, CURRENT_SEED in enumerate(SEEDS):
    print(f"\n{'='*60}")
    print(f"SEED {seed_i+1}/{len(SEEDS)}: {CURRENT_SEED}")
    print(f"{'='*60}")
    
    kf = KFold(N_FOLDS, shuffle=True, random_state=CURRENT_SEED)
    splits = list(kf.split(X))
    
    oof_lgb = np.zeros(len(tr)); test_lgb = np.zeros(len(te))
    oof_cat = np.zeros(len(tr)); test_cat = np.zeros(len(te))
    oof_xgb = np.zeros(len(tr)); test_xgb = np.zeros(len(te))
    oof_mlp = np.zeros(len(tr)); test_mlp = np.zeros(len(te))
    
    for fold_i, (a, b) in enumerate(splits, 1):
        print(f"  Fold {fold_i}...", end=" ", flush=True)
        
        # Confident pseudo-labeled test data
        X_test_conf = Xt.iloc[confident_idx]
        y_test_conf = y_pseudo_all[confident_idx]
        
        # Temporal weights for train fold + pseudo weights
        w_fold = temporal_w[a]
        w_pseudo = np.full(len(y_test_conf), PSEUDO_WEIGHT, dtype="float32")
        
        # ── LightGBM ──
        lgb_p = {**LGB_BASE, "random_state": CURRENT_SEED}
        X_comb = pd.concat([X.iloc[a], X_test_conf], ignore_index=True)
        y_comb = np.concatenate([y_full[a], y_test_conf])
        w_comb = np.concatenate([w_fold, w_pseudo])
        m = lgb_lib.LGBMRegressor(**lgb_p)
        m.fit(X_comb, y_comb, sample_weight=w_comb, eval_set=[(X.iloc[b], y_full[b])],
              categorical_feature=CATEGORICAL, callbacks=[lgb_lib.early_stopping(100, verbose=False)])
        oof_lgb[b] = clip(m.predict(X.iloc[b]))
        test_lgb += clip(m.predict(Xt)) / N_FOLDS
        
        # ── CatBoost ──
        cat_p = {**CAT_BASE, "random_seed": CURRENT_SEED}
        Xc_conf = Xtc.iloc[confident_idx]
        Xc_comb = pd.concat([Xc.iloc[a], Xc_conf], ignore_index=True)
        tp = Pool(Xc_comb, y_comb, cat_features=cat_idx, weight=w_comb)
        vp = Pool(Xc.iloc[b], y_full[b], cat_features=cat_idx)
        cm = CatBoostRegressor(**cat_p)
        cm.fit(tp, eval_set=vp, use_best_model=True)
        oof_cat[b] = clip(cm.predict(Xc.iloc[b]))
        test_cat += clip(cm.predict(Xtc)) / N_FOLDS
        
        # ── XGBoost ──
        Xxgb_conf = Xtxgb.iloc[confident_idx]
        Xxgb_comb = pd.concat([Xxgb.iloc[a], Xxgb_conf], ignore_index=True)
        xgb_p = {**XGB_PARAMS, "random_state": CURRENT_SEED}
        es_rounds = xgb_p.pop("early_stopping_rounds")
        xm = xgb.XGBRegressor(**xgb_p)
        xm.fit(Xxgb_comb, y_comb, sample_weight=w_comb,
               eval_set=[(Xxgb.iloc[b], y_full[b])], verbose=False)
        oof_xgb[b] = clip(xm.predict(Xxgb.iloc[b]))
        test_xgb += clip(xm.predict(Xtxgb)) / N_FOLDS
        
        # ── MLP ──
        X_np_conf = X_te_np[confident_idx]
        X_np_comb = np.concatenate([X_tr_np[a], X_np_conf], axis=0)
        w_np_comb = np.concatenate([w_fold, w_pseudo])
        torch.manual_seed(CURRENT_SEED)
        model = MLP(mlp_params["hidden_dim1"], mlp_params["hidden_dim2"], mlp_params["dropout"])
        preds = train_nn(model, X_np_comb, y_comb, w_np_comb, X_tr_np[b], y_full[b],
                        mlp_params["lr"], mlp_params["weight_decay"], epochs=25)
        oof_mlp[b] = clip(preds)
        test_mlp += clip(predict_nn(model, X_te_np)) / N_FOLDS
        
        print("OK")
    
    # ── Level-2 Stacking ──
    X_meta_tr = np.column_stack([oof_lgb, oof_cat, oof_xgb, oof_mlp])
    X_meta_te = np.column_stack([test_lgb, test_cat, test_xgb, test_mlp])
    
    meta = Ridge(alpha=1.0, positive=True)
    meta.fit(X_meta_tr, y_full, sample_weight=temporal_w)
    
    oof_stack = clip(meta.predict(X_meta_tr))
    test_stack = clip(meta.predict(X_meta_te))
    
    seed_woof = wmse(oof_stack)
    print(f"  Seed {CURRENT_SEED} Stacking wOOF: {seed_woof:.3f}")
    print(f"  Meta weights: LGB={meta.coef_[0]:.3f} CAT={meta.coef_[1]:.3f} XGB={meta.coef_[2]:.3f} MLP={meta.coef_[3]:.3f}")
    
    final_test_preds += test_stack / len(SEEDS)

# Final submission
sub_path = os.path.join(BASE, "best_model_mega_upgrade", "submission_mega_upgrade.csv")
pd.DataFrame({ID_COL: te[ID_COL], TARGET: clip(final_test_preds)}).to_csv(sub_path, index=False)
print(f"\n{'='*60}")
print(f"FINAL: Saved {sub_path}")
print(f"Multi-seed average prediction saved.")
print(f"{'='*60}")
