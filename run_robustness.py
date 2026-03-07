"""
Robustness analyses requested by AE reviewers.
1. has_broken_image removal robustness
2. RF max_depth sensitivity (5, 7, 10, 15)
3. Fraud users' channel distribution
4. Cohort period specification
5. Zero-activity user subgroup AUC
6. SA/DA feature importance decay comparison
7. PSM Rosenbaum bounds (sensitivity analysis)
8. Permutation importance vs RF importance
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

print("Loading data...")
df_raw = pd.read_csv('coldstart_dataset_260304.csv', low_memory=False)
df = df_raw[df_raw['IS_HAS_FRAUD'] != 1].copy()
print(f"Users (fraud excluded): {len(df):,}")

# Parse InApp m10 (leakage-safe)
INAPP_KEYS_RAW = ['active', 'ad_engagement', 'core_engagement', 'deeplink_count',
                  'open_count', 'purchase_engagement', 'totalEventCount']
INAPP_KEYS = ['active', 'ad_engagement', 'core_engagement', 'deeplink_count',
              'open_count', 'adjusted_totalEventCount']

col = 'inapp_m10'
if col in df.columns:
    parsed = df[col].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else {})
    for key in INAPP_KEYS_RAW:
        df[f'inapp_m10_{key}'] = parsed.apply(lambda d: d.get(key, 0))
    df['inapp_m10_adjusted_totalEventCount'] = (
        df['inapp_m10_totalEventCount'] - df['inapp_m10_purchase_engagement']
    )
    df.drop(columns=[col, 'inapp_m10_purchase_engagement', 'inapp_m10_totalEventCount'], inplace=True)

inapp_m10_cols = [f'inapp_m10_{k}' for k in INAPP_KEYS]

# Feature setup
DEVICE_FEATURES = ['OS_NAME', 'DEVICE_MANUFACTURER', 'DEVICE_LANGUAGE',
                   'DEVICE_TIMEZONE', 'DEVICE_OSVERSION', 'DEVICE_CARRIER']
exclude_prefixes = ['IS_', 'TARGET_', 'inapp_', 'ltv']
meta_cols = ['IDFA', 'IDFV', 'GAID', 'INSTALL_TIMESTAMP', 'USER_ID']
creative_image_cols = [
    'creative_brightness_mean', 'creative_saturation_mean', 'creative_hue_mean',
    'creative_brightness_std', 'creative_saturation_std', 'creative_hue_std',
    'creative_colorfulness', 'creative_symmetry_score',
    'brightness_mean', 'saturation_mean', 'color_entropy', 'edge_density',
    'hue_cos', 'hue_sin', 'symmetry_score', 'vertical_symmetry_score'
]
exclude_exact = set(meta_cols + DEVICE_FEATURES + creative_image_cols + ['media_type', 'keyword_list', 'ocr_text'])
original_cols = pd.read_csv('coldstart_dataset_260304.csv', nrows=0).columns.tolist()
ua_cols = [col for col in original_cols
           if not any(col.startswith(p) for p in exclude_prefixes) and col not in exclude_exact]

for col in DEVICE_FEATURES:
    df[col] = df[col].fillna('unknown')
device_dummies = pd.get_dummies(df[DEVICE_FEATURES], dtype=int)
device_cols = device_dummies.columns.tolist()
df = pd.concat([df, device_dummies], axis=1)
df[ua_cols] = df[ua_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)

target = 'IS_D7_PURCHASE'
y_full = df[target].values
print(f"UA: {len(ua_cols)}, Device: {len(device_cols)}")

# Sample for faster analyses
N_SAMPLE = 50_000
np.random.seed(42)
sample_idx = np.random.choice(len(df), N_SAMPLE, replace=False)
df_sample = df.iloc[sample_idx].reset_index(drop=True)
y = df_sample[target].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 1. has_broken_image removal robustness
# ============================================================
print("\n" + "=" * 60)
print("[1/8] HAS_BROKEN_IMAGE REMOVAL ROBUSTNESS")
print("=" * 60)

ua_cols_no_broken = [c for c in ua_cols if c != 'has_broken_image']
rf_params = dict(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

for label, cols in [("With has_broken_image", ua_cols), ("Without has_broken_image", ua_cols_no_broken)]:
    X = df_sample[device_cols + cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    aucs = []
    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1]))
    print(f"  {label}: AUC = {np.mean(aucs):.4f} (±{np.std(aucs):.4f})")

# ============================================================
# 2. RF max_depth sensitivity
# ============================================================
print("\n" + "=" * 60)
print("[2/8] RF MAX_DEPTH SENSITIVITY")
print("=" * 60)

X_devua = df_sample[device_cols + ua_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
X_dev = df_sample[device_cols].values

for depth in [5, 7, 10, 15, None]:
    aucs_devua = []
    aucs_dev = []
    for tr, te in skf.split(X_devua, y):
        rf1 = RandomForestClassifier(n_estimators=200, max_depth=depth, random_state=42, n_jobs=-1)
        rf1.fit(X_devua[tr], y[tr])
        aucs_devua.append(roc_auc_score(y[te], rf1.predict_proba(X_devua[te])[:, 1]))

        rf2 = RandomForestClassifier(n_estimators=200, max_depth=depth, random_state=42, n_jobs=-1)
        rf2.fit(X_dev[tr], y[tr])
        aucs_dev.append(roc_auc_score(y[te], rf2.predict_proba(X_dev[te])[:, 1]))

    lift = np.mean(aucs_devua) - np.mean(aucs_dev)
    depth_str = str(depth) if depth else "None(unlimited)"
    print(f"  depth={depth_str:>15}: Dev={np.mean(aucs_dev):.4f}, Dev+UA={np.mean(aucs_devua):.4f}, lift={lift:+.4f}")

# ============================================================
# 3. Fraud users' channel distribution
# ============================================================
print("\n" + "=" * 60)
print("[3/8] FRAUD USERS' CHANNEL DISTRIBUTION")
print("=" * 60)

df_fraud = df_raw[df_raw['IS_HAS_FRAUD'] == 1].copy()
df_clean = df_raw[df_raw['IS_HAS_FRAUD'] != 1].copy()

print(f"  Fraud users: {len(df_fraud):,}")
print(f"  Clean users: {len(df_clean):,}")

for label, dff in [("Fraud", df_fraud), ("Clean", df_clean)]:
    n = len(dff)
    sa = (dff.get('last_touch_is_sa', pd.Series(dtype=float)).fillna(0) == 1).sum()
    da = (dff.get('last_touch_is_da', pd.Series(dtype=float)).fillna(0) == 1).sum()
    organic = (dff.get('has_touchpoint', pd.Series(dtype=float)).fillna(0) == 0).sum()
    paid = n - organic
    print(f"\n  {label}:")
    print(f"    Paid: {paid:,} ({paid/n*100:.1f}%)")
    print(f"    Organic: {organic:,} ({organic/n*100:.1f}%)")
    print(f"    SA (last touch): {sa:,} ({sa/n*100:.1f}%)")
    print(f"    DA (last touch): {da:,} ({da/n*100:.1f}%)")
    if label == "Fraud":
        d7_purchase_rate = dff['IS_D7_PURCHASE'].mean() * 100
        d7_churn_rate = dff['IS_D7_CHURN'].mean() * 100 if 'IS_D7_CHURN' in dff.columns else float('nan')
        print(f"    D7 purchase rate: {d7_purchase_rate:.1f}%")
        print(f"    D7 churn rate: {d7_churn_rate:.1f}%")

# ============================================================
# 4. Cohort period specification
# ============================================================
print("\n" + "=" * 60)
print("[4/8] COHORT PERIOD SPECIFICATION")
print("=" * 60)

if 'INSTALL_TIMESTAMP' in df.columns:
    ts = pd.to_datetime(df['INSTALL_TIMESTAMP'], errors='coerce')
    print(f"  Earliest install: {ts.min()}")
    print(f"  Latest install: {ts.max()}")
    print(f"  Period: {(ts.max() - ts.min()).days} days")
    print(f"  Missing timestamps: {ts.isna().sum():,}")

    # Weekly distribution
    ts_week = ts.dt.isocalendar().week
    print(f"\n  Weekly distribution:")
    for week, count in ts_week.value_counts().sort_index().items():
        print(f"    Week {week}: {count:,} ({count/len(df)*100:.1f}%)")
else:
    print("  INSTALL_TIMESTAMP not found")

# ============================================================
# 5. Zero-activity user subgroup AUC
# ============================================================
print("\n" + "=" * 60)
print("[5/8] ZERO-ACTIVITY USER SUBGROUP AUC")
print("=" * 60)

# Users with zero core_engagement in first 10 min
zero_mask = df_sample['inapp_m10_core_engagement'] == 0
n_zero = zero_mask.sum()
n_active = (~zero_mask).sum()
print(f"  Zero activity (10min): {n_zero:,} ({n_zero/len(df_sample)*100:.1f}%)")
print(f"  Active (10min): {n_active:,} ({n_active/len(df_sample)*100:.1f}%)")
print(f"  Zero-activity D7 purchase rate: {y[zero_mask].mean()*100:.1f}%")
print(f"  Active D7 purchase rate: {y[~zero_mask].mean()*100:.1f}%")

# Model B AUC for zero-activity subgroup only
X_full = df_sample[device_cols + ua_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

# Use full CV predictions then subset
all_probs_devua = np.zeros(len(y))
all_probs_dev = np.zeros(len(y))

for tr, te in skf.split(X_full, y):
    rf1 = RandomForestClassifier(**rf_params)
    rf1.fit(X_full[tr], y[tr])
    all_probs_devua[te] = rf1.predict_proba(X_full[te])[:, 1]

    rf2 = RandomForestClassifier(**rf_params)
    rf2.fit(X_dev[tr], y[tr])
    all_probs_dev[te] = rf2.predict_proba(X_dev[te])[:, 1]

# AUC for zero subgroup
if y[zero_mask].sum() > 10 and (~(y[zero_mask].astype(bool))).sum() > 10:
    auc_zero_devua = roc_auc_score(y[zero_mask], all_probs_devua[zero_mask])
    auc_zero_dev = roc_auc_score(y[zero_mask], all_probs_dev[zero_mask])
    print(f"\n  Zero-activity subgroup:")
    print(f"    Device-only AUC: {auc_zero_dev:.4f}")
    print(f"    Device+UA AUC: {auc_zero_devua:.4f}")
    print(f"    UA lift: {auc_zero_devua - auc_zero_dev:+.4f}")

    # Top/Bottom 10% for zero-activity
    top10 = np.percentile(all_probs_devua[zero_mask], 90)
    bot10 = np.percentile(all_probs_devua[zero_mask], 10)
    top_rate = y[zero_mask][all_probs_devua[zero_mask] >= top10].mean() * 100
    bot_rate = y[zero_mask][all_probs_devua[zero_mask] <= bot10].mean() * 100
    ratio = top_rate / bot_rate if bot_rate > 0 else float('inf')
    print(f"    Top 10% purchase rate: {top_rate:.1f}%")
    print(f"    Bot 10% purchase rate: {bot_rate:.1f}%")
    print(f"    Ratio: {ratio:.1f}x")

# AUC for active subgroup
if y[~zero_mask].sum() > 10:
    auc_active_devua = roc_auc_score(y[~zero_mask], all_probs_devua[~zero_mask])
    auc_active_dev = roc_auc_score(y[~zero_mask], all_probs_dev[~zero_mask])
    print(f"\n  Active subgroup:")
    print(f"    Device-only AUC: {auc_active_dev:.4f}")
    print(f"    Device+UA AUC: {auc_active_devua:.4f}")
    print(f"    UA lift: {auc_active_devua - auc_active_dev:+.4f}")

# ============================================================
# 6. SA/DA information decay comparison
# ============================================================
print("\n" + "=" * 60)
print("[6/8] SA/DA FEATURE IMPORTANCE DECAY COMPARISON")
print("=" * 60)

# Parse additional inapp windows for full data
WINDOWS = ['m10', 'm30', 'm60', 'm120', 'm360', 'd1']
for window in WINDOWS:
    inapp_col = f'inapp_{window}'
    if inapp_col in df.columns:
        parsed = df[inapp_col].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else {})
        for key in ['active', 'ad_engagement', 'core_engagement', 'deeplink_count', 'open_count', 'purchase_engagement', 'totalEventCount']:
            df[f'inapp_{window}_{key}'] = parsed.apply(lambda d: d.get(key, 0))
        df[f'inapp_{window}_adjusted_totalEventCount'] = (
            df[f'inapp_{window}_totalEventCount'] - df[f'inapp_{window}_purchase_engagement']
        )
        for drop_col in [inapp_col, f'inapp_{window}_purchase_engagement', f'inapp_{window}_totalEventCount']:
            if drop_col in df.columns:
                df.drop(columns=[drop_col], inplace=True, errors='ignore')

# SA vs DA subgroup FI decay
sa_mask_full = df['last_touch_is_sa'] == 1
da_mask_full = df['last_touch_is_da'] == 1

print(f"\n  SA users: {sa_mask_full.sum():,}")
print(f"  DA users: {da_mask_full.sum():,}")

print(f"\n  {'Window':<10} {'SA_UA%':>8} {'SA_InApp%':>10} {'DA_UA%':>8} {'DA_InApp%':>10}")
print("-" * 50)

for window in ['m10', 'm30', 'm60', 'm360', 'd1']:
    inapp_w_cols = [f'inapp_{window}_{k}' for k in INAPP_KEYS if f'inapp_{window}_{k}' in df.columns]
    all_feat = device_cols + ua_cols + inapp_w_cols

    for label, mask in [("SA", sa_mask_full), ("DA", da_mask_full)]:
        df_sub = df[mask]
        if len(df_sub) < 1000:
            continue
        # Sample for speed
        n_sub = min(30000, len(df_sub))
        sub_idx = np.random.choice(len(df_sub), n_sub, replace=False)
        df_sub_s = df_sub.iloc[sub_idx]
        X_sub = df_sub_s[all_feat].replace([np.inf, -np.inf], np.nan).fillna(0).values
        y_sub = df_sub_s[target].values

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_sub, y_sub)
        imp = pd.Series(rf.feature_importances_, index=all_feat)
        total = imp.sum()

        if label == "SA":
            sa_ua = imp[ua_cols].sum() / total * 100
            sa_inapp = imp[inapp_w_cols].sum() / total * 100
        else:
            da_ua = imp[ua_cols].sum() / total * 100
            da_inapp = imp[inapp_w_cols].sum() / total * 100

    print(f"  {window:<10} {sa_ua:>7.1f}% {sa_inapp:>9.1f}% {da_ua:>7.1f}% {da_inapp:>9.1f}%")

# ============================================================
# 7. PSM Rosenbaum Bounds (Sensitivity Analysis)
# ============================================================
print("\n" + "=" * 60)
print("[7/8] PSM ROSENBAUM BOUNDS (SENSITIVITY ANALYSIS)")
print("=" * 60)

# Simplified Rosenbaum bounds: test how large an unmeasured confounder
# would need to be to nullify the PSM result
# The original PSM showed +5.0%p effect after matching

# We'll compute the PSM effect and then calculate critical Gamma
# using the approach: at what odds ratio of hidden bias does the p-value become > 0.05?

# First, replicate PSM
from sklearn.neighbors import NearestNeighbors

# Define high/low activity based on median core_engagement in m10
# Only for users who survived 10 min
survived_mask = df_sample['inapp_m10_active'] > 0
df_surv = df_sample[survived_mask].copy()
y_surv = y[survived_mask]

median_ce = df_surv['inapp_m10_core_engagement'].median()
treatment = (df_surv['inapp_m10_core_engagement'] >= median_ce).astype(int).values

# Propensity score model using UA + device features
X_ps = df_surv[device_cols + ua_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
scaler = StandardScaler()
X_ps_scaled = scaler.fit_transform(X_ps)

lr_ps = LogisticRegression(max_iter=1000, random_state=42)
lr_ps.fit(X_ps_scaled, treatment)
pscore = lr_ps.predict_proba(X_ps_scaled)[:, 1]

# Nearest neighbor matching
treated_idx = np.where(treatment == 1)[0]
control_idx = np.where(treatment == 0)[0]

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(pscore[control_idx].reshape(-1, 1))
distances, indices = nn.kneighbors(pscore[treated_idx].reshape(-1, 1))

# Caliper matching (0.05)
caliper = 0.05
valid_matches = distances.flatten() < caliper
n_matched = valid_matches.sum()

matched_treated = treated_idx[valid_matches]
matched_control = control_idx[indices.flatten()[valid_matches]]

purchase_treated = y_surv[matched_treated]
purchase_control = y_surv[matched_control]

effect = purchase_treated.mean() - purchase_control.mean()
print(f"  Matched pairs: {n_matched:,}")
print(f"  Treated D7 purchase rate: {purchase_treated.mean()*100:.1f}%")
print(f"  Control D7 purchase rate: {purchase_control.mean()*100:.1f}%")
print(f"  PSM effect: {effect*100:+.1f}%p")

# Rosenbaum bounds: sensitivity analysis
# For McNemar's test on matched pairs
# Discordant pairs
disc_10 = ((purchase_treated == 1) & (purchase_control == 0)).sum()  # treated yes, control no
disc_01 = ((purchase_treated == 0) & (purchase_control == 1)).sum()  # treated no, control yes

print(f"\n  Discordant pairs: {disc_10 + disc_01:,}")
print(f"    Treated=1, Control=0: {disc_10:,}")
print(f"    Treated=0, Control=1: {disc_01:,}")

# Under Rosenbaum bounds, for a given Gamma, the test statistic changes
# Gamma = odds ratio of differential treatment assignment due to unmeasured confounder
from scipy.stats import norm

print(f"\n  Rosenbaum Sensitivity (Gamma = odds of hidden bias):")
print(f"  {'Gamma':>8} {'Upper p-value':>15} {'Significant?':>14}")
print("  " + "-" * 40)

for gamma in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0]:
    # Upper bound p-value under Gamma
    T_plus = disc_10  # observed test statistic
    n_disc = disc_10 + disc_01
    if n_disc == 0:
        continue
    # Under Gamma, E[T+] = n_disc * gamma/(1+gamma), Var[T+] = n_disc * gamma/(1+gamma)^2
    p_gamma = gamma / (1 + gamma)
    E_T = n_disc * p_gamma
    V_T = n_disc * p_gamma * (1 - p_gamma)
    z = (T_plus - E_T) / np.sqrt(V_T)
    p_upper = 1 - norm.cdf(z)
    sig = "Yes" if p_upper < 0.05 else "NO"
    print(f"  {gamma:>8.1f} {p_upper:>15.6f} {sig:>14}")

# ============================================================
# 8. Permutation Importance vs RF Importance
# ============================================================
print("\n" + "=" * 60)
print("[8/8] PERMUTATION IMPORTANCE VS RF IMPORTANCE")
print("=" * 60)

# Use 50K sample, train on 80%, compute permutation importance on 20%
from sklearn.model_selection import train_test_split

X_perm = df_sample[device_cols + ua_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
X_train, X_test, y_train, y_test = train_test_split(X_perm, y, test_size=0.2, random_state=42, stratify=y)

rf_perm = RandomForestClassifier(**rf_params)
rf_perm.fit(X_train, y_train)

# RF (MDI) importance
mdi_imp = pd.Series(rf_perm.feature_importances_, index=device_cols + ua_cols)

# Permutation importance
perm_result = permutation_importance(rf_perm, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_imp = pd.Series(perm_result.importances_mean, index=device_cols + ua_cols)

# Compare group-level
print("\n  Group-level importance comparison:")
print(f"  {'Group':<12} {'MDI %':>8} {'Permutation %':>15}")
print("  " + "-" * 38)

mdi_total = mdi_imp.sum()
perm_total = perm_imp[perm_imp > 0].sum()  # only positive contributions

for group_name, group_cols in [("Device", device_cols), ("UA", ua_cols)]:
    mdi_pct = mdi_imp[group_cols].sum() / mdi_total * 100
    perm_pct = perm_imp[group_cols].sum() / perm_total * 100 if perm_total > 0 else 0
    print(f"  {group_name:<12} {mdi_pct:>7.1f}% {perm_pct:>14.1f}%")

# Top 10 comparison
print("\n  Top 10 UA features comparison:")
mdi_ua = mdi_imp[ua_cols].sort_values(ascending=False)
perm_ua = perm_imp[ua_cols].sort_values(ascending=False)

print(f"\n  {'Rank':>4} {'MDI Feature':<30} {'MDI%':>6} {'Perm Feature':<30} {'Perm imp':>10}")
print("  " + "-" * 85)
for i in range(10):
    mdi_name = mdi_ua.index[i]
    mdi_val = mdi_ua.iloc[i] / mdi_total * 100
    perm_name = perm_ua.index[i]
    perm_val = perm_ua.iloc[i]
    print(f"  {i+1:>4} {mdi_name:<30} {mdi_val:>5.1f}% {perm_name:<30} {perm_val:>10.4f}")

# Rank correlation between MDI and Permutation for UA features
from scipy.stats import spearmanr
mdi_ranks = mdi_ua.rank(ascending=False)
perm_ranks = perm_ua.reindex(mdi_ua.index).rank(ascending=False)
rho, p_val = spearmanr(mdi_ranks, perm_ranks)
print(f"\n  Spearman rank correlation (MDI vs Permutation, UA features): rho={rho:.3f}, p={p_val:.4f}")

print("\n" + "=" * 60)
print("ALL ROBUSTNESS ANALYSES COMPLETE")
print("=" * 60)
