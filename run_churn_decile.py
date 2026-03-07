"""
D7 Churn prediction: top/bottom 10% decile table (same format as purchase prediction).
Models: RF (Device+UA), RF (Device only), LR (Device+UA), LR (Device only).
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

print("Loading data...")
df_raw = pd.read_csv('coldstart_dataset_260304.csv', low_memory=False)
df = df_raw[df_raw['IS_HAS_FRAUD'] != 1].copy()
print(f"Users: {len(df):,}")

# Parse InApp m10 (leakage-safe)
col = 'inapp_m10'
parsed = df[col].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else {})
INAPP_KEYS = ['active', 'ad_engagement', 'core_engagement', 'deeplink_count', 'open_count']
for key in INAPP_KEYS:
    df[f'inapp_m10_{key}'] = parsed.apply(lambda d: d.get(key, 0))
total_evt = parsed.apply(lambda d: d.get('totalEventCount', 0))
purch_evt = parsed.apply(lambda d: d.get('purchase_engagement', 0))
df['inapp_m10_adjusted_total'] = total_evt - purch_evt

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
ua_cols = [c for c in original_cols
           if not any(c.startswith(p) for p in exclude_prefixes) and c not in exclude_exact]

for c in DEVICE_FEATURES:
    df[c] = df[c].fillna('unknown')
device_dummies = pd.get_dummies(df[DEVICE_FEATURES], dtype=int)
device_cols = device_dummies.columns.tolist()
df = pd.concat([df, device_dummies], axis=1)
df[ua_cols] = df[ua_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)

target = 'IS_D7_CHURN'
y = df[target].values
n = len(y)

# Models
rf_params = dict(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def run_cv(X, y, model_type='rf'):
    preds = np.zeros(n)
    for train_idx, test_idx in skf.split(X, y):
        if model_type == 'rf':
            m = RandomForestClassifier(**rf_params)
            m.fit(X[train_idx], y[train_idx])
            preds[test_idx] = m.predict_proba(X[test_idx])[:, 1]
        else:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            m = LogisticRegression(max_iter=1000, random_state=42)
            m.fit(X_tr, y[train_idx])
            preds[test_idx] = m.predict_proba(X_te)[:, 1]
    return preds

# Feature sets
device_X = df[device_cols].values
ua_X = df[device_cols + ua_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

# For LR: sample 50K
np.random.seed(42)
lr_idx = np.random.choice(n, 50000, replace=False)

print("\n=== D7 CHURN PREDICTION: TOP/BOTTOM 10% ===\n")

configs = [
    ('Random Forest (Device+UA)', ua_X, y, 'rf', None),
    ('Random Forest (Device only)', device_X, y, 'rf', None),
    ('Logistic Regression (Device+UA)', ua_X, y, 'lr', lr_idx),
    ('Logistic Regression (Device only)', device_X, y, 'lr', lr_idx),
]

from sklearn.metrics import roc_auc_score

print(f"{'Model':<35} {'Top 10% Churn':>14} {'Bot 10% Churn':>14} {'Ratio':>8} {'AUC':>7}")
print("-" * 82)

for name, X_full, y_full, mtype, idx in configs:
    if idx is not None:
        X_use = X_full[idx]
        y_use = y_full[idx]
        n_use = len(idx)
    else:
        X_use = X_full
        y_use = y_full
        n_use = n

    preds_cv = np.zeros(n_use)
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_i, test_i in skf2.split(X_use, y_use):
        if mtype == 'rf':
            m = RandomForestClassifier(**rf_params)
            m.fit(X_use[train_i], y_use[train_i])
            preds_cv[test_i] = m.predict_proba(X_use[test_i])[:, 1]
        else:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_use[train_i])
            Xte = scaler.transform(X_use[test_i])
            m = LogisticRegression(max_iter=1000, random_state=42)
            m.fit(Xtr, y_use[train_i])
            preds_cv[test_i] = m.predict_proba(Xte)[:, 1]

    auc = roc_auc_score(y_use, preds_cv)
    top10 = preds_cv >= np.percentile(preds_cv, 90)
    bot10 = preds_cv <= np.percentile(preds_cv, 10)
    top_rate = y_use[top10].mean() * 100
    bot_rate = y_use[bot10].mean() * 100
    ratio = top_rate / bot_rate if bot_rate > 0 else float('inf')
    print(f"  {name:<33} {top_rate:>13.1f}% {bot_rate:>13.1f}% {ratio:>7.1f}x {auc:>6.3f}")

print("\nDONE")
