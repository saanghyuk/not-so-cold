"""
Full-data (385K) verification for the key numbers in research_framing_kor.md.
Only runs Model A, B (no InApp) since those are what's cited in the framing.
Also runs corrected Model C, D with leakage-fixed InApp.
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

print("Loading data...")
df_raw = pd.read_csv('coldstart_dataset_260304.csv', low_memory=False)
df = df_raw[df_raw['IS_HAS_FRAUD'] != 1].copy()
print(f"Users: {len(df):,}")

# Parse InApp (leakage-safe)
INAPP_WINDOWS = ['m10']
INAPP_KEYS_RAW = ['active', 'ad_engagement', 'core_engagement', 'deeplink_count',
                  'open_count', 'purchase_engagement', 'totalEventCount']
INAPP_KEYS = ['active', 'ad_engagement', 'core_engagement', 'deeplink_count',
              'open_count', 'adjusted_totalEventCount']

for window in INAPP_WINDOWS:
    col = f'inapp_{window}'
    if col in df.columns:
        parsed = df[col].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else {})
        for key in INAPP_KEYS_RAW:
            df[f'inapp_{window}_{key}'] = parsed.apply(lambda d: d.get(key, 0))
        df[f'inapp_{window}_adjusted_totalEventCount'] = (
            df[f'inapp_{window}_totalEventCount'] - df[f'inapp_{window}_purchase_engagement']
        )
        df.drop(columns=[col, f'inapp_{window}_purchase_engagement', f'inapp_{window}_totalEventCount'], inplace=True)

def get_inapp_features(window):
    return [f'inapp_{window}_{k}' for k in INAPP_KEYS]

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

print(f"UA: {len(ua_cols)}, Device dummies: {len(device_cols)}")

target = 'IS_D7_PURCHASE'
y = df[target].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_params = dict(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

# ============================================================
# 1. 4-Model AUC (Full Data, RF)
# ============================================================
print("\n" + "=" * 60)
print("4-MODEL AUC — FULL DATA (385K), RF")
print("=" * 60)

inapp_m10_cols = [c for c in get_inapp_features('m10') if c in df.columns]

configs = {
    'A: Device Only': device_cols,
    'B: Device + UA': device_cols + ua_cols,
    'C: Device + UA + InApp(M10)': device_cols + ua_cols + inapp_m10_cols,
    'D: Device + InApp(M10)': device_cols + inapp_m10_cols,
}

for name, cols_list in configs.items():
    X = df[cols_list].replace([np.inf, -np.inf], np.nan).fillna(0).values
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X[train_idx], y[train_idx])
        y_prob = rf.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], y_prob))
    print(f'  {name}: AUC = {np.mean(aucs):.4f} (±{np.std(aucs):.4f})')

# ============================================================
# 2. Top/Bottom 10% (Full Data, RF, CV-based)
# ============================================================
print("\n" + "=" * 60)
print("TOP/BOTTOM 10% — FULL DATA, RF, CV-BASED")
print("=" * 60)

for name, cols_list in [('B: Device + UA', device_cols + ua_cols),
                         ('A: Device Only', device_cols)]:
    X = df[cols_list].replace([np.inf, -np.inf], np.nan).fillna(0).values
    all_probs = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X[train_idx], y[train_idx])
        all_probs[test_idx] = rf.predict_proba(X[test_idx])[:, 1]

    top10 = np.percentile(all_probs, 90)
    bot10 = np.percentile(all_probs, 10)
    top_rate = y[all_probs >= top10].mean() * 100
    bot_rate = y[all_probs <= bot10].mean() * 100
    ratio = top_rate / bot_rate if bot_rate > 0 else float('inf')
    print(f'  {name}: Top10%={top_rate:.1f}%, Bot10%={bot_rate:.1f}%, Ratio={ratio:.1f}x')

# ============================================================
# 3. Feature Importance (Full Data, RF) — m10
# ============================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE — FULL DATA, RF, M10")
print("=" * 60)

all_cols_m10 = device_cols + ua_cols + inapp_m10_cols
X_all = df[all_cols_m10].replace([np.inf, -np.inf], np.nan).fillna(0).values

rf_fi = RandomForestClassifier(**rf_params)
rf_fi.fit(X_all, y)
importance = pd.Series(rf_fi.feature_importances_, index=all_cols_m10)

d_pct = importance[device_cols].sum() / importance.sum() * 100
u_pct = importance[ua_cols].sum() / importance.sum() * 100
i_pct = importance[inapp_m10_cols].sum() / importance.sum() * 100
print(f'  Device: {d_pct:.1f}%, UA: {u_pct:.1f}%, InApp(M10): {i_pct:.1f}%')

# Top UA features
print('\nTop 10 UA features:')
ua_imp = importance[ua_cols].sort_values(ascending=False)
for feat, val in ua_imp.head(10).items():
    print(f'  {feat}: {val:.4f}')

print("\n=== DONE ===")
