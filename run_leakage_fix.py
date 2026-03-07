"""
Run the leakage-fixed InApp models to get corrected AUC values.
Fixes applied:
1. purchase_engagement EXCLUDED from features
2. adjusted_totalEventCount = totalEventCount - purchase_engagement
3. get_inapp_features() returns only clean keys
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

print("=" * 60)
print("LEAKAGE-FIX VERIFICATION SCRIPT")
print("=" * 60)

# --- Load data ---
print("\n[1/7] Loading data...")
df_raw = pd.read_csv('coldstart_dataset_260304.csv', low_memory=False)
df = df_raw[df_raw['IS_HAS_FRAUD'] != 1].copy()
print(f"Analysis users: {len(df):,}")

# --- Parse InApp JSON (FIXED) ---
print("\n[2/7] Parsing InApp JSON (leakage-safe)...")
INAPP_WINDOWS = ['m10', 'm30', 'm60', 'm90', 'm120', 'm150', 'm180',
                 'm210', 'm240', 'm270', 'm300', 'm330', 'm360',
                 'd1', 'd2', 'd3', 'd7', 'd14', 'd30']

INAPP_KEYS_RAW = ['active', 'ad_engagement', 'core_engagement', 'deeplink_count',
                  'open_count', 'purchase_engagement', 'totalEventCount']

# Clean keys for modeling
INAPP_KEYS = ['active', 'ad_engagement', 'core_engagement', 'deeplink_count',
              'open_count', 'adjusted_totalEventCount']

for window in INAPP_WINDOWS:
    col = f'inapp_{window}'
    if col in df.columns:
        parsed = df[col].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else {})
        for key in INAPP_KEYS_RAW:
            df[f'inapp_{window}_{key}'] = parsed.apply(lambda d: d.get(key, 0))
        # Compute adjusted_totalEventCount
        df[f'inapp_{window}_adjusted_totalEventCount'] = (
            df[f'inapp_{window}_totalEventCount'] - df[f'inapp_{window}_purchase_engagement']
        )
        # Drop leaky columns
        df.drop(columns=[col, f'inapp_{window}_purchase_engagement', f'inapp_{window}_totalEventCount'], inplace=True)

def get_inapp_features(window):
    return [f'inapp_{window}_{k}' for k in INAPP_KEYS]

print(f"Clean InApp keys: {INAPP_KEYS}")
print(f"Features per window: {len(INAPP_KEYS)}")

# --- Verify purchase_engagement correlation is gone ---
print("\n[2b] Correlation check (m10 features vs IS_D7_PURCHASE):")
for k in INAPP_KEYS:
    col = f'inapp_m10_{k}'
    if col in df.columns:
        corr = df[col].corr(df['IS_D7_PURCHASE'])
        print(f"  {k}: r={corr:.4f}")

# --- Feature setup ---
print("\n[3/7] Setting up features...")
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
exclude_exact = set(meta_cols + DEVICE_FEATURES + creative_image_cols + ['media_type', 'keyword_list', 'ocr_text', 'channel'])

original_cols = pd.read_csv('coldstart_dataset_260304.csv', nrows=0).columns.tolist()
ua_cols = []
for col in original_cols:
    if any(col.startswith(p) for p in exclude_prefixes):
        continue
    if col in exclude_exact:
        continue
    ua_cols.append(col)

print(f"UA features: {len(ua_cols)}")

# Device encoding
for col in DEVICE_FEATURES:
    df[col] = df[col].fillna('unknown')
device_dummies = pd.get_dummies(df[DEVICE_FEATURES], dtype=int)
device_cols = device_dummies.columns.tolist()
df = pd.concat([df, device_dummies], axis=1)

df[ua_cols] = df[ua_cols].apply(pd.to_numeric, errors='coerce')
df[ua_cols] = df[ua_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# Sample
N_SAMPLE = 50_000
np.random.seed(42)
sample_idx = np.random.choice(len(df), N_SAMPLE, replace=False)
df_sample = df.iloc[sample_idx].reset_index(drop=True)

print(f"Device dummies: {len(device_cols)}")
print(f"Sample: {N_SAMPLE:,}")

# --- CV AUC utility ---
def compute_cv_auc(X, y, model_type='rf', n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if model_type == 'lr':
            model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, y_prob))
    return np.mean(aucs), np.std(aucs)

# ============================================================
# [4/7] 4-Model AUC Comparison (THE KEY TEST)
# ============================================================
print("\n" + "=" * 60)
print("[4/7] 4-MODEL AUC COMPARISON — LEAKAGE-FIXED")
print("=" * 60)

target = 'IS_D7_PURCHASE'
y = df_sample[target].values

inapp_m10_cols = [c for c in get_inapp_features('m10') if c in df_sample.columns]
print(f"\nInApp M10 features used: {inapp_m10_cols}")

configs = {
    'A: Device Only': device_cols,
    'B: Device + UA': device_cols + ua_cols,
    'C: Device + UA + InApp(M10)': device_cols + ua_cols + inapp_m10_cols,
    'D: Device + InApp(M10)': device_cols + inapp_m10_cols,
}

for algo in ['rf']:
    print(f'\nAlgorithm: {algo.upper()}')
    for name, cols_list in configs.items():
        X = df_sample[cols_list].replace([np.inf, -np.inf], np.nan).fillna(0).values
        mean_auc, std_auc = compute_cv_auc(X, y, model_type=algo)
        print(f'  {name}: AUC = {mean_auc:.4f} (±{std_auc:.4f})')

print(f'\nOLD (with leakage): C=0.723, D=0.723')

# ============================================================
# [5/7] Feature Importance Decay (FIXED — no d7 for D7 target)
# ============================================================
print("\n" + "=" * 60)
print("[5/7] FEATURE IMPORTANCE DECAY — LEAKAGE-FIXED")
print("=" * 60)

decay_windows = ['m10', 'm30', 'm60', 'm360', 'd1']
y_d7 = df_sample[target].values

print(f'\n{"Window":<10} {"Device %":>10} {"UA %":>10} {"InApp %":>10}')
print('-' * 45)

for window in decay_windows:
    inapp_w_cols = [c for c in get_inapp_features(window) if c in df_sample.columns]
    all_cols = device_cols + ua_cols + inapp_w_cols
    X_all = df_sample[all_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_all, y_d7)
    importance = pd.Series(rf.feature_importances_, index=all_cols)

    device_pct = importance[device_cols].sum() / importance.sum() * 100
    ua_pct = importance[ua_cols].sum() / importance.sum() * 100
    inapp_pct = importance[inapp_w_cols].sum() / importance.sum() * 100

    print(f'{window:<10} {device_pct:>9.1f}% {ua_pct:>9.1f}% {inapp_pct:>9.1f}%')

print(f'\nOLD (with leakage): m10: 7.1%/29.3%/63.6%, d7: 0.5%/2.3%/97.2%')

# ============================================================
# [6/7] Top/Bottom 10% for Models C and D
# ============================================================
print("\n" + "=" * 60)
print("[6/7] TOP/BOTTOM 10% — LEAKAGE-FIXED")
print("=" * 60)

for name, cols_list in configs.items():
    X = df_sample[cols_list].replace([np.inf, -np.inf], np.nan).fillna(0).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_probs = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X[train_idx], y[train_idx])
        all_probs[test_idx] = rf.predict_proba(X[test_idx])[:, 1]

    top10 = np.percentile(all_probs, 90)
    bot10 = np.percentile(all_probs, 10)
    top_rate = y[all_probs >= top10].mean() * 100
    bot_rate = y[all_probs <= bot10].mean() * 100
    ratio = top_rate / bot_rate if bot_rate > 0 else float('inf')
    print(f'{name}: Top10%={top_rate:.1f}%, Bot10%={bot_rate:.1f}%, Ratio={ratio:.1f}x')

# ============================================================
# [7/7] AUC Decay (first 6 windows only, for speed)
# ============================================================
print("\n" + "=" * 60)
print("[7/7] AUC DECAY (UA incremental value) — LEAKAGE-FIXED")
print("=" * 60)

target_d30 = 'IS_D30_PURCHASE'
y_d30 = df_sample[target_d30].values

short_windows = ['m10', 'm30', 'm60', 'm360', 'd1', 'd3', 'd7']
print(f'\n{"Window":<10} {"AUC (no UA)":>12} {"AUC (+UA)":>12} {"ΔUA":>8}')
print('-' * 48)

for window in short_windows:
    inapp_cols_w = [c for c in get_inapp_features(window) if c in df_sample.columns]

    X_no = df_sample[device_cols + inapp_cols_w].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_with = df_sample[device_cols + ua_cols + inapp_cols_w].replace([np.inf, -np.inf], np.nan).fillna(0).values

    auc_no, _ = compute_cv_auc(X_no, y_d30, 'lr')
    auc_with, _ = compute_cv_auc(X_with, y_d30, 'lr')
    delta = auc_with - auc_no

    print(f'{window:<10} {auc_no:>11.4f} {auc_with:>11.4f} {delta:>+7.4f}')

print("\n" + "=" * 60)
print("DONE — Compare above with framing document values")
print("=" * 60)
