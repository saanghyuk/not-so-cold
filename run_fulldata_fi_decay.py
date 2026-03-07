"""
Full-data Feature Importance Decay table (leakage-fixed).
For research_framing_kor.md lines 182-191.
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier

print("Loading data...")
df_raw = pd.read_csv('coldstart_dataset_260304.csv', low_memory=False)
df = df_raw[df_raw['IS_HAS_FRAUD'] != 1].copy()
print(f"Users: {len(df):,}")

# Parse InApp (leakage-safe) for windows needed
WINDOWS_NEEDED = ['m10', 'm30', 'm60', 'm90', 'm120', 'm180', 'm360', 'd1']
INAPP_KEYS_RAW = ['active', 'ad_engagement', 'core_engagement', 'deeplink_count',
                  'open_count', 'purchase_engagement', 'totalEventCount']
INAPP_KEYS = ['active', 'ad_engagement', 'core_engagement', 'deeplink_count',
              'open_count', 'adjusted_totalEventCount']

for window in WINDOWS_NEEDED:
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

# Use 100K sample for speed (still representative)
N = 100_000
np.random.seed(42)
idx = np.random.choice(len(df), N, replace=False)
df_s = df.iloc[idx].reset_index(drop=True)

target = 'IS_D7_PURCHASE'
y = df_s[target].values
rf_params = dict(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE DECAY — LEAKAGE-FIXED, 100K SAMPLE")
print("=" * 60)

# Note: d7 excluded because target is D7 purchase
decay_windows = ['m10', 'm30', 'm60', 'm90', 'm120', 'm180', 'm360', 'd1']

print(f'\n{"Window":<12} {"Device %":>10} {"UA %":>10} {"InApp %":>10}')
print('-' * 47)

for window in decay_windows:
    inapp_w_cols = [c for c in get_inapp_features(window) if c in df_s.columns]
    all_cols = device_cols + ua_cols + inapp_w_cols
    X_all = df_s[all_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_all, y)
    importance = pd.Series(rf.feature_importances_, index=all_cols)

    d_pct = importance[device_cols].sum() / importance.sum() * 100
    u_pct = importance[ua_cols].sum() / importance.sum() * 100
    i_pct = importance[inapp_w_cols].sum() / importance.sum() * 100

    print(f'{window:<12} {d_pct:>9.1f}% {u_pct:>9.1f}% {i_pct:>9.1f}%')

print("\n=== DONE ===")
