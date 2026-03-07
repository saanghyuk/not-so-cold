"""
Fix verification for:
1. Top/Bottom 10% — now using CV predictions (not in-sample)
2. Bootstrap CI — now using OOB evaluation (not in-bag)
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

print("Loading data...")
df_raw = pd.read_csv('coldstart_dataset_260304.csv', low_memory=False)
df = df_raw[df_raw['IS_HAS_FRAUD'] != 1].copy()
print(f"Users: {len(df):,}")

# Feature setup (same as notebook)
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

N_SAMPLE = 50_000
np.random.seed(42)
sample_idx = np.random.choice(len(df), N_SAMPLE, replace=False)
df_sample = df.iloc[sample_idx].reset_index(drop=True)

print(f"UA: {len(ua_cols)}, Device: {len(device_cols)}, Sample: {N_SAMPLE:,}")

# ============================================================
# 1. Top/Bottom 10% — CV-based
# ============================================================
print("\n" + "=" * 60)
print("TOP/BOTTOM 10% — CV-BASED (FIXED)")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for target_label, target_col in [('D7 Purchase', 'IS_D7_PURCHASE'),
                                  ('D14 Purchase', 'IS_D14_PURCHASE'),
                                  ('D30 Purchase', 'IS_D30_PURCHASE')]:
    X_dev_ua = df_sample[device_cols + ua_cols].values
    X_dev = df_sample[device_cols].values
    y_t = df_sample[target_col].values

    # Device + UA (CV predictions)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    probs = cross_val_predict(lr, X_dev_ua, y_t, cv=skf, method='predict_proba')[:, 1]
    top10 = np.percentile(probs, 90)
    bot10 = np.percentile(probs, 10)
    top_rate = y_t[probs >= top10].mean() * 100
    bot_rate = y_t[probs <= bot10].mean() * 100
    ratio = top_rate / bot_rate if bot_rate > 0 else float('inf')
    print(f'{target_label} (Dev+UA): Top={top_rate:.1f}%, Bot={bot_rate:.1f}%, Ratio={ratio:.1f}x')

    # Device only (CV predictions)
    lr2 = LogisticRegression(max_iter=1000, random_state=42)
    probs2 = cross_val_predict(lr2, X_dev, y_t, cv=skf, method='predict_proba')[:, 1]
    top10_2 = np.percentile(probs2, 90)
    bot10_2 = np.percentile(probs2, 10)
    top_rate_2 = y_t[probs2 >= top10_2].mean() * 100
    bot_rate_2 = y_t[probs2 <= bot10_2].mean() * 100
    ratio_2 = top_rate_2 / bot_rate_2 if bot_rate_2 > 0 else float('inf')
    print(f'{target_label} (Dev only): Top={top_rate_2:.1f}%, Bot={bot_rate_2:.1f}%, Ratio={ratio_2:.1f}x')
    print()

# Also with RF
print("--- RF Model ---")
for target_label, target_col in [('D7 Purchase', 'IS_D7_PURCHASE')]:
    X_dev_ua = df_sample[device_cols + ua_cols].values
    y_t = df_sample[target_col].values

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    probs_rf = cross_val_predict(rf, X_dev_ua, y_t, cv=skf, method='predict_proba')[:, 1]
    top10 = np.percentile(probs_rf, 90)
    bot10 = np.percentile(probs_rf, 10)
    top_rate = y_t[probs_rf >= top10].mean() * 100
    bot_rate = y_t[probs_rf <= bot10].mean() * 100
    ratio = top_rate / bot_rate if bot_rate > 0 else float('inf')
    print(f'{target_label} RF (Dev+UA): Top={top_rate:.1f}%, Bot={bot_rate:.1f}%, Ratio={ratio:.1f}x')

# ============================================================
# 2. Bootstrap CI — OOB evaluation
# ============================================================
print("\n" + "=" * 60)
print("BOOTSTRAP CI — OOB EVALUATION (FIXED)")
print("=" * 60)

n_bootstrap = 200
lifts = []
y_d7 = df_sample['IS_D7_PURCHASE'].values
N = len(df_sample)

for i in range(n_bootstrap):
    idx = np.random.choice(N, N, replace=True)
    oob = np.setdiff1d(np.arange(N), np.unique(idx))

    if len(oob) < 100:
        continue

    X_dev_train = df_sample.iloc[idx][device_cols].values
    X_devua_train = df_sample.iloc[idx][device_cols + ua_cols].values
    y_train = y_d7[idx]

    X_dev_oob = df_sample.iloc[oob][device_cols].values
    X_devua_oob = df_sample.iloc[oob][device_cols + ua_cols].values
    y_oob = y_d7[oob]

    lr_dev = LogisticRegression(max_iter=1000, random_state=42)
    lr_dev.fit(X_dev_train, y_train)
    auc_dev = roc_auc_score(y_oob, lr_dev.predict_proba(X_dev_oob)[:, 1])

    lr_ua = LogisticRegression(max_iter=1000, random_state=42)
    lr_ua.fit(X_devua_train, y_train)
    auc_ua = roc_auc_score(y_oob, lr_ua.predict_proba(X_devua_oob)[:, 1])

    lifts.append(auc_ua - auc_dev)

lifts = np.array(lifts)
print(f'Iterations: {len(lifts)}')
print(f'Mean lift: {lifts.mean():.4f}')
print(f'95% CI: [{np.percentile(lifts, 2.5):.4f}, {np.percentile(lifts, 97.5):.4f}]')
print(f'P(lift>0): {(lifts > 0).mean()*100:.1f}%')
print(f'\nOLD (in-bag): CI [0.049, 0.070]')

print("\nDONE")
