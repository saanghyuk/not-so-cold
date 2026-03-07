"""
Complete re-verification with correct methodology:
1. RF for Top/Bottom 10% (CV-based)
2. RF for Bootstrap CI (OOB)
3. LR with StandardScaler for Bootstrap CI (OOB) — for comparison
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("Loading data...")
df_raw = pd.read_csv('coldstart_dataset_260304.csv', low_memory=False)
df = df_raw[df_raw['IS_HAS_FRAUD'] != 1].copy()
print(f"Users: {len(df):,}")

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

N_SAMPLE = 50_000
np.random.seed(42)
sample_idx = np.random.choice(len(df), N_SAMPLE, replace=False)
df_sample = df.iloc[sample_idx].reset_index(drop=True)
print(f"Sample: {N_SAMPLE:,}")

# ============================================================
# 1. Top/Bottom 10% — RF with CV
# ============================================================
print("\n" + "=" * 60)
print("1. TOP/BOTTOM 10% — RF, CV-BASED")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for target_label, target_col in [('D7 Purchase', 'IS_D7_PURCHASE')]:
    X_dev_ua = df_sample[device_cols + ua_cols].values
    X_dev = df_sample[device_cols].values
    y_t = df_sample[target_col].values

    # RF Device + UA (CV)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    probs_rf = cross_val_predict(rf, X_dev_ua, y_t, cv=skf, method='predict_proba')[:, 1]
    top10 = np.percentile(probs_rf, 90)
    bot10 = np.percentile(probs_rf, 10)
    top_rate = y_t[probs_rf >= top10].mean() * 100
    bot_rate = y_t[probs_rf <= bot10].mean() * 100
    ratio = top_rate / bot_rate if bot_rate > 0 else float('inf')
    print(f'  RF Dev+UA: Top={top_rate:.1f}%, Bot={bot_rate:.1f}%, Ratio={ratio:.1f}x')

    # RF Device only (CV)
    rf2 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    probs_rf2 = cross_val_predict(rf2, X_dev, y_t, cv=skf, method='predict_proba')[:, 1]
    top10_2 = np.percentile(probs_rf2, 90)
    bot10_2 = np.percentile(probs_rf2, 10)
    top_rate_2 = y_t[probs_rf2 >= top10_2].mean() * 100
    bot_rate_2 = y_t[probs_rf2 <= bot10_2].mean() * 100
    ratio_2 = top_rate_2 / bot_rate_2 if bot_rate_2 > 0 else float('inf')
    print(f'  RF Dev only: Top={top_rate_2:.1f}%, Bot={bot_rate_2:.1f}%, Ratio={ratio_2:.1f}x')

    # LR with StandardScaler Device + UA (CV)
    pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, random_state=42))])
    probs_lr = cross_val_predict(pipe, X_dev_ua, y_t, cv=skf, method='predict_proba')[:, 1]
    top10_lr = np.percentile(probs_lr, 90)
    bot10_lr = np.percentile(probs_lr, 10)
    top_rate_lr = y_t[probs_lr >= top10_lr].mean() * 100
    bot_rate_lr = y_t[probs_lr <= bot10_lr].mean() * 100
    ratio_lr = top_rate_lr / bot_rate_lr if bot_rate_lr > 0 else float('inf')
    print(f'  LR+Scaler Dev+UA: Top={top_rate_lr:.1f}%, Bot={bot_rate_lr:.1f}%, Ratio={ratio_lr:.1f}x')

# ============================================================
# 2. Bootstrap CI — RF, OOB
# ============================================================
print("\n" + "=" * 60)
print("2. BOOTSTRAP CI — RF, OOB")
print("=" * 60)

n_bootstrap = 100  # RF is slower, use 100
lifts_rf = []
y_d7 = df_sample['IS_D7_PURCHASE'].values
N = len(df_sample)

for i in range(n_bootstrap):
    idx = np.random.choice(N, N, replace=True)
    oob = np.setdiff1d(np.arange(N), np.unique(idx))
    if len(oob) < 500:
        continue

    y_train = y_d7[idx]
    y_oob = y_d7[oob]

    # Device only — RF
    X_dev_train = df_sample.iloc[idx][device_cols].values
    X_dev_oob = df_sample.iloc[oob][device_cols].values
    rf_dev = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_dev.fit(X_dev_train, y_train)
    auc_dev = roc_auc_score(y_oob, rf_dev.predict_proba(X_dev_oob)[:, 1])

    # Device + UA — RF
    X_devua_train = df_sample.iloc[idx][device_cols + ua_cols].values
    X_devua_oob = df_sample.iloc[oob][device_cols + ua_cols].values
    rf_ua = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_ua.fit(X_devua_train, y_train)
    auc_ua = roc_auc_score(y_oob, rf_ua.predict_proba(X_devua_oob)[:, 1])

    lifts_rf.append(auc_ua - auc_dev)
    if (i+1) % 20 == 0:
        print(f'  RF iter {i+1}/{n_bootstrap}: current mean lift = {np.mean(lifts_rf):.4f}')

lifts_rf = np.array(lifts_rf)
print(f'\nRF Bootstrap (n={len(lifts_rf)}):')
print(f'  Mean lift: {lifts_rf.mean():.4f}')
print(f'  95% CI: [{np.percentile(lifts_rf, 2.5):.4f}, {np.percentile(lifts_rf, 97.5):.4f}]')
print(f'  P(lift>0): {(lifts_rf > 0).mean()*100:.1f}%')

# ============================================================
# 3. Bootstrap CI — LR with StandardScaler, OOB
# ============================================================
print("\n" + "=" * 60)
print("3. BOOTSTRAP CI — LR+SCALER, OOB")
print("=" * 60)

lifts_lr = []
for i in range(200):
    idx = np.random.choice(N, N, replace=True)
    oob = np.setdiff1d(np.arange(N), np.unique(idx))
    if len(oob) < 500:
        continue

    y_train = y_d7[idx]
    y_oob = y_d7[oob]

    # Device only — LR+Scaler
    X_dev_train = df_sample.iloc[idx][device_cols].values
    X_dev_oob = df_sample.iloc[oob][device_cols].values
    sc1 = StandardScaler()
    lr_dev = LogisticRegression(max_iter=1000, random_state=42)
    lr_dev.fit(sc1.fit_transform(X_dev_train), y_train)
    auc_dev = roc_auc_score(y_oob, lr_dev.predict_proba(sc1.transform(X_dev_oob))[:, 1])

    # Device + UA — LR+Scaler
    X_devua_train = df_sample.iloc[idx][device_cols + ua_cols].values
    X_devua_oob = df_sample.iloc[oob][device_cols + ua_cols].values
    sc2 = StandardScaler()
    lr_ua = LogisticRegression(max_iter=1000, random_state=42)
    lr_ua.fit(sc2.fit_transform(X_devua_train), y_train)
    auc_ua = roc_auc_score(y_oob, lr_ua.predict_proba(sc2.transform(X_devua_oob))[:, 1])

    lifts_lr.append(auc_ua - auc_dev)

lifts_lr = np.array(lifts_lr)
print(f'LR+Scaler Bootstrap (n={len(lifts_lr)}):')
print(f'  Mean lift: {lifts_lr.mean():.4f}')
print(f'  95% CI: [{np.percentile(lifts_lr, 2.5):.4f}, {np.percentile(lifts_lr, 97.5):.4f}]')
print(f'  P(lift>0): {(lifts_lr > 0).mean()*100:.1f}%')

print(f'\nOLD (in-bag, no scaler): CI [0.049, 0.070]')
print("\n=== ALL DONE ===")
