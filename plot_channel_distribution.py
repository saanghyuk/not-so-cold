"""
Channel-wise distribution of core_engagement (1h) — SA vs DA vs Organic.
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df_raw = pd.read_csv('coldstart_dataset_260304.csv', low_memory=False)
df = df_raw[df_raw['IS_HAS_FRAUD'] != 1].copy()
print(f"Users: {len(df):,}")

# Parse inapp_m60 for core_engagement
col = 'inapp_m60'
parsed = df[col].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else {})
df['core_engagement_1h'] = parsed.apply(lambda d: d.get('core_engagement', 0))

# Channel classification
df['channel'] = 'Organic'
df.loc[df['SA_count'] > 0, 'channel'] = 'SA'
df.loc[(df['SA_count'] == 0) & (df['DA_count'] > 0), 'channel'] = 'DA'
# If has_touchpoint but neither SA nor DA
df.loc[(df['has_touchpoint'] == 1) & (df['SA_count'] == 0) & (df['DA_count'] == 0), 'channel'] = 'Other Paid'

print("\nChannel counts:")
print(df['channel'].value_counts())

print("\nCore engagement 1h stats by channel:")
for ch in ['SA', 'DA', 'Organic']:
    subset = df[df['channel'] == ch]['core_engagement_1h']
    print(f"  {ch}: mean={subset.mean():.2f}, median={subset.median():.1f}, "
          f"std={subset.std():.2f}, max={subset.max():.0f}, "
          f"zero_pct={((subset == 0).sum() / len(subset) * 100):.1f}%")

# --- Plot 1: Histogram (truncated at 95th percentile for visibility) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

channels = ['SA (Search Ad)', 'Organic', 'DA (Display Ad)']
channel_keys = ['SA', 'Organic', 'DA']
colors = ['#2196F3', '#4CAF50', '#FF9800']

# Get 95th percentile for x-axis limit
cap = df['core_engagement_1h'].quantile(0.95)
cap = max(cap, 30)
bins = np.arange(0, cap + 2, 1)

for i, (ch_label, ch_key, color) in enumerate(zip(channels, channel_keys, colors)):
    ax = axes[i]
    data = df[df['channel'] == ch_key]['core_engagement_1h']
    data_capped = data[data <= cap]

    ax.hist(data_capped, bins=bins, color=color, alpha=0.7, edgecolor='white', density=True)
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f}')
    ax.axvline(data.median(), color='darkred', linestyle=':', linewidth=2, label=f'Median: {data.median():.0f}')
    ax.set_title(f'{ch_label}\n(n={len(data):,})', fontsize=13)
    ax.set_xlabel('Core Events (1 Hour)', fontsize=11)
    ax.legend(fontsize=9)
    if i == 0:
        ax.set_ylabel('Density', fontsize=11)

fig.suptitle('Distribution of First-Hour Core Engagement by Channel', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_channel_core_engagement_dist.png', dpi=150, bbox_inches='tight')
print("\nSaved: figures/fig_channel_core_engagement_dist.png")

# --- Plot 2: Box plot comparison ---
fig2, ax2 = plt.subplots(figsize=(8, 6))

data_list = []
labels = []
for ch_key, ch_label, color in zip(channel_keys, channels, colors):
    data = df[df['channel'] == ch_key]['core_engagement_1h']
    data_list.append(data.values)
    labels.append(ch_label)

bp = ax2.boxplot(data_list, labels=labels, patch_artist=True, showfliers=False,
                 medianprops=dict(color='red', linewidth=2))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Add mean markers
means = [d.mean() for d in data_list]
ax2.scatter(range(1, 4), means, color='red', marker='D', s=80, zorder=5, label='Mean')

for i, m in enumerate(means):
    ax2.annotate(f'{m:.1f}', (i+1, m), textcoords="offset points", xytext=(15, 5), fontsize=11, color='red')

ax2.set_ylabel('Core Events (core_engagement)', fontsize=12)
ax2.set_title('First-Hour Core Engagement by Channel (Outliers Excluded)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
plt.tight_layout()
plt.savefig('figures/fig_channel_core_engagement_box.png', dpi=150, bbox_inches='tight')
print("Saved: figures/fig_channel_core_engagement_box.png")

# --- Plot 3: CDF comparison (on same axes) ---
fig3, ax3 = plt.subplots(figsize=(10, 6))

for ch_key, ch_label, color in zip(channel_keys, channels, colors):
    data = df[df['channel'] == ch_key]['core_engagement_1h'].values
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    # Subsample for speed
    step = max(len(sorted_data) // 5000, 1)
    ax3.plot(sorted_data[::step], cdf[::step], color=color, linewidth=2, label=ch_label)

ax3.set_xlim(0, 50)
ax3.set_xlabel('Core Events (core_engagement, 1 Hour)', fontsize=12)
ax3.set_ylabel('Cumulative Proportion (CDF)', fontsize=12)
ax3.set_title('Cumulative Distribution of Core Events by Channel — SA Users Are More Active', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Add reference lines
ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax3.text(48, 0.51, '50%', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('figures/fig_channel_core_engagement_cdf.png', dpi=150, bbox_inches='tight')
print("Saved: figures/fig_channel_core_engagement_cdf.png")

print("\n=== DONE ===")
