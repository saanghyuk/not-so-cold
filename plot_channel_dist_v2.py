"""
Channel-wise distribution of core_engagement (1h) — cleaner density plot in English.
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df_raw = pd.read_csv('coldstart_dataset_260304.csv', low_memory=False)
df = df_raw[df_raw['IS_HAS_FRAUD'] != 1].copy()

# Parse inapp_m60 for core_engagement
col = 'inapp_m60'
parsed = df[col].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else {})
df['core_engagement_1h'] = parsed.apply(lambda d: d.get('core_engagement', 0))

# Channel classification
df['channel'] = 'Organic'
df.loc[df['SA_count'] > 0, 'channel'] = 'SA'
df.loc[(df['SA_count'] == 0) & (df['DA_count'] > 0), 'channel'] = 'DA'
df.loc[(df['has_touchpoint'] == 1) & (df['SA_count'] == 0) & (df['DA_count'] == 0), 'channel'] = 'Other Paid'

# Filter to SA, DA, Organic only
df_plot = df[df['channel'].isin(['SA', 'DA', 'Organic'])].copy()

print("Stats:")
for ch in ['SA', 'Organic', 'DA']:
    s = df_plot[df_plot['channel'] == ch]['core_engagement_1h']
    print(f"  {ch}: n={len(s):,}, mean={s.mean():.2f}, median={s.median():.0f}, zero%={((s==0).sum()/len(s)*100):.1f}%")

# --- Overlaid density histogram ---
fig, ax = plt.subplots(figsize=(10, 6))

channels = {'SA': '#2196F3', 'Organic': '#4CAF50', 'DA': '#FF9800'}
bins = np.arange(0, 36, 1)

for ch, color in channels.items():
    data = df_plot[df_plot['channel'] == ch]['core_engagement_1h']
    n = len(data)
    mean_val = data.mean()
    ax.hist(data, bins=bins, color=color, alpha=0.45, density=True,
            label=f'{ch} (n={n:,}, mean={mean_val:.1f})', edgecolor='white', linewidth=0.3)
    # Add mean line
    ax.axvline(mean_val, color=color, linestyle='--', linewidth=2, alpha=0.8)

ax.set_xlabel('Core Engagement Events (first 1 hour)', fontsize=13)
ax.set_ylabel('Density', fontsize=13)
ax.set_title('Distribution of First-Hour App Activity by Acquisition Channel', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.set_xlim(-0.5, 35)
ax.grid(True, alpha=0.2)

# Annotate key insight
ax.annotate('SA users: fewer zeros,\nlonger right tail',
            xy=(12, 0.02), fontsize=10, style='italic', color='#555555')

plt.tight_layout()
plt.savefig('fig_channel_activity_dist.png', dpi=150, bbox_inches='tight')
print("\nSaved: fig_channel_activity_dist.png")

# --- Violin plot (better for showing shape) ---
fig2, ax2 = plt.subplots(figsize=(8, 6))

data_dict = {}
for ch in ['SA', 'Organic', 'DA']:
    data_dict[ch] = df_plot[df_plot['channel'] == ch]['core_engagement_1h'].values

# Cap at 95th percentile for visualization
cap = 30
data_capped = [np.clip(d, 0, cap) for d in data_dict.values()]
labels = list(data_dict.keys())
colors = ['#2196F3', '#4CAF50', '#FF9800']

parts = ax2.violinplot(data_capped, positions=[1, 2, 3], showmedians=True, showextrema=False)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.6)
parts['cmedians'].set_color('red')

# Add mean markers
means = [d.mean() for d in data_dict.values()]
ax2.scatter([1, 2, 3], means, color='red', marker='D', s=60, zorder=5, label='Mean')
for i, (m, label) in enumerate(zip(means, labels)):
    ax2.annotate(f'{m:.1f}', (i + 1.15, m), fontsize=11, color='red', fontweight='bold')

ax2.set_xticks([1, 2, 3])
full_labels = ['SA\n(Search Ad)', 'Organic\n(Direct)', 'DA\n(Display Ad)']
ax2.set_xticklabels(full_labels, fontsize=12)
ax2.set_ylabel('Core Engagement Events (1h)', fontsize=12)
ax2.set_title('First-Hour App Activity by Channel — SA Users Are More Active', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('fig_channel_activity_violin.png', dpi=150, bbox_inches='tight')
print("Saved: fig_channel_activity_violin.png")

print("\n=== DONE ===")
