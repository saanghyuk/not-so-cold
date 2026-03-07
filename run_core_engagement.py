"""
Core engagement analysis: zero vs active users in first 10 minutes.
core_engagement = product.viewed + page_view + home.viewed
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

df_raw = pd.read_csv('coldstart_dataset_260304.csv', low_memory=False)
df = df_raw[df_raw['IS_HAS_FRAUD'] != 1].copy()

# Parse inapp_m10
col = 'inapp_m10'
parsed = df[col].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else {})
df['m10_core_engagement'] = parsed.apply(lambda d: d.get('core_engagement', 0))
df['m10_active'] = parsed.apply(lambda d: d.get('active', 0))
df['m10_total'] = parsed.apply(lambda d: d.get('totalEventCount', 0))
df['m10_purchase'] = parsed.apply(lambda d: d.get('purchase_engagement', 0))
df['m10_adjusted_total'] = df['m10_total'] - df['m10_purchase']

n = len(df)
print(f'Total users: {n:,}')
print()

# core_engagement = product.viewed + page_view + home.viewed
print('=== CORE ENGAGEMENT (product.viewed + page_view + home.viewed) ===')
zero_ce = df['m10_core_engagement'] == 0
n_zero = zero_ce.sum()
n_active = (~zero_ce).sum()

print(f'Zero core engagement (10min): {n_zero:,} ({n_zero/n*100:.1f}%)')
print(f'1+ core engagement (10min): {n_active:,} ({n_active/n*100:.1f}%)')
print()

# D7 purchase rate
pr_zero = df[zero_ce]['IS_D7_PURCHASE'].mean()*100
pr_active = df[~zero_ce]['IS_D7_PURCHASE'].mean()*100
print('D7 Purchase Rate:')
print(f'  Zero: {pr_zero:.1f}%')
print(f'  Active: {pr_active:.1f}%')
print(f'  Ratio: {pr_active / pr_zero:.1f}x')

# D7 churn rate
print()
ch_zero = df[zero_ce]['IS_D7_CHURN'].mean()*100
ch_active = df[~zero_ce]['IS_D7_CHURN'].mean()*100
print('D7 Churn Rate:')
print(f'  Zero: {ch_zero:.1f}%')
print(f'  Active: {ch_active:.1f}%')
print(f'  Diff: {ch_zero - ch_active:+.1f}%p')

# M10 churn
print()
m10_zero = df[zero_ce]['IS_M10_CHURN'].mean()*100
m10_active = df[~zero_ce]['IS_M10_CHURN'].mean()*100
print('M10 Churn Rate:')
print(f'  Zero: {m10_zero:.1f}%')
print(f'  Active: {m10_active:.1f}%')

# D30 churn
print()
d30_zero = df[zero_ce]['IS_D30_CHURN'].mean()*100
d30_active = df[~zero_ce]['IS_D30_CHURN'].mean()*100
print('D30 Churn Rate:')
print(f'  Zero: {d30_zero:.1f}%')
print(f'  Active: {d30_active:.1f}%')

# D14, D30 purchase
print()
print('D14 Purchase Rate:')
p14_z = df[zero_ce]['IS_D14_PURCHASE'].mean()*100
p14_a = df[~zero_ce]['IS_D14_PURCHASE'].mean()*100
print(f'  Zero: {p14_z:.1f}%')
print(f'  Active: {p14_a:.1f}%')
print(f'  Ratio: {p14_a/p14_z:.1f}x')

print()
print('D30 Purchase Rate:')
p30_z = df[zero_ce]['IS_D30_PURCHASE'].mean()*100
p30_a = df[~zero_ce]['IS_D30_PURCHASE'].mean()*100
print(f'  Zero: {p30_z:.1f}%')
print(f'  Active: {p30_a:.1f}%')
print(f'  Ratio: {p30_a/p30_z:.1f}x')

# Among zero core engagement, what do they have?
print()
print('=== AMONG ZERO CORE ENGAGEMENT USERS ===')
print(f'  m10_adjusted_total mean: {df[zero_ce]["m10_adjusted_total"].mean():.2f}')
zero_total = (df[zero_ce]['m10_adjusted_total'] == 0).sum()
print(f'  m10_adjusted_total == 0: {zero_total:,} ({zero_total/n_zero*100:.1f}%)')
print(f'  m10_active mean: {df[zero_ce]["m10_active"].mean():.2f}')
zero_active = (df[zero_ce]['m10_active'] == 0).sum()
print(f'  m10_active == 0: {zero_active:,} ({zero_active/n_zero*100:.1f}%)')

# Gradient
print()
print('=== GRADIENT BY CORE ENGAGEMENT COUNT ===')
print(f'{"Count":>8} {"Users":>10} {"Pct":>7} {"D7 Purch":>10} {"D7 Churn":>10}')
print('-' * 50)

bins = [(-0.5, 0.5, '0'), (0.5, 1.5, '1'), (1.5, 3.5, '2-3'),
        (3.5, 5.5, '4-5'), (5.5, 10.5, '6-10'), (10.5, 9999, '11+')]
for low, high, label in bins:
    mask = (df['m10_core_engagement'] > low) & (df['m10_core_engagement'] <= high)
    cnt = mask.sum()
    purch = df[mask]['IS_D7_PURCHASE'].mean() * 100
    churn = df[mask]['IS_D7_CHURN'].mean() * 100
    print(f'{label:>8} {cnt:>10,} {cnt/n*100:>6.1f}% {purch:>9.1f}% {churn:>9.1f}%')

print()
print('DONE')
