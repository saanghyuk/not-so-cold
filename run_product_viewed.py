"""
Product.viewed analysis from v2 data (new_inapp_m10).
"""
import duckdb
import json
import pandas as pd
import numpy as np

print("Loading v2 data...")
result = duckdb.sql("""
    SELECT
        new_inapp_m10,
        CAST(IS_D7_PURCHASE AS INTEGER) as d7_purchase,
        CAST(IS_D7_CHURN AS INTEGER) as d7_churn,
        CAST(IS_D14_PURCHASE AS INTEGER) as d14_purchase,
        CAST(IS_D30_PURCHASE AS INTEGER) as d30_purchase,
        CAST(IS_D30_CHURN AS INTEGER) as d30_churn,
        CAST(IS_M10_CHURN AS INTEGER) as m10_churn,
        CAST(IS_HAS_FRAUD AS INTEGER) as is_fraud
    FROM 'coldstart_v2.parquet'
""").fetchdf()

# Filter fraud
result = result[result['is_fraud'] != 1].reset_index(drop=True)
n = len(result)
print(f"Users (fraud excluded): {n:,}")

def count_event(json_str, event_key):
    try:
        d = json.loads(json_str) if pd.notna(json_str) and json_str else {}
    except:
        return 0
    for k, v in d.items():
        if event_key in k:
            return len(v) if isinstance(v, list) else 1
    return 0

print("Counting individual events...")
result['pv'] = result['new_inapp_m10'].apply(lambda x: count_event(x, 'product.viewed'))
result['home'] = result['new_inapp_m10'].apply(lambda x: count_event(x, 'home.viewed'))
result['cart'] = result['new_inapp_m10'].apply(lambda x: count_event(x, 'addedtocart'))
result['page'] = result['new_inapp_m10'].apply(lambda x: count_event(x, 'page_view'))

# Event rates
print()
print("=== EVENT RATES (first 10 min) ===")
for name, col in [('product.viewed', 'pv'), ('home.viewed', 'home'),
                   ('page_view', 'page'), ('addedtocart', 'cart')]:
    has = (result[col] > 0).sum()
    print(f"  {name:>20}: {has:,} ({has/n*100:.1f}%) have 1+")

# Product.viewed analysis
print()
print("=== PRODUCT.VIEWED == 0 vs 1+ ===")
zero_pv = result['pv'] == 0
n_zero = zero_pv.sum()
print(f"  product.viewed == 0: {n_zero:,} ({n_zero/n*100:.1f}%)")
print(f"  product.viewed >= 1: {(~zero_pv).sum():,} ({(~zero_pv).sum()/n*100:.1f}%)")

print()
metrics = [
    ('D7 Purchase', 'd7_purchase'),
    ('D7 Churn', 'd7_churn'),
    ('M10 Churn', 'm10_churn'),
    ('D30 Purchase', 'd30_purchase'),
    ('D30 Churn', 'd30_churn'),
]

print(f"{'Metric':<15} {'Zero PV':>10} {'Has PV':>10} {'Diff/Ratio':>12}")
print("-" * 50)
for name, col in metrics:
    vz = result[zero_pv][col].mean() * 100
    va = result[~zero_pv][col].mean() * 100
    if 'Purchase' in name:
        diff = f"{va/vz:.1f}x"
    else:
        diff = f"{vz-va:+.1f}%p"
    print(f"  {name:<13} {vz:>9.1f}% {va:>9.1f}% {diff:>12}")

# Gradient
print()
print("=== GRADIENT BY PRODUCT.VIEWED COUNT ===")
print(f"{'Count':>8} {'Users':>10} {'Pct':>7} {'D7 Purch':>10} {'D7 Churn':>10}")
print("-" * 50)
bins = [(0, 0, '0'), (1, 1, '1'), (2, 3, '2-3'), (4, 5, '4-5'), (6, 10, '6-10'), (11, 9999, '11+')]
for low, high, label in bins:
    mask = (result['pv'] >= low) & (result['pv'] <= high)
    cnt = mask.sum()
    if cnt == 0:
        continue
    purch = result[mask]['d7_purchase'].mean() * 100
    churn = result[mask]['d7_churn'].mean() * 100
    print(f"  {label:>6} {cnt:>10,} {cnt/n*100:>6.1f}% {purch:>9.1f}% {churn:>9.1f}%")

# Among zero PV users, what else?
print()
print("=== ZERO PRODUCT.VIEWED USERS: other events ===")
zu = result[zero_pv]
print(f"  home.viewed 1+: {(zu['home']>0).sum():,} ({(zu['home']>0).mean()*100:.1f}%)")
print(f"  page_view 1+: {(zu['page']>0).sum():,} ({(zu['page']>0).mean()*100:.1f}%)")
print(f"  addedtocart 1+: {(zu['cart']>0).sum():,} ({(zu['cart']>0).mean()*100:.1f}%)")

# Total event count for zero PV users
def total_events(json_str):
    try:
        d = json.loads(json_str) if pd.notna(json_str) and json_str else {}
    except:
        return 0
    total = 0
    for k, v in d.items():
        total += len(v) if isinstance(v, list) else 1
    return total

zu_total = zu['new_inapp_m10'].apply(total_events)
print(f"  Total events mean: {zu_total.mean():.2f}")
print(f"  Total events == 1: {(zu_total==1).sum():,} ({(zu_total==1).mean()*100:.1f}%)")
print(f"  Total events == 0: {(zu_total==0).sum():,} ({(zu_total==0).mean()*100:.1f}%)")

print()
print("DONE")
