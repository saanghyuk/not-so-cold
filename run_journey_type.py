"""
Journey typology: classify users by touchpoint sequence pattern.
Uses 'type' field (SA/DA) from touchpoint_sequence JSON.
"""
import duckdb
import orjson
import pandas as pd
import numpy as np
from collections import Counter

def parse_json(s):
    if not s or s in ('[]', '{}', ''):
        return None
    try:
        return orjson.loads(s)
    except:
        return None

print("Loading data...")
df = duckdb.sql("""
    SELECT
        touchpoint_sequence,
        CAST(CAST(IS_D7_PURCHASE AS FLOAT) AS INT) as d7_purchase,
        CAST(CAST(IS_D7_CHURN AS FLOAT) AS INT) as d7_churn,
        CAST(CAST(IS_HAS_FRAUD AS FLOAT) AS INT) as is_fraud,
        CAST(CAST(last_touch_is_sa AS FLOAT) AS INT) as last_touch_is_sa,
        CAST(CAST(last_touch_is_da AS FLOAT) AS INT) as last_touch_is_da
    FROM 'coldstart_v2.parquet'
    WHERE CAST(CAST(IS_HAS_FRAUD AS FLOAT) AS INT) = 0
""").df()
n = len(df)
print(f"Users: {n:,}")

# Parse touchpoint_sequence and extract type sequence (SA/DA)
df['tp_parsed'] = df['touchpoint_sequence'].apply(parse_json)

def get_type_sequence(tp_list):
    """Extract ordered list of SA/DA types."""
    if not tp_list or not isinstance(tp_list, list):
        return []
    types = []
    for tp in tp_list:
        if isinstance(tp, dict):
            t = tp.get('type', None)
            if t:
                types.append(t)
    return types

df['type_seq'] = df['tp_parsed'].apply(get_type_sequence)
df['n_touches'] = df['type_seq'].apply(len)

# Check what types exist
type_counts = Counter()
for seq in df['type_seq']:
    type_counts.update(seq)
print(f"\nType values: {dict(type_counts)}")

# Classify journey type
def classify_journey(type_seq):
    if len(type_seq) == 0:
        return 'organic'

    unique_types = set(type_seq)
    has_sa = 'SA' in unique_types
    has_da = 'DA' in unique_types
    n = len(type_seq)

    # Single touch
    if n == 1:
        if type_seq[0] == 'SA':
            return 'single_SA'
        elif type_seq[0] == 'DA':
            return 'single_DA'
        else:
            return 'single_other'

    # Multi-touch, SA and DA both present
    if has_sa and has_da:
        # Find first SA and first DA positions
        first_sa = type_seq.index('SA')
        first_da = type_seq.index('DA')
        if first_sa < first_da:
            return 'SA_then_DA'
        else:
            return 'DA_then_SA'

    # Multi-touch, only SA
    if has_sa and not has_da:
        return 'repeat_SA'

    # Multi-touch, only DA
    if has_da and not has_sa:
        return 'repeat_DA'

    # Multi-touch, neither SA nor DA (e.g., trackinglink only)
    return 'repeat_other'

df['journey_type'] = df['type_seq'].apply(classify_journey)

# But wait - the original table had "SA_mixed" and "repeat_trackinglink"
# Let me check what "other" types exist
other_types = Counter()
for seq in df['type_seq']:
    for t in seq:
        if t not in ('SA', 'DA'):
            other_types[t] += 1
print(f"\nNon-SA/DA types: {dict(other_types.most_common(10))}")

# Reclassify with more granularity
def classify_journey_v2(type_seq):
    if len(type_seq) == 0:
        return 'organic'

    unique_types = set(type_seq)
    has_sa = 'SA' in unique_types
    has_da = 'DA' in unique_types
    n = len(type_seq)
    other_types = unique_types - {'SA', 'DA'}

    if n == 1:
        if type_seq[0] == 'SA':
            return 'single_SA'
        elif type_seq[0] == 'DA':
            return 'single_DA'
        else:
            return 'single_other'

    # Has both SA and DA
    if has_sa and has_da:
        first_sa = type_seq.index('SA')
        first_da = type_seq.index('DA')
        if first_sa < first_da:
            return 'SA_then_DA'
        else:
            return 'DA_then_SA'

    # SA only (possibly with other types like trackinglink)
    if has_sa and not has_da:
        if other_types:
            return 'SA_mixed'
        return 'repeat_SA'

    # DA only (possibly with other types)
    if has_da and not has_sa:
        if other_types:
            return 'DA_mixed'
        return 'repeat_DA'

    # No SA, no DA — e.g., trackinglink, affiliate
    return 'repeat_trackinglink'

df['journey_type'] = df['type_seq'].apply(classify_journey_v2)

print("\n=== JOURNEY TYPOLOGY ===")
print(f"{'Journey Type':<35} {'Users':>12} {'%':>7} {'D7 Purch':>10} {'D7 Churn':>10} {'Avg Touch':>10}")
print("-" * 87)

# Sort by purchase rate descending
for jtype in df['journey_type'].value_counts().index:
    mask = df['journey_type'] == jtype
    cnt = mask.sum()
    purch = df[mask]['d7_purchase'].mean() * 100
    churn = df[mask]['d7_churn'].mean() * 100
    avg_touch = df[mask]['n_touches'].mean()
    print(f"  {jtype:<33} {cnt:>10,} ({cnt/n*100:>5.1f}%) {purch:>9.1f}% {churn:>9.1f}% {avg_touch:>9.1f}")

# Also check: last_touch based channel classification
print("\n\n=== CHANNEL CLASSIFICATION BASIS ===")
print("For the 'SA/DA/Organic' table in the document:")
# Last-touch based
sa_lt = df['last_touch_is_sa'] == 1
da_lt = (df['last_touch_is_da'] == 1) & (df['last_touch_is_sa'] != 1)
org = (df['last_touch_is_sa'] != 1) & (df['last_touch_is_da'] != 1) & (df['n_touches'] == 0)

print(f"\nLast-touch based:")
print(f"  SA: {sa_lt.sum():,} ({sa_lt.mean()*100:.1f}%) - D7 purch: {df[sa_lt]['d7_purchase'].mean()*100:.1f}%, D7 churn: {df[sa_lt]['d7_churn'].mean()*100:.1f}%")
print(f"  DA: {da_lt.sum():,} ({da_lt.mean()*100:.1f}%) - D7 purch: {df[da_lt]['d7_purchase'].mean()*100:.1f}%, D7 churn: {df[da_lt]['d7_churn'].mean()*100:.1f}%")
print(f"  Organic: {org.sum():,} ({org.mean()*100:.1f}%) - D7 purch: {df[org]['d7_purchase'].mean()*100:.1f}%, D7 churn: {df[org]['d7_churn'].mean()*100:.1f}%")

# Journey type is based on the FULL sequence, not last touch
print("\n\n=== SA_then_DA vs DA_then_SA: DETAILED ===")
for jt in ['SA_then_DA', 'DA_then_SA']:
    mask = df['journey_type'] == jt
    sub = df[mask]
    print(f"\n{jt}: {mask.sum():,} users")
    print(f"  Avg touches: {sub['n_touches'].mean():.1f}")
    print(f"  Last touch is SA: {sub['last_touch_is_sa'].mean()*100:.1f}%")
    print(f"  Last touch is DA: {sub['last_touch_is_da'].mean()*100:.1f}%")
    # Show touch count distribution
    for low, high, label in [(2,2,'2'), (3,5,'3-5'), (6,10,'6-10'), (11,50,'11-50'), (51,9999,'51+')]:
        m2 = mask & (df['n_touches'] >= low) & (df['n_touches'] <= high)
        if m2.sum() > 0:
            print(f"  Touches {label}: {m2.sum():,} users, purch {df[m2]['d7_purchase'].mean()*100:.1f}%")

print("\nDONE")
