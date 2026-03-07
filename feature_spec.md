# Feature Specification — coldstart_dataset_260304.csv

> Dataset: 438,519 users × 154 columns
> Analysis subset: 385,025 users (IS_HAS_FRAUD == 0)
> Source: MMP + Snowflake

---

## 1. Identifiers & Meta (4 columns — excluded from modeling)

| Column | Description |
|--------|------------|
| USER_ID | Unique user identifier |
| IDFA | iOS Advertising ID |
| IDFV | iOS Vendor ID |
| GAID | Google Advertising ID |

---

## 2. Device Features (6 columns → 700 one-hot dummies)

Used as Model A baseline. Encoded via `pd.get_dummies(fillna('unknown'))`.

| Column | Description | Example values |
|--------|------------|----------------|
| OS_NAME | Operating system | android, iOS |
| DEVICE_LANGUAGE | Device language setting | ko (97.0%), en, ja |
| DEVICE_CARRIER | Telecom carrier | carrier_A, carrier_B, carrier_C |
| DEVICE_MANUFACTURER | Device manufacturer | samsung, Apple, LGE |
| DEVICE_TIMEZONE | Device timezone | e.g., UTC+9 |
| DEVICE_OSVERSION | OS version string | 14.0, 13.0, etc. |

---

## 3. UA (User Acquisition) Features (77 columns — numeric)

Core predictive features for cold-start personalization. Used in Model B (Device + UA).

### 3.1 Channel & Touch Counts (8)

| Column | Description |
|--------|------------|
| SA_count | Number of Search Ad touchpoints |
| DA_count | Number of Display Ad touchpoints |
| total_touch_count | Total touchpoint count (all types) |
| click_count | Total click touchpoints |
| impression_count | Total impression touchpoints |
| trackinglink_count | Tracking link touchpoints (manual URL setup) |
| unique_channel_count | Number of distinct channels in the journey |
| has_touchpoint | Binary: has any paid touchpoint (1=Paid, 0=Organic) |

### 3.2 Time Features (4)

| Column | Description | Unit |
|--------|------------|------|
| latency | First ad exposure → install | Seconds |
| recency | Last touchpoint → install | Seconds |
| touch_window | First touchpoint → last touchpoint | Seconds |
| INSTALL_TIMESTAMP | Install timestamp | Excluded from modeling |

### 3.3 Ratio & Density Features (11)

| Column | Description |
|--------|------------|
| click_ratio | click_count / total_touch_count |
| touch_count_ratio_0_30m | % of touches in first 30 min before install |
| touch_count_ratio_0_1h | % of touches in first 1 hour before install |
| touch_count_ratio_30m_1h | % of touches between 30min-1h before install |
| touch_count_ratio_1h_3h | % of touches between 1h-3h before install |
| touch_per_latency_hour | Touches per hour across latency period |
| touch_per_latency_day | Touches per day across latency period |
| touch_per_window_hour | Touches per hour across touch window |
| recent_24h_ratio | % of touches in last 24h before install |
| recent_24h_multiple | Ratio of last-24h touches to earlier touches |
| recent_touch_pressure | Touch density in the period right before install |

### 3.4 Recency Touch Counts (5)

| Column | Description |
|--------|------------|
| last30min_touch_count | Touches in last 30 min before install |
| last1h_touch_count | Touches in last 1 hour before install |
| last3h_touch_count | Touches in last 3 hours before install |
| last12h_touch_count | Touches in last 12 hours before install |
| last24h_touch_count | Touches in last 24 hours before install |

### 3.5 Touch Pattern (8)

| Column | Description |
|--------|------------|
| is_single_touch_install | Binary: installed after single touch |
| first_is_click | Binary: first touchpoint was a click |
| first_is_impression | Binary: first touchpoint was an impression |
| last_is_click | Binary: last touchpoint was a click |
| last_is_impression | Binary: last touchpoint was an impression |
| last_touch_is_da | Binary: last touch was Display Ad |
| last_touch_is_sa | Binary: last touch was Search Ad |
| last_touch_is_trackinglink | Binary: last touch was Tracking Link |

### 3.6 Channel & Last Touch (3)

| Column | Description |
|--------|------------|
| has_last_touch | Binary: has a recorded last touchpoint |
| channel_entropy | Shannon entropy of channel distribution |
| has_gm_touchpoint | Binary: has Google or Meta touchpoint |

### 3.7 Search Keywords (5)

| Column | Description |
|--------|------------|
| has_term | Binary: has search keyword |
| kw_brand_search | Binary: keyword is brand name search |
| kw_product_search | Binary: keyword is product search |
| kw_promo_season_search | Binary: keyword is promo/season search |
| term_total_count | Total keyword occurrences |
| term_unique_count | Unique keyword count |

Note: `keyword_list` (text column) and `media_type` are excluded from modeling.

### 3.8 Creative Text Features — LLM (8)

GPT-classified message types from ad creative OCR text. Binary (0/1).

| Column | Description |
|--------|------------|
| llm_brand_trust | "Brand trust" message detected |
| llm_call_to_action | "Call to action" message detected |
| llm_price_discount | "Price discount" message detected |
| llm_product_attribute | "Product attribute" message detected |
| llm_reward_benefit | "Reward/benefit" message detected |
| llm_social_proof | "Social proof" message detected |
| llm_target_specific | "Target specific" message detected |
| llm_urgency_scarcity | "Urgency/scarcity" message detected |

### 3.9 Creative Text Features — Rule-based (15)

Keyword-matching rules applied to OCR text. Binary (0/1).

| Column | Description |
|--------|------------|
| rule_amount_won | Contains monetary amount in local currency |
| rule_benefit_word | Contains benefit-related word |
| rule_black_friday | Contains "Black Friday" (local language) |
| rule_coupon | Contains "coupon" (local language) |
| rule_date_format | Contains date pattern |
| rule_discount_word | Contains "discount" (local language) |
| rule_first_come | Contains "first come first served" (local language) |
| rule_half_price | Contains "half price" (local language) |
| rule_lowest_price | Contains "lowest price" (local language) |
| rule_percent | Contains "%" |
| rule_sale | Contains "sale" (local language) |
| rule_sold_out | Contains "sold out" (local language) |
| rule_special_price | Contains "special price" (local language) |
| rule_today_only | Contains "today only" (local language) |
| rule_up_to | Contains "up to" (local language) |

### 3.10 Creative Metadata (4)

| Column | Description |
|--------|------------|
| has_any_creative_url | Binary: has any creative URL |
| has_broken_image | Binary: creative URL exists but image broken (catalog ads) |
| has_usable_creative | Binary: has usable (non-broken) creative |
| has_ocr_text | Binary: OCR text extracted from creative |

### 3.11 Install Time (6)

| Column | Description |
|--------|------------|
| is_installed_02_06 | Installed between 02:00-06:00 |
| is_installed_06_10 | Installed between 06:00-10:00 |
| is_installed_10_14 | Installed between 10:00-14:00 |
| is_installed_14_18 | Installed between 14:00-18:00 |
| is_installed_18_22 | Installed between 18:00-22:00 |
| is_installed_22_02 | Installed between 22:00-02:00 |

---

## 4. Creative Image Features (16 columns — excluded from UA model)

Excluded because they show no predictive value (correlation ±0.03, AUC contribution +0.001).

| Column | Prefix | Description |
|--------|--------|------------|
| creative_brightness_mean | creative_ | Mean brightness of ad image |
| creative_saturation_mean | creative_ | Mean saturation of ad image |
| creative_hue_mean | creative_ | Mean hue of ad image |
| creative_brightness_std | creative_ | Brightness standard deviation |
| creative_saturation_std | creative_ | Saturation standard deviation |
| creative_hue_std | creative_ | Hue standard deviation |
| creative_colorfulness | creative_ | Colorfulness metric |
| creative_symmetry_score | creative_ | Symmetry score |
| brightness_mean | (none) | Brightness mean (alt extraction) |
| saturation_mean | (none) | Saturation mean (alt extraction) |
| color_entropy | (none) | Color entropy |
| edge_density | (none) | Edge density |
| hue_cos | (none) | Hue cosine component |
| hue_sin | (none) | Hue sine component |
| symmetry_score | (none) | Symmetry score (alt extraction) |
| vertical_symmetry_score | (none) | Vertical symmetry score |

---

## 5. InApp Features (19 JSON columns → 5 numeric features per window)

Each `inapp_*` column contains a JSON string with 7 sub-fields. We use 5 features (excluding `purchase_engagement` for data leakage, and replacing `totalEventCount` with `adjusted_totalEventCount`).

### Time Windows (19)

| Column | Window |
|--------|--------|
| inapp_m10 | First 10 minutes |
| inapp_m30 | First 30 minutes |
| inapp_m60 | First 1 hour |
| inapp_m90 | First 1.5 hours |
| inapp_m120 | First 2 hours |
| inapp_m150 | First 2.5 hours |
| inapp_m180 | First 3 hours |
| inapp_m210 | First 3.5 hours |
| inapp_m240 | First 4 hours |
| inapp_m270 | First 4.5 hours |
| inapp_m300 | First 5 hours |
| inapp_m330 | First 5.5 hours |
| inapp_m360 | First 6 hours |
| inapp_d1 | First 1 day |
| inapp_d2 | First 2 days |
| inapp_d3 | First 3 days |
| inapp_d7 | First 7 days |
| inapp_d14 | First 14 days |
| inapp_d30 | First 30 days |

### JSON Sub-fields (7 → 5 used)

| Sub-field | Used? | Description |
|-----------|-------|------------|
| active | Yes | Active session count |
| core_engagement | Yes | Core events: product.viewed + page_view + home.viewed |
| deeplink_count | Yes | Deep link open count |
| open_count | Yes | App open count |
| totalEventCount | Modified | All event types combined |
| purchase_engagement | No | Purchase-related events — **excluded (data leakage)** |
| adjusted_totalEventCount | Yes | = totalEventCount - purchase_engagement (corr with D7 purchase: 0.109 vs 0.475 for raw total) |

---

## 6. Target Variables (42 columns)

### Churn (IS_*_CHURN) — 15 windows

"Permanent churn" = last activity before this window, never returns through D60.

| Column | Window |
|--------|--------|
| IS_M10_CHURN | 10 minutes |
| IS_M30_CHURN | 30 minutes |
| IS_M60_CHURN ... IS_M360_CHURN | 60-360 minutes (30-min intervals) |
| IS_D1_CHURN ... IS_D30_CHURN | 1-30 days |

### Purchase (IS_*_PURCHASE) — 15 windows

Binary: made at least one purchase within the window.

### InApp Targets (TARGET_*) — 12 columns

| Pattern | Description |
|---------|------------|
| TARGET_M60_INAPP, TARGET_M360_INAPP, TARGET_D1_INAPP | InApp activity target at 1h, 6h, 1d |
| TARGET_M60_TARGET, TARGET_M360_TARGET, TARGET_D1_TARGET | Engagement target at 1h, 6h, 1d |

### Fraud Flag

| Column | Description |
|--------|------------|
| IS_HAS_FRAUD | Binary: flagged as fraud (53,494 users excluded) |

---

## 7. LTV (1 JSON column)

| Column | Description |
|--------|------------|
| ltv | Nested JSON with two sub-objects: `purchase_LTV` and `ad_LTV` |

### JSON Structure

```json
{
  "purchase_LTV": {
    "M10_LTV": 627800, "M30_LTV": 627800, "M60_LTV": ...,
    "M90_LTV": ..., "M120_LTV": ..., "M150_LTV": ..., "M180_LTV": ...,
    "M210_LTV": ..., "M240_LTV": ..., "M270_LTV": ..., "M300_LTV": ...,
    "M330_LTV": ..., "M360_LTV": ...,
    "D1_LTV": ..., "D2_LTV": ..., "D3_LTV": ...,
    "D7_LTV": ..., "D14_LTV": ..., "D30_LTV": ...
  },
  "ad_LTV": {
    "M10_LTV": ..., "M30_LTV": ..., ...
  }
}
```

- `purchase_LTV`: Revenue from purchase/subscribe/order events (`%subscribe%`, `%order%`, `%purchase%`)
- `ad_LTV`: Revenue from in-app ad impressions (`%adimpression%`)
- Time windows match InApp windows (M10 through D30)
- Values are cumulative amounts in local currency

---

## 8. Other Columns (excluded from modeling)

| Column | Type | Reason for exclusion |
|--------|------|---------------------|
| media_type | Categorical | Redundant with channel features |
| keyword_list | Text | Raw keyword text, processed into kw_* features |
| ocr_text | Text | Raw OCR text, processed into llm_* and rule_* features |

---

## Model Specifications

### Model A: Device-only
- Features: 6 device columns → 700 one-hot dummies
- AUC (5-fold CV): 0.555

### Model B: Device + UA
- Features: 700 device dummies + 77 UA numeric features
- AUC (5-fold CV): 0.622
- UA lift: +0.067, Bootstrap 95% CI [0.051, 0.075] (100 iterations, RF OOB)
- Top/Bottom 10% (CV predictions): 26.1% / 2.6% = 9.9x

### Model C: Device + UA + InApp (m10)
- Features: 700 + 77 + 6 = 783 (leakage-fixed: purchase_engagement excluded, adjusted_totalEventCount added)
- AUC (5-fold CV): 0.723

### Model D: InApp-only (m10)
- Features: 6 InApp features (active, ad_engagement, core_engagement, deeplink_count, open_count, adjusted_totalEventCount)
- AUC (5-fold CV): 0.703

### Common Parameters
- Algorithm: Random Forest (n_estimators=200, max_depth=10, random_state=42)
- CV: 5-Fold Stratified, random_state=42
- Bootstrap: 100 iterations, 50K samples each, n_estimators=100
- LR: LogisticRegression (max_iter=1000, random_state=42), StandardScaler
