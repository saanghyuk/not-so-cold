# Not So Cold: How Ad Journeys Warm Up New User Personalization

## Research Framing

> **Updated**: 2026-03-02
>
> Author: Noel Son (Data Scientist, AB180 Inc.), Roi Nam (CEO, AB180 Inc.)

---

## The Story in 30 Seconds

Mobile apps spend money to acquire users, yet half of those users leave within a day and never return. To retain users, apps need to deliver personalized experiences — but they know nothing about new users because there is no behavioral data.

However, apps actually **already know something** about these users. Which ads they saw, what path they took, how long they deliberated before installing — this **ad journey data** exists before the user ever opens the app.

We show that this data can **distinguish high-purchase-probability users from low-purchase-probability users by a factor of 5.2x for purchases within 7 days of install, at the very moment the user first opens the app**. We then propose an RL system that enables seamless personalization from the moment of install. It relies on ad journey signals immediately after install and naturally transitions as in-app behavior accumulates.

---

## 1. Limitation: Half of Users Disappear Before Generating Any Revenue

There is a structural problem in the mobile app business.

Companies invest advertising spend to acquire users (User Acquisition). Yet **half of these acquired users permanently churn within a single day**.

| Time Since Install | Permanent Churn Rate | Implication |
|------|-----------|------|
| 10 minutes | 24.9% | 1 in 4 users permanently churn within 10 minutes |
| 6 hours | 44.1% | Nearly half permanently churn within 6 hours |
| 1 day | 50.3% | A majority permanently churn within a single day |

> (438,519 users observed, tracked for 60 days post-install. "Permanent churn" = no return activity from last event through D60)

There is an even more serious problem. **User behavior varies dramatically by acquisition channel** — and the direction is counterintuitive:

| Acquisition Channel | 7-Day Purchase Rate | 7-Day Churn Rate |
|----------|---------|----------|
| SA (Search Ads) | **18.6%** | 43.0% |
| Organic (Self-directed) | 13.4% | 43.6% |
| DA (Display Ads) | 11.2% | **51.2%** |

Users acquired through display ads (DA) have a **lower purchase rate and 7.7pp higher churn rate** than Organic users — who cost nothing to acquire. Even among ad-driven users, search ad (SA) users outperform Organic, while display users underperform Organic. **The advertising channel decisively determines user quality.**

Moreover, this churn is **determined within the first few minutes**. A full **24.9% of all users permanently churn within 10 minutes of install**, accounting for approximately 37% of all churn that occurs within the first 30 days — the remaining 63% is distributed across the 10-minute to 30-day window. The purchase rate of users who churn within 10 minutes is **2.9%**, approximately **1/9th** of users who remain active beyond 10 minutes (26.3%). **The first 10 minutes are effectively an irreversible fork.**

The solution is clear: **deliver a personalized initial experience** to reduce early churn. That personalization improves churn and conversion rates has been widely demonstrated in the mobile app domain (Huang & Ascarza, 2024; Li et al., 2011). However, existing research on personalization has been limited to scenarios where in-app behavioral data has already accumulated. And many users churn before that happens.

But personalization requires information about the user. **What does the app currently know about a new user?**

Only automatically collected device information and install time. This information cannot distinguish who will purchase:

| Information Type | Category | 7-Day Purchase Rate |
|----------|------|---------------------|
| OS | Android | 14.6% |
| | iOS | 17.6% |
| Manufacturer | Samsung | 14.6% |
| | Apple | 17.2% |
| Language | Korean (97.3%) | 15.9% |
| | English | 13.2% |
| Install Hour | 10–14h (highest) | 17.3% |
| | 02–06h (lowest) | 12.1% |

Regardless of whether we look at OS, manufacturer, language, or install hour, purchase rate differences are only 2–5pp. Even when all these variables are included in a model, the ability to distinguish purchasers from non-purchasers (AUC) is 0.545 — barely different from a coin flip (0.500).

As a result, **during the first 6 hours when 44% of users leave forever, the app has no choice but to show every user the same screen.**

---

## 2. What To Do: Personalize the First Experience Using Ad Journey Data

### Data That Already Exists but Goes Unused

**Before** a user installs the app, the company already possesses rich information about that user.

MMPs (Mobile Measurement Partners — attribution services such as Airbridge, AppsFlyer, and Adjust) record the user's **entire pre-install ad journey**:

- **Which ad channel** brought them (search ads? display? Instagram?)
- **How many times** they were exposed to ads before installing (once and immediately? 10 times after much deliberation?)
- **How quickly** they installed (within 1 hour of seeing the ad? after a week?)
- **Did they search for the brand**, or **search for the product**?

This data is already collected by every app, but it is **used solely for acquisition cost attribution and then discarded.** No one leverages it for post-install personalization.

### This Data Actually Distinguishes Users

We validated whether ad journey data is genuinely useful.

At install time (t=0), we compared a model using only device information versus a model augmented with ad journey data. **No in-app behavioral data was used whatsoever** — the prediction was made simultaneously as the user opened the app.

**Result: Actual purchase rates for top-10% vs. bottom-10% users as predicted by the "device info + ad journey" model**

(Comparing actual purchase rates between the top 10% of users the model predicted as "high purchase probability" and the bottom 10% predicted as "low purchase probability")

| Prediction Target | Top 10% Purchase Rate | Bottom 10% Purchase Rate | Ratio | AUC |
|----------|---------------|---------------|------|-----|
| Purchase within 10 min | 8.5% | 1.3% | **6.7x** | 0.658 |
| Purchase within 6 hours | 15.3% | 2.8% | **5.4x** | 0.625 |
| Purchase within 7 days | 22.8% | 4.4% | **5.2x** | 0.604 |

The same analysis using device information only:

| Prediction Target | Top 10% Purchase Rate | Bottom 10% Purchase Rate | Ratio | AUC |
|----------|---------------|---------------|------|-----|
| Purchase within 10 min | 4.7% | 4.0% | **1.2x** | 0.548 |
| Purchase within 6 hours | 10.2% | 9.0% | **1.1x** | 0.531 |
| Purchase within 7 days | 17.0% | 15.4% | **1.1x** | 0.545 |

> (Logistic Regression, 5-Fold Stratified CV, Bootstrap 1000-iteration test with 95% CI not including 0 — fully statistically significant)

With device information alone, the purchase rate difference between top and bottom is 1.1–1.2x — essentially no discrimination. **Adding the ad journey enables 5–7x discrimination.**

### What Exactly Does the Ad Journey Distinguish?

Let us examine intuitively how the model internally distinguishes users.

**Channel type**: Even among "ad-acquired users," behavior varies completely depending on which ad brought them.

| Acquisition Channel | 7-Day Purchase Rate | 7-Day Churn Rate | Core Events within 1 Hour of Install |
|----------|---------------------|-------------|-------------------------------|
| SA (Search Ads) | **18.6%** | 43.0% | **7.84 events** (+29%) |
| DA (Display Ads) | 11.2% | **51.2%** | 6.08 events (baseline) |

> "Core events" refers to the average number of key in-app events — product views, searches, wishlists, cart additions, etc. — during the first hour after install (Mann-Whitney U, p ≈ 0).

Search ad (SA) users are those who actively searched for the brand or product name, clicked on a search ad, and installed. Among SA users, brand search users have a 7-day purchase rate of 25.3%, and product search users have 22.5%. They generate an average of 7.84 core events in the first hour after install. Display ad (DA) users are those who clicked on a banner ad in a social media feed. In the same time frame, they generated 6.08 events — **29% less** activity.

Organic users fall in between. With a 7-day purchase rate of 13.4% and a 7-day churn rate of 43.6%, they perform better than DA but worse than SA. A comparison across all three channels:

| Acquisition Channel | 7-Day Purchase Rate | 7-Day Churn Rate | Core Events within 1 Hour |
|----------|-------------|-------------|----------------------|
| SA (Search Ads) | **18.6%** | 43.0% | **7.84 events** |
| Organic (Self-directed) | 13.4% | 43.6% | — |
| DA (Display Ads) | 11.2% | **51.2%** | 6.08 events |

Purchase rates differ by 1.66x and churn rates by 8.2pp. Showing the same first screen to all three segments is irrational.

**Search keyword intent**: Whether a search query exists at the time of the ad click, and **what was searched**, also provides a powerful signal.

| Search Intent | 7-Day Purchase Rate | vs. No-Keyword Users |
|----------|---------|-----------------|
| Brand search (e.g., "[App Name]") | **25.3%** | 1.66x |
| Product search (e.g., "[Product Category]") | **22.5%** | 1.48x |
| Promo search (e.g., "[App Name] discount") | **32.3%** | 2.13x |
| No keyword | 15.2% | Baseline |

Promo search users have a purchase rate of 32.3% — **more than double** that of no-keyword users. Simply knowing "what this user searched for" can dramatically increase first-screen personalization precision.

Beyond these, the model utilizes a total of 46 ad journey variables. Key variables by category include:

- **Channel information**: Whether the last touch was search vs. display, keyword presence, number of unique channels, channel diversity (Shannon entropy)
- **Temporal information**: Time from first ad exposure to install (latency), time from last touch to install (recency), total ad journey duration (touch window)
- **Behavioral density**: Total touch count, click-to-impression ratio, touches within 30 min/1 hour/24 hours before install, hourly touch density
- **Interaction patterns**: Single-touch install flag, type of first/last touch (click vs. impression), proportion of touches in last 24 hours

These variables combine to enable the model to distinguish the top and bottom 10% by a factor of 5–7x.

### At the Moment of Install, the Ad Journey Is the Only Available Signal

A user has opened the app for the first time. The app must decide what to show. **What does it know about this user right now?**

In-app behavioral data? **None.** The user has done nothing yet.

Device information? The app knows whether it is Android or iOS. But the 7-day purchase rate for Android users is 14.6% and for iOS is 17.6% — **effectively no difference.** The same holds for manufacturer, language, and time of day. Device information cannot distinguish purchasers from non-purchasers (AUC 0.545, nearly identical to random at 0.500).

**The ad journey is the only data source.** What ads this user saw, whether they came through a search, how quickly they installed — this is the only meaningfully predictive data available at the moment of install.

As time passes, in-app behavior begins to accumulate. Within 10 minutes there are a few clicks, within 1 hour there are product views and searches, and by 6 hours a substantial amount of behavioral data has been collected. In-app behavior naturally becomes the superior information source, and the role of the ad journey diminishes.

We confirmed this with data. We predicted **"Will this user purchase within 7 days?"** while varying the prediction time point and using **only data accumulated up to that point**. For example, the "10 minutes post-install" row uses only device info + ad journey + **10 minutes of in-app behavior** to predict 7-day purchase (Random Forest feature importance):

| Prediction Time | Data Used | Device Info | Ad Journey | In-App Behavior |
|----------|------------|----------|----------|----------|
| **10 min post-install** | Device + Ad Journey + 10 min in-app | 7.1% | **29.3%** | 63.6% |
| 30 min post-install | Device + Ad Journey + 30 min in-app | 3.9% | 16.9% | 79.3% |
| 1 hour post-install | Device + Ad Journey + 1 hour in-app | 3.0% | 11.6% | 85.4% |
| 6 hours post-install | Device + Ad Journey + 6 hours in-app | 1.8% | 7.4% | 90.8% |
| 7 days post-install | Device + Ad Journey + 7 days in-app | 0.5% | 2.3% | 97.2% |

**What this table reveals is that the optimal weighting continuously shifts over time.**

Immediately after install, 10 minutes of in-app behavior already accounts for 63.6% of predictive power, but these 10 minutes of behavior are only available **after** 10 minutes have elapsed — and by then, 24.9% of users have already permanently churned. In contrast, ad journey data (29.3%) and device information (7.1%) are **already available the moment the user opens the app, making them the only meaningful signals for immediate personalization**. As time passes and in-app behavior accumulates, the ad journey's weight naturally diminishes: 11.6% at 1 hour, 7.4% at 6 hours, 2.3% at 7 days.

We also confirmed this decay in terms of **model performance (AUC)**. At each time point, we compared the 30-day purchase prediction AUC between a model using "in-app data only" and a model using "in-app + ad journey":

| Prediction Time | AUC Lift from Adding Ad Journey |
|----------|--------------------------|
| Immediately post-install (10 min) | **+0.0106** |
| 30 min | +0.0087 |
| 1 hour | +0.0081 |
| 6 hours | +0.0052 |
| 1 day | +0.0021 (80% decline from install) |
| 7 days | +0.0011 (90% decline from install) |
| 30 days | +0.0000 (fully dissipated) |

The incremental AUC from the ad journey is largest immediately post-install (at the 10-minute mark), 80% of it disappears by day 1, and it reaches exactly 0 by day 30.

One important nuance: **In-app behavior already has higher predictive power than the ad journey at 10 minutes post-install** (in-app-only AUC 0.701 vs. ad-journey-only AUC 0.571). This does not render the ad journey unnecessary — combining both **lifts AUC at every time point**. The ad journey and in-app behavior contain different information and are complementary. The ad journey captures "**how** this user arrived," while in-app behavior captures "**what** they did after arriving."

This structure is also confirmed by the **prediction gap between Paid and Organic users**. Paid users, who have ad journey data, achieve an AUC of 0.765 at 10 minutes post-install, while Organic users reach only 0.731. However, as in-app data accumulates, this gap narrows rapidly and nearly disappears after 6 hours (0.877 vs. 0.850). Ad journey data **delivers its greatest value during the early period when in-app behavior is scarce**.

This dynamic transition is precisely **why an RL approach — rather than a static model — is necessary**.

Let us examine the limitations of static rules concretely. Suppose we create a fixed rule: "30% ad journey, 70% in-app." This rule assigns **too much weight to in-app data immediately after install** (when only 10 minutes of in-app data exist), and **too much weight to the ad journey after 7 days** (when 2.3% would be sufficient). It is suboptimal at every time point.

The more fundamental problem is that **even at the same time point, the meaning of ad journey signals varies entirely by user type**. Three pieces of EDA evidence support this:

1. **Key predictive variables differ by channel**: Random Forest feature importance analysis by channel reveals that for DA (display) users, recency (time elapsed since last ad exposure) is the top predictor, while for SA (search ad) users, SA_count (number of searches) ranks first. Even within the same ad journey data, **the relevant signal itself differs by user type**.

2. **The same variable operates in opposite directions across channels**: Taking latency (time from ad exposure to install) as an example, DA users exhibit an inverted-U pattern where purchase rate is lowest at medium latency (1–24 hours, 9.3%), while SA users show a monotonic increase with longer latency (15.9% to 22.5%). No single rule of "longer latency is better/worse" can be established.

3. **OS x channel interaction effects**: The 7-day purchase rate for Android x DA users is 9.4%, while for iOS x SA users it is 25.0% — a **2.66x** difference. Neither OS nor channel alone can explain this gap.

What these three pieces of evidence demonstrate is that the interpretation of ad journey data must fundamentally differ depending on the user's acquisition channel. For users who searched for a specific product via search ads, the ad journey signal directly reflects purchase intent and can be trusted longer. For users who incidentally clicked a display banner, the signal is weak and the system should quickly transition to in-app behavior.

RL automatically learns this user-specific, time-specific optimal weighting transition from data.

---

## 3. How To Make More Money: Seamless Personalization from Install

### Ultimate Goal

**Increase first conversion rate within 7 days of install.** In this study, we define conversion as purchase, though the definition can be adapted to subscriptions, bookings, content consumption, etc., depending on the app type. The rationale for a 7-day window is data-driven: **74.4% of users who convert within 30 days have already converted within the first 7 days, and among users who have not converted by day 7, only 6.3% convert by day 30**.

To achieve this, we propose an **RL (reinforcement learning) system that selects the most appropriate page for each user every time they open the app**. The RL **reward is purchase within 7 days of install**, but the **system itself operates continuously from the moment of install** — determining the optimal page at every app open. The same system can extend beyond 7 days to optimize for 14-day or 30-day purchases. However, this study sets 7-day post-install conversion as the core metric for validation.

### The Specific Decision a Manager Can Make

"A new user has opened the app. **Which page should we show them?**"

This question arises hundreds of thousands of times every day. Currently, all new users see the same home screen. Our system automatically personalizes this decision on a per-user basis.

```
[User A installs the app for the first time] — RL begins operating immediately

  What RL knows: Ad journey data (channel, latency, touch count...)
  What RL doesn't know: What this user likes (no in-app behavior yet)

  RL's judgment: "This user searched for a specific product and came via search ads.
              Purchase intent is high. Send them to that category."
  → Route to the bestseller page for that category


[Same user opens app again 3 hours later] — RL still operating

  What RL knows: Ad journey + "the user browsed a different category earlier"

  RL's judgment: "This user is interested in a different category than the one
              they arrived from. In-app behavior is starting to emerge,
              so let's increase its weight."
  → Route to a curated collection page for the explored category


[Same user opens app again 3 days later] — RL still operating

  What RL knows: Ad journey + 3 days of in-app behavior (wishlists, cart, view history)

  RL's judgment: "Sufficient in-app data has accumulated.
              The user's behavioral data is now far more accurate than the ad journey."
  → Route to a personalized page based on cart/wishlist items
```

The key point: **RL runs continuously from start to finish.** Only the weighting of the information it references changes over time.

- Immediately after install: Relies heavily on the ad journey (no other information available)
- As time passes: In-app behavior accumulates and the weight on the ad journey naturally decreases
- After several days: Nearly 100% in-app behavior-based — identical to conventional personalized recommendations

### Why RL

As demonstrated above, the optimal weighting between ad journey and in-app behavior continuously shifts over time (10 min: 29.3% → 1 hour: 11.6% → 7 days: 2.3%). Yet this weighting cannot be predetermined with fixed rules. There are two reasons.

**First, the reliability and validity period of ad journey signals differ by user type.** Examples confirmed through EDA:

- **The 7-day purchase rate for SA (search ad) users is 18.6%, while for DA (display) users it is 11.2%.** Since search ad users have clear purchase intent, the ad journey signal can be trusted longer. For DA users, the signal is weak and the system should quickly transition to in-app behavioral data.
- **Even at the same 10-minute mark, the amount of accumulated in-app data varies by user.** The average number of core events within 1 hour of install is 7.84 for SA users and 6.08 for DA users — a 29% difference. For highly active users, in-app data accumulates rapidly and ad journey weighting can be reduced quickly. For less active users, in-app data remains sparse and the system should rely on ad journey signals longer.

Since the optimal weighting differs entirely by user type even at the same time point, an average rule like "29.3% ad journey at 10 minutes post-install" is suboptimal for any individual user.

**Second, the relationships between individual variables are inherently nonlinear.** Consider latency (time from ad exposure to install):

| Latency Range | 7-Day Churn Rate | 7-Day Purchase Rate |
|-------------|----------|---------|
| < 1 hour (immediate install) | **51.6%** | 12.8% |
| 1–15 hours | 47.4% | 14.1% |
| 15–23 hours | 45.0% | 15.6% |
| **23–24 hours** | **42.7%** | **16.2%** |
| 24+ hours | 46.8% | 15.9% |

Intuitively, one would expect "faster install = better," but in reality, **users who install immediately have the highest churn rate** (51.6%). Users who deliberate for about a day show the best outcomes. In such nonlinear relationships, a simple rule like "shorter latency is better/worse" cannot be established. Furthermore, **even at the same latency, purchase rate patterns are opposite across channels**:

| Channel x Latency | < 1 hour (7-Day Purchase Rate) | 1–24 hours (7-Day Purchase Rate) | 24+ hours (7-Day Purchase Rate) |
|---------------|---------|---------|--------|
| DA (Display) | 12.8% | **9.3%** (lowest) | 14.0% |
| SA (Search Ads) | 15.9% | 19.1% | **22.5%** (highest) |

DA users exhibit a U-shaped pattern with the lowest purchase rate at 1–24 hours (9.3%), while SA users show a monotonically increasing pattern with the highest purchase rate at 24+ hours (22.5%). A single rule "longer latency is better/worse" cannot be formulated — the direction itself differs by channel. With dozens of variables — latency, channel, touch count, entropy, etc. — each exhibiting nonlinear behavior and interacting with one another, the combinatorial space explodes.

RL learns all of these patterns from data. It optimizes **"For this user, at this time point, which page maximizes the probability of purchase within 7 days of install?"** — at every app open. Changing the reward function to 14-day or 30-day purchase enables optimization over longer horizons, though this study targets 7-day post-install conversion.

### Operating Mechanism

```
Install ──→ RL begins ──→ RL continues ──→ RL continues ──→ Purchase!
       (Ad journey-based)  (Weight transition)  (In-app behavior-based)

Time:      t=0              ~1 hour later        ~7 days later
Reference: Ad journey 29.3%  Ad journey 11.6%   Ad journey 2.3%
           In-app 63.6%      In-app 85.4%       In-app 97.2%
```

### Ad Journey → 7-Day Purchase: The Causal Mechanism

Ad journey data does not **directly** increase purchases within 7 days of install. The mechanism is indirect, operating through a virtuous cycle:

```
1. Personalize the first experience using the ad journey
   (Based on ad journey data, RL selects the most appropriate
    first landing page for this user)
        ↓
2. The user stays instead of churning
   (A user who would have left upon seeing an irrelevant home screen
    now spends time browsing relevant products)
        ↓
3. In-app behavioral data accumulates
   (Clicks, views, searches, wishlists — data that reveals
    what this user likes)
        ↓
4. RL personalizes with increasing accuracy
   (As in-app data accumulates, RL shifts weight from ad journey
    to in-app behavior, improving recommendation accuracy)
        ↓
5. Purchase within 7 days of install
```

**The critical step is #2.** What the ad journey does is not directly predict "this user will purchase." The role of the ad journey is to **prevent early churn and push more users into the personalization funnel**. Specifically:

- Currently, for every 100 installs, 44 users leave permanently within 6 hours
- Only the remaining 56 accumulate in-app behavior, and only a fraction of those convert to purchases
- If ad journey-based personalization reduces early churn by just 10pp (from 44% to 34%), **users remaining in the funnel increase from 56 to 66** — an 18% increase
- These additional 10 users generate in-app data, RL personalizes more accurately, and some of them purchase

Users must stay for in-app data to accumulate, in-app data must accumulate for accurate personalization, and accurate personalization must occur for purchases to happen. **The ad journey is the starting point of this virtuous cycle.**

Without the ad journey? The only thing RL can reference is device information, whose predictive power is barely above a coin flip (AUC 0.545, random = 0.500). It is effectively recommending blind. While recommending blind, 44% of users permanently churn in the first 6 hours — by the time any personalization system activates, **nearly half the users are already lost.**

---

## 4. Why This Research Matters

### For Managers

1. **Immediately actionable**: Every app already collects this data through their MMP. Implementation is possible with existing infrastructure, no additional data collection required.

2. **Enables specific decisions**: Automates and personalizes the decision of "which page to show a new user" — a decision that occurs hundreds of thousands of times daily.

3. **Directly monetizable**: Currently, 44.1% of users permanently churn within 6 hours of install. Retaining even 10% of these users means tens of thousands of additional retained users, some of whom will convert to purchases. Given that customer acquisition cost (CAC) typically runs several dollars per user, this directly improves the return on acquisition spend already invested.

### For Academics

1. **A novel data source**: MMP multitouch data generates billions of records annually, yet has been virtually unexplored in academic research. Leveraging this data for cold-start personalization is itself a new approach.

2. **Temporal structure of information value**: We provide the first empirical evidence of a systematic decay pattern where ad journey information accounts for 29.3% of 7-day purchase prediction immediately post-install, declining to 2.3% after 7 days. This structure represents a dynamic characteristic missed by existing static feature selection approaches.

3. **A new application context for RL**: Dynamic optimization in an environment where the information source transitions over time (ad journey → in-app behavior). Existing marketing RL research has optimized actions with fixed information sources; we jointly learn the weighting of the information sources themselves.

### vs. Existing Literature

| Existing Research | Core Contribution | How This Research Differs |
|-----------|----------|----------------|
| Padilla & Ascarza (2021, JMR) | Using acquisition channel data to predict cold-start churn | They stop at "prediction." We connect prediction to **action (personalization)** and apply **dynamic optimization** via RL |
| Huang & Ascarza (2024, Marketing Science) | Using short-term behavioral signals for long-term targeting ("Doing More with Less") | Their "short-term signals" are early in-app behavior. We go one step further — starting **before any behavior occurs**, using only the ad journey |
| Ma, Huang, Ascarza & Israeli (2025, WP) | RL-based multi-signal dynamic personalization | Our methodological foundation. We add a **new information source (ad journey)** and introduce a structure where the **information source itself transitions over time** |
| Ascarza (2018, JMR) | Churn prediction-based targeting can be ineffective | Requires behavioral data. We solve the problem of starting at **t=0, when no behavioral data exists** |

---

## 5. One-Line Pitch

> **Existing personalization can only begin after in-app behavior accumulates. But apps already know which ads brought each user. By using this information to move the personalization start point to the moment of install, we can retain more users and convert them into purchasers during the critical first hours when half would otherwise leave.**

---

## Appendix: Key Numbers

| Metric | Value | Implication |
|------|------|------|
| Users analyzed | 385,025 | Partner mobile app, fraud excluded |
| Permanent churn within 6 hours | 44.1% | Nearly half leave forever within 6 hours |
| DA vs. Organic 7-day churn rate | 51.2% vs. 43.6% | Ad-acquired users churn more |
| SA vs. DA 7-day purchase rate | 18.6% vs. 11.2% | Knowing the channel alone yields 1.66x difference |
| Android x DA vs. iOS x SA 7-day purchase rate | 9.4% vs. 25.0% | Interaction effect: 2.66x |
| Ad journey top/bottom 10% 7-day purchase rate ratio | 5.2x | Distinguishable before the app is even opened |
| Keyword-bearing users 7-day purchase rate | 25.3% vs. 15.2% | 1.66x — keyword intent is a powerful signal |
| Ad journey information contribution at install | 29.3% | Primary information source for 7-day purchase prediction at install |
| Ad journey information contribution at 7 days | 2.3% | Replaced by in-app data |
| UA incremental AUC decay | 10 min +0.0106 → 7 days +0.0011 | 90% decline; fully dissipated at 30 days |
| Optimal latency range (23–24 hours) | Churn 42.7%, Purchase 16.2% | Lower churn than immediate install (51.6%) — nonlinear |
| 7-day purchase prediction AUC lift (LR) | +0.061 | Improvement over device-info-only baseline |
| 7-day purchase prediction AUC lift (RF) | +0.070 | Nonlinear model captures 16% more UA value |
| Statistical significance | 95% CI [+0.049, +0.070] | Bootstrap 200 iterations, P(lift>0) = 100% |
| Purchase rate of users who churn within 10 min | 2.9% vs. 26.3% | 1/9th of active user level |
| 74.4% of 30-day purchases occur within 7 days | 6.3% | Probability that non-purchasers at D7 convert by D30 |
| Fraud robustness | +0.059–0.061 | UA lift stable regardless of fraud inclusion/exclusion |

### Appendix: Purchase Rates by Search Keyword Intent

This analysis is limited to users with a search query at the time of ad click (approximately 5% of all users), but purchase behavior varies dramatically by keyword **intent**:

| Search Intent | Users | 7-Day Purchase Rate | vs. No-Keyword Users |
|----------|--------|---------|---------------|
| Brand search ("[App Name]") | 18,573 | **25.3%** | 1.66x |
| Product search ("oak desk") | 9,601 | **22.5%** | 1.45x |
| Promo search ("[App Name] discount") | 424 | **32.3%** | 2.06x |
| No keyword | — | 15.2% | Baseline |

Promo search users have a 7-day purchase rate of 32.3% — **more than double** that of no-keyword users. While current coverage (5%) is too low for inclusion in the primary analysis, this is a high-potential variable that could substantially improve personalization precision as MMP data expands.

---

### Appendix: The Paid vs. Organic Paradox — The Core Target for Personalization

> **Question: Do ad-acquired users and self-directed users behave differently?**

**Yes. And in the opposite direction from intuition.**

| Metric | Paid (Ad-acquired) | Organic (Self-directed) | Difference |
|------|----------------|-------------------|------|
| 7-day purchase rate | **15.8%** | 13.4% | Paid is +2.3pp higher |
| 7-day permanent churn rate | **46.5%** | 43.6% | Paid is +2.9pp higher |
| Permanent churn within 10 min | **25.1%** | 19.2% | Paid is **+5.9pp higher** |
| 30-day permanent churn rate | **64.7%** | 61.3% | Paid is +3.4pp higher |

Paid users have a **higher** purchase rate while also having a higher churn rate. Advertising acquires users who "have purchase intent but low loyalty." The largest gap of 5.9pp appears in **churn within 10 minutes of install** — demonstrating that Paid user churn is concentrated in the first few minutes after install.

**This is the core target for cold-start personalization**: Paid users are high-value if retained because they have purchase intent, but they must be retained quickly because their early churn is rapid. And these users have ad journey data available for personalization. Organic users lack ad journey data, making personalization more difficult, but their churn is relatively less urgent.

### Appendix: The Power of the First 10 Minutes — Both Predictor and Outcome

> **Question: Is behavior in the first 10 minutes a predictor variable or an outcome variable?**

**Both.** First-10-minute behavior is both a powerful predictor of long-term outcomes and simultaneously an outcome that we can **influence** through personalization.

| Model | Data Used | 7-Day Purchase AUC |
|------|-----------|------------|
| A | Device info only | 0.546 |
| B | Device + Ad journey | 0.600 **(+0.054)** |
| D | Device + 10-min in-app | 0.748 **(+0.202)** |
| C | Device + Ad journey + 10-min in-app | **0.757** **(+0.211)** |

First-10-minute in-app behavior (Model D) has **4x stronger** predictive power than the ad journey (Model B) (AUC +0.202 vs. +0.054).

**This is the causal mechanism hypothesis we propose** (to be further validated via field experiment):

1. Personalize the first screen using the ad journey (t=0)
2. The user sees relevant content and **engages more actively during the first 10 minutes**
3. This 10-minute activity strongly predicts long-term purchase (AUC 0.748)
4. In other words, the ad journey does not directly drive purchase — it creates a **richer 10-minute experience**, which in turn leads to purchase

We partially validated the causal relationship in step 2 (whether the 10-minute experience actually changes purchase behavior) using observational data. Using **Propensity Score Matching (PSM)**, we matched users with identical ad journey + device information (= pre-install intent proxy) and estimated the effect of 10-minute activity differences on purchase:

| Comparison | High-Activity Purchase Rate | Low-Activity Purchase Rate | Difference |
|------|-------------|-------------|------|
| Pre-matching (naive) | 22.0% | 16.8% | +5.2pp |
| **Post-matching (PSM)** | **21.8%** | **17.0%** | **+4.9pp** |

> (292,708 users surviving beyond 10 minutes, 1:1 nearest-neighbor matching, caliper 0.05, 40,164 matched pairs, all key variables SMD < 0.1 confirming balance, chi-squared = 302.2, p = 1.07e-67)

After matching, **93% of the naive gap persists** — selection (inherently likely purchasers being more active) explains only 7% of the gap. This supports a causal interpretation that the first 10-minute experience itself influences purchase. However, since unobserved variables (e.g., a user's immediate purchase need) cannot be fully ruled out, definitive causal confirmation requires a field experiment (random assignment of landing pages → changes in 10-minute activity → changes in purchase).

**Roles within the RL framework:**

- **t=0 (first app open)**: Ad journey data → RL state → first landing page decision (action). At this point, post-install 10-minute behavior is a **yet-to-exist future outcome**.
- **t=10 minutes onward**: 10 minutes of accumulated in-app behavior → added to RL state → more accurate personalization at the next app open. The first 10-minute behavior is now an **observed state input**.
- Key insight: RL's first action determines the quality of the first 10-minute behavior, and that behavior in turn becomes RL's next state. **Ad journey → first action → 10-minute behavior (outcome) → next state (input) → next action** — a virtuous cycle.

### Appendix: Depth of the Ad Journey — Touch Count, Channel Diversity, and User Intent

> **Question: Do more ad exposures and greater channel diversity produce better users?**

#### Purchase and Churn Patterns by Touch Count

Users were divided into 5 groups by the number of ad touchpoints (clicks, impressions):

| Touch Count | Users | 7-Day Purchase Rate | 7-Day Churn Rate |
|----------|--------|-------------|-------------|
| 1 (single touch) | 58,369 (18%) | 14.3% | 46.2% |
| 2–3 | 72,887 (22%) | 14.5% | **50.4%** |
| 4–10 | 65,864 (20%) | 16.1% | 49.2% |
| 11–30 | 46,641 (14%) | 16.8% | 46.0% |
| 31+ | 85,016 (26%) | **17.0%** | **41.7%** |

Two patterns emerge:

1. **Higher touch count correlates with higher purchase rate** (14.3% → 17.0%). This does not imply a causal relationship where "more ad exposure leads to purchase." Ad platform targeting algorithms expose more ads to users who are more likely to purchase. **What matters for us is not causality, but the fact that touch count can be leveraged for personalization** — whether the signal was created by platform targeting or by the user's voluntary exploration, touch count remains a valid predictive variable that reflects purchase probability.
2. **The 2–3 touch segment has the highest churn rate** (50.4%). This is counterintuitive. Interpretation: Single-touch users may have had strong intent from the start (searched and clicked immediately), while 2–3 touch users are likely "impulsive installers who saw the ad a couple of times and installed out of casual curiosity." Beyond 4 touches, the journey to install lengthens and intent solidifies.

Overall, 82% of Paid users are multitouch (2+). Most users install only after multiple exposures across multiple channels, and the **depth of this journey** itself reflects user intent.

#### Does Channel Diversity (Multi-channel) Affect Purchase?

We compared users who went through multiple **channels** (search ads, display, social media, etc.) versus single-channel users:

| User Type | Users | 7-Day Purchase Rate | 7-Day Churn Rate |
|----------|--------|-------------|-------------|
| Single channel | 162,918 | 15.9% | 46.2% |
| Multi-channel (2+) | 165,859 | 15.7% | 46.8% |

**The difference is negligible** (purchase rate 0.2pp, churn rate 0.6pp). **Which** channel matters far more than the **number** of channels (or their combination). A user who went through SA (search ads) alone has a 1.66x higher purchase rate than a user who went through 3 different DA (display) channels.

An interesting finding: While channel diversity (channel entropy) appears among the top 10 important features in Random Forest, it does not appear in Logistic Regression. This indicates that the effect of channel diversity manifests **only through nonlinear interactions with other variables** — for example, the same 2-channel combination means entirely different things depending on which channels are included. As confirmed earlier, the 7-day purchase rate for users who went through SA alone is 18.6% versus 11.2% for DA, and the OS x channel interaction shows iOS x SA (25.0%) vs. Android x DA (9.4%) — a 2.66x gap. It is the "quality of the combination," not the "number of channels," that matters. Capturing such complex interactions requires nonlinear learning models (including RL) rather than linear rules.

### Appendix: Coefficient Directions of Ad Journey Variables — Which Signals Work in Which Direction

Beyond prediction accuracy, understanding the **direction** in which each variable moves purchase probability is practically important. For this analysis, we examined the **standardized coefficients** from Logistic Regression.

#### Methodology

In the Logistic Regression, the dependent variable is purchase within 7 days of install (0 or 1), and the independent variables are 46 ad journey variables. All independent variables were **standardized using StandardScaler** (transformed to mean 0, standard deviation 1) before model training. The interpretation of standardized coefficients is as follows:

- **Sign (+/-)**: Whether a 1 standard deviation increase in that variable increases or decreases the purchase **log-odds**
- **Absolute value**: Larger values indicate greater influence on purchase probability
- **Purpose of standardization**: To compare variables on the same scale despite differing original units (time vs. count vs. ratio). A standardized coefficient of 0.113 means "a 1 standard deviation increase in this variable increases purchase log-odds by 0.113"

Validation: 5-Fold Stratified Cross-Validation, 50,000 random sample, fraud users excluded.

#### Variables That Increase Purchase Probability (+)

| Variable | Standardized Coefficient | Interpretation |
|------|-----------|------|
| SA touch count | +0.113 | More search ad clicks → higher purchase probability — users who actively searched and explored have higher purchase intent |
| Ad journey duration | +0.101 | Longer time from first ad exposure to install → higher purchase probability — users who deliberated at length before installing have firm intent |
| Recent 30-min touch proportion | +0.097 | Higher proportion of total touches concentrated in the 30 minutes before install → higher purchase probability — intensive exploration just before install signals strong purchase intent |
| Recent 30-min touch count | +0.087 | More absolute touches in the 30 minutes before install → higher purchase probability |

#### Variables That Decrease Purchase Probability (-)

| Variable | Standardized Coefficient | Interpretation |
|------|-----------|------|
| **Latency** | **-0.323** | Longer time from first ad exposure to install → lower purchase probability — **the single most powerful variable**. See note below |
| DA last touch | -0.131 | If the last ad touched before install was display (DA) → lower purchase probability — reflects DA users' lower purchase intent |
| Hourly touch density | -0.085 | Excessively high ad exposure frequency relative to time → lower purchase probability — **possible ad fatigue**. Bombarding the same user with too many ads in a short period can backfire |
| Single-touch install | -0.081 | Users who installed after only a single ad exposure → lower purchase probability — likely an impulsive install |

#### Three Notable Findings

**1. Latency: The nonlinear reality that linear models miss**

Logistic Regression estimates latency's coefficient at -0.323, implying "longer latency → lower purchase" as a linear relationship. However, looking at the actual data from the "Why RL" section above tells a different story:

| Latency Range | 7-Day Churn Rate |
|-------------|----------|
| < 1 hour (immediate install) | **51.6%** (worst) |
| 23–24 hours (one day of deliberation) | **42.7%** (best) |
| 24+ hours (extended delay) | 46.8% (worse again) |

The actual relationship is an **inverted-U**: both very fast and very slow installs produce poor outcomes. The best results come from **users who deliberated for about a day**. Immediate installers tend to be impulsive, while those who waited too long have lost interest.

Logistic Regression cannot capture this inverted-U. Linear models can only express "increases are good" or "increases are bad." This is one concrete reason why a model capable of automatically learning nonlinear relationships — such as RL — is necessary. Indeed, Random Forest (a nonlinear model) captures 16% more of the UA predictive contribution than LR on the same data (AUC lift: RF +0.070 vs. LR +0.061).

**2. Ad bombardment effect**: Higher hourly touch density correlates with lower purchase probability. Excessive ad exposure to the same user in a short time frame produces a negative effect. This has direct implications for ad campaign optimization.

**3. SA vs. DA direction is unambiguous**: SA touches are positive (+), DA last touch is negative (-). Whether a user **actively searched and arrived** versus **passively encountered an ad** is the fundamental discriminator of purchase behavior.

### Appendix: OS x Channel Interaction Effects — The Power of Variable Combinations

> **Question: What patterns emerge when we examine OS and ad channel simultaneously?**

**Interaction effects** exist that cannot be explained by any single variable (OS or channel) alone. Here, DA/SA refers to the user's **primary acquisition channel** (based on last touch):

| OS x Channel | 7-Day Churn Rate | 7-Day Purchase Rate |
|-----------|-------------|-------------|
| **Android x DA** | **54.7%** | **9.4%** |
| Android x SA | 43.6% | 17.2% |
| Android x Organic | 36.0% | 11.5% |
| iOS x DA | 40.4% | 16.9% |
| iOS x SA | 38.5% | **25.0%** |
| iOS x Organic | 41.5% | 16.0% |

**The worst combination is Android x DA** — churn rate 54.7%, purchase rate 9.4%. **The best combination is iOS x SA** — churn rate 38.5%, purchase rate 25.0%. The purchase rate gap is **2.66x**.

Notably, Android x Organic (zero ad spend) has the **lowest** churn rate at 36.0% across all combinations. Meanwhile, Android x DA (with ad spend) has the highest churn rate at 54.7%. Even among Android users, the churn rate differs by 18.7pp depending on which channel brought them.

What this interaction effect demonstrates: **Personalization must be based on variable combinations, not single variables.** Simple rules like "show A to Android users and B to iOS users" are insufficient. Capturing the complex patterns created by the **combination** of OS, channel, latency, touch count, and other variables requires nonlinear learning models like RL.

---
