# Not So Cold: How Ad Journeys Warm Up New User Personalization

## Abstract

Mobile apps face a critical challenge: the first few days after installation determine whether a user stays or churns, yet apps know nothing about new users at the moment of install. This "cold start" problem forces apps to provide the same onboarding experience to every new user, regardless of their underlying preferences and intent. We challenge the assumption that all new users are equally unknown. For users acquired through paid advertising — who comprise the majority of installs for most apps — a rich pre-install ad journey exists in Mobile Measurement Partner (MMP) data: which ads they saw, which channels they came through, what creatives they engaged with, and how long their journey took. This ad journey reflects users' self-selected preferences and intent, providing a "warm signal" at the coldest moment. We propose a **reinforcement learning framework** that leverages this pre-install ad journey as the starting point of personalization, learning optimal intervention policies (e.g., subscription offer timing, content recommendation, push notification strategy) that maximize long-term outcomes such as subscription conversion and retention. Our framework accommodates two distinct user types: paid users, whose ad journeys provide immediate personalization signals at install, and organic users, who enter as truly cold starts and must rely solely on device context until in-app behavior accumulates. We show that as in-app behavior builds over time, the informational value of ad journeys gradually diminishes, and the personalization performance of paid and organic users converges — identifying the precise window during which ad journey data serves as a critical bridge. Using data from real mobile app deployments, we demonstrate that incorporating ad journey information yields incrementally better personalization compared to using channel-level indicators alone or device context alone, and that the optimal intervention strategy differs systematically between paid and organic users. Our work introduces a novel perspective that connects the pre-install advertising world to post-install app experience as one continuous user journey, offering both a new data source and a practical framework for cold-start personalization.

---

## Data

- **dataset_purchase_retention_device.csv** (~155MB)
  - 438K users with device, UA, and in-app features
  - Targets: D3/D7 purchase, D3/D7 churn
  - Split: 83.8% paid users, 16.2% organic

- **dataset_purchase_retention.csv** (~140MB)
  - Alternative feature set

*Note: CSV files are ignored by git.*

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or with one command (macOS/Linux):

```bash
pip install -r requirements.txt && jupyter notebook
```

### 2. Run the Analysis

```bash
jupyter notebook multitouch.ipynb
```

Then click **multitouch.ipynb** and run all cells.

## Output

The notebook generates three visualizations:

- **incremental_value.png** — AUC comparison across feature sets
- **ua_feature_importance.png** — Top 15 UA features for D3 purchase prediction
- **cold_start_timeline.png** — Timeline showing ad journey bridge effect

## Project Structure

```
.
├── multitouch.ipynb                          # Main analysis notebook
├── dataset_purchase_retention.csv            # Original dataset
├── dataset_purchase_retention_device.csv     # Device-enriched dataset
├── requirements.txt                          # Python dependencies
├── .gitignore                                # Git ignore rules
└── README.md                                 # This file
```

## Technical Details

- **Models**: Random Forest (100 trees, max_depth=10)
- **Metrics**: AUC-ROC for imbalanced targets
- **Validation**: Stratified train/test split (80/20)
- **Feature Sets**:
  - Device only (baseline)
  - Device + UA (ad journey)
  - Device + InApp (first 10 min)
  - Device + UA + InApp (full)

## Environment

- Python 3.8+
- Jupyter Notebook
- See `requirements.txt` for package versions

---

*Last updated: 2025-02-17*
