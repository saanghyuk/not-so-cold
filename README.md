# Not So Cold: Leveraging Ad Journey Data for Cold-Start Personalization

Mobile apps face a cold-start problem: new users must be treated identically until enough in-app behavior accumulates. This research demonstrates that **pre-install ad journey data** — touchpoint sequences, channel types, creative exposures, and search keywords captured by Mobile Measurement Partners (MMPs) — can predict post-install outcomes (purchase, churn) **before the user takes any in-app action**.

## Key Findings

- **Ad journey features lift AUC by +0.067** over device-only baseline (0.555 → 0.622) for D7 purchase prediction
- **Top/Bottom decile ratio of 9.9×** — the highest-value 10% of users are 9.9× more likely to purchase than the lowest 10%, using only pre-install signals
- **Search Ad users show 37% higher revenue** and 34% higher AOV than Display Ad users
- **First 10 minutes of in-app data** push AUC to 0.723, but ad journey signals remain valuable for the critical window before any behavior is observed

## Repository Structure

```
.
├── research_framing.md              # Research framing document
├── feature_spec.md                  # Feature dictionary (438K users × 154 cols)
├── multitouch.ipynb                 # Core analysis: multitouch attribution modeling
├── analysis_test.ipynb              # EDA with survival curves, feature importance, decay
├── analysis_260304.ipynb            # v1 analysis notebook
├── analysis_v2.ipynb                # v2 analysis with event-level data
├── creative_analysis.ipynb          # Ad creative feature analysis
├── plot_channel_distribution.py     # Channel-wise engagement distribution plots
├── plot_channel_dist_v2.py          # Channel activity distribution (violin + hist)
├── run_*.py                         # Robustness checks and supplementary analyses
├── figures/                         # All generated figures
└── requirements.txt                 # Python dependencies
```

## Data

The dataset contains **438,519 users** with:
- **Device features**: OS, manufacturer, language, carrier, timezone
- **UA (User Acquisition) features**: touchpoint counts, timing, channel entropy, search keywords, creative text/image features
- **In-app behavioral features**: activity counts across 19 time windows (10min–30days)
- **Targets**: purchase and churn flags at multiple horizons

*Note: Raw data files (CSV/Parquet) are excluded from this repository.*

## Models

| Model | Features | AUC (5-fold CV) |
|-------|----------|-----------------|
| A: Device only | 700 one-hot dummies | 0.555 |
| B: Device + UA | 700 + 77 numeric | 0.622 |
| C: Device + UA + InApp (10min) | 700 + 77 + 6 | 0.723 |
| D: InApp only (10min) | 6 features | 0.703 |

- Algorithm: Random Forest (n_estimators=200, max_depth=10)
- Validation: 5-Fold Stratified CV

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook multitouch.ipynb
```

## Environment

- Python 3.8+
- Key packages: scikit-learn, pandas, numpy, matplotlib, seaborn
- See `requirements.txt` for full dependency list
