# Not So Cold: How Ad Journeys Warm Up New User Personalization

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
