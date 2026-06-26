# bKash Churn Prediction — NSU CEC Datathon

End-to-end churn prediction pipeline for a mobile wallet platform, built for the
**bKash x NSU CEC Datathon**. Goes beyond model accuracy — covers big-data ingestion,
feature engineering, class imbalance, hyperparameter tuning, and SHAP-based
explainability, finishing with business-ready recommendations.

**Result:** LightGBM model tuned via Optuna achieving **0.985 AUC** and **0.911
Precision@10%** on a held-out validation set of ~1 million accounts with a 12.68%
churn rate.

---

## The Problem

A 12.68% churn rate sounds manageable — until you realize that's roughly **108,000
customers** out of the dataset's 850,000 active wallet users leaving over the
observation window. A model that just predicts "never churn" would score 87.32%
accuracy while being completely useless for retention targeting.

The challenge: build a full pipeline — not just a model — that can identify
at-risk users early enough for the business to act, using only their transaction
and balance behavior.

## What This Project Covers

| Notebook | Phase | What's Inside |
|---|---|---|
| `01_data_loading.ipynb` | Big data handling | Polars lazy scanning, column pruning across 1M+ KYC, transaction, and balance rows |
| `02_feature_engineering.ipynb` | Feature engineering | 36 behavioral features across recency, frequency, monetary, balance, and velocity windows |
| `03_feature_quality.ipynb` | Distribution handling | Skewness analysis, winsorising, log1p transforms |
| `04_class_imbalance.ipynb` | Class imbalance | Churn rate quantification, stratified train/val split, imbalance-handling strategy |
| `05_model_training.ipynb` | Model training | LightGBM baseline — AUC, Avg Precision, Precision@K, Recall@K |
| `06_hyperparameter_tuning.ipynb` | Tuning | Optuna Bayesian optimization, 5-fold CV, convergence tracking |
| `07_explainability.ipynb` | Explainability | SHAP beeswarm + dependence plots, leakage audit |

## Dataset

- **~1,000,000 accounts** total (850K Customer / 100K Merchant / 50K Biller)
- **Transaction** and **DayEndBalance** tables at a scale too large to load eagerly —
  handled with Polars' lazy API rather than pandas
- **Churn rate: 12.68%** on the labeled training set

> The dataset is proprietary to the bKash x NSU CEC Datathon and is not included in
> this repository. See [`data/README.md`](data/README.md) for the expected folder
> structure if you want to run this pipeline yourself.

## Key Features

36 features were engineered across five behavioral dimensions — recency, frequency,
monetary volume, balance stability, and usage velocity, each at 7/14/30-day windows.
A few examples:

| Feature | Hypothesis |
|---|---|
| `recency` | Days since last transaction — longer gaps signal disengagement |
| `freq_last_7D` / `freq_last_14D` / `freq_last_30D` | Recent activity level across multiple time horizons captures both sudden drop-off and gradual decline |
| `bill_pay_ratio` | Bill payments anchor a customer to the platform |
| `balance_trend` | Declining balance relative to lifetime average signals cash drain before churn |
| `inactivity_multiplier` | How many multiples of a user's normal rhythm they've been dormant |
| `velocity_last_7d` / `velocity_last_14d` | Share of lifetime activity concentrated in the most recent window |

Full feature list and rationale in [`02_feature_engineering.ipynb`](02_feature_engineering.ipynb).

## Results

| Model | AUC | Avg Precision | Precision@10% | Recall@10% |
|---|---|---|---|---|
| LightGBM (baseline params) | 0.9847 | 0.9224 | 0.9113 | 0.7188 |
| LightGBM (Optuna-tuned) | 0.9848 | 0.9226 | 0.9110 | 0.7186 |
| LightGBM (tuned, 5-fold CV) | **0.9850** | — | — | — |

Hyperparameter tuning via Optuna (50 trials, TPE sampler) found a learning rate of
~0.043, 26 leaves, and a positive-class weight of ~1.44, converging to a marginal
but consistent improvement over default parameters — most of the model's strength
came from the feature set rather than tuning headroom.

### Top SHAP Drivers

![SHAP Summary](shap_summary.png)

| Rank | Feature | Mean \|SHAP\| | Direction |
|---|---|---|---|
| 1 | `freq_last_14D` | 0.985 | Low values (blue, right side) push toward churn |
| 2 | `freq_last_7D` | 0.678 | Same pattern at a shorter window — confirms recency of drop-off matters most |
| 3 | `recency` | 0.332 | Higher recency (longer gap since last activity) pushes toward churn |
| 4 | `balance_std_dev_last7D` | 0.328 | Low recent balance volatility associates with churn risk |
| 5 | `freq_last_30D` | 0.323 | Confirms the signal holds across the full month, not just short windows |

The strongest churn signal across the board is **recent transaction frequency**
collapsing — all three frequency-window features (`freq_last_7D/14D/30D`) rank in
the top 5, suggesting a short, sharp activity drop-off is the clearest early-warning
sign, more so than slow balance decline.

Dependence plots for the top two frequency features are in
[`shap_dependence_freq_last_7D.png`](shap_dependence_freq_last_7D.png) and
[`shap_dependence_freq_last_14D.png`](shap_dependence_freq_last_14D.png).

## Business Recommendation

With **91.1% precision at the top 10% risk decile**, the model lets the retention
team focus on a small, highly accurate slice of accounts rather than the full user
base — flagging the top 10% by predicted churn probability captures **71.9% of all
churners** (Recall@10%) while keeping false positives low.

**Suggested intervention, tied to the SHAP findings above:** trigger a re-engagement
nudge when `freq_last_14D` drops sharply relative to a user's `freq_last_30D`
baseline — this is the earliest and strongest signal in the model, appearing before
`recency` alone would flag the account as inactive.

## Setup

```bash
git clone https://github.com/<your-username>/bkash-churn-datathon.git
cd bkash-churn-datathon
pip install -r requirements.txt
```

Run notebooks in order from `01_data_loading.ipynb` through `07_explainability.ipynb`.
Each notebook saves its output (parquet/model files) for the next one to load, so
they can also be run independently if you already have the intermediate files.

## Tech Stack

Polars (lazy big-data handling) · LightGBM · Optuna (Bayesian hyperparameter tuning)
· SHAP (explainability) · scikit-learn

## Reflections & Future Improvements

- Frequency-window features dominate SHAP importance; a natural next step is
  modeling frequency as an explicit time series (e.g. survival analysis) rather
  than fixed 7/14/30-day snapshots
- Tuning gains over default parameters were marginal (+0.0001 AUC) — most model
  strength came from feature engineering, suggesting future effort is better spent
  on features (e.g. network/graph features from P2P transfers) than further tuning
- Production deployment would need drift monitoring, since frequency-based features
  are sensitive to seasonal usage shifts

---

Built in 5 days for the bKash x NSU CEC Datathon.
