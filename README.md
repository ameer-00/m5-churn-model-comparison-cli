# compare_models — Petra Telecom Churn Model Comparison CLI

A production-quality command-line tool that runs the full 6-model churn prediction pipeline from Integration 5B. Trains classifiers via stratified cross-validation and saves a complete comparison report — metrics table, PR curves, calibration plots, and the best model — to disk.

---

## Installation

**Python 3.9+** is required.

```bash
# Clone the repository
git clone https://github.com/<your-username>/stretch-5b-compare-models.git
cd stretch-5b-compare-models

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
scikit-learn>=1.5,<1.10
pandas
numpy
matplotlib
joblib
```

---

## Usage

```
python compare_models.py --data-path PATH [OPTIONS]
```

### Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `--data-path` | `str` | ✅ Yes | — | Path to the input CSV dataset. Must contain `churned` column and the 8 numeric feature columns. |
| `--output-dir` | `str` | No | `./output` | Directory where all results and plots are saved. Created automatically if it does not exist. |
| `--n-folds` | `int` | No | `5` | Number of stratified cross-validation folds. |
| `--random-seed` | `int` | No | `42` | Random seed for reproducible splits and model training. |
| `--dry-run` | flag | No | `False` | Validates the data and prints the full pipeline configuration without training any models. |
| `--verbose` | flag | No | `False` | Enables DEBUG-level logging for detailed fold-by-fold diagnostics. |

---

## Example Commands

### Normal run (default settings)
```bash
python compare_models.py --data-path data/telecom_churn.csv
```

### Dry run — validate data and preview config before training
```bash
python compare_models.py --data-path data/telecom_churn.csv --dry-run
```

### Custom output directory and 10-fold CV
```bash
python compare_models.py \
  --data-path data/telecom_churn.csv \
  --output-dir ./results/experiment_01 \
  --n-folds 10 \
  --random-seed 99
```

### Verbose mode for debugging
```bash
python compare_models.py --data-path data/telecom_churn.csv --verbose
```

---

## Output Files

All artefacts are saved to `--output-dir` (default: `./output`):

| File | Description |
|---|---|
| `comparison_table.csv` | Mean ± std for 5 metrics × 6 models (CV results) |
| `experiment_log.csv` | Timestamped log row per model for experiment tracking |
| `pr_curves.png` | Precision-Recall curves for the top 3 models |
| `calibration.png` | Calibration curves for the top 3 models |
| `best_model.joblib` | Best model (by CV PR-AUC) serialised with joblib |
| `tree_vs_linear_disagreement.md` | Analysis of the test sample where RF and LR disagree most |

---

## Models Compared

| Name | Scaler | Estimator |
|---|---|---|
| `Dummy` | — | DummyClassifier (most_frequent) |
| `LR_default` | StandardScaler | LogisticRegression |
| `LR_balanced` | StandardScaler | LogisticRegression (class_weight=balanced) |
| `DT_depth5` | — | DecisionTreeClassifier (max_depth=5) |
| `RF_default` | — | RandomForestClassifier (100 trees, max_depth=10) |
| `RF_balanced` | — | RandomForestClassifier (balanced class weights) |

---

## Expected Dataset Schema

The input CSV must contain at minimum these columns:

```
tenure, monthly_charges, total_charges, num_support_calls,
senior_citizen, has_partner, has_dependents, contract_months, churned
```

---

## Sample Output

Running the script on `telecom_churn.csv` (4,500 rows, 16.4% churn rate):

```
      model  accuracy_mean  precision_mean  recall_mean  f1_mean  pr_auc_mean
      Dummy         0.8364          0.0000       0.0000   0.0000       0.1636
 LR_default         0.8542          0.7135       0.1952   0.3032       0.4656
LR_balanced         0.7075          0.3209       0.7028   0.4404       0.4630
  DT_depth5         0.8572          0.6444       0.2852   0.3935       0.4496
 RF_default         0.8611          0.7341       0.2394   0.3593       0.5137
RF_balanced         0.8086          0.4267       0.4720   0.4475       0.4707
```

**Best model:** `RF_default` (PR-AUC = 0.514)

---

## Project Structure

```
stretch-5b-compare-models/
├── compare_models.py        # Production CLI script
├── requirements.txt
├── README.md
└── output/                  # Generated artefacts (git-ignored)
    ├── comparison_table.csv
    ├── experiment_log.csv
    ├── pr_curves.png
    ├── calibration.png
    ├── best_model.joblib
    └── tree_vs_linear_disagreement.md
```

---

## License

Educational use only. See [LICENSE](LICENSE) for terms.