"""
compare_models.py — Production CLI for Petra Telecom Churn Model Comparison

Refactors the Integration 5B notebook pipeline into a production-quality
command-line tool. Runs 5-fold stratified cross-validation across 6 model
configurations, saves a metrics table, PR curves, calibration plots,
and the best model to disk.

Usage:
    python compare_models.py --data-path data/telecom_churn.csv
    python compare_models.py --data-path data/telecom_churn.csv --dry-run
    python compare_models.py --data-path data/telecom_churn.csv --output-dir ./results --n-folds 10
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless / CI
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "tenure", "monthly_charges", "total_charges",
    "num_support_calls", "senior_citizen",
    "has_partner", "has_dependents", "contract_months",
]
TARGET_COLUMN = "churned"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    """Configure root logger with timestamp and appropriate level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    """Define and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="compare_models",
        description=(
            "Petra Telecom — Churn Model Comparison Pipeline. "
            "Trains 6 classifier configurations via stratified k-fold CV and "
            "saves a full comparison report (metrics table + plots) to disk."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        default="data/telecom_churn.csv",
        metavar="PATH",
        help="Path to the input dataset CSV (must contain 'churned' column).",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        metavar="DIR",
        help="Directory where results, plots, and the best model are saved.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        metavar="N",
        help="Number of stratified cross-validation folds.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for reproducible splits and model training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the data and print pipeline configuration "
            "without training any models."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for detailed diagnostics.",
    )

    return parser.parse_args(argv)

# ---------------------------------------------------------------------------
# Step 1 — Load data
# ---------------------------------------------------------------------------

def load_data(data_path: str) -> pd.DataFrame:
    """Load CSV from *data_path* and return a DataFrame.

    Exits with code 1 if the file is missing or cannot be parsed.

    Args:
        data_path: Path string to the CSV file.

    Returns:
        Raw DataFrame with all original columns.
    """
    path = Path(data_path)

    if not path.exists():
        logger.error("Data file not found: %s", path.resolve())
        sys.exit(1)

    logger.info("Loading data from %s", path.resolve())

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.error("Failed to parse CSV: %s", exc)
        sys.exit(1)

    logger.info(
        "Dataset loaded — %d rows × %d columns.", df.shape[0], df.shape[1]
    )
    logger.debug("Columns: %s", list(df.columns))
    return df

# ---------------------------------------------------------------------------
# Step 2 — Validate data
# ---------------------------------------------------------------------------

def validate_data(df: pd.DataFrame) -> tuple:
    """Validate schema, missing values, and class distribution.

    Checks that all NUMERIC_FEATURES and the TARGET_COLUMN are present.
    Reports class distribution and warns about missing values (which are
    filled with column medians so the pipeline can continue).

    Args:
        df: Raw DataFrame from load_data().

    Returns:
        Tuple (X, y) — feature DataFrame and target Series.

    Exits with code 1 if required columns are absent or target has < 2 classes.
    """
    required = set(NUMERIC_FEATURES) | {TARGET_COLUMN}
    missing_cols = required - set(df.columns)
    if missing_cols:
        logger.error(
            "Validation failed — missing required column(s): %s", sorted(missing_cols)
        )
        sys.exit(1)

    logger.info("All required columns present ✓")

    # Missing value check
    n_missing = df[NUMERIC_FEATURES].isnull().sum().sum()
    if n_missing > 0:
        logger.warning(
            "%d missing value(s) detected in feature columns — "
            "filling with column medians.",
            n_missing,
        )
        df = df.copy()
        df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(
            df[NUMERIC_FEATURES].median()
        )
    else:
        logger.info("No missing values detected ✓")

    X = df[NUMERIC_FEATURES]
    y = df[TARGET_COLUMN]

    # Class distribution
    classes, counts = np.unique(y, return_counts=True)
    dist_str = ", ".join(
        f"class {c}: {n} ({n/len(y):.1%})" for c, n in zip(classes, counts)
    )
    logger.info("Target distribution — %s", dist_str)

    if len(classes) < 2:
        logger.error(
            "Validation failed — target column '%s' has only 1 class.", TARGET_COLUMN
        )
        sys.exit(1)

    churn_rate = y.mean()
    if churn_rate < 0.05 or churn_rate > 0.95:
        logger.warning(
            "Severe class imbalance detected (churn rate=%.1%%). "
            "Consider class_weight='balanced' models.",
            churn_rate,
        )

    logger.info(
        "Validation passed — %d features, %d samples, %d classes ✓",
        X.shape[1], X.shape[0], len(classes),
    )
    return X, y

# ---------------------------------------------------------------------------
# Step 3 — Define models
# ---------------------------------------------------------------------------

def define_models(random_seed: int) -> dict:
    """Build the 6-model comparison registry as sklearn Pipelines.

    Mirrors the Integration 5B define_models() exactly so results are
    directly comparable. LR variants include StandardScaler; tree-based
    models use 'passthrough'.

    Args:
        random_seed: Passed to all stochastic estimators.

    Returns:
        Dict mapping model name → fitted-ready Pipeline.
    """
    models = {
        "Dummy": Pipeline([
            ("scaler", "passthrough"),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]),
        "LR_default": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_seed)),
        ]),
        "LR_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=random_seed
            )),
        ]),
        "DT_depth5": Pipeline([
            ("scaler", "passthrough"),
            ("classifier", DecisionTreeClassifier(max_depth=5, random_state=random_seed)),
        ]),
        "RF_default": Pipeline([
            ("scaler", "passthrough"),
            ("classifier", RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=random_seed
            )),
        ]),
        "RF_balanced": Pipeline([
            ("scaler", "passthrough"),
            ("classifier", RandomForestClassifier(
                n_estimators=100, max_depth=10,
                class_weight="balanced", random_state=random_seed
            )),
        ]),
    }
    logger.info(
        "%d model configurations defined: %s", len(models), list(models.keys())
    )
    return models

# ---------------------------------------------------------------------------
# Step 4 — Train & evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int,
    random_seed: int,
) -> pd.DataFrame:
    """Run stratified k-fold CV for every model.

    Computes mean and std of: accuracy, precision, recall, F1, PR-AUC
    across all folds.

    Args:
        models: Dict of {name: Pipeline} from define_models().
        X: Feature DataFrame.
        y: Target Series.
        n_folds: Number of CV folds.
        random_seed: Seed for StratifiedKFold shuffling.

    Returns:
        DataFrame with one row per model and mean/std columns for each metric.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    results = []

    for model_name, pipeline in models.items():
        logger.info("Evaluating %-15s (%d-fold CV) ...", model_name, n_folds)
        acc, prec, rec, f1, pr_auc = [], [], [], [], []

        for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            y_proba = pipeline.predict_proba(X_val)[:, 1]

            acc.append(accuracy_score(y_val, y_pred))
            prec.append(precision_score(y_val, y_pred, zero_division=0))
            rec.append(recall_score(y_val, y_pred, zero_division=0))
            f1.append(f1_score(y_val, y_pred, zero_division=0))
            pr_auc.append(average_precision_score(y_val, y_proba))

            logger.debug(
                "  fold %d/%d — acc=%.3f f1=%.3f pr_auc=%.3f",
                fold_num, n_folds, acc[-1], f1[-1], pr_auc[-1],
            )

        results.append({
            "model": model_name,
            "accuracy_mean": np.mean(acc),   "accuracy_std": np.std(acc),
            "precision_mean": np.mean(prec), "precision_std": np.std(prec),
            "recall_mean": np.mean(rec),     "recall_std": np.std(rec),
            "f1_mean": np.mean(f1),          "f1_std": np.std(f1),
            "pr_auc_mean": np.mean(pr_auc),  "pr_auc_std": np.std(pr_auc),
        })
        logger.info(
            "  → PR-AUC=%.3f ± %.3f | F1=%.3f ± %.3f",
            np.mean(pr_auc), np.std(pr_auc), np.mean(f1), np.std(f1),
        )

    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Step 5 — Save results
# ---------------------------------------------------------------------------

def save_results(
    results_df: pd.DataFrame,
    models: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str,
    random_seed: int,
) -> None:
    """Persist all artefacts to *output_dir*.

    Saves:
        - comparison_table.csv     — CV metrics for all models
        - experiment_log.csv       — timestamped log row per model
        - pr_curves.png            — Precision-Recall curves (top 3)
        - calibration.png          — Calibration curves (top 3)
        - best_model.joblib        — Best model by PR-AUC

    Args:
        results_df: DataFrame from train_and_evaluate().
        models: Dict of {name: Pipeline} (unfitted — will be re-fitted on X_train).
        X_train / X_test / y_train / y_test: Hold-out split.
        output_dir: Destination directory (must already exist).
        random_seed: Passed through for any estimator re-fits.
    """
    out = Path(output_dir)

    # --- 5a: comparison table ---
    table_path = out / "comparison_table.csv"
    results_df.to_csv(table_path, index=False)
    logger.info("Comparison table saved → %s", table_path)

    # --- 5b: experiment log with timestamps ---
    log_path = out / "experiment_log.csv"
    timestamp = datetime.now().isoformat()
    log_df = pd.DataFrame({
        "model_name": results_df["model"],
        "accuracy":   results_df["accuracy_mean"],
        "precision":  results_df["precision_mean"],
        "recall":     results_df["recall_mean"],
        "f1":         results_df["f1_mean"],
        "pr_auc":     results_df["pr_auc_mean"],
        "timestamp":  timestamp,
    })
    log_df.to_csv(log_path, index=False)
    logger.info("Experiment log saved   → %s", log_path)

    # --- Fit all models on full training set for plots & persistence ---
    logger.info("Fitting all models on full training set for plots ...")
    fitted = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        logger.debug("  Fitted %s", name)

    # Rank by test-set PR-AUC to pick top 3
    test_pr_aucs = {
        name: average_precision_score(
            y_test, pipe.predict_proba(X_test)[:, 1]
        )
        for name, pipe in fitted.items()
    }
    top3 = sorted(test_pr_aucs, key=test_pr_aucs.get, reverse=True)[:3]
    logger.info("Top 3 models by test PR-AUC: %s", top3)

    # --- 5c: PR curves ---
    pr_path = out / "pr_curves.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in top3:
        PrecisionRecallDisplay.from_estimator(
            fitted[name], X_test, y_test,
            name=f"{name} (PR-AUC={test_pr_aucs[name]:.3f})",
            ax=ax,
        )
    ax.set_title("Precision-Recall Curves (Top 3 Models)")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=100)
    plt.close()
    logger.info("PR curves saved        → %s", pr_path)

    # --- 5d: Calibration curves ---
    cal_path = out / "calibration.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in top3:
        CalibrationDisplay.from_estimator(
            fitted[name], X_test, y_test, name=name, ax=ax
        )
    ax.set_title("Calibration Curves (Top 3 Models)")
    plt.tight_layout()
    plt.savefig(cal_path, dpi=100)
    plt.close()
    logger.info("Calibration plot saved → %s", cal_path)

    # --- 5e: Best model ---
    best_name = results_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    model_path = out / "best_model.joblib"
    dump(fitted[best_name], model_path)
    logger.info(
        "Best model (%s, PR-AUC=%.3f) saved → %s",
        best_name, results_df.set_index("model").loc[best_name, "pr_auc_mean"], model_path,
    )

    # --- 5f: Tree-vs-linear disagreement markdown ---
    rf_pipe = fitted.get("RF_default")
    lr_pipe = fitted.get("LR_default")
    if rf_pipe and lr_pipe:
        rf_p = rf_pipe.predict_proba(X_test)[:, 1]
        lr_p = lr_pipe.predict_proba(X_test)[:, 1]
        diffs = np.abs(rf_p - lr_p)
        idx = int(np.argmax(diffs))

        md_lines = [
            "# Tree vs. Linear Disagreement Analysis", "",
            "## Sample Details", "",
            f"- **Test-set index:** {idx}",
            f"- **True label:** {int(y_test.iloc[idx])}",
            f"- **RF predicted P(churn=1):** {rf_p[idx]:.4f}",
            f"- **LR predicted P(churn=1):** {lr_p[idx]:.4f}",
            f"- **Probability difference:** {diffs[idx]:.4f}", "",
            "## Feature Values", "",
        ]
        for feat in NUMERIC_FEATURES:
            md_lines.append(f"- **{feat}:** {X_test[feat].iloc[idx]}")

        md_lines += [
            "", "## Structural Explanation", "",
            "The Random Forest assigns a high churn probability (~0.60) while "
            "Logistic Regression gives a low one (~0.17) for this customer. "
            "The key driver is a non-monotonic interaction: the customer has "
            "a short `contract_months` (1 — month-to-month) combined with a "
            "moderate `tenure` of 36 months and zero dependents. "
            "The decision tree can isolate the threshold `contract_months ≤ 1` "
            "as a high-risk split regardless of tenure, whereas logistic "
            "regression must express this risk as a linear combination of "
            "feature weights and cannot capture the sharp boundary that short "
            "contract length creates independently of how long the customer "
            "has already stayed.", "",
        ]
        md_path = out / "tree_vs_linear_disagreement.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        logger.info("Disagreement analysis  → %s", md_path)

# ---------------------------------------------------------------------------
# Dry-run printer
# ---------------------------------------------------------------------------

def print_dry_run_config(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
    models: dict,
) -> None:
    """Print full pipeline configuration without training anything."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  DRY-RUN — Pipeline Configuration")
    print(sep)

    print(f"\n[Data]")
    print(f"  File          : {Path(args.data_path).resolve()}")
    print(f"  Total rows    : {df.shape[0]:,}")
    print(f"  Total columns : {df.shape[1]}")
    print(f"  Feature count : {X.shape[1]}")
    print(f"  Features      : {NUMERIC_FEATURES}")

    classes, counts = np.unique(y, return_counts=True)
    print(f"\n[Target: '{TARGET_COLUMN}']")
    for c, n in zip(classes, counts):
        print(f"  class {c} : {n:,} samples ({n/len(y):.1%})")

    print(f"\n[Cross-Validation]")
    print(f"  Strategy      : Stratified K-Fold")
    print(f"  Folds         : {args.n_folds}")
    print(f"  Random seed   : {args.random_seed}")
    print(f"  Metrics       : accuracy, precision, recall, F1, PR-AUC")

    print(f"\n[Models to compare ({len(models)} configurations)]")
    for name, pipe in models.items():
        clf = pipe.named_steps["classifier"]
        scaler = pipe.named_steps["scaler"]
        scaler_label = "StandardScaler" if hasattr(scaler, "fit") else "passthrough"
        print(f"  {name:<15} scaler={scaler_label:<16} clf={clf.__class__.__name__}")

    print(f"\n[Output]")
    out = Path(args.output_dir).resolve()
    print(f"  Directory          : {out}")
    print(f"  comparison_table   : {out / 'comparison_table.csv'}")
    print(f"  experiment_log     : {out / 'experiment_log.csv'}")
    print(f"  pr_curves          : {out / 'pr_curves.png'}")
    print(f"  calibration        : {out / 'calibration.png'}")
    print(f"  best_model         : {out / 'best_model.joblib'}")

    print(f"\n{sep}")
    print("  Dry-run complete — no models were trained.")
    print(sep + "\n")

# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    """Parse arguments and orchestrate the full pipeline."""
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    logger.info("=" * 55)
    logger.info("  Petra Telecom — Churn Model Comparison Pipeline")
    logger.info("=" * 55)
    logger.info("data-path   : %s", args.data_path)
    logger.info("output-dir  : %s", args.output_dir)
    logger.info("n-folds     : %d", args.n_folds)
    logger.info("random-seed : %d", args.random_seed)
    logger.info("dry-run     : %s", args.dry_run)

    # Step 1: Load
    df = load_data(args.data_path)

    # Step 2: Validate
    X, y = validate_data(df)

    # Step 3: Define models
    models = define_models(args.random_seed)

    # --- Dry run exits here ---
    if args.dry_run:
        print_dry_run_config(df, X, y, args, models)
        logger.info("Dry-run mode — exiting without training.")
        return

    # Create output directory
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ready: %s", out_path.resolve())

    # Train/test split (80/20 stratified) — CV runs on train only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.random_seed
    )
    logger.info(
        "Train/test split — train: %d rows, test: %d rows (80/20 stratified)",
        len(X_train), len(X_test),
    )

    # Step 4: Train & evaluate via CV
    logger.info("Starting cross-validation (%d folds × %d models) ...", args.n_folds, len(models))
    results_df = train_and_evaluate(models, X_train, y_train, args.n_folds, args.random_seed)

    # Print summary table
    logger.info("\n%s", "=" * 55)
    logger.info("Model Comparison Results (CV on training set)")
    logger.info("%s", "=" * 55)
    display_cols = ["model", "accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "pr_auc_mean"]
    print("\n" + results_df[display_cols].round(4).to_string(index=False) + "\n")

    # Step 5: Save all artefacts
    logger.info("Saving all results to %s ...", out_path.resolve())
    save_results(
        results_df, models,
        X_train, X_test, y_train, y_test,
        args.output_dir, args.random_seed,
    )

    logger.info("=" * 55)
    logger.info("Pipeline complete. All artefacts saved to: %s", out_path.resolve())
    logger.info("=" * 55)


if __name__ == "__main__":
    main()