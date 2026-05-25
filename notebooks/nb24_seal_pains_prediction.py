# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anndata",
#     "duckdb==1.5.3",
#     "loguru==0.7.3",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==2.3.3",
#     "polars==1.40.1",
#     "python-dotenv",
#     "scikit-learn==1.8.0",
# ]
# ///

# ruff: noqa: N803, N806  # Allow uppercase X, X_train, X_test (ML convention)

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    from loguru import logger
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import (
        DEFAULT_DPI,
        METADATA_DB,
        PROCESSED_DATA_DIR,
    )
    from nb03_ss_profiles import load_profiles

    # Pre-Harmony parquet retains original CellProfiler feature names
    FEATSELECT_URL = "https://cellpainting-gallery.s3.amazonaws.com/cpg0042-chandrasekaran-jump/source_all/workspace/profiles_assembled/compound_no_source7/v1.0/profiles_var_mad_int_featselect.parquet"

    OUTPUT_DIR = PROCESSED_DATA_DIR / "pains-prediction"
    N_FOLDS = 5
    RANDOM_STATE = 42


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PAINS Prediction from Morphological Profiles

    Investigates whether Cell Painting morphological phenotypes can predict
    PAINS (Pan-Assay INterference compoundS) structural alerts.

    **Research question:** Do PAINS compounds produce distinctive morphological signatures?

    Uses scaffold splitting (via StratifiedGroupKFold on precomputed Murcko scaffolds)
    for realistic generalization assessment, compared against random stratified splits
    to quantify optimistic bias.

    **Models:**
    - HistGradientBoostingClassifier with balanced class weights
    - Baselines: dummy (stratified random), cell count only, molecular properties only (MW, LogP, TPSA)

    **Outputs:** `data/processed/pains-prediction/{dataset}/`
    """)
    return


@app.cell
def _(mo):
    dataset_dropdown = mo.ui.dropdown(
        options=[
            "compound_no_source7",
            "compound_DL_CPCNN_no_source7",
        ],
        value="compound_no_source7",
        label="Dataset",
    )
    sample_slider = mo.ui.slider(
        start=0.1,
        stop=1.0,
        step=0.1,
        value=0.1,
        label="Sample fraction",
    )
    skip_eda_toggle = mo.ui.switch(value=False, label="Skip EDA plots")
    mo.hstack(
        [dataset_dropdown, sample_slider, skip_eda_toggle],
        justify="start",
    )
    return dataset_dropdown, sample_slider, skip_eda_toggle


# =========================================================================
# Data loading
# =========================================================================


@app.function
def load_pains_labels_and_metadata() -> pd.DataFrame:
    """Load PAINS labels, SMILES, scaffolds, and properties from metadata database."""
    _con = duckdb.connect(str(METADATA_DB), read_only=True)
    _df = _con.execute("""
        SELECT
            Metadata_JCP2022 as JCP2022,
            Metadata_SMILES as SMILES,
            Metadata_MurckoScaffold as Scaffold,
            Metadata_HasPAINS as HasPAINS,
            Metadata_MW as MW,
            Metadata_LogP as LogP,
            Metadata_TPSA as TPSA,
            Metadata_median_cell_count as median_cell_count
        FROM compound_metadata
        WHERE Metadata_SMILES IS NOT NULL
          AND Metadata_ValidMol = TRUE
          AND Metadata_HasPAINS IS NOT NULL
    """).df()
    _con.close()
    _n_pains = _df["HasPAINS"].sum()
    logger.info(f"Loaded PAINS labels: {len(_df):,} compounds, {_n_pains:,} PAINS ({_n_pains / len(_df) * 100:.1f}%)")
    return _df


@app.function
def align_pains_data(
    adata,
    metadata_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align morphology features with PAINS labels by JCP2022 ID.

    Returns:
        X: Feature matrix (morphology profiles)
        y: Binary labels (HasPAINS)
        scaffolds: Murcko scaffold strings for scaffold splitting
        X_props: Simple properties (MW, LogP, TPSA) for baseline model
        X_cellcount: Median cell count (shape (n, 1)) for cell count baseline
        scaffold_ids: Integer group IDs for scaffold splitting
    """
    _profile_jcp = adata.obs["JCP2022"].values
    _profile_set = set(_profile_jcp)
    _metadata_set = set(metadata_df["JCP2022"])
    _common_jcp = sorted(_profile_set & _metadata_set)
    logger.info(f"Common compounds: {len(_common_jcp):,}")

    _profile_idx = {k: i for i, k in enumerate(_profile_jcp)}
    _metadata_idx = metadata_df.set_index("JCP2022")

    _feature_order = [_profile_idx[k] for k in _common_jcp]
    X = adata.X[_feature_order].astype(np.float32)
    y = _metadata_idx.loc[_common_jcp, "HasPAINS"].values.astype(np.int32)
    _scaffolds = _metadata_idx.loc[_common_jcp, "Scaffold"].values
    X_props = _metadata_idx.loc[_common_jcp, ["MW", "LogP", "TPSA"]].values.astype(np.float32)
    X_cellcount = _metadata_idx.loc[_common_jcp, "median_cell_count"].values.astype(np.float32).reshape(-1, 1)

    _feature_norms = np.linalg.norm(X, axis=1)
    _props_valid = ~np.isnan(X_props).any(axis=1)
    _cellcount_valid = ~np.isnan(X_cellcount).any(axis=1)
    _valid_mask = (_feature_norms > 0) & _props_valid & _cellcount_valid
    logger.info(f"Valid compounds: {_valid_mask.sum():,}")

    X = X[_valid_mask]
    y = y[_valid_mask]
    _scaffolds = _scaffolds[_valid_mask]
    X_props = X_props[_valid_mask]
    X_cellcount = X_cellcount[_valid_mask]

    _unique_scaffolds = list(set(_scaffolds))
    _scaffold_to_id = {s: i for i, s in enumerate(_unique_scaffolds)}
    _scaffold_ids = np.array([_scaffold_to_id[s] for s in _scaffolds])
    logger.info(f"Unique scaffolds in aligned data: {len(_unique_scaffolds):,}")

    return X, y, _scaffolds, X_props, X_cellcount, _scaffold_ids


# =========================================================================
# EDA
# =========================================================================


@app.function
def run_eda(
    y: np.ndarray,
    X: np.ndarray,
    X_props: np.ndarray,
    scaffolds: np.ndarray,
    output_dir: Path,
) -> tuple[dict, list[plt.Figure]]:
    """Run exploratory data analysis; return stats dict and list of figures."""
    _n_total = len(y)
    _n_pains = int(y.sum())
    _n_non_pains = _n_total - _n_pains
    _pct_pains = _n_pains / _n_total * 100

    _eda_stats: dict = {
        "n_total": _n_total,
        "n_pains": _n_pains,
        "n_non_pains": _n_non_pains,
        "pct_pains": float(_pct_pains),
    }
    _figures: list[plt.Figure] = []

    # 1. Class distribution
    _fig1, _axes1 = plt.subplots(1, 2, figsize=(10, 4))
    _ax = _axes1[0]
    _bars = _ax.bar(
        ["Non-PAINS", "PAINS"],
        [_n_non_pains, _n_pains],
        color=["#1f77b4", "#ff7f0e"],
    )
    _ax.set_ylabel("Count")
    _ax.set_title("A. Class Distribution")
    for _bar, _count in zip(_bars, [_n_non_pains, _n_pains]):
        _ax.text(
            _bar.get_x() + _bar.get_width() / 2,
            _bar.get_height(),
            f"{_count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _ax = _axes1[1]
    _ax.pie(
        [_n_non_pains, _n_pains],
        labels=["Non-PAINS", "PAINS"],
        autopct="%1.1f%%",
        colors=["#1f77b4", "#ff7f0e"],
    )
    _ax.set_title("B. Class Proportion")
    _fig1.tight_layout()
    _fig1.savefig(output_dir / "eda_class_distribution.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    _figures.append(_fig1)

    # 2. Most discriminative features (Cohen's d)
    _pains_mask = y == 1
    _non_pains_mask = y == 0
    _pains_means = X[_pains_mask].mean(axis=0)
    _non_pains_means = X[_non_pains_mask].mean(axis=0)
    _pooled_std = np.sqrt(
        (
            (X[_pains_mask].std(axis=0) ** 2) * (_pains_mask.sum() - 1)
            + (X[_non_pains_mask].std(axis=0) ** 2) * (_non_pains_mask.sum() - 1)
        )
        / (_n_total - 2)
    )
    _pooled_std = np.where(_pooled_std == 0, 1e-10, _pooled_std)
    _effect_sizes = np.abs(_pains_means - _non_pains_means) / _pooled_std
    _top_feature_idx = np.argsort(_effect_sizes)[-6:][::-1]
    _eda_stats["top_discriminative_features"] = [
        {"index": int(i), "effect_size": float(_effect_sizes[i])} for i in _top_feature_idx
    ]

    _fig2, _axes2 = plt.subplots(2, 3, figsize=(12, 8))
    _axes2_flat = _axes2.flatten()
    for _ax_idx, _feat_idx in enumerate(_top_feature_idx):
        _ax = _axes2_flat[_ax_idx]
        _ax.hist(X[_non_pains_mask, _feat_idx], bins=50, alpha=0.6, label="Non-PAINS", color="#1f77b4", density=True)
        _ax.hist(X[_pains_mask, _feat_idx], bins=50, alpha=0.6, label="PAINS", color="#ff7f0e", density=True)
        _ax.set_xlabel(f"Feature {_feat_idx}")
        _ax.set_ylabel("Density")
        _ax.set_title(f"d = {_effect_sizes[_feat_idx]:.2f}")
        if _ax_idx == 0:
            _ax.legend()
    _fig2.suptitle("Top 6 Discriminative Morphology Features (by effect size)")
    _fig2.tight_layout()
    _fig2.savefig(output_dir / "eda_feature_distributions.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    _figures.append(_fig2)

    # 3. Properties vs PAINS
    _prop_names = ["MW", "LogP", "TPSA"]
    _fig3, _axes3 = plt.subplots(1, 3, figsize=(12, 4))
    for _ax_idx, _prop_name in enumerate(_prop_names):
        _ax = _axes3[_ax_idx]
        _ax.hist(
            X_props[_non_pains_mask, _ax_idx], bins=50, alpha=0.6, label="Non-PAINS", color="#1f77b4", density=True
        )
        _ax.hist(X_props[_pains_mask, _ax_idx], bins=50, alpha=0.6, label="PAINS", color="#ff7f0e", density=True)
        _ax.set_xlabel(_prop_name)
        _ax.set_ylabel("Density")
        _ax.set_title(f"{_prop_name} Distribution")
        if _ax_idx == 0:
            _ax.legend()
    _fig3.suptitle("Molecular Properties by PAINS Status")
    _fig3.tight_layout()
    _fig3.savefig(output_dir / "eda_properties_distribution.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    _figures.append(_fig3)

    # 4. Scaffold analysis
    _pains_scaffolds = set(scaffolds[_pains_mask])
    _non_pains_scaffolds = set(scaffolds[_non_pains_mask])
    _shared_scaffolds = len(_pains_scaffolds & _non_pains_scaffolds)
    _eda_stats["scaffold_analysis"] = {
        "unique_scaffolds": int(len(set(scaffolds))),
        "pains_scaffolds": int(len(_pains_scaffolds)),
        "non_pains_scaffolds": int(len(_non_pains_scaffolds)),
        "shared_scaffolds": int(_shared_scaffolds),
    }

    return _eda_stats, _figures


# =========================================================================
# Model training and evaluation
# =========================================================================


@app.function
def evaluate_dummy_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate dummy baseline (stratified random prediction)."""
    _dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    _dummy.fit(np.zeros((len(y_train), 1)), y_train)
    _y_pred_proba = _dummy.predict_proba(np.zeros((len(y_test), 1)))[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, _y_pred_proba)),
        "avg_precision": float(average_precision_score(y_test, _y_pred_proba)),
    }


@app.function
def train_evaluate_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_weight: str | dict | None = None,
) -> tuple[dict, HistGradientBoostingClassifier]:
    """Train HistGradientBoostingClassifier and evaluate."""
    _clf = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    _clf.fit(X_train, y_train)
    _y_pred_proba = _clf.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, _y_pred_proba)),
        "avg_precision": float(average_precision_score(y_test, _y_pred_proba)),
    }, _clf


@app.function
def run_pains_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    scaffold_ids: np.ndarray,
    X_props: np.ndarray,
    X_cellcount: np.ndarray,
    split_type: str = "scaffold",
) -> dict:
    """Run cross-validation with specified split type.

    Args:
        X: Feature matrix (morphology profiles)
        y: Binary labels (HasPAINS)
        scaffold_ids: Integer scaffold group IDs for StratifiedGroupKFold
        X_props: Simple properties (MW, LogP, TPSA)
        X_cellcount: Median cell count (shape (n, 1))
        split_type: "scaffold" or "random"

    Returns:
        Dict with classification metrics across folds.
    """
    logger.info(f"Running {N_FOLDS}-fold CV with {split_type} split...")

    if split_type == "scaffold":
        _splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        _split_iter = list(_splitter.split(X, y, groups=scaffold_ids))
    else:
        _splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        _split_iter = list(_splitter.split(X, y))

    _morphology_metrics: dict[str, list[float]] = {"roc_auc": [], "avg_precision": []}
    _dummy_metrics: dict[str, list[float]] = {"roc_auc": [], "avg_precision": []}
    _cellcount_metrics: dict[str, list[float]] = {"roc_auc": [], "avg_precision": []}
    _props_metrics: dict[str, list[float]] = {"roc_auc": [], "avg_precision": []}
    _all_models: list[HistGradientBoostingClassifier] = []

    for _fold_idx, (_train_idx, _test_idx) in enumerate(_split_iter):
        logger.info(f"  Fold {_fold_idx + 1}/{N_FOLDS}: train={len(_train_idx)}, test={len(_test_idx)}")
        X_train, y_train = X[_train_idx], y[_train_idx]
        X_test, y_test = X[_test_idx], y[_test_idx]

        # Dummy
        _dummy_result = evaluate_dummy_baseline(y_train, y_test)
        for _k, _v in _dummy_result.items():
            _dummy_metrics[_k].append(_v)

        # Cell count
        _cc_result, _ = train_evaluate_classifier(
            X_cellcount[_train_idx],
            y_train,
            X_cellcount[_test_idx],
            y_test,
            "balanced",
        )
        for _k, _v in _cc_result.items():
            _cellcount_metrics[_k].append(_v)

        # Properties
        _props_result, _ = train_evaluate_classifier(
            X_props[_train_idx],
            y_train,
            X_props[_test_idx],
            y_test,
            "balanced",
        )
        for _k, _v in _props_result.items():
            _props_metrics[_k].append(_v)

        # Morphology
        _morph_result, _clf = train_evaluate_classifier(X_train, y_train, X_test, y_test, "balanced")
        for _k, _v in _morph_result.items():
            _morphology_metrics[_k].append(_v)
        _all_models.append(_clf)

    _results = {
        "morphology": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in _morphology_metrics.items()},
        "baseline_dummy": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in _dummy_metrics.items()},
        "baseline_cellcount": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in _cellcount_metrics.items()
        },
        "baseline_properties": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in _props_metrics.items()
        },
        "models": _all_models,
    }

    logger.info(
        f"  Morphology ROC-AUC: {_results['morphology']['roc_auc']['mean']:.3f} "
        f"+/- {_results['morphology']['roc_auc']['std']:.3f}"
    )
    return _results


# =========================================================================
# Visualization
# =========================================================================


@app.function
def plot_pains_comparison(
    random_results: dict,
    scaffold_results: dict,
) -> plt.Figure:
    """Create comparison plot: baselines vs models across split types."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: ROC-AUC comparison across all models (scaffold split)
    _ax = axes[0]
    _model_keys = ["baseline_dummy", "baseline_cellcount", "baseline_properties", "morphology"]
    _model_labels = ["Dummy", "Cell Count", "Properties", "Morphology"]
    _colors = ["#999999", "#2ca02c", "#ff7f0e", "#1f77b4"]

    _scaffold_roc = [scaffold_results[k]["roc_auc"]["mean"] for k in _model_keys]
    _scaffold_roc_std = [scaffold_results[k]["roc_auc"]["std"] for k in _model_keys]

    _bars = _ax.bar(_model_labels, _scaffold_roc, yerr=_scaffold_roc_std, capsize=3, color=_colors)
    _ax.set_ylabel("ROC-AUC")
    _ax.set_title("A. Model Comparison (Scaffold Split)")
    _ax.set_ylim(0, 1)
    for _bar, _val in zip(_bars, _scaffold_roc):
        _ax.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.02, f"{_val:.3f}", ha="center", fontsize=9)

    # Panel B: Random vs scaffold split (morphology only)
    _ax = axes[1]
    _metrics = ["roc_auc", "avg_precision"]
    _x = np.arange(len(_metrics))
    _width = 0.35

    _random_means = [random_results["morphology"][m]["mean"] for m in _metrics]
    _random_stds = [random_results["morphology"][m]["std"] for m in _metrics]
    _scaffold_means = [scaffold_results["morphology"][m]["mean"] for m in _metrics]
    _scaffold_stds = [scaffold_results["morphology"][m]["std"] for m in _metrics]

    _ax.bar(_x - _width / 2, _random_means, _width, yerr=_random_stds, label="Random Split", capsize=3, color="#1f77b4")
    _ax.bar(
        _x + _width / 2,
        _scaffold_means,
        _width,
        yerr=_scaffold_stds,
        label="Scaffold Split",
        capsize=3,
        color="#ff7f0e",
    )
    _ax.set_ylabel("Score")
    _ax.set_title("B. Generalization Gap (Morphology)")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(["ROC-AUC", "Avg Precision"])
    _ax.legend(loc="lower right")
    _ax.set_ylim(0, 1)

    _gap = random_results["morphology"]["roc_auc"]["mean"] - scaffold_results["morphology"]["roc_auc"]["mean"]
    _ax.annotate(
        f"Gap: {_gap:.2f}",
        xy=(0, _scaffold_means[0]),
        xytext=(0.5, 0.3),
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    fig.tight_layout()
    return fig


@app.function
def plot_roc_pr_curves(
    X: np.ndarray,
    y: np.ndarray,
    scaffold_ids: np.ndarray,
) -> plt.Figure:
    """Plot ROC and PR curves using a single train/test scaffold split."""
    _splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    _train_idx, _test_idx = next(_splitter.split(X, y, groups=scaffold_ids))

    _clf = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=0,
    )
    _clf.fit(X[_train_idx], y[_train_idx])
    _y_pred_proba = _clf.predict_proba(X[_test_idx])[:, 1]
    _y_test = y[_test_idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ROC
    _fpr, _tpr, _ = roc_curve(_y_test, _y_pred_proba)
    _roc_auc = roc_auc_score(_y_test, _y_pred_proba)
    axes[0].plot(_fpr, _tpr, color="#1f77b4", lw=2, label=f"Morphology (AUC = {_roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("A. ROC Curve (Scaffold Split)")
    axes[0].legend(loc="lower right")

    # PR
    _precision, _recall, _ = precision_recall_curve(_y_test, _y_pred_proba)
    _avg_precision = average_precision_score(_y_test, _y_pred_proba)
    _baseline_precision = _y_test.sum() / len(_y_test)
    axes[1].plot(_recall, _precision, color="#1f77b4", lw=2, label=f"Morphology (AP = {_avg_precision:.3f})")
    axes[1].axhline(y=_baseline_precision, color="gray", linestyle="--", label=f"Baseline ({_baseline_precision:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("B. Precision-Recall Curve (Scaffold Split)")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    return fig


@app.function
def get_feature_names() -> list[str]:
    """Fetch CellProfiler feature names from pre-Harmony parquet schema (no data download)."""
    _cols = pl.scan_parquet(FEATSELECT_URL).collect_schema().names()
    _feat_cols = [c for c in _cols if not c.startswith("Metadata_")]
    logger.info(f"Loaded {len(_feat_cols)} CellProfiler feature names from parquet schema")
    return _feat_cols


@app.function
def plot_feature_importance(
    models: list[HistGradientBoostingClassifier],
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    top_n: int = 20,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Compute and plot permutation-based feature importance."""
    _feat_names = get_feature_names()
    if len(_feat_names) != X.shape[1]:
        logger.info(f"Feature count mismatch ({len(_feat_names)} vs {X.shape[1]}), using generic names")
        _feat_names = [f"X_{i + 1}" for i in range(X.shape[1])]

    _model = models[0]
    _n_sample = min(5000, len(X))
    _rng = np.random.default_rng(RANDOM_STATE)
    _sample_idx = _rng.choice(len(X), size=_n_sample, replace=False)

    _result = permutation_importance(
        _model,
        X[_sample_idx],
        y[_sample_idx],
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    _importance_df = pd.DataFrame(
        {
            "feature_index": np.arange(len(_result.importances_mean)),
            "feature_name": _feat_names,
            "importance_mean": _result.importances_mean,
            "importance_std": _result.importances_std,
        }
    )
    _importance_df = _importance_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    _importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    _top = _importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    _y_pos = np.arange(len(_top))
    ax.barh(
        _y_pos,
        _top["importance_mean"],
        xerr=_top["importance_std"],
        align="center",
        color="#1f77b4",
        capsize=2,
    )
    ax.set_yticks(_y_pos)
    ax.set_yticklabels(_top["feature_name"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"Top {top_n} Morphology Features for PAINS Prediction")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig.savefig(output_dir / "feature_importance.pdf", bbox_inches="tight")

    return _importance_df, fig


# =========================================================================
# Summary
# =========================================================================


@app.function
def generate_pains_summary(
    random_results: dict,
    scaffold_results: dict,
    eda_stats: dict,
    n_compounds: int,
    n_pains: int,
    n_features: int,
) -> dict:
    """Generate summary dict with all results."""
    _roc_gap = random_results["morphology"]["roc_auc"]["mean"] - scaffold_results["morphology"]["roc_auc"]["mean"]
    _scaffold_roc = scaffold_results["morphology"]["roc_auc"]["mean"]
    _roc_overestimate_pct = (_roc_gap / (_scaffold_roc - 0.5)) * 100 if _scaffold_roc > 0.5 else 0

    _dummy_roc = scaffold_results["baseline_dummy"]["roc_auc"]["mean"]
    _cc_roc = scaffold_results["baseline_cellcount"]["roc_auc"]["mean"]
    _props_roc = scaffold_results["baseline_properties"]["roc_auc"]["mean"]

    _lift_dummy = _scaffold_roc - _dummy_roc
    _lift_cc = _scaffold_roc - _cc_roc
    _lift_props = _scaffold_roc - _props_roc

    if _scaffold_roc > 0.6:
        _conclusion = f"Morphology profiles show predictive signal for PAINS (ROC-AUC={_scaffold_roc:.3f})"
    elif _scaffold_roc > 0.55:
        _conclusion = f"Weak predictive signal for PAINS from morphology (ROC-AUC={_scaffold_roc:.3f})"
    else:
        _conclusion = f"Limited predictive signal for PAINS from morphology (ROC-AUC={_scaffold_roc:.3f})"

    return {
        "task": "PAINS prediction from morphology",
        "research_question": "Can morphological phenotypes predict PAINS structural alerts?",
        "n_compounds": n_compounds,
        "n_pains": n_pains,
        "pct_pains": float(n_pains / n_compounds * 100),
        "n_features": n_features,
        "n_folds": N_FOLDS,
        "model": "HistGradientBoostingClassifier",
        "eda": eda_stats,
        "baselines": {
            "dummy_roc_auc": float(_dummy_roc),
            "cellcount_roc_auc": float(_cc_roc),
            "properties_roc_auc": float(_props_roc),
        },
        "results": {
            "random_split": {
                "roc_auc": random_results["morphology"]["roc_auc"]["mean"],
                "roc_auc_std": random_results["morphology"]["roc_auc"]["std"],
                "avg_precision": random_results["morphology"]["avg_precision"]["mean"],
                "avg_precision_std": random_results["morphology"]["avg_precision"]["std"],
            },
            "scaffold_split": {
                "roc_auc": scaffold_results["morphology"]["roc_auc"]["mean"],
                "roc_auc_std": scaffold_results["morphology"]["roc_auc"]["std"],
                "avg_precision": scaffold_results["morphology"]["avg_precision"]["mean"],
                "avg_precision_std": scaffold_results["morphology"]["avg_precision"]["std"],
            },
        },
        "generalization_gap": {
            "roc_auc_drop": float(_roc_gap),
            "interpretation": f"Random split overestimates by {_roc_overestimate_pct:.0f}% of signal",
        },
        "lift_over_baselines": {
            "over_dummy": float(_lift_dummy),
            "over_cellcount": float(_lift_cc),
            "over_properties": float(_lift_props),
            "interpretation": (
                f"Morphology adds {_lift_cc:.3f} ROC-AUC over cell count, {_lift_props:.3f} over properties"
            ),
        },
        "conclusion": _conclusion,
    }


# =========================================================================
# Main pipeline cells
# =========================================================================


@app.cell
def _(dataset_dropdown, mo, sample_slider):
    _dataset = dataset_dropdown.value
    _sample_frac = sample_slider.value

    mo.stop(
        not METADATA_DB.exists(),
        mo.md(f"**Missing database:** `{METADATA_DB}` not found."),
    )

    # Load data
    adata = load_profiles(_dataset, level="perturbation")
    metadata_df = load_pains_labels_and_metadata()

    # Align
    X, y, scaffolds, X_props, X_cellcount, scaffold_ids = align_pains_data(adata, metadata_df)

    # Sample if requested
    if _sample_frac < 1.0:
        _n_sample = int(len(X) * _sample_frac)
        _rng = np.random.default_rng(RANDOM_STATE)
        _sample_idx = _rng.choice(len(X), size=_n_sample, replace=False)
        X = X[_sample_idx]
        y = y[_sample_idx]
        scaffolds = scaffolds[_sample_idx]
        X_props = X_props[_sample_idx]
        X_cellcount = X_cellcount[_sample_idx]
        _unique_sc = list(set(scaffolds))
        _sc_to_id = {s: i for i, s in enumerate(_unique_sc)}
        scaffold_ids = np.array([_sc_to_id[s] for s in scaffolds])
        logger.info(f"Sampled {_sample_frac:.0%}: {_n_sample:,} compounds")

    n_compounds = len(X)
    n_pains = int(y.sum())

    mo.md(f"""
    ## Data Summary

    | | |
    |---|---|
    | Dataset | `{_dataset}` |
    | Sample fraction | {_sample_frac:.0%} |
    | Compounds | {n_compounds:,} |
    | PAINS | {n_pains:,} ({n_pains / n_compounds * 100:.1f}%) |
    | Features | {X.shape[1]:,} dims |
    """)
    return X, X_cellcount, X_props, n_compounds, n_pains, scaffold_ids, scaffolds, y


@app.cell
def _(
    X,
    X_cellcount,
    X_props,
    dataset_dropdown,
    n_compounds,
    n_pains,
    scaffold_ids,
    scaffolds,
    skip_eda_toggle,
    y,
    mo,
):
    _dataset = dataset_dropdown.value
    _output_dir = OUTPUT_DIR / _dataset
    _output_dir.mkdir(parents=True, exist_ok=True)

    # EDA
    if not skip_eda_toggle.value:
        eda_stats, _eda_figs = run_eda(y, X, X_props, scaffolds, _output_dir)
    else:
        eda_stats = {"n_total": n_compounds, "n_pains": n_pains}
        _eda_figs = []

    # Cross-validation
    random_results = run_pains_cross_validation(X, y, scaffold_ids, X_props, X_cellcount, split_type="random")
    scaffold_results = run_pains_cross_validation(X, y, scaffold_ids, X_props, X_cellcount, split_type="scaffold")

    # Summary
    summary = generate_pains_summary(
        random_results,
        scaffold_results,
        eda_stats,
        n_compounds,
        n_pains,
        X.shape[1],
    )

    mo.md(f"""
    ## Cross-Validation Results

    ### Classification (ROC-AUC, scaffold split)

    | Model | ROC-AUC |
    |-------|---------|
    | Dummy | {scaffold_results["baseline_dummy"]["roc_auc"]["mean"]:.3f} |
    | Cell count | {scaffold_results["baseline_cellcount"]["roc_auc"]["mean"]:.3f} |
    | Properties | {scaffold_results["baseline_properties"]["roc_auc"]["mean"]:.3f} |
    | Morphology | {scaffold_results["morphology"]["roc_auc"]["mean"]:.3f} +/- {scaffold_results["morphology"]["roc_auc"]["std"]:.3f} |

    ### Generalization Gap

    | | Random Split | Scaffold Split | Gap |
    |---|---|---|---|
    | ROC-AUC | {random_results["morphology"]["roc_auc"]["mean"]:.3f} | {scaffold_results["morphology"]["roc_auc"]["mean"]:.3f} | {summary["generalization_gap"]["roc_auc_drop"]:.3f} |
    | Avg Precision | {random_results["morphology"]["avg_precision"]["mean"]:.3f} | {scaffold_results["morphology"]["avg_precision"]["mean"]:.3f} | |

    **Conclusion:** {summary["conclusion"]}
    """)
    return eda_stats, random_results, scaffold_results, summary


@app.cell
def _(random_results, scaffold_results):
    comparison_fig = plot_pains_comparison(random_results, scaffold_results)
    comparison_fig
    return (comparison_fig,)


@app.cell
def _(X, scaffold_ids, y):
    roc_pr_fig = plot_roc_pr_curves(X, y, scaffold_ids)
    roc_pr_fig
    return (roc_pr_fig,)


@app.cell
def _(
    X,
    comparison_fig,
    dataset_dropdown,
    mo,
    random_results,
    roc_pr_fig,
    scaffold_results,
    summary,
    y,
):
    _dataset = dataset_dropdown.value
    _output_dir = OUTPUT_DIR / _dataset
    _output_dir.mkdir(parents=True, exist_ok=True)

    # Feature importance
    _importance_df, importance_fig = plot_feature_importance(
        scaffold_results["models"],
        X,
        y,
        _output_dir,
    )

    # Save result JSONs
    _random_save = {k: v for k, v in random_results.items() if k != "models"}
    _scaffold_save = {k: v for k, v in scaffold_results.items() if k != "models"}
    with open(_output_dir / "random_split_results.json", "w") as _f:
        json.dump(_random_save, _f, indent=2)
    with open(_output_dir / "scaffold_split_results.json", "w") as _f:
        json.dump(_scaffold_save, _f, indent=2)
    with open(_output_dir / "summary.json", "w") as _f:
        json.dump(summary, _f, indent=2)

    # Save figures
    comparison_fig.savefig(_output_dir / "comparison_plot.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    comparison_fig.savefig(_output_dir / "comparison_plot.pdf", bbox_inches="tight")
    roc_pr_fig.savefig(_output_dir / "roc_pr_curves.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    roc_pr_fig.savefig(_output_dir / "roc_pr_curves.pdf", bbox_inches="tight")

    mo.md(f"""
    **Saved outputs to** `{_output_dir}`

    - `summary.json` - aggregated summary
    - `random_split_results.json` / `scaffold_split_results.json` - detailed CV results
    - `comparison_plot.png/.pdf` - bar chart comparison figure
    - `roc_pr_curves.png/.pdf` - ROC and PR curve figure
    - `feature_importance.csv/.png/.pdf` - permutation feature importance
    """)
    return (importance_fig,)


@app.function
def run_pains_prediction(
    dataset: str = "compound_no_source7",
    sample=1.0,
    output_dir=None,
) -> str:
    """Run PAINS prediction analysis with scaffold + random CV splits.

    Composes load_profiles, load_pains_labels_and_metadata, align_pains_data,
    run_pains_cross_validation, plot_pains_comparison, and
    generate_pains_summary. Saves comparison plot, result JSONs, summary.json,
    and .complete marker.

    Called from workflow.py via run_task.py.

    Args:
        dataset: Dataset name (e.g., "compound_no_source7")
        sample: Fraction of data to use (0.0-1.0). Accepts str from subprocess.
        output_dir: Output directory (default: data/processed/pains-prediction/...)

    Returns:
        Path to .complete marker file.
    """
    sample = float(sample) if isinstance(sample, str) else sample

    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "pains-prediction" / dataset
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    _adata = load_profiles(dataset, level="perturbation")
    _metadata_df = load_pains_labels_and_metadata()

    # Align
    X, y, _scaffolds, X_props, X_cellcount, scaffold_ids = align_pains_data(_adata, _metadata_df)

    # Sample if requested
    _sample_frac = sample
    if _sample_frac < 1.0:
        _n_sample = int(len(X) * _sample_frac)
        _rng = np.random.default_rng(RANDOM_STATE)
        _sample_idx = _rng.choice(len(X), size=_n_sample, replace=False)
        X = X[_sample_idx]
        y = y[_sample_idx]
        _scaffolds = _scaffolds[_sample_idx]
        X_props = X_props[_sample_idx]
        X_cellcount = X_cellcount[_sample_idx]
        _unique_sc = list(set(_scaffolds))
        _sc_to_id = {s: i for i, s in enumerate(_unique_sc)}
        scaffold_ids = np.array([_sc_to_id[s] for s in _scaffolds])
        logger.info(f"Sampled {_sample_frac:.0%}: {_n_sample:,} compounds")

    _n_compounds = len(X)
    _n_pains = int(y.sum())

    # Cross-validation
    _random_results = run_pains_cross_validation(X, y, scaffold_ids, X_props, X_cellcount, split_type="random")
    _scaffold_results = run_pains_cross_validation(X, y, scaffold_ids, X_props, X_cellcount, split_type="scaffold")

    # Summary (skip EDA for headless run)
    _eda_stats = {"n_total": _n_compounds, "n_pains": _n_pains}
    _summary = generate_pains_summary(
        _random_results, _scaffold_results, _eda_stats, _n_compounds, _n_pains, X.shape[1]
    )

    # Plot and save
    _fig = plot_pains_comparison(
        {k: v for k, v in _random_results.items() if k != "models"},
        {k: v for k, v in _scaffold_results.items() if k != "models"},
    )
    _fig.savefig(output_dir / "comparison_plot.png", dpi=150, bbox_inches="tight", facecolor="white")
    _fig.savefig(output_dir / "comparison_plot.pdf", bbox_inches="tight")
    plt.close(_fig)

    # Save result JSONs (strip non-serializable model objects)
    _random_save = {k: v for k, v in _random_results.items() if k != "models"}
    _scaffold_save = {k: v for k, v in _scaffold_results.items() if k != "models"}
    with open(output_dir / "random_split_results.json", "w") as _f:
        json.dump(_random_save, _f, indent=2)
    with open(output_dir / "scaffold_split_results.json", "w") as _f:
        json.dump(_scaffold_save, _f, indent=2)
    Path(output_dir, "summary.json").write_text(json.dumps(_summary, indent=2))

    # Write .complete marker
    _marker = output_dir / ".complete"
    _marker.touch()
    logger.success(f"Saved PAINS prediction outputs to {output_dir}")
    return str(_marker)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
