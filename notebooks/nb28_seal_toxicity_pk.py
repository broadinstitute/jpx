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
#     "python-dotenv",
#     "scikit-learn==1.8.0",
#     "scipy==1.17.1",
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
    from loguru import logger
    from scipy.stats import spearmanr
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import KFold, StratifiedKFold

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import (
        DEFAULT_DPI,
        INTERIM_DATA_DIR,
        METADATA_DB,
        PROCESSED_DATA_DIR,
    )
    from nb03_ss_profiles import load_profiles

    OUTPUT_DIR = PROCESSED_DATA_DIR / "toxicity-pk-prediction"
    MORGAN_FP_FILE = INTERIM_DATA_DIR / "compound_featurization" / "morgan_fp.npz"

    PK_TARGETS = ["human_VDss_L_kg", "human_CL_mL_min_kg", "human_fup", "human_mrt", "human_thalf"]
    PK_TARGET_NAMES = {
        "human_VDss_L_kg": "VDss (L/kg)",
        "human_CL_mL_min_kg": "CL (mL/min/kg)",
        "human_fup": "fup",
        "human_mrt": "MRT (h)",
        "human_thalf": "t1/2 (h)",
    }
    PK_LOG_SCALE_TARGETS = {"human_VDss_L_kg", "human_CL_mL_min_kg", "human_mrt", "human_thalf"}

    N_FOLDS = 5
    RANDOM_STATE = 42
    MODALITIES = ["morphology", "fingerprint", "combined"]
    MODALITY_COLORS = {"morphology": "#1f77b4", "fingerprint": "#2ca02c", "combined": "#d62728"}
    MODALITY_LABELS = {"morphology": "Morphology", "fingerprint": "Fingerprint", "combined": "Combined"}


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Toxicity and PK Endpoint Prediction

    Predicts DILI, DICT, MitoTox (classification) and pharmacokinetic endpoints
    (regression) from three feature modalities:

    - **Morphology**: Cell Painting profiles
    - **Fingerprint**: Morgan ECFP4 (2048-bit, radius 2) from precomputed npz
    - **Combined**: Concatenation of morphology + fingerprint features

    All three modalities use the same compound set and CV fold indices for fair comparison.
    Labels come from the augmented metadata database (pre-matched to JUMP compounds).

    **PK regression targets:** VDss, clearance, fraction unbound, MRT, half-life.
    Log-scale targets are trained in log space and evaluated on original scale.

    **Outputs:** `data/processed/toxicity-pk-prediction/{dataset}/`
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
    skip_plots_toggle = mo.ui.switch(value=False, label="Skip plots")
    mo.hstack([dataset_dropdown, skip_plots_toggle], justify="start")
    return dataset_dropdown, skip_plots_toggle


# =========================================================================
# Data loading (all from augmented metadata database)
# =========================================================================


@app.function
def load_dict_data() -> pd.DataFrame:
    """Load DICT labels from augmented database.

    Positive = 'less' or 'most' concern, Negative = 'no' concern.
    """
    _con = duckdb.connect(str(METADATA_DB), read_only=True)
    _df = _con.execute("""
        SELECT Metadata_JCP2022 AS JCP2022,
               CASE WHEN Metadata_dict_concern IN ('less', 'most') THEN 1 ELSE 0 END AS label
        FROM toxicity_pk_annotations
        WHERE Metadata_dict_concern IN ('less', 'most', 'no')
    """).df()
    _con.close()
    _n_pos = (_df["label"] == 1).sum()
    _n_neg = (_df["label"] == 0).sum()
    logger.info(f"DICT: {len(_df):,} compounds ({_n_pos} positive, {_n_neg} negative)")
    return _df


@app.function
def load_dili_data() -> pd.DataFrame:
    """Load DILI labels from augmented database.

    Positive = 'less' or 'most' concern, Negative = 'no' concern.
    """
    _con = duckdb.connect(str(METADATA_DB), read_only=True)
    _df = _con.execute("""
        SELECT Metadata_JCP2022 AS JCP2022,
               CASE WHEN Metadata_dili_concern IN ('less', 'most') THEN 1 ELSE 0 END AS label
        FROM toxicity_pk_annotations
        WHERE Metadata_dili_concern IN ('less', 'most', 'no')
    """).df()
    _con.close()
    _n_pos = (_df["label"] == 1).sum()
    _n_neg = (_df["label"] == 0).sum()
    logger.info(f"DILI: {len(_df):,} compounds ({_n_pos} positive, {_n_neg} negative)")
    return _df


@app.function
def load_mitotox_data() -> pd.DataFrame:
    """Load MitoTox labels from augmented database."""
    _con = duckdb.connect(str(METADATA_DB), read_only=True)
    _df = _con.execute("""
        SELECT Metadata_JCP2022 AS JCP2022, Metadata_mitotox_toxic AS label
        FROM mitotox_annotations
    """).df()
    _con.close()
    _n_pos = (_df["label"] == 1).sum()
    _n_neg = (_df["label"] == 0).sum()
    logger.info(f"MitoTox: {len(_df):,} compounds ({_n_pos} toxic, {_n_neg} non-toxic)")
    return _df


@app.function
def load_pk_data() -> pd.DataFrame:
    """Load PK data from augmented database."""
    _con = duckdb.connect(str(METADATA_DB), read_only=True)
    _df = _con.execute("""
        SELECT Metadata_JCP2022 AS JCP2022,
               Metadata_pk_vdss_l_kg AS human_VDss_L_kg,
               Metadata_pk_cl_ml_min_kg AS human_CL_mL_min_kg,
               Metadata_pk_fup AS human_fup,
               Metadata_pk_mrt_h AS human_mrt,
               Metadata_pk_thalf_h AS human_thalf
        FROM toxicity_pk_annotations
        WHERE Metadata_pk_vdss_l_kg IS NOT NULL
           OR Metadata_pk_cl_ml_min_kg IS NOT NULL
           OR Metadata_pk_fup IS NOT NULL
           OR Metadata_pk_mrt_h IS NOT NULL
           OR Metadata_pk_thalf_h IS NOT NULL
    """).df()
    _con.close()
    logger.info(f"PK: {len(_df):,} compounds with any PK data")
    return _df


@app.function
def load_morgan_fingerprints() -> dict[str, np.ndarray]:
    """Load precomputed Morgan fingerprints (2048-bit, radius 2) from npz."""
    _data = np.load(MORGAN_FP_FILE, allow_pickle=True)
    _jcp_ids = _data["jcp2022"]
    _fps = _data["fingerprints"]
    _valid_mask = _data["valid_mask"]

    _valid_fps = _fps[_valid_mask].astype(np.float32)
    _valid_ids = _jcp_ids[_valid_mask]

    _fp_dict = {jcp: _valid_fps[i] for i, jcp in enumerate(_valid_ids)}
    logger.info(f"Morgan FPs: {len(_fp_dict):,} valid compounds (of {len(_jcp_ids):,} total)")
    return _fp_dict


@app.function
def get_features_multi(
    adata,
    matched_df: pd.DataFrame,
    value_col: str,
    fp_dict: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    """Extract features for morphology, fingerprint, and combined modalities.

    All three modalities use the SAME compound set and ordering,
    enabling direct fold-to-fold comparison.

    Returns:
        Dict with keys "morphology", "fingerprint", "combined", each mapping to
        (X, y, jcp_ids) tuple. Empty dict if no compounds match.
    """
    _profile_jcp = adata.obs["JCP2022"].values
    _profile_idx = {k: i for i, k in enumerate(_profile_jcp)}
    _matched_jcp = list(matched_df["JCP2022"])
    _matched_idx = matched_df.set_index("JCP2022")

    _common_jcp = [jcp for jcp in _matched_jcp if jcp in _profile_idx and jcp in fp_dict]
    _n_missing_fp = sum(1 for jcp in _matched_jcp if jcp in _profile_idx and jcp not in fp_dict)
    if _n_missing_fp > 0:
        logger.warning(f"  {_n_missing_fp} compounds have profiles but no valid fingerprint")
    logger.info(f"  Common compound set: {len(_common_jcp):,} / {len(_matched_jcp):,}")

    if not _common_jcp:
        return {}

    _morph_indices = [_profile_idx[jcp] for jcp in _common_jcp]
    X_morph = adata.X[_morph_indices].astype(np.float32)
    X_fp = np.stack([fp_dict[jcp] for jcp in _common_jcp], axis=0)
    X_combined = np.concatenate([X_morph, X_fp], axis=1)

    _y = _matched_idx.loc[_common_jcp, value_col].values
    if value_col == "label":
        _y = _y.astype(np.int32)
    else:
        _y = _y.astype(np.float32)

    _morph_norms = np.linalg.norm(X_morph, axis=1)
    _valid_mask = _morph_norms > 0
    if not _valid_mask.all():
        _n_invalid = (~_valid_mask).sum()
        logger.warning(f"  Removed {_n_invalid} compounds with zero-norm morphology features")
        X_morph = X_morph[_valid_mask]
        X_fp = X_fp[_valid_mask]
        X_combined = X_combined[_valid_mask]
        _y = _y[_valid_mask]
        _common_jcp = [jcp for jcp, v in zip(_common_jcp, _valid_mask) if v]

    return {
        "morphology": (X_morph, _y, _common_jcp),
        "fingerprint": (X_fp, _y, _common_jcp),
        "combined": (X_combined, _y, _common_jcp),
    }


# =========================================================================
# Classification model training
# =========================================================================


@app.function
def train_classification_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_FOLDS,
    fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """Train classification model with stratified K-fold CV.

    Returns dict with per-fold metrics for model and baseline.
    """
    if fold_indices is None:
        _skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        fold_indices = list(_skf.split(X, y))

    _fold_results = []
    _all_y_true: list = []
    _all_y_pred_model: list = []
    _all_y_pred_baseline: list = []

    for _fold_idx, (_train_idx, _test_idx) in enumerate(fold_indices):
        X_train, y_train = X[_train_idx], y[_train_idx]
        X_test, y_test = X[_test_idx], y[_test_idx]

        _model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=6,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=0,
        )
        _model.fit(X_train, y_train)
        _y_pred_proba = _model.predict_proba(X_test)[:, 1]
        _y_pred = _model.predict(X_test)

        _baseline = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
        _baseline.fit(X_train, y_train)
        _y_pred_bl_proba = _baseline.predict_proba(X_test)[:, 1]
        _y_pred_bl = _baseline.predict(X_test)

        _fold_results.append(
            {
                "fold": _fold_idx + 1,
                "n_train": len(_train_idx),
                "n_test": len(_test_idx),
                "model_auc": roc_auc_score(y_test, _y_pred_proba),
                "model_ba": balanced_accuracy_score(y_test, _y_pred),
                "model_f1": f1_score(y_test, _y_pred),
                "model_precision": precision_score(y_test, _y_pred, zero_division=0),
                "model_recall": recall_score(y_test, _y_pred, zero_division=0),
                "baseline_auc": roc_auc_score(y_test, _y_pred_bl_proba),
                "baseline_ba": balanced_accuracy_score(y_test, _y_pred_bl),
                "baseline_f1": f1_score(y_test, _y_pred_bl),
            }
        )

        _all_y_true.extend(y_test)
        _all_y_pred_model.extend(_y_pred_proba)
        _all_y_pred_baseline.extend(_y_pred_bl_proba)

    _results_df = pd.DataFrame(_fold_results)
    _summary = {
        "model_auc_mean": _results_df["model_auc"].mean(),
        "model_auc_std": _results_df["model_auc"].std(),
        "model_ba_mean": _results_df["model_ba"].mean(),
        "model_ba_std": _results_df["model_ba"].std(),
        "model_f1_mean": _results_df["model_f1"].mean(),
        "model_f1_std": _results_df["model_f1"].std(),
        "baseline_auc_mean": _results_df["baseline_auc"].mean(),
        "baseline_ba_mean": _results_df["baseline_ba"].mean(),
    }

    return {
        "fold_results": _results_df,
        "summary": _summary,
        "y_true": np.array(_all_y_true),
        "y_pred_model": np.array(_all_y_pred_model),
        "y_pred_baseline": np.array(_all_y_pred_baseline),
    }


# =========================================================================
# Regression model training
# =========================================================================


@app.function
def train_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    target_name: str,
    n_splits: int = N_FOLDS,
    use_log_scale: bool = False,
    fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """Train regression model with K-fold CV.

    If use_log_scale, trains on log(y) and evaluates on original scale.
    """
    if use_log_scale:
        _valid = y > 0
        if not _valid.all():
            _n_invalid = (~_valid).sum()
            logger.warning(f"Removing {_n_invalid} non-positive values for log transform")
            X = X[_valid]
            y = y[_valid]
            if fold_indices is not None:
                _old_to_new = np.full(len(_valid), -1, dtype=int)
                _old_to_new[_valid] = np.arange(_valid.sum())
                fold_indices = [
                    (
                        _old_to_new[ti[ti < len(_valid)]][_old_to_new[ti[ti < len(_valid)]] >= 0],
                        _old_to_new[te[te < len(_valid)]][_old_to_new[te[te < len(_valid)]] >= 0],
                    )
                    for ti, te in fold_indices
                ]
        _y_transformed = np.log(y)
        logger.info(f"    Using log scale for {target_name}")
    else:
        _y_transformed = y

    if fold_indices is None:
        _kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        fold_indices = list(_kf.split(X))

    _fold_results = []
    _all_y_true: list = []
    _all_y_pred_model: list = []
    _all_y_pred_baseline: list = []

    for _fold_idx, (_train_idx, _test_idx) in enumerate(fold_indices):
        X_train, X_test = X[_train_idx], X[_test_idx]
        _y_train_t = _y_transformed[_train_idx]
        _y_test_orig = y[_test_idx]

        _model = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbose=0,
        )
        _model.fit(X_train, _y_train_t)
        _y_pred_t = _model.predict(X_test)

        _baseline = DummyRegressor(strategy="mean")
        _baseline.fit(X_train, _y_train_t)
        _y_pred_bl_t = _baseline.predict(X_test)

        if use_log_scale:
            _y_pred = np.exp(_y_pred_t)
            _y_pred_bl = np.exp(_y_pred_bl_t)
        else:
            _y_pred = _y_pred_t
            _y_pred_bl = _y_pred_bl_t

        _fold_results.append(
            {
                "fold": _fold_idx + 1,
                "target": target_name,
                "n_train": len(_train_idx),
                "n_test": len(_test_idx),
                "log_scale": use_log_scale,
                "model_r2": r2_score(_y_test_orig, _y_pred),
                "model_rmse": np.sqrt(mean_squared_error(_y_test_orig, _y_pred)),
                "model_mae": mean_absolute_error(_y_test_orig, _y_pred),
                "model_spearman": spearmanr(_y_test_orig, _y_pred).correlation,
                "baseline_r2": r2_score(_y_test_orig, _y_pred_bl),
                "baseline_rmse": np.sqrt(mean_squared_error(_y_test_orig, _y_pred_bl)),
                "baseline_mae": mean_absolute_error(_y_test_orig, _y_pred_bl),
            }
        )

        if use_log_scale:
            _all_y_true.extend(np.log(_y_test_orig))
            _all_y_pred_model.extend(_y_pred_t)
            _all_y_pred_baseline.extend(_y_pred_bl_t)
        else:
            _all_y_true.extend(_y_test_orig)
            _all_y_pred_model.extend(_y_pred)
            _all_y_pred_baseline.extend(_y_pred_bl)

    _results_df = pd.DataFrame(_fold_results)
    _summary = {
        "model_r2_mean": _results_df["model_r2"].mean(),
        "model_r2_std": _results_df["model_r2"].std(),
        "model_rmse_mean": _results_df["model_rmse"].mean(),
        "model_mae_mean": _results_df["model_mae"].mean(),
        "model_spearman_mean": _results_df["model_spearman"].mean(),
        "model_spearman_std": _results_df["model_spearman"].std(),
        "baseline_r2_mean": _results_df["baseline_r2"].mean(),
        "baseline_rmse_mean": _results_df["baseline_rmse"].mean(),
    }

    return {
        "fold_results": _results_df,
        "summary": _summary,
        "y_true": np.array(_all_y_true),
        "y_pred_model": np.array(_all_y_pred_model),
        "y_pred_baseline": np.array(_all_y_pred_baseline),
        "use_log_scale": use_log_scale,
    }


# =========================================================================
# Visualization
# =========================================================================


@app.function
def plot_classification_roc(
    modality_results: dict[str, dict],
    task_name: str,
) -> plt.Figure:
    """Plot ROC curves for all modalities for a classification task."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for _modality in MODALITIES:
        if _modality not in modality_results:
            continue
        _results = modality_results[_modality]
        _fpr, _tpr, _ = roc_curve(_results["y_true"], _results["y_pred_model"])
        _auc = _results["summary"]["model_auc_mean"]
        _auc_std = _results["summary"]["model_auc_std"]
        ax.plot(
            _fpr,
            _tpr,
            color=MODALITY_COLORS[_modality],
            lw=2,
            label=f"{MODALITY_LABELS[_modality]} (AUC = {_auc:.3f} +/- {_auc_std:.3f})",
        )

    _first = next(iter(modality_results.values()))
    _baseline_auc = _first["summary"]["baseline_auc_mean"]
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label=f"Baseline (AUC = {_baseline_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{task_name} Prediction - ROC Curves by Modality")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    return fig


@app.function
def plot_regression_scatter(
    modality_results: dict[str, dict],
    target_name: str,
    target_display: str,
) -> plt.Figure:
    """Plot predicted vs actual scatter for all modalities (1x3 grid)."""
    _active = [m for m in MODALITIES if m in modality_results]
    _n = len(_active)
    fig, axes = plt.subplots(1, _n, figsize=(6 * _n, 5))
    if _n == 1:
        axes = [axes]

    for _ax, _modality in zip(axes, _active):
        _results = modality_results[_modality]
        _y_true = _results["y_true"]
        _y_pred = _results["y_pred_model"]
        _r2 = _results["summary"]["model_r2_mean"]
        _spearman_val = _results["summary"]["model_spearman_mean"]
        _use_log = _results.get("use_log_scale", False)

        _ax.scatter(_y_true, _y_pred, alpha=0.5, s=20, c=MODALITY_COLORS[_modality])
        _min_val = min(_y_true.min(), _y_pred.min())
        _max_val = max(_y_true.max(), _y_pred.max())
        _ax.plot([_min_val, _max_val], [_min_val, _max_val], "k--", alpha=0.5)
        _ax.text(
            0.05,
            0.95,
            f"R2 = {_r2:.3f}\nrho = {_spearman_val:.3f}",
            transform=_ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        _prefix = "log " if _use_log else ""
        _ax.set_xlabel(f"Actual {_prefix}{target_display}")
        _ax.set_ylabel(f"Predicted {_prefix}{target_display}")
        _ax.set_title(MODALITY_LABELS[_modality])

    fig.suptitle(f"{target_display} Prediction by Modality", fontsize=12)
    fig.tight_layout()
    return fig


@app.function
def plot_summary_bar(
    classification_results: dict,
    regression_results: dict,
) -> plt.Figure:
    """Create summary bar chart comparing modalities and baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _width = 0.2

    # Classification
    _ax = axes[0]
    _tasks = list(classification_results.keys())
    _x = np.arange(len(_tasks))

    for _i, _modality in enumerate(MODALITIES):
        _aucs = [classification_results[t][_modality]["summary"]["model_auc_mean"] for t in _tasks]
        _stds = [classification_results[t][_modality]["summary"]["model_auc_std"] for t in _tasks]
        _offset = (_i - 1.5) * _width
        _ax.bar(
            _x + _offset,
            _aucs,
            _width,
            yerr=_stds,
            label=MODALITY_LABELS[_modality],
            capsize=3,
            color=MODALITY_COLORS[_modality],
        )

    _bl_aucs = [classification_results[t]["morphology"]["summary"]["baseline_auc_mean"] for t in _tasks]
    _ax.bar(_x + 1.5 * _width, _bl_aucs, _width, label="Baseline", color="gray", alpha=0.7)
    _ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    _ax.set_ylabel("AUC-ROC")
    _ax.set_title("Classification Tasks")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_tasks)
    _ax.legend(fontsize=8)
    _ax.set_ylim([0, 1])

    # Regression
    _ax = axes[1]
    _targets = list(regression_results.keys())
    _x = np.arange(len(_targets))

    for _i, _modality in enumerate(MODALITIES):
        _r2s = [regression_results[t][_modality]["summary"]["model_r2_mean"] for t in _targets]
        _stds = [regression_results[t][_modality]["summary"]["model_r2_std"] for t in _targets]
        _offset = (_i - 1.5) * _width
        _ax.bar(
            _x + _offset,
            _r2s,
            _width,
            yerr=_stds,
            label=MODALITY_LABELS[_modality],
            capsize=3,
            color=MODALITY_COLORS[_modality],
        )

    _bl_r2s = [regression_results[t]["morphology"]["summary"]["baseline_r2_mean"] for t in _targets]
    _ax.bar(_x + 1.5 * _width, _bl_r2s, _width, label="Baseline", color="gray", alpha=0.7)
    _ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    _ax.set_ylabel("R2")
    _ax.set_title("Regression Tasks (PK Endpoints)")
    _ax.set_xticks(_x)
    _ax.set_xticklabels([PK_TARGET_NAMES.get(t, t) for t in _targets], rotation=45, ha="right")
    _ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# =========================================================================
# Summary generation
# =========================================================================


@app.function
def generate_toxpk_summary(
    classification_results: dict,
    regression_results: dict,
    matching_stats: dict,
) -> dict:
    """Generate summary dict with all results nested by modality."""
    _summary: dict = {
        "task": "DILI/DICT/MitoTox/PK prediction: morphology vs fingerprint vs combined",
        "research_question": "Do morphology, chemical structure, or their combination best predict tox/PK endpoints?",
        "method": "5-fold stratified CV for classification, 5-fold CV for regression",
        "model": "HistGradientBoostingClassifier/Regressor",
        "modalities": {
            "morphology": "Cell Painting profiles",
            "fingerprint": "Morgan ECFP4 (2048-bit, radius 2)",
            "combined": "Morphology + Fingerprint concatenation",
        },
        "compound_matching": {
            "method": "Augmented metadata database (pre-matched via InChIKey)",
            "stats": matching_stats,
        },
        "classification": {},
        "regression": {},
    }

    _conclusions = []

    for _task_name, _modality_results in classification_results.items():
        _first = next(iter(_modality_results.values()))
        _n = int(_first["fold_results"]["n_train"].iloc[0] + _first["fold_results"]["n_test"].iloc[0])
        _task_summary: dict = {"n_compounds": _n}

        _best_auc = -1.0
        _best_mod = None
        for _modality, _results in _modality_results.items():
            _s = _results["summary"]
            _task_summary[_modality] = {
                "auc_mean": float(_s["model_auc_mean"]),
                "auc_std": float(_s["model_auc_std"]),
                "balanced_accuracy_mean": float(_s["model_ba_mean"]),
                "f1_mean": float(_s["model_f1_mean"]),
            }
            if _s["model_auc_mean"] > _best_auc:
                _best_auc = _s["model_auc_mean"]
                _best_mod = _modality

        _bl_s = _first["summary"]
        _task_summary["baseline"] = {
            "auc_mean": float(_bl_s["baseline_auc_mean"]),
            "balanced_accuracy_mean": float(_bl_s["baseline_ba_mean"]),
        }
        _task_summary["best_modality"] = _best_mod
        _summary["classification"][_task_name] = _task_summary

        _parts = [f"{m}={_task_summary[m]['auc_mean']:.3f}" for m in MODALITIES if m in _task_summary]
        _conclusions.append(f"{_task_name}: best={_best_mod} (AUC={_best_auc:.3f}), {', '.join(_parts)}")

    for _target_name, _modality_results in regression_results.items():
        _first = next(iter(_modality_results.values()))
        _n = int(_first["fold_results"]["n_train"].iloc[0] + _first["fold_results"]["n_test"].iloc[0])
        _target_summary: dict = {"n_compounds": _n}

        _best_r2 = -np.inf
        _best_mod = None
        for _modality, _results in _modality_results.items():
            _s = _results["summary"]
            _target_summary[_modality] = {
                "r2_mean": float(_s["model_r2_mean"]),
                "r2_std": float(_s["model_r2_std"]),
                "rmse_mean": float(_s["model_rmse_mean"]),
                "mae_mean": float(_s["model_mae_mean"]),
                "spearman_mean": float(_s["model_spearman_mean"]),
            }
            if _s["model_r2_mean"] > _best_r2:
                _best_r2 = _s["model_r2_mean"]
                _best_mod = _modality

        _bl_s = _first["summary"]
        _target_summary["baseline"] = {
            "r2_mean": float(_bl_s["baseline_r2_mean"]),
            "rmse_mean": float(_bl_s["baseline_rmse_mean"]),
        }
        _target_summary["best_modality"] = _best_mod
        _summary["regression"][_target_name] = _target_summary

        _parts = [f"{m}={_target_summary[m]['r2_mean']:.3f}" for m in MODALITIES if m in _target_summary]
        _conclusions.append(f"{_target_name}: best={_best_mod} (R2={_best_r2:.3f}), {', '.join(_parts)}")

    _summary["conclusions"] = _conclusions
    return _summary


# =========================================================================
# Main pipeline cells
# =========================================================================


@app.cell
def _(dataset_dropdown, mo):
    _dataset = dataset_dropdown.value

    mo.stop(
        not METADATA_DB.exists(),
        mo.md(f"**Missing database:** `{METADATA_DB}` not found."),
    )
    mo.stop(
        not MORGAN_FP_FILE.exists(),
        mo.md(f"**Missing file:** `{MORGAN_FP_FILE}` not found. Run featurization first."),
    )

    # Load profiles and fingerprints
    adata = load_profiles(_dataset, level="perturbation")
    fp_dict = load_morgan_fingerprints()

    # Load labels
    dict_matched = load_dict_data()
    dili_matched = load_dili_data()
    mitotox_matched = load_mitotox_data()
    pk_df = load_pk_data()

    matching_stats = {
        "DICT": {"matched": len(dict_matched)},
        "DILI": {"matched": len(dili_matched)},
        "MitoTox": {"matched": len(mitotox_matched)},
        "PK": {"matched": len(pk_df)},
    }

    mo.md(f"""
    ## Data Loaded

    | Endpoint | Matched Compounds |
    |----------|-------------------|
    | DICT | {len(dict_matched):,} |
    | DILI | {len(dili_matched):,} |
    | MitoTox | {len(mitotox_matched):,} |
    | PK (any target) | {len(pk_df):,} |

    Profiles: {adata.n_obs:,} compounds, {adata.n_vars:,} features.
    Fingerprints: {len(fp_dict):,} valid compounds.
    """)
    return adata, dict_matched, dili_matched, fp_dict, matching_stats, mitotox_matched, pk_df


@app.cell
def _(adata, dict_matched, dili_matched, fp_dict, mitotox_matched, mo):
    # Classification tasks
    classification_results = {}

    for _task_name, _task_matched in [("DICT", dict_matched), ("DILI", dili_matched), ("MitoTox", mitotox_matched)]:
        logger.info(f"\n--- {_task_name} ---")
        _mod_features = get_features_multi(adata, _task_matched, "label", fp_dict)
        if not _mod_features:
            logger.warning(f"No {_task_name} compounds matched, skipping")
            continue

        _X_morph, _y, _ = _mod_features["morphology"]
        logger.info(f"  Training on {len(_y):,} compounds")

        _skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        _fold_indices = list(_skf.split(_X_morph, _y))

        classification_results[_task_name] = {}
        for _modality in MODALITIES:
            logger.info(f"  Modality: {_modality}")
            _X, _y_mod, _ = _mod_features[_modality]
            classification_results[_task_name][_modality] = train_classification_cv(
                _X,
                _y_mod,
                N_FOLDS,
                fold_indices=_fold_indices,
            )

    # Build results table
    _lines = ["## Classification Results\n", "| Task | Modality | AUC (mean +/- std) |", "|---|---|---|"]
    for _task_name, _mod_results in classification_results.items():
        for _modality in MODALITIES:
            if _modality in _mod_results:
                _s = _mod_results[_modality]["summary"]
                _lines.append(
                    f"| {_task_name} | {MODALITY_LABELS[_modality]} | "
                    f"{_s['model_auc_mean']:.3f} +/- {_s['model_auc_std']:.3f} |"
                )
    mo.md("\n".join(_lines))
    return (classification_results,)


@app.cell
def _(adata, fp_dict, mo, pk_df):
    # Regression tasks (PK)
    regression_results = {}

    for _target in PK_TARGETS:
        logger.info(f"\n--- {PK_TARGET_NAMES[_target]} ---")
        _pk_target = pk_df.dropna(subset=[_target])
        if len(_pk_target) == 0:
            logger.warning(f"No PK compounds for {_target}, skipping")
            continue

        _mod_features = get_features_multi(adata, _pk_target, _target, fp_dict)
        if not _mod_features:
            logger.warning(f"No valid compounds for {_target}, skipping")
            continue

        _X_morph, _y, _ = _mod_features["morphology"]
        if len(_y) < N_FOLDS:
            logger.warning(f"Not enough samples ({len(_y)}) for {N_FOLDS}-fold CV, skipping {_target}")
            continue

        _use_log = _target in PK_LOG_SCALE_TARGETS

        # For log-scale, filter non-positive before generating folds
        if _use_log:
            _valid = _y > 0
            if not _valid.all():
                _n_inv = (~_valid).sum()
                logger.warning(f"  Removing {_n_inv} non-positive values for log transform")
                _mod_features = {
                    k: (_X[_valid], _yv[_valid], [j for j, v in zip(_jcp, _valid) if v])
                    for k, (_X, _yv, _jcp) in _mod_features.items()
                }
                _X_morph, _y, _ = _mod_features["morphology"]

        _kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        _fold_indices = list(_kf.split(_X_morph))

        regression_results[_target] = {}
        for _modality in MODALITIES:
            logger.info(f"  Modality: {_modality}")
            _X, _y_mod, _ = _mod_features[_modality]
            regression_results[_target][_modality] = train_regression_cv(
                _X,
                _y_mod,
                _target,
                N_FOLDS,
                _use_log,
                fold_indices=_fold_indices,
            )

    # Build results table
    _lines = [
        "## Regression Results (PK)\n",
        "| Target | Modality | R2 (mean +/- std) | Spearman |",
        "|---|---|---|---|",
    ]
    for _target, _mod_results in regression_results.items():
        for _modality in MODALITIES:
            if _modality in _mod_results:
                _s = _mod_results[_modality]["summary"]
                _lines.append(
                    f"| {PK_TARGET_NAMES.get(_target, _target)} | {MODALITY_LABELS[_modality]} | "
                    f"{_s['model_r2_mean']:.3f} +/- {_s['model_r2_std']:.3f} | "
                    f"{_s['model_spearman_mean']:.3f} |"
                )
    mo.md("\n".join(_lines))
    return (regression_results,)


@app.cell
def _(classification_results, mo, skip_plots_toggle):
    _figs = []
    if not skip_plots_toggle.value:
        for _task_name, _mod_results in classification_results.items():
            _figs.append(plot_classification_roc(_mod_results, _task_name))

    if _figs:
        mo.vstack(_figs)
    else:
        mo.md("*Classification ROC plots skipped.*")
    return


@app.cell
def _(mo, regression_results, skip_plots_toggle):
    _figs = []
    if not skip_plots_toggle.value:
        for _target, _mod_results in regression_results.items():
            _figs.append(
                plot_regression_scatter(
                    _mod_results,
                    _target,
                    PK_TARGET_NAMES.get(_target, _target),
                )
            )

    if _figs:
        mo.vstack(_figs)
    else:
        mo.md("*Regression scatter plots skipped.*")
    return


@app.cell
def _(classification_results, mo, regression_results, skip_plots_toggle):
    if not skip_plots_toggle.value and classification_results and regression_results:
        summary_bar_fig = plot_summary_bar(classification_results, regression_results)
        summary_bar_fig
    else:
        mo.md("*Summary bar chart skipped.*")
    return


@app.cell
def _(
    classification_results,
    dataset_dropdown,
    matching_stats,
    mo,
    regression_results,
    skip_plots_toggle,
):
    _dataset = dataset_dropdown.value
    _output_dir = OUTPUT_DIR / _dataset
    _output_dir.mkdir(parents=True, exist_ok=True)

    # Save classification fold results
    if classification_results:
        _all_cls_folds = []
        for _task_name, _mod_results in classification_results.items():
            for _modality, _results in _mod_results.items():
                _df = _results["fold_results"].copy()
                _df["task"] = _task_name
                _df["modality"] = _modality
                _all_cls_folds.append(_df)
        _cls_df = pd.concat(_all_cls_folds, ignore_index=True)
        _cls_df.to_csv(_output_dir / "classification_results.csv", index=False)

    # Save regression fold results
    if regression_results:
        _all_reg_folds = []
        for _target, _mod_results in regression_results.items():
            for _modality, _results in _mod_results.items():
                _df = _results["fold_results"].copy()
                _df["modality"] = _modality
                _all_reg_folds.append(_df)
        _reg_df = pd.concat(_all_reg_folds, ignore_index=True)
        _reg_df.to_csv(_output_dir / "regression_results.csv", index=False)

    # Save plots
    if not skip_plots_toggle.value:
        for _task_name, _mod_results in classification_results.items():
            _fig = plot_classification_roc(_mod_results, _task_name)
            _fig.savefig(_output_dir / f"roc_{_task_name.lower()}.png", dpi=DEFAULT_DPI, bbox_inches="tight")
            plt.close(_fig)
        for _target, _mod_results in regression_results.items():
            _fig = plot_regression_scatter(_mod_results, _target, PK_TARGET_NAMES.get(_target, _target))
            _fig.savefig(_output_dir / f"scatter_{_target}.png", dpi=DEFAULT_DPI, bbox_inches="tight")
            plt.close(_fig)
        if classification_results and regression_results:
            _fig = plot_summary_bar(classification_results, regression_results)
            _fig.savefig(_output_dir / "summary_comparison.png", dpi=DEFAULT_DPI, bbox_inches="tight")
            plt.close(_fig)

    # Generate and save summary
    summary = generate_toxpk_summary(classification_results, regression_results, matching_stats)
    with open(_output_dir / "summary.json", "w") as _f:
        json.dump(summary, _f, indent=2)

    _conclusion_lines = [f"- {c}" for c in summary.get("conclusions", [])]

    mo.md(f"""
    ## Saved Outputs

    **Directory:** `{_output_dir}`

    - `summary.json` - all results nested by endpoint and modality
    - `classification_results.csv` - fold results with modality column
    - `regression_results.csv` - fold results with modality column
    - ROC and scatter plot PNGs (if not skipped)
    - `summary_comparison.png` - bar chart overview (if not skipped)

    ### Conclusions

    {chr(10).join(_conclusion_lines)}
    """)
    return


@app.function
def run_toxicity_prediction(
    dataset,
    output_dir=None,
) -> str:
    """Run the full toxicity/PK prediction pipeline.

    Parameters
    ----------
    dataset : str
        Dataset name, e.g. "compound_no_source7".
    output_dir : str or Path, optional
        Output directory. Defaults to OUTPUT_DIR / dataset.

    Returns
    -------
    str
        Path to the output directory.
    """
    dataset = str(dataset)

    if output_dir is None:
        output_dir = OUTPUT_DIR / dataset
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running toxicity/PK prediction: dataset={dataset}")

    # Step 1: Load profiles and fingerprints
    adata = load_profiles(dataset, level="perturbation")
    fp_dict = load_morgan_fingerprints()

    # Step 2: Load labels
    dict_matched = load_dict_data()
    dili_matched = load_dili_data()
    mitotox_matched = load_mitotox_data()
    pk_df = load_pk_data()

    matching_stats = {
        "DICT": {"matched": len(dict_matched)},
        "DILI": {"matched": len(dili_matched)},
        "MitoTox": {"matched": len(mitotox_matched)},
        "PK": {"matched": len(pk_df)},
    }

    # Step 3: Classification tasks
    classification_results = {}
    for _task_name, _task_matched in [
        ("DICT", dict_matched),
        ("DILI", dili_matched),
        ("MitoTox", mitotox_matched),
    ]:
        logger.info(f"--- {_task_name} ---")
        _mod_features = get_features_multi(adata, _task_matched, "label", fp_dict)
        if not _mod_features:
            logger.warning(f"No {_task_name} compounds matched, skipping")
            continue

        _X_morph, _y, _ = _mod_features["morphology"]
        logger.info(f"  Training on {len(_y):,} compounds")

        _skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        _fold_indices = list(_skf.split(_X_morph, _y))

        classification_results[_task_name] = {}
        for _modality in MODALITIES:
            logger.info(f"  Modality: {_modality}")
            _X, _y_mod, _ = _mod_features[_modality]
            classification_results[_task_name][_modality] = train_classification_cv(
                _X, _y_mod, N_FOLDS, fold_indices=_fold_indices
            )

    # Step 4: Regression tasks (PK)
    regression_results = {}
    for _target in PK_TARGETS:
        logger.info(f"--- {PK_TARGET_NAMES[_target]} ---")
        _pk_target = pk_df.dropna(subset=[_target])
        if len(_pk_target) == 0:
            logger.warning(f"No PK compounds for {_target}, skipping")
            continue

        _mod_features = get_features_multi(adata, _pk_target, _target, fp_dict)
        if not _mod_features:
            logger.warning(f"No valid compounds for {_target}, skipping")
            continue

        _X_morph, _y, _ = _mod_features["morphology"]
        if len(_y) < N_FOLDS:
            logger.warning(f"Not enough samples ({len(_y)}) for {N_FOLDS}-fold CV, skipping {_target}")
            continue

        _use_log = _target in PK_LOG_SCALE_TARGETS

        # For log-scale, filter non-positive before generating folds
        if _use_log:
            _valid = _y > 0
            if not _valid.all():
                _mod_features = {
                    k: (_X[_valid], _yv[_valid], [j for j, v in zip(_jcp, _valid) if v])
                    for k, (_X, _yv, _jcp) in _mod_features.items()
                }
                _X_morph, _y, _ = _mod_features["morphology"]

        _kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        _fold_indices = list(_kf.split(_X_morph))

        regression_results[_target] = {}
        for _modality in MODALITIES:
            logger.info(f"  Modality: {_modality}")
            _X, _y_mod, _ = _mod_features[_modality]
            regression_results[_target][_modality] = train_regression_cv(
                _X, _y_mod, _target, N_FOLDS, _use_log, fold_indices=_fold_indices
            )

    # Step 5: Save classification fold results
    if classification_results:
        _all_cls_folds = []
        for _task_name, _mod_results in classification_results.items():
            for _modality, _results in _mod_results.items():
                _df = _results["fold_results"].copy()
                _df["task"] = _task_name
                _df["modality"] = _modality
                _all_cls_folds.append(_df)
        _cls_df = pd.concat(_all_cls_folds, ignore_index=True)
        _cls_df.to_csv(output_dir / "classification_results.csv", index=False)

    # Step 6: Save regression fold results
    if regression_results:
        _all_reg_folds = []
        for _target_name, _mod_results in regression_results.items():
            for _modality, _results in _mod_results.items():
                _df = _results["fold_results"].copy()
                _df["modality"] = _modality
                _all_reg_folds.append(_df)
        _reg_df = pd.concat(_all_reg_folds, ignore_index=True)
        _reg_df.to_csv(output_dir / "regression_results.csv", index=False)

    # Step 7: Save plots
    for _task_name, _mod_results in classification_results.items():
        fig = plot_classification_roc(_mod_results, _task_name)
        fig.savefig(output_dir / f"roc_{_task_name.lower()}.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    for _target_name, _mod_results in regression_results.items():
        fig = plot_regression_scatter(_mod_results, _target_name, PK_TARGET_NAMES.get(_target_name, _target_name))
        fig.savefig(output_dir / f"scatter_{_target_name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    if classification_results and regression_results:
        fig = plot_summary_bar(classification_results, regression_results)
        fig.savefig(output_dir / "summary_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # Step 8: Generate and save summary JSON
    summary = generate_toxpk_summary(classification_results, regression_results, matching_stats)
    Path(output_dir, "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info(f"Toxicity/PK prediction complete: {output_dir}")
    return str(output_dir)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
