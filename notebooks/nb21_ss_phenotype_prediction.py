# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb==1.5.3",
#     "loguru==0.7.3",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "python-dotenv",
#     "scikit-learn==1.8.0",
#     "scipy==1.17.1",
#     "xgboost==3.2.0",
# ]
# # NOTE: Run with pixi run -e deepchem marimo edit/run
# # rdkit and deepchem come from the pixi deepchem env (conda-only).
# # Do NOT add them to PEP 723 deps.
# ///

# ruff: noqa: N803, N806  # Allow uppercase X, X_train, X_test (ML convention)

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import json
    import sys
    import time
    from collections import defaultdict
    from pathlib import Path
    from typing import Any

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from loguru import logger
    from scipy import stats
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import (
        COPAIRS_RESULTS_DB,
        DATA_DIR,
        DEFAULT_DPI,
        METADATA_DB,
        PROCESSED_DATA_DIR,
    )
    from nb02_ss_queries import query_activity_results

    # GPU detection (same logic as jump_production.gpu.has_gpu)
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        HAS_GPU = True
    except Exception:
        HAS_GPU = False

    logger.info(f"GPU acceleration: {'enabled' if HAS_GPU else 'disabled'}")

    N_FOLDS = 5
    RANDOM_STATE = 42
    OUTPUT_DIR = PROCESSED_DATA_DIR / "phenotype-prediction"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Phenotype Prediction from Chemical Structure

    Predicts phenotypic outcomes (mAP activity) from chemical structure using
    proper molecular scaffold splits vs random splits.

    **Split strategies:**
    - Scaffold split using precomputed Murcko scaffolds (from metadata DB)
    - Random split with stratification (StratifiedKFold)

    **Structural representations:**
    - Morgan fingerprints (2048-bit)
    - ChemBERTa embeddings (384-dim)

    **Models:**
    - XGBoost (classification + regression)
    - MLP (optional, better for dense embeddings)
    - Baselines: dummy (majority class / mean) and simple properties (MW, LogP, TPSA)

    Uses 5-fold cross-validation. Supports GPU acceleration for XGBoost.

    **Environment:** Requires `pixi run -e deepchem marimo edit/run`

    *Outputs:* `data/processed/phenotype-prediction/{dataset}/{preprocessing}/{structure_rep}/`
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
    preprocessing_dropdown = mo.ui.dropdown(
        options=[
            "activity_no_target2",
        ],
        value="activity_no_target2",
        label="Preprocessing",
    )
    filter_dropdown = mo.ui.dropdown(
        options=[
            "all_sources",
        ],
        value="all_sources",
        label="Filter",
    )
    activity_params_dropdown = mo.ui.dropdown(
        options=[
            "default",
            "withinsource",
            "crosssource",
        ],
        value="default",
        label="Activity params",
    )
    structure_dropdown = mo.ui.dropdown(
        options=["morgan", "chemberta"],
        value="morgan",
        label="Structure representation",
    )
    sample_slider = mo.ui.slider(
        start=0.1,
        stop=1.0,
        step=0.1,
        value=0.1,
        label="Sample fraction",
    )
    mlp_toggle = mo.ui.switch(value=False, label="Also train MLP")
    mo.hstack(
        [
            dataset_dropdown,
            preprocessing_dropdown,
            filter_dropdown,
            activity_params_dropdown,
        ],
        justify="start",
    )
    mo.hstack(
        [structure_dropdown, sample_slider, mlp_toggle],
        justify="start",
    )
    return (
        activity_params_dropdown,
        dataset_dropdown,
        filter_dropdown,
        mlp_toggle,
        preprocessing_dropdown,
        sample_slider,
        structure_dropdown,
    )


@app.function
def scaffold_k_fold_split(smiles: np.ndarray, k: int = 5) -> list[tuple[list[str], list[str]]]:
    """Generate k-fold CV splits grouping compounds by Murcko scaffold.

    Uses precomputed scaffolds from compound_metadata.Metadata_MurckoScaffold.
    Implements the same algorithm as DeepChem's ScaffoldSplitter:
    1. Group compounds by scaffold
    2. Sort groups by size (largest first)
    3. Distribute to k buckets using greedy balancing

    Args:
        smiles: Array of SMILES strings
        k: Number of folds

    Returns:
        List of k tuples: (train_smiles, test_smiles)
    """
    _con = duckdb.connect(str(METADATA_DB), read_only=True)
    _df = _con.execute(
        """
        SELECT Metadata_SMILES, Metadata_MurckoScaffold
        FROM compound_metadata
        WHERE Metadata_SMILES = ANY(?)
          AND Metadata_MurckoScaffold IS NOT NULL
    """,
        [list(smiles)],
    ).df()
    _con.close()

    _scaffold_mapping = dict(zip(_df["Metadata_SMILES"], _df["Metadata_MurckoScaffold"]))
    logger.debug(f"Loaded scaffolds for {len(_scaffold_mapping):,} of {len(smiles):,} compounds")

    # Group indices by scaffold
    _scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    _no_scaffold_indices: list[int] = []

    for _idx, _smi in enumerate(smiles):
        _scaffold = _scaffold_mapping.get(_smi)
        if _scaffold and _scaffold != "":
            _scaffold_to_indices[_scaffold].append(_idx)
        else:
            _no_scaffold_indices.append(_idx)

    # Sort scaffolds by group size (largest first), then alphabetically for determinism
    _sorted_scaffolds = sorted(
        _scaffold_to_indices.keys(),
        key=lambda s: (-len(_scaffold_to_indices[s]), s),
    )

    # Initialize k buckets and distribute scaffold groups (greedy balancing)
    _buckets: list[list[int]] = [[] for _ in range(k)]
    for _scaffold in _sorted_scaffolds:
        _min_bucket = min(range(k), key=lambda i: len(_buckets[i]))
        _buckets[_min_bucket].extend(_scaffold_to_indices[_scaffold])

    # Distribute compounds without scaffolds
    for _idx in _no_scaffold_indices:
        _min_bucket = min(range(k), key=lambda i: len(_buckets[i]))
        _buckets[_min_bucket].append(_idx)

    # Generate train/test splits
    _smiles_list = list(smiles)
    _fold_splits = []
    for _fold_idx in range(k):
        _test_ids = [_smiles_list[i] for i in _buckets[_fold_idx]]
        _train_ids = [_smiles_list[i] for i in sum((_buckets[j] for j in range(k) if j != _fold_idx), [])]
        _fold_splits.append((_train_ids, _test_ids))

    logger.debug(f"Created {k} folds with sizes: {[len(b) for b in _buckets]}")
    return _fold_splits


@app.function
def load_fingerprints() -> tuple[np.ndarray, np.ndarray]:
    """Load compound fingerprints (Morgan FP, 2048-bit)."""
    _fp_file = DATA_DIR / "interim/compound_featurization/morgan_fp.npz"
    _fp_data = np.load(_fp_file, allow_pickle=True)
    _fp_array = _fp_data["fingerprints"]
    _fp_jcp = _fp_data["jcp2022"]
    logger.info(f"Morgan fingerprints: {_fp_array.shape[0]:,} compounds x {_fp_array.shape[1]} bits")
    return _fp_array, _fp_jcp


@app.function
def load_chemberta_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Load ChemBERTa-77M-MLM embeddings (384-dim)."""
    _chemberta_file = DATA_DIR / "interim/compound_featurization/chemberta_77m_mlm.npz"
    _emb_data = np.load(_chemberta_file, allow_pickle=True)
    _embeddings = _emb_data["embeddings"]
    _jcp = _emb_data["jcp2022"]
    logger.info(f"ChemBERTa embeddings: {_embeddings.shape[0]:,} compounds x {_embeddings.shape[1]} dims")
    return _embeddings, _jcp


@app.function
def load_smiles_and_properties() -> pd.DataFrame:
    """Load SMILES and simple properties from metadata database."""
    _con = duckdb.connect(str(METADATA_DB), read_only=True)
    _df = _con.execute("""
        SELECT
            Metadata_JCP2022 as JCP2022,
            Metadata_SMILES as SMILES,
            Metadata_MW as MW,
            Metadata_LogP as LogP,
            Metadata_TPSA as TPSA
        FROM compound_metadata
        WHERE Metadata_SMILES IS NOT NULL
          AND Metadata_MW IS NOT NULL
    """).df()
    _con.close()
    logger.info(f"SMILES + properties: {len(_df):,} compounds")
    return _df


@app.function
def load_activity_data(
    dataset: str,
    preprocessing: str,
    filter_name: str,
    activity_params: str,
) -> pd.DataFrame:
    """Load activity status from copairs results."""
    _df = query_activity_results(
        dataset,
        preprocessing=preprocessing,
        filter_name=filter_name,
        activity_params=activity_params,
    )
    _df = _df.rename(
        columns={
            "Metadata_JCP2022": "JCP2022",
            "mean_average_precision": "mAP",
            "mean_normalized_average_precision": "nmAP",
            "below_corrected_p": "active",
        }
    )
    logger.info(f"Activity: {len(_df):,} compounds, {_df.active.sum():,} active ({_df.active.mean() * 100:.1f}%)")
    return _df[["JCP2022", "mAP", "nmAP", "active"]]


@app.function
def align_data(
    features: np.ndarray,
    feature_jcp: np.ndarray,
    activity_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align features, activity labels, SMILES, and properties by JCP2022 ID.

    Returns:
        X: Feature matrix
        y_class: Binary labels (active/inactive)
        y_reg: Continuous labels (nmAP)
        smiles: SMILES strings for scaffold splitting
        X_props: Simple properties (MW, LogP, TPSA) for baseline model
    """
    _feature_set = set(feature_jcp)
    _activity_set = set(activity_df["JCP2022"])
    _metadata_set = set(metadata_df["JCP2022"])

    _common_jcp = sorted(_feature_set & _activity_set & _metadata_set)
    logger.info(f"Common compounds: {len(_common_jcp):,}")

    _feature_idx = {k: i for i, k in enumerate(feature_jcp)}
    _activity_idx = activity_df.set_index("JCP2022")
    _metadata_idx = metadata_df.set_index("JCP2022")

    _feature_order = [_feature_idx[k] for k in _common_jcp]
    X = features[_feature_order].astype(np.float32)
    y_class = _activity_idx.loc[_common_jcp, "active"].values.astype(np.int32)
    y_reg = _activity_idx.loc[_common_jcp, "nmAP"].values.astype(np.float32)
    _smiles = _metadata_idx.loc[_common_jcp, "SMILES"].values
    X_props = _metadata_idx.loc[_common_jcp, ["MW", "LogP", "TPSA"]].values.astype(np.float32)

    # Filter out compounds with zero-norm features or NaN values
    _feature_norms = np.linalg.norm(X, axis=1)
    _props_valid = ~np.isnan(X_props).any(axis=1)
    _valid_mask = (_feature_norms > 0) & ~np.isnan(y_reg) & _props_valid
    logger.info(f"Valid compounds: {_valid_mask.sum():,}")

    return X[_valid_mask], y_class[_valid_mask], y_reg[_valid_mask], _smiles[_valid_mask], X_props[_valid_mask]


@app.function
def evaluate_baselines(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_reg_train: np.ndarray,
    y_reg_test: np.ndarray,
) -> dict:
    """Evaluate dummy baseline (majority class / mean prediction)."""
    _dummy_clf = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    _dummy_clf.fit(np.zeros((len(y_train), 1)), y_train)
    _y_pred_proba = _dummy_clf.predict_proba(np.zeros((len(y_test), 1)))[:, 1]

    _dummy_reg = DummyRegressor(strategy="mean")
    _dummy_reg.fit(np.zeros((len(y_reg_train), 1)), y_reg_train)
    _y_pred_reg = _dummy_reg.predict(np.zeros((len(y_reg_test), 1)))

    return {
        "classification": {
            "roc_auc": float(roc_auc_score(y_test, _y_pred_proba)),
            "avg_precision": float(average_precision_score(y_test, _y_pred_proba)),
        },
        "regression": {
            "r2": float(r2_score(y_reg_test, _y_pred_reg)),
        },
    }


@app.function
def train_evaluate_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
) -> dict:
    """Train XGBoost classifier and evaluate with threshold-independent metrics."""
    _clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        device=device,
        verbosity=0,
    )
    _clf.fit(X_train, y_train)
    _y_pred_proba = _clf.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_test, _y_pred_proba)),
        "avg_precision": float(average_precision_score(y_test, _y_pred_proba)),
    }


@app.function
def train_evaluate_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
) -> dict:
    """Train XGBoost regressor and evaluate."""
    _reg = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        device=device,
        verbosity=0,
    )
    _reg.fit(X_train, y_train)
    _y_pred = _reg.predict(X_test)
    _spearman_result = stats.spearmanr(y_test, _y_pred)

    return {
        "r2": float(r2_score(y_test, _y_pred)),
        "spearman_r": float(_spearman_result.statistic),
        "spearman_p": float(_spearman_result.pvalue),
    }


@app.function
def train_evaluate_mlp_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train MLP classifier - better suited for dense embeddings like ChemBERTa."""
    _scaler = StandardScaler()
    _X_train_scaled = _scaler.fit_transform(X_train)
    _X_test_scaled = _scaler.transform(X_test)

    _clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    _clf.fit(_X_train_scaled, y_train)
    _y_pred_proba = _clf.predict_proba(_X_test_scaled)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_test, _y_pred_proba)),
        "avg_precision": float(average_precision_score(y_test, _y_pred_proba)),
    }


@app.function
def train_evaluate_mlp_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train MLP regressor - better suited for dense embeddings like ChemBERTa."""
    _scaler = StandardScaler()
    _X_train_scaled = _scaler.fit_transform(X_train)
    _X_test_scaled = _scaler.transform(X_test)

    _reg = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    _reg.fit(_X_train_scaled, y_train)
    _y_pred = _reg.predict(_X_test_scaled)
    _spearman_result = stats.spearmanr(y_test, _y_pred)

    return {
        "r2": float(r2_score(y_test, _y_pred)),
        "spearman_r": float(_spearman_result.statistic),
        "spearman_p": float(_spearman_result.pvalue),
    }


@app.function
def process_single_fold(
    fold_idx: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    X: np.ndarray,
    y_class: np.ndarray,
    y_reg: np.ndarray,
    X_props: np.ndarray,
    use_mlp: bool = False,
) -> dict:
    """Process a single CV fold.

    Args:
        fold_idx: Index of this fold (0-based)
        train_indices: Array of indices for training set
        test_indices: Array of indices for test set
        X: Full feature matrix
        y_class: Full binary labels
        y_reg: Full continuous labels
        X_props: Full properties matrix
        use_mlp: Whether to also train MLP models

    Returns:
        Dict with all metrics for this fold
    """
    _device = "cuda" if HAS_GPU else "cpu"
    _timings: dict[str, float] = {}

    X_train, y_class_train, y_reg_train = X[train_indices], y_class[train_indices], y_reg[train_indices]
    X_test, y_class_test, y_reg_test = X[test_indices], y_class[test_indices], y_reg[test_indices]
    X_props_train, X_props_test = X_props[train_indices], X_props[test_indices]

    # 1. Dummy baseline
    _t0 = time.perf_counter()
    _baseline_result = evaluate_baselines(y_class_train, y_class_test, y_reg_train, y_reg_test)
    _timings["baseline"] = time.perf_counter() - _t0

    # 2. Properties-only model (MW, LogP, TPSA)
    _t0 = time.perf_counter()
    _props_cls = train_evaluate_classification(X_props_train, y_class_train, X_props_test, y_class_test, _device)
    _props_reg = train_evaluate_regression(X_props_train, y_reg_train, X_props_test, y_reg_test, _device)
    _timings["props"] = time.perf_counter() - _t0

    # 3. Full model with XGBoost
    _t0 = time.perf_counter()
    _cls_result = train_evaluate_classification(X_train, y_class_train, X_test, y_class_test, _device)
    _timings["fp_cls"] = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    _reg_result = train_evaluate_regression(X_train, y_reg_train, X_test, y_reg_test, _device)
    _timings["fp_reg"] = time.perf_counter() - _t0

    _result: dict[str, Any] = {
        "fold_idx": fold_idx,
        "device": _device,
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "baseline": _baseline_result,
        "props_cls": _props_cls,
        "props_reg": {k: v for k, v in _props_reg.items() if k != "spearman_p"},
        "cls": _cls_result,
        "reg": {k: v for k, v in _reg_result.items() if k != "spearman_p"},
        "timings": _timings,
    }

    # 4. MLP model (optional)
    if use_mlp:
        _t0 = time.perf_counter()
        _mlp_cls = train_evaluate_mlp_classification(X_train, y_class_train, X_test, y_class_test)
        _timings["mlp_cls"] = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        _mlp_reg = train_evaluate_mlp_regression(X_train, y_reg_train, X_test, y_reg_test)
        _timings["mlp_reg"] = time.perf_counter() - _t0

        _result["mlp_cls"] = _mlp_cls
        _result["mlp_reg"] = {k: v for k, v in _mlp_reg.items() if k != "spearman_p"}

    return _result


@app.function
def run_cross_validation(
    X: np.ndarray,
    y_class: np.ndarray,
    y_reg: np.ndarray,
    smiles: np.ndarray,
    X_props: np.ndarray,
    split_type: str = "scaffold",
    use_mlp: bool = False,
) -> dict:
    """Run cross-validation with specified split type.

    Args:
        X: Feature matrix (fingerprints or embeddings)
        y_class: Binary labels
        y_reg: Continuous labels
        smiles: SMILES for scaffold splitting
        X_props: Simple properties (MW, LogP, TPSA) for baseline
        split_type: "scaffold" or "random"
        use_mlp: Whether to also train MLP models

    Returns:
        Dict with classification and regression metrics across folds
    """
    _mlp_str = " + MLP" if use_mlp else ""
    logger.info(f"Running {N_FOLDS}-fold CV with {split_type} split{_mlp_str}...")

    _fold_splits: list[tuple[np.ndarray, np.ndarray]] = []
    _smiles_to_idx = {s: i for i, s in enumerate(smiles)}

    if split_type == "scaffold":
        _t0 = time.perf_counter()
        _smiles_splits = scaffold_k_fold_split(smiles, k=N_FOLDS)
        for _train_smiles, _test_smiles in _smiles_splits:
            _train_idx = np.array([_smiles_to_idx[s] for s in _train_smiles if s in _smiles_to_idx])
            _test_idx = np.array([_smiles_to_idx[s] for s in _test_smiles if s in _smiles_to_idx])
            _fold_splits.append((_train_idx, _test_idx))
        logger.info(f"  Scaffold splitting took {time.perf_counter() - _t0:.1f}s")
    else:
        _skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for _train_idx, _test_idx in _skf.split(X, y_class):
            _fold_splits.append((_train_idx, _test_idx))

    # Log class distribution per fold
    for _fold_idx, (_train_idx, _test_idx) in enumerate(_fold_splits):
        _train_pct = y_class[_train_idx].mean() * 100
        _test_pct = y_class[_test_idx].mean() * 100
        logger.debug(f"  Fold {_fold_idx + 1}: train active={_train_pct:.1f}%, test active={_test_pct:.1f}%")

    # Sequential fold execution
    _fold_results: list[dict[str, Any]] = []
    for _fold_idx, (_train_idx, _test_idx) in enumerate(_fold_splits):
        logger.info(f"  Fold {_fold_idx + 1}/{N_FOLDS}")
        _result = process_single_fold(_fold_idx, _train_idx, _test_idx, X, y_class, y_reg, X_props, use_mlp)
        _fold_results.append(_result)

        _t = _result["timings"]
        _timing_parts = [
            f"baseline={_t['baseline']:.1f}s",
            f"props={_t['props']:.1f}s",
            f"fp_cls={_t['fp_cls']:.1f}s",
            f"fp_reg={_t['fp_reg']:.1f}s",
        ]
        if use_mlp and "mlp_cls" in _t:
            _timing_parts.extend([f"mlp_cls={_t['mlp_cls']:.1f}s", f"mlp_reg={_t['mlp_reg']:.1f}s"])
        logger.info(f"    Timing: {', '.join(_timing_parts)}")

    # Aggregate results
    _class_metrics: dict[str, list[float]] = {"roc_auc": [], "avg_precision": []}
    _reg_metrics: dict[str, list[float]] = {"r2": [], "spearman_r": []}
    _dummy_class_metrics: dict[str, list[float]] = {"roc_auc": [], "avg_precision": []}
    _dummy_reg_metrics: dict[str, list[float]] = {"r2": []}
    _props_class_metrics: dict[str, list[float]] = {"roc_auc": [], "avg_precision": []}
    _props_reg_metrics: dict[str, list[float]] = {"r2": [], "spearman_r": []}
    _mlp_class_metrics: dict[str, list[float]] = {"roc_auc": [], "avg_precision": []} if use_mlp else {}
    _mlp_reg_metrics: dict[str, list[float]] = {"r2": [], "spearman_r": []} if use_mlp else {}

    for _r in _fold_results:
        for _k, _v in _r["baseline"]["classification"].items():
            _dummy_class_metrics[_k].append(_v)
        _dummy_reg_metrics["r2"].append(_r["baseline"]["regression"]["r2"])

        for _k, _v in _r["props_cls"].items():
            _props_class_metrics[_k].append(_v)
        for _k, _v in _r["props_reg"].items():
            _props_reg_metrics[_k].append(_v)

        for _k, _v in _r["cls"].items():
            _class_metrics[_k].append(_v)
        for _k, _v in _r["reg"].items():
            _reg_metrics[_k].append(_v)

        if use_mlp and "mlp_cls" in _r:
            for _k, _v in _r["mlp_cls"].items():
                _mlp_class_metrics[_k].append(_v)
            for _k, _v in _r["mlp_reg"].items():
                _mlp_reg_metrics[_k].append(_v)

    _results = {
        "classification": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in _class_metrics.items()},
        "regression": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in _reg_metrics.items()},
        "baseline_dummy": {
            "classification": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in _dummy_class_metrics.items()
            },
            "regression": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in _dummy_reg_metrics.items()
            },
        },
        "baseline_properties": {
            "classification": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in _props_class_metrics.items()
            },
            "regression": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in _props_reg_metrics.items()
            },
        },
    }

    if use_mlp and _mlp_class_metrics:
        _results["mlp"] = {
            "classification": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in _mlp_class_metrics.items()
            },
            "regression": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in _mlp_reg_metrics.items()
            },
        }

    logger.info(
        f"  XGB ROC-AUC: {_results['classification']['roc_auc']['mean']:.3f} +/- {_results['classification']['roc_auc']['std']:.3f}"
    )
    logger.info(f"  XGB R2: {_results['regression']['r2']['mean']:.3f} +/- {_results['regression']['r2']['std']:.3f}")

    return _results


@app.function
def plot_comparison(
    random_results: dict,
    scaffold_results: dict,
    structure_rep: str,
    n_compounds: int,
    sample_frac: float = 1.0,
) -> plt.Figure:
    """Create comparison plot of random vs scaffold split performance."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _rep_label = "Morgan FP" if structure_rep == "morgan" else "ChemBERTa"

    if sample_frac < 1.0:
        fig.suptitle(f"[{sample_frac:.0%} sample, n={n_compounds:,}]", fontsize=10, style="italic", color="gray")

    # Classification metrics
    _ax = axes[0]
    _metrics = ["roc_auc", "avg_precision"]
    _x = np.arange(len(_metrics))
    _width = 0.35

    _random_means = [random_results["classification"][m]["mean"] for m in _metrics]
    _random_stds = [random_results["classification"][m]["std"] for m in _metrics]
    _scaffold_means = [scaffold_results["classification"][m]["mean"] for m in _metrics]
    _scaffold_stds = [scaffold_results["classification"][m]["std"] for m in _metrics]

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
    _ax.set_title(f"A. Classification ({_rep_label})")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(["ROC-AUC", "Avg Precision"])
    _ax.legend(loc="lower right")
    _ax.set_ylim(0, 1)

    _gap = random_results["classification"]["roc_auc"]["mean"] - scaffold_results["classification"]["roc_auc"]["mean"]
    _ax.annotate(
        f"Gap: {_gap:.2f}",
        xy=(0, _scaffold_means[0]),
        xytext=(0.5, 0.3),
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    # Regression metrics
    _ax = axes[1]
    _metrics = ["r2", "spearman_r"]
    _x = np.arange(len(_metrics))

    _random_means = [random_results["regression"][m]["mean"] for m in _metrics]
    _random_stds = [random_results["regression"][m]["std"] for m in _metrics]
    _scaffold_means = [scaffold_results["regression"][m]["mean"] for m in _metrics]
    _scaffold_stds = [scaffold_results["regression"][m]["std"] for m in _metrics]

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
    _ax.set_title(f"B. Regression ({_rep_label})")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(["R^2", "Spearman rho"])
    _ax.legend(loc="lower right")
    _ax.set_ylim(0, 1)

    plt.tight_layout()
    return fig


@app.function
def generate_summary(
    random_results: dict,
    scaffold_results: dict,
    n_compounds: int,
    n_active: int,
    structure_rep: str,
    dataset: str,
    sample_frac: float = 1.0,
) -> dict:
    """Generate summary dict with all results."""
    _roc_gap = (
        random_results["classification"]["roc_auc"]["mean"] - scaffold_results["classification"]["roc_auc"]["mean"]
    )
    _r2_gap = random_results["regression"]["r2"]["mean"] - scaffold_results["regression"]["r2"]["mean"]

    _scaffold_roc = scaffold_results["classification"]["roc_auc"]["mean"]
    _roc_overestimate_pct = (_roc_gap / _scaffold_roc * 100) if _scaffold_roc > 0 else 0

    _dummy_roc = scaffold_results["baseline_dummy"]["classification"]["roc_auc"]["mean"]
    _props_roc = scaffold_results["baseline_properties"]["classification"]["roc_auc"]["mean"]
    _model_roc = _scaffold_roc

    _lift_over_dummy = _model_roc - _dummy_roc
    _lift_over_props = _model_roc - _props_roc

    return {
        "dataset": dataset,
        "structure_rep": structure_rep,
        "sample_frac": sample_frac,
        "n_compounds": n_compounds,
        "n_active": n_active,
        "n_folds": N_FOLDS,
        "use_gpu": HAS_GPU,
        "baselines": {
            "dummy_roc_auc": float(_dummy_roc),
            "properties_roc_auc": float(_props_roc),
        },
        "classification": {
            "random_split": {k: v["mean"] for k, v in random_results["classification"].items()},
            "scaffold_split": {k: v["mean"] for k, v in scaffold_results["classification"].items()},
        },
        "regression": {
            "random_split": {k: v["mean"] for k, v in random_results["regression"].items()},
            "scaffold_split": {k: v["mean"] for k, v in scaffold_results["regression"].items()},
        },
        "generalization_gap": {
            "roc_auc_drop": float(_roc_gap),
            "r2_drop": float(_r2_gap),
            "interpretation": f"Random split overestimates ROC-AUC by {_roc_overestimate_pct:.0f}%",
        },
        "lift_over_baselines": {
            "over_dummy": float(_lift_over_dummy),
            "over_properties": float(_lift_over_props),
            "interpretation": f"Fingerprints add {_lift_over_props:.3f} ROC-AUC over simple properties",
        },
    }


@app.cell
def _(
    activity_params_dropdown,
    dataset_dropdown,
    filter_dropdown,
    mlp_toggle,
    mo,
    preprocessing_dropdown,
    sample_slider,
    structure_dropdown,
):
    _dataset = dataset_dropdown.value
    _preprocessing = preprocessing_dropdown.value
    _filter_name = filter_dropdown.value
    _activity_params = activity_params_dropdown.value
    _structure_rep = structure_dropdown.value
    _sample_frac = sample_slider.value
    _use_mlp = mlp_toggle.value

    # Check data files exist
    _fp_file = DATA_DIR / "interim/compound_featurization/morgan_fp.npz"
    _chemberta_file = DATA_DIR / "interim/compound_featurization/chemberta_77m_mlm.npz"
    _needed_file = _fp_file if _structure_rep == "morgan" else _chemberta_file
    mo.stop(
        not _needed_file.exists(),
        mo.md(f"**Missing data:** `{_needed_file}` not found. Run featurization pipeline first."),
    )
    mo.stop(
        not COPAIRS_RESULTS_DB.exists(),
        mo.md(f"**Missing database:** `{COPAIRS_RESULTS_DB}` not found."),
    )
    mo.stop(
        not METADATA_DB.exists(),
        mo.md(f"**Missing database:** `{METADATA_DB}` not found."),
    )

    # Load data
    if _structure_rep == "morgan":
        _features, _feature_jcp = load_fingerprints()
    else:
        _features, _feature_jcp = load_chemberta_embeddings()

    _activity_df = load_activity_data(_dataset, _preprocessing, _filter_name, _activity_params)
    _metadata_df = load_smiles_and_properties()

    # Align
    X, y_class, y_reg, smiles, X_props = align_data(_features, _feature_jcp, _activity_df, _metadata_df)

    # Sample if requested
    if _sample_frac < 1.0:
        _n_sample = int(len(X) * _sample_frac)
        _rng = np.random.default_rng(RANDOM_STATE)
        _sample_idx = _rng.choice(len(X), size=_n_sample, replace=False)
        X = X[_sample_idx]
        y_class = y_class[_sample_idx]
        y_reg = y_reg[_sample_idx]
        smiles = smiles[_sample_idx]
        X_props = X_props[_sample_idx]
        logger.info(f"Sampled {_sample_frac:.0%} of data: {_n_sample:,} compounds")

    _n_compounds = len(X)
    _n_active = int(y_class.sum())

    mo.md(f"""
    ## Data Summary

    | | |
    |---|---|
    | Dataset | `{_dataset}` |
    | Structure | `{_structure_rep}` |
    | Sample fraction | {_sample_frac:.0%} |
    | Compounds | {_n_compounds:,} |
    | Active | {_n_active:,} ({_n_active / _n_compounds * 100:.1f}%) |
    | Features | {X.shape[1]:,} dims |
    | GPU | {"enabled" if HAS_GPU else "disabled"} |
    | MLP | {"yes" if _use_mlp else "no"} |
    """)
    return X, X_props, smiles, y_class, y_reg


@app.cell
def _(
    X,
    X_props,
    dataset_dropdown,
    mlp_toggle,
    mo,
    sample_slider,
    smiles,
    structure_dropdown,
    y_class,
    y_reg,
):
    _use_mlp = mlp_toggle.value

    # Run cross-validation with both split types
    _random_results = run_cross_validation(X, y_class, y_reg, smiles, X_props, split_type="random", use_mlp=_use_mlp)
    _scaffold_results = run_cross_validation(
        X, y_class, y_reg, smiles, X_props, split_type="scaffold", use_mlp=_use_mlp
    )

    # Store for downstream cells
    random_results = _random_results
    scaffold_results = _scaffold_results

    _dataset = dataset_dropdown.value
    _structure_rep = structure_dropdown.value
    _sample_frac = sample_slider.value
    _n_compounds = len(X)
    _n_active = int(y_class.sum())

    # Generate summary
    summary = generate_summary(
        _random_results,
        _scaffold_results,
        _n_compounds,
        _n_active,
        _structure_rep,
        _dataset,
        _sample_frac,
    )

    mo.md(f"""
    ## Cross-Validation Results

    ### Classification (ROC-AUC)

    | Model | Random Split | Scaffold Split | Gap |
    |-------|-------------|---------------|-----|
    | Dummy | {_random_results["baseline_dummy"]["classification"]["roc_auc"]["mean"]:.3f} | {_scaffold_results["baseline_dummy"]["classification"]["roc_auc"]["mean"]:.3f} | |
    | Properties | {_random_results["baseline_properties"]["classification"]["roc_auc"]["mean"]:.3f} | {_scaffold_results["baseline_properties"]["classification"]["roc_auc"]["mean"]:.3f} | |
    | XGBoost | {_random_results["classification"]["roc_auc"]["mean"]:.3f} +/- {_random_results["classification"]["roc_auc"]["std"]:.3f} | {_scaffold_results["classification"]["roc_auc"]["mean"]:.3f} +/- {_scaffold_results["classification"]["roc_auc"]["std"]:.3f} | {summary["generalization_gap"]["roc_auc_drop"]:.3f} |

    ### Regression (R^2)

    | Model | Random Split | Scaffold Split | Gap |
    |-------|-------------|---------------|-----|
    | XGBoost | {_random_results["regression"]["r2"]["mean"]:.3f} +/- {_random_results["regression"]["r2"]["std"]:.3f} | {_scaffold_results["regression"]["r2"]["mean"]:.3f} +/- {_scaffold_results["regression"]["r2"]["std"]:.3f} | {summary["generalization_gap"]["r2_drop"]:.3f} |

    ### Key Findings

    - {summary["generalization_gap"]["interpretation"]}
    - {summary["lift_over_baselines"]["interpretation"]}
    """)
    return random_results, scaffold_results, summary


@app.cell
def _(X, random_results, sample_slider, scaffold_results, structure_dropdown):
    _structure_rep = structure_dropdown.value
    _sample_frac = sample_slider.value
    _n_compounds = len(X)

    fig = plot_comparison(
        random_results,
        scaffold_results,
        _structure_rep,
        _n_compounds,
        _sample_frac,
    )
    fig
    return (fig,)


@app.cell
def _(
    dataset_dropdown,
    fig,
    mo,
    preprocessing_dropdown,
    random_results,
    sample_slider,
    scaffold_results,
    structure_dropdown,
    summary,
):
    _dataset = dataset_dropdown.value
    _preprocessing = preprocessing_dropdown.value
    _structure_rep = structure_dropdown.value
    _sample_frac = sample_slider.value

    _output_dir = OUTPUT_DIR / _dataset / _preprocessing / _structure_rep
    _output_dir.mkdir(parents=True, exist_ok=True)

    # Save results JSON
    with open(_output_dir / "random_split_results.json", "w") as _f:
        json.dump(random_results, _f, indent=2)
    with open(_output_dir / "scaffold_split_results.json", "w") as _f:
        json.dump(scaffold_results, _f, indent=2)
    with open(_output_dir / "summary.json", "w") as _f:
        json.dump(summary, _f, indent=2)

    # Save figure
    fig.savefig(_output_dir / "comparison_plot.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig.savefig(_output_dir / "comparison_plot.pdf", bbox_inches="tight")

    mo.md(f"""
    **Saved outputs to** `{_output_dir}`

    - `random_split_results.json` - detailed random split CV results
    - `scaffold_split_results.json` - detailed scaffold split CV results
    - `summary.json` - aggregated summary with generalization gaps
    - `comparison_plot.png` / `.pdf` - bar chart comparison figure
    """)
    return


@app.function
def run_phenotype_prediction(
    dataset: str = "compound_no_source7",
    preprocessing: str = "activity_no_target2",
    structure_rep: str = "morgan",
    filter_name: str = "all_sources",
    activity_params: str = "default",
    sample=0.2,
    output_dir=None,
) -> str:
    """Run XGBoost scaffold vs random split CV for phenotype prediction.

    Composes load_fingerprints/load_chemberta_embeddings, load_activity_data,
    load_smiles_and_properties, align_data, run_cross_validation,
    plot_comparison, and generate_summary. Saves comparison plot, result
    JSONs, summary.json, and .complete marker.

    Called from workflow.py via run_task.py.

    Args:
        dataset: Dataset name (e.g., "compound_no_source7")
        preprocessing: Preprocessing name (e.g., "activity_no_target2")
        structure_rep: "morgan" or "chemberta"
        filter_name: Filter name (e.g., "all_sources")
        activity_params: Activity params (e.g., "default")
        sample: Fraction of data to use (0.0-1.0). Accepts str from subprocess.
        output_dir: Output directory (default: data/processed/phenotype-prediction/...)

    Returns:
        Path to .complete marker file.
    """
    import matplotlib.pyplot as plt

    from nb00_ss_config import PROCESSED_DATA_DIR

    sample = float(sample) if isinstance(sample, str) else sample

    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "phenotype-prediction" / dataset / preprocessing / structure_rep
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load structural features
    if structure_rep == "morgan":
        _features, _feature_jcp = load_fingerprints()
    else:
        _features, _feature_jcp = load_chemberta_embeddings()

    # Load activity and metadata
    _activity_df = load_activity_data(dataset, preprocessing, filter_name, activity_params)
    _metadata_df = load_smiles_and_properties()

    # Align
    X, y_class, y_reg, _smiles, X_props = align_data(_features, _feature_jcp, _activity_df, _metadata_df)

    # Sample if requested
    _sample_frac = sample
    if _sample_frac < 1.0:
        _n_sample = int(len(X) * _sample_frac)
        _rng = np.random.default_rng(RANDOM_STATE)
        _sample_idx = _rng.choice(len(X), size=_n_sample, replace=False)
        X = X[_sample_idx]
        y_class = y_class[_sample_idx]
        y_reg = y_reg[_sample_idx]
        _smiles = _smiles[_sample_idx]
        X_props = X_props[_sample_idx]
        logger.info(f"Sampled {_sample_frac:.0%} of data: {_n_sample:,} compounds")

    _n_compounds = len(X)
    _n_active = int(y_class.sum())

    # Run cross-validation with both split types
    _random_results = run_cross_validation(X, y_class, y_reg, _smiles, X_props, split_type="random", use_mlp=False)
    _scaffold_results = run_cross_validation(X, y_class, y_reg, _smiles, X_props, split_type="scaffold", use_mlp=False)

    # Generate summary
    _summary = generate_summary(
        _random_results, _scaffold_results, _n_compounds, _n_active, structure_rep, dataset, _sample_frac
    )

    # Plot and save
    _fig = plot_comparison(_random_results, _scaffold_results, structure_rep, _n_compounds, _sample_frac)
    _fig.savefig(output_dir / "comparison_plot.png", dpi=150, bbox_inches="tight", facecolor="white")
    _fig.savefig(output_dir / "comparison_plot.pdf", bbox_inches="tight")
    plt.close(_fig)

    # Save result JSONs
    with open(output_dir / "random_split_results.json", "w") as _f:
        json.dump(_random_results, _f, indent=2)
    with open(output_dir / "scaffold_split_results.json", "w") as _f:
        json.dump(_scaffold_results, _f, indent=2)
    Path(output_dir, "summary.json").write_text(json.dumps(_summary, indent=2))

    # Write .complete marker
    _marker = output_dir / ".complete"
    _marker.touch()
    logger.success(f"Saved phenotype prediction outputs to {output_dir}")
    return str(_marker)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
