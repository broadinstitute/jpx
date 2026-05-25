# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anndata",
#     "duckdb",
#     "numpy",
#     "pandas",
#     "pyyaml",
#     "python-dotenv",
#     "loguru",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    import anndata as ad
    import duckdb
    import numpy as np
    import pandas as pd
    import yaml
    from loguru import logger

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import INTERIM_DATA_DIR, PROJ_ROOT

    try:
        import rapids_singlecell as rsc  # noqa: F401

        HAS_GPU = True
    except ImportError:
        HAS_GPU = False

    CANONICAL_METADATA = {
        "Metadata_JCP2022",
        "Metadata_Plate",
        "Metadata_Source",
        "Metadata_Well",
    }


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Batch Correction (PCA + Harmony)

    GPU-accelerated batch correction using rapids_singlecell.
    Migrated from `src/jump_production/processing/batch_correct.py`.

    **Exported function:**
    - `run_batch_correct(dataset, variant, db_path, config_path)` -
      PCA + Harmony batch correction on GPU

    **Pipeline usage:**
    ```
    pixi run -e rapids python run_task.py nb38_ss_batch_correction run_batch_correct compound_no_source7 rsc
    ```
    """)
    return


@app.function
def join_batch_metadata(df: pd.DataFrame, db_path: Path) -> pd.DataFrame:
    """Standardize metadata columns and join Metadata_Batch from DuckDB.

    Drops non-canonical metadata columns, then joins Metadata_Batch fresh
    from DuckDB for consistency. Some pre-Harmony parquets have extra
    metadata or already contain Metadata_Batch - this normalizes them.
    """
    extra_meta = [c for c in df.columns if c.startswith("Metadata_") and c not in CANONICAL_METADATA]
    if extra_meta:
        df = df.drop(columns=extra_meta)
        logger.info(f"Dropped {len(extra_meta)} extra metadata columns: {extra_meta}")

    with duckdb.connect(str(db_path), read_only=True) as con:
        plate_df = con.sql("SELECT Metadata_Source, Metadata_Plate, Metadata_Batch FROM plate").fetchdf()

    n_before = len(df)
    df = df.merge(plate_df, on=["Metadata_Source", "Metadata_Plate"], how="left")

    n_missing = df["Metadata_Batch"].isna().sum()
    if n_missing > 0:
        logger.warning(f"{n_missing:,} rows have no Metadata_Batch after join")

    logger.info(f"Joined Metadata_Batch: {n_before:,} rows, {df['Metadata_Batch'].nunique()} unique batches")
    return df


@app.function
def parquet_to_anndata(df: pd.DataFrame) -> ad.AnnData:
    """Convert a profile DataFrame to AnnData.

    Metadata_* columns go to obs (with prefix stripped),
    everything else becomes the feature matrix X.
    """
    metadata_cols = [c for c in df.columns if c.startswith("Metadata_")]
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]

    obs = df[metadata_cols].copy()
    obs.columns = [c.replace("Metadata_", "") for c in obs.columns]
    obs.index = obs.index.astype(str)

    for col in obs.columns:
        if obs[col].dtype == "string" or str(obs[col].dtype).startswith("String"):
            obs[col] = obs[col].astype(str)

    X = df[feature_cols].values.astype(np.float32)
    var = pd.DataFrame(index=feature_cols)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    logger.info(f"AnnData: {adata.n_obs:,} obs x {adata.n_vars} vars")
    return adata


@app.function
def run_harmony(
    adata: ad.AnnData,
    batch_key: str,
    n_pcs: int,
    n_clusters: int,
    max_iter_harmony: int,
) -> np.ndarray:
    """Run PCA + Harmony via rapids_singlecell (GPU required).

    Args:
        adata: AnnData with features in X and batch key(s) in obs.
            Config batch_key uses Metadata_ prefix (e.g., "Metadata_Batch");
            obs columns have prefix stripped (e.g., "Batch").
        batch_key: Column name with Metadata_ prefix (stripped internally)
        n_pcs: Number of PCA components
        n_clusters: Harmony n_clusters parameter
        max_iter_harmony: Harmony max iterations

    Returns:
        Corrected components as numpy array (n_obs, n_pcs)
    """
    import rapids_singlecell as rsc

    obs_key = batch_key.replace("Metadata_", "") if batch_key.startswith("Metadata_") else batch_key
    if obs_key not in adata.obs.columns:
        msg = f"Batch key '{obs_key}' not found in obs. Available: {list(adata.obs.columns)}"
        raise ValueError(msg)

    n_nan = np.isnan(adata.X).sum()
    if n_nan > 0:
        logger.warning(f"Replacing {n_nan:,} NaN values with 0")
        adata.X = np.nan_to_num(adata.X, nan=0.0)

    n_comps = min(n_pcs, adata.n_vars - 1)

    logger.info("Transferring data to GPU")
    rsc.get.anndata_to_GPU(adata)

    logger.info(f"Computing PCA with {n_comps} components (GPU)")
    rsc.pp.pca(adata, n_comps=n_comps)

    logger.info(f"Running Harmony on '{obs_key}' (n_clusters={n_clusters}, max_iter={max_iter_harmony})")
    rsc.pp.harmony_integrate(
        adata,
        key=obs_key,
        n_clusters=n_clusters,
        max_iter_harmony=max_iter_harmony,
    )

    logger.info("Transferring data back to CPU")
    rsc.get.anndata_to_CPU(adata)

    corrected = np.asarray(adata.obsm["X_pca_harmony"], dtype=np.float32)
    logger.info(f"Harmony complete: {corrected.shape}")
    return corrected


@app.function
def build_output_df(adata: ad.AnnData, corrected: np.ndarray) -> pd.DataFrame:
    """Build output DataFrame from corrected components + metadata.

    Restores Metadata_ prefix on obs columns and names features X_1..X_N.
    Only canonical metadata columns are kept in output.
    """
    n_pcs = corrected.shape[1]
    feature_names = [f"X_{i + 1}" for i in range(n_pcs)]

    meta_df = adata.obs.copy()
    meta_df.columns = [f"Metadata_{c}" for c in meta_df.columns]
    meta_df.index = range(len(meta_df))

    canonical = ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"]
    meta_df = meta_df[canonical]

    feat_df = pd.DataFrame(corrected, columns=feature_names)
    return pd.concat([meta_df, feat_df], axis=1)


@app.function
def run_batch_correct(
    dataset: str,
    variant: str = "rsc",
    db_path: str | None = None,
    config_path: str | None = None,
) -> str:
    """Run GPU PCA + Harmony batch correction on pre-Harmony profiles.

    Reads pre-Harmony profiles, joins batch metadata from DuckDB, runs PCA
    then Harmony via rapids_singlecell, and writes corrected PCA-Harmony
    components as parquet.

    Args:
        dataset: Dataset name (e.g., "compound_no_source7")
        variant: Output suffix (default "rsc" for rapids_singlecell Harmony)
        db_path: Path to augmented metadata DuckDB (default: standard location)
        config_path: Path to batch_correction.yaml (default: configs/batch_correction.yaml)

    Returns:
        Output parquet path as string
    """
    if not HAS_GPU:
        msg = "GPU required for batch correction. Run with: pixi run -e rapids"
        raise RuntimeError(msg)

    resolved_db = Path(db_path) if db_path else INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
    resolved_config = Path(config_path) if config_path else PROJ_ROOT / "configs" / "batch_correction.yaml"

    input_path = PROJ_ROOT / "data" / "raw" / "profiles" / f"{dataset}_pre_harmony.parquet"
    output_path = PROJ_ROOT / "data" / "raw" / "profiles" / f"{dataset}_{variant}.parquet"

    if not input_path.exists():
        msg = f"Input not found: {input_path}"
        raise FileNotFoundError(msg)
    if not resolved_db.exists():
        msg = f"Database not found: {resolved_db}"
        raise FileNotFoundError(msg)

    with open(resolved_config) as f:
        cfg = yaml.safe_load(f)
    batch_key = cfg["batch_key"]
    n_pcs = int(cfg["n_pcs"])
    n_clusters = int(cfg["n_clusters"])
    max_iter_harmony = int(cfg["max_iter_harmony"])
    logger.info(f"Config: batch_key={batch_key}, n_pcs={n_pcs}, n_clusters={n_clusters}")

    logger.info(f"Loading {input_path}")
    df = pd.read_parquet(input_path)
    n_input_features = len([c for c in df.columns if not c.startswith("Metadata_")])
    logger.info(f"Loaded {len(df):,} rows, {n_input_features} features")

    df = join_batch_metadata(df, resolved_db)
    adata = parquet_to_anndata(df)

    corrected = run_harmony(
        adata,
        batch_key=batch_key,
        n_pcs=n_pcs,
        n_clusters=n_clusters,
        max_iter_harmony=max_iter_harmony,
    )

    out_df = build_output_df(adata, corrected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    logger.info(f"Saved corrected profiles to {output_path}: {len(out_df):,} rows x {corrected.shape[1]} features")
    return str(output_path)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo

    This notebook requires GPU (rapids_singlecell). The function above is
    called from the pipeline via:

    ```
    pixi run -e rapids python run_task.py nb38_ss_batch_correction \
        run_batch_correct compound_no_source7 rsc
    ```
    """)
    return


@app.cell
def _():
    # Show GPU status
    logger.info(f"GPU available: {HAS_GPU}")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
