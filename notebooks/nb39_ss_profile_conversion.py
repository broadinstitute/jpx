# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anndata",
#     "numpy",
#     "pandas",
#     "scanpy",
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
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from loguru import logger

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import ANNDATA_DIR, PROJ_ROOT

    try:
        import rapids_singlecell as rsc  # noqa: F401

        HAS_GPU = True
    except ImportError:
        HAS_GPU = False


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Profile Conversion (Parquet to AnnData)

    Convert parquet profile files to h5ad format for scanpy/rapids_singlecell
    integration. Uses GPU-accelerated aggregation when available.
    Migrated from `src/jump_production/processing/profiles_to_anndata.py`.

    **Exported function:**
    - `convert_profiles(dataset, output_dir)` -
      Convert parquet profiles to well-level and perturbation-level h5ad

    **Pipeline usage:**
    ```
    pixi run -e rapids python run_task.py nb39_ss_profile_conversion convert_profiles compound_no_source7
    ```
    """)
    return


@app.function
def aggregate(adata: ad.AnnData, by: str = "JCP2022", func: str = "mean") -> ad.AnnData:
    """Aggregate profiles by perturbation identifier.

    Uses GPU-accelerated rapids_singlecell when available, falls back to scanpy.

    Args:
        adata: AnnData with well-level profiles
        by: Column in obs to group by (default: JCP2022)
        func: Aggregation function ('mean' or 'sum'; 'median' only on CPU)

    Returns:
        Aggregated AnnData with one row per unique perturbation
    """
    logger.info(f"Aggregating {adata.n_obs:,} observations by {by} using {func}")

    if HAS_GPU:
        import rapids_singlecell as rsc

        logger.info("Using GPU-accelerated aggregation (rapids_singlecell)")
        rsc.get.anndata_to_GPU(adata)
        adata_agg = rsc.get.aggregate(adata, by=by, func=func)
    else:
        logger.info("Using CPU aggregation (scanpy)")
        adata_agg = sc.get.aggregate(adata, by=by, func=func)

    # Both scanpy and rapids store result in layers[func], move to X
    layer_data = adata_agg.layers[func]
    del adata_agg.layers[func]

    # Convert to numpy array (handle cupy arrays from GPU)
    if hasattr(layer_data, "get"):
        # cupy array - use .get() to transfer to CPU
        adata_agg.X = layer_data.get().astype(np.float32)
    elif hasattr(layer_data, "toarray"):
        # sparse matrix
        adata_agg.X = layer_data.toarray().astype(np.float32)
    else:
        adata_agg.X = np.asarray(layer_data, dtype=np.float32)

    # Restore grouping column and reset index for h5ad compatibility
    adata_agg.obs[by] = adata_agg.obs.index
    adata_agg.obs.index = [str(i) for i in range(adata_agg.n_obs)]

    logger.info(f"Aggregated to {adata_agg.n_obs:,} unique perturbations")
    return adata_agg


@app.function
def convert_profiles(dataset: str, output_dir: str | None = None) -> list[str]:
    """Convert parquet profiles to AnnData h5ad (well + perturbation levels).

    Reads a parquet profile, separates Metadata_* columns into obs and
    feature columns into X, creates well-level and perturbation-level
    h5ad files.

    Args:
        dataset: Dataset name (e.g., "compound_no_source7")
        output_dir: Output directory for h5ad files (default: data/interim/anndata/)

    Returns:
        List of output h5ad paths as strings
    """
    input_path = PROJ_ROOT / "data" / "raw" / "profiles" / f"{dataset}.parquet"
    resolved_output_dir = Path(output_dir) if output_dir else ANNDATA_DIR

    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        raise FileNotFoundError(msg)

    logger.info(f"Loading parquet from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} samples with {len(df.columns)} columns")

    # Separate metadata columns (Metadata_*) from feature columns
    metadata_cols = [c for c in df.columns if c.startswith("Metadata_")]
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]
    logger.info(f"Found {len(feature_cols)} feature columns, {len(metadata_cols)} metadata columns")

    # Extract feature matrix
    X = df[feature_cols].values.astype(np.float32)

    # Extract observation metadata
    obs = df[metadata_cols].copy()
    obs.columns = [c.replace("Metadata_", "") for c in obs.columns]
    obs.index = obs.index.astype(str)

    # Convert string columns to object dtype for anndata compatibility
    for col in obs.columns:
        if obs[col].dtype == "string" or str(obs[col].dtype).startswith("String"):
            obs[col] = obs[col].astype(str)

    # Create variable annotations (feature names)
    var = pd.DataFrame(index=feature_cols)
    var.index.name = "feature"

    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs, var=var)
    logger.info(f"Created AnnData: {adata.n_obs:,} obs x {adata.n_vars} vars")

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    # Save well-level h5ad
    well_path = resolved_output_dir / f"{dataset}.h5ad"
    logger.info(f"Writing well-level to {well_path}")
    adata.write_h5ad(well_path)
    logger.info(f"Saved well-level AnnData to {well_path}")
    output_paths.append(str(well_path))

    # Create and save perturbation-level aggregated version
    adata_agg = aggregate(adata, by="JCP2022", func="mean")
    perturbation_path = resolved_output_dir / f"{dataset}_perturbation.h5ad"
    logger.info(f"Writing perturbation-level to {perturbation_path}")
    adata_agg.write_h5ad(perturbation_path)
    logger.info(f"Saved perturbation-level AnnData to {perturbation_path}")
    output_paths.append(str(perturbation_path))

    return output_paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo

    This notebook uses GPU when available for aggregation. Called from the
    pipeline via:

    ```
    pixi run -e rapids python run_task.py nb39_ss_profile_conversion \
        convert_profiles compound_no_source7
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
