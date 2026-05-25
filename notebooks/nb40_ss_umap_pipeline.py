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
    import scanpy as sc
    from loguru import logger

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import ANNDATA_DIR
    from nb03_ss_profiles import join_activity, load_profiles

    try:
        import rapids_singlecell as rsc  # noqa: F401

        HAS_GPU = True
    except ImportError:
        HAS_GPU = False

    # Reduce scanpy verbosity
    sc.settings.verbosity = 1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # UMAP Pipeline (PCA + Neighbors + UMAP)

    GPU-accelerated UMAP embedding computation using rapids_singlecell
    with CPU fallback via scanpy.
    Migrated from `src/jump_production/processing/umap.py`.

    **Exported function:**
    - `compute_umap(dataset, metric, filter_type, output_dir, n_neighbors, n_pcs, min_dist, seed)` -
      PCA + neighbors + UMAP at well and perturbation levels

    **Pipeline usage:**
    ```
    pixi run -e rapids python run_task.py nb40_ss_umap_pipeline compute_umap compound_no_source7 cosine all
    ```
    """)
    return


@app.function
def run_umap_embedding(
    adata: ad.AnnData,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    min_dist: float = 0.5,
    metric: str = "cosine",
    random_state: int = 0,
) -> ad.AnnData:
    """Compute PCA + neighbors + UMAP embedding.

    Uses GPU-accelerated rapids_singlecell when available,
    falls back to scanpy on CPU.
    """
    if HAS_GPU:
        import rapids_singlecell as rsc

        logger.info("Transferring data to GPU")
        rsc.get.anndata_to_GPU(adata)

        logger.info(f"Computing PCA with {n_pcs} components (GPU)")
        rsc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1), random_state=random_state)

        # CAGRA only supports euclidean, sqeuclidean, inner_product
        # IVF-Flat supports more metrics including cosine
        cagra_metrics = {"euclidean", "sqeuclidean", "inner_product"}
        if metric in cagra_metrics:
            logger.info(f"Computing neighbors n_neighbors={n_neighbors}, metric={metric} (GPU, CAGRA)")
            rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric, algorithm="cagra")
        else:
            logger.info(f"Computing neighbors n_neighbors={n_neighbors}, metric={metric} (GPU, IVF-Flat)")
            rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric, algorithm="ivfflat")

        logger.info(f"Computing UMAP with min_dist={min_dist} (GPU)")
        rsc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

        logger.info("Transferring data back to CPU")
        rsc.get.anndata_to_CPU(adata)
    else:
        logger.info(f"Computing PCA with {n_pcs} components (CPU)")
        sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1), random_state=random_state)

        logger.info(f"Computing neighbors n_neighbors={n_neighbors}, metric={metric} (CPU)")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric)

        logger.info(f"Computing UMAP with min_dist={min_dist} (CPU)")
        sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

    return adata


@app.function
def process_level(
    dataset: str,
    level: str,
    output_dir: Path,
    n_neighbors: int,
    n_pcs: int,
    min_dist: float,
    metric: str,
    seed: int,
    filter_active: bool = False,
) -> str:
    """Process a single level (well or perturbation) through UMAP pipeline.

    Saves a minimal h5ad with only obs metadata and obsm embeddings (no X
    matrix) for efficient storage. Activity data is joined at visualization
    time, not here.

    Args:
        dataset: Dataset name
        level: 'well' or 'perturbation'
        output_dir: Output directory for h5ad files
        n_neighbors: Number of neighbors for UMAP
        n_pcs: Number of PCA components
        min_dist: UMAP min_dist parameter
        metric: Distance metric for neighbors
        seed: Random seed
        filter_active: If True, filter to phenotypically active compounds

    Returns:
        Output h5ad path as string
    """
    prefix = f"{dataset}_perturbation" if level == "perturbation" else dataset
    filter_name = "active" if filter_active else "all"
    h5ad_output = output_dir / f"{prefix}_{metric}_{filter_name}_umap.h5ad"

    logger.info(f"Processing {level}-level for dataset: {dataset}")
    adata = load_profiles(dataset, level=level)

    # Filter to phenotypically active compounds if requested
    if filter_active:
        logger.info("Filtering to phenotypically active compounds...")
        adata = join_activity(adata, dataset)
        if "below_corrected_p" not in adata.obs.columns:
            msg = "Activity data not found - cannot filter. Run activity analysis first."
            raise RuntimeError(msg)
        mask = adata.obs["below_corrected_p"] == "True"
        n_before = adata.n_obs
        adata = adata[mask].copy()
        logger.info(f"Filtered to {adata.n_obs:,} active observations (from {n_before:,})")

        # Drop activity columns - joined fresh at visualization time
        activity_cols = [
            "mean_average_precision",
            "mean_normalized_average_precision",
            "p_value",
            "corrected_p_value",
            "below_corrected_p",
        ]
        adata.obs = adata.obs.drop(columns=[c for c in activity_cols if c in adata.obs.columns])

    logger.info(f"Dataset shape: {adata.n_obs:,} obs x {adata.n_vars} vars")

    # Compute UMAP
    adata = run_umap_embedding(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )

    # Create minimal AnnData for storage (no X matrix, only obs + obsm)
    output_dir.mkdir(parents=True, exist_ok=True)
    adata_minimal = sc.AnnData(
        obs=adata.obs,
        obsm={"X_pca": adata.obsm["X_pca"], "X_umap": adata.obsm["X_umap"]},
    )

    if "pca" in adata.uns:
        adata_minimal.uns["pca"] = adata.uns["pca"]
    if "umap" in adata.uns:
        adata_minimal.uns["umap"] = adata.uns["umap"]

    logger.info(f"Saving minimal UMAP h5ad to {h5ad_output}")
    logger.info(f"  obs: {adata_minimal.n_obs:,} x {len(adata_minimal.obs.columns)} columns")
    logger.info(f"  obsm: {list(adata_minimal.obsm.keys())}")
    adata_minimal.write_h5ad(h5ad_output)

    return str(h5ad_output)


@app.function
def compute_umap(
    dataset: str,
    metric: str = "cosine",
    filter_type: str = "all",
    output_dir: str | None = None,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    min_dist: float = 0.5,
    seed: int = 0,
) -> list[str]:
    """Compute GPU UMAP + PCA for well and perturbation levels.

    Loads h5ad profiles, computes PCA + neighbors + UMAP using
    rapids_singlecell (GPU) or scanpy (CPU fallback). Optionally
    filters to phenotypically active compounds.

    Saves minimal h5ad files containing only obs + obsm (no X matrix)
    for efficient storage. Metadata is joined at visualization time.

    Args:
        dataset: Dataset name (e.g., "compound_no_source7")
        metric: Distance metric for neighbors (default "cosine")
        filter_type: "all" or "active" (filter to active compounds)
        output_dir: Output directory for h5ad files (default: data/interim/anndata/)
        n_neighbors: Number of neighbors for UMAP (default 15)
        n_pcs: Number of PCA components (default 50)
        min_dist: UMAP min_dist parameter (default 0.5)
        seed: Random seed (default 0)

    Returns:
        List of output h5ad paths as strings
    """
    # Handle string-to-type conversion for pipeline argv calls
    n_neighbors = int(n_neighbors)
    n_pcs = int(n_pcs)
    min_dist = float(min_dist)
    seed = int(seed)

    resolved_output_dir = Path(output_dir) if output_dir else ANNDATA_DIR
    filter_active = filter_type == "active"

    output_paths = []
    for level in ["well", "perturbation"]:
        path = process_level(
            dataset=dataset,
            level=level,
            output_dir=resolved_output_dir,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            min_dist=min_dist,
            metric=metric,
            seed=seed,
            filter_active=filter_active,
        )
        output_paths.append(path)

    logger.info("Done computing UMAP for both well-level and perturbation-level")
    return output_paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo

    This notebook uses GPU when available for PCA, neighbors, and UMAP.
    Called from the pipeline via:

    ```
    pixi run -e rapids python run_task.py nb40_ss_umap_pipeline \
        compute_umap compound_no_source7 cosine all
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
