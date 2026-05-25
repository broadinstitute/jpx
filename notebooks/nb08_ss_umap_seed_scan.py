# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anndata==0.12.16",
#     "duckdb==1.5.3",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==2.3.3",
#     "scanpy==1.12.1",
#     "scipy==1.17.1",
#     "python-dotenv",
#     "loguru==0.7.3",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from itertools import combinations
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import scanpy as sc
    from loguru import logger
    from matplotlib.lines import Line2D
    from scipy.spatial import procrustes

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import DEFAULT_DPI, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

    SEED_SCAN_DATA_DIR = INTERIM_DATA_DIR / "anndata" / "seed_scan"
    SEED_SCAN_OUTPUT_DIR = PROCESSED_DATA_DIR / "exploration" / "0.04"

    SIG_PALETTE = {
        "Significant": "#e41a1c",
        "Not significant": "#377eb8",
        "No data": "#cccccc",
    }


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # UMAP Seed Stability Scan

    Runs multiple UMAP seeds for cosine and euclidean metrics to verify whether
    the "encircling" pattern in cosine UMAP is reproducible or seed-dependent.

    Computes 10 UMAP embeddings (5 seeds x 2 metrics) on perturbation-level profiles,
    creates a comparison grid, and measures layout stability via Procrustes analysis.

    *Related: [GitHub Issue #17](https://github.com/broadinstitute/jpx/issues/17)*
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
    skip_compute_switch = mo.ui.switch(value=False, label="Skip compute (reuse existing h5ad)")
    mo.hstack([dataset_dropdown, skip_compute_switch])
    return dataset_dropdown, skip_compute_switch


@app.function
def compute_umap(
    adata: sc.AnnData,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> sc.AnnData:
    """Compute PCA and UMAP embedding with GPU/CPU fallback."""
    has_gpu = False
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        has_gpu = True
    except Exception:
        pass

    if has_gpu:
        try:
            import rapids_singlecell as rsc
        except ImportError:
            has_gpu = False

    if has_gpu:
        import rapids_singlecell as rsc

        logger.info("Transferring data to GPU")
        rsc.get.anndata_to_GPU(adata)

        logger.info(f"Computing PCA with {n_pcs} components (GPU)")
        rsc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1), random_state=random_state)

        cagra_metrics = {"euclidean", "sqeuclidean", "inner_product"}
        if metric in cagra_metrics:
            logger.info(f"Computing neighbors: metric={metric} (GPU, CAGRA)")
            rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric, algorithm="cagra")
        else:
            logger.info(f"Computing neighbors: metric={metric} (GPU, IVF-Flat)")
            rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric, algorithm="ivfflat")

        logger.info(f"Computing UMAP with min_dist={min_dist} (GPU)")
        rsc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

        logger.info("Transferring data back to CPU")
        rsc.get.anndata_to_CPU(adata)
    else:
        logger.info(f"Computing PCA with {n_pcs} components (CPU)")
        sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1), random_state=random_state)

        logger.info(f"Computing neighbors: metric={metric} (CPU)")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric)

        logger.info(f"Computing UMAP with min_dist={min_dist} (CPU)")
        sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

    return adata


@app.function
def save_minimal_h5ad(adata: sc.AnnData, output_path: Path) -> None:
    """Save minimal h5ad with only obs and obsm (no X matrix)."""
    adata_minimal = sc.AnnData(
        obs=adata.obs.copy(),
        obsm={"X_pca": adata.obsm["X_pca"], "X_umap": adata.obsm["X_umap"]},
    )
    if "pca" in adata.uns:
        adata_minimal.uns["pca"] = adata.uns["pca"]
    if "umap" in adata.uns:
        adata_minimal.uns["umap"] = adata.uns["umap"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_minimal.write_h5ad(output_path)
    logger.info(f"Saved: {output_path}")


@app.function
def compute_layout_similarity(
    results: dict[tuple[str, int], sc.AnnData],
    metrics: list[str],
    seeds: list[int],
) -> dict[str, dict[str, float]]:
    """Procrustes similarity between seed pairs within each metric.

    Returns mean Procrustes correlation per metric (1.0 = identical layout).
    """
    similarities = {}

    for metric in metrics:
        correlations = []
        seed_adatas = [(seed, results[(metric, seed)]) for seed in seeds]

        for (seed1, adata1), (seed2, adata2) in combinations(seed_adatas, 2):
            coords1 = adata1.obsm["X_umap"]
            coords2 = adata2.obsm["X_umap"]
            _, mtx2_aligned, _ = procrustes(coords1, coords2)
            corr = np.corrcoef(coords1.flatten(), mtx2_aligned.flatten())[0, 1]
            correlations.append(corr)

        similarities[metric] = {
            "mean_correlation": float(np.mean(correlations)),
            "std_correlation": float(np.std(correlations)),
            "min_correlation": float(np.min(correlations)),
        }

    return similarities


@app.cell
def _(dataset_dropdown, mo, skip_compute_switch):
    from nb03_ss_profiles import join_activity, load_profiles

    seeds = [42, 123, 456, 789, 1000]
    umap_metrics = ["cosine", "euclidean"]
    dataset = dataset_dropdown.value
    results: dict[tuple[str, int], sc.AnnData] = {}

    if skip_compute_switch.value:
        for _metric in umap_metrics:
            for _seed in seeds:
                _h5ad_path = SEED_SCAN_DATA_DIR / f"{dataset}_perturbation_{_metric}_seed{_seed}_umap.h5ad"
                if not _h5ad_path.exists():
                    mo.stop(True, mo.md(f"**Missing:** `{_h5ad_path}`. Turn off skip-compute."))
                _adata = sc.read_h5ad(_h5ad_path)
                _adata = join_activity(_adata, dataset)
                results[(_metric, _seed)] = _adata
        logger.success(f"Loaded {len(results)} existing h5ad files")
    else:
        _adata_base = load_profiles(dataset, level="perturbation")
        logger.info(f"Loaded {_adata_base.n_obs:,} obs x {_adata_base.n_vars} vars")

        _total = len(seeds) * len(umap_metrics)
        for _idx, (_metric, _seed) in enumerate([(m, s) for m in umap_metrics for s in seeds], start=1):
            logger.info(f"[{_idx}/{_total}] UMAP: metric={_metric}, seed={_seed}")
            _adata_result = compute_umap(_adata_base.copy(), metric=_metric, random_state=_seed)

            _h5ad_path = SEED_SCAN_DATA_DIR / f"{dataset}_perturbation_{_metric}_seed{_seed}_umap.h5ad"
            save_minimal_h5ad(_adata_result, _h5ad_path)

            _adata_result = join_activity(_adata_result, dataset)
            results[(_metric, _seed)] = _adata_result

        logger.success(f"Completed {_total} UMAP computations")

    mo.md(f"**Computed {len(results)} UMAP embeddings** for {dataset}")
    return dataset, results, seeds, umap_metrics


@app.cell
def _(
    dataset,
    mo,
    results: dict[tuple[str, int], sc.AnnData],
    seeds,
    umap_metrics,
):
    from nb04_ss_visualization import compute_umap_bounds

    _n_metrics = len(umap_metrics)
    _n_seeds = len(seeds)
    grid_fig, _axes = plt.subplots(
        _n_metrics,
        _n_seeds,
        figsize=(4 * _n_seeds, 4 * _n_metrics),
        squeeze=False,
    )

    _metric_bounds = {}
    for _metric in umap_metrics:
        _adata = results[(_metric, seeds[0])]
        _metric_bounds[_metric] = compute_umap_bounds(_adata)

    for _i, _metric in enumerate(umap_metrics):
        _xlim, _ylim = _metric_bounds[_metric]
        for _j, _seed in enumerate(seeds):
            _ax = _axes[_i, _j]
            _adata = results[(_metric, _seed)]

            if "phenotypic_significant" not in _adata.obs.columns:
                if "below_corrected_p" in _adata.obs.columns:
                    _adata.obs["phenotypic_significant"] = _adata.obs["below_corrected_p"].map(
                        {"True": "Significant", "False": "Not significant", "NA": "No data"}
                    )
                else:
                    _adata.obs["phenotypic_significant"] = "No data"

            sc.pl.umap(
                _adata,
                color="phenotypic_significant",
                ax=_ax,
                show=False,
                title="",
                palette=SIG_PALETTE,
                legend_loc=None,
            )
            _ax.set_xlim(_xlim)
            _ax.set_ylim(_ylim)
            _ax.set_xlabel("")
            _ax.set_ylabel("")

            if _i == 0:
                _ax.set_title(f"seed={_seed}", fontsize=12)
            if _j == 0:
                _ax.set_ylabel(_metric, fontsize=14, fontweight="bold")

    _legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SIG_PALETTE["Significant"],
            markersize=8,
            label="Significant (p<0.05)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SIG_PALETTE["Not significant"],
            markersize=8,
            label="Not significant",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SIG_PALETTE["No data"],
            markersize=8,
            label="No activity data",
        ),
    ]
    grid_fig.legend(
        handles=_legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=10,
        frameon=False,
    )
    grid_fig.suptitle(
        f"UMAP Seed Stability: {dataset}\n(perturbation-level, colored by phenotypic significance)",
        fontsize=14,
        fontweight="bold",
        y=1.06,
    )
    plt.tight_layout()

    _output_path = SEED_SCAN_OUTPUT_DIR / dataset / "seed_comparison.png"
    _output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_fig.savefig(_output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    logger.info(f"Saved: {_output_path}")

    mo.md(f"**Saved:** `{_output_path}`")
    grid_fig
    return


@app.cell
def _(mo, results: dict[tuple[str, int], sc.AnnData], seeds, umap_metrics):
    _similarities = compute_layout_similarity(results, umap_metrics, seeds)

    _rows = []
    for _metric, _stats in _similarities.items():
        _rows.append(
            {
                "Metric": _metric,
                "Mean correlation": f"{_stats['mean_correlation']:.4f}",
                "Std": f"{_stats['std_correlation']:.4f}",
                "Min": f"{_stats['min_correlation']:.4f}",
            }
        )

    mo.md(
        "## Procrustes Seed Stability\n\n"
        "Correlation close to 1.0 means layouts are stable across seeds.\n\n" + mo.as_html(mo.ui.table(_rows)).text
    )
    return


@app.function
def run_seed_scan(dataset: str = "compound_no_source7", output_dir=None) -> str:
    """Run UMAP seed stability scan for a dataset.

    Loads perturbation-level profiles, computes UMAP embeddings for multiple
    seeds and metrics, creates a comparison grid, and saves seed_comparison.png.
    Called from workflow.py via run_task.py in the rapids pixi env.
    """
    from nb03_ss_profiles import join_activity, load_profiles
    from nb04_ss_visualization import compute_umap_bounds

    seeds = [42, 123, 456, 789, 1000]
    umap_metrics = ["cosine", "euclidean"]

    if output_dir is None:
        _output_dir = SEED_SCAN_OUTPUT_DIR / dataset
    else:
        _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)

    # Compute UMAPs
    adata_base = load_profiles(dataset, level="perturbation")
    logger.info(f"Loaded {adata_base.n_obs:,} obs x {adata_base.n_vars} vars")

    results: dict[tuple[str, int], sc.AnnData] = {}
    total = len(seeds) * len(umap_metrics)
    for idx, (metric, seed) in enumerate([(m, s) for m in umap_metrics for s in seeds], start=1):
        logger.info(f"[{idx}/{total}] UMAP: metric={metric}, seed={seed}")
        adata_result = compute_umap(adata_base.copy(), metric=metric, random_state=seed)

        h5ad_path = SEED_SCAN_DATA_DIR / f"{dataset}_perturbation_{metric}_seed{seed}_umap.h5ad"
        save_minimal_h5ad(adata_result, h5ad_path)

        adata_result = join_activity(adata_result, dataset)
        results[(metric, seed)] = adata_result

    logger.success(f"Completed {total} UMAP computations")

    # Create comparison grid
    n_metrics = len(umap_metrics)
    n_seeds = len(seeds)
    grid_fig, axes = plt.subplots(
        n_metrics,
        n_seeds,
        figsize=(4 * n_seeds, 4 * n_metrics),
        squeeze=False,
    )

    metric_bounds = {}
    for metric in umap_metrics:
        metric_bounds[metric] = compute_umap_bounds(results[(metric, seeds[0])])

    for i, metric in enumerate(umap_metrics):
        xlim, ylim = metric_bounds[metric]
        for j, seed in enumerate(seeds):
            ax = axes[i, j]
            adata = results[(metric, seed)]

            if "phenotypic_significant" not in adata.obs.columns:
                if "below_corrected_p" in adata.obs.columns:
                    adata.obs["phenotypic_significant"] = adata.obs["below_corrected_p"].map(
                        {"True": "Significant", "False": "Not significant", "NA": "No data"}
                    )
                else:
                    adata.obs["phenotypic_significant"] = "No data"

            sc.pl.umap(
                adata,
                color="phenotypic_significant",
                ax=ax,
                show=False,
                title="",
                palette=SIG_PALETTE,
                legend_loc=None,
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel("")
            ax.set_ylabel("")
            if i == 0:
                ax.set_title(f"seed={seed}", fontsize=12)
            if j == 0:
                ax.set_ylabel(metric, fontsize=14, fontweight="bold")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SIG_PALETTE["Significant"],
            markersize=8,
            label="Significant (p<0.05)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SIG_PALETTE["Not significant"],
            markersize=8,
            label="Not significant",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SIG_PALETTE["No data"],
            markersize=8,
            label="No activity data",
        ),
    ]
    grid_fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=10,
        frameon=False,
    )
    grid_fig.suptitle(
        f"UMAP Seed Stability: {dataset}\n(perturbation-level, colored by phenotypic significance)",
        fontsize=14,
        fontweight="bold",
        y=1.06,
    )
    plt.tight_layout()

    output_path = _output_dir / "seed_comparison.png"
    grid_fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(grid_fig)
    logger.success(f"Saved: {output_path}")

    # Log layout similarity
    similarities = compute_layout_similarity(results, umap_metrics, seeds)
    for metric, stats in similarities.items():
        logger.info(f"  {metric}: mean_corr={stats['mean_correlation']:.4f} std={stats['std_correlation']:.4f}")

    return str(output_path)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
