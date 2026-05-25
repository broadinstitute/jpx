# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anndata==0.12.16",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==2.3.3",
#     "python-dotenv",
#     "loguru==0.7.3",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from collections import OrderedDict
    from pathlib import Path

    import anndata as ad
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from loguru import logger
    from scipy.stats import chi2

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import ANNDATA_DIR, DEFAULT_DPI, PROCESSED_DATA_DIR

    OUTPUT_DIR = PROCESSED_DATA_DIR / "batch-effect"

    DATASETS = OrderedDict(
        [
            ("CP_no_s7", "compound_no_source7"),
            ("DL_no_s7", "compound_DL_CPCNN_no_source7"),
            ("CP_with_s7", "compound_with_source7"),
            ("DL_with_s7", "compound_DL_CPCNN_with_source7"),
        ]
    )

    FEATURE_COLORS = {"CP": "#2171b5", "DL": "#e6550d"}


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Batch Effect: CP vs DL x Source 7

    Computes local batch mixing metrics on X_pca (50-dim) for 4 dataset variants:
    CP no_s7 / CP with_s7 / DL no_s7 / DL with_s7.

    **Metrics** (on X_pca via k-NN graph, batch_key="Source", higher = better mixing):
    - **iLISI** - Inverse Simpson on k-NN batch distribution (median, normalized)
    - **kBET acceptance** - Chi-squared test of batch proportions in k-NN neighborhoods

    Both are local neighborhood metrics from the scIB benchmark (Luecken et al. 2022).

    **Requires GPU** (rapids_singlecell for kNN).
    Run with: `pixi run -e rapids marimo run notebooks/nb09_ss_source7_batch.py`
    """)
    return


@app.cell
def _(mo):
    n_neighbors_slider = mo.ui.slider(
        start=30,
        stop=150,
        step=10,
        value=90,
        label="k (neighbors)",
    )
    subsample_kbet_slider = mo.ui.slider(
        start=10_000,
        stop=100_000,
        step=10_000,
        value=50_000,
        label="kBET subsample",
    )
    mo.hstack([n_neighbors_slider, subsample_kbet_slider])
    return n_neighbors_slider, subsample_kbet_slider


@app.function
def compute_knn_graph(adata: ad.AnnData, n_neighbors: int = 90) -> ad.AnnData:
    """Compute k-NN graph on X_pca using GPU via rapids_singlecell."""
    import rapids_singlecell as rsc

    logger.info(f"Computing kNN graph (k={n_neighbors}) on {adata.n_obs:,} obs")
    adata_tmp = ad.AnnData(X=adata.obsm["X_pca"].copy())
    adata_tmp.obsm["X_pca"] = adata_tmp.X.copy()
    rsc.get.anndata_to_GPU(adata_tmp)
    rsc.pp.neighbors(
        adata_tmp,
        n_neighbors=n_neighbors,
        use_rep="X_pca",
        metric="cosine",
        algorithm="ivfflat",
    )
    rsc.get.anndata_to_CPU(adata_tmp)
    adata.obsp["distances"] = adata_tmp.obsp["distances"]
    del adata_tmp
    logger.info("kNN graph computed")
    return adata


@app.function
def compute_ilisi(adata: ad.AnnData, batch_key: str = "Source") -> tuple[float, float]:
    """Integration Local Inverse Simpson's Index.

    Returns (median_lisi, normalized_lisi) where normalized = (median - 1) / (n_batches - 1).
    """
    labels = (
        adata.obs[batch_key].cat.codes.values
        if hasattr(adata.obs[batch_key], "cat")
        else pd.Categorical(adata.obs[batch_key]).codes
    )
    n_batches = len(np.unique(labels))

    dist_matrix = adata.obsp["distances"]
    n_cells = dist_matrix.shape[0]
    indptr = dist_matrix.indptr
    indices = dist_matrix.indices

    lisi_values = np.empty(n_cells, dtype=np.float64)
    for i in range(n_cells):
        neighbor_idx = indices[indptr[i] : indptr[i + 1]]
        if len(neighbor_idx) == 0:
            lisi_values[i] = 1.0
            continue
        neighbor_labels = labels[neighbor_idx]
        _, counts = np.unique(neighbor_labels, return_counts=True)
        props = counts / counts.sum()
        lisi_values[i] = 1.0 / np.sum(props**2)

    median_lisi = float(np.median(lisi_values))
    normalized_lisi = (median_lisi - 1) / (n_batches - 1) if n_batches > 1 else 0.0
    logger.info(f"iLISI: median={median_lisi:.4f}, normalized={normalized_lisi:.4f} (n_batches={n_batches})")
    return median_lisi, normalized_lisi


@app.function
def compute_kbet(
    adata: ad.AnnData,
    batch_key: str = "Source",
    subsample: int = 50_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> float:
    """kBET acceptance rate via chi-squared test on k-NN batch proportions."""
    labels = (
        adata.obs[batch_key].cat.codes.values
        if hasattr(adata.obs[batch_key], "cat")
        else pd.Categorical(adata.obs[batch_key]).codes
    )
    n_batches = len(np.unique(labels))
    dist_matrix = adata.obsp["distances"]
    n_cells = dist_matrix.shape[0]

    _, global_counts = np.unique(labels, return_counts=True)
    global_props = global_counts / global_counts.sum()

    rng = np.random.default_rng(seed)
    n_query = min(subsample, n_cells)
    query_idx = rng.choice(n_cells, size=n_query, replace=False)

    indptr = dist_matrix.indptr
    indices = dist_matrix.indices
    df = n_batches - 1

    accepted = 0
    for i in query_idx:
        neighbor_idx = indices[indptr[i] : indptr[i + 1]]
        k = len(neighbor_idx)
        if k == 0:
            continue
        neighbor_labels = labels[neighbor_idx]
        obs_counts = np.bincount(neighbor_labels, minlength=n_batches)
        exp_counts = global_props * k
        mask = exp_counts > 0
        chi2_stat = np.sum((obs_counts[mask] - exp_counts[mask]) ** 2 / exp_counts[mask])
        p_value = chi2.sf(chi2_stat, df)
        if p_value > alpha:
            accepted += 1

    acceptance_rate = accepted / n_query
    logger.info(f"kBET acceptance={acceptance_rate:.4f} (n_query={n_query:,})")
    return acceptance_rate


@app.function
def process_dataset(
    label: str,
    dataset: str,
    n_neighbors: int,
    subsample_kbet: int,
) -> dict:
    """Load h5ad, compute kNN, run iLISI and kBET for one dataset."""
    h5ad_path = ANNDATA_DIR / f"{dataset}_cosine_all_umap.h5ad"
    logger.info(f"Loading {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    logger.info(f"  {adata.n_obs:,} obs, obsm keys: {list(adata.obsm.keys())}")

    if not hasattr(adata.obs["Source"], "cat"):
        adata.obs["Source"] = pd.Categorical(adata.obs["Source"])
    n_sources = adata.obs["Source"].nunique()

    adata = compute_knn_graph(adata, n_neighbors=n_neighbors)

    feature_type = "CP" if label.startswith("CP") else "DL"
    has_source7 = "with_s7" in label

    logger.info(f"--- {label}: Computing iLISI ---")
    ilisi_median, ilisi_norm = compute_ilisi(adata)

    logger.info(f"--- {label}: Computing kBET ---")
    kbet = compute_kbet(adata, subsample=subsample_kbet)

    return {
        "name": label,
        "feature_type": feature_type,
        "source7": has_source7,
        "n_obs": adata.n_obs,
        "n_sources": n_sources,
        "iLISI": ilisi_median,
        "iLISI_norm": ilisi_norm,
        "kBET_acceptance": kbet,
    }


@app.cell
def _(mo, n_neighbors_slider, subsample_kbet_slider):
    _results_list = []
    _skipped = []
    for _label, _dataset in DATASETS.items():
        _h5ad_path = ANNDATA_DIR / f"{_dataset}_cosine_all_umap.h5ad"
        if not _h5ad_path.exists():
            logger.warning(f"Skipping {_label}: {_h5ad_path} not found")
            _skipped.append(_label)
            continue
        logger.info(f"=== Processing {_label} ({_dataset}) ===")
        _row = process_dataset(_label, _dataset, n_neighbors_slider.value, subsample_kbet_slider.value)
        _results_list.append(_row)

    df = pd.DataFrame(_results_list)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _csv_path = OUTPUT_DIR / "batch_effect_summary.csv"
    df.to_csv(_csv_path, index=False)
    logger.info(f"Saved CSV: {_csv_path}")

    _skip_msg = f" Skipped (missing h5ad): {', '.join(_skipped)}" if _skipped else ""
    mo.md(f"**Processed {len(_results_list)} of {len(DATASETS)} datasets.** Saved to `{_csv_path}`{_skip_msg}")
    return (df,)


@app.cell
def _(df, mo):
    mo.ui.dataframe(df)
    return


@app.cell
def _(df, mo):
    _METRIC_COLS = ["iLISI_norm", "kBET_acceptance"]
    _METRIC_LABELS = {
        "iLISI_norm": "iLISI\n(normalized)",
        "kBET_acceptance": "kBET\nacceptance",
    }

    batch_fig, _axes = plt.subplots(1, 2, figsize=(8, 4.5), sharey=False)

    _x_labels = ["no Source 7", "with Source 7"]
    _x_pos = np.array([0, 1])
    _bar_width = 0.35

    for _ax, _metric in zip(_axes, _METRIC_COLS):
        _all_vals = []
        for _i, (_feat, _color) in enumerate(FEATURE_COLORS.items()):
            _vals = []
            for _s7 in [False, True]:
                _row = df[(df["feature_type"] == _feat) & (df["source7"] == _s7)]
                _vals.append(_row[_metric].values[0] if len(_row) else 0)
            _all_vals.extend(_vals)
            _offset = -_bar_width / 2 if _i == 0 else _bar_width / 2
            _bars = _ax.bar(_x_pos + _offset, _vals, _bar_width, label=_feat, color=_color, alpha=0.85)
            for _bar, _val in zip(_bars, _vals):
                _ax.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _bar.get_height(),
                    f"{_val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        _vmin, _vmax = min(_all_vals), max(_all_vals)
        _spread = _vmax - _vmin
        if _spread < 0.02:
            _pad = max(_spread * 2, 0.005)
            _y_lo = max(0, _vmin - _pad)
            _y_hi = min(1, _vmax + _pad * 3)
        else:
            _y_lo = 0
            _y_hi = _vmax + _spread * 0.35
        _ax.set_ylim(_y_lo, _y_hi)

        _ax.set_xticks(_x_pos)
        _ax.set_xticklabels(_x_labels, fontsize=9)
        _ax.set_ylabel(_METRIC_LABELS[_metric], fontsize=9)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

    _axes[0].legend(fontsize=9, loc="upper left")
    batch_fig.suptitle("Batch Effect Metrics (Source) - CP vs DL x Source 7", fontsize=12, y=1.02)
    batch_fig.tight_layout()

    _output_path = OUTPUT_DIR / "batch_effect_comparison.png"
    batch_fig.savefig(_output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    logger.info(f"Saved: {_output_path}")

    mo.md(f"**Saved:** `{_output_path}`")
    batch_fig
    return


@app.function
def run_batch_quantification(output_dir=None) -> str:
    """Run batch effect quantification across all 4 dataset variants.

    Processes CP/DL x no_s7/with_s7, computes iLISI and kBET metrics,
    saves batch_effect_summary.csv and batch_effect_comparison.png.
    Called from workflow.py via run_task.py in the rapids pixi env.
    """
    from nb00_ss_config import DEFAULT_DPI, PROCESSED_DATA_DIR

    if output_dir is None:
        _output_dir = PROCESSED_DATA_DIR / "source7-comparison"
    else:
        _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)

    n_neighbors = 90
    subsample_kbet = 50_000

    # Process all datasets
    results_list = []
    skipped = []
    for label, dataset in DATASETS.items():
        h5ad_path = ANNDATA_DIR / f"{dataset}_cosine_all_umap.h5ad"
        if not h5ad_path.exists():
            logger.warning(f"Skipping {label}: {h5ad_path} not found")
            skipped.append(label)
            continue
        logger.info(f"=== Processing {label} ({dataset}) ===")
        row = process_dataset(label, dataset, n_neighbors, subsample_kbet)
        results_list.append(row)

    df = pd.DataFrame(results_list)
    csv_path = _output_dir / "batch_effect_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")

    if skipped:
        logger.warning(f"Skipped (missing h5ad): {', '.join(skipped)}")

    # Generate comparison plot
    METRIC_COLS = ["iLISI_norm", "kBET_acceptance"]
    METRIC_LABELS = {
        "iLISI_norm": "iLISI\n(normalized)",
        "kBET_acceptance": "kBET\nacceptance",
    }

    batch_fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), sharey=False)
    x_labels = ["no Source 7", "with Source 7"]
    x_pos = np.array([0, 1])
    bar_width = 0.35

    for ax, metric_col in zip(axes, METRIC_COLS):
        all_vals = []
        for i, (feat, color) in enumerate(FEATURE_COLORS.items()):
            vals = []
            for s7 in [False, True]:
                row = df[(df["feature_type"] == feat) & (df["source7"] == s7)]
                vals.append(row[metric_col].values[0] if len(row) else 0)
            all_vals.extend(vals)
            offset = -bar_width / 2 if i == 0 else bar_width / 2
            bars = ax.bar(x_pos + offset, vals, bar_width, label=feat, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        vmin, vmax = min(all_vals), max(all_vals)
        spread = vmax - vmin
        if spread < 0.02:
            pad = max(spread * 2, 0.005)
            y_lo = max(0, vmin - pad)
            y_hi = min(1, vmax + pad * 3)
        else:
            y_lo = 0
            y_hi = vmax + spread * 0.35
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric_col], fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(fontsize=9, loc="upper left")
    batch_fig.suptitle("Batch Effect Metrics (Source) - CP vs DL x Source 7", fontsize=12, y=1.02)
    batch_fig.tight_layout()

    png_path = _output_dir / "batch_effect_comparison.png"
    batch_fig.savefig(png_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(batch_fig)
    logger.success(f"Saved: {png_path}")

    return str(csv_path)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
