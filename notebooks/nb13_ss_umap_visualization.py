# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anndata",
#     "duckdb",
#     "loguru",
#     "matplotlib",
#     "numpy",
#     "python-dotenv",
#     "scanpy",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import scanpy as sc
    from loguru import logger

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import DEFAULT_DPI, PROCESSED_DATA_DIR
    from nb03_ss_profiles import join_activity, load_umap
    from nb04_ss_visualization import compute_umap_bounds

    sc.settings.verbosity = 1
    sc.settings.set_figure_params(dpi=DEFAULT_DPI, frameon=False, figsize=(8, 8))

    UMAP_OUTPUT_DIR = PROCESSED_DATA_DIR / "umap"

    SIG_PALETTE = {
        "Significant": "#e41a1c",
        "Not significant": "#377eb8",
        "No data": "#cccccc",
    }


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # UMAP Visualization

    Interactive UMAP visualizations of JUMP compound profiles colored by
    different annotations: source, target presence, phenotypic activity
    significance, normalized mAP, and cell counts.

    Loads precomputed UMAP embeddings from h5ad files and joins metadata
    and activity data from DuckDB at visualization time. Supports both
    well-level (preserves Source information) and perturbation-level
    (aggregated by compound) views.

    Plots can be saved to `data/processed/umap/{dataset}/{metric}_{filter}/`.

    *Requires UMAP embeddings computed via `jump_production.processing.umap`.*
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
    level_dropdown = mo.ui.dropdown(
        options=["well", "perturbation"],
        value="perturbation",
        label="Level",
    )
    metric_dropdown = mo.ui.dropdown(
        options=["cosine", "euclidean"],
        value="cosine",
        label="Metric",
    )
    filter_dropdown = mo.ui.dropdown(
        options=["all", "active"],
        value="all",
        label="Filter",
    )
    clip_switch = mo.ui.switch(value=True, label="Clip outliers (IQR)")
    mo.hstack([dataset_dropdown, level_dropdown, metric_dropdown, filter_dropdown, clip_switch])
    return (clip_switch, dataset_dropdown, filter_dropdown, level_dropdown, metric_dropdown)


@app.cell
def _(dataset_dropdown, filter_dropdown, level_dropdown, metric_dropdown, mo):
    _adata = load_umap(
        dataset_dropdown.value,
        level=level_dropdown.value,
        metric=metric_dropdown.value,
        filter_name=filter_dropdown.value,
    )

    mo.stop(
        "X_umap" not in _adata.obsm,
        mo.md(f"**Error:** `X_umap` not found for {dataset_dropdown.value}. Run UMAP computation first."),
    )

    _adata = join_activity(
        _adata,
        dataset_dropdown.value,
        preprocessing="activity_no_target2",
        filter_name="all_sources",
    )

    # Derive annotation columns used by downstream plots
    if "repurposing_target" in _adata.obs.columns and "has_repurposing_target" not in _adata.obs.columns:
        _adata.obs["has_repurposing_target"] = _adata.obs["repurposing_target"].notna().astype(str)
    if "Uniprot_target" in _adata.obs.columns and "has_uniprot_target" not in _adata.obs.columns:
        _adata.obs["has_uniprot_target"] = _adata.obs["Uniprot_target"].notna().astype(str)
    if "below_corrected_p" in _adata.obs.columns and "phenotypic_significant" not in _adata.obs.columns:
        _adata.obs["phenotypic_significant"] = _adata.obs["below_corrected_p"].map(
            {"True": "Significant", "False": "Not significant", "NA": "No data"}
        )

    adata = _adata

    mo.md(f"""
    ## Loaded UMAP data

    **Dataset:** {dataset_dropdown.value} | **Level:** {level_dropdown.value}
    | **Metric:** {metric_dropdown.value} | **Filter:** {filter_dropdown.value}

    **Shape:** {adata.n_obs:,} observations x {adata.n_vars} variables

    **obs columns:** {", ".join(adata.obs.columns.tolist())}
    """)
    return (adata,)


@app.cell
def _(adata, clip_switch):
    _xlim, _ylim, _clip_label = None, None, ""
    if clip_switch.value:
        _xlim, _ylim = compute_umap_bounds(adata)
        _coords = adata.obsm["X_umap"]
        _n_outside = np.sum(
            (_coords[:, 0] < _xlim[0])
            | (_coords[:, 0] > _xlim[1])
            | (_coords[:, 1] < _ylim[0])
            | (_coords[:, 1] > _ylim[1])
        )
        _pct_outside = 100 * _n_outside / len(_coords)
        _clip_label = f" [{_pct_outside:.1f}% clipped]" if _n_outside > 0 else ""
        logger.info(f"IQR clipping: {_n_outside} points ({_pct_outside:.2f}%) outside bounds")

    umap_xlim = _xlim
    umap_ylim = _ylim
    clip_label = _clip_label
    return (clip_label, umap_xlim, umap_ylim)


@app.function
def apply_bounds(ax, xlim, ylim):
    """Apply axis limits if set."""
    if xlim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


@app.function
def make_title(dataset: str, level: str, suffix: str, clip_label: str) -> str:
    """Build a plot title with dataset, level, suffix, and clip info."""
    _level_label = "Perturbation" if level == "perturbation" else "Well"
    return f"{dataset} ({_level_label}) - {suffix}{clip_label}"


# -- Generate all UMAP plots --


@app.cell
def _(adata, clip_label, dataset_dropdown, level_dropdown, umap_xlim, umap_ylim):
    _dataset = dataset_dropdown.value
    _level = level_dropdown.value
    _plots = {}

    # Source coloring (well-level only)
    if _level == "well" and "Source" in adata.obs.columns:
        _fig, _ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color="Source",
            ax=_ax,
            show=False,
            title=make_title(_dataset, _level, "Source", clip_label),
        )
        apply_bounds(_ax, umap_xlim, umap_ylim)
        _plots["source"] = _fig

    # Repurposing target
    if "has_repurposing_target" in adata.obs.columns:
        _fig, _ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color="has_repurposing_target",
            ax=_ax,
            show=False,
            title=make_title(_dataset, _level, "Has Repurposing Target", clip_label),
            palette={"True": "#e41a1c", "False": "#cccccc"},
        )
        apply_bounds(_ax, umap_xlim, umap_ylim)
        _plots["repurposing"] = _fig

    # Uniprot target
    if "has_uniprot_target" in adata.obs.columns:
        _fig, _ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color="has_uniprot_target",
            ax=_ax,
            show=False,
            title=make_title(_dataset, _level, "Has Uniprot Target", clip_label),
            palette={"True": "#377eb8", "False": "#cccccc"},
        )
        apply_bounds(_ax, umap_xlim, umap_ylim)
        _plots["uniprot"] = _fig

    # Phenotypic significance
    if "phenotypic_significant" in adata.obs.columns:
        _fig, _ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color="phenotypic_significant",
            ax=_ax,
            show=False,
            title=make_title(_dataset, _level, "Phenotypic Activity (corrected p < 0.05)", clip_label),
            palette=SIG_PALETTE,
        )
        apply_bounds(_ax, umap_xlim, umap_ylim)
        _plots["significance"] = _fig

    # Significant only
    if "phenotypic_significant" in adata.obs.columns:
        _sig_mask = adata.obs["phenotypic_significant"] == "Significant"
        _n_sig = _sig_mask.sum()
        _obs_sig = adata.obs.loc[_sig_mask].copy()
        _obs_sig["phenotypic_significant"] = _obs_sig["phenotypic_significant"].cat.remove_unused_categories()
        _adata_sig = sc.AnnData(
            obs=_obs_sig,
            obsm={"X_umap": adata.obsm["X_umap"][_sig_mask.values]},
        )
        _fig, _ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            _adata_sig,
            color="phenotypic_significant",
            ax=_ax,
            show=False,
            title=make_title(_dataset, _level, f"Significant Only ({_n_sig:,})", clip_label),
            palette={"Significant": "#e41a1c"},
        )
        apply_bounds(_ax, umap_xlim, umap_ylim)
        _plots["sig_only"] = _fig

    # Normalized mAP
    if "mean_normalized_average_precision" in adata.obs.columns:
        _fig, _ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color="mean_normalized_average_precision",
            ax=_ax,
            show=False,
            title=make_title(_dataset, _level, "Normalized Mean Average Precision", clip_label),
            cmap="viridis",
        )
        apply_bounds(_ax, umap_xlim, umap_ylim)
        _plots["nmap"] = _fig

    # Cell count (well-level)
    if _level == "well" and "Count_Cells" in adata.obs.columns:
        _fig, _ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color="Count_Cells",
            ax=_ax,
            show=False,
            title=make_title(_dataset, _level, "Cell Count per Well", clip_label),
            cmap="viridis",
        )
        apply_bounds(_ax, umap_xlim, umap_ylim)
        _plots["cell_count"] = _fig

    # Median cell count
    if "median_cell_count" in adata.obs.columns:
        _fig, _ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color="median_cell_count",
            ax=_ax,
            show=False,
            title=make_title(_dataset, _level, "Median Cell Count per Perturbation", clip_label),
            cmap="viridis",
        )
        apply_bounds(_ax, umap_xlim, umap_ylim)
        _plots["median_cells"] = _fig

    # Combined multi-panel
    _color_keys = []
    if _level == "well" and "Source" in adata.obs.columns:
        _color_keys.append("Source")
    if "has_repurposing_target" in adata.obs.columns:
        _color_keys.append("has_repurposing_target")
    if "has_uniprot_target" in adata.obs.columns:
        _color_keys.append("has_uniprot_target")
    if "phenotypic_significant" in adata.obs.columns:
        _color_keys.append("phenotypic_significant")
    if "mean_normalized_average_precision" in adata.obs.columns:
        _color_keys.append("mean_normalized_average_precision")
    if _level == "well" and "Count_Cells" in adata.obs.columns:
        _color_keys.append("Count_Cells")
    if "median_cell_count" in adata.obs.columns:
        _color_keys.append("median_cell_count")

    if _color_keys:
        _level_label = "Perturbation" if _level == "perturbation" else "Well"
        _fig = sc.pl.umap(
            adata,
            color=_color_keys,
            ncols=min(len(_color_keys), 3),
            show=False,
            return_fig=True,
            na_color="none",
        )
        _fig.suptitle(
            f"{_dataset} ({_level_label}){clip_label}",
            fontsize=14,
            y=1.02,
        )
        for _a in _fig.axes:
            apply_bounds(_a, umap_xlim, umap_ylim)
        _plots["combined"] = _fig

    umap_plots = _plots
    return (umap_plots,)


# -- Display individual plots --


@app.cell(hide_code=True)
def _(mo, umap_plots):
    _tabs = {}
    _labels = {
        "source": "Source",
        "repurposing": "Repurposing Target",
        "uniprot": "Uniprot Target",
        "significance": "Phenotypic Significance",
        "sig_only": "Significant Only",
        "nmap": "Normalized mAP",
        "cell_count": "Cell Count",
        "median_cells": "Median Cell Count",
        "combined": "Combined",
    }
    for _key, _label in _labels.items():
        if _key in umap_plots:
            _tabs[_label] = umap_plots[_key]
        else:
            _tabs[_label] = mo.md(f"*{_label} not available for this level/dataset.*")

    mo.ui.tabs(_tabs)
    return


# -- Save all plots --


@app.cell
def _(dataset_dropdown, filter_dropdown, level_dropdown, metric_dropdown, mo):
    save_button = mo.ui.button(label="Save all plots", kind="warn")
    mo.hstack(
        [
            save_button,
            mo.md(
                f"Saves to `data/processed/umap/{dataset_dropdown.value}/"
                f"{metric_dropdown.value}_{filter_dropdown.value}/`"
            ),
        ]
    )
    return (save_button,)


@app.cell
def _(
    dataset_dropdown,
    filter_dropdown,
    level_dropdown,
    metric_dropdown,
    mo,
    save_button,
    umap_plots,
):
    mo.stop(not save_button.value)

    _output_dir = UMAP_OUTPUT_DIR / dataset_dropdown.value / f"{metric_dropdown.value}_{filter_dropdown.value}"
    _output_dir.mkdir(parents=True, exist_ok=True)
    _prefix = level_dropdown.value

    _key_to_filename = {
        "source": f"{_prefix}_umap_source.png",
        "repurposing": f"{_prefix}_umap_repurposing_target.png",
        "uniprot": f"{_prefix}_umap_uniprot_target.png",
        "significance": f"{_prefix}_umap_phenotypic_significance.png",
        "sig_only": f"{_prefix}_umap_significant_only.png",
        "nmap": f"{_prefix}_umap_nmap.png",
        "cell_count": f"{_prefix}_umap_cell_count.png",
        "median_cells": f"{_prefix}_umap_median_cell_count.png",
        "combined": f"{_prefix}_umap_combined.png",
    }

    _saved = []
    for _key, _filename in _key_to_filename.items():
        _fig = umap_plots.get(_key)
        if _fig is not None and hasattr(_fig, "savefig"):
            _fig.savefig(_output_dir / _filename, dpi=DEFAULT_DPI, bbox_inches="tight")
            _saved.append(_filename)

    logger.success(f"Saved {len(_saved)} plots to {_output_dir}")
    mo.md(f"**Saved {len(_saved)} plots** to `{_output_dir}`:\n" + "\n".join(f"- `{f}`" for f in _saved))
    return


@app.function
def plot_umap_combined(
    dataset: str,
    metric: str = "cosine",
    filter_type: str = "all",
    output_dir=None,
) -> dict[str, str]:
    """Generate combined UMAP plots for both well and perturbation levels.

    Loads precomputed UMAP h5ad, joins metadata and activity data, creates
    multi-panel combined plots for each level, and saves PNGs.

    Called from workflow.py. Returns dict mapping level to saved PNG path.

    Args:
        dataset: e.g. "compound_no_source7"
        metric: "cosine" or "euclidean"
        filter_type: "all" or "active"
        output_dir: Override output directory (default: PROCESSED_DATA_DIR/umap/{dataset}/{metric}_{filter})
    """
    import matplotlib.pyplot as plt
    import scanpy as sc

    from nb00_ss_config import DEFAULT_DPI, PROCESSED_DATA_DIR
    from nb03_ss_profiles import join_activity, load_umap
    from nb04_ss_visualization import compute_umap_bounds

    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "umap" / dataset / f"{metric}_{filter_type}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _saved = {}

    for _level in ("well", "perturbation"):
        _adata = load_umap(dataset, level=_level, metric=metric, filter_name=filter_type)
        if "X_umap" not in _adata.obsm:
            logger.warning(f"X_umap not found for {dataset} {_level} {metric} {filter_type}, skipping")
            continue

        _adata = join_activity(_adata, dataset, preprocessing="activity_no_target2", filter_name="all_sources")

        # Derive annotation columns
        if "repurposing_target" in _adata.obs.columns and "has_repurposing_target" not in _adata.obs.columns:
            _adata.obs["has_repurposing_target"] = _adata.obs["repurposing_target"].notna().astype(str)
        if "Uniprot_target" in _adata.obs.columns and "has_uniprot_target" not in _adata.obs.columns:
            _adata.obs["has_uniprot_target"] = _adata.obs["Uniprot_target"].notna().astype(str)
        if "below_corrected_p" in _adata.obs.columns and "phenotypic_significant" not in _adata.obs.columns:
            _adata.obs["phenotypic_significant"] = _adata.obs["below_corrected_p"].map(
                {"True": "Significant", "False": "Not significant", "NA": "No data"}
            )

        # IQR clipping bounds
        _xlim, _ylim = compute_umap_bounds(_adata)
        _coords = _adata.obsm["X_umap"]
        _n_outside = np.sum(
            (_coords[:, 0] < _xlim[0])
            | (_coords[:, 0] > _xlim[1])
            | (_coords[:, 1] < _ylim[0])
            | (_coords[:, 1] > _ylim[1])
        )
        _pct_outside = 100 * _n_outside / len(_coords)
        _clip_label = f" [{_pct_outside:.1f}% clipped]" if _n_outside > 0 else ""

        # Build color keys for combined plot
        _color_keys = []
        if _level == "well" and "Source" in _adata.obs.columns:
            _color_keys.append("Source")
        if "has_repurposing_target" in _adata.obs.columns:
            _color_keys.append("has_repurposing_target")
        if "has_uniprot_target" in _adata.obs.columns:
            _color_keys.append("has_uniprot_target")
        if "phenotypic_significant" in _adata.obs.columns:
            _color_keys.append("phenotypic_significant")
        if "mean_normalized_average_precision" in _adata.obs.columns:
            _color_keys.append("mean_normalized_average_precision")
        if _level == "well" and "Count_Cells" in _adata.obs.columns:
            _color_keys.append("Count_Cells")
        if "median_cell_count" in _adata.obs.columns:
            _color_keys.append("median_cell_count")

        if not _color_keys:
            logger.warning(f"No color keys available for {dataset} {_level}, skipping")
            continue

        _level_label = "Perturbation" if _level == "perturbation" else "Well"
        _fig = sc.pl.umap(
            _adata,
            color=_color_keys,
            ncols=min(len(_color_keys), 3),
            show=False,
            return_fig=True,
            na_color="none",
        )
        _fig.suptitle(f"{dataset} ({_level_label}){_clip_label}", fontsize=14, y=1.02)
        for _a in _fig.axes:
            apply_bounds(_a, _xlim, _ylim)

        # Save individual plots too
        _key_to_filename = {
            "source": f"{_level}_umap_source.png",
            "repurposing": f"{_level}_umap_repurposing_target.png",
            "uniprot": f"{_level}_umap_uniprot_target.png",
            "significance": f"{_level}_umap_phenotypic_significance.png",
            "sig_only": f"{_level}_umap_significant_only.png",
            "nmap": f"{_level}_umap_nmap.png",
            "cell_count": f"{_level}_umap_cell_count.png",
            "median_cells": f"{_level}_umap_median_cell_count.png",
        }

        # Generate and save individual plots
        if _level == "well" and "Source" in _adata.obs.columns:
            _ifig, _iax = plt.subplots(figsize=(10, 8))
            sc.pl.umap(
                _adata, color="Source", ax=_iax, show=False, title=make_title(dataset, _level, "Source", _clip_label)
            )
            apply_bounds(_iax, _xlim, _ylim)
            _ifig.savefig(output_dir / _key_to_filename["source"], dpi=DEFAULT_DPI, bbox_inches="tight")
            plt.close(_ifig)

        if "phenotypic_significant" in _adata.obs.columns:
            _ifig, _iax = plt.subplots(figsize=(10, 8))
            sc.pl.umap(
                _adata,
                color="phenotypic_significant",
                ax=_iax,
                show=False,
                title=make_title(dataset, _level, "Phenotypic Activity (corrected p < 0.05)", _clip_label),
                palette=SIG_PALETTE,
            )
            apply_bounds(_iax, _xlim, _ylim)
            _ifig.savefig(output_dir / _key_to_filename["significance"], dpi=DEFAULT_DPI, bbox_inches="tight")
            plt.close(_ifig)

        if "mean_normalized_average_precision" in _adata.obs.columns:
            _ifig, _iax = plt.subplots(figsize=(10, 8))
            sc.pl.umap(
                _adata,
                color="mean_normalized_average_precision",
                ax=_iax,
                show=False,
                title=make_title(dataset, _level, "Normalized Mean Average Precision", _clip_label),
                cmap="viridis",
            )
            apply_bounds(_iax, _xlim, _ylim)
            _ifig.savefig(output_dir / _key_to_filename["nmap"], dpi=DEFAULT_DPI, bbox_inches="tight")
            plt.close(_ifig)

        # Save combined
        _combined_path = output_dir / f"{_level}_umap_combined.png"
        _fig.savefig(_combined_path, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
        plt.close(_fig)
        _saved[_level] = str(_combined_path)

        logger.success(f"Saved {_level}-level UMAP plots to {output_dir}")

    return _saved


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
