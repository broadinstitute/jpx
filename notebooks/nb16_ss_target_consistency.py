# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "adjusttext==1.3.0",
#     "anndata==0.12.16",
#     "duckdb==1.5.3",
#     "loguru==0.7.3",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==2.3.3",
#     "python-dotenv",
#     "scanpy==1.12.1",
#     "scipy==1.17.1",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import re
    import sys
    from pathlib import Path

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from adjustText import adjust_text
    from loguru import logger
    from matplotlib.lines import Line2D

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import COPAIRS_RESULTS_DB, DEFAULT_DPI, PROCESSED_DATA_DIR
    from nb02_ss_queries import query_activity_results, query_consistency_results
    from nb03_ss_profiles import load_umap
    from nb04_ss_visualization import (
        NONSIG_COLOR,
        SIG_COLOR,
        compute_umap_bounds,
        plot_dotplot,
    )

    sc.settings.verbosity = 1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Target Consistency

    Summarizes copairs consistency results with volcano plots, dot plots, and UMAP
    visualizations. For each annotation source (repurposing, uniprot, moa, etc.),
    the consistency analysis asks whether compounds sharing a target annotation
    cluster together in morphological space.

    **Workflow:**

    1. Select dataset, group type, and distance metric
    2. View volcano plot (effect size vs significance) and dot plot (top targets)
    3. Explore UMAP grid highlighting the top consistent targets
    4. Optionally save all outputs to `data/processed/target-consistency/`

    Metadata is joined at visualization time from DuckDB (not stored in h5ad files).
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
    group_type_dropdown = mo.ui.dropdown(
        options=[
            "repurposing",
            "uniprot",
            "chemical_probes",
            "moa",
            "disease_area",
            "motive_biokg",
            "motive_opentargets",
            "motive_primekg",
            "toxcast_assay",
        ],
        value="repurposing",
        label="Group type",
    )
    distance_dropdown = mo.ui.dropdown(
        options=["cosine", "abs_cosine"],
        value="cosine",
        label="Distance",
    )
    umap_filter_dropdown = mo.ui.dropdown(
        options=["all", "active"],
        value="all",
        label="UMAP filter",
    )
    label_top_n_slider = mo.ui.slider(start=5, stop=40, step=5, value=20, label="Top N labels (volcano)")
    mo.hstack([dataset_dropdown, group_type_dropdown, distance_dropdown, umap_filter_dropdown, label_top_n_slider])
    return (dataset_dropdown, distance_dropdown, group_type_dropdown, label_top_n_slider, umap_filter_dropdown)


# -- Query helpers --


@app.function
def get_group_type_column(group_type: str) -> str | None:
    """Get h5ad column name for a group type from copairs database.

    The mapping is stored in consistency_results._group_column. We strip the
    Metadata_ prefix since load_profiles() strips it when loading into h5ad.
    """
    _con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    _result = _con.execute(
        "SELECT DISTINCT _group_column FROM consistency_results WHERE _group_type = ? AND _preprocessing NOT LIKE '%_sweep'",
        [group_type],
    ).fetchone()
    _con.close()
    if _result:
        return _result[0].replace("Metadata_", "")
    return None


# -- Load data --


@app.cell
def _(dataset_dropdown, distance_dropdown, group_type_dropdown, mo):
    mo.stop(
        not COPAIRS_RESULTS_DB.exists(),
        mo.md(f"**Error:** Copairs database not found at `{COPAIRS_RESULTS_DB}`. Run `just run` first."),
    )

    _dataset = dataset_dropdown.value
    _group_type = group_type_dropdown.value
    _distance = distance_dropdown.value

    # Activity results
    activity_df = query_activity_results(_dataset)
    mo.stop(
        len(activity_df) == 0,
        mo.md(f"**Error:** No activity results found for dataset `{_dataset}`."),
    )

    # Consistency results
    _consistency_preprocessing = "consistency_no_target2"
    consistency_df = query_consistency_results(
        _dataset,
        preprocessing=_consistency_preprocessing,
        filter_name="all_sources",
        group_type=_group_type,
        distance=_distance,
    )

    _n_compounds = len(activity_df)
    _n_active = activity_df["below_corrected_p"].sum()
    _n_targets = len(consistency_df)
    _n_consistent = consistency_df["below_corrected_p"].sum() if _n_targets > 0 else 0

    mo.md(f"""
    ## Data loaded

    **Dataset:** {_dataset} | **Group type:** {_group_type} | **Distance:** {_distance}

    | Metric | Count |
    |--------|-------|
    | Total compounds | {_n_compounds:,} |
    | Active compounds (p < 0.05) | {_n_active:,} ({_n_active / _n_compounds:.1%}) |
    | Total targets ({_group_type}) | {_n_targets:,} |
    | Consistent targets (p < 0.05) | {_n_consistent:,} ({_n_consistent / _n_targets:.1%} of targets) |
    """)
    return (activity_df, consistency_df)


# -- Volcano plot --


@app.function
def plot_volcano(
    df: pd.DataFrame,
    title: str = "Target Consistency",
    p_threshold: float = 0.05,
    label_top_n: int = 20,
) -> plt.Figure:
    """Create volcano plot of consistency results.

    x-axis: normalized mAP (effect size)
    y-axis: -log10(corrected p-value) (significance)
    Labels: top significant targets
    Size: n_perturbations
    """
    _df = df.copy()
    _df["neg_log10_p"] = -np.log10(_df["corrected_p_value"].clip(lower=1e-300))

    _sig_threshold = -np.log10(p_threshold)

    fig, ax = plt.subplots(figsize=(10, 8))

    _sig_mask = _df["below_corrected_p"]
    _nonsig = _df[~_sig_mask]
    _sig = _df[_sig_mask]

    _size_scale = 20
    _min_size = 20
    _max_size = 200

    def _get_sizes(n_perturb):
        return (n_perturb * _size_scale).clip(_min_size, _max_size)

    if len(_nonsig) > 0:
        ax.scatter(
            _nonsig["mean_normalized_average_precision"],
            _nonsig["neg_log10_p"],
            s=_get_sizes(_nonsig["n_perturbations"]),
            c=NONSIG_COLOR,
            alpha=0.6,
            edgecolors="none",
            label=f"Not significant (n={len(_nonsig)})",
        )

    if len(_sig) > 0:
        ax.scatter(
            _sig["mean_normalized_average_precision"],
            _sig["neg_log10_p"],
            s=_get_sizes(_sig["n_perturbations"]),
            c=SIG_COLOR,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            label=f"Significant p<{p_threshold} (n={len(_sig)})",
        )

    ax.axhline(y=_sig_threshold, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(
        ax.get_xlim()[1] * 0.98,
        _sig_threshold + 0.1,
        f"p={p_threshold}",
        ha="right",
        va="bottom",
        color="red",
        fontsize=9,
    )

    if len(_sig) > 0:
        _top_targets = _sig.nlargest(label_top_n, "mean_normalized_average_precision")
        _texts = []
        for _, _row in _top_targets.iterrows():
            _texts.append(
                ax.text(
                    _row["mean_normalized_average_precision"],
                    _row["neg_log10_p"],
                    _row["group_value"],
                    fontsize=8,
                )
            )
        adjust_text(
            _texts,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            ax=ax,
        )

    ax.set_xlabel("Normalized mAP", fontsize=11)
    ax.set_ylabel("-log10(corrected p-value)", fontsize=11)
    ax.set_title(title, fontsize=12)

    # Size legend
    _size_handles = []
    _size_labels = []
    for _n in [2, 5, 10]:
        _size_handles.append(ax.scatter([], [], s=_get_sizes(pd.Series([_n]))[0], c="gray", alpha=0.6))
        _size_labels.append(f"n={_n}")
    _size_legend = ax.legend(
        _size_handles,
        _size_labels,
        title="# compounds",
        loc="lower right",
        framealpha=0.9,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(_size_legend)

    ax.legend(loc="center right", framealpha=0.9)

    plt.tight_layout()
    return fig


@app.cell
def _(consistency_df, dataset_dropdown, group_type_dropdown, label_top_n_slider, mo):
    mo.stop(
        len(consistency_df) == 0,
        mo.md(f"No consistency results for **{group_type_dropdown.value}**. Try a different group type."),
    )

    fig_volcano = plot_volcano(
        consistency_df,
        title=f"Target Consistency: {group_type_dropdown.value} ({dataset_dropdown.value})",
        label_top_n=label_top_n_slider.value,
    )
    fig_volcano
    return (fig_volcano,)


# -- Dot plot --


@app.cell
def _(consistency_df, dataset_dropdown, group_type_dropdown, mo):
    mo.stop(
        len(consistency_df) == 0,
        mo.md("No consistency data for dot plot."),
    )

    fig_dotplot = plot_dotplot(
        consistency_df,
        value_col="mean_normalized_average_precision",
        label_col="group_value",
        count_col="n_perturbations",
        sig_col="below_corrected_p",
        title=f"Top Consistent Targets: {group_type_dropdown.value} ({dataset_dropdown.value})",
        xlabel="Normalized mAP",
    )
    fig_dotplot
    return (fig_dotplot,)


# -- UMAP grid --


@app.function
def plot_target_umap_grid(
    consistency_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    dataset: str,
    group_type: str,
    umap_filter: str = "all",
    metric: str = "cosine",
    top_n: int = 12,
) -> plt.Figure | None:
    """Create 3x4 grid of UMAPs highlighting top consistent targets.

    Each panel shows one target's compounds highlighted against gray background.
    Blue: compounds in consistency analysis (has target + phenotypically active).
    Purple: compounds with target annotation but NOT phenotypically active.
    """
    _target_column = get_group_type_column(group_type)
    if _target_column is None:
        logger.warning(f"No column mapping found for group_type={group_type}")
        return None

    try:
        _adata = load_umap(dataset, level="perturbation", metric=metric, filter_name=umap_filter)
    except FileNotFoundError as e:
        logger.warning(f"{e}, skipping UMAP grid")
        return None

    if _target_column not in _adata.obs.columns:
        logger.warning(f"Column {_target_column} not found in h5ad")
        return None

    _active_ids = set(activity_df.loc[activity_df["below_corrected_p"], "Metadata_JCP2022"].tolist())
    logger.info(f"Found {len(_active_ids)} phenotypically active compounds")

    _top_targets = consistency_df.nlargest(top_n, "mean_normalized_average_precision")["group_value"].tolist()
    logger.info(f"Plotting top {len(_top_targets)} targets on UMAP")

    _iqr_k = 1.5 if umap_filter == "all" else 3
    _xlim, _ylim = compute_umap_bounds(_adata, iqr_k=_iqr_k)

    _coords = _adata.obsm["X_umap"]

    _nrows, _ncols = 3, 4
    fig, _axes = plt.subplots(_nrows, _ncols, figsize=(16, 12))
    _axes = _axes.flatten()

    for _idx, _target in enumerate(_top_targets):
        _ax = _axes[_idx]

        _target_row = consistency_df[consistency_df["group_value"] == _target].iloc[0]
        _n_in_analysis = _target_row["n_perturbations"]
        _nmap = _target_row["mean_normalized_average_precision"]

        _pattern = rf"(?:^|\|){re.escape(_target)}(?:\||$)"
        _has_target = _adata.obs[_target_column].str.contains(_pattern, na=False, regex=True)

        _is_active = _adata.obs["JCP2022"].isin(_active_ids)
        _mask_in_analysis = _has_target & _is_active
        _mask_not_in_analysis = _has_target & ~_is_active

        _n_total = _has_target.sum()

        _highlight = pd.Categorical(
            ["Other"] * len(_adata),
            categories=["Other", "Inactive", "Active"],
        )
        _highlight[_mask_not_in_analysis] = "Inactive"
        _highlight[_mask_in_analysis] = "Active"
        _adata.obs["_highlight"] = _highlight

        sc.pl.umap(
            _adata,
            color="_highlight",
            groups=["Active", "Inactive"],
            palette={"Active": "#0072B2", "Inactive": "#CC79A7", "Other": "#E5E5E5"},
            na_color="#E5E5E5",
            size=20,
            ax=_ax,
            show=False,
            title=f"{_target}\n({_n_in_analysis}/{_n_total} active, mAP={_nmap:.2f})",
        )

        if _n_total < 10:
            _texts = []
            for _mask, _text_color in [(_mask_in_analysis, "#0072B2"), (_mask_not_in_analysis, "#CC79A7")]:
                if _mask.sum() > 0:
                    _indices = np.where(_mask)[0]
                    for _i in _indices:
                        _jcp_id = _adata.obs["JCP2022"].iloc[_i]
                        _short_id = _jcp_id.replace("JCP2022_", "")
                        _texts.append(
                            _ax.text(
                                _coords[_i, 0],
                                _coords[_i, 1],
                                _short_id,
                                fontsize=8,
                                fontweight="bold",
                                color=_text_color,
                                zorder=11,
                            )
                        )
            if _texts:
                adjust_text(_texts, ax=_ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

        _ax.set_xlim(_xlim)
        _ax.set_ylim(_ylim)
        _ax.set_xlabel("")
        _ax.set_ylabel("")
        if _ax.get_legend():
            _ax.get_legend().remove()

    for _idx in range(len(_top_targets), len(_axes)):
        _axes[_idx].set_visible(False)

    _legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#0072B2",
            markersize=10,
            markeredgecolor="white",
            label="Active",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#CC79A7",
            markersize=10,
            markeredgecolor="white",
            label="Inactive",
        ),
    ]
    fig.legend(handles=_legend_elements, loc="upper right", bbox_to_anchor=(0.99, 0.99))

    fig.suptitle(f"Top Consistent Targets: {group_type} ({dataset}, {umap_filter})", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


@app.cell
def _(activity_df, consistency_df, dataset_dropdown, distance_dropdown, group_type_dropdown, mo, umap_filter_dropdown):
    mo.stop(
        len(consistency_df) == 0,
        mo.md("No consistency data for UMAP grid."),
    )

    fig_umap_grid = plot_target_umap_grid(
        consistency_df,
        activity_df,
        dataset=dataset_dropdown.value,
        group_type=group_type_dropdown.value,
        umap_filter=umap_filter_dropdown.value,
        metric=distance_dropdown.value if distance_dropdown.value in ("cosine", "euclidean") else "cosine",
    )
    mo.stop(
        fig_umap_grid is None,
        mo.md("UMAP grid could not be generated. Check that UMAP embeddings and target column exist."),
    )
    fig_umap_grid
    return (fig_umap_grid,)


# -- Summary statistics --


@app.function
def format_summary_stats(
    dataset: str,
    group_type: str,
    activity_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    distance: str = "cosine",
) -> str:
    """Format summary statistics as markdown."""
    _lines = []
    _lines.append(f"# Copairs Summary: {dataset}")
    _lines.append(f"- **Group type**: {group_type}")
    _lines.append(f"- **Distance metric**: {distance}")
    _lines.append("")

    _lines.append("## Activity Results")
    _lines.append("")
    _n_total = len(activity_df)
    _n_active = activity_df["below_corrected_p"].sum()
    _rate = _n_active / _n_total if _n_total > 0 else 0
    _lines.append(f"- Total compounds: {_n_total:,}")
    _lines.append(f"- Active compounds (p<0.05): {_n_active:,} ({_rate:.1%})")
    if _n_total > 0:
        _lines.append(f"- Mean normalized mAP: {activity_df['mean_normalized_average_precision'].mean():.4f}")
        _lines.append(f"- Median normalized mAP: {activity_df['mean_normalized_average_precision'].median():.4f}")
    _lines.append("")

    _lines.append(f"## Consistency Results ({group_type}, {distance})")
    _lines.append("")
    _n_targets = len(consistency_df)
    _n_consistent = consistency_df["below_corrected_p"].sum()
    _rate = _n_consistent / _n_targets if _n_targets > 0 else 0
    _lines.append(f"- Total targets: {_n_targets:,}")
    _lines.append(f"- Consistent targets (p<0.05): {_n_consistent:,} ({_rate:.1%})")
    _lines.append("")

    if len(consistency_df) > 0:
        _lines.append("### Top 10 Consistent Targets")
        _lines.append("")
        _lines.append("| Target | Normalized mAP | p-value | # Compounds |")
        _lines.append("|--------|----------------|---------|-------------|")
        for _, _row in consistency_df.head(10).iterrows():
            _lines.append(
                f"| {_row['group_value']} | {_row['mean_normalized_average_precision']:.3f} | "
                f"{_row['corrected_p_value']:.2e} | {_row['n_perturbations']} |"
            )

    return "\n".join(_lines)


@app.cell
def _(activity_df, consistency_df, dataset_dropdown, distance_dropdown, group_type_dropdown, mo):
    _summary = format_summary_stats(
        dataset_dropdown.value,
        group_type_dropdown.value,
        activity_df,
        consistency_df,
        distance_dropdown.value,
    )
    mo.md(_summary)
    return


# -- Save outputs --


@app.cell
def _(dataset_dropdown, distance_dropdown, group_type_dropdown, mo):
    save_button = mo.ui.button(label="Save all outputs", kind="warn")
    _output_path = (
        PROCESSED_DATA_DIR
        / "target-consistency"
        / dataset_dropdown.value
        / group_type_dropdown.value
        / distance_dropdown.value
    )
    mo.hstack([save_button, mo.md(f"Saves to `{_output_path}`")])
    return (save_button,)


@app.cell
def _(
    activity_df,
    consistency_df,
    dataset_dropdown,
    distance_dropdown,
    fig_dotplot,
    fig_umap_grid,
    fig_volcano,
    group_type_dropdown,
    mo,
    save_button,
    umap_filter_dropdown,
):
    mo.stop(not save_button.value)

    _dataset = dataset_dropdown.value
    _group_type = group_type_dropdown.value
    _distance = distance_dropdown.value
    _umap_filter = umap_filter_dropdown.value

    _output_dir = PROCESSED_DATA_DIR / "target-consistency" / _dataset / _group_type / _distance
    _output_dir.mkdir(parents=True, exist_ok=True)

    _saved = []

    # Save consistency results CSV
    consistency_df.to_csv(_output_dir / "results.csv", index=False)
    _saved.append("results.csv")

    # Save volcano plot
    if fig_volcano is not None:
        fig_volcano.savefig(_output_dir / "volcano.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        _saved.append("volcano.png")

    # Save dot plot
    if fig_dotplot is not None:
        fig_dotplot.savefig(_output_dir / "dotplot.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        _saved.append("dotplot.png")

    # Save UMAP grid
    if fig_umap_grid is not None:
        fig_umap_grid.savefig(_output_dir / f"umap_grid_{_umap_filter}.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        _saved.append(f"umap_grid_{_umap_filter}.png")

    # Save summary stats
    _summary_text = format_summary_stats(_dataset, _group_type, activity_df, consistency_df, _distance)
    (_output_dir / "summary_stats.md").write_text(_summary_text)
    _saved.append("summary_stats.md")

    logger.success(f"Saved {len(_saved)} outputs to {_output_dir}")
    mo.md(f"**Saved {len(_saved)} outputs** to `{_output_dir}`:\n" + "\n".join(f"- `{f}`" for f in _saved))
    return


@app.function
def run_target_consistency(
    dataset: str,
    group_type: str,
    umap_filter: str = "all",
    distance: str = "cosine",
    output_dir: str | Path | None = None,
) -> list[str]:
    """Run target consistency analysis and save all outputs.

    Produces volcano plot, dot plot, UMAP grid, results CSV, and summary
    stats markdown. The old Snakemake pipeline had two rules:
    target_consistency_base (all outputs + "all" UMAP) and
    target_consistency_umap_active (only active UMAP grid). The
    umap_filter parameter handles both: pass "all" for the base case
    or "active" for active-only UMAP.

    Parameters
    ----------
    dataset : str
        e.g. "compound_no_source7"
    group_type : str
        Annotation source, e.g. "repurposing", "uniprot", "moa"
    umap_filter : str
        "all" or "active" - controls which UMAP embeddings to use
    distance : str
        Distance metric, e.g. "cosine", "abs_cosine"
    output_dir : str | Path | None
        Output directory. Defaults to
        PROCESSED_DATA_DIR / "target-consistency" / dataset / group_type / distance

    Returns
    -------
    list[str]
        Paths of saved output files.
    """
    if output_dir is None:
        _out = PROCESSED_DATA_DIR / "target-consistency" / dataset / group_type / distance
    else:
        _out = Path(output_dir)
    _out.mkdir(parents=True, exist_ok=True)

    _saved: list[str] = []

    # Query data
    _activity_df = query_activity_results(dataset)
    _consistency_preprocessing = "consistency_no_target2"
    _consistency_df = query_consistency_results(
        dataset,
        preprocessing=_consistency_preprocessing,
        filter_name="all_sources",
        group_type=group_type,
        distance=distance,
    )
    logger.info(
        f"Loaded {len(_activity_df)} activity results, "
        f"{len(_consistency_df)} consistency results for {group_type}/{distance}"
    )

    # Save consistency results CSV
    if len(_consistency_df) > 0:
        _csv_path = _out / "results.csv"
        _consistency_df.to_csv(_csv_path, index=False)
        _saved.append(str(_csv_path))

    # Volcano plot
    if len(_consistency_df) > 0:
        _fig_volcano = plot_volcano(
            _consistency_df,
            title=f"Target Consistency: {group_type} ({dataset})",
        )
        _volcano_path = _out / "volcano.png"
        _fig_volcano.savefig(_volcano_path, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
        plt.close(_fig_volcano)
        _saved.append(str(_volcano_path))

    # Dot plot
    if len(_consistency_df) > 0:
        _fig_dotplot = plot_dotplot(
            _consistency_df,
            value_col="mean_normalized_average_precision",
            label_col="group_value",
            count_col="n_perturbations",
            sig_col="below_corrected_p",
            title=f"Top Consistent Targets: {group_type} ({dataset})",
            xlabel="Normalized mAP",
        )
        _dotplot_path = _out / "dotplot.png"
        _fig_dotplot.savefig(_dotplot_path, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
        plt.close(_fig_dotplot)
        _saved.append(str(_dotplot_path))

    # UMAP grid
    if len(_consistency_df) > 0 and len(_activity_df) > 0:
        _metric = distance if distance in ("cosine", "euclidean") else "cosine"
        _fig_umap = plot_target_umap_grid(
            _consistency_df,
            _activity_df,
            dataset=dataset,
            group_type=group_type,
            umap_filter=umap_filter,
            metric=_metric,
        )
        if _fig_umap is not None:
            _umap_path = _out / f"umap_grid_{umap_filter}.png"
            _fig_umap.savefig(_umap_path, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
            plt.close(_fig_umap)
            _saved.append(str(_umap_path))

    # Summary stats markdown
    _summary_text = format_summary_stats(dataset, group_type, _activity_df, _consistency_df, distance)
    _summary_path = _out / "summary_stats.md"
    _summary_path.write_text(_summary_text)
    _saved.append(str(_summary_path))

    # .complete marker
    (_out / ".complete").touch()
    _saved.append(str(_out / ".complete"))

    logger.success(f"run_target_consistency: saved {len(_saved)} outputs to {_out}")
    return _saved


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
