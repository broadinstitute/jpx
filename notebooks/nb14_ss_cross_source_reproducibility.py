# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb",
#     "loguru",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "python-dotenv",
#     "seaborn",
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
    import pandas as pd
    import seaborn as sns
    from loguru import logger

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import COPAIRS_RESULTS_DB, DEFAULT_DPI, PROCESSED_DATA_DIR
    from nb02_ss_queries import query_cross_source_reproducibility

    # Color palettes for consistent styling
    PALETTE_BOX = {
        "Within-source": "#3498db",
        "Cross-source": "#e74c3c",
        "Difference": "#3498db",
    }
    PALETTE_STRIP = {
        "Within-source": "#1a5276",
        "Cross-source": "#922b21",
        "Difference": "#1a5276",
    }

    OUTPUT_BASE = PROCESSED_DATA_DIR / "cross-source-reproducibility"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cross-Source Reproducibility

    Compares within-source vs cross-source activity (normalized mAP) to measure
    how reproducible compound phenotypes are across different lab sites.

    **Not "consistency"** (shared biology clustering) - this measures reproducibility
    of the same compound's phenotypic activity across different lab sites.

    - **Within-source:** replicates of the same compound at the same source (same lab)
    - **Cross-source:** same compound compared across different sources (different labs)

    Uses normalized mAP (not raw mAP) because within-source has ~25 positive pairs
    while cross-source has ~122. Normalized mAP = (AP - mu_0) / (1 - mu_0) is
    scale-independent.

    *Outputs:* `data/processed/cross-source-reproducibility/{dataset}/{preprocessing}/`
    """)
    return


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


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
            "activity_only_target2",
            "activity_poscon_only",
        ],
        value="activity_only_target2",
        label="Preprocessing",
    )
    filter_dropdown = mo.ui.dropdown(
        options=[
            "all_sources",
            "no_source9",
        ],
        value="all_sources",
        label="Filter",
    )
    mo.hstack(
        [dataset_dropdown, preprocessing_dropdown, filter_dropdown],
        justify="start",
    )
    return (dataset_dropdown, filter_dropdown, preprocessing_dropdown)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@app.cell
def _(dataset_dropdown, filter_dropdown, mo, preprocessing_dropdown):
    mo.stop(
        not COPAIRS_RESULTS_DB.exists(),
        mo.md(f"**Copairs database not found:** `{COPAIRS_RESULTS_DB}`"),
    )
    df_repro = query_cross_source_reproducibility(
        dataset_dropdown.value,
        preprocessing_dropdown.value,
        filter_dropdown.value,
    )
    mo.stop(
        len(df_repro) == 0,
        mo.md("**No data found** for the selected configuration."),
    )
    _n_compounds = df_repro["Metadata_JCP2022"].nunique()
    _n_sources = df_repro["Metadata_Source"].nunique()
    mo.md(f"""
    ## Data loaded

    **{len(df_repro):,}** compound-source pairs |
    **{_n_compounds}** unique compounds |
    **{_n_sources}** sources
    """)
    return (df_repro,)


# ---------------------------------------------------------------------------
# Reusable plot helpers
# ---------------------------------------------------------------------------


@app.function
def plot_boxstrip(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    order: list | None = None,
    hue_order: list | None = None,
    palette_box: dict | None = None,
    palette_strip: dict | None = None,
    show_mean: bool = False,
    strip_size: float = 3,
    strip_alpha: float = 0.5,
    color_box: str = "#3498db",
    color_strip: str = "#1a5276",
) -> None:
    """Plot boxplot with stripplot overlay using consistent styling."""
    if hue:
        _box_palette = palette_box or PALETTE_BOX
        _strip_palette = palette_strip or PALETTE_STRIP
        _box_color = None
        _strip_color = None
    else:
        _box_palette = None
        _strip_palette = None
        _box_color = color_box
        _strip_color = color_strip

    sns.boxplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        ax=ax,
        palette=_box_palette,
        color=_box_color,
        linewidth=0.8,
        fliersize=0,
        legend="auto" if hue else False,
    )

    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        ax=ax,
        palette=_strip_palette,
        color=_strip_color,
        size=strip_size,
        alpha=strip_alpha,
        dodge=True if hue else False,
        legend=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_mean:
        if hue:
            for _i, _x_val in enumerate(order or data[x].unique()):
                for _j, _h_val in enumerate(hue_order or data[hue].unique()):
                    _subset = data[(data[x] == _x_val) & (data[hue] == _h_val)]
                    if len(_subset) > 0:
                        _mean_val = _subset[y].mean()
                        _n_hue = len(hue_order or data[hue].unique())
                        _offset = (_j - (_n_hue - 1) / 2) * 0.4
                        ax.annotate(
                            f"mu={_mean_val:.2f}",
                            xy=(_i + _offset, ax.get_ylim()[1] * 0.98),
                            ha="center",
                            va="top",
                            fontsize=8,
                        )
        else:
            for _i, _x_val in enumerate(order or data[x].unique()):
                _mean_val = data[data[x] == _x_val][y].mean()
                ax.annotate(
                    f"mu={_mean_val:.2f}",
                    xy=(_i, ax.get_ylim()[1] * 0.98),
                    ha="center",
                    va="top",
                    fontsize=9,
                )


@app.function
def plot_within_vs_cross_scatter(
    df: pd.DataFrame,
    title_suffix: str = "",
) -> plt.Figure:
    """Scatter plot of within-source vs cross-source normalized mAP.

    Left panel: colored by significance status.
    Right panel: colored by source (for significant pairs).

    Returns the figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: color by significance
    _ax = axes[0]
    _both_sig = df[df["sig_within"] & df["sig_cross"]]
    _one_sig = df[df["sig_within"] ^ df["sig_cross"]]
    _neither_sig = df[~df["sig_within"] & ~df["sig_cross"]]

    for _subset, _color, _alpha, _label in [
        (_neither_sig, "#a0a0a0", 0.3, f"Neither sig (n={len(_neither_sig)})"),
        (_one_sig, "#ff7f0e", 0.5, f"One sig (n={len(_one_sig)})"),
        (_both_sig, "#1f77b4", 0.6, f"Both sig (n={len(_both_sig)})"),
    ]:
        if len(_subset) > 0:
            _ax.scatter(
                _subset["nmAP_within"],
                _subset["nmAP_cross"],
                c=_color,
                alpha=_alpha,
                s=15,
                label=_label,
                linewidths=0,
                rasterized=True,
            )

    _ax.set_xlabel("Within-source normalized mAP")
    _ax.set_ylabel("Cross-source normalized mAP")
    _all_vals = pd.concat([df["nmAP_within"], df["nmAP_cross"]])
    _vmin = min(-0.1, _all_vals.min() - 0.05)
    _vmax = max(1.02, _all_vals.max() + 0.02)
    _ax.set_xlim(_vmin, _vmax)
    _ax.set_ylim(_vmin, _vmax)
    _ax.plot([_vmin, _vmax], [_vmin, _vmax], "k--", alpha=0.5, linewidth=1)
    _ax.set_aspect("equal")
    _ax.legend(loc="lower right", fontsize=8, frameon=True)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    # Statistics for significant pairs only
    _sig_df = _both_sig if len(_both_sig) > 10 else df
    _corr = _sig_df["nmAP_within"].corr(_sig_df["nmAP_cross"])
    _n_above = (_sig_df["nmAP_cross"] > _sig_df["nmAP_within"]).sum()
    _n_below = (_sig_df["nmAP_cross"] < _sig_df["nmAP_within"]).sum()
    _sig_label = "both sig" if len(_both_sig) > 10 else "all"
    _ax.annotate(
        f"r = {_corr:.2f} ({_sig_label})\nAbove: {_n_above}, Below: {_n_below}",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
    )
    _ax.set_title(f"By Significance{title_suffix}")

    # Right: color by source (significant pairs)
    _ax2 = axes[1]
    _plot_df = _both_sig if len(_both_sig) > 10 else df
    _sources = sorted(_plot_df["Metadata_Source"].unique(), key=lambda s: int(s.split("_")[1]))
    _colors = plt.cm.tab10(np.linspace(0, 1, len(_sources)))
    _source_colors = dict(zip(_sources, _colors))

    for _source in _sources:
        _source_df = _plot_df[_plot_df["Metadata_Source"] == _source]
        _ax2.scatter(
            _source_df["nmAP_within"],
            _source_df["nmAP_cross"],
            c=[_source_colors[_source]],
            alpha=0.5,
            s=20,
            label=_source.replace("_", " ").title(),
            linewidths=0,
            rasterized=True,
        )

    _ax2.set_xlabel("Within-source normalized mAP")
    _ax2.set_ylabel("Cross-source normalized mAP")
    _ax2.set_xlim(_vmin, _vmax)
    _ax2.set_ylim(_vmin, _vmax)
    _ax2.plot([_vmin, _vmax], [_vmin, _vmax], "k--", alpha=0.5, linewidth=1, label="Identity")
    _ax2.set_aspect("equal")
    _ax2.legend(loc="lower right", fontsize=8, frameon=True)
    _ax2.spines["top"].set_visible(False)
    _ax2.spines["right"].set_visible(False)
    _filter_label = "Both Significant" if len(_both_sig) > 10 else "All"
    _ax2.set_title(f"By Source ({_filter_label}){title_suffix}")

    fig.tight_layout()
    return fig


@app.function
def plot_map_comparison_box(
    df: pd.DataFrame,
    title_suffix: str = "",
) -> plt.Figure:
    """Boxplot comparing mean within-source vs cross-source mAP.

    Left panel: overall comparison.
    Right panel: per-source breakdown.

    Returns the figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 3]})

    # Reshape for plotting
    _df_melted = pd.melt(
        df,
        id_vars=["Metadata_Source", "Metadata_JCP2022"],
        value_vars=["nmAP_within", "nmAP_cross"],
        var_name="comparison",
        value_name="nmAP",
    )
    _df_melted["comparison"] = _df_melted["comparison"].map(
        {"nmAP_within": "Within-source", "nmAP_cross": "Cross-source"}
    )

    # Left: overall
    _ax1 = axes[0]
    _order = ["Within-source", "Cross-source"]
    plot_boxstrip(
        _ax1,
        _df_melted,
        x="comparison",
        y="nmAP",
        order=_order,
        strip_size=2,
        strip_alpha=0.4,
    )
    _ax1.set_xticks([0, 1])
    _ax1.set_xticklabels(["Within", "Cross"])
    _ax1.set_xlabel("")
    _ax1.set_ylabel("Normalized mAP")
    _ax1.set_ylim(-0.2, 1.05)
    _ax1.set_title(f"Overall{title_suffix}")

    for _i, _comp in enumerate(_order):
        _mean_val = _df_melted[_df_melted["comparison"] == _comp]["nmAP"].mean()
        _ax1.annotate(f"mu={_mean_val:.2f}", xy=(_i, 1.02), ha="center", va="bottom", fontsize=9)

    # Right: per-source
    _ax2 = axes[1]
    _sources = sorted(df["Metadata_Source"].unique(), key=lambda s: int(s.split("_")[1]))
    _n_compounds = df.groupby("Metadata_Source")["Metadata_JCP2022"].nunique().reindex(_sources)

    plot_boxstrip(
        _ax2,
        _df_melted,
        x="Metadata_Source",
        y="nmAP",
        hue="comparison",
        order=_sources,
        hue_order=_order,
        strip_alpha=0.6,
    )

    _ax2.set_xticks(range(len(_sources)))
    _labels = [f"{s.replace('_', ' ').title()}\n(n={_n_compounds[s]})" for s in _sources]
    _ax2.set_xticklabels(_labels, rotation=45, ha="right")
    _ax2.set_xlabel("")
    _ax2.set_ylabel("Normalized mAP")
    _ax2.set_ylim(-0.2, 1.05)
    _ax2.legend(loc="upper right", fontsize=9, title="")
    _ax2.set_title(f"Per-Source Comparison{title_suffix}")

    fig.tight_layout()
    return fig


@app.function
def plot_map_diff_distribution(
    df: pd.DataFrame,
    title_suffix: str = "",
) -> plt.Figure:
    """Distribution of mAP difference (within - cross) per compound-source.

    Positive = within-source better (expected).
    Negative = cross-source better (unexpected, suggests batch effects).

    Returns the figure object.
    """
    _df = df.copy()
    _df["nmAP_diff"] = _df["nmAP_within"] - _df["nmAP_cross"]
    _df["_category"] = "Difference"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 3]})

    # Left: overall difference
    _ax1 = axes[0]
    plot_boxstrip(
        _ax1,
        _df,
        x="_category",
        y="nmAP_diff",
        order=["Difference"],
        strip_size=2,
        strip_alpha=0.4,
    )
    _ax1.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    _ax1.set_xlabel("")
    _ax1.set_ylabel("Normalized mAP difference\n(within - cross)")
    _ax1.set_title(f"Overall{title_suffix}")

    _n_positive = (_df["nmAP_diff"] > 0).sum()
    _n_negative = (_df["nmAP_diff"] < 0).sum()
    _mean_val = _df["nmAP_diff"].mean()
    _ax1.annotate(
        f"mu={_mean_val:.2f}\n"
        f"Within > Cross: {_n_positive} ({100 * _n_positive / len(_df):.1f}%)\n"
        f"Cross > Within: {_n_negative} ({100 * _n_negative / len(_df):.1f}%)",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
    )

    # Right: per-source
    _ax2 = axes[1]
    _sources = sorted(_df["Metadata_Source"].unique(), key=lambda s: int(s.split("_")[1]))
    plot_boxstrip(
        _ax2,
        _df,
        x="Metadata_Source",
        y="nmAP_diff",
        order=_sources,
        strip_alpha=0.5,
    )
    _ax2.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    _ax2.set_xticks(range(len(_sources)))
    _ax2.set_xticklabels([s.replace("_", " ").title() for s in _sources], rotation=45, ha="right")
    _ax2.set_xlabel("")
    _ax2.set_ylabel("Normalized mAP difference (within - cross)")
    _ax2.set_title(f"Per-Source Difference{title_suffix}")

    fig.tight_layout()
    return fig


@app.function
def compute_summary_stats(
    df: pd.DataFrame,
    preprocessing: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute summary statistics for the reproducibility analysis.

    Returns (overall_df, per_source_df, problematic_df).
    """
    _df = df.copy()
    _df["nmAP_diff"] = _df["nmAP_within"] - _df["nmAP_cross"]

    # Significance counts
    _n_both_sig = (_df["sig_within"] & _df["sig_cross"]).sum()
    _n_within_only = (_df["sig_within"] & ~_df["sig_cross"]).sum()
    _n_cross_only = (~_df["sig_within"] & _df["sig_cross"]).sum()
    _n_neither = (~_df["sig_within"] & ~_df["sig_cross"]).sum()

    _both_sig = _df[_df["sig_within"] & _df["sig_cross"]]

    _overall = {
        "preprocessing": preprocessing,
        "n_compound_source_pairs": len(_df),
        "n_compounds": _df["Metadata_JCP2022"].nunique(),
        "n_sources": _df["Metadata_Source"].nunique(),
        "n_both_significant": _n_both_sig,
        "n_within_only_significant": _n_within_only,
        "n_cross_only_significant": _n_cross_only,
        "n_neither_significant": _n_neither,
        "pct_both_significant": 100 * _n_both_sig / len(_df),
        "mean_nmAP_within_all": _df["nmAP_within"].mean(),
        "mean_nmAP_cross_all": _df["nmAP_cross"].mean(),
        "median_nmAP_diff_all": _df["nmAP_diff"].median(),
        "pct_within_better_all": 100 * (_df["nmAP_diff"] > 0).mean(),
        "correlation_all": _df["nmAP_within"].corr(_df["nmAP_cross"]),
    }

    if len(_both_sig) > 0:
        _both_sig_diff = _both_sig["nmAP_within"] - _both_sig["nmAP_cross"]
        _overall.update(
            {
                "mean_nmAP_within_bothsig": _both_sig["nmAP_within"].mean(),
                "mean_nmAP_cross_bothsig": _both_sig["nmAP_cross"].mean(),
                "median_nmAP_diff_bothsig": _both_sig_diff.median(),
                "pct_within_better_bothsig": 100 * (_both_sig_diff > 0).mean(),
                "correlation_bothsig": _both_sig["nmAP_within"].corr(_both_sig["nmAP_cross"]),
            }
        )

    _overall_df = pd.DataFrame([_overall])

    _per_source = (
        _df.groupby("Metadata_Source")
        .agg(
            n_compounds=("Metadata_JCP2022", "nunique"),
            mean_nmAP_within=("nmAP_within", "mean"),
            mean_nmAP_cross=("nmAP_cross", "mean"),
            median_nmAP_diff=("nmAP_diff", "median"),
            pct_within_better=("nmAP_diff", lambda x: 100 * (x > 0).mean()),
        )
        .reset_index()
    )

    _problematic = _df[_df["nmAP_diff"] < -0.1].sort_values("nmAP_diff")

    return _overall_df, _per_source, _problematic


# ---------------------------------------------------------------------------
# Build title suffix
# ---------------------------------------------------------------------------


@app.function
def build_title_suffix(preprocessing: str, filter_name: str) -> str:
    """Build a human-readable title suffix from config names."""
    if "target2" in preprocessing.lower():
        _suffix = " (TARGET2)"
        if filter_name == "no_source9":
            _suffix = " (TARGET2, no source_9)"
    elif "poscon" in preprocessing.lower():
        _suffix = " (Poscons)"
    else:
        _suffix = f" ({preprocessing})"
    return _suffix


# ---------------------------------------------------------------------------
# Scatter plot: within vs cross
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Within-source vs cross-source scatter")
    return


@app.cell
def _(df_repro, filter_dropdown, preprocessing_dropdown):
    _title_suffix = build_title_suffix(preprocessing_dropdown.value, filter_dropdown.value)
    fig_scatter = plot_within_vs_cross_scatter(df_repro, _title_suffix)
    fig_scatter
    return (fig_scatter,)


# ---------------------------------------------------------------------------
# Box plot: mAP comparison
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Within vs cross-source mAP comparison")
    return


@app.cell
def _(df_repro, filter_dropdown, preprocessing_dropdown):
    _title_suffix = build_title_suffix(preprocessing_dropdown.value, filter_dropdown.value)
    fig_box = plot_map_comparison_box(df_repro, _title_suffix)
    fig_box
    return (fig_box,)


# ---------------------------------------------------------------------------
# Difference distribution
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md("## mAP difference distribution (within - cross)")
    return


@app.cell
def _(df_repro, filter_dropdown, preprocessing_dropdown):
    _title_suffix = build_title_suffix(preprocessing_dropdown.value, filter_dropdown.value)
    fig_diff = plot_map_diff_distribution(df_repro, _title_suffix)
    fig_diff
    return (fig_diff,)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Summary statistics")
    return


@app.cell
def _(df_repro, filter_dropdown, mo, preprocessing_dropdown):
    _subdir_name = (
        preprocessing_dropdown.value
        if filter_dropdown.value == "all_sources"
        else f"{preprocessing_dropdown.value}_{filter_dropdown.value}"
    )
    overall_stats, per_source_stats, problematic_pairs = compute_summary_stats(df_repro, _subdir_name)
    mo.md(f"""
    ### Overall

    {mo.as_html(overall_stats)}

    ### Per-source

    {mo.as_html(per_source_stats)}

    ### Problematic pairs (cross > within by > 0.1)

    **{len(problematic_pairs)}** compound-source pairs where cross-source mAP exceeds
    within-source mAP by more than 0.1.
    """)
    return (overall_stats, per_source_stats, problematic_pairs)


@app.cell
def _(mo, problematic_pairs):
    mo.stop(len(problematic_pairs) == 0, mo.md("No problematic pairs found."))
    mo.ui.dataframe(problematic_pairs)
    return


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Save outputs")
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save plots and statistics")
    save_button
    return (save_button,)


@app.cell
def _(
    dataset_dropdown,
    df_repro,
    fig_box,
    fig_diff,
    fig_scatter,
    filter_dropdown,
    mo,
    overall_stats,
    per_source_stats,
    preprocessing_dropdown,
    problematic_pairs,
    save_button,
):
    mo.stop(not save_button.value)

    _subdir_name = (
        preprocessing_dropdown.value
        if filter_dropdown.value == "all_sources"
        else f"{preprocessing_dropdown.value}_{filter_dropdown.value}"
    )
    _output_dir = OUTPUT_BASE / dataset_dropdown.value / _subdir_name
    _output_dir.mkdir(parents=True, exist_ok=True)

    fig_scatter.savefig(_output_dir / "within_vs_cross_scatter.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_box.savefig(_output_dir / "map_comparison_box.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_diff.savefig(_output_dir / "map_diff_distribution.png", dpi=DEFAULT_DPI, bbox_inches="tight")

    overall_stats.to_csv(_output_dir / "summary_overall.csv", index=False)
    per_source_stats.to_csv(_output_dir / "summary_per_source.csv", index=False)
    if len(problematic_pairs) > 0:
        problematic_pairs.to_csv(_output_dir / "problematic_pairs.csv", index=False)
        logger.warning(f"Found {len(problematic_pairs)} compound-source pairs where cross > within by >0.1")

    logger.info(f"Saved outputs to {_output_dir}")
    mo.md(f"Saved all outputs to `{_output_dir}`")
    return


@app.function
def run_cross_source(
    dataset: str = "compound_no_source7",
    preprocessing: str = "activity_only_target2",
    filter_name: str = "all_sources",
    output_dir=None,
) -> str:
    """Run cross-source reproducibility analysis for one config combo.

    Queries within-source vs cross-source data, creates scatter, box, and
    difference plots, saves CSVs and PNGs.

    Called from workflow.py. Returns the output directory path.

    Args:
        dataset: e.g. "compound_no_source7"
        preprocessing: e.g. "activity_only_target2"
        filter_name: e.g. "all_sources" or "no_source9"
        output_dir: Override base output directory (default: PROCESSED_DATA_DIR/cross-source-reproducibility)
    """
    import matplotlib.pyplot as plt

    from nb00_ss_config import DEFAULT_DPI, PROCESSED_DATA_DIR

    _subdir_name = preprocessing if filter_name == "all_sources" else f"{preprocessing}_{filter_name}"

    if output_dir is None:
        _out = PROCESSED_DATA_DIR / "cross-source-reproducibility" / dataset / _subdir_name
    else:
        _out = Path(output_dir) / dataset / _subdir_name
    _out.mkdir(parents=True, exist_ok=True)

    _df = query_cross_source_reproducibility(dataset, preprocessing, filter_name)
    if len(_df) == 0:
        logger.warning(f"No cross-source data for {dataset}/{preprocessing}/{filter_name}")
        return str(_out)

    _title_suffix = build_title_suffix(preprocessing, filter_name)

    # Scatter plot
    _fig_scatter = plot_within_vs_cross_scatter(_df, _title_suffix)
    _fig_scatter.savefig(_out / "within_vs_cross_scatter.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(_fig_scatter)

    # Box plot
    _fig_box = plot_map_comparison_box(_df, _title_suffix)
    _fig_box.savefig(_out / "map_comparison_box.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(_fig_box)

    # Difference distribution
    _fig_diff = plot_map_diff_distribution(_df, _title_suffix)
    _fig_diff.savefig(_out / "map_diff_distribution.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(_fig_diff)

    # Summary stats
    _overall, _per_source, _problematic = compute_summary_stats(_df, _subdir_name)
    _overall.to_csv(_out / "summary_overall.csv", index=False)
    _per_source.to_csv(_out / "summary_per_source.csv", index=False)
    if len(_problematic) > 0:
        _problematic.to_csv(_out / "problematic_pairs.csv", index=False)

    logger.success(f"Saved cross-source outputs to {_out}")
    return str(_out)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
