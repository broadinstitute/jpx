# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "scipy==1.17.1",
#     "seaborn==0.13.2",
#     "scikit-learn==1.8.0",
#     "scanpy==1.12.1",
#     "python-dotenv",
#     "loguru==0.7.3",
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
    import scanpy as sc
    import seaborn as sns
    from scipy import stats

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import DEFAULT_DPI

    SIG_COLOR = "#2171b5"
    NONSIG_COLOR = "#a0a0a0"

    BBOX_STYLE = {
        "boxstyle": "round,pad=0.3",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.8,
    }

    MAP_COL = "mean_normalized_average_precision"
    SIG_COL = "below_corrected_p"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualization

    Foundation notebook providing plotting functions, constants, and statistical helpers
    for activity and consistency analysis. Other notebooks import via
    `from nb04_ss_visualization import plot_scatter_with_marginals, ...`

    **Constants:** `SIG_COLOR`, `NONSIG_COLOR`, `BBOX_STYLE`, `MAP_COL`, `SIG_COL`

    **Dataclass:** `TraitConfig` (col, label, xlim, panel_label)

    **Statistical helpers:**
    - `split_by_significance()` - split DataFrame by significance flag
    - `compute_activity_stats()` - Spearman + Mann-Whitney for trait vs activity
    - `compute_partial_correlation()` - partial Spearman controlling for covariates
    - `format_stat_annotation()` - format stats for plot annotation

    **Plot functions:**
    - `plot_categorical_box()` - box plot of activity by categorical variable
    - `plot_scatter_with_marginals()` - scatter + marginal histograms, colored by significance
    - `plot_hexbin_with_marginals()` - hexbin density + marginal histograms
    - `compute_umap_bounds()` - IQR-based axis limits for UMAP
    - `plot_violin_by_group()` - violin comparing distributions
    - `plot_survival_curve()` - complementary CDF curves
    - `plot_activity_rate_by_bin()` - bar chart of activity rate by binned variable
    - `plot_dotplot()` - Cleveland-style dot plot with significance coloring
    - `plot_plate_heatmap()` - multiwell plate heatmap
    """)
    return


@app.function
def split_by_significance(
    df: pd.DataFrame,
    sig_col: str = SIG_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into significant and non-significant subsets."""
    return df[df[sig_col]], df[~df[sig_col]]


@app.function
def compute_activity_stats(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = MAP_COL,
    sig_col: str = SIG_COL,
) -> dict:
    """Compute Spearman correlation and Mann-Whitney U test for trait vs activity."""
    valid = df[[x_col, y_col, sig_col]].dropna()
    sig, nonsig = split_by_significance(valid, sig_col)

    r, p = stats.spearmanr(valid[x_col], valid[y_col])
    mw_stat, mw_p = stats.mannwhitneyu(
        sig[x_col].dropna(),
        nonsig[x_col].dropna(),
        alternative="two-sided",
    )

    return {
        "spearman_r": r,
        "spearman_p": p,
        "mw_stat": mw_stat,
        "mw_p": mw_p,
        "n": len(valid),
        "n_sig": len(sig),
        "n_nonsig": len(nonsig),
    }


@app.function
def compute_partial_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    covariate_cols: list[str],
) -> dict:
    """Compute partial Spearman correlation controlling for covariates."""
    from sklearn.linear_model import LinearRegression

    cols_needed = [x_col, y_col] + covariate_cols
    valid = df[cols_needed].dropna()
    if len(valid) < 10:
        return {"partial_r": np.nan, "n": len(valid)}

    x = valid[x_col].values
    y = valid[y_col].values
    covariates = valid[covariate_cols].values

    model_x = LinearRegression().fit(covariates, x)
    x_resid = x - model_x.predict(covariates)

    model_y = LinearRegression().fit(covariates, y)
    y_resid = y - model_y.predict(covariates)

    r, _ = stats.spearmanr(x_resid, y_resid)

    return {"partial_r": r, "n": len(valid)}


@app.function
def format_stat_annotation(
    spearman_r: float,
    mw_p: float,
    n: int | None = None,
    compact: bool = True,
) -> str:
    """Format Spearman correlation and Mann-Whitney test for plot annotation."""
    mw_str = f"p < {mw_p:.0e}" if mw_p < 0.001 else f"p = {mw_p:.3f}"

    if compact:
        text = f"ρ = {spearman_r:.2f}\nMW {mw_str}"
    else:
        text = f"Spearman ρ = {spearman_r:.2f}\nMann-Whitney {mw_str}"

    if n is not None:
        text += f"\nn = {n:,}"

    return text


@app.function
def plot_categorical_box(
    df: pd.DataFrame,
    cat_col: str,
    y_col: str = MAP_COL,
    order: list | None = None,
    labels: list[str] | None = None,
    colors: list[str] | dict | None = None,
    xlabel: str = "",
    ylabel: str = "Normalized mAP",
    ax: plt.Axes | None = None,
    output_dir: Path | None = None,
    filename: str = "categorical_box.png",
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Generic box plot of activity by categorical variable."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 4))

    valid = df[[cat_col, y_col]].dropna()

    if order is None:
        order = sorted(valid[cat_col].unique())

    if colors is None:
        palette = "#3498db"
    elif isinstance(colors, dict):
        palette = colors
    else:
        palette = dict(zip(order, colors))

    sns.boxplot(
        data=valid,
        x=cat_col,
        y=y_col,
        order=order,
        hue=cat_col if isinstance(palette, dict) else None,
        palette=palette if isinstance(palette, dict) else None,
        color=palette if isinstance(palette, str) else None,
        legend=False,
        width=0.6,
        ax=ax,
    )

    if labels is not None:
        ax.set_xticklabels(labels)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    y_max = valid[y_col].max()
    for i, val in enumerate(order):
        n = int((valid[cat_col] == val).sum())
        n_str = f"{n // 1000}k" if n >= 1000 else str(n)
        ax.annotate(n_str, xy=(i, y_max + 0.08), ha="center", fontsize=8, color="gray")
    ax.set_ylim(None, y_max + 0.15)

    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        plt.tight_layout()
        if output_dir:
            fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
            if save_pdf:
                fig.savefig(output_dir / filename.replace(".png", ".pdf"), bbox_inches="tight")
        return fig


@app.function
def plot_scatter_with_marginals(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = MAP_COL,
    sig_col: str = SIG_COL,
    x_label: str = "",
    y_label: str = "Normalized mAP",
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] = (0, 1),
    title: str = "",
    panel_label: str = "",
    fig: plt.Figure | None = None,
    gs_parent=None,
    output_dir: Path | None = None,
    filename: str = "scatter_marginals.png",
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> dict:
    """Scatter plot with marginal histograms, colored by significance."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    standalone = gs_parent is None

    valid = df[[x_col, y_col, sig_col]].dropna()
    sig, nonsig = split_by_significance(valid, sig_col)

    activity_stats = compute_activity_stats(df, x_col, y_col, sig_col)

    if x_lim is None:
        x_min, x_max = valid[x_col].quantile(0.01), valid[x_col].quantile(0.99)
        x_lim = (x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))

    if standalone:
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    else:
        gs = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs_parent, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05
        )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    scatter_alpha = (0.3, 0.5) if standalone else (0.2, 0.4)
    scatter_size = (6, 8) if standalone else (4, 6)
    font_size = 9 if standalone else 7
    n_bins = 50 if standalone else 40

    ax_main.scatter(
        nonsig[x_col],
        nonsig[y_col],
        c=NONSIG_COLOR,
        alpha=scatter_alpha[0],
        s=scatter_size[0],
        linewidths=0,
        rasterized=True,
    )
    ax_main.scatter(
        sig[x_col],
        sig[y_col],
        c=SIG_COLOR,
        alpha=scatter_alpha[1],
        s=scatter_size[1],
        linewidths=0,
        rasterized=True,
    )

    ax_main.set_xlabel(x_label, fontsize=font_size)
    ax_main.set_ylabel(y_label, fontsize=font_size)
    ax_main.set_xlim(x_lim)
    ax_main.set_ylim(y_lim)
    ax_main.tick_params(labelsize=font_size - 2)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    bins_x = np.linspace(x_lim[0], x_lim[1], n_bins)
    ax_histx.hist(
        nonsig[x_col],
        bins=bins_x,
        alpha=0.5,
        color=NONSIG_COLOR,
        density=True,
        label="Not significant" if standalone else None,
    )
    ax_histx.hist(
        sig[x_col],
        bins=bins_x,
        alpha=0.7,
        color=SIG_COLOR,
        density=True,
        label="Significant" if standalone else None,
    )
    ax_histx.tick_params(labelbottom=False, labelsize=font_size - 3)
    ax_histx.spines["top"].set_visible(False)
    ax_histx.spines["right"].set_visible(False)

    if standalone:
        ax_histx.set_ylabel("Density")
        ax_histx.legend(loc="upper right", fontsize=8, frameon=False)
    else:
        ax_histx.spines["left"].set_visible(False)
        ax_histx.set_yticks([])
        if panel_label:
            ax_histx.set_title(panel_label, fontsize=10, fontweight="bold", loc="left")

    bins_y = np.linspace(y_lim[0], y_lim[1], n_bins)
    ax_histy.hist(
        nonsig[y_col],
        bins=bins_y,
        alpha=0.5,
        color=NONSIG_COLOR,
        orientation="horizontal",
        density=True,
    )
    ax_histy.hist(
        sig[y_col],
        bins=bins_y,
        alpha=0.7,
        color=SIG_COLOR,
        orientation="horizontal",
        density=True,
    )
    ax_histy.tick_params(labelleft=False, labelsize=font_size - 3)
    ax_histy.spines["top"].set_visible(False)
    ax_histy.spines["right"].set_visible(False)

    if standalone:
        ax_histy.set_xlabel("Density")
    else:
        ax_histy.spines["bottom"].set_visible(False)
        ax_histy.set_xticks([])

    annotation = format_stat_annotation(
        activity_stats["spearman_r"],
        activity_stats["mw_p"],
        n=activity_stats["n"] if standalone else None,
        compact=not standalone,
    )
    ax_main.annotate(
        annotation,
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=font_size,
        bbox=BBOX_STYLE,
    )

    if standalone:
        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)
        if output_dir:
            fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
            if save_pdf:
                fig.savefig(output_dir / filename.replace(".png", ".pdf"), bbox_inches="tight")

    return fig, activity_stats


@app.function
def plot_hexbin_with_marginals(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = MAP_COL,
    x_label: str = "",
    y_label: str = "Normalized mAP",
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] = (0, 1),
    title: str = "",
    stats_annotation: str | None = None,
    fig: plt.Figure | None = None,
    gs_parent=None,
    output_dir: Path | None = None,
    filename: str = "hexbin_marginals.png",
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> plt.Figure:
    """Hexbin density plot with marginal histograms (count-based)."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    standalone = gs_parent is None
    valid = df[[x_col, y_col]].dropna()

    if x_lim is None:
        x_lim = (valid[x_col].quantile(0.01), valid[x_col].quantile(0.99))

    if standalone:
        fig = plt.figure(figsize=(7, 6))
        gs = fig.add_gridspec(2, 3, width_ratios=[4, 1, 0.2], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    else:
        gs = GridSpecFromSubplotSpec(
            2, 3, subplot_spec=gs_parent, width_ratios=[4, 1, 0.2], height_ratios=[1, 4], wspace=0.05, hspace=0.05
        )

    ax_cbar = fig.add_subplot(gs[1, 2])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    hb = ax_main.hexbin(
        valid[x_col],
        valid[y_col],
        gridsize=50,
        cmap="Blues",
        mincnt=1,
        bins="log",
        rasterized=True,
    )
    cb = fig.colorbar(hb, cax=ax_cbar)
    cb.set_label("Count (log scale)", fontsize=9)

    ax_main.set_xlabel(x_label, fontsize=10)
    ax_main.set_ylabel(y_label, fontsize=10)
    ax_main.set_xlim(x_lim)
    ax_main.set_ylim(y_lim)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    bins_x = np.linspace(x_lim[0], x_lim[1], 50)
    ax_histx.hist(valid[x_col], bins=bins_x, color="#3498db", alpha=0.7)
    ax_histx.set_ylabel("Count")
    ax_histx.tick_params(labelbottom=False)
    ax_histx.spines["top"].set_visible(False)
    ax_histx.spines["right"].set_visible(False)

    bins_y = np.linspace(y_lim[0], y_lim[1], 50)
    ax_histy.hist(
        valid[y_col],
        bins=bins_y,
        orientation="horizontal",
        color="#3498db",
        alpha=0.7,
    )
    ax_histy.set_xlabel("Count")
    ax_histy.tick_params(labelleft=False)
    ax_histy.spines["top"].set_visible(False)
    ax_histy.spines["right"].set_visible(False)

    if stats_annotation:
        ax_main.annotate(
            stats_annotation,
            xy=(0.97, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
        )

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    if output_dir:
        fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
        if save_pdf:
            fig.savefig(output_dir / filename.replace(".png", ".pdf"), bbox_inches="tight")

    return fig


@app.function
def compute_umap_bounds(adata: sc.AnnData, iqr_k: float = 3.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute axis limits excluding outliers via IQR method."""
    coords = adata.obsm["X_umap"]

    def iqr_bounds(values: np.ndarray) -> tuple[float, float]:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        return (q1 - iqr_k * iqr, q3 + iqr_k * iqr)

    xlim = iqr_bounds(coords[:, 0])
    ylim = iqr_bounds(coords[:, 1])

    return xlim, ylim


@app.function
def plot_violin_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    group_order: list[str] | None = None,
    group_colors: dict[str, str] | None = None,
    ylabel: str = "",
    title: str = "",
    ax: plt.Axes | None = None,
    output_dir: Path | None = None,
    filename: str = "violin_by_group.png",
    dpi: int = DEFAULT_DPI,
) -> plt.Axes:
    """Violin plot comparing distributions between groups."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))

    if group_order is None:
        group_order = sorted(df[group_col].unique())

    if group_colors is None:
        group_colors = {group_order[0]: NONSIG_COLOR, group_order[1]: SIG_COLOR}

    sns.violinplot(
        data=df,
        x=group_col,
        y=value_col,
        hue=group_col,
        palette=group_colors,
        order=group_order,
        cut=0,
        inner="box",
        legend=False,
        ax=ax,
    )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    if title:
        ax.set_title(title)

    y_max = ax.get_ylim()[1]
    for i, group in enumerate(group_order):
        n = (df[group_col] == group).sum()
        ax.text(i, y_max * 0.95, f"n={n:,}", ha="center", fontsize=9, color="gray")

    if standalone:
        plt.tight_layout()
        if output_dir:
            fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
        return fig

    return ax


@app.function
def plot_survival_curve(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    group_order: list[str] | None = None,
    group_colors: dict[str, str] | None = None,
    xlabel: str = "Threshold",
    ylabel: str = "Fraction above threshold",
    title: str = "",
    xlim: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    output_dir: Path | None = None,
    filename: str = "survival_curve.png",
    dpi: int = DEFAULT_DPI,
) -> plt.Axes:
    """Plot survival/complementary CDF curves for multiple groups."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))

    if group_order is None:
        group_order = sorted(df[group_col].unique())

    if group_colors is None:
        group_colors = {group_order[0]: NONSIG_COLOR, group_order[1]: SIG_COLOR}

    for group in group_order:
        values = df[df[group_col] == group][value_col].values
        values_sorted = np.sort(values)
        survival = 1 - np.arange(1, len(values) + 1) / len(values)
        ax.plot(
            values_sorted,
            survival,
            label=f"{group} (n={len(values):,})",
            color=group_colors.get(group, None),
            linewidth=2,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    ax.legend(loc="upper right")

    if standalone:
        plt.tight_layout()
        if output_dir:
            fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
        return fig

    return ax


@app.function
def plot_activity_rate_by_bin(
    df: pd.DataFrame,
    value_col: str,
    activity_col: str = SIG_COL,
    bins: list[float] | None = None,
    bin_labels: list[str] | None = None,
    xlabel: str = "",
    ylabel: str = "% Phenotypically Active",
    title: str = "",
    cmap: str = "Blues",
    ax: plt.Axes | None = None,
    output_dir: Path | None = None,
    filename: str = "activity_rate_by_bin.png",
    dpi: int = DEFAULT_DPI,
) -> tuple[plt.Axes, pd.DataFrame]:
    """Bar chart showing activity rate by binned continuous variable."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))

    if bins is None:
        bins = [-0.001, 0, 0.02, 0.04, 0.08, 0.15, 1]
    if bin_labels is None:
        bin_labels = ["0", "0-2%", "2-4%", "4-8%", "8-15%", ">15%"]

    df = df.copy()
    df["_bin"] = pd.cut(df[value_col], bins=bins, labels=bin_labels)
    bin_stats = df.groupby("_bin", observed=False).agg({activity_col: ["mean", "count"]}).reset_index()
    bin_stats.columns = ["bin", "active_frac", "count"]
    bin_stats = bin_stats[bin_stats["count"] > 0]

    colors = plt.colormaps[cmap](np.linspace(0.3, 0.9, len(bin_stats)))
    bars = ax.bar(range(len(bin_stats)), bin_stats["active_frac"] * 100, color=colors, edgecolor="black")

    ax.set_xticks(range(len(bin_stats)))
    ax.set_xticklabels(bin_stats["bin"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    for i, (bar, count) in enumerate(zip(bars, bin_stats["count"])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"n={count:,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )

    ax.set_ylim(0, 105)

    baseline = df[activity_col].mean() * 100
    ax.axhline(baseline, color="red", linestyle="--", label=f"Baseline: {baseline:.1f}%")
    ax.legend()

    if standalone:
        plt.tight_layout()
        if output_dir:
            fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
        return fig, bin_stats

    return ax, bin_stats


@app.function
def plot_dotplot(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    count_col: str | None = None,
    sig_col: str | None = None,
    top_n: int = 40,
    title: str = "",
    xlabel: str = "Value",
    p_threshold: float = 0.05,
    size_scale: float = 15.0,
    size_range: tuple[float, float] = (30, 150),
    ax: plt.Axes | None = None,
    output_dir: Path | None = None,
    filename: str = "dotplot.png",
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> plt.Axes:
    """Cleveland-style dot plot with optional significance coloring."""
    from matplotlib.lines import Line2D

    standalone = ax is None

    plot_df = df.nlargest(top_n, value_col).copy()
    plot_df = plot_df.sort_values(value_col, ascending=True)

    if standalone:
        fig, ax = plt.subplots(figsize=(8, max(6, len(plot_df) * 0.25)))

    if count_col is not None:
        sizes = (plot_df[count_col] * size_scale).clip(size_range[0], size_range[1])
    else:
        sizes = 60

    if sig_col is not None:
        colors = [SIG_COLOR if sig else NONSIG_COLOR for sig in plot_df[sig_col]]
    else:
        colors = SIG_COLOR

    y_pos = range(len(plot_df))
    ax.scatter(
        plot_df[value_col],
        y_pos,
        s=sizes,
        c=colors,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
    )

    for y in y_pos:
        ax.axhline(y=y, color="#ecf0f1", linewidth=0.5, zorder=0)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df[label_col], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)

    if count_col is not None:
        x_max = plot_df[value_col].max()
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax.text(
                x_max * 1.02,
                i,
                f"n={int(row[count_col])}",
                va="center",
                fontsize=8,
                color="#7f8c8d",
            )

    if sig_col is not None:
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=SIG_COLOR, markersize=10, label=f"p<{p_threshold}"),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=NONSIG_COLOR, markersize=10, label="Not significant"
            ),
        ]
        ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    ax.set_xlim(left=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        plt.tight_layout()
        if output_dir:
            fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
            if save_pdf:
                fig.savefig(output_dir / filename.replace(".png", ".pdf"), bbox_inches="tight")
        return fig

    return ax


@app.function
def plot_plate_heatmap(
    df: pd.DataFrame,
    row_col: str = "row",
    col_col: str = "col",
    value_col: str = "value",
    title: str | None = None,
    cbar_label: str | None = None,
    cmap: str = "YlGnBu",
    figsize: tuple[float, float] = (14, 7),
    output_path: Path | str | None = None,
    ax: plt.Axes | None = None,
    dpi: int = DEFAULT_DPI,
) -> plt.Axes:
    """Plot a heatmap styled like a multiwell plate diagram."""
    pivot = df.pivot(index=row_col, columns=col_col, values=value_col)

    unique_rows = sorted(pivot.index, key=lambda x: (len(x), x))
    pivot = pivot.reindex(unique_rows)
    pivot = pivot[sorted(pivot.columns)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": cbar_label or "", "shrink": 0.6},
        xticklabels=True,
        yticklabels=True,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if title:
        ax.set_title(title, fontsize=12, pad=10)

    if output_path:
        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return ax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo gallery

    Synthetic data showing each plot function. These are importable via
    `from nb04_ss_visualization import plot_scatter_with_marginals, SIG_COLOR, ...`
    """)
    return


@app.cell
def _():
    np.random.seed(42)
    n = 500
    demo_df = pd.DataFrame(
        {
            MAP_COL: np.clip(np.random.beta(2, 5, n), 0, 1),
            "trait_x": np.random.randn(n) * 2 + 3,
            "category": np.random.choice(["A", "B", "C"], n),
            "group": np.random.choice(["Active", "Inactive"], n, p=[0.15, 0.85]),
        }
    )
    demo_df[SIG_COL] = demo_df[MAP_COL] > demo_df[MAP_COL].quantile(0.85)
    return (demo_df,)


@app.cell
def _(mo):
    mo.md("""
    ### Scatter with marginals
    """)
    return


@app.cell
def _(demo_df):
    fig_scatter, stats_result = plot_scatter_with_marginals(
        demo_df,
        x_col="trait_x",
        x_label="Synthetic Trait",
        title="Demo: Scatter + Marginals",
    )
    fig_scatter
    return


@app.cell
def _(mo):
    mo.md("""
    ### Box plot by category
    """)
    return


@app.cell
def _(demo_df):
    fig_box, ax_box = plt.subplots(figsize=(5, 4))
    plot_categorical_box(
        demo_df,
        cat_col="category",
        ax=ax_box,
        ylabel="Normalized mAP",
    )
    ax_box.set_title("Demo: Categorical Box Plot")
    fig_box
    return


@app.cell
def _(mo):
    mo.md("""
    ### Violin by group
    """)
    return


@app.cell
def _(demo_df):
    fig_violin, ax_violin = plt.subplots(figsize=(5, 4))
    plot_violin_by_group(
        demo_df,
        value_col=MAP_COL,
        group_col="group",
        ylabel="Normalized mAP",
        title="Demo: Violin by Group",
        ax=ax_violin,
    )
    fig_violin
    return


@app.cell
def _(mo):
    mo.md("""
    ### Survival curve (complementary CDF)
    """)
    return


@app.cell
def _(demo_df):
    fig_surv, ax_surv = plt.subplots(figsize=(5, 4))
    plot_survival_curve(
        demo_df,
        value_col=MAP_COL,
        group_col="group",
        xlabel="Normalized mAP",
        title="Demo: Survival Curve",
        ax=ax_surv,
    )
    fig_surv
    return


@app.cell
def _(mo):
    mo.md("""
    ### Dot plot (top entries)
    """)
    return


@app.cell
def _():
    dotplot_df = pd.DataFrame(
        {
            "target": [f"Target_{i}" for i in range(20)],
            "nmap": np.random.beta(3, 5, 20),
            "n_compounds": np.random.randint(2, 30, 20),
            "significant": np.random.random(20) > 0.5,
        }
    )
    fig_dot, ax_dot = plt.subplots(figsize=(7, 5))
    plot_dotplot(
        dotplot_df,
        value_col="nmap",
        label_col="target",
        count_col="n_compounds",
        sig_col="significant",
        top_n=15,
        title="Demo: Dot Plot",
        xlabel="Normalized mAP",
        ax=ax_dot,
    )
    fig_dot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Activity rate by bin
    """)
    return


@app.cell(hide_code=True)
def _(demo_df):
    _bin_df = demo_df.copy()
    _bin_df["cv_pct"] = np.random.uniform(0, 0.2, len(_bin_df))
    fig_bins, ax_bins = plt.subplots(figsize=(7, 5))
    plot_activity_rate_by_bin(
        _bin_df,
        value_col="cv_pct",
        activity_col=SIG_COL,
        xlabel="CV (%)",
        title="Demo: Activity Rate by Bin",
        ax=ax_bins,
    )
    fig_bins
    return


@app.cell
def _(mo):
    mo.md("""
    ### Plate heatmap
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Hexbin density plot
    """)
    return


@app.cell(hide_code=True)
def _(demo_df):
    fig_hex = plot_hexbin_with_marginals(
        demo_df,
        x_col="trait_x",
        x_label="Synthetic Trait",
        title="Demo: Hexbin + Marginals",
    )
    fig_hex
    return


@app.cell
def _():
    rows = list("ABCDEFGH")
    cols = list(range(1, 13))
    plate_data = pd.DataFrame(
        [(r, c, np.random.randint(500, 2000)) for r in rows for c in cols],
        columns=["row", "col", "value"],
    )
    fig_plate, ax_plate = plt.subplots(figsize=(10, 5))
    plot_plate_heatmap(
        plate_data,
        title="Demo: 96-Well Plate Heatmap",
        cbar_label="Cell Count",
        ax=ax_plate,
    )
    fig_plate
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
