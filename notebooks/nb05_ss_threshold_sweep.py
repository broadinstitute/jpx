# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb==1.5.2",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
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

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from nb00_ss_config import COPAIRS_RESULTS_DB, DEFAULT_DPI


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Activity Threshold Sensitivity

    How sensitive are consistency results to the activity p-value threshold used
    to define "active" compounds? This notebook sweeps thresholds (0.05 - 0.30)
    and measures impact on the number of significant consistency groups and their
    mean nMAP effect size.

    Sweep results live in `consistency_results` with `_preprocessing LIKE '%_sweep'`.
    The default threshold (0.10) matches regular (non-sweep) runs.

    **Verdict:** Relaxing the threshold increases discovery count modestly but
    dilutes effect size. The 0.10 default is a reasonable balance.

    *Related: [GitHub Issue #22](https://github.com/broadinstitute/jpx/issues/22)*
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
    dataset_dropdown
    return (dataset_dropdown,)


@app.function
def query_threshold_sweep(dataset: str) -> pd.DataFrame:
    """Query threshold sweep results aggregated by annotation and threshold."""
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    query = """
    SELECT
        _activity_threshold as threshold,
        _group_type as annotation,
        COUNT(*) as n_groups,
        SUM(below_corrected_p::int) as n_significant,
        ROUND(100.0 * SUM(below_corrected_p::int) / COUNT(*), 1) as pct_significant,
        ROUND(AVG(mean_normalized_average_precision), 4) as avg_nmap_all,
        ROUND(STDDEV(mean_normalized_average_precision), 4) as std_nmap_all,
        ROUND(AVG(CASE WHEN below_corrected_p THEN mean_normalized_average_precision END), 4) as avg_nmap_significant,
        ROUND(STDDEV(CASE WHEN below_corrected_p THEN mean_normalized_average_precision END), 4) as std_nmap_significant,
        SUM(n_perturbations) as total_perturbations
    FROM consistency_results
    WHERE _preprocessing LIKE '%_sweep'
      AND _dataset = ?
    GROUP BY _activity_threshold, _group_type
    ORDER BY _group_type, _activity_threshold
    """
    df = con.execute(query, [dataset]).df()
    con.close()
    return df


@app.cell
def _(dataset_dropdown, mo):
    df = query_threshold_sweep(dataset_dropdown.value)
    if df.empty:
        mo.stop(True, mo.md("**No threshold sweep results found.** Run `just run` first."))
    mo.md(
        f"**{len(df)} rows** across {df['annotation'].nunique()} annotations and {df['threshold'].nunique()} thresholds"
    )
    return (df,)


@app.function
def plot_threshold_summary(df: pd.DataFrame) -> plt.Figure:
    """Three-panel summary: discovery count, significance rate, effect size."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    annotations = df["annotation"].unique()
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    color_map = dict(zip(annotations, colors))

    for annotation in annotations:
        subset = df[df["annotation"] == annotation]
        c = color_map[annotation]

        axes[0].plot(
            subset["threshold"], subset["n_significant"], "o-", color=c, label=annotation, markersize=4, linewidth=1
        )
        axes[1].plot(
            subset["threshold"], subset["pct_significant"], "o-", color=c, label=annotation, markersize=4, linewidth=1
        )
        axes[2].errorbar(
            subset["threshold"],
            subset["avg_nmap_significant"],
            yerr=subset["std_nmap_significant"] / np.sqrt(subset["n_significant"]),
            fmt="o-",
            color=c,
            label=annotation,
            markersize=4,
            linewidth=1,
            capsize=0,
            elinewidth=0.5,
            alpha=0.8,
        )

    titles = ["Discovery Count", "Significance Rate", "Effect Size"]
    ylabels = ["N Significant", "% Significant", "Mean nMAP (sig.)"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_xticks(sorted(df["threshold"].unique()))
        ax.tick_params(labelsize=7)

    fig.supxlabel("Activity Threshold (p-value)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(annotations), fontsize=6, bbox_to_anchor=(0.5, -0.12))
    plt.suptitle("Threshold Sensitivity Analysis", fontsize=10, y=1.02)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    return fig


@app.cell
def _(df):
    fig_summary = plot_threshold_summary(df)
    fig_summary
    return (fig_summary,)


@app.function
def plot_threshold_tradeoff(df: pd.DataFrame) -> plt.Figure:
    """Trade-off plot: N significant vs mean nMAP per annotation trajectory."""
    fig, ax = plt.subplots(figsize=(6, 4))

    annotations = df["annotation"].unique()
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    color_map = dict(zip(annotations, colors))

    for annotation in annotations:
        subset = df[df["annotation"] == annotation].sort_values("threshold")
        c = color_map[annotation]

        ax.plot(
            subset["n_significant"],
            subset["avg_nmap_significant"],
            "-",
            color=c,
            linewidth=1,
            alpha=0.8,
            zorder=2,
            label=annotation,
        )
        ax.scatter(
            subset["n_significant"],
            subset["avg_nmap_significant"],
            c=[c],
            s=15,
            edgecolor="white",
            linewidth=0.2,
            zorder=3,
        )

        for _, row in subset.iterrows():
            sem = row["std_nmap_significant"] / np.sqrt(row["n_significant"])
            ax.errorbar(
                row["n_significant"],
                row["avg_nmap_significant"],
                yerr=sem,
                fmt="none",
                color=c,
                alpha=0.3,
                linewidth=0.5,
                capsize=0,
                zorder=1,
            )

    ax.set_xlabel("N Significant Groups", fontsize=8)
    ax.set_ylabel("Mean nMAP (sig. only)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.legend(title="Annotation", loc="upper right", fontsize=8, title_fontsize=9)
    plt.title("Trade-off: Discovery Count vs Effect Size", fontsize=9)
    return fig


@app.cell
def _(df):
    fig_tradeoff = plot_threshold_tradeoff(df)
    fig_tradeoff
    return (fig_tradeoff,)


@app.cell
def _(dataset_dropdown, fig_summary, fig_tradeoff, mo):
    outdir = Path("data/processed/exploration/0.05") / dataset_dropdown.value
    outdir.mkdir(parents=True, exist_ok=True)

    fig_summary.savefig(outdir / "threshold_sweep_summary.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    fig_tradeoff.savefig(
        outdir / "threshold_sweep_tradeoff.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white"
    )

    mo.md(f"Saved to `{outdir}/`")
    return


@app.cell
def _(df, mo):
    lines = ["## Trade-off Summary (most stringent - most lenient)\n"]
    for annotation in df["annotation"].unique():
        subset = df[df["annotation"] == annotation].sort_values("threshold")
        low = subset.iloc[0]
        high = subset.iloc[-1]

        sig_change = high["n_significant"] - low["n_significant"]
        sig_pct = 100 * sig_change / low["n_significant"] if low["n_significant"] > 0 else 0
        nmap_pct = (
            100 * (high["avg_nmap_significant"] - low["avg_nmap_significant"]) / low["avg_nmap_significant"]
            if low["avg_nmap_significant"] > 0
            else 0
        )

        lines.append(f"### {annotation}")
        lines.append(
            f"- N significant: {int(low['n_significant'])} - {int(high['n_significant'])} "
            f"({'+' if sig_change >= 0 else ''}{int(sig_change)}, "
            f"{'+' if sig_pct >= 0 else ''}{sig_pct:.0f}%)"
        )
        lines.append(
            f"- Avg nMAP: {low['avg_nmap_significant']:.4f} - {high['avg_nmap_significant']:.4f} "
            f"({'+' if nmap_pct >= 0 else ''}{nmap_pct:.0f}%)\n"
        )

    mo.md("\n".join(lines))
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
