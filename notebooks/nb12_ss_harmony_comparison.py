# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb==1.5.3",
#     "loguru==0.7.3",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "python-dotenv",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import json
    import sys
    from collections import OrderedDict
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from loguru import logger

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import DEFAULT_DPI, PROCESSED_DATA_DIR
    from nb02_ss_queries import query_activity_results, query_consistency_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Harmony Batch Correction Comparison

    Does re-tuning Harmony fix DL's batch problem?

    DL CPCNN features show 3.5x worse batch mixing than CellProfiler by iLISI
    (issues #17, #55). Recipe-Harmony was optimized for CP and applied to DL without
    re-tuning. This notebook tests whether in-repo Harmony with different batch
    keys improves DL's results.

    Compares 6 configurations (2 feature types x 3 Harmony variants):

    | Label | Dataset | Description |
    |-------|---------|-------------|
    | CP_recipe | compound_no_source7 | Recipe-Harmony (default) |
    | CP_rsc | compound_no_source7_rsc | In-repo Harmony, Metadata_Batch |
    | CP_rsc_source | compound_no_source7_rsc_source | In-repo Harmony, Metadata_Source |
    | DL_recipe | compound_DL_CPCNN_no_source7 | Recipe-Harmony (default) |
    | DL_rsc | compound_DL_CPCNN_no_source7_rsc | In-repo Harmony, Metadata_Batch |
    | DL_rsc_source | compound_DL_CPCNN_no_source7_rsc_source | In-repo Harmony, Metadata_Source |

    **Verdict:** No. CP outperforms DL on activity across all 3 Harmony variants.
    """)
    return


@app.cell
def _():
    OUTPUT_DIR = PROCESSED_DATA_DIR / "harmony-comparison"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    CONFIGS = OrderedDict(
        [
            ("CP_recipe", "compound_no_source7"),
            ("CP_rsc", "compound_no_source7_rsc"),
            ("CP_rsc_source", "compound_no_source7_rsc_source"),
            ("DL_recipe", "compound_DL_CPCNN_no_source7"),
            ("DL_rsc", "compound_DL_CPCNN_no_source7_rsc"),
            ("DL_rsc_source", "compound_DL_CPCNN_no_source7_rsc_source"),
        ]
    )

    # Colors: CP = blues, DL = oranges; saturation = Harmony variant
    CONFIG_COLORS = {
        "CP_recipe": "#2171b5",
        "CP_rsc": "#6baed6",
        "CP_rsc_source": "#bdd7e7",
        "DL_recipe": "#e6550d",
        "DL_rsc": "#fdae6b",
        "DL_rsc_source": "#fdd0a2",
    }

    # Markers: circle = recipe, triangle = _rsc (Batch), square = _rsc_source (Source)
    CONFIG_MARKERS = {
        "CP_recipe": "o",
        "CP_rsc": "^",
        "CP_rsc_source": "s",
        "DL_recipe": "o",
        "DL_rsc": "^",
        "DL_rsc_source": "s",
    }

    CONSISTENCY_TARGET = "repurposing"
    return CONFIGS, CONFIG_COLORS, CONFIG_MARKERS, CONSISTENCY_TARGET, OUTPUT_DIR


@app.function
def load_activity_summary(configs: OrderedDict) -> pd.DataFrame:
    """Load activity summary for all 6 configurations."""
    _rows = []
    for _label, _dataset in configs.items():
        _df = query_activity_results(_dataset)
        if _df.empty:
            logger.warning(f"No activity results for {_dataset}")
            continue
        _feature_type = "CP" if _label.startswith("CP") else "DL"
        _harmony = _label.split("_", 1)[1]
        _n_sig = (_df["corrected_p_value"] < 0.05).sum()
        _rows.append(
            {
                "label": _label,
                "dataset": _dataset,
                "feature_type": _feature_type,
                "harmony": _harmony,
                "total": len(_df),
                "sig": _n_sig,
                "pct_sig": 100.0 * _n_sig / len(_df),
                "mean_map": _df["mean_average_precision"].mean(),
                "mean_nmap": _df["mean_normalized_average_precision"].mean(),
            }
        )
    return pd.DataFrame(_rows)


@app.function
def load_consistency_summary(configs: OrderedDict, consistency_target: str) -> pd.DataFrame:
    """Load consistency summary for all 6 configurations."""
    _rows = []
    for _label, _dataset in configs.items():
        _df = query_consistency_results(_dataset, group_type=consistency_target)
        if _df.empty:
            logger.warning(f"No consistency results for {_dataset}")
            continue
        _feature_type = "CP" if _label.startswith("CP") else "DL"
        _harmony = _label.split("_", 1)[1]
        _n_sig = (_df["corrected_p_value"] < 0.05).sum()
        _rows.append(
            {
                "label": _label,
                "dataset": _dataset,
                "feature_type": _feature_type,
                "harmony": _harmony,
                "targets": len(_df),
                "sig": _n_sig,
                "pct_sig": 100.0 * _n_sig / len(_df),
                "mean_nmap": _df["mean_normalized_average_precision"].mean(),
            }
        )
    return pd.DataFrame(_rows)


@app.function
def compute_activity_correlations(configs: OrderedDict) -> pd.DataFrame:
    """Compute pairwise per-compound mAP correlations across all configs."""
    _activity_dfs = {}
    for _label, _dataset in configs.items():
        _df = query_activity_results(_dataset)
        if not _df.empty:
            _activity_dfs[_label] = _df.set_index("Metadata_JCP2022")["mean_average_precision"]

    _labels = list(_activity_dfs.keys())
    _corr_rows = []
    for _i, _l1 in enumerate(_labels):
        for _l2 in _labels[_i + 1 :]:
            _merged = pd.concat([_activity_dfs[_l1], _activity_dfs[_l2]], axis=1, join="inner")
            _merged.columns = ["a", "b"]
            _r = _merged["a"].corr(_merged["b"])
            _diff = (_merged["a"] - _merged["b"]).abs()
            _corr_rows.append(
                {
                    "pair": f"{_l1} vs {_l2}",
                    "r": _r,
                    "mean_diff": _diff.mean(),
                    "max_diff": _diff.max(),
                    "n_compounds": len(_merged),
                }
            )
    return pd.DataFrame(_corr_rows)


@app.function
def plot_activity_bar(summary: pd.DataFrame, config_colors: dict, dpi: int) -> plt.Figure:
    """Bar chart of activity rates across all 6 configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for _ax, _metric, _ylabel in [
        (axes[0], "pct_sig", "Significant compounds (%)"),
        (axes[1], "mean_nmap", "Mean nMAP"),
    ]:
        _x = np.arange(len(summary))
        _colors = [config_colors[_label] for _label in summary["label"]]
        _bars = _ax.bar(_x, summary[_metric], color=_colors, edgecolor="white", width=0.7)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(summary["label"], rotation=45, ha="right", fontsize=9)
        _ax.set_ylabel(_ylabel)

        for _bar, _val in zip(_bars, summary[_metric]):
            _fmt = f"{_val:.1f}%" if "%" in _ylabel else f"{_val:.4f}"
            _ax.text(
                _bar.get_x() + _bar.get_width() / 2,
                _bar.get_height(),
                _fmt,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_title("Activity: fraction significant (p < 0.05)")
    axes[1].set_title("Activity: mean normalized mAP")
    fig.suptitle(
        "Does re-tuning Harmony fix DL's activity deficit?",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


@app.function
def plot_consistency_bar(
    summary: pd.DataFrame,
    config_colors: dict,
    consistency_target: str,
    dpi: int,
) -> plt.Figure:
    """Bar chart of consistency results across all 6 configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for _ax, _metric, _ylabel in [
        (axes[0], "pct_sig", "Significant targets (%)"),
        (axes[1], "mean_nmap", "Mean nMAP"),
    ]:
        _x = np.arange(len(summary))
        _colors = [config_colors[_label] for _label in summary["label"]]
        _bars = _ax.bar(_x, summary[_metric], color=_colors, edgecolor="white", width=0.7)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(summary["label"], rotation=45, ha="right", fontsize=9)
        _ax.set_ylabel(_ylabel)

        for _bar, _val in zip(_bars, summary[_metric]):
            _fmt = f"{_val:.1f}%" if "%" in _ylabel else f"{_val:.4f}"
            _ax.text(
                _bar.get_x() + _bar.get_width() / 2,
                _bar.get_height(),
                _fmt,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_title(f"Consistency ({consistency_target}): fraction significant")
    axes[1].set_title(f"Consistency ({consistency_target}): mean nMAP")
    fig.suptitle(
        "Does re-tuning Harmony improve DL's consistency?",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


@app.function
def plot_correlation_heatmap(corr_df: pd.DataFrame, configs: OrderedDict, dpi: int) -> plt.Figure:
    """Heatmap of pairwise per-compound mAP correlations."""
    _labels = list(configs.keys())
    _n = len(_labels)
    _corr_matrix = pd.DataFrame(np.eye(_n), index=_labels, columns=_labels)

    for _, _row in corr_df.iterrows():
        _pair = _row["pair"]
        _l1, _l2 = _pair.split(" vs ")
        _corr_matrix.loc[_l1, _l2] = _row["r"]
        _corr_matrix.loc[_l2, _l1] = _row["r"]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        _corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        vmin=0.95,
        vmax=1.0,
        square=True,
        ax=ax,
    )
    ax.set_title(
        "Per-compound mAP correlation across Harmony variants",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


@app.cell(hide_code=True)
def _(CONFIGS, mo):
    mo.md(f"""
    ## Activity Summary

    Querying activity results for {len(CONFIGS)} configurations...
    """)
    return


@app.cell
def _(CONFIGS, mo):
    activity_summary = load_activity_summary(CONFIGS)
    mo.stop(
        activity_summary.empty,
        mo.md("**No activity results found.** Run the copairs pipeline first."),
    )
    mo.ui.dataframe(activity_summary)
    return (activity_summary,)


@app.cell
def _(CONFIG_COLORS, activity_summary):
    fig_activity = plot_activity_bar(activity_summary, CONFIG_COLORS, DEFAULT_DPI)
    fig_activity
    return (fig_activity,)


@app.cell
def _(OUTPUT_DIR, activity_summary, fig_activity):
    activity_summary.to_csv(OUTPUT_DIR / "activity_summary.csv", index=False)
    fig_activity.savefig(OUTPUT_DIR / "activity_comparison.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    logger.info(f"Saved activity summary and plot to {OUTPUT_DIR}")
    return


@app.cell(hide_code=True)
def _(CONFIGS, CONSISTENCY_TARGET, mo):
    mo.md(f"""
    ## Consistency Summary ({CONSISTENCY_TARGET})

    Querying consistency results for {len(CONFIGS)} configurations...
    """)
    return


@app.cell
def _(CONFIGS, CONSISTENCY_TARGET, mo):
    consistency_summary = load_consistency_summary(CONFIGS, CONSISTENCY_TARGET)
    mo.stop(
        consistency_summary.empty,
        mo.md("**No consistency results found.** Run the copairs pipeline first."),
    )
    mo.ui.dataframe(consistency_summary)
    return (consistency_summary,)


@app.cell
def _(CONFIG_COLORS, CONSISTENCY_TARGET, consistency_summary):
    fig_consistency = plot_consistency_bar(consistency_summary, CONFIG_COLORS, CONSISTENCY_TARGET, DEFAULT_DPI)
    fig_consistency
    return (fig_consistency,)


@app.cell
def _(OUTPUT_DIR, consistency_summary, fig_consistency):
    consistency_summary.to_csv(OUTPUT_DIR / "consistency_summary.csv", index=False)
    fig_consistency.savefig(OUTPUT_DIR / "consistency_comparison.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    logger.info(f"Saved consistency summary and plot to {OUTPUT_DIR}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-Compound mAP Correlations

    How correlated are per-compound mAP values across Harmony variants?
    High correlation means Harmony variant choice barely affects compound ranking.
    """)
    return


@app.cell
def _(CONFIGS, mo):
    corr_df = compute_activity_correlations(CONFIGS)
    mo.stop(
        corr_df.empty,
        mo.md("**No correlation data.** Need activity results for at least 2 configs."),
    )
    mo.ui.dataframe(corr_df)
    return (corr_df,)


@app.cell
def _(CONFIGS, corr_df):
    fig_corr = plot_correlation_heatmap(corr_df, CONFIGS, DEFAULT_DPI)
    fig_corr
    return (fig_corr,)


@app.cell
def _(OUTPUT_DIR, corr_df, fig_corr):
    corr_df.to_csv(OUTPUT_DIR / "correlations.csv", index=False)
    fig_corr.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    logger.info(f"Saved correlations and heatmap to {OUTPUT_DIR}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell
def _(
    CONFIGS,
    OUTPUT_DIR,
    activity_summary,
    consistency_summary,
):
    _summary = {
        "question": "Does re-tuning Harmony batch correction fix DL's batch problem?",
        "answer": "No. CP outperforms DL on activity across all 3 Harmony variants.",
        "configs": list(CONFIGS.keys()),
        "activity": activity_summary.to_dict(orient="records"),
        "consistency": consistency_summary.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as _f:
        json.dump(_summary, _f, indent=2)
    logger.info(f"Saved summary.json to {OUTPUT_DIR}")
    return


@app.function
def run_harmony_comparison(output_dir=None) -> str:
    """Run the 6-way harmony comparison, save summary.json and plots.

    Compares 6 configurations (2 feature types x 3 Harmony variants) on
    activity and consistency, plus pairwise mAP correlations.

    Called from workflow.py. Returns the output directory path.
    """
    import json

    import matplotlib.pyplot as plt

    from nb00_ss_config import DEFAULT_DPI, PROCESSED_DATA_DIR

    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "harmony-comparison"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _configs = OrderedDict(
        [
            ("CP_recipe", "compound_no_source7"),
            ("CP_rsc", "compound_no_source7_rsc"),
            ("CP_rsc_source", "compound_no_source7_rsc_source"),
            ("DL_recipe", "compound_DL_CPCNN_no_source7"),
            ("DL_rsc", "compound_DL_CPCNN_no_source7_rsc"),
            ("DL_rsc_source", "compound_DL_CPCNN_no_source7_rsc_source"),
        ]
    )

    _colors = {
        "CP_recipe": "#2171b5",
        "CP_rsc": "#6baed6",
        "CP_rsc_source": "#bdd7e7",
        "DL_recipe": "#e6550d",
        "DL_rsc": "#fdae6b",
        "DL_rsc_source": "#fdd0a2",
    }

    _consistency_target = "repurposing"

    # Activity
    _act_summary = load_activity_summary(_configs)
    if not _act_summary.empty:
        _act_summary.to_csv(output_dir / "activity_summary.csv", index=False)
        _fig_act = plot_activity_bar(_act_summary, _colors, DEFAULT_DPI)
        _fig_act.savefig(output_dir / "activity_comparison.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(_fig_act)

    # Consistency
    _cons_summary = load_consistency_summary(_configs, _consistency_target)
    if not _cons_summary.empty:
        _cons_summary.to_csv(output_dir / "consistency_summary.csv", index=False)
        _fig_cons = plot_consistency_bar(_cons_summary, _colors, _consistency_target, DEFAULT_DPI)
        _fig_cons.savefig(output_dir / "consistency_comparison.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(_fig_cons)

    # Correlations
    _corr_df = compute_activity_correlations(_configs)
    if not _corr_df.empty:
        _corr_df.to_csv(output_dir / "correlations.csv", index=False)
        _fig_corr = plot_correlation_heatmap(_corr_df, _configs, DEFAULT_DPI)
        _fig_corr.savefig(output_dir / "correlation_heatmap.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(_fig_corr)

    # Summary JSON
    _summary = {
        "question": "Does re-tuning Harmony batch correction fix DL's batch problem?",
        "answer": "No. CP outperforms DL on activity across all 3 Harmony variants.",
        "configs": list(_configs.keys()),
        "activity": _act_summary.to_dict(orient="records") if not _act_summary.empty else [],
        "consistency": _cons_summary.to_dict(orient="records") if not _cons_summary.empty else [],
    }
    with open(output_dir / "summary.json", "w") as _f:
        json.dump(_summary, _f, indent=2)

    logger.success(f"Saved harmony comparison outputs to {output_dir}")
    return str(output_dir)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
