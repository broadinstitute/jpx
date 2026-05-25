# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb==1.5.3",
#     "loguru==0.7.3",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "plotly==6.7.0",
#     "python-dotenv",
#     "scipy==1.17.1",
#     "tabulate==0.10.0",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import stats

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import COPAIRS_RESULTS_DB, DEFAULT_DPI, PROCESSED_DATA_DIR


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Activity-Consistency Relationship

    Explores whether marginally active compounds (0.05 <= p < 0.10) still contribute
    meaningful consistency signal. Addresses Anne's question from Issue #35:
    "How come those seemingly low activity compounds still have signal to be consistent?"

    **Key questions:**

    1. Is there correlation between compound activity and consistency contribution?
    2. Do marginally active compounds have high-consistency outliers?
    3. How does signal-to-noise change across activity p-value bins?
    4. At each threshold transition, how many groups are gained vs lost?
    5. Why do marginal compounds with high p-values still show high consistency?
       (Investigates whether these compounds have high effect size but low replicates.)

    *Related: [GitHub Issue #35](https://github.com/broadinstitute/jpx/issues/35)*
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
    threshold_dropdown = mo.ui.dropdown(
        options=["0.05", "0.10", "0.15", "0.20", "0.25", "0.30"],
        value="0.30",
        label="Activity threshold",
    )
    annotation_dropdown = mo.ui.dropdown(
        options=[
            "all",
            "Metadata_motive_gene_opentargets",
            "Metadata_motive_gene_biokg",
            "Metadata_motive_gene_primekg",
            "Metadata_repurposing_target",
            "Metadata_uniprot_gene",
        ],
        value="all",
        label="Annotation filter",
    )
    mo.hstack([dataset_dropdown, threshold_dropdown, annotation_dropdown])
    return (annotation_dropdown, dataset_dropdown, threshold_dropdown)


# -- Query functions --


@app.function
def get_activity_consistency_data(
    dataset: str,
    activity_threshold: float,
    annotation: str | None = None,
) -> pd.DataFrame:
    """Join activity results with consistency scores at compound level.

    Each compound appears once, with its maximum consistency score across all
    targets/annotations. This avoids duplication from compounds hitting multiple
    targets.
    """
    _annotation_filter = ""
    params = [dataset, activity_threshold, dataset]
    if annotation:
        _annotation_filter = "AND _group_column = ?"
        params.append(annotation)

    _query = f"""
    WITH activity AS (
        SELECT
            Metadata_JCP2022,
            corrected_p_value as activity_p,
            mean_normalized_average_precision as activity_nap
        FROM activity_results
        WHERE _preprocessing = 'activity_no_target2'
          AND _filter = 'all_sources'
          AND _activity_params = 'default'
          AND _dataset = ?
    ),
    consistency_per_compound AS (
        SELECT
            Metadata_JCP2022,
            MAX(normalized_average_precision) as consistency_nap
        FROM consistency_scores
        WHERE _preprocessing = 'consistency_no_target2_sweep'
          AND _filter = 'all_sources'
          AND _activity_threshold = ?
          AND _dataset = ?
          {_annotation_filter}
        GROUP BY Metadata_JCP2022
    )
    SELECT
        c.Metadata_JCP2022,
        a.activity_p,
        a.activity_nap,
        c.consistency_nap
    FROM consistency_per_compound c
    JOIN activity a ON c.Metadata_JCP2022 = a.Metadata_JCP2022
    """
    _con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    _df = _con.execute(_query, params).df()
    _con.close()
    return _df


@app.function
def get_activity_with_replicates(dataset: str) -> pd.DataFrame:
    """Get activity results with replicate counts.

    The replicate count is computed from the length of the `indices` column,
    which contains the list of row indices used in the mAP aggregation.
    """
    _query = """
    SELECT
        Metadata_JCP2022,
        corrected_p_value as activity_p,
        mean_normalized_average_precision as activity_nap,
        len(string_split(indices, ',')) as n_replicates
    FROM activity_results
    WHERE _preprocessing = 'activity_no_target2'
      AND _filter = 'all_sources'
      AND _activity_params = 'default'
      AND _dataset = ?
    """
    _con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    _df = _con.execute(_query, [dataset]).df()
    _con.close()
    return _df


@app.function
def get_effect_size_replicate_analysis(
    dataset: str,
    activity_threshold: float,
) -> pd.DataFrame:
    """Analyze effect size and replicate counts for high-consistency marginal compounds.

    Addresses Anne's question: why do marginal activity compounds still show
    high consistency? Investigates whether these compounds have high effect size
    but low replicates (insufficient power for significance).

    Statistics are computed at the COMPOUND level, not the pair level.
    A promiscuous compound hitting many targets would otherwise inflate the median
    replicate count.
    """
    _query = """
    WITH activity AS (
        SELECT
            Metadata_JCP2022,
            corrected_p_value as activity_p,
            mean_normalized_average_precision as activity_nap,
            len(string_split(indices, ',')) as n_replicates
        FROM activity_results
        WHERE _preprocessing = 'activity_no_target2'
          AND _filter = 'all_sources'
          AND _activity_params = 'default'
          AND _dataset = ?
    ),
    consistency_per_compound AS (
        SELECT
            Metadata_JCP2022,
            MAX(normalized_average_precision) as max_consistency_nap,
            COUNT(*) as n_target_pairs
        FROM consistency_scores
        WHERE _preprocessing = 'consistency_no_target2_sweep'
          AND _filter = 'all_sources'
          AND _activity_threshold = ?
          AND _dataset = ?
        GROUP BY Metadata_JCP2022
    ),
    compound_level AS (
        SELECT
            a.Metadata_JCP2022,
            a.activity_p,
            a.activity_nap,
            a.n_replicates,
            c.max_consistency_nap,
            c.n_target_pairs
        FROM activity a
        JOIN consistency_per_compound c ON a.Metadata_JCP2022 = c.Metadata_JCP2022
    )
    SELECT
        CASE
            WHEN activity_p < 0.05 THEN '1_p<0.05'
            WHEN activity_p < 0.10 THEN '2_0.05-0.10'
            WHEN activity_p < 0.15 THEN '3_0.10-0.15'
            WHEN activity_p < 0.20 THEN '4_0.15-0.20'
            WHEN activity_p < 0.25 THEN '5_0.20-0.25'
            WHEN activity_p < 0.30 THEN '6_0.25-0.30'
            ELSE '7_p>=0.30'
        END as activity_p_bin,
        CASE
            WHEN max_consistency_nap >= 0.3 THEN 'high_consistency'
            WHEN max_consistency_nap >= 0.0 THEN 'low_consistency'
            ELSE 'negative_consistency'
        END as consistency_level,
        COUNT(*) as n_compounds,
        ROUND(AVG(activity_nap), 4) as avg_activity_nap,
        ROUND(MEDIAN(activity_nap), 4) as median_activity_nap,
        ROUND(AVG(n_replicates), 2) as avg_replicates,
        ROUND(MEDIAN(n_replicates), 0) as median_replicates,
        ROUND(AVG(max_consistency_nap), 4) as avg_max_consistency_nap
    FROM compound_level
    GROUP BY activity_p_bin, consistency_level
    ORDER BY activity_p_bin, consistency_level DESC
    """
    _con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    _df = _con.execute(_query, [dataset, activity_threshold, dataset]).df()
    _con.close()
    return _df


@app.function
def get_high_consistency_marginal_details(
    dataset: str,
    activity_threshold: float,
    n_examples: int = 20,
) -> pd.DataFrame:
    """Get detailed examples of marginal activity compounds with high consistency."""
    _query = """
    WITH activity AS (
        SELECT
            Metadata_JCP2022,
            corrected_p_value as activity_p,
            mean_normalized_average_precision as activity_nap,
            len(string_split(indices, ',')) as n_replicates
        FROM activity_results
        WHERE _preprocessing = 'activity_no_target2'
          AND _filter = 'all_sources'
          AND _activity_params = 'default'
          AND _dataset = ?
    ),
    consistency AS (
        SELECT
            Metadata_JCP2022,
            normalized_average_precision as consistency_nap,
            group_value as target,
            _group_column as annotation
        FROM consistency_scores
        WHERE _preprocessing = 'consistency_no_target2_sweep'
          AND _filter = 'all_sources'
          AND _activity_threshold = ?
          AND _dataset = ?
    )
    SELECT
        a.Metadata_JCP2022,
        c.target,
        REPLACE(c.annotation, 'Metadata_', '') as annotation,
        ROUND(a.activity_p, 4) as activity_p,
        ROUND(a.activity_nap, 4) as activity_nap,
        a.n_replicates,
        ROUND(c.consistency_nap, 4) as consistency_nap
    FROM consistency c
    JOIN activity a ON c.Metadata_JCP2022 = a.Metadata_JCP2022
    WHERE a.activity_p >= 0.05 AND a.activity_p < 0.30
      AND c.consistency_nap >= 0.3
    ORDER BY c.consistency_nap DESC
    LIMIT ?
    """
    _con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    _df = _con.execute(_query, [dataset, activity_threshold, dataset, n_examples]).df()
    _con.close()
    return _df


@app.function
def get_compound_level_data(
    dataset: str,
    activity_threshold: float,
) -> pd.DataFrame:
    """Get compound-level data for detailed plotting.

    Returns one row per compound with activity metrics and max consistency.
    """
    _query = """
    WITH activity AS (
        SELECT
            Metadata_JCP2022,
            corrected_p_value as activity_p,
            mean_normalized_average_precision as activity_nap,
            len(string_split(indices, ',')) as n_replicates
        FROM activity_results
        WHERE _preprocessing = 'activity_no_target2'
          AND _filter = 'all_sources'
          AND _activity_params = 'default'
          AND _dataset = ?
    ),
    consistency_per_compound AS (
        SELECT
            Metadata_JCP2022,
            MAX(normalized_average_precision) as max_consistency_nap
        FROM consistency_scores
        WHERE _preprocessing = 'consistency_no_target2_sweep'
          AND _filter = 'all_sources'
          AND _activity_threshold = ?
          AND _dataset = ?
        GROUP BY Metadata_JCP2022
    )
    SELECT
        a.Metadata_JCP2022,
        a.activity_p,
        a.activity_nap,
        a.n_replicates,
        c.max_consistency_nap
    FROM activity a
    JOIN consistency_per_compound c ON a.Metadata_JCP2022 = c.Metadata_JCP2022
    """
    _con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    _df = _con.execute(_query, [dataset, activity_threshold, dataset]).df()
    _con.close()
    return _df


@app.function
def get_significance_by_threshold(dataset: str) -> pd.DataFrame:
    """Get group significance status at each threshold."""
    _con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    _df = _con.execute(
        """
        SELECT
            group_value,
            _group_column as annotation,
            _activity_threshold as threshold,
            below_corrected_p as significant,
            mean_normalized_average_precision as nmap,
            corrected_p_value as corrected_p,
            n_perturbations
        FROM consistency_results
        WHERE _preprocessing LIKE '%_sweep'
          AND _dataset = ?
        ORDER BY group_value, _group_column, _activity_threshold
        """,
        [dataset],
    ).df()
    _con.close()
    return _df


# -- Summary computation --


@app.function
def compute_summary_by_activity_bin(
    df: pd.DataFrame,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compute summary statistics by activity p-value bin."""
    if bins is None:
        bins = [0, 0.05, 0.10, 0.20, 0.30, 1.0]
    if labels is None:
        labels = ["p<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30", "p>0.30"]

    _df = df.copy()
    _df["activity_bin"] = pd.cut(_df["activity_p"], bins=bins, labels=labels)

    _summary = (
        _df.groupby("activity_bin", observed=True)
        .agg(
            n=("consistency_nap", "count"),
            mean=("consistency_nap", "mean"),
            median=("consistency_nap", "median"),
            std=("consistency_nap", "std"),
            n_above_03=("consistency_nap", lambda x: (x > 0.3).sum()),
        )
        .round(3)
    )
    _summary["pct_above_03"] = (100 * _summary["n_above_03"] / _summary["n"]).round(1)

    return _summary


@app.function
def compute_transitions(df: pd.DataFrame, thresholds: list[float]) -> dict:
    """Compute gains and losses at each threshold transition.

    Returns a dict with:
    - summary: list of {from, to, gained, lost, net, ratio} for each transition
    - details: list of {from, to, lost_groups: [...]} with details of lost groups
    """
    _annotations = df["annotation"].unique().tolist()

    _summary = []
    _details = []

    for i in range(len(thresholds) - 1):
        _t1, _t2 = thresholds[i], thresholds[i + 1]

        _total_gained = 0
        _total_lost = 0
        _gained_groups = []
        _lost_groups = []

        for _ann in _annotations:
            _ann_df = df[df["annotation"] == _ann]
            _df1 = _ann_df[_ann_df["threshold"] == _t1].set_index("group_value")
            _df2 = _ann_df[_ann_df["threshold"] == _t2].set_index("group_value")

            _sig1 = set(_df1[_df1["significant"]].index)
            _sig2 = set(_df2[_df2["significant"]].index)

            _gained = _sig2 - _sig1
            _lost = _sig1 - _sig2

            _total_gained += len(_gained)
            _total_lost += len(_lost)

            for _g in _gained:
                if _g in _df1.index and _g in _df2.index:
                    _gained_groups.append(
                        {
                            "annotation": _ann.replace("Metadata_", ""),
                            "group": _g,
                            "nmap_before": round(float(_df1.loc[_g, "nmap"]), 4),
                            "nmap_after": round(float(_df2.loc[_g, "nmap"]), 4),
                            "n_before": int(_df1.loc[_g, "n_perturbations"]),
                            "n_after": int(_df2.loc[_g, "n_perturbations"]),
                        }
                    )
                elif _g in _df2.index:
                    _gained_groups.append(
                        {
                            "annotation": _ann.replace("Metadata_", ""),
                            "group": _g,
                            "nmap_before": None,
                            "nmap_after": round(float(_df2.loc[_g, "nmap"]), 4),
                            "n_before": None,
                            "n_after": int(_df2.loc[_g, "n_perturbations"]),
                        }
                    )

            for _g in _lost:
                if _g in _df2.index:
                    _lost_groups.append(
                        {
                            "annotation": _ann.replace("Metadata_", ""),
                            "group": _g,
                            "nmap_before": round(float(_df1.loc[_g, "nmap"]), 4),
                            "nmap_after": round(float(_df2.loc[_g, "nmap"]), 4),
                            "n_before": int(_df1.loc[_g, "n_perturbations"]),
                            "n_after": int(_df2.loc[_g, "n_perturbations"]),
                        }
                    )

        _ratio = _total_gained / _total_lost if _total_lost > 0 else float("inf")

        _summary.append(
            {
                "from": _t1,
                "to": _t2,
                "gained": _total_gained,
                "lost": _total_lost,
                "net": _total_gained - _total_lost,
                "ratio": round(_ratio, 1) if _ratio != float("inf") else "inf",
            }
        )

        _details.append(
            {
                "from": _t1,
                "to": _t2,
                "gained_groups": _gained_groups,
                "lost_groups": _lost_groups,
            }
        )

    return {"summary": _summary, "details": _details}


# -- Plotting functions --


@app.function
def plot_activity_vs_consistency(
    df: pd.DataFrame,
    max_p: float = 0.35,
    figsize: tuple[int, int] = (16, 5),
) -> plt.Figure:
    """Create 3-panel figure showing activity vs consistency relationship."""
    _bins = [0, 0.05, 0.10, 0.20, 0.30, 1.0]
    _labels = ["p<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30", "p>0.30"]
    _colors = {
        "p<0.05": "#2ecc71",
        "0.05-0.10": "#e74c3c",
        "0.10-0.20": "#9b59b6",
        "0.20-0.30": "#3498db",
        "p>0.30": "#95a5a6",
    }

    _df = df.copy()
    _df["group"] = pd.cut(_df["activity_p"], bins=_bins, labels=_labels)

    fig, _axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Scatter of activity p-value vs consistency nAP
    _ax1 = _axes[0]
    for _grp in _labels[:-1]:
        _mask = _df["group"] == _grp
        if _mask.sum() > 0:
            _ax1.scatter(
                _df.loc[_mask, "activity_p"],
                _df.loc[_mask, "consistency_nap"],
                c=_colors[_grp],
                alpha=0.3,
                s=10,
                label=_grp,
            )

    for _p, _color in [(0.05, "#2ecc71"), (0.10, "#e74c3c"), (0.20, "#9b59b6"), (0.30, "#3498db")]:
        _ax1.axvline(x=_p, color=_color, linestyle="--", linewidth=1.5, alpha=0.7)

    _ax1.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    _ax1.axhline(y=0.3, color="orange", linestyle=":", linewidth=1.5, label="High consistency")
    _ax1.set_xlabel("Activity p-value", fontsize=12)
    _ax1.set_ylabel("Max Consistency nAP", fontsize=12)
    _ax1.set_title(
        "Activity p-value vs Consistency\n(each dot = one compound, max consistency across targets)", fontsize=12
    )
    _ax1.legend(loc="upper right", fontsize=9)
    _ax1.set_xlim(-0.01, max_p)

    # Plot 2: Box plots comparing all groups
    _ax2 = _axes[1]
    _groups_to_plot = [g for g in _labels[:-1] if (_df["group"] == g).sum() > 0]
    _groups_data = [_df[_df["group"] == g]["consistency_nap"] for g in _groups_to_plot]
    _box_colors = [_colors[g] for g in _groups_to_plot]

    if _groups_data:
        _bp = _ax2.boxplot(
            _groups_data,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=2, alpha=0.3),
        )
        for _patch, _color in zip(_bp["boxes"], _box_colors):
            _patch.set_facecolor(_color)

    _ax2.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    _ax2.axhline(y=0.3, color="orange", linestyle=":", linewidth=1.5)
    _ax2.set_ylabel("Max Consistency nAP", fontsize=12)
    _ax2.set_title("Distribution by activity group", fontsize=12)

    _tick_labels_with_n = [f"{g}\n(n={len(_df[_df['group'] == g])})" for g in _groups_to_plot]
    _ax2.set_xticks(range(1, len(_groups_to_plot) + 1))
    _ax2.set_xticklabels(_tick_labels_with_n, fontsize=9)

    # Plot 3: Histogram overlay
    _ax3 = _axes[2]
    _hist_bins = np.linspace(-0.1, 0.8, 25)
    for _grp in _groups_to_plot:
        _data = _df[_df["group"] == _grp]["consistency_nap"]
        _ax3.hist(
            _data,
            bins=_hist_bins,
            alpha=0.5,
            color=_colors[_grp],
            label=f"{_grp} (n={len(_data)})",
            density=True,
        )

    _ax3.axvline(x=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    _ax3.axvline(x=0.3, color="orange", linestyle=":", linewidth=2)
    _ax3.set_xlabel("Max Consistency nAP", fontsize=12)
    _ax3.set_ylabel("Density", fontsize=12)
    _ax3.set_title("Overlaid distributions", fontsize=12)
    _ax3.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig


@app.function
def plot_transitions(transitions: dict, figsize: tuple[int, int] = (12, 5)) -> plt.Figure:
    """Create plot showing gains vs losses at each threshold transition."""
    _summary = transitions["summary"]

    fig, _axes = plt.subplots(1, 2, figsize=figsize)

    _labels = [f"{s['from']:.2f}->{s['to']:.2f}" for s in _summary]
    _gained = [s["gained"] for s in _summary]
    _lost = [s["lost"] for s in _summary]
    _net = [s["net"] for s in _summary]

    _x = np.arange(len(_labels))

    # Plot 1: Stacked bar showing gained vs lost
    _ax1 = _axes[0]
    _ax1.bar(_x, _gained, 0.6, label="Gained", color="#2ecc71", alpha=0.8)
    _ax1.bar(_x, [-val for val in _lost], 0.6, label="Lost", color="#e74c3c", alpha=0.8)
    _ax1.axhline(y=0, color="black", linewidth=0.5)
    _ax1.set_xlabel("Threshold transition", fontsize=11)
    _ax1.set_ylabel("Number of groups", fontsize=11)
    _ax1.set_title("Groups gained vs lost at each transition", fontsize=12)
    _ax1.set_xticks(_x)
    _ax1.set_xticklabels(_labels, fontsize=9)
    _ax1.legend(loc="upper right")

    for _i, _s in enumerate(_summary):
        _ratio_str = f"{_s['ratio']}:1" if _s["ratio"] != "inf" else "inf:1"
        _ax1.annotate(_ratio_str, xy=(_i, _gained[_i] + 2), ha="center", fontsize=9, color="#666666")

    # Plot 2: Cumulative net change
    _ax2 = _axes[1]
    _cumulative = np.cumsum(_net)
    _thresholds_end = [s["to"] for s in _summary]

    _ax2.plot(_thresholds_end, _cumulative, "o-", color="#3498db", linewidth=2, markersize=8)
    _ax2.fill_between(_thresholds_end, 0, _cumulative, alpha=0.3, color="#3498db")
    _ax2.axhline(y=0, color="black", linewidth=0.5)

    if len(_net) > 0:
        _ax2.annotate(
            f"+{_net[0]} at 0.05->0.10",
            xy=(0.10, _cumulative[0]),
            xytext=(0.15, _cumulative[0] - 10),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#666666"),
        )

    _ax2.set_xlabel("Activity threshold", fontsize=11)
    _ax2.set_ylabel("Cumulative net change in significant groups", fontsize=11)
    _ax2.set_title("Cumulative effect of relaxing threshold", fontsize=12)
    _ax2.set_xticks(_thresholds_end)

    plt.tight_layout()
    return fig


@app.function
def plot_effect_size_vs_replicates(
    summary_df: pd.DataFrame,
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Create plot showing effect size and replicate counts across activity bins."""
    fig, _axes = plt.subplots(1, 2, figsize=figsize)

    _high_cons = summary_df[summary_df["consistency_level"] == "high_consistency"].copy()
    _high_cons["activity_p_bin_clean"] = _high_cons["activity_p_bin"].str.replace(r"^\d_", "", regex=True)

    _colors = ["#2ecc71", "#e74c3c", "#9b59b6", "#8e44ad", "#3498db", "#2980b9"]
    _x = range(len(_high_cons))

    # Plot 1: Median activity nMAP by p-value bin (high-consistency only)
    _ax1 = _axes[0]
    _bars = _ax1.bar(_x, _high_cons["median_activity_nap"], color=_colors[: len(_high_cons)], alpha=0.8)

    _ax1.set_xlabel("Activity p-value bin", fontsize=11)
    _ax1.set_ylabel("Median Activity Effect Size (nMAP)", fontsize=11)
    _ax1.set_title(
        "Activity Effect Size for High-Consistency Compounds\n(per compound, max consistency nAP >= 0.3)", fontsize=12
    )
    _ax1.set_xticks(list(_x))
    _ax1.set_xticklabels(_high_cons["activity_p_bin_clean"], fontsize=9, rotation=15)
    _ax1.axhline(y=0.3, color="orange", linestyle=":", linewidth=1.5, label="Strong effect (0.3)")
    _ax1.legend(loc="upper right", fontsize=9)

    for _i, (_bar, _n) in enumerate(zip(_bars, _high_cons["n_compounds"])):
        _ax1.annotate(
            f"n={_n}",
            xy=(_bar.get_x() + _bar.get_width() / 2, _bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    # Plot 2: Median replicates by p-value bin (high-consistency only)
    _ax2 = _axes[1]
    _bars2 = _ax2.bar(_x, _high_cons["median_replicates"], color=_colors[: len(_high_cons)], alpha=0.8)

    _ax2.set_xlabel("Activity p-value bin", fontsize=11)
    _ax2.set_ylabel("Median Replicate Count", fontsize=11)
    _ax2.set_title(
        "Replicate Count for High-Consistency Compounds\n(per compound, max consistency nAP >= 0.3)", fontsize=12
    )
    _ax2.set_xticks(list(_x))
    _ax2.set_xticklabels(_high_cons["activity_p_bin_clean"], fontsize=9, rotation=15)
    _ax2.axhline(y=5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Typical (5)")
    _ax2.legend(loc="upper right", fontsize=9)

    for _i, (_bar, _n) in enumerate(zip(_bars2, _high_cons["n_compounds"])):
        _ax2.annotate(
            f"n={_n}",
            xy=(_bar.get_x() + _bar.get_width() / 2, _bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    return fig


@app.function
def plot_replicate_distribution_by_consistency(
    compound_df: pd.DataFrame,
    figsize: tuple[int, int] = (16, 5),
) -> plt.Figure:
    """Show relationship between activity p-value, replicates, and consistency.

    Directly visualizes whether high-consistency compounds with high p-values
    (marginal activity) are the ones with low replicate counts.
    """
    fig, _axes = plt.subplots(1, 2, figsize=figsize)

    _df = compound_df.copy()

    _df["consistency_group"] = pd.cut(
        _df["max_consistency_nap"],
        bins=[-np.inf, 0.0, 0.3, np.inf],
        labels=["Negative (<0)", "Low (0-0.3)", "High (>=0.3)"],
    )

    _colors = {"High (>=0.3)": "#2ecc71", "Low (0-0.3)": "#3498db", "Negative (<0)": "#95a5a6"}

    _df_filtered = _df[_df["activity_p"] < 0.30].copy()

    # Panel 1: Scatter of activity_p vs n_replicates, colored by consistency
    _ax1 = _axes[0]

    np.random.seed(42)
    _df_filtered = _df_filtered.copy()
    _df_filtered["_jitter"] = np.random.uniform(-0.35, 0.35, len(_df_filtered))

    for _cons_group in ["Negative (<0)", "Low (0-0.3)", "High (>=0.3)"]:
        _subset = _df_filtered[_df_filtered["consistency_group"] == _cons_group]
        if len(_subset) > 0:
            _ax1.scatter(
                _subset["activity_p"],
                _subset["n_replicates"] + _subset["_jitter"],
                alpha=0.6,
                s=30,
                color=_colors[_cons_group],
                edgecolor="white",
                linewidth=0.3,
                label=f"{_cons_group} (n={len(_subset)})",
            )

    _ax1.axvline(x=0.05, color="black", linestyle="--", linewidth=1, alpha=0.5, label="p=0.05")
    _ax1.axvline(x=0.10, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="p=0.10")

    _ax1.set_xlabel("Activity p-value", fontsize=11)
    _ax1.set_ylabel("Number of Replicates", fontsize=11)
    _ax1.set_title("Activity p-value vs Replicates\n(colored by consistency level)", fontsize=12)
    _ax1.set_xlim(-0.01, 0.31)
    _ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: Focus on high-consistency only, show activity_p vs replicates
    _ax2 = _axes[1]
    _high_cons = _df_filtered[_df_filtered["consistency_group"] == "High (>=0.3)"].copy()

    if len(_high_cons) > 0:
        _high_cons["p_bin"] = pd.cut(
            _high_cons["activity_p"],
            bins=[0, 0.05, 0.10, 0.20, 0.30],
            labels=["p<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30"],
        )
        _p_colors = {
            "p<0.05": "#2ecc71",
            "0.05-0.10": "#e74c3c",
            "0.10-0.20": "#9b59b6",
            "0.20-0.30": "#3498db",
        }

        for _p_bin in ["p<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30"]:
            _subset = _high_cons[_high_cons["p_bin"] == _p_bin]
            if len(_subset) > 0:
                _ax2.scatter(
                    _subset["activity_nap"],
                    _subset["n_replicates"] + _subset["_jitter"],
                    alpha=0.7,
                    s=50,
                    color=_p_colors[_p_bin],
                    edgecolor="white",
                    linewidth=0.5,
                    label=f"{_p_bin} (n={len(_subset)}, med_rep={_subset['n_replicates'].median():.0f})",
                )

        _ax2.set_xlabel("Activity Effect Size (nMAP)", fontsize=11)
        _ax2.set_ylabel("Number of Replicates", fontsize=11)
        _ax2.set_title(
            "High-Consistency Compounds Only\n(activity effect size vs replicates, colored by p-value)",
            fontsize=12,
        )
        _ax2.legend(loc="upper right", fontsize=8)
        _ax2.axhline(y=5, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    _y_max = 16
    for _ax in _axes:
        _ax.set_ylim(0, _y_max)
        _ax.set_yticks(range(0, _y_max + 1, 2))

    _n_above = (_df_filtered["n_replicates"] > _y_max).sum()
    if _n_above > 0:
        _axes[0].text(
            0.98,
            0.02,
            f"({_n_above} compounds with >{_y_max} replicates not shown)",
            transform=_axes[0].transAxes,
            fontsize=7,
            ha="right",
            va="bottom",
            style="italic",
            color="gray",
        )

    plt.tight_layout()
    return fig


@app.function
def plot_replicate_distribution_interactive(
    compound_df: pd.DataFrame,
) -> go.Figure:
    """Create interactive version of replicate distribution plot with hover tooltips."""
    _df = compound_df.copy()

    _df["consistency_group"] = pd.cut(
        _df["max_consistency_nap"],
        bins=[-np.inf, 0.0, 0.3, np.inf],
        labels=["Negative (<0)", "Low (0-0.3)", "High (>=0.3)"],
    )

    _df["p_bin"] = pd.cut(
        _df["activity_p"],
        bins=[0, 0.05, 0.10, 0.20, 0.30, 1.0],
        labels=["p<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30", "p>=0.30"],
    )

    _df_filtered = _df[_df["activity_p"] < 0.30].copy()

    np.random.seed(42)
    _df_filtered["n_replicates_jittered"] = _df_filtered["n_replicates"] + np.random.uniform(
        -0.35, 0.35, len(_df_filtered)
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Activity p-value vs Replicates<br>(colored by consistency level)",
            "High-Consistency Compounds Only<br>(colored by p-value bin)",
        ),
        horizontal_spacing=0.08,
    )

    _cons_colors = {"High (>=0.3)": "#2ecc71", "Low (0-0.3)": "#3498db", "Negative (<0)": "#95a5a6"}
    _p_colors = {
        "p<0.05": "#2ecc71",
        "0.05-0.10": "#e74c3c",
        "0.10-0.20": "#9b59b6",
        "0.20-0.30": "#3498db",
    }

    # Panel 1: All compounds colored by consistency
    for _cons_group in ["Negative (<0)", "Low (0-0.3)", "High (>=0.3)"]:
        _subset = _df_filtered[_df_filtered["consistency_group"] == _cons_group]
        if len(_subset) > 0:
            fig.add_trace(
                go.Scatter(
                    x=_subset["activity_p"],
                    y=_subset["n_replicates_jittered"],
                    mode="markers",
                    name=f"{_cons_group} (n={len(_subset)})",
                    marker=dict(color=_cons_colors[_cons_group], size=8, opacity=0.7),
                    customdata=np.stack(
                        [
                            _subset["Metadata_JCP2022"],
                            _subset["n_replicates"],
                            _subset["activity_nap"],
                            _subset["max_consistency_nap"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Activity p: %{x:.4f}<br>"
                        "Replicates: %{customdata[1]}<br>"
                        "Activity nMAP: %{customdata[2]:.3f}<br>"
                        "Max Consistency nAP: %{customdata[3]:.3f}"
                        "<extra></extra>"
                    ),
                    legendgroup="panel1",
                ),
                row=1,
                col=1,
            )

    fig.add_vline(x=0.05, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)
    fig.add_vline(x=0.10, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)

    # Panel 2: High-consistency only, colored by p-value bin
    _high_cons = _df_filtered[_df_filtered["consistency_group"] == "High (>=0.3)"].copy()

    for _p_bin in ["p<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30"]:
        _subset = _high_cons[_high_cons["p_bin"] == _p_bin]
        if len(_subset) > 0:
            _med_rep = _subset["n_replicates"].median()
            fig.add_trace(
                go.Scatter(
                    x=_subset["activity_nap"],
                    y=_subset["n_replicates_jittered"],
                    mode="markers",
                    name=f"{_p_bin} (n={len(_subset)}, med_rep={_med_rep:.0f})",
                    marker=dict(color=_p_colors[_p_bin], size=10, opacity=0.7),
                    customdata=np.stack(
                        [
                            _subset["Metadata_JCP2022"],
                            _subset["n_replicates"],
                            _subset["activity_p"],
                            _subset["max_consistency_nap"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Activity nMAP: %{x:.3f}<br>"
                        "Replicates: %{customdata[1]}<br>"
                        "Activity p: %{customdata[2]:.4f}<br>"
                        "Max Consistency nAP: %{customdata[3]:.3f}"
                        "<extra></extra>"
                    ),
                    legendgroup="panel2",
                ),
                row=1,
                col=2,
            )

    fig.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)

    fig.update_xaxes(title_text="Activity p-value", range=[-0.01, 0.31], row=1, col=1)
    fig.update_xaxes(title_text="Activity Effect Size (nMAP)", row=1, col=2)
    fig.update_yaxes(title_text="Number of Replicates", range=[0, 16], dtick=2, row=1, col=1)
    fig.update_yaxes(title_text="Number of Replicates", range=[0, 16], dtick=2, row=1, col=2)

    fig.update_layout(
        height=500,
        width=1200,
        title_text="Replicate Distribution by Consistency Level (Interactive)",
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.02),
    )

    return fig


# -- Load data --


@app.cell
def _(annotation_dropdown, dataset_dropdown, mo, threshold_dropdown):
    mo.stop(
        not COPAIRS_RESULTS_DB.exists(),
        mo.md(f"**Error:** Copairs database not found at `{COPAIRS_RESULTS_DB}`. Run `just run` first."),
    )

    _dataset = dataset_dropdown.value
    _threshold = float(threshold_dropdown.value)
    _annotation = annotation_dropdown.value if annotation_dropdown.value != "all" else None

    activity_consistency_df = get_activity_consistency_data(_dataset, _threshold, _annotation)

    mo.stop(
        len(activity_consistency_df) == 0,
        mo.md("**Error:** No data returned. Check dataset/threshold/annotation combination."),
    )

    _n_compounds = len(activity_consistency_df)
    _max_p = _threshold + 0.05
    _df_vis = activity_consistency_df[activity_consistency_df["activity_p"] < _max_p].copy()

    mo.md(f"""
    ## Data loaded

    **Dataset:** {_dataset} | **Activity threshold:** {_threshold} | **Annotation:** {_annotation or "all"}

    - Loaded {_n_compounds:,} compounds (using max consistency per compound)
    - Showing {len(_df_vis):,} compounds with activity p < {_max_p}
    """)
    return (activity_consistency_df,)


# -- Summary statistics --


@app.cell
def _(activity_consistency_df, mo, threshold_dropdown):
    _threshold = float(threshold_dropdown.value)
    _max_p = _threshold + 0.05
    _df_vis = activity_consistency_df[activity_consistency_df["activity_p"] < _max_p].copy()

    summary_df = compute_summary_by_activity_bin(_df_vis)

    _rho, _pval = stats.spearmanr(_df_vis["activity_p"], _df_vis["consistency_nap"])

    mo.md(f"""
    ## Summary by activity bin

    **Spearman correlation** (activity p vs consistency nAP): rho={_rho:.3f}, p={_pval:.2e}

    {summary_df.to_markdown()}
    """)
    return (summary_df,)


# -- Activity vs Consistency plot --


@app.cell
def _(activity_consistency_df, threshold_dropdown):
    _threshold = float(threshold_dropdown.value)
    _max_p = _threshold + 0.05
    _df_vis = activity_consistency_df[activity_consistency_df["activity_p"] < _max_p].copy()

    fig_activity_vs_consistency = plot_activity_vs_consistency(_df_vis, max_p=_max_p)
    fig_activity_vs_consistency
    return (fig_activity_vs_consistency,)


# -- Threshold Transition Analysis --


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Threshold Transition Analysis

    At each threshold step, how many target groups become significant (gained) vs lose
    significance (lost)? A high gain-to-loss ratio indicates the marginal compounds
    entering the analysis carry real signal rather than adding noise.
    """)
    return


@app.cell
def _(dataset_dropdown, mo):
    _dataset = dataset_dropdown.value
    _sig_df = get_significance_by_threshold(_dataset)

    mo.stop(
        len(_sig_df) == 0,
        mo.md("**Error:** No sweep data found. Run the threshold sweep pipeline first."),
    )

    _thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    transitions = compute_transitions(_sig_df, _thresholds)

    _rows = []
    for _s in transitions["summary"]:
        _ratio_str = f"{_s['ratio']}:1" if _s["ratio"] != "inf" else "inf:1"
        _rows.append(
            f"| {_s['from']:.2f} -> {_s['to']:.2f} | {_s['gained']} | {_s['lost']} | {_s['net']:+d} | {_ratio_str} |"
        )

    _table = "\n".join(_rows)

    mo.md(f"""
    ### Transition summary

    | Transition | Gained | Lost | Net | Ratio |
    |-----------|--------|------|-----|-------|
    {_table}
    """)
    return (transitions,)


@app.cell
def _(transitions):
    fig_transitions = plot_transitions(transitions)
    fig_transitions
    return (fig_transitions,)


# -- Effect Size vs Replicates Analysis --


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effect Size vs Replicates

    **Hypothesis:** Marginal activity compounds with high consistency have HIGH effect
    size (nMAP) but LOW replicate counts - insufficient statistical power for
    individual significance, yet they contribute real signal to target groups.
    """)
    return


@app.cell
def _(dataset_dropdown, mo, threshold_dropdown):
    _dataset = dataset_dropdown.value
    _threshold = float(threshold_dropdown.value)

    effect_rep_df = get_effect_size_replicate_analysis(_dataset, _threshold)

    _high_cons_marginal = effect_rep_df[
        (effect_rep_df["consistency_level"] == "high_consistency") & (effect_rep_df["activity_p_bin"] != "7_p>=0.30")
    ].copy()

    _detail_rows = []
    if len(_high_cons_marginal) > 0:
        for _, _row in _high_cons_marginal.iterrows():
            _bin_label = _row["activity_p_bin"].replace("_", " ").lstrip("0123456789 ")
            _detail_rows.append(
                f"| {_bin_label} | {_row['n_compounds']} | {_row['median_activity_nap']:.3f} | {_row['median_replicates']:.0f} |"
            )

    _detail_table = "\n".join(_detail_rows) if _detail_rows else "| (no data) | - | - | - |"

    mo.md(f"""
    ### High-consistency compounds by activity bin

    | Activity p-bin | N compounds | Effect Size (median nAP) | Replicates (median) |
    |---------------|-------------|-------------------------|---------------------|
    {_detail_table}
    """)
    return (effect_rep_df,)


@app.cell
def _(effect_rep_df):
    fig_effect_size = plot_effect_size_vs_replicates(effect_rep_df)
    fig_effect_size
    return (fig_effect_size,)


# -- Replicate distribution --


@app.cell
def _(dataset_dropdown, threshold_dropdown):
    _dataset = dataset_dropdown.value
    _threshold = float(threshold_dropdown.value)

    compound_data = get_compound_level_data(_dataset, _threshold)
    fig_replicate_dist = plot_replicate_distribution_by_consistency(compound_data)
    fig_replicate_dist
    return (compound_data, fig_replicate_dist)


# -- Interactive plot --


@app.cell
def _(compound_data):
    fig_interactive = plot_replicate_distribution_interactive(compound_data)
    fig_interactive
    return (fig_interactive,)


# -- Top examples --


@app.cell
def _(dataset_dropdown, mo, threshold_dropdown):
    _dataset = dataset_dropdown.value
    _threshold = float(threshold_dropdown.value)

    _examples = get_high_consistency_marginal_details(_dataset, _threshold, n_examples=20)

    mo.stop(
        len(_examples) == 0,
        mo.md("No marginal compounds with high consistency found at this threshold."),
    )

    mo.md(f"""
    ## Top examples: Marginal activity + High consistency

    Compounds with activity p in [0.05, 0.30) and consistency nAP >= 0.3.
    These are the puzzling cases - individually non-significant but
    contributing to target-group consistency.

    {_examples.to_markdown(index=False)}
    """)
    return


# -- Observations --


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observations

    1. Strong activity (p<0.05) compounds show the highest consistency signal.
    2. Marginal compounds (0.05-0.30) are noisier but some show real consistency signal.
    3. Threshold transition analysis shows gain/loss ratios at each step -
       the first relaxation (0.05->0.10) gives by far the best gain ratio.
    4. **Marginal compounds with high consistency** tend to have high activity effect
       size (nAP) but low replicate counts - insufficient power for individual
       significance, yet contributing real morphological signal to target groups.
    """)
    return


# -- Save outputs --


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save all outputs")
    save_button
    return (save_button,)


@app.cell
def _(
    compound_data,
    dataset_dropdown,
    effect_rep_df,
    fig_activity_vs_consistency,
    fig_effect_size,
    fig_interactive,
    fig_replicate_dist,
    fig_transitions,
    mo,
    save_button,
    summary_df,
    threshold_dropdown,
    transitions,
):
    mo.stop(not save_button.value)

    _dataset = dataset_dropdown.value
    _threshold = float(threshold_dropdown.value)

    _outdir = PROCESSED_DATA_DIR / "exploration" / "nb22" / _dataset
    _outdir.mkdir(parents=True, exist_ok=True)

    fig_activity_vs_consistency.savefig(_outdir / "activity_vs_consistency.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    summary_df.to_csv(_outdir / "activity_vs_consistency_summary.csv")

    fig_transitions.savefig(_outdir / "threshold_transitions.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    with open(_outdir / "threshold_transitions.json", "w") as _f:
        json.dump(transitions, _f, indent=2)

    fig_effect_size.savefig(_outdir / "effect_size_vs_replicates.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    effect_rep_df.to_csv(_outdir / "effect_size_vs_replicates.csv", index=False)

    fig_replicate_dist.savefig(
        _outdir / "replicate_distribution_by_consistency.png", dpi=DEFAULT_DPI, bbox_inches="tight"
    )

    fig_interactive.write_html(str(_outdir / "replicate_distribution_interactive.html"))

    mo.md(f"""
    **Saved all outputs to:** `{_outdir}`

    - `activity_vs_consistency.png` / `.csv`
    - `threshold_transitions.png` / `.json`
    - `effect_size_vs_replicates.png` / `.csv`
    - `replicate_distribution_by_consistency.png`
    - `replicate_distribution_interactive.html`
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
