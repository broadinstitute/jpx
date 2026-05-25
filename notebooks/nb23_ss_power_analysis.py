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
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Patch

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import COPAIRS_RESULTS_DB, DEFAULT_DPI, PROCESSED_DATA_DIR


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Statistical Power Analysis

    Addresses Anne's question from Issue #35: "Why do marginal activity compounds
    (p >= 0.10) still contribute consistency signal?"

    **Key insight:** Activity significance depends on three factors:

    1. Effect size (activity nAP)
    2. Number of replicates
    3. Number of negative controls per replicate

    This analysis investigates whether high-consistency marginal compounds have high
    effect sizes but insufficient statistical power (due to low replicates), or whether
    they represent genuinely moderate signal that contributes to group-level consistency.

    **Outputs:**

    - Power analysis scatter (replicates vs effect size, colored by significance/consistency)
    - Target consistency scatter (power vs effect size at target level)
    - Faceted by-target panels
    - Summary statistics CSV

    *Related: [GitHub Issue #35](https://github.com/broadinstitute/jpx/issues/35)*
    """)
    return


# -- Controls --


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
    target_system_dropdown = mo.ui.dropdown(
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
        label="Target system",
    )
    activity_threshold_slider = mo.ui.slider(start=0.05, stop=0.20, step=0.01, value=0.10, label="Activity p threshold")
    consistency_threshold_slider = mo.ui.slider(
        start=0.10, stop=0.50, step=0.05, value=0.30, label="Consistency threshold"
    )
    mo.hstack([dataset_dropdown, target_system_dropdown, activity_threshold_slider, consistency_threshold_slider])
    return (activity_threshold_slider, consistency_threshold_slider, dataset_dropdown, target_system_dropdown)


# -- Query functions --


@app.function
def compute_consistency_threshold(
    dataset: str,
    target_system: str = "repurposing",
    consistency_threshold: float = 0.30,
    target_significance: float = 0.05,
    percentile: float = 50,
) -> float:
    """Compute empirical consistency threshold from significant targets.

    The threshold is derived from the distribution of target-level nAP for
    targets that passed the consistency significance test. At median, this
    tells us "what does a consistent target's nAP look like?"

    This empirical threshold differs from the fixed exploratory threshold of 0.3
    used in notebook 0.06. The fixed 0.3 is stricter, useful for identifying only
    the strongest consistency signals, while this data-driven threshold captures
    the typical consistency level of significant targets.
    """
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    query = """
    SELECT mean_normalized_average_precision as target_nap
    FROM consistency_results
    WHERE _preprocessing = 'consistency_no_target2_sweep'
      AND _filter = 'all_sources'
      AND _activity_threshold = ?
      AND _dataset = ?
      AND _group_type = ?
      AND corrected_p_value < ?
    """
    _df = con.execute(query, [consistency_threshold, dataset, target_system, target_significance]).df()
    con.close()

    if len(_df) == 0:
        return 0.12

    return float(np.percentile(_df["target_nap"], percentile))


@app.function
def get_compound_power_data(
    dataset: str,
    consistency_threshold: float = 0.30,
    max_median_neg_pairs: int = 40,
) -> pd.DataFrame:
    """Get compound-level data with activity, consistency, and power metrics.

    Returns compound-level data with columns:
    - Metadata_JCP2022, activity_p, activity_nap, n_replicates, median_neg_pairs
    - consistency_median, consistency_std, consistency_max, n_targets
    """
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    query = """
    WITH scores_agg AS (
        SELECT
            Metadata_JCP2022,
            COUNT(*) as n_replicates,
            MEDIAN(n_total_pairs - n_pos_pairs) as median_neg_pairs
        FROM activity_scores
        WHERE _preprocessing = 'activity_no_target2'
          AND _filter = 'all_sources'
          AND _activity_params = 'default'
          AND _dataset = ?
        GROUP BY Metadata_JCP2022
    ),
    consistency_agg AS (
        SELECT
            Metadata_JCP2022,
            MEDIAN(normalized_average_precision) as consistency_median,
            STDDEV(normalized_average_precision) as consistency_std,
            MAX(normalized_average_precision) as consistency_max,
            COUNT(*) as n_targets
        FROM consistency_scores
        WHERE _preprocessing = 'consistency_no_target2_sweep'
          AND _filter = 'all_sources'
          AND _activity_threshold = ?
          AND _dataset = ?
        GROUP BY Metadata_JCP2022
    )
    SELECT
        r.Metadata_JCP2022,
        r.corrected_p_value as activity_p,
        r.mean_normalized_average_precision as activity_nap,
        -log10(r.corrected_p_value) as neg_log10_p,
        s.n_replicates,
        s.median_neg_pairs,
        c.consistency_median,
        c.consistency_std,
        c.consistency_max,
        c.n_targets
    FROM activity_results r
    JOIN scores_agg s ON r.Metadata_JCP2022 = s.Metadata_JCP2022
    JOIN consistency_agg c ON r.Metadata_JCP2022 = c.Metadata_JCP2022
    WHERE r._preprocessing = 'activity_no_target2'
      AND r._filter = 'all_sources'
      AND r._activity_params = 'default'
      AND r._dataset = ?
      AND s.median_neg_pairs <= ?
    """
    _df = con.execute(query, [dataset, consistency_threshold, dataset, dataset, max_median_neg_pairs]).df()
    con.close()
    return _df


@app.function
def get_target_consistency_stats(
    dataset: str,
    target_system: str = "repurposing",
    consistency_threshold: float = 0.30,
    target_significance: float = 0.05,
) -> pd.DataFrame:
    """Get target-level consistency statistics (p-value, nMAP, significance).

    Returns target-level stats: target_name, target_p, target_nap, target_significant, n_perturbations
    """
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    query = """
    SELECT
        group_value as target_name,
        corrected_p_value as target_p,
        mean_normalized_average_precision as target_nap,
        corrected_p_value < ? as target_significant,
        n_perturbations
    FROM consistency_results
    WHERE _preprocessing = 'consistency_no_target2_sweep'
      AND _filter = 'all_sources'
      AND _activity_threshold = ?
      AND _dataset = ?
      AND _group_type = ?
    """
    _df = con.execute(query, [target_significance, consistency_threshold, dataset, target_system]).df()
    con.close()
    return _df


@app.function
def get_target_level_power_data(
    dataset: str,
    target_system: str = "repurposing",
    consistency_threshold: float = 0.30,
    max_median_neg_pairs: int = 40,
) -> pd.DataFrame:
    """Get target-level data with one row per compound-target pair.

    Unlike get_compound_power_data which aggregates consistency across targets,
    this function preserves the compound-target pair granularity to enable
    per-target analysis.

    Returns target-level data with columns:
    - Metadata_JCP2022, target_name, activity_p, activity_nap, n_replicates, consistency_nap
    """
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    query = """
    WITH scores_agg AS (
        SELECT
            Metadata_JCP2022,
            COUNT(*) as n_replicates,
            MEDIAN(n_total_pairs - n_pos_pairs) as median_neg_pairs
        FROM activity_scores
        WHERE _preprocessing = 'activity_no_target2'
          AND _filter = 'all_sources'
          AND _activity_params = 'default'
          AND _dataset = ?
        GROUP BY Metadata_JCP2022
    )
    SELECT
        c.Metadata_JCP2022,
        c.group_value as target_name,
        r.corrected_p_value as activity_p,
        r.mean_normalized_average_precision as activity_nap,
        s.n_replicates,
        c.normalized_average_precision as consistency_nap
    FROM consistency_scores c
    JOIN activity_results r ON c.Metadata_JCP2022 = r.Metadata_JCP2022
    JOIN scores_agg s ON c.Metadata_JCP2022 = s.Metadata_JCP2022
    WHERE c._preprocessing = 'consistency_no_target2_sweep'
      AND c._filter = 'all_sources'
      AND c._activity_threshold = ?
      AND c._group_type = ?
      AND c._dataset = ?
      AND r._preprocessing = 'activity_no_target2'
      AND r._filter = 'all_sources'
      AND r._activity_params = 'default'
      AND r._dataset = ?
      AND s.median_neg_pairs <= ?
    """
    _df = con.execute(
        query, [dataset, consistency_threshold, target_system, dataset, dataset, max_median_neg_pairs]
    ).df()
    con.close()
    return _df


# -- Plot functions --


@app.function
def plot_power_analysis(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    activity_threshold: float = 0.10,
    replicate_cap: int | None = 6,
    replicate_floor: int | None = 3,
    figsize: tuple[int, int] = (12, 5),
    show_legend: bool = True,
    consistency_col: str = "consistency_max",
    consistency_high_threshold: float = 0.3,
    title: str | None = None,
    show_counts_in_legend: bool = True,
) -> plt.Figure | plt.Axes:
    """Single focused scatter: replicates vs effect size.

    Creates a visualization showing compounds in four categories:
    - Gray: not significant, low consistency (background)
    - Green: significant, low consistency (background)
    - Blue: significant AND high consistency (foreground)
    - Red: not significant, high consistency (foreground)

    Blue and red are shown with larger markers for direct comparison.
    """
    _df = df.copy()
    _df["significant"] = _df["activity_p"] < activity_threshold
    _df["high_consistency"] = _df[consistency_col] >= consistency_high_threshold

    # Four mutually exclusive categories
    _df["sig_high_cons"] = _df["significant"] & _df["high_consistency"]
    _df["sig_low_cons"] = _df["significant"] & (~_df["high_consistency"])
    _df["nonsig_high_cons"] = (~_df["significant"]) & _df["high_consistency"]
    _df["nonsig_low_cons"] = (~_df["significant"]) & (~_df["high_consistency"])

    # Bin replicates: floor (low values) and cap (high values)
    _df["n_replicates_plot"] = _df["n_replicates"].copy()
    if replicate_floor is not None:
        _df["n_replicates_plot"] = _df["n_replicates_plot"].clip(lower=replicate_floor)
    if replicate_cap is not None:
        _df["n_replicates_plot"] = _df["n_replicates_plot"].clip(upper=replicate_cap)

    # Create figure if no axes provided
    _created_figure = ax is None
    if _created_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Separate groups
    _sig_high = _df[_df["sig_high_cons"]]
    _sig_low = _df[_df["sig_low_cons"]]
    _nonsig_high = _df[_df["nonsig_high_cons"]]
    _nonsig_low = _df[_df["nonsig_low_cons"]]

    # Jitter function for discrete x-values
    _rng = np.random.default_rng(42)

    def _jitter(x, scale=0.25):
        return x + _rng.uniform(-scale, scale, len(x))

    # Build labels with or without counts
    if show_counts_in_legend:
        _label_sig_low = f"Significant, low consistency (n={len(_sig_low)})"
        _label_nonsig_low = f"Not significant, low consistency (n={len(_nonsig_low)})"
        _label_sig_high = f"Significant, high consistency (n={len(_sig_high)})"
        _label_nonsig_high = f"Not significant, high consistency (n={len(_nonsig_high)})"
    else:
        _label_sig_low = "Significant, low consistency"
        _label_nonsig_low = "Not significant, low consistency"
        _label_sig_high = f"Significant, high consistency (nAP >= {consistency_high_threshold:.2f})"
        _label_nonsig_high = f"Not significant, high consistency (nAP >= {consistency_high_threshold:.2f})"

    # All dots same size for cleaner visualization
    _dot_size = 30

    # Background: significant, low-consistency (green)
    ax.scatter(
        _jitter(_sig_low["n_replicates_plot"]),
        _sig_low["activity_nap"],
        c="#2e8b57",
        s=_dot_size,
        alpha=0.6,
        edgecolor="none",
        label=_label_sig_low,
    )

    # Background: non-significant, low-consistency (gray)
    ax.scatter(
        _jitter(_nonsig_low["n_replicates_plot"]),
        _nonsig_low["activity_nap"],
        c="#888888",
        s=_dot_size,
        alpha=0.6,
        edgecolor="none",
        label=_label_nonsig_low,
    )

    # Foreground: significant + high consistency (blue)
    ax.scatter(
        _jitter(_sig_high["n_replicates_plot"]),
        _sig_high["activity_nap"],
        c="#3498db",
        s=_dot_size,
        alpha=0.8,
        edgecolor="none",
        zorder=10,
        label=_label_sig_high,
    )

    # Foreground: not significant + high consistency (red)
    ax.scatter(
        _jitter(_nonsig_high["n_replicates_plot"]),
        _nonsig_high["activity_nap"],
        c="#e74c3c",
        s=_dot_size,
        alpha=0.8,
        edgecolor="none",
        zorder=11,
        label=_label_nonsig_high,
    )

    ax.set_xlabel("Number of Replicates")
    ax.set_ylabel("Activity Effect Size (nMAP)")

    # Set x-axis limits based on floor and cap
    _x_min = (replicate_floor - 0.5) if replicate_floor else 1.5
    _x_max = (replicate_cap + 0.5) if replicate_cap else 20
    ax.set_xlim(_x_min, _x_max)
    ax.set_ylim(0, 1.05)

    # Set x-axis ticks with "<=N" for floor and "N+" for cap
    if replicate_floor is not None or replicate_cap is not None:
        _floor_val = replicate_floor or 2
        _cap_val = replicate_cap or 20
        _ticks = list(range(_floor_val, _cap_val + 1))
        _tick_labels = []
        for _t in _ticks:
            if replicate_floor is not None and _t == replicate_floor:
                _tick_labels.append(f"<={_t}")
            elif replicate_cap is not None and _t == replicate_cap:
                _tick_labels.append(f"{_t}+")
            else:
                _tick_labels.append(str(_t))
        ax.set_xticks(_ticks)
        ax.set_xticklabels(_tick_labels)
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Minimal styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Title if provided
    if title:
        ax.set_title(title, fontsize=10)

    # Legend at bottom (only for standalone figure)
    if show_legend and _created_figure:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=False,
            fontsize=8,
        )

    if _created_figure:
        plt.tight_layout()
        return fig
    return ax


@app.function
def plot_power_analysis_by_target(
    df: pd.DataFrame,
    target_stats: pd.DataFrame | None = None,
    targets: list[str] | None = None,
    max_targets: int | None = None,
    min_compounds: int = 5,
    ncols: int = 8,
    activity_threshold: float = 0.10,
    replicate_cap: int | None = 6,
    replicate_floor: int | None = 3,
    consistency_high_threshold: float = 0.3,
    panel_size: tuple[float, float] = (3.0, 2.5),
) -> plt.Figure:
    """Create faceted power analysis plot with one panel per target.

    Each panel shows the replicates-vs-effect-size scatter for a single target.
    Panels are highlighted (light blue background) if the target is significant
    in the consistency test.
    """
    # Select targets if not specified
    if targets is None:
        _target_counts = df.groupby("target_name")["Metadata_JCP2022"].nunique()
        _target_counts = _target_counts[_target_counts >= min_compounds]
        _target_counts = _target_counts.sort_values(ascending=False)
        if max_targets is not None:
            _target_counts = _target_counts.head(max_targets)
        targets = _target_counts.index.tolist()

    # Filter to selected targets
    _df_filtered = df[df["target_name"].isin(targets)].copy()

    # Build target stats lookup if provided
    _stats_lookup = {}
    if target_stats is not None:
        for _, _row in target_stats.iterrows():
            _stats_lookup[_row["target_name"]] = {
                "p": _row["target_p"],
                "nap": _row["target_nap"],
                "sig": _row["target_significant"],
            }

    # Calculate grid dimensions
    _n_targets = len(targets)
    _nrows = (_n_targets + ncols - 1) // ncols

    # Create figure
    fig, _axes = plt.subplots(
        _nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * _nrows),
        squeeze=False,
    )
    _axes = _axes.flatten()

    # Plot each target
    for _i, _target in enumerate(targets):
        _target_df = _df_filtered[_df_filtered["target_name"] == _target]
        _n_compounds = _target_df["Metadata_JCP2022"].nunique()

        # Build title with target stats if available
        if _target in _stats_lookup:
            _stats = _stats_lookup[_target]
            _p_val = _stats["p"]
            if _p_val < 0.001:
                _p_str = f"p={_p_val:.1e}"
            else:
                _p_str = f"p={_p_val:.3f}"
            _title = f"{_target}\nn={_n_compounds}, nMAP={_stats['nap']:.2f}, {_p_str}"
            if _stats["sig"]:
                _axes[_i].set_facecolor("#e8f4f8")
        else:
            _title = f"{_target}\nn={_n_compounds}"

        plot_power_analysis(
            _target_df,
            ax=_axes[_i],
            activity_threshold=activity_threshold,
            replicate_cap=replicate_cap,
            replicate_floor=replicate_floor,
            show_legend=False,
            consistency_col="consistency_nap",
            consistency_high_threshold=consistency_high_threshold,
            title=_title,
            show_counts_in_legend=False,
        )

        # Reduce label clutter for inner panels
        if _i % ncols != 0:
            _axes[_i].set_ylabel("")
        if _i < (_nrows - 1) * ncols:
            _axes[_i].set_xlabel("")

    # Hide unused axes
    for _i in range(_n_targets, len(_axes)):
        _axes[_i].set_visible(False)

    # Add shared legend at bottom
    _handles, _labels = _axes[0].get_legend_handles_labels()
    fig.legend(
        _handles,
        _labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    return fig


@app.function
def plot_target_consistency_scatter(
    target_stats: pd.DataFrame,
    figsize: tuple[float, float] = (12, 7),
    jitter_scale: float = 0.25,
    compound_bins: list[int] | None = None,
    show_boxplots: bool = True,
) -> plt.Figure:
    """Create scatter plot showing target consistency power relationship.

    Shows how target-level consistency significance depends on both:
    1. Number of compounds (statistical power) - X-axis (binned)
    2. nMAP (effect size) - Y-axis

    Key insight: "More compounds -> more likely significant, but nMAP also matters"
    """
    _df = target_stats.copy()

    # Default bins: 2, 3, 4, 5-6, 7-9, 10-14, 15+
    if compound_bins is None:
        compound_bins = [2, 3, 4, 5, 7, 10, 15, 100]

    # Create bin labels and assign bins
    _bin_labels = []
    for _i in range(len(compound_bins) - 1):
        _lo, _hi = compound_bins[_i], compound_bins[_i + 1]
        if _hi - _lo == 1:
            _bin_labels.append(str(_lo))
        elif _hi == 100:
            _bin_labels.append(f"{_lo}+")
        else:
            _bin_labels.append(f"{_lo}-{_hi - 1}")

    _df["compound_bin"] = pd.cut(
        _df["n_perturbations"],
        bins=compound_bins,
        labels=_bin_labels,
        right=False,
    )

    # Drop any rows that didn't fit in bins
    _df = _df.dropna(subset=["compound_bin"])

    # Create numeric x positions for bins
    _bin_positions = {label: i for i, label in enumerate(_bin_labels)}
    _df["x_pos"] = _df["compound_bin"].map(_bin_positions).astype(float)

    # Offset for side-by-side: non-significant left, significant right
    _box_offset = 0.27
    _df["x_offset"] = _df["target_significant"].map({False: -_box_offset, True: _box_offset})

    # Add jitter within each side
    _rng = np.random.default_rng(42)
    _jitter_within_box = jitter_scale * 0.5
    _df["x_jitter"] = (
        _df["x_pos"].values + _df["x_offset"].values + _rng.uniform(-_jitter_within_box, _jitter_within_box, len(_df))
    )

    fig, _ax = plt.subplots(figsize=figsize)

    # Separate significant and non-significant
    _sig_mask = _df["target_significant"]
    _nonsig = _df[~_sig_mask]
    _sig = _df[_sig_mask]

    # Draw side-by-side box plots if requested
    if show_boxplots:
        _box_width = 0.24

        # Non-significant box plots (left, gray)
        _nonsig_data = []
        _nonsig_positions = []
        for _i, _label in enumerate(_bin_labels):
            _bin_data = _nonsig[_nonsig["compound_bin"] == _label]["target_nap"].values
            if len(_bin_data) > 0:
                _nonsig_data.append(_bin_data)
                _nonsig_positions.append(_i - _box_offset)

        if _nonsig_data:
            _bp_nonsig = _ax.boxplot(
                _nonsig_data,
                positions=_nonsig_positions,
                widths=_box_width,
                patch_artist=True,
                showfliers=False,
                zorder=1,
            )
            for _patch in _bp_nonsig["boxes"]:
                _patch.set_facecolor("#e0e0e0")
                _patch.set_edgecolor("#999999")
                _patch.set_alpha(0.8)
            for _element in ["whiskers", "caps", "medians"]:
                for _line in _bp_nonsig[_element]:
                    _line.set_color("#777777")
                    _line.set_linewidth(1)

        # All targets box plots (center, light green/teal)
        _all_data = []
        _all_positions = []
        for _i, _label in enumerate(_bin_labels):
            _bin_data = _df[_df["compound_bin"] == _label]["target_nap"].values
            if len(_bin_data) > 0:
                _all_data.append(_bin_data)
                _all_positions.append(_i)

        if _all_data:
            _bp_all = _ax.boxplot(
                _all_data,
                positions=_all_positions,
                widths=_box_width,
                patch_artist=True,
                showfliers=False,
                zorder=0,
            )
            for _patch in _bp_all["boxes"]:
                _patch.set_facecolor("#e8f5e9")
                _patch.set_edgecolor("#66bb6a")
                _patch.set_alpha(0.7)
            for _element in ["whiskers", "caps", "medians"]:
                for _line in _bp_all[_element]:
                    _line.set_color("#43a047")
                    _line.set_linewidth(1)

        # Significant box plots (right, blue)
        _sig_data = []
        _sig_positions = []
        for _i, _label in enumerate(_bin_labels):
            _bin_data = _sig[_sig["compound_bin"] == _label]["target_nap"].values
            if len(_bin_data) > 0:
                _sig_data.append(_bin_data)
                _sig_positions.append(_i + _box_offset)

        if _sig_data:
            _bp_sig = _ax.boxplot(
                _sig_data,
                positions=_sig_positions,
                widths=_box_width,
                patch_artist=True,
                showfliers=False,
                zorder=1,
            )
            for _patch in _bp_sig["boxes"]:
                _patch.set_facecolor("#c6dbef")
                _patch.set_edgecolor("#2166ac")
                _patch.set_alpha(0.8)
            for _element in ["whiskers", "caps", "medians"]:
                for _line in _bp_sig[_element]:
                    _line.set_color("#2166ac")
                    _line.set_linewidth(1)

    # Fixed dot size
    _dot_size = 35

    # Plot non-significant first (gray, background)
    _ax.scatter(
        _nonsig["x_jitter"],
        _nonsig["target_nap"],
        c="#969696",
        s=_dot_size,
        alpha=0.6,
        edgecolors="none",
        label=f"Not significant (n={len(_nonsig)})",
        zorder=5,
    )

    # Plot significant on top (blue, foreground)
    _ax.scatter(
        _sig["x_jitter"],
        _sig["target_nap"],
        c="#2166ac",
        s=_dot_size,
        alpha=0.7,
        edgecolors="none",
        label=f"Significant (n={len(_sig)})",
        zorder=10,
    )

    # Add annotation with significance percentage at bottom center
    _significant_ratio = _df["target_significant"].mean()
    _n_sig = _sig_mask.sum()
    _ax.text(
        0.5,
        -0.18,
        f"Overall: {100 * _significant_ratio:.0f}% significant ({_n_sig} of {len(_df)} targets)",
        transform=_ax.transAxes,
        va="top",
        ha="center",
        fontsize=10,
        color="#525252",
    )

    # Add counts per bin at bottom
    for _i, _label in enumerate(_bin_labels):
        _bin_count = (_df["compound_bin"] == _label).sum()
        _bin_sig = ((_df["compound_bin"] == _label) & _df["target_significant"]).sum()
        _sig_pct = 100 * _bin_sig / _bin_count if _bin_count > 0 else 0
        _ax.text(
            _i,
            -0.12,
            f"n={_bin_count}, {_sig_pct:.0f}% sig",
            ha="center",
            va="top",
            fontsize=8,
            color="#666666",
            transform=_ax.get_xaxis_transform(),
        )

    # Styling
    _ax.set_xlabel("Number of Compounds per Target", fontsize=12)
    _ax.set_ylabel("Effect Size (nMAP)", fontsize=12)
    _ax.set_title("Target Consistency: Power vs Effect Size", fontsize=13)

    # Set x-axis ticks
    _ax.set_xticks(range(len(_bin_labels)))
    _ax.set_xticklabels(_bin_labels)
    _ax.set_xlim(-0.5, len(_bin_labels) - 0.5)

    # Y-axis
    _y_min = min(-0.05, _df["target_nap"].min() - 0.05)
    _y_max = max(1.05, _df["target_nap"].max() + 0.05)
    _ax.set_ylim(_y_min, _y_max)

    # Reference line at nMAP = 0
    _ax.axhline(0, color="#cccccc", linestyle="-", linewidth=1, alpha=0.5)

    # Grid (only horizontal)
    _ax.yaxis.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    _ax.set_axisbelow(True)

    # Remove top and right spines
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    # Legend with custom entries for the three boxplot types
    _legend_elements = [
        Patch(facecolor="#e0e0e0", edgecolor="#999999", label="Not significant"),
        Patch(facecolor="#e8f5e9", edgecolor="#66bb6a", label="All"),
        Patch(facecolor="#c6dbef", edgecolor="#2166ac", label="Significant"),
    ]
    _ax.legend(
        handles=_legend_elements,
        loc="upper right",
        frameon=True,
        fontsize=9,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    return fig


@app.function
def compute_summary_statistics(
    df: pd.DataFrame,
    activity_threshold: float = 0.10,
    consistency_high_threshold: float = 0.15,
) -> pd.DataFrame:
    """Compute summary statistics by significance and consistency status.

    Returns a DataFrame with group-level aggregates: n_compounds, median/mean/std
    of activity_nap, n_replicates, consistency_median, and n_targets.
    """
    _df = df.copy()
    _df["significant"] = _df["activity_p"] < activity_threshold
    _df["high_consistency"] = _df["consistency_max"] >= consistency_high_threshold

    _summary = (
        _df.groupby(["significant", "high_consistency"])
        .agg(
            {
                "Metadata_JCP2022": "count",
                "activity_nap": ["median", "mean", "std"],
                "n_replicates": ["median", "mean"],
                "consistency_median": ["median", "mean"],
                "n_targets": ["median", "mean"],
            }
        )
        .round(3)
    )

    _summary.columns = ["_".join(col).strip() for col in _summary.columns.values]
    _summary = _summary.rename(columns={"Metadata_JCP2022_count": "n_compounds"})

    return _summary.reset_index()


# -- Load data --


@app.cell
def _(activity_threshold_slider, consistency_threshold_slider, dataset_dropdown, mo, target_system_dropdown):
    mo.stop(
        not COPAIRS_RESULTS_DB.exists(),
        mo.md(f"**Error:** Copairs database not found at `{COPAIRS_RESULTS_DB}`. Run `just run` first."),
    )

    _dataset = dataset_dropdown.value
    _target_system = target_system_dropdown.value
    _activity_threshold = activity_threshold_slider.value
    _consistency_threshold = consistency_threshold_slider.value

    # Compute empirical high-consistency threshold
    consistency_high = compute_consistency_threshold(_dataset, _target_system, _consistency_threshold)

    # Load compound-level data
    compound_df = get_compound_power_data(_dataset, _consistency_threshold)

    mo.stop(
        len(compound_df) == 0,
        mo.md(f"**Error:** No compound data found for dataset `{_dataset}`."),
    )

    # Classify compounds into 4 categories
    compound_df["significant"] = compound_df["activity_p"] < _activity_threshold
    compound_df["high_consistency"] = compound_df["consistency_max"] >= consistency_high
    compound_df["sig_high_cons"] = compound_df["significant"] & compound_df["high_consistency"]
    compound_df["sig_low_cons"] = compound_df["significant"] & (~compound_df["high_consistency"])
    compound_df["nonsig_high_cons"] = (~compound_df["significant"]) & compound_df["high_consistency"]
    compound_df["nonsig_low_cons"] = (~compound_df["significant"]) & (~compound_df["high_consistency"])

    _n_sig_high = compound_df["sig_high_cons"].sum()
    _n_sig_low = compound_df["sig_low_cons"].sum()
    _n_nonsig_high = compound_df["nonsig_high_cons"].sum()
    _n_nonsig_low = compound_df["nonsig_low_cons"].sum()

    mo.md(f"""
    ## Data loaded

    **Dataset:** {_dataset} | **Target system:** {_target_system}
    | **Activity threshold:** p < {_activity_threshold}
    | **Consistency threshold (empirical):** nAP >= {consistency_high:.3f}

    | Category | Count | Color |
    |----------|-------|-------|
    | Significant, high consistency | {_n_sig_high:,} | Blue |
    | Significant, low consistency | {_n_sig_low:,} | Green |
    | Not significant, high consistency | {_n_nonsig_high:,} | Red |
    | Not significant, low consistency | {_n_nonsig_low:,} | Gray |
    | **Total** | **{len(compound_df):,}** | |
    """)
    return (compound_df, consistency_high)


# -- Main power analysis scatter --


@app.cell
def _(activity_threshold_slider, compound_df, consistency_high):
    fig_power = plot_power_analysis(
        compound_df,
        activity_threshold=activity_threshold_slider.value,
        replicate_cap=6,
        replicate_floor=3,
        consistency_high_threshold=consistency_high,
    )
    fig_power
    return (fig_power,)


# -- Summary statistics --


@app.cell
def _(activity_threshold_slider, compound_df, consistency_high, mo):
    summary_df = compute_summary_statistics(
        compound_df,
        activity_threshold=activity_threshold_slider.value,
        consistency_high_threshold=consistency_high,
    )
    mo.ui.table(summary_df)
    return (summary_df,)


# -- Red vs Blue comparison --


@app.cell
def _(compound_df, mo):
    _nonsig_high = compound_df[compound_df["nonsig_high_cons"]]
    _sig_high = compound_df[compound_df["sig_high_cons"]]

    mo.stop(
        len(_nonsig_high) == 0 or len(_sig_high) == 0,
        mo.md("Not enough data in both groups for comparison."),
    )

    mo.md(f"""
    ## Key Insight: Red vs Blue Comparison

    Not significant, high consistency (**RED**, n={len(_nonsig_high)}) vs
    Significant, high consistency (**BLUE**, n={len(_sig_high)}):

    | Metric | RED (not sig) | BLUE (sig) |
    |--------|---------------|------------|
    | Activity effect size (nMAP) | {_nonsig_high["activity_nap"].median():.3f} | {_sig_high["activity_nap"].median():.3f} |
    | Replicate count (median) | {_nonsig_high["n_replicates"].median():.0f} | {_sig_high["n_replicates"].median():.0f} |
    | Consistency (median nAP) | {_nonsig_high["consistency_median"].median():.3f} | {_sig_high["consistency_median"].median():.3f} |

    The red dots (not significant but high consistency) have lower activity effect
    size and fewer replicates than blue, but similar consistency signal.
    This suggests they may be underpowered detections of real biological signal.
    """)
    return


# -- Not significant, high consistency details --


@app.cell
def _(compound_df, mo):
    _nonsig_high = compound_df[compound_df["nonsig_high_cons"]].sort_values("activity_p")
    _display_cols = [
        "Metadata_JCP2022",
        "activity_p",
        "activity_nap",
        "n_replicates",
        "consistency_median",
        "consistency_max",
        "n_targets",
    ]
    mo.md("## Not Significant, High Consistency Compounds (red dots)")
    return


@app.cell
def _(compound_df, mo):
    _nonsig_high = compound_df[compound_df["nonsig_high_cons"]].sort_values("activity_p")
    _display_cols = [
        "Metadata_JCP2022",
        "activity_p",
        "activity_nap",
        "n_replicates",
        "consistency_median",
        "consistency_max",
        "n_targets",
    ]
    mo.ui.table(_nonsig_high[_display_cols])
    return


# -- Target consistency scatter --


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Target-Level Analysis

    The target consistency scatter shows how significance depends on both the
    number of compounds per target (power) and the target-level nMAP (effect size).
    Targets with more compounds are more likely to reach significance, but
    effect size also matters.
    """)
    return


@app.cell
def _(consistency_threshold_slider, dataset_dropdown, target_system_dropdown):
    target_stats = get_target_consistency_stats(
        dataset_dropdown.value,
        target_system_dropdown.value,
        consistency_threshold_slider.value,
    )
    return (target_stats,)


@app.cell
def _(target_stats):
    fig_target_scatter = plot_target_consistency_scatter(target_stats)
    fig_target_scatter
    return (fig_target_scatter,)


# -- By-target faceted plot --


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## By-Target Faceted Analysis

    Each panel shows the replicates-vs-effect-size scatter for a single target.
    Panels with light blue backgrounds indicate targets whose consistency
    reached significance (p < 0.05).
    """)
    return


@app.cell
def _(mo):
    min_compounds_slider = mo.ui.slider(start=3, stop=15, step=1, value=5, label="Min compounds per target")
    max_targets_slider = mo.ui.slider(start=8, stop=64, step=8, value=32, label="Max targets to show")
    ncols_slider = mo.ui.slider(start=4, stop=12, step=2, value=8, label="Columns")
    mo.hstack([min_compounds_slider, max_targets_slider, ncols_slider])
    return (max_targets_slider, min_compounds_slider, ncols_slider)


@app.cell
def _(
    activity_threshold_slider,
    consistency_high,
    consistency_threshold_slider,
    dataset_dropdown,
    max_targets_slider,
    min_compounds_slider,
    ncols_slider,
    target_stats,
    target_system_dropdown,
):
    _target_df = get_target_level_power_data(
        dataset_dropdown.value,
        target_system_dropdown.value,
        consistency_threshold_slider.value,
    )

    fig_by_target = plot_power_analysis_by_target(
        _target_df,
        target_stats=target_stats,
        max_targets=max_targets_slider.value,
        min_compounds=min_compounds_slider.value,
        ncols=ncols_slider.value,
        activity_threshold=activity_threshold_slider.value,
        replicate_cap=6,
        replicate_floor=3,
        consistency_high_threshold=consistency_high,
    )
    fig_by_target
    return (fig_by_target,)


# -- Save outputs --


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Save Outputs")
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save all outputs")
    save_button
    return (save_button,)


@app.cell
def _(
    dataset_dropdown,
    fig_by_target,
    fig_power,
    fig_target_scatter,
    mo,
    save_button,
    summary_df,
):
    mo.stop(not save_button.value)

    _dataset = dataset_dropdown.value
    _outdir = PROCESSED_DATA_DIR / "exploration" / "0.07" / _dataset
    _outdir.mkdir(parents=True, exist_ok=True)

    fig_power.savefig(_outdir / "power_analysis.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_target_scatter.savefig(_outdir / "target_consistency_scatter.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_by_target.savefig(_outdir / "power_analysis_by_target.png", dpi=DEFAULT_DPI, bbox_inches="tight")

    _summary_path = _outdir / "power_analysis_summary.csv"
    summary_df.to_csv(_summary_path, index=False)

    mo.md(f"""
    **Saved outputs to** `{_outdir}/`:

    - `power_analysis.png`
    - `target_consistency_scatter.png`
    - `power_analysis_by_target.png`
    - `power_analysis_summary.csv`
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
