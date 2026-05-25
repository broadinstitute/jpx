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
#     "upsetplot==0.9.0",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import sys
    import warnings
    from pathlib import Path

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from upsetplot import UpSet, from_indicators

    from nb00_ss_config import COPAIRS_RESULTS_DB, DEFAULT_DPI, PROCESSED_DATA_DIR
    from nb02_ss_queries import query_activity_results, query_consistency_results

    warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot")
    pd.set_option("future.no_silent_downcasting", True)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Profile Comparison

    Compare phenotypic activity and consistency across two profile types
    (e.g., `compound_DL_CPCNN_no_source7` vs `compound_no_source7`).

    Produces:
    - **Threshold curves**: fraction significant at each p-value threshold
    - **Overlap analysis**: UpSet plot and concordance metrics
    - Saves results to `data/processed/profile-comparison/`

    *Converted from `1.01-jfh-profile-comparison.py`*
    """)
    return


@app.cell
def _(mo):
    profile1_dropdown = mo.ui.dropdown(
        options=[
            "compound_no_source7",
            "compound_DL_CPCNN_no_source7",
        ],
        value="compound_DL_CPCNN_no_source7",
        label="Profile 1",
    )
    profile2_dropdown = mo.ui.dropdown(
        options=[
            "compound_no_source7",
            "compound_DL_CPCNN_no_source7",
        ],
        value="compound_no_source7",
        label="Profile 2",
    )
    mo.hstack([profile1_dropdown, profile2_dropdown])
    return (profile1_dropdown, profile2_dropdown)


@app.cell
def _(mo):
    _analysis_options = get_available_analysis_types()
    analysis_type_dropdown = mo.ui.dropdown(
        options=_analysis_options,
        value=_analysis_options[0] if _analysis_options else "activity",
        label="Analysis Type",
    )
    distance_dropdown = mo.ui.dropdown(
        options=["cosine", "abs_cosine"],
        value="cosine",
        label="Distance (consistency only)",
    )
    activity_params_dropdown = mo.ui.dropdown(
        options=["default", "withinsource", "crosssource"],
        value="default",
        label="Activity Params",
    )
    mo.hstack([analysis_type_dropdown, distance_dropdown, activity_params_dropdown])
    return (activity_params_dropdown, analysis_type_dropdown, distance_dropdown)


@app.function
def get_available_analysis_types() -> list[str]:
    """Get available analysis types from the copairs results database."""
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    types = ["activity"]
    try:
        targets = con.execute(
            "SELECT DISTINCT _group_type FROM consistency_results "
            "WHERE _preprocessing NOT LIKE '%_sweep' ORDER BY _group_type"
        ).fetchall()
        types.extend([f"consistency_{t[0]}" for t in targets])
    except duckdb.CatalogException:
        pass
    con.close()
    return types


@app.function
def query_results_for_profiles(
    analysis_type: str,
    profiles: list[str],
    preprocessing: str = "activity_no_target2",
    filter_name: str = "all_sources",
    distance: str = "cosine",
    activity_params: str = "default",
) -> dict[str, pd.DataFrame]:
    """Query results from copairs DB for the given profiles.

    Uses foundation query functions from nb02_ss_queries.
    """
    results = {}
    if analysis_type == "activity":
        for profile in profiles:
            _df = query_activity_results(profile, preprocessing, filter_name, activity_params)
            if len(_df) > 0:
                results[profile] = _df
    else:
        _target = analysis_type.removeprefix("consistency_")
        _cons_preprocessing = preprocessing.replace("activity_", "consistency_")
        for profile in profiles:
            _df = query_consistency_results(profile, _cons_preprocessing, filter_name, _target, distance)
            if len(_df) > 0:
                results[profile] = _df
    return results


@app.function
def get_significance_summary(data: dict[str, pd.DataFrame], thresholds: list[float]) -> pd.DataFrame:
    """Calculate fraction significant at each p-value threshold."""
    rows = []
    for name, df in data.items():
        if "corrected_p_value" not in df.columns:
            continue
        row = {"profile": name}
        for t in thresholds:
            row[f"p<{t}"] = (df["corrected_p_value"] < t).mean()
        rows.append(row)
    return pd.DataFrame(rows)


@app.function
def plot_threshold_curves(
    summary_df: pd.DataFrame,
    thresholds: list[float],
    title: str,
) -> plt.Figure:
    """Plot fraction significant vs p-value threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in summary_df.iterrows():
        _fractions = [row[f"p<{t}"] for t in thresholds]
        ax.plot(thresholds, _fractions, marker="o", label=row["profile"], linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Corrected p-value threshold")
    ax.set_ylabel("Fraction significant")
    ax.set_title(f"Significance by threshold: {title}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    return fig


@app.function
def create_overlap_analysis(
    data: dict[str, pd.DataFrame],
    analysis_type: str,
    title: str,
) -> tuple[pd.DataFrame | None, plt.Figure | None]:
    """Create overlap analysis between two profiles.

    Returns (summary_df, upset_fig) tuple. Either may be None if
    the analysis cannot be performed.
    """
    if len(data) != 2:
        return None, None

    _names = list(data.keys())
    _dfs = list(data.values())

    # Determine key column based on analysis type
    if analysis_type == "activity":
        _key_col = "Metadata_JCP2022"
    else:
        _key_col = "group_value"

    if _key_col not in _dfs[0].columns or "below_corrected_p" not in _dfs[0].columns:
        return None, None

    # Merge the two dataframes on the key column
    _merged = (
        _dfs[0][[_key_col, "below_corrected_p"]]
        .rename(columns={"below_corrected_p": _names[0]})
        .merge(
            _dfs[1][[_key_col, "below_corrected_p"]].rename(columns={"below_corrected_p": _names[1]}),
            on=_key_col,
            how="outer",
        )
    )
    _merged[_names[0]] = _merged[_names[0]].fillna(False).astype(bool)
    _merged[_names[1]] = _merged[_names[1]].fillna(False).astype(bool)

    # Vectorized category assignment
    _merged["category"] = np.select(
        [
            _merged[_names[0]] & _merged[_names[1]],
            _merged[_names[0]] & ~_merged[_names[1]],
            ~_merged[_names[0]] & _merged[_names[1]],
        ],
        ["both", f"only_{_names[0]}", f"only_{_names[1]}"],
        default="neither",
    )

    # Summary with concordance
    _summary = _merged["category"].value_counts().reset_index()
    _summary.columns = ["category", "count"]

    _counts = dict(zip(_summary["category"], _summary["count"]))
    _total = _summary["count"].sum()
    _both = _counts.get("both", 0)
    _neither = _counts.get("neither", 0)
    _concordance = int((_both + _neither) / _total) if _total > 0 else 0

    _concordance_row = pd.DataFrame([{"category": "concordance", "count": _concordance}])
    _summary = pd.concat([_summary, _concordance_row], ignore_index=True)

    # UpSet plot (exclude items not significant in either)
    _upset_fig = None
    _sig_items = _merged[_merged[_names[0]] | _merged[_names[1]]]
    if len(_sig_items) > 0:
        try:
            _upset_data = from_indicators(_names, data=_sig_items)
            _upset_fig = plt.figure(figsize=(14, 6))
            _upset = UpSet(_upset_data, show_counts=True)
            _axes = _upset.plot(fig=_upset_fig)
            _axes["matrix"].tick_params(axis="y", labelsize=8)
            plt.suptitle(f"Overlap: {title}")
        except (AttributeError, ValueError, TypeError):
            _upset_fig = None

    return _summary, _upset_fig


# --- Query data ---


@app.cell
def _(
    activity_params_dropdown,
    analysis_type_dropdown,
    distance_dropdown,
    mo,
    profile1_dropdown,
    profile2_dropdown,
):
    _p1 = profile1_dropdown.value
    _p2 = profile2_dropdown.value
    mo.stop(_p1 == _p2, mo.md("**Select two different profiles to compare.**"))

    data = query_results_for_profiles(
        analysis_type_dropdown.value,
        [_p1, _p2],
        distance=distance_dropdown.value,
        activity_params=activity_params_dropdown.value,
    )
    mo.stop(
        len(data) < 2,
        mo.md(f"**Only found data for {list(data.keys())}. Need both profiles.**"),
    )
    mo.md(f"Loaded **{sum(len(v) for v in data.values())}** total rows across **{len(data)}** profiles")
    return (data,)


# --- Threshold analysis ---


@app.cell
def _(analysis_type_dropdown, data):
    _thresholds = [0.5, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.001, 0.0001]
    threshold_summary = get_significance_summary(data, _thresholds)
    fig_threshold = plot_threshold_curves(
        threshold_summary,
        _thresholds,
        analysis_type_dropdown.value,
    )
    fig_threshold
    return (fig_threshold, threshold_summary)


# --- Overlap / UpSet analysis ---


@app.cell
def _(analysis_type_dropdown, data, mo):
    overlap_summary, fig_upset = create_overlap_analysis(
        data,
        analysis_type_dropdown.value,
        analysis_type_dropdown.value,
    )
    mo.stop(overlap_summary is None, mo.md("**Could not compute overlap.**"))
    mo.vstack(
        [
            mo.md("## Overlap Summary"),
            overlap_summary,
            fig_upset,
        ]
    )
    return (fig_upset, overlap_summary)


# --- Save outputs ---


@app.cell
def _(
    activity_params_dropdown,
    analysis_type_dropdown,
    distance_dropdown,
    fig_threshold,
    fig_upset,
    mo,
    overlap_summary,
    profile1_dropdown,
    profile2_dropdown,
    threshold_summary,
):
    _p1 = profile1_dropdown.value
    _p2 = profile2_dropdown.value
    _analysis = analysis_type_dropdown.value

    _comparison_dir = PROCESSED_DATA_DIR / "profile-comparison" / f"{_p1}_vs_{_p2}"
    if _analysis == "activity":
        _output_dir = _comparison_dir / "activity" / activity_params_dropdown.value
    else:
        _grouping = _analysis.removeprefix("consistency_")
        _output_dir = _comparison_dir / "consistency" / _grouping / distance_dropdown.value

    _output_dir.mkdir(parents=True, exist_ok=True)

    # Save threshold CSV and plot
    threshold_summary.to_csv(_output_dir / "active_at_threshold.csv", index=False)
    fig_threshold.savefig(
        _output_dir / "threshold_curve.png",
        dpi=DEFAULT_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    # Save overlap outputs
    if overlap_summary is not None:
        overlap_summary.to_csv(_output_dir / "overlap_summary.csv", index=False)
    if fig_upset is not None:
        fig_upset.savefig(
            _output_dir / "overlap_upset.png",
            dpi=DEFAULT_DPI,
            bbox_inches="tight",
            facecolor="white",
        )

    mo.md(f"Saved outputs to `{_output_dir}/`")
    return


@app.function
def run_profile_comparison(output_dir=None) -> str:
    """Run all pairwise profile comparisons for activity and consistency.

    Iterates over COMPARISON_PROFILES_ACTIVITY (activity) and
    COMPARISON_PROFILES_CONSISTENCY (all group_type x distance combos).
    Mirrors the old scripts/profile_comparison_runner.py logic.

    Called from workflow.py. Returns the base output directory path.
    """
    import matplotlib.pyplot as plt

    from nb00_ss_config import DEFAULT_DPI, PROCESSED_DATA_DIR

    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "profile-comparison"
    else:
        output_dir = Path(output_dir)

    # Profile pairs (mirrors rules/common.smk defaults)
    comparison_activity = [
        ("compound_no_source7", "compound_DL_CPCNN_no_source7"),
    ]
    comparison_consistency = [
        ("compound_no_source7_active_union", "compound_DL_CPCNN_no_source7_active_union"),
    ]

    # Discover (group_type, distance) combos from copairs DB
    _con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    _group_distance_combos = _con.execute(
        "SELECT DISTINCT _group_type, _distance FROM consistency_results "
        "WHERE _preprocessing NOT LIKE '%_sweep' ORDER BY _group_type, _distance"
    ).fetchall()
    _con.close()

    _thresholds = [0.5, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.001, 0.0001]

    def _run_one(
        analysis_type, p1, p2, activity_params="default", distance="cosine", preprocessing="activity_no_target2"
    ):
        """Run a single profile comparison and save outputs."""
        _data = query_results_for_profiles(
            analysis_type, [p1, p2], preprocessing=preprocessing, distance=distance, activity_params=activity_params
        )
        if len(_data) < 2:
            return

        # Determine output subdirectory
        _comparison_dir = output_dir / f"{p1}_vs_{p2}"
        if analysis_type == "activity":
            _out = _comparison_dir / "activity" / activity_params
        else:
            _grouping = analysis_type.removeprefix("consistency_")
            _out = _comparison_dir / "consistency" / _grouping / distance
        _out.mkdir(parents=True, exist_ok=True)

        # Threshold curves
        _summary = get_significance_summary(_data, _thresholds)
        _summary.to_csv(_out / "active_at_threshold.csv", index=False)
        _fig = plot_threshold_curves(_summary, _thresholds, analysis_type)
        _fig.savefig(_out / "threshold_curve.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
        plt.close(_fig)

        # Overlap / UpSet
        _overlap, _upset_fig = create_overlap_analysis(_data, analysis_type, analysis_type)
        if _overlap is not None:
            _overlap.to_csv(_out / "overlap_summary.csv", index=False)
        if _upset_fig is not None:
            try:
                _upset_fig.savefig(_out / "overlap_upset.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
            except TypeError:
                pass
            plt.close(_upset_fig)

    # Activity comparisons
    for _p1, _p2 in comparison_activity:
        _run_one("activity", _p1, _p2)

    # Consistency comparisons (active_union uses prefiltered preprocessing)
    for _p1, _p2 in comparison_consistency:
        for _group_type, _distance in _group_distance_combos:
            _run_one(f"consistency_{_group_type}", _p1, _p2, distance=_distance, preprocessing="activity_prefiltered")

    # Touch .complete marker
    (output_dir / ".complete").touch()
    return str(output_dir)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
