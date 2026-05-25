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
#     "scanpy==1.12.1",
#     "scipy==1.17.1",
#     "seaborn==0.13.2",
#     "statsmodels==0.14.6",
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
    import seaborn as sns
    import statsmodels.formula.api as smf
    from loguru import logger
    from scipy import stats

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import (
        COPAIRS_RESULTS_DB,
        DEFAULT_DPI,
        METADATA_DB,
        PROCESSED_DATA_DIR,
    )
    from nb02_ss_queries import query_activity_results
    from nb04_ss_visualization import (
        compute_activity_stats,
        format_stat_annotation,
        plot_hexbin_with_marginals,
        plot_scatter_with_marginals,
    )

    OUTPUT_DIR = PROCESSED_DATA_DIR / "data-quality"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell Count Quality

    Two analyses of JUMP Cell Painting cell count data:

    1. **Cell density vs phenotypic activity** - Is measured activity driven by
       toxicity (low cell count)? Compares raw, source-excluded, and
       plate-normalized cell density against compound nMAP.

    2. **Edge well effects** - Do wells at the plate perimeter show
       systematically different cell counts? Heatmaps, marginals, and violins
       per source, plus a statistical edge-vs-interior comparison.

    *Outputs:* `data/processed/data-quality/{dataset}/{preprocessing}/`
    """)
    return


# ---------------------------------------------------------------------------
# Interactive controls
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
            "activity_no_target2",
        ],
        value="activity_no_target2",
        label="Preprocessing",
    )
    filter_dropdown = mo.ui.dropdown(
        options=[
            "all_sources",
        ],
        value="all_sources",
        label="Filter",
    )
    activity_params_dropdown = mo.ui.dropdown(
        options=[
            "default",
            "withinsource",
            "crosssource",
        ],
        value="default",
        label="Activity params",
    )
    mo.hstack(
        [dataset_dropdown, preprocessing_dropdown, filter_dropdown, activity_params_dropdown],
        justify="start",
    )
    return (
        activity_params_dropdown,
        dataset_dropdown,
        filter_dropdown,
        preprocessing_dropdown,
    )


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------


@app.function
def query_cell_counts(
    plate_format: int = 384,
    plate_type: str = "COMPOUND",
    normalize_by_area: bool = True,
) -> pd.DataFrame:
    """Query cell counts from metadata database.

    Args:
        plate_format: Filter to sources using this plate format (384, 1536, or 0 for all).
        plate_type: Filter to this plate type (e.g., "COMPOUND"). Use None for all.
        normalize_by_area: If True, divide cell counts by well area (in mm2).
    """
    sources_384 = ["source_2", "source_3", "source_5", "source_6", "source_8", "source_10", "source_11"]
    sources_1536 = ["source_1", "source_9"]

    if plate_format == 384:
        source_filter = sources_384
    elif plate_format == 1536:
        source_filter = sources_1536
    else:
        source_filter = None

    _params = []
    _where_clauses = []

    if source_filter:
        _placeholders = ", ".join(["?" for _ in source_filter])
        _where_clauses.append(f"c.Metadata_Source IN ({_placeholders})")
        _params.extend(source_filter)

    if plate_type:
        _where_clauses.append("p.Metadata_PlateType = ?")
        _params.append(plate_type)

    _where_sql = " AND ".join(_where_clauses) if _where_clauses else "1=1"

    _query = f"""
        SELECT
            c.Metadata_Source,
            c.Metadata_Plate,
            c.Metadata_Well,
            c.Metadata_Count_Cells,
            m.Metadata_Sites_Per_Well,
            m.Metadata_Well_Area_Microns2,
            p.Metadata_PlateType
        FROM cell_counts c
        LEFT JOIN microscope_config_fov m ON c.Metadata_Source = m.Metadata_Source
        LEFT JOIN plate p ON c.Metadata_Plate = p.Metadata_Plate
        WHERE {_where_sql}
    """
    _con = duckdb.connect(str(METADATA_DB), read_only=True)
    _df = _con.execute(_query, _params).df()
    _con.close()

    if normalize_by_area and len(_df) > 0:
        _df["Metadata_Well_Area_mm2"] = _df["Metadata_Well_Area_Microns2"] / 1e6
        _df["Metadata_Count_Cells_Per_mm2"] = _df["Metadata_Count_Cells"] / _df["Metadata_Well_Area_mm2"]
        logger.info(
            f"Normalized cell counts by well area (range: "
            f"{_df['Metadata_Well_Area_mm2'].min():.2f}-{_df['Metadata_Well_Area_mm2'].max():.2f} mm2)"
        )

    return _df


@app.function
def query_cell_density(
    exclude_sources: list[str] | None = None,
    aggregate: bool = True,
) -> pd.DataFrame:
    """Query cell density (cells/mm2) from the metadata database.

    Cell counts are normalized by well area to account for different imaging
    configurations across sources (well areas range from 1.67 to 5.26 mm2).

    Args:
        exclude_sources: List of sources to exclude (e.g., ["source_2"])
        aggregate: If True, return median per compound. If False, return well-level data.
    """
    _where_clause = "w.Metadata_JCP2022 IS NOT NULL"
    _params = []

    if exclude_sources:
        _placeholders = ", ".join(["?" for _ in exclude_sources])
        _where_clause += f" AND c.Metadata_Source NOT IN ({_placeholders})"
        _params.extend(exclude_sources)

    if aggregate:
        _query = f"""
            SELECT
                w.Metadata_JCP2022,
                MEDIAN(c.Metadata_Count_Cells / (m.Metadata_Well_Area_Microns2 / 1e6)) as Metadata_median_cell_density
            FROM well w
            JOIN cell_counts c ON w.Metadata_Plate = c.Metadata_Plate AND w.Metadata_Well = c.Metadata_Well
            JOIN microscope_config_fov m ON c.Metadata_Source = m.Metadata_Source
            WHERE {_where_clause}
            GROUP BY w.Metadata_JCP2022
        """
    else:
        _query = f"""
            SELECT
                w.Metadata_JCP2022,
                c.Metadata_Source,
                c.Metadata_Plate,
                c.Metadata_Well,
                c.Metadata_Count_Cells / (m.Metadata_Well_Area_Microns2 / 1e6) as cell_density
            FROM well w
            JOIN cell_counts c ON w.Metadata_Plate = c.Metadata_Plate AND w.Metadata_Well = c.Metadata_Well
            JOIN microscope_config_fov m ON c.Metadata_Source = m.Metadata_Source
            WHERE {_where_clause}
        """

    _con = duckdb.connect(str(METADATA_DB), read_only=True)
    _df = _con.execute(_query, _params).df()
    _con.close()
    return _df


@app.function
def normalize_cell_density(df: pd.DataFrame, value_col: str = "cell_density") -> pd.DataFrame:
    """Two-step normalization: regress out source, then robust z-score by plate.

    Step 1: Regress out source effects (removes systematic lab/protocol differences)
    Step 2: Robust z-score by plate (centers and scales within each plate)

    Args:
        df: DataFrame with Metadata_Source, Metadata_Plate, and value_col columns
        value_col: Column name containing raw cell density values

    Returns:
        DataFrame with {value_col}_normalized column
    """
    _df = df.copy()

    # Step 1: Regress out source effects
    _model = smf.ols(f"{value_col} ~ C(Metadata_Source)", data=_df).fit()
    _source_adj = _model.resid
    logger.info(f"Regressed out source effects (R2 = {_model.rsquared:.3f})")

    # Step 2: Robust z-score by plate
    def _robust_zscore(x: pd.Series) -> pd.Series:
        _median = x.median()
        _mad = (x - _median).abs().median()
        if _mad > 0:
            return (x - _median) / (_mad * 1.4826)
        _std = x.std()
        if _std > 0:
            return (x - _median) / _std
        return x - _median

    _df[f"{value_col}_normalized"] = (
        pd.Series(_source_adj, index=_df.index).groupby(_df["Metadata_Plate"]).transform(_robust_zscore)
    )

    return _df


@app.function
def aggregate_compound_cell_density(
    df: pd.DataFrame,
    value_col: str = "cell_density_normalized",
    output_col: str = "Metadata_median_cell_density",
) -> pd.DataFrame:
    """Aggregate well-level data to compound-level by taking the median.

    Args:
        df: Well-level DataFrame with Metadata_JCP2022 and value_col
        value_col: Column to aggregate (default matches normalize_cell_density output)
        output_col: Name for the output column
    """
    return df.groupby("Metadata_JCP2022")[value_col].median().reset_index(name=output_col)


@app.function
def parse_well_position(well: str) -> tuple[str, int]:
    """Parse well ID like 'A01' into row letter and column number."""
    _match = re.match(r"([A-Z]+)(\d+)", well)
    if _match:
        return _match.group(1), int(_match.group(2))
    return well, 0


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


@app.function
def analyze_cell_density_vs_activity(
    activity_df: pd.DataFrame,
    cell_density_df: pd.DataFrame,
    density_label: str = "Median cell density (cells/mm2)",
) -> tuple[pd.DataFrame, dict]:
    """Analyze relationship between cell density and phenotypic activity.

    Returns the merged dataframe and a dict of summary statistics.
    """
    _merged = activity_df.merge(cell_density_df, on="Metadata_JCP2022", how="inner")
    logger.info(f"Merged {len(_merged):,} compounds with both activity and cell density data")

    _valid = _merged.dropna(subset=["Metadata_median_cell_density", "mean_normalized_average_precision"])
    _r, _p = stats.spearmanr(_valid["Metadata_median_cell_density"], _valid["mean_normalized_average_precision"])

    _sig = _merged[_merged["below_corrected_p"]]
    _nonsig = _merged[~_merged["below_corrected_p"]]

    _mw_stat, _mw_p = stats.mannwhitneyu(
        _sig["Metadata_median_cell_density"].dropna(),
        _nonsig["Metadata_median_cell_density"].dropna(),
        alternative="two-sided",
    )
    logger.info(f"Mann-Whitney U test: statistic={_mw_stat:.0f}, p={_mw_p:.2e}")

    _summary = {
        "n_compounds": len(_merged),
        "spearman_r": _r,
        "spearman_p": _p,
        "n_significant": len(_sig),
        "n_not_significant": len(_nonsig),
        "median_cell_density_sig": _sig["Metadata_median_cell_density"].median(),
        "median_cell_density_nonsig": _nonsig["Metadata_median_cell_density"].median(),
        "mannwhitney_statistic": _mw_stat,
        "mannwhitney_p": _mw_p,
        "density_label": density_label,
    }

    return _merged, _summary


@app.function
def plot_cell_density_distribution(
    raw_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
) -> plt.Figure:
    """Compare compound-level cell density distributions: raw vs plate-normalized.

    Shows whether bimodality in raw cell density is due to plate/source batch effects
    or reflects true compound-level differences.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Raw cell density distribution
    _ax1 = axes[0]
    _raw_vals = raw_df["Metadata_median_cell_density"].dropna()
    _bins_raw = np.linspace(_raw_vals.quantile(0.01), _raw_vals.quantile(0.99), 50)
    _ax1.hist(_raw_vals, bins=_bins_raw, color="#3498db", alpha=0.7, edgecolor="white", linewidth=0.5)
    _ax1.axvline(
        _raw_vals.median(), color="#e74c3c", linestyle="--", linewidth=2, label=f"Median: {_raw_vals.median():.0f}"
    )
    _ax1.set_xlabel("Cell density (cells/mm2)")
    _ax1.set_ylabel("Count")
    _ax1.set_title("Raw cell density")
    _ax1.legend(loc="upper right", fontsize=9)
    _ax1.spines["top"].set_visible(False)
    _ax1.spines["right"].set_visible(False)

    # Right: Plate-normalized distribution
    _ax2 = axes[1]
    _norm_vals = normalized_df["Metadata_median_cell_density"].dropna()
    _bins_norm = np.linspace(_norm_vals.quantile(0.01), _norm_vals.quantile(0.99), 50)
    _ax2.hist(_norm_vals, bins=_bins_norm, color="#2ecc71", alpha=0.7, edgecolor="white", linewidth=0.5)
    _ax2.axvline(
        _norm_vals.median(), color="#e74c3c", linestyle="--", linewidth=2, label=f"Median: {_norm_vals.median():.2f}"
    )
    _ax2.axvline(0, color="#7f8c8d", linestyle=":", linewidth=1, alpha=0.7)
    _ax2.set_xlabel("Plate-normalized cell density (robust z-score)")
    _ax2.set_ylabel("Count")
    _ax2.set_title("Plate-normalized cell density")
    _ax2.legend(loc="upper right", fontsize=9)
    _ax2.spines["top"].set_visible(False)
    _ax2.spines["right"].set_visible(False)

    fig.suptitle(f"Compound-level cell density distribution (n={len(_raw_vals):,} compounds)", fontsize=11, y=1.02)
    plt.tight_layout()

    logger.info(f"Raw cell density: median={_raw_vals.median():.0f}, std={_raw_vals.std():.0f}")
    logger.info(f"Plate-normalized: median={_norm_vals.median():.2f}, std={_norm_vals.std():.2f}")

    return fig


@app.function
def plot_edge_heatmaps(
    viz_df: pd.DataFrame,
    count_col: str,
    count_label: str,
) -> plt.Figure:
    """Plot per-source heatmaps of cell counts by well position."""
    _sources = sorted(viz_df["Metadata_Source"].unique(), key=lambda x: int(x.split("_")[1]))
    _n_sources = len(_sources)
    _vmin = viz_df[count_col].quantile(0.01)
    _vmax = viz_df[count_col].quantile(0.99)

    fig, axes = plt.subplots(_n_sources, 1, figsize=(8, 1.5 * _n_sources), gridspec_kw={"hspace": 0.3})
    if _n_sources == 1:
        axes = [axes]

    for _i, _source in enumerate(_sources):
        _source_df = viz_df[viz_df["Metadata_Source"] == _source]
        _source_means = _source_df.groupby(["row", "col"])[count_col].mean().reset_index()
        _source_means.columns = ["row", "col", "mean_cell_count"]
        _src_rows = sorted(_source_means["row"].unique(), key=lambda x: (len(x), x))
        _well_area_mm2 = _source_df["Metadata_Well_Area_mm2"].iloc[0]

        _pivot = _source_means.pivot(index="row", columns="col", values="mean_cell_count")
        _pivot = _pivot.reindex(_src_rows)[sorted(_pivot.columns)]

        _ax = axes[_i]
        sns.heatmap(
            _pivot,
            ax=_ax,
            cmap="YlGnBu",
            square=True,
            linewidths=0.3,
            linecolor="white",
            vmin=_vmin,
            vmax=_vmax,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        _ax.set_ylabel("")
        _ax.set_xlabel(f"{_source.replace('_', ' ').title()} ({_well_area_mm2:.1f} mm2)", fontsize=8)
        _ax.tick_params(axis="both", which="both", length=0, labelsize=6)

    fig.subplots_adjust(bottom=0.06)
    _cbar_ax = fig.add_axes([0.4, 0.015, 0.2, 0.008])
    _sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=plt.Normalize(vmin=_vmin, vmax=_vmax))
    fig.colorbar(_sm, cax=_cbar_ax, orientation="horizontal").set_label(count_label, fontsize=9)

    return fig


@app.function
def plot_edge_marginals(
    viz_df: pd.DataFrame,
    count_col: str,
    count_label: str,
) -> plt.Figure:
    """Plot row and column marginal bar charts for cell counts per source."""
    _sources = sorted(viz_df["Metadata_Source"].unique(), key=lambda x: int(x.split("_")[1]))
    _n_sources = len(_sources)
    _vmin = viz_df[count_col].quantile(0.01)
    _vmax = viz_df[count_col].quantile(0.99)

    fig, axes = plt.subplots(_n_sources, 2, figsize=(8, 1.5 * _n_sources), gridspec_kw={"hspace": 0.1, "wspace": 0.15})
    if _n_sources == 1:
        axes = axes.reshape(1, -1)

    for _i, _source in enumerate(_sources):
        _source_df = viz_df[viz_df["Metadata_Source"] == _source]
        _src_rows = sorted(_source_df["row"].unique(), key=lambda x: (len(x), x))
        _well_area_mm2 = _source_df["Metadata_Well_Area_mm2"].iloc[0]

        _row_means = _source_df.groupby("row")[count_col].mean().reindex(_src_rows)
        _ax1 = axes[_i, 0]
        _ax1.barh(range(len(_row_means)), _row_means.values, color="#3498db", alpha=0.8, height=0.8)
        _ax1.set_yticks(range(0, len(_row_means), max(1, len(_row_means) // 8)))
        _ax1.set_yticklabels(
            [_src_rows[j] for j in range(0, len(_row_means), max(1, len(_row_means) // 8))], fontsize=6
        )
        _ax1.set_xlim(_vmin * 0.9, _vmax * 1.1)
        _ax1.invert_yaxis()
        _ax1.axvline(_row_means.mean(), color="red", linestyle="--", alpha=0.5, linewidth=1)
        _ax1.set_ylabel(f"{_source.replace('_', ' ').title()} ({_well_area_mm2:.1f} mm2)", fontsize=7)
        _ax1.tick_params(axis="x", labelsize=6)

        _col_means = _source_df.groupby("col")[count_col].mean().sort_index()
        _ax2 = axes[_i, 1]
        _ax2.bar(_col_means.index.values, _col_means.values, width=0.8, color="#3498db", alpha=0.8)
        _ax2.set_xlim(0, _col_means.index.max() + 1)
        _ax2.set_ylim(_vmin * 0.9, _vmax * 1.1)
        _ax2.axhline(_col_means.mean(), color="red", linestyle="--", alpha=0.5, linewidth=1)
        _ax2.tick_params(axis="both", labelsize=6)

        if _i < _n_sources - 1:
            _ax1.set_xticklabels([])
            _ax2.set_xticklabels([])
        else:
            _ax2.set_xticks(range(0, _col_means.index.max() + 1, 4))

    axes[-1, 0].set_xlabel(count_label, fontsize=8)
    axes[-1, 1].set_xlabel("Column", fontsize=8)
    axes[0, 0].set_title("Row marginals", fontsize=9)
    axes[0, 1].set_title("Column marginals", fontsize=9)
    fig.text(0.98, 0.5, count_label, va="center", rotation=-90, fontsize=8)

    return fig


@app.function
def plot_edge_violins(
    viz_df: pd.DataFrame,
    count_col: str,
    count_label: str,
) -> plt.Figure:
    """Plot row and column violin distributions for cell counts per source."""
    _sources = sorted(viz_df["Metadata_Source"].unique(), key=lambda x: int(x.split("_")[1]))
    _n_sources = len(_sources)
    _vmin = viz_df[count_col].quantile(0.01)
    _vmax = viz_df[count_col].quantile(0.99)

    fig, axes = plt.subplots(_n_sources, 2, figsize=(14, 2 * _n_sources), gridspec_kw={"hspace": 0.4, "wspace": 0.2})
    if _n_sources == 1:
        axes = axes.reshape(1, -1)

    for _i, _source in enumerate(_sources):
        _source_df = viz_df[viz_df["Metadata_Source"] == _source]
        _src_rows = sorted(_source_df["row"].unique(), key=lambda x: (len(x), x))
        _src_cols = sorted(_source_df["col"].unique())
        _well_area_mm2 = _source_df["Metadata_Well_Area_mm2"].iloc[0]

        for _ax, _x_var, _order in [
            (axes[_i, 0], "row", _src_rows),
            (axes[_i, 1], "col", _src_cols),
        ]:
            sns.violinplot(
                data=_source_df,
                x=_x_var,
                y=count_col,
                order=_order,
                ax=_ax,
                color="#3498db",
                linewidth=0.5,
                inner="box",
                density_norm="width",
            )
            _ax.axhline(_source_df[count_col].mean(), color="red", linestyle="--", alpha=0.7, linewidth=1)
            _ax.set_xlabel("")
            _ax.tick_params(axis="x", labelsize=6)
            _ax.tick_params(axis="y", labelsize=6)
            _ax.set_ylim(_vmin * 0.8, _vmax * 1.2)

        axes[_i, 0].set_ylabel(f"{_source.replace('_', ' ').title()}\n({_well_area_mm2:.1f} mm2)", fontsize=8)
        axes[_i, 1].set_ylabel("")

        if _i < _n_sources - 1:
            axes[_i, 0].set_xticklabels([])
            axes[_i, 1].set_xticklabels([])

    axes[-1, 0].set_xlabel("Row", fontsize=9)
    axes[-1, 1].set_xlabel("Column", fontsize=9)
    axes[0, 0].set_title("Row distribution", fontsize=10)
    axes[0, 1].set_title("Column distribution", fontsize=10)
    fig.text(0.02, 0.5, count_label, va="center", rotation=90, fontsize=9)

    return fig


@app.function
def prepare_edge_analysis(
    cell_count_df: pd.DataFrame,
    plate_format: int = 384,
    treatment_cols_only: bool = True,
) -> tuple[pd.DataFrame, str, str]:
    """Prepare cell count data for edge effects analysis.

    Returns (prepared_df, count_col, count_label).
    """
    _df = cell_count_df.copy()
    _positions = _df["Metadata_Well"].apply(parse_well_position)
    _df["row"] = _positions.apply(lambda x: x[0])
    _df["col"] = _positions.apply(lambda x: x[1])

    if treatment_cols_only and plate_format == 384:
        _control_cols = {1, 2, 23, 24}
        _n_before = len(_df)
        _df = _df[~_df["col"].isin(_control_cols)]
        logger.info(f"Filtered to treatment columns (3-22): {_n_before:,} -> {len(_df):,} wells")

    if "Metadata_Count_Cells_Per_mm2" in _df.columns:
        _count_col = "Metadata_Count_Cells_Per_mm2"
        _count_label = "Cell Density (cells/mm2)"
        logger.info("Using normalized cell counts (cells per mm2 of imaged area)")
    else:
        _count_col = "Metadata_Count_Cells"
        _count_label = "Cell Count"

    return _df, _count_col, _count_label


@app.function
def compute_edge_statistics(
    df: pd.DataFrame,
    count_col: str,
) -> dict:
    """Compute edge vs interior well statistics.

    Args:
        df: DataFrame with row, col, and count_col columns
        count_col: Column containing cell count/density values

    Returns:
        Dict of edge effect summary statistics.
    """
    _unique_rows = sorted(df["row"].unique(), key=lambda x: (len(x), x))
    _unique_cols = sorted(df["col"].unique())

    _edge_rows = set([_unique_rows[0], _unique_rows[-1]])
    _edge_cols = set([_unique_cols[0], _unique_cols[-1]])

    _is_edge = df["row"].isin(_edge_rows) | df["col"].isin(_edge_cols)

    _edge_mean = df[_is_edge][count_col].mean()
    _interior_mean = df[~_is_edge][count_col].mean()
    _edge_n = _is_edge.sum()
    _interior_n = (~_is_edge).sum()

    _t_stat, _t_pval = stats.ttest_ind(
        df[_is_edge][count_col],
        df[~_is_edge][count_col],
    )

    return {
        "n_wells": len(df),
        "n_rows": len(_unique_rows),
        "n_cols": len(_unique_cols),
        "edge_mean_cell_density": _edge_mean,
        "interior_mean_cell_density": _interior_mean,
        "edge_n_wells": _edge_n,
        "interior_n_wells": _interior_n,
        "edge_effect_pct": (_interior_mean - _edge_mean) / _interior_mean * 100,
        "ttest_statistic": _t_stat,
        "ttest_pvalue": _t_pval,
    }


# ---------------------------------------------------------------------------
# Guard: check databases exist
# ---------------------------------------------------------------------------


@app.cell
def _(
    activity_params_dropdown,
    dataset_dropdown,
    filter_dropdown,
    mo,
    preprocessing_dropdown,
):
    mo.stop(
        not COPAIRS_RESULTS_DB.exists(),
        mo.md(f"**Copairs database not found:** `{COPAIRS_RESULTS_DB}`"),
    )
    mo.stop(
        not METADATA_DB.exists(),
        mo.md(f"**Metadata database not found:** `{METADATA_DB}`"),
    )

    activity_df = query_activity_results(
        dataset_dropdown.value,
        preprocessing_dropdown.value,
        filter_dropdown.value,
        activity_params_dropdown.value,
    )
    mo.md(f"""
    ## Data loaded

    **Dataset:** {dataset_dropdown.value} | **Preprocessing:** {preprocessing_dropdown.value}
    | **Filter:** {filter_dropdown.value} | **Activity params:** {activity_params_dropdown.value}

    **Compounds with activity scores:** {len(activity_df):,}
    """)
    return (activity_df,)


# ===========================================================================
# Part 1: Cell Density vs Activity
# ===========================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Cell Density vs Phenotypic Activity

    Three progressively more rigorous analyses:

    1. **Raw** - Simple median cell density per compound (baseline, shows confounding)
    2. **Excluding source_2** - Remove outlier source to test if it drives the pattern
    3. **Normalized** - Regress out source effects, then robust z-score by plate
    """)
    return


# ---------------------------------------------------------------------------
# 1a. Raw cell density (all sources)
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 1a. Raw cell density (all sources)""")
    return


@app.cell
def _(activity_df):
    cell_density_raw = query_cell_density(aggregate=True)
    logger.info(f"Loaded {len(cell_density_raw):,} compound cell densities (all sources)")

    merged_raw, summary_raw = analyze_cell_density_vs_activity(activity_df, cell_density_raw)

    _stats = compute_activity_stats(merged_raw, "Metadata_median_cell_density")
    _annotation = format_stat_annotation(_stats["spearman_r"], _stats["mw_p"], n=_stats["n"])

    fig_hexbin_raw = plot_hexbin_with_marginals(
        merged_raw,
        x_col="Metadata_median_cell_density",
        x_label="Median cell density (cells/mm2)",
        stats_annotation=_annotation,
    )
    fig_hexbin_raw
    return (cell_density_raw, fig_hexbin_raw, merged_raw, summary_raw)


@app.cell
def _(merged_raw):
    fig_scatter_raw, _ = plot_scatter_with_marginals(
        merged_raw,
        x_col="Metadata_median_cell_density",
        x_label="Median cell density (cells/mm2)",
    )
    fig_scatter_raw
    return (fig_scatter_raw,)


# ---------------------------------------------------------------------------
# 1b. Excluding source_2 (outlier with ~3x higher cell density)
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 1b. Excluding source_2""")
    return


@app.cell
def _(activity_df):
    cell_density_excl = query_cell_density(exclude_sources=["source_2"], aggregate=True)
    logger.info(f"Loaded {len(cell_density_excl):,} compound cell densities (excluding source_2)")

    merged_excl, summary_excl = analyze_cell_density_vs_activity(activity_df, cell_density_excl)

    _stats = compute_activity_stats(merged_excl, "Metadata_median_cell_density")
    _annotation = format_stat_annotation(_stats["spearman_r"], _stats["mw_p"], n=_stats["n"])

    fig_hexbin_excl = plot_hexbin_with_marginals(
        merged_excl,
        x_col="Metadata_median_cell_density",
        x_label="Median cell density (cells/mm2)",
        stats_annotation=_annotation,
    )
    fig_hexbin_excl
    return (cell_density_excl, fig_hexbin_excl, merged_excl, summary_excl)


@app.cell
def _(merged_excl):
    fig_scatter_excl, _ = plot_scatter_with_marginals(
        merged_excl,
        x_col="Metadata_median_cell_density",
        x_label="Median cell density (cells/mm2)",
    )
    fig_scatter_excl
    return (fig_scatter_excl,)


# ---------------------------------------------------------------------------
# 1c. Source-adjusted, plate-normalized cell density
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1c. Source-adjusted, plate-normalized cell density

    Two-step normalization: regress out source effects, then robust z-score by plate.
    This removes systematic lab/protocol differences and plate-level batch effects.
    """)
    return


@app.cell
def _(activity_df):
    _well_level_df = query_cell_density(aggregate=False)
    logger.info(f"Loaded {len(_well_level_df):,} well-level cell densities")

    _well_level_df = normalize_cell_density(_well_level_df, value_col="cell_density")
    _norm_col = "cell_density_normalized"
    logger.info(
        f"Applied two-step normalization "
        f"(range: {_well_level_df[_norm_col].min():.2f} to {_well_level_df[_norm_col].max():.2f})"
    )

    cell_density_normalized = aggregate_compound_cell_density(
        _well_level_df, value_col=_norm_col, output_col="Metadata_median_cell_density"
    )
    logger.info(f"Aggregated to {len(cell_density_normalized):,} compounds (source-adjusted, plate-normalized)")

    merged_norm, summary_norm = analyze_cell_density_vs_activity(
        activity_df,
        cell_density_normalized,
        density_label="Normalized cell density (source-adjusted, plate z-score)",
    )

    _stats = compute_activity_stats(merged_norm, "Metadata_median_cell_density")
    _annotation = format_stat_annotation(_stats["spearman_r"], _stats["mw_p"], n=_stats["n"])

    fig_hexbin_norm = plot_hexbin_with_marginals(
        merged_norm,
        x_col="Metadata_median_cell_density",
        x_label="Normalized cell density (source-adjusted, plate z-score)",
        stats_annotation=_annotation,
    )
    fig_hexbin_norm
    return (cell_density_normalized, fig_hexbin_norm, merged_norm, summary_norm)


@app.cell
def _(merged_norm):
    fig_scatter_norm, _ = plot_scatter_with_marginals(
        merged_norm,
        x_col="Metadata_median_cell_density",
        x_label="Normalized cell density (source-adjusted, plate z-score)",
    )
    fig_scatter_norm
    return (fig_scatter_norm,)


# ---------------------------------------------------------------------------
# 1d. Cell density distribution comparison
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1d. Cell density distribution comparison

    Raw vs plate-normalized distributions. If the raw bimodality disappears
    after normalization, the two peaks are driven by plate/source batch effects
    rather than genuine compound-level toxicity differences.
    """)
    return


@app.cell
def _(cell_density_normalized, cell_density_raw):
    fig_dist = plot_cell_density_distribution(cell_density_raw, cell_density_normalized)
    fig_dist
    return (fig_dist,)


# ---------------------------------------------------------------------------
# Summary table for Part 1
# ---------------------------------------------------------------------------


@app.cell
def _(mo, summary_excl, summary_norm, summary_raw):
    mo.md(f"""
    ### Cell density vs activity summary

    | Variant | n compounds | Spearman rho | MW p-value | Median density (sig) | Median density (nonsig) |
    |---------|------------|-------------|-----------|---------------------|------------------------|
    | Raw (all sources) | {summary_raw["n_compounds"]:,} | {summary_raw["spearman_r"]:.3f} | {summary_raw["mannwhitney_p"]:.2e} | {summary_raw["median_cell_density_sig"]:.0f} | {summary_raw["median_cell_density_nonsig"]:.0f} |
    | Excl. source_2 | {summary_excl["n_compounds"]:,} | {summary_excl["spearman_r"]:.3f} | {summary_excl["mannwhitney_p"]:.2e} | {summary_excl["median_cell_density_sig"]:.0f} | {summary_excl["median_cell_density_nonsig"]:.0f} |
    | Normalized | {summary_norm["n_compounds"]:,} | {summary_norm["spearman_r"]:.3f} | {summary_norm["mannwhitney_p"]:.2e} | {summary_norm["median_cell_density_sig"]:.2f} | {summary_norm["median_cell_density_nonsig"]:.2f} |
    """)
    return


# ===========================================================================
# Part 2: Edge Well Effects
# ===========================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Edge Well Effects

    Do wells at the plate perimeter show systematically different cell counts?
    Heatmaps, row/column marginals, and violins for each source, with a
    statistical edge-vs-interior comparison on 384-well plates.
    """)
    return


@app.cell
def _():
    well_cell_counts_all = query_cell_counts(plate_format=0, plate_type="COMPOUND")
    logger.info(f"Loaded {len(well_cell_counts_all):,} well-level cell counts from all COMPOUND plates")

    well_cell_counts_384 = query_cell_counts(plate_format=384, plate_type="COMPOUND")
    logger.info(f"Loaded {len(well_cell_counts_384):,} well-level cell counts from 384-well COMPOUND plates")

    # Prepare the 384-well data for edge statistics
    edge_df, count_col, count_label = prepare_edge_analysis(
        well_cell_counts_384, plate_format=384, treatment_cols_only=False
    )
    return (count_col, count_label, edge_df, well_cell_counts_all, well_cell_counts_384)


@app.cell
def _(edge_df, count_col, mo):
    edge_stats = compute_edge_statistics(edge_df, count_col)

    logger.info(f"Edge wells: mean={edge_stats['edge_mean_cell_density']:.1f}, n={edge_stats['edge_n_wells']:,}")
    logger.info(
        f"Interior wells: mean={edge_stats['interior_mean_cell_density']:.1f}, n={edge_stats['interior_n_wells']:,}"
    )
    logger.info(
        f"Edge effect: {edge_stats['edge_effect_pct']:.1f}% fewer cells on edges (p={edge_stats['ttest_pvalue']:.2e})"
    )

    mo.md(f"""
    ### Edge vs interior statistics (384-well plates)

    | Metric | Value |
    |--------|-------|
    | Total wells | {edge_stats["n_wells"]:,} |
    | Grid | {edge_stats["n_rows"]} rows x {edge_stats["n_cols"]} cols |
    | Edge mean | {edge_stats["edge_mean_cell_density"]:.1f} |
    | Interior mean | {edge_stats["interior_mean_cell_density"]:.1f} |
    | Edge effect | {edge_stats["edge_effect_pct"]:.1f}% fewer cells on edges |
    | t-test p-value | {edge_stats["ttest_pvalue"]:.2e} |
    """)
    return (edge_stats,)


# ---------------------------------------------------------------------------
# Edge effects: all sources heatmap
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Per-source cell count heatmaps (all sources)""")
    return


@app.cell
def _(count_col, count_label, well_cell_counts_all):
    # Prepare all-sources data for visualization
    _viz_all = well_cell_counts_all.copy()
    _positions = _viz_all["Metadata_Well"].apply(parse_well_position)
    _viz_all["row"] = _positions.apply(lambda x: x[0])
    _viz_all["col"] = _positions.apply(lambda x: x[1])

    fig_heatmap_all = plot_edge_heatmaps(_viz_all, count_col, count_label)
    fig_heatmap_all
    return (fig_heatmap_all,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Row and column marginals (all sources)""")
    return


@app.cell
def _(count_col, count_label, well_cell_counts_all):
    _viz_all = well_cell_counts_all.copy()
    _positions = _viz_all["Metadata_Well"].apply(parse_well_position)
    _viz_all["row"] = _positions.apply(lambda x: x[0])
    _viz_all["col"] = _positions.apply(lambda x: x[1])

    fig_marginals_all = plot_edge_marginals(_viz_all, count_col, count_label)
    fig_marginals_all
    return (fig_marginals_all,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Violin distributions (all sources)""")
    return


@app.cell
def _(count_col, count_label, well_cell_counts_all):
    _viz_all = well_cell_counts_all.copy()
    _positions = _viz_all["Metadata_Well"].apply(parse_well_position)
    _viz_all["row"] = _positions.apply(lambda x: x[0])
    _viz_all["col"] = _positions.apply(lambda x: x[1])

    fig_violin_all = plot_edge_violins(_viz_all, count_col, count_label)
    fig_violin_all
    return (fig_violin_all,)


# ---------------------------------------------------------------------------
# Edge effects: excluding source_2
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Per-source cell count heatmaps (excluding source_2)""")
    return


@app.cell
def _(count_col, count_label, well_cell_counts_all):
    _viz_excl = well_cell_counts_all[well_cell_counts_all["Metadata_Source"] != "source_2"].copy()
    _positions = _viz_excl["Metadata_Well"].apply(parse_well_position)
    _viz_excl["row"] = _positions.apply(lambda x: x[0])
    _viz_excl["col"] = _positions.apply(lambda x: x[1])

    fig_heatmap_excl = plot_edge_heatmaps(_viz_excl, count_col, count_label)
    fig_heatmap_excl
    return (fig_heatmap_excl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Row and column marginals (excluding source_2)""")
    return


@app.cell
def _(count_col, count_label, well_cell_counts_all):
    _viz_excl = well_cell_counts_all[well_cell_counts_all["Metadata_Source"] != "source_2"].copy()
    _positions = _viz_excl["Metadata_Well"].apply(parse_well_position)
    _viz_excl["row"] = _positions.apply(lambda x: x[0])
    _viz_excl["col"] = _positions.apply(lambda x: x[1])

    fig_marginals_excl = plot_edge_marginals(_viz_excl, count_col, count_label)
    fig_marginals_excl
    return (fig_marginals_excl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Violin distributions (excluding source_2)""")
    return


@app.cell
def _(count_col, count_label, well_cell_counts_all):
    _viz_excl = well_cell_counts_all[well_cell_counts_all["Metadata_Source"] != "source_2"].copy()
    _positions = _viz_excl["Metadata_Well"].apply(parse_well_position)
    _viz_excl["row"] = _positions.apply(lambda x: x[0])
    _viz_excl["col"] = _positions.apply(lambda x: x[1])

    fig_violin_excl = plot_edge_violins(_viz_excl, count_col, count_label)
    fig_violin_excl
    return (fig_violin_excl,)


# ===========================================================================
# Save outputs
# ===========================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save outputs

    Saves all figures and summary CSVs to
    `data/processed/data-quality/{dataset}/{preprocessing}/`.
    """)
    return


@app.cell
def _(
    count_col,
    dataset_dropdown,
    edge_stats,
    fig_dist,
    fig_heatmap_all,
    fig_heatmap_excl,
    fig_hexbin_excl,
    fig_hexbin_norm,
    fig_hexbin_raw,
    fig_marginals_all,
    fig_marginals_excl,
    fig_scatter_excl,
    fig_scatter_norm,
    fig_scatter_raw,
    fig_violin_all,
    fig_violin_excl,
    mo,
    preprocessing_dropdown,
    summary_excl,
    summary_norm,
    summary_raw,
):
    _base_dir = OUTPUT_DIR / dataset_dropdown.value / preprocessing_dropdown.value

    # Cell density vs activity outputs
    _density_dir = _base_dir / "cell_density_activity"
    _density_dir.mkdir(parents=True, exist_ok=True)

    fig_hexbin_raw.savefig(_density_dir / "cell_density_vs_map_hexbin.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_scatter_raw.savefig(_density_dir / "cell_density_vs_map_scatter.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_hexbin_excl.savefig(
        _density_dir / "cell_density_vs_map_hexbin_excl_source2.png", dpi=DEFAULT_DPI, bbox_inches="tight"
    )
    fig_scatter_excl.savefig(
        _density_dir / "cell_density_vs_map_scatter_excl_source2.png", dpi=DEFAULT_DPI, bbox_inches="tight"
    )
    fig_hexbin_norm.savefig(
        _density_dir / "cell_density_vs_map_hexbin_normalized.png", dpi=DEFAULT_DPI, bbox_inches="tight"
    )
    fig_scatter_norm.savefig(
        _density_dir / "cell_density_vs_map_scatter_normalized.png", dpi=DEFAULT_DPI, bbox_inches="tight"
    )
    fig_dist.savefig(_density_dir / "cell_density_distribution_comparison.png", dpi=DEFAULT_DPI, bbox_inches="tight")

    # Save summary CSVs for each variant
    pd.DataFrame([summary_raw]).to_csv(_density_dir / "summary.csv", index=False)
    pd.DataFrame([summary_excl]).to_csv(_density_dir / "summary_excl_source2.csv", index=False)
    pd.DataFrame([summary_norm]).to_csv(_density_dir / "summary_normalized.csv", index=False)

    # Edge effects outputs
    _edge_dir = _base_dir / "edge_effects"
    _edge_dir.mkdir(parents=True, exist_ok=True)

    fig_heatmap_all.savefig(_edge_dir / "cell_count_heatmap_by_source.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_marginals_all.savefig(_edge_dir / "position_marginals.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_violin_all.savefig(_edge_dir / "position_marginals_violin.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_heatmap_excl.savefig(
        _edge_dir / "cell_count_heatmap_by_source_excl_source2.png", dpi=DEFAULT_DPI, bbox_inches="tight"
    )
    fig_marginals_excl.savefig(_edge_dir / "position_marginals_excl_source2.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_violin_excl.savefig(
        _edge_dir / "position_marginals_violin_excl_source2.png", dpi=DEFAULT_DPI, bbox_inches="tight"
    )

    # Edge statistics summary
    _edge_summary = {**edge_stats, "count_col": count_col}
    pd.DataFrame([_edge_summary]).to_csv(_edge_dir / "summary.csv", index=False)

    logger.info(f"Saved all outputs to {_base_dir}")

    mo.md(f"""
    **Saved outputs to** `{_base_dir}`

    **Cell density vs activity** (`cell_density_activity/`):
    - `cell_density_vs_map_hexbin.png`, `cell_density_vs_map_scatter.png` (raw)
    - `*_excl_source2.png` (excluding source_2)
    - `*_normalized.png` (source-adjusted, plate-normalized)
    - `cell_density_distribution_comparison.png`
    - `summary.csv`, `summary_excl_source2.csv`, `summary_normalized.csv`

    **Edge effects** (`edge_effects/`):
    - `cell_count_heatmap_by_source.png`, `position_marginals.png`, `position_marginals_violin.png`
    - `*_excl_source2.png` variants
    - `summary.csv`
    """)
    return


@app.function
def run_data_quality(
    dataset: str = "compound_no_source7",
    preprocessing: str = "activity_no_target2",
    filter_name: str = "all_sources",
    activity_params: str = "default",
    output_dir=None,
) -> str:
    """Run the full data quality analysis: cell density vs activity and edge effects.

    Produces all plots and summary CSVs that the Snakemake data_quality rule did.

    Called from workflow.py. Returns the base output directory path.

    Args:
        dataset: e.g. "compound_no_source7"
        preprocessing: e.g. "activity_no_target2"
        filter_name: e.g. "all_sources"
        activity_params: e.g. "default"
        output_dir: Override base output directory
    """
    import matplotlib.pyplot as plt

    from nb00_ss_config import DEFAULT_DPI, PROCESSED_DATA_DIR
    from nb04_ss_visualization import (
        compute_activity_stats,
        format_stat_annotation,
        plot_hexbin_with_marginals,
        plot_scatter_with_marginals,
    )

    if output_dir is None:
        _base_dir = PROCESSED_DATA_DIR / "data-quality" / dataset / preprocessing
    else:
        _base_dir = Path(output_dir)
    _density_dir = _base_dir / "cell_density_activity"
    _edge_dir = _base_dir / "edge_effects"
    _density_dir.mkdir(parents=True, exist_ok=True)
    _edge_dir.mkdir(parents=True, exist_ok=True)

    # Load activity data
    _activity_df = query_activity_results(dataset, preprocessing, filter_name, activity_params)

    # -----------------------------------------------------------------------
    # Part 1: Cell density vs activity (3 variants)
    # -----------------------------------------------------------------------

    # 1a. Raw cell density (all sources)
    _cd_raw = query_cell_density(aggregate=True)
    _merged_raw, _summary_raw = analyze_cell_density_vs_activity(_activity_df, _cd_raw)
    _stats_raw = compute_activity_stats(_merged_raw, "Metadata_median_cell_density")
    _ann_raw = format_stat_annotation(_stats_raw["spearman_r"], _stats_raw["mw_p"], n=_stats_raw["n"])

    _fig = plot_hexbin_with_marginals(
        _merged_raw,
        x_col="Metadata_median_cell_density",
        x_label="Median cell density (cells/mm2)",
        stats_annotation=_ann_raw,
    )
    _fig.savefig(_density_dir / "cell_density_vs_map_hexbin.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(_fig)

    _fig, _ = plot_scatter_with_marginals(
        _merged_raw,
        x_col="Metadata_median_cell_density",
        x_label="Median cell density (cells/mm2)",
    )
    _fig.savefig(_density_dir / "cell_density_vs_map_scatter.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(_fig)

    # 1b. Excluding source_2
    _cd_excl = query_cell_density(exclude_sources=["source_2"], aggregate=True)
    _merged_excl, _summary_excl = analyze_cell_density_vs_activity(_activity_df, _cd_excl)
    _stats_excl = compute_activity_stats(_merged_excl, "Metadata_median_cell_density")
    _ann_excl = format_stat_annotation(_stats_excl["spearman_r"], _stats_excl["mw_p"], n=_stats_excl["n"])

    _fig = plot_hexbin_with_marginals(
        _merged_excl,
        x_col="Metadata_median_cell_density",
        x_label="Median cell density (cells/mm2)",
        stats_annotation=_ann_excl,
    )
    _fig.savefig(_density_dir / "cell_density_vs_map_hexbin_excl_source2.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(_fig)

    _fig, _ = plot_scatter_with_marginals(
        _merged_excl,
        x_col="Metadata_median_cell_density",
        x_label="Median cell density (cells/mm2)",
    )
    _fig.savefig(_density_dir / "cell_density_vs_map_scatter_excl_source2.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(_fig)

    # 1c. Normalized (source-adjusted, plate z-score)
    _well_df = query_cell_density(aggregate=False)
    _well_df = normalize_cell_density(_well_df, value_col="cell_density")
    _cd_norm = aggregate_compound_cell_density(
        _well_df, value_col="cell_density_normalized", output_col="Metadata_median_cell_density"
    )
    _merged_norm, _summary_norm = analyze_cell_density_vs_activity(
        _activity_df,
        _cd_norm,
        density_label="Normalized cell density (source-adjusted, plate z-score)",
    )
    _stats_norm = compute_activity_stats(_merged_norm, "Metadata_median_cell_density")
    _ann_norm = format_stat_annotation(_stats_norm["spearman_r"], _stats_norm["mw_p"], n=_stats_norm["n"])

    _fig = plot_hexbin_with_marginals(
        _merged_norm,
        x_col="Metadata_median_cell_density",
        x_label="Normalized cell density (source-adjusted, plate z-score)",
        stats_annotation=_ann_norm,
    )
    _fig.savefig(_density_dir / "cell_density_vs_map_hexbin_normalized.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(_fig)

    _fig, _ = plot_scatter_with_marginals(
        _merged_norm,
        x_col="Metadata_median_cell_density",
        x_label="Normalized cell density (source-adjusted, plate z-score)",
    )
    _fig.savefig(_density_dir / "cell_density_vs_map_scatter_normalized.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(_fig)

    # 1d. Distribution comparison
    _fig_dist = plot_cell_density_distribution(_cd_raw, _cd_norm)
    _fig_dist.savefig(_density_dir / "cell_density_distribution_comparison.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(_fig_dist)

    # Save summary CSVs
    pd.DataFrame([_summary_raw]).to_csv(_density_dir / "summary.csv", index=False)
    pd.DataFrame([_summary_excl]).to_csv(_density_dir / "summary_excl_source2.csv", index=False)
    pd.DataFrame([_summary_norm]).to_csv(_density_dir / "summary_normalized.csv", index=False)

    # -----------------------------------------------------------------------
    # Part 2: Edge well effects
    # -----------------------------------------------------------------------

    _cc_all = query_cell_counts(plate_format=0, plate_type="COMPOUND")
    _cc_384 = query_cell_counts(plate_format=384, plate_type="COMPOUND")

    _edge_df, _count_col, _count_label = prepare_edge_analysis(_cc_384, plate_format=384, treatment_cols_only=False)

    # Edge statistics
    _edge_stats = compute_edge_statistics(_edge_df, _count_col)

    # Prepare all-sources visualization data
    _viz_all = _cc_all.copy()
    _positions = _viz_all["Metadata_Well"].apply(parse_well_position)
    _viz_all["row"] = _positions.apply(lambda x: x[0])
    _viz_all["col"] = _positions.apply(lambda x: x[1])

    for _plot_fn, _fname in [
        (plot_edge_heatmaps, "cell_count_heatmap_by_source.png"),
        (plot_edge_marginals, "position_marginals.png"),
        (plot_edge_violins, "position_marginals_violin.png"),
    ]:
        _fig = _plot_fn(_viz_all, _count_col, _count_label)
        _fig.savefig(_edge_dir / _fname, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(_fig)

    # Excluding source_2
    _viz_excl = _viz_all[_viz_all["Metadata_Source"] != "source_2"].copy()
    for _plot_fn, _fname in [
        (plot_edge_heatmaps, "cell_count_heatmap_by_source_excl_source2.png"),
        (plot_edge_marginals, "position_marginals_excl_source2.png"),
        (plot_edge_violins, "position_marginals_violin_excl_source2.png"),
    ]:
        _fig = _plot_fn(_viz_excl, _count_col, _count_label)
        _fig.savefig(_edge_dir / _fname, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(_fig)

    # Edge statistics CSV
    pd.DataFrame([{**_edge_stats, "count_col": _count_col}]).to_csv(_edge_dir / "summary.csv", index=False)

    # Touch .complete marker
    (_base_dir / ".complete").touch()
    logger.success(f"Saved all data quality outputs to {_base_dir}")
    return str(_base_dir)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
