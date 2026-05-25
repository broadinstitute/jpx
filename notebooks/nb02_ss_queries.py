# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb==1.5.2",
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
    import pandas as pd

    from nb00_ss_config import COPAIRS_RESULTS_DB


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Queries

    Foundation notebook providing DuckDB query functions for copairs results and metadata.
    Other notebooks import via `from nb02_ss_queries import query_activity_results, ...`

    **Exported functions:**
    - `query_activity_results()` - compound phenotypic activity (nMAP, p-values)
    - `query_cross_source_reproducibility()` - within-source vs cross-source activity
    - `query_consistency_results()` - target-group consistency clustering

    Always use `mean_normalized_average_precision` (not raw mAP)
    and `corrected_p_value` (not raw p-value).
    """)
    return


@app.function
def query_activity_results(
    dataset: str,
    preprocessing: str = "activity_no_target2",
    filter_name: str = "all_sources",
    activity_params: str = "default",
) -> pd.DataFrame:
    """Query activity results from copairs_results.duckdb."""
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    query = """
        SELECT
            Metadata_JCP2022,
            mean_average_precision,
            mean_normalized_average_precision,
            p_value,
            corrected_p_value,
            below_p,
            below_corrected_p
        FROM activity_results
        WHERE _dataset = ?
          AND _preprocessing = ?
          AND _filter = ?
          AND _activity_params = ?
        ORDER BY mean_normalized_average_precision DESC
    """
    df = con.execute(query, [dataset, preprocessing, filter_name, activity_params]).df()
    con.close()
    return df


@app.function
def query_cross_source_reproducibility(
    dataset: str,
    preprocessing: str,
    filter_name: str = "all_sources",
) -> pd.DataFrame:
    """Query within-source vs cross-source activity for reproducibility analysis.

    NOT "consistency" (shared biology clustering) - this measures
    reproducibility of the same compound across different lab sites.
    """
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    query = """
        WITH within_src AS (
            SELECT
                Metadata_JCP2022,
                Metadata_Source,
                mean_normalized_average_precision as nmAP_within,
                corrected_p_value as p_within,
                below_corrected_p as sig_within
            FROM activity_results
            WHERE _dataset = ?
              AND _activity_params = 'withinsource'
              AND _preprocessing = ?
              AND _filter = ?
        ),
        cross_src AS (
            SELECT
                Metadata_JCP2022,
                Metadata_Source,
                mean_normalized_average_precision as nmAP_cross,
                corrected_p_value as p_cross,
                below_corrected_p as sig_cross
            FROM activity_results
            WHERE _dataset = ?
              AND _activity_params = 'crosssource'
              AND _preprocessing = ?
              AND _filter = ?
        )
        SELECT
            w.Metadata_JCP2022,
            w.Metadata_Source,
            w.nmAP_within,
            w.p_within,
            w.sig_within,
            c.nmAP_cross,
            c.p_cross,
            c.sig_cross
        FROM within_src w
        JOIN cross_src c ON w.Metadata_JCP2022 = c.Metadata_JCP2022
                        AND w.Metadata_Source = c.Metadata_Source
        ORDER BY w.Metadata_JCP2022, w.Metadata_Source
    """
    df = con.execute(query, [dataset, preprocessing, filter_name, dataset, preprocessing, filter_name]).df()
    con.close()
    return df


@app.function
def query_consistency_results(
    dataset: str,
    preprocessing: str = "consistency_no_target2",
    filter_name: str = "all_sources",
    group_type: str = "repurposing",
    distance: str = "cosine",
) -> pd.DataFrame:
    """Query consistency results, excluding threshold sweep runs."""
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    query = """
        SELECT
            group_value,
            mean_average_precision,
            mean_normalized_average_precision,
            p_value,
            corrected_p_value,
            below_p,
            below_corrected_p,
            n_perturbations
        FROM consistency_results
        WHERE _dataset = ?
          AND _preprocessing = ?
          AND _filter = ?
          AND _group_type = ?
          AND _distance = ?
          AND _preprocessing NOT LIKE '%_sweep'
        ORDER BY mean_normalized_average_precision DESC
    """
    df = con.execute(query, [dataset, preprocessing, filter_name, group_type, distance]).df()
    con.close()
    return df


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


@app.cell
def _(dataset_dropdown, mo):
    df_activity = query_activity_results(dataset_dropdown.value)
    mo.md(f"""
    ## Activity results preview

    **Dataset:** {dataset_dropdown.value}
    **Rows:** {len(df_activity):,}
    **Significant (corrected p < 0.05):** {df_activity["below_corrected_p"].sum():,}
    """)
    return (df_activity,)


@app.cell
def _(df_activity, mo):
    mo.ui.dataframe(df_activity.head(20))
    return


@app.cell
def _(dataset_dropdown, mo):
    df_consistency = query_consistency_results(dataset_dropdown.value)
    mo.md(f"""
    ## Consistency results preview

    **Dataset:** {dataset_dropdown.value} | **Group type:** repurposing
    **Rows:** {len(df_consistency):,}
    **Significant:** {df_consistency["below_corrected_p"].sum():,}
    """)
    return (df_consistency,)


@app.cell
def _(df_consistency, mo):
    mo.ui.dataframe(df_consistency.head(20))
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
