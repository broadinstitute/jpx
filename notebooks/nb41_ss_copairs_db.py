# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb==1.5.2",
#     "omegaconf==2.3.0",
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

    import duckdb
    import pandas as pd
    from loguru import logger
    from omegaconf import OmegaConf

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import COPAIRS_RESULTS_DB, PROCESSED_DATA_DIR

    COPAIRS_RUNS_DIR = PROCESSED_DATA_DIR / "copairs/runs"

    # Columns that can have purely numeric values but should be strings
    CSV_DTYPES = {
        "Metadata_Plate": "str",
        "Metadata_Batch": "str",
    }

    # Table documentation
    TABLE_COMMENTS = {
        "activity_scores": "Per-replicate Average Precision scores for phenotypic activity measurement",
        "activity_results": "Per-perturbation mean Average Precision (mAP) for phenotypic activity measurement",
        "consistency_scores": "Per-perturbation-per-group Average Precision scores for phenotypic consistency measurement",
        "consistency_results": (
            "Per-group mean Average Precision (mAP) for phenotypic consistency measurement. "
            "Contains both regular runs and threshold sweep runs. "
            "Filter with: _preprocessing NOT LIKE '%_sweep' for regular runs only, "
            "or _preprocessing LIKE '%_sweep' for sweep runs only."
        ),
    }

    # Column documentation
    COLUMN_COMMENTS = {
        # Config metadata columns (from Hydra)
        "_dataset": "Dataset config name (e.g., 'compound_no_source7')",
        "_columns": "Feature columns config name (e.g., 'feat_all')",
        "_preprocessing": "Preprocessing config name (e.g., 'activity_no_target2')",
        "_filter": "Filter config name (e.g., 'all_sources', 'source_2')",
        "_activity_params": "Activity parameters config (e.g., 'default', 'withinsource', 'crosssource')",
        "_distance": "Distance metric for similarity computation (e.g., 'cosine', 'abs_cosine')",
        "_activity_threshold": (
            "P-value threshold used for activity filtering. "
            "To distinguish sweep vs regular runs, check _preprocessing: "
            "names ending in '_sweep' are threshold sweep runs."
        ),
        # Group columns (consistency only)
        "_group_type": "Group type config name (e.g., 'repurposing', 'uniprot', 'chemical_probes')",
        "_group_column": "Original column name in source data (e.g., 'Metadata_repurposing_target')",
        "group_value": "Normalized group value extracted from the group-specific column",
        # Copairs output columns
        "Metadata_JCP2022": "JUMP perturbation ID",
        "mean_average_precision": "Mean AP score for each group",
        "mean_normalized_average_precision": "Mean normalized AP score (scale-independent)",
        "p_value": "P-value comparing mAP to the null distribution",
        "corrected_p_value": "Adjusted p-value after multiple testing correction (FDR, Benjamini-Hochberg)",
        "below_p": "True if p_value < threshold",
        "below_corrected_p": "True if corrected_p_value < threshold (use this for significance)",
        "-log10(p-value)": "Negative log10 of p-value for visualization",
        "indices": "List of row indices used in the mAP aggregation",
        "average_precision": "Calculated average precision score for each profile",
        "normalized_average_precision": "Normalized AP score (scale-independent)",
        "n_pos_pairs": "Number of positive pairs for each profile",
        "n_total_pairs": "Total number of pairs (positive + negative) for each profile",
        "n_perturbations": "Number of perturbations in this group (= n_pos_pairs + 1 from scores)",
    }


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Copairs Results Database Builder

    Scans all copairs run directories under `data/processed/copairs/runs/`,
    reads Hydra configs (`.hydra/config.yaml`) for run metadata, and combines
    activity/consistency results CSVs into a unified DuckDB with documented tables.

    **Exported function:**
    - `build_results_db(output_path)` - build the copairs results DuckDB

    **Tables created:**
    - `activity_results` - per-perturbation mAP for phenotypic activity
    - `activity_scores` - per-replicate AP scores for activity
    - `consistency_results` - per-group mAP for phenotypic consistency
    - `consistency_scores` - per-perturbation-per-group AP scores

    Migrated from `src/jump_production/processing/build_copairs_results_db.py`.
    """)
    return


@app.function
def _add_config_metadata(df: pd.DataFrame, config: OmegaConf, result_type: str) -> pd.DataFrame:
    """Add config metadata columns to a DataFrame from Hydra config.

    Extracts dataset, columns, preprocessing, filter, activity_params,
    group info (consistency), distance metric, and activity threshold
    from the OmegaConf config object and adds them as underscore-prefixed columns.
    """
    df["_dataset"] = config.dataset.name
    df["_columns"] = config.columns.name
    df["_preprocessing"] = config.preprocessing_02_core.name
    df["_filter"] = config.filter.name

    # Activity params (activity runs only)
    if hasattr(config, "activity_params"):
        df["_activity_params"] = config.activity_params.name

    # Group info (consistency runs only)
    # "target_column" is the Hydra config key; stored as _group_* for genericity
    if hasattr(config, "target_column"):
        df["_group_type"] = config.target_column.name
        df["_group_column"] = config.target_column.column
        group_col = config.target_column.column
        if group_col in df.columns:
            df["group_value"] = df[group_col]

    # Distance metric from average_precision config (consistency only)
    if hasattr(config, "average_precision") and hasattr(config.average_precision, "params"):
        if "distance" in config.average_precision.params:
            df["_distance"] = config.average_precision.params.distance

    # Activity threshold from preprocessing steps (consistency only)
    if hasattr(config, "preprocessing_02_core") and hasattr(config.preprocessing_02_core, "steps"):
        for step in config.preprocessing_02_core.steps:
            if step.get("type") == "filter_active":
                threshold = step.get("params", {}).get("threshold")
                if threshold is not None:
                    df["_activity_threshold"] = threshold
                break

    return df


@app.function
def _collect_results(result_type: str) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Collect all runs of a given result type with both map_results and ap_scores.

    Traverses `COPAIRS_RUNS_DIR/<result_type>/*/results/.hydra/config.yaml`,
    loads the corresponding CSVs, and attaches Hydra config metadata.

    Returns:
        Tuple of (map_results DataFrames, ap_scores DataFrames)
    """
    runs_dir = COPAIRS_RUNS_DIR / result_type
    if not runs_dir.exists():
        logger.warning(f"No {result_type} runs directory found: {runs_dir}")
        return [], []

    map_results_dfs = []
    ap_scores_dfs = []

    for config_path in runs_dir.glob("*/results/.hydra/config.yaml"):
        run_dir = config_path.parent.parent
        map_results_file = run_dir / f"{result_type}_map_results.csv"
        ap_scores_file = run_dir / f"{result_type}_ap_scores.csv"

        config = OmegaConf.load(config_path)
        run_name = run_dir.name

        # Load map_results (aggregated with p-values)
        if map_results_file.exists():
            df = pd.read_csv(map_results_file, dtype=CSV_DTYPES)
            df = _add_config_metadata(df, config, result_type)

            # Compute n_perturbations from indices for consistency results
            if result_type == "consistency" and "indices" in df.columns:
                df["n_perturbations"] = df["indices"].apply(lambda x: 0 if x == "[]" else x.count(",") + 1)

            map_results_dfs.append(df)
            logger.info(f"Loaded {len(df):,} map_results rows from {run_name}")
        else:
            logger.warning(f"Missing map_results: {map_results_file}")

        # Load ap_scores (per-entity scores)
        if ap_scores_file.exists():
            df = pd.read_csv(ap_scores_file, dtype=CSV_DTYPES)
            df = _add_config_metadata(df, config, result_type)
            ap_scores_dfs.append(df)
            logger.info(f"Loaded {len(df):,} ap_scores rows from {run_name}")
        else:
            logger.warning(f"Missing ap_scores: {ap_scores_file}")

    return map_results_dfs, ap_scores_dfs


@app.function
def _add_comments(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
    """Add COMMENT ON TABLE and COMMENT ON COLUMN statements to DuckDB."""
    if table_name in TABLE_COMMENTS:
        comment = TABLE_COMMENTS[table_name].replace("'", "''")
        con.execute(f"COMMENT ON TABLE {table_name} IS '{comment}'")

    columns = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]
    for col in columns:
        if col in COLUMN_COMMENTS:
            comment = COLUMN_COMMENTS[col].replace("'", "''")
            col_quoted = f'"{col}"' if not col.isidentifier() else col
            con.execute(f"COMMENT ON COLUMN {table_name}.{col_quoted} IS '{comment}'")


@app.function
def _build_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    dfs: list[pd.DataFrame],
) -> None:
    """Build a DuckDB table from collected DataFrames.

    Concatenates all DataFrames, creates the table, and adds documentation comments.
    """
    if not dfs:
        logger.warning(f"No data to build {table_name} table")
        return

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(combined):,} total rows for {table_name}")

    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM combined")
    _add_comments(con, table_name)

    logger.success(f"Created {table_name} with {len(combined):,} rows")


@app.function
def build_results_db(output_path: Path | None = None) -> Path:
    """Scan data/processed/copairs/runs/, aggregate all activity + consistency CSVs into DuckDB.

    Discovers result types from the directory structure (activity, consistency),
    reads Hydra configs for run metadata, combines all CSVs, and writes a
    documented DuckDB database.

    Args:
        output_path: Output DuckDB path. Defaults to COPAIRS_RESULTS_DB.

    Returns:
        Path to the created DuckDB file.
    """
    if output_path is None:
        output_path = COPAIRS_RESULTS_DB

    output_path = Path(output_path)

    # Discover result types from directory structure
    result_types = []
    if COPAIRS_RUNS_DIR.exists():
        result_types = [d.name for d in COPAIRS_RUNS_DIR.iterdir() if d.is_dir()]

    if not result_types:
        msg = f"No result types found in {COPAIRS_RUNS_DIR}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Found result types: {result_types}")

    # Remove existing DB and create fresh
    output_path.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(output_path))

    for result_type in result_types:
        logger.info(f"Processing {result_type} runs...")
        map_results_dfs, ap_scores_dfs = _collect_results(result_type)

        if map_results_dfs:
            _build_table(con, f"{result_type}_results", map_results_dfs)

        if ap_scores_dfs:
            _build_table(con, f"{result_type}_scores", ap_scores_dfs)

    # Summary
    tables = con.execute("SHOW TABLES").fetchall()
    logger.info("Database summary:")
    for (table_name,) in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        cols = con.execute(f"SELECT COUNT(*) FROM duckdb_columns() WHERE table_name = '{table_name}'").fetchone()[0]
        logger.info(f"  {table_name}: {count:,} rows x {cols} columns")

    con.close()
    logger.success(f"Wrote {output_path}")
    return output_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: inspect the existing copairs results database

    Shows tables, row counts, and sample rows from the current database
    (if it exists). To rebuild, call `build_results_db()` in a cell.
    """)
    return


@app.cell
def _():
    # Check if the database exists and show summary
    if COPAIRS_RESULTS_DB.exists():
        _con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
        db_summary = _con.execute("""
            SELECT
                table_name,
                (SELECT COUNT(*) FROM query_table(table_name)) as row_count,
                column_count
            FROM duckdb_tables()
            ORDER BY table_name
        """).df()
        _con.close()
        db_summary
    else:
        logger.warning(f"Database not found: {COPAIRS_RESULTS_DB}")
        db_summary = pd.DataFrame(columns=["table_name", "row_count", "column_count"])
        db_summary
    return (db_summary,)


@app.cell
def _():
    # Show sample rows from activity_results (if available)
    if COPAIRS_RESULTS_DB.exists():
        _con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
        _tables = [t[0] for t in _con.execute("SHOW TABLES").fetchall()]
        if "activity_results" in _tables:
            activity_sample = _con.execute("""
                SELECT
                    Metadata_JCP2022,
                    _dataset,
                    _preprocessing,
                    _filter,
                    _activity_params,
                    mean_normalized_average_precision,
                    corrected_p_value,
                    below_corrected_p
                FROM activity_results
                ORDER BY mean_normalized_average_precision DESC
                LIMIT 10
            """).df()
        else:
            activity_sample = pd.DataFrame()
        _con.close()
        activity_sample
    else:
        activity_sample = pd.DataFrame()
        activity_sample
    return (activity_sample,)


@app.cell
def _():
    # Show distinct run configurations
    if COPAIRS_RESULTS_DB.exists():
        _con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
        _tables = [t[0] for t in _con.execute("SHOW TABLES").fetchall()]
        if "activity_results" in _tables:
            activity_configs = _con.execute("""
                SELECT
                    _dataset,
                    _preprocessing,
                    _filter,
                    _activity_params,
                    COUNT(*) as n_rows
                FROM activity_results
                GROUP BY ALL
                ORDER BY _dataset, _preprocessing
            """).df()
        else:
            activity_configs = pd.DataFrame()

        if "consistency_results" in _tables:
            consistency_configs = _con.execute("""
                SELECT
                    _dataset,
                    _preprocessing,
                    _filter,
                    _group_type,
                    _distance,
                    COUNT(*) as n_rows
                FROM consistency_results
                GROUP BY ALL
                ORDER BY _dataset, _preprocessing, _group_type
            """).df()
        else:
            consistency_configs = pd.DataFrame()
        _con.close()
    else:
        activity_configs = pd.DataFrame()
        consistency_configs = pd.DataFrame()
    return activity_configs, consistency_configs


@app.cell(hide_code=True)
def _(activity_configs, consistency_configs, mo):
    mo.md(f"""
    ### Run configurations

    **Activity runs**: {len(activity_configs)} configurations
    **Consistency runs**: {len(consistency_configs)} configurations
    """)
    return


@app.cell
def _(activity_configs):
    activity_configs
    return


@app.cell
def _(consistency_configs):
    consistency_configs
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
