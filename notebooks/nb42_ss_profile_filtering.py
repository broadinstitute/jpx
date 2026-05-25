# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb==1.5.2",
#     "pandas==3.0.3",
#     "pyarrow",
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

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import COPAIRS_RESULTS_DB, METADATA_DB, RAW_DATA_DIR

    # Canonical metadata columns expected by copairs-runner (joined from DuckDB at runtime).
    # Extra metadata in the parquet causes conflicts with the DB join.
    CANONICAL_METADATA = {"Metadata_JCP2022", "Metadata_Plate", "Metadata_Source", "Metadata_Well"}


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Profile Filtering

    Utilities for filtering JUMP profiles before copairs analysis.

    **Exported functions:**

    - `create_union_profiles(activity1, activity2, parquet1, parquet2, output_dir, suffix)` -
      Filter two parquet profile files to the union of active perturbations from both,
      enabling fair comparison between feature extraction methods (CP vs DL).

    - `filter_source7_exclusive(input_parquet, db_path, output)` -
      Keep all non-source_7 wells plus source_7 wells for compounds profiled
      ONLY at source_7. Avoids dose confounds (source_7 used 0.625 uM vs 10 uM elsewhere).

    Migrated from:
    - `src/jump_production/processing/create_union_profiles.py`
    - `src/jump_production/processing/filter_source7_exclusive.py`
    """)
    return


@app.function
def create_union_profiles(
    activity1: str | Path,
    activity2: str | Path,
    parquet1: str | Path,
    parquet2: str | Path,
    output_dir: str | Path | None = None,
    suffix: str = "active_union",
    on_column: str = "Metadata_JCP2022",
    filter_column: str = "below_corrected_p",
) -> list[Path]:
    """Filter profiles to union of active perturbations from two activity result sets.

    Creates pre-filtered parquet files containing all perturbations active in
    either method, enabling comprehensive comparison of feature extraction
    methods (e.g., CellProfiler vs Deep Learning).

    Args:
        activity1: Activity results CSV for the first profile set.
        activity2: Activity results CSV for the second profile set.
        parquet1: Parquet file for the first profile set.
        parquet2: Parquet file for the second profile set.
        output_dir: Output directory for filtered parquets. Default:
            ``RAW_DATA_DIR / "profiles"``
        suffix: Suffix for output filenames (underscore prepended automatically).
        on_column: Column to match perturbations on.
        filter_column: Boolean column in activity CSVs indicating active perturbations.

    Returns:
        List of output Paths (one per input parquet).
    """
    activity1 = Path(activity1)
    activity2 = Path(activity2)
    parquet1 = Path(parquet1)
    parquet2 = Path(parquet2)
    if output_dir is None:
        output_dir = RAW_DATA_DIR / "profiles"
    output_dir = Path(output_dir)

    # Load activity results and get active sets
    act1 = pd.read_csv(activity1)
    act2 = pd.read_csv(activity2)

    active1 = set(act1[act1[filter_column]][on_column].unique())
    active2 = set(act2[act2[filter_column]][on_column].unique())

    logger.info(f"{parquet1.stem}: {len(active1):,} active perturbations")
    logger.info(f"{parquet2.stem}: {len(active2):,} active perturbations")

    # Compute union (all perturbations active in either method)
    union = active1 | active2
    logger.info(f"Union: {len(union):,} perturbations")
    logger.info(f"  Intersection (both): {len(active1 & active2):,}")
    logger.info(f"  Only {parquet1.stem}: {len(active1 - active2):,}")
    logger.info(f"  Only {parquet2.stem}: {len(active2 - active1):,}")

    # Normalize suffix
    _suffix = suffix if suffix.startswith("_") else f"_{suffix}"

    # Filter and save each parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for parquet_path in [parquet1, parquet2]:
        df = pd.read_parquet(parquet_path)
        original_rows = len(df)

        df_filtered = df[df[on_column].isin(union)]

        output_path = output_dir / f"{parquet_path.stem}{_suffix}.parquet"
        df_filtered.to_parquet(output_path, index=False)

        logger.info(
            f"{parquet_path.stem}: {original_rows:,} -> {len(df_filtered):,} rows "
            f"({len(df_filtered) / original_rows * 100:.1f}%) -> {output_path.name}"
        )
        outputs.append(output_path)

    return outputs


@app.function
def _get_source7_exclusive_compounds(db_path: Path) -> set[str]:
    """Query DuckDB for compounds profiled only at source_7.

    Returns JCP2022 IDs for compounds that appear in source_7 wells
    but never in any other source's wells.
    """
    with duckdb.connect(str(db_path), read_only=True) as con:
        result = con.sql("""
            SELECT DISTINCT s.Metadata_JCP2022
            FROM (SELECT DISTINCT Metadata_JCP2022 FROM well WHERE Metadata_Source = 'source_7') s
            LEFT JOIN (SELECT DISTINCT Metadata_JCP2022 FROM well WHERE Metadata_Source != 'source_7') o
                USING (Metadata_JCP2022)
            WHERE o.Metadata_JCP2022 IS NULL
        """).fetchdf()
    return set(result["Metadata_JCP2022"])


@app.function
def filter_source7_exclusive(
    input_parquet: str | Path,
    db_path: str | Path | None = None,
    output: str | Path | None = None,
) -> Path:
    """Keep non-source_7 wells + source_7-exclusive compounds.

    Creates a new profile parquet that includes all non-source_7 wells plus
    source_7 wells for compounds profiled ONLY at source_7 (no cross-site
    replicates). This avoids dose confounds since source_7 used 0.625 uM
    while other sources used 10 uM.

    Also standardizes metadata columns to the canonical set expected by
    copairs-runner (Metadata_JCP2022, Metadata_Plate, Metadata_Source,
    Metadata_Well), dropping extra Metadata_* columns that would conflict
    with the DuckDB join at runtime.

    Args:
        input_parquet: Full profile parquet (with source_7 wells).
        db_path: Augmented metadata DuckDB path. Default:
            ``METADATA_DB``
        output: Output parquet path. Default:
            derived from input name with ``_with_source7`` suffix.

    Returns:
        Path to the output parquet file.
    """
    input_parquet = Path(input_parquet)
    if db_path is None:
        db_path = METADATA_DB
    db_path = Path(db_path)
    if output is None:
        output = input_parquet.parent / f"{input_parquet.stem}_with_source7.parquet"
    output = Path(output)

    if not input_parquet.exists():
        msg = f"Input file not found: {input_parquet}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    if not db_path.exists():
        msg = f"Database not found: {db_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    exclusive = _get_source7_exclusive_compounds(db_path)
    logger.info(f"Found {len(exclusive):,} source_7-exclusive compounds in database")

    df = pd.read_parquet(input_parquet)
    is_source7 = df["Metadata_Source"] == "source_7"
    is_exclusive = df["Metadata_JCP2022"].isin(exclusive)

    # Keep: all non-source_7 wells + source_7 wells for exclusive compounds
    df_filtered = df[~is_source7 | is_exclusive]

    # Standardize metadata columns to canonical set
    extra_meta = [c for c in df_filtered.columns if c.startswith("Metadata_") and c not in CANONICAL_METADATA]
    if extra_meta:
        df_filtered = df_filtered.drop(columns=extra_meta)
        logger.info(f"Dropped {len(extra_meta)} extra metadata columns: {extra_meta}")

    logger.info(f"Input: {len(df):,} rows")
    logger.info(f"Source_7 wells kept (exclusive): {(is_source7 & is_exclusive).sum():,}")
    logger.info(f"Source_7 wells dropped (non-exclusive): {(is_source7 & ~is_exclusive).sum():,}")
    logger.info(f"Output: {len(df_filtered):,} rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(output, index=False)
    logger.success(f"Saved to {output}")

    return output


# ---------------------------------------------------------------------------
# Demo cells
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: source_7-exclusive compound counts

    Queries the metadata database to show how many compounds are
    profiled only at source_7 (the ones we keep when filtering).
    """)
    return


@app.cell
def _():
    # Count source_7-exclusive compounds from the metadata database
    if METADATA_DB.exists():
        _con = duckdb.connect(str(METADATA_DB), read_only=True)
        source7_stats = _con.execute("""
            WITH source7_compounds AS (
                SELECT DISTINCT Metadata_JCP2022
                FROM well
                WHERE Metadata_Source = 'source_7'
            ),
            other_source_compounds AS (
                SELECT DISTINCT Metadata_JCP2022
                FROM well
                WHERE Metadata_Source != 'source_7'
            ),
            classified AS (
                SELECT
                    s.Metadata_JCP2022,
                    CASE WHEN o.Metadata_JCP2022 IS NULL THEN 'exclusive' ELSE 'shared' END as status
                FROM source7_compounds s
                LEFT JOIN other_source_compounds o USING (Metadata_JCP2022)
            )
            SELECT status, COUNT(*) as n_compounds
            FROM classified
            GROUP BY status
            ORDER BY status
        """).df()
        _con.close()
        source7_stats
    else:
        logger.warning(f"Metadata database not found: {METADATA_DB}")
        source7_stats = pd.DataFrame(columns=["status", "n_compounds"])
        source7_stats
    return (source7_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: existing filtered profile files

    Lists the profile parquet files that currently exist, showing both
    the base profiles and any union/source7-filtered variants.
    """)
    return


@app.cell
def _():
    profiles_dir = RAW_DATA_DIR / "profiles"
    if profiles_dir.exists():
        profile_files = sorted(profiles_dir.glob("*.parquet"))
        profile_info = []
        for p in profile_files:
            size_mb = p.stat().st_size / (1024 * 1024)
            profile_info.append(
                {
                    "filename": p.name,
                    "size_mb": round(size_mb, 1),
                }
            )
        profile_listing = pd.DataFrame(profile_info)
        profile_listing
    else:
        logger.warning(f"Profiles directory not found: {profiles_dir}")
        profile_listing = pd.DataFrame(columns=["filename", "size_mb"])
        profile_listing
    return (profile_listing,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: activity result overlap

    If activity results exist for both CP and DL (no source_7), shows
    the overlap between active perturbation sets - the union is what
    `create_union_profiles` produces.
    """)
    return


@app.cell
def _():
    # Show overlap between CP and DL active perturbations (if results exist)
    if COPAIRS_RESULTS_DB.exists():
        _con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
        _tables = [t[0] for t in _con.execute("SHOW TABLES").fetchall()]
        if "activity_results" in _tables:
            overlap = _con.execute("""
                WITH cp AS (
                    SELECT DISTINCT Metadata_JCP2022
                    FROM activity_results
                    WHERE _dataset = 'compound_no_source7'
                      AND _preprocessing = 'activity_no_target2'
                      AND _filter = 'all_sources'
                      AND _activity_params = 'default'
                      AND below_corrected_p = true
                ),
                dl AS (
                    SELECT DISTINCT Metadata_JCP2022
                    FROM activity_results
                    WHERE _dataset = 'compound_DL_CPCNN_no_source7'
                      AND _preprocessing = 'activity_no_target2'
                      AND _filter = 'all_sources'
                      AND _activity_params = 'default'
                      AND below_corrected_p = true
                )
                SELECT
                    (SELECT COUNT(*) FROM cp) as cp_active,
                    (SELECT COUNT(*) FROM dl) as dl_active,
                    (SELECT COUNT(*) FROM cp INTERSECT SELECT COUNT(*) FROM dl) as both_active,
                    (SELECT COUNT(*) FROM (SELECT * FROM cp UNION SELECT * FROM dl)) as union_active,
                    (SELECT COUNT(*) FROM cp WHERE Metadata_JCP2022 NOT IN (SELECT * FROM dl)) as cp_only,
                    (SELECT COUNT(*) FROM dl WHERE Metadata_JCP2022 NOT IN (SELECT * FROM cp)) as dl_only
            """).df()
        else:
            overlap = pd.DataFrame()
        _con.close()
        overlap
    else:
        overlap = pd.DataFrame()
        overlap
    return (overlap,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
