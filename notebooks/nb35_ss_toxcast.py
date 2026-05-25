# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars==1.41.0",
#     "pandas==3.0.3",
#     "openpyxl==3.1.5",
#     "python-dotenv",
#     "loguru==0.7.3",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import sys
    import zipfile
    from pathlib import Path

    import pandas as pd
    import polars as pl
    from loguru import logger

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

    # Columns to keep from mc5-6 data
    MC5_COLUMNS = [
        "dsstox_substance_id",
        "chid",
        "chnm",
        "aenm",
        "hitc",
        "ac50",
        "acc",
        "conc_max",
        "conc_min",
        "nconc",
    ]

    # Cell-based cell formats (for cytotoxicity correction)
    CELL_BASED_FORMATS = [
        "cell line",
        "primary cell",
        "secondary cell",
        "primary cell co-culture",
        "cell-based",
    ]

    # Cell-free cell formats (no cytotoxicity correction needed)
    CELL_FREE_FORMATS = [
        "cell-free",
        "tissue-based cell-free",
    ]

    # Assays to exclude (from OASIS filtering)
    EXCLUDED_ASSAY_PATTERNS = [
        "Followup",  # Follow-up experiments
        "TRANS",  # Transporter assays
        "EcoTox",  # Ecological toxicity
    ]

    EXCLUDED_ASSAY_NAMES = [
        "CLD_6hr",
        "CLD_24hr",
        "APR_HepG2_1hr",
        "APR_HepG2_72hr",
    ]

    # Assay component patterns to exclude
    EXCLUDED_COMPONENT_PATTERNS = [
        "_ch1",  # Redundant channel 1
        "_ch2",  # Redundant channel 2
    ]

    # Time-series viability assays to exclude (OASIS excludes these)
    EXCLUDED_VIABILITY_ASSAYS = [
        "TOX21_RT_HEPG2_FLO_00hr_viability",
        "TOX21_RT_HEPG2_FLO_08hr_viability",
        "TOX21_RT_HEPG2_FLO_16hr_viability",
        "TOX21_RT_HEPG2_FLO_24hr_viability",
        "TOX21_RT_HEPG2_FLO_32hr_viability",
        "TOX21_RT_HEPG2_GLO_00hr_viability",
        "TOX21_RT_HEPG2_GLO_08hr_viability",
        "TOX21_RT_HEPG2_GLO_16hr_viability",
        "TOX21_RT_HEPG2_GLO_24hr_viability",
        "TOX21_RT_HEPG2_GLO_32hr_viability",
        "TOX21_RT_HEK293_FLO_00hr_viability",
        "TOX21_RT_HEK293_FLO_08hr_viability",
        "TOX21_RT_HEK293_FLO_16hr_viability",
        "TOX21_RT_HEK293_FLO_24hr_viability",
        "TOX21_RT_HEK293_FLO_32hr_viability",
        "TOX21_RT_HEK293_GLO_00hr_viability",
        "TOX21_RT_HEK293_GLO_08hr_viability",
        "TOX21_RT_HEK293_GLO_16hr_viability",
        "TOX21_RT_HEK293_GLO_24hr_viability",
        "TOX21_RT_HEK293_GLO_32hr_viability",
    ]


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ToxCast Bioactivity Processing

    Process EPA ToxCast invitrodb v4.3 bioactivity data for JUMP compound matching.

    ToxCast (Toxicity Forecaster) is EPA's high-throughput screening program that tests
    chemicals across ~1,600 assay endpoints to predict potential toxicity and biological
    activity.

    **Exported function:**

    - `process_toxcast(toxcast_zip, dsstox_zip, compound_file, output_file, apply_filters, apply_cytotox)` -
      End-to-end pipeline: extract from ZIP files, apply OASIS-style filtering,
      cytotoxicity correction, map DTXSID to InChIKey, match to JUMP compounds,
      output long-format table for SQL import.

    Filtering follows the OASIS project methodology:
    - OASIS repo: https://github.com/broadinstitute/2025_04_13_OASIS_CellPainting
    - Notebook: `00_prepare_data/05_extract_invitrodb.ipynb`

    Data Sources:
    - ToxCast invitrodb v4.3: https://doi.org/10.23645/epacomptox.6062623.v14
    - DSSTox identifiers: https://www.epa.gov/comptox-tools/distributed-structure-searchable-toxicity-dsstox-database

    Migrated from: `src/jump_production/processing/toxcast.py`
    """)
    return


# ---------------------------------------------------------------------------
# Internal helpers for loading data from ZIP files
# ---------------------------------------------------------------------------


@app.function
def _load_dsstox_mapping(zip_path: Path) -> pl.DataFrame:
    """Load DTXSID to InChIKey mapping from DSSTox zip.

    Reads DSSToxCCDdump.csv from the zip, filters to valid entries,
    and creates an InChIKey14 prefix column for stereoisomer-tolerant matching.
    """
    logger.info(f"Loading DSSTox mapping from {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [f for f in zf.namelist() if f.endswith("DSSToxCCDdump.csv")]
        if not csv_files:
            raise ValueError("Could not find DSSToxCCDdump.csv in zip")

        with zf.open(csv_files[0]) as f:
            df = pl.read_csv(
                f,
                columns=["DTXSID", "INCHIKEY"],
                infer_schema=False,
            )

    df = df.filter((pl.col("INCHIKEY").is_not_null()) & (pl.col("INCHIKEY") != "") & (pl.col("DTXSID").is_not_null()))
    df = df.with_columns(pl.col("INCHIKEY").str.slice(0, 14).alias("InChIKey14"))

    logger.info(f"Loaded {len(df)} DTXSID to InChIKey mappings")
    return df


@app.function
def _load_assay_annotations(zip_path: Path) -> pl.DataFrame:
    """Load assay annotations from ToxCast summary zip.

    Reads the assay_annotations Excel file embedded in the zip.
    Selects columns needed for OASIS-style filtering.
    """
    logger.info("Loading assay annotations...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        xlsx_files = [f for f in zf.namelist() if "assay_annotations" in f and f.endswith(".xlsx")]
        if not xlsx_files:
            raise ValueError("Could not find assay_annotations xlsx in zip")

        with zf.open(xlsx_files[0]) as f:
            df_pd = pd.read_excel(f)
            df = pl.from_pandas(df_pd)

    cols = [
        "aeid",
        "assay_component_endpoint_name",
        "organism",
        "cell_format",
        "tissue",
        "cell_short_name",
        "assay_design_type",
        "assay_function_type",
        "cell_viability_assay",
        "assay_name",
    ]
    available_cols = [c for c in cols if c in df.columns]
    df = df.select(available_cols)

    logger.info(f"Loaded {len(df)} assay annotations")
    return df


@app.function
def _load_toxcast_data(zip_path: Path) -> pl.DataFrame:
    """Load mc5-6 bioactivity data from ToxCast summary zip.

    Reads the mc5-6 CSV from the zip, selects relevant columns,
    and casts numeric types. Filters to entries with valid DTXSID.
    """
    logger.info(f"Loading ToxCast mc5-6 data from {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        mc5_files = [f for f in zf.namelist() if "mc5-6" in f and f.endswith(".csv")]
        if not mc5_files:
            raise ValueError("Could not find mc5-6 CSV in zip")

        mc5_file = mc5_files[0]
        logger.info(f"Reading {mc5_file}")

        with zf.open(mc5_file) as f:
            df = pl.read_csv(
                f,
                infer_schema=False,
                null_values=["NA", ""],
            )

    available_cols = [c for c in MC5_COLUMNS if c in df.columns]
    df = df.select(available_cols)

    numeric_cols = ["hitc", "ac50", "acc", "conc_max", "conc_min"]
    for col in numeric_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    if "nconc" in df.columns:
        df = df.with_columns(pl.col("nconc").cast(pl.Int32, strict=False))

    if "chid" in df.columns:
        df = df.with_columns(pl.col("chid").cast(pl.Int64, strict=False))

    df = df.filter((pl.col("dsstox_substance_id").is_not_null()) & (pl.col("dsstox_substance_id") != ""))

    logger.info(f"Loaded {len(df)} ToxCast records")
    logger.info(f"Unique chemicals: {df['dsstox_substance_id'].n_unique()}")
    logger.info(f"Unique assays: {df['aenm'].n_unique()}")

    return df


@app.function
def _load_jump_compounds(compound_file: Path) -> pl.DataFrame:
    """Load JUMP compounds with InChIKey for matching.

    Creates InChIKey14 prefix for stereoisomer-tolerant matching.
    """
    logger.info(f"Loading JUMP compounds from {compound_file}")

    df = pl.read_csv(
        compound_file,
        columns=["Metadata_JCP2022", "Metadata_InChIKey"],
    )

    df = df.filter(pl.col("Metadata_InChIKey").is_not_null())
    df = df.with_columns(pl.col("Metadata_InChIKey").str.slice(0, 14).alias("InChIKey14"))

    logger.info(f"Loaded {len(df)} JUMP compounds with InChIKey")
    return df


# ---------------------------------------------------------------------------
# Internal helpers for filtering and cytotoxicity correction
# ---------------------------------------------------------------------------


@app.function
def _filter_assays(
    toxcast_df: pl.DataFrame,
    assay_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Apply OASIS-style assay filtering.

    Filtering follows OASIS methodology from 05_extract_invitrodb.ipynb:
    - Common filters applied to all: human only, exclude whole embryo,
      exclude redundant channels
    - Cell-based only: exclude Followup/TRANS/EcoTox, exclude CLD_*/APR_*,
      exclude background reporter/control, exclude cell_short_name == "NA"
    - Cell-free only: exclude Followup

    Returns:
        Tuple of (cell_based_df, cell_free_df, viability_df)
    """
    logger.info("Applying OASIS-style assay filtering...")

    # Join toxcast data with assay annotations
    merged = toxcast_df.join(
        assay_df,
        left_on="aenm",
        right_on="assay_component_endpoint_name",
        how="inner",
    )
    logger.info(f"After joining with assay annotations: {len(merged)} records")

    # === COMMON FILTERS (applied to all assays) ===

    if "organism" in merged.columns:
        merged = merged.filter(pl.col("organism") == "human")
        logger.info(f"After human filter: {len(merged)} records")

    if "cell_format" in merged.columns:
        merged = merged.filter(pl.col("cell_format") != "whole embryo")
        logger.info(f"After excluding whole embryo: {len(merged)} records")

    merged = merged.filter(~pl.col("aenm").is_in(EXCLUDED_VIABILITY_ASSAYS))
    for pattern in EXCLUDED_COMPONENT_PATTERNS:
        merged = merged.filter(~pl.col("aenm").str.contains(pattern))
    logger.info(f"After excluding redundant channels: {len(merged)} records")

    if "assay_function_type" in merged.columns:
        merged = merged.filter(pl.col("assay_function_type") != "background control")
        logger.info(f"After excluding background control: {len(merged)} records")

    # === SPLIT INTO CELL-BASED VS CELL-FREE ===
    if "cell_format" in merged.columns:
        cell_based = merged.filter(pl.col("cell_format").is_in(CELL_BASED_FORMATS))
        cell_free = merged.filter(pl.col("cell_format").is_in(CELL_FREE_FORMATS))
    else:
        cell_based = merged
        cell_free = pl.DataFrame()

    # === CELL-BASED ONLY FILTERS ===

    if "assay_name" in cell_based.columns:
        for pattern in EXCLUDED_ASSAY_PATTERNS:
            cell_based = cell_based.filter(~pl.col("assay_name").str.contains(pattern))
        cell_based = cell_based.filter(~pl.col("assay_name").is_in(EXCLUDED_ASSAY_NAMES))
        logger.info(f"Cell-based after excluding followup/TRANS/EcoTox: {len(cell_based)} records")

    if "assay_design_type" in cell_based.columns:
        cell_based = cell_based.filter(pl.col("assay_design_type") != "background reporter")
        logger.info(f"Cell-based after excluding background reporters: {len(cell_based)} records")

    # === CELL-FREE ONLY FILTERS ===

    if not cell_free.is_empty() and "assay_name" in cell_free.columns:
        cell_free = cell_free.filter(~pl.col("assay_name").str.contains("Followup"))
        logger.info(f"Cell-free after excluding followup: {len(cell_free)} records")

    # === SEPARATE VIABILITY FROM PRIMARY ENDPOINTS ===

    if "cell_viability_assay" in cell_based.columns:
        viability = cell_based.filter(pl.col("cell_viability_assay") == 1)
        cell_based = cell_based.filter(pl.col("cell_viability_assay") != 1)
    else:
        viability = pl.DataFrame()

    if "cell_short_name" in cell_based.columns:
        cell_based = cell_based.filter(pl.col("cell_short_name") != "NA")
        logger.info(f"Cell-based after excluding NA cell_short_name: {len(cell_based)} records")

    logger.info(f"Cell-based assays: {len(cell_based)} records ({cell_based['aenm'].n_unique()} endpoints)")
    logger.info(
        f"Cell-free assays: {len(cell_free)} records "
        f"({cell_free['aenm'].n_unique() if not cell_free.is_empty() else 0} endpoints)"
    )
    logger.info(f"Viability assays: {len(viability)} records")

    return cell_based, cell_free, viability


@app.function
def _calculate_cytotox_context(
    viability_df: pl.DataFrame,
    hitcall_threshold: float = 0.9,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Calculate cytotoxicity AC50 for each chemical by cell line and tissue.

    Following OASIS methodology:
    - Group viability assay results by (chemical, cell_short_name) and (chemical, tissue)
    - For chemicals with hits (hitcall > 0.9), calculate median AC50
    - A compound must be hit in >=20% of tested instances to count
    - AC50 must be < 100 uM (highest tested dose)

    Returns:
        Tuple of (cell_cytotox, tissue_cytotox) DataFrames
    """
    if viability_df.is_empty() or "chid" not in viability_df.columns:
        logger.warning("No viability data available for cytotox calculation")
        return pl.DataFrame(), pl.DataFrame()

    logger.info("Calculating cytotoxicity context from viability assays...")

    viability_df = viability_df.with_columns((pl.col("hitc") > hitcall_threshold).cast(pl.Int64).alias("is_hit"))

    def _apply_thresholds(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
        return df.with_columns(
            pl.when((pl.col("nhit") / pl.col("ntested")) < 0.2)
            .then(None)
            .when(pl.col("cytotox_median_ac50") > 100)
            .then(None)
            .otherwise(pl.col("cytotox_median_ac50"))
            .alias("cytotox_median_ac50")
        )

    # Group by cell_short_name
    if "cell_short_name" in viability_df.columns:
        cell_cytotox = (
            viability_df.filter(pl.col("cell_short_name").is_not_null())
            .group_by(["chid", "cell_short_name"])
            .agg(
                [
                    pl.len().alias("ntested"),
                    pl.col("is_hit").sum().alias("nhit"),
                    pl.col("ac50").filter(pl.col("is_hit") == 1).median().alias("cytotox_median_ac50"),
                ]
            )
            .with_columns(pl.lit("cell_short_name").alias("cytotox_source"))
        )
        cell_cytotox = _apply_thresholds(cell_cytotox)
    else:
        cell_cytotox = pl.DataFrame()

    # Group by tissue
    if "tissue" in viability_df.columns:
        tissue_cytotox = (
            viability_df.filter(pl.col("tissue").is_not_null())
            .group_by(["chid", "tissue"])
            .agg(
                [
                    pl.len().alias("ntested"),
                    pl.col("is_hit").sum().alias("nhit"),
                    pl.col("ac50").filter(pl.col("is_hit") == 1).median().alias("cytotox_median_ac50"),
                ]
            )
            .with_columns(pl.lit("tissue").alias("cytotox_source"))
        )
        tissue_cytotox = _apply_thresholds(tissue_cytotox)
    else:
        tissue_cytotox = pl.DataFrame()

    logger.info(f"Cytotox by cell_short_name: {len(cell_cytotox)} chemical-cell pairs")
    logger.info(f"Cytotox by tissue: {len(tissue_cytotox)} chemical-tissue pairs")

    return cell_cytotox, tissue_cytotox


@app.function
def _apply_cytotox_correction(
    cell_based_df: pl.DataFrame,
    cell_cytotox: pl.DataFrame,
    tissue_cytotox: pl.DataFrame,
    hitcall_threshold: float = 0.9,
) -> pl.DataFrame:
    """Apply 2-fold selectivity filter to cell-based assays.

    A hit is only considered specific if: ac50 < cytotox_median_ac50 / 2.
    This removes non-specific toxic compounds from the active set.
    """
    if cell_based_df.is_empty():
        return cell_based_df

    logger.info("Applying cytotoxicity correction (2-fold selectivity filter)...")

    # First try to match by cell_short_name
    if not cell_cytotox.is_empty() and "cell_short_name" in cell_based_df.columns:
        matched_cell = cell_based_df.join(
            cell_cytotox.select(["chid", "cell_short_name", "cytotox_median_ac50"]),
            on=["chid", "cell_short_name"],
            how="left",
        )
    else:
        matched_cell = cell_based_df.with_columns(pl.lit(None).alias("cytotox_median_ac50"))

    # For unmatched, try tissue
    if not tissue_cytotox.is_empty() and "tissue" in matched_cell.columns:
        unmatched = matched_cell.filter(pl.col("cytotox_median_ac50").is_null())
        matched = matched_cell.filter(pl.col("cytotox_median_ac50").is_not_null())

        unmatched = unmatched.drop("cytotox_median_ac50").join(
            tissue_cytotox.select(["chid", "tissue", "cytotox_median_ac50"]),
            on=["chid", "tissue"],
            how="left",
        )

        matched_cell = pl.concat([matched, unmatched], how="diagonal")

    # Apply 2-fold selectivity filter
    # Hit is only valid if ac50 < cytotox_median_ac50 / 2
    # If no cytotox data, keep original hitcall
    # Note: OASIS uses strict > for hitcall threshold (not >=)
    matched_cell = matched_cell.with_columns(
        pl.when(pl.col("cytotox_median_ac50").is_null())
        .then(pl.col("hitc") > hitcall_threshold)
        .when((pl.col("cytotox_median_ac50") / 2) < pl.col("ac50"))
        .then(False)
        .otherwise(pl.col("hitc") > hitcall_threshold)
        .alias("corrected_active")
    )

    n_original = matched_cell.filter(pl.col("hitc") > hitcall_threshold).height
    n_corrected = matched_cell.filter(pl.col("corrected_active")).height
    n_removed = n_original - n_corrected

    logger.info(f"Original active: {n_original}, After correction: {n_corrected}, Removed as non-specific: {n_removed}")

    return matched_cell


@app.function
def _match_and_format(
    toxcast_df: pl.DataFrame,
    dsstox_df: pl.DataFrame,
    jump_df: pl.DataFrame,
    assay_df: pl.DataFrame | None,
    apply_filters: bool,
    apply_cytotox: bool,
    hitcall_threshold: float,
) -> pl.DataFrame:
    """Match ToxCast data to JUMP compounds and format for output.

    Internal workhorse that handles DSSTox mapping, JUMP matching,
    optional OASIS filtering, cytotoxicity correction, majority voting,
    and final column selection.
    """
    # Step 1: Map DTXSID to InChIKey via DSSTox
    toxcast_with_inchi = toxcast_df.join(
        dsstox_df.select(["DTXSID", "InChIKey14"]),
        left_on="dsstox_substance_id",
        right_on="DTXSID",
        how="inner",
    )
    logger.info(f"After DSSTox mapping: {toxcast_with_inchi['dsstox_substance_id'].n_unique()} chemicals")

    # Step 2: Match to JUMP compounds via InChIKey14
    matched = toxcast_with_inchi.join(
        jump_df.select(["Metadata_JCP2022", "InChIKey14"]),
        on="InChIKey14",
        how="inner",
    )
    logger.info(f"Matched to {matched['Metadata_JCP2022'].n_unique()} JUMP compounds")

    # Step 3: Apply assay filtering if requested
    if apply_filters and assay_df is not None:
        cell_based, cell_free, viability = _filter_assays(matched, assay_df)

        # Step 4: Apply cytotoxicity correction for cell-based assays
        if apply_cytotox and not viability.is_empty():
            cell_cytotox, tissue_cytotox = _calculate_cytotox_context(viability, hitcall_threshold)
            cell_based = _apply_cytotox_correction(cell_based, cell_cytotox, tissue_cytotox, hitcall_threshold)
            cell_based = cell_based.with_columns(pl.col("corrected_active").cast(pl.Int8).alias("is_active"))
        else:
            # Note: OASIS uses strict > for hitcall threshold (not >=)
            cell_based = cell_based.with_columns((pl.col("hitc") > hitcall_threshold).cast(pl.Int8).alias("is_active"))

        # Cell-free: no cytotox correction
        if not cell_free.is_empty():
            cell_free = cell_free.with_columns((pl.col("hitc") > hitcall_threshold).cast(pl.Int8).alias("is_active"))

        # Combine cell-based and cell-free
        if cell_free.is_empty():
            matched = cell_based
        elif cell_based.is_empty():
            matched = cell_free
        else:
            common_cols = list(set(cell_based.columns) & set(cell_free.columns))
            matched = pl.concat(
                [cell_based.select(common_cols), cell_free.select(common_cols)],
                how="vertical",
            )
    else:
        # No filtering - use raw hitcall
        matched = matched.with_columns((pl.col("hitc") > hitcall_threshold).cast(pl.Int8).alias("is_active"))

    # Step 5: Create output columns with proper naming
    result = matched.select(
        [
            "Metadata_JCP2022",
            pl.col("aenm").alias("Metadata_txcst_assay"),
            pl.col("hitc").alias("Metadata_txcst_hitcall"),
            pl.col("is_active").alias("Metadata_txcst_active"),
            pl.when(pl.col("is_active") == 1).then(pl.col("ac50")).otherwise(None).alias("Metadata_txcst_ac50"),
            pl.col("acc").alias("Metadata_txcst_acc"),
        ]
    )

    # Step 6: Handle duplicates - majority voting (OASIS-style)
    result = result.group_by(["Metadata_JCP2022", "Metadata_txcst_assay"]).agg(
        [
            pl.col("Metadata_txcst_hitcall").max(),
            (pl.col("Metadata_txcst_active").sum() > (pl.len() / 2)).cast(pl.Int8).alias("Metadata_txcst_active"),
            pl.col("Metadata_txcst_ac50").drop_nulls().min(),
            pl.col("Metadata_txcst_acc").max(),
        ]
    )

    # Step 7: Ensure AC50 is null for inactive records after majority voting
    result = result.with_columns(
        pl.when(pl.col("Metadata_txcst_active") == 0)
        .then(None)
        .otherwise(pl.col("Metadata_txcst_ac50"))
        .alias("Metadata_txcst_ac50")
    )

    logger.info(f"Final output: {len(result)} compound-assay records")
    logger.info(
        f"Active records (hitcall > {hitcall_threshold}): {result.filter(pl.col('Metadata_txcst_active') == 1).height}"
    )

    return result


# ---------------------------------------------------------------------------
# Public exported function
# ---------------------------------------------------------------------------


@app.function
def process_toxcast(
    toxcast_zip: str | Path | None = None,
    dsstox_zip: str | Path | None = None,
    compound_file: str | Path | None = None,
    output_file: str | Path | None = None,
    apply_filters: bool = True,
    apply_cytotox: bool = True,
    hitcall_threshold: float = 0.9,
) -> Path:
    """Process ToxCast bioactivity data end-to-end and save to parquet.

    Extracts mc5-6 bioactivity data from the ToxCast summary zip, maps
    DTXSID to InChIKey via DSSTox identifiers, matches to JUMP compounds
    via InChIKey14 (stereoisomer-tolerant), and optionally applies
    OASIS-style assay filtering with cytotoxicity correction.

    Args:
        toxcast_zip: Path to ToxCast summary zip. Default:
            ``EXTERNAL_DATA_DIR / "toxcast_invitrodb_v4_3_summary.zip"``
        dsstox_zip: Path to DSSTox identifiers zip. Default:
            ``EXTERNAL_DATA_DIR / "dsstox_identifiers.zip"``
        compound_file: Path to JUMP compounds CSV. Default:
            ``EXTERNAL_DATA_DIR / "compound.csv.gz"``
        output_file: Output parquet path. Default:
            ``INTERIM_DATA_DIR / "toxcast_processed.parquet"``
        apply_filters: Whether to apply OASIS-style assay filtering
            (human only, exclude background, etc.). Default True.
        apply_cytotox: Whether to apply cytotoxicity correction
            (2-fold selectivity filter) for cell-based assays. Default True.
        hitcall_threshold: Minimum hitcall value to consider active.
            Default 0.9 (OASIS uses strict >).

    Returns:
        Path to the output parquet file.
    """
    logger.info("Processing EPA ToxCast bioactivity annotations...")

    # Set default paths
    toxcast_zip = (
        Path(toxcast_zip) if toxcast_zip is not None else EXTERNAL_DATA_DIR / "toxcast_invitrodb_v4_3_summary.zip"
    )
    dsstox_zip = Path(dsstox_zip) if dsstox_zip is not None else EXTERNAL_DATA_DIR / "dsstox_identifiers.zip"
    compound_file = Path(compound_file) if compound_file is not None else EXTERNAL_DATA_DIR / "compound.csv.gz"
    output_file = Path(output_file) if output_file is not None else INTERIM_DATA_DIR / "toxcast_processed.parquet"

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Check input files exist
    missing = []
    for f, name in [
        (toxcast_zip, "ToxCast summary zip"),
        (dsstox_zip, "DSSTox identifiers zip"),
        (compound_file, "JUMP compounds"),
    ]:
        if not f.exists():
            missing.append(f"{name}: {f}")

    if missing:
        msg = "Missing input files: " + "; ".join(missing)
        raise FileNotFoundError(msg)

    # Load data
    dsstox_df = _load_dsstox_mapping(dsstox_zip)
    toxcast_df = _load_toxcast_data(toxcast_zip)
    jump_df = _load_jump_compounds(compound_file)

    # Load assay annotations if filtering requested
    assay_df = None
    if apply_filters:
        logger.info("Loading assay annotations for filtering...")
        assay_df = _load_assay_annotations(toxcast_zip)

    # Process and match
    result = _match_and_format(
        toxcast_df,
        dsstox_df,
        jump_df,
        assay_df=assay_df,
        apply_filters=apply_filters,
        apply_cytotox=apply_cytotox,
        hitcall_threshold=hitcall_threshold,
    )

    # Log statistics
    n_compounds = result["Metadata_JCP2022"].n_unique()
    n_assays = result["Metadata_txcst_assay"].n_unique()
    n_active = result.filter(pl.col("Metadata_txcst_active") == 1).height

    logger.info("Output statistics:")
    logger.info(f"  JUMP compounds with ToxCast data: {n_compounds}")
    logger.info(f"  Unique assay endpoints: {n_assays}")
    logger.info(f"  Total compound-assay records: {len(result)}")
    logger.info(f"  Active records (hitcall > {hitcall_threshold}): {n_active}")

    # Sort for deterministic output
    result = result.sort(["Metadata_JCP2022", "Metadata_txcst_assay"])

    # Save results as compressed parquet
    result.write_parquet(output_file, compression="zstd")
    logger.success(f"Saved {len(result)} records to {output_file}")

    return output_file


# ---------------------------------------------------------------------------
# Demo cells
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: check input files

    Verifies that the required ToxCast and DSSTox zip files are present
    in the external data directory.
    """)
    return


@app.cell
def _(mo):
    _toxcast_zip = EXTERNAL_DATA_DIR / "toxcast_invitrodb_v4_3_summary.zip"
    _dsstox_zip = EXTERNAL_DATA_DIR / "dsstox_identifiers.zip"
    _compound_csv = EXTERNAL_DATA_DIR / "compound.csv.gz"

    _files = {
        "ToxCast summary zip": _toxcast_zip,
        "DSSTox identifiers zip": _dsstox_zip,
        "JUMP compounds CSV": _compound_csv,
    }

    _status_rows = []
    for name, path in _files.items():
        exists = path.exists()
        size_mb = f"{path.stat().st_size / (1024 * 1024):.1f} MB" if exists else "N/A"
        _status_rows.append(f"| {name} | `{path.name}` | {'yes' if exists else 'no'} | {size_mb} |")

    _table = "\n".join(_status_rows)
    mo.md(f"""
    | File | Name | Exists | Size |
    |------|------|--------|------|
    {_table}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: process ToxCast data

    Runs the full pipeline if all input files are present. Shows summary
    statistics of the output: matched JUMP compounds, assay endpoints,
    active records.
    """)
    return


@app.cell
def _(mo):
    _toxcast_zip = EXTERNAL_DATA_DIR / "toxcast_invitrodb_v4_3_summary.zip"
    _dsstox_zip = EXTERNAL_DATA_DIR / "dsstox_identifiers.zip"
    _compound_csv = EXTERNAL_DATA_DIR / "compound.csv.gz"

    if all(p.exists() for p in (_toxcast_zip, _dsstox_zip, _compound_csv)):
        _output = process_toxcast()

        # Read back for display
        _result = pl.read_parquet(_output)
        _n_compounds = _result["Metadata_JCP2022"].n_unique()
        _n_assays = _result["Metadata_txcst_assay"].n_unique()
        _n_active = _result.filter(pl.col("Metadata_txcst_active") == 1).height
        _n_total = len(_result)

        mo.md(f"""
    **ToxCast processing complete:**

    - Output: `{_output}`
    - JUMP compounds matched: **{_n_compounds:,}**
    - Unique assay endpoints: **{_n_assays:,}**
    - Total compound-assay records: **{_n_total:,}**
    - Active records: **{_n_active:,}** ({_n_active / _n_total * 100:.1f}%)
        """)
    else:
        mo.md("Skipped: one or more input files not found (run `just get-inputs`)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: output preview

    Shows the first rows of the processed ToxCast output if it exists.
    """)
    return


@app.cell
def _(mo):
    _output_path = INTERIM_DATA_DIR / "toxcast_processed.parquet"
    if _output_path.exists():
        _preview = pl.read_parquet(_output_path).head(20).to_pandas()
        mo.ui.dataframe(_preview)
    else:
        mo.md(f"Output not yet generated: `{_output_path}`")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
