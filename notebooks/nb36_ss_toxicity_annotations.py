# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas==3.0.3",
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
    from loguru import logger

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
    from nb32_ss_processing_utils import (
        generate_inchikeys_fast,
        load_jump_compounds,
        match_to_jump,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Toxicity Annotations

    Migrated from `src/jump_production/processing/toxicity_pk.py` and
    `src/jump_production/processing/mitotox.py` into reusable `@app.function` helpers.

    **Exported functions:**

    - `process_toxicity_pk(input_zip, compound_file, output_file, fast)` - DILI/DICT/PK from ZIP
    - `process_mitotox(input_parquet, compound_file, output_file)` - MitoTox binary labels

    All functions accept optional path overrides (default `None` resolves from `nb00_ss_config`).

    **Data sources:**

    - DILI (Drug-Induced Liver Injury): DILIrank v2 binary classification
    - DICT (Drug-Induced CardioToxicity): DICTrank binary classification
    - PK (Pharmacokinetic parameters): PKSmart regression values (VDss, CL, fup, MRT, t1/2)
    - MitoTox: Mitochondrial toxicity binary labels and functional mechanisms
    """)
    return


# ---------------------------------------------------------------------------
# Internal helpers for toxicity_pk processing
# ---------------------------------------------------------------------------


@app.function
def _process_dili(
    zip_path: Path,
    fast: bool = True,
) -> pd.DataFrame:
    """Process DILI (Drug-Induced Liver Injury) annotations from ZIP.

    DILIrank v2 provides concern levels for drug-induced liver injury.

    Args:
        zip_path: Path to zip file containing DILIrankv2_smiles.csv
        fast: Use RDKit directly instead of jump_smiles (faster but fewer matches)

    Returns:
        DataFrame with standardized_inchikey, Metadata_dili_concern, Metadata_dili_positive
    """
    logger.info(f"Loading DILI data from {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [f for f in zf.namelist() if "DILIrank" in f and f.endswith(".csv")]
        if not csv_files:
            raise ValueError("Could not find DILIrankv2_smiles.csv in zip")
        with zf.open(csv_files[0]) as f:
            df = pd.read_csv(f)
    logger.info(f"Loaded {len(df)} DILI records")

    # Generate InChIKeys using shared utilities
    if fast:
        df = generate_inchikeys_fast(df, "standardized_smiles", source_name="DILI")
    else:
        from nb32_ss_processing_utils import standardize_smiles

        df = standardize_smiles(df, "standardized_smiles", source_name="DILI")

    # Map concern levels to canonical lowercase names
    concern_map = {
        "vMOST-DILI-concern": "most",
        "vMost-DILI-concern": "most",
        "vLess-DILI-concern": "less",
        "vNo-DILI-concern": "no",
        "Ambiguous-DILI-concern": "ambiguous",
    }
    df["Metadata_dili_concern"] = df["vDILI-Concern"].map(concern_map)
    df["Metadata_dili_positive"] = (df["Metadata_dili_concern"] == "most").astype(int)

    result = df[["standardized_inchikey", "Metadata_dili_concern", "Metadata_dili_positive"]].copy()
    result = result.dropna(subset=["standardized_inchikey"])

    logger.info(f"DILI concern distribution:\n{df['Metadata_dili_concern'].value_counts()}")
    return result


@app.function
def _process_dict(
    zip_path: Path,
    fast: bool = True,
) -> pd.DataFrame:
    """Process DICT (Drug-Induced CardioToxicity) annotations from ZIP.

    DICTrank provides concern levels for drug-induced cardiotoxicity.

    Args:
        zip_path: Path to zip file containing DICTrank_smiles.csv
        fast: Use RDKit directly instead of jump_smiles (faster but fewer matches)

    Returns:
        DataFrame with standardized_inchikey, Metadata_dict_concern, Metadata_dict_positive
    """
    logger.info(f"Loading DICT data from {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [f for f in zf.namelist() if "DICTrank" in f and f.endswith(".csv")]
        if not csv_files:
            raise ValueError("Could not find DICTrank_smiles.csv in zip")
        with zf.open(csv_files[0]) as f:
            df = pd.read_csv(f)
    logger.info(f"Loaded {len(df)} DICT records")

    if fast:
        df = generate_inchikeys_fast(df, "Standardized_SMILES", source_name="DICT")
    else:
        from nb32_ss_processing_utils import standardize_smiles

        df = standardize_smiles(df, "Standardized_SMILES", source_name="DICT")

    df["Metadata_dict_concern"] = df["DICT _ Concern"].str.lower().str.strip()
    df["Metadata_dict_positive"] = (df["Metadata_dict_concern"] == "most").astype(int)

    result = df[["standardized_inchikey", "Metadata_dict_concern", "Metadata_dict_positive"]].copy()
    result = result.dropna(subset=["standardized_inchikey"])

    logger.info(f"DICT concern distribution:\n{df['Metadata_dict_concern'].value_counts()}")
    return result


@app.function
def _process_pk(
    zip_path: Path,
    fast: bool = True,
) -> pd.DataFrame:
    """Process PK (Pharmacokinetic) annotations from ZIP.

    PKSmart provides human pharmacokinetic parameters:
    VDss, CL, fup, MRT, t1/2.

    Args:
        zip_path: Path to zip file containing Human_PK_data.csv
        fast: Use RDKit directly instead of jump_smiles (faster but fewer matches)

    Returns:
        DataFrame with standardized_inchikey and Metadata_pk_* columns
    """
    logger.info(f"Loading PK data from {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [f for f in zf.namelist() if "Human_PK" in f and f.endswith(".csv")]
        if not csv_files:
            raise ValueError("Could not find Human_PK_data.csv in zip")
        with zf.open(csv_files[0]) as f:
            df = pd.read_csv(f)
    logger.info(f"Loaded {len(df)} PK records")

    if fast:
        df = generate_inchikeys_fast(df, "smiles_r", source_name="PK")
    else:
        from nb32_ss_processing_utils import standardize_smiles

        df = standardize_smiles(df, "smiles_r", source_name="PK")

    result = df[
        [
            "standardized_inchikey",
            "human_VDss_L_kg",
            "human_CL_mL_min_kg",
            "human_fup",
            "human_mrt",
            "human_thalf",
        ]
    ].copy()

    result = result.rename(
        columns={
            "human_VDss_L_kg": "Metadata_pk_vdss_l_kg",
            "human_CL_mL_min_kg": "Metadata_pk_cl_ml_min_kg",
            "human_fup": "Metadata_pk_fup",
            "human_mrt": "Metadata_pk_mrt_h",
            "human_thalf": "Metadata_pk_thalf_h",
        }
    )

    result = result.dropna(subset=["standardized_inchikey"])
    return result


@app.function
def _match_dataset_to_jump(
    df: pd.DataFrame,
    jump_compounds: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Match a dataset to JUMP compounds via InChIKey14 prefix.

    Args:
        df: DataFrame with standardized_inchikey column
        jump_compounds: JUMP compounds with Metadata_InChIKey14 column
        dataset_name: Name for logging

    Returns:
        DataFrame with Metadata_JCP2022 column added
    """
    df = df.copy()
    df["inchikey_prefix"] = df["standardized_inchikey"].str[:14]

    matched = df.merge(
        jump_compounds[["Metadata_JCP2022", "Metadata_InChIKey14"]],
        left_on="inchikey_prefix",
        right_on="Metadata_InChIKey14",
        how="inner",
    )

    matched = matched.drop(columns=["inchikey_prefix", "Metadata_InChIKey14", "standardized_inchikey"])
    logger.info(f"{dataset_name}: {matched['Metadata_JCP2022'].nunique()} JUMP compounds matched")
    return matched


# ---------------------------------------------------------------------------
# Exported function 1: process_toxicity_pk
# ---------------------------------------------------------------------------


@app.function
def process_toxicity_pk(
    input_zip: Path | None = None,
    compound_file: Path | None = None,
    output_file: Path | None = None,
    fast: bool = True,
) -> Path:
    """Process DILI, DICT, and PK annotations from ZIP, match to JUMP, and save.

    Integrates three SMILES-based datasets:
    - DILI (Drug-Induced Liver Injury): binary classification from DILIrank v2
    - DICT (Drug-Induced CardioToxicity): binary classification from DICTrank
    - PK (Pharmacokinetic parameters): regression values from PKSmart

    All datasets use SMILES for compound identification, matched to JUMP via
    InChIKey14 prefix matching.

    Args:
        input_zip: Path to toxicity_pk_data.zip. Default: EXTERNAL_DATA_DIR / "toxicity_pk_data.zip"
        compound_file: Path to compound.csv.gz. Default: EXTERNAL_DATA_DIR / "compound.csv.gz"
        output_file: Output CSV path. Default: INTERIM_DATA_DIR / "toxicity_pk_processed.csv"
        fast: Use RDKit directly instead of jump_smiles (~5s vs ~10min). Default True.

    Returns:
        Path to the saved output CSV file.
    """
    if input_zip is None:
        input_zip = EXTERNAL_DATA_DIR / "toxicity_pk_data.zip"
    if compound_file is None:
        compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"
    if output_file is None:
        output_file = INTERIM_DATA_DIR / "toxicity_pk_processed.csv"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_zip.exists():
        raise FileNotFoundError(f"Toxicity/PK data zip not found: {input_zip}")
    if not compound_file.exists():
        raise FileNotFoundError(f"JUMP compound metadata not found: {compound_file}")

    # Load JUMP compound metadata
    jump_compounds = load_jump_compounds(compound_file)

    # Process each dataset from the zip file
    logger.info("Processing DILI annotations")
    dili_df = _process_dili(zip_path=input_zip, fast=fast)

    logger.info("Processing DICT annotations")
    dict_df = _process_dict(zip_path=input_zip, fast=fast)

    logger.info("Processing PK annotations")
    pk_df = _process_pk(zip_path=input_zip, fast=fast)

    # Match each dataset to JUMP compounds
    logger.info("Matching to JUMP compounds via InChIKey14 prefix")
    dili_matched = _match_dataset_to_jump(dili_df, jump_compounds, "DILI")
    dict_matched = _match_dataset_to_jump(dict_df, jump_compounds, "DICT")
    pk_matched = _match_dataset_to_jump(pk_df, jump_compounds, "PK")

    # Combine all matched data (outer join on Metadata_JCP2022)
    all_jcp_ids = (
        set(dili_matched["Metadata_JCP2022"])
        | set(dict_matched["Metadata_JCP2022"])
        | set(pk_matched["Metadata_JCP2022"])
    )
    result = pd.DataFrame({"Metadata_JCP2022": sorted(all_jcp_ids)})

    # Merge DILI columns
    dili_cols = dili_matched[["Metadata_JCP2022", "Metadata_dili_concern", "Metadata_dili_positive"]].drop_duplicates(
        subset=["Metadata_JCP2022"]
    )
    result = result.merge(dili_cols, on="Metadata_JCP2022", how="left")

    # Merge DICT columns
    dict_cols = dict_matched[["Metadata_JCP2022", "Metadata_dict_concern", "Metadata_dict_positive"]].drop_duplicates(
        subset=["Metadata_JCP2022"]
    )
    result = result.merge(dict_cols, on="Metadata_JCP2022", how="left")

    # Merge PK columns (take first if duplicates from stereoisomers)
    pk_cols = pk_matched[
        [
            "Metadata_JCP2022",
            "Metadata_pk_vdss_l_kg",
            "Metadata_pk_cl_ml_min_kg",
            "Metadata_pk_fup",
            "Metadata_pk_mrt_h",
            "Metadata_pk_thalf_h",
        ]
    ].drop_duplicates(subset=["Metadata_JCP2022"])
    result = result.merge(pk_cols, on="Metadata_JCP2022", how="left")

    # Save results
    result.to_csv(output_file, index=False)

    # Summary
    logger.info(f"Saved {len(result)} records to {output_file}")
    logger.info(f"  DILI annotations: {result['Metadata_dili_concern'].notna().sum()}")
    logger.info(f"  DICT annotations: {result['Metadata_dict_concern'].notna().sum()}")
    logger.info(f"  PK annotations: {result['Metadata_pk_vdss_l_kg'].notna().sum()}")

    has_dili = result["Metadata_dili_concern"].notna()
    has_dict = result["Metadata_dict_concern"].notna()
    has_pk = result["Metadata_pk_vdss_l_kg"].notna()
    logger.info(f"  DILI + DICT overlap: {(has_dili & has_dict).sum()}")
    logger.info(f"  DILI + PK overlap: {(has_dili & has_pk).sum()}")
    logger.info(f"  DICT + PK overlap: {(has_dict & has_pk).sum()}")
    logger.info(f"  All three: {(has_dili & has_dict & has_pk).sum()}")

    return output_file


# ---------------------------------------------------------------------------
# Exported function 2: process_mitotox
# ---------------------------------------------------------------------------


@app.function
def process_mitotox(
    input_parquet: Path | None = None,
    compound_file: Path | None = None,
    output_file: Path | None = None,
) -> Path:
    """Process MitoTox mitochondrial toxicity annotations and match to JUMP.

    MitoTox (https://www.mitotox.org/) provides binary toxicity labels and
    functional mechanism annotations for compounds with mitochondrial toxicity
    evidence. Compounds are classified as toxic if any experimental record shows
    a positive result across 8 top-level functional categories (F01-F08).

    Matching to JUMP uses InChIKey14 prefix (stereoisomer-tolerant).

    Args:
        input_parquet: Path to mitotox_compounds.parquet.
            Default: EXTERNAL_DATA_DIR / "mitotox" / "mitotox_compounds.parquet"
        compound_file: Path to compound.csv.gz.
            Default: EXTERNAL_DATA_DIR / "compound.csv.gz"
        output_file: Output CSV path.
            Default: INTERIM_DATA_DIR / "mitotox_processed.csv"

    Returns:
        Path to the saved output CSV file.
    """
    if input_parquet is None:
        input_parquet = EXTERNAL_DATA_DIR / "mitotox" / "mitotox_compounds.parquet"
    if compound_file is None:
        compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"
    if output_file is None:
        output_file = INTERIM_DATA_DIR / "mitotox_processed.csv"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_parquet.exists():
        raise FileNotFoundError(f"MitoTox parquet not found: {input_parquet}")
    if not compound_file.exists():
        raise FileNotFoundError(f"JUMP compound metadata not found: {compound_file}")

    df = pd.read_parquet(input_parquet)
    logger.info(f"Loaded {len(df)} MitoTox compounds")

    # Generate InChIKeys from SMILES
    df = generate_inchikeys_fast(df, smiles_col="smiles", source_name="MitoTox")

    # Deduplicate source data on InChIKey14 before matching
    df = df.dropna(subset=["standardized_inchikey"])
    df["_ik14"] = df["standardized_inchikey"].str[:14]
    n_before = len(df)
    df = df.drop_duplicates(subset=["_ik14"], keep="first").drop(columns=["_ik14"])
    logger.info(f"Deduplicated InChIKey14: {n_before} -> {len(df)}")

    # Match to JUMP compounds via InChIKey14 prefix
    jump_compounds = load_jump_compounds(compound_file)
    matched = match_to_jump(df, jump_compounds, inchikey_col="standardized_inchikey")

    # Build output with Metadata_ prefix convention
    result = pd.DataFrame(
        {
            "Metadata_JCP2022": matched["Metadata_JCP2022"],
            "Metadata_mitotox_label": matched["mitotox_label"],
            "Metadata_mitotox_toxic": (matched["mitotox_label"] == "toxic").astype(int),
            "Metadata_mitotox_mechanisms": matched["functional_mechanism"],
        }
    )

    # One row per JUMP compound
    result = result.drop_duplicates(subset=["Metadata_JCP2022"], keep="first")

    result.to_csv(output_file, index=False)

    toxic_count = result["Metadata_mitotox_toxic"].sum()
    logger.info(
        f"Saved {len(result)} compounds ({toxic_count} toxic, {len(result) - toxic_count} non-toxic) to {output_file}"
    )

    return output_file


# ---------------------------------------------------------------------------
# Demo cells
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: Toxicity and PK annotations

    Processes DILI, DICT, and PK datasets from the toxicity_pk_data.zip,
    matches to JUMP compounds via InChIKey14 prefix, and combines results.
    """)
    return


@app.cell
def _(mo):
    _input_zip = EXTERNAL_DATA_DIR / "toxicity_pk_data.zip"
    _compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"

    if _input_zip.exists() and _compound_file.exists():
        _output = process_toxicity_pk()
        _df = pd.read_csv(_output)

        _has_dili = _df["Metadata_dili_concern"].notna().sum()
        _has_dict = _df["Metadata_dict_concern"].notna().sum()
        _has_pk = _df["Metadata_pk_vdss_l_kg"].notna().sum()

        mo.md(f"""
    **Toxicity/PK results:**

    - Total JUMP compounds with any annotation: **{len(_df):,}**
    - DILI annotations: **{_has_dili:,}**
    - DICT annotations: **{_has_dict:,}**
    - PK annotations: **{_has_pk:,}**
    - Output: `{_output}`
        """)
    else:
        _df = None
        _missing = []
        if not _input_zip.exists():
            _missing.append(f"`{_input_zip}`")
        if not _compound_file.exists():
            _missing.append(f"`{_compound_file}`")
        mo.md(f"Skipped: missing {', '.join(_missing)}")
    return


@app.cell
def _(mo):
    _output = INTERIM_DATA_DIR / "toxicity_pk_processed.csv"
    if _output.exists():
        _df = pd.read_csv(_output)
        mo.ui.dataframe(_df.head(20))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: MitoTox mitochondrial toxicity

    Processes MitoTox binary toxicity labels and functional mechanisms,
    matches to JUMP compounds via InChIKey14 prefix.
    """)
    return


@app.cell
def _(mo):
    _input_parquet = EXTERNAL_DATA_DIR / "mitotox" / "mitotox_compounds.parquet"
    _compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"

    if _input_parquet.exists() and _compound_file.exists():
        _output = process_mitotox()
        _df = pd.read_csv(_output)

        _toxic = _df["Metadata_mitotox_toxic"].sum()
        _nontoxic = len(_df) - _toxic

        mo.md(f"""
    **MitoTox results:**

    - Total JUMP compounds matched: **{len(_df):,}**
    - Toxic: **{_toxic:,}**
    - Non-toxic: **{_nontoxic:,}**
    - Output: `{_output}`
        """)
    else:
        _df = None
        _missing = []
        if not _input_parquet.exists():
            _missing.append(f"`{_input_parquet}`")
        if not _compound_file.exists():
            _missing.append(f"`{_compound_file}`")
        mo.md(f"Skipped: missing {', '.join(_missing)}")
    return


@app.cell
def _(mo):
    _output = INTERIM_DATA_DIR / "mitotox_processed.csv"
    if _output.exists():
        _df = pd.read_csv(_output)
        mo.ui.dataframe(_df.head(20))
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
