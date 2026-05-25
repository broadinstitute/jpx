# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "python-dotenv",
#     "loguru==0.7.3",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import os
    import sys
    from pathlib import Path

    import pandas as pd
    from loguru import logger

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import EXTERNAL_DATA_DIR


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Processing Utilities

    Shared utility functions for data processing, migrated from
    `src/jump_production/processing/utils.py`.

    Other notebooks import via `from nb32_ss_processing_utils import load_jump_compounds, match_to_jump, ...`

    **Exported functions:**
    - `load_jump_compounds(compound_file, columns)` - load JUMP compound metadata with InChIKey14 prefix
    - `match_to_jump(df, jump_compounds, inchikey_col, how, drop_inchikey14)` - match external data to JUMP via InChIKey14
    - `validate_inputs(*paths, names)` - check file existence with logging
    - `standardize_smiles(df, smiles_col, num_cpus, source_name)` - SMILES standardization (requires jump_smiles)
    - `generate_inchikeys_fast(df, smiles_col, source_name)` - RDKit InChIKey generation
    """)
    return


@app.function
def load_jump_compounds(
    compound_file: Path | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load JUMP compound metadata with InChIKey14 prefix for matching.

    Args:
        compound_file: Path to compound.csv.gz. Defaults to EXTERNAL_DATA_DIR / "compound.csv.gz"
        columns: Additional columns to load beyond Metadata_JCP2022 and Metadata_InChIKey.
                 These columns are always included.

    Returns:
        DataFrame with Metadata_JCP2022, Metadata_InChIKey, Metadata_InChIKey14, and any
        additional requested columns. Rows with null InChIKey are excluded.

    Example:
        >>> jump_compounds = load_jump_compounds()
        >>> jump_compounds.columns.tolist()
        ['Metadata_JCP2022', 'Metadata_InChIKey', 'Metadata_InChIKey14']
    """
    if compound_file is None:
        compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"

    # Always include these base columns
    base_cols = ["Metadata_JCP2022", "Metadata_InChIKey"]
    usecols = list(set(base_cols + (columns or [])))

    logger.info(f"Loading JUMP compounds from {compound_file}")
    df = pd.read_csv(compound_file, usecols=usecols)

    # Filter to compounds with InChIKey and create prefix
    df = df[df["Metadata_InChIKey"].notna()].copy()
    df["Metadata_InChIKey14"] = df["Metadata_InChIKey"].str[:14]

    logger.info(f"Loaded {len(df)} JUMP compounds with InChIKey")
    return df


@app.function
def match_to_jump(
    df: pd.DataFrame,
    jump_compounds: pd.DataFrame | None = None,
    inchikey_col: str = "inchikey",
    how: str = "inner",
    drop_inchikey14: bool = True,
) -> pd.DataFrame:
    """Match external data to JUMP compounds via InChIKey14 prefix.

    Uses the first 14 characters of InChIKey (connectivity layer) for matching,
    which provides tolerance for stereochemistry differences between labs.

    Args:
        df: DataFrame with InChIKey column to match
        jump_compounds: JUMP compound metadata. If None, loads automatically.
        inchikey_col: Name of the InChIKey column in df
        how: Join type ('inner', 'left', 'right', 'outer'). Default 'inner'.
        drop_inchikey14: If True, drop the temporary InChIKey14 columns after merge

    Returns:
        DataFrame with Metadata_JCP2022 column added (and optionally Metadata_InChIKey)

    Example:
        >>> external_df = pd.DataFrame({"compound": ["A", "B"], "inchikey": ["ABC...", "XYZ..."]})
        >>> matched = match_to_jump(external_df, inchikey_col="inchikey")
    """
    if jump_compounds is None:
        jump_compounds = load_jump_compounds()

    # Create InChIKey14 prefix for matching
    df = df.copy()
    df["_inchikey14"] = df[inchikey_col].str[:14]

    # Merge on InChIKey14
    result = df.merge(
        jump_compounds[["Metadata_JCP2022", "Metadata_InChIKey14"]],
        left_on="_inchikey14",
        right_on="Metadata_InChIKey14",
        how=how,
    )

    n_matched = result["Metadata_JCP2022"].notna().sum()
    n_unique = result["Metadata_JCP2022"].nunique()
    logger.info(f"Matched {n_matched} records to {n_unique} unique JUMP compounds")

    # Clean up temporary columns
    if drop_inchikey14:
        result = result.drop(columns=["_inchikey14", "Metadata_InChIKey14"], errors="ignore")

    return result


@app.function
def validate_inputs(*paths: Path, names: list[str] | None = None) -> bool:
    """Validate that input files exist, logging errors for missing files.

    Args:
        *paths: Paths to validate
        names: Optional descriptive names for each path (for error messages).
               If not provided, uses the filename.

    Returns:
        True if all files exist, False otherwise

    Example:
        >>> if not validate_inputs(input_file, compound_file, names=["input data", "compound metadata"]):
        ...     print("Missing files!")
    """
    if names is None:
        names = [p.name for p in paths]

    missing = []
    for path, name in zip(paths, names, strict=False):
        if not path.exists():
            logger.error(f"{name} not found: {path}")
            missing.append(path)

    return len(missing) == 0


@app.function
def standardize_smiles(
    df: pd.DataFrame,
    smiles_col: str,
    num_cpus: int | None = None,
    source_name: str = "compounds",
) -> pd.DataFrame:
    """Standardize SMILES and generate InChIKeys using jump_smiles.

    Uses the jump_canonical method which is the JUMP consortium standard for
    consistent compound matching.

    Args:
        df: DataFrame with SMILES column
        smiles_col: Name of the SMILES column
        num_cpus: Number of CPUs for parallel processing. Defaults to 50% of available.
        source_name: Name for logging (e.g., "DILI", "kinase probes")

    Returns:
        df with standardized_smiles and standardized_inchikey columns added

    Note:
        Requires the jump-smiles environment. For fast testing, use
        generate_inchikeys_fast() instead.
    """
    # Lazy import - only needed when this function is called
    from jump_smiles.standardize_smiles import StandardizeMolecule

    logger.info(f"Standardizing SMILES for {source_name}...")

    # Prepare DataFrame for jump_smiles (requires 'SMILES' column)
    smiles_df = pd.DataFrame({"SMILES": df[smiles_col].dropna()})

    if len(smiles_df) == 0:
        logger.warning(f"No valid SMILES to standardize for {source_name}")
        df = df.copy()
        df["standardized_smiles"] = None
        df["standardized_inchikey"] = None
        return df

    # Set number of CPUs
    if num_cpus is None:
        num_cpus = max((os.cpu_count() or 1) // 2, 1)
    logger.info(f"Using {num_cpus} CPUs for SMILES standardization")

    # Run standardization
    standardizer = StandardizeMolecule(
        input=smiles_df,
        method="jump_canonical",
        num_cpu=num_cpus,
    )
    standardized_results = standardizer.run()

    # Create mappings from original to standardized
    smiles_map = dict(
        zip(
            standardized_results["SMILES_original"],
            standardized_results["SMILES_standardized"],
            strict=False,
        )
    )
    inchikey_map = dict(
        zip(
            standardized_results["SMILES_original"],
            standardized_results["InChIKey_standardized"],
            strict=False,
        )
    )

    # Apply to DataFrame
    df = df.copy()
    df["standardized_smiles"] = df[smiles_col].map(smiles_map)
    df["standardized_inchikey"] = df[smiles_col].map(inchikey_map)

    success_count = df["standardized_smiles"].notna().sum()
    logger.success(f"Standardized {success_count}/{len(df)} SMILES for {source_name}")

    return df


@app.function
def generate_inchikeys_fast(
    df: pd.DataFrame,
    smiles_col: str,
    source_name: str = "compounds",
) -> pd.DataFrame:
    """Generate InChIKeys using RDKit directly (fast mode).

    This is ~240x faster than standardize_smiles() but may find ~10-15% fewer
    matches due to different standardization than jump_canonical.

    Args:
        df: DataFrame with SMILES column
        smiles_col: Name of the SMILES column
        source_name: Name for logging (e.g., "DILI", "kinase probes")

    Returns:
        df with standardized_inchikey column added (no standardized_smiles)

    Note:
        Use for development/testing. For production, use standardize_smiles()
        with jump_canonical method for maximum JUMP compound matches.
    """
    # Lazy import - only needed when this function is called
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchiKey

    logger.info(f"Generating InChIKeys with RDKit for {source_name} (fast mode)...")

    def smiles_to_inchikey(smiles):
        """Convert SMILES to InChIKey using RDKit."""
        if pd.isna(smiles):
            return None
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return None
            return MolToInchiKey(mol)
        except Exception:
            return None

    df = df.copy()
    df["standardized_inchikey"] = df[smiles_col].apply(smiles_to_inchikey)

    success_count = df["standardized_inchikey"].notna().sum()
    logger.success(f"Generated {success_count}/{len(df)} InChIKeys for {source_name}")

    return df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo

    Usage examples for the exported functions.
    """)
    return


@app.cell
def _(mo):
    # Demo: load JUMP compounds and inspect
    compounds = load_jump_compounds()
    mo.vstack(
        [
            mo.md(f"**Loaded {len(compounds):,} JUMP compounds**"),
            mo.md(f"Columns: `{compounds.columns.tolist()}`"),
            compounds.head(),
        ]
    )
    return (compounds,)


@app.cell
def _(mo, compounds):
    # Demo: validate_inputs
    compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"
    fake_file = EXTERNAL_DATA_DIR / "does_not_exist.csv"

    valid = validate_inputs(compound_file, names=["compound metadata"])
    invalid = validate_inputs(fake_file, names=["fake file"])

    mo.md(f"""
    **validate_inputs demo:**
    - compound.csv.gz exists: `{valid}`
    - does_not_exist.csv exists: `{invalid}`
    """)
    return


@app.cell
def _(mo, compounds):
    # Demo: match_to_jump with synthetic data
    sample_inchikeys = compounds["Metadata_InChIKey"].dropna().head(5).tolist()
    demo_df = pd.DataFrame(
        {
            "compound_name": [f"compound_{i}" for i in range(5)],
            "inchikey": sample_inchikeys,
        }
    )

    matched = match_to_jump(demo_df, jump_compounds=compounds, inchikey_col="inchikey")
    mo.vstack(
        [
            mo.md(f"**Matched {len(matched)} compounds to JUMP**"),
            matched,
        ]
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
