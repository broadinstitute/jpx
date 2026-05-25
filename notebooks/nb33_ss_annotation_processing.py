# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb==1.5.3",
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
    from pathlib import Path

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    import duckdb
    import pandas as pd

    from nb00_ss_config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, METADATA_DB


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Annotation Processing

    Migrated from `src/jump_production/processing/` scripts into reusable `@app.function` helpers.
    Each function transforms a raw external annotation source into a clean intermediate table
    suitable for SQL import into the augmented metadata database.

    **Exported functions:**

    - `process_chembl()` - ChEMBL protein targets: wide-to-long, pipe-delimited gene lists
    - `process_repurposing_hub()` - Drug Repurposing Hub: clinical phase, MOA, target, disease area
    - `process_motive()` - MOTIVE compound-gene annotations from 8 curated databases

    All functions accept optional path overrides (default `None` resolves from `nb00_ss_config`).
    """)
    return


# ---------------------------------------------------------------------------
# Shared helpers (inlined from processing/utils.py for base-env portability)
# ---------------------------------------------------------------------------


@app.function
def _load_jump_compounds_csv(compound_file: Path | None = None) -> pd.DataFrame:
    """Load JUMP compound metadata from compound.csv.gz with InChIKey14 prefix.

    Used by process_repurposing_hub which matches via the CSV file.
    """
    if compound_file is None:
        compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"
    df = pd.read_csv(
        compound_file,
        usecols=["Metadata_JCP2022", "Metadata_InChIKey"],
    )
    df = df[df["Metadata_InChIKey"].notna()].copy()
    df["Metadata_InChIKey14"] = df["Metadata_InChIKey"].str[:14]
    return df


@app.function
def _load_jump_compounds_db(db_path: Path | None = None) -> pd.DataFrame:
    """Load JUMP compounds with InChIKey from the metadata DuckDB.

    Used by process_motive which matches via the database.
    """
    if db_path is None:
        db_path = METADATA_DB
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(
        "SELECT Metadata_JCP2022, Metadata_InChIKey FROM compound WHERE Metadata_InChIKey IS NOT NULL"
    ).fetchdf()
    con.close()
    df["inchikey_prefix"] = df["Metadata_InChIKey"].str[:14]
    return df


@app.function
def _validate_inputs(*paths: Path | str, names: list[str] | None = None) -> bool:
    """Check that all input files exist. Returns True if all present."""
    paths = [Path(p) for p in paths]
    if names is None:
        names = [p.name for p in paths]
    missing = [(name, path) for path, name in zip(paths, names, strict=False) if not path.exists()]
    for name, path in missing:
        print(f"Missing {name}: {path}")
    return len(missing) == 0


# ---------------------------------------------------------------------------
# 1. ChEMBL protein targets
# ---------------------------------------------------------------------------


@app.function
def process_chembl(
    input_csv: Path | None = None,
    output_csv: Path | None = None,
) -> Path:
    """Transform ChEMBL protein targets from wide to long format.

    The input CSV has one row per JUMP compound with binary columns for each
    gene target. This function melts it to long format, groups by compound,
    and creates a pipe-delimited target list per compound.

    Returns:
        Path to the output CSV.
    """
    if input_csv is None:
        input_csv = EXTERNAL_DATA_DIR / "targetannotations_singleproteins_jumpcompounds_all_5.csv"
    if output_csv is None:
        output_csv = INTERIM_DATA_DIR / "chembl_protein_targets_processed.csv"

    if not _validate_inputs(input_csv, names=["ChEMBL protein targets file"]):
        raise FileNotFoundError(f"ChEMBL input not found: {input_csv}")

    # Load and drop SMILES column
    df = pd.read_csv(input_csv).set_index("JUMP_ID").drop(columns=["Standardized_SMILES"])

    # Wide to long: melt binary matrix, keep only non-zero connections
    df = (
        df.stack()
        .reset_index()
        .rename(columns={0: "connection", "level_1": "gene", "JUMP_ID": "Metadata_JCP2022"})
        .query("connection != 0")
        .drop(columns=["connection"])
    )

    # Aggregate: pipe-delimited unique genes per compound
    df = (
        df.groupby("Metadata_JCP2022")
        .apply(lambda x: "|".join(pd.Series(x["gene"]).unique()), include_groups=False)
        .reset_index()
        .rename(columns={0: "Metadata_Uniprot_target"})
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


# ---------------------------------------------------------------------------
# 2. Drug Repurposing Hub
# ---------------------------------------------------------------------------


@app.function
def process_repurposing_hub(
    samples_file: Path | None = None,
    drugs_file: Path | None = None,
    compound_file: Path | None = None,
    output_file: Path | None = None,
) -> Path:
    """Map Drug Repurposing Hub annotations to JUMP IDs via InChIKey14.

    Extracts clinical_phase, moa, target, disease_area, and indication.
    Compounds are matched to JUMP via the first 14 characters of InChIKey
    (connectivity layer), providing tolerance for stereoisomer differences.

    Args:
        samples_file: Repurposing Hub samples TSV. Default:
            ``EXTERNAL_DATA_DIR / "repurposing_samples_20200324.txt"``
        drugs_file: Repurposing Hub drugs TSV. Default:
            ``EXTERNAL_DATA_DIR / "repurposing_drugs_20200324.txt"``
        compound_file: JUMP compound CSV. Default:
            ``EXTERNAL_DATA_DIR / "compound.csv.gz"``
        output_file: Where to write the result. Default:
            ``INTERIM_DATA_DIR / "repurposing_hub_annotations_processed.tsv"``

    Returns:
        DataFrame with columns ``[Metadata_JCP2022, Metadata_repurposing_name,
        Metadata_repurposing_clinical_phase, Metadata_repurposing_moa,
        Metadata_repurposing_target, Metadata_repurposing_disease_area,
        Metadata_repurposing_indication]``.
        Only rows with a non-null target are kept.
    """
    if samples_file is None:
        samples_file = EXTERNAL_DATA_DIR / "repurposing_samples_20200324.txt"
    if drugs_file is None:
        drugs_file = EXTERNAL_DATA_DIR / "repurposing_drugs_20200324.txt"
    if compound_file is None:
        compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"
    if output_file is None:
        output_file = INTERIM_DATA_DIR / "repurposing_hub_annotations_processed.tsv"

    if not _validate_inputs(
        samples_file,
        drugs_file,
        compound_file,
        names=["Repurposing Hub samples", "Repurposing Hub drugs", "JUMP compounds"],
    ):
        missing = [p for p in (samples_file, drugs_file, compound_file) if not p.exists()]
        raise FileNotFoundError(f"Missing input(s): {missing}")

    # Load samples: extract InChIKey14 for matching
    rephub_samples_df = (
        pd.read_csv(
            samples_file,
            sep="\t",
            comment="!",
            usecols=["InChIKey", "pert_iname"],
        )
        .assign(InChIKey14=lambda x: x["InChIKey"].str[:14])
        .drop_duplicates(subset=["pert_iname", "InChIKey14"])
        .drop(columns="InChIKey")
        .rename(columns={"InChIKey14": "Metadata_InChIKey14"})
        .reset_index(drop=True)
    )

    # Load drugs: clinical metadata
    rephub_drugs_df = (
        pd.read_csv(
            drugs_file,
            sep="\t",
            comment="!",
            usecols=[
                "pert_iname",
                "clinical_phase",
                "moa",
                "target",
                "disease_area",
                "indication",
            ],
        )
        .drop_duplicates(subset="pert_iname")
        .rename(
            columns={
                "clinical_phase": "Metadata_repurposing_clinical_phase",
                "moa": "Metadata_repurposing_moa",
                "target": "Metadata_repurposing_target",
                "disease_area": "Metadata_repurposing_disease_area",
                "indication": "Metadata_repurposing_indication",
            }
        )
        .reset_index(drop=True)
    )

    # Merge samples + drugs
    rephub_merged_df = rephub_samples_df.merge(rephub_drugs_df, on="pert_iname", how="inner").rename(
        columns={"pert_iname": "Metadata_repurposing_name"}
    )

    # Load JUMP compounds and match via InChIKey14
    compound_metadata_df = _load_jump_compounds_csv(compound_file)

    result = (
        compound_metadata_df.merge(rephub_merged_df, on="Metadata_InChIKey14")
        .drop(columns=["Metadata_InChIKey", "Metadata_InChIKey14"])
        .dropna(subset=["Metadata_repurposing_target"])
        .reset_index(drop=True)
    )

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False, sep="\t")
    return output_file


# ---------------------------------------------------------------------------
# 3. MOTIVE compound-gene annotations
# ---------------------------------------------------------------------------


@app.function
def _standardize_rel_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize MOTIVE relationship types to canonical lowercase names."""
    rel_type_mapping = {
        "targets": "targets",
        "binds": "binds",
        "enzyme": "enzyme",
        "transports": "transports",
        "carries": "carries",
        "upregulates": "upregulates",
        "downregulates": "downregulates",
        "unknown": "unknown",
        "inhibitor": "inhibitor",
        "blocker": "blocker",
        "antagonist": "antagonist",
        "inverse agonist": "inverse_agonist",
        "agonist": "agonist",
        "activator": "activator",
        "partial agonist": "partial_agonist",
        "modulator": "modulator",
        "allosteric modulator": "allosteric_modulator",
        "positive modulator": "positive_modulator",
        "negative modulator": "negative_modulator",
        "inhibitory allosteric modulator": "inhibitory_allosteric_modulator",
        "DRUG_CATALYSIS_GENE": "catalysis",
        "DRUG_REACTION_GENE": "reaction",
        "DRUG_INHIBITION_GENE": "inhibition",
        "DRUG_ACTIVATION_GENE": "activation",
        "DRUG_BINDACT_GENE": "binding_activation",
        "ASSOCIATES_CHaG": "associates",
        "IS_ACTIVE_ON_DNA_OR_RNA_LEVEL_CHiaodorlG": "dna_rna_active",
        "IS_ACTIVE_IN_METABOLISM_CHiaimG": "metabolism_active",
        "IS_ACTIVE_ON_CELLULAR_LEVEL_CHiaoclG": "cellular_active",
        "INCREASES_DEGENERATION_CHidG": "increases_degeneration",
        "DECREASES_DEGENERATION_CHddG": "decreases_degeneration",
    }
    df["rel_type"] = df["rel_type"].map(lambda x: rel_type_mapping.get(x, x))
    df["rel_type"] = df["rel_type"].str.replace(",", "_", regex=False)
    df["rel_type"] = df["rel_type"].str.lower()
    return df


@app.function
def process_motive(
    input_parquet: Path | None = None,
    db_path: Path | None = None,
    output_file: Path | None = None,
) -> Path:
    """Load MOTIVE compound-gene annotations and match to JUMP compounds.

    MOTIVE integrates curated drug-target interaction data from 8 databases:
    BioKG, DGIdb, DrugRep, Hetionet, OpenBioLink, OpenTargets, PharmeBiNet, PrimeKG.

    Compounds are matched to JUMP via InChIKey prefix (first 14 chars) for
    stereoisomer tolerance. Genes are aggregated per (JCP2022, rel_type, database)
    with pipe-separated gene lists.

    Args:
        input_parquet: MOTIVE annotations parquet. Default:
            ``EXTERNAL_DATA_DIR / "motive_cpd_gene_annot.parquet"``
        db_path: JUMP metadata DuckDB for compound lookup. Default:
            ``INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"``
        output_file: Where to write the result CSV. Default:
            ``INTERIM_DATA_DIR / "motive_annotations.csv"``

    Returns:
        DataFrame with columns ``[Metadata_JCP2022, Metadata_rel_type,
        Metadata_database, Metadata_motive_gene, Metadata_n_genes]``.
    """
    if input_parquet is None:
        input_parquet = EXTERNAL_DATA_DIR / "motive_cpd_gene_annot.parquet"
    if db_path is None:
        db_path = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
    if output_file is None:
        output_file = INTERIM_DATA_DIR / "motive_annotations.csv"

    if not _validate_inputs(
        input_parquet,
        names=["MOTIVE annotations parquet"],
    ):
        raise FileNotFoundError(f"MOTIVE input not found: {input_parquet}")
    if not _validate_inputs(db_path, names=["JUMP metadata database"]):
        raise FileNotFoundError(f"JUMP metadata DB not found: {db_path}")

    # Load annotations
    annotations = pd.read_parquet(input_parquet)
    if "__index_level_0__" in annotations.columns:
        annotations = annotations.drop(columns=["__index_level_0__"])
    annotations["inchikey_prefix"] = annotations["inchikey"].str[:14]

    # Load JUMP compounds from DB
    jump_compounds = _load_jump_compounds_db(db_path)

    # Match via InChIKey prefix
    merged = annotations.merge(
        jump_compounds[["Metadata_JCP2022", "inchikey_prefix"]],
        on="inchikey_prefix",
        how="inner",
    )

    # Standardize relationship types
    merged = _standardize_rel_types(merged)

    # Aggregate: one row per (compound, rel_type, database)
    aggregated = (
        merged.groupby(["Metadata_JCP2022", "rel_type", "database"])
        .agg(
            genes=("target", lambda x: "|".join(sorted(set(x)))),
            n_genes=("target", "nunique"),
        )
        .reset_index()
        .rename(
            columns={
                "rel_type": "Metadata_rel_type",
                "database": "Metadata_database",
                "genes": "Metadata_motive_gene",
                "n_genes": "Metadata_n_genes",
            }
        )
    )

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_file, index=False)
    return output_file


# ---------------------------------------------------------------------------
# Demo cells
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: ChEMBL protein targets

    Transforms the wide binary matrix (one column per gene) into a compact
    long-format table with pipe-delimited target lists per JUMP compound.
    """)
    return


@app.cell
def _(mo):
    _chembl_input = EXTERNAL_DATA_DIR / "targetannotations_singleproteins_jumpcompounds_all_5.csv"

    if _chembl_input.exists():
        df_chembl = process_chembl()
        _n_compounds = len(df_chembl)
        _avg_targets = df_chembl["Metadata_Uniprot_target"].str.count(r"\|").add(1).mean()
        mo.md(f"""
    **ChEMBL results:**

    - Compounds with targets: **{_n_compounds:,}**
    - Average targets per compound: **{_avg_targets:.1f}**
    - Columns: `{list(df_chembl.columns)}`
        """)
    else:
        df_chembl = None
        mo.md(f"Skipped: input file not found at `{_chembl_input}`")
    return (df_chembl,)


@app.cell
def _(df_chembl, mo):
    if df_chembl is not None:
        mo.ui.dataframe(df_chembl.head(20))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: Drug Repurposing Hub

    Matches Drug Repurposing Hub compounds to JUMP IDs via InChIKey14.
    Extracts clinical phase, MOA, target, disease area, and indication.
    """)
    return


@app.cell
def _(mo):
    _rephub_samples = EXTERNAL_DATA_DIR / "repurposing_samples_20200324.txt"
    _rephub_drugs = EXTERNAL_DATA_DIR / "repurposing_drugs_20200324.txt"
    _compound_csv = EXTERNAL_DATA_DIR / "compound.csv.gz"

    if all(p.exists() for p in (_rephub_samples, _rephub_drugs, _compound_csv)):
        df_rephub = process_repurposing_hub()
        _n_compounds = df_rephub["Metadata_JCP2022"].nunique()
        _n_rows = len(df_rephub)
        _cols = [c for c in df_rephub.columns if c.startswith("Metadata_repurposing_")]
        _non_null = {c.replace("Metadata_repurposing_", ""): df_rephub[c].notna().sum() for c in _cols}
        mo.md(f"""
    **Repurposing Hub results:**

    - Rows: **{_n_rows:,}** (compounds x annotations)
    - Unique JUMP compounds matched: **{_n_compounds:,}**
    - Non-null counts: {_non_null}
        """)
    else:
        df_rephub = None
        mo.md("Skipped: one or more Repurposing Hub input files not found")
    return (df_rephub,)


@app.cell
def _(df_rephub, mo):
    if df_rephub is not None:
        mo.ui.dataframe(df_rephub.head(20))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: MOTIVE compound-gene annotations

    Integrates curated drug-target interactions from 8 databases (BioKG, DGIdb,
    DrugRep, Hetionet, OpenBioLink, OpenTargets, PharmeBiNet, PrimeKG).
    Matches to JUMP via InChIKey prefix and aggregates genes per
    (compound, relationship type, database).
    """)
    return


@app.cell
def _(mo):
    _motive_input = EXTERNAL_DATA_DIR / "motive_cpd_gene_annot.parquet"
    _db = METADATA_DB

    if _motive_input.exists() and _db.exists():
        df_motive = process_motive()
        _n_compounds = df_motive["Metadata_JCP2022"].nunique()
        _n_rows = len(df_motive)
        _n_databases = df_motive["Metadata_database"].nunique()
        _db_counts = df_motive["Metadata_database"].value_counts().to_dict()
        _rel_counts = df_motive["Metadata_rel_type"].value_counts().head(10).to_dict()
        mo.md(f"""
    **MOTIVE results:**

    - Records (compound x rel_type x database): **{_n_rows:,}**
    - Unique JUMP compounds matched: **{_n_compounds:,}**
    - Databases ({_n_databases}): {_db_counts}
    - Top 10 relationship types: {_rel_counts}
        """)
    else:
        df_motive = None
        _missing = []
        if not _motive_input.exists():
            _missing.append(f"`{_motive_input}`")
        if not _db.exists():
            _missing.append(f"`{_db}`")
        mo.md(f"Skipped: missing {', '.join(_missing)}")
    return (df_motive,)


@app.cell
def _(df_motive, mo):
    if df_motive is not None:
        mo.ui.dataframe(df_motive.head(20))
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
