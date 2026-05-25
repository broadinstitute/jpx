# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "openpyxl",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import json
    import sys
    import urllib.error
    import urllib.request
    from pathlib import Path

    import pandas as pd
    from loguru import logger

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
    from nb32_ss_processing_utils import (  # noqa: F401
        generate_inchikeys_fast,
        load_jump_compounds,
        match_to_jump,
        standardize_smiles,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Probe Curation

    Migrated from `src/jump_production/processing/chemical_probes.py` and
    `src/jump_production/processing/kinase_probes.py` into reusable `@app.function` helpers.

    **Exported functions:**

    - `process_chemical_probes()` - Probe & Drugs Portal chemical probes: target aggregation,
      SMILES standardization, deduplication, JUMP matching, quality flags
    - `process_kinase_probes()` - KCGS and PKIS kinase probe sets: load from Excel/ChEMBL,
      InChIKey generation, JUMP matching

    All functions accept optional path overrides (default `None` resolves from `nb00_ss_config`).
    This is a cheminformatics-env notebook (uses RDKit via nb32 utils).
    """)
    return


# ---------------------------------------------------------------------------
# ChEMBL API config for PKIS
# ---------------------------------------------------------------------------

CHEMBL_BASE_URL = "https://www.ebi.ac.uk"
CHEMBL_PKIS_URL = f"{CHEMBL_BASE_URL}/chembl/api/data/molecule.json?document_chembl_id=CHEMBL2303647&limit=1000"


# ---------------------------------------------------------------------------
# Internal helpers for chemical probes
# ---------------------------------------------------------------------------


@app.function
def _process_targets_and_merge(df: pd.DataFrame, targets_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate target genes per probe and merge with compounds data.

    Groups targets by pdid, flattens comma-separated gene names, removes
    uncharacterized protein placeholders ("-"), and creates a pipe-delimited
    gene list column (Metadata_chmprb_target_genes).
    """

    def flatten_and_dedupe_genes(gene_series):
        """Flatten comma-separated genes, deduplicate, and sort.

        Note: "-" entries represent uncharacterized proteins. We remove these
        because they would incorrectly match between compounds that have
        different uncharacterized proteins as targets.
        """
        all_genes = set()
        for gene_entry in gene_series.dropna():
            genes = [g.strip() for g in gene_entry.split(",")]
            all_genes.update(genes)
        all_genes.discard("")
        all_genes.discard("-")
        return "|".join(sorted(all_genes)) if all_genes else None

    target_aggregation = (
        targets_df.groupby("pdid")["gene_name"].apply(flatten_and_dedupe_genes).reset_index(name="target_genes")
    )

    df = df.merge(target_aggregation, on="pdid", how="left")
    df = df.rename(columns={"target_genes": "Metadata_chmprb_target_genes"})

    probes_with_targets = df[df["probe"] == 1]["Metadata_chmprb_target_genes"].notna().sum()
    total_probes = (df["probe"] == 1).sum()
    logger.info(f"{probes_with_targets}/{total_probes} probes have target gene annotations")

    return df


@app.function
def _deduplicate_by_inchikey(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate probes that standardize to the same InChIKey.

    When multiple probes map to the same InChIKey:
    1. Merge target genes (union)
    2. Keep probe with best quality metrics (P&D approved > experimental > highest QED)
    3. Track which pdids were merged
    """
    dup_mask = df["standardized_inchikey"].duplicated(keep=False)
    n_dups = dup_mask.sum()

    if n_dups == 0:
        logger.info("No duplicate InChIKeys found after standardization")
        return df

    logger.warning(f"Found {n_dups} probes with duplicate InChIKeys")

    deduped_rows = []
    merged_count = 0

    for inchikey, group in df.groupby("standardized_inchikey"):
        if len(group) > 1:
            merged_count += 1

            # Merge target genes (union of all targets)
            all_targets = set()
            for _, row in group.iterrows():
                if pd.notna(row["Metadata_chmprb_target_genes"]):
                    all_targets.update(row["Metadata_chmprb_target_genes"].split("|"))
            all_targets.discard("")
            all_targets.discard("-")
            merged_targets = "|".join(sorted(all_targets)) if all_targets else None

            # Pick best row based on quality metrics
            best_row = (
                group.sort_values(
                    ["P&D approved", "experimental probe", "qed"],
                    ascending=False,
                    na_position="last",
                )
                .iloc[0]
                .copy()
            )

            best_row["Metadata_chmprb_target_genes"] = merged_targets
            best_row["no. targets"] = len(all_targets) if all_targets else 0

            other_pdids = group[group["pdid"] != best_row["pdid"]]["pdid"].tolist()
            if other_pdids:
                best_row["merged_from_pdids"] = "|".join(other_pdids)

            logger.info(
                f"Merged {len(group)} probes with InChIKey {inchikey}: keeping {best_row['pdid']} ({best_row['name']})"
            )

            deduped_rows.append(best_row)
        else:
            deduped_rows.append(group.iloc[0])

    result_df = pd.DataFrame(deduped_rows).reset_index(drop=True)
    logger.info(f"Deduplication complete: {len(df)} -> {len(result_df)} probes ({merged_count} merges)")

    return result_df


# ---------------------------------------------------------------------------
# Internal helpers for kinase probes
# ---------------------------------------------------------------------------


@app.function
def _load_kcgs(input_file: Path) -> pd.DataFrame:
    """Load KCGS compounds from Zenodo Excel file.

    Returns DataFrame with columns: probe_id, probe_name, smiles,
    probe_set, selectivity_s10, is_original_kcgs, target_info.
    """
    logger.info(f"Loading KCGS from {input_file}")
    df = pd.read_excel(input_file, sheet_name="Data summary")

    # Filter out empty SMILES
    df = df[df["SMILES string"].notna() & (df["SMILES string"] != "EMPTY")].copy()

    # Convert yes/no to 1/0 for is_original_kcgs
    is_original_col = df.get("member of original KCGS", pd.Series([0] * len(df)))
    is_original = is_original_col.map({"yes": 1, "no": 0, 1: 1, 0: 0}).fillna(0).astype(int)

    result = pd.DataFrame(
        {
            "probe_id": df["compound"],
            "probe_name": df["compound"],
            "smiles": df["SMILES string"],
            "probe_set": "KCGS",
            "selectivity_s10": df.get("S10 (1 uM) from Discoverx data"),
            "is_original_kcgs": is_original,
            "target_info": df.get(
                "target data from our lab: generally  Discoverx Kd<100 nM "
                "and/or Nanosyn %I>90 (screened at 1 uM); some rows have "
                "literature data only, or a combination of literature and "
                "our data"
            ),
        }
    )

    logger.info(f"Loaded {len(result)} KCGS compounds")
    return result


@app.function
def _load_pkis_raw(cache_file: Path | None = None, max_compounds: int = 2000) -> list:
    """Load raw PKIS molecule data from cache or ChEMBL API.

    Returns list of raw molecule dicts from ChEMBL API.
    """
    if cache_file and cache_file.exists():
        logger.info(f"Loading PKIS from cache: {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    logger.info("Fetching PKIS from ChEMBL API...")
    molecules = []

    url = CHEMBL_PKIS_URL
    while url and len(molecules) < max_compounds:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode())
            molecules.extend(data.get("molecules", []))
            next_url = data.get("page_meta", {}).get("next")
            if next_url and not next_url.startswith("http"):
                url = f"{CHEMBL_BASE_URL}{next_url}"
            else:
                url = next_url

    molecules = molecules[:max_compounds]
    logger.info(f"Fetched {len(molecules)} molecules from ChEMBL")

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(molecules, f)
        logger.info(f"Cached PKIS data to {cache_file}")

    return molecules


@app.function
def _fetch_pkis_from_chembl(cache_file: Path | None = None, max_compounds: int = 2000) -> pd.DataFrame:
    """Fetch PKIS compounds from ChEMBL API and return as DataFrame.

    Returns DataFrame with columns: probe_id, probe_name, smiles,
    probe_set, selectivity_s10, is_original_kcgs, target_info.
    """
    molecules = _load_pkis_raw(cache_file, max_compounds)

    pkis_data = []
    for mol in molecules:
        chembl_id = mol.get("molecule_chembl_id")
        structures = mol.get("molecule_structures") or {}
        smiles = structures.get("canonical_smiles")
        pref_name = mol.get("pref_name")

        if smiles:
            pkis_data.append(
                {
                    "probe_id": chembl_id,
                    "probe_name": pref_name or chembl_id,
                    "smiles": smiles,
                    "probe_set": "PKIS",
                    "selectivity_s10": None,
                    "is_original_kcgs": 0,
                    "target_info": None,
                }
            )

    result = pd.DataFrame(pkis_data)
    logger.info(f"Loaded {len(result)} PKIS compounds with SMILES")
    return result


@app.function
def _generate_inchikeys_fast_for_kinase_probes(kcgs_df: pd.DataFrame, pkis_mols: list) -> pd.DataFrame:
    """Generate InChIKeys without jump_canonical standardization (fast mode).

    Fast mode (~1 sec vs ~4 min) with ~12% fewer matches:
    - KCGS: Uses RDKit to generate InChIKey directly from SMILES
    - PKIS: Uses ChEMBL's pre-computed standard_inchi_key
    """
    logger.info("Fast mode: generating InChIKeys without jump_canonical standardization")
    logger.warning("Fast mode finds ~12% fewer matches (106 vs 120) but runs 240x faster")

    # KCGS: Use shared utility for RDKit InChIKey generation
    kcgs_with_inchikey = generate_inchikeys_fast(kcgs_df, "smiles", source_name="KCGS")

    # PKIS: Use ChEMBL's standard_inchi_key directly (pre-computed)
    pkis_data = []
    for mol in pkis_mols:
        chembl_id = mol.get("molecule_chembl_id")
        structures = mol.get("molecule_structures") or {}
        smiles = structures.get("canonical_smiles")
        inchikey = structures.get("standard_inchi_key")
        pref_name = mol.get("pref_name")

        if smiles:
            pkis_data.append(
                {
                    "probe_id": chembl_id,
                    "probe_name": pref_name or chembl_id,
                    "smiles": smiles,
                    "probe_set": "PKIS",
                    "selectivity_s10": None,
                    "is_original_kcgs": 0,
                    "target_info": None,
                    "standardized_inchikey": inchikey,
                }
            )
    pkis_df = pd.DataFrame(pkis_data)
    logger.info(f"PKIS: {pkis_df['standardized_inchikey'].notna().sum()}/{len(pkis_df)} with ChEMBL InChIKey")

    return pd.concat([kcgs_with_inchikey, pkis_df], ignore_index=True)


@app.function
def _match_probes_to_jump(df: pd.DataFrame, jump_compounds: pd.DataFrame) -> pd.DataFrame:
    """Match probe compounds to JUMP via InChIKey connectivity layer (first 14 chars).

    Deduplicates on (probe_id, Metadata_JCP2022) to handle stereoisomer matches.
    """
    df = df[df["standardized_inchikey"].notna()].copy()
    df["connectivity"] = df["standardized_inchikey"].str[:14]

    matched = df.merge(
        jump_compounds[["Metadata_JCP2022", "Metadata_InChIKey14"]],
        left_on="connectivity",
        right_on="Metadata_InChIKey14",
        how="inner",
    )

    matched = matched.drop_duplicates(subset=["probe_id", "Metadata_JCP2022"])
    matched = matched.drop(columns=["connectivity", "Metadata_InChIKey14"])

    logger.info(
        f"Matched {matched['probe_id'].nunique()} probes to {matched['Metadata_JCP2022'].nunique()} JUMP compounds"
    )

    return matched


# ---------------------------------------------------------------------------
# 1. Chemical probes (Probe & Drugs Portal)
# ---------------------------------------------------------------------------


@app.function
def process_chemical_probes(
    input_xlsx: Path | None = None,
    compound_file: Path | None = None,
    output_probes: Path | None = None,
    output_targets: Path | None = None,
    skip_standardization: bool = False,
) -> dict[str, Path]:
    """Process Probe & Drugs Portal chemical probes.

    Loads the multi-sheet Excel export from probes-drugs.org, aggregates target
    genes per probe, optionally standardizes SMILES, deduplicates by InChIKey,
    matches to JUMP compounds, and adds quality flags.

    Args:
        input_xlsx: Probe & Drugs Portal Excel export. Default:
            ``EXTERNAL_DATA_DIR / "pd_export_01_2025_875_targets_standardized.xlsx"``
        compound_file: JUMP compound CSV. Default:
            ``EXTERNAL_DATA_DIR / "compound.csv.gz"``
        output_probes: Where to write processed probes CSV. Default:
            ``INTERIM_DATA_DIR / "chemical_probes_processed.csv"``
        output_targets: Where to write probe-target pairs CSV. Default:
            ``INTERIM_DATA_DIR / "chemical_probes_targets.csv"``
        skip_standardization: If True (default), use original InChIKeys instead of
            running slow SMILES standardization via jump_canonical.

    Returns:
        Dict with keys ``probes`` and ``targets`` mapping to output file Paths.
    """
    if input_xlsx is None:
        input_xlsx = EXTERNAL_DATA_DIR / "pd_export_01_2025_875_targets_standardized.xlsx"
    if compound_file is None:
        compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"
    if output_probes is None:
        output_probes = INTERIM_DATA_DIR / "chemical_probes_processed.csv"
    if output_targets is None:
        output_targets = INTERIM_DATA_DIR / "chemical_probes_targets.csv"

    for path, name in [
        (input_xlsx, "Chemical probes Excel file"),
        (compound_file, "JUMP compound metadata"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    output_probes.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Processing chemical probes data...")
    if skip_standardization:
        logger.info("FAST MODE: Skipping SMILES standardization")

    # Load chemical probes data
    logger.info(f"Loading data from {input_xlsx}")
    df = pd.read_excel(input_xlsx, sheet_name="COMPOUNDS")
    targets_df = pd.read_excel(input_xlsx, sheet_name="TARGETS")
    logger.info(f"Loaded {len(df)} chemical probes and {len(targets_df)} probe-target pairs")

    # Load JUMP compound metadata
    jump_compounds = load_jump_compounds(compound_file)

    # STEP 1: Process targets (fast)
    df = _process_targets_and_merge(df, targets_df)

    # STEP 2: Standardize SMILES (slow - can be skipped)
    if skip_standardization:
        logger.info("Using original InChIKeys (not standardized)")
        df["standardized_smiles"] = df["smiles"]
        df["standardized_inchikey"] = df["inchikey"]
    else:
        df = standardize_smiles(df, "smiles", source_name="chemical probes")

    # Filter for actual chemical probes
    probes_df = df[df["probe"] == 1].copy()
    logger.info(f"Found {len(probes_df)} chemical probes")

    # STEP 3: Deduplicate probes with same InChIKey
    probes_df = _deduplicate_by_inchikey(probes_df)

    # STEP 4: Match to JUMP compounds via InChIKey
    logger.info("Matching chemical probes to JUMP compounds via InChIKey...")
    probes_df = probes_df.merge(
        jump_compounds,
        left_on="standardized_inchikey",
        right_on="Metadata_InChIKey",
        how="left",
    )

    matched_probes = probes_df["Metadata_JCP2022"].notna().sum()
    logger.info(f"{matched_probes}/{len(probes_df)} chemical probes matched to JUMP compounds")

    # Drop redundant InChIKey columns from merge
    probes_df = probes_df.drop(columns=["Metadata_InChIKey", "Metadata_InChIKey14"], errors="ignore")

    # STEP 5: Clean up columns and add Metadata_chmprb_ prefix
    probes_df = probes_df.rename(
        columns={
            "experimental probe": "experimental_probe",
            "calculated probe": "calculated_probe",
            "approved drug": "approved_drug",
            "P&D approved": "pd_approved",
            "covalent binder": "covalent_binder",
            "biased GPCR ligand": "biased_gpcr_ligand",
            "structural alert": "structural_alert",
            "PAINS Family A": "pains_family_a",
            "PAINS Family B": "pains_family_b",
            "PAINS Family C": "pains_family_c",
            "Drug Status": "drug_status",
            "no. targets": "num_targets",
            "PROTAC": "protac",
            "Aggregator": "aggregator",
            "Obsolete": "obsolete",
            "Nuisance": "nuisance",
        }
    )

    # Add Metadata_chmprb_ prefix to all columns except Metadata_JCP2022
    # and Metadata_chmprb_target_genes
    columns_to_rename = {}
    for col in probes_df.columns:
        if col in ("Metadata_JCP2022", "Metadata_chmprb_target_genes"):
            continue
        columns_to_rename[col] = f"Metadata_chmprb_{col}"

    probes_df = probes_df.rename(columns=columns_to_rename)

    # Convert binary flags to proper integers (0/1)
    binary_columns = [
        "Metadata_chmprb_probe",
        "Metadata_chmprb_experimental_probe",
        "Metadata_chmprb_calculated_probe",
        "Metadata_chmprb_available",
        "Metadata_chmprb_approved_drug",
        "Metadata_chmprb_pd_approved",
        "Metadata_chmprb_protac",
        "Metadata_chmprb_covalent_binder",
        "Metadata_chmprb_biased_gpcr_ligand",
        "Metadata_chmprb_inorganic",
        "Metadata_chmprb_structural_alert",
        "Metadata_chmprb_pains_family_a",
        "Metadata_chmprb_pains_family_b",
        "Metadata_chmprb_pains_family_c",
        "Metadata_chmprb_aggregator",
        "Metadata_chmprb_obsolete",
        "Metadata_chmprb_nuisance",
    ]

    for col in binary_columns:
        if col in probes_df.columns:
            probes_df[col] = probes_df[col].fillna(0).astype(int)

    # Add computed quality flags
    probes_df["Metadata_chmprb_has_quality_alerts"] = (
        (probes_df.get("Metadata_chmprb_structural_alert", 0) > 0)
        | (probes_df.get("Metadata_chmprb_pains_family_a", 0) > 0)
        | (probes_df.get("Metadata_chmprb_pains_family_b", 0) > 0)
        | (probes_df.get("Metadata_chmprb_pains_family_c", 0) > 0)
        | (probes_df.get("Metadata_chmprb_aggregator", 0) > 0)
        | (probes_df.get("Metadata_chmprb_obsolete", 0) > 0)
        | (probes_df.get("Metadata_chmprb_nuisance", 0) > 0)
    ).astype(int)

    probes_df["Metadata_chmprb_is_high_quality"] = (
        (probes_df["Metadata_chmprb_experimental_probe"] == 1)
        & (probes_df["Metadata_chmprb_pd_approved"] == 1)
        & (probes_df["Metadata_chmprb_qed"] >= 0.5)
        & (probes_df["Metadata_chmprb_has_quality_alerts"] == 0)
    ).astype(int)

    # Flag for whether probe is in JUMP compound library
    probes_df["Metadata_chmprb_in_jump"] = probes_df["Metadata_JCP2022"].notna().astype(int)

    # Reorganize: Metadata_JCP2022 first
    cols = list(probes_df.columns)
    if "Metadata_JCP2022" in cols:
        cols.remove("Metadata_JCP2022")
        cols = ["Metadata_JCP2022"] + cols
        probes_df = probes_df[cols]

    # Save processed probes
    probes_df.to_csv(output_probes, index=False)
    logger.info(f"Saved {len(probes_df)} chemical probes to {output_probes}")

    # Save detailed targets file (filter to probes that remain after dedup)
    remaining_pdids = set(probes_df["Metadata_chmprb_pdid"])
    if "Metadata_chmprb_merged_from_pdids" in probes_df.columns:
        remaining_pdids.update(probes_df["Metadata_chmprb_merged_from_pdids"].dropna().str.split("|").explode())

    targets_filtered = targets_df[targets_df["pdid"].isin(remaining_pdids)].copy()

    targets_output = targets_filtered[
        [
            "pdid",
            "name",
            "gene_name",
            "target_name",
            "target_type",
            "moa",
            "activity_biochemical",
        ]
    ].rename(
        columns={
            "pdid": "Metadata_chmprb_pdid",
            "name": "Metadata_chmprb_probe_name",
            "gene_name": "Metadata_chmprb_gene_name",
            "target_name": "Metadata_chmprb_target_name",
            "target_type": "Metadata_chmprb_target_type",
            "moa": "Metadata_chmprb_moa",
            "activity_biochemical": "Metadata_chmprb_activity_biochemical",
        }
    )

    targets_output.to_csv(output_targets, index=False)
    logger.info(f"Saved {len(targets_output)} probe-target pairs to {output_targets}")

    # Summary
    logger.info(f"Total probes processed: {len(probes_df)}")
    logger.info(f"Total probe-target pairs: {len(targets_output)}")
    logger.info(f"Unique target genes: {targets_df['gene_name'].nunique()}")

    target_counts = probes_df["Metadata_chmprb_target_genes"].str.count(r"\|").fillna(-1) + 1
    logger.info(f"Single-target probes: {(target_counts == 1).sum()}")
    logger.info(f"Multi-target probes (2-5): {((target_counts >= 2) & (target_counts <= 5)).sum()}")
    logger.info(f"Promiscuous probes (>5): {(target_counts > 5).sum()}")

    return {"probes": output_probes, "targets": output_targets}


# ---------------------------------------------------------------------------
# 2. Kinase probes (KCGS + PKIS)
# ---------------------------------------------------------------------------


@app.function
def process_kinase_probes(
    kcgs_file: Path | None = None,
    compound_file: Path | None = None,
    output_file: Path | None = None,
    fast: bool = True,
) -> Path:
    """Process KCGS and PKIS kinase probe sets and match to JUMP compounds.

    KCGS (Kinase Chemogenomic Set) v2.0: 295 highly selective kinase inhibitors
    (Kd < 100 nM, S10 < 0.04) from SGC-UNC.

    PKIS (Published Kinase Inhibitor Set): 367 compounds from GSK drug discovery
    programs, fetched from ChEMBL (document CHEMBL2303647).

    Args:
        kcgs_file: Path to KCGS Excel file. Default:
            ``EXTERNAL_DATA_DIR / "kcgs_v2.xlsx"``
        compound_file: JUMP compound CSV. Default:
            ``EXTERNAL_DATA_DIR / "compound.csv.gz"``
        output_file: Where to write the result CSV. Default:
            ``INTERIM_DATA_DIR / "kinase_probes.csv"``
        fast: If True (default), use RDKit/ChEMBL InChIKeys without
            jump_canonical standardization (~1s vs ~4min, ~12% fewer matches).

    Returns:
        Path to the output CSV file.
    """
    if kcgs_file is None:
        kcgs_file = EXTERNAL_DATA_DIR / "kcgs_v2.xlsx"
    if compound_file is None:
        compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"
    if output_file is None:
        output_file = INTERIM_DATA_DIR / "kinase_probes.csv"

    pkis_cache = EXTERNAL_DATA_DIR / "pkis_chembl_cache.json"

    for path, name in [
        (kcgs_file, "KCGS Excel file"),
        (compound_file, "JUMP compound metadata"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Processing kinase probe sets (KCGS and PKIS)...")
    if fast:
        logger.info("Fast mode enabled: using RDKit/ChEMBL InChIKeys without jump_canonical")

    # Load JUMP compounds
    jump_compounds = load_jump_compounds(compound_file)

    # Load KCGS
    kcgs_df = _load_kcgs(kcgs_file)

    # Load/fetch PKIS data
    pkis_raw = _load_pkis_raw(cache_file=pkis_cache)

    if fast:
        # Fast mode: RDKit for KCGS, ChEMBL InChIKey for PKIS
        combined = _generate_inchikeys_fast_for_kinase_probes(kcgs_df, pkis_raw)
    else:
        # Full mode: standardize all SMILES with jump_canonical
        pkis_df = _fetch_pkis_from_chembl(cache_file=pkis_cache)
        combined = pd.concat([kcgs_df, pkis_df], ignore_index=True)
        logger.info(f"Combined: {len(combined)} compounds ({len(kcgs_df)} KCGS + {len(pkis_df)} PKIS)")
        combined = standardize_smiles(combined, "smiles", source_name="kinase probes")

    # Match to JUMP
    matched = _match_probes_to_jump(combined, jump_compounds)

    # Create output with probe set flags
    output = matched[
        [
            "Metadata_JCP2022",
            "probe_set",
            "probe_id",
            "probe_name",
            "standardized_inchikey",
            "selectivity_s10",
            "is_original_kcgs",
            "target_info",
        ]
    ].copy()

    output = output.rename(
        columns={
            "probe_set": "Metadata_kinase_probe_set",
            "probe_id": "Metadata_kinase_probe_id",
            "probe_name": "Metadata_kinase_probe_name",
            "standardized_inchikey": "Metadata_kinase_probe_inchikey",
            "selectivity_s10": "Metadata_kinase_selectivity_s10",
            "is_original_kcgs": "Metadata_kinase_is_original_kcgs",
            "target_info": "Metadata_kinase_target_info",
        }
    )

    output.to_csv(output_file, index=False)
    logger.info(f"Saved {len(output)} kinase probe annotations to {output_file}")

    # Summary
    kcgs_matched = output[output["Metadata_kinase_probe_set"] == "KCGS"]
    pkis_matched = output[output["Metadata_kinase_probe_set"] == "PKIS"]
    pkis_with_smiles = sum(1 for m in pkis_raw if (m.get("molecule_structures") or {}).get("canonical_smiles"))
    logger.info(f"KCGS: {len(kcgs_matched)} probes matched to JUMP ({len(kcgs_df)} total)")
    logger.info(f"PKIS: {len(pkis_matched)} probes matched to JUMP ({pkis_with_smiles} total)")
    logger.info(f"Unique JUMP compounds: {output['Metadata_JCP2022'].nunique()}")

    return output_file


# ---------------------------------------------------------------------------
# Demo cells
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: Chemical Probes (Probe & Drugs Portal)

    Processes the Probe & Drugs Portal export, aggregates target genes,
    deduplicates by InChIKey, matches to JUMP, and computes quality flags.
    """)
    return


@app.cell
def _(mo):
    _input_xlsx = EXTERNAL_DATA_DIR / "pd_export_01_2025_875_targets_standardized.xlsx"
    _compound_csv = EXTERNAL_DATA_DIR / "compound.csv.gz"

    if _input_xlsx.exists() and _compound_csv.exists():
        _result = process_chemical_probes()
        _probes_path = _result["probes"]
        _targets_path = _result["targets"]

        _probes_df = pd.read_csv(_probes_path)
        _targets_df = pd.read_csv(_targets_path)
        _n_probes = len(_probes_df)
        _n_in_jump = _probes_df["Metadata_chmprb_in_jump"].sum()
        _n_hq = _probes_df["Metadata_chmprb_is_high_quality"].sum()
        _n_targets = len(_targets_df)

        mo.md(f"""
    **Chemical probes results:**

    - Total probes: **{_n_probes:,}**
    - Matched to JUMP: **{_n_in_jump:,}**
    - High-quality probes: **{_n_hq:,}**
    - Probe-target pairs: **{_n_targets:,}**
    - Output: `{_probes_path}`
        """)
    else:
        _probes_df = None
        _missing = []
        if not _input_xlsx.exists():
            _missing.append(f"`{_input_xlsx}`")
        if not _compound_csv.exists():
            _missing.append(f"`{_compound_csv}`")
        mo.md(f"Skipped: missing {', '.join(_missing)}")
    return


@app.cell
def _(mo):
    _probes_path = INTERIM_DATA_DIR / "chemical_probes_processed.csv"
    if _probes_path.exists():
        _df = pd.read_csv(_probes_path, nrows=20)
        mo.ui.dataframe(_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: Kinase Probes (KCGS + PKIS)

    Loads KCGS from Excel (Zenodo) and PKIS from ChEMBL API (cached),
    generates InChIKeys, and matches to JUMP compounds.
    """)
    return


@app.cell
def _(mo):
    _kcgs_file = EXTERNAL_DATA_DIR / "kcgs_v2.xlsx"
    _compound_csv = EXTERNAL_DATA_DIR / "compound.csv.gz"

    if _kcgs_file.exists() and _compound_csv.exists():
        try:
            _output_path = process_kinase_probes()
            _kinase_df = pd.read_csv(_output_path)
            _n_total = len(_kinase_df)
            _n_kcgs = (_kinase_df["Metadata_kinase_probe_set"] == "KCGS").sum()
            _n_pkis = (_kinase_df["Metadata_kinase_probe_set"] == "PKIS").sum()
            _n_unique = _kinase_df["Metadata_JCP2022"].nunique()

            mo.md(f"""
    **Kinase probes results:**

    - Total probe-JUMP matches: **{_n_total:,}**
    - KCGS matches: **{_n_kcgs:,}**
    - PKIS matches: **{_n_pkis:,}**
    - Unique JUMP compounds: **{_n_unique:,}**
    - Output: `{_output_path}`
            """)
        except (urllib.error.URLError, OSError) as e:
            mo.md(f"Skipped PKIS fetch (network error): `{e}`")
    else:
        _missing = []
        if not _kcgs_file.exists():
            _missing.append(f"`{_kcgs_file}`")
        if not _compound_csv.exists():
            _missing.append(f"`{_compound_csv}`")
        mo.md(f"Skipped: missing {', '.join(_missing)}")
    return


@app.cell
def _(mo):
    _kinase_path = INTERIM_DATA_DIR / "kinase_probes.csv"
    if _kinase_path.exists():
        _df = pd.read_csv(_kinase_path, nrows=20)
        mo.ui.dataframe(_df)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
