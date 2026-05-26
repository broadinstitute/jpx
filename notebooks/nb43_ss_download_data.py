# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pooch==1.9.0",
#     "pandas==3.0.3",
#     "requests==2.34.2",
#     "loguru==0.7.3",
#     "python-dotenv>=1.0",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import json
    import sys
    import time
    from pathlib import Path

    import pooch
    import requests
    from loguru import logger

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import EXTERNAL_DATA_DIR, RAW_DATA_DIR

    PROFILES_DIR = RAW_DATA_DIR / "profiles"

    EXTERNAL_FILES = {
        "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt": (
            "repurposing_drugs_20200324.txt",
            "9c3a08b9d4369fe4257dc9aa763d682fc5ef6b7dd0615534bb44f92b48bc287b",
        ),
        "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_samples_20200324.txt": (
            "repurposing_samples_20200324.txt",
            "3324dfac93d19ceaeafbe7d9e466127709328777cb2f6c5a465740cd327088e9",
        ),
        "https://github.com/jump-cellpainting/datasets/raw/main/metadata/compound.csv.gz": (
            "compound.csv.gz",
            "8885960e92ebd99eb33699a79129f517e668d78dd94f0d7478d39c9825bd3c0a",
        ),
        "https://github.com/jump-cellpainting/datasets/raw/main/metadata/plate.csv.gz": (
            "plate.csv.gz",
            "541ada1f64816166509a4e2328316d2a6662ba67e257b7ae134cbec9d7079319",
        ),
        "https://github.com/jump-cellpainting/datasets/raw/main/metadata/well.csv.gz": (
            "well.csv.gz",
            "fde9fe0678b8d9973f17e106e4cf7e62017110ee3b700f10e27122e1d5f291a1",
        ),
        "https://github.com/jump-cellpainting/datasets/releases/download/v0.12/jump_metadata.duckdb": (
            "jump_metadata.duckdb",
            "ea77a4234cd01060bdf0c8c7f4a18e64dac283c8630a25de4a205936413f6a12",
        ),
        "https://drive.usercontent.google.com/download?id=1lxoLEG6lyDRZnFlxWy3NTkIR9l6dmwPA&export=download&authuser=0&confirm=t": (
            "targetannotations_singleproteins_jumpcompounds_all_5.csv",
            "b299aabfd07a4b6685c9eea0327d762a3d20b71434c82a646a24d76c32c46480",
        ),
        "https://github.com/broadinstitute/jump-profiling-recipe/raw/d2512d978ca17aafead0e99de66386337e6312e4/inputs/cell_counts/all_cell_counts.csv.gz": (
            "all_cell_counts.csv.gz",
            "bf0b2252f316ae96cf857052f1e7739fd1aa6a1f1fee92eb9b1bcf89c89e6897",
        ),
        "https://github.com/broadinstitute/jump-profiling-recipe/raw/d2512d978ca17aafead0e99de66386337e6312e4/inputs/cell_counts/compound_cell_counts.csv.gz": (
            "compound_cell_counts.csv.gz",
            "799a8be0ff54becb86f731467147f1ac0bfbe1936c6fefdf0964dd7876ef9676",
        ),
        "https://github.com/broadinstitute/jump-profiling-recipe/raw/d2512d978ca17aafead0e99de66386337e6312e4/inputs/cell_counts/crispr_cell_counts.csv.gz": (
            "crispr_cell_counts.csv.gz",
            "67a137376cfb0725c22e5d71ae2b2543cc38efd2ad435f4d24de0b5078b7cbb0",
        ),
        "https://github.com/broadinstitute/jump-profiling-recipe/raw/d2512d978ca17aafead0e99de66386337e6312e4/inputs/cell_counts/orf_cell_counts.csv.gz": (
            "orf_cell_counts.csv.gz",
            "809be69372a5e19c53f3c921b49d20f768ee660c3a037e437a57bfecc7deb691",
        ),
        "https://zenodo.org/records/18197517/files/annotations_compound_gene_curated.parquet?download=1": (
            "motive_cpd_gene_annot.parquet",
            "164057bf957c7829ab929d5598b605e00b98024691406742101d8b5a32f36536",
        ),
        "https://zenodo.org/records/18197517/files/mappings_pointers.csv?download=1": (
            "unichem_pointers.csv",
            "03400b14411b561d4a8a2eb4a90b71c393bffc849dcaa52b653d37907cedbf89",
        ),
        "https://clowder.edap-cluster.com/files/68af6b70e4b02565fc7c3a98/blob": (
            "toxcast_invitrodb_v4_3_summary.zip",
            "c61d46cf685d5d84e72d0d4f6f46746ab1714d6f924e43480d390925ae827c59",
        ),
        "https://clowder.edap-cluster.com/files/69529775e4b0731a616efc4b/blob": (
            "dsstox_identifiers.zip",
            "66fc9d4d3bda053ab60ee01e2ff0ed4ba030339588c40711b6939faede6cf4c3",
        ),
        "https://zenodo.org/records/7830352/files/KCGS2.0%20plate%20map%20and%20kinase%20data%20and%20coverage%20summary.xlsx?download=1": (
            "kcgs_v2.xlsx",
            "4b2ab160808c8491d3bc1d830166db7d767771855b3a23a9bfc5cbde4d8cd61a",
        ),
        "https://drive.usercontent.google.com/download?id=1Hrjs7qFlrRoAEYs9-kh6eXx7oX2Dh4Fg&export=download&confirm=t": (
            "toxicity_pk_data.zip",
            "80316a23707f6fb0fbabfcaf133dda1957459f753ad87f9b72d14de5ee4eb61b",
        ),
        "https://zenodo.org/records/20388583/files/pd_export_01_2025_875_targets_standardized.xlsx?download=1": (
            "pd_export_01_2025_875_targets_standardized.xlsx",
            "20459a1fadfc1cbaa47bb7d3b1a16054ab3a7017d229f271966e5d9854a14f9b",
        ),
        "https://zenodo.org/records/20388583/files/molport_batch_search.zip?download=1": (
            "molport_batch_search.zip",
            "a9431123ea353d1d1b36cf61afb968e2bad041d08d6708e003784658a4c23960",
        ),
        "https://zenodo.org/records/20388583/files/pkis_chembl_cache.json?download=1": (
            "pkis_chembl_cache.json",
            "bf8c53323a8f7cb3fd5bfd35e6a9d08aed11690a254ac0c8e024507bf2e8f2e0",
        ),
        "https://zenodo.org/records/20388583/files/mitotox_compounds.parquet?download=1": (
            "mitotox_compounds.parquet",
            "4b23940e0793d8c805f04e0db74a9adc7003bc78c8f3ab5859ce7e2c4b3640bc",
        ),
    }

    MANUAL_FILES: list[str] = []

    PROFILE_FILES = {
        "https://cellpainting-gallery.s3.amazonaws.com/cpg0042-chandrasekaran-jump/source_all/workspace/profiles_assembled/compound/v1.0/profiles_var_mad_int_featselect_harmony.parquet": (
            "compound.parquet",
            "85c2af21852866418d9cacbf7483c98d93001e3818b8527e810381ce6894735f",
        ),
        "https://cellpainting-gallery.s3.amazonaws.com/cpg0042-chandrasekaran-jump/source_all/workspace/profiles_assembled/compound_no_source7/v1.0/profiles_var_mad_int_featselect_harmony.parquet": (
            "compound_no_source7.parquet",
            "8e1e5d9e50c8c7c95ed406981b02adb57c7ff9d253192ed52e661115379be6f1",
        ),
        "https://cellpainting-gallery.s3.amazonaws.com/cpg0042-chandrasekaran-jump/source_all/workspace/profiles_assembled/compound_DL_CPCNN/v1.0/profiles_dropna_var_mad_int_featselect_harmony.parquet": (
            "compound_DL_CPCNN.parquet",
            "3fbf22bb0a8f45aefb924ce024f6b3abf64e33f4b1dc17a6ef916a2534e966bc",
        ),
        "https://cellpainting-gallery.s3.amazonaws.com/cpg0042-chandrasekaran-jump/source_all/workspace/profiles_assembled/compound_DL_CPCNN_no_source7/v1.0/profiles_dropna_var_mad_int_featselect_harmony.parquet": (
            "compound_DL_CPCNN_no_source7.parquet",
            "da9485973e74e22d22f48239ede414fe872efb53fab28aa4eb1e17022a062b00",
        ),
    }

    PRE_HARMONY_FILES = {
        "https://cellpainting-gallery.s3.amazonaws.com/cpg0042-chandrasekaran-jump/source_all/workspace/profiles_assembled/compound_no_source7/v1.0/profiles_var_mad_int_featselect.parquet": (
            "compound_no_source7_pre_harmony.parquet",
            "30655596ee3b6b06ce18a13342f429249dbd7b2351ea0649609384721571d02b",
        ),
        "https://cellpainting-gallery.s3.amazonaws.com/cpg0042-chandrasekaran-jump/source_all/workspace/profiles_assembled/compound_DL_CPCNN_no_source7/v1.0/profiles_dropna_var_mad_int_featselect.parquet": (
            "compound_DL_CPCNN_no_source7_pre_harmony.parquet",
            "f0ac49023c6c21e94352e468806e8edd2e4a3e93a403c60fa10dbf009b68e4a3",
        ),
        "https://cellpainting-gallery.s3.amazonaws.com/cpg0042-chandrasekaran-jump/source_all/workspace/profiles_assembled/compound/v1.0/profiles_var_mad_int_featselect.parquet": (
            "compound_pre_harmony.parquet",
            "9ee4862e8bfc9b08ba5d8675f50cd224f7a4d1ec83848b9fe4396224960b2936",
        ),
        "https://cellpainting-gallery.s3.amazonaws.com/cpg0042-chandrasekaran-jump/source_all/workspace/profiles_assembled/compound_DL_CPCNN/v1.0/profiles_dropna_var_mad_int_featselect.parquet": (
            "compound_DL_CPCNN_pre_harmony.parquet",
            "843fabab93495e69c22bd7e52102a9c70212e7e099a4c008f4f9c3d740b15f70",
        ),
    }


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Download Data

    Fetch all external annotation files and morphological profiles from their
    original public sources with SHA-256 hash verification via Pooch.

    **Three acquisition paths exist:**
    - `just get-results`: pre-computed pipeline outputs only (skip the pipeline entirely)
    - `just get-inputs`: bulk rclone sync from pre-staged S3 (faster, no per-file verification)
    - `just get-from-sources` (this notebook): downloads each file from its canonical URL (full provenance)

    `get-from-sources` downloads everything needed to run the pipeline.

    **Exported functions:**
    - `download_external_files()` - annotation sources (Repurposing Hub, ToxCast, Zenodo, etc.)
    - `download_profiles()` - recipe-Harmony profiles from CellPainting Gallery
    - `download_pre_harmony_profiles()` - pre-batch-correction profiles
    - `download_pkis_chembl()` - PKIS molecule data from ChEMBL API
    - `download_mitotox()` - MitoTox database from mitotox.org API + PubChem SMILES
    - `download_all()` - all of the above
    """)
    return


# ---------------------------------------------------------------------------
# File registries
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------


@app.function
def download_external_files(
    external_files: dict | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Download annotation source files with hash verification.

    Skips files already present with correct hash (Pooch handles this).
    Returns list of downloaded file paths.
    """
    if external_files is None:
        external_files = EXTERNAL_FILES
    if output_dir is None:
        output_dir = EXTERNAL_DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for url, (filename, hash_value) in external_files.items():
        logger.info(f"Downloading {filename}")
        path = pooch.retrieve(url=url, known_hash=f"sha256:{hash_value}", path=output_dir, fname=filename)
        paths.append(Path(path))

    logger.success(f"Downloaded {len(paths)} external files to {output_dir}")
    return paths


@app.function
def download_profiles(
    profile_files: dict | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Download recipe-Harmony profiles from CellPainting Gallery.

    These are the default batch-corrected profiles used by copairs/anndata.
    """
    if profile_files is None:
        profile_files = PROFILE_FILES
    if output_dir is None:
        output_dir = PROFILES_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for url, (filename, hash_value) in profile_files.items():
        logger.info(f"Downloading {filename}")
        path = pooch.retrieve(url=url, known_hash=f"sha256:{hash_value}", path=output_dir, fname=filename)
        paths.append(Path(path))

    logger.success(f"Downloaded {len(paths)} profile files to {output_dir}")
    return paths


@app.function
def download_pre_harmony_profiles(
    pre_harmony_files: dict | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Download pre-batch-correction profiles from CellPainting Gallery.

    Used as input for in-repo Harmony batch correction.
    """
    if pre_harmony_files is None:
        pre_harmony_files = PRE_HARMONY_FILES
    if output_dir is None:
        output_dir = PROFILES_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for url, (filename, hash_value) in pre_harmony_files.items():
        logger.info(f"Downloading {filename}")
        path = pooch.retrieve(url=url, known_hash=f"sha256:{hash_value}", path=output_dir, fname=filename)
        paths.append(Path(path))

    logger.success(f"Downloaded {len(paths)} pre-Harmony profiles to {output_dir}")
    return paths


# ---------------------------------------------------------------------------
# PKIS download (ChEMBL API)
# ---------------------------------------------------------------------------

CHEMBL_BASE_URL = "https://www.ebi.ac.uk"
CHEMBL_PKIS_URL = f"{CHEMBL_BASE_URL}/chembl/api/data/molecule.json?document_chembl_id=CHEMBL2303647&limit=1000"


@app.function
def download_pkis_chembl(
    output_dir: Path | None = None,
    max_compounds: int = 2000,
    force: bool = False,
) -> Path:
    """Fetch PKIS molecule data from ChEMBL API (document CHEMBL2303647).

    Paginates through the API and caches the raw JSON response.
    Used downstream by nb34 for kinase probe curation.
    """
    if output_dir is None:
        output_dir = EXTERNAL_DATA_DIR

    output_path = output_dir / "pkis_chembl_cache.json"
    if output_path.exists() and not force:
        logger.info(f"Output exists: {output_path}. Pass force=True to overwrite.")
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching PKIS from ChEMBL API...")
    molecules = []
    url = CHEMBL_PKIS_URL
    while url and len(molecules) < max_compounds:
        for attempt in range(3):
            response = requests.get(url, timeout=60)
            if response.status_code < 500:
                break
            logger.warning(f"ChEMBL returned {response.status_code}, retry {attempt + 1}/3")
            time.sleep(2 ** attempt)
        response.raise_for_status()
        data = response.json()
        molecules.extend(data.get("molecules", []))
        next_url = data.get("page_meta", {}).get("next")
        if next_url and not next_url.startswith("http"):
            url = f"{CHEMBL_BASE_URL}{next_url}"
        else:
            url = next_url

    molecules = molecules[:max_compounds]
    output_path.write_text(json.dumps(molecules))
    logger.success(f"Cached {len(molecules)} PKIS molecules to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# MitoTox download (API-based)
# ---------------------------------------------------------------------------

MITOTOX_BASE_URL = "https://www.mitotox.org/api"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"

MITOTOX_FUNCTIONS = {
    "F01": "Transmembrane potential",
    "F02": "Function of mitochondria",
    "F03": "Organization of mitochondria",
    "F04": "Movement of mitochondria",
    "F05": "Oxidative stress",
    "F06": "Mitochondrial DNA",
    "F07": "Cell death",
    "F08": "Signaling",
}


@app.function
def _fetch_all_pages(url: str) -> list[dict]:
    """Fetch all pages from a paginated API endpoint."""
    results = []
    while url:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        results.extend(data.get("results", []))
        url = data.get("next")
        time.sleep(0.3)
    return results


@app.function
def _load_or_fetch(cache_path: Path, url: str, name: str) -> list[dict]:
    """Load from cache or fetch from API."""
    if cache_path.exists():
        logger.info(f"Loading cached {name}")
        return json.loads(cache_path.read_text())

    logger.info(f"Fetching {name} from API...")
    data = _fetch_all_pages(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))
    return data


@app.function
def _fetch_smiles(cache_path: Path, cids: list[str]) -> dict[str, str]:
    """Fetch SMILES from PubChem for a list of CIDs."""
    if cache_path.exists():
        logger.info("Loading cached SMILES")
        return json.loads(cache_path.read_text())

    cid_to_smiles = {}
    valid_cids = [c for c in cids if c and str(c).isdigit()]

    for i in range(0, len(valid_cids), 100):
        batch = valid_cids[i : i + 100]
        url = f"{PUBCHEM_URL}/{','.join(map(str, batch))}/property/CanonicalSMILES/JSON"
        try:
            logger.info(f"Fetching SMILES {i + 1}-{i + len(batch)} of {len(valid_cids)}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            for prop in response.json().get("PropertyTable", {}).get("Properties", []):
                smiles = prop.get("CanonicalSMILES") or prop.get("ConnectivitySMILES")
                cid_to_smiles[str(prop["CID"])] = smiles
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch SMILES batch {i}: {e}")
        time.sleep(0.3)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cid_to_smiles))
    return cid_to_smiles


@app.function
def download_mitotox(
    output_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download MitoTox database from mitotox.org API + PubChem SMILES.

    Uses a local cache to avoid repeated API calls. Pass force=True to
    re-download even if the output parquet already exists.
    """
    import pandas as pd

    if output_dir is None:
        output_dir = EXTERNAL_DATA_DIR

    output_path = output_dir / "mitotox_compounds.parquet"
    if output_path.exists() and not force:
        logger.info(f"Output exists: {output_path}. Pass force=True to overwrite.")
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "mitotox_cache"

    compounds = _load_or_fetch(cache_dir / "compounds.json", f"{MITOTOX_BASE_URL}/compounds/list", "compounds")
    records = _load_or_fetch(cache_dir / "records.json", f"{MITOTOX_BASE_URL}/records/list", "records")

    compound_lookup = {
        c["compound_ID"]: {"name": c.get("name"), "cid": c.get("cid")} for c in compounds if c.get("compound_ID")
    }

    compound_toxicity: dict[str, dict] = {}
    for rec in records:
        compound_field = rec.get("compound", "")
        if ": " not in compound_field:
            continue
        compound_id = compound_field.split(": ")[0]

        if compound_id not in compound_toxicity:
            compound_toxicity[compound_id] = {"toxic": False, "functions": set()}

        if rec.get("result"):
            compound_toxicity[compound_id]["toxic"] = True
            func_code = rec.get("func") or ""
            if len(func_code) >= 3 and func_code[:3] in MITOTOX_FUNCTIONS:
                compound_toxicity[compound_id]["functions"].add(MITOTOX_FUNCTIONS[func_code[:3]])

    all_cids = [str(c.get("cid")) for c in compounds if c.get("cid")]
    cid_to_smiles = _fetch_smiles(cache_dir / "smiles.json", all_cids)

    rows = []
    for compound_id, info in compound_lookup.items():
        cid = info.get("cid")
        smiles = cid_to_smiles.get(str(cid)) if cid else None
        if not smiles:
            continue

        tox = compound_toxicity.get(compound_id, {"toxic": False, "functions": set()})
        rows.append(
            {
                "smiles": smiles,
                "compound_name": info.get("name"),
                "compound_id": compound_id,
                "cid": cid,
                "mitotox_label": "toxic" if tox["toxic"] else "non-toxic",
                "functional_mechanism": "|".join(sorted(tox["functions"])) or None,
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)

    toxic_count = (df["mitotox_label"] == "toxic").sum()
    logger.success(f"Saved {len(df)} compounds ({toxic_count} toxic) to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Combined download
# ---------------------------------------------------------------------------


@app.function
def download_all() -> dict[str, list[Path] | Path]:
    """Download all external data and profiles from original sources.

    Equivalent to `just get-inputs` but with per-file hash verification
    and full URL provenance.
    """
    return {
        "external": download_external_files(),
        "profiles": download_profiles(),
        "pre_harmony": download_pre_harmony_profiles(),
        "pkis_chembl": download_pkis_chembl(),
        "mitotox": download_mitotox(),
    }


# ---------------------------------------------------------------------------
# Manual files check
# ---------------------------------------------------------------------------


@app.function
def check_manual_files(manual_files: list[str] | None = None) -> list[str]:
    """Report which manually-sourced files are missing from data/external/.

    These cannot be downloaded programmatically (portal exports, batch searches).
    """
    if manual_files is None:
        manual_files = MANUAL_FILES

    missing = [f for f in manual_files if not (EXTERNAL_DATA_DIR / f).exists()]
    if missing:
        logger.warning(f"{len(missing)} manual files missing: {missing}")
    else:
        logger.success("All manual files present")
    return missing


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
