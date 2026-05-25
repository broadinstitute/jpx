# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb==1.5.3",
#     "numpy==2.4.6",
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

    import numpy as np
    import pandas as pd
    from loguru import logger

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

    # Paths - read from base DB, write to interim files
    BASE_METADATA_DB = EXTERNAL_DATA_DIR / "jump_metadata.duckdb"
    FEATURIZATION_DIR = INTERIM_DATA_DIR / "compound_featurization"
    PROPERTIES_FILE = FEATURIZATION_DIR / "properties.parquet"
    MORGAN_FP_FILE = FEATURIZATION_DIR / "morgan_fp.npz"
    CHEMBERTA_FILE = FEATURIZATION_DIR / "chemberta_77m_mlm.npz"

    # Model constants
    CHEMBERTA_EMBED_DIM = 384


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compound Featurization

    Migrated from `src/jump_production/processing/compound_properties.py` into
    reusable `@app.function` helpers. Each function computes a different molecular
    representation for JUMP compounds.

    **Exported functions:**

    - `compute_properties(db_path, output_file, force)` - Physicochemical properties (MW, LogP, TPSA, QED, Ro5)
    - `compute_morgan(db_path, output_file, force)` - Morgan fingerprints (2048-bit, radius 2)
    - `compute_chemberta(db_path, output_file, device, force)` - ChemBERTa-77M embeddings (384-dim)

    **Environments:**

    - `compute_properties` and `compute_morgan`: `pixi run -e cheminformatics`
    - `compute_chemberta`: `pixi run -e chemberta` (CUDA support)
    """)
    return


# ---------------------------------------------------------------------------
# Shared: load SMILES from base metadata DB
# ---------------------------------------------------------------------------


@app.function
def _load_smiles_from_db(
    db_path: Path | None = None,
    sample: float = 1.0,
) -> pd.DataFrame:
    """Load SMILES from the base metadata database.

    Args:
        db_path: Path to jump_metadata.duckdb. Default: BASE_METADATA_DB
        sample: Fraction of compounds to load (1.0 = all, 0.1 = 10%)

    Returns:
        DataFrame with Metadata_JCP2022 and Metadata_SMILES columns
    """
    import duckdb

    if db_path is None:
        db_path = BASE_METADATA_DB

    if not db_path.exists():
        raise FileNotFoundError(f"Base metadata database not found: {db_path}")

    logger.info(f"Loading SMILES from {db_path}...")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT Metadata_JCP2022, Metadata_SMILES
            FROM compound
            WHERE Metadata_SMILES IS NOT NULL
        """).fetchdf()
    finally:
        con.close()

    logger.info(f"Loaded {len(df):,} compounds with SMILES")

    if sample < 1.0:
        n_sample = int(len(df) * sample)
        df = df.sample(n=n_sample, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {len(df):,} compounds ({sample * 100:.0f}%)")

    return df


# ---------------------------------------------------------------------------
# Internal: single-compound property computation
# ---------------------------------------------------------------------------


@app.function
def _compute_single_compound(smiles: str) -> dict:
    """Compute physicochemical properties for a single compound.

    Returns dict with Metadata_* property columns. RDKit imports are lazy.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED, Descriptors, Lipinski
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")

    result = {
        "Metadata_MW": np.nan,
        "Metadata_LogP": np.nan,
        "Metadata_TPSA": np.nan,
        "Metadata_HBD": np.nan,
        "Metadata_HBA": np.nan,
        "Metadata_RotatableBonds": np.nan,
        "Metadata_NumRings": np.nan,
        "Metadata_NumHeavyAtoms": np.nan,
        "Metadata_NumAromaticRings": np.nan,
        "Metadata_FractionCSP3": np.nan,
        "Metadata_Lipinski_Violations": np.nan,
        "Metadata_QED": np.nan,
        "Metadata_MurckoScaffold": None,
        "Metadata_HasPAINS": None,
        "Metadata_ValidMol": False,
    }

    if pd.isna(smiles):
        return result

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return result

    result["Metadata_ValidMol"] = True

    # Basic properties
    result["Metadata_MW"] = Descriptors.MolWt(mol)
    result["Metadata_LogP"] = Descriptors.MolLogP(mol)
    result["Metadata_TPSA"] = Descriptors.TPSA(mol)
    result["Metadata_HBD"] = Lipinski.NumHDonors(mol)
    result["Metadata_HBA"] = Lipinski.NumHAcceptors(mol)
    result["Metadata_RotatableBonds"] = Lipinski.NumRotatableBonds(mol)
    result["Metadata_NumRings"] = Descriptors.RingCount(mol)
    result["Metadata_NumHeavyAtoms"] = Lipinski.HeavyAtomCount(mol)
    result["Metadata_NumAromaticRings"] = Descriptors.NumAromaticRings(mol)
    result["Metadata_FractionCSP3"] = Descriptors.FractionCSP3(mol)

    # Lipinski violations
    result["Metadata_Lipinski_Violations"] = sum(
        [
            result["Metadata_MW"] > 500,
            result["Metadata_LogP"] > 5,
            result["Metadata_HBD"] > 5,
            result["Metadata_HBA"] > 10,
        ]
    )

    # QED (drug-likeness score 0-1)
    try:
        result["Metadata_QED"] = QED.qed(mol)
    except Exception:
        pass

    # Murcko scaffold
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        result["Metadata_MurckoScaffold"] = Chem.MolToSmiles(scaffold)
    except Exception:
        pass

    # PAINS filter
    try:
        pains_params = FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        pains_catalog = FilterCatalog(pains_params)
        result["Metadata_HasPAINS"] = pains_catalog.HasMatch(mol)
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Exported function 1: compute_properties
# ---------------------------------------------------------------------------


@app.function
def compute_properties(
    db_path: Path | None = None,
    output_file: Path | None = None,
    force: bool = True,
) -> Path:
    """Compute physicochemical properties for JUMP compounds.

    Properties computed per compound: MW, LogP, TPSA, HBD, HBA,
    RotatableBonds, NumRings, NumHeavyAtoms, NumAromaticRings,
    FractionCSP3, Lipinski violations, QED, Murcko scaffold, PAINS flag.

    Args:
        db_path: Path to jump_metadata.duckdb. Default: BASE_METADATA_DB
        output_file: Output parquet path. Default: FEATURIZATION_DIR / "properties.parquet"
        force: Overwrite existing output. Default True.

    Returns:
        Path to saved properties parquet file.
    """
    if output_file is None:
        output_file = PROPERTIES_FILE

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not force:
        logger.info(f"Output already exists, skipping: {output_file}")
        return output_file

    df = _load_smiles_from_db(db_path)
    smiles_list = df["Metadata_SMILES"].tolist()

    logger.info(f"Computing properties for {len(smiles_list):,} compounds...")
    from multiprocessing import Pool, cpu_count

    n_workers = cpu_count()
    logger.info(f"Using {n_workers} workers")
    with Pool(processes=n_workers) as pool:
        results = pool.map(_compute_single_compound, smiles_list)

    props_df = pd.DataFrame(results)
    props_df.insert(0, "Metadata_JCP2022", df["Metadata_JCP2022"].values)

    n_valid = props_df["Metadata_ValidMol"].sum()
    n_pains = props_df["Metadata_HasPAINS"].sum()
    logger.info(f"Valid molecules: {n_valid:,}/{len(props_df):,} ({100 * n_valid / len(props_df):.1f}%)")
    if n_valid > 0 and n_pains > 0:
        logger.info(f"PAINS compounds: {n_pains:,} ({100 * n_pains / n_valid:.1f}% of valid)")

    props_df.to_parquet(output_file, index=False)
    props_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Saved properties: {len(props_df):,} rows ({props_size_mb:.1f} MB) to {output_file}")

    return output_file


# ---------------------------------------------------------------------------
# Exported function 2: compute_morgan
# ---------------------------------------------------------------------------


@app.function
def compute_morgan(
    db_path: Path | None = None,
    output_file: Path | None = None,
    force: bool = True,
) -> Path:
    """Compute Morgan fingerprints (2048-bit, radius 2) for JUMP compounds.

    Uses RDKit Morgan fingerprint generator for consistent, reproducible
    structure-based fingerprints suitable for similarity searches and UMAP.

    Args:
        db_path: Path to jump_metadata.duckdb. Default: BASE_METADATA_DB
        output_file: Output npz path. Default: FEATURIZATION_DIR / "morgan_fp.npz"
        force: Overwrite existing output. Default True.

    Returns:
        Path to saved fingerprint npz file.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    RDLogger.DisableLog("rdApp.*")

    if output_file is None:
        output_file = MORGAN_FP_FILE

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not force:
        logger.info(f"Output already exists, skipping: {output_file}")
        return output_file

    df = _load_smiles_from_db(db_path)
    smiles_list = df["Metadata_SMILES"].tolist()

    logger.info(f"Computing Morgan fingerprints for {len(smiles_list):,} compounds...")
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

    fp_list = []
    valid_list = []
    for smiles in smiles_list:
        if pd.isna(smiles):
            fp_list.append(np.zeros(2048, dtype=np.uint8))
            valid_list.append(False)
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fp_list.append(np.zeros(2048, dtype=np.uint8))
            valid_list.append(False)
            continue
        fp = morgan_gen.GetFingerprint(mol)
        fp_list.append(np.array(fp, dtype=np.uint8))
        valid_list.append(True)

    fp_array = np.array(fp_list, dtype=np.uint8)
    valid_mask = np.array(valid_list, dtype=bool)

    logger.info(f"Valid molecules for fingerprints: {valid_mask.sum():,}/{len(smiles_list):,}")

    np.savez_compressed(
        output_file,
        fingerprints=fp_array,
        valid_mask=valid_mask,
        jcp2022=df["Metadata_JCP2022"].values,
        featurizer="morgan",
        radius=2,
        nbits=2048,
    )
    fp_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Saved Morgan fingerprints: {fp_array.shape} ({fp_size_mb:.1f} MB) to {output_file}")

    return output_file


# ---------------------------------------------------------------------------
# Exported function 3: compute_chemberta
# ---------------------------------------------------------------------------


@app.function
def compute_chemberta(
    db_path: Path | None = None,
    output_file: Path | None = None,
    device: str = "cuda",
    force: bool = True,
) -> Path:
    """Compute ChemBERTa-77M-MLM embeddings (384-dim) for JUMP compounds.

    Uses CLS token embedding from the pretrained ChemBERTa model via
    HuggingFace transformers. Produces 384-dimensional embeddings.

    Requires torch and transformers - use the chemberta pixi env for CUDA.

    Args:
        db_path: Path to jump_metadata.duckdb. Default: BASE_METADATA_DB
        output_file: Output npz path. Default: FEATURIZATION_DIR / "chemberta_77m_mlm.npz"
        device: Device for inference ('cpu', 'cuda', 'cuda:0'). Default 'cuda'.
        force: Overwrite existing output. Default True.

    Returns:
        Path to saved embeddings npz file.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if output_file is None:
        output_file = CHEMBERTA_FILE

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not force:
        logger.info(f"Output already exists, skipping: {output_file}")
        return output_file

    # Check CUDA availability
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    df = _load_smiles_from_db(db_path)
    smiles_list = df["Metadata_SMILES"].tolist()

    # Load model
    logger.info("Loading ChemBERTa-77M-MLM from HuggingFace...")
    model_name = "DeepChem/ChemBERTa-77M-MLM"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device).eval()

    logger.info(f"Computing embeddings for {len(smiles_list):,} compounds on {device}...")

    batch_size = 64
    embeddings = []
    valid_mask = np.ones(len(smiles_list), dtype=bool)

    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]

            try:
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                # CLS token is first token
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
            except Exception as e:
                logger.warning(f"Batch {i // batch_size} failed: {e}")
                valid_mask[i : i + len(batch)] = False
                embeddings.append(np.full((len(batch), CHEMBERTA_EMBED_DIM), np.nan, dtype=np.float32))

            if (i // batch_size) % 100 == 0 and i > 0:
                logger.info(f"  Processed {i:,}/{len(smiles_list):,} compounds")

    embed_array = np.vstack(embeddings).astype(np.float32)
    logger.info(f"Valid embeddings: {valid_mask.sum():,}/{len(smiles_list):,}")

    np.savez_compressed(
        output_file,
        embeddings=embed_array,
        valid_mask=valid_mask,
        jcp2022=df["Metadata_JCP2022"].values,
        featurizer="chemberta-77m-mlm",
        embed_dim=CHEMBERTA_EMBED_DIM,
    )
    embed_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Saved ChemBERTa embeddings: {embed_array.shape} ({embed_size_mb:.1f} MB) to {output_file}")

    return output_file


# ---------------------------------------------------------------------------
# Demo cells
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: Physicochemical properties

    Computes MW, LogP, TPSA, HBD, HBA, rotatable bonds, ring counts,
    FractionCSP3, Lipinski violations, QED, Murcko scaffold, and PAINS flag
    for all JUMP compounds with SMILES in the base metadata database.
    """)
    return


@app.cell
def _(mo):
    if BASE_METADATA_DB.exists():
        if PROPERTIES_FILE.exists():
            _props = pd.read_parquet(PROPERTIES_FILE)
            _n_valid = _props["Metadata_ValidMol"].sum()
            _n_pains = _props["Metadata_HasPAINS"].sum()
            _lipinski = _props["Metadata_Lipinski_Violations"].value_counts().sort_index().to_dict()
            mo.md(f"""
    **Properties (pre-computed):**

    - Compounds: **{len(_props):,}**
    - Valid molecules: **{_n_valid:,}** ({100 * _n_valid / len(_props):.1f}%)
    - PAINS compounds: **{_n_pains:,}**
    - Lipinski violations distribution: {_lipinski}
    - File: `{PROPERTIES_FILE}`
            """)
        else:
            mo.md(f"Properties file not found at `{PROPERTIES_FILE}`. Run `compute_properties()` to generate.")
    else:
        mo.md(f"Skipped: base metadata DB not found at `{BASE_METADATA_DB}`")
    return


@app.cell
def _(mo):
    if PROPERTIES_FILE.exists():
        _props = pd.read_parquet(PROPERTIES_FILE)
        mo.ui.dataframe(_props.head(20))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: Morgan fingerprints

    2048-bit Morgan fingerprints (radius 2) for structural similarity
    and UMAP visualization.
    """)
    return


@app.cell
def _(mo):
    if MORGAN_FP_FILE.exists():
        _data = np.load(MORGAN_FP_FILE, allow_pickle=True)
        _fp = _data["fingerprints"]
        _valid = _data["valid_mask"]
        _jcp = _data["jcp2022"]
        mo.md(f"""
    **Morgan fingerprints (pre-computed):**

    - Compounds: **{len(_jcp):,}**
    - Valid molecules: **{_valid.sum():,}**
    - Array shape: `{_fp.shape}`
    - File: `{MORGAN_FP_FILE}`
    - Size: **{MORGAN_FP_FILE.stat().st_size / (1024 * 1024):.1f} MB**
        """)
    else:
        mo.md(f"Morgan FP file not found at `{MORGAN_FP_FILE}`. Run `compute_morgan()` to generate.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo: ChemBERTa embeddings

    384-dimensional ChemBERTa-77M-MLM embeddings (CLS token) for learned
    molecular representations. Requires `pixi run -e chemberta` for CUDA.
    """)
    return


@app.cell
def _(mo):
    if CHEMBERTA_FILE.exists():
        _data = np.load(CHEMBERTA_FILE, allow_pickle=True)
        _emb = _data["embeddings"]
        _valid = _data["valid_mask"]
        _jcp = _data["jcp2022"]
        mo.md(f"""
    **ChemBERTa embeddings (pre-computed):**

    - Compounds: **{len(_jcp):,}**
    - Valid embeddings: **{_valid.sum():,}**
    - Array shape: `{_emb.shape}`
    - File: `{CHEMBERTA_FILE}`
    - Size: **{CHEMBERTA_FILE.stat().st_size / (1024 * 1024):.1f} MB**
        """)
    else:
        mo.md(f"ChemBERTa file not found at `{CHEMBERTA_FILE}`. Run `compute_chemberta()` to generate.")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
