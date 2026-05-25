# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb",
#     "loguru",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "python-dotenv",
#     "scanpy",
#     "scikit-learn",
#     "scipy",
# ]
# NOTE: Run with pixi run -e cheminformatics marimo edit/run
# (rdkit is conda-only and comes from the pixi env)
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import json
    import sys
    import warnings
    from pathlib import Path

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from loguru import logger
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from sklearn.preprocessing import StandardScaler

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import (
        DEFAULT_DPI,
        INTERIM_DATA_DIR,
        METADATA_DB,
        PROCESSED_DATA_DIR,
    )
    from nb04_ss_visualization import compute_umap_bounds

    MORGAN_FP_FILE = INTERIM_DATA_DIR / "compound_featurization" / "morgan_fp.npz"
    OUTPUT_DIR = PROCESSED_DATA_DIR / "chemical-space"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chemical Space Characterization

    Analyzes the chemical properties and diversity of the JUMP compound collection:

    1. Annotation coverage (Drug Repurposing Hub, ChEMBL, Chemical Probes, MOTIVE)
    2. Physicochemical properties (MW, logP, TPSA, etc.)
    3. Structure UMAP (Morgan fingerprints, cosine - Tanimoto proxy)
    4. Property UMAP (z-scored descriptors, Euclidean)
    5. Scaffold diversity (Murcko scaffolds)
    6. PAINS / structural alerts
    7. Property space coverage

    Properties are precomputed by `workflow.py` via `compute_compound_properties()` task
    (calls `nb37_ss_compound_featurization`).

    **Environment:** Requires `pixi run -e cheminformatics marimo edit/run` (rdkit is conda-only).

    *Outputs:* `data/processed/chemical-space/`
    """)
    return


@app.cell
def _(mo):
    skip_umap_toggle = mo.ui.switch(value=False, label="Skip UMAP computation")
    subsample_slider = mo.ui.slider(
        start=0.1,
        stop=1.0,
        step=0.1,
        value=1.0,
        label="UMAP subsample fraction",
    )
    mo.hstack([skip_umap_toggle, subsample_slider], justify="start")
    return skip_umap_toggle, subsample_slider


@app.function
def load_compounds() -> pd.DataFrame:
    """Load compound metadata with precomputed properties from DuckDB.

    Returns DataFrame with Metadata_JCP2022, SMILES, annotations, and physicochemical properties.
    """
    _con = duckdb.connect(str(METADATA_DB), read_only=True)
    _query = """
    SELECT * FROM compound_metadata
    WHERE Metadata_SMILES IS NOT NULL
    """
    _df = _con.execute(_query).fetchdf()
    _con.close()
    _df = _df.astype({c: "object" for c in _df.select_dtypes("string").columns})
    logger.info(f"Loaded {len(_df):,} compounds with SMILES and properties")

    _n_valid = _df["Metadata_ValidMol"].sum() if "Metadata_ValidMol" in _df.columns else 0
    logger.info(f"Valid molecules (RDKit): {_n_valid:,} ({100 * _n_valid / len(_df):.1f}%)")

    return _df


@app.function
def load_annotations() -> dict[str, pd.DataFrame]:
    """Load annotation tables from DuckDB.

    Returns dict of DataFrames keyed by annotation source.
    """
    _con = duckdb.connect(str(METADATA_DB), read_only=True)
    _tables = [row[0] for row in _con.execute("SHOW TABLES").fetchall()]

    _annotation_tables = {
        "repurposing_hub": "repurposing_hub_annotations",
        "chemical_probes": "chemical_probes",
        "motive": "motive_annotations",
        "toxcast": "toxcast_annotations",
    }

    _annotations = {}
    for _name, _table in _annotation_tables.items():
        if _table in _tables:
            _annotations[_name] = _con.execute(f"SELECT * FROM {_table}").fetchdf()
            logger.info(f"Loaded {len(_annotations[_name]):,} rows from {_table}")

    _con.close()
    return _annotations


@app.function
def plot_annotation_coverage(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Figure | None:
    """Bar chart showing annotation coverage by source.

    Returns figure only when standalone (ax is None).
    """
    _standalone = ax is None
    if _standalone:
        _fig, ax = plt.subplots(figsize=(4, 3))

    _annotation_cols = {
        "Metadata_repurposing_target": "Rep. Hub\n(Target)",
        "Metadata_repurposing_moa": "Rep. Hub\n(MOA)",
        "Metadata_Uniprot_target": "ChEMBL",
        "Metadata_chmprb_target_genes": "Chem.\nProbes",
        "Metadata_motive_gene_biokg": "MOTIVE",
    }

    _coverage = {label: df[col].notna().sum() for col, label in _annotation_cols.items() if col in df.columns}

    if not _coverage:
        logger.warning("No annotation columns found in compound metadata")
        return _fig if _standalone else None

    _bars = ax.bar(_coverage.keys(), _coverage.values(), color="#3498db", edgecolor="none", width=0.7)
    for _bar, _count in zip(_bars, _coverage.values()):
        ax.annotate(
            f"{100 * _count / len(df):.0f}%",
            xy=(_bar.get_x() + _bar.get_width() / 2, _bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_ylabel("Compounds", fontsize=9)
    ax.set_title("Annotation Coverage", fontsize=10, fontweight="bold", loc="left")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if _standalone:
        _fig.tight_layout()
        return _fig
    return None


@app.function
def plot_property_distributions(df: pd.DataFrame, axes: list[plt.Axes] | None = None) -> plt.Figure | None:
    """Faceted histograms of physicochemical property distributions (2x2 grid).

    Returns figure only when standalone (axes is None).
    """
    _standalone = axes is None
    if _standalone:
        _fig, _axes_arr = plt.subplots(2, 2, figsize=(6, 4))
        axes = _axes_arr.flatten()

    _props = [
        ("Metadata_MW", "MW (Da)", (100, 700), 40),
        ("Metadata_LogP", "LogP", (-3, 8), 40),
        ("Metadata_TPSA", "TPSA (A2)", (0, 200), 40),
        ("Metadata_HBD", "HBD", (-0.5, 8.5), 9),
    ]

    for _ax, (_col, _label, _xlim, _bins) in zip(axes, _props):
        if _col not in df.columns:
            continue

        _data = df[_col].dropna()
        _ax.hist(_data, bins=_bins, range=_xlim, color="#3498db", edgecolor="none", alpha=0.8)

        _median = _data.median()
        _ax.axvline(_median, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8)
        _ax.annotate(
            f"med={_median:.1f}",
            xy=(_median, _ax.get_ylim()[1] * 0.9),
            fontsize=7,
            color="#e74c3c",
            ha="left" if _median < (_xlim[0] + _xlim[1]) / 2 else "right",
        )

        _ax.set_xlabel(_label, fontsize=9)
        _ax.set_ylabel("Count", fontsize=8)
        _ax.set_xlim(_xlim)
        _ax.tick_params(labelsize=7)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

    if _standalone:
        _fig.suptitle("Property Distributions", fontsize=10, fontweight="bold", y=1.02)
        _fig.tight_layout()
        return _fig
    return None


@app.function
def plot_lipinski_violations(df: pd.DataFrame, ax: plt.Axes | None = None, style: str = "pie") -> plt.Figure | None:
    """Pie chart of Lipinski rule violations.

    Returns figure only when standalone (ax is None).
    """
    _standalone = ax is None
    if _standalone:
        _fig, ax = plt.subplots(figsize=(3, 3))

    if "Metadata_Lipinski_Violations" not in df.columns:
        logger.warning("Metadata_Lipinski_Violations column not found")
        return _fig if _standalone else None

    _counts = df["Metadata_Lipinski_Violations"].value_counts().sort_index()
    _violation_colors = {0: "#2ca02c", 1: "#98df8a", 2: "#ffbb78", 3: "#ff7f0e", 4: "#d62728"}
    _pie_colors = [_violation_colors.get(int(v), "#999999") for v in _counts.index]
    _labels = [f"{int(v)} viol." for v in _counts.index]

    if style == "pie":
        _total = _counts.sum()
        _wedges, _texts = ax.pie(
            _counts.values,
            colors=_pie_colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )
        _legend_labels = [f"{lbl}: {cnt:,} ({100 * cnt / _total:.0f}%)" for lbl, cnt in zip(_labels, _counts.values)]
        ax.legend(
            _wedges,
            _legend_labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=7,
            frameon=False,
        )
        ax.set_title("Lipinski Rule-of-5", fontsize=10, fontweight="bold", loc="center")
    else:
        ax.bar(_counts.index.astype(int), _counts.values, color="#3498db", edgecolor="none", width=0.7)
        for _i, _count in zip(_counts.index, _counts.values):
            ax.annotate(f"{100 * _count / len(df):.0f}%", xy=(_i, _count), ha="center", va="bottom", fontsize=7)
        ax.set_xlabel("Violations", fontsize=9)
        ax.set_ylabel("Compounds", fontsize=9)
        ax.set_title("Lipinski Rule-of-5", fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if _standalone:
        _fig.tight_layout()
        return _fig
    return None


@app.function
def plot_scaffold_diversity(df: pd.DataFrame, ax: plt.Axes | None = None, top_n: int = 10) -> plt.Figure | None:
    """Horizontal bar chart of most common Murcko scaffolds.

    Returns figure only when standalone (ax is None).
    """
    _standalone = ax is None
    if _standalone:
        _fig, ax = plt.subplots(figsize=(4, 3))

    if "Metadata_MurckoScaffold" not in df.columns:
        logger.warning("Metadata_MurckoScaffold column not found")
        return _fig if _standalone else None

    _scaffold_col = df["Metadata_MurckoScaffold"]
    _n_acyclic = _scaffold_col.isna().sum() + (_scaffold_col == "").sum()

    _valid_scaffolds = _scaffold_col[_scaffold_col.notna() & (_scaffold_col != "")]
    _scaffold_counts = _valid_scaffolds.value_counts()
    _n_unique = len(_scaffold_counts)
    _n_singletons = (_scaffold_counts == 1).sum()

    logger.info(f"Unique scaffolds: {_n_unique:,}")
    logger.info(f"Singleton scaffolds: {_n_singletons:,} ({100 * _n_singletons / _n_unique:.1f}%)")
    logger.info(f"Acyclic molecules (no scaffold): {_n_acyclic:,}")

    _top_scaffolds = _scaffold_counts.head(top_n)

    ax.barh(range(len(_top_scaffolds)), _top_scaffolds.values, color="#3498db", edgecolor="none")
    ax.set_yticks(range(len(_top_scaffolds)))
    ax.set_yticklabels(
        [s[:15] + "..." if len(s) > 15 else s for s in _top_scaffolds.index],
        fontsize=6,
        family="monospace",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Compounds", fontsize=9)
    ax.set_title(f"Top Scaffolds ({_n_unique:,} unique)", fontsize=10, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if _standalone:
        _fig.tight_layout()
        return _fig
    return None


@app.function
def plot_property_space(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Figure | None:
    """Scatter plot of MW vs LogP (property space coverage).

    Returns figure only when standalone (ax is None).
    """
    _standalone = ax is None
    if _standalone:
        _fig, ax = plt.subplots(figsize=(4, 3))

    if "Metadata_MW" not in df.columns or "Metadata_LogP" not in df.columns:
        logger.warning("Metadata_MW or Metadata_LogP columns not found")
        return _fig if _standalone else None

    _mask = df["Metadata_MW"].notna() & df["Metadata_LogP"].notna()
    _x = df.loc[_mask, "Metadata_MW"]
    _y = df.loc[_mask, "Metadata_LogP"]

    ax.scatter(_x, _y, s=0.5, alpha=0.1, c="#3498db", rasterized=True)

    ax.axvline(500, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(5, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("MW (Da)", fontsize=9)
    ax.set_ylabel("LogP", fontsize=9)
    ax.set_xlim(0, 800)
    ax.set_ylim(-5, 10)
    ax.set_title("Property Space", fontsize=10, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if _standalone:
        _fig.tight_layout()
        return _fig
    return None


@app.function
def plot_clinical_phase(df: pd.DataFrame, ax: plt.Axes | None = None, style: str = "pie") -> plt.Figure | None:
    """Pie chart of clinical development phases.

    Returns figure only when standalone (ax is None).
    """
    _standalone = ax is None
    if _standalone:
        _fig, ax = plt.subplots(figsize=(4, 3))

    _phase_col = "Metadata_repurposing_clinical_phase"
    if _phase_col not in df.columns:
        logger.warning(f"{_phase_col} column not found")
        return _fig if _standalone else None

    _phase_order = ["Preclinical", "Phase 1", "Phase 2", "Phase 3", "Launched"]
    _phase_labels = ["Pre", "P1", "P2", "P3", "Launch"]
    _colors = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#2ca02c"]

    _phase_counts = df[_phase_col].value_counts().reindex(_phase_order).fillna(0).astype(int)
    _nonzero_mask = _phase_counts.values > 0
    _counts = _phase_counts.values[_nonzero_mask]
    _labels = [_phase_labels[i] for i in range(len(_phase_labels)) if _nonzero_mask[i]]
    _pie_colors = [_colors[i] for i in range(len(_colors)) if _nonzero_mask[i]]

    if style == "pie":
        _total = _counts.sum()
        _wedges, _texts = ax.pie(
            _counts,
            colors=_pie_colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )
        _legend_labels = [f"{lbl}: {cnt:,} ({100 * cnt / _total:.0f}%)" for lbl, cnt in zip(_labels, _counts)]
        ax.legend(
            _wedges,
            _legend_labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=7,
            frameon=False,
        )
        ax.set_title("Clinical Phase", fontsize=10, fontweight="bold", loc="center")
    else:
        ax.bar(range(len(_phase_counts)), _phase_counts.values, color="#3498db", edgecolor="none", width=0.7)
        ax.set_xticks(range(len(_phase_counts)))
        ax.set_xticklabels(_phase_labels, fontsize=8)
        for _i, _count in enumerate(_phase_counts.values):
            if _count > 0:
                ax.annotate(f"{_count:,}", xy=(_i, _count), ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("Compounds", fontsize=9)
        ax.set_title("Clinical Phase", fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if _standalone:
        _fig.tight_layout()
        return _fig
    return None


@app.function
def plot_pains_summary(df: pd.DataFrame, ax: plt.Axes | None = None, style: str = "pie") -> plt.Figure | None:
    """Pie chart showing PAINS structural alerts.

    Returns figure only when standalone (ax is None).
    """
    _standalone = ax is None
    if _standalone:
        _fig, ax = plt.subplots(figsize=(4, 3))

    if "Metadata_HasPAINS" not in df.columns:
        logger.warning("Metadata_HasPAINS column not found")
        return _fig if _standalone else None

    _n_pains = df["Metadata_HasPAINS"].sum()
    _n_clean = len(df) - _n_pains
    _total = len(df)

    if style == "pie":
        _counts = [_n_clean, _n_pains]
        _labels = ["Clean", "PAINS"]
        _colors = ["#2ca02c", "#d62728"]

        _wedges, _texts = ax.pie(
            _counts,
            colors=_colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
            explode=(0, 0.05),
        )
        _legend_labels = [f"{lbl}: {cnt:,} ({100 * cnt / _total:.1f}%)" for lbl, cnt in zip(_labels, _counts)]
        ax.legend(
            _wedges,
            _legend_labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=7,
            frameon=False,
        )
        ax.set_title("PAINS Alerts", fontsize=10, fontweight="bold", loc="center")
    else:
        _bar_height = 0.18
        ax.barh([0], [_n_clean], height=_bar_height, color="#3498db", edgecolor="none", label=f"Clean ({_n_clean:,})")
        ax.barh(
            [0],
            [_n_pains],
            height=_bar_height,
            left=[_n_clean],
            color="#e74c3c",
            edgecolor="none",
            label=f"PAINS ({_n_pains:,}, {100 * _n_pains / len(df):.1f}%)",
        )
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k" if x > 0 else "0"))
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("Compounds", fontsize=9)
        ax.set_title("PAINS Structural Alerts", fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if _standalone:
        _fig.tight_layout()
        return _fig
    return None


@app.function
def compute_structure_umap(df: pd.DataFrame, subsample_frac: float = 1.0, seed: int = 42) -> sc.AnnData | None:
    """Compute UMAP from precomputed Morgan fingerprints.

    Uses L2 normalization + cosine similarity as a proxy for Tanimoto/Jaccard.
    Achieves 99.8% rank correlation with Tanimoto (see issue #26) and
    enables GPU acceleration via rapids_singlecell.

    Returns AnnData with UMAP coordinates, or None if fingerprints not found.
    """
    logger.info("Computing structure UMAP from precomputed fingerprints...")

    if not MORGAN_FP_FILE.exists():
        logger.error(f"Fingerprints file not found: {MORGAN_FP_FILE}")
        logger.info("Run: just redun main  # compute_compound_properties task")
        return None

    _fp_data = np.load(MORGAN_FP_FILE, allow_pickle=True)
    _fp_array = _fp_data["fingerprints"]
    _valid_mask = _fp_data["valid_mask"]
    _jcp2022_fps = _fp_data["jcp2022"]

    logger.info(f"Loaded fingerprints: {_fp_array.shape}")
    logger.info(f"Valid fingerprints: {_valid_mask.sum():,}/{len(_valid_mask):,}")

    _fp_valid = _fp_array[_valid_mask]
    _jcp_valid = _jcp2022_fps[_valid_mask]

    _jcp_to_idx = {jcp: i for i, jcp in enumerate(df["Metadata_JCP2022"])}

    _matched_fps = []
    _matched_idx = []
    for _i, _jcp in enumerate(_jcp_valid):
        if _jcp in _jcp_to_idx:
            _matched_fps.append(_fp_valid[_i])
            _matched_idx.append(_jcp_to_idx[_jcp])

    logger.info(f"Matched {len(_matched_fps):,} fingerprints to compound metadata")

    _fp_matched = np.array(_matched_fps, dtype=np.float32)
    _obs_df = df.iloc[_matched_idx].reset_index(drop=True)

    if subsample_frac < 1.0:
        _n_total = len(_fp_matched)
        _n_sample = int(_n_total * subsample_frac)
        _rng = np.random.default_rng(seed)
        _sample_idx = _rng.choice(_n_total, size=_n_sample, replace=False)
        _sample_idx = np.sort(_sample_idx)
        _fp_matched = _fp_matched[_sample_idx]
        _obs_df = _obs_df.iloc[_sample_idx].reset_index(drop=True)
        logger.info(f"Subsampled to {_n_sample:,} compounds ({subsample_frac:.0%})")

    # L2 normalize fingerprints for cosine similarity
    logger.info("L2 normalizing fingerprints for cosine similarity...")
    _norms = np.linalg.norm(_fp_matched, axis=1, keepdims=True)
    _norms = np.maximum(_norms, 1e-10)
    _fp_normalized = _fp_matched / _norms

    warnings.filterwarnings("ignore", category=FutureWarning, message=".*Transforming to str index.*")
    try:
        from anndata import ImplicitModificationWarning

        warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
    except ImportError:
        pass

    _adata = sc.AnnData(X=_fp_normalized)
    _adata.obs = _obs_df

    # Check for GPU
    _has_gpu = False
    try:
        import cupy  # noqa: F401

        _has_gpu = True
    except ImportError:
        pass

    if _has_gpu:
        import rapids_singlecell as rsc

        logger.info("Using GPU acceleration via rapids_singlecell")
        rsc.get.anndata_to_GPU(_adata)
        logger.info("Computing neighbors (cosine on L2-normalized, GPU IVF-Flat)...")
        rsc.pp.neighbors(_adata, n_neighbors=15, use_rep="X", metric="cosine", algorithm="ivfflat", random_state=seed)
        logger.info("Computing UMAP (GPU)...")
        rsc.tl.umap(_adata, min_dist=0.1, random_state=seed)
        logger.info("Transferring data back to CPU")
        rsc.get.anndata_to_CPU(_adata)
    else:
        logger.info("Using CPU-only scanpy")
        logger.info("Computing neighbors (cosine on L2-normalized, CPU)...")
        sc.pp.neighbors(_adata, n_neighbors=15, use_rep="X", metric="cosine", random_state=seed)
        logger.info("Computing UMAP (CPU)...")
        sc.tl.umap(_adata, min_dist=0.1, random_state=seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return _adata


@app.function
def compute_property_umap(df: pd.DataFrame, seed: int = 42) -> sc.AnnData | None:
    """Compute UMAP from z-scored physicochemical descriptors.

    Uses Euclidean distance on standardized continuous descriptors
    to visualize property space coverage.

    Returns AnnData with UMAP coordinates, or None if descriptors not available.
    """
    logger.info("Computing property UMAP from z-scored descriptors...")

    _descriptor_cols = [
        "Metadata_MW",
        "Metadata_LogP",
        "Metadata_TPSA",
        "Metadata_HBA",
        "Metadata_HBD",
        "Metadata_RotatableBonds",
        "Metadata_NumRings",
        "Metadata_NumAromaticRings",
        "Metadata_NumHeavyAtoms",
        "Metadata_FractionCSP3",
        "Metadata_QED",
    ]

    _available_cols = [c for c in _descriptor_cols if c in df.columns]
    if len(_available_cols) < 5:
        logger.warning(f"Only {len(_available_cols)} descriptor columns found, skipping property UMAP")
        return None

    logger.info(f"Using {len(_available_cols)} descriptors: {_available_cols}")

    _desc_df = df[_available_cols + ["Metadata_JCP2022"]].dropna()
    logger.info(f"Compounds with all descriptors: {len(_desc_df):,}")

    _features = _desc_df[_available_cols].values.astype(np.float32)
    _features_scaled = StandardScaler().fit_transform(_features)

    warnings.filterwarnings("ignore", category=FutureWarning, message=".*Transforming to str index.*")

    _adata = sc.AnnData(X=_features_scaled, obs=_desc_df.reset_index(drop=True))

    _has_gpu = False
    try:
        import cupy  # noqa: F401

        _has_gpu = True
    except ImportError:
        pass

    if _has_gpu:
        import rapids_singlecell as rsc

        logger.info("Computing property UMAP (GPU, Euclidean on z-scored descriptors)...")
        rsc.get.anndata_to_GPU(_adata)
        rsc.pp.neighbors(_adata, n_neighbors=15, use_rep="X", metric="euclidean", random_state=seed)
        rsc.tl.umap(_adata, min_dist=0.1, random_state=seed)
        rsc.get.anndata_to_CPU(_adata)
    else:
        logger.info("Computing property UMAP (CPU, Euclidean on z-scored descriptors)...")
        sc.pp.neighbors(_adata, n_neighbors=15, use_rep="X", metric="euclidean", random_state=seed)
        sc.tl.umap(_adata, min_dist=0.1, random_state=seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return _adata


@app.function
def plot_structure_umap(adata: sc.AnnData) -> plt.Figure:
    """Plot structure UMAP with IQR-based outlier clipping.

    Returns the figure.
    """
    _xlim, _ylim = compute_umap_bounds(adata, iqr_k=3.0)
    _coords = adata.obsm["X_umap"]

    _fig, _ax = plt.subplots(figsize=(10, 8))
    _ax.scatter(_coords[:, 0], _coords[:, 1], s=0.5, alpha=0.3, c="#3498db", rasterized=True)
    _ax.set_xlim(_xlim)
    _ax.set_ylim(_ylim)
    _ax.set_xlabel("UMAP 1")
    _ax.set_ylabel("UMAP 2")
    _ax.set_title(f"Chemical Structure Space (n={len(adata):,} compounds)", fontweight="bold")
    _ax.set_aspect("equal")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _fig.tight_layout()
    return _fig


@app.function
def plot_property_umap(adata: sc.AnnData) -> plt.Figure:
    """Plot property UMAP.

    Returns the figure.
    """
    _coords = adata.obsm["X_umap"]

    _fig, _ax = plt.subplots(figsize=(8, 8))
    _ax.scatter(_coords[:, 0], _coords[:, 1], s=0.5, alpha=0.3, c="#3498db", rasterized=True)
    _ax.set_xlabel("UMAP 1")
    _ax.set_ylabel("UMAP 2")
    _ax.set_title(f"Property Space UMAP (n={len(adata):,} compounds)", fontweight="bold")
    _ax.set_aspect("equal")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _fig.tight_layout()
    return _fig


@app.function
def plot_umap_gallery(adata: sc.AnnData) -> plt.Figure:
    """Create a 2x3 UMAP gallery figure with all colorings.

    Layout:
        Base (overview)     | Clinical Phase      | Has Target
        Molecular Weight    | LogP                | PAINS

    Returns the figure.
    """
    _xlim, _ylim = compute_umap_bounds(adata, iqr_k=3.0)
    _coords = adata.obsm["X_umap"]
    _fig, _axes = plt.subplots(2, 3, figsize=(15, 10))

    _point_size = 0.3
    _alpha = 0.3

    # A: Base (overview)
    _ax = _axes[0, 0]
    _ax.scatter(_coords[:, 0], _coords[:, 1], s=_point_size, alpha=_alpha, c="#3498db", rasterized=True)
    _ax.set_xlim(_xlim)
    _ax.set_ylim(_ylim)
    _ax.set_xlabel("UMAP 1", fontsize=9)
    _ax.set_ylabel("UMAP 2", fontsize=9)
    _ax.set_title("A. Chemical Structure Space", fontsize=10, fontweight="bold", loc="left")
    _ax.tick_params(labelsize=8)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    # B: Clinical Phase
    _ax = _axes[0, 1]
    _phase_col = "Metadata_repurposing_clinical_phase"
    if _phase_col in adata.obs.columns:
        _mask_na = adata.obs[_phase_col].isna()
        _ax.scatter(_coords[_mask_na, 0], _coords[_mask_na, 1], s=_point_size, alpha=0.1, c="#cccccc", rasterized=True)
        _phase_order = ["Preclinical", "Phase 1", "Phase 2", "Phase 3", "Launched"]
        _phase_colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#e31a1c"]
        for _phase, _color in zip(_phase_order, _phase_colors):
            _mask = adata.obs[_phase_col] == _phase
            if _mask.any():
                _ax.scatter(
                    _coords[_mask, 0], _coords[_mask, 1], s=1, alpha=0.6, c=_color, rasterized=True, label=_phase
                )
        _ax.legend(fontsize=6, loc="upper right", markerscale=3)
    _ax.set_xlim(_xlim)
    _ax.set_ylim(_ylim)
    _ax.set_xlabel("UMAP 1", fontsize=9)
    _ax.set_ylabel("UMAP 2", fontsize=9)
    _ax.set_title("B. Clinical Phase", fontsize=10, fontweight="bold", loc="left")
    _ax.tick_params(labelsize=8)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    # C: Has Target
    _ax = _axes[0, 2]
    _target_cols = ["Metadata_repurposing_target", "Metadata_Uniprot_target", "Metadata_chmprb_target_genes"]
    _has_target = pd.Series(False, index=adata.obs.index)
    for _col in _target_cols:
        if _col in adata.obs.columns:
            _has_target = _has_target | adata.obs[_col].notna()
    _mask_no_target = ~_has_target
    _ax.scatter(
        _coords[_mask_no_target, 0],
        _coords[_mask_no_target, 1],
        s=_point_size,
        alpha=0.1,
        c="#cccccc",
        rasterized=True,
    )
    _ax.scatter(_coords[_has_target, 0], _coords[_has_target, 1], s=0.8, alpha=0.5, c="#e74c3c", rasterized=True)
    _ax.set_xlim(_xlim)
    _ax.set_ylim(_ylim)
    _ax.set_xlabel("UMAP 1", fontsize=9)
    _ax.set_ylabel("UMAP 2", fontsize=9)
    _ax.set_title(f"C. Has Target ({_has_target.sum():,})", fontsize=10, fontweight="bold", loc="left")
    _ax.tick_params(labelsize=8)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    # D: Molecular Weight
    _ax = _axes[1, 0]
    if "Metadata_MW" in adata.obs.columns:
        _mw = adata.obs["Metadata_MW"].values
        _mw_clipped = np.clip(_mw, 150, 700)
        _scatter = _ax.scatter(
            _coords[:, 0], _coords[:, 1], s=_point_size, alpha=_alpha, c=_mw_clipped, cmap="viridis", rasterized=True
        )
        _cbar = plt.colorbar(_scatter, ax=_ax, shrink=0.3)
        _cbar.set_label("MW (Da)", fontsize=8)
        _cbar.ax.tick_params(labelsize=7)
    _ax.set_xlim(_xlim)
    _ax.set_ylim(_ylim)
    _ax.set_xlabel("UMAP 1", fontsize=9)
    _ax.set_ylabel("UMAP 2", fontsize=9)
    _ax.set_title("D. Molecular Weight", fontsize=10, fontweight="bold", loc="left")
    _ax.tick_params(labelsize=8)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    # E: LogP
    _ax = _axes[1, 1]
    if "Metadata_LogP" in adata.obs.columns:
        _logp = adata.obs["Metadata_LogP"].values
        _logp_clipped = np.clip(_logp, -2, 8)
        _scatter = _ax.scatter(
            _coords[:, 0], _coords[:, 1], s=_point_size, alpha=_alpha, c=_logp_clipped, cmap="RdYlBu_r", rasterized=True
        )
        _cbar = plt.colorbar(_scatter, ax=_ax, shrink=0.3)
        _cbar.set_label("LogP", fontsize=8)
        _cbar.ax.tick_params(labelsize=7)
    _ax.set_xlim(_xlim)
    _ax.set_ylim(_ylim)
    _ax.set_xlabel("UMAP 1", fontsize=9)
    _ax.set_ylabel("UMAP 2", fontsize=9)
    _ax.set_title("E. LogP (Lipophilicity)", fontsize=10, fontweight="bold", loc="left")
    _ax.tick_params(labelsize=8)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    # F: PAINS
    _ax = _axes[1, 2]
    _n_pains_gallery = 0
    if "Metadata_HasPAINS" in adata.obs.columns:
        _has_pains = adata.obs["Metadata_HasPAINS"].astype(bool)
        _mask_clean = ~_has_pains
        _ax.scatter(
            _coords[_mask_clean, 0],
            _coords[_mask_clean, 1],
            s=_point_size,
            alpha=0.1,
            c="#cccccc",
            rasterized=True,
        )
        _ax.scatter(_coords[_has_pains, 0], _coords[_has_pains, 1], s=0.8, alpha=0.5, c="#e74c3c", rasterized=True)
        _n_pains_gallery = _has_pains.sum()
    _ax.set_xlim(_xlim)
    _ax.set_ylim(_ylim)
    _ax.set_xlabel("UMAP 1", fontsize=9)
    _ax.set_ylabel("UMAP 2", fontsize=9)
    _ax.set_title(f"F. PAINS Alerts ({_n_pains_gallery:,})", fontsize=10, fontweight="bold", loc="left")
    _ax.tick_params(labelsize=8)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    _fig.suptitle("Structure UMAP Gallery", fontsize=12, fontweight="bold", y=0.98)
    _fig.tight_layout(rect=[0, 0, 1, 0.96])

    return _fig


@app.function
def plot_combined_figure(df: pd.DataFrame, adata_umap: sc.AnnData | None = None) -> plt.Figure:
    """Create combined publication-ready figure with all chemical space panels.

    Layout (3x3 grid):
        A: Annotation coverage    B: Clinical phase       C: Property space
        D: Property distributions (2x2, spans 2 cols)    E: Lipinski violations
        F: Scaffold diversity     G: PAINS summary        H: Structure UMAP

    Returns the figure.
    """
    _fig = plt.figure(figsize=(14, 10))
    _gs = _fig.add_gridspec(3, 3, hspace=0.4, wspace=0.5)

    # Row 1: A, B, C
    _ax_a = _fig.add_subplot(_gs[0, 0])
    plot_annotation_coverage(df, ax=_ax_a)
    _ax_a.set_title("A. Annotation Coverage", fontsize=10, fontweight="bold", loc="left")

    _ax_b = _fig.add_subplot(_gs[0, 1])
    plot_clinical_phase(df, ax=_ax_b, style="pie")
    _ax_b.set_title("B. Clinical Phase", fontsize=10, fontweight="bold", loc="center")

    _ax_c = _fig.add_subplot(_gs[0, 2])
    plot_property_space(df, ax=_ax_c)
    _ax_c.set_title("C. Property Space", fontsize=10, fontweight="bold", loc="left")

    # Row 2: D (2x2 faceted histograms spanning 2 cols), E
    _gs_d = GridSpecFromSubplotSpec(2, 2, subplot_spec=_gs[1, :2], hspace=0.55, wspace=0.25)
    _axes_d = [_fig.add_subplot(_gs_d[i, j]) for i in range(2) for j in range(2)]
    plot_property_distributions(df, axes=_axes_d)
    _axes_d[0].annotate(
        "D. Property Distributions",
        xy=(0, 1.25),
        xycoords="axes fraction",
        fontsize=10,
        fontweight="bold",
    )

    _ax_e = _fig.add_subplot(_gs[1, 2])
    plot_lipinski_violations(df, ax=_ax_e, style="pie")
    _ax_e.set_title("E. Lipinski Rule-of-5", fontsize=10, fontweight="bold", loc="center")

    # Row 3: F, G, H
    _ax_f = _fig.add_subplot(_gs[2, 0])
    plot_scaffold_diversity(df, ax=_ax_f)
    _ax_f.set_title("F. Top Scaffolds", fontsize=10, fontweight="bold", loc="left")

    _ax_g = _fig.add_subplot(_gs[2, 1])
    plot_pains_summary(df, ax=_ax_g, style="pie")
    _ax_g.set_title("G. PAINS Alerts", fontsize=10, fontweight="bold", loc="center")

    # Panel H: Structure UMAP
    _ax_h = _fig.add_subplot(_gs[2, 2])
    if adata_umap is not None and "X_umap" in adata_umap.obsm:
        _coords = adata_umap.obsm["X_umap"]
        _xlim, _ylim = compute_umap_bounds(adata_umap, iqr_k=3.0)
        _ax_h.scatter(_coords[:, 0], _coords[:, 1], s=0.5, alpha=0.3, c="#3498db", rasterized=True)
        _ax_h.set_xlim(_xlim)
        _ax_h.set_ylim(_ylim)
        _ax_h.set_xlabel("UMAP 1", fontsize=8)
        _ax_h.set_ylabel("UMAP 2", fontsize=8)
        _ax_h.tick_params(labelsize=7)
        _ax_h.set_aspect("equal")
    else:
        _ax_h.text(0.5, 0.5, "UMAP not computed", ha="center", va="center", fontsize=9, color="gray")
        _ax_h.set_xticks([])
        _ax_h.set_yticks([])
    _ax_h.set_title("H. Structure UMAP", fontsize=10, fontweight="bold", loc="left")
    _ax_h.spines["top"].set_visible(False)
    _ax_h.spines["right"].set_visible(False)

    _fig.suptitle(f"Chemical Space Characterization (n={len(df):,} compounds)", fontsize=12, fontweight="bold", y=0.98)

    return _fig


@app.function
def generate_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics for LLM consumption.

    Returns a dict with all key statistics from the chemical space analysis.
    Also saves to JSON file.
    """
    _n_total = len(df)

    _annotation_cols = {
        "Metadata_repurposing_target": "repurposing_hub_target",
        "Metadata_repurposing_moa": "repurposing_hub_moa",
        "Metadata_Uniprot_target": "chembl",
        "Metadata_chmprb_target_genes": "chemical_probes",
        "Metadata_motive_gene_biokg": "motive",
    }
    _annotations = {}
    for _col, _name in _annotation_cols.items():
        if _col in df.columns:
            _n = int(df[_col].notna().sum())
            _annotations[_name] = {"count": _n, "percent": round(100 * _n / _n_total, 1)}

    _phase_col = "Metadata_repurposing_clinical_phase"
    _clinical_phases = {}
    if _phase_col in df.columns:
        for _phase in ["Preclinical", "Phase 1", "Phase 2", "Phase 3", "Launched"]:
            _clinical_phases[_phase] = int((df[_phase_col] == _phase).sum())

    _properties = {}
    _prop_cols = [
        ("Metadata_MW", "molecular_weight_da"),
        ("Metadata_LogP", "logp"),
        ("Metadata_TPSA", "tpsa_angstrom2"),
        ("Metadata_HBD", "hbd"),
        ("Metadata_HBA", "hba"),
        ("Metadata_RotatableBonds", "rotatable_bonds"),
        ("Metadata_NumRings", "num_rings"),
        ("Metadata_QED", "qed_drug_likeness"),
    ]
    for _col, _name in _prop_cols:
        if _col in df.columns:
            _data = df[_col].dropna()
            _properties[_name] = {
                "median": round(float(_data.median()), 2),
                "mean": round(float(_data.mean()), 2),
                "std": round(float(_data.std()), 2),
                "min": round(float(_data.min()), 2),
                "max": round(float(_data.max()), 2),
                "q25": round(float(_data.quantile(0.25)), 2),
                "q75": round(float(_data.quantile(0.75)), 2),
            }

    _lipinski = {}
    if "Metadata_Lipinski_Violations" in df.columns:
        _counts = df["Metadata_Lipinski_Violations"].value_counts().sort_index()
        for _violations, _count in _counts.items():
            _lipinski[f"{int(_violations)}_violations"] = {
                "count": int(_count),
                "percent": round(100 * _count / _n_total, 1),
            }
        _lipinski["rule_of_5_compliant"] = {
            "count": int((df["Metadata_Lipinski_Violations"] == 0).sum()),
            "percent": round(100 * (df["Metadata_Lipinski_Violations"] == 0).sum() / _n_total, 1),
        }

    _scaffolds = {}
    if "Metadata_MurckoScaffold" in df.columns:
        _scaffold_col = df["Metadata_MurckoScaffold"]
        _n_acyclic = int(_scaffold_col.isna().sum() + (_scaffold_col == "").sum())
        _valid_scaffolds = _scaffold_col[_scaffold_col.notna() & (_scaffold_col != "")]
        _scaffold_counts = _valid_scaffolds.value_counts()
        _n_unique = len(_scaffold_counts)
        _n_singletons = int((_scaffold_counts == 1).sum())

        _scaffolds = {
            "unique_scaffolds": _n_unique,
            "singleton_scaffolds": _n_singletons,
            "singleton_percent": round(100 * _n_singletons / _n_unique, 1) if _n_unique > 0 else 0,
            "acyclic_molecules": _n_acyclic,
            "diversity_ratio": round(_n_unique / _n_total, 3),
            "top_5_scaffolds": [{"smiles": str(s), "count": int(c)} for s, c in _scaffold_counts.head(5).items()],
        }

    _pains = {}
    if "Metadata_HasPAINS" in df.columns:
        _n_pains = int(df["Metadata_HasPAINS"].sum())
        _pains = {
            "pains_flagged": _n_pains,
            "pains_percent": round(100 * _n_pains / _n_total, 1),
            "clean": _n_total - _n_pains,
        }

    _summary = {
        "dataset": "JUMP Cell Painting",
        "total_compounds": _n_total,
        "valid_molecules": int(df["Metadata_ValidMol"].sum()) if "Metadata_ValidMol" in df.columns else _n_total,
        "annotation_coverage": _annotations,
        "clinical_development": _clinical_phases,
        "physicochemical_properties": _properties,
        "lipinski_rule_of_5": _lipinski,
        "scaffold_diversity": _scaffolds,
        "pains_structural_alerts": _pains,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _output_file = OUTPUT_DIR / "chemical_space_summary.json"
    with open(_output_file, "w") as f:
        json.dump(_summary, f, indent=2)
    logger.info(f"Saved summary statistics to {_output_file}")

    return _summary


@app.cell
def _(mo):
    mo.stop(
        not METADATA_DB.exists(),
        mo.md(f"**Metadata database not found:** `{METADATA_DB}`"),
    )

    df = load_compounds()
    _annotations = load_annotations()

    mo.md(f"Loaded **{len(df):,}** compounds with SMILES and properties.")
    return (df,)


@app.cell
def _(df):
    _fig = plot_annotation_coverage(df)
    _fig
    return


@app.cell
def _(df):
    _fig = plot_clinical_phase(df)
    _fig
    return


@app.cell
def _(df):
    _fig = plot_property_distributions(df)
    _fig
    return


@app.cell
def _(df):
    _fig = plot_lipinski_violations(df)
    _fig
    return


@app.cell
def _(df):
    _fig = plot_scaffold_diversity(df)
    _fig
    return


@app.cell
def _(df):
    _fig = plot_property_space(df)
    _fig
    return


@app.cell
def _(df):
    _fig = plot_pains_summary(df)
    _fig
    return


@app.cell
def _(df, mo, skip_umap_toggle, subsample_slider):
    mo.stop(
        skip_umap_toggle.value,
        mo.md("**UMAP computation skipped** (toggle above to enable)."),
    )
    mo.stop(
        not MORGAN_FP_FILE.exists(),
        mo.md(
            f"**Morgan fingerprints not found:** `{MORGAN_FP_FILE}`\n\n"
            "Run: `just redun main  # compute_compound_properties task`"
        ),
    )

    adata_structure = compute_structure_umap(df, subsample_frac=subsample_slider.value)
    return (adata_structure,)


@app.cell
def _(adata_structure, mo):
    mo.stop(adata_structure is None, mo.md("Structure UMAP not available."))
    _fig = plot_structure_umap(adata_structure)
    _fig
    return


@app.cell
def _(adata_structure, mo):
    mo.stop(adata_structure is None, mo.md("Structure UMAP not available for gallery."))
    _fig = plot_umap_gallery(adata_structure)
    _fig
    return


@app.cell
def _(df, mo, skip_umap_toggle):
    mo.stop(
        skip_umap_toggle.value,
        mo.md("**Property UMAP skipped** (toggle above to enable)."),
    )

    adata_property = compute_property_umap(df)
    return (adata_property,)


@app.cell
def _(adata_property, mo):
    mo.stop(adata_property is None, mo.md("Property UMAP not available."))
    _fig = plot_property_umap(adata_property)
    _fig
    return


@app.cell
def _(adata_structure, df, skip_umap_toggle):
    # Access adata_structure only if UMAP was computed
    _adata_for_combined = None
    if not skip_umap_toggle.value:
        try:
            _adata_for_combined = adata_structure  # noqa: F821
        except NameError:
            pass

    _fig = plot_combined_figure(df, adata_umap=_adata_for_combined)
    _fig
    return


@app.cell
def _(adata_structure, df, mo, skip_umap_toggle):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual plots
    _fig_annot = plot_annotation_coverage(df)
    _fig_annot.savefig(OUTPUT_DIR / "annotations_coverage.png", dpi=DEFAULT_DPI, bbox_inches="tight")

    _fig_phase = plot_clinical_phase(df)
    _fig_phase.savefig(OUTPUT_DIR / "clinical_phase.png", dpi=DEFAULT_DPI, bbox_inches="tight")

    _fig_props = plot_property_distributions(df)
    _fig_props.savefig(OUTPUT_DIR / "properties_distributions.png", dpi=DEFAULT_DPI, bbox_inches="tight")

    _fig_lipinski = plot_lipinski_violations(df)
    _fig_lipinski.savefig(OUTPUT_DIR / "lipinski_violations.png", dpi=DEFAULT_DPI, bbox_inches="tight")

    _fig_scaffold = plot_scaffold_diversity(df)
    _fig_scaffold.savefig(OUTPUT_DIR / "scaffold_diversity.png", dpi=DEFAULT_DPI, bbox_inches="tight")

    _fig_propspace = plot_property_space(df)
    _fig_propspace.savefig(OUTPUT_DIR / "property_space.png", dpi=DEFAULT_DPI, bbox_inches="tight")

    _fig_pains = plot_pains_summary(df)
    _fig_pains.savefig(OUTPUT_DIR / "pains_summary.png", dpi=DEFAULT_DPI, bbox_inches="tight")

    # Save PAINS compound list
    if "Metadata_HasPAINS" in df.columns:
        _pains_cols = ["Metadata_JCP2022", "Metadata_SMILES", "Metadata_MurckoScaffold"]
        _pains_out = df[df["Metadata_HasPAINS"]][[c for c in _pains_cols if c in df.columns]].copy()
        if len(_pains_out) > 0:
            _pains_out.to_csv(OUTPUT_DIR / "pains_compounds.csv", index=False)

    # Save scaffold stats
    if "Metadata_MurckoScaffold" in df.columns:
        _scaffold_col = df["Metadata_MurckoScaffold"]
        _n_acyclic = _scaffold_col.isna().sum() + (_scaffold_col == "").sum()
        _valid_scaffolds = _scaffold_col[_scaffold_col.notna() & (_scaffold_col != "")]
        _scaffold_counts = _valid_scaffolds.value_counts()
        _n_unique = len(_scaffold_counts)
        _n_singletons = (_scaffold_counts == 1).sum()
        _stats_df = pd.DataFrame(
            [
                {
                    "n_compounds": len(df),
                    "n_unique_scaffolds": _n_unique,
                    "n_singleton_scaffolds": int(_n_singletons),
                    "n_acyclic": int(_n_acyclic),
                    "scaffold_diversity_ratio": _n_unique / len(df),
                }
            ]
        )
        _stats_df.to_csv(OUTPUT_DIR / "scaffold_stats.csv", index=False)

    # Combined figure
    _adata_for_save = None
    if not skip_umap_toggle.value:
        try:
            _adata_for_save = adata_structure  # noqa: F821
        except NameError:
            pass

    _fig_combined = plot_combined_figure(df, adata_umap=_adata_for_save)
    _fig_combined.savefig(OUTPUT_DIR / "chemical_space_combined.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    _fig_combined.savefig(OUTPUT_DIR / "chemical_space_combined.pdf", bbox_inches="tight")

    # UMAP gallery
    if not skip_umap_toggle.value:
        try:
            _adata_gallery = adata_structure  # noqa: F821
            if _adata_gallery is not None:
                _fig_gallery = plot_umap_gallery(_adata_gallery)
                _fig_gallery.savefig(OUTPUT_DIR / "structure_umap_gallery.png", dpi=DEFAULT_DPI, bbox_inches="tight")
                _fig_gallery.savefig(OUTPUT_DIR / "structure_umap_gallery.pdf", bbox_inches="tight")
        except NameError:
            pass

    # Summary JSON
    _summary = generate_summary(df)

    mo.md(f"""
    **Saved all outputs to** `{OUTPUT_DIR}`

    - `annotations_coverage.png` - annotation coverage bar chart
    - `clinical_phase.png` - clinical development phases
    - `properties_distributions.png` - physicochemical property histograms
    - `lipinski_violations.png` - Lipinski rule-of-5 pie chart
    - `scaffold_diversity.png` - top Murcko scaffolds
    - `property_space.png` - MW vs LogP scatter
    - `pains_summary.png` - PAINS structural alerts
    - `chemical_space_combined.png/pdf` - combined publication figure
    - `structure_umap_gallery.png/pdf` - 2x3 UMAP gallery (if computed)
    - `structure_umap.h5ad` - structure UMAP coordinates (if computed)
    - `property_umap.h5ad` - property UMAP coordinates (if computed)
    - `chemical_space_summary.json` - all statistics
    - `scaffold_stats.csv` - scaffold diversity statistics
    - `pains_compounds.csv` - list of PAINS-flagged compounds
    """)
    return


@app.function
def run_chemical_space(output_dir: str | Path | None = None) -> str:
    """Run the full chemical space analysis and save all outputs.

    Produces combined figure (PNG + PDF), structure UMAP gallery,
    individual plots, summary JSON, scaffold stats CSV, PAINS compound
    list, and a .complete marker.

    Parameters
    ----------
    output_dir : str | Path | None
        Output directory. Defaults to PROCESSED_DATA_DIR / "chemical-space".

    Returns
    -------
    str
        Path to the output directory.
    """
    if output_dir is None:
        _out = OUTPUT_DIR
    else:
        _out = Path(output_dir)
    _out.mkdir(parents=True, exist_ok=True)

    # Load data
    _df = load_compounds()

    # Individual plots
    _fig_annot = plot_annotation_coverage(_df)
    _fig_annot.savefig(_out / "annotations_coverage.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(_fig_annot)

    _fig_phase = plot_clinical_phase(_df)
    _fig_phase.savefig(_out / "clinical_phase.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(_fig_phase)

    _fig_props = plot_property_distributions(_df)
    _fig_props.savefig(_out / "properties_distributions.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(_fig_props)

    _fig_lipinski = plot_lipinski_violations(_df)
    _fig_lipinski.savefig(_out / "lipinski_violations.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(_fig_lipinski)

    _fig_scaffold = plot_scaffold_diversity(_df)
    _fig_scaffold.savefig(_out / "scaffold_diversity.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(_fig_scaffold)

    _fig_propspace = plot_property_space(_df)
    _fig_propspace.savefig(_out / "property_space.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(_fig_propspace)

    _fig_pains = plot_pains_summary(_df)
    _fig_pains.savefig(_out / "pains_summary.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(_fig_pains)

    # PAINS compound list
    if "Metadata_HasPAINS" in _df.columns:
        _pains_cols = ["Metadata_JCP2022", "Metadata_SMILES", "Metadata_MurckoScaffold"]
        _pains_out = _df[_df["Metadata_HasPAINS"]][[c for c in _pains_cols if c in _df.columns]].copy()
        if len(_pains_out) > 0:
            _pains_out.to_csv(_out / "pains_compounds.csv", index=False)

    # Scaffold stats
    if "Metadata_MurckoScaffold" in _df.columns:
        _scaffold_col = _df["Metadata_MurckoScaffold"]
        _n_acyclic = _scaffold_col.isna().sum() + (_scaffold_col == "").sum()
        _valid_scaffolds = _scaffold_col[_scaffold_col.notna() & (_scaffold_col != "")]
        _scaffold_counts = _valid_scaffolds.value_counts()
        _n_unique = len(_scaffold_counts)
        _n_singletons = (_scaffold_counts == 1).sum()
        _stats_df = pd.DataFrame(
            [
                {
                    "n_compounds": len(_df),
                    "n_unique_scaffolds": _n_unique,
                    "n_singleton_scaffolds": int(_n_singletons),
                    "n_acyclic": int(_n_acyclic),
                    "scaffold_diversity_ratio": _n_unique / len(_df),
                }
            ]
        )
        _stats_df.to_csv(_out / "scaffold_stats.csv", index=False)

    # Structure UMAP (may return None if fingerprints not available)
    _adata_structure = compute_structure_umap(_df)

    # Combined figure
    _fig_combined = plot_combined_figure(_df, adata_umap=_adata_structure)
    _fig_combined.savefig(_out / "chemical_space_combined.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    _fig_combined.savefig(_out / "chemical_space_combined.pdf", bbox_inches="tight")
    plt.close(_fig_combined)

    # UMAP gallery and h5ad (if structure UMAP available)
    if _adata_structure is not None:
        _adata_structure.write_h5ad(_out / "structure_umap.h5ad")
        _fig_gallery = plot_umap_gallery(_adata_structure)
        _fig_gallery.savefig(
            _out / "structure_umap_gallery.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white"
        )
        _fig_gallery.savefig(_out / "structure_umap_gallery.pdf", bbox_inches="tight")
        plt.close(_fig_gallery)

    # Summary JSON
    generate_summary(_df)

    # .complete marker
    (_out / ".complete").touch()

    logger.success(f"run_chemical_space: saved all outputs to {_out}")
    return str(_out)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
