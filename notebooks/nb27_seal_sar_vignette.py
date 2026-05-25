# NOTE: Run with pixi run -e cheminformatics marimo edit/run
# (rdkit is conda-only and comes from the pixi env)
#
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
#     "tqdm",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from loguru import logger
    from rdkit import Chem
    from rdkit.Chem import BRICS, Draw
    from tqdm import tqdm

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import (
        COPAIRS_RESULTS_DB,
        DEFAULT_DPI,
        METADATA_DB,
        PROCESSED_DATA_DIR,
    )
    from nb02_ss_queries import query_activity_results
    from nb04_ss_visualization import compute_umap_bounds

    OUTPUT_DIR = PROCESSED_DATA_DIR / "sar-vignette"

    # Default thresholds
    DEFAULT_P_THRESHOLD = 0.10
    DEFAULT_MIN_ATOMS = 2
    DEFAULT_MAX_ATOMS = 18
    DEFAULT_MIN_HITS = 5
    DEFAULT_MIN_PPV = 0.5

    # Common substructures to filter out (SMARTS patterns)
    COMMON_SUBSTRUCTURES = [
        "c1ccccc1",  # benzene
        "C1CCCCC1",  # cyclohexane
        "[CH3]",  # methyl
        "[OH]",  # hydroxyl
    ]


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SAR Vignette - Structural Alert Discovery

    Discovers structural alerts (substructure patterns) that predict phenotypic
    activity in Cell Painting using RDKit's BRICS fragment enumeration.

    **Research question:** What molecular substructures are associated with
    phenotypically active compounds in Cell Painting?

    **Algorithm (SARpy-inspired):**
    1. Fragment enumeration: Generate BRICS fragments for all compounds
    2. Evaluate each fragment: Compute PPV (precision) for predicting activity
    3. Filter alerts: Keep fragments with PPV > threshold, min occurrences >= 5
    4. Greedy selection: Build ruleset by iteratively selecting highest-PPV fragments
    5. Post-filter: Remove small fragments and common substructures

    **Environment:** Requires `pixi run -e cheminformatics marimo edit/run` (rdkit is conda-only).

    *Outputs:* `data/processed/sar-vignette/{dataset}/{preprocessing}/`
    """)
    return


@app.cell
def _(mo):
    dataset_input = mo.ui.text(
        value="compound_no_source7",
        label="Dataset",
    )
    preprocessing_input = mo.ui.text(
        value="activity_no_target2",
        label="Preprocessing",
    )
    p_threshold_slider = mo.ui.slider(
        start=0.01,
        stop=0.20,
        step=0.01,
        value=0.10,
        label="P-value threshold for activity",
    )
    min_ppv_slider = mo.ui.slider(
        start=0.3,
        stop=0.9,
        step=0.05,
        value=0.50,
        label="Minimum PPV for alerts",
    )
    mo.vstack(
        [
            mo.hstack([dataset_input, preprocessing_input], justify="start"),
            mo.hstack([p_threshold_slider, min_ppv_slider], justify="start"),
        ]
    )
    return dataset_input, min_ppv_slider, p_threshold_slider, preprocessing_input


# =============================================================================
# Data Loading
# =============================================================================


@app.function
def load_activity_labels(
    dataset: str,
    preprocessing: str,
    filter_name: str = "all_sources",
    activity_params: str = "default",
    p_threshold: float = DEFAULT_P_THRESHOLD,
) -> pd.DataFrame:
    """Load activity labels from copairs results using shared query utility."""
    df = query_activity_results(
        dataset=dataset,
        preprocessing=preprocessing,
        filter_name=filter_name,
        activity_params=activity_params,
    )

    # Add is_active label based on p_threshold
    df["is_active"] = (df["corrected_p_value"] < p_threshold).astype(int)

    _n_active = df["is_active"].sum()
    _pct_active = _n_active / len(df) * 100
    logger.info(f"Loaded activity: {len(df):,} compounds, {_n_active:,} active ({_pct_active:.1f}%)")
    return df


@app.function
def load_smiles() -> pd.DataFrame:
    """Load SMILES from metadata database."""
    _con = duckdb.connect(str(METADATA_DB), read_only=True)
    df = _con.execute(
        """
        SELECT
            Metadata_JCP2022 as JCP2022,
            Metadata_SMILES as SMILES
        FROM compound
        WHERE Metadata_SMILES IS NOT NULL
    """
    ).df()
    _con.close()
    logger.info(f"Loaded SMILES: {len(df):,} compounds")
    return df


@app.function
def prepare_dataset(
    activity_df: pd.DataFrame,
    smiles_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join activity labels with SMILES and create RDKit molecules."""
    df = activity_df.merge(smiles_df, left_on="Metadata_JCP2022", right_on="JCP2022", how="inner")
    logger.info(f"Joined dataset: {len(df):,} compounds")

    # Create RDKit molecule objects
    logger.info("Creating RDKit molecule objects...")
    _mols = []
    _valid_mask = []
    for _smiles in tqdm(df["SMILES"], desc="Parsing SMILES"):
        _mol = Chem.MolFromSmiles(_smiles)
        if _mol is not None:
            _mols.append(_mol)
            _valid_mask.append(True)
        else:
            _mols.append(None)
            _valid_mask.append(False)

    df["mol"] = _mols
    df = df[_valid_mask].reset_index(drop=True)
    logger.info(f"Valid molecules: {len(df):,}")

    return df


# =============================================================================
# Fragment Enumeration
# =============================================================================


@app.function
def count_heavy_atoms(smiles: str) -> int:
    """Count heavy atoms in a SMILES string."""
    _mol = Chem.MolFromSmiles(smiles)
    if _mol is None:
        return 0
    return _mol.GetNumHeavyAtoms()


@app.function
def enumerate_brics_fragments(
    dataset: pd.DataFrame,
    min_atoms: int = DEFAULT_MIN_ATOMS,
    max_atoms: int = DEFAULT_MAX_ATOMS,
) -> dict[str, list[str]]:
    """Enumerate BRICS fragments for all compounds.

    Returns:
        Dictionary mapping fragment SMILES to list of JCP2022 IDs
    """
    logger.info("Enumerating BRICS fragments...")
    fragment_to_compounds: dict[str, list[str]] = {}

    for _, _row in tqdm(dataset.iterrows(), total=len(dataset), desc="BRICS decomposition"):
        _mol = _row["mol"]
        _jcp = _row["Metadata_JCP2022"]

        try:
            _frags = BRICS.BRICSDecompose(_mol, returnMols=False)
            for _frag in _frags:
                # Clean up BRICS dummy atoms for counting
                _clean_frag = _frag
                for _i in range(20):  # Remove dummy atom markers [1*] through [19*]
                    _clean_frag = _clean_frag.replace(f"[{_i}*]", "[*]")

                _n_atoms = count_heavy_atoms(_clean_frag)
                if min_atoms <= _n_atoms <= max_atoms:
                    if _frag not in fragment_to_compounds:
                        fragment_to_compounds[_frag] = []
                    fragment_to_compounds[_frag].append(_jcp)
        except Exception:
            continue  # Skip problematic molecules

    logger.info(f"Found {len(fragment_to_compounds):,} unique fragments")
    return fragment_to_compounds


# =============================================================================
# Alert Evaluation
# =============================================================================


@app.function
def evaluate_fragments(
    fragment_to_compounds: dict[str, list[str]],
    dataset: pd.DataFrame,
    min_hits: int = DEFAULT_MIN_HITS,
) -> pd.DataFrame:
    """Evaluate PPV for each fragment.

    Returns DataFrame with columns: smiles, n_hits, tp, fp, ppv
    """
    logger.info("Evaluating fragment PPV...")

    # Create JCP to activity mapping
    _jcp_to_active = dict(zip(dataset["Metadata_JCP2022"], dataset["is_active"]))

    _results = []
    for _frag_smiles, _compound_list in tqdm(fragment_to_compounds.items(), desc="Evaluating fragments"):
        _n_hits = len(_compound_list)
        if _n_hits < min_hits:
            continue

        _tp = sum(_jcp_to_active.get(_jcp, 0) for _jcp in _compound_list)
        _fp = _n_hits - _tp

        _ppv = _tp / _n_hits if _n_hits > 0 else 0

        _results.append(
            {
                "smiles": _frag_smiles,
                "n_hits": _n_hits,
                "tp": _tp,
                "fp": _fp,
                "ppv": _ppv,
            }
        )

    df = pd.DataFrame(_results)
    logger.info(f"Evaluated {len(df):,} fragments with >= {min_hits} hits")
    return df


@app.function
def filter_common_substructures(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """Remove common/trivial substructures from alerts."""
    logger.info("Filtering common substructures...")

    # Parse common substructures
    _common_patterns = []
    for _smarts in COMMON_SUBSTRUCTURES:
        _pattern = Chem.MolFromSmarts(_smarts)
        if _pattern is not None:
            _common_patterns.append(_pattern)

    def _is_common(smiles: str) -> bool:
        _mol = Chem.MolFromSmiles(smiles)
        if _mol is None:
            return True

        # Check exact match with common substructures
        for _pattern in _common_patterns:
            if _mol.HasSubstructMatch(_pattern) and _mol.GetNumHeavyAtoms() <= _pattern.GetNumHeavyAtoms() + 2:
                return True

        return False

    _initial_count = len(alerts_df)
    alerts_df = alerts_df[~alerts_df["smiles"].apply(_is_common)].copy()
    _removed = _initial_count - len(alerts_df)
    logger.info(f"Removed {_removed} common substructures, {len(alerts_df)} remaining")

    return alerts_df


@app.function
def filter_small_fragments(alerts_df: pd.DataFrame, min_atoms: int = 5) -> pd.DataFrame:
    """Remove fragments with too few atoms."""
    logger.info(f"Filtering fragments with < {min_atoms} heavy atoms...")

    _initial_count = len(alerts_df)
    alerts_df = alerts_df[alerts_df["smiles"].apply(count_heavy_atoms) >= min_atoms].copy()
    _removed = _initial_count - len(alerts_df)
    logger.info(f"Removed {_removed} small fragments, {len(alerts_df)} remaining")

    return alerts_df


# =============================================================================
# Greedy Rule Selection
# =============================================================================


@app.function
def greedy_select_alerts(
    alerts_df: pd.DataFrame,
    fragment_to_compounds: dict[str, list[str]],
    dataset: pd.DataFrame,
    min_ppv: float = DEFAULT_MIN_PPV,
    max_alerts: int = 100,
) -> pd.DataFrame:
    """Greedily select non-redundant alerts.

    Iteratively selects the highest-PPV alert and removes covered compounds.
    """
    logger.info(f"Greedy selection with min_ppv={min_ppv}...")

    # Filter by PPV threshold
    _candidates = alerts_df[alerts_df["ppv"] >= min_ppv].copy()
    logger.info(f"Candidates with PPV >= {min_ppv}: {len(_candidates)}")

    if len(_candidates) == 0:
        logger.warning("No alerts meet the PPV threshold")
        return pd.DataFrame(columns=["smiles", "n_hits", "tp", "fp", "ppv", "rank"])

    # Sort by PPV descending, then by n_hits descending, then by SMILES for deterministic ties
    _candidates = _candidates.sort_values(["ppv", "n_hits", "smiles"], ascending=[False, False, True])

    # Greedy selection
    _selected = []
    _covered_compounds = set()
    _jcp_to_active = dict(zip(dataset["Metadata_JCP2022"], dataset["is_active"]))

    for _, _row in _candidates.iterrows():
        if len(_selected) >= max_alerts:
            break

        _frag_smiles = _row["smiles"]
        _compounds = fragment_to_compounds.get(_frag_smiles, [])

        # Recompute PPV on uncovered compounds
        _uncovered = [c for c in _compounds if c not in _covered_compounds]
        if len(_uncovered) < 3:  # Skip if too few uncovered
            continue

        _tp_new = sum(_jcp_to_active.get(_jcp, 0) for _jcp in _uncovered)
        _ppv_new = _tp_new / len(_uncovered) if _uncovered else 0

        if _ppv_new >= min_ppv:
            _selected.append(
                {
                    "smiles": _frag_smiles,
                    "n_hits": len(_uncovered),
                    "tp": _tp_new,
                    "fp": len(_uncovered) - _tp_new,
                    "ppv": _ppv_new,
                    "rank": len(_selected) + 1,
                }
            )
            _covered_compounds.update(_uncovered)

    logger.info(f"Selected {len(_selected)} structural alerts")
    return pd.DataFrame(_selected)


# =============================================================================
# Visualization
# =============================================================================


@app.function
def plot_ppv_distribution(alerts_df: pd.DataFrame) -> plt.Figure:
    """Plot distribution of PPV values across all fragments."""
    _fig, _ax = plt.subplots(figsize=(8, 5))

    _ax.hist(alerts_df["ppv"], bins=50, edgecolor="black", alpha=0.7)
    _ax.axvline(DEFAULT_MIN_PPV, color="red", linestyle="--", label=f"Threshold = {DEFAULT_MIN_PPV}")
    _ax.set_xlabel("Positive Predictive Value (PPV)")
    _ax.set_ylabel("Number of Fragments")
    _ax.set_title("PPV Distribution of BRICS Fragments")
    _ax.legend()

    _fig.tight_layout()
    return _fig


@app.function
def plot_top_alerts(selected_alerts: pd.DataFrame, top_n: int = 20) -> plt.Figure | None:
    """Plot bar chart of top alerts by PPV."""
    if len(selected_alerts) == 0:
        logger.warning("No alerts to plot")
        return None

    _top = selected_alerts.head(top_n)

    _fig, _ax = plt.subplots(figsize=(10, 6))

    _y_pos = np.arange(len(_top))
    _bars = _ax.barh(_y_pos, _top["ppv"], color="steelblue", edgecolor="black")

    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels([f"Alert {i + 1}" for i in range(len(_top))])
    _ax.invert_yaxis()
    _ax.set_xlabel("Positive Predictive Value (PPV)")
    _ax.set_title(f"Top {len(_top)} Structural Alerts by PPV")
    _ax.axvline(DEFAULT_MIN_PPV, color="red", linestyle="--", alpha=0.7)

    # Add hit counts as labels
    for _bar, _hits in zip(_bars, _top["n_hits"]):
        _ax.text(_bar.get_width() + 0.01, _bar.get_y() + _bar.get_height() / 2, f"n={_hits}", va="center", fontsize=8)

    _fig.tight_layout()
    return _fig


@app.function
def plot_alert_structures(selected_alerts: pd.DataFrame, top_n: int = 20):
    """Plot grid of top alert structures. Returns a PIL Image."""
    if len(selected_alerts) == 0:
        logger.warning("No alerts to plot")
        return None

    _top = selected_alerts.head(top_n)

    _mols = []
    _legends = []
    for _, _row in _top.iterrows():
        _mol = Chem.MolFromSmiles(_row["smiles"])
        if _mol is not None:
            _mols.append(_mol)
            _legends.append(f"PPV={_row['ppv']:.2f}\nn={_row['n_hits']}")

    if not _mols:
        logger.warning("No valid molecules to draw")
        return None

    # Calculate grid dimensions
    _n_cols = min(5, len(_mols))

    _img = Draw.MolsToGridImage(
        _mols,
        molsPerRow=_n_cols,
        subImgSize=(250, 200),
        legends=_legends,
        useSVG=False,
    )

    return _img


@app.function
def plot_coverage(
    selected_alerts: pd.DataFrame,
    fragment_to_compounds: dict[str, list[str]],
    dataset: pd.DataFrame,
) -> plt.Figure | None:
    """Plot cumulative coverage of active compounds by alerts."""
    if len(selected_alerts) == 0:
        logger.warning("No alerts for coverage plot")
        return None

    _total_active = dataset["is_active"].sum()
    _jcp_to_active = dict(zip(dataset["Metadata_JCP2022"], dataset["is_active"]))

    _covered = set()
    _coverage = []

    for _, _row in selected_alerts.iterrows():
        _compounds = fragment_to_compounds.get(_row["smiles"], [])
        _active_compounds = {c for c in _compounds if _jcp_to_active.get(c, 0) == 1}
        _covered.update(_active_compounds)
        _coverage.append(len(_covered) / _total_active * 100)

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.plot(range(1, len(_coverage) + 1), _coverage, marker="o", markersize=3)
    _ax.set_xlabel("Number of Alerts")
    _ax.set_ylabel("Coverage of Active Compounds (%)")
    _ax.set_title("Cumulative Coverage of Active Compounds")
    _ax.grid(True, alpha=0.3)

    _fig.tight_layout()
    return _fig


@app.function
def plot_alerts_on_structure_umap(
    selected_alerts: pd.DataFrame,
    fragment_to_compounds: dict[str, list[str]],
    top_n: int = 10,
) -> plt.Figure | None:
    """Plot top alerts highlighted on structure UMAP.

    Creates a single UMAP plot with all compounds containing any of the
    top structural alerts highlighted in red. Uses IQR-based outlier clipping.
    """
    _structure_umap_path = PROCESSED_DATA_DIR / "chemical-space" / "structure_umap.h5ad"

    if not _structure_umap_path.exists():
        logger.warning(f"Structure UMAP not found at {_structure_umap_path}")
        return None

    if len(selected_alerts) == 0:
        logger.warning("No alerts to plot on UMAP")
        return None

    logger.info("Loading structure UMAP...")
    _adata = sc.read_h5ad(_structure_umap_path)

    # Use IQR-based bounds for outlier clipping
    _xlim, _ylim = compute_umap_bounds(_adata, iqr_k=3.0)
    _umap_coords = _adata.obsm["X_umap"]
    _jcp_ids = _adata.obs["Metadata_JCP2022"].values

    _jcp_to_idx = {jcp: i for i, jcp in enumerate(_jcp_ids)}

    # Collect all compounds from top alerts
    _top = selected_alerts.head(top_n)
    _all_alert_compounds = set()
    for _, _row in _top.iterrows():
        _compounds = fragment_to_compounds.get(_row["smiles"], [])
        _all_alert_compounds.update(_compounds)

    _highlight_idx = [_jcp_to_idx[jcp] for jcp in _all_alert_compounds if jcp in _jcp_to_idx]
    logger.info(f"Highlighting {len(_highlight_idx)} compounds from top {len(_top)} alerts")

    # Create single UMAP plot
    _fig, _ax = plt.subplots(figsize=(10, 10))

    # Plot all points in gray
    _ax.scatter(
        _umap_coords[:, 0],
        _umap_coords[:, 1],
        c="lightgray",
        s=1,
        alpha=0.5,
        rasterized=True,
        label=f"All compounds (n={len(_umap_coords):,})",
    )

    # Highlight compounds with alerts
    if _highlight_idx:
        _ax.scatter(
            _umap_coords[_highlight_idx, 0],
            _umap_coords[_highlight_idx, 1],
            c="red",
            s=20,
            alpha=0.9,
            edgecolors="darkred",
            linewidths=0.5,
            label=f"Alert compounds (n={len(_highlight_idx)})",
        )

    _ax.set_xlim(_xlim)
    _ax.set_ylim(_ylim)
    _ax.set_xlabel("UMAP 1", fontsize=12)
    _ax.set_ylabel("UMAP 2", fontsize=12)
    _ax.set_title(
        f"Structural Alerts on Structure UMAP\n(Top {len(_top)} alerts, {len(_highlight_idx)} compounds)", fontsize=14
    )
    _ax.legend(loc="upper right", markerscale=1.5, fontsize=10)
    _ax.set_aspect("equal")

    _fig.tight_layout()
    return _fig


# =============================================================================
# Pipeline cells
# =============================================================================


@app.cell
def _(dataset_input, mo, preprocessing_input):
    mo.stop(
        not METADATA_DB.exists(),
        mo.md(f"**Metadata database not found:** `{METADATA_DB}`"),
    )
    mo.stop(
        not COPAIRS_RESULTS_DB.exists(),
        mo.md(f"**Copairs results database not found:** `{COPAIRS_RESULTS_DB}`"),
    )

    _dataset = dataset_input.value
    _preprocessing = preprocessing_input.value

    smiles_df = load_smiles()

    mo.md(f"Loaded **{len(smiles_df):,}** compounds with SMILES from metadata database.")
    return (smiles_df,)


@app.cell
def _(dataset_input, p_threshold_slider, preprocessing_input):
    activity_df = load_activity_labels(
        dataset=dataset_input.value,
        preprocessing=preprocessing_input.value,
        p_threshold=p_threshold_slider.value,
    )
    return (activity_df,)


@app.cell
def _(activity_df, mo, smiles_df):
    compound_df = prepare_dataset(activity_df, smiles_df)

    _n_active = compound_df["is_active"].sum()
    _pct_active = _n_active / len(compound_df) * 100
    mo.md(f"**Dataset:** {len(compound_df):,} compounds, {_n_active:,} active ({_pct_active:.1f}%)")
    return (compound_df,)


@app.cell
def _(compound_df, mo):
    mo.md("### Fragment Enumeration")
    fragment_to_compounds = enumerate_brics_fragments(compound_df)
    mo.md(f"Found **{len(fragment_to_compounds):,}** unique BRICS fragments.")
    return (fragment_to_compounds,)


@app.cell
def _(compound_df, fragment_to_compounds, mo):
    mo.md("### Fragment Evaluation and Filtering")
    alerts_df = evaluate_fragments(fragment_to_compounds, compound_df)
    alerts_df = filter_small_fragments(alerts_df, min_atoms=5)
    alerts_df = filter_common_substructures(alerts_df)
    mo.md(f"**{len(alerts_df):,}** fragments remaining after filtering.")
    return (alerts_df,)


@app.cell
def _(alerts_df):
    _fig = plot_ppv_distribution(alerts_df)
    _fig
    return


@app.cell
def _(alerts_df, compound_df, fragment_to_compounds, min_ppv_slider, mo):
    selected_alerts = greedy_select_alerts(alerts_df, fragment_to_compounds, compound_df, min_ppv=min_ppv_slider.value)

    if len(selected_alerts) > 0:
        _total_active = compound_df["is_active"].sum()
        _jcp_to_active = dict(zip(compound_df["Metadata_JCP2022"], compound_df["is_active"]))
        _covered = set()
        for _, _row in selected_alerts.iterrows():
            _compounds = fragment_to_compounds.get(_row["smiles"], [])
            _active_compounds = {c for c in _compounds if _jcp_to_active.get(c, 0) == 1}
            _covered.update(_active_compounds)
        coverage_pct = len(_covered) / _total_active * 100
    else:
        coverage_pct = 0.0

    mo.md(
        f"### Greedy Selection Results\n\n"
        f"- **{len(selected_alerts)}** structural alerts selected (PPV >= {min_ppv_slider.value})\n"
        f"- **Coverage:** {coverage_pct:.1f}% of active compounds"
    )
    return coverage_pct, selected_alerts


@app.cell
def _(selected_alerts):
    _fig = plot_top_alerts(selected_alerts)
    _fig
    return


@app.cell
def _(selected_alerts):
    _img = plot_alert_structures(selected_alerts)
    _img
    return


@app.cell
def _(compound_df, fragment_to_compounds, selected_alerts):
    _fig = plot_coverage(selected_alerts, fragment_to_compounds, compound_df)
    _fig
    return


@app.cell
def _(fragment_to_compounds, selected_alerts):
    _fig = plot_alerts_on_structure_umap(selected_alerts, fragment_to_compounds, top_n=10)
    _fig
    return


# =============================================================================
# Save outputs
# =============================================================================


@app.cell
def _(
    alerts_df,
    compound_df,
    coverage_pct,
    dataset_input,
    min_ppv_slider,
    mo,
    p_threshold_slider,
    preprocessing_input,
    selected_alerts,
):
    _dataset = dataset_input.value
    _preprocessing = preprocessing_input.value
    _output_dir = OUTPUT_DIR / _dataset / _preprocessing
    _output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    alerts_df.to_csv(_output_dir / "all_fragments.csv", index=False)
    selected_alerts.to_csv(_output_dir / "structural_alerts.csv", index=False)

    # Save plots
    _fig_ppv = plot_ppv_distribution(alerts_df)
    _fig_ppv.savefig(_output_dir / "ppv_distribution.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(_fig_ppv)

    _fig_top = plot_top_alerts(selected_alerts)
    if _fig_top is not None:
        _fig_top.savefig(_output_dir / "top_alerts.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(_fig_top)

    _img = plot_alert_structures(selected_alerts)
    if _img is not None:
        _img.save(str(_output_dir / "alert_structures.png"))

    # Summary JSON
    _summary = {
        "task": "SAR Vignette - Structural Alert Discovery",
        "method": "BRICS decomposition",
        "parameters": {
            "p_threshold": p_threshold_slider.value,
            "min_atoms": DEFAULT_MIN_ATOMS,
            "max_atoms": DEFAULT_MAX_ATOMS,
            "min_hits": DEFAULT_MIN_HITS,
            "min_ppv": min_ppv_slider.value,
        },
        "dataset_statistics": {
            "n_compounds": len(compound_df),
            "n_active": int(compound_df["is_active"].sum()),
            "pct_active": float(compound_df["is_active"].mean() * 100),
        },
        "fragment_statistics": {
            "n_unique_fragments": len(alerts_df) + 0,  # after filtering
            "n_fragments_evaluated": len(alerts_df),
        },
        "results": {
            "n_structural_alerts": len(selected_alerts),
            "coverage_pct": coverage_pct,
        },
    }

    if len(selected_alerts) > 0:
        _summary["top_alerts"] = selected_alerts.head(10).to_dict(orient="records")

    with open(_output_dir / "summary.json", "w") as _f:
        json.dump(_summary, _f, indent=2)

    mo.md(f"""
    ### Outputs Saved

    All results saved to `{_output_dir}`:

    - `all_fragments.csv` - all evaluated fragments with PPV
    - `structural_alerts.csv` - selected alerts after greedy selection
    - `ppv_distribution.png` - PPV distribution histogram
    - `top_alerts.png` - bar chart of top alerts
    - `alert_structures.png` - molecular structure grid
    - `summary.json` - summary statistics
    """)
    return


@app.function
def run_sar_vignette(
    dataset,
    preprocessing,
    output_dir=None,
) -> str:
    """Run the full BRICS fragment SAR analysis pipeline.

    Parameters
    ----------
    dataset : str
        Dataset name, e.g. "compound_no_source7".
    preprocessing : str
        Preprocessing name, e.g. "activity_no_target2".
    output_dir : str or Path, optional
        Output directory. Defaults to OUTPUT_DIR / dataset / preprocessing.

    Returns
    -------
    str
        Path to the output directory.
    """
    dataset = str(dataset)
    preprocessing = str(preprocessing)

    if output_dir is None:
        output_dir = OUTPUT_DIR / dataset / preprocessing
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running SAR vignette: dataset={dataset}, preprocessing={preprocessing}")

    # Step 1: Load data
    smiles_df = load_smiles()
    activity_df = load_activity_labels(
        dataset=dataset,
        preprocessing=preprocessing,
        p_threshold=DEFAULT_P_THRESHOLD,
    )
    compound_df = prepare_dataset(activity_df, smiles_df)

    # Step 2: Enumerate BRICS fragments
    fragment_to_compounds = enumerate_brics_fragments(compound_df)

    # Step 3: Evaluate and filter fragments
    alerts_df = evaluate_fragments(fragment_to_compounds, compound_df)
    alerts_df = filter_small_fragments(alerts_df, min_atoms=5)
    alerts_df = filter_common_substructures(alerts_df)

    # Step 4: Greedy selection
    selected_alerts = greedy_select_alerts(alerts_df, fragment_to_compounds, compound_df, min_ppv=DEFAULT_MIN_PPV)

    # Compute coverage
    if len(selected_alerts) > 0:
        _total_active = compound_df["is_active"].sum()
        _jcp_to_active = dict(zip(compound_df["Metadata_JCP2022"], compound_df["is_active"]))
        _covered = set()
        for _, _row in selected_alerts.iterrows():
            _compounds = fragment_to_compounds.get(_row["smiles"], [])
            _active_compounds = {c for c in _compounds if _jcp_to_active.get(c, 0) == 1}
            _covered.update(_active_compounds)
        coverage_pct = len(_covered) / _total_active * 100
    else:
        coverage_pct = 0.0

    # Step 5: Save CSVs
    alerts_df.to_csv(output_dir / "all_fragments.csv", index=False)
    selected_alerts.to_csv(output_dir / "structural_alerts.csv", index=False)

    # Step 6: Save plots
    fig_ppv = plot_ppv_distribution(alerts_df)
    fig_ppv.savefig(output_dir / "ppv_distribution.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_ppv)

    fig_top = plot_top_alerts(selected_alerts)
    if fig_top is not None:
        fig_top.savefig(output_dir / "top_alerts.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig_top)

    img = plot_alert_structures(selected_alerts)
    if img is not None:
        img.save(str(output_dir / "alert_structures.png"))

    fig_cov = plot_coverage(selected_alerts, fragment_to_compounds, compound_df)
    if fig_cov is not None:
        fig_cov.savefig(output_dir / "coverage.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig_cov)

    fig_umap = plot_alerts_on_structure_umap(selected_alerts, fragment_to_compounds, top_n=10)
    if fig_umap is not None:
        fig_umap.savefig(output_dir / "alerts_on_umap.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig_umap)

    # Step 7: Save summary JSON
    summary = {
        "task": "SAR Vignette - Structural Alert Discovery",
        "method": "BRICS decomposition",
        "parameters": {
            "p_threshold": DEFAULT_P_THRESHOLD,
            "min_atoms": DEFAULT_MIN_ATOMS,
            "max_atoms": DEFAULT_MAX_ATOMS,
            "min_hits": DEFAULT_MIN_HITS,
            "min_ppv": DEFAULT_MIN_PPV,
        },
        "dataset_statistics": {
            "n_compounds": len(compound_df),
            "n_active": int(compound_df["is_active"].sum()),
            "pct_active": float(compound_df["is_active"].mean() * 100),
        },
        "fragment_statistics": {
            "n_unique_fragments": len(alerts_df),
            "n_fragments_evaluated": len(alerts_df),
        },
        "results": {
            "n_structural_alerts": len(selected_alerts),
            "coverage_pct": coverage_pct,
        },
    }
    if len(selected_alerts) > 0:
        summary["top_alerts"] = selected_alerts.head(10).to_dict(orient="records")

    Path(output_dir, "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info(f"SAR vignette complete: {len(selected_alerts)} alerts, {coverage_pct:.1f}% coverage")
    return str(output_dir)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
