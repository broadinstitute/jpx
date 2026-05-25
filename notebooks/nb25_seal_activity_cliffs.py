# NOTE: Run with pixi run -e rapids marimo edit/run
# (cupy/rapids_singlecell are conda-only and come from the pixi env)
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anndata==0.12.16",
#     "duckdb==1.5.3",
#     "loguru==0.7.3",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==2.3.3",
#     "python-dotenv",
#     "scanpy==1.12.1",
#     "scipy==1.17.1",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import json
    import sys
    from functools import lru_cache
    from pathlib import Path

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from loguru import logger
    from scipy.stats import spearmanr

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import DEFAULT_DPI, INTERIM_DATA_DIR, METADATA_DB, PROCESSED_DATA_DIR
    from nb03_ss_profiles import load_profiles
    from nb04_ss_visualization import plot_hexbin_with_marginals


# -- Inlined helpers from src/jump_production/gpu.py and statistics.py --


@app.function
@lru_cache(maxsize=1)
def has_gpu() -> bool:
    """Check whether a CUDA GPU is available via CuPy."""
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"CuPy installed but GPU unavailable ({type(e).__name__}: {e}), falling back to CPU")
        return False


@app.function
def permutation_null_cliff_rate(
    struct_sim: np.ndarray,
    morph_sim: np.ndarray,
    struct_threshold: float = 0.7,
    morph_threshold: float = 0.3,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> dict:
    """Compute permutation null for activity cliff rate.

    Uses CuPy for GPU acceleration when available, falls back to NumPy.
    """
    xp = np
    use_cupy = False
    if has_gpu():
        import cupy as cp

        xp = cp
        use_cupy = True
        struct_sim = cp.asarray(struct_sim)
        morph_sim = cp.asarray(morph_sim)
        logger.debug("Using CuPy GPU acceleration for permutation test")
    else:
        logger.debug("No GPU available, using NumPy for permutation test")

    xp.random.seed(random_state)

    high_struct_mask = struct_sim >= struct_threshold
    n_high_struct = int(high_struct_mask.sum())

    if n_high_struct == 0:
        return {
            "observed_rate": 0.0,
            "null_mean": 0.0,
            "null_std": 0.0,
            "z_score": 0.0,
            "p_value": 1.0,
            "n_permutations": n_permutations,
        }

    observed_cliffs = int((high_struct_mask & (morph_sim < morph_threshold)).sum())
    observed_rate = observed_cliffs / n_high_struct

    null_rates = xp.zeros(n_permutations, dtype=xp.float32)
    n_samples = len(morph_sim)

    for i in range(n_permutations):
        perm_idx = xp.random.permutation(n_samples)
        morph_sim_shuffled = morph_sim[perm_idx]
        null_cliffs = (high_struct_mask & (morph_sim_shuffled < morph_threshold)).sum()
        null_rates[i] = null_cliffs / n_high_struct

    if use_cupy:
        null_rates = null_rates.get()
        observed_rate = float(observed_rate)

    null_mean = float(np.mean(null_rates))
    null_std = float(np.std(null_rates))

    if null_std > 0:
        z_score = (observed_rate - null_mean) / null_std
        p_value = float(np.mean(np.abs(null_rates - null_mean) >= np.abs(observed_rate - null_mean)))
    else:
        z_score = 0.0 if observed_rate == null_mean else float("inf") * np.sign(observed_rate - null_mean)
        p_value = 1.0 if observed_rate == null_mean else 0.0

    return {
        "observed_rate": float(observed_rate),
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


@app.function
def permutation_null_correlation(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 100,
    subsample_size: int | None = 100000,
    random_state: int = 42,
) -> dict:
    """Compute permutation null for Spearman correlation."""
    rng = np.random.default_rng(random_state)

    if subsample_size is not None and len(x) > subsample_size:
        idx = rng.choice(len(x), size=subsample_size, replace=False)
        x = x[idx]
        y = y[idx]

    observed_r, _ = spearmanr(x, y)

    null_rs = np.zeros(n_permutations)
    y_copy = y.copy()

    for i in range(n_permutations):
        rng.shuffle(y_copy)
        null_rs[i], _ = spearmanr(x, y_copy)

    null_mean = float(np.mean(null_rs))
    null_std = float(np.std(null_rs))

    if null_std > 0:
        z_score = (observed_r - null_mean) / null_std
    else:
        z_score = 0.0 if observed_r == null_mean else float("inf") * np.sign(observed_r - null_mean)

    return {
        "observed_r": float(observed_r),
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "n_permutations": n_permutations,
    }


# -- Configuration constants --

OUTPUT_DIR = PROCESSED_DATA_DIR / "activity-cliffs"

STRUCT_HIGH_THRESHOLD = 0.7
MORPH_LOW_THRESHOLD = 0.3
K_NEIGHBORS = 100
RANDOM_STATE = 42


# -- Data loading functions --


@app.function
def load_fingerprints() -> tuple[np.ndarray, np.ndarray]:
    """Load Morgan fingerprints (2048-bit)."""
    fp_data = np.load(INTERIM_DATA_DIR / "compound_featurization" / "morgan_fp.npz", allow_pickle=True)
    fp_array = fp_data["fingerprints"]
    fp_jcp = fp_data["jcp2022"]
    logger.info(f"Morgan fingerprints: {fp_array.shape[0]:,} compounds x {fp_array.shape[1]} bits")
    return fp_array, fp_jcp


@app.function
def load_chembl_targets(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load ChEMBL target annotations from metadata database."""
    df = con.execute("""
        SELECT
            Metadata_JCP2022 as JCP2022,
            Metadata_Uniprot_target as targets_str
        FROM chembl_protein_targets
        WHERE Metadata_Uniprot_target IS NOT NULL
    """).df()

    df["targets"] = df["targets_str"].str.split("|")
    df = df.drop(columns=["targets_str"])

    _n_compounds = len(df)
    _n_unique_targets = len(set(t for targets in df["targets"] for t in targets))
    logger.info(f"ChEMBL targets: {_n_compounds:,} compounds, {_n_unique_targets:,} unique targets")

    return df


@app.function
def align_data(
    adata_profiles,
    fp_array: np.ndarray,
    fp_jcp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align fingerprints and morphology profiles by JCP2022 ID."""
    morph_jcp = adata_profiles.obs["JCP2022"].values
    common_jcp = sorted(set(fp_jcp) & set(morph_jcp))
    logger.info(f"Common compounds: {len(common_jcp):,}")

    fp_idx = {k: i for i, k in enumerate(fp_jcp)}
    morph_idx = {k: i for i, k in enumerate(morph_jcp)}

    fp_order = [fp_idx[k] for k in common_jcp]
    morph_order = [morph_idx[k] for k in common_jcp]

    X_fp = fp_array[fp_order].astype(np.float32)
    X_morph = adata_profiles.X[morph_order].astype(np.float32)
    jcp_ids = np.array(common_jcp)

    fp_norms = np.linalg.norm(X_fp, axis=1)
    morph_norms = np.linalg.norm(X_morph, axis=1)
    valid_mask = (fp_norms > 0) & (morph_norms > 0) & ~np.isnan(morph_norms)
    logger.info(f"Valid compounds: {valid_mask.sum():,}")

    return X_fp[valid_mask], X_morph[valid_mask], jcp_ids[valid_mask]


# -- k-NN based pairwise similarity computation --


@app.function
def compute_structural_knn(
    X_fp: np.ndarray,
    jcp_ids: np.ndarray,
    k: int = K_NEIGHBORS,
) -> sc.AnnData:
    """Compute k-NN in structural (fingerprint) space.

    Uses rapids_singlecell for GPU acceleration when available, falls back to scanpy.
    """
    X_fp_norm = X_fp / np.linalg.norm(X_fp, axis=1, keepdims=True)

    logger.info(f"Computing structural k-NN (k={k})...")
    adata = sc.AnnData(X=X_fp_norm)
    adata.obs["JCP2022"] = jcp_ids
    adata.obs_names = jcp_ids.astype(str)

    if has_gpu():
        import rapids_singlecell as rsc

        neighbors_fn = rsc.pp.neighbors
        logger.info("Using GPU-accelerated k-NN (rapids_singlecell)")
    else:
        neighbors_fn = sc.pp.neighbors
        logger.info("Using CPU k-NN (scanpy)")

    neighbors_fn(adata, n_neighbors=k, use_rep="X", metric="cosine")
    logger.info(f"  Edges: {adata.obsp['connectivities'].nnz:,}")

    return adata


@app.function
def extract_pairwise_similarities(
    adata_struct: sc.AnnData,
    X_fp: np.ndarray,
    X_morph: np.ndarray,
    jcp_ids: np.ndarray,
) -> pd.DataFrame:
    """Extract pairwise similarities for all k-NN pairs."""
    logger.info("Extracting pairwise similarities for k-NN pairs...")

    X_fp_norm = X_fp / np.linalg.norm(X_fp, axis=1, keepdims=True)
    X_morph_norm = X_morph / np.linalg.norm(X_morph, axis=1, keepdims=True)

    dist_matrix = adata_struct.obsp["distances"]
    rows, cols = dist_matrix.nonzero()

    upper_mask = rows < cols
    rows = rows[upper_mask]
    cols = cols[upper_mask]

    logger.info(f"  Computing similarities for {len(rows):,} pairs...")

    struct_sim = np.einsum("ij,ij->i", X_fp_norm[rows], X_fp_norm[cols])
    morph_sim = np.einsum("ij,ij->i", X_morph_norm[rows], X_morph_norm[cols])

    pairs_df = pd.DataFrame(
        {
            "i": rows,
            "j": cols,
            "jcp_i": jcp_ids[rows],
            "jcp_j": jcp_ids[cols],
            "struct_sim": struct_sim,
            "morph_sim": morph_sim,
        }
    )

    logger.info(f"  Extracted {len(pairs_df):,} unique pairs")

    return pairs_df


# -- Q1: Structure-Morphology cliff analysis --


@app.function
def analyze_structure_morphology_relationship(
    pairs_df: pd.DataFrame,
    struct_high: float = STRUCT_HIGH_THRESHOLD,
    morph_low: float = MORPH_LOW_THRESHOLD,
) -> dict:
    """Analyze the relationship between structural and morphological similarity."""
    logger.info("Analyzing structure-morphology relationship...")

    n_pairs = len(pairs_df)
    struct_sim = pairs_df["struct_sim"].values
    morph_sim = pairs_df["morph_sim"].values

    spearman_r, spearman_p = spearmanr(struct_sim, morph_sim)
    logger.info(f"  Spearman correlation: rho = {spearman_r:.4f} (p = {spearman_p:.2e})")

    morph_high = 1 - morph_low

    high_struct_mask = struct_sim >= struct_high
    low_struct_mask = struct_sim < struct_high
    high_morph_mask = morph_sim >= morph_high
    low_morph_mask = morph_sim < morph_low

    n_concordant = int((high_struct_mask & high_morph_mask).sum())
    n_cliffs = int((high_struct_mask & low_morph_mask).sum())
    n_reverse = int((low_struct_mask & high_morph_mask).sum())
    n_neutral = int((low_struct_mask & low_morph_mask).sum())

    pct_cliffs = n_cliffs / n_pairs * 100 if n_pairs > 0 else 0
    pct_concordant = n_concordant / n_pairs * 100 if n_pairs > 0 else 0

    stats = {
        "n_pairs": n_pairs,
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "thresholds": {
            "struct_high": struct_high,
            "morph_low": morph_low,
            "morph_high": float(morph_high),
        },
        "quadrants": {
            "concordant": n_concordant,
            "cliffs": n_cliffs,
            "reverse_cliffs": n_reverse,
            "neutral": n_neutral,
        },
        "percentages": {
            "pct_cliffs": pct_cliffs,
            "pct_concordant": pct_concordant,
        },
        "summary_stats": {
            "struct_sim_mean": float(struct_sim.mean()),
            "struct_sim_std": float(struct_sim.std()),
            "morph_sim_mean": float(morph_sim.mean()),
            "morph_sim_std": float(morph_sim.std()),
        },
    }

    logger.info("  Quadrant counts:")
    logger.info(f"    Concordant (high-high): {n_concordant:,} ({pct_concordant:.2f}%)")
    logger.info(f"    Cliffs (high struct, low morph): {n_cliffs:,} ({pct_cliffs:.2f}%)")
    logger.info(f"    Reverse cliffs: {n_reverse:,}")
    logger.info(f"    Neutral: {n_neutral:,}")

    return stats


@app.function
def identify_cliff_pairs(
    pairs_df: pd.DataFrame,
    struct_high: float = STRUCT_HIGH_THRESHOLD,
    morph_low: float = MORPH_LOW_THRESHOLD,
    top_n: int = 100,
) -> pd.DataFrame:
    """Identify the most extreme activity cliff pairs."""
    logger.info(f"Identifying top {top_n} cliff pairs...")

    cliff_candidates = pairs_df[pairs_df["struct_sim"] >= struct_high].copy()
    logger.info(f"  Candidates (struct >= {struct_high}): {len(cliff_candidates):,}")

    cliff_pairs = cliff_candidates[cliff_candidates["morph_sim"] < morph_low].copy()
    logger.info(f"  Cliffs (morph < {morph_low}): {len(cliff_pairs):,}")

    if len(cliff_pairs) == 0:
        logger.warning("  No cliff pairs found with current thresholds")
        return pd.DataFrame()

    cliff_pairs["cliff_score"] = cliff_pairs["struct_sim"] - cliff_pairs["morph_sim"]
    cliff_pairs = cliff_pairs.sort_values("cliff_score", ascending=False).head(top_n)

    return cliff_pairs[["jcp_i", "jcp_j", "struct_sim", "morph_sim", "cliff_score"]]


# -- Q2: Target-informed cliff analysis --


@app.function
def analyze_target_informed_cliffs(
    pairs_df: pd.DataFrame,
    target_df: pd.DataFrame,
    struct_high: float = STRUCT_HIGH_THRESHOLD,
    morph_low: float = MORPH_LOW_THRESHOLD,
    min_compounds: int = 5,
) -> dict:
    """Analyze cliffs within compounds sharing the same ChEMBL target."""
    logger.info("Analyzing target-informed cliffs...")

    expanded = target_df.explode("targets")
    expanded = expanded.rename(columns={"targets": "target"})
    logger.info(f"  Expanded annotations: {len(expanded):,} compound-target pairs")

    target_counts = expanded.groupby("target").size().reset_index(name="n_compounds")
    targets_with_coverage = target_counts[target_counts["n_compounds"] >= min_compounds]
    logger.info(f"  Targets with >= {min_compounds} compounds: {len(targets_with_coverage):,}")

    jcp_to_targets = expanded.groupby("JCP2022")["target"].apply(set).to_dict()

    logger.info("  Finding shared targets for pairs...")
    shared_targets = []
    for _, row in pairs_df.iterrows():
        jcp_i, jcp_j = row["jcp_i"], row["jcp_j"]
        targets_i = jcp_to_targets.get(jcp_i, set())
        targets_j = jcp_to_targets.get(jcp_j, set())
        shared = targets_i & targets_j
        shared_targets.append(shared if shared else None)

    pairs_with_targets = pairs_df.copy()
    pairs_with_targets["shared_targets"] = shared_targets

    target_pairs = pairs_with_targets[pairs_with_targets["shared_targets"].notna()].copy()
    logger.info(f"  Pairs with shared targets: {len(target_pairs):,}")

    if len(target_pairs) == 0:
        logger.warning("  No pairs with shared targets found")
        return {"n_target_pairs": 0}

    target_pair_stats = analyze_structure_morphology_relationship(target_pairs, struct_high, morph_low)
    target_pair_stats["n_target_pairs"] = len(target_pairs)

    target_cliffs = target_pairs[(target_pairs["struct_sim"] >= struct_high) & (target_pairs["morph_sim"] < morph_low)]
    target_pair_stats["n_target_cliffs"] = len(target_cliffs)

    target_cliff_counts = {}
    for _, row in target_cliffs.iterrows():
        for target in row["shared_targets"]:
            target_cliff_counts[target] = target_cliff_counts.get(target, 0) + 1

    top_cliff_targets = sorted(target_cliff_counts.items(), key=lambda x: -x[1])[:20]
    target_pair_stats["top_cliff_targets"] = [{"target": t, "n_cliffs": n} for t, n in top_cliff_targets]

    logger.info(f"  Target cliffs found: {len(target_cliffs):,}")
    if top_cliff_targets:
        logger.info(f"  Top cliff target: {top_cliff_targets[0][0]} ({top_cliff_targets[0][1]} cliffs)")

    return target_pair_stats


# -- Visualization functions --


@app.function
def plot_structure_morphology_hexbin(
    pairs_df: pd.DataFrame,
    struct_high: float = STRUCT_HIGH_THRESHOLD,
    morph_low: float = MORPH_LOW_THRESHOLD,
) -> plt.Figure:
    """Create hexbin plot of structural vs morphological similarity with marginals."""
    _spearman_r, _ = spearmanr(pairs_df["struct_sim"], pairs_df["morph_sim"])
    _stats_annotation = f"Spearman rho = {_spearman_r:.3f}\nn = {len(pairs_df):,}"

    _n_pairs = len(pairs_df)
    _high_struct = pairs_df["struct_sim"] >= struct_high
    _low_morph = pairs_df["morph_sim"] < morph_low
    _n_cliffs = (_high_struct & _low_morph).sum()
    _stats_annotation += f"\nCliffs: {_n_cliffs:,} ({_n_cliffs / _n_pairs * 100:.2f}%)"

    fig = plot_hexbin_with_marginals(
        pairs_df,
        x_col="struct_sim",
        y_col="morph_sim",
        x_label="Structural Similarity (Tanimoto - cosine)",
        y_label="Morphological Similarity (cosine)",
        x_lim=(0, 1),
        y_lim=(-0.5, 1),
        title="Structure vs Morphology Similarity",
        stats_annotation=_stats_annotation,
    )
    return fig


@app.function
def plot_cliff_distribution(
    pairs_df: pd.DataFrame,
    struct_high: float = STRUCT_HIGH_THRESHOLD,
) -> plt.Figure:
    """Plot distribution of morphological similarity for structurally similar pairs."""
    similar_pairs = pairs_df[pairs_df["struct_sim"] >= struct_high]

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(similar_pairs) == 0:
        ax.text(0.5, 0.5, "No structurally similar pairs found", transform=ax.transAxes, ha="center")
        return fig

    ax.hist(
        similar_pairs["morph_sim"],
        bins=50,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )

    ax.set_xlabel("Morphological Similarity (cosine)")
    ax.set_ylabel("Count")
    ax.set_title(f"Morphological Similarity for Structurally Similar Pairs (struct >= {struct_high})")

    _mean_morph = similar_pairs["morph_sim"].mean()
    ax.axvline(_mean_morph, color="red", linestyle="--", label=f"Mean = {_mean_morph:.3f}")
    ax.legend()

    _stats_text = f"n = {len(similar_pairs):,}\nMean = {_mean_morph:.3f}\nStd = {similar_pairs['morph_sim'].std():.3f}"
    ax.text(
        0.95,
        0.95,
        _stats_text,
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    return fig


@app.function
def plot_target_cliff_analysis(target_stats: dict) -> plt.Figure:
    """Plot bar chart of targets with most cliffs."""
    top_targets = target_stats.get("top_cliff_targets", [])

    fig, ax = plt.subplots(figsize=(10, 6))

    if not top_targets:
        ax.text(0.5, 0.5, "No target cliff data to plot", transform=ax.transAxes, ha="center")
        return fig

    _targets = [t["target"] for t in top_targets]
    _n_cliffs = [t["n_cliffs"] for t in top_targets]

    _y_pos = np.arange(len(_targets))
    ax.barh(_y_pos, _n_cliffs, color="#ff7f0e", edgecolor="black", linewidth=0.5)

    ax.set_yticks(_y_pos)
    ax.set_yticklabels(_targets, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Cliff Pairs")
    ax.set_title("Targets with Most Activity Cliff Pairs")

    plt.tight_layout()
    return fig


# -- Summary generation --


@app.function
def generate_summary(
    q1_stats: dict,
    q2_stats: dict,
    cliff_pairs_df: pd.DataFrame,
    n_compounds: int,
    n_pairs: int,
    null_cliff_results: dict | None = None,
    null_corr_results: dict | None = None,
) -> dict:
    """Generate summary dict with all results."""
    summary = {
        "task": "Activity cliffs analysis",
        "research_questions": [
            "Q1: Do structurally similar compounds have similar morphological profiles?",
            "Q2: For compounds sharing ChEMBL targets, do similar structures yield similar phenotypes?",
        ],
        "n_compounds": n_compounds,
        "n_pairs_analyzed": n_pairs,
        "use_gpu": has_gpu(),
        "q1_structure_morphology": {
            "spearman_correlation": q1_stats["spearman_r"],
            "spearman_p_value": q1_stats["spearman_p"],
            "thresholds": q1_stats["thresholds"],
            "quadrants": q1_stats["quadrants"],
            "interpretation": "",
        },
        "q2_target_informed": {
            "n_target_pairs": q2_stats.get("n_target_pairs", 0),
            "n_target_cliffs": q2_stats.get("n_target_cliffs", 0),
            "spearman_correlation": q2_stats.get("spearman_r"),
            "top_cliff_targets": q2_stats.get("top_cliff_targets", [])[:10],
        },
        "cliff_pairs": {
            "n_cliff_pairs": len(cliff_pairs_df) if cliff_pairs_df is not None else 0,
            "top_cliff_score": float(cliff_pairs_df["cliff_score"].max()) if len(cliff_pairs_df) > 0 else None,
        },
    }

    if null_cliff_results is not None or null_corr_results is not None:
        summary["null_model"] = {}
        if null_cliff_results is not None:
            summary["null_model"]["n_permutations"] = null_cliff_results.get("n_permutations", 1000)
            summary["null_model"]["cliff_rate"] = {
                "observed": null_cliff_results["observed_rate"],
                "null_mean": null_cliff_results["null_mean"],
                "null_std": null_cliff_results["null_std"],
                "z_score": null_cliff_results["z_score"],
                "p_value": null_cliff_results["p_value"],
            }
        if null_corr_results is not None:
            summary["null_model"]["correlation"] = {
                "observed": null_corr_results["observed_r"],
                "null_mean": null_corr_results["null_mean"],
                "null_std": null_corr_results["null_std"],
                "z_score": null_corr_results["z_score"],
            }

    r = q1_stats["spearman_r"]
    if null_corr_results is not None:
        z = null_corr_results["z_score"]
        if abs(z) > 10:
            _interpretation = (
                f"Correlation (rho={r:.3f}) is highly significant vs null (z={z:.1f}): "
                "structure-morphology relationship is real but weak"
            )
        elif abs(z) > 3:
            _interpretation = (
                f"Correlation (rho={r:.3f}) is significant vs null (z={z:.1f}): structure modestly predicts morphology"
            )
        else:
            _interpretation = (
                f"Correlation (rho={r:.3f}) is not significant vs null (z={z:.1f}): "
                "no meaningful structure-morphology relationship"
            )
    else:
        if r > 0.5:
            _interpretation = f"Strong positive correlation (rho={r:.3f}): similar structure -> similar morphology"
        elif r > 0.3:
            _interpretation = f"Moderate positive correlation (rho={r:.3f}): some structure-morphology relationship"
        elif r > 0.1:
            _interpretation = f"Weak positive correlation (rho={r:.3f}): limited structure-morphology relationship"
        else:
            _interpretation = f"Very weak/no correlation (rho={r:.3f}): structure does not predict morphology"

    summary["q1_structure_morphology"]["interpretation"] = _interpretation

    return summary


# =====================================================================
# Notebook cells
# =====================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Activity Cliffs Analysis

    Investigates the relationship between structural similarity and morphological
    similarity to identify "activity cliffs" - compound pairs that are structurally
    similar but morphologically different, or vice versa.

    **Research questions:**

    - Q1: Do structurally similar compounds have similar morphological profiles?
    - Q2: For compounds sharing ChEMBL targets, do similar structures yield similar phenotypes?

    Note: ChEMBL data here is binary target annotations (compound->target: yes/no),
    not activity potency values. This analyzes "annotation cliffs" rather than
    traditional IC50-based activity cliffs.
    """)
    return


@app.cell
def _(mo):
    dataset_dropdown = mo.ui.dropdown(
        options=[
            "compound_no_source7",
            "compound_DL_CPCNN_no_source7",
        ],
        value="compound_no_source7",
        label="Dataset",
    )
    sample_slider = mo.ui.slider(
        start=0.1,
        stop=1.0,
        step=0.1,
        value=0.1,
        label="Sample fraction",
    )
    mo.hstack([dataset_dropdown, sample_slider])
    return (dataset_dropdown, sample_slider)


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run analysis")
    run_button
    return (run_button,)


@app.cell
def _(dataset_dropdown, mo, run_button, sample_slider):
    mo.stop(not run_button.value)

    mo.stop(
        not METADATA_DB.exists(),
        mo.md(f"**Error:** Metadata database not found at `{METADATA_DB}`. Run `just run` first."),
    )

    _dataset = dataset_dropdown.value
    _sample_frac = sample_slider.value

    logger.info(f"GPU acceleration: {'enabled' if has_gpu() else 'disabled'}")

    # Load data
    logger.info("Loading data...")
    fp_array, fp_jcp = load_fingerprints()
    adata = load_profiles(_dataset, level="perturbation")

    _metadata_con = duckdb.connect(str(METADATA_DB), read_only=True)
    target_df = load_chembl_targets(_metadata_con)
    _metadata_con.close()

    # Align data
    X_fp, X_morph, jcp_ids = align_data(adata, fp_array, fp_jcp)

    # Sample if requested
    if _sample_frac < 1.0:
        _n_sample = int(len(X_fp) * _sample_frac)
        _rng = np.random.default_rng(RANDOM_STATE)
        _sample_idx = _rng.choice(len(X_fp), size=_n_sample, replace=False)
        X_fp = X_fp[_sample_idx]
        X_morph = X_morph[_sample_idx]
        jcp_ids = jcp_ids[_sample_idx]
        logger.info(f"Sampled {_sample_frac:.0%} of data: {_n_sample:,} compounds")

    n_compounds = len(jcp_ids)
    logger.info(f"Final dataset: {n_compounds:,} compounds")

    mo.md(f"""
    ## Data loaded

    **Dataset:** {_dataset} | **Sample:** {_sample_frac:.0%} | **GPU:** {"yes" if has_gpu() else "no"}

    - Fingerprints: {fp_array.shape[0]:,} compounds
    - Profiles: {adata.n_obs:,} compounds
    - ChEMBL targets: {len(target_df):,} compounds
    - Aligned (after sampling): {n_compounds:,} compounds
    """)
    return (X_fp, X_morph, adata, fp_array, fp_jcp, jcp_ids, n_compounds, target_df)


# -- Compute k-NN and pairwise similarities --


@app.cell
def _(X_fp, X_morph, jcp_ids):
    adata_struct = compute_structural_knn(X_fp, jcp_ids, k=K_NEIGHBORS)
    pairs_df = extract_pairwise_similarities(adata_struct, X_fp, X_morph, jcp_ids)
    n_pairs = len(pairs_df)
    logger.info(f"Total pairs: {n_pairs:,}")
    return (adata_struct, n_pairs, pairs_df)


# -- Q1: Structure-morphology analysis --


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q1: Structure-Morphology Cliff Analysis

    Classifies compound pairs into quadrants based on structural and morphological
    similarity thresholds to identify concordant pairs, cliffs, reverse cliffs,
    and neutral pairs.
    """)
    return


@app.cell
def _(mo, pairs_df):
    q1_stats = analyze_structure_morphology_relationship(
        pairs_df,
        struct_high=STRUCT_HIGH_THRESHOLD,
        morph_low=MORPH_LOW_THRESHOLD,
    )

    _q = q1_stats["quadrants"]
    _p = q1_stats["percentages"]
    mo.md(f"""
    ### Quadrant Analysis

    | Quadrant | Count | Percentage |
    |----------|-------|------------|
    | Concordant (high struct, high morph) | {_q["concordant"]:,} | {_p["pct_concordant"]:.2f}% |
    | Cliffs (high struct, low morph) | {_q["cliffs"]:,} | {_p["pct_cliffs"]:.2f}% |
    | Reverse cliffs (low struct, high morph) | {_q["reverse_cliffs"]:,} | - |
    | Neutral (low struct, low morph) | {_q["neutral"]:,} | - |

    **Spearman rho:** {q1_stats["spearman_r"]:.4f} (p = {q1_stats["spearman_p"]:.2e})
    """)
    return (q1_stats,)


@app.cell
def _(pairs_df):
    fig_hexbin = plot_structure_morphology_hexbin(pairs_df)
    fig_hexbin
    return (fig_hexbin,)


@app.cell
def _(pairs_df):
    fig_cliff_dist = plot_cliff_distribution(pairs_df)
    fig_cliff_dist
    return (fig_cliff_dist,)


# -- Null model analysis --


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Null Model Analysis

    Permutation tests to assess whether the observed cliff rate and
    structure-morphology correlation are significant compared to
    a null model that shuffles morphological similarities.
    """)
    return


@app.cell
def _(mo, pairs_df):
    logger.info("Running permutation null model for cliff rate...")
    null_cliff_results = permutation_null_cliff_rate(
        pairs_df["struct_sim"].values,
        pairs_df["morph_sim"].values,
        struct_threshold=STRUCT_HIGH_THRESHOLD,
        morph_threshold=MORPH_LOW_THRESHOLD,
        n_permutations=1000,
    )

    logger.info("Running permutation null model for correlation...")
    null_corr_results = permutation_null_correlation(
        pairs_df["struct_sim"].values,
        pairs_df["morph_sim"].values,
        n_permutations=100,
    )

    mo.md(f"""
    ### Permutation test results

    **Cliff rate:**
    - Observed: {null_cliff_results["observed_rate"]:.4f}
    - Null: {null_cliff_results["null_mean"]:.4f} +/- {null_cliff_results["null_std"]:.4f}
    - z-score: {null_cliff_results["z_score"]:.1f}, p-value: {null_cliff_results["p_value"]:.4f}

    **Correlation:**
    - Observed Spearman rho: {null_corr_results["observed_r"]:.4f}
    - Null: {null_corr_results["null_mean"]:.4f} +/- {null_corr_results["null_std"]:.4f}
    - z-score: {null_corr_results["z_score"]:.1f}
    """)
    return (null_cliff_results, null_corr_results)


# -- Top cliff pairs --


@app.cell
def _(mo, pairs_df):
    cliff_pairs_df = identify_cliff_pairs(
        pairs_df,
        struct_high=STRUCT_HIGH_THRESHOLD,
        morph_low=MORPH_LOW_THRESHOLD,
        top_n=100,
    )

    if len(cliff_pairs_df) > 0:
        mo.md(f"""
        ### Top Cliff Pairs

        Top 20 most extreme activity cliff pairs (out of {len(cliff_pairs_df)} total):

        {cliff_pairs_df.head(20).to_markdown(index=False)}
        """)
    else:
        mo.md("No cliff pairs found with current thresholds.")
    return (cliff_pairs_df,)


# -- Q2: Target-informed analysis --


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q2: Target-Informed Cliff Analysis

    For compounds sharing the same ChEMBL target, do structurally similar
    compounds yield similar phenotypes? Identifies which targets have the
    most cliff pairs.
    """)
    return


@app.cell
def _(jcp_ids, mo, pairs_df, target_df):
    _target_df_filtered = target_df[target_df["JCP2022"].isin(jcp_ids)]
    logger.info(f"Compounds with target annotations: {len(_target_df_filtered):,}")

    q2_stats = analyze_target_informed_cliffs(
        pairs_df,
        _target_df_filtered,
        struct_high=STRUCT_HIGH_THRESHOLD,
        morph_low=MORPH_LOW_THRESHOLD,
    )

    mo.md(f"""
    ### Target-informed results

    - Pairs with shared targets: {q2_stats.get("n_target_pairs", 0):,}
    - Target cliffs: {q2_stats.get("n_target_cliffs", 0):,}
    - Spearman rho (target pairs): {q2_stats.get("spearman_r", "N/A")}
    """)
    return (q2_stats,)


@app.cell
def _(q2_stats):
    fig_target_cliffs = plot_target_cliff_analysis(q2_stats)
    fig_target_cliffs
    return (fig_target_cliffs,)


# -- Save outputs --


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save all outputs")
    save_button
    return (save_button,)


@app.cell
def _(
    cliff_pairs_df,
    dataset_dropdown,
    fig_cliff_dist,
    fig_hexbin,
    fig_target_cliffs,
    mo,
    n_compounds,
    n_pairs,
    null_cliff_results,
    null_corr_results,
    pairs_df,
    q1_stats,
    q2_stats,
    save_button,
):
    mo.stop(not save_button.value)

    _dataset = dataset_dropdown.value
    _outdir = OUTPUT_DIR / _dataset
    _outdir.mkdir(parents=True, exist_ok=True)

    fig_hexbin.savefig(_outdir / "structure_morphology_hexbin.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_hexbin.savefig(_outdir / "structure_morphology_hexbin.pdf", bbox_inches="tight")

    fig_cliff_dist.savefig(_outdir / "cliff_distribution.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_cliff_dist.savefig(_outdir / "cliff_distribution.pdf", bbox_inches="tight")

    if q2_stats.get("top_cliff_targets"):
        fig_target_cliffs.savefig(_outdir / "target_cliff_analysis.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        fig_target_cliffs.savefig(_outdir / "target_cliff_analysis.pdf", bbox_inches="tight")

    pairs_df.to_parquet(_outdir / "all_pairs.parquet", index=False)

    if len(cliff_pairs_df) > 0:
        cliff_pairs_df.to_csv(_outdir / "cliff_pairs.csv", index=False)

    _summary = generate_summary(
        q1_stats,
        q2_stats,
        cliff_pairs_df,
        n_compounds,
        n_pairs,
        null_cliff_results=null_cliff_results,
        null_corr_results=null_corr_results,
    )

    with open(_outdir / "summary.json", "w") as _f:
        json.dump(_summary, _f, indent=2)

    mo.md(f"""
    **Saved all outputs to:** `{_outdir}`

    - `structure_morphology_hexbin.png/.pdf`
    - `cliff_distribution.png/.pdf`
    - `target_cliff_analysis.png/.pdf`
    - `all_pairs.parquet`
    - `cliff_pairs.csv`
    - `summary.json`
    """)
    return


@app.function
def run_activity_cliffs(
    dataset: str = "compound_no_source7",
    sample=1.0,
    output_dir=None,
) -> str:
    """Run the full activity cliffs analysis.

    Composes load_fingerprints, load_profiles, load_chembl_targets,
    align_data, compute_structural_knn, extract_pairwise_similarities,
    analyze_structure_morphology_relationship, identify_cliff_pairs,
    analyze_target_informed_cliffs, permutation_null_cliff_rate,
    permutation_null_correlation, plot_structure_morphology_hexbin,
    and generate_summary. Saves hexbin plot, cliff pairs, summary.json,
    and .complete marker.

    Called from workflow.py via run_task.py.

    Args:
        dataset: Dataset name (e.g., "compound_no_source7")
        sample: Fraction of data to use (0.0-1.0). Accepts str from subprocess.
        output_dir: Output directory (default: data/processed/activity-cliffs/...)

    Returns:
        Path to .complete marker file.
    """
    sample = float(sample) if isinstance(sample, str) else sample

    if output_dir is None:
        output_dir = OUTPUT_DIR / dataset
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    _fp_array, _fp_jcp = load_fingerprints()
    _adata = load_profiles(dataset, level="perturbation")

    _metadata_con = duckdb.connect(str(METADATA_DB), read_only=True)
    _target_df = load_chembl_targets(_metadata_con)
    _metadata_con.close()

    # Align data
    _X_fp, _X_morph, _jcp_ids = align_data(_adata, _fp_array, _fp_jcp)

    # Sample if requested
    _sample_frac = sample
    if _sample_frac < 1.0:
        _n_sample = int(len(_X_fp) * _sample_frac)
        _rng = np.random.default_rng(RANDOM_STATE)
        _sample_idx = _rng.choice(len(_X_fp), size=_n_sample, replace=False)
        _X_fp = _X_fp[_sample_idx]
        _X_morph = _X_morph[_sample_idx]
        _jcp_ids = _jcp_ids[_sample_idx]
        logger.info(f"Sampled {_sample_frac:.0%} of data: {_n_sample:,} compounds")

    _n_compounds = len(_jcp_ids)

    # Compute k-NN and pairwise similarities
    _adata_struct = compute_structural_knn(_X_fp, _jcp_ids, k=K_NEIGHBORS)
    _pairs_df = extract_pairwise_similarities(_adata_struct, _X_fp, _X_morph, _jcp_ids)
    _n_pairs = len(_pairs_df)

    # Q1: Structure-morphology analysis
    _q1_stats = analyze_structure_morphology_relationship(
        _pairs_df, struct_high=STRUCT_HIGH_THRESHOLD, morph_low=MORPH_LOW_THRESHOLD
    )

    # Null model analysis
    _null_cliff_results = permutation_null_cliff_rate(
        _pairs_df["struct_sim"].values,
        _pairs_df["morph_sim"].values,
        struct_threshold=STRUCT_HIGH_THRESHOLD,
        morph_threshold=MORPH_LOW_THRESHOLD,
        n_permutations=1000,
    )
    _null_corr_results = permutation_null_correlation(
        _pairs_df["struct_sim"].values,
        _pairs_df["morph_sim"].values,
        n_permutations=100,
    )

    # Cliff pairs
    _cliff_pairs_df = identify_cliff_pairs(
        _pairs_df, struct_high=STRUCT_HIGH_THRESHOLD, morph_low=MORPH_LOW_THRESHOLD, top_n=100
    )

    # Q2: Target-informed analysis
    _target_df_filtered = _target_df[_target_df["JCP2022"].isin(_jcp_ids)]
    _q2_stats = analyze_target_informed_cliffs(
        _pairs_df, _target_df_filtered, struct_high=STRUCT_HIGH_THRESHOLD, morph_low=MORPH_LOW_THRESHOLD
    )

    # Save hexbin plot
    _fig_hexbin = plot_structure_morphology_hexbin(_pairs_df)
    _fig_hexbin.savefig(output_dir / "structure_morphology_hexbin.png", dpi=150, bbox_inches="tight", facecolor="white")
    _fig_hexbin.savefig(output_dir / "structure_morphology_hexbin.pdf", bbox_inches="tight")
    plt.close(_fig_hexbin)

    # Save cliff distribution plot
    _fig_cliff_dist = plot_cliff_distribution(_pairs_df)
    _fig_cliff_dist.savefig(output_dir / "cliff_distribution.png", dpi=150, bbox_inches="tight", facecolor="white")
    _fig_cliff_dist.savefig(output_dir / "cliff_distribution.pdf", bbox_inches="tight")
    plt.close(_fig_cliff_dist)

    # Save data
    _pairs_df.to_parquet(output_dir / "all_pairs.parquet", index=False)
    if len(_cliff_pairs_df) > 0:
        _cliff_pairs_df.to_csv(output_dir / "cliff_pairs.csv", index=False)

    # Generate and save summary
    _summary = generate_summary(
        _q1_stats,
        _q2_stats,
        _cliff_pairs_df,
        _n_compounds,
        _n_pairs,
        null_cliff_results=_null_cliff_results,
        null_corr_results=_null_corr_results,
    )
    Path(output_dir, "summary.json").write_text(json.dumps(_summary, indent=2))

    # Write .complete marker
    _marker = output_dir / ".complete"
    _marker.touch()
    logger.success(f"Saved activity cliffs outputs to {output_dir}")
    return str(_marker)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
