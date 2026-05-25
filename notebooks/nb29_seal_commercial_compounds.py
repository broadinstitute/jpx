# NOTE: Run with pixi run -e cheminformatics marimo edit/run
# (rdkit is conda-only and comes from the pixi env)
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anndata",
#     "duckdb",
#     "loguru",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "python-dotenv",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import json
    import sys
    import zipfile
    from pathlib import Path

    import anndata as ad
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from loguru import logger
    from matplotlib.colors import Normalize
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.cluster import KMeans

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import (
        DEFAULT_DPI,
        EXTERNAL_DATA_DIR,
        PROCESSED_DATA_DIR,
    )
    from nb03_ss_profiles import join_activity, join_metadata, load_profiles, load_umap
    from nb04_ss_visualization import compute_umap_bounds

    CHEMICAL_SPACE_DIR = PROCESSED_DATA_DIR / "chemical-space"
    OUTPUT_DIR = PROCESSED_DATA_DIR / "commercial-compounds"

    # Molport batch search data (zip of batch inputs + Molport quote results)
    MOLPORT_ZIP = EXTERNAL_DATA_DIR / "molport_batch_search.zip"

    # Price cutoff for Molport compounds (applied early in filtering)
    MAX_PRICE_USD = 200

    # Clustering configurations: (n_clusters, n_per_cluster) -> output name
    # 320 compounds fits a 384-well plate with room for controls
    CLUSTER_CONFIGS = [
        (160, 1, "160c_1pc"),  # 160 clusters x 1 = 160 compounds
        (160, 2, "160c_2pc"),  # 160 clusters x 2 = 320 compounds
        (320, 1, "320c_1pc"),  # 320 clusters x 1 = 320 compounds
        (320, 2, "320c_2pc"),  # 320 clusters x 2 = 640 compounds
    ]

    RANDOM_STATE = 42


@app.function
def has_gpu():
    """Check whether a CUDA GPU is available via CuPy.

    Imports cupy and probes the CUDA runtime driver. Returns False when cupy is
    not installed or when the driver/device is unavailable.
    """
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        return True
    except ImportError:
        return False
    except Exception:
        return False


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Commercial Compounds Curation

    Curates commercially available compounds with distinctive Cell Painting profiles
    for experimental follow-up.

    **Objectives:**
    1. Aggregate Molport batch search results (batches were split for website limitations)
    2. Curate multiple compound lists for different plate sizes

    **Approach:**
    - Filter to phenotypically active compounds (reproducible profiles via FDR-corrected p-value)
    - Filter to Molport-available compounds first
    - Cluster profile data using K-means (GPU-accelerated when available)
    - Select N compounds per cluster (closest to centroid)

    **Output lists (4 total):**
    - 160 clusters x 1 compound = 160 compounds
    - 160 clusters x 2 compounds = 320 compounds
    - 320 clusters x 1 compound = 320 compounds
    - 320 clusters x 2 compounds = 640 compounds

    The 320-compound lists fit a 384-well plate with room for controls.

    **Environment:** Requires `pixi run -e cheminformatics marimo edit/run` (rdkit is conda-only).
    GPU optional (cuML for faster clustering).

    *Outputs:* `data/processed/commercial-compounds/`
    """)
    return


@app.cell
def _(mo):
    skip_plots_toggle = mo.ui.switch(value=False, label="Skip plot generation")
    mo.hstack([skip_plots_toggle], justify="start")
    return (skip_plots_toggle,)


# =============================================================================
# Visualization Functions
# =============================================================================


@app.function
def plot_price_distribution(
    curated_df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Histogram of Molport prices for curated compounds.

    Returns figure only when standalone (ax is None).
    """
    _standalone = ax is None
    if _standalone:
        _fig, ax = plt.subplots(figsize=(6, 4))

    _prices = curated_df["Price, USD"].dropna()

    if len(_prices) > 0:
        # Clip extreme values for visualization
        _prices_clipped = _prices.clip(upper=_prices.quantile(0.99))

        ax.hist(_prices_clipped, bins=30, color="#3498db", edgecolor="white", alpha=0.8)

        # Add median line
        _median_price = _prices.median()
        ax.axvline(_median_price, color="#e74c3c", linestyle="--", linewidth=2, label=f"Median: ${_median_price:.0f}")

        # Add statistics annotation
        ax.annotate(
            f"n = {len(_prices):,}\nMedian: ${_median_price:.0f}\nMean: ${_prices.mean():.0f}",
            xy=(0.97, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )

    ax.set_xlabel("Price (USD)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Price Distribution of Curated Compounds", fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if _standalone:
        _fig.tight_layout()
        return _fig
    return None


@app.function
def plot_cluster_coverage(
    curated_df: pd.DataFrame,
    n_clusters: int,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Bar chart showing compounds per cluster.

    Returns figure only when standalone (ax is None).
    """
    _standalone = ax is None
    if _standalone:
        _fig, ax = plt.subplots(figsize=(10, 4))

    # Count compounds per cluster
    _cluster_counts = curated_df["cluster_id"].value_counts().sort_index()

    # Fill in missing clusters with 0
    _all_clusters = pd.Series(0, index=range(n_clusters))
    _all_clusters.update(_cluster_counts)

    # Color bars by count
    _colors = plt.cm.Blues(Normalize(vmin=0, vmax=_all_clusters.max())(_all_clusters.values))

    ax.bar(_all_clusters.index, _all_clusters.values, color=_colors, edgecolor="none", width=0.8)

    # Add horizontal line for mean
    _mean_count = curated_df.shape[0] / n_clusters
    ax.axhline(_mean_count, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean: {_mean_count:.1f}")

    # Add statistics
    _n_represented = (_all_clusters > 0).sum()
    ax.annotate(
        f"Clusters represented: {_n_represented}/{n_clusters}\nTotal compounds: {len(curated_df):,}",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )

    ax.set_xlabel("Cluster ID", fontsize=10)
    ax.set_ylabel("Compounds", fontsize=10)
    ax.set_title("Cluster Coverage of Curated Compounds", fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if _standalone:
        _fig.tight_layout()
        return _fig
    return None


@app.function
def plot_activity_distribution(
    curated_df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Histogram of normalized mAP values for curated compounds.

    Returns figure only when standalone (ax is None).
    """
    _standalone = ax is None
    if _standalone:
        _fig, ax = plt.subplots(figsize=(6, 4))

    _activity_col = "mean_normalized_average_precision"
    if _activity_col not in curated_df.columns:
        logger.warning(f"Column {_activity_col} not found in curated_df")
        if _standalone:
            plt.close(_fig)
        return None

    _activity = curated_df[_activity_col].dropna()

    if len(_activity) > 0:
        ax.hist(_activity, bins=30, color="#2ca02c", edgecolor="white", alpha=0.8)

        _median_activity = _activity.median()
        ax.axvline(
            _median_activity, color="#e74c3c", linestyle="--", linewidth=2, label=f"Median: {_median_activity:.3f}"
        )

        ax.annotate(
            f"n = {len(_activity):,}\nMedian: {_median_activity:.3f}\nMean: {_activity.mean():.3f}",
            xy=(0.97, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )

    ax.set_xlabel("Normalized mAP", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Phenotypic Activity of Curated Compounds", fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if _standalone:
        _fig.tight_layout()
        return _fig
    return None


@app.function
def plot_comparison_distributions(
    all_results: dict[str, dict],
) -> plt.Figure:
    """Plot comparison of activity and price distributions across all configurations."""
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Color palette for configurations
    _colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Left panel: Activity distribution
    _ax = _axes[0]
    _activity_col = "mean_normalized_average_precision"

    for _i, (_config_name, _result) in enumerate(all_results.items()):
        _curated_df = _result["curated_df"]
        if _activity_col in _curated_df.columns:
            _activity = _curated_df[_activity_col].dropna()
            _n_compounds = _result["n_compounds"]
            _label = f"{_config_name} (n={_n_compounds})"
            _ax.hist(_activity, bins=25, alpha=0.5, color=_colors[_i], label=_label, edgecolor="white")

    _ax.set_xlabel("Normalized mAP", fontsize=11)
    _ax.set_ylabel("Count", fontsize=11)
    _ax.set_title("Phenotypic Activity Distribution", fontsize=12, fontweight="bold", loc="left")
    _ax.legend(fontsize=9, loc="upper left")
    _ax.tick_params(labelsize=10)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    # Right panel: Price distribution
    _ax = _axes[1]

    for _i, (_config_name, _result) in enumerate(all_results.items()):
        _curated_df = _result["curated_df"]
        if "Price, USD" in _curated_df.columns:
            _prices = _curated_df["Price, USD"].dropna()
            _prices_clipped = _prices.clip(upper=_prices.quantile(0.99))
            _n_compounds = _result["n_compounds"]
            _label = f"{_config_name} (n={_n_compounds})"
            _ax.hist(_prices_clipped, bins=25, alpha=0.5, color=_colors[_i], label=_label, edgecolor="white")

    _ax.set_xlabel("Price (USD)", fontsize=11)
    _ax.set_ylabel("Count", fontsize=11)
    _ax.set_title("Price Distribution", fontsize=12, fontweight="bold", loc="left")
    _ax.legend(fontsize=9, loc="upper right")
    _ax.tick_params(labelsize=10)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    _fig.tight_layout()
    return _fig


@app.function
def plot_single_umap(
    adata: ad.AnnData,
    curated_jcp: set[str],
    space_name: str = "Phenotypic",
    color_col: str | None = None,
    color_label: str = "",
    cmap: str = "viridis",
    title_suffix: str = "",
    curated_df: pd.DataFrame | None = None,
) -> plt.Figure:
    """Create a single UMAP figure with curated compounds highlighted and outliers clipped.

    Returns the figure.
    """
    # Compute IQR-based bounds for outlier clipping
    _xlim, _ylim = compute_umap_bounds(adata, iqr_k=3.0)

    _coords = adata.obsm["X_umap"]
    _jcp_ids = adata.obs["JCP2022"].values

    # Log outlier statistics
    _outlier_mask = (
        (_coords[:, 0] < _xlim[0])
        | (_coords[:, 0] > _xlim[1])
        | (_coords[:, 1] < _ylim[0])
        | (_coords[:, 1] > _ylim[1])
    )
    _n_outliers = _outlier_mask.sum()
    logger.info(f"{space_name} UMAP: {_n_outliers:,} outliers ({100 * _n_outliers / len(_coords):.1f}%) clipped")

    # Create mask for curated compounds
    _curated_mask = np.array([jcp in curated_jcp for jcp in _jcp_ids])

    _fig, _ax = plt.subplots(figsize=(10, 8))

    # Plot background (all compounds in gray)
    _ax.scatter(
        _coords[~_curated_mask, 0],
        _coords[~_curated_mask, 1],
        s=0.3,
        alpha=0.1,
        c="#cccccc",
        rasterized=True,
    )

    # Plot curated compounds
    _curated_coords = _coords[_curated_mask]
    _curated_jcp_arr = _jcp_ids[_curated_mask]

    if color_col is not None and curated_df is not None:
        # Get color values aligned with curated_coords
        _jcp_to_color = dict(zip(curated_df["JCP2022"], curated_df[color_col]))
        _color_values = np.array([_jcp_to_color.get(jcp, np.nan) for jcp in _curated_jcp_arr])

        # Filter out NaN values
        _valid_mask = ~np.isnan(_color_values)
        if _valid_mask.sum() > 0:
            _scatter = _ax.scatter(
                _curated_coords[_valid_mask, 0],
                _curated_coords[_valid_mask, 1],
                s=8,
                alpha=0.8,
                c=_color_values[_valid_mask],
                cmap=cmap,
                edgecolor="black",
                linewidth=0.3,
                rasterized=True,
            )
            _cbar = plt.colorbar(_scatter, ax=_ax, shrink=0.5)
            _cbar.set_label(color_label, fontsize=10)
            _cbar.ax.tick_params(labelsize=9)
    else:
        # Simple colored scatter without continuous coloring
        _ax.scatter(
            _curated_coords[:, 0],
            _curated_coords[:, 1],
            s=8,
            alpha=0.8,
            c="#e74c3c",
            edgecolor="black",
            linewidth=0.3,
            rasterized=True,
        )

    # Set axis limits (clips outliers)
    _ax.set_xlim(_xlim)
    _ax.set_ylim(_ylim)

    _ax.set_xlabel("UMAP 1", fontsize=11)
    _ax.set_ylabel("UMAP 2", fontsize=11)

    # Build title
    if title_suffix:
        _title = f"{space_name} Space - {title_suffix}"
    else:
        _title = f"{space_name} Space"
    _ax.set_title(_title, fontsize=12, fontweight="bold")

    # Add count annotation
    _ax.annotate(
        f"Curated: n = {len(curated_jcp):,}",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )

    _ax.tick_params(labelsize=10)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    _fig.tight_layout()
    return _fig


# =============================================================================
# Part 1: Molport Search Validation
# =============================================================================


@app.function
def load_batch_smiles(zf: zipfile.ZipFile) -> dict[int, list[str]]:
    """Load all batch files from zip and return dict of batch_id -> SMILES list."""
    _batches = {}
    _batch_files = sorted(n for n in zf.namelist() if "/to_search/batch_" in n and n.endswith(".txt"))
    for _name in _batch_files:
        _batch_id = int(Path(_name).stem.split("_")[1])
        _smiles_list = [line.strip() for line in zf.read(_name).decode().splitlines() if line.strip()]
        _batches[_batch_id] = _smiles_list
        logger.info(f"Batch {_batch_id:02d}: {len(_smiles_list):,} SMILES")
    return _batches


@app.function
def load_molport_results(zf: zipfile.ZipFile) -> pd.DataFrame:
    """Load all Molport result Excel files from zip."""
    _all_results = []
    _xls_files = sorted(n for n in zf.namelist() if "/search_results/Quote_" in n and n.endswith(".xls"))

    if not _xls_files:
        raise FileNotFoundError("No Molport result files (Quote_*.xls) found in zip")

    for _name in _xls_files:
        _df = pd.read_excel(zf.open(_name))
        _df["_source_file"] = Path(_name).name
        _all_results.append(_df)
        logger.info(f"Loaded {Path(_name).name}: {len(_df):,} rows")

    _combined = pd.concat(_all_results, ignore_index=True)
    logger.info(f"Total Molport results: {len(_combined):,} rows")
    return _combined


@app.function
def validate_molport_search(batches: dict[int, list[str]], results: pd.DataFrame) -> dict:
    """Cross-validate batch inputs against Molport results.

    Returns:
        stats: Dictionary with aggregate statistics
    """
    # Get unique SMILES that got hits
    _smiles_with_hits = set(results["Search Criteria"].dropna().unique())

    # Aggregate all batches
    _all_searched = set()
    for _batch_id in sorted(batches.keys()):
        _smiles_list = batches[_batch_id]
        _all_searched.update(_smiles_list)

    _total_searched = len(_all_searched)
    _total_hits = len(_smiles_with_hits)

    _unexpected = _smiles_with_hits - _all_searched
    if _unexpected:
        logger.warning(f"Found {len(_unexpected)} SMILES in results not in batches")

    _stats = {
        "n_batches": len(batches),
        "n_result_files": results["_source_file"].nunique(),
        "total_searched": _total_searched,
        "total_hits": _total_hits,
        "overall_hit_rate_pct": _total_hits / _total_searched * 100 if _total_searched > 0 else 0,
    }

    return _stats


# =============================================================================
# Part 2: Profile Clustering
# =============================================================================


@app.function
def cluster_profiles(
    adata: ad.AnnData,
    n_clusters: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """K-means clustering of profile data (not UMAP coordinates).

    Uses rapids cuML for GPU acceleration when available,
    falls back to sklearn on CPU.

    Returns:
        labels: Cluster labels for each sample
        centroids: (n_clusters, n_features) array of cluster centroids
    """
    _n_samples = adata.n_obs
    logger.info(f"Clustering {_n_samples:,} profiles into {n_clusters} clusters")

    _use_gpu = has_gpu()

    if _use_gpu:
        try:
            import cupy as cp
            from cuml.cluster import KMeans as cuKMeans

            logger.info("Using GPU-accelerated clustering (cuML)")
            _X = cp.asarray(adata.X)
            _kmeans = cuKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
            _kmeans.fit(_X)
            _labels = _kmeans.labels_.get()
            _centroids = _kmeans.cluster_centers_.get()

            _sizes = np.bincount(_labels)
            logger.info(f"Cluster sizes: min={_sizes.min()}, median={np.median(_sizes):.0f}, max={_sizes.max()}")
            return _labels, _centroids
        except ImportError:
            logger.warning("cuML not available, falling back to CPU")

    logger.info("Using CPU clustering (sklearn)")
    _kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    _labels = _kmeans.fit_predict(adata.X)
    _centroids = _kmeans.cluster_centers_

    _sizes = np.bincount(_labels)
    logger.info(f"Cluster sizes: min={_sizes.min()}, median={np.median(_sizes):.0f}, max={_sizes.max()}")

    return _labels, _centroids


@app.function
def select_from_clusters(
    adata: ad.AnnData,
    labels: np.ndarray,
    centroids: np.ndarray,
    n_per_cluster: int = 2,
) -> pd.DataFrame:
    """Select compounds closest to each cluster centroid.

    Returns:
        DataFrame with columns: JCP2022, cluster_id, centroid_distance
    """
    _jcp_ids = adata.obs["JCP2022"].values
    _X = adata.X
    _n_clusters = len(centroids)

    _candidates = []
    for _cluster_id in range(_n_clusters):
        _mask = labels == _cluster_id
        _cluster_indices = np.where(_mask)[0]

        if len(_cluster_indices) == 0:
            continue

        _cluster_X = _X[_mask]

        # Calculate distance to centroid (cosine distance)
        _centroid = centroids[_cluster_id]
        _cluster_norms = np.linalg.norm(_cluster_X, axis=1, keepdims=True)
        _centroid_norm = np.linalg.norm(_centroid)
        _cluster_X_norm = _cluster_X / np.maximum(_cluster_norms, 1e-10)
        _centroid_norm_vec = _centroid / max(_centroid_norm, 1e-10)

        # Cosine similarity -> distance
        _cos_sim = _cluster_X_norm @ _centroid_norm_vec
        _distances = 1 - _cos_sim

        # Get closest compounds
        _n_select = min(n_per_cluster, len(_cluster_indices))
        _closest_local = np.argsort(_distances)[:_n_select]

        for _local_idx in _closest_local:
            _global_idx = _cluster_indices[_local_idx]
            _candidates.append(
                {
                    "JCP2022": _jcp_ids[_global_idx],
                    "cluster_id": _cluster_id,
                    "centroid_distance": float(_distances[_local_idx]),
                }
            )

    df = pd.DataFrame(_candidates)
    logger.info(f"Selected {len(df):,} compounds from {_n_clusters} clusters ({n_per_cluster} per cluster)")
    return df


# =============================================================================
# Part 3: MaxMin Chemical Diversity Selection
# =============================================================================


@app.function
def compute_morgan_fingerprints(smiles_list: list[str], radius: int = 2, n_bits: int = 2048):
    """Compute Morgan fingerprints for a list of SMILES.

    Returns:
        fps: List of fingerprint objects (or None for invalid SMILES)
        valid_mask: Boolean array indicating valid molecules
    """
    _fps = []
    _valid_mask = []

    _fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    for _smiles in smiles_list:
        _mol = Chem.MolFromSmiles(_smiles)
        if _mol is not None:
            _fp = _fpgen.GetFingerprint(_mol)
            _fps.append(_fp)
            _valid_mask.append(True)
        else:
            _fps.append(None)
            _valid_mask.append(False)

    return _fps, np.array(_valid_mask)


@app.function
def maxmin_diversity_selection(
    fps: list,
    valid_mask: np.ndarray,
    target_count: int,
    seed_idx: int | None = None,
) -> list[int]:
    """MaxMin algorithm for chemical diversity selection.

    Greedily selects compounds that maximize the minimum Tanimoto distance
    to the already-selected set.

    Returns:
        List of selected indices
    """
    _valid_indices = np.where(valid_mask)[0]
    _n_valid = len(_valid_indices)

    if _n_valid == 0:
        return []

    if target_count >= _n_valid:
        logger.warning(f"Target count {target_count} >= valid compounds {_n_valid}, returning all")
        return _valid_indices.tolist()

    logger.info(f"Running MaxMin diversity selection: {_n_valid:,} candidates -> {target_count} compounds")

    # Start with seed or first valid compound
    if seed_idx is not None and valid_mask[seed_idx]:
        _first_idx = seed_idx
    else:
        _first_idx = _valid_indices[0]

    _selected = [_first_idx]
    _selected_set = {_first_idx}

    # Track minimum distance to selected set for each candidate
    _min_distances = np.full(len(fps), np.inf)
    for _idx in _valid_indices:
        if _idx != _first_idx:
            _min_distances[_idx] = 1 - DataStructs.TanimotoSimilarity(fps[_idx], fps[_first_idx])

    # Greedy selection
    for _i in range(1, target_count):
        if _i % 100 == 0:
            logger.info(f"  Selected {_i}/{target_count}")

        _best_idx = None
        _best_dist = -1

        for _idx in _valid_indices:
            if _idx in _selected_set:
                continue
            if _min_distances[_idx] > _best_dist:
                _best_dist = _min_distances[_idx]
                _best_idx = _idx

        if _best_idx is None:
            break

        _selected.append(_best_idx)
        _selected_set.add(_best_idx)

        # Update minimum distances
        for _idx in _valid_indices:
            if _idx not in _selected_set:
                _dist = 1 - DataStructs.TanimotoSimilarity(fps[_idx], fps[_best_idx])
                if _dist < _min_distances[_idx]:
                    _min_distances[_idx] = _dist

    logger.info(f"Selected {len(_selected)} compounds with MaxMin diversity")
    return _selected


@app.function
def plot_umap_clusters(
    umap_coords: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    selected_mask: np.ndarray | None = None,
) -> plt.Figure:
    """Plot UMAP with cluster assignments and optionally highlight selected compounds."""
    _fig, _ax = plt.subplots(figsize=(10, 10))

    # Plot all points colored by cluster
    _ax.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=labels,
        cmap="tab20",
        s=1,
        alpha=0.3,
        rasterized=True,
    )

    # Plot centroids
    _ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        s=50,
        marker="x",
        linewidths=2,
        label="Centroids",
    )

    # Highlight selected compounds if provided
    if selected_mask is not None:
        _ax.scatter(
            umap_coords[selected_mask, 0],
            umap_coords[selected_mask, 1],
            facecolors="none",
            edgecolors="red",
            s=20,
            linewidths=1,
            label=f"Selected ({selected_mask.sum():,})",
        )

    _ax.set_xlabel("UMAP 1")
    _ax.set_ylabel("UMAP 2")
    _ax.set_title("Cell Painting UMAP with K-means Clusters")
    _ax.legend(loc="upper right")

    _fig.tight_layout()
    return _fig


# =============================================================================
# Pipeline cells
# =============================================================================


@app.cell
def _(mo):
    mo.stop(
        not MOLPORT_ZIP.exists(),
        mo.md(f"**Molport batch search zip not found:** `{MOLPORT_ZIP}`"),
    )

    mo.md("### Part 1: Molport Search Aggregation")
    return


@app.cell
def _(mo):
    with zipfile.ZipFile(MOLPORT_ZIP, "r") as _zf:
        batches = load_batch_smiles(_zf)
        molport_results = load_molport_results(_zf)

    validation_stats = validate_molport_search(batches, molport_results)

    mo.md(
        f"**Molport search summary:**\n\n"
        f"- Batches submitted: {validation_stats['n_batches']}\n"
        f"- Result files: {validation_stats['n_result_files']}\n"
        f"- Total compounds searched: {validation_stats['total_searched']:,}\n"
        f"- Compounds with hits: {validation_stats['total_hits']:,}\n"
        f"- Overall hit rate: {validation_stats['overall_hit_rate_pct']:.1f}%"
    )
    return batches, molport_results, validation_stats


@app.cell
def _(mo, molport_results):
    # Get all SMILES with hits and their best matches (Perfect > Isomer)
    molport_hits = (
        molport_results.groupby("Search Criteria")
        .agg(
            {
                "Molport ID": "first",
                "Match Type": lambda x: "Perfect" if "Perfect" in x.values else "Isomer",
                "Price, USD": "min",
                "Supplier Name": "first",
            }
        )
        .reset_index()
        .rename(columns={"Search Criteria": "SMILES"})
    )

    # Apply price cutoff early
    _n_before_price = len(molport_hits)
    molport_hits = molport_hits[molport_hits["Price, USD"] <= MAX_PRICE_USD]

    mo.md(f"**Price cutoff** (<= ${MAX_PRICE_USD}): {_n_before_price:,} -> {len(molport_hits):,} compounds")
    return (molport_hits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("### Part 2: Load Profiles and Filter to Active + Available")
    return


@app.cell
def _(mo):
    adata_profiles = load_profiles("compound_no_source7", level="perturbation")
    adata_profiles = join_metadata(adata_profiles, modality="compound", level="perturbation")

    # Join phenotypic activity data
    adata_profiles = join_activity(adata_profiles, "compound_no_source7", preprocessing="activity_no_target2")
    n_total = adata_profiles.n_obs
    n_significant = (adata_profiles.obs["below_corrected_p"] == "True").sum()

    mo.md(
        f"Loaded **{adata_profiles.n_obs:,}** compounds with **{adata_profiles.n_vars:,}** features.\n\n"
        f"Phenotypically active: **{n_significant:,}** of {n_total:,}"
    )
    return adata_profiles, n_significant, n_total


@app.cell
def _(adata_profiles, mo, molport_hits):
    # Filter to phenotypically active
    _active_mask = adata_profiles.obs["below_corrected_p"] == "True"
    _adata_active = adata_profiles[_active_mask].copy()

    # Filter to Molport available
    _molport_smiles_set = set(molport_hits["SMILES"])
    _adata_active.obs["molport_available"] = _adata_active.obs["SMILES"].isin(_molport_smiles_set)
    _n_available = _adata_active.obs["molport_available"].sum()

    # Filter to Molport available FIRST (before clustering)
    _available_mask = _adata_active.obs["molport_available"]
    adata_available = _adata_active[_available_mask].copy()

    mo.md(
        f"Active compounds: **{_adata_active.n_obs:,}**\n\n"
        f"Molport available among active: **{_n_available:,}**\n\n"
        f"Compounds for clustering: **{adata_available.n_obs:,}** (active + Molport available)"
    )
    return (adata_available,)


@app.cell
def _(mo):
    # Load UMAPs for visualization
    adata_pheno = load_umap("compound_no_source7", level="perturbation", metric="cosine", filter_name="all")
    adata_pheno = join_metadata(adata_pheno, modality="compound", level="perturbation")

    # Load structure UMAP
    _structure_umap_path = CHEMICAL_SPACE_DIR / "structure_umap.h5ad"
    if _structure_umap_path.exists():
        logger.info(f"Loading structure UMAP from {_structure_umap_path}")
        adata_structure = ad.read_h5ad(_structure_umap_path)
        if "Metadata_JCP2022" in adata_structure.obs.columns:
            adata_structure.obs["JCP2022"] = adata_structure.obs["Metadata_JCP2022"]
        logger.info(f"Loaded structure UMAP with {adata_structure.n_obs:,} compounds")
    else:
        logger.warning(f"Structure UMAP not found at {_structure_umap_path}")
        adata_structure = adata_pheno

    mo.md(
        f"Loaded phenotypic UMAP: **{adata_pheno.n_obs:,}** compounds\n\n"
        f"Loaded structure UMAP: **{adata_structure.n_obs:,}** compounds"
    )
    return adata_pheno, adata_structure


@app.cell(hide_code=True)
def _(mo):
    mo.md("### Part 3: Generate Multiple Compound Lists")
    return


@app.cell
def _(adata_available, mo):
    mo.stop(
        adata_available.n_obs == 0,
        mo.md("**No active Molport-available compounds. Cannot proceed.**"),
    )

    # Cluster once per unique n_clusters value
    _unique_n_clusters = sorted(set(cfg[0] for cfg in CLUSTER_CONFIGS))
    cluster_results = {}

    for _n_clusters in _unique_n_clusters:
        logger.info(f"Clustering into {_n_clusters} clusters...")
        _labels, _centroids = cluster_profiles(adata_available, n_clusters=_n_clusters)
        cluster_results[_n_clusters] = (_labels, _centroids)

    mo.md(f"Computed K-means for cluster sizes: {_unique_n_clusters}")
    return (cluster_results,)


@app.cell
def _(adata_available, cluster_results, mo, molport_hits):
    # Generate each configuration
    all_results = {}

    for _n_clusters, _n_per_cluster, _config_name in CLUSTER_CONFIGS:
        logger.info(f"Generating: {_config_name} ({_n_clusters} clusters x {_n_per_cluster} per cluster)")

        _labels, _centroids = cluster_results[_n_clusters]

        # Select compounds
        _curated_df = select_from_clusters(adata_available, _labels, _centroids, n_per_cluster=_n_per_cluster)

        # Add metadata
        _curated_df = _curated_df.merge(
            adata_available.obs[["JCP2022", "SMILES", "InChIKey"]].drop_duplicates(),
            on="JCP2022",
            how="left",
        )

        # Add Molport info
        _curated_df = _curated_df.merge(
            molport_hits[["SMILES", "Molport ID", "Match Type", "Price, USD", "Supplier Name"]],
            on="SMILES",
            how="left",
        )

        # Add activity data
        _activity_cols = ["mean_normalized_average_precision", "corrected_p_value"]
        _activity_data = adata_available.obs[
            ["JCP2022"] + [c for c in _activity_cols if c in adata_available.obs.columns]
        ].drop_duplicates()
        _curated_df = _curated_df.merge(_activity_data, on="JCP2022", how="left")

        # Sort by cluster_id then centroid_distance
        _curated_df = _curated_df.sort_values(["cluster_id", "centroid_distance"]).reset_index(drop=True)

        all_results[_config_name] = {
            "n_clusters": _n_clusters,
            "n_per_cluster": _n_per_cluster,
            "n_compounds": len(_curated_df),
            "n_clusters_represented": _curated_df["cluster_id"].nunique(),
            "curated_df": _curated_df,
        }

    _lines = [
        "| Config | Clusters | Per cluster | Compounds | Clusters represented |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _name, _res in all_results.items():
        _lines.append(
            f"| {_name} | {_res['n_clusters']} | {_res['n_per_cluster']} | {_res['n_compounds']:,} | {_res['n_clusters_represented']} |"
        )
    mo.md("**Configuration results:**\n\n" + "\n".join(_lines))
    return (all_results,)


@app.cell
def _(all_results):
    _fig = plot_comparison_distributions(all_results)
    _fig
    return


@app.cell
def _(
    adata_pheno,
    adata_structure,
    all_results,
    mo,
    skip_plots_toggle,
):
    mo.stop(
        skip_plots_toggle.value,
        mo.md("**UMAP visualization skipped** (toggle above to enable)."),
    )

    # Show phenotypic UMAP for the 320c_1pc config as representative example
    _config_name = "320c_1pc"
    if _config_name in all_results:
        _result = all_results[_config_name]
        _curated_jcp = set(_result["curated_df"]["JCP2022"])

        _fig_pheno = plot_single_umap(
            adata_pheno,
            _curated_jcp,
            space_name=f"Phenotypic ({_config_name})",
        )

        _fig_struct = plot_single_umap(
            adata_structure,
            _curated_jcp,
            space_name=f"Structure ({_config_name})",
        )

        mo.vstack([_fig_pheno, _fig_struct])
    else:
        mo.md("No 320c_1pc config found.")
    return


# =============================================================================
# Save outputs
# =============================================================================


@app.cell
def _(
    adata_pheno,
    adata_structure,
    all_results,
    mo,
    molport_hits,
    n_significant,
    n_total,
    skip_plots_toggle,
    validation_stats,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save Molport hits
    molport_hits.to_csv(OUTPUT_DIR / "molport_hits.csv", index=False)

    # Save each configuration
    for _config_name, _result in all_results.items():
        _curated_df = _result["curated_df"]
        _curated_df.to_csv(OUTPUT_DIR / f"curated_compounds_{_config_name}.csv", index=False)

        # Generate UMAP plots for each config
        if not skip_plots_toggle.value:
            _curated_jcp = set(_curated_df["JCP2022"])

            # Phenotypic UMAP - overview
            _fig = plot_single_umap(
                adata_pheno,
                _curated_jcp,
                space_name=f"Phenotypic ({_config_name})",
            )
            _fig.savefig(OUTPUT_DIR / f"umap_phenotypic_{_config_name}.png", dpi=DEFAULT_DPI, bbox_inches="tight")
            plt.close(_fig)

            # Phenotypic UMAP - by price
            _fig = plot_single_umap(
                adata_pheno,
                _curated_jcp,
                space_name=f"Phenotypic ({_config_name})",
                color_col="Price, USD",
                color_label="Price (USD)",
                cmap="YlOrRd",
                title_suffix="by Price",
                curated_df=_curated_df,
            )
            _fig.savefig(OUTPUT_DIR / f"umap_phenotypic_{_config_name}_price.png", dpi=DEFAULT_DPI, bbox_inches="tight")
            plt.close(_fig)

            # Phenotypic UMAP - by activity
            _fig = plot_single_umap(
                adata_pheno,
                _curated_jcp,
                space_name=f"Phenotypic ({_config_name})",
                color_col="mean_normalized_average_precision",
                color_label="Normalized mAP",
                cmap="viridis",
                title_suffix="by Activity",
                curated_df=_curated_df,
            )
            _fig.savefig(
                OUTPUT_DIR / f"umap_phenotypic_{_config_name}_activity.png", dpi=DEFAULT_DPI, bbox_inches="tight"
            )
            plt.close(_fig)

            # Structure UMAP - overview
            _fig = plot_single_umap(
                adata_structure,
                _curated_jcp,
                space_name=f"Structure ({_config_name})",
            )
            _fig.savefig(OUTPUT_DIR / f"umap_structure_{_config_name}.png", dpi=DEFAULT_DPI, bbox_inches="tight")
            plt.close(_fig)

            # Structure UMAP - by price
            _fig = plot_single_umap(
                adata_structure,
                _curated_jcp,
                space_name=f"Structure ({_config_name})",
                color_col="Price, USD",
                color_label="Price (USD)",
                cmap="YlOrRd",
                title_suffix="by Price",
                curated_df=_curated_df,
            )
            _fig.savefig(OUTPUT_DIR / f"umap_structure_{_config_name}_price.png", dpi=DEFAULT_DPI, bbox_inches="tight")
            plt.close(_fig)

            # Structure UMAP - by activity
            _fig = plot_single_umap(
                adata_structure,
                _curated_jcp,
                space_name=f"Structure ({_config_name})",
                color_col="mean_normalized_average_precision",
                color_label="Normalized mAP",
                cmap="viridis",
                title_suffix="by Activity",
                curated_df=_curated_df,
            )
            _fig.savefig(
                OUTPUT_DIR / f"umap_structure_{_config_name}_activity.png", dpi=DEFAULT_DPI, bbox_inches="tight"
            )
            plt.close(_fig)

    # Comparison distribution plot
    if not skip_plots_toggle.value:
        _fig_comp = plot_comparison_distributions(all_results)
        _fig_comp.savefig(OUTPUT_DIR / "comparison_distributions.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(_fig_comp)

    # Build summary JSON
    _configs_summary = {}
    for _config_name, _result in all_results.items():
        _curated_df = _result["curated_df"]

        _activity_col = "mean_normalized_average_precision"
        if _activity_col in _curated_df.columns:
            _activity_stats = {
                "median": round(float(_curated_df[_activity_col].median()), 4),
                "mean": round(float(_curated_df[_activity_col].mean()), 4),
                "min": round(float(_curated_df[_activity_col].min()), 4),
                "max": round(float(_curated_df[_activity_col].max()), 4),
            }
        else:
            _activity_stats = {}

        if "Price, USD" in _curated_df.columns:
            _prices = _curated_df["Price, USD"].dropna()
            _price_stats = {
                "median": round(float(_prices.median()), 2),
                "mean": round(float(_prices.mean()), 2),
                "min": round(float(_prices.min()), 2),
                "max": round(float(_prices.max()), 2),
            }
        else:
            _price_stats = {}

        _configs_summary[_config_name] = {
            "n_clusters": _result["n_clusters"],
            "n_per_cluster": _result["n_per_cluster"],
            "n_compounds": _result["n_compounds"],
            "n_clusters_represented": _result["n_clusters_represented"],
            "activity_stats": _activity_stats,
            "price_stats": _price_stats,
        }

    _summary = {
        "task": "Commercial compounds curation - multiple configurations",
        "molport_search": {
            "total_searched": int(validation_stats["total_searched"]),
            "total_hits": int(validation_stats["total_hits"]),
            "hit_rate_pct": round(float(validation_stats["overall_hit_rate_pct"]), 2),
            "price_cutoff_usd": MAX_PRICE_USD,
            "n_after_price_cutoff": len(molport_hits),
        },
        "phenotypic_activity_prefilter": {
            "n_total_compounds": int(n_total),
            "n_active_compounds": int(n_significant),
            "active_rate_pct": round(float(n_significant / n_total * 100), 2),
        },
        "configurations": _configs_summary,
    }

    with open(OUTPUT_DIR / "summary.json", "w") as _f:
        json.dump(_summary, _f, indent=2)

    _out_files = [f"curated_compounds_{name}.csv" for name in all_results]
    mo.md(
        f"### Outputs Saved\n\n"
        f"All results saved to `{OUTPUT_DIR}`:\n\n"
        + "\n".join(f"- `{f}`" for f in _out_files)
        + "\n- `molport_hits.csv`\n"
        + "- `comparison_distributions.png`\n"
        + "- `summary.json`\n"
        + "- UMAP plots for each configuration"
    )
    return


@app.function
def run_commercial_analysis(
    output_dir=None,
) -> str:
    """Run the full commercial compound curation pipeline.

    Parameters
    ----------
    output_dir : str or Path, optional
        Output directory. Defaults to OUTPUT_DIR.

    Returns
    -------
    str
        Path to the output directory.
    """
    import zipfile as _zipfile

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running commercial compound curation")

    # Step 1: Load and validate Molport results
    with _zipfile.ZipFile(MOLPORT_ZIP, "r") as zf:
        batches = load_batch_smiles(zf)
        molport_results = load_molport_results(zf)

    validation_stats = validate_molport_search(batches, molport_results)

    # Aggregate Molport hits
    molport_hits = (
        molport_results.groupby("Search Criteria")
        .agg(
            {
                "Molport ID": "first",
                "Match Type": lambda x: "Perfect" if "Perfect" in x.values else "Isomer",
                "Price, USD": "min",
                "Supplier Name": "first",
            }
        )
        .reset_index()
        .rename(columns={"Search Criteria": "SMILES"})
    )
    molport_hits = molport_hits[molport_hits["Price, USD"] <= MAX_PRICE_USD]

    # Step 2: Load profiles and filter to active + available
    adata_profiles = load_profiles("compound_no_source7", level="perturbation")
    adata_profiles = join_metadata(adata_profiles, modality="compound", level="perturbation")
    adata_profiles = join_activity(adata_profiles, "compound_no_source7", preprocessing="activity_no_target2")

    n_total = adata_profiles.n_obs
    n_significant = (adata_profiles.obs["below_corrected_p"] == "True").sum()

    # Filter to active
    _active_mask = adata_profiles.obs["below_corrected_p"] == "True"
    _adata_active = adata_profiles[_active_mask].copy()

    # Filter to Molport available
    _molport_smiles_set = set(molport_hits["SMILES"])
    _adata_active.obs["molport_available"] = _adata_active.obs["SMILES"].isin(_molport_smiles_set)
    _available_mask = _adata_active.obs["molport_available"]
    adata_available = _adata_active[_available_mask].copy()

    if adata_available.n_obs == 0:
        logger.error("No active Molport-available compounds. Cannot proceed.")
        summary = {
            "task": "Commercial compounds curation",
            "error": "No active Molport-available compounds",
        }
        Path(output_dir, "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return str(output_dir)

    # Step 3: Cluster per unique n_clusters value
    _unique_n_clusters = sorted(set(cfg[0] for cfg in CLUSTER_CONFIGS))
    cluster_results = {}
    for _n_clusters in _unique_n_clusters:
        logger.info(f"Clustering into {_n_clusters} clusters...")
        _labels, _centroids = cluster_profiles(adata_available, n_clusters=_n_clusters)
        cluster_results[_n_clusters] = (_labels, _centroids)

    # Step 4: Generate each configuration
    all_results = {}
    for _n_clusters, _n_per_cluster, _config_name in CLUSTER_CONFIGS:
        logger.info(f"Generating: {_config_name}")
        _labels, _centroids = cluster_results[_n_clusters]
        _curated_df = select_from_clusters(adata_available, _labels, _centroids, n_per_cluster=_n_per_cluster)

        # Add metadata
        _curated_df = _curated_df.merge(
            adata_available.obs[["JCP2022", "SMILES", "InChIKey"]].drop_duplicates(),
            on="JCP2022",
            how="left",
        )
        _curated_df = _curated_df.merge(
            molport_hits[["SMILES", "Molport ID", "Match Type", "Price, USD", "Supplier Name"]],
            on="SMILES",
            how="left",
        )
        _activity_cols = ["mean_normalized_average_precision", "corrected_p_value"]
        _activity_data = adata_available.obs[
            ["JCP2022"] + [c for c in _activity_cols if c in adata_available.obs.columns]
        ].drop_duplicates()
        _curated_df = _curated_df.merge(_activity_data, on="JCP2022", how="left")
        _curated_df = _curated_df.sort_values(["cluster_id", "centroid_distance"]).reset_index(drop=True)

        all_results[_config_name] = {
            "n_clusters": _n_clusters,
            "n_per_cluster": _n_per_cluster,
            "n_compounds": len(_curated_df),
            "n_clusters_represented": _curated_df["cluster_id"].nunique(),
            "curated_df": _curated_df,
        }

    # Step 5: Save CSVs
    molport_hits.to_csv(output_dir / "molport_hits.csv", index=False)
    for _config_name, _result in all_results.items():
        _result["curated_df"].to_csv(output_dir / f"curated_compounds_{_config_name}.csv", index=False)

    # Step 6: Save comparison distributions plot
    fig_comp = plot_comparison_distributions(all_results)
    fig_comp.savefig(output_dir / "comparison_distributions.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_comp)

    # Step 7: Build and save summary JSON
    _configs_summary = {}
    for _config_name, _result in all_results.items():
        _curated_df = _result["curated_df"]
        _activity_col = "mean_normalized_average_precision"
        if _activity_col in _curated_df.columns:
            _activity_stats = {
                "median": round(float(_curated_df[_activity_col].median()), 4),
                "mean": round(float(_curated_df[_activity_col].mean()), 4),
                "min": round(float(_curated_df[_activity_col].min()), 4),
                "max": round(float(_curated_df[_activity_col].max()), 4),
            }
        else:
            _activity_stats = {}

        if "Price, USD" in _curated_df.columns:
            _prices = _curated_df["Price, USD"].dropna()
            _price_stats = {
                "median": round(float(_prices.median()), 2),
                "mean": round(float(_prices.mean()), 2),
                "min": round(float(_prices.min()), 2),
                "max": round(float(_prices.max()), 2),
            }
        else:
            _price_stats = {}

        _configs_summary[_config_name] = {
            "n_clusters": _result["n_clusters"],
            "n_per_cluster": _result["n_per_cluster"],
            "n_compounds": _result["n_compounds"],
            "n_clusters_represented": _result["n_clusters_represented"],
            "activity_stats": _activity_stats,
            "price_stats": _price_stats,
        }

    summary = {
        "task": "Commercial compounds curation - multiple configurations",
        "molport_search": {
            "total_searched": int(validation_stats["total_searched"]),
            "total_hits": int(validation_stats["total_hits"]),
            "hit_rate_pct": round(float(validation_stats["overall_hit_rate_pct"]), 2),
            "price_cutoff_usd": MAX_PRICE_USD,
            "n_after_price_cutoff": len(molport_hits),
        },
        "phenotypic_activity_prefilter": {
            "n_total_compounds": int(n_total),
            "n_active_compounds": int(n_significant),
            "active_rate_pct": round(float(n_significant / n_total * 100), 2),
        },
        "configurations": _configs_summary,
    }

    Path(output_dir, "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info(f"Commercial compound curation complete: {output_dir}")
    return str(output_dir)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
