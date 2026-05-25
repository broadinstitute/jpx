# copairs & copairs-runner API Reference

## copairs Library

### Installation

```bash
pip install copairs
```

### Module Structure

```
copairs/
├── matching.py          # Pair identification
├── compute.py           # Core computations (AP, distances, p-values)
├── map/                 # High-level API
│   ├── average_precision.py
│   ├── map.py           # mean_average_precision
│   ├── multilabel.py    # For items with multiple labels
│   ├── filter.py
│   ├── normalization.py # Normalized AP computation
│   └── hierarchical_fdr.py
├── plot.py              # Visualization
└── replicating.py       # Legacy API
```

---

## copairs.matching

### find_pairs

```python
from copairs.matching import find_pairs

pairs = find_pairs(
    dframe: pd.DataFrame | duckdb.DuckDBPyRelation,
    sameby: str | List[str],   # Columns that must match
    diffby: str | List[str],   # Columns that must differ
    rev: bool = False          # If True, swap sameby/diffby
) -> np.ndarray  # Shape (N, 2) array of index pairs
```

Find indices of pairs where rows share values in `sameby` columns but differ in `diffby` columns.

### find_pairs_multilabel

```python
from copairs.matching import find_pairs_multilabel

pairs = find_pairs_multilabel(
    dframe: pd.DataFrame | duckdb.DuckDBPyRelation,
    sameby: str | List[str],
    diffby: str | List[str],
    multilabel_col: str        # Column containing lists of labels
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Handle columns with multiple labels (e.g., `["EGFR", "HER2"]`). The `multilabel_col` must be in either `sameby` or `diffby`.

### assign_reference_index

```python
from copairs.matching import assign_reference_index

df = assign_reference_index(
    df: pd.DataFrame,
    condition: str | pd.Index,     # Boolean column or query
    reference_col: str = 'Metadata_Reference_Index',
    default_value: int = -1,       # Value for reference rows
    inplace: bool = False
) -> pd.DataFrame
```

Mark rows as reference (e.g., negative controls) so they won't be ranked as positives. Rows with `reference_col = -1` are excluded from being "query" profiles.

---

## copairs.map

### average_precision

```python
from copairs.map import average_precision

ap_results = average_precision(
    meta: pd.DataFrame,            # Metadata columns
    feats: np.ndarray,             # Feature matrix (n_samples, n_features)
    pos_sameby: List[str],         # Positive pairs share these values
    pos_diffby: List[str],         # Positive pairs differ on these (optional)
    neg_sameby: List[str],         # Negative pairs share these (optional)
    neg_diffby: List[str],         # Negative pairs differ on these
    batch_size: int = 20000,       # Batch size for distance computation
    distance: str = 'cosine',      # Distance metric
    progress_bar: bool = True
) -> pd.DataFrame
```

**Returns DataFrame with columns:**
- `average_precision`: AP score for each profile
- `normalized_average_precision`: Scale-independent AP (see normalization below)
- `n_pos_pairs`: Number of positive pairs
- `n_total_pairs`: Total pairs (positive + negative)
- Plus all metadata columns

**Distance options:** `cosine`, `abs_cosine`, `correlation`, `euclidean`, `manhattan`, `chebyshev`

### mean_average_precision

```python
from copairs.map import mean_average_precision

map_results = mean_average_precision(
    ap_scores: pd.DataFrame,       # Output from average_precision
    sameby: List[str],             # Group AP scores by these columns
    null_size: int,                # Permutations for p-value (e.g., 100000)
    threshold: float,              # Significance threshold (e.g., 0.05)
    seed: int,                     # Random seed for reproducibility
    progress_bar: bool = True,
    max_workers: int = None,       # Parallel workers
    cache_dir: str | Path = None   # Cache null distributions
) -> pd.DataFrame
```

**Returns DataFrame with columns:**
- `mean_average_precision`: Mean AP for each group
- `mean_normalized_average_precision`: Mean normalized AP (scale-independent)
- `p_value`: Raw p-value from permutation test
- `corrected_p_value`: BH FDR-corrected p-value
- `below_p`: True if p_value < threshold
- `below_corrected_p`: True if corrected_p_value < threshold
- `-log10(p-value)`: For visualization
- `indices`: List of row indices in the group

### Multilabel API

```python
from copairs.map.multilabel import average_precision as multilabel_ap

ap_results = multilabel_ap(
    meta: pd.DataFrame,
    feats: np.ndarray,
    pos_sameby: List[str],
    pos_diffby: List[str],
    neg_sameby: List[str],
    neg_diffby: List[str],
    multilabel_col: str,           # Column with list values
    batch_size: int = 1000,
    distance: str = 'cosine'
) -> pd.DataFrame
```

For items with multiple labels (e.g., `["EGFR", "HER2", "CDK4"]`). Returns AP scores for each item-label pair.

---

## copairs.map.normalization

Normalized AP allows comparison across groups with different sizes.

### normalize_ap

```python
from copairs.map.normalization import normalize_ap

normalized = normalize_ap(
    ap: float | np.ndarray,    # AP score(s) to normalize
    M: int | np.ndarray,       # Number of positive items
    N: int | np.ndarray,       # Number of negative items
    eps: float = 1e-10         # Avoid division by zero
) -> float | np.ndarray
```

**Formula:** `(AP - μ₀) / (1 - μ₀)` where μ₀ is expected AP under random ranking.

**Interpretation:**
- Normalized AP = 0: Performance equals random chance
- Normalized AP = 1: Perfect performance
- Normalized AP < 0: Worse than random

### expected_ap

```python
from copairs.map.normalization import expected_ap

mu_0 = expected_ap(
    M: int,   # Number of positive items
    N: int    # Number of negative items
) -> float
```

Computes the expected Average Precision under random ranking using the exact finite-sample formula.

---

## copairs.compute

Lower-level functions for custom workflows:

```python
from copairs.compute import (
    get_similarity_fn,       # Get distance function by name
    pairwise_cosine,         # (n, d) x (m, d) -> (n, m) distances
    pairwise_abs_cosine,
    pairwise_corr,
    pairwise_euclidean,
    pairwise_manhattan,
    average_precision,       # Compute AP from relevance array
    random_ap,               # Generate null AP distribution
    p_values,                # Compute p-values from null
    get_null_dists,          # Generate null distributions
)
```

---

## copairs-runner

### Installation

copairs-runner is not on PyPI. Install from the Broad monorepo:

```bash
# Using uv
uv add "git+https://github.com/broadinstitute/monorepo.git@copairs-runner#subdirectory=libs/copairs_runner"

# Using pip
pip install "git+https://github.com/broadinstitute/monorepo.git@copairs-runner#subdirectory=libs/copairs_runner"
```

**Note:** The package is currently on the `copairs-runner` branch and will eventually be merged to main.

### CLI Usage

```bash
copairs-runner --config-dir <path> --config-name <name> [overrides...]

# Examples:
copairs-runner --config-dir configs --config-name my_analysis \
  distance=abs_cosine

# Override nested params:
copairs-runner ... mean_average_precision.params.null_size=50000
```

### CopairsRunner Class

```python
from copairs_runner import CopairsRunner
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")
runner = CopairsRunner(config)

# Full pipeline
results = runner.run()  # Returns dict with ap_scores, map_results, map_plot

# Or step by step:
df = runner.load_data()
df = runner.preprocess_data(df)
metadata = df.filter(regex="^Metadata")
features = df[[c for c in df.columns if not c.startswith("Metadata")]].values
ap_results = runner.run_average_precision(metadata, features)
map_results = runner.run_mean_average_precision(ap_results)
runner.save_results({"ap_scores": ap_results, "map_results": map_results}, "output_name")
```

### Preprocessing Steps

To get the current list of available preprocessing steps:

```bash
python -c "
from copairs_runner import CopairsRunner
methods = [m.replace('_preprocess_', '') for m in dir(CopairsRunner) if m.startswith('_preprocess_')]
print('\n'.join(sorted(methods)))
"
```

Common preprocessing step types:

| Type | Required Params | Description |
|------|-----------------|-------------|
| `filter` | `query` | Pandas query syntax row filtering |
| `dropna` | `columns` | Drop rows with NaN in specified columns |
| `remove_nan_features` | - | Remove feature columns containing any NaN |
| `merge_metadata` | `source`, `table`, `on_columns`, `how` | Join metadata from CSV or DuckDB |
| `add_column` | `query`, `column` | Create boolean column from query |
| `apply_assign_reference` | `condition`, `reference_col`, `default_value` | Mark controls as reference |
| `filter_active` | `activity_file`, `on_columns`, `threshold` | Keep only active items |
| `aggregate_replicates` | `groupby` | Median aggregation by group |
| `split_multilabel` | `column`, `separator` | Convert "A\|B\|C" to ["A","B","C"] |
| `filter_single_replicates` | `min_replicates`, `groupby` | Remove groups with < N members |

### Config Schema

```yaml
# Required
input:
  path: "path/to/profiles.parquet"  # Also supports CSV, URLs, S3 paths
  columns: null  # null = all, or list of column names
  filter_query: null  # SQL WHERE clause for lazy filtering
  use_lazy_filter: true  # Filter before loading (memory efficient)

output:
  directory: "${hydra:runtime.output_dir}"
  name: "analysis"  # Prefix for output files
  format: "csv"  # or "parquet"

# Optional preprocessing (list of steps, executed in order)
preprocessing:
  steps:
    - type: filter
      params:
        query: "some_column > 0"
    - type: merge_metadata
      params:
        source: "metadata.duckdb"
        table: "annotations"
        on_columns: "id_column"
        how: "left"

# Analysis parameters
average_precision:
  multilabel: false  # true for multilabel analysis
  params:
    pos_sameby: ["group_column"]
    pos_diffby: []
    neg_sameby: []
    neg_diffby: ["group_column"]
    batch_size: 1000
    distance: "cosine"

mean_average_precision:
  params:
    sameby: ["group_column"]
    null_size: 100000
    threshold: 0.05
    seed: 42
```

### Output Files

Each run produces:
- `{name}_ap_scores.csv` (or `.parquet`) - Per-entity AP scores
- `{name}_map_results.csv` (or `.parquet`) - Aggregated mAP with p-values
- `{name}_map_plot.png` - mAP vs -log10(p) scatter plot (also shows normalized mAP)
- `.hydra/config.yaml` - Full resolved configuration
