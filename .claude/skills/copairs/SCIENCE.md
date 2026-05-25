# Scientific Background: The mAP Framework for Phenotypic Profiling

Based on: Kalinin et al. "A versatile information retrieval framework for evaluating profile strength and similarity" Nature Communications (2025). doi:10.1038/s41467-025-60306-2

## The Problem

Large-scale profiling assays (Cell Painting, Perturb-seq, proteomics) generate high-dimensional data where:
- Each perturbation has multiple replicates
- Feature spaces are massive (100s to 1000s of features)
- Traditional statistical tests make assumptions that don't hold:
  - MANOVA/Hotelling's T² assume normality
  - Sample sizes often smaller than feature dimensions
  - Non-linear, heterogeneous relationships

**Key questions the mAP framework addresses:**

1. **Phenotypic Activity**: Does this perturbation cause a detectable phenotypic change?
2. **Phenotypic Consistency**: Do perturbations with shared biology produce similar phenotypes?
3. **Phenotypic Distinctiveness**: Is this perturbation distinguishable from all other perturbations?

## The Solution: Information Retrieval

Reframe profile evaluation as a retrieval problem:
- Given a query profile, can we retrieve its relevant matches from a collection?
- Use **mean Average Precision (mAP)** as the metric

### Why mAP?

| Property | Benefit |
|----------|---------|
| No distributional assumptions | Works with non-normal, heavy-tailed data |
| Scales with dimensions | Performance improves as features increase |
| Top-heavy bias | Early correct matches contribute more |
| Works with few replicates | Only needs 2+ replicates per perturbation |
| Probabilistic interpretation | AP ≈ P(random relevant > random irrelevant) |

### The Algorithm (General Form)

The same algorithm underlies all three metrics - they differ only in how "relevant" (positive) and "irrelevant" (negative) items are defined:

```
For each profile i in the query group:
  1. Compute distances to all other profiles (query group + reference group)
  2. Rank by similarity (closest = rank 1)
  3. Create binary relevance list: 1 = same group, 0 = different group
  4. Compute Average Precision:
     AP = Σ (Precision@k × ΔRecall@k) for k where relevance = 1

mAP = mean(AP scores for all profiles in query group)
```

**Precision@k** = (# relevant in top k) / k
**Recall@k** = (# relevant in top k) / (total relevant)

### Statistical Significance

P-values via permutation testing:
1. Generate null distribution by shuffling labels (typically 100,000 permutations)
2. p-value = fraction of null mAPs ≥ observed mAP
3. Apply Benjamini-Hochberg FDR correction for multiple testing

## The Three Metrics

### Phenotypic Activity

**Question:** Does perturbation X cause a phenotypic change distinguishable from controls?

**Setup:**
- Query group: Replicates of perturbation X
- Reference group: Negative control replicates (e.g., DMSO)
- Positive pairs: Same perturbation replicates
- Negative pairs: Perturbation vs control

**Use cases:**
- Filter inactive perturbations before downstream analysis
- Assess dataset quality (% retrieved indicates assay sensitivity)
- Compare feature extraction methods

### Phenotypic Consistency

**Question:** Do perturbations sharing an annotation (target, MOA, disease area) cluster together?

**Setup:**
- First, aggregate replicates → one profile per perturbation (typically median)
- Filter to only phenotypically active perturbations
- Query group: Perturbations with same label (target/MOA)
- Reference group: Perturbations with different labels
- Uses multilabel handling (items can have multiple labels)

**Use cases:**
- Validate annotation quality
- Discover new perturbation-annotation associations
- Compare different annotation sources

### Phenotypic Distinctiveness

**Question:** Is this perturbation distinguishable from all other perturbations (not just controls)?

**Setup:**
- Query group: Replicates of perturbation X
- Reference group: All other perturbations' replicates
- Positive pairs: Same perturbation replicates
- Negative pairs: Different perturbations

**Use cases:**
- Identify perturbations with unique phenotypic signatures
- Find perturbations that are highly specific vs broadly acting
- Quality control for perturbation libraries

## Normalized mAP

Raw mAP values depend on the number of positive and negative items. **Normalized mAP** allows comparison across groups with different sizes.

**Formula:** `(AP - μ₀) / (1 - μ₀)`

Where μ₀ is the expected AP under random ranking (computed from group sizes).

**Interpretation:**
- Normalized mAP = 0: Performance equals random chance
- Normalized mAP = 1: Perfect retrieval
- Normalized mAP < 0: Worse than random (indicates anti-correlation or systematic bias)

Use normalized mAP when comparing groups with different numbers of members.

## Blocking and Experimental Design

**Blocking** controls for confounding variables by constraining pair formation.

**Example: Account for plate effects in activity**
```yaml
average_precision:
  params:
    pos_sameby: ["perturbation_id"]     # Same perturbation = positive
    neg_sameby: ["plate_id"]            # Same plate (blocking)
    neg_diffby: ["is_control"]          # Perturbation vs control
```

This ensures negative pairs come from the same plate, controlling for plate-to-plate technical variation.

**Example: Cross-source reproducibility**
```yaml
average_precision:
  params:
    pos_sameby: ["perturbation_id"]
    pos_diffby: ["source"]              # Different sources required
```

This forces positive pairs to come from different lab sites, testing cross-site reproducibility.

## Benchmarking Results

From the paper's simulation study comparing mAP to alternatives:

| Method | Pros | Cons |
|--------|------|------|
| **mAP** | Best sensitivity at high dimensions, no assumptions | Depends on distance metric choice |
| mp-value | Established method | Performance degrades with high dimensions |
| MMD | Nonparametric kernel test | Requires careful kernel bandwidth tuning |
| k-means | Simple clustering | Needs restarts, less stable |

**Key finding:** mAP sensitivity *improves* as feature count increases, while mp-value and MMD can degrade.

## Practical Recommendations

### Preprocessing
1. **Normalize features**: MAD robustize per plate recommended
2. **Feature selection**: Remove low-variance, highly correlated features
3. **Batch correction**: Consider Harmony or Seurat v3 before mAP

### Distance Metrics
- **Cosine** (default): Measures pattern similarity, ignores magnitude
- **Abs cosine**: Ignores sign (useful for bidirectional effects)
- **Correlation**: Similar to cosine, robust to scaling
- **Euclidean/Manhattan**: Magnitude-sensitive

### Interpreting Results

**mAP range:** 0 to 1 (higher is better)
- Random performance is NOT 0.5 - it depends on the ratio of positive to negative pairs
- Use normalized mAP for scale-independent comparison

**p-value interpretation:**
- Use `corrected_p_value` (FDR-corrected), not raw p-value
- Typical thresholds are 0.05 or 0.10 depending on application
- Low percent retrieved across a dataset may indicate batch effects or weak signal

### Interpreting Low Retrieval

If percent retrieved is low:
1. **Batch effects**: Try different normalization/batch correction
2. **Feature quality**: Consider feature selection or different extraction
3. **Weak perturbations**: Assay may not detect the phenotype
4. **Annotation quality**: External annotations may be incomplete/incorrect

## References

1. Kalinin AA, Arevalo J, Serrano E, et al. A versatile information retrieval framework for evaluating profile strength and similarity. Nat Commun. 2025;16:5181.
2. Bray MA, Singh S, Han H, et al. Cell Painting, a high-content image-based assay for morphological profiling using multiplexed fluorescent dyes. Nat Protoc. 2016;11:1757-1774.
3. Corsello SM, et al. The Drug Repurposing Hub: a next-generation drug library and information resource. Nat Med. 2017;23:405-408.
