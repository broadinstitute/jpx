# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb",
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "scanpy",
#     "anndata",
#     "python-dotenv",
#     "loguru",
#     "requests",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    import duckdb
    import numpy as np
    import requests

    from nb00_ss_config import EXTERNAL_DATA_DIR, RAW_DATA_DIR

    # --- jpx (THIS repo) inputs ---
    PROFILES_PARQUET = RAW_DATA_DIR / "profiles" / "all_modalities.parquet"
    METADATA_BASE_DB = EXTERNAL_DATA_DIR / "jump_metadata.duckdb"  # base, canonical
    MOTIVE_ANNOT = EXTERNAL_DATA_DIR / "motive_cpd_gene_annot.parquet"  # LEAKY label key

    # --- MOTIVE paper repo (empirical hook + pair identity only) ---
    MOTIVE_ROOT = Path("/work/users/shsingh/GitHub/jump_voa/2024_Arevalo_NeurIPS_MotiVE")
    # gin + CP-init cold-source config (verified config.json initialization: cp)
    MOTIVE_RUN = MOTIVE_ROOT / "outputs/orf/source/bipartite/gin/2c5c7a"
    MOTIVE_AP = MOTIVE_RUN / "cartesian/test/metrics/average_precision.parquet"
    MOTIVE_SOURCE_MAP = MOTIVE_ROOT / "data/bipartite/orf/source_map.parquet"
    MOTIVE_TARGET_MAP = MOTIVE_ROOT / "data/bipartite/orf/target_map.parquet"

    FEATURE_COLS = [f"X_{i}" for i in range(1, 615)]  # 614 features

    # --- DepMap Breadbox (public, no key) ---
    BREADBOX = "https://depmap.org/portal/breadbox"
    BB_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    DEPMAP_CHRONOS = "a2a0a725-b585-40c8-8c45-a924f8178656"  # gene KO dependency
    DEPMAP_PRISM2 = "07b7bda9-ae00-43b3-bca1-336b9607f8f5"  # PRISM Secondary AUC

    # Lazy cache for the all-gene consensus matrices (built once per modality).
    CONSENSUS_CACHE: dict = {}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MOTIVE per-pair interaction evidence (agentic alternative to the one-shot GNN)

    > This cell is a **complete build brief**. It is the prompt. Read it top to bottom,
    > then compose the rest of this notebook live in the marimo editor (do NOT hand-author
    > marimo dataflow blind - the auto-managed cell signatures fail silently). Build
    > cell-by-cell, run each cell, confirm real output before moving on. Follow the jpx
    > `compose-notebook` skill contract (`.claude/skills/compose-notebook/SKILL.md`).

    ## What MOTIVE is

    MOTIVE is the 2024 Arevalo et al. NeurIPS paper, *"MOTIVE: A Drug-Target Interaction
    Graph For Inductive Link Prediction"* (repo: `2024_Arevalo_NeurIPS_MotiVE`). It builds
    a heterogeneous **graph** from JUMP Cell Painting data - nodes are **compounds** and
    **genes**, and an edge means a known **compound-gene (drug-target) interaction** (the
    compound binds/modulates that gene's protein). Each node carries its **Cell Painting
    morphological profile** as features. A graph neural network is trained for **link
    prediction**: given the graph, score whether a held-out compound-gene pair interacts.
    The emphasis is **inductive** generalization to entities unseen in training - the
    "cold" splits hold out entire compounds (cold-source) or entire genes (cold-target),
    so the model must predict interactions for nodes it never saw an edge for. That cold
    regime is exactly where a single forward pass has the least graph structure to lean on,
    which is what makes it the interesting test bed for the agentic alternative below.

    Glossary for the terms this brief uses as understood:
    - **source = compound, target = gene** (the code is written generically). The
      `binds`/interaction edge is `source-binds-target`.
    - **cold-source split** = hold out whole compounds from training; **cold-target** =
      hold out whole genes. (Transductive `random` split is the easy case.)
    - **CP init** = node embeddings initialized from Cell Painting features (vs learned
      `embs`); on cold splits `embs` is degenerate (no embedding for unseen nodes), so CP
      init is the meaningful one.
    - **GIN / GraphSAGE / GAT** = the GNN backbone variants; GIN tends to win here.
    - **cartesian inference** = score ALL compound x gene pairs (the full ranking table),
      vs **sampled** inference which scores neighbor-sampled subgraphs like training.
    - **ORF vs CRISPR** = the two gene-perturbation modalities supplying gene features:
      ORF over-expresses the gene, CRISPR knocks it out.
    - **per-node AP / mAP** = average precision of a node's ranked interaction list (the
      retrieval metric MOTIVE is judged by).

    ## Why this project (the stake) - read this before optimizing anything

    This notebook exists to make the argument of a **specific talk** concrete:
    "Cell Bio @ Scale", SciLifeLab Stockholm, ~June 3 2026 (10 min + 10 Q&A). The talk's
    thesis - **Shantanu's own**, not an external reviewer's - is that MOTIVE, and the
    broader virtual-cell field, does **one-shot inference** (map input to output in a
    single non-interactive step, no reasoning loop) and that this is the wrong move; the
    path forward is an **agentic loop** that, per gene-compound pair, fetches and reasons
    over evidence (databases, phenocopy/morphology, drug-likeness) working backwards from
    the functional endpoint you act on. (Echoes two recurring points in the field:
    squeezing everything into one model is "information loss", and a perfect prediction
    you can't interpret is useless.) MOTIVE is the lineage's "deep slide"; this artifact
    is the turn from one-shot to loop.

    **What this notebook needs to produce: a talk demo/figure.** One or two worked pairs,
    rendered as a rich, legible per-pair analysis, that a viewer immediately reads as
    "this is what the agentic loop does, and the one-shot scalar cannot."

    **What counts as a win: the artifact's existence and legibility - NOT a benchmark.**
    Success is that a rich, auditable per-pair analysis CAN be produced at all and reads
    clearly. We are explicitly NOT trying to beat MOTIVE's AP/recall, not chasing coverage,
    not optimizing a score. So:
    - Optimize for a **clean, readable, well-sourced single worked pair**, then a second
      for contrast. Depth and clarity over breadth. Do not scale to all 20 pairs for the
      talk - 2 done beautifully beats 20 done thinly.
    - Use BRD4/JQ1 as a **known-true reference** and a GNN-missed sparse pair as the
      **contrast**. NOTE (verified 2026-06-01, see "Reality check" below): with the simple
      signals as specified, BRD4/JQ1 does NOT light up - so the punchline is NOT "every
      stream fires." The punchline is **decomposability**: the GNN emits one scalar
      (0.668) with no account of itself, while the loop lays out exactly which evidence
      exists, which is weak, and WHY (e.g. BRD4 is pan-essential so DepMap can't
      deconvolve it) - an honest, auditable breakdown a single forward pass cannot give.
      That legibility, including where evidence is thin, IS the argument.
    - The labeled pairs **validate the judge / sanity-check the call**; they are not a
      target to chase. Keep the leakage split visible precisely because legibility - "you
      can see what the call rests on" - IS the deliverable.

    Terminology note: say **"one-shot inference"** / "a single non-interactive prediction
    step, no reasoning loop", NOT "single forward pass" (that literal claim is breakable -
    MOTIVE is literally a GNN forward pass; the defensible axis is interaction/iteration:
    act-observe-adjust in a loop vs map-input-to-output once).

    ## Why this exists (the design - per-pair analysis + judge)

    MOTIVE's model is `f_GNN: (gene, compound) -> scalar logit`. We want
    `f_agent: (gene, compound) -> a rich, inspectable analysis` - a worked reasoning
    artifact that pulls in whatever evidence bears on the interaction and lays it out.
    A second function `judge: analysis -> assessment` (human or judge-LLM) reads it and
    decides whether the interaction is supported and whether the reasoning is sound.

    Pipeline: **pair -> analysis -> judgment.** The scalar appears only at the judge,
    carrying its justification. This is the answer to the "one-shot inference" critique:
    reasoning is externalized, auditable, and can fetch evidence per pair (the loop),
    instead of being compressed into GNN weights.

    Two consequences that shape the build:
    - We are **NOT** optimizing a scalar. Do not design around a ranking metric. Define
      (a) what the analysis *contains* and (b) what the judge *assesses*. Labeled pairs
      validate the judge; they are not a target the analysis chases.
    - The **leakage audit falls out for free**: a "rich enough to be judged" analysis
      shows its sources, so the judge can see whether a call leaned on label-bearing
      annotations vs genuinely predictive evidence. No separate instrumentation needed.

    ## Paths on spirit (two repo roots - everything below is under one of these)

    This work spans two repos, both on spirit:
    - **`<JPX>`** = `/work/users/shsingh/GitHub/jump/jpx` - THIS notebook's repo and the
      working directory. All `data/...` paths in this brief (e.g.
      `data/raw/profiles/all_modalities.parquet`, `data/external/jump_metadata.duckdb`) are
      relative to here.
    - **`<MOTIVE>`** = `/work/users/shsingh/GitHub/jump_voa/2024_Arevalo_NeurIPS_MotiVE` -
      the MOTIVE paper repo. Used only for the empirical hook (the failing-gene results)
      and for pair identity. All `<MOTIVE>/...` paths below are relative to here.

    ## What grounds it (the empirical hook)

    On the ORF cold-**source** split (gin + CP init, cartesian inference), GNN per-node AP
    is bimodal and **failure concentrates on low-connectivity genes**: degree-1 genes
    (the majority, 1196 of 2229) have mean AP ~0.15 and 84% fail (<0.1); degree 4-10 ->
    0.85; degree 11+ -> ~1.0. Where the graph is dense the GNN is near-perfect; where it
    is sparse the single pass collapses (CP features alone don't carry it). **1002 genes
    have exactly one known test compound AND AP < 0.05** (near-total miss), spanning 226
    distinct compounds. These sparse misses are the regime the agentic loop should beat.
    (MOTIVE outputs on spirit:
    `<MOTIVE>/outputs/orf/source/bipartite/gin/2c5c7a/cartesian/test/`. Hash `2c5c7a` is the
    gin + **CP-init** cold-source config (verified: `config.json` has `initialization: cp`;
    the sibling `28f7da` is the `embs`-init run, degenerate on cold splits - no embedding
    for unseen nodes). Per-node AP: `.../2c5c7a/cartesian/test/metrics/average_precision.parquet`
    (cols: node_id, node_type, average_precision, n_pos_pairs). Node-id -> name maps:
    `<MOTIVE>/data/bipartite/orf/source_map.parquet` and `.../target_map.parquet`, col
    `"0"` = `Metadata_InChIKey` (source) / `Metadata_Symbol` (target). Test edges (with
    subset in {message,train,valid,test}): `<MOTIVE>/data/bipartite/orf/source/s_t_labels.parquet`.)

    ## The deliverable: an evidence portfolio of pluggable streams

    Each **stream** is an `@app.function` that takes a pair and returns a structured record:
    `{source, signal, value, leaky, summary}`. Keep **predictive** and **prior-knowledge**
    evidence split so the judge sees what a call rests on. Build morphology first
    end-to-end, then add DepMap, then structure. Architect for collation now, execute
    simple first.

    ### Stream 1 - Morphology (predictive, non-leaky) -- BUILD THIS FIRST

    Profiles live in ONE shared feature space (JUMP `ALL/v1.0b`: joint
    featselect+sphering+harmony over compound + ORF + CRISPR), so compound-vs-gene cosine
    is valid directly, no per-modality alignment.

    - File: `data/raw/profiles/all_modalities.parquet` (2.5 GB, 946,431 well-level rows).
    - Columns: `Metadata_Source, Metadata_Plate, Metadata_Well, Metadata_JCP2022` +
      **614 feature columns named `X_1 .. X_614`** (float). Keyed only by `Metadata_JCP2022`.
    - **Consensus** = median over all wells of a given `Metadata_JCP2022`.

    Name/ID join via `data/external/jump_metadata.duckdb` (base, canonical; NOT the augmented
    one - target mappings there are still being populated):
    - `compound(Metadata_JCP2022, Metadata_InChIKey, Metadata_InChI, Metadata_SMILES)`
    - `orf(Metadata_JCP2022, Metadata_Symbol, Metadata_NCBI_Gene_ID, ...)`
    - `crispr(Metadata_JCP2022, Metadata_NCBI_Gene_ID, Metadata_Symbol)`
    - `well(Metadata_Source, Metadata_Plate, Metadata_Well, Metadata_JCP2022)`

    A compound is identified by `Metadata_InChIKey`, a gene by `Metadata_Symbol`. A gene
    has up to TWO perturbation profiles: an **ORF** (overexpression) JCP2022 and a **CRISPR**
    (knockout) JCP2022 - both are interesting and the SIGN difference is the point.

    Signals the morphology stream should compute:
    - **Directional connectivity** = cosine(compound consensus, gene ORF consensus) AND
      cosine(compound consensus, gene CRISPR consensus). The SIGN PATTERN is the signal,
      not a naive "high phenocopy cosine": an inhibitor should anti-correlate the ORF
      (overexpression) and correlate the CRISPR (knockout). Report both cosines and the
      sign pattern. (Logged correction: "phenocopy" alone is too crude and sign-ambiguous
      against ORF overexpression.)
    - **Percentile rank** of the true gene's connectivity vs ALL genes (where does this
      pair sit in the distribution of the compound against every gene profile?).
    - **Morphological neighbors** of the compound (and of the gene) - nearest profiles.
    - **Phenotypic activity** of both the compound and the gene perturbations (is each
      perturbation morphologically active at all? an inactive perturbation explains a
      dead pair). Activity lives in the jpx DuckDB layer - see `nb02_ss_queries` /
      `nb03_ss_profiles.join_activity`; or recompute simply.

    ### Stream 2 - DepMap (predictive, ORTHOGONAL, essentially non-leaky) -- ADD SECOND

    Target deconvolution: does the compound's PRISM drug-sensitivity profile across cell
    lines track the gene's Chronos knockout-dependency profile? Derived from independent
    assays, NOT the MOTIVE label DBs - so it is the most convincing form of the thesis.
    Implement as an inline ~10-line **Breadbox REST** client (public, no key:
    `https://depmap.org/portal/breadbox`). Helpers: `bb_get`/`bb_post`;
    `POST datasets/matrix/{id}` with `{features:[label], feature_identifier:"label"}` for
    profiles; `POST temp/associations/query-slice` for cross-modality associations.
    Dataset IDs: Chronos dependency `a2a0a725-b585-40c8-8c45-a924f8178656`; PRISM Secondary
    AUC dose-response `07b7bda9-ae00-43b3-bca1-336b9607f8f5` (1482 cpds, UPPERCASE labels);
    PRISM Primary Viability `a3b600c8-1abf-4ff3-bad1-2b310cbbdc28` (6562 cpds).

    Coverage (checked, 20 selected pairs): **gene side 17/17** (all in Chronos);
    **compound side ~11/17** have a retrievable PRISM profile, matched by normalized
    Repurposing `pert_iname` -> PRISM UPPERCASE label. Only 4/17 are also in the richer
    Secondary dose-response set: BRD4/JQ1, ACTB/ethinyl-estradiol, ADIPOR2/colforsin,
    ACTG1/dihydroartemisinin. So DepMap is a **"fires-when-available" majority-coverage
    stream** with graceful fallback; morphology is the backbone (covers all). Do NOT try
    the InChIKey/jump_smiles route via Breadbox - it exposes SMILES for only ~3% of cpds.

    ### Stream 3 - Structure (lighter, ADD THIRD)

    Does the compound resemble known ligands of the target/family? Use jpx fingerprints
    (`nb17`/`nb18`/`nb37`). Cheminformatics pixi env (rdkit is conda-only).

    ### Synthesis = the object the judge reads

    Streams kept apart (predictive vs prior-knowledge), with a provisional call +
    confidence + explicit grounds. The judge (human first, judge-LLM later) reads this.

    ## Leakage rules (critical - this is half the point)

    jpx literally contains the answer key. Quarantine and LABEL these as prior-knowledge;
    the predictive streams must NOT consult them:
    - `data/external/motive_cpd_gene_annot.parquet` - the literal MOTIVE compound-gene
      label (`source` cpd name, `target` gene Symbol, `rel_type`, `inchikey`, `database`).
    - `data/external/repurposing_{drugs,samples}_20200324.txt` - Drug Repurposing Hub
      MoA/target/clinical-phase.
    - `targetannotations_*`, `pd_export_*targets*`, ChEMBL/Probes target mappings.

    Using these to **pick/identify** the demo pairs (compound name <-> InChIKey, gene
    Symbol) is fine - that is experiment design, not evidence. The rule is that the
    analysis/streams the judge reads as PREDICTIVE evidence must not read them. Mark every
    record's `leaky` field accordingly.

    ## Demo pairs (start with 2, then scale to the 20)

    - **Known-true reference:** `BRD4` + `JQ1`. A real, famous BET-bromodomain interaction
      and in DepMap on both sides. CORRECTED per-node AP on the live CP-init run `2c5c7a`
      (NOT 1.0, the earlier claim was wrong): BRD4 gene **0.341**, JQ1 compound **0.668**.
      See "Reality check" - the simple streams do not strongly support it; that is expected
      and is part of the legibility point, not a bug to hide.
    - **Miss:** `ACTG1` + `dihydroartemisinin` (gene AP **0.006**) OR `ACTB` +
      `ethinyl-estradiol` (gene AP **0.011**) - both confirmed misses, both in PRISM so the
      DepMap stream fires. Do NOT use ACVR2B's compound for a DepMap-inclusive demo - it
      isn't in Repurposing/DepMap.

    ### Reality check (verified 2026-06-01 by running the notebook end-to-end)

    With the signals exactly as specified above, the **positive control comes out weak**:
    - Morphology: directional cosines ~+0.035 (ORF) / +0.072 (CRISPR), BRD4 at the ~49.6th
      connectivity percentile (~random), sign pattern NOT inhibitor-consistent.
    - DepMap: naive PRISM-vs-Chronos Pearson r ~ +0.07 over ~383-577 shared cell lines -
      weak because **BRD4 is broadly (pan-)essential**, so its Chronos dependency profile
      is nearly flat and carries little line-to-line signal to correlate against. BRD4 is
      therefore a structurally POOR positive control for the DepMap deconvolution stream
      (known a priori - a selectively-essential gene would be a fairer test).
    - So `provisional_call` returns "not supported by predictive evidence" even for the
      true pair.

    RESOLVED diagnostic (2026-06-01): the weak morphology is NOT an inactive-profile or
    bad-match artifact. JQ1 (`JCP2022_017085`) is morphologically ACTIVE - phenotypic
    activity mAP 1.0, corrected p 0.0001 (`activity_results`, `compound_with_source7`).
    JUMP stored it as the flat `DNVXATUJJDPFDM-UHFFFAOYSA-N` (no enantiomer layer), but the
    plated material gives a reproducible, distinct profile. So the weak BRD4 cosines are a
    GENUINE result. CAVEAT worth surfacing in the analysis (good for the legibility point):
    JQ1's profile is **source-7-only, 5 wells / 5 plates**. Source 7 is the documented
    batch-effect source in JUMP (jpx nb09/nb11/nb12); BRD4's ORF/CRISPR profiles come from
    other sources, so the cross-modality cosine is a 5-well single-source consensus compared
    across sources in a harmonized-but-imperfect space - a real reason to down-weight the
    morphology signal, and exactly the provenance the one-shot scalar 0.668 hides.

    Still worth deciding before the talk (each is a one-function swap): (1) replace the raw
    PRISM/Chronos Pearson with Breadbox's precomputed `temp/associations/query-slice` (more
    robust than hand-rolled correlation) - though it won't rescue BRD4, which is
    pan-essential. (2) If you want a positive where streams fire STRONGLY, pick one whose
    gene is selectively (not pan-) essential and whose compound is confirmed active with
    good (non-source-7) well coverage. Otherwise ship BRD4/JQ1 as the "auditable null":
    the loop shows active compound + pan-essential gene + cross-source caveat = honest
    "evidence is thin and here's why," which the scalar cannot express.

    To resolve a compound name -> `Metadata_InChIKey` for the demo, the cleanest in-repo
    source is `motive_cpd_gene_annot.parquet` (filter `target` = the gene Symbol; the
    `inchikey` column gives the paired compound's key) or the Repurposing samples file.
    This is pair-identity resolution, NOT evidence. Then join InChIKey -> JCP2022 via
    `jump_metadata.duckdb.compound`, and JCP2022 -> profiles via `all_modalities.parquet`.

    ## jpx composition contract (follow exactly)

    - `with app.setup:` for shared imports + `sys.path.insert(0, notebooks/)`; import
      foundation helpers: `nb00_ss_config` (paths: `RAW_DATA_DIR`, `EXTERNAL_DATA_DIR`,
      `METADATA_DB`, `ANNDATA_DIR`, `DEFAULT_DPI`, `JUMP_CPUS`), `nb02_ss_queries`
      (activity/consistency DuckDB queries), `nb03_ss_profiles`
      (`load_profiles`, `join_metadata`, `join_activity`), `nb04_ss_visualization` (plots).
      NOTE: `all_modalities.parquet` is a raw parquet (not an h5ad in `ANNDATA_DIR`), so
      load it directly with duckdb/pandas - `load_profiles` does not cover it.
    - Reusable logic as `@app.function` (only references `app.setup` symbols or other
      `@app.function`s) so other notebooks can import it. Each stream is one such function.
    - Narrative first: every section opens with `mo.md(...)` stating question + verdict.
    - Populate any dropdowns from the data, never hardcode option lists.
    - PEP 723 inline deps header (sandbox can't see the outer env). Minimum:
      `marimo, duckdb, pandas, numpy, matplotlib, scipy, scanpy, python-dotenv, loguru,
      anndata, requests`. No exact pins. No conda-only pkgs (rdkit/cupy) - those need the
      pixi env, so the structure stream notebook/section runs under
      `pixi run -e cheminformatics`, not `--sandbox`.
    - Unique `_`-prefixed throwaway cell vars; `hide_code=True` on narrative cells.

    ## Launch + validate (on spirit, reachable via Tailscale `spirit:<PORT>`)

    This is a pure-Python notebook (morphology + DepMap streams): launch with the sandbox.
    ```bash
    nohup uvx marimo@latest edit notebooks/nb43_ss_motive_interaction_evidence.py \
        --port 2731 --host 0.0.0.0 --headless --sandbox --no-token \
        > /tmp/marimo-nb43.log 2>&1 &
    ```
    Validation rule (jpx): after composing, run ALL cells and confirm real output (not
    empty tables / broken plots) before declaring done. Then `pixi run ruff check/format
    notebooks/`.

    ## Build order (recommended)

    1. `app.setup` + foundation imports + PEP 723 header. Run it.
    2. Helper: load consensus profile for a `Metadata_JCP2022` (median over wells) from
       `all_modalities.parquet`. Verify it returns a 614-vector for one known JCP2022.
    3. Helpers: InChIKey -> compound JCP2022; Symbol -> ORF JCP2022 and CRISPR JCP2022.
    4. `@app.function morphology_stream(inchikey, symbol)` -> list of
       `{source, signal, value, leaky, summary}` records (directional cosines + sign
       pattern, percentile rank vs all genes, neighbors, activity). Test on BRD4/JQ1.
    5. Render the per-pair analysis for the 2 demo pairs (a table + a short narrative).
    6. Stub `depmap_stream` and `structure_stream` with the interfaces above; fill DepMap
       next (coverage-gated, graceful fallback), structure last.
    7. Synthesis cell: collate streams (predictive vs prior-knowledge), provisional call.

    Status coming in: pairs selected; profiles downloaded; metadata join + DepMap coverage
    (~11/17) confirmed; composition contract read. This notebook is the scaffold to fill.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How this notebook is organized

    Below the brief, the notebook is built bottom-up as importable `@app.function`
    helpers, then a small interactive viewer:

    1. **ID resolution** - InChIKey -> compound JCP2022 (matched on the 14-char InChIKey
       skeleton, since JUMP stores the connectivity key and the label files carry stereo
       suffixes), Symbol -> ORF/CRISPR consensus profile.
    2. **Consensus profiles** - median over wells, in the shared `ALL/v1.0b` feature space.
    3. **Streams** - `morphology_stream` (predictive, non-leaky), `depmap_stream`
       (predictive, orthogonal), `structure_stream` (stub). Each returns
       `{source, signal, value, leaky, summary}` records.
    4. **GNN baseline** - `motive_baseline` reads the one-shot GNN's per-node AP **live**
       from the MOTIVE run. This is the thing we contrast against, not evidence.
    5. **Viewer** - pick a pair, see the assembled portfolio + a provisional call.

    **A correction surfaced while wiring this up:** the brief's "BRD4/JQ1 AP 1.0" does not
    match the live `2c5c7a` (CP-init) run. Actual per-node AP there is BRD4 gene **0.34**,
    JQ1 compound **0.67**; the misses (ACTG1 gene 0.006, ACTB gene 0.011) do hold. The
    notebook reads AP live so the figure never drifts from the file.
    """)
    return


@app.function
def evidence_record(source, signal, value, leaky, summary):
    """One stream output row. `leaky=True` marks prior-knowledge / label-bearing evidence."""
    return {
        "source": source,
        "signal": signal,
        "value": value,
        "leaky": leaky,
        "summary": summary,
    }


@app.function
def consensus_profile(jcp):
    """Median 614-vector over all wells of a Metadata_JCP2022, or None if absent."""
    if jcp is None:
        return None
    cols = ", ".join(f"median({c}) AS {c}" for c in FEATURE_COLS)
    df = duckdb.sql(f"SELECT {cols} FROM read_parquet('{PROFILES_PARQUET}') WHERE Metadata_JCP2022 = '{jcp}'").df()
    if df.empty or df.iloc[0].isna().all():
        return None
    return df.iloc[0].to_numpy(dtype=float)


@app.function
def resolve_compound_jcp(inchikey):
    """InChIKey -> a compound JCP2022 present in the profiles (match 14-char skeleton)."""
    pfx = inchikey[:14]
    con = duckdb.connect()
    con.execute(f"ATTACH '{METADATA_BASE_DB}' AS meta (READ_ONLY)")
    df = con.execute(f"SELECT Metadata_JCP2022 AS jcp FROM meta.compound WHERE Metadata_InChIKey LIKE '{pfx}%'").df()
    con.close()
    for jcp in df["jcp"]:
        n = duckdb.sql(
            f"SELECT COUNT(*) AS n FROM read_parquet('{PROFILES_PARQUET}') WHERE Metadata_JCP2022 = '{jcp}'"
        ).fetchone()[0]
        if n > 0:
            return jcp
    return df["jcp"].iloc[0] if not df.empty else None


@app.function
def gene_consensus(symbol, modality):
    """Median consensus over all wells of all constructs of `symbol` in {orf, crispr}."""
    cols = ", ".join(f"median(p.{c}) AS {c}" for c in FEATURE_COLS)
    con = duckdb.connect()
    con.execute(f"ATTACH '{METADATA_BASE_DB}' AS meta (READ_ONLY)")
    df = con.execute(
        f"SELECT {cols} FROM read_parquet('{PROFILES_PARQUET}') p "
        f"JOIN meta.{modality} t ON p.Metadata_JCP2022 = t.Metadata_JCP2022 "
        f"WHERE t.Metadata_Symbol = '{symbol}'"
    ).df()
    con.close()
    if df.empty or df.iloc[0].isna().all():
        return None
    return df.iloc[0].to_numpy(dtype=float)


@app.function
def gene_consensus_matrix(modality):
    """All-gene consensus (pooled per Symbol) for percentile rank + neighbours. Cached."""
    if modality in CONSENSUS_CACHE:
        return CONSENSUS_CACHE[modality]
    cols = ", ".join(f"median(p.{c}) AS {c}" for c in FEATURE_COLS)
    con = duckdb.connect()
    con.execute(f"ATTACH '{METADATA_BASE_DB}' AS meta (READ_ONLY)")
    df = con.execute(
        f"SELECT t.Metadata_Symbol AS symbol, {cols} "
        f"FROM read_parquet('{PROFILES_PARQUET}') p "
        f"JOIN meta.{modality} t ON p.Metadata_JCP2022 = t.Metadata_JCP2022 "
        f"GROUP BY t.Metadata_Symbol"
    ).df()
    con.close()
    symbols = df["symbol"].to_numpy()
    mat = df[FEATURE_COLS].to_numpy(dtype=float)
    CONSENSUS_CACHE[modality] = (symbols, mat)
    return CONSENSUS_CACHE[modality]


@app.function
def cosine(a, b):
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))


@app.function
def cosine_to_all(vec, mat):
    """Cosine of `vec` against every row of `mat` -> 1-D array."""
    norms = np.linalg.norm(mat, axis=1)
    vn = np.linalg.norm(vec)
    denom = norms * vn
    out = mat @ vec
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, out / denom, np.nan)
    return out


@app.function
def compound_provenance(jcp):
    """Well count + source(s) for a compound JCP2022 - a data-quality / generalization caveat.

    source_7 is the documented JUMP batch-effect source; a source-7-only profile means
    cross-source generalization is unverified (a thing the GNN scalar cannot surface).
    """
    if jcp is None:
        return None
    df = duckdb.sql(
        f"SELECT Metadata_Source AS source, COUNT(*) AS n "
        f"FROM read_parquet('{PROFILES_PARQUET}') "
        f"WHERE Metadata_JCP2022 = '{jcp}' GROUP BY Metadata_Source ORDER BY n DESC"
    ).df()
    if df.empty:
        return None
    sources = {row.source: int(row.n) for row in df.itertuples()}
    return {"n_wells": int(df["n"].sum()), "sources": sources}


@app.function
def morphology_stream(inchikey, symbol, k_neighbors=5):
    """Predictive, non-leaky morphology evidence for a (compound, gene) pair."""
    records = []
    cpd_jcp = resolve_compound_jcp(inchikey)
    cpd = consensus_profile(cpd_jcp)
    orf = gene_consensus(symbol, "orf")
    crispr = gene_consensus(symbol, "crispr")

    if cpd is None:
        records.append(
            evidence_record(
                "morphology",
                "compound_profile",
                None,
                False,
                f"no JUMP profile for {inchikey[:14]} (compound side dead)",
            )
        )
        return records

    # 0. Provenance / generalization caveat (legibility payload - the GNN can't state this)
    prov = compound_provenance(cpd_jcp)
    if prov is not None:
        src_str = ", ".join(f"{s} ({n}w)" for s, n in prov["sources"].items())
        source7_only = set(prov["sources"]) == {"source_7"}
        records.append(
            evidence_record(
                "morphology",
                "compound_provenance",
                {
                    "n_wells": prov["n_wells"],
                    "sources": prov["sources"],
                    "source7_only": source7_only,
                },
                False,
                f"compound profile: {prov['n_wells']} wells from {src_str}"
                + (
                    "  [CAVEAT: source-7-only - documented batch-effect source; cross-source generalization unverified]"
                    if source7_only
                    else ""
                ),
            )
        )

    # 1. Directional connectivity + sign pattern
    cos_orf = cosine(cpd, orf)
    cos_crispr = cosine(cpd, crispr)
    records.append(
        evidence_record(
            "morphology",
            "cosine_orf",
            cos_orf,
            False,
            "compound vs ORF (over-expression): " + ("n/a (no ORF profile)" if cos_orf is None else f"{cos_orf:+.3f}"),
        )
    )
    records.append(
        evidence_record(
            "morphology",
            "cosine_crispr",
            cos_crispr,
            False,
            "compound vs CRISPR (knockout): "
            + ("n/a (no CRISPR profile)" if cos_crispr is None else f"{cos_crispr:+.3f}"),
        )
    )
    if cos_orf is not None and cos_crispr is not None:
        consistent = cos_orf < 0 < cos_crispr
        records.append(
            evidence_record(
                "morphology",
                "sign_pattern",
                {
                    "orf": cos_orf,
                    "crispr": cos_crispr,
                    "inhibitor_consistent": consistent,
                },
                False,
                "inhibitor-consistent (anti-correlates over-expression, correlates knockout)"
                if consistent
                else "sign pattern NOT inhibitor-consistent",
            )
        )

    # 2. Percentile rank of the true gene vs all ORF genes
    symbols, mat = gene_consensus_matrix("orf")
    sims = cosine_to_all(cpd, mat)
    idx = np.where(symbols == symbol)[0]
    if idx.size:
        true_sim = sims[idx[0]]
        valid = ~np.isnan(sims)
        pct = float((sims[valid] < true_sim).mean() * 100)
        records.append(
            evidence_record(
                "morphology",
                "percentile_rank_orf",
                pct,
                False,
                f"{symbol} ranks at the {pct:.1f}th percentile of this compound's "
                f"connectivity against all {valid.sum()} ORF genes (cosine {true_sim:+.3f})",
            )
        )

    # 3. Morphological neighbours of the compound (nearest ORF genes)
    order = np.argsort(np.where(np.isnan(sims), -np.inf, sims))[::-1][:k_neighbors]
    neigh = [(symbols[i], float(sims[i])) for i in order]
    records.append(
        evidence_record(
            "morphology",
            "compound_neighbors_orf",
            neigh,
            False,
            "top ORF-gene neighbours: " + ", ".join(f"{s} ({v:+.2f})" for s, v in neigh),
        )
    )

    # 4. Phenotypic activity proxy (consensus L2 norm + percentile vs all genes)
    gene_norms = np.linalg.norm(mat, axis=1)
    for label, vec in (("compound", cpd), ("gene_orf", orf), ("gene_crispr", crispr)):
        if vec is None:
            continue
        norm = float(np.linalg.norm(vec))
        pct = float((gene_norms < norm).mean() * 100)
        records.append(
            evidence_record(
                "morphology",
                f"activity_{label}",
                norm,
                False,
                f"{label} consensus L2 norm {norm:.1f} "
                f"({pct:.0f}th pct vs ORF genes; higher = more morphologically active)",
            )
        )
    return records


@app.function
def bb_matrix(dataset_id, label):
    """POST Breadbox datasets/matrix -> {model_id: value} for a single feature label."""
    r = requests.post(
        f"{BREADBOX}/datasets/matrix/{dataset_id}",
        json={"features": [label], "feature_identifier": "label"},
        headers=BB_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    return j.get(label, {}) if isinstance(j, dict) else {}


@app.function
def prism_label_candidates(name, explicit=None):
    """Candidate PRISM UPPERCASE labels for a compound name."""
    cands = []
    if explicit:
        cands.append(explicit)
    if name:
        up = name.upper()
        cands += [up, up.replace(" ", "-"), up.replace("-", " "), up.replace(" ", "")]
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


@app.function
def depmap_stream(symbol, compound_name=None, prism_label=None):
    """Predictive, orthogonal: does PRISM drug sensitivity track Chronos KO dependency?

    Fires when available; degrades gracefully to a labelled 'unavailable' record.
    """
    try:
        gene = bb_matrix(DEPMAP_CHRONOS, symbol)
    except Exception as e:  # noqa: BLE001 - network is best-effort
        return [evidence_record("depmap", "chronos", None, False, f"Chronos unavailable: {e}")]
    if not gene:
        return [evidence_record("depmap", "chronos", None, False, f"{symbol} not in Chronos")]

    records = []

    # Chronos dependency distribution. A pan-essential gene (very negative mean, low spread)
    # is the same in nearly every line -> a near-flat profile DepMap cannot deconvolve. This
    # record EXPLAINS a weak correlation rather than hiding it (the legibility payload).
    gvals = np.array([v for v in gene.values() if v is not None], dtype=float)
    gvals = gvals[np.isfinite(gvals)]
    g_mean, g_std = float(gvals.mean()), float(gvals.std())
    pan_essential = g_mean < -0.5 and g_std < 0.25
    records.append(
        evidence_record(
            "depmap",
            "chronos_profile",
            {"mean": g_mean, "std": g_std, "pan_essential": pan_essential},
            False,
            f"{symbol} Chronos dependency: mean {g_mean:+.2f}, std {g_std:.2f} "
            f"across {gvals.size} lines"
            + (
                "  [pan-essential - near-flat profile, little for cross-line deconvolution to grip]"
                if pan_essential
                else ""
            ),
        )
    )

    labels = prism_label_candidates(compound_name, prism_label)
    cpd, used = {}, None
    for lab in labels:
        try:
            v = bb_matrix(DEPMAP_PRISM2, lab)
        except Exception:  # noqa: BLE001
            v = {}
        if v:
            cpd, used = v, lab
            break
    if not cpd:
        records.append(
            evidence_record(
                "depmap",
                "prism",
                None,
                False,
                f"compound not in PRISM Secondary (tried {labels})",
            )
        )
        return records

    common = [c for c in (set(gene) & set(cpd)) if gene[c] is not None and cpd[c] is not None]
    g = np.array([gene[c] for c in common], dtype=float)
    d = np.array([cpd[c] for c in common], dtype=float)
    mask = np.isfinite(g) & np.isfinite(d)
    g, d = g[mask], d[mask]
    if g.size < 10:
        records.append(
            evidence_record(
                "depmap",
                "prism_chronos_corr",
                None,
                False,
                f"insufficient valid cell-line overlap ({g.size})",
            )
        )
        return records
    r = float(np.corrcoef(g, d)[0, 1])
    records.append(
        evidence_record(
            "depmap",
            "prism_chronos_corr",
            {
                "r": r,
                "n_lines": len(g),
                "prism_label": used,
                "pan_essential": pan_essential,
                "g": g.tolist(),
                "d": d.tolist(),
            },
            False,
            f"Pearson r={r:+.3f} across {len(g)} cell lines "
            f"(PRISM '{used}' sensitivity vs Chronos {symbol} dependency)"
            + ("  - weak, expected given pan-essentiality above" if pan_essential else ""),
        )
    )
    return records


@app.function
def structure_stream(inchikey, symbol):
    """Stub: chemical-structure similarity to known target ligands.

    Needs RDKit (conda-only) -> run the structure section under
    `pixi run -e cheminformatics`, not the pure-Python sandbox. Interface only here.
    """
    return [
        evidence_record(
            "structure",
            "ligand_similarity",
            None,
            False,
            "not computed in the sandbox (RDKit is conda-only; wire via nb17/nb18/nb37 under pixi -e cheminformatics)",
        )
    ]


@app.function
def motive_baseline(inchikey, symbol):
    """The one-shot GNN's per-node AP, read LIVE from the MOTIVE 2c5c7a run.

    This is the baseline we contrast against - context, NOT predictive evidence.
    """
    out = {}
    if not MOTIVE_AP.exists():
        return out
    pfx = inchikey[:14]
    gene = duckdb.sql(
        f"SELECT a.average_precision AS ap, a.n_pos_pairs AS npos "
        f"FROM read_parquet('{MOTIVE_TARGET_MAP}') t "
        f"JOIN read_parquet('{MOTIVE_AP}') a "
        f"  ON a.node_id = t.\"0\" AND a.node_type = 'target' "
        f"WHERE t.Metadata_Symbol = '{symbol}'"
    ).df()
    cpd = duckdb.sql(
        f"SELECT a.average_precision AS ap, a.n_pos_pairs AS npos "
        f"FROM read_parquet('{MOTIVE_SOURCE_MAP}') s "
        f"JOIN read_parquet('{MOTIVE_AP}') a "
        f"  ON a.node_id = s.\"0\" AND a.node_type = 'source' "
        f"WHERE s.Metadata_InChIKey = '{pfx}'"
    ).df()
    if not gene.empty:
        out["gene_ap"] = float(gene["ap"].iloc[0])
        out["gene_npos"] = int(gene["npos"].iloc[0])
    if not cpd.empty:
        out["compound_ap"] = float(cpd["ap"].iloc[0])
        out["compound_npos"] = int(cpd["npos"].iloc[0])
    return out


@app.function
def provisional_call(records):
    """Transparent stand-in for the judge: a call + confidence + explicit grounds.

    Predictive evidence only. Lists exactly what it rested on (this IS the deliverable).
    """
    by = {(r["source"], r["signal"]): r["value"] for r in records if not r["leaky"]}
    grounds, score = [], 0.0

    prov = by.get(("morphology", "compound_provenance"))
    if isinstance(prov, dict) and prov.get("source7_only"):
        grounds.append("CAVEAT: compound profile is source-7-only (batch-effect source)")

    sign = by.get(("morphology", "sign_pattern"))
    if isinstance(sign, dict):
        if sign["inhibitor_consistent"]:
            score += 1.0
            grounds.append("morphology sign pattern is inhibitor-consistent")
        else:
            grounds.append("morphology sign pattern not inhibitor-consistent")

    pct = by.get(("morphology", "percentile_rank_orf"))
    if isinstance(pct, (int, float)):
        if pct >= 90:
            score += 1.0
            grounds.append(f"gene in top decile of compound connectivity ({pct:.0f}th pct)")
        elif pct >= 75:
            score += 0.5
            grounds.append(f"gene above median connectivity ({pct:.0f}th pct)")
        else:
            grounds.append(f"gene at {pct:.0f}th pct of connectivity (unremarkable)")

    dm = by.get(("depmap", "prism_chronos_corr"))
    if isinstance(dm, dict):
        if abs(dm["r"]) >= 0.2:
            score += 1.0
            grounds.append(f"DepMap PRISM/Chronos correlate (r={dm['r']:+.2f}, {dm['n_lines']} lines)")
        elif dm.get("pan_essential"):
            grounds.append(
                f"DepMap correlation weak (r={dm['r']:+.2f}) - gene is pan-essential, "
                f"so the stream cannot deconvolve (not evidence against the pair)"
            )
        else:
            grounds.append(f"DepMap correlation weak (r={dm['r']:+.2f})")
    else:
        grounds.append("DepMap stream did not fire (coverage)")

    if score >= 2.0:
        call, conf = "supported", "high" if score >= 2.5 else "moderate"
    elif score >= 1.0:
        call, conf = "weakly supported", "low"
    else:
        call, conf = "not supported by predictive evidence", "low"
    return {"call": call, "confidence": conf, "score": score, "grounds": grounds}


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-pair evidence portfolio

    Pick a pair. **BRD4 / (+)-JQ1** is the known-true reference - and, with the simple signals
    here, an **auditable null**: it does *not* light up (weak morphology, weak DepMap), and the
    provisional call comes back "not supported." That is the point, not a bug. The loop names
    *why* the evidence is thin - JQ1's JUMP profile is **source-7-only** (the documented
    batch-effect source) and **BRD4 is pan-essential**, so its near-flat Chronos profile gives
    the deconvolution nothing to grip. The one-shot GNN emits a single scalar (compound AP
    0.668) that can express none of that. **Decomposability - including honest "the evidence is
    thin, and here's exactly which part and why" - is the anti-one-shot argument.** The other
    pairs are GNN-missed genes (per-node AP ~0.01) shown for contrast. Depth over breadth.
    """)
    return


@app.cell
def _(mo):
    # Demo pairs are a DESIGNED selection (pair identity is experiment design, not evidence).
    # InChIKeys carry stereo; resolution matches the 14-char skeleton against JUMP.
    demo_pairs = {
        "BRD4 / (+)-JQ1  [known-true / auditable null]": {
            "symbol": "BRD4",
            "inchikey": "DNVXATUJJDPFDM-KRWDZBQOSA-N",
            "compound_name": "JQ1",
            "prism_label": "JQ1-(+)",
        },
        "ACTG1 / dihydroartemisinin  [GNN miss]": {
            "symbol": "ACTG1",
            "inchikey": "BJDCWCLMFKKGEE-ITYWPALDSA-N",
            "compound_name": "dihydroartemisinin",
            "prism_label": "DIHYDROARTEMISININ",
        },
        "ACTB / ethinyl-estradiol  [GNN miss]": {
            "symbol": "ACTB",
            "inchikey": "BFPYWIDHMRZLRN-SLHNCBLASA-N",
            "compound_name": "ethinyl-estradiol",
            "prism_label": "ETHINYL-ESTRADIOL",
        },
    }
    pair_dropdown = mo.ui.dropdown(options=demo_pairs, value=list(demo_pairs)[0], label="Pair")
    run_depmap = mo.ui.checkbox(value=True, label="query DepMap (network)")
    mo.hstack([pair_dropdown, run_depmap], justify="start")
    return pair_dropdown, run_depmap


@app.cell
def _(mo, pair_dropdown, run_depmap):
    import pandas as pd

    _pair = pair_dropdown.value
    _sym, _ik = _pair["symbol"], _pair["inchikey"]

    _records = morphology_stream(_ik, _sym)
    if run_depmap.value:
        _records += depmap_stream(_sym, _pair["compound_name"], _pair["prism_label"])
    else:
        _records += [evidence_record("depmap", "prism_chronos_corr", None, False, "skipped (toggle off)")]
    _records += structure_stream(_ik, _sym)

    _baseline = motive_baseline(_ik, _sym)
    _call = provisional_call(_records)

    # --- evidence table ---
    def _fmt(v):
        if isinstance(v, dict):
            if "r" in v:
                return f"r={v['r']:+.3f}, n={v['n_lines']}"
            if "inhibitor_consistent" in v:
                return f"ORF {v['orf']:+.2f} / CRISPR {v['crispr']:+.2f}"
            if "pan_essential" in v and "mean" in v:
                return f"mean {v['mean']:+.2f}, std {v['std']:.2f}"
            if "n_wells" in v:
                return f"{v['n_wells']}w / {'+'.join(v['sources'])}"
            return str(v)
        if isinstance(v, list):
            return ", ".join(f"{s} {x:+.2f}" for s, x in v[:3])
        if isinstance(v, float):
            return f"{v:+.3f}"
        return "-" if v is None else str(v)

    _tbl = pd.DataFrame(
        [
            {
                "stream": r["source"],
                "signal": r["signal"],
                "value": _fmt(r["value"]),
                "leaky": "yes" if r["leaky"] else "no",
                "summary": r["summary"],
            }
            for r in _records
        ]
    )

    _gnn = "GNN baseline unavailable"
    if _baseline:
        _parts = []
        if "gene_ap" in _baseline:
            _parts.append(f"gene AP **{_baseline['gene_ap']:.3f}** (deg {_baseline['gene_npos']})")
        if "compound_ap" in _baseline:
            _parts.append(f"compound AP **{_baseline['compound_ap']:.3f}** (deg {_baseline['compound_npos']})")
        _gnn = "MOTIVE one-shot GNN: " + "; ".join(_parts)

    _grounds_md = "\n".join(f"- {g}" for g in _call["grounds"])
    _header = mo.md(
        f"""
        ### {pair_dropdown.selected_key}

        **One-shot baseline (what we contrast against):** {_gnn}

        **Agentic provisional call:** **{_call["call"]}** (confidence: {_call["confidence"]},
        score {_call["score"]:.1f}/3). Grounds (predictive evidence only - this is what a
        judge reads):

        {_grounds_md}
        """
    )
    mo.vstack([_header, mo.ui.table(_tbl, selection=None, pagination=False)])
    return


@app.cell
def _(pair_dropdown, run_depmap):
    import matplotlib.pyplot as plt

    _pair = pair_dropdown.value
    _sym, _ik = _pair["symbol"], _pair["inchikey"]
    _records = morphology_stream(_ik, _sym)
    _dm = None
    if run_depmap.value:
        _dmrec = depmap_stream(_sym, _pair["compound_name"], _pair["prism_label"])
        for _rec in _dmrec:
            _v = _rec["value"]
            if isinstance(_v, dict) and "g" in _v:
                _dm = _v
                break

    _by = {(r["source"], r["signal"]): r["value"] for r in _records}
    _cos_orf = _by.get(("morphology", "cosine_orf"))
    _cos_crispr = _by.get(("morphology", "cosine_crispr"))

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=120)

    # left: directional cosines
    _labels, _vals = [], []
    if _cos_orf is not None:
        _labels.append("vs ORF\n(over-expr)")
        _vals.append(_cos_orf)
    if _cos_crispr is not None:
        _labels.append("vs CRISPR\n(knockout)")
        _vals.append(_cos_crispr)
    _ax = _axes[0]
    if _vals:
        _colors = ["#c0392b" if v < 0 else "#27ae60" for v in _vals]
        _ax.barh(_labels, _vals, color=_colors)
        _ax.axvline(0, color="k", lw=0.8)
        _ax.set_xlabel("cosine (shared ALL/v1.0b space)")
        _ax.set_title("Directional morphology connectivity")
    else:
        _ax.text(0.5, 0.5, "no gene profile", ha="center")
        _ax.set_axis_off()

    # right: PRISM vs Chronos scatter
    _ax2 = _axes[1]
    if _dm is not None:
        _ax2.scatter(_dm["g"], _dm["d"], s=8, alpha=0.5, color="#2c3e50")
        _ax2.set_xlabel(f"Chronos {_sym} dependency")
        _ax2.set_ylabel(f"PRISM '{_dm['prism_label']}' sensitivity")
        _ax2.set_title(f"DepMap (r={_dm['r']:+.2f}, n={_dm['n_lines']})")
    else:
        _ax2.text(0.5, 0.5, "DepMap not fired", ha="center")
        _ax2.set_axis_off()

    _fig.suptitle(f"{_sym} / {_pair['compound_name']}", fontsize=11)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Synthesis - what the judge reads, and why this answers the one-shot critique

    The portfolio above is the object. What makes it the turn from **one-shot inference** (map
    a pair to a scalar in a single non-interactive step) to an **agentic loop** (fetch and
    reason over evidence per pair, working back from the functional endpoint):

    - **Decomposability beats a green checkmark.** On BRD4/JQ1 the call is "not supported," yet
      the loop says *why*: the compound IS morphologically active (high activity proxy) but its
      profile is source-7-only, and the gene is pan-essential so DepMap has nothing to
      deconvolve. "Active compound + pan-essential gene + cross-source caveat = the evidence is
      genuinely thin, and here is which part" is a stronger argument than a confident scalar.
      The GNN's 0.668 can express none of it.
    - **Evidence is externalized and sourced.** Each row names its `source` and whether it is
      `leaky` (label-bearing prior knowledge) or genuinely predictive. The call lists its
      grounds. A reader sees what it rests on - and where it is thin.
    - **The leakage audit is free.** The predictive vs prior-knowledge split is visible without
      separate instrumentation. The predictive streams (morphology, DepMap) never read the
      MOTIVE label key; that file is used only to *identify* the demo pairs.
    - **The loop fetches per pair.** DepMap fires when the compound is in PRISM and the gene in
      Chronos; morphology is the backbone that always covers. On GNN-missed sparse genes, the
      loop assembles orthogonal evidence the single forward pass had no graph structure for.

    **Win condition (restated):** that a rich, auditable per-pair analysis *can be produced and
    reads clearly, including where the evidence is weak* - not a higher AP. Two pairs done well
    beat twenty done thinly.
    """)
    return


if __name__ == "__main__":
    app.run()
