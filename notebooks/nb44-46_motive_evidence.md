# MOTIVE per-pair evidence notebooks (nb44 / nb45 / nb46)

Three notebooks, one idea, three levels of condensation.
They all answer the same question for one compound-gene pair - *does this drug actually engage this target?* - and they all do it the same way underneath.
What differs is how much narrative and interactivity sits on top, depending on whether you are reading the source, giving a talk, or just want the answer.

## The idea

MOTIVE (Arevalo et al., NeurIPS 2024) scores a compound-gene interaction in one shot: a graph neural network maps `(gene, compound) -> a single scalar`.
These notebooks do the other thing.
They treat the same question as an **agentic evidence loop**: for one pair, fetch several independent, non-leaky signals, lay out what each one says, and emit a transparent verdict with its grounds attached.

The worked pair is **BRD4 / (+)-JQ1**, the famous BET-bromodomain interaction.
It is a deliberate **auditable null**: with these simple signals the verdict comes back "not supported," and the loop names exactly why the evidence is thin (JQ1's JUMP profile is source-7-only; BRD4 is common-essential so DepMap has little differential dependency to deconvolve; JQ1 is genuinely morphologically active, so the weak connectivity is real, not a dead profile).
The point is **decomposability**: the one-shot GNN emits a scalar (compound AP 0.668) that accounts for none of this, while the loop shows which part of the evidence is weak and why.
A higher score is explicitly *not* the goal.

The evidence streams (all `@app.function` helpers, identical across the three files):

- **morphology** (predictive, non-leaky) - directional cosine of the compound's Cell Painting profile against the gene's ORF and CRISPR profiles, sign pattern, percentile rank vs all genes, nearest neighbours, and a real `compound_activity()` query (copairs `activity_results`).
- **depmap** (predictive, orthogonal) - does the compound's PRISM drug-sensitivity profile track the gene's Chronos knockout-dependency profile across cell lines? Live Breadbox REST, fires when available.
- **motive_baseline** - the one-shot GNN's per-node AP, read live from the MOTIVE run, shown only as the contrast (not evidence).
- **provisional_call** - a transparent stand-in for the judge: a verdict + confidence + the explicit grounds it rests on, predictive evidence only.

The `leaky` column on every record keeps the leakage audit free: the MOTIVE label key is used only to *identify* the pair, never as evidence.

## What is what

| Notebook | Role | Pairs | Narrative | Interactivity | Run with |
|---|---|---|---|---|---|
| `nb44_ss_motive_interaction_evidence.py` | **Full - source of truth** | 3 (dropdown) | Full build brief + synthesis | dropdown + DepMap checkbox | `marimo edit` |
| `nb45_ss_motive_evidence_slide.py` | **Linear talk artifact** | 1 (BRD4/JQ1) | Short intro + synthesis | none (linear) | `marimo run` |
| `nb46_ss_motive_evidence_answer.py` | **Bare answer** | 1 (BRD4/JQ1) | Title + one framing line | none (linear) | `marimo run` |

**nb44 - the full version.**
This is where the work lives and is documented.
The first cell is a complete build brief (the prompt the notebook was composed from): what MOTIVE is, why the project exists, the leakage rules, the demo pairs, and a "reality check" recording the live numbers.
Below it are all the `@app.function` stream helpers, a `structure_stream` stub (chemical-similarity, not yet wired - needs RDKit under pixi), the live GNN baseline, an evidence table + a 2-panel figure, and a closing synthesis.
A dropdown switches between three pairs (BRD4/JQ1 plus two GNN-missed contrast pairs), and a checkbox toggles the DepMap network call.
Read or edit this one when you want to understand or change the logic.

**nb45 - the slide.**
A single-pair (BRD4/JQ1) linear cut for a talk.
No dropdown, no checkbox - it runs straight through, DepMap always queried, the pair hardcoded.
Keeps a short narrative intro and a "why this answers the one-shot critique" synthesis so the slide reads on its own.
`marimo run` shows output-only (code hidden), which is what you screenshot or present.

**nb46 - the answer.**
The most condensed.
Same single pair, but the narrative is stripped to a title heading ("Does JQ1 interact with BRD4?") plus one framing line.
The body is just the verdict in plain prose, its grounds, the evidence table, and the figure - the view you get if an agent simply answered the question and showed its work.
The GNN scalar is demoted to a one-line "for reference" footnote.

## Relationship and maintenance

nb44 is the **source of truth**.
The three files **copy** the same `@app.function` helpers rather than importing them (each is a self-contained, sandbox-runnable notebook with its own PEP 723 header), so a change to a stream's logic must be made in nb44 and then propagated to nb45 and nb46 by hand.
nb45 and nb46 only differ from nb44 in the demo-pair selection (fixed vs dropdown) and the amount of narrative - the compute is the same.

If you edit a stream, re-validate all three with `uvx marimo@latest export html --sandbox notebooks/<file>` and `pixi run ruff check/format notebooks/`.

## Data it needs (none of it ships in-repo)

- `data/raw/profiles/all_modalities.parquet` - JUMP well-level profiles in the shared ALL/v1.0b feature space (614 features).
- `data/external/jump_metadata.duckdb` - InChIKey/Symbol -> JCP2022 joins.
- `data/processed/copairs_results.duckdb` - the `activity_results` table for `compound_activity()`.
- the MOTIVE run on disk (`2024_Arevalo_NeurIPS_MotiVE/.../gin/2c5c7a/`) for the live GNN baseline.
- a network call to DepMap Breadbox (`https://depmap.org/portal/breadbox`, needs a `User-Agent` header or it 403s).

Because of these dependencies the notebooks cannot re-run in a wasm/molab sandbox; what molab can show is the saved cell outputs.
Those live as committed marimo session snapshots under `notebooks/__marimo__/session/` (that directory is otherwise gitignored; the snapshots are force-added).

## Key live numbers (BRD4 / JQ1, for sanity-checking a run)

- GNN per-node AP: compound (JQ1) 0.668, gene (BRD4) 0.341.
- Morphology: cosine vs ORF +0.035, vs CRISPR +0.072 (sign NOT inhibitor-consistent); BRD4 at ~49.6th connectivity percentile.
- Activity: JQ1 mAP 1.00, corrected p 0.0001 (morphologically active).
- DepMap: BRD4 Chronos mean ~-1.01 (common-essential); PRISM/Chronos Pearson r ~+0.07 across ~383 lines (weak, as expected).
- Verdict: "not supported by predictive evidence," confidence low, score 0.0/3.
