# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb",
#     "pandas",
#     "numpy",
#     "matplotlib",
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

    from nb00_ss_config import COPAIRS_RESULTS_DB, EXTERNAL_DATA_DIR, RAW_DATA_DIR

    # --- jpx (THIS repo) inputs ---
    PROFILES_PARQUET = RAW_DATA_DIR / "profiles" / "all_modalities.parquet"
    METADATA_BASE_DB = EXTERNAL_DATA_DIR / "jump_metadata.duckdb"  # base, canonical

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

    # The single worked pair. (Narrated versions: nb45_ss_motive_evidence_slide.py /
    # nb44_ss_motive_interaction_evidence.py.)
    PAIR = {
        "symbol": "BRD4",
        "inchikey": "DNVXATUJJDPFDM-KRWDZBQOSA-N",
        "compound_name": "JQ1",
        "prism_label": "JQ1-(+)",
    }


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        f"""
        # Does {PAIR["compound_name"]} interact with {PAIR["symbol"]}?

        An agentic evidence check: instead of one model score, the loop fetches
        predictive, non-leaky signals for this pair and lays out what each one says.
        """
    )
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
def compound_activity(jcp):
    """Phenotypic activity of the compound: is its morphology reproducible / distinct?

    From copairs activity_results (compound_with_source7 / activity_no_target2 /
    all_sources). Non-leaky: activity is a compound-only property, not the compound-gene
    label. Returns None (gracefully) if the DB or row is absent.
    """
    if jcp is None or not Path(COPAIRS_RESULTS_DB).exists():
        return None
    try:
        con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
        df = con.execute(
            "SELECT mean_average_precision AS map, corrected_p_value AS p, below_corrected_p AS active "
            "FROM activity_results "
            "WHERE Metadata_JCP2022 = ? AND _dataset = 'compound_with_source7' "
            "AND _preprocessing = 'activity_no_target2' AND _filter = 'all_sources' LIMIT 1",
            [jcp],
        ).df()
        con.close()
    except Exception:  # noqa: BLE001 - activity is best-effort context
        return None
    if df.empty:
        return None
    row = df.iloc[0]
    return {"map": float(row["map"]), "p": float(row["p"]), "active": bool(row["active"])}


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

    # 0.5 Phenotypic activity (is the morphology real? rules out a dead-profile explanation)
    act = compound_activity(cpd_jcp)
    if act is not None:
        records.append(
            evidence_record(
                "morphology",
                "phenotypic_activity",
                {"map": act["map"], "p": act["p"], "active": act["active"]},
                False,
                f"compound is morphologically {'ACTIVE' if act['active'] else 'inactive'} "
                f"(activity mAP {act['map']:.2f}, corrected p {act['p']:.4f}) - "
                "so weak connectivity below is a real result, not a dead profile",
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

    # 4. Activity proxy for the gene perturbations (consensus L2 norm vs all genes)
    gene_norms = np.linalg.norm(mat, axis=1)
    for label, vec in (("gene_orf", orf), ("gene_crispr", crispr)):
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

    # Chronos dependency distribution. A common-essential gene (mean dependency ~ -1, essential
    # in nearly every line) offers little differential dependency to track drug sensitivity
    # against. This record EXPLAINS a weak correlation rather than hiding it (legibility payload).
    gvals = np.array([v for v in gene.values() if v is not None], dtype=float)
    gvals = gvals[np.isfinite(gvals)]
    g_mean, g_std = float(gvals.mean()), float(gvals.std())
    common_essential = g_mean < -0.5
    records.append(
        evidence_record(
            "depmap",
            "chronos_profile",
            {"mean": g_mean, "std": g_std, "common_essential": common_essential},
            False,
            f"{symbol} Chronos dependency: mean {g_mean:+.2f}, std {g_std:.2f} "
            f"across {gvals.size} lines"
            + (
                "  [common-essential - essential across nearly all lines; little differential dependency to grip]"
                if common_essential
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
                "common_essential": common_essential,
                "g": g.tolist(),
                "d": d.tolist(),
            },
            False,
            f"Pearson r={r:+.3f} across {len(g)} cell lines "
            f"(PRISM '{used}' sensitivity vs Chronos {symbol} dependency)"
            + (
                "  - weak, expected: common-essential gene (little differential dependency)" if common_essential else ""
            ),
        )
    )
    return records


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
        elif dm.get("common_essential"):
            grounds.append(
                f"DepMap correlation weak (r={dm['r']:+.2f}) - gene is common-essential, "
                "so this assay has little differential signal to deconvolve (not evidence against the pair)"
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


@app.cell
def _():
    import pandas as pd

    _sym, _ik = PAIR["symbol"], PAIR["inchikey"]

    records = morphology_stream(_ik, _sym)
    records += depmap_stream(_sym, PAIR["compound_name"], PAIR["prism_label"])

    baseline = motive_baseline(_ik, _sym)
    call = provisional_call(records)

    # cosines + DepMap scatter for the figure (computed once, here)
    _by = {(r["source"], r["signal"]): r["value"] for r in records}
    cos_orf = _by.get(("morphology", "cosine_orf"))
    cos_crispr = _by.get(("morphology", "cosine_crispr"))
    dm_scatter = _by.get(("depmap", "prism_chronos_corr"))
    if not isinstance(dm_scatter, dict) or "g" not in dm_scatter:
        dm_scatter = None
    return baseline, call, cos_crispr, cos_orf, dm_scatter, pd, records


@app.cell(hide_code=True)
def _(baseline, call, mo, pd, records):
    def _fmt(v):
        if isinstance(v, dict):
            if "r" in v:
                return f"r={v['r']:+.3f}, n={v['n_lines']}"
            if "inhibitor_consistent" in v:
                return f"ORF {v['orf']:+.2f} / CRISPR {v['crispr']:+.2f}"
            if "common_essential" in v and "mean" in v:
                return f"mean {v['mean']:+.2f}, std {v['std']:.2f}"
            if "map" in v and "active" in v:
                return f"mAP {v['map']:.2f}, p {v['p']:.4f}"
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
            for r in records
        ]
    )

    _gnn = "n/a"
    if baseline:
        _parts = []
        if "compound_ap" in baseline:
            _parts.append(f"compound AP {baseline['compound_ap']:.3f} (deg {baseline['compound_npos']})")
        if "gene_ap" in baseline:
            _parts.append(f"gene AP {baseline['gene_ap']:.3f} (deg {baseline['gene_npos']})")
        _gnn = "; ".join(_parts)

    _grounds_md = "\n".join(f"- {g}" for g in call["grounds"])
    _answer = mo.md(
        f"""
        ## Verdict: {call["call"]}

        Confidence {call["confidence"]} (score {call["score"]:.1f} of 3). Here is what each
        predictive evidence stream says:

        {_grounds_md}

        For reference, the one-shot MOTIVE GNN gives only: {_gnn}.
        """
    )
    mo.vstack([_answer, mo.ui.table(_tbl, selection=None, pagination=False)])
    return


@app.cell(hide_code=True)
def _(cos_crispr, cos_orf, dm_scatter):
    import matplotlib.pyplot as plt

    _sym = PAIR["symbol"]
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=120)

    # left: directional cosines
    _labels, _vals = [], []
    if cos_orf is not None:
        _labels.append("vs ORF\n(over-expr)")
        _vals.append(cos_orf)
    if cos_crispr is not None:
        _labels.append("vs CRISPR\n(knockout)")
        _vals.append(cos_crispr)
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
    if dm_scatter is not None:
        _ax2.scatter(dm_scatter["g"], dm_scatter["d"], s=8, alpha=0.5, color="#2c3e50")
        _ax2.set_xlabel(f"Chronos {_sym} dependency")
        _ax2.set_ylabel(f"PRISM '{dm_scatter['prism_label']}' sensitivity")
        _ax2.set_title(f"DepMap (r={dm_scatter['r']:+.2f}, n={dm_scatter['n_lines']})")
    else:
        _ax2.text(0.5, 0.5, "DepMap not fired", ha="center")
        _ax2.set_axis_off()

    _fig.suptitle(f"{_sym} / {PAIR['compound_name']}", fontsize=11)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
