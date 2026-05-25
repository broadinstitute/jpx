import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import sqlite3
    from collections import Counter
    from pathlib import Path

    import marimo as mo
    import networkx as nx

    DB_PATH = Path(".redun/redun.db")
    return Counter, DB_PATH, Path, mo, nx, sqlite3


@app.cell(hide_code=True)
def load_db(DB_PATH, mo, sqlite3):
    conn = sqlite3.connect(str(DB_PATH))

    def query_table_counts():
        result = {}
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            n = r[0]
            c = conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]
            result[n] = c
        return result

    table_counts = query_table_counts()
    mo.md(f"## redun.db Overview\n\nLoaded `{DB_PATH}` with **{len(table_counts)} tables**")
    return conn, table_counts


@app.cell(hide_code=True)
def stats_table(mo, table_counts):
    mo.ui.table(
        [{"table": k, "rows": v} for k, v in sorted(table_counts.items(), key=lambda x: -x[1])],
        label="Table row counts",
    )
    return


@app.cell(hide_code=True)
def build_dag(conn, mo, nx):
    def build_task_dag():
        nodes_map = {}
        for r in conn.execute("SELECT call_hash, task_name FROM call_node"):
            nodes_map[r[0]] = r[1]

        raw_edges = []
        for r in conn.execute("SELECT parent_id, child_id FROM call_edge"):
            pt = nodes_map.get(r[0])
            ct = nodes_map.get(r[1])
            if pt and ct and pt != ct:
                raw_edges.append((pt, ct))

        def short(s):
            return s.split(".")[-1] if "." in s else s

        deduped = list(set((short(a), short(b)) for a, b in raw_edges))

        dag = nx.DiGraph()
        dag.add_edges_from(deduped)

        for r in conn.execute("SELECT DISTINCT name FROM task"):
            if r[0] not in dag:
                dag.add_node(r[0])
        return dag

    G = build_task_dag()
    mo.md(
        f"**DAG**: {G.number_of_nodes()} tasks, {G.number_of_edges()} edges, {nx.number_weakly_connected_components(G)} components"
    )
    return (G,)


@app.cell(hide_code=True)
def render_dag(G, mo, nx):
    def safe_id(nd):
        return nd.replace("_", "X")

    def classify(nd):
        if nd in ("main", "core", "explore", "srijit_all", "batch_source7"):
            return "entry"
        if nd.startswith("process_") or nd.startswith("compute_") or nd.startswith("fetch_"):
            return "processing"
        if "copairs" in nd or "activity" in nd or "consistency" in nd:
            return "copairs"
        if "umap" in nd or "profile" in nd or "chemical" in nd:
            return "profiles"
        if nd.startswith("explore_"):
            return "exploration"
        return "other"

    STYLES = """    classDef entry fill:#FF6B6B,stroke:#c0392b,color:#fff,font-weight:bold
        classDef processing fill:#4ECDC4,stroke:#16a085,color:#fff
        classDef copairs fill:#45B7D1,stroke:#2980b9,color:#fff
        classDef profiles fill:#96CEB4,stroke:#27ae60,color:#fff
        classDef exploration fill:#FFEAA7,stroke:#f39c12,color:#333
        classDef other fill:#DDA0DD,stroke:#8e44ad,color:#fff"""

    def subdag_for(root):
        descendants = nx.descendants(G, root) | {root}
        sub = G.subgraph(descendants)
        lines = ["flowchart TD", STYLES]
        for src, dst in sub.edges():
            lines.append(f"    {safe_id(src)}[{src}] --> {safe_id(dst)}[{dst}]")
        for nd in sub.nodes():
            if sub.in_degree(nd) == 0 and sub.out_degree(nd) == 0:
                lines.append(f"    {safe_id(nd)}[{nd}]")
        by_class = {}
        for nd in sub.nodes():
            cls = classify(nd)
            by_class.setdefault(cls, []).append(safe_id(nd))
        for cls, ids in by_class.items():
            lines.append(f"    class {','.join(ids)} {cls}")
        return "\n".join(lines)

    def overview_dag():
        lines = ["flowchart LR", STYLES]
        entry_points = ["main", "core", "explore", "srijit_all", "batch_source7"]
        shown = set()
        for ep in entry_points:
            if ep not in G:
                continue
            for child in G.successors(ep):
                lines.append(f"    {safe_id(ep)}[{ep}] --> {safe_id(child)}[{child}]")
                shown.add(ep)
                shown.add(child)
        by_class = {}
        for nd in shown:
            cls = classify(nd)
            by_class.setdefault(cls, []).append(safe_id(nd))
        for cls, ids in by_class.items():
            lines.append(f"    class {','.join(ids)} {cls}")
        return "\n".join(lines)

    entry_points = sorted([n for n in G.nodes() if G.in_degree(n) == 0])
    options = {"overview": "overview"}
    for ep in entry_points:
        options[ep] = ep

    view_selector = mo.ui.dropdown(
        options=options,
        value="overview",
        label="View",
    )
    view_selector
    return overview_dag, subdag_for, view_selector


@app.cell(hide_code=True)
def render_selected(G, mo, nx, overview_dag, subdag_for, view_selector):
    selected_view = view_selector.value

    if selected_view == "overview":
        diagram = overview_dag()
        title = "Top-level DAG: entry points and their direct children"
    else:
        diagram = subdag_for(selected_view)
        n_nodes = len(nx.descendants(G, selected_view)) + 1
        title = f"Subgraph rooted at **{selected_view}** ({n_nodes} tasks)"

    legend = mo.md("""
    | Color | Group |
    |-------|-------|
    | <span style="background:#FF6B6B;color:#fff;padding:2px 8px;border-radius:3px">\u2588\u2588</span> | Entry points (main, core, explore, srijit_all, batch_source7) |
    | <span style="background:#4ECDC4;color:#fff;padding:2px 8px;border-radius:3px">\u2588\u2588</span> | Data processing (process_\\*, compute_\\*, fetch_\\*) |
    | <span style="background:#45B7D1;color:#fff;padding:2px 8px;border-radius:3px">\u2588\u2588</span> | Copairs / analysis (copairs, activity, consistency) |
    | <span style="background:#96CEB4;color:#fff;padding:2px 8px;border-radius:3px">\u2588\u2588</span> | Profiles / chem space (umap, profile, chemical) |
    | <span style="background:#FFEAA7;color:#333;padding:2px 8px;border-radius:3px">\u2588\u2588</span> | Exploration (explore_\\*) |
    | <span style="background:#DDA0DD;color:#fff;padding:2px 8px;border-radius:3px">\u2588\u2588</span> | Other |
    """)

    mo.vstack(
        [
            mo.md(f"## {title}"),
            legend,
            mo.mermaid(diagram),
        ]
    )
    return


@app.cell(hide_code=True)
def exec_history(conn, mo):
    exec_data = conn.execute("""
        SELECT e.id, e.args,
               COUNT(j.id) as total_jobs,
               SUM(CASE WHEN j.cached = 1 THEN 1 ELSE 0 END) as cached_jobs,
               MIN(j.start_time) as started,
               MAX(j.end_time) as ended
        FROM execution e
        LEFT JOIN job j ON j.execution_id = e.id
        GROUP BY e.id
        ORDER BY started
    """).fetchall()

    exec_rows = [
        {
            "execution": r[0][:12] + "...",
            "command": r[1] or "",
            "jobs": r[2],
            "cached": r[3],
            "computed": r[2] - r[3],
            "started": r[4],
            "ended": r[5],
        }
        for r in exec_data
    ]

    mo.vstack(
        [
            mo.md("## Execution History"),
            mo.ui.table(exec_rows, label="All pipeline runs recorded in redun.db"),
        ]
    )
    return


@app.cell(hide_code=True)
def file_provenance(Counter, Path, conn, mo):
    def count_files_by_dir():
        dirs = Counter()
        for r in conn.execute("SELECT path FROM file"):
            p = Path(r[0])
            parts = p.parts
            try:
                idx = list(parts).index("data")
                key = "/".join(parts[idx : idx + 3])
            except (ValueError, IndexError):
                key = str(p.parent)
            dirs[key] += 1
        return dirs

    file_dir_counts = count_files_by_dir()

    mo.vstack(
        [
            mo.md(
                f"## Tracked Files\n\n**{sum(file_dir_counts.values())}** files across **{len(file_dir_counts)}** directories"
            ),
            mo.ui.table(
                [{"directory": d, "files": c} for d, c in file_dir_counts.most_common()],
                label="Files per directory",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
