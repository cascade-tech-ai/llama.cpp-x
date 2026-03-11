#!/usr/bin/env python3

import argparse
import html
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: python3 -m pip install pyyaml") from exc


def load_trace(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def proposal_tree(cycle: dict[str, Any]) -> list[dict[str, Any]]:
    return list(cycle.get("proposal_tree") or [])


def greedy_path(cycle: dict[str, Any]) -> list[dict[str, Any]]:
    tree = proposal_tree(cycle)
    if not tree:
        return []
    result: list[dict[str, Any]] = []
    node = tree[0]
    while node:
        result.append(
            {
                "token": int(node.get("token", -1)),
                "text_escaped": node.get("text_escaped", ""),
                "prob": float(node.get("prob", 0.0)),
                "cum_prob": float(node.get("cum_prob", 0.0)),
                "selected": bool(node.get("selected", False)),
                "accepted": bool(node.get("accepted", False)),
            }
        )
        children = list(node.get("children") or [])
        node = children[0] if children else None
    return result


def target_tokens(cycle: dict[str, Any]) -> list[dict[str, Any]]:
    appended = cycle.get("appended_tokens") or []
    passes = cycle.get("passes") or []
    result: list[dict[str, Any]] = []
    for i, token in enumerate(appended):
        p = passes[i] if i < len(passes) else {}
        result.append(
            {
                "token": int(token),
                "text_escaped": p.get("sampled_text_escaped", ""),
                "prob": float(p.get("sampled_target_prob", 0.0)),
            }
        )
    return result


def token_cell_html(token: dict[str, Any] | None, *, extra: str = "") -> str:
    if not token:
        return "<div class=\"token empty\">-</div>"
    text = html.escape(str(token.get("text_escaped", "")))
    prob = token.get("prob")
    prob_html = ""
    if prob is not None:
        prob_html = f"<div class=\"prob\">p={float(prob):.4g}</div>"
    extra_html = f"<div class=\"extra\">{extra}</div>" if extra else ""
    return (
        "<div class=\"token\">"
        f"<div class=\"tokid\">{int(token['token'])}</div>"
        f"<div class=\"toktext\">{text}</div>"
        f"{prob_html}"
        f"{extra_html}"
        "</div>"
    )


def tree_lookup_path(
    roots: list[dict[str, Any]],
    tokens: list[int],
) -> list[dict[str, Any] | None]:
    out: list[dict[str, Any] | None] = []
    level = roots
    for token in tokens:
        node = next((child for child in level if int(child.get("token", -1)) == int(token)), None)
        out.append(node)
        if node is None:
            level = []
            continue
        level = list(node.get("children") or [])
    return out


def render_graph_foldout(label: str, cycle: dict[str, Any], details_id: str) -> str:
    tree = proposal_tree(cycle)
    if not tree:
        return (
            f"<details id=\"{details_id}\" class=\"graph-details\">"
            f"<summary>{html.escape(label)} graph</summary>"
            "<p>No proposal tree.</p></details>"
        )

    rows: list[tuple[int, list[dict[str, Any]]]] = []

    def walk(node: dict[str, Any], depth: int) -> None:
        chain: list[dict[str, Any]] = []
        cur = node
        while cur:
            chain.append(cur)
            children = list(cur.get("children") or [])
            cur = children[0] if children else None
        rows.append((depth, chain))

        cur = node
        cur_depth = depth
        while cur:
            children = list(cur.get("children") or [])
            for alt in children[1:]:
                walk(alt, cur_depth + 1)
            cur = children[0] if children else None
            cur_depth += 1

    for root in tree:
        walk(root, 0)

    max_cols = 0
    for start_depth, chain in rows:
        max_cols = max(max_cols, start_depth + len(chain))

    parts = [
        f"<details id=\"{details_id}\" class=\"graph-details\">",
        f"<summary>{html.escape(label)} graph</summary>",
        "<table class=\"graph-table\"><tr>",
    ]
    for depth in range(max_cols):
        parts.append(f"<th>d{depth}</th>")
    parts.append("</tr>")

    for row_idx, (start_depth, chain) in enumerate(rows):
        parts.append("<tr>")
        for depth in range(max_cols):
            rel = depth - start_depth
            if rel < 0 or rel >= len(chain):
                parts.append("<td class=\"graph-empty\">&nbsp;</td>")
                continue
            node = chain[rel]
            cls = "graph-greedy" if row_idx == 0 else "graph-node"
            if node.get("selected"):
                cls += " graph-selected"
            extra = f"cp={float(node.get('cum_prob', 0.0)):.4g}"
            if node.get("selected"):
                extra += " selected"
            if node.get("accepted"):
                extra += " accepted"
            parts.append(
                f"<td class=\"{cls}\">"
                f"{token_cell_html(node, extra=extra)}"
                "</td>"
            )
        parts.append("</tr>")
    parts.append("</table></details>")
    return "".join(parts)


def flatten_target_stream(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
    stream: list[dict[str, Any]] = []
    cycle_offsets: list[int] = []
    for cycle in trace.get("cycles") or []:
        cycle_offsets.append(len(stream))
        stream.extend(target_tokens(cycle))
    return stream, cycle_offsets


def shared_prefix_len(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i]["token"] != b[i]["token"]:
            return i
    return n


def compare_greedy(trace_a: dict[str, Any], trace_b: dict[str, Any]) -> dict[str, Any]:
    cycles_a = trace_a.get("cycles") or []
    cycles_b = trace_b.get("cycles") or []
    n = min(len(cycles_a), len(cycles_b))
    per_cycle = []
    exact = 0
    total_prefix = 0
    total_tokens = 0
    total_matching_tokens = 0
    first_mismatch = None

    for idx in range(n):
        path_a = greedy_path(cycles_a[idx])
        path_b = greedy_path(cycles_b[idx])
        prefix = shared_prefix_len(path_a, path_b)
        exact_match = len(path_a) == len(path_b) and prefix == len(path_a)
        exact += 1 if exact_match else 0
        total_prefix += prefix
        total_tokens += max(len(path_a), len(path_b))
        total_matching_tokens += prefix
        row = {
            "cycle": idx + 1,
            "len_a": len(path_a),
            "len_b": len(path_b),
            "prefix": prefix,
            "exact": exact_match,
        }
        per_cycle.append(row)
        if first_mismatch is None and not exact_match:
            mismatch_idx = prefix
            first_mismatch = {
                "cycle": idx + 1,
                "depth": mismatch_idx,
                "a": path_a[mismatch_idx] if mismatch_idx < len(path_a) else None,
                "b": path_b[mismatch_idx] if mismatch_idx < len(path_b) else None,
            }

    return {
        "cycles_compared": n,
        "exact_cycles": exact,
        "avg_prefix": (total_prefix / n) if n else 0.0,
        "token_match_rate": (total_matching_tokens / total_tokens) if total_tokens else 1.0,
        "first_mismatch": first_mismatch,
        "per_cycle": per_cycle,
    }


def render_cycle_pair_table(
    cycle_idx: int,
    truth_stream: list[dict[str, Any]],
    truth_offset: int,
    label_a: str,
    cycle_a: dict[str, Any] | None,
    label_b: str,
    cycle_b: dict[str, Any] | None,
) -> str:
    truth = truth_stream[truth_offset : truth_offset + 8]
    cols = 8
    truth_tokens = [int(tok["token"]) for tok in truth if tok]
    lookup_a = tree_lookup_path(proposal_tree(cycle_a or {}), truth_tokens)
    lookup_b = tree_lookup_path(proposal_tree(cycle_b or {}), truth_tokens)

    parts: list[str] = [
        f"<section><h3>Cycle {cycle_idx + 1}</h3>",
        "<table class=\"cycle-table\">",
        "<tr><th class=\"row-label\">trace</th>",
    ]
    parts.append("<th class=\"target-label\">t0</th>")
    for j in range(1, cols):
        parts.append(f"<th class=\"draft-col\">d{j}</th>")
    parts.append("</tr>")

    def add_truth_row() -> None:
        parts.append("<tr class=\"trace-truth\">")
        parts.append("<th class=\"row-label\"><div class=\"trace-name\">target</div></th>")
        for j in range(cols):
            token = truth[j] if j < len(truth) else None
            cls = "anchor" if j == 0 else "truth"
            parts.append(f"<td class=\"{cls}\">{token_cell_html(token)}</td>")
        parts.append("</tr>")

    def add_status_row(
        label: str,
        row_class: str,
        cycle: dict[str, Any] | None,
        looked_up: list[dict[str, Any] | None],
    ) -> None:
        accepted = int((cycle or {}).get("accepted_count", 0))
        proposals = int((cycle or {}).get("proposal_count", 0))
        parts.append(
            f"<tr class=\"{row_class}\">"
            f"<th class=\"row-label\"><div class=\"trace-name\">{html.escape(label)}</div>"
            f"<div class=\"trace-meta\">accepted={accepted}<br>proposals={proposals}</div></th>"
        )
        for j in range(cols):
            expect = truth[j] if j < len(truth) else None
            node = looked_up[j] if j < len(looked_up) else None
            if expect is None:
                cls = "beyond"
                extra = "no future target"
                cell = None
            elif node is None:
                cls = "missing"
                extra = "missing from tree"
                cell = expect
            else:
                cls = "selected" if bool(node.get("selected")) else "present"
                extra = f"tree p={float(node.get('prob', 0.0)):.4g} cp={float(node.get('cum_prob', 0.0)):.4g}"
                cell = {
                    "token": int(expect["token"]),
                    "text_escaped": expect.get("text_escaped", ""),
                    "prob": expect.get("prob"),
                }
            if j == 0 and cls == "missing":
                cls = "missing anchor"
            elif j == 0:
                cls += " anchor"
            parts.append(f"<td class=\"{cls}\">{token_cell_html(cell, extra=extra)}</td>")
        parts.append("</tr>")

    add_truth_row()
    add_status_row(label_a, "trace-a", cycle_a, lookup_a)
    add_status_row(label_b, "trace-b", cycle_b, lookup_b)
    parts.append("</table>")
    parts.append("<div class=\"graph-foldouts\">")
    parts.append(render_graph_foldout(label_a, cycle_a or {}, f"graph-{cycle_idx}-{label_a}"))
    parts.append(render_graph_foldout(label_b, cycle_b or {}, f"graph-{cycle_idx}-{label_b}"))
    parts.append("</div></section>")
    return "".join(parts)


def write_report(path: Path, label_a: str, trace_a: dict[str, Any], label_b: str, trace_b: dict[str, Any], stats: dict[str, Any]) -> None:
    first = stats.get("first_mismatch")
    mismatch_html = "<p>No mismatch.</p>"
    if first:
        mismatch_html = (
            f"<p>First mismatch: cycle {first['cycle']} depth {first['depth']}"
            f"<br>{html.escape(label_a)}: {html.escape(str(first['a']))}"
            f"<br>{html.escape(label_b)}: {html.escape(str(first['b']))}</p>"
        )

    cycles_a = trace_a.get("cycles") or []
    cycles_b = trace_b.get("cycles") or []
    n = min(len(cycles_a), len(cycles_b))
    truth_stream, truth_offsets = flatten_target_stream(trace_a)
    cycle_html = "".join(
        render_cycle_pair_table(i, truth_stream, truth_offsets[i], label_a, cycles_a[i], label_b, cycles_b[i])
        for i in range(n)
    )

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>EAGLE Trace Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; margin-bottom: 24px; width: 100%; }}
    th, td {{ border: 1px solid #bbb; padding: 6px; vertical-align: top; }}
    th {{ background: #f4f4f4; }}
    .cycle-table .row-label {{ width: 150px; }}
    .cycle-table .target-label {{ width: 220px; }}
    .draft-col {{ text-align: center; min-width: 150px; }}
    td.anchor {{ background: #eef3f8; }}
    td.truth {{ background: #f7f7f7; }}
    td.selected {{ background: #dff2d8; }}
    td.present {{ background: #f8d7da; }}
    td.missing {{ background: #8b1e1e; color: #fff; }}
    td.missing .prob, td.missing .extra, td.missing .tokid {{ color: #f9d8d8; }}
    td.beyond {{ background: #f2f2f2; color: #666; }}
    td.empty {{ background: #fafafa; color: #999; }}
    tr.trace-truth > th.row-label {{ background: #ececec; }}
    tr.trace-a > th.row-label {{ background: #e8f1fb; }}
    tr.trace-b > th.row-label {{ background: #f5ead7; }}
    .graph-foldouts {{ display: flex; gap: 12px; margin-top: 8px; flex-wrap: wrap; }}
    .graph-details {{ min-width: 420px; }}
    .graph-table {{ width: auto; }}
    .graph-table th {{ text-align: center; min-width: 140px; }}
    .graph-greedy {{ background: #eaf6e4; }}
    .graph-node {{ background: #fafafa; }}
    .graph-selected {{ outline: 2px solid #2e7d32; outline-offset: -2px; }}
    .graph-empty {{ background: #fff; color: #aaa; text-align: center; }}
    .trace-name {{ font-weight: 700; }}
    .trace-meta {{ font-weight: 400; font-size: 12px; color: #555; margin-top: 6px; }}
    .token .tokid {{ font-weight: 700; font-family: monospace; }}
    .token .toktext {{ margin-top: 4px; white-space: pre-wrap; word-break: break-word; }}
    .token .prob, .token .extra {{ margin-top: 4px; font-size: 12px; color: #555; white-space: pre-wrap; word-break: break-word; }}
    section {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>EAGLE Trace Report</h1>
  <p>Cycles compared: {stats['cycles_compared']}<br>
     Exact greedy matches: {stats['exact_cycles']}<br>
     Average shared greedy prefix: {stats['avg_prefix']:.3f}<br>
     Greedy token match rate: {stats['token_match_rate']:.3%}</p>
  {mismatch_html}
  {cycle_html}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare EAGLE rollout traces")
    ap.add_argument("trace_a")
    ap.add_argument("trace_b")
    ap.add_argument("--mode", default="greedy", choices=["greedy"])
    ap.add_argument("--report", help="Write HTML report")
    ap.add_argument("--label-a", default="trace_a")
    ap.add_argument("--label-b", default="trace_b")
    args = ap.parse_args()

    trace_a = load_trace(Path(args.trace_a))
    trace_b = load_trace(Path(args.trace_b))

    stats = compare_greedy(trace_a, trace_b)
    print(f"cycles_compared={stats['cycles_compared']}")
    print(f"exact_cycles={stats['exact_cycles']}")
    print(f"avg_prefix={stats['avg_prefix']:.3f}")
    print(f"token_match_rate={stats['token_match_rate']:.3%}")
    if stats["first_mismatch"]:
        first = stats["first_mismatch"]
        print(f"first_mismatch_cycle={first['cycle']} depth={first['depth']}")
        print(f"{args.label_a}={first['a']}")
        print(f"{args.label_b}={first['b']}")

    if args.report:
        write_report(Path(args.report), args.label_a, trace_a, args.label_b, trace_b, stats)
        print(f"wrote_report={args.report}")


if __name__ == "__main__":
    main()
