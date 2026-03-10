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


def best_graph_node(cycle: dict[str, Any], depth: int, token: int) -> dict[str, Any]:
    graph = cycle.get("proposal_graph") or []
    if depth >= len(graph):
        return {}
    best: dict[str, Any] = {}
    best_cum = -1.0
    for node in graph[depth] or []:
        if int(node.get("token", -1)) != token:
            continue
        cum = float(node.get("cum_prob", 0.0))
        if cum > best_cum:
            best = node
            best_cum = cum
    return best


def greedy_path(cycle: dict[str, Any]) -> list[dict[str, Any]]:
    paths = cycle.get("proposal_paths") or []
    if not paths:
        return []
    path = paths[0] or []
    result: list[dict[str, Any]] = []
    for depth, token in enumerate(path):
        node = best_graph_node(cycle, depth, int(token))
        result.append(
            {
                "token": int(token),
                "text_escaped": node.get("text_escaped", ""),
                "prob": float(node.get("prob", 0.0)),
                "cum_prob": float(node.get("cum_prob", 0.0)),
            }
        )
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


def render_trace_matrix(label: str, trace: dict[str, Any]) -> str:
    parts: list[str] = [f"<h2>{html.escape(label)}</h2>"]
    for idx, cycle in enumerate(trace.get("cycles") or []):
        greedy = greedy_path(cycle)
        target = target_tokens(cycle)
        parts.append(
            f"<section><h3>Cycle {idx + 1}</h3>"
            f"<p>accepted={int(cycle.get('accepted_count', 0))} "
            f"proposals={int(cycle.get('proposal_count', 0))}</p>"
        )
        if not greedy:
            parts.append("<p>No proposal paths.</p></section>")
            continue
        parts.append("<table><tr><th>Target \\ Draft</th>")
        for j, draft in enumerate(greedy):
            parts.append(
                "<th>"
                f"d{j}<br>{draft['token']}<br>{html.escape(str(draft.get('text_escaped', '')))}"
                f"<br>p={draft['prob']:.4g}"
                "</th>"
            )
        parts.append("</tr>")
        for i, tgt in enumerate(target):
            parts.append(
                "<tr><th>"
                f"t{i}<br>{tgt['token']}<br>{html.escape(str(tgt.get('text_escaped', '')))}"
                f"<br>p={tgt['prob']:.4g}"
                "</th>"
            )
            for draft in greedy:
                cls = "match" if draft["token"] == tgt["token"] else "miss"
                parts.append(
                    f"<td class=\"{cls}\">"
                    f"{draft['token']}<br>{html.escape(str(draft.get('text_escaped', '')))}"
                    f"<br>p={draft['prob']:.4g}<br>cp={draft['cum_prob']:.4g}"
                    "</td>"
                )
            parts.append("</tr>")
        parts.append("</table></section>")
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

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>EAGLE Trace Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #bbb; padding: 6px; vertical-align: top; }}
    td.match {{ background: #dff2d8; }}
    td.miss {{ background: #f8d7da; }}
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
  {render_trace_matrix(label_a, trace_a)}
  {render_trace_matrix(label_b, trace_b)}
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
