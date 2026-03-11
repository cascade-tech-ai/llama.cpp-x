#!/usr/bin/env python3

import argparse
import html
import json
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any

from verify_trace import compare_greedy, greedy_path, load_trace, write_report


def run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def render_summary_html(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for row in summary["rows"]:
        report_rel = html.escape(row["report"])
        prompt_rel = html.escape(row["prompt_file"])
        rows.append(
            "<tr>"
            f"<td>{row['dataset_entry']}</td>"
            f"<td>{row['cycles_compared']}</td>"
            f"<td>{row['exact_cycles']}</td>"
            f"<td>{row['avg_prefix']:.3f}</td>"
            f"<td>{row['token_match_rate']:.3%}</td>"
            f"<td>{row['divergent_positions']}</td>"
            f"<td>{row['mean_prob_delta']:.6f}</td>"
            f"<td>{row['min_prob_delta']:.6f}</td>"
            f"<td>{row['max_prob_delta']:.6f}</td>"
            f"<td>{html.escape(row['first_mismatch_desc'])}</td>"
            f"<td><a href=\"{report_rel}\">report</a></td>"
            f"<td><a href=\"{prompt_rel}\">prompt</a></td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>EAGLE Trace Batch Compare</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #bbb; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f2f2f2; }}
  </style>
</head>
<body>
  <h1>EAGLE Trace Batch Compare</h1>
  <p>count={summary['count']}<br>
     divergent_positions={summary['divergent_positions_total']}<br>
     mean_prob_delta={summary['mean_prob_delta']:.6f}<br>
     min_prob_delta={summary['min_prob_delta']:.6f}<br>
     max_prob_delta={summary['max_prob_delta']:.6f}</p>
  <table>
    <tr>
      <th>entry</th>
      <th>cycles</th>
      <th>exact</th>
      <th>avg prefix</th>
      <th>match rate</th>
      <th>divergences</th>
      <th>mean dP</th>
      <th>min dP</th>
      <th>max dP</th>
      <th>first mismatch</th>
      <th>html</th>
      <th>prompt</th>
    </tr>
    {''.join(rows)}
  </table>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def describe_first_mismatch(stats: dict[str, Any]) -> str:
    first = stats.get("first_mismatch")
    if not first:
        return "none"
    a = first.get("a")
    b = first.get("b")
    return (
        f"cycle {first['cycle']} depth {first['depth']}: "
        f"{a['token'] if a else 'None'}->{b['token'] if b else 'None'}"
    )


def divergence_stats(trace_a: dict[str, Any], trace_b: dict[str, Any]) -> dict[str, Any]:
    cycles_a = trace_a.get("cycles") or []
    cycles_b = trace_b.get("cycles") or []
    n = min(len(cycles_a), len(cycles_b))
    deltas: list[float] = []
    divergent_positions = 0

    for idx in range(n):
        path_a = greedy_path(cycles_a[idx])
        path_b = greedy_path(cycles_b[idx])
        m = min(len(path_a), len(path_b))
        for depth in range(m):
            if path_a[depth]["token"] == path_b[depth]["token"]:
                continue
            divergent_positions += 1
            deltas.append(abs(float(path_a[depth]["prob"]) - float(path_b[depth]["prob"])))

    if not deltas:
        deltas = [0.0]

    return {
        "divergent_positions": divergent_positions,
        "mean_prob_delta": mean(deltas),
        "min_prob_delta": min(deltas),
        "max_prob_delta": max(deltas),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate and compare Kestrel vs llama.cpp EAGLE traces for a dataset slice.")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-model-hf", required=True)
    ap.add_argument("--head-model-hf", required=True)
    ap.add_argument("--base-model-gguf", required=True)
    ap.add_argument("--head-model-gguf", required=True)
    ap.add_argument("--llama-binary", default="./build-cuda/bin/llama-speculative-simple")
    ap.add_argument("--kestrel-train", default="/home/alvion/projects/kestrel/train.py")
    ap.add_argument("--llama-cwd", default=".")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--max-proposals", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--prob-threshold", type=float, default=1.0 / 1024.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    llama_cwd = Path(args.llama_cwd).resolve()

    rows: list[dict[str, Any]] = []
    all_deltas: list[float] = []
    total_divergent_positions = 0

    for idx in range(args.count):
        item_dir = out_dir / f"entry_{idx:02d}"
        ensure_dir(item_dir)

        kestrel_yaml = item_dir / "kestrel.yaml"
        llama_yaml = item_dir / "llama.yaml"
        prompt_file = item_dir / "prompt.txt"
        report_file = item_dir / "compare.html"

        run(
            [
                "python3",
                args.kestrel_train,
                "analyze",
                "--dataset",
                args.dataset,
                "--dataset-entry",
                str(idx),
                "--base-model",
                args.base_model_hf,
                "--head-model",
                args.head_model_hf,
                "--yaml-out",
                str(kestrel_yaml),
                "--max-proposals",
                str(args.max_proposals),
                "--max-depth",
                str(args.max_depth),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--prob-threshold",
                str(args.prob_threshold),
                "--temp",
                "0",
                "--top-k",
                "1",
            ],
            cwd=llama_cwd,
        )

        kestrel_trace = load_trace(kestrel_yaml)
        prompt_file.write_text(kestrel_trace["prompt"], encoding="utf-8")

        run(
            [
                args.llama_binary,
                "-m",
                args.base_model_gguf,
                "--override-kv",
                "tokenizer.ggml.add_bos_token=bool:false",
                "--model-draft",
                args.head_model_gguf,
                "--spec-type",
                "eagle3",
                "--eagle-max-depth",
                str(args.max_depth),
                "--eagle-max-proposals",
                str(args.max_proposals),
                "--eagle-beam-width",
                str(args.max_proposals),
                "--eagle-per-beam-topk-candidates",
                "1024",
                "--temp",
                "0",
                "--top-k",
                "1",
                "-fa",
                "on",
                "-n",
                str(args.max_new_tokens),
                "--no-conversation",
                "-f",
                str(prompt_file),
                "--eagle-trace-yaml",
                str(llama_yaml),
            ],
            cwd=llama_cwd,
        )

        llama_trace = load_trace(llama_yaml)
        stats = compare_greedy(kestrel_trace, llama_trace)
        div = divergence_stats(llama_trace, kestrel_trace)
        write_report(report_file, "kestrel", kestrel_trace, "llama.cpp", llama_trace, stats)

        total_divergent_positions += div["divergent_positions"]
        if div["divergent_positions"] > 0:
            cycles_a = llama_trace.get("cycles") or []
            cycles_b = kestrel_trace.get("cycles") or []
            n = min(len(cycles_a), len(cycles_b))
            for cidx in range(n):
                path_a = greedy_path(cycles_a[cidx])
                path_b = greedy_path(cycles_b[cidx])
                m = min(len(path_a), len(path_b))
                for depth in range(m):
                    if path_a[depth]["token"] != path_b[depth]["token"]:
                        all_deltas.append(abs(float(path_a[depth]["prob"]) - float(path_b[depth]["prob"])))

        row = {
            "dataset_entry": idx,
            "cycles_compared": stats["cycles_compared"],
            "exact_cycles": stats["exact_cycles"],
            "avg_prefix": stats["avg_prefix"],
            "token_match_rate": stats["token_match_rate"],
            "divergent_positions": div["divergent_positions"],
            "mean_prob_delta": div["mean_prob_delta"],
            "min_prob_delta": div["min_prob_delta"],
            "max_prob_delta": div["max_prob_delta"],
            "first_mismatch_desc": describe_first_mismatch(stats),
            "report": str(report_file.relative_to(out_dir)),
            "prompt_file": str(prompt_file.relative_to(out_dir)),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    if not all_deltas:
        all_deltas = [0.0]

    summary = {
        "count": len(rows),
        "divergent_positions_total": total_divergent_positions,
        "mean_prob_delta": mean(all_deltas),
        "min_prob_delta": min(all_deltas),
        "max_prob_delta": max(all_deltas),
        "rows": rows,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_summary_html(out_dir / "summary.html", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
