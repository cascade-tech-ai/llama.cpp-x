#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


RUNS = [
    {
        "name": "baseline_none",
        "label": "Baseline",
        "mode": "none",
        "max_depth": None,
        "adaptive_depth": None,
        "command": (
            "GGML_CUDA_NVTX=1 "
            "llama-speculative-simple --no-conversation "
            "-m /home/alvion/models/unsloth-Qwen3-4B-Instruct-2507-bf16.gguf "
            "--spec-type none "
            "-n 64 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on "
            "-f cascade/reports/eagle_trace_compare_first10_11297aac_selected/entry_00/prompt.txt"
        ),
    },
    {
        "name": "serial_d7_a005",
        "label": "Serial d=7 a=0.05",
        "mode": "serial",
        "max_depth": 7,
        "adaptive_depth": 0.05,
        "command": (
            "GGML_CUDA_NVTX=1 CASCADE_SPEC_PROFILE=1 CASCADE_EAGLE_PROFILE=1 "
            "llama-speculative-simple --no-conversation "
            "-m /home/alvion/models/unsloth-Qwen3-4B-Instruct-2507-bf16.gguf "
            "-md /home/alvion/projects/kestrel/models/gguf/qwen35-4b-phase2-asstmask-h100x4_eagle.bf16.gguf "
            "--spec-type eagle3 --eagle-serial --eagle-max-depth 7 --eagle-adaptive-depth 0.05 "
            "-ngld 999 -n 64 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on "
            "-f cascade/reports/eagle_trace_compare_first10_11297aac_selected/entry_00/prompt.txt"
        ),
    },
    {
        "name": "serial_d7_a020",
        "label": "Serial d=7 a=0.20",
        "mode": "serial",
        "max_depth": 7,
        "adaptive_depth": 0.20,
        "command": (
            "GGML_CUDA_NVTX=1 CASCADE_SPEC_PROFILE=1 CASCADE_EAGLE_PROFILE=1 "
            "llama-speculative-simple --no-conversation "
            "-m /home/alvion/models/unsloth-Qwen3-4B-Instruct-2507-bf16.gguf "
            "-md /home/alvion/projects/kestrel/models/gguf/qwen35-4b-phase2-asstmask-h100x4_eagle.bf16.gguf "
            "--spec-type eagle3 --eagle-serial --eagle-max-depth 7 --eagle-adaptive-depth 0.20 "
            "-ngld 999 -n 64 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on "
            "-f cascade/reports/eagle_trace_compare_first10_11297aac_selected/entry_00/prompt.txt"
        ),
    },
    {
        "name": "serial_d5_a005",
        "label": "Serial d=5 a=0.05",
        "mode": "serial",
        "max_depth": 5,
        "adaptive_depth": 0.05,
        "command": (
            "GGML_CUDA_NVTX=1 CASCADE_SPEC_PROFILE=1 CASCADE_EAGLE_PROFILE=1 "
            "llama-speculative-simple --no-conversation "
            "-m /home/alvion/models/unsloth-Qwen3-4B-Instruct-2507-bf16.gguf "
            "-md /home/alvion/projects/kestrel/models/gguf/qwen35-4b-phase2-asstmask-h100x4_eagle.bf16.gguf "
            "--spec-type eagle3 --eagle-serial --eagle-max-depth 5 --eagle-adaptive-depth 0.05 "
            "-ngld 999 -n 64 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on "
            "-f cascade/reports/eagle_trace_compare_first10_11297aac_selected/entry_00/prompt.txt"
        ),
    },
    {
        "name": "serial_d3_a005",
        "label": "Serial d=3 a=0.05",
        "mode": "serial",
        "max_depth": 3,
        "adaptive_depth": 0.05,
        "command": (
            "GGML_CUDA_NVTX=1 CASCADE_SPEC_PROFILE=1 CASCADE_EAGLE_PROFILE=1 "
            "llama-speculative-simple --no-conversation "
            "-m /home/alvion/models/unsloth-Qwen3-4B-Instruct-2507-bf16.gguf "
            "-md /home/alvion/projects/kestrel/models/gguf/qwen35-4b-phase2-asstmask-h100x4_eagle.bf16.gguf "
            "--spec-type eagle3 --eagle-serial --eagle-max-depth 3 --eagle-adaptive-depth 0.05 "
            "-ngld 999 -n 64 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on "
            "-f cascade/reports/eagle_trace_compare_first10_11297aac_selected/entry_00/prompt.txt"
        ),
    },
    {
        "name": "serial_d7_a000",
        "label": "Serial d=7 a=0.00",
        "mode": "serial",
        "max_depth": 7,
        "adaptive_depth": 0.00,
        "command": (
            "GGML_CUDA_NVTX=1 CASCADE_SPEC_PROFILE=1 CASCADE_EAGLE_PROFILE=1 "
            "llama-speculative-simple --no-conversation "
            "-m /home/alvion/models/unsloth-Qwen3-4B-Instruct-2507-bf16.gguf "
            "-md /home/alvion/projects/kestrel/models/gguf/qwen35-4b-phase2-asstmask-h100x4_eagle.bf16.gguf "
            "--spec-type eagle3 --eagle-serial --eagle-max-depth 7 --eagle-adaptive-depth 0.0 "
            "-ngld 999 -n 64 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on "
            "-f cascade/reports/eagle_trace_compare_first10_11297aac_selected/entry_00/prompt.txt"
        ),
    },
]

TARGET_RANGE = "spec/target_pass/validation_linear/forward"
DRAFT_RANGE = "spec/target_pass/eagle_total"
SERIAL_ROLLOUT_RANGE = "eagle3/serial_rollout"
LOGITS_RANGE = "eagle3/logits"


def fmt_ms(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f} ms"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}%"


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def to_ms(ns: str | None) -> float | None:
    if not ns:
        return None
    return float(ns) / 1_000_000.0


def clean_range(name: str | None) -> str:
    if not name:
        return ""
    return name[1:] if name.startswith(":") else name


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_range_rows(path: Path, range_col: str) -> dict[str, dict[str, float | int | str]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, float | int | str]] = {}
    for row in read_csv_rows(path):
        name = clean_range(row.get(range_col, ""))
        if not name:
            continue
        out[name] = {
            "name": name,
            "total_ms": to_ms(row.get("Total Time (ns)") or row.get("Total Proj Time (ns)")),
            "avg_ms": to_ms(row.get("Avg (ns)") or row.get("Proj Avg (ns)")),
            "instances": int(float(row.get("Instances") or row.get("Range Instances") or 0)),
            "gpu_ops": float(row.get("Total GPU Ops") or 0),
        }
    return out


def parse_kernels(path: Path) -> dict[str, list[dict[str, float | int | str]]]:
    if not path.exists():
        return {}
    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in read_csv_rows(path):
        range_name = clean_range(row.get("NVTX Range", ""))
        kernel = row.get("Kernel Name", "").strip()
        if not range_name or not kernel:
            continue
        grouped[range_name].append(
            {
                "kernel": kernel,
                "total_ms": to_ms(row.get("Total Time (ns)")) or 0.0,
                "instances": int(float(row.get("Kern Inst") or 0)),
            }
        )
    for items in grouped.values():
        items.sort(key=lambda item: item["total_ms"], reverse=True)
    return dict(grouped)


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text()
    data: dict[str, object] = {
        "success": "main: llama_decode failed" not in text and "failed to create context" not in text,
        "failure_reason": None,
    }

    decoded = re.search(r"decoded\s+(\d+) tokens in\s+([\d.]+)\s+seconds,\s+speed:\s+([\d.]+)\s+t/s", text)
    if decoded:
        data["decoded_tokens"] = int(decoded.group(1))
        data["decode_seconds"] = float(decoded.group(2))
        data["decode_tps"] = float(decoded.group(3))

    encoded = re.search(r"encoded\s+(\d+) tokens in\s+([\d.]+)\s+seconds,\s+speed:\s+([\d.]+)\s+t/s", text)
    if encoded:
        data["prompt_tokens"] = int(encoded.group(1))
        data["prompt_seconds"] = float(encoded.group(2))
        data["prompt_tps"] = float(encoded.group(3))

    totals = re.search(
        r"spec profile total:\s+target_passes=(\d+)\s+target_total=([\d.]+)ms\s+target_forward=([\d.]+)ms\s+"
        r"target_sampling=([\d.]+)ms\s+eagle_total=([\d.]+)ms",
        text,
    )
    if totals:
        data["target_passes"] = int(totals.group(1))
        data["target_total_ms"] = float(totals.group(2))
        data["target_forward_ms"] = float(totals.group(3))
        data["target_sampling_ms"] = float(totals.group(4))
        data["eagle_total_ms"] = float(totals.group(5))

    for key in ["n_draft", "n_predict", "n_drafted", "n_accept"]:
        match = re.search(rf"{key}\s*=\s*([0-9]+)", text)
        if match:
            data[key] = int(match.group(1))

    accept = re.search(r"accept\s*=\s*([-\d.]+)%", text)
    if accept:
        data["accept_pct"] = float(accept.group(1))

    acc_len = re.search(r"acc_len\s*=\s*([-\d.]+)", text)
    if acc_len:
        data["acc_len"] = float(acc_len.group(1))

    avg_depth = re.search(r"avg_depth\s*=\s*([-\d.]+)", text)
    if avg_depth:
        data["avg_depth"] = float(avg_depth.group(1))

    perf_patterns = {
        "draft_time_ms": r"speculative_perf:\s+draft time =\s+([\d.]+)\s+ms",
        "target_fwd_perf_ms": r"speculative_perf:\s+target fwd time =\s+([\d.]+)\s+ms",
        "target_time_perf_ms": r"speculative_perf:\s+target time =\s+([\d.]+)\s+ms",
        "decode_time_ms": r"speculative_perf:\s+decode time =\s+([\d.]+)\s+ms",
        "sampling_time_ms": r"speculative_perf:\s+sampling time =\s+([\d.]+)\s+ms",
    }
    for key, pattern in perf_patterns.items():
        match = re.search(pattern, text)
        if match:
            data[key] = float(match.group(1))

    graphs = re.search(r"speculative_perf:\s+graphs reused =\s+(\d+)", text)
    if graphs:
        data["graphs_reused"] = int(graphs.group(1))

    for pattern in [
        r"init: invalid token\[[^\n]+",
        r"decode: failed to initialize batch",
        r"main: llama_decode failed with -1",
        r"GGML_ASSERT[^\n]+",
    ]:
        match = re.search(pattern, text)
        if match:
            data["failure_reason"] = match.group(0)
            break

    return data


def make_run_entry(report_dir: Path, meta: dict[str, object]) -> dict[str, object]:
    name = str(meta["name"])
    logs_dir = report_dir / "logs"
    stats_dir = report_dir / "stats"
    log_path = logs_dir / f"{name}.log"
    pushpop_path = stats_dir / f"{name}_nvtx_pushpop_sum.csv"
    gpu_path = stats_dir / f"{name}_nvtx_gpu_proj_sum.csv"
    kernel_path = stats_dir / f"{name}_nvtx_kern_sum_base.csv"

    entry: dict[str, object] = dict(meta)
    entry["paths"] = {
        "log": str(log_path.relative_to(report_dir)),
        "pushpop": str(pushpop_path.relative_to(report_dir)) if pushpop_path.exists() else None,
        "gpu": str(gpu_path.relative_to(report_dir)) if gpu_path.exists() else None,
        "kernels": str(kernel_path.relative_to(report_dir)) if kernel_path.exists() else None,
    }
    entry["log"] = parse_log(log_path)
    entry["pushpop"] = parse_range_rows(pushpop_path, "Range")
    entry["gpu"] = parse_range_rows(gpu_path, "Range")
    entry["kernels"] = parse_kernels(kernel_path)

    log = entry["log"]
    target_passes = log.get("target_passes") or 0
    if target_passes:
        target_passes = int(target_passes)
        entry["derived"] = {
            "target_host_ms_per_pass": float(log.get("target_total_ms", 0.0)) / target_passes,
            "target_forward_host_ms_per_pass": float(log.get("target_forward_ms", 0.0)) / target_passes,
            "target_sampling_host_ms_per_pass": float(log.get("target_sampling_ms", 0.0)) / target_passes,
            "draft_host_ms_per_pass": float(log.get("eagle_total_ms", 0.0)) / target_passes,
            "target_gpu_ms_per_pass": entry["gpu"].get("spec/target_pass/total", {}).get("avg_ms"),
            "target_validation_gpu_ms_per_pass": entry["gpu"].get(TARGET_RANGE, {}).get("avg_ms"),
            "draft_gpu_ms_per_pass": entry["gpu"].get(DRAFT_RANGE, {}).get("avg_ms"),
            "serial_rollout_gpu_ms_per_pass": entry["gpu"].get(SERIAL_ROLLOUT_RANGE, {}).get("avg_ms"),
            "logits_gpu_ms_per_pass": entry["gpu"].get(LOGITS_RANGE, {}).get("avg_ms"),
        }
    else:
        entry["derived"] = {}

    return entry


def kernel_rows(entry: dict[str, object], range_name: str, limit: int = 5) -> list[dict[str, object]]:
    items = list(entry["kernels"].get(range_name, []))
    total = sum(float(item["total_ms"]) for item in items[:limit]) or 1.0
    rows = []
    for item in items[:limit]:
        share = 100.0 * float(item["total_ms"]) / total
        rows.append(
            {
                "kernel": item["kernel"],
                "total_ms": float(item["total_ms"]),
                "instances": int(item["instances"]),
                "share_pct": share,
            }
        )
    return rows


def rel(path: str | None) -> str:
    if not path:
        return "#"
    return html.escape(path)


def code(text: str) -> str:
    return f"<code>{html.escape(text)}</code>"


def render_findings(by_name: dict[str, dict[str, object]]) -> list[str]:
    baseline = by_name["baseline_none"]
    rep = by_name["serial_d7_a005"]
    high_adapt = by_name["serial_d7_a020"]
    depth3 = by_name["serial_d3_a005"]
    depth5 = by_name["serial_d5_a005"]
    broken = by_name["serial_d7_a000"]

    baseline_tps = float(baseline["log"].get("decode_tps", 0.0))
    rep_tps = float(rep["log"].get("decode_tps", 0.0))
    speed_delta = 100.0 * (rep_tps / baseline_tps - 1.0)

    baseline_target_gpu = float(baseline["derived"].get("target_validation_gpu_ms_per_pass") or 0.0)
    rep_target_gpu = float(rep["derived"].get("target_validation_gpu_ms_per_pass") or 0.0)
    rep_draft_gpu = float(rep["derived"].get("draft_gpu_ms_per_pass") or 0.0)
    rep_forward_host = float(rep["derived"].get("target_forward_host_ms_per_pass") or 0.0)
    counter_ratio = rep_target_gpu / rep_forward_host if rep_forward_host else None

    depth_tps = [
        float(depth3["log"].get("decode_tps", 0.0)),
        float(depth5["log"].get("decode_tps", 0.0)),
        float(rep["log"].get("decode_tps", 0.0)),
    ]
    depth_spread = max(depth_tps) - min(depth_tps)

    findings = [
        (
            f"On this 64-token serial microprofile, speculation is slower than baseline: "
            f"{rep_tps:.2f} tok/s vs {baseline_tps:.2f} tok/s ({speed_delta:.1f}%)."
        ),
        (
            f"Nsight projects nearly the same target-validation GPU work per pass with and without serial speculation: "
            f"{baseline_target_gpu:.2f} ms/pass baseline vs {rep_target_gpu:.2f} ms/pass for `serial_d7_a005`."
        ),
        (
            f"The serial draft path adds about {rep_draft_gpu:.2f} ms/pass of GPU work and "
            f"{float(rep['derived'].get('draft_host_ms_per_pass') or 0.0):.2f} ms/pass of host wall time, "
            f"but this prompt only drafted {int(rep['log'].get('n_drafted', 0))} token and accepted "
            f"{int(rep['log'].get('n_accept', 0))}."
        ),
        (
            f"The in-app `target_forward` host counter is not a device-time proxy here: it reports "
            f"{rep_forward_host:.2f} ms/pass while Nsight places about {rep_target_gpu:.2f} ms/pass of GPU work in "
            f"`{TARGET_RANGE}`" + (f" ({counter_ratio:.1f}x larger)." if counter_ratio else ".")
        ),
        (
            f"With `adaptive_depth=0.05`, changing `max_depth` from 3 to 7 barely changes throughput or timing "
            f"(spread {depth_spread:.2f} tok/s). On this prompt the rollout almost always stops before depth becomes relevant."
        ),
        (
            f"`serial_d7_a020` is marginally better than `serial_d7_a005` ({float(high_adapt['log'].get('decode_tps', 0.0)):.2f} vs "
            f"{rep_tps:.2f} tok/s) because it prunes draft work slightly harder, but it still does not beat baseline."
        ),
        (
            f"`serial_d7_a000` is currently broken on this model/head pair. The run fails immediately after prefill with "
            f"{code(str(broken['log'].get('failure_reason') or 'an unknown decode failure'))}."
        ),
    ]
    return findings


def render_run_table(entries: list[dict[str, object]]) -> str:
    rows = []
    for entry in entries:
        log = entry["log"]
        derived = entry["derived"]
        status = "ok" if log.get("success") else "failed"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(entry['label']))}</td>"
            f"<td>{status}</td>"
            f"<td>{fmt_num(log.get('decode_tps'))}</td>"
            f"<td>{log.get('n_drafted', 'N/A')}</td>"
            f"<td>{log.get('n_accept', 'N/A')}</td>"
            f"<td>{fmt_num(log.get('avg_depth'))}</td>"
            f"<td>{fmt_ms(derived.get('target_host_ms_per_pass'))}</td>"
            f"<td>{fmt_ms(derived.get('target_forward_host_ms_per_pass'))}</td>"
            f"<td>{fmt_ms(derived.get('target_sampling_host_ms_per_pass'))}</td>"
            f"<td>{fmt_ms(derived.get('draft_host_ms_per_pass'))}</td>"
            f"<td>{fmt_ms(derived.get('target_validation_gpu_ms_per_pass'))}</td>"
            f"<td>{fmt_ms(derived.get('draft_gpu_ms_per_pass'))}</td>"
            f"<td>{fmt_ms(derived.get('serial_rollout_gpu_ms_per_pass'))}</td>"
            f"<td>{fmt_ms(derived.get('logits_gpu_ms_per_pass'))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_kernel_table(entry: dict[str, object], range_name: str, title: str) -> str:
    rows = kernel_rows(entry, range_name)
    if not rows:
        return f"<section class='card'><h3>{html.escape(title)}</h3><p>No kernel data.</p></section>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['kernel']))}</td>"
            f"<td>{fmt_ms(float(row['total_ms']), 3)}</td>"
            f"<td>{row['instances']}</td>"
            f"<td>{fmt_pct(float(row['share_pct']), 1)}</td>"
            "</tr>"
        )
    return (
        "<section class='card'>"
        f"<h3>{html.escape(title)}</h3>"
        f"<div class='subtle'>{code(range_name)}</div>"
        "<table><thead><tr><th>Kernel</th><th>Total</th><th>Instances</th><th>Share of Top-5</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
        "</section>"
    )


def build_html(report_dir: Path, entries: list[dict[str, object]], metadata: dict[str, str]) -> str:
    by_name = {str(entry["name"]): entry for entry in entries}
    representative = by_name["serial_d7_a005"]

    findings = "".join(f"<li>{item}</li>" for item in render_findings(by_name))
    table_rows = render_run_table(entries)

    artifact_rows = []
    for entry in entries:
        paths = entry["paths"]
        pushpop_link = (
            f"<a href='{rel(paths['pushpop'])}'>{html.escape(str(paths['pushpop']))}</a>"
            if paths["pushpop"]
            else "N/A"
        )
        gpu_link = (
            f"<a href='{rel(paths['gpu'])}'>{html.escape(str(paths['gpu']))}</a>"
            if paths["gpu"]
            else "N/A"
        )
        kernel_link = (
            f"<a href='{rel(paths['kernels'])}'>{html.escape(str(paths['kernels']))}</a>"
            if paths["kernels"]
            else "N/A"
        )
        artifact_rows.append(
            "<tr>"
            f"<td>{html.escape(str(entry['label']))}</td>"
            f"<td><a href='{rel(paths['log'])}'>{html.escape(str(paths['log']))}</a></td>"
            f"<td>{pushpop_link}</td>"
            f"<td>{gpu_link}</td>"
            f"<td>{kernel_link}</td>"
            "</tr>"
        )

    command_rows = []
    for entry in entries:
        command_rows.append(
            "<details class='card command'>"
            f"<summary>{html.escape(str(entry['label']))}</summary>"
            f"<pre>{html.escape(str(entry['command']))}</pre>"
            "</details>"
        )

    broken = by_name["serial_d7_a000"]
    broken_reason = html.escape(str(broken["log"].get("failure_reason") or "unknown failure"))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Qwen3.5-4B Serial EAGLE Nsight Report</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --fg: #221d18;
      --muted: #6a6158;
      --line: #d4c3ad;
      --card: #fffaf2;
      --accent: #9a4f2a;
      --accent-soft: #f3ddd0;
      --good: #2f7d32;
      --bad: #a52c2c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Iosevka Etoile", sans-serif;
      background:
        radial-gradient(circle at top left, #fff8ef 0%, var(--bg) 45%, #eadccc 100%);
      color: var(--fg);
    }}
    main {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1, h2, h3 {{
      font-family: "IBM Plex Serif", serif;
      margin: 0 0 12px;
    }}
    p, li {{
      line-height: 1.45;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin: 18px 0 24px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 18px;
      box-shadow: 0 12px 28px rgba(62, 39, 21, 0.08);
    }}
    .hero {{
      padding: 22px 24px;
      background:
        linear-gradient(135deg, rgba(154, 79, 42, 0.12), rgba(154, 79, 42, 0.02)),
        var(--card);
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 18px;
    }}
    .meta-line {{
      margin: 4px 0;
      color: var(--muted);
    }}
    code, pre {{
      font-family: "Iosevka", monospace;
      font-size: 12px;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 12px 28px rgba(62, 39, 21, 0.08);
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #efe2d0;
      position: sticky;
      top: 0;
    }}
    tr:nth-child(even) td {{
      background: rgba(154, 79, 42, 0.035);
    }}
    .subtle {{
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin: 18px 0 24px;
    }}
    .status-ok {{ color: var(--good); }}
    .status-failed {{ color: var(--bad); }}
    .command summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    ul {{
      margin: 10px 0 0 18px;
      padding: 0;
    }}
    @media (max-width: 1000px) {{
      .hero-grid, .two-col {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
<main>
  <section class="card hero">
    <div class="hero-grid">
      <div>
        <h1>Qwen3.5-4B Serial EAGLE Nsight Report</h1>
        <p>This report compares baseline decode against serial EAGLE runs using the new non-invasive NVTX coverage. It combines in-app pass counters with Nsight Systems projected GPU time so the host/device split is visible without adding extra synchronization.</p>
        <div class="meta-line"><strong>Scope:</strong> serial mode only, single prompt, 64 decode tokens, greedy decode, BF16 base + BF16 EAGLE head</div>
        <div class="meta-line"><strong>Prompt:</strong> {code("cascade/reports/eagle_trace_compare_first10_11297aac_selected/entry_00/prompt.txt")}</div>
        <div class="meta-line"><strong>Repo:</strong> {code(metadata["commit"])}</div>
        <div class="meta-line"><strong>GPU:</strong> {code(metadata["gpu"])}</div>
        <div class="meta-line"><strong>Profiler:</strong> {code(metadata["nsight"])}</div>
        <div class="meta-line"><strong>Generated:</strong> {html.escape(metadata["generated_at"])}</div>
      </div>
      <div class="card">
        <h3>Interpretation Notes</h3>
        <ul>
          <li><strong>Host ranges</strong> come from <code>nvtx_pushpop_sum</code> and the in-app counters. They are wall-time, not synchronized GPU elapsed time.</li>
          <li><strong>Projected GPU ranges</strong> come from <code>nvtx_gpu_proj_sum</code>. They estimate device work launched inside a range without forcing sync.</li>
          <li><strong>Nested NVTX ranges overlap.</strong> Do not sum parent and child projected GPU totals.</li>
          <li><strong>This prompt is effectively zero-acceptance.</strong> It exposes overhead well, but it is not a best-case speculative speedup scenario.</li>
        </ul>
      </div>
    </div>
  </section>

  <section class="card">
    <h2>Key Findings</h2>
    <ul>{findings}</ul>
  </section>

  <section class="card">
    <h2>Per-Run Summary</h2>
    <table>
      <thead>
        <tr>
          <th>Run</th>
          <th>Status</th>
          <th>Tok/s</th>
          <th>Drafted</th>
          <th>Accepted</th>
          <th>Avg Depth</th>
          <th>Target Host / Pass</th>
          <th>Target Forward Counter / Pass</th>
          <th>Target Sampling Counter / Pass</th>
          <th>Draft Host / Pass</th>
          <th>Validation GPU / Pass</th>
          <th>Draft GPU / Pass</th>
          <th>Serial Rollout GPU / Pass</th>
          <th>Logits GPU / Pass</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </section>

  <section class="two-col">
    <section class="card">
      <h2>Representative Host vs Device View</h2>
      <p>The representative serial run is <strong>{html.escape(str(representative["label"]))}</strong>. The key mismatch is that the host-side <code>target_forward</code> counter stays tiny even though Nsight shows the target validation range occupying almost the entire pass on device.</p>
      <table>
        <thead><tr><th>Metric</th><th>Baseline</th><th>Serial d=7 a=0.05</th></tr></thead>
        <tbody>
          <tr><td>Decode throughput</td><td>{fmt_num(by_name["baseline_none"]["log"].get("decode_tps"))} tok/s</td><td>{fmt_num(representative["log"].get("decode_tps"))} tok/s</td></tr>
          <tr><td>Target pass host / pass</td><td>{fmt_ms(by_name["baseline_none"]["derived"].get("target_host_ms_per_pass"))}</td><td>{fmt_ms(representative["derived"].get("target_host_ms_per_pass"))}</td></tr>
          <tr><td>Target forward counter / pass</td><td>{fmt_ms(by_name["baseline_none"]["derived"].get("target_forward_host_ms_per_pass"))}</td><td>{fmt_ms(representative["derived"].get("target_forward_host_ms_per_pass"))}</td></tr>
          <tr><td>Validation GPU / pass</td><td>{fmt_ms(by_name["baseline_none"]["derived"].get("target_validation_gpu_ms_per_pass"))}</td><td>{fmt_ms(representative["derived"].get("target_validation_gpu_ms_per_pass"))}</td></tr>
          <tr><td>Draft host / pass</td><td>{fmt_ms(by_name["baseline_none"]["derived"].get("draft_host_ms_per_pass"))}</td><td>{fmt_ms(representative["derived"].get("draft_host_ms_per_pass"))}</td></tr>
          <tr><td>Draft GPU / pass</td><td>{fmt_ms(by_name["baseline_none"]["derived"].get("draft_gpu_ms_per_pass"))}</td><td>{fmt_ms(representative["derived"].get("draft_gpu_ms_per_pass"))}</td></tr>
          <tr><td>Drafted / accepted</td><td>{by_name["baseline_none"]["log"].get("n_drafted", 0)} / {by_name["baseline_none"]["log"].get("n_accept", 0)}</td><td>{representative["log"].get("n_drafted", 0)} / {representative["log"].get("n_accept", 0)}</td></tr>
        </tbody>
      </table>
    </section>

    <section class="card">
      <h2>Parameter Sweep Readout</h2>
      <p>For this prompt, the stable serial runs cluster tightly together. Increasing <code>max_depth</code> from 3 to 7 at <code>adaptive_depth=0.05</code> barely moves the totals because the rollout almost always stops before depth matters.</p>
      <table>
        <thead><tr><th>Run</th><th>Tok/s</th><th>Draft Host / Pass</th><th>Validation GPU / Pass</th><th>Draft GPU / Pass</th></tr></thead>
        <tbody>
          <tr><td>Serial d=3 a=0.05</td><td>{fmt_num(by_name["serial_d3_a005"]["log"].get("decode_tps"))}</td><td>{fmt_ms(by_name["serial_d3_a005"]["derived"].get("draft_host_ms_per_pass"))}</td><td>{fmt_ms(by_name["serial_d3_a005"]["derived"].get("target_validation_gpu_ms_per_pass"))}</td><td>{fmt_ms(by_name["serial_d3_a005"]["derived"].get("draft_gpu_ms_per_pass"))}</td></tr>
          <tr><td>Serial d=5 a=0.05</td><td>{fmt_num(by_name["serial_d5_a005"]["log"].get("decode_tps"))}</td><td>{fmt_ms(by_name["serial_d5_a005"]["derived"].get("draft_host_ms_per_pass"))}</td><td>{fmt_ms(by_name["serial_d5_a005"]["derived"].get("target_validation_gpu_ms_per_pass"))}</td><td>{fmt_ms(by_name["serial_d5_a005"]["derived"].get("draft_gpu_ms_per_pass"))}</td></tr>
          <tr><td>Serial d=7 a=0.05</td><td>{fmt_num(by_name["serial_d7_a005"]["log"].get("decode_tps"))}</td><td>{fmt_ms(by_name["serial_d7_a005"]["derived"].get("draft_host_ms_per_pass"))}</td><td>{fmt_ms(by_name["serial_d7_a005"]["derived"].get("target_validation_gpu_ms_per_pass"))}</td><td>{fmt_ms(by_name["serial_d7_a005"]["derived"].get("draft_gpu_ms_per_pass"))}</td></tr>
          <tr><td>Serial d=7 a=0.20</td><td>{fmt_num(by_name["serial_d7_a020"]["log"].get("decode_tps"))}</td><td>{fmt_ms(by_name["serial_d7_a020"]["derived"].get("draft_host_ms_per_pass"))}</td><td>{fmt_ms(by_name["serial_d7_a020"]["derived"].get("target_validation_gpu_ms_per_pass"))}</td><td>{fmt_ms(by_name["serial_d7_a020"]["derived"].get("draft_gpu_ms_per_pass"))}</td></tr>
        </tbody>
      </table>
    </section>
  </section>

  <section class="card">
    <h2>Kernel Mix</h2>
    <p>The target path is still dominated by the usual matmul kernels. The serial draft path is much smaller on device, but it is still pure overhead on this prompt because acceptance is zero.</p>
  </section>
  <section class="two-col">
    {render_kernel_table(by_name["baseline_none"], TARGET_RANGE, "Baseline Target Validation")}
    {render_kernel_table(by_name["serial_d7_a005"], TARGET_RANGE, "Serial Target Validation")}
  </section>
  <section class="two-col">
    {render_kernel_table(by_name["serial_d7_a005"], DRAFT_RANGE, "Serial Draft Total")}
    {render_kernel_table(by_name["serial_d7_a005"], SERIAL_ROLLOUT_RANGE, "Serial Rollout")}
  </section>

  <section class="card">
    <h2>Broken Configuration</h2>
    <p><strong>Run:</strong> {html.escape(str(broken["label"]))}</p>
    <p><strong>Failure:</strong> {broken_reason}</p>
    <p>This is not a tuning datapoint. The current serial path fails as soon as <code>adaptive_depth=0</code> is used on this base/head/prompt combination, so that should be treated as a correctness bug before using it for performance exploration.</p>
  </section>

  <section class="card">
    <h2>Commands</h2>
    {''.join(command_rows)}
  </section>

  <section class="card">
    <h2>Artifacts</h2>
    <table>
      <thead><tr><th>Run</th><th>Log</th><th>Push/Pop CSV</th><th>GPU Proj CSV</th><th>Kernel CSV</th></tr></thead>
      <tbody>{''.join(artifact_rows)}</tbody>
    </table>
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an HTML report for serial EAGLE Nsight captures.")
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    entries = [make_run_entry(report_dir, meta) for meta in RUNS]

    metadata = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "commit": "e873901ef663",
        "gpu": "NVIDIA GeForce RTX 5090, driver 590.48.01, 32607 MiB",
        "nsight": "NVIDIA Nsight Systems 2024.6.2",
    }

    output = {
        "metadata": metadata,
        "runs": entries,
    }
    json_path = report_dir / "qwen35_4b_serial_nsight_report.json"
    html_path = report_dir / "qwen35_4b_serial_nsight_report.html"
    json_path.write_text(json.dumps(output, indent=2))
    html_path.write_text(build_html(report_dir, entries, metadata))


if __name__ == "__main__":
    main()
