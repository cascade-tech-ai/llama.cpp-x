#!/usr/bin/env python3
# AI-GENERATED: This file was created with AI assistance for an experimental fork.
# DO NOT SUBMIT upstream unless rewritten or exhaustively reviewed by a human.
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


RE_DECODED = re.compile(r"decoded\s+(\d+)\s+tokens.*speed:\s+([0-9.]+)\s+t/s")
RE_ENCODED = re.compile(r"encoded\s+(\d+)\s+tokens.*speed:\s+([0-9.]+)\s+t/s")
RE_ACCEPT = re.compile(r"accept\s+=\s+([0-9.]+)%")
RE_N_VAL = re.compile(r"^(n_predict|n_drafted|n_accept)\s+=\s+(\d+)$", re.MULTILINE)
RE_PROFILE_SUMMARY = re.compile(
    r"eagle3 profile: prompt=(\d+) depth=(\d+) beam=(\d+) props=(\d+) out=(\d+) "
    r"total=([0-9.]+)ms prefill=([0-9.]+)ms logits=([0-9.]+)ms score=([0-9.]+)ms step=([0-9.]+)ms "
    r"calls\(logits=(\d+),step=(\d+)\)(?: gpu\(logits=([0-9.]+)ms,step=([0-9.]+)ms\))?"
)
RE_PROFILE_ROOT = re.compile(r"eagle3 profile root: prompt=(\d+) root_step=([0-9.]+)ms")
RE_PROFILE_DEPTH = re.compile(
    r"eagle3 profile depth=(\d+) active_beams=(\d+) expansions=(\d+) "
    r"select=([0-9.]+)ms score=([0-9.]+)ms step=([0-9.]+)ms"
    r"(?: gpu\(select=([0-9.]+)ms,step=([0-9.]+)ms\))?"
)


@dataclass
class ProfileDepth:
    depth: int
    active_beams: int
    expansions: int
    select_ms: float
    score_ms: float
    step_ms: float
    select_gpu_ms: float | None = None
    step_gpu_ms: float | None = None


@dataclass
class ProfileSummary:
    prompt_tokens: int
    max_depth: int
    beam_width: int
    max_proposals: int
    output_tokens: int
    total_ms: float
    prefill_ms: float
    logits_ms: float
    score_ms: float
    step_ms: float
    logits_calls: int
    step_calls: int
    logits_gpu_ms: float | None = None
    step_gpu_ms: float | None = None
    root_step_ms: float | None = None


@dataclass
class ProfileBlock:
    summary: ProfileSummary
    depths: list[ProfileDepth] = field(default_factory=list)


@dataclass
class RunResult:
    family: str
    base_quant: str
    head_quant: str
    drafting: str
    prompt_tokens: int | None
    output_tokens: int | None
    output_tps: float | None
    n_predict: int | None
    n_drafted: int | None
    n_accept: int | None
    acceptance_rate: float | None
    log_path: str
    command: str
    returncode: int
    profile_blocks: list[ProfileBlock] = field(default_factory=list)


def parse_metrics(text: str) -> tuple[int | None, int | None, float | None, dict[str, int], float | None]:
    prompt_tokens = None
    output_tokens = None
    output_tps = None
    acceptance_rate = None
    n_vals: dict[str, int] = {}

    m = RE_ENCODED.search(text)
    if m:
        prompt_tokens = int(m.group(1))

    m = RE_DECODED.search(text)
    if m:
        output_tokens = int(m.group(1))
        output_tps = float(m.group(2))

    for key, value in RE_N_VAL.findall(text):
        n_vals[key] = int(value)

    m = RE_ACCEPT.search(text)
    if m:
        acceptance_rate = float(m.group(1))

    return prompt_tokens, output_tokens, output_tps, n_vals, acceptance_rate


def parse_profile(text: str) -> list[ProfileBlock]:
    blocks: list[ProfileBlock] = []
    current: ProfileBlock | None = None

    for line in text.splitlines():
        summary_match = RE_PROFILE_SUMMARY.search(line)
        if summary_match:
            current = ProfileBlock(summary=ProfileSummary(
                prompt_tokens=int(summary_match.group(1)),
                max_depth=int(summary_match.group(2)),
                beam_width=int(summary_match.group(3)),
                max_proposals=int(summary_match.group(4)),
                output_tokens=int(summary_match.group(5)),
                total_ms=float(summary_match.group(6)),
                prefill_ms=float(summary_match.group(7)),
                logits_ms=float(summary_match.group(8)),
                score_ms=float(summary_match.group(9)),
                step_ms=float(summary_match.group(10)),
                logits_calls=int(summary_match.group(11)),
                step_calls=int(summary_match.group(12)),
                logits_gpu_ms=float(summary_match.group(13)) if summary_match.group(13) else None,
                step_gpu_ms=float(summary_match.group(14)) if summary_match.group(14) else None,
            ))
            blocks.append(current)
            continue

        root_match = RE_PROFILE_ROOT.search(line)
        if root_match and current is not None:
            current.summary.root_step_ms = float(root_match.group(2))
            continue

        depth_match = RE_PROFILE_DEPTH.search(line)
        if depth_match and current is not None:
            current.depths.append(ProfileDepth(
                depth=int(depth_match.group(1)),
                active_beams=int(depth_match.group(2)),
                expansions=int(depth_match.group(3)),
                select_ms=float(depth_match.group(4)),
                score_ms=float(depth_match.group(5)),
                step_ms=float(depth_match.group(6)),
                select_gpu_ms=float(depth_match.group(7)) if depth_match.group(7) else None,
                step_gpu_ms=float(depth_match.group(8)) if depth_match.group(8) else None,
            ))

    return blocks


def run_case(cmd: list[str], log_path: Path, *, profile: bool) -> tuple[int, str]:
    env = os.environ.copy()
    for key in [
        "CASCADE_EAGLE_PROFILE",
        "CASCADE_EAGLE_PROFILE_GPU",
        "CASCADE_EAGLE_PROFILE_DEEP",
        "CASCADE_EAGLE_PROFILE_SPLIT",
        "CASCADE_EAGLE_DUMP_DIR",
        "GGML_CUDA_TIMING",
        "GGML_CUDA_NVTX",
    ]:
        env.pop(key, None)

    if profile:
        env["CASCADE_EAGLE_PROFILE"] = "1"
        env["CASCADE_EAGLE_PROFILE_GPU"] = "1"
        env["GGML_CUDA_TIMING"] = "1"

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd="/home/alvion/projects/llama.cpp-x",
    )
    log_path.write_text(proc.stdout)
    return proc.returncode, proc.stdout


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_build(build_dir: Path, archive_dir: Path) -> dict[str, object]:
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    bin_src = build_dir / "bin"
    if not bin_src.is_dir():
        raise FileNotFoundError(f"missing build bin directory: {bin_src}")

    bin_dst = archive_dir / "bin"
    shutil.copytree(bin_src, bin_dst, symlinks=True)

    cmake_cache = build_dir / "CMakeCache.txt"
    if cmake_cache.exists():
        shutil.copy2(cmake_cache, archive_dir / "CMakeCache.txt")

    manifest = {
        "archived_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "build_dir": str(build_dir),
        "files": [],
    }
    for path in sorted(bin_dst.rglob("*")):
        if path.is_file() and not path.is_symlink():
            manifest["files"].append({
                "path": str(path.relative_to(archive_dir)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })

    manifest_path = archive_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {
        "archive_dir": str(archive_dir),
        "manifest": str(manifest_path),
    }


def render_summary_html(results: list[RunResult], report_path: Path, meta: dict[str, object]) -> None:
    rows = []
    for item in results:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.family)}</td>"
            f"<td>{html.escape(item.base_quant)}</td>"
            f"<td>{html.escape(item.head_quant)}</td>"
            f"<td>{html.escape(item.drafting)}</td>"
            f"<td>{'' if item.prompt_tokens is None else item.prompt_tokens}</td>"
            f"<td>{'' if item.output_tokens is None else item.output_tokens}</td>"
            f"<td>{'' if item.output_tps is None else f'{item.output_tps:.2f}'}</td>"
            f"<td>{'' if item.n_predict is None else item.n_predict}</td>"
            f"<td>{'' if item.n_drafted is None else item.n_drafted}</td>"
            f"<td>{'' if item.n_accept is None else item.n_accept}</td>"
            f"<td>{'N/A' if item.acceptance_rate is None else f'{item.acceptance_rate:.3f}%'}</td>"
            "</tr>"
        )

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EAGLE Performance Report</title>
  <style>
    :root {{
      --bg: #f6f2ea;
      --fg: #1f1b18;
      --muted: #6b6259;
      --line: #d7c7b3;
      --card: #fffaf3;
    }}
    body {{
      margin: 0;
      font-family: "Iosevka Etoile", "IBM Plex Sans", sans-serif;
      background: radial-gradient(circle at top left, #fff8ef 0%, var(--bg) 50%, #efe4d5 100%);
      color: var(--fg);
    }}
    main {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px;
    }}
    h1, h2 {{
      font-family: "IBM Plex Serif", serif;
      margin: 0 0 12px;
    }}
    .meta, .note {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px 18px;
      margin-bottom: 18px;
      box-shadow: 0 12px 30px rgba(70, 45, 18, 0.06);
    }}
    .meta code {{
      font-family: "Iosevka", monospace;
      font-size: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 12px 30px rgba(70, 45, 18, 0.06);
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f0e3d2;
      position: sticky;
      top: 0;
    }}
    tr:nth-child(even) td {{
      background: rgba(143, 59, 27, 0.03);
    }}
    .muted {{
      color: var(--muted);
    }}
  </style>
</head>
<body>
<main>
  <h1>EAGLE CUDA Performance</h1>
  <div class="meta">
    <div><strong>Method:</strong> CUDA-only, greedy decode, chat templating through <code>llama-speculative-simple</code>, coarse profiling disabled.</div>
    <div><strong>Prompt:</strong> <code>{html.escape(str(meta["prompt"]))}</code></div>
    <div><strong>Binary:</strong> <code>{html.escape(str(meta["binary"]))}</code></div>
    <div><strong>Args:</strong> <code>{html.escape(str(meta["args"]))}</code></div>
    <div><strong>Build Archive:</strong> <code>{html.escape(str(meta["build_archive"]))}</code></div>
    <div><strong>Detailed Report:</strong> <code>{html.escape(str(meta["detail_report"]))}</code></div>
    <div><strong>Generated:</strong> {html.escape(str(meta["generated_at"]))}</div>
  </div>
  <div class="note muted">
    Acceptance is reported from the example binary as extra accepted draft tokens / drafted tokens.
    Baseline non-EAGLE rows do not have acceptance metrics.
  </div>
  <table>
    <thead>
      <tr>
        <th>Family</th>
        <th>Base</th>
        <th>Head</th>
        <th>Mode</th>
        <th>Prompt Tok</th>
        <th>Output Tok</th>
        <th>Output Tok/s</th>
        <th>n_predict</th>
        <th>n_drafted</th>
        <th>n_accept</th>
        <th>Acceptance</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</main>
</body>
</html>
"""
    report_path.write_text(report)


def render_detail_html(results: list[RunResult], report_path: Path, meta: dict[str, object]) -> None:
    sections: list[str] = []
    current_family = None

    for item in results:
        if item.family != current_family:
            current_family = item.family
            sections.append(f"<h2>{html.escape(item.family)}</h2>")

        metrics = [
            f"<li><strong>Mode:</strong> {html.escape(item.drafting)}</li>",
            f"<li><strong>Base:</strong> {html.escape(item.base_quant)}</li>",
            f"<li><strong>Head:</strong> {html.escape(item.head_quant)}</li>",
            f"<li><strong>Output Tok/s:</strong> {'N/A' if item.output_tps is None else f'{item.output_tps:.2f}'}</li>",
            f"<li><strong>n_predict:</strong> {'' if item.n_predict is None else item.n_predict}</li>",
            f"<li><strong>n_drafted:</strong> {'' if item.n_drafted is None else item.n_drafted}</li>",
            f"<li><strong>n_accept:</strong> {'' if item.n_accept is None else item.n_accept}</li>",
            f"<li><strong>Acceptance:</strong> {'N/A' if item.acceptance_rate is None else f'{item.acceptance_rate:.3f}%'} </li>",
            f"<li><strong>Log:</strong> <code>{html.escape(item.log_path)}</code></li>",
            f"<li><strong>Command:</strong> <code>{html.escape(item.command)}</code></li>",
        ]

        profile_html = ""
        if item.profile_blocks:
            total_ms = sum(block.summary.total_ms for block in item.profile_blocks)
            total_logits_ms = sum(block.summary.logits_ms for block in item.profile_blocks)
            total_score_ms = sum(block.summary.score_ms for block in item.profile_blocks)
            total_step_ms = sum(block.summary.step_ms for block in item.profile_blocks)
            total_output_tokens = sum(block.summary.output_tokens for block in item.profile_blocks)
            total_logits_calls = sum(block.summary.logits_calls for block in item.profile_blocks)
            total_step_calls = sum(block.summary.step_calls for block in item.profile_blocks)
            total_logits_gpu_ms = sum(block.summary.logits_gpu_ms or 0.0 for block in item.profile_blocks)
            total_step_gpu_ms = sum(block.summary.step_gpu_ms or 0.0 for block in item.profile_blocks)

            profile_bits = [
                f"<li><strong>Profile Blocks:</strong> {len(item.profile_blocks)}</li>",
                f"<li><strong>Profiled Output Tokens:</strong> {total_output_tokens}</li>",
                f"<li><strong>Total:</strong> {total_ms:.3f} ms</li>",
                f"<li><strong>Select:</strong> {total_logits_ms:.3f} ms</li>",
                f"<li><strong>Score:</strong> {total_score_ms:.3f} ms</li>",
                f"<li><strong>Step:</strong> {total_step_ms:.3f} ms</li>",
                f"<li><strong>Calls:</strong> logits={total_logits_calls}, step={total_step_calls}</li>",
            ]
            if any(block.summary.logits_gpu_ms is not None for block in item.profile_blocks):
                profile_bits.append(
                    f"<li><strong>GPU Totals:</strong> select={total_logits_gpu_ms:.3f} ms, step={total_step_gpu_ms:.3f} ms</li>"
                )

            block_rows = []
            block_sections = []
            for idx, block in enumerate(item.profile_blocks, start=1):
                ps = block.summary
                block_rows.append(
                    "<tr>"
                    f"<td>{idx}</td>"
                    f"<td>{ps.prompt_tokens}</td>"
                    f"<td>{ps.output_tokens}</td>"
                    f"<td>{ps.total_ms:.3f}</td>"
                    f"<td>{ps.logits_ms:.3f}</td>"
                    f"<td>{ps.score_ms:.3f}</td>"
                    f"<td>{ps.step_ms:.3f}</td>"
                    f"<td>{'N/A' if ps.root_step_ms is None else f'{ps.root_step_ms:.3f}'}</td>"
                    f"<td>{'' if ps.logits_gpu_ms is None else f'{ps.logits_gpu_ms:.3f}'}</td>"
                    f"<td>{'' if ps.step_gpu_ms is None else f'{ps.step_gpu_ms:.3f}'}</td>"
                    "</tr>"
                )

                depth_rows = []
                for depth in block.depths:
                    depth_rows.append(
                        "<tr>"
                        f"<td>{depth.depth}</td>"
                        f"<td>{depth.active_beams}</td>"
                        f"<td>{depth.expansions}</td>"
                        f"<td>{depth.select_ms:.3f}</td>"
                        f"<td>{depth.score_ms:.3f}</td>"
                        f"<td>{depth.step_ms:.3f}</td>"
                        f"<td>{'' if depth.select_gpu_ms is None else f'{depth.select_gpu_ms:.3f}'}</td>"
                        f"<td>{'' if depth.step_gpu_ms is None else f'{depth.step_gpu_ms:.3f}'}</td>"
                        "</tr>"
                    )

                block_sections.append(f"""
                <div class="block">
                  <h3>Profile Block {idx}</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>Depth</th>
                        <th>Active Beams</th>
                        <th>Expansions</th>
                        <th>Select ms</th>
                        <th>Score ms</th>
                        <th>Step ms</th>
                        <th>Select GPU ms</th>
                        <th>Step GPU ms</th>
                      </tr>
                    </thead>
                    <tbody>
                      {''.join(depth_rows)}
                    </tbody>
                  </table>
                </div>
                """)

            profile_html = f"""
            <div class="block">
              <h3>Profile Aggregate</h3>
              <ul>{''.join(profile_bits)}</ul>
            </div>
            <div class="block">
              <h3>Profile Blocks</h3>
              <table>
                <thead>
                  <tr>
                    <th>Block</th>
                    <th>Prompt Tok</th>
                    <th>Out Tok</th>
                    <th>Total ms</th>
                    <th>Select ms</th>
                    <th>Score ms</th>
                    <th>Step ms</th>
                    <th>Root Step ms</th>
                    <th>Select GPU ms</th>
                    <th>Step GPU ms</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(block_rows)}
                </tbody>
              </table>
            </div>
            {''.join(block_sections)}
            """

        sections.append(f"""
        <section class="card">
          <header>
            <h3>{html.escape(item.family)} / {html.escape(item.base_quant)} / {html.escape(item.head_quant)}</h3>
          </header>
          <div class="block">
            <h3>Run</h3>
            <ul>{''.join(metrics)}</ul>
          </div>
          {profile_html}
        </section>
        """)

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EAGLE Performance Detail Report</title>
  <style>
    :root {{
      --bg: #f2efe9;
      --fg: #1d1b18;
      --muted: #6c655c;
      --line: #d7cdc2;
      --card: #fffdf9;
      --accent: #8a4b27;
    }}
    body {{
      margin: 0;
      font-family: "Iosevka Etoile", "IBM Plex Sans", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(138, 75, 39, 0.08), transparent 30%),
        linear-gradient(180deg, #faf6f0 0%, var(--bg) 100%);
      color: var(--fg);
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px;
    }}
    h1, h2, h3 {{
      font-family: "IBM Plex Serif", serif;
    }}
    .meta, .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 12px 30px rgba(70, 45, 18, 0.06);
    }}
    .meta {{
      padding: 16px 18px;
      margin-bottom: 24px;
    }}
    .card {{
      padding: 20px 22px;
      margin: 18px 0 28px;
    }}
    .block {{
      margin-top: 14px;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
    }}
    code {{
      font-family: "Iosevka", monospace;
      font-size: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: rgba(138, 75, 39, 0.08);
    }}
    .muted {{
      color: var(--muted);
    }}
  </style>
</head>
<body>
<main>
  <h1>EAGLE CUDA Detailed Performance</h1>
  <div class="meta">
    <div><strong>Method:</strong> same benchmark matrix as the summary report, rerun with coarse EAGLE profiling enabled.</div>
    <div><strong>Prompt:</strong> <code>{html.escape(str(meta["prompt"]))}</code></div>
    <div><strong>Binary:</strong> <code>{html.escape(str(meta["binary"]))}</code></div>
    <div><strong>Build Archive:</strong> <code>{html.escape(str(meta["build_archive"]))}</code></div>
    <div><strong>Generated:</strong> {html.escape(str(meta["generated_at"]))}</div>
    <div class="muted">This report is section-based on purpose so the per-model counters are readable without scanning a wide table.</div>
  </div>
  {''.join(sections)}
</main>
</body>
</html>
"""
    report_path.write_text(report)


def build_cases(binary: Path, prompt: Path, n_predict: int, ctx: int) -> tuple[list[dict[str, str | list[str]]], str]:
    families = [
        {
            "family": "Llama-3.2-1B",
            "base": {
                "bf16": "/home/alvion/models/unsloth-Llama-3.2-1B-Instruct-bf16.gguf",
                "q8_0": "/home/alvion/models/unsloth-Llama-3.2-1B-Instruct-Q8_0.gguf",
                "q4_k_m": "/home/alvion/models/unsloth-Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            },
            "head": {
                "bf16": "/home/alvion/projects/kestrel/models/llama3-1b_eagle.bf16.gguf",
                "q8_0": "/home/alvion/projects/kestrel/models/llama3-1b_eagle.Q8_0.gguf",
                "q4_k_m": "/home/alvion/projects/kestrel/models/llama3-1b_eagle.Q4_K_M.gguf",
            },
        },
        {
            "family": "Qwen3-4B",
            "base": {
                "bf16": "/home/alvion/models/unsloth-Qwen3-4B-Instruct-2507-bf16.gguf",
                "q8_0": "/home/alvion/models/unsloth-Qwen3-4B-Instruct-2507-Q8_0.gguf",
                "q4_k_m": "/home/alvion/models/unsloth-Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            },
            "head": {
                "bf16": "/home/alvion/projects/kestrel/models/qwen3-4b_eagle.bf16.gguf",
                "q8_0": "/home/alvion/projects/kestrel/models/qwen3-4b_eagle.Q8_0.gguf",
                "q4_k_m": "/home/alvion/projects/kestrel/models/qwen3-4b_eagle.Q4_K_M.gguf",
            },
        },
        {
            "family": "Llama-3.1-8B",
            "base": {
                "bf16": "/home/alvion/models/unsloth-Meta-Llama-3.1-8B-Instruct-bf16.gguf",
                "q8_0": "/home/alvion/models/unsloth-Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
                "q4_k_m": "/home/alvion/models/unsloth-Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            },
            "head": {
                "bf16": "/home/alvion/projects/kestrel/models/llama3.1-8b_eagle.bf16.gguf",
                "q8_0": "/home/alvion/projects/kestrel/models/llama3.1-8b_eagle.Q8_0.gguf",
                "q4_k_m": "/home/alvion/projects/kestrel/models/llama3.1-8b_eagle.Q4_K_M.gguf",
            },
        },
    ]

    diagonal = {
        "bf16": ["bf16", "q8_0", "q4_k_m"],
        "q8_0": ["q8_0", "q4_k_m"],
        "q4_k_m": ["q4_k_m"],
    }

    base_common = [
        str(binary),
        "-f", str(prompt),
        "-n", str(n_predict),
        "-c", str(ctx),
        "-ngl", "999",
        "--temp", "0",
        "--seed", "123",
    ]

    cases: list[dict[str, str | list[str]]] = []
    for family in families:
        for base_quant in ["bf16", "q8_0", "q4_k_m"]:
            base_model = family["base"][base_quant]
            cases.append({
                "family": family["family"],
                "base_quant": base_quant,
                "head_quant": "none",
                "drafting": "off",
                "cmd": base_common + [
                    "-m", base_model,
                    "--spec-type", "none",
                ],
            })

            for head_quant in diagonal[base_quant]:
                head_model = family["head"][head_quant]
                cases.append({
                    "family": family["family"],
                    "base_quant": base_quant,
                    "head_quant": head_quant,
                    "drafting": "on",
                    "cmd": base_common + [
                        "-m", base_model,
                        "-md", head_model,
                        "--spec-type", "eagle3",
                        "--eagle-max-depth", "7",
                        "--eagle-max-proposals", "16",
                        "--eagle-beam-width", "8",
                        "--eagle-per-beam-topk-candidates", "128",
                        "-ngld", "999",
                    ],
                })

    args_text = (
        f"-n {n_predict} -c {ctx} -ngl 999 --temp 0 --seed 123; "
        "eagle: --eagle-max-depth 7 --eagle-max-proposals 16 --eagle-beam-width 8 "
        "--eagle-per-beam-topk-candidates 128 -ngld 999"
    )
    return cases, args_text


def benchmark_cases(cases: list[dict[str, str | list[str]]], log_dir: Path, *, profile: bool) -> list[RunResult]:
    results: list[RunResult] = []
    total_cases = len(cases)

    for idx, case in enumerate(cases, start=1):
        family = str(case["family"])
        base_quant = str(case["base_quant"])
        head_quant = str(case["head_quant"])
        drafting = str(case["drafting"])
        cmd = list(case["cmd"])
        log_path = log_dir / f"{family}_{base_quant}_{head_quant}.log"
        print(
            f"[{idx}/{total_cases}] {'profile' if profile else 'summary'} "
            f"{family} base={base_quant} head={head_quant}",
            flush=True,
        )
        rc, text = run_case(cmd, log_path, profile=profile)
        prompt_tokens, output_tokens, output_tps, n_vals, acceptance_rate = parse_metrics(text)
        profile_blocks = parse_profile(text) if profile else []
        results.append(RunResult(
            family=family,
            base_quant=base_quant,
            head_quant=head_quant,
            drafting=drafting,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            output_tps=output_tps,
            n_predict=n_vals.get("n_predict"),
            n_drafted=n_vals.get("n_drafted"),
            n_accept=n_vals.get("n_accept"),
            acceptance_rate=acceptance_rate,
            log_path=str(log_path),
            command=shlex.join(cmd),
            returncode=rc,
            profile_blocks=profile_blocks,
        ))

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-html", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-detail-html", required=True)
    parser.add_argument("--out-detail-json", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--detail-log-dir", required=True)
    parser.add_argument("--build-archive-dir", required=True)
    parser.add_argument("--build-dir", default="/home/alvion/projects/llama.cpp-x/build-cuda")
    parser.add_argument("--binary", default=None)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--ctx", type=int, default=4096)
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    binary = Path(args.binary) if args.binary else build_dir / "bin" / "llama-speculative-simple"
    prompt = Path(args.prompt)
    log_dir = Path(args.log_dir)
    detail_log_dir = Path(args.detail_log_dir)
    report_html = Path(args.out_html)
    report_json = Path(args.out_json)
    detail_html = Path(args.out_detail_html)
    detail_json = Path(args.out_detail_json)
    build_archive_dir = Path(args.build_archive_dir)

    log_dir.mkdir(parents=True, exist_ok=True)
    detail_log_dir.mkdir(parents=True, exist_ok=True)
    report_html.parent.mkdir(parents=True, exist_ok=True)
    detail_html.parent.mkdir(parents=True, exist_ok=True)

    build_meta = archive_build(build_dir, build_archive_dir)
    cases, args_text = build_cases(binary, prompt, args.n_predict, args.ctx)

    summary_results = benchmark_cases(cases, log_dir, profile=False)
    detail_results = benchmark_cases(cases, detail_log_dir, profile=True)

    report_json.write_text(json.dumps([asdict(item) for item in summary_results], indent=2))
    detail_json.write_text(json.dumps([asdict(item) for item in detail_results], indent=2))

    meta = {
        "prompt": str(prompt),
        "binary": str(binary),
        "args": args_text,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "build_archive": build_meta["archive_dir"],
        "detail_report": detail_html.name,
    }

    render_summary_html(summary_results, report_html, meta)
    render_detail_html(detail_results, detail_html, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
