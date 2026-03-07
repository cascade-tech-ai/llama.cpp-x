#!/usr/bin/env python3
# AI-GENERATED: This file was created with AI assistance for an experimental fork.
# DO NOT SUBMIT upstream unless rewritten or exhaustively reviewed by a human.
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


RE_DECODED = re.compile(r"decoded\s+(\d+)\s+tokens.*speed:\s+([0-9.]+)\s+t/s")
RE_ENCODED = re.compile(r"encoded\s+(\d+)\s+tokens.*speed:\s+([0-9.]+)\s+t/s")
RE_ACCEPT = re.compile(r"accept\s+=\s+([0-9.]+)%")
RE_N_VAL = re.compile(r"^(n_predict|n_drafted|n_accept)\s+=\s+(\d+)$", re.MULTILINE)


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


def run_case(cmd: list[str], log_path: Path) -> tuple[int, str]:
    env = os.environ.copy()
    for key in [
        "CASCADE_EAGLE_PROFILE",
        "CASCADE_EAGLE_PROFILE_GPU",
        "CASCADE_EAGLE_DUMP_DIR",
        "GGML_CUDA_TIMING",
    ]:
        env.pop(key, None)

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


def render_html(results: list[RunResult], report_path: Path, meta: dict[str, object]) -> None:
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
            f"<td>{item.returncode}</td>"
            f"<td><code>{html.escape(item.log_path)}</code></td>"
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
      --accent: #8f3b1b;
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
    .meta code, td code {{
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
    <div><strong>Method:</strong> CUDA-only, no debug sync env vars, greedy decode, chat templating through <code>llama-speculative-simple</code>.</div>
    <div><strong>Prompt:</strong> <code>{html.escape(str(meta["prompt"]))}</code></div>
    <div><strong>Binary:</strong> <code>{html.escape(str(meta["binary"]))}</code></div>
    <div><strong>Args:</strong> <code>{html.escape(meta["args"])}</code></div>
    <div><strong>Generated:</strong> {html.escape(meta["generated_at"])}</div>
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
        <th>RC</th>
        <th>Log</th>
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-html", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--ctx", type=int, default=4096)
    args = parser.parse_args()

    binary = Path("/home/alvion/projects/llama.cpp-x/build-cuda/bin/llama-speculative-simple")
    prompt = Path(args.prompt)
    log_dir = Path(args.log_dir)
    report_html = Path(args.out_html)
    report_json = Path(args.out_json)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_html.parent.mkdir(parents=True, exist_ok=True)

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
        "-n", str(args.n_predict),
        "-c", str(args.ctx),
        "-ngl", "999",
        "--temp", "0",
        "--seed", "123",
    ]

    results: list[RunResult] = []
    total_cases = sum(1 + len(diagonal[bq]) for _ in families for bq in ["bf16", "q8_0", "q4_k_m"])
    case_idx = 0

    for family in families:
        for base_quant in ["bf16", "q8_0", "q4_k_m"]:
            case_idx += 1
            base_model = family["base"][base_quant]
            baseline_cmd = base_common + [
                "-m", base_model,
                "--spec-type", "none",
            ]
            log_path = log_dir / f"{family['family']}_{base_quant}_none.log"
            print(f"[{case_idx}/{total_cases}] {family['family']} base={base_quant} head=none", flush=True)
            rc, text = run_case(baseline_cmd, log_path)
            prompt_tokens, output_tokens, output_tps, n_vals, acceptance_rate = parse_metrics(text)
            results.append(RunResult(
                family=family["family"],
                base_quant=base_quant,
                head_quant="none",
                drafting="off",
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                output_tps=output_tps,
                n_predict=n_vals.get("n_predict"),
                n_drafted=n_vals.get("n_drafted"),
                n_accept=n_vals.get("n_accept"),
                acceptance_rate=acceptance_rate,
                log_path=str(log_path),
                command=shlex.join(baseline_cmd),
                returncode=rc,
            ))

            for head_quant in diagonal[base_quant]:
                case_idx += 1
                head_model = family["head"][head_quant]
                eagle_cmd = base_common + [
                    "-m", base_model,
                    "-md", head_model,
                    "--spec-type", "eagle3",
                    "--eagle-max-depth", "7",
                    "--eagle-max-proposals", "16",
                    "--eagle-beam-width", "8",
                    "-ngld", "999",
                ]
                log_path = log_dir / f"{family['family']}_{base_quant}_{head_quant}.log"
                print(f"[{case_idx}/{total_cases}] {family['family']} base={base_quant} head={head_quant}", flush=True)
                rc, text = run_case(eagle_cmd, log_path)
                prompt_tokens, output_tokens, output_tps, n_vals, acceptance_rate = parse_metrics(text)
                results.append(RunResult(
                    family=family["family"],
                    base_quant=base_quant,
                    head_quant=head_quant,
                    drafting="on",
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    output_tps=output_tps,
                    n_predict=n_vals.get("n_predict"),
                    n_drafted=n_vals.get("n_drafted"),
                    n_accept=n_vals.get("n_accept"),
                    acceptance_rate=acceptance_rate,
                    log_path=str(log_path),
                    command=shlex.join(eagle_cmd),
                    returncode=rc,
                ))

    payload = [asdict(item) for item in results]
    report_json.write_text(json.dumps(payload, indent=2))
    render_html(
        results,
        report_html,
        {
            "prompt": str(prompt),
            "binary": str(binary),
            "args": "-n %d -c %d -ngl 999 --temp 0 --seed 123; eagle: --eagle-max-depth 7 --eagle-max-proposals 16 --eagle-beam-width 8 -ngld 999" % (args.n_predict, args.ctx),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
