#!/usr/bin/env python3
# AI-GENERATED: This file was created with AI assistance for an experimental fork.
# DO NOT SUBMIT upstream unless rewritten or exhaustively reviewed by a human.
"""
Run sequential benchmark requests from a Kestrel dataset against a running llama-server.

The script runs the same N prompts twice:
1) Baseline mode (`speculative.type=none`)
2) EAGLE3 mode (`speculative.type=eagle3`)

It compares end-to-end latency and reports speculative accepted/rejected counts.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class SampleResult:
    dataset_index: int
    request_index: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int
    draft_n: int
    draft_n_accepted: int
    prompt_ms: float
    predicted_ms: float


@dataclass
class ModeSummary:
    name: str
    n: int
    total_latency_ms: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int
    acceptance_rate: float
    requests_per_second: float
    completion_tokens_per_second: float
    server_prompt_ms_total: float
    server_predicted_ms_total: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs EAGLE3 using a running llama-server.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080", help="Base URL of llama-server.")
    parser.add_argument(
        "--mode",
        choices=("both", "baseline", "eagle3"),
        default="both",
        help="Which benchmark mode(s) to run.",
    )
    parser.add_argument(
        "--endpoint",
        default="/v1/chat/completions",
        help="HTTP endpoint to call (default: /v1/chat/completions).",
    )
    parser.add_argument("--api-key", default=None, help="Optional OpenAI-style bearer token.")
    parser.add_argument("--model", default="qwen3-4b", help="Model name sent in request body.")
    parser.add_argument(
        "--dataset",
        default="perfect_10k",
        help="Dataset name (resolved under kestrel/datasets) or explicit .jsonl path.",
    )
    parser.add_argument(
        "--message-mode",
        choices=("last-user", "full"),
        default="last-user",
        help="How to build chat messages from each dataset row.",
    )
    parser.add_argument(
        "--max-message-chars",
        type=int,
        default=None,
        help="Skip dataset rows whose combined message content exceeds this many characters.",
    )
    parser.add_argument(
        "--kestrel-path",
        default="~/projects/kestrel",
        help="Path to kestrel repo (default: ~/projects/kestrel).",
    )
    parser.add_argument("-n", "--num-prompts", type=int, default=100, help="Measured prompts per mode.")
    parser.add_argument("--offset", type=int, default=0, help="Start offset in dataset.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup requests per mode (not measured).")
    parser.add_argument("--max-tokens", type=int, default=128, help="`max_tokens` request field.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for reproducibility.")
    parser.add_argument("--timeout-s", type=float, default=600.0, help="HTTP timeout per request.")
    parser.add_argument("--progress-every", type=int, default=10, help="Progress print interval.")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")

    # Optional EAGLE3 request overrides
    parser.add_argument("--eagle-n-min", type=int, default=None)
    parser.add_argument("--eagle-n-max", type=int, default=None)
    parser.add_argument("--eagle-p-min", type=float, default=None)
    parser.add_argument("--eagle-max-depth", type=int, default=None)
    parser.add_argument("--eagle-max-proposals", type=int, default=None)
    parser.add_argument("--eagle-beam-width", type=int, default=None)
    parser.add_argument("--eagle-per-beam-topk-candidates", type=int, default=None)

    args = parser.parse_args()
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    return args


def resolve_dataset_path(dataset: str, kestrel_path: str) -> Path:
    dataset_path = Path(dataset).expanduser()
    if dataset_path.exists():
        return dataset_path.resolve()

    base = Path(kestrel_path).expanduser().resolve() / "datasets"
    if dataset_path.suffix == "":
        candidate = base / f"{dataset}.jsonl"
    else:
        candidate = base / dataset
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not find dataset path for '{dataset}' (checked {candidate})")


def _extract_messages(messages: List[Dict[str, Any]], mode: str, line_number: int) -> List[Dict[str, Any]]:
    if mode == "full":
        return messages

    if mode == "last-user":
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
                return [{"role": "user", "content": msg["content"]}]
        raise ValueError(f"Line {line_number}: no user message found for --message-mode=last-user")

    raise ValueError(f"Unsupported message mode: {mode}")


def _messages_char_len(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(str(part.get("text", "")))
                else:
                    total += len(str(part))
        else:
            total += len(str(content))
    return total


def load_dataset_entries(
    dataset_path: Path,
    offset: int,
    count: int,
    message_mode: str,
    max_message_chars: int | None,
) -> List[Tuple[int, List[Dict[str, Any]]]]:
    out: List[Tuple[int, List[Dict[str, Any]]]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx < offset:
                continue
            if len(out) >= count:
                break

            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"Line {line_idx + 1}: missing list field 'messages'")
            selected_messages = _extract_messages(messages, message_mode, line_idx + 1)
            if max_message_chars is not None:
                if _messages_char_len(selected_messages) > max_message_chars:
                    continue
            out.append((line_idx, selected_messages))

    if len(out) < count:
        raise ValueError(
            f"Dataset only yielded {len(out)} entries from offset {offset}, need {count}. "
            f"File: {dataset_path}"
        )
    return out


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    w = rank - lo
    return ordered[lo] * (1.0 - w) + ordered[hi] * w


def post_json(url: str, payload: Dict[str, Any], api_key: str | None, timeout_s: float) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return int(resp.status), json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"raw": raw}
        return int(e.code), body
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError) as e:
        raise RuntimeError(f"Request failed: {e}") from e


def build_payload(args: argparse.Namespace, messages: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "stream": False,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        # disable prefix cache reuse for clean latency comparison
        "cache_prompt": False,
        "n_cache_reuse": 0,
        "timings_per_token": True,
    }

    if mode == "baseline":
        payload["speculative.type"] = "none"
        # In current server scheduling, n_max controls whether drafting runs.
        # Set to 0 to force true non-speculative baseline when a draft model is loaded.
        payload["speculative.n_max"] = 0
        payload["speculative.n_min"] = 0
    elif mode == "eagle3":
        payload["speculative.type"] = "eagle3"
        optional = {
            "speculative.n_min": args.eagle_n_min,
            "speculative.n_max": args.eagle_n_max,
            "speculative.p_min": args.eagle_p_min,
            "speculative.eagle_max_depth": args.eagle_max_depth,
            "speculative.eagle_max_proposals": args.eagle_max_proposals,
            "speculative.eagle_beam_width": args.eagle_beam_width,
            "speculative.eagle_per_beam_topk_candidates": args.eagle_per_beam_topk_candidates,
        }
        for key, value in optional.items():
            if value is not None:
                payload[key] = value
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return payload


def parse_response_metrics(body: Dict[str, Any]) -> Dict[str, Any]:
    usage = body.get("usage") or {}
    completion_tokens_details = usage.get("completion_tokens_details") or {}
    timings = body.get("timings") or {}

    prompt_tokens = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)

    accepted = int(completion_tokens_details.get("accepted_prediction_tokens", 0) or 0)
    rejected = int(completion_tokens_details.get("rejected_prediction_tokens", 0) or 0)

    draft_n = int(timings.get("draft_n", 0) or 0)
    draft_n_accepted = int(timings.get("draft_n_accepted", 0) or 0)

    # Fallback when completion_tokens_details is absent.
    if accepted == 0 and rejected == 0 and draft_n > 0:
        accepted = max(0, draft_n_accepted)
        rejected = max(0, draft_n - accepted)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "accepted_prediction_tokens": accepted,
        "rejected_prediction_tokens": rejected,
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
        "prompt_ms": float(timings.get("prompt_ms", 0.0) or 0.0),
        "predicted_ms": float(timings.get("predicted_ms", 0.0) or 0.0),
    }


def summarize_mode(name: str, samples: List[SampleResult]) -> ModeSummary:
    if not samples:
        raise ValueError(f"No samples recorded for mode: {name}")

    latencies = [s.latency_ms for s in samples]
    total_latency_ms = sum(latencies)
    total_prompt_tokens = sum(s.prompt_tokens for s in samples)
    total_completion_tokens = sum(s.completion_tokens for s in samples)
    total_tokens = sum(s.total_tokens for s in samples)
    accepted_prediction_tokens = sum(s.accepted_prediction_tokens for s in samples)
    rejected_prediction_tokens = sum(s.rejected_prediction_tokens for s in samples)
    denom = accepted_prediction_tokens + rejected_prediction_tokens
    acceptance_rate = (accepted_prediction_tokens / denom) if denom > 0 else 0.0
    requests_per_second = len(samples) * 1000.0 / total_latency_ms
    completion_tokens_per_second = total_completion_tokens * 1000.0 / total_latency_ms
    server_prompt_ms_total = sum(s.prompt_ms for s in samples)
    server_predicted_ms_total = sum(s.predicted_ms for s in samples)

    return ModeSummary(
        name=name,
        n=len(samples),
        total_latency_ms=total_latency_ms,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=percentile(latencies, 0.50),
        p95_latency_ms=percentile(latencies, 0.95),
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        accepted_prediction_tokens=accepted_prediction_tokens,
        rejected_prediction_tokens=rejected_prediction_tokens,
        acceptance_rate=acceptance_rate,
        requests_per_second=requests_per_second,
        completion_tokens_per_second=completion_tokens_per_second,
        server_prompt_ms_total=server_prompt_ms_total,
        server_predicted_ms_total=server_predicted_ms_total,
    )


def run_mode(
    mode: str,
    args: argparse.Namespace,
    url: str,
    entries: Iterable[Tuple[int, List[Dict[str, Any]]]],
) -> Tuple[ModeSummary, List[SampleResult]]:
    all_entries = list(entries)
    measured: List[SampleResult] = []

    print(f"\n[{mode}] warmup={args.warmup}, measured={args.num_prompts}")
    for i, (dataset_idx, messages) in enumerate(all_entries):
        payload = build_payload(args, messages, mode)
        t0 = time.perf_counter()
        status, body = post_json(url, payload, args.api_key, args.timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if status != 200:
            raise RuntimeError(
                f"{mode} request failed at dataset_index={dataset_idx} request={i + 1}/{len(all_entries)} "
                f"status={status} body={json.dumps(body)[:2000]}"
            )

        if i < args.warmup:
            continue

        metrics = parse_response_metrics(body)
        sample = SampleResult(
            dataset_index=dataset_idx,
            request_index=i - args.warmup,
            latency_ms=latency_ms,
            prompt_tokens=metrics["prompt_tokens"],
            completion_tokens=metrics["completion_tokens"],
            total_tokens=metrics["total_tokens"],
            accepted_prediction_tokens=metrics["accepted_prediction_tokens"],
            rejected_prediction_tokens=metrics["rejected_prediction_tokens"],
            draft_n=metrics["draft_n"],
            draft_n_accepted=metrics["draft_n_accepted"],
            prompt_ms=metrics["prompt_ms"],
            predicted_ms=metrics["predicted_ms"],
        )
        measured.append(sample)

        if args.progress_every > 0 and (len(measured) % args.progress_every == 0 or len(measured) == args.num_prompts):
            avg_so_far = statistics.mean([s.latency_ms for s in measured])
            print(
                f"[{mode}] {len(measured):4d}/{args.num_prompts} "
                f"last={latency_ms:8.2f} ms avg={avg_so_far:8.2f} ms"
            )

    summary = summarize_mode(mode, measured)
    return summary, measured


def print_summary(summary: ModeSummary) -> None:
    print(f"\n== {summary.name.upper()} ==")
    print(f"requests:                 {summary.n}")
    print(f"total latency:            {summary.total_latency_ms:.2f} ms")
    print(f"avg latency:              {summary.avg_latency_ms:.2f} ms")
    print(f"p50 latency:              {summary.p50_latency_ms:.2f} ms")
    print(f"p95 latency:              {summary.p95_latency_ms:.2f} ms")
    print(f"prompt tokens:            {summary.total_prompt_tokens}")
    print(f"completion tokens:        {summary.total_completion_tokens}")
    print(f"total tokens:             {summary.total_tokens}")
    print(f"accepted pred tokens:     {summary.accepted_prediction_tokens}")
    print(f"rejected pred tokens:     {summary.rejected_prediction_tokens}")
    print(f"acceptance rate:          {summary.acceptance_rate * 100.0:.2f}%")
    print(f"requests/sec:             {summary.requests_per_second:.3f}")
    print(f"completion toks/sec:      {summary.completion_tokens_per_second:.3f}")
    print(f"server prompt ms total:   {summary.server_prompt_ms_total:.2f}")
    print(f"server predict ms total:  {summary.server_predicted_ms_total:.2f}")


def main() -> int:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset, args.kestrel_path)
    total_needed = args.warmup + args.num_prompts
    entries = load_dataset_entries(
        dataset_path,
        args.offset,
        total_needed,
        args.message_mode,
        args.max_message_chars,
    )
    url = f"{args.server_url.rstrip('/')}{args.endpoint}"

    print("Benchmark configuration:")
    print(f"  server:        {url}")
    print(f"  dataset:       {dataset_path}")
    print(f"  offset:        {args.offset}")
    print(f"  warmup:        {args.warmup}")
    print(f"  measured N:    {args.num_prompts}")
    print(f"  max_tokens:    {args.max_tokens}")
    print(f"  temperature:   {args.temperature}")
    print(f"  message_mode:  {args.message_mode}")
    if args.max_message_chars is not None:
        print(f"  max_msg_chars: {args.max_message_chars}")
    print("  cache_prompt:  false")
    print("  n_cache_reuse: 0")

    baseline_summary = None
    baseline_samples: List[SampleResult] = []
    eagle_summary = None
    eagle_samples: List[SampleResult] = []

    if args.mode in ("both", "baseline"):
        baseline_summary, baseline_samples = run_mode("baseline", args, url, entries)
        print_summary(baseline_summary)

    if args.mode in ("both", "eagle3"):
        eagle_summary, eagle_samples = run_mode("eagle3", args, url, entries)
        print_summary(eagle_summary)

    latency_speedup = None
    req_per_sec_ratio = None
    tok_per_sec_ratio = None
    if baseline_summary is not None and eagle_summary is not None:
        latency_speedup = baseline_summary.total_latency_ms / eagle_summary.total_latency_ms
        req_per_sec_ratio = eagle_summary.requests_per_second / baseline_summary.requests_per_second
        tok_per_sec_ratio = eagle_summary.completion_tokens_per_second / baseline_summary.completion_tokens_per_second

        print("\n== COMPARISON (EAGLE3 vs BASELINE) ==")
        print(f"total latency speedup:    {latency_speedup:.4f}x")
        print(f"requests/sec ratio:       {req_per_sec_ratio:.4f}x")
        print(f"completion toks/sec:      {tok_per_sec_ratio:.4f}x")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        payload = {
            "config": {
                "server_url": args.server_url,
                "endpoint": args.endpoint,
                "dataset": str(dataset_path),
                "offset": args.offset,
                "warmup": args.warmup,
                "num_prompts": args.num_prompts,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
                "message_mode": args.message_mode,
                "max_message_chars": args.max_message_chars,
                "cache_prompt": False,
                "n_cache_reuse": 0,
            },
            "mode": args.mode,
            "comparison": {
                "total_latency_speedup": latency_speedup,
                "requests_per_second_ratio": req_per_sec_ratio,
                "completion_tokens_per_second_ratio": tok_per_sec_ratio,
            },
        }
        if baseline_summary is not None:
            payload["baseline"] = {
                "summary": asdict(baseline_summary),
                "samples": [asdict(s) for s in baseline_samples],
            }
        if eagle_summary is not None:
            payload["eagle3"] = {
                "summary": asdict(eagle_summary),
                "samples": [asdict(s) for s in eagle_samples],
            }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote benchmark report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
