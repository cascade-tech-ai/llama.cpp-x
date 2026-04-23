#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_enable_thinking(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid --enable-thinking value: {value!r}")


def _normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages") or record.get("conversations") or []
    out: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role") or message.get("from")
        content = message.get("content") or message.get("value") or ""
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        out.append({"role": str(role), "content": str(content)})
    return out


def _strip_trailing_assistant(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    trimmed = list(messages)
    while trimmed and trimmed[-1].get("role") == "assistant":
        trimmed.pop()
    return trimmed


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], *, enable_thinking: bool | None, **kwargs: Any) -> Any:
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def _load_records(dataset_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {dataset_path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object in {dataset_path}:{line_no}")
            records.append(payload)
    if not records:
        raise ValueError(f"dataset is empty: {dataset_path}")
    return records


def _select_records(records: list[dict[str, Any]], *, max_items: int, item_ids: list[int] | None) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(records, start=1))
    if item_ids:
        selected: list[tuple[int, dict[str, Any]]] = []
        for item_id in item_ids:
            if item_id < 1 or item_id > len(records):
                raise IndexError(f"--item-ids entries must be in [1, {len(records)}]; got {item_id}")
            selected.append((item_id, records[item_id - 1]))
        indexed = selected
    if max_items > 0:
        indexed = indexed[:max_items]
    return indexed


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_tokens(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_llama_metrics(output: str) -> dict[str, Any]:
    patterns: dict[str, tuple[str, Any]] = {
        "generated_count": (r"decoded\s+(\d+)\s+tokens in\s+[0-9.]+\s+seconds, speed:\s+[0-9.]+\s+t/s", int),
        "decode_seconds": (r"decoded\s+\d+\s+tokens in\s+([0-9.]+)\s+seconds, speed:\s+[0-9.]+\s+t/s", float),
        "generation_tps": (r"decoded\s+\d+\s+tokens in\s+[0-9.]+\s+seconds, speed:\s+([0-9.]+)\s+t/s", float),
        "n_predict": (r"n_predict\s+=\s+(\d+)", int),
        "n_drafted": (r"n_drafted\s+=\s+(\d+)", int),
        "n_accept": (r"n_accept\s+=\s+(\d+)", int),
        "accept": (r"accept\s+=\s+([0-9.]+)%", float),
        "acc_len": (r"acc_len\s+=\s+([0-9.]+)", float),
        "avg_depth": (r"avg_depth\s+=\s+([0-9.]+)", float),
    }
    parsed: dict[str, Any] = {}
    for key, (pattern, cast) in patterns.items():
        match = re.search(pattern, output)
        if not match:
            raise ValueError(f"could not parse {key} from llama-speculative-simple output")
        parsed[key] = cast(match.group(1))
    parsed["passes"] = max(0, int(parsed["n_predict"]) - int(parsed["n_accept"]))
    parsed["proposal_accept"] = (
        float(parsed["n_accept"]) / float(parsed["n_drafted"])
        if int(parsed["n_drafted"]) > 0
        else 0.0
    )
    return parsed


def _build_command(args: argparse.Namespace, *, prompt_text: str) -> list[str]:
    cmd = [
        str(args.llama_binary),
        "-m", str(args.model),
        "-p", prompt_text,
        "--no-conversation",
        "-n", str(args.max_new_tokens),
        "-c", str(args.ctx_size),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-ngl", str(args.gpu_layers),
        "--temp", str(args.temp),
        "--top-k", str(args.top_k),
        "--seed", str(args.seed),
        "-fa", args.flash_attn,
    ]

    if args.gpu_layers_draft is not None:
        cmd.extend(["-ngld", str(args.gpu_layers_draft)])
    if args.cache_type_k is not None:
        cmd.extend(["-ctk", args.cache_type_k])
    if args.cache_type_v is not None:
        cmd.extend(["-ctv", args.cache_type_v])
    if args.cache_type_k_draft is not None:
        cmd.extend(["-ctkd", args.cache_type_k_draft])
    if args.cache_type_v_draft is not None:
        cmd.extend(["-ctvd", args.cache_type_v_draft])
    if args.fit is not None:
        cmd.extend(["-fit", args.fit])

    if args.spec_type == "none":
        cmd.extend(["--spec-type", "none"])
    else:
        if not args.draft_model:
            raise ValueError("--draft-model is required when --spec-type=eagle3")
        cmd.extend([
            "--spec-type", "eagle3",
            "-md", str(args.draft_model),
            "--eagle-max-depth", str(args.max_depth),
            "--eagle-adaptive-depth", str(args.eagle_adaptive_depth),
        ])
        if args.eval_mode == "serial":
            cmd.append("--eagle-serial")
        elif args.eval_mode == "beam_prefix":
            cmd.extend([
                "--eagle-max-proposals", str(args.max_proposals),
                "--eagle-beam-width", str(args.eagle_beam_width),
                "--eagle-per-beam-topk-candidates", str(args.eagle_per_beam_topk_candidates),
            ])
        else:
            raise ValueError("--eval-mode=legacy_topk is not supported by llama-speculative-simple")

    extra: list[str] = []
    if args.llama_args:
        extra = list(args.llama_args)
        if extra and extra[0] == "--":
            extra = extra[1:]

    # The wrapper renders the full chat template itself, including BOS when the
    # HF tokenizer emits it. Avoid a duplicate BOS from GGUF tokenization unless
    # the caller explicitly overrides that metadata key.
    has_bos_override = False
    for i, token in enumerate(extra):
        if token == "--override-kv" and i + 1 < len(extra):
            if "tokenizer.ggml.add_bos_token" in extra[i + 1]:
                has_bos_override = True
                break
        if token.startswith("tokenizer.ggml.add_bos_token="):
            has_bos_override = True
            break
    if not has_bos_override:
        cmd.extend(["--override-kv", "tokenizer.ggml.add_bos_token=bool:false"])

    cmd.extend(extra)
    return cmd


def _prompt_payload(tokenizer: Any, record: dict[str, Any], *, enable_thinking: bool | None) -> dict[str, Any]:
    messages = _strip_trailing_assistant(_normalize_messages(record))
    if not messages:
        raise ValueError("record produced no prompt messages after stripping trailing assistant turns")
    prompt_text = _apply_chat_template(
        tokenizer,
        messages,
        enable_thinking=enable_thinking,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = _apply_chat_template(
        tokenizer,
        messages,
        enable_thinking=enable_thinking,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(prompt_tokens, "tolist"):
        prompt_tokens = prompt_tokens.tolist()
    if prompt_tokens and isinstance(prompt_tokens[0], list):
        prompt_tokens = prompt_tokens[0]
    prompt_tokens = [int(tok) for tok in prompt_tokens]
    return {
        "messages": messages,
        "prompt_text": str(prompt_text),
        "prompt_tokens": prompt_tokens,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark llama-speculative-simple over a JSONL dataset using the same prompt construction "
            "semantics as kestrel train.py eval. Shared defaults intentionally match Kestrel eval; "
            "llama.cpp runtime knobs have explicit local defaults."
        )
    )
    parser.add_argument("--dataset", required=True, help="JSONL dataset path.")
    parser.add_argument("--base-model", required=True, help="HF model id or path used for chat templating.")
    parser.add_argument("--model", required=True, help="Base GGUF passed to llama-speculative-simple.")
    parser.add_argument("--draft-model", default=None, help="Draft/head GGUF for speculative decoding.")
    parser.add_argument("--llama-binary", default="build/bin/llama-speculative-simple", help="llama-speculative-simple path.")
    parser.add_argument("--out-json", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--out-dir", default=None, help="Optional directory for per-sample logs and prompts.")
    parser.add_argument("--trace-yaml-dir", default=None, help="Optional directory for per-sample trace YAML/JSON files.")

    # Shared semantics: keep these aligned with kestrel train.py eval.
    parser.add_argument("--spec-type", choices=("eagle3", "none"), default="eagle3", help="Speculative mode. Default matches train.py eval behavior.")
    parser.add_argument("--eval-mode", choices=("beam_prefix", "legacy_topk", "serial"), default="beam_prefix", help="Shared acceptance mode. legacy_topk is unsupported in llama.cpp and will error.")
    parser.add_argument("--max-proposals", type=int, default=8, help="Shared default aligned with kestrel train.py eval.")
    parser.add_argument("--max-depth", type=int, default=7, help="Shared default aligned with kestrel train.py eval.")
    parser.add_argument("--max-items", type=int, default=-1, help="Shared default aligned with kestrel train.py eval.")
    parser.add_argument("--item-ids", type=int, nargs="+", default=None, help="1-based dataset item indices, same convention as kestrel train.py eval.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Shared default aligned with kestrel train.py eval.")
    parser.add_argument("--prob-threshold", type=float, default=1e-4, help="Recorded for parity metadata. llama.cpp does not expose the exact kestrel threshold control.")
    parser.add_argument("--seed", type=int, default=42, help="Shared default aligned with kestrel train.py eval.")
    parser.add_argument("--temp", type=float, default=0.0, help="Shared default aligned with kestrel train.py eval.")
    parser.add_argument("--top-k", type=int, default=1024, help="Top-k is a truncation threshold, not a filter. Default 1024 bounds sampler work; 0 would enumerate the full vocab (~150k on Qwen) at every step.")
    parser.add_argument("--enable-thinking", default=None, help="Pass true/false/empty through tokenizer.apply_chat_template like kestrel train.py eval.")

    # llama.cpp-only defaults.
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ubatch-size", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--gpu-layers-draft", type=int, default=999)
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"), default="on")
    parser.add_argument("--eagle-adaptive-depth", type=float, default=0.05)
    parser.add_argument("--eagle-beam-width", type=int, default=8)
    parser.add_argument("--eagle-per-beam-topk-candidates", type=int, default=128)
    parser.add_argument("--cache-type-k", default=None)
    parser.add_argument("--cache-type-v", default=None)
    parser.add_argument("--cache-type-k-draft", default=None)
    parser.add_argument("--cache-type-v-draft", default=None)
    parser.add_argument("--fit", choices=("on", "off"), default=None)
    parser.add_argument("llama_args", nargs=argparse.REMAINDER, help="Additional args forwarded verbatim to llama-speculative-simple after '--'.")

    args = parser.parse_args()
    args.enable_thinking = _parse_enable_thinking(args.enable_thinking)

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset file not found: {dataset_path}")
    args.model = Path(args.model).expanduser().resolve()
    if not args.model.is_file():
        raise FileNotFoundError(f"model file not found: {args.model}")
    if args.draft_model is not None:
        args.draft_model = Path(args.draft_model).expanduser().resolve()
        if not args.draft_model.is_file():
            raise FileNotFoundError(f"draft model file not found: {args.draft_model}")
    args.llama_binary = Path(args.llama_binary).expanduser().resolve()
    if not args.llama_binary.is_file():
        raise FileNotFoundError(f"llama binary not found: {args.llama_binary}")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required for chat templating in benchmark_llama_speculative.py") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    records = _load_records(dataset_path)
    selected_records = _select_records(records, max_items=args.max_items, item_ids=args.item_ids)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    trace_dir = Path(args.trace_yaml_dir).expanduser().resolve() if args.trace_yaml_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)

    sample_metrics: list[dict[str, Any]] = []
    total_passes = 0
    total_proposed = 0
    total_accepted = 0
    total_generated = 0
    total_decode_seconds = 0.0

    for sample_ordinal, (item_id, record) in enumerate(selected_records, start=1):
        prompt = _prompt_payload(tokenizer, record, enable_thinking=args.enable_thinking)
        sample_name = f"sample_{item_id:03d}"

        prompt_path = out_dir / f"{sample_name}.prompt.txt" if out_dir else None
        if prompt_path:
            prompt_path.write_text(prompt["prompt_text"], encoding="utf-8")

        command_for_report: list[str] | None = None

        cmd = _build_command(args, prompt_text=prompt["prompt_text"])
        command_for_report = list(cmd)
        try:
            prompt_index = command_for_report.index("-p") + 1
            command_for_report[prompt_index] = str(prompt_path) if prompt_path else "<rendered-prompt>"
        except (ValueError, IndexError):
            pass
        if trace_dir and args.spec_type != "none":
            cmd.extend(["--eagle-trace-yaml", str(trace_dir / f"{sample_name}.trace.yaml")])
            if command_for_report is not None:
                command_for_report.extend(["--eagle-trace-yaml", str(trace_dir / f"{sample_name}.trace.yaml")])

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        output = proc.stdout

        if out_dir:
            (out_dir / f"{sample_name}.log").write_text(output, encoding="utf-8")

        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-speculative-simple failed for dataset item {item_id} with exit code {proc.returncode}\n"
                f"{output[-4000:]}"
            )

        metrics = _parse_llama_metrics(output)
        total_passes += int(metrics["passes"])
        total_proposed += int(metrics["n_drafted"])
        total_accepted += int(metrics["n_accept"])
        total_generated += int(metrics["generated_count"])
        total_decode_seconds += float(metrics["decode_seconds"])

        sample_payload = {
            "sample": sample_ordinal,
            "item_id": item_id,
            "prompt_char_count": len(prompt["prompt_text"]),
            "prompt_token_count": len(prompt["prompt_tokens"]),
            "prompt_sha256": _sha256_text(prompt["prompt_text"]),
            "prompt_token_sha256": _sha256_tokens(prompt["prompt_tokens"]),
            "messages": prompt["messages"],
            "command": command_for_report,
            "spec_type": args.spec_type,
            "eval_mode": args.eval_mode,
            **metrics,
        }
        sample_metrics.append(sample_payload)

        print(
            f"{sample_name}: passes={metrics['passes']} acc_len={metrics['acc_len']:.3f} "
            f"proposed={metrics['n_drafted']} accepted={metrics['n_accept']} "
            f"proposal_accept={metrics['proposal_accept']:.4f} tps={metrics['generation_tps']:.3f}"
        )

    overall_accept_len = total_accepted / max(1, total_passes)
    overall_proposal_accept = total_accepted / max(1, total_proposed)
    overall_generation_tps = total_generated / total_decode_seconds if total_decode_seconds > 0 else 0.0

    summary = {
        "dataset": str(dataset_path),
        "base_model": args.base_model,
        "model": str(args.model),
        "draft_model": str(args.draft_model) if args.draft_model else None,
        "llama_binary": str(args.llama_binary),
        "spec_type": args.spec_type,
        "eval_mode": args.eval_mode,
        "seed": args.seed,
        "max_items": args.max_items,
        "item_ids": args.item_ids,
        "max_new_tokens": args.max_new_tokens,
        "max_depth": args.max_depth,
        "max_proposals": args.max_proposals,
        "prob_threshold": args.prob_threshold,
        "temperature": args.temp,
        "top_k": args.top_k,
        "enable_thinking": args.enable_thinking,
        "ctx_size": args.ctx_size,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "gpu_layers": args.gpu_layers,
        "gpu_layers_draft": args.gpu_layers_draft,
        "flash_attn": args.flash_attn,
        "eagle_adaptive_depth": args.eagle_adaptive_depth,
        "eagle_beam_width": args.eagle_beam_width,
        "eagle_per_beam_topk_candidates": args.eagle_per_beam_topk_candidates,
        "cache_type_k": args.cache_type_k,
        "cache_type_v": args.cache_type_v,
        "cache_type_k_draft": args.cache_type_k_draft,
        "cache_type_v_draft": args.cache_type_v_draft,
        "fit": args.fit,
        "llama_args": args.llama_args[1:] if args.llama_args and args.llama_args[0] == "--" else args.llama_args,
        "total_samples": len(sample_metrics),
        "total_passes": total_passes,
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
        "overall_accept_len": overall_accept_len,
        "overall_proposal_accept": overall_proposal_accept,
        "overall_generation_tps": overall_generation_tps,
        "sample_metrics": sample_metrics,
    }

    if args.out_json:
        _write_json(Path(args.out_json).expanduser().resolve(), summary)

    print(
        f"total: passes={total_passes} acc_len={overall_accept_len:.3f} proposed={total_proposed} "
        f"accepted={total_accepted} proposal_accept={overall_proposal_accept:.4f} "
        f"tps={overall_generation_tps:.3f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[benchmark] interrupted", file=sys.stderr)
        raise SystemExit(130)
