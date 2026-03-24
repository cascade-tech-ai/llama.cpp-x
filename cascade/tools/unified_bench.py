#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml


def sha256_tokens(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_messages(dataset_path: Path, entry_idx: int) -> list[dict[str, str]]:
    records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    record = records[entry_idx]
    messages = record.get("messages") or []
    out: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", ""))
        if role:
            out.append({"role": role, "content": content})
    while out and out[-1]["role"].lower() == "assistant":
        out.pop()
    if not out:
        raise ValueError(f"dataset entry {entry_idx} produced no prompt messages")
    return out


def render_prompt(base_model_hf: str, dataset_path: Path, entry_idx: int) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_hf, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    messages = load_messages(dataset_path, entry_idx)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return {
        "messages": messages,
        "prompt": prompt_text,
        "prompt_tokens": [int(tok) for tok in prompt_tokens],
    }


def parse_llama_spec_stdout(text: str) -> dict[str, Any]:
    speed = re.search(r"decoded\s+(\d+)\s+tokens in\s+([0-9.]+)\s+seconds, speed:\s+([0-9.]+)\s+t/s", text)
    n_accept = re.search(r"n_accept\s+=\s+(\d+)", text)
    n_drafted = re.search(r"n_drafted\s+=\s+(\d+)", text)
    if not speed:
        raise ValueError("could not parse speculative llama decode speed")
    return {
        "generated_count": int(speed.group(1)),
        "decode_seconds": float(speed.group(2)),
        "generation_tps": float(speed.group(3)),
        "n_accept": int(n_accept.group(1)) if n_accept else None,
        "n_drafted": int(n_drafted.group(1)) if n_drafted else None,
    }


def run_llama_spec(args: argparse.Namespace, prompt_info: dict[str, Any], persist_trace_path: Path | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="eagle-trace-") as tmpdir:
        trace_path = Path(tmpdir) / "llama_trace.yaml"
        cmd = [
            args.llama_spec_binary,
            "-m", args.base_model_gguf,
            "--override-kv", "tokenizer.ggml.add_bos_token=bool:false",
            "--model-draft", args.head_model_gguf,
            "--spec-type", "eagle3",
            "--eagle-max-depth", str(args.max_depth),
            "--eagle-max-proposals", str(args.max_proposals),
            "--eagle-beam-width", str(args.max_proposals),
            "--eagle-per-beam-topk-candidates", str(args.per_beam_topk_candidates),
            "--temp", str(args.temp),
            "--top-k", str(args.top_k),
            "-fa", args.flash_attn,
            "-n", str(args.max_new_tokens),
            "--no-conversation",
            "--prompt", prompt_info["prompt"],
            "--eagle-trace-yaml", str(trace_path),
        ]
        proc = subprocess.run(cmd, cwd=args.llama_cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout[-4000:])
        trace = yaml.safe_load(trace_path.read_text(encoding="utf-8"))
        saved_trace_path = None
        if persist_trace_path is not None:
            persist_trace_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trace_path, persist_trace_path)
            saved_trace_path = str(persist_trace_path)
        parsed = parse_llama_spec_stdout(proc.stdout)
        cycles = trace.get("cycles") or []
        accepted = sum(int(cycle.get("accepted_count", 0)) for cycle in cycles)
        return {
            **parsed,
            "acceptance_len": (accepted / len(cycles)) if cycles else 0.0,
            "prompt_tokens_observed": trace.get("prompt_tokens") or [],
            "trace_path": saved_trace_path,
        }


def run_llama_base(args: argparse.Namespace, prompt_info: dict[str, Any]) -> dict[str, Any]:
    cmd = [
        args.llama_spec_binary,
        "-m", args.base_model_gguf,
        "--spec-type", "none",
        "--override-kv", "tokenizer.ggml.add_bos_token=bool:false",
        "-fa", args.flash_attn,
        "--temp", str(args.temp),
        "--top-k", str(args.top_k),
        "-n", str(args.max_new_tokens),
        "--no-conversation",
        "--prompt", prompt_info["prompt"],
    ]
    proc = subprocess.run(cmd, cwd=args.llama_cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-4000:])
    parsed = parse_llama_spec_stdout(proc.stdout)
    return {
        **parsed,
        "acceptance_len": None,
        "prompt_tokens_observed": prompt_info["prompt_tokens"],
        "trace_path": None,
    }


def run_kestrel_spec(args: argparse.Namespace, prompt_info: dict[str, Any], persist_trace_path: Path | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="eagle-kestrel-") as tmpdir:
        trace_path = Path(tmpdir) / "kestrel_trace.yaml"
        cmd = [
            "python3", args.kestrel_train,
            "analyze",
            "--dataset", str(args.dataset),
            "--dataset-entry", str(args.dataset_entry),
            "--base-model", args.base_model_hf,
            "--head-model", args.head_model_hf,
            "--yaml-out", str(trace_path),
            "--max-proposals", str(args.max_proposals),
            "--max-depth", str(args.max_depth),
            "--max-new-tokens", str(args.max_new_tokens),
            "--prob-threshold", str(args.prob_threshold),
            "--temp", str(args.temp),
            "--top-k", str(args.top_k),
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=args.llama_cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout[-4000:])
        trace = yaml.safe_load(trace_path.read_text(encoding="utf-8"))
        saved_trace_path = None
        if persist_trace_path is not None:
            persist_trace_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trace_path, persist_trace_path)
            saved_trace_path = str(persist_trace_path)
        cycles = trace.get("cycles") or []
        accepted = sum(int(cycle.get("accepted_count", 0)) for cycle in cycles)
        generated = int(trace.get("generated_count", 0))
        return {
            "generated_count": generated,
            "decode_seconds": elapsed,
            "generation_tps": (generated / elapsed) if elapsed > 0 else None,
            "n_accept": accepted,
            "n_drafted": None,
            "acceptance_len": (accepted / len(cycles)) if cycles else 0.0,
            "prompt_tokens_observed": trace.get("prompt_tokens") or [],
            "trace_path": saved_trace_path,
        }


def run_kestrel_base(args: argparse.Namespace, prompt_info: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_hf, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_hf,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    input_ids = torch.tensor([prompt_info["prompt_tokens"]], device=device, dtype=torch.long)
    attn = torch.ones_like(input_ids)
    generated = []
    past_key_values = None
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
        next_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        for _ in range(args.max_new_tokens):
            tok = int(torch.argmax(next_logits, dim=-1).item())
            generated.append(tok)
            step = model(input_ids=torch.tensor([[tok]], device=device), use_cache=True, past_key_values=past_key_values)
            past_key_values = step.past_key_values
            next_logits = step.logits[:, -1, :]
    elapsed = time.perf_counter() - t0
    return {
        "generated_count": len(generated),
        "decode_seconds": elapsed,
        "generation_tps": (len(generated) / elapsed) if elapsed > 0 else None,
        "n_accept": None,
        "n_drafted": None,
        "acceptance_len": None,
        "prompt_tokens_observed": prompt_info["prompt_tokens"],
        "trace_path": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Kestrel or llama.cpp with a unified prompt pipeline and metrics output.")
    ap.add_argument("--backend", choices=["llama", "kestrel"], required=True)
    ap.add_argument("--mode", choices=["spec", "base"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-entry", type=int, default=0)
    ap.add_argument("--base-model-hf", required=True)
    ap.add_argument("--head-model-hf", default="/home/alvion/projects/kestrel/models/llama3-1b_eagle")
    ap.add_argument("--base-model-gguf", default="/home/alvion/models/unsloth-Llama-3.2-1B-Instruct-bf16.gguf")
    ap.add_argument("--head-model-gguf", default="/home/alvion/projects/kestrel/models/llama3-1b_eagle.bf16.gguf")
    ap.add_argument("--llama-cwd", default=".")
    ap.add_argument("--llama-spec-binary", default="./build/bin/llama-speculative-simple")
    ap.add_argument("--kestrel-train", default="/home/alvion/projects/kestrel/train.py")
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--max-proposals", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--prob-threshold", type=float, default=1.0 / 1024.0)
    ap.add_argument("--per-beam-topk-candidates", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--flash-attn", choices=["on", "off", "auto"], default="on")
    ap.add_argument("--out")
    args = ap.parse_args()

    prompt_info = render_prompt(args.base_model_hf, Path(args.dataset), args.dataset_entry)
    expected_hash = sha256_tokens(prompt_info["prompt_tokens"])
    persist_trace_path = None
    if args.out:
        persist_trace_path = Path(args.out).with_suffix(".trace.yaml")

    if args.backend == "llama" and args.mode == "spec":
        result = run_llama_spec(args, prompt_info, persist_trace_path)
    elif args.backend == "llama" and args.mode == "base":
        result = run_llama_base(args, prompt_info)
    elif args.backend == "kestrel" and args.mode == "spec":
        result = run_kestrel_spec(args, prompt_info, persist_trace_path)
    else:
        result = run_kestrel_base(args, prompt_info)

    observed_tokens = [int(tok) for tok in (result.get("prompt_tokens_observed") or [])]
    observed_hash = sha256_tokens(observed_tokens) if observed_tokens else None

    payload = {
        "backend": args.backend,
        "mode": args.mode,
        "dataset": str(args.dataset),
        "dataset_entry": args.dataset_entry,
        "prompt_length_chars": len(prompt_info["prompt"]),
        "prompt_token_count_expected": len(prompt_info["prompt_tokens"]),
        "prompt_token_sha256_expected": expected_hash,
        "prompt_token_count_observed": len(observed_tokens) if observed_tokens else None,
        "prompt_token_sha256_observed": observed_hash,
        "prompt_tokens_match": (observed_hash == expected_hash) if observed_hash is not None else None,
        "generated_count": result.get("generated_count"),
        "decode_seconds": result.get("decode_seconds"),
        "generation_tps": result.get("generation_tps"),
        "acceptance_len": result.get("acceptance_len"),
        "n_accept": result.get("n_accept"),
        "n_drafted": result.get("n_drafted"),
        "trace_path": result.get("trace_path"),
    }

    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
