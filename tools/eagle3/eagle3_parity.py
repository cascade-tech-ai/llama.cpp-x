#!/usr/bin/env python3
"""EAGLE3 head parity helper.

This script is intended to compare a llama.cpp EAGLE3 dump (CASCADE_EAGLE_DUMP_DIR)
against a PyTorch/Kestrel reference computation.

It does NOT require running the base LLM in PyTorch: it patches a tiny embedding
matrix using `head_embd_by_step.npy` so the head sees the exact embeddings that
llama.cpp used.

Usage example:
  python3 tools/eagle3/eagle3_parity.py \
    --llama-dump /tmp/eagle_llama \
    --head-model /home/alvion/models/llama3-1b_eagle_001
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def _load_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected ndarray from {path}, got {type(arr)}")
    return arr


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def _align_for_compare(name: str, llama_arr: np.ndarray, torch_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if llama_arr.shape == torch_arr.shape:
        return llama_arr, torch_arr

    # Some heads (e.g. Qwen3) can have q_proj width != hidden_size.
    # Keep parity checks running by comparing the shared prefix.
    if (
        llama_arr.ndim == 2
        and torch_arr.ndim == 2
        and llama_arr.shape[0] == torch_arr.shape[0]
    ):
        w = min(llama_arr.shape[1], torch_arr.shape[1])
        print(
            f"{name}: shape mismatch llama={llama_arr.shape} torch={torch_arr.shape}; "
            f"comparing first {w} columns"
        )
        return llama_arr[:, :w], torch_arr[:, :w]

    raise SystemExit(f"{name} shape mismatch: llama={llama_arr.shape} torch={torch_arr.shape}")


def _topk_from_logits(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (idx, logits, probs) sorted by prob desc."""
    if logits.ndim != 1:
        raise ValueError(f"expected 1D logits, got shape={logits.shape}")
    k = min(k, logits.shape[0])

    # stable-ish softmax
    m = float(np.max(logits))
    ex = np.exp(logits - m)
    probs = ex / float(np.sum(ex))

    idx = np.argpartition(-probs, kth=k - 1)[:k]
    idx = idx[np.argsort(-probs[idx])]
    return idx.astype(np.int64), logits[idx], probs[idx]


def _patch_embedding_weight(embed_weight: "np.ndarray", input_ids: np.ndarray, embd_by_step: np.ndarray) -> None:
    """Patch rows of embed_weight in-place based on step-aligned (token_id, embd_vector)."""
    assert input_ids.ndim == 1
    assert embd_by_step.ndim == 2
    assert embd_by_step.shape[0] == input_ids.shape[0]

    seen: dict[int, np.ndarray] = {}
    for tid, vec in zip(input_ids.tolist(), embd_by_step, strict=True):
        tid_i = int(tid)
        if tid_i in seen:
            # sanity: embeddings should be identical for the same token id
            if _max_abs_diff(seen[tid_i], vec) > 1e-5:
                raise ValueError(f"embedding mismatch for token id {tid_i}")
            continue
        seen[tid_i] = vec

    for tid_i, vec in seen.items():
        embed_weight[tid_i, :] = vec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-dump", type=Path, required=True)
    ap.add_argument("--head-model", type=str, default="/home/alvion/models/llama3-1b_eagle_001")
    ap.add_argument("--kestrel-repo", type=Path, default=Path.home() / "projects" / "kestrel")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--save-torch-dump", type=Path, default=None)
    args = ap.parse_args()

    dump_dir: Path = args.llama_dump
    meta_path = dump_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"missing {meta_path} (did you set CASCADE_EAGLE_DUMP_DIR?)")

    meta = json.loads(meta_path.read_text())
    hidden_in_dim = int(meta["hidden_in_dim"])
    hidden_size = int(meta["hidden_size"])
    draft_vocab_size = int(meta["draft_vocab_size"])

    prompt_tgt = _load_npy(dump_dir / "prompt_tgt.npy").astype(np.int32)
    id_last = int(_load_npy(dump_dir / "id_last.npy").astype(np.int32).reshape(-1)[0])

    head_input_ids = _load_npy(dump_dir / "head_input_ids.npy").astype(np.int32)
    teacher_hidden = _load_npy(dump_dir / "teacher_hidden_concat_by_step.npy").astype(np.float32)
    llama_head_hidden = _load_npy(dump_dir / "head_hidden_after_step.npy").astype(np.float32)
    llama_root_logits = _load_npy(dump_dir / "head_root_logits_draft.npy").astype(np.float32)

    embd_by_step = _load_npy(dump_dir / "head_embd_by_step.npy").astype(np.float32)
    embd_norm_by_step = _load_npy(dump_dir / "head_embd_norm_by_step.npy").astype(np.float32)
    hidden_proj_by_step = _load_npy(dump_dir / "head_hidden_proj_by_step.npy").astype(np.float32)
    hidden_norm_by_step = _load_npy(dump_dir / "head_hidden_norm_by_step.npy").astype(np.float32)
    cat_by_step = _load_npy(dump_dir / "head_cat_by_step.npy").astype(np.float32)
    q_by_step = _load_npy(dump_dir / "head_q_by_step.npy").astype(np.float32)
    k_by_step = _load_npy(dump_dir / "head_k_by_step.npy").astype(np.float32)
    v_by_step = _load_npy(dump_dir / "head_v_by_step.npy").astype(np.float32)

    if teacher_hidden.shape != (head_input_ids.shape[0], hidden_in_dim):
        raise SystemExit(f"teacher_hidden shape mismatch: {teacher_hidden.shape} vs ({head_input_ids.shape[0]}, {hidden_in_dim})")
    if llama_head_hidden.shape != (head_input_ids.shape[0], hidden_size):
        raise SystemExit(f"llama_head_hidden shape mismatch: {llama_head_hidden.shape} vs ({head_input_ids.shape[0]}, {hidden_size})")
    if embd_by_step.shape != (head_input_ids.shape[0], hidden_size):
        raise SystemExit(f"embd_by_step shape mismatch: {embd_by_step.shape} vs ({head_input_ids.shape[0]}, {hidden_size})")
    if embd_norm_by_step.shape != (head_input_ids.shape[0], hidden_size):
        raise SystemExit(
            f"embd_norm_by_step shape mismatch: {embd_norm_by_step.shape} vs ({head_input_ids.shape[0]}, {hidden_size})"
        )
    if hidden_proj_by_step.shape != (head_input_ids.shape[0], hidden_size):
        raise SystemExit(
            f"hidden_proj_by_step shape mismatch: {hidden_proj_by_step.shape} vs ({head_input_ids.shape[0]}, {hidden_size})"
        )
    if hidden_norm_by_step.shape != (head_input_ids.shape[0], hidden_size):
        raise SystemExit(
            f"hidden_norm_by_step shape mismatch: {hidden_norm_by_step.shape} vs ({head_input_ids.shape[0]}, {hidden_size})"
        )
    if cat_by_step.shape != (head_input_ids.shape[0], hidden_size * 2):
        raise SystemExit(
            f"cat_by_step shape mismatch: {cat_by_step.shape} vs ({head_input_ids.shape[0]}, {hidden_size * 2})"
        )
    if q_by_step.shape != (head_input_ids.shape[0], hidden_size):
        raise SystemExit(f"q_by_step shape mismatch: {q_by_step.shape} vs ({head_input_ids.shape[0]}, {hidden_size})")
    if k_by_step.ndim != 2 or k_by_step.shape[0] != head_input_ids.shape[0]:
        raise SystemExit(f"k_by_step shape mismatch: {k_by_step.shape} vs ({head_input_ids.shape[0]}, kv_dim)")
    if v_by_step.shape != k_by_step.shape:
        raise SystemExit(f"v_by_step shape mismatch: {v_by_step.shape} vs {k_by_step.shape}")
    if llama_root_logits.shape != (draft_vocab_size,):
        raise SystemExit(f"llama_root_logits shape mismatch: {llama_root_logits.shape} vs ({draft_vocab_size},)")

    print(f"prompt_tgt_len={prompt_tgt.shape[0]} id_last={id_last} head_steps={head_input_ids.shape[0]}")

    # Import Kestrel head implementation
    kestrel_repo = args.kestrel_repo
    if not kestrel_repo.exists():
        raise SystemExit(f"kestrel repo not found: {kestrel_repo}")

    import sys

    sys.path.insert(0, str(kestrel_repo))

    import torch
    import torch.nn as nn

    from kestrel.head import apply_rotary_pos_emb, load_eagle_head

    device = torch.device("cpu")
    dtype = torch.float32

    # Build a minimal embedding table and patch only needed rows.
    max_id = int(np.max(head_input_ids))
    embed_tokens = nn.Embedding(max_id + 1, hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        embed_tokens.weight.zero_()
        w = embed_tokens.weight.detach().cpu().numpy()
        _patch_embedding_weight(w, head_input_ids, embd_by_step)
        embed_tokens.weight.copy_(torch.from_numpy(w).to(device=device, dtype=dtype))

    head = load_eagle_head(args.head_model, base_embed_tokens=embed_tokens, device=device, dtype=dtype)

    # Run the head in one go (equivalent to token-by-token w/ causal mask)
    hs = torch.from_numpy(teacher_hidden).to(device=device, dtype=dtype).unsqueeze(0)  # [1, T, hidden_in_dim]
    ids = torch.from_numpy(head_input_ids.astype(np.int64)).to(device=device).unsqueeze(0)  # [1, T]
    attn = torch.ones_like(ids, device=device)

    with torch.no_grad():
        # Pre-attn intermediates (these should match llama.cpp dumps closely).
        torch_embd = torch.nn.functional.embedding(ids, head.embed_weight).to(dtype)  # [1, T, hidden]
        torch_hidden_proj = head.fc(hs)                                               # [1, T, hidden]
        torch_embd_norm = head.midlayer.input_layernorm(torch_embd)                   # [1, T, hidden]
        torch_hidden_norm = head.midlayer.hidden_norm(torch_hidden_proj)              # [1, T, hidden]
        torch_cat = torch.cat((torch_embd_norm, torch_hidden_norm), dim=-1)           # [1, T, 2*hidden]

        # Attention projections (q/k after RoPE), flattened to match ggml layout.
        attn_mod = head.midlayer.self_attn
        bsz, t_len, _ = torch_cat.shape
        q_lin = attn_mod.q_proj(torch_cat)
        k_lin = attn_mod.k_proj(torch_cat)
        v_lin = attn_mod.v_proj(torch_cat)

        q_states = q_lin.view(bsz, t_len, attn_mod.num_heads, attn_mod.head_dim).transpose(1, 2)
        k_states = k_lin.view(bsz, t_len, attn_mod.num_key_value_heads, attn_mod.head_dim).transpose(1, 2)
        v_states = v_lin.view(bsz, t_len, attn_mod.num_key_value_heads, attn_mod.head_dim).transpose(1, 2)

        position_ids = torch.arange(0, t_len, dtype=torch.long, device=device).unsqueeze(0)
        cos, sin = attn_mod.rotary_emb(v_states, position_ids)
        q_rope, k_rope = apply_rotary_pos_emb(q_states, k_states, cos, sin)

        # ggml tensors are laid out with ne0 (head_dim) as the fastest-changing dimension.
        # In torch (row-major), the last dim is fastest. So we keep head_dim last.
        torch_q_flat = q_rope[0].permute(1, 0, 2).contiguous().view(t_len, -1)
        torch_k_flat = k_rope[0].permute(1, 0, 2).contiguous().view(t_len, -1)
        torch_v_flat = v_states[0].permute(1, 0, 2).contiguous().view(t_len, -1)

        torch_hidden, _ = head(hidden_states=hs, input_ids=ids, attention_mask=attn, use_cache=False)
        torch_logits = head.logits_from_hidden(torch_hidden).squeeze(0)

    torch_embd_np = torch_embd.squeeze(0).cpu().numpy().astype(np.float32)
    torch_hidden_proj_np = torch_hidden_proj.squeeze(0).cpu().numpy().astype(np.float32)
    torch_embd_norm_np = torch_embd_norm.squeeze(0).cpu().numpy().astype(np.float32)
    torch_hidden_norm_np = torch_hidden_norm.squeeze(0).cpu().numpy().astype(np.float32)
    torch_cat_np = torch_cat.squeeze(0).cpu().numpy().astype(np.float32)
    torch_q_flat_np = torch_q_flat.cpu().numpy().astype(np.float32)
    torch_k_flat_np = torch_k_flat.cpu().numpy().astype(np.float32)
    torch_v_flat_np = torch_v_flat.cpu().numpy().astype(np.float32)

    torch_hidden_np = torch_hidden.squeeze(0).cpu().numpy().astype(np.float32)
    torch_logits_np = torch_logits.cpu().numpy().astype(np.float32)

    def _print_stage(name: str, llama_arr: np.ndarray, torch_arr: np.ndarray) -> None:
        llama_arr, torch_arr = _align_for_compare(name, llama_arr, torch_arr)
        d = torch_arr - llama_arr
        per_step = np.max(np.abs(d).reshape(d.shape[0], -1), axis=1)
        worst = int(np.argmax(per_step))
        print(
            f"{name}: max_abs={float(np.max(per_step)):.6g} rms={_rms(d):.6g} "
            f"worst_step={worst} worst_step_max_abs={float(per_step[worst]):.6g}"
        )

    _print_stage("embd_by_step", embd_by_step, torch_embd_np)
    _print_stage("embd_norm_by_step", embd_norm_by_step, torch_embd_norm_np)
    _print_stage("hidden_proj_by_step", hidden_proj_by_step, torch_hidden_proj_np)
    _print_stage("hidden_norm_by_step", hidden_norm_by_step, torch_hidden_norm_np)
    _print_stage("cat_by_step", cat_by_step, torch_cat_np)
    _print_stage("q_by_step", q_by_step, torch_q_flat_np)
    _print_stage("k_by_step", k_by_step, torch_k_flat_np)
    _print_stage("v_by_step", v_by_step, torch_v_flat_np)

    # Compare hidden sequence (per step)
    per_step_max = np.max(np.abs(torch_hidden_np - llama_head_hidden), axis=1)
    worst_step = int(np.argmax(per_step_max))
    print(f"head_hidden_after_step: max_abs={float(np.max(per_step_max)):.6g} rms={_rms(torch_hidden_np - llama_head_hidden):.6g} worst_step={worst_step} worst_step_max_abs={float(per_step_max[worst_step]):.6g}")

    # Compare final logits
    logit_diff = torch_logits_np - llama_root_logits
    print(f"head_root_logits_draft: max_abs={_max_abs_diff(torch_logits_np, llama_root_logits):.6g} rms={_rms(logit_diff):.6g}")

    # Show top-k (draft vocab indices)
    k = int(args.topk)
    idx_l, log_l, p_l = _topk_from_logits(llama_root_logits, k)
    idx_t, log_t, p_t = _topk_from_logits(torch_logits_np, k)

    print("\nTop-k draft tokens (llama.cpp vs torch):")
    for i in range(max(len(idx_l), len(idx_t))):
        left = ""
        right = ""
        if i < len(idx_l):
            left = f"{int(idx_l[i]):6d} logit={float(log_l[i]):9.4f} p={float(p_l[i])*100:6.2f}%"
        if i < len(idx_t):
            right = f"{int(idx_t[i]):6d} logit={float(log_t[i]):9.4f} p={float(p_t[i])*100:6.2f}%"
        print(f"{left:<34}    {right}")

    # Probability of base token id 22 ("7") if it exists in the draft vocab directly.
    token7_id = 22
    if token7_id < llama_root_logits.shape[0]:
        m = float(np.max(llama_root_logits))
        p7 = float(np.exp(llama_root_logits[token7_id] - m) / np.sum(np.exp(llama_root_logits - m)))
        mt = float(np.max(torch_logits_np))
        p7t = float(np.exp(torch_logits_np[token7_id] - mt) / np.sum(np.exp(torch_logits_np - mt)))
        print(f"\nP(draft_idx=22 aka token_id 22): llama={p7*100:.3f}% torch={p7t*100:.3f}%")

    if args.save_torch_dump is not None:
        out_dir = args.save_torch_dump
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "torch_head_hidden_after_step.npy", torch_hidden_np)
        np.save(out_dir / "torch_head_root_logits_draft.npy", torch_logits_np)
        (out_dir / "meta.json").write_text(json.dumps({
            "format": "torch_eagle3_parity_v1",
            "source_llama_dump": str(dump_dir),
            "head_model": args.head_model,
        }, indent=2) + "\n")
        print(f"\nwrote torch dump to {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
