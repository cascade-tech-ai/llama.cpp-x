#!/usr/bin/env python3
# AI-GENERATED: This file was created with AI assistance for an experimental fork.
# DO NOT SUBMIT upstream unless rewritten or exhaustively reviewed by a human.
"""
Convert a Kestrel/speculators EAGLE3 head checkpoint to GGUF for this fork.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch


def _add_repo_paths(kestrel_path: Path, repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "gguf-py"))
    sys.path.insert(0, str(kestrel_path))


def _to_numpy(tensor: torch.Tensor, dtype: str) -> tuple[np.ndarray, str | None]:
    tensor = tensor.detach().cpu().contiguous()

    if dtype == "f16":
        arr = tensor.to(dtype=torch.float16).numpy()
        return np.ascontiguousarray(arr), None
    if dtype == "f32":
        arr = tensor.to(dtype=torch.float32).numpy()
        return np.ascontiguousarray(arr), None
    if dtype == "bf16":
        # NumPy does not expose a native bfloat16 dtype here, so store raw BF16 payload bytes.
        arr = tensor.to(dtype=torch.bfloat16).view(torch.uint16).numpy()
        return np.ascontiguousarray(arr), "bf16"

    raise ValueError(f"Unsupported dtype: {dtype}")


def _require_keys(state: Dict[str, torch.Tensor], keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in state]
    if missing:
        raise KeyError(f"Missing required weights: {', '.join(missing)}")


def _resolve_layer_ids(config, base_n_layers: int | None) -> List[int]:
    """Resolve hidden state layer IDs from the head config.

    Layer IDs follow the vLLM/kestrel convention (adopted Dec 2025): each ID
    refers to the OUTPUT of that layer, i.e. the tensor produced after layer N
    finishes.  In llama.cpp's internal numbering the l_out callback fires at the
    end of layer ``il``, so we store l_out(il) under key ``il + 1`` to match
    this convention.  See llm_graph_context::try_capture_eagle3_hidden() in
    llama-graph.cpp for the mapping.
    """
    layer_ids = getattr(config, "eagle_aux_hidden_state_layer_ids", None)
    if layer_ids:
        return [int(x) for x in layer_ids]

    layer_spec = getattr(config, "eagle_aux_hidden_state_layers", None)
    if layer_spec is None:
        raise ValueError("Missing eagle_aux_hidden_state_layer_ids/layers in head config.")
    if base_n_layers is None:
        raise ValueError("Need --base-n-layers to resolve eagle_aux_hidden_state_layers.")

    from kestrel.runtime import resolve_eagle_hidden_layers

    return resolve_eagle_hidden_layers(layer_spec, base_n_layers)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert EAGLE3 head to GGUF.")
    parser.add_argument("head", help="Path or HF repo id for the EAGLE3 head")
    parser.add_argument("--out", required=True, help="Output GGUF path")
    parser.add_argument(
        "--kestrel-path",
        default=os.path.expanduser("~/projects/kestrel"),
        help="Path to kestrel repo (default: ~/projects/kestrel)",
    )
    parser.add_argument(
        "--dtype",
        choices=("f16", "bf16", "f32"),
        default="bf16",
        help="Output tensor dtype (default: bf16). Use bf16 to preserve bfloat16 precision from PyTorch.",
    )
    parser.add_argument(
        "--base-n-layers",
        type=int,
        default=None,
        help="Base model layer count (needed if config lacks eagle_aux_hidden_state_layer_ids)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    kestrel_path = Path(args.kestrel_path).resolve()
    if not kestrel_path.exists():
        raise FileNotFoundError(f"kestrel path not found: {kestrel_path}")

    _add_repo_paths(kestrel_path, repo_root)

    from gguf import GGMLQuantizationType, GGUFWriter, GGUFValueType
    from kestrel.draft_loader import load_draft_head

    state, d2t, _t2d, config = load_draft_head(args.head)

    layer_ids = _resolve_layer_ids(config, args.base_n_layers)
    hidden_concat = len(layer_ids)
    cfg_hidden_concat = int(getattr(config, "hidden_concat", hidden_concat))
    if cfg_hidden_concat != hidden_concat:
        print(
            f"warning: config.hidden_concat={cfg_hidden_concat} does not match "
            f"layer_ids length {hidden_concat}; using {hidden_concat}"
        )

    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    if int(getattr(config, "num_hidden_layers", 1)) != 1:
        raise ValueError("EAGLE3 head conversion expects num_hidden_layers == 1")
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.head_dim or (hidden_size // num_heads))
    target_hidden = int(config.target_hidden_size or hidden_size)
    draft_vocab_size = int(config.draft_vocab_size)
    vocab_size = int(config.vocab_size)
    rms_eps = float(config.rms_norm_eps)
    norm_before_residual = bool(getattr(config, "norm_before_residual", False))

    required = [
        "fc.weight",
        "norm.weight",
        "lm_head.weight",
        "midlayer.hidden_norm.weight",
        "midlayer.input_layernorm.weight",
        "midlayer.post_attention_layernorm.weight",
        "midlayer.self_attn.q_proj.weight",
        "midlayer.self_attn.k_proj.weight",
        "midlayer.self_attn.v_proj.weight",
        "midlayer.self_attn.o_proj.weight",
        "midlayer.mlp.gate_proj.weight",
        "midlayer.mlp.up_proj.weight",
        "midlayer.mlp.down_proj.weight",
    ]
    _require_keys(state, required)

    mapping = {
        "fc.weight": "fc.weight",
        "norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
        "midlayer.hidden_norm.weight": "hidden_norm.weight",
        "midlayer.input_layernorm.weight": "attn_norm.weight",
        "midlayer.post_attention_layernorm.weight": "ffn_norm.weight",
        "midlayer.self_attn.q_proj.weight": "attn_q.weight",
        "midlayer.self_attn.k_proj.weight": "attn_k.weight",
        "midlayer.self_attn.v_proj.weight": "attn_v.weight",
        "midlayer.self_attn.o_proj.weight": "attn_output.weight",
        "midlayer.mlp.gate_proj.weight": "ffn_gate.weight",
        "midlayer.mlp.up_proj.weight": "ffn_up.weight",
        "midlayer.mlp.down_proj.weight": "ffn_down.weight",
    }

    optional = {
        "midlayer.self_attn.q_proj.bias": "attn_q.bias",
        "midlayer.self_attn.k_proj.bias": "attn_k.bias",
        "midlayer.self_attn.v_proj.bias": "attn_v.bias",
        "midlayer.self_attn.o_proj.bias": "attn_output.bias",
        "midlayer.mlp.gate_proj.bias": "ffn_gate.bias",
        "midlayer.mlp.up_proj.bias": "ffn_up.bias",
        "midlayer.mlp.down_proj.bias": "ffn_down.bias",
    }

    writer = GGUFWriter(args.out, arch="eagle3")

    writer.add_int32("eagle3.hidden_size", hidden_size)
    writer.add_int32("eagle3.intermediate_size", intermediate_size)
    writer.add_int32("eagle3.num_attention_heads", num_heads)
    writer.add_int32("eagle3.num_key_value_heads", num_kv_heads)
    writer.add_int32("eagle3.head_dim", head_dim)
    writer.add_int32("eagle3.hidden_concat", hidden_concat)
    writer.add_int32("eagle3.target_hidden_size", target_hidden)
    writer.add_int32("eagle3.draft_vocab_size", draft_vocab_size)
    writer.add_int32("eagle3.vocab_size", vocab_size)
    writer.add_float32("eagle3.rms_norm_eps", rms_eps)
    writer.add_bool("eagle3.norm_before_residual", norm_before_residual)

    # rope_theta and partial_rotary_factor may live on config directly (older
    # transformers versions / Llama) or inside config.rope_parameters (newer
    # HF configs, Qwen3, etc.). Check both.
    rope_parameters = getattr(config, "rope_parameters", None)
    if not isinstance(rope_parameters, dict):
        rope_parameters = {}
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = rope_parameters.get("rope_theta")
    if rope_theta is not None:
        writer.add_float32("eagle3.rope_theta", float(rope_theta))

    partial_rotary = getattr(config, "partial_rotary_factor", None)
    if partial_rotary is None:
        partial_rotary = rope_parameters.get("partial_rotary_factor")
    if partial_rotary is not None:
        # HF convention: rotary is applied to the first partial_rotary_factor
        # fraction of each head. Store that fraction; the consumer turns it
        # into an integer n_rot at load time.
        writer.add_float32("eagle3.partial_rotary_factor", float(partial_rotary))

    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("type")
        rope_factor = rope_scaling.get("factor")
        if rope_type:
            writer.add_string("eagle3.rope_scaling.type", str(rope_type))
        if rope_factor is not None:
            writer.add_float32("eagle3.rope_scaling.factor", float(rope_factor))

    writer.add_key_value(
        "eagle3.hidden_state_layer_ids",
        [int(x) for x in layer_ids],
        GGUFValueType.ARRAY,
        GGUFValueType.INT32,
    )
    d2t_list = d2t.to(dtype=torch.int32).cpu().tolist()
    if len(d2t_list) != draft_vocab_size:
        raise ValueError("d2t length does not match draft_vocab_size")
    writer.add_key_value(
        "eagle3.d2t",
        d2t_list,
        GGUFValueType.ARRAY,
        GGUFValueType.INT32,
    )

    for src, dst in mapping.items():
        tensor, raw_dtype = _to_numpy(state[src], args.dtype)
        writer.add_tensor(
            dst,
            tensor,
            raw_dtype=GGMLQuantizationType.BF16 if raw_dtype == "bf16" else None,
        )

    for src, dst in optional.items():
        if src in state:
            tensor, raw_dtype = _to_numpy(state[src], args.dtype)
            writer.add_tensor(
                dst,
                tensor,
                raw_dtype=GGMLQuantizationType.BF16 if raw_dtype == "bf16" else None,
            )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Wrote GGUF to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
