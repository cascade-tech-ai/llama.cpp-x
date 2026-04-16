#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


COMMON_ARRAYS = [
    "prompt_tgt.npy",
    "id_last.npy",
    "layer_ids.npy",
    "teacher_hidden_concat_by_step.npy",
    "head_input_ids.npy",
    "head_embd_by_step.npy",
    "head_embd_norm_by_step.npy",
    "head_hidden_proj_by_step.npy",
    "head_hidden_norm_by_step.npy",
    "head_cat_by_step.npy",
    "head_q_by_step.npy",
    "head_k_by_step.npy",
    "head_v_by_step.npy",
    "head_hidden_after_step.npy",
    "head_root_logits_draft.npy",
]


def _load(path: Path) -> np.ndarray:
    return np.load(path)


def _compare(name: str, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "shape_match": False}
    diff = b.astype(np.float64) - a.astype(np.float64)
    metrics: dict[str, Any] = {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "shape_match": True,
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(diff)))) if diff.size else 0.0,
    }
    if diff.ndim >= 2:
        flat = diff.reshape(diff.shape[0], -1)
        per_step = np.max(np.abs(flat), axis=1)
        worst = int(np.argmax(per_step))
        metrics["worst_step"] = worst
        metrics["worst_step_max_abs"] = float(per_step[worst])
    if name == "head_root_logits_draft.npy" and a.ndim == 1:
        topk = min(10, a.shape[0])
        idx_a = np.argsort(-a)[:topk].astype(np.int64).tolist()
        idx_b = np.argsort(-b)[:topk].astype(np.int64).tolist()
        metrics["top10_a"] = idx_a
        metrics["top10_b"] = idx_b
        metrics["top10_overlap"] = len(set(idx_a) & set(idx_b))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two EAGLE first-step dump directories.")
    parser.add_argument("--left", required=True, help="Left dump dir.")
    parser.add_argument("--right", required=True, help="Right dump dir.")
    parser.add_argument("--out-json", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    left = Path(args.left).expanduser().resolve()
    right = Path(args.right).expanduser().resolve()
    if not left.is_dir():
        raise FileNotFoundError(f"left dump dir not found: {left}")
    if not right.is_dir():
        raise FileNotFoundError(f"right dump dir not found: {right}")

    results: dict[str, Any] = {
        "left": str(left),
        "right": str(right),
        "files": {},
    }

    meta_left = left / "meta.json"
    meta_right = right / "meta.json"
    if meta_left.is_file():
        results["meta_left"] = json.loads(meta_left.read_text(encoding="utf-8"))
    if meta_right.is_file():
        results["meta_right"] = json.loads(meta_right.read_text(encoding="utf-8"))

    for name in COMMON_ARRAYS:
        lp = left / name
        rp = right / name
        if not lp.is_file() or not rp.is_file():
            results["files"][name] = {
                "present_left": lp.is_file(),
                "present_right": rp.is_file(),
                "shape_match": False,
            }
            continue
        results["files"][name] = _compare(name, _load(lp), _load(rp))

    if meta_left.is_file() and meta_right.is_file():
        meta_l = results["meta_left"]
        meta_r = results["meta_right"]
        teacher_layers = []
        for layer_id in meta_l.get("layer_ids", []):
            name = f"teacher_hidden_layer_{int(layer_id)}.npy"
            lp = left / name
            rp = right / name
            if lp.is_file() and rp.is_file():
                results["files"][name] = _compare(name, _load(lp), _load(rp))
                teacher_layers.append(name)
        results["teacher_hidden_layer_files"] = teacher_layers

    lines: list[str] = []
    for name, metrics in results["files"].items():
        if not metrics.get("shape_match"):
            lines.append(f"{name}: shape/presence mismatch")
            continue
        line = (
            f"{name}: max_abs={metrics['max_abs']:.6g} "
            f"mean_abs={metrics['mean_abs']:.6g} rmse={metrics['rmse']:.6g}"
        )
        if "worst_step" in metrics:
            line += f" worst_step={metrics['worst_step']} worst_step_max_abs={metrics['worst_step_max_abs']:.6g}"
        if "top10_overlap" in metrics:
            line += f" top10_overlap={metrics['top10_overlap']}/10"
        lines.append(line)

    print("\n".join(lines))

    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"[compare] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
