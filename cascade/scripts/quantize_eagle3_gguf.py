#!/usr/bin/env python3
# AI-GENERATED: This file was created with AI assistance for an experimental fork.
# DO NOT SUBMIT upstream unless rewritten or exhaustively reviewed by a human.
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import numpy as np


def copy_kv(reader, writer, gguf_mod) -> None:
    vt = gguf_mod.GGUFValueType

    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if key == "general.architecture":
            continue

        main_type = field.types[0]
        value = field.contents()

        if main_type == vt.STRING:
            writer.add_string(key, value)
        elif main_type == vt.INT32:
            writer.add_int32(key, int(value))
        elif main_type == vt.FLOAT32:
            writer.add_float32(key, float(value))
        elif main_type == vt.BOOL:
            writer.add_bool(key, bool(value))
        elif main_type == vt.ARRAY and field.types[-1] == vt.INT32:
            writer.add_key_value(key, [int(x) for x in value], vt.ARRAY, vt.INT32)
        else:
            raise ValueError(f"Unsupported KV type for {key}: {field.types}")


def load_ggml(repo_root: Path) -> ctypes.CDLL:
    candidates = [
        repo_root / "build" / "bin" / "libggml.so",
        repo_root / "build-cuda" / "bin" / "libggml.so",
    ]

    last_error = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            return ctypes.CDLL(str(path))
        except OSError as err:
            last_error = err

    detail = f": {last_error}" if last_error is not None else ""
    raise FileNotFoundError(f"Could not load libggml.so from {candidates}{detail}")


def configure_ggml(lib: ctypes.CDLL) -> None:
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    lib.ggml_quantize_init.restype = None

    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    lib.ggml_row_size.restype = ctypes.c_size_t

    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t


def quantize_with_ggml(
    lib: ctypes.CDLL,
    data_f32: np.ndarray,
    qtype: int,
) -> np.ndarray:
    data_f32 = np.ascontiguousarray(data_f32.astype(np.float32, copy=False))
    if data_f32.ndim == 1:
        data_f32 = data_f32.reshape(1, -1)

    nrows = int(np.prod(data_f32.shape[:-1], dtype=np.int64))
    n_per_row = int(data_f32.shape[-1])

    lib.ggml_quantize_init(qtype)

    row_size = int(lib.ggml_row_size(qtype, n_per_row))
    out = np.empty((nrows, row_size), dtype=np.uint8)

    used = int(
        lib.ggml_quantize_chunk(
            qtype,
            data_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_void_p(out.ctypes.data),
            0,
            nrows,
            n_per_row,
            None,
        )
    )

    expected = nrows * row_size
    if used != expected:
        raise ValueError(f"Quantizer returned {used} bytes, expected {expected}")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize an EAGLE3 GGUF head.")
    parser.add_argument("input", help="Input EAGLE3 GGUF")
    parser.add_argument("output", help="Output GGUF")
    parser.add_argument("qtype", choices=("Q8_0", "Q4_K_M"), help="Quantization type")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(repo_root / "gguf-py"))

    import gguf
    from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
    from gguf.quants import dequantize

    ggml = load_ggml(repo_root)
    configure_ggml(ggml)

    reader = GGUFReader(args.input)

    arch = reader.get_field("general.architecture")
    if arch is None or arch.contents() != "eagle3":
        raise ValueError(f"{args.input} is not an eagle3 GGUF")

    writer = GGUFWriter(args.output, arch="eagle3")
    copy_kv(reader, writer, gguf)

    def pick_qtype(name: str, shape: np.ndarray) -> GGMLQuantizationType:
        # Keep small 1D tensors in higher precision; they do not matter for size/perf.
        if len(shape) == 1:
            return GGMLQuantizationType.F32

        if args.qtype == "Q8_0":
            return GGMLQuantizationType.Q8_0

        if name.endswith(("attn_v.weight", "ffn_down.weight", "lm_head.weight")):
            return GGMLQuantizationType.Q6_K
        return GGMLQuantizationType.Q4_K

    for tensor in reader.tensors:
        tensor_qtype = pick_qtype(tensor.name, tensor.shape)
        data_f32 = dequantize(np.asarray(tensor.data), tensor.tensor_type)
        if tensor_qtype == GGMLQuantizationType.F32:
            quantized = np.ascontiguousarray(data_f32.astype(np.float32, copy=False))
            writer.add_tensor(tensor.name, np.ascontiguousarray(quantized))
        else:
            quantized = quantize_with_ggml(ggml, data_f32, int(tensor_qtype))
            writer.add_tensor(tensor.name, np.ascontiguousarray(quantized), raw_dtype=tensor_qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
