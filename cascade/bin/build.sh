#!/usr/bin/env bash
# AI-GENERATED: This file was created with AI assistance for an experimental fork.
# DO NOT SUBMIT upstream unless rewritten or exhaustively reviewed by a human.
set -euo pipefail

# Default CUDA build for this fork (override with env vars as needed)
: "${GGML_CUDA:=ON}"
: "${GGML_CUDA_FA:=ON}"
: "${CMAKE_CUDA_ARCHITECTURES:=80;86}"

cmake -S . -B build \
  -DGGML_CUDA="${GGML_CUDA}" \
  -DGGML_CUDA_FA="${GGML_CUDA_FA}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}"

cmake --build build --target llama-speculative-simple
