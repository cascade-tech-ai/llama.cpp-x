# EAGLE3 Speculative Decoding — Design Notes

## What this is

EAGLE3 is a speculative decoding method where a small draft head predicts multiple future tokens per target model evaluation. The draft head takes hidden states captured from intermediate layers of the target model and uses them — along with the embedding of the last accepted token — to predict what the target model would produce next, without running the full target model.

The draft head runs a beam search over multiple depths, building a tree of candidate continuations. The target model then evaluates the entire tree in one pass and accepts the longest prefix that matches greedy decoding.

Our implementation lives in this fork. The reference training and evaluation system is **kestrel** (`~/projects/kestrel`), which trains EAGLE3 heads in PyTorch. Our llama.cpp implementation must produce numerically compatible results with kestrel's inference path.

## Hidden state capture

The EAGLE3 head requires hidden states from specific intermediate layers of the target model (e.g., layers 2, 16, 29 for Qwen3.5-4B). Which layers are stored in the EAGLE head's GGUF metadata as `eagle_aux_hidden_state_layer_ids`.

We use a **generic capture mechanism** that hooks on the `l_out` callback emitted at the end of every transformer layer in every model. The callback fires at the end of layer `il`, producing the output of that layer, which equals the input to layer `il+1`. Since kestrel uses input-to-layer semantics (matching HuggingFace's `hidden_states[i]`), we store `l_out(il)` as `eagle3_hidden[il+1]`.

The capture always creates a distinct tensor copy via `ggml_cast` + `ggml_cont`. Without this, the ggml allocator may reuse the buffer for downstream operations, silently overwriting the captured data.

This generic mechanism means **no per-model code is needed** to support EAGLE3 — any model that emits `l_out` (99+ models in llama.cpp) works automatically. The only model-specific fix so far is adding the missing `l_out` callback to Qwen3.5, which didn't have one (1 line).

## Optimized prefill

The draft head needs to process the full prompt before it can start drafting. Naively this means running the draft head on every prompt token, which is expensive. Our optimized prefill path splits this into two cached graphs:

1. **KV-only pass**: Processes all prompt tokens through the draft head's attention layers to populate its KV cache, but skips the output projection. This can be chunked for long prompts.
2. **Root pass**: Processes only the last token through the full draft head (including output projection) to get the initial draft logits.

Both graphs are pre-built and cached on the runtime, so the first speculative cycle doesn't pay graph construction costs. This is the main latency optimization for the draft head — without it, prefill would be the bottleneck.

## Tree verification: flat tree vs coupled batches

After the draft head proposes a tree of candidates, the target model must evaluate all of them to decide which to accept. There are two approaches:

**Coupled batches** (legacy, env var `COUPLED_TREE=1`): Each branch of the tree gets its own sequence via `seq_cp`. All branches share the prefix KV cache. The cost is O(n_branches × n_prefix_cells) for the sequence copies.

**Flat tree** (default): All tree tokens go on a single sequence with unique cache positions. A tree attention mask ensures each token only attends to its ancestors. The cost is O(1) for sequence management — no copies needed.

Flat tree is strictly faster and produces identical outputs. Coupled batches cannot work with hybrid models (e.g., Qwen3.5's delta-net layers don't support `seq_cp`). We keep coupled as a fallback but flat tree is the default for all non-hybrid models. Hybrid models use linear (non-tree) verification.

## Key files

| File | Role |
|------|------|
| `src/llama-eagle3.cpp` / `.h` | Draft head engine: forward pass, beam search, tree building, state management |
| `common/speculative.cpp` | Integration layer: orchestrates draft/verify cycles, trace recording |
| `src/llama-graph.cpp` | Generic hidden state capture mechanism |
| `examples/speculative-simple/speculative-simple.cpp` | Main example binary with flat tree verification and trace output |
| `cascade/tools/` | Benchmarking, trace verification, parity checking, GGUF conversion |

## Kestrel compatibility

Kestrel is the reference. When our outputs diverge from kestrel, we assume we have a bug until proven otherwise. Key compatibility points:

- **Hidden state indexing**: Both use input-to-layer semantics. `layer_id=2` means input to block 2 = output of block 1 = HF `hidden_states[2]`.
- **Draft logits**: Root logits should match within bf16 tolerance (~0.05 max abs diff, same top-10 ordering). Verified via `CASCADE_EAGLE_DUMP_DIR` env var which dumps all intermediate tensors as `.npy` files.
- **Acceptance length**: Should be in the same ballpark for the same model and prompts. Exact trace match is not expected since the beam search implementations differ.
- **GGUF conversion**: `cascade/tools/convert_eagle3_to_gguf.py` converts kestrel checkpoints to GGUF.
