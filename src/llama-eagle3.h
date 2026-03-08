// AI-GENERATED: This file was created with AI assistance for an experimental fork.
// DO NOT SUBMIT upstream unless rewritten or exhaustively reviewed by a human.
#pragma once

#include "ggml-cpp.h"
#include "llama.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_eagle3_hparams {
    int32_t hidden_size        = 0;
    int32_t intermediate_size  = 0;
    int32_t num_heads          = 0;
    int32_t num_kv_heads       = 0;
    int32_t head_dim           = 0;
    int32_t hidden_concat      = 0;
    int32_t target_hidden_size = 0;
    int32_t draft_vocab_size   = 0;
    int32_t vocab_size         = 0;

    float rms_norm_eps = 1e-6f;

    bool norm_before_residual = false;
};

struct llama_eagle3_tensors {
    ggml_tensor * fc_w        = nullptr;
    ggml_tensor * norm_w      = nullptr;
    ggml_tensor * lm_head_w   = nullptr;

    ggml_tensor * hidden_norm_w = nullptr;
    ggml_tensor * input_norm_w  = nullptr;
    ggml_tensor * post_norm_w   = nullptr;

    ggml_tensor * attn_q_w = nullptr;
    ggml_tensor * attn_k_w = nullptr;
    ggml_tensor * attn_v_w = nullptr;
    ggml_tensor * attn_o_w = nullptr;

    ggml_tensor * attn_q_b = nullptr;
    ggml_tensor * attn_k_b = nullptr;
    ggml_tensor * attn_v_b = nullptr;
    ggml_tensor * attn_o_b = nullptr;

    ggml_tensor * ffn_gate_w = nullptr;
    ggml_tensor * ffn_up_w   = nullptr;
    ggml_tensor * ffn_down_w = nullptr;

    ggml_tensor * ffn_gate_b = nullptr;
    ggml_tensor * ffn_up_b   = nullptr;
    ggml_tensor * ffn_down_b = nullptr;
};

struct llama_eagle3_model {
    llama_eagle3_hparams hparams;
    llama_eagle3_tensors tensors;

    ggml_context_ptr         ctx_weights;
    ggml_backend_buffer_ptr  buf_weights;

    std::vector<int32_t> hidden_layer_ids;
    std::vector<int32_t> d2t;
};

struct llama_eagle3_state {
    std::vector<float> hidden; // [hidden_size]
    std::vector<float> k;      // [head_dim, n_kv_heads, past_len]
    std::vector<float> v;      // [head_dim, n_kv_heads, past_len]
    int32_t past_len = 0;

    struct device_state;
    std::shared_ptr<device_state> dev;
};

// Optional debugging outputs for llama_eagle3_step().
//
// When a pointer is non-null, the implementation appends the current step's tensor values
// (as contiguous f32) to the referenced vector.
struct llama_eagle3_step_debug {
    std::vector<float> * embd       = nullptr; // token embedding (pre-norm), shape: [hidden_size]
    std::vector<float> * embd_norm  = nullptr; // token embedding after RMSNorm*input_norm_w, shape: [hidden_size]

    std::vector<float> * hidden_proj = nullptr; // teacher hidden after optional fc projection, shape: [hidden_size]
    std::vector<float> * hidden_norm = nullptr; // teacher hidden after RMSNorm*hidden_norm_w, shape: [hidden_size]

    std::vector<float> * cat        = nullptr; // concat([embd_norm, hidden_norm]), shape: [hidden_size*2]

    // Attention projections (after reshape + RoPE for q/k, no RoPE for v).
    // Layout matches ggml's contiguous [head_dim, n_head, 1] (q) and [head_dim, n_kv, 1] (k/v).
    std::vector<float> * q = nullptr; // [head_dim * n_head]
    std::vector<float> * k = nullptr; // [head_dim * n_kv_heads]
    std::vector<float> * v = nullptr; // [head_dim * n_kv_heads]
};

struct llama_eagle3_logits_graph {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf_compute;
    ggml_cgraph *           gf = nullptr;

    ggml_tensor * t_hidden = nullptr;
    ggml_tensor * t_logits = nullptr;
};

struct llama_eagle3_step_graph {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf_compute;
    ggml_cgraph *           gf = nullptr;

    ggml_tensor * t_hidden_in = nullptr;
    ggml_tensor * t_tok       = nullptr;
    ggml_tensor * t_pos       = nullptr;
    ggml_tensor * t_k_past_input = nullptr;
    ggml_tensor * t_v_past_input = nullptr;

    ggml_tensor * t_hidden_out = nullptr;
    ggml_tensor * t_k_curr     = nullptr;
    ggml_tensor * t_v_curr     = nullptr;
    ggml_tensor * t_k_total    = nullptr;
    ggml_tensor * t_v_total    = nullptr;
    ggml_tensor * t_logits     = nullptr;
};

struct llama_eagle3_step_batch_graph {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf_compute;
    ggml_cgraph *           gf = nullptr;
    int32_t                 n_beams = 0;

    // Batched inputs (backing storage for per-beam views below).
    ggml_tensor * t_hidden_in_b = nullptr; // [hidden_in_dim, n_beams]
    ggml_tensor * t_tok_b       = nullptr; // [n_beams]
    ggml_tensor * t_pos_b       = nullptr; // [n_beams]

    std::vector<ggml_tensor *> t_hidden_in;
    std::vector<ggml_tensor *> t_tok;
    std::vector<ggml_tensor *> t_pos;
    std::vector<ggml_tensor *> t_k_past_input;
    std::vector<ggml_tensor *> t_v_past_input;

    std::vector<ggml_tensor *> t_hidden_out;
    std::vector<ggml_tensor *> t_k_total;
    std::vector<ggml_tensor *> t_v_total;
    std::vector<ggml_tensor *> t_logits;
};

struct llama_eagle3_topk_graph {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf_compute;
    ggml_cgraph *           gf = nullptr;
    int32_t                 k = 0;

    ggml_tensor * t_hidden    = nullptr;
    ggml_tensor * t_probs     = nullptr;
    ggml_tensor * t_topk_idx  = nullptr;
    ggml_tensor * t_topk_prob = nullptr;
};

struct llama_eagle3_topk_batch_graph {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf_compute;
    ggml_cgraph *           gf = nullptr;
    int32_t                 k = 0;
    int32_t                 n_beams = 0;

    ggml_tensor * t_hidden    = nullptr; // [hidden_size, n_beams]
    ggml_tensor * t_probs     = nullptr; // [draft_vocab_size, n_beams]
    ggml_tensor * t_topk_idx  = nullptr; // [k, n_beams]
    ggml_tensor * t_topk_prob = nullptr; // [k, n_beams]

    // Views into t_hidden for convenient per-beam filling without host staging.
    std::vector<ggml_tensor *> t_hidden_cols; // [n_beams], each [hidden_size, 1]
};

struct llama_eagle3_select_batch_graph {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf_compute;
    ggml_cgraph *           gf = nullptr;
    int32_t                 k = 0;
    int32_t                 n_beams = 0;
    int32_t                 n_select = 0;

    ggml_tensor * t_hidden            = nullptr; // [hidden_size, n_beams]
    ggml_tensor * t_beam_logprob      = nullptr; // [n_beams]
    ggml_tensor * t_d2t               = nullptr; // [draft_vocab_size]
    ggml_tensor * t_selected_linear   = nullptr; // [n_select, 1]
    ggml_tensor * t_selected_parent   = nullptr; // [n_select, 1]
    ggml_tensor * t_selected_draft    = nullptr; // [n_select, 1]
    ggml_tensor * t_selected_base     = nullptr; // [n_select, 1]
    ggml_tensor * t_selected_logprob  = nullptr; // [n_select, 1]

    std::vector<ggml_tensor *> t_hidden_cols; // [n_beams], each [hidden_size, 1]
};

struct llama_eagle3_runtime {
    const llama_model * base_model = nullptr;
    ggml_tensor * tok_embd = nullptr;

    llama_rope_type rope_type = LLAMA_ROPE_TYPE_NONE;
    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;
    ggml_context_ptr        rope_factors_ctx;
    std::vector<uint8_t>    rope_factors_buf;
    ggml_tensor *           rope_factors = nullptr;

    float yarn_ext_factor  = 1.0f;
    float yarn_attn_factor = 1.0f;
    float yarn_beta_fast   = 32.0f;
    float yarn_beta_slow   = 1.0f;
    int32_t n_ctx_orig     = 0;

    int32_t n_threads = 0;

    // Preferred backend for EAGLE head execution. Falls back to CPU path if unavailable.
    ggml_backend_ptr backend_compute;
    ggml_backend_buffer_type_t buft_compute = nullptr;
    ggml_backend_t target_backend = nullptr; // non-owning; source backend for target hidden capture tensors

    // Backend-local copies of EAGLE head weights (same layout as llama_eagle3_model::tensors).
    ggml_context_ptr        ctx_weights_compute;
    ggml_backend_buffer_ptr buf_weights_compute;
    llama_eagle3_tensors    tensors_compute;

    // Reusable compute graphs for backend execution.
    mutable llama_eagle3_logits_graph logits_graph;
    mutable llama_eagle3_topk_graph topk_graph;
    mutable llama_eagle3_topk_batch_graph topk_batch_graph;
    mutable llama_eagle3_select_batch_graph select_batch_graph;
    mutable uint64_t step_graph_key = ~uint64_t(0);
    mutable llama_eagle3_step_graph step_graph;
    mutable uint64_t step_batch_graph_key = ~uint64_t(0);
    mutable llama_eagle3_step_batch_graph step_batch_graph;
};

struct llama_eagle3_select_batch_device_result {
    ggml_backend_t backend = nullptr; // non-owning
    const ggml_tensor * t_selected_linear = nullptr;
    const ggml_tensor * t_selected_parent = nullptr;
    const ggml_tensor * t_selected_draft = nullptr;
    const ggml_tensor * t_selected_base = nullptr;
    const ggml_tensor * t_selected_logprob = nullptr;
    int32_t n_beams = 0;
    int32_t k = 0;
    int32_t n_select = 0;
};

llama_eagle3_model * llama_eagle3_load(const std::string & path, std::string & err);
void llama_eagle3_free(llama_eagle3_model * model);

llama_eagle3_runtime llama_eagle3_make_runtime(
        const llama_context * ctx_tgt,
        const llama_eagle3_model * model,
        int32_t n_threads);

bool llama_eagle3_step(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        llama_eagle3_state & state,
        const float * hidden_in,
        int32_t hidden_in_dim,
        llama_token input_id,
        std::vector<float> * logits_out,
        llama_eagle3_step_debug * dbg = nullptr);

bool llama_eagle3_step_from_hidden_capture(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        llama_eagle3_state & state,
        const std::vector<const ggml_tensor *> & hidden_capture,
        size_t token_idx,
        llama_token input_id,
        std::vector<float> * logits_out,
        llama_eagle3_step_debug * dbg = nullptr);

bool llama_eagle3_logits(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const float * hidden,
        std::vector<float> & logits_out);

bool llama_eagle3_topk(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const float * hidden,
        int32_t k,
        std::vector<int32_t> & topk_idx_out,
        std::vector<float> & topk_prob_out);

bool llama_eagle3_topk_state(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const llama_eagle3_state & state,
        int32_t k,
        std::vector<int32_t> & topk_idx_out,
        std::vector<float> & topk_prob_out);

bool llama_eagle3_topk_state_batch(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const std::vector<const llama_eagle3_state *> & states,
        int32_t k,
        std::vector<int32_t> & topk_idx_out,
        std::vector<float> & topk_prob_out);

bool llama_eagle3_select_state_batch(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const std::vector<const llama_eagle3_state *> & states,
        const std::vector<float> & beam_logprob,
        int32_t k,
        std::vector<int32_t> & selected_linear_out,
        std::vector<int32_t> & selected_draft_idx_out,
        std::vector<float> & selected_logprob_out);

bool llama_eagle3_select_state_batch_device(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const std::vector<const llama_eagle3_state *> & states,
        const std::vector<float> & beam_logprob,
        int32_t k,
        llama_eagle3_select_batch_device_result & out);

bool llama_eagle3_select_state_slots(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const std::vector<const llama_eagle3_state *> & states,
        const std::vector<uint8_t> & active_mask,
        const std::vector<float> & beam_logprob,
        int32_t k,
        std::vector<int32_t> & selected_linear_out,
        std::vector<int32_t> & selected_draft_idx_out,
        std::vector<float> & selected_logprob_out);

bool llama_eagle3_select_state_slots_device(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const std::vector<const llama_eagle3_state *> & states,
        const std::vector<uint8_t> & active_mask,
        const std::vector<float> & beam_logprob,
        int32_t k,
        llama_eagle3_select_batch_device_result & out);

bool llama_eagle3_state_has_hidden(const llama_eagle3_state & state);
bool llama_eagle3_state_get_hidden(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        llama_eagle3_state & state,
        std::vector<float> & hidden_out);

bool llama_eagle3_step_batch(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const std::vector<llama_eagle3_state *> & states,
        int32_t hidden_in_dim,
        const std::vector<llama_token> & input_ids);

bool llama_eagle3_step_batch_from_parents(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const std::vector<const llama_eagle3_state *> & parent_states,
        int32_t hidden_in_dim,
        const std::vector<llama_token> & input_ids,
        const std::vector<llama_eagle3_state *> & out_states,
        int32_t reserve_kv);
