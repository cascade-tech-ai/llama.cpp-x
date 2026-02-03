// AI-GENERATED: This file was created with AI assistance for an experimental fork.
// DO NOT SUBMIT upstream unless rewritten or exhaustively reviewed by a human.
#pragma once

#include "ggml-cpp.h"
#include "llama.h"

#include <cstdint>
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
};

struct llama_eagle3_runtime {
    const llama_model * base_model = nullptr;

    llama_rope_type rope_type = LLAMA_ROPE_TYPE_NONE;
    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;

    float yarn_ext_factor  = 1.0f;
    float yarn_attn_factor = 1.0f;
    float yarn_beta_fast   = 32.0f;
    float yarn_beta_slow   = 1.0f;
    int32_t n_ctx_orig     = 0;

    int32_t n_threads = 0;
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
        std::vector<float> * logits_out);

bool llama_eagle3_logits(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const float * hidden,
        std::vector<float> & logits_out);
