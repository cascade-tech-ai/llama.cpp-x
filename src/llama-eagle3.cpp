// AI-GENERATED: This file was created with AI assistance for an experimental fork.
// DO NOT SUBMIT upstream unless rewritten or exhaustively reviewed by a human.
#include "llama-eagle3.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-context.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

struct llama_eagle3_state::device_state {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor *           t_hidden = nullptr;
    ggml_tensor *           t_k = nullptr;
    ggml_tensor *           t_v = nullptr;
    int32_t                 past_len = 0;
};

namespace {

constexpr const char * EAGLE3_ARCH = "eagle3";

constexpr const char * EAGLE3_KEY_HIDDEN_SIZE        = "eagle3.hidden_size";
constexpr const char * EAGLE3_KEY_INTERMEDIATE_SIZE  = "eagle3.intermediate_size";
constexpr const char * EAGLE3_KEY_NUM_HEADS          = "eagle3.num_attention_heads";
constexpr const char * EAGLE3_KEY_NUM_KV_HEADS       = "eagle3.num_key_value_heads";
constexpr const char * EAGLE3_KEY_RMS_EPS            = "eagle3.rms_norm_eps";
constexpr const char * EAGLE3_KEY_HIDDEN_CONCAT      = "eagle3.hidden_concat";
constexpr const char * EAGLE3_KEY_TARGET_HIDDEN_SIZE = "eagle3.target_hidden_size";
constexpr const char * EAGLE3_KEY_DRAFT_VOCAB_SIZE    = "eagle3.draft_vocab_size";
constexpr const char * EAGLE3_KEY_VOCAB_SIZE          = "eagle3.vocab_size";
constexpr const char * EAGLE3_KEY_HEAD_DIM           = "eagle3.head_dim";
constexpr const char * EAGLE3_KEY_NORM_BEFORE_RESIDUAL = "eagle3.norm_before_residual";
constexpr const char * EAGLE3_KEY_HIDDEN_LAYER_IDS    = "eagle3.hidden_state_layer_ids";
constexpr const char * EAGLE3_KEY_D2T                 = "eagle3.d2t";

// fallback keys aligned with gguf LLM conventions (optional)
constexpr const char * EAGLE3_KEY_EMBEDDING_LENGTH      = "eagle3.embedding_length";
constexpr const char * EAGLE3_KEY_FFN_LENGTH            = "eagle3.feed_forward_length";
constexpr const char * EAGLE3_KEY_ATTN_HEAD_COUNT       = "eagle3.attention.head_count";
constexpr const char * EAGLE3_KEY_ATTN_HEAD_COUNT_KV    = "eagle3.attention.head_count_kv";
constexpr const char * EAGLE3_KEY_ATTN_RMS_EPS          = "eagle3.attention.layer_norm_rms_epsilon";

constexpr const char * EAGLE3_TENSOR_FC          = "eagle3.fc.weight";
constexpr const char * EAGLE3_TENSOR_NORM        = "eagle3.norm.weight";
constexpr const char * EAGLE3_TENSOR_LM_HEAD     = "eagle3.lm_head.weight";
constexpr const char * EAGLE3_TENSOR_HIDDEN_NORM = "eagle3.hidden_norm.weight";
constexpr const char * EAGLE3_TENSOR_INPUT_NORM  = "eagle3.input_layernorm.weight";
constexpr const char * EAGLE3_TENSOR_POST_NORM   = "eagle3.post_attention_layernorm.weight";
constexpr const char * EAGLE3_TENSOR_ATTN_Q      = "eagle3.attn_q.weight";
constexpr const char * EAGLE3_TENSOR_ATTN_K      = "eagle3.attn_k.weight";
constexpr const char * EAGLE3_TENSOR_ATTN_V      = "eagle3.attn_v.weight";
constexpr const char * EAGLE3_TENSOR_ATTN_O      = "eagle3.attn_o.weight";
constexpr const char * EAGLE3_TENSOR_ATTN_Q_B    = "eagle3.attn_q.bias";
constexpr const char * EAGLE3_TENSOR_ATTN_K_B    = "eagle3.attn_k.bias";
constexpr const char * EAGLE3_TENSOR_ATTN_V_B    = "eagle3.attn_v.bias";
constexpr const char * EAGLE3_TENSOR_ATTN_O_B    = "eagle3.attn_o.bias";
constexpr const char * EAGLE3_TENSOR_FFN_GATE    = "eagle3.ffn_gate.weight";
constexpr const char * EAGLE3_TENSOR_FFN_UP      = "eagle3.ffn_up.weight";
constexpr const char * EAGLE3_TENSOR_FFN_DOWN    = "eagle3.ffn_down.weight";
constexpr const char * EAGLE3_TENSOR_FFN_GATE_B  = "eagle3.ffn_gate.bias";
constexpr const char * EAGLE3_TENSOR_FFN_UP_B    = "eagle3.ffn_up.bias";
constexpr const char * EAGLE3_TENSOR_FFN_DOWN_B  = "eagle3.ffn_down.bias";

int32_t get_kv_i32(const gguf_context * ctx, const char * key, bool required, int32_t fallback = 0) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        if (required) {
            throw std::runtime_error(std::string("missing required key: ") + key);
        }
        return fallback;
    }
    return gguf_get_val_i32(ctx, kid);
}

float get_kv_f32(const gguf_context * ctx, const char * key, bool required, float fallback = 0.0f) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        if (required) {
            throw std::runtime_error(std::string("missing required key: ") + key);
        }
        return fallback;
    }
    return gguf_get_val_f32(ctx, kid);
}

bool get_kv_bool(const gguf_context * ctx, const char * key, bool required, bool fallback = false) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        if (required) {
            throw std::runtime_error(std::string("missing required key: ") + key);
        }
        return fallback;
    }
    return gguf_get_val_bool(ctx, kid);
}

std::vector<int32_t> get_kv_arr_i32(const gguf_context * ctx, const char * key, bool required) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        if (required) {
            throw std::runtime_error(std::string("missing required key: ") + key);
        }
        return {};
    }
    if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY) {
        throw std::runtime_error(std::string("invalid array type for key: ") + key);
    }
    const auto arr_type = gguf_get_arr_type(ctx, kid);
    const size_t n = gguf_get_arr_n(ctx, kid);
    if (arr_type == GGUF_TYPE_INT32) {
        const int32_t * data = reinterpret_cast<const int32_t *>(gguf_get_arr_data(ctx, kid));
        return std::vector<int32_t>(data, data + n);
    }
    if (arr_type == GGUF_TYPE_INT64) {
        const int64_t * data = reinterpret_cast<const int64_t *>(gguf_get_arr_data(ctx, kid));
        std::vector<int32_t> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (data[i] < std::numeric_limits<int32_t>::min() || data[i] > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error(std::string("array value out of int32 range for key: ") + key);
            }
            out.push_back(static_cast<int32_t>(data[i]));
        }
        return out;
    }
    throw std::runtime_error(std::string("invalid array element type for key: ") + key);
}

ggml_tensor * require_tensor(ggml_context * ctx, const gguf_context * gguf_ctx, const char * name) {
    const int64_t tid = gguf_find_tensor(gguf_ctx, name);
    if (tid < 0) {
        throw std::runtime_error(std::string("missing tensor: ") + name);
    }
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        throw std::runtime_error(std::string("tensor not found in context: ") + name);
    }
    return t;
}

ggml_tensor * optional_tensor(ggml_context * ctx, const gguf_context * gguf_ctx, const char * name) {
    const int64_t tid = gguf_find_tensor(gguf_ctx, name);
    if (tid < 0) {
        return nullptr;
    }
    return ggml_get_tensor(ctx, name);
}

void check_shape(const ggml_tensor * t, int64_t n0, int64_t n1, int64_t n2 = 1, int64_t n3 = 1) {
    if (t->ne[0] != n0 || t->ne[1] != n1 || t->ne[2] != n2 || t->ne[3] != n3) {
        throw std::runtime_error("tensor shape mismatch for " + std::string(t->name));
    }
}

void load_tensor_data(
        const gguf_context * gguf_ctx,
        llama_file & file,
        ggml_tensor * t,
        std::vector<uint8_t> & scratch) {
    const int64_t tid = gguf_find_tensor(gguf_ctx, t->name);
    if (tid < 0) {
        throw std::runtime_error(std::string("missing tensor data for ") + t->name);
    }
    const size_t offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tid);
    const size_t size = ggml_nbytes(t);
    scratch.resize(size);
    file.seek(offs, SEEK_SET);
    file.read_raw(scratch.data(), size);
    ggml_backend_tensor_set(t, scratch.data(), 0, size);
}

size_t estimate_step_mem(const llama_eagle3_hparams & hp, int32_t past_len) {
    const size_t kv_bytes = (size_t) (past_len + 1) * hp.head_dim * hp.num_kv_heads * sizeof(float);
    const size_t base = 32ull * 1024ull * 1024ull;
    const size_t extra = kv_bytes * 4;
    return base + extra;
}

ggml_context_ptr make_ctx_with_buf(std::vector<uint8_t> & buf, size_t bytes) {
    if (buf.size() < bytes) {
        buf.resize(bytes);
    }
    ggml_init_params params = {
        /* .mem_size   = */ buf.size(),
        /* .mem_buffer = */ buf.data(),
        /* .no_alloc   = */ false,
    };
    return ggml_context_ptr(ggml_init(params));
}

ggml_context_ptr make_ctx_no_alloc(size_t max_nodes) {
    const size_t mem_size =
            ggml_tensor_overhead() * max_nodes +
            ggml_graph_overhead_custom(max_nodes, /* grads = */ false);
    ggml_init_params params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    return ggml_context_ptr(ggml_init(params));
}

uint64_t make_step_graph_key(int32_t past_len, int32_t hidden_in_dim, bool with_logits) {
    return (uint64_t(uint32_t(past_len)) << 32) |
           (uint64_t(uint32_t(hidden_in_dim)) << 1) |
           (with_logits ? 1ull : 0ull);
}

void maybe_set_backend_threads(ggml_backend_t backend, int32_t n_threads) {
    if (!backend || n_threads <= 0) {
        return;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) {
        return;
    }

    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads) {
        set_n_threads(backend, n_threads);
    }
}

template<typename T>
void set_name_same(T * dst, const T * src) {
    if (dst && src && src->name) {
        ggml_set_name(dst, src->name);
    }
}

const llama_eagle3_tensors & get_runtime_tensors(const llama_eagle3_model & model, const llama_eagle3_runtime & rt) {
    if (rt.tensors_compute.fc_w) {
        return rt.tensors_compute;
    }
    return model.tensors;
}

bool alloc_state_device(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t past_len,
        std::shared_ptr<llama_eagle3_state::device_state> & out) {
    if (!rt.buft_compute || past_len < 0) {
        return false;
    }

    const auto & hp = model.hparams;
    auto ctx = make_ctx_no_alloc(/* max_nodes = */ 32);
    if (!ctx) {
        return false;
    }

    ggml_tensor * t_hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hp.hidden_size, 1);
    ggml_tensor * t_k = nullptr;
    ggml_tensor * t_v = nullptr;
    if (past_len > 0) {
        t_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, past_len);
        t_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, past_len);
    }

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    if (!buf) {
        return false;
    }

    auto dev = std::make_shared<llama_eagle3_state::device_state>();
    dev->ctx = std::move(ctx);
    dev->buf = std::move(buf);
    dev->t_hidden = t_hidden;
    dev->t_k = t_k;
    dev->t_v = t_v;
    dev->past_len = past_len;
    out = std::move(dev);
    return true;
}

bool copy_weight_tensors_to_backend(const llama_eagle3_model & model, llama_eagle3_runtime & rt) {
    if (!rt.backend_compute || !rt.buft_compute) {
        return false;
    }

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 64 + 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    rt.ctx_weights_compute.reset(ggml_init(params));
    if (!rt.ctx_weights_compute) {
        return false;
    }

    auto dup = [&](ggml_tensor * src) -> ggml_tensor * {
        if (!src) {
            return nullptr;
        }
        ggml_tensor * dst = ggml_dup_tensor(rt.ctx_weights_compute.get(), src);
        set_name_same(dst, src);
        return dst;
    };

    rt.tensors_compute.fc_w          = dup(model.tensors.fc_w);
    rt.tensors_compute.norm_w        = dup(model.tensors.norm_w);
    rt.tensors_compute.lm_head_w     = dup(model.tensors.lm_head_w);
    rt.tensors_compute.hidden_norm_w = dup(model.tensors.hidden_norm_w);
    rt.tensors_compute.input_norm_w  = dup(model.tensors.input_norm_w);
    rt.tensors_compute.post_norm_w   = dup(model.tensors.post_norm_w);
    rt.tensors_compute.attn_q_w      = dup(model.tensors.attn_q_w);
    rt.tensors_compute.attn_k_w      = dup(model.tensors.attn_k_w);
    rt.tensors_compute.attn_v_w      = dup(model.tensors.attn_v_w);
    rt.tensors_compute.attn_o_w      = dup(model.tensors.attn_o_w);
    rt.tensors_compute.ffn_gate_w    = dup(model.tensors.ffn_gate_w);
    rt.tensors_compute.ffn_up_w      = dup(model.tensors.ffn_up_w);
    rt.tensors_compute.ffn_down_w    = dup(model.tensors.ffn_down_w);

    rt.tensors_compute.attn_q_b      = dup(model.tensors.attn_q_b);
    rt.tensors_compute.attn_k_b      = dup(model.tensors.attn_k_b);
    rt.tensors_compute.attn_v_b      = dup(model.tensors.attn_v_b);
    rt.tensors_compute.attn_o_b      = dup(model.tensors.attn_o_b);
    rt.tensors_compute.ffn_gate_b    = dup(model.tensors.ffn_gate_b);
    rt.tensors_compute.ffn_up_b      = dup(model.tensors.ffn_up_b);
    rt.tensors_compute.ffn_down_b    = dup(model.tensors.ffn_down_b);

    rt.buf_weights_compute.reset(ggml_backend_alloc_ctx_tensors_from_buft(rt.ctx_weights_compute.get(), rt.buft_compute));
    if (!rt.buf_weights_compute) {
        return false;
    }

    auto copy = [&](ggml_tensor * src, ggml_tensor * dst) {
        if (src && dst) {
            ggml_backend_tensor_copy(src, dst);
        }
    };

    copy(model.tensors.fc_w,          rt.tensors_compute.fc_w);
    copy(model.tensors.norm_w,        rt.tensors_compute.norm_w);
    copy(model.tensors.lm_head_w,     rt.tensors_compute.lm_head_w);
    copy(model.tensors.hidden_norm_w, rt.tensors_compute.hidden_norm_w);
    copy(model.tensors.input_norm_w,  rt.tensors_compute.input_norm_w);
    copy(model.tensors.post_norm_w,   rt.tensors_compute.post_norm_w);
    copy(model.tensors.attn_q_w,      rt.tensors_compute.attn_q_w);
    copy(model.tensors.attn_k_w,      rt.tensors_compute.attn_k_w);
    copy(model.tensors.attn_v_w,      rt.tensors_compute.attn_v_w);
    copy(model.tensors.attn_o_w,      rt.tensors_compute.attn_o_w);
    copy(model.tensors.ffn_gate_w,    rt.tensors_compute.ffn_gate_w);
    copy(model.tensors.ffn_up_w,      rt.tensors_compute.ffn_up_w);
    copy(model.tensors.ffn_down_w,    rt.tensors_compute.ffn_down_w);

    copy(model.tensors.attn_q_b,      rt.tensors_compute.attn_q_b);
    copy(model.tensors.attn_k_b,      rt.tensors_compute.attn_k_b);
    copy(model.tensors.attn_v_b,      rt.tensors_compute.attn_v_b);
    copy(model.tensors.attn_o_b,      rt.tensors_compute.attn_o_b);
    copy(model.tensors.ffn_gate_b,    rt.tensors_compute.ffn_gate_b);
    copy(model.tensors.ffn_up_b,      rt.tensors_compute.ffn_up_b);
    copy(model.tensors.ffn_down_b,    rt.tensors_compute.ffn_down_b);

    return true;
}

bool build_logits_graph(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        llama_eagle3_logits_graph & graph) {
    if (graph.ctx && graph.buf_compute && graph.gf && graph.t_hidden && graph.t_logits) {
        return true;
    }

    const auto & hp = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);

    auto ctx = make_ctx_no_alloc(/* max_nodes = */ 256);
    if (!ctx) {
        return false;
    }

    ggml_tensor * t_hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hp.hidden_size, 1);
    ggml_set_input(t_hidden);

    ggml_tensor * t_norm = ggml_rms_norm(ctx.get(), t_hidden, hp.rms_norm_eps);
    ggml_tensor * t_norm_w = ggml_cast(ctx.get(), tensors.norm_w, GGML_TYPE_F32);
    t_norm = ggml_mul(ctx.get(), t_norm, t_norm_w);

    ggml_tensor * t_logits = ggml_mul_mat(ctx.get(), tensors.lm_head_w, t_norm);

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, t_logits);

    ggml_backend_buffer_ptr buf_compute;
    if (rt.buft_compute) {
        buf_compute.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    }

    if (!buf_compute) {
        return false;
    }

    graph.ctx = std::move(ctx);
    graph.buf_compute = std::move(buf_compute);
    graph.gf = gf;
    graph.t_hidden = t_hidden;
    graph.t_logits = t_logits;
    return true;
}

bool build_topk_graph(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t k,
        llama_eagle3_topk_graph & graph) {
    if (graph.ctx && graph.buf_compute && graph.gf && graph.t_hidden && graph.t_probs && graph.t_topk_idx && graph.k == k) {
        return true;
    }

    const auto & hp = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);

    auto ctx = make_ctx_no_alloc(/* max_nodes = */ 320);
    if (!ctx) {
        return false;
    }

    ggml_tensor * t_hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hp.hidden_size, 1);
    ggml_set_input(t_hidden);

    ggml_tensor * t_norm = ggml_rms_norm(ctx.get(), t_hidden, hp.rms_norm_eps);
    ggml_tensor * t_norm_w = ggml_cast(ctx.get(), tensors.norm_w, GGML_TYPE_F32);
    t_norm = ggml_mul(ctx.get(), t_norm, t_norm_w);

    ggml_tensor * t_logits = ggml_mul_mat(ctx.get(), tensors.lm_head_w, t_norm);
    ggml_tensor * t_probs = ggml_soft_max(ctx.get(), t_logits);
    t_probs = ggml_cont(ctx.get(), t_probs);
    ggml_tensor * t_topk_idx = ggml_top_k(ctx.get(), t_probs, k);
    t_topk_idx = ggml_cont(ctx.get(), t_topk_idx);
    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, t_topk_idx);

    ggml_backend_buffer_ptr buf_compute;
    if (rt.buft_compute) {
        buf_compute.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    }

    if (!buf_compute) {
        return false;
    }

    graph.ctx = std::move(ctx);
    graph.buf_compute = std::move(buf_compute);
    graph.gf = gf;
    graph.k = k;
    graph.t_hidden = t_hidden;
    graph.t_probs = t_probs;
    graph.t_topk_idx = t_topk_idx;
    graph.t_topk_prob = nullptr;
    return true;
}

bool build_topk_batch_graph(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t n_beams,
        int32_t k,
        llama_eagle3_topk_batch_graph & graph) {
    if (graph.ctx && graph.buf_compute && graph.gf &&
        graph.t_hidden && graph.t_probs && graph.t_topk_idx &&
        graph.n_beams == n_beams && graph.k == k) {
        return true;
    }

    const auto & hp = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);

    auto ctx = make_ctx_no_alloc(/* max_nodes = */ 384);
    if (!ctx) {
        return false;
    }

    ggml_tensor * t_hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hp.hidden_size, n_beams);
    ggml_set_input(t_hidden);

    ggml_tensor * t_norm = ggml_rms_norm(ctx.get(), t_hidden, hp.rms_norm_eps);
    ggml_tensor * t_norm_w = ggml_cast(ctx.get(), tensors.norm_w, GGML_TYPE_F32);
    t_norm = ggml_mul(ctx.get(), t_norm, t_norm_w);

    ggml_tensor * t_logits = ggml_mul_mat(ctx.get(), tensors.lm_head_w, t_norm);
    ggml_tensor * t_probs = ggml_soft_max(ctx.get(), t_logits);
    t_probs = ggml_cont(ctx.get(), t_probs);
    ggml_tensor * t_topk_idx = ggml_top_k(ctx.get(), t_probs, k);
    t_topk_idx = ggml_cont(ctx.get(), t_topk_idx);

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, t_topk_idx);

    ggml_backend_buffer_ptr buf_compute;
    if (rt.buft_compute) {
        buf_compute.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    }
    if (!buf_compute) {
        return false;
    }

    graph.ctx = std::move(ctx);
    graph.buf_compute = std::move(buf_compute);
    graph.gf = gf;
    graph.k = k;
    graph.n_beams = n_beams;
    graph.t_hidden = t_hidden;
    graph.t_probs = t_probs;
    graph.t_topk_idx = t_topk_idx;
    return true;
}

bool build_step_graph(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t past_len,
        int32_t hidden_in_dim,
        bool with_logits,
        llama_eagle3_step_graph & graph) {
    if (graph.ctx && graph.buf_compute && graph.gf && graph.t_hidden_in && graph.t_hidden_out && graph.t_k_total && graph.t_v_total) {
        return true;
    }

    if (!rt.tok_embd) {
        return false;
    }

    const auto & hp = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);

    auto ctx = make_ctx_no_alloc(/* max_nodes = */ 2048);
    if (!ctx) {
        return false;
    }

    ggml_tensor * t_hidden_in = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden_in_dim, 1);
    ggml_set_input(t_hidden_in);

    ggml_tensor * t_tok = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_set_input(t_tok);

    ggml_tensor * t_embd = ggml_get_rows(ctx.get(), rt.tok_embd, t_tok);
    t_embd = ggml_cast(ctx.get(), t_embd, GGML_TYPE_F32);

    ggml_tensor * t_hidden = t_hidden_in;
    if (hidden_in_dim != hp.hidden_size) {
        t_hidden = ggml_mul_mat(ctx.get(), tensors.fc_w, t_hidden_in);
    }

    ggml_tensor * t_hidden_norm = ggml_rms_norm(ctx.get(), t_hidden, hp.rms_norm_eps);
    ggml_tensor * t_hidden_norm_w = ggml_cast(ctx.get(), tensors.hidden_norm_w, GGML_TYPE_F32);
    t_hidden_norm = ggml_mul(ctx.get(), t_hidden_norm, t_hidden_norm_w);

    ggml_tensor * t_embd_norm = ggml_rms_norm(ctx.get(), t_embd, hp.rms_norm_eps);
    ggml_tensor * t_input_norm_w = ggml_cast(ctx.get(), tensors.input_norm_w, GGML_TYPE_F32);
    t_embd_norm = ggml_mul(ctx.get(), t_embd_norm, t_input_norm_w);

    ggml_tensor * t_cat = ggml_concat(ctx.get(), t_embd_norm, t_hidden_norm, 0);

    ggml_tensor * t_q = ggml_mul_mat(ctx.get(), tensors.attn_q_w, t_cat);
    if (tensors.attn_q_b) {
        ggml_tensor * t_q_b = ggml_cast(ctx.get(), tensors.attn_q_b, GGML_TYPE_F32);
        t_q = ggml_add(ctx.get(), t_q, t_q_b);
    }
    ggml_tensor * t_k = ggml_mul_mat(ctx.get(), tensors.attn_k_w, t_cat);
    if (tensors.attn_k_b) {
        ggml_tensor * t_k_b = ggml_cast(ctx.get(), tensors.attn_k_b, GGML_TYPE_F32);
        t_k = ggml_add(ctx.get(), t_k, t_k_b);
    }
    ggml_tensor * t_v = ggml_mul_mat(ctx.get(), tensors.attn_v_w, t_cat);
    if (tensors.attn_v_b) {
        ggml_tensor * t_v_b = ggml_cast(ctx.get(), tensors.attn_v_b, GGML_TYPE_F32);
        t_v = ggml_add(ctx.get(), t_v, t_v_b);
    }

    t_q = ggml_reshape_3d(ctx.get(), t_q, hp.head_dim, hp.num_heads, 1);
    t_k = ggml_reshape_3d(ctx.get(), t_k, hp.head_dim, hp.num_kv_heads, 1);
    t_v = ggml_reshape_3d(ctx.get(), t_v, hp.head_dim, hp.num_kv_heads, 1);

    ggml_tensor * t_pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_set_input(t_pos);

    t_q = ggml_rope_ext(
            ctx.get(), t_q, t_pos, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);

    t_k = ggml_rope_ext(
            ctx.get(), t_k, t_pos, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);

    ggml_tensor * t_k_total = t_k;
    ggml_tensor * t_v_total = t_v;
    ggml_tensor * t_k_past_input = nullptr;
    ggml_tensor * t_v_past_input = nullptr;

    if (past_len > 0) {
        t_k_past_input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, past_len);
        t_v_past_input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, past_len);
        ggml_set_input(t_k_past_input);
        ggml_set_input(t_v_past_input);

        t_k_total = ggml_concat(ctx.get(), t_k_past_input, t_k, 2);
        t_v_total = ggml_concat(ctx.get(), t_v_past_input, t_v, 2);
    }

    ggml_tensor * t_k_attn = t_k_total;
    ggml_tensor * t_v_attn = t_v_total;

    if (hp.num_kv_heads != hp.num_heads) {
        const int32_t n_rep = hp.num_heads / hp.num_kv_heads;

        ggml_tensor * k4 = ggml_reshape_4d(ctx.get(), t_k_attn, hp.head_dim, hp.num_kv_heads, t_k_attn->ne[2], 1);
        ggml_tensor * v4 = ggml_reshape_4d(ctx.get(), t_v_attn, hp.head_dim, hp.num_kv_heads, t_v_attn->ne[2], 1);

        k4 = ggml_permute(ctx.get(), k4, 0, 2, 3, 1);
        v4 = ggml_permute(ctx.get(), v4, 0, 2, 3, 1);

        k4 = ggml_repeat_4d(ctx.get(), k4, hp.head_dim, n_rep, hp.num_kv_heads, t_k_attn->ne[2]);
        v4 = ggml_repeat_4d(ctx.get(), v4, hp.head_dim, n_rep, hp.num_kv_heads, t_v_attn->ne[2]);

        k4 = ggml_cont(ctx.get(), k4);
        v4 = ggml_cont(ctx.get(), v4);

        t_k_attn = ggml_reshape_3d(ctx.get(), k4, hp.head_dim, hp.num_heads, t_k_attn->ne[2]);
        t_v_attn = ggml_reshape_3d(ctx.get(), v4, hp.head_dim, hp.num_heads, t_v_attn->ne[2]);
    }

    ggml_tensor * qv = ggml_view_4d(ctx.get(), t_q, t_q->ne[0], t_q->ne[1], t_q->ne[2], 1, t_q->nb[1], t_q->nb[2], t_q->nb[3], 0);
    ggml_tensor * kv = ggml_view_4d(ctx.get(), t_k_attn, t_k_attn->ne[0], t_k_attn->ne[1], t_k_attn->ne[2], 1,
                                    t_k_attn->nb[1], t_k_attn->nb[2], t_k_attn->nb[3], 0);
    ggml_tensor * vv = ggml_view_4d(ctx.get(), t_v_attn, t_v_attn->ne[0], t_v_attn->ne[1], t_v_attn->ne[2], 1,
                                    t_v_attn->nb[1], t_v_attn->nb[2], t_v_attn->nb[3], 0);

    qv = ggml_permute(ctx.get(), qv, 0, 2, 1, 3);
    kv = ggml_permute(ctx.get(), kv, 0, 2, 1, 3);
    vv = ggml_permute(ctx.get(), vv, 0, 2, 1, 3);

    ggml_tensor * kq = ggml_mul_mat(ctx.get(), kv, qv);
    const float kq_scale = 1.0f / std::sqrt(float(hp.head_dim));
    kq = ggml_scale(ctx.get(), kq, kq_scale);
    kq = ggml_cont(ctx.get(), kq);
    kq = ggml_soft_max(ctx.get(), kq);

    ggml_tensor * vv_t = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), vv));
    ggml_tensor * kqv = ggml_mul_mat(ctx.get(), vv_t, kq);
    ggml_tensor * attn_out = ggml_permute(ctx.get(), kqv, 0, 2, 1, 3);
    attn_out = ggml_cont_2d(ctx.get(), attn_out, attn_out->ne[0]*attn_out->ne[1], attn_out->ne[2]*attn_out->ne[3]);

    ggml_tensor * t_attn = ggml_mul_mat(ctx.get(), tensors.attn_o_w, attn_out);
    if (tensors.attn_o_b) {
        ggml_tensor * t_attn_b = ggml_cast(ctx.get(), tensors.attn_o_b, GGML_TYPE_F32);
        t_attn = ggml_add(ctx.get(), t_attn, t_attn_b);
    }

    ggml_tensor * t_resid = hp.norm_before_residual ? t_hidden_norm : t_hidden;
    ggml_tensor * t_hidden_attn = ggml_add(ctx.get(), t_attn, t_resid);

    ggml_tensor * t_post = ggml_rms_norm(ctx.get(), t_hidden_attn, hp.rms_norm_eps);
    ggml_tensor * t_post_norm_w = ggml_cast(ctx.get(), tensors.post_norm_w, GGML_TYPE_F32);
    t_post = ggml_mul(ctx.get(), t_post, t_post_norm_w);

    ggml_tensor * t_gate = ggml_mul_mat(ctx.get(), tensors.ffn_gate_w, t_post);
    if (tensors.ffn_gate_b) {
        ggml_tensor * t_gate_b = ggml_cast(ctx.get(), tensors.ffn_gate_b, GGML_TYPE_F32);
        t_gate = ggml_add(ctx.get(), t_gate, t_gate_b);
    }
    ggml_tensor * t_up = ggml_mul_mat(ctx.get(), tensors.ffn_up_w, t_post);
    if (tensors.ffn_up_b) {
        ggml_tensor * t_up_b = ggml_cast(ctx.get(), tensors.ffn_up_b, GGML_TYPE_F32);
        t_up = ggml_add(ctx.get(), t_up, t_up_b);
    }
    ggml_tensor * t_act = ggml_mul(ctx.get(), ggml_silu(ctx.get(), t_gate), t_up);
    ggml_tensor * t_ffn = ggml_mul_mat(ctx.get(), tensors.ffn_down_w, t_act);
    if (tensors.ffn_down_b) {
        ggml_tensor * t_down_b = ggml_cast(ctx.get(), tensors.ffn_down_b, GGML_TYPE_F32);
        t_ffn = ggml_add(ctx.get(), t_ffn, t_down_b);
    }

    ggml_tensor * t_hidden_out = ggml_add(ctx.get(), t_hidden_attn, t_ffn);
    t_hidden_out = ggml_cont(ctx.get(), t_hidden_out);
    ggml_tensor * t_k_total_out = ggml_cont(ctx.get(), t_k_total);
    ggml_tensor * t_v_total_out = ggml_cont(ctx.get(), t_v_total);

    ggml_tensor * t_logits = nullptr;
    if (with_logits) {
        ggml_tensor * t_norm = ggml_rms_norm(ctx.get(), t_hidden_out, hp.rms_norm_eps);
        ggml_tensor * t_norm_w = ggml_cast(ctx.get(), tensors.norm_w, GGML_TYPE_F32);
        t_norm = ggml_mul(ctx.get(), t_norm, t_norm_w);
        t_logits = ggml_mul_mat(ctx.get(), tensors.lm_head_w, t_norm);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, t_hidden_out);
    ggml_build_forward_expand(gf, t_k_total_out);
    ggml_build_forward_expand(gf, t_v_total_out);
    if (t_logits) {
        ggml_build_forward_expand(gf, t_logits);
    }

    ggml_backend_buffer_ptr buf_compute;
    if (rt.buft_compute) {
        buf_compute.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    }

    if (!buf_compute) {
        return false;
    }

    graph.ctx = std::move(ctx);
    graph.buf_compute = std::move(buf_compute);
    graph.gf = gf;
    graph.t_hidden_in = t_hidden_in;
    graph.t_tok = t_tok;
    graph.t_pos = t_pos;
    graph.t_k_past_input = t_k_past_input;
    graph.t_v_past_input = t_v_past_input;
    graph.t_hidden_out = t_hidden_out;
    graph.t_k_curr = t_k;
    graph.t_v_curr = t_v;
    graph.t_k_total = t_k_total_out;
    graph.t_v_total = t_v_total_out;
    graph.t_logits = t_logits;
    return true;
}

} // namespace

llama_eagle3_model * llama_eagle3_load(const std::string & path, std::string & err) {
    try {
        ggml_context * ctx_meta = nullptr;
        gguf_init_params meta_params = {
            /* .no_alloc = */ true,
            /* .ctx      = */ &ctx_meta,
        };

        gguf_context_ptr ctx_gguf { gguf_init_from_file(path.c_str(), meta_params) };
        if (!ctx_gguf) {
            err = "failed to open GGUF";
            return nullptr;
        }

        const int64_t arch_id = gguf_find_key(ctx_gguf.get(), "general.architecture");
        if (arch_id < 0) {
            err = "missing general.architecture";
            return nullptr;
        }
        const std::string arch = gguf_get_val_str(ctx_gguf.get(), arch_id);
        if (arch != EAGLE3_ARCH) {
            err = "unexpected architecture: " + arch;
            return nullptr;
        }

        auto model = std::make_unique<llama_eagle3_model>();
        model->ctx_weights.reset(ctx_meta);

        llama_eagle3_hparams hp;

        hp.hidden_size = get_kv_i32(ctx_gguf.get(), EAGLE3_KEY_HIDDEN_SIZE, false, 0);
        if (hp.hidden_size == 0) {
            hp.hidden_size = get_kv_i32(ctx_gguf.get(), EAGLE3_KEY_EMBEDDING_LENGTH, true);
        }
        hp.intermediate_size = get_kv_i32(ctx_gguf.get(), EAGLE3_KEY_INTERMEDIATE_SIZE, false, 0);
        if (hp.intermediate_size == 0) {
            hp.intermediate_size = get_kv_i32(ctx_gguf.get(), EAGLE3_KEY_FFN_LENGTH, true);
        }
        hp.num_heads = get_kv_i32(ctx_gguf.get(), EAGLE3_KEY_NUM_HEADS, false, 0);
        if (hp.num_heads == 0) {
            hp.num_heads = get_kv_i32(ctx_gguf.get(), EAGLE3_KEY_ATTN_HEAD_COUNT, true);
        }
        hp.num_kv_heads = get_kv_i32(ctx_gguf.get(), EAGLE3_KEY_NUM_KV_HEADS, false, 0);
        if (hp.num_kv_heads == 0) {
            hp.num_kv_heads = get_kv_i32(ctx_gguf.get(), EAGLE3_KEY_ATTN_HEAD_COUNT_KV, true);
        }
        hp.rms_norm_eps = get_kv_f32(ctx_gguf.get(), EAGLE3_KEY_RMS_EPS, false, 0.0f);
        if (hp.rms_norm_eps == 0.0f) {
            hp.rms_norm_eps = get_kv_f32(ctx_gguf.get(), EAGLE3_KEY_ATTN_RMS_EPS, true);
        }
        hp.hidden_concat      = get_kv_i32(ctx_gguf.get(), EAGLE3_KEY_HIDDEN_CONCAT, true);
        hp.target_hidden_size = get_kv_i32(ctx_gguf.get(), EAGLE3_KEY_TARGET_HIDDEN_SIZE, false, hp.hidden_size);
        hp.draft_vocab_size   = get_kv_i32(ctx_gguf.get(), EAGLE3_KEY_DRAFT_VOCAB_SIZE, true);
        hp.vocab_size         = get_kv_i32(ctx_gguf.get(), EAGLE3_KEY_VOCAB_SIZE, true);
        hp.head_dim           = get_kv_i32(ctx_gguf.get(), EAGLE3_KEY_HEAD_DIM, false, 0);
        hp.norm_before_residual = get_kv_bool(ctx_gguf.get(), EAGLE3_KEY_NORM_BEFORE_RESIDUAL, false, false);

        if (hp.hidden_size <= 0 || hp.intermediate_size <= 0 || hp.num_heads <= 0 || hp.num_kv_heads <= 0) {
            throw std::runtime_error("invalid head config values");
        }
        if (hp.num_heads % hp.num_kv_heads != 0) {
            throw std::runtime_error("num_attention_heads must be divisible by num_key_value_heads");
        }
        if (hp.head_dim <= 0) {
            if (hp.hidden_size % hp.num_heads != 0) {
                throw std::runtime_error("hidden_size not divisible by num_heads");
            }
            hp.head_dim = hp.hidden_size / hp.num_heads;
        }
        if (hp.hidden_concat <= 0) {
            throw std::runtime_error("hidden_concat must be >= 1");
        }
        if (hp.norm_before_residual) {
            throw std::runtime_error("norm_before_residual is not supported");
        }

        model->hparams = hp;

        model->hidden_layer_ids = get_kv_arr_i32(ctx_gguf.get(), EAGLE3_KEY_HIDDEN_LAYER_IDS, true);
        if (model->hidden_layer_ids.empty()) {
            throw std::runtime_error("hidden_state_layer_ids must be non-empty");
        }
        if ((int32_t) model->hidden_layer_ids.size() != hp.hidden_concat) {
            throw std::runtime_error("hidden_state_layer_ids size does not match hidden_concat");
        }

        model->d2t = get_kv_arr_i32(ctx_gguf.get(), EAGLE3_KEY_D2T, false);
        if (model->d2t.empty()) {
            if (hp.draft_vocab_size != hp.vocab_size) {
                throw std::runtime_error("missing d2t mapping for subset draft vocab");
            }
            model->d2t.assign(hp.draft_vocab_size, 0);
        } else if ((int32_t) model->d2t.size() != hp.draft_vocab_size) {
            throw std::runtime_error("d2t length does not match draft_vocab_size");
        }

        model->buf_weights.reset(ggml_backend_alloc_ctx_tensors_from_buft(model->ctx_weights.get(), ggml_backend_cpu_buffer_type()));
        if (!model->buf_weights) {
            throw std::runtime_error("failed to allocate eagle3 weight buffer");
        }

        model->tensors.fc_w          = require_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_FC);
        model->tensors.norm_w        = require_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_NORM);
        model->tensors.lm_head_w     = require_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_LM_HEAD);
        model->tensors.hidden_norm_w = require_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_HIDDEN_NORM);
        model->tensors.input_norm_w  = require_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_INPUT_NORM);
        model->tensors.post_norm_w   = require_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_POST_NORM);
        model->tensors.attn_q_w      = require_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_ATTN_Q);
        model->tensors.attn_k_w      = require_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_ATTN_K);
        model->tensors.attn_v_w      = require_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_ATTN_V);
        model->tensors.attn_o_w      = require_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_ATTN_O);
        model->tensors.ffn_gate_w    = require_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_FFN_GATE);
        model->tensors.ffn_up_w      = require_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_FFN_UP);
        model->tensors.ffn_down_w    = require_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_FFN_DOWN);

        model->tensors.attn_q_b   = optional_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_ATTN_Q_B);
        model->tensors.attn_k_b   = optional_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_ATTN_K_B);
        model->tensors.attn_v_b   = optional_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_ATTN_V_B);
        model->tensors.attn_o_b   = optional_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_ATTN_O_B);
        model->tensors.ffn_gate_b = optional_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_FFN_GATE_B);
        model->tensors.ffn_up_b   = optional_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_FFN_UP_B);
        model->tensors.ffn_down_b = optional_tensor(model->ctx_weights.get(), ctx_gguf.get(), EAGLE3_TENSOR_FFN_DOWN_B);

        check_shape(model->tensors.fc_w,          hp.hidden_concat * hp.target_hidden_size, hp.hidden_size);
        check_shape(model->tensors.norm_w,        hp.hidden_size, 1);
        check_shape(model->tensors.lm_head_w,     hp.hidden_size, hp.draft_vocab_size);
        check_shape(model->tensors.hidden_norm_w, hp.hidden_size, 1);
        check_shape(model->tensors.input_norm_w,  hp.hidden_size, 1);
        check_shape(model->tensors.post_norm_w,   hp.hidden_size, 1);
        check_shape(model->tensors.attn_q_w,      2 * hp.hidden_size, hp.head_dim * hp.num_heads);
        check_shape(model->tensors.attn_k_w,      2 * hp.hidden_size, hp.head_dim * hp.num_kv_heads);
        check_shape(model->tensors.attn_v_w,      2 * hp.hidden_size, hp.head_dim * hp.num_kv_heads);
        check_shape(model->tensors.attn_o_w,      hp.head_dim * hp.num_heads, hp.hidden_size);
        check_shape(model->tensors.ffn_gate_w,    hp.hidden_size, hp.intermediate_size);
        check_shape(model->tensors.ffn_up_w,      hp.hidden_size, hp.intermediate_size);
        check_shape(model->tensors.ffn_down_w,    hp.intermediate_size, hp.hidden_size);

        if (model->tensors.attn_q_b) { check_shape(model->tensors.attn_q_b, hp.head_dim * hp.num_heads, 1); }
        if (model->tensors.attn_k_b) { check_shape(model->tensors.attn_k_b, hp.head_dim * hp.num_kv_heads, 1); }
        if (model->tensors.attn_v_b) { check_shape(model->tensors.attn_v_b, hp.head_dim * hp.num_kv_heads, 1); }
        if (model->tensors.attn_o_b) { check_shape(model->tensors.attn_o_b, hp.hidden_size, 1); }
        if (model->tensors.ffn_gate_b) { check_shape(model->tensors.ffn_gate_b, hp.intermediate_size, 1); }
        if (model->tensors.ffn_up_b)   { check_shape(model->tensors.ffn_up_b,   hp.intermediate_size, 1); }
        if (model->tensors.ffn_down_b) { check_shape(model->tensors.ffn_down_b, hp.hidden_size, 1); }

        llama_file gguf_file(path.c_str(), "rb");
        std::vector<uint8_t> scratch;
        load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.fc_w, scratch);
        load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.norm_w, scratch);
        load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.lm_head_w, scratch);
        load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.hidden_norm_w, scratch);
        load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.input_norm_w, scratch);
        load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.post_norm_w, scratch);
        load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.attn_q_w, scratch);
        load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.attn_k_w, scratch);
        load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.attn_v_w, scratch);
        load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.attn_o_w, scratch);
        load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.ffn_gate_w, scratch);
        load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.ffn_up_w, scratch);
        load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.ffn_down_w, scratch);

        if (model->tensors.attn_q_b)   { load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.attn_q_b, scratch); }
        if (model->tensors.attn_k_b)   { load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.attn_k_b, scratch); }
        if (model->tensors.attn_v_b)   { load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.attn_v_b, scratch); }
        if (model->tensors.attn_o_b)   { load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.attn_o_b, scratch); }
        if (model->tensors.ffn_gate_b) { load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.ffn_gate_b, scratch); }
        if (model->tensors.ffn_up_b)   { load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.ffn_up_b, scratch); }
        if (model->tensors.ffn_down_b) { load_tensor_data(ctx_gguf.get(), gguf_file, model->tensors.ffn_down_b, scratch); }

        return model.release();
    } catch (const std::exception & e) {
        err = e.what();
        return nullptr;
    }
}

void llama_eagle3_free(llama_eagle3_model * model) {
    delete model;
}

llama_eagle3_runtime llama_eagle3_make_runtime(
        const llama_context * ctx_tgt,
        const llama_eagle3_model * model,
        int32_t n_threads) {
    llama_eagle3_runtime rt;
    const auto * ctx_impl = static_cast<const llama_context *>(ctx_tgt);
    const auto rope = ctx_impl->get_rope_params(0);
    const auto & cparams = ctx_impl->get_cparams();

    rt.base_model     = llama_get_model(const_cast<llama_context *>(ctx_tgt));
    rt.tok_embd       = rt.base_model ? rt.base_model->tok_embd : nullptr;
    // The EAGLE head is exported from HF Transformers and uses the same RoPE convention as
    // transformers' LlamaRotaryEmbedding/apply_rotary_pos_emb, which corresponds to GGML_ROPE_TYPE_NEOX
    // (pair first-half with second-half). The base model in llama.cpp may use a different internal
    // convention due to weight permutations, so do not inherit rope.rope_type here.
    rt.rope_type      = LLAMA_ROPE_TYPE_NEOX;
    rt.rope_freq_base = rope.freq_base;
    rt.rope_freq_scale = rope.freq_scale;
    rt.rope_factors    = nullptr;
    rt.yarn_ext_factor  = rope.ext_factor;
    rt.yarn_attn_factor = rope.attn_factor;
    rt.yarn_beta_fast   = rope.beta_fast;
    rt.yarn_beta_slow   = rope.beta_slow;
    rt.n_ctx_orig       = rope.n_ctx_orig;
    rt.n_threads        = n_threads;

    // Keep rope factors in host memory so fallback CPU execution stays valid.
    if (rt.base_model) {
        ggml_tensor * src = rt.base_model->get_rope_factors(cparams, /* il */ 0);
        if (src) {
            const size_t n_bytes = ggml_nbytes(src);
            const size_t mem_size = ggml_tensor_overhead() * 4 + n_bytes + 1024;
            rt.rope_factors_buf.resize(mem_size);

            ggml_init_params params = {
                /* .mem_size   = */ mem_size,
                /* .mem_buffer = */ rt.rope_factors_buf.data(),
                /* .no_alloc   = */ false,
            };
            rt.rope_factors_ctx.reset(ggml_init(params));
            if (rt.rope_factors_ctx) {
                ggml_tensor * dst = ggml_new_tensor(rt.rope_factors_ctx.get(), src->type, ggml_n_dims(src), src->ne);
                std::vector<uint8_t> tmp(n_bytes);
                ggml_backend_tensor_get(src, tmp.data(), 0, n_bytes);
                std::memcpy(dst->data, tmp.data(), n_bytes);
                rt.rope_factors = dst;
            }
        }
    }

    // Prefer GPU backend for EAGLE head execution.
    rt.backend_compute.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr));
    if (!rt.backend_compute) {
        rt.backend_compute.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr));
    }
    if (rt.backend_compute) {
        rt.buft_compute = ggml_backend_get_default_buffer_type(rt.backend_compute.get());
        maybe_set_backend_threads(rt.backend_compute.get(), n_threads);
        if (!copy_weight_tensors_to_backend(*model, rt)) {
            rt.backend_compute.reset();
            rt.buft_compute = nullptr;
            rt.ctx_weights_compute.reset();
            rt.buf_weights_compute.reset();
            rt.tensors_compute = {};
        }
    }

    // Fallback backend for environments without a GPU backend.
    if (!rt.backend_compute) {
        rt.backend_compute.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        if (rt.backend_compute) {
            rt.buft_compute = ggml_backend_get_default_buffer_type(rt.backend_compute.get());
            maybe_set_backend_threads(rt.backend_compute.get(), n_threads);
        }
    }

    return rt;
}

bool llama_eagle3_logits(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const float * hidden,
        std::vector<float> & logits_out) {
    if (!hidden) {
        return false;
    }

    const auto & hp = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);

    if (rt.backend_compute && rt.buft_compute) {
        if (build_logits_graph(model, rt, rt.logits_graph)) {
            ggml_backend_tensor_set(rt.logits_graph.t_hidden, hidden, 0, hp.hidden_size * sizeof(float));
            const ggml_status status = ggml_backend_graph_compute(rt.backend_compute.get(), rt.logits_graph.gf);
            if (status == GGML_STATUS_SUCCESS) {
                logits_out.resize(hp.draft_vocab_size);
                ggml_backend_tensor_get(rt.logits_graph.t_logits, logits_out.data(), 0, hp.draft_vocab_size * sizeof(float));
                return true;
            }
        }
    }

    const size_t mem_size = 8ull * 1024ull * 1024ull;
    std::vector<uint8_t> buf(mem_size);
    auto ctx = make_ctx_with_buf(buf, mem_size);
    if (!ctx) {
        return false;
    }

    ggml_tensor * t_hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hp.hidden_size, 1);
    ggml_set_input(t_hidden);

    ggml_tensor * t_norm = ggml_rms_norm(ctx.get(), t_hidden, hp.rms_norm_eps);
    ggml_tensor * t_norm_w = ggml_cast(ctx.get(), tensors.norm_w, GGML_TYPE_F32);
    t_norm = ggml_mul(ctx.get(), t_norm, t_norm_w);

    ggml_tensor * t_logits = ggml_mul_mat(ctx.get(), tensors.lm_head_w, t_norm);

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, t_logits);

    std::memcpy(t_hidden->data, hidden, hp.hidden_size * sizeof(float));
    ggml_graph_compute_with_ctx(ctx.get(), gf, std::max(1, rt.n_threads));

    logits_out.resize(hp.draft_vocab_size);
    std::memcpy(logits_out.data(), t_logits->data, hp.draft_vocab_size * sizeof(float));
    return true;
}

bool llama_eagle3_topk(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const float * hidden,
        int32_t k,
        std::vector<int32_t> & topk_idx_out,
        std::vector<float> & topk_prob_out) {
    if (!hidden || k <= 0) {
        return false;
    }

    const auto & hp = model.hparams;
    k = std::min(k, hp.draft_vocab_size);
    if (k <= 0) {
        return false;
    }

    if (rt.backend_compute && rt.buft_compute) {
        if (build_topk_graph(model, rt, k, rt.topk_graph)) {
            ggml_backend_tensor_set(rt.topk_graph.t_hidden, hidden, 0, hp.hidden_size * sizeof(float));
            const ggml_status status = ggml_backend_graph_compute(rt.backend_compute.get(), rt.topk_graph.gf);
            if (status == GGML_STATUS_SUCCESS) {
                topk_idx_out.resize(k);
                topk_prob_out.resize(k);
                ggml_backend_tensor_get(rt.topk_graph.t_topk_idx, topk_idx_out.data(), 0, (size_t) k * sizeof(int32_t));

                const size_t n_probs = (size_t) hp.draft_vocab_size;
                for (int i = 0; i < k; ++i) {
                    int32_t idx = topk_idx_out[i];
                    if (idx < 0 || (size_t) idx >= n_probs) {
                        topk_prob_out[i] = 0.0f;
                        continue;
                    }
                    float p = 0.0f;
                    ggml_backend_tensor_get(rt.topk_graph.t_probs, &p, (size_t) idx * sizeof(float), sizeof(float));
                    topk_prob_out[i] = p;
                }
                return true;
            }
        }
    }

    std::vector<float> logits;
    if (!llama_eagle3_logits(model, rt, hidden, logits) || logits.empty()) {
        return false;
    }

    float max_logit = logits[0];
    for (float v : logits) {
        max_logit = std::max(max_logit, v);
    }

    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    if (sum <= 0.0f) {
        return false;
    }
    for (float & p : probs) {
        p /= sum;
    }

    std::vector<int32_t> idx(probs.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [&](int32_t a, int32_t b) { return probs[a] > probs[b]; });

    topk_idx_out.resize(k);
    topk_prob_out.resize(k);
    for (int i = 0; i < k; ++i) {
        topk_idx_out[i] = idx[i];
        topk_prob_out[i] = probs[idx[i]];
    }
    return true;
}

bool llama_eagle3_topk_state(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const llama_eagle3_state & state,
        int32_t k,
        std::vector<int32_t> & topk_idx_out,
        std::vector<float> & topk_prob_out) {
    const auto & hp = model.hparams;
    k = std::min(k, hp.draft_vocab_size);
    if (k <= 0) {
        return false;
    }

    if (rt.backend_compute && rt.buft_compute && state.dev && state.dev->t_hidden) {
        if (build_topk_graph(model, rt, k, rt.topk_graph)) {
            ggml_backend_tensor_copy(state.dev->t_hidden, rt.topk_graph.t_hidden);
            const ggml_status status = ggml_backend_graph_compute(rt.backend_compute.get(), rt.topk_graph.gf);
            if (status == GGML_STATUS_SUCCESS) {
                topk_idx_out.resize(k);
                topk_prob_out.resize(k);
                ggml_backend_tensor_get(rt.topk_graph.t_topk_idx, topk_idx_out.data(), 0, (size_t) k * sizeof(int32_t));

                const size_t n_probs = (size_t) hp.draft_vocab_size;
                for (int i = 0; i < k; ++i) {
                    int32_t idx = topk_idx_out[i];
                    if (idx < 0 || (size_t) idx >= n_probs) {
                        topk_prob_out[i] = 0.0f;
                        continue;
                    }
                    float p = 0.0f;
                    ggml_backend_tensor_get(rt.topk_graph.t_probs, &p, (size_t) idx * sizeof(float), sizeof(float));
                    topk_prob_out[i] = p;
                }
                return true;
            }
        }
    }

    if (!state.hidden.empty()) {
        return llama_eagle3_topk(model, rt, state.hidden.data(), k, topk_idx_out, topk_prob_out);
    }

    return false;
}

bool llama_eagle3_topk_state_batch(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const std::vector<const llama_eagle3_state *> & states,
        int32_t k,
        std::vector<int32_t> & topk_idx_out,
        std::vector<float> & topk_prob_out) {
    const auto & hp = model.hparams;
    const int32_t n_beams = (int32_t) states.size();
    if (n_beams <= 0) {
        return false;
    }

    k = std::min(k, hp.draft_vocab_size);
    if (k <= 0) {
        return false;
    }

    if (rt.backend_compute && rt.buft_compute) {
        if (build_topk_batch_graph(model, rt, n_beams, k, rt.topk_batch_graph)) {
            std::vector<float> hidden_in((size_t) hp.hidden_size * n_beams);
            for (int32_t ib = 0; ib < n_beams; ++ib) {
                const llama_eagle3_state * st = states[ib];
                if (st == nullptr) {
                    return false;
                }
                float * dst = hidden_in.data() + (size_t) ib * hp.hidden_size;
                if (!st->hidden.empty()) {
                    std::memcpy(dst, st->hidden.data(), (size_t) hp.hidden_size * sizeof(float));
                } else if (st->dev && st->dev->t_hidden) {
                    ggml_backend_tensor_get(st->dev->t_hidden, dst, 0, (size_t) hp.hidden_size * sizeof(float));
                } else {
                    return false;
                }
            }

            ggml_backend_tensor_set(rt.topk_batch_graph.t_hidden, hidden_in.data(), 0, hidden_in.size() * sizeof(float));
            const ggml_status status = ggml_backend_graph_compute(rt.backend_compute.get(), rt.topk_batch_graph.gf);
            if (status == GGML_STATUS_SUCCESS) {
                topk_idx_out.resize((size_t) n_beams * k);
                topk_prob_out.resize((size_t) n_beams * k);

                ggml_backend_tensor_get(
                        rt.topk_batch_graph.t_topk_idx,
                        topk_idx_out.data(),
                        0,
                        topk_idx_out.size() * sizeof(int32_t));

                std::vector<float> probs((size_t) hp.draft_vocab_size * n_beams);
                ggml_backend_tensor_get(
                        rt.topk_batch_graph.t_probs,
                        probs.data(),
                        0,
                        probs.size() * sizeof(float));

                for (int32_t ib = 0; ib < n_beams; ++ib) {
                    const size_t probs_off = (size_t) ib * hp.draft_vocab_size;
                    const size_t topk_off = (size_t) ib * k;
                    for (int32_t j = 0; j < k; ++j) {
                        const int32_t idx = topk_idx_out[topk_off + j];
                        if (idx < 0 || idx >= hp.draft_vocab_size) {
                            topk_prob_out[topk_off + j] = 0.0f;
                        } else {
                            topk_prob_out[topk_off + j] = probs[probs_off + (size_t) idx];
                        }
                    }
                }
                return true;
            }
        }
    }

    topk_idx_out.resize((size_t) n_beams * k);
    topk_prob_out.resize((size_t) n_beams * k);
    for (int32_t ib = 0; ib < n_beams; ++ib) {
        const llama_eagle3_state * st = states[ib];
        if (st == nullptr) {
            return false;
        }

        std::vector<int32_t> idx;
        std::vector<float> prob;
        if (!llama_eagle3_topk_state(model, rt, *st, k, idx, prob) || (int32_t) idx.size() != k || (int32_t) prob.size() != k) {
            return false;
        }
        std::memcpy(topk_idx_out.data() + (size_t) ib * k, idx.data(), (size_t) k * sizeof(int32_t));
        std::memcpy(topk_prob_out.data() + (size_t) ib * k, prob.data(), (size_t) k * sizeof(float));
    }
    return true;
}

bool llama_eagle3_state_has_hidden(const llama_eagle3_state & state) {
    return !state.hidden.empty() || (state.dev && state.dev->t_hidden != nullptr);
}

bool llama_eagle3_state_get_hidden(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        llama_eagle3_state & state,
        std::vector<float> & hidden_out) {
    GGML_UNUSED(rt);
    const auto & hp = model.hparams;
    if (!state.hidden.empty()) {
        hidden_out = state.hidden;
        return hidden_out.size() == (size_t) hp.hidden_size;
    }

    if (state.dev && state.dev->t_hidden) {
        hidden_out.resize(hp.hidden_size);
        ggml_backend_tensor_get(state.dev->t_hidden, hidden_out.data(), 0, (size_t) hp.hidden_size * sizeof(float));
        state.hidden = hidden_out;
        return true;
    }

    return false;
}

bool llama_eagle3_step(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        llama_eagle3_state & state,
        const float * hidden_in,
        int32_t hidden_in_dim,
        llama_token input_id,
        std::vector<float> * logits_out,
        llama_eagle3_step_debug * dbg) {
    if (!rt.tok_embd) {
        return false;
    }

    const auto & hp = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);
    const bool hidden_from_state = hidden_in == nullptr;
    if (hidden_from_state && hidden_in_dim != hp.hidden_size) {
        return false;
    }
    if (hidden_from_state && state.hidden.empty() && (!state.dev || !state.dev->t_hidden)) {
        return false;
    }

    const bool with_logits = logits_out != nullptr;
    if (rt.backend_compute && rt.buft_compute && dbg == nullptr) {
        const uint64_t key = make_step_graph_key(state.past_len, hidden_in_dim, with_logits);
        if (rt.step_graph_key != key) {
            rt.step_graph = {};
            if (build_step_graph(model, rt, state.past_len, hidden_in_dim, with_logits, rt.step_graph)) {
                rt.step_graph_key = key;
            }
        }

        if (rt.step_graph_key == key && rt.step_graph.gf) {
            auto & graph = rt.step_graph;
            const size_t block = (size_t) hp.head_dim * hp.num_kv_heads;
            const size_t past_bytes = (size_t) state.past_len * block * sizeof(float);

            if (hidden_in) {
                ggml_backend_tensor_set(graph.t_hidden_in, hidden_in, 0, hidden_in_dim * sizeof(float));
            } else if (state.dev && state.dev->t_hidden) {
                ggml_backend_tensor_copy(state.dev->t_hidden, graph.t_hidden_in);
            } else {
                ggml_backend_tensor_set(graph.t_hidden_in, state.hidden.data(), 0, hidden_in_dim * sizeof(float));
            }
            ggml_backend_tensor_set(graph.t_tok, &input_id, 0, sizeof(input_id));
            ggml_backend_tensor_set(graph.t_pos, &state.past_len, 0, sizeof(state.past_len));

            if (graph.t_k_past_input && state.past_len > 0) {
                if (state.dev && state.dev->t_k && state.dev->past_len == state.past_len) {
                    ggml_backend_tensor_copy(state.dev->t_k, graph.t_k_past_input);
                } else if (state.k.size() * sizeof(float) >= past_bytes) {
                    ggml_backend_tensor_set(graph.t_k_past_input, state.k.data(), 0, past_bytes);
                } else {
                    return false;
                }
            }
            if (graph.t_v_past_input && state.past_len > 0) {
                if (state.dev && state.dev->t_v && state.dev->past_len == state.past_len) {
                    ggml_backend_tensor_copy(state.dev->t_v, graph.t_v_past_input);
                } else if (state.v.size() * sizeof(float) >= past_bytes) {
                    ggml_backend_tensor_set(graph.t_v_past_input, state.v.data(), 0, past_bytes);
                } else {
                    return false;
                }
            }

            const ggml_status status = ggml_backend_graph_compute(rt.backend_compute.get(), graph.gf);
            if (status == GGML_STATUS_SUCCESS) {
                std::shared_ptr<llama_eagle3_state::device_state> next_dev;
                if (alloc_state_device(model, rt, state.past_len + 1, next_dev)) {
                    ggml_backend_tensor_copy(graph.t_hidden_out, next_dev->t_hidden);
                    if (next_dev->t_k && next_dev->t_v && graph.t_k_total && graph.t_v_total) {
                        ggml_backend_tensor_copy(graph.t_k_total, next_dev->t_k);
                        ggml_backend_tensor_copy(graph.t_v_total, next_dev->t_v);
                    }
                    state.dev = std::move(next_dev);
                } else {
                    state.dev.reset();
                }

                state.hidden.clear();
                state.k.clear();
                state.v.clear();

                if (with_logits && graph.t_logits) {
                    logits_out->resize(hp.draft_vocab_size);
                    ggml_backend_tensor_get(graph.t_logits, logits_out->data(), 0, hp.draft_vocab_size * sizeof(float));
                }

                state.past_len += 1;
                return true;
            }
        }
    }

    const size_t mem_size = estimate_step_mem(hp, state.past_len);
    std::vector<uint8_t> buf(mem_size);
    auto ctx = make_ctx_with_buf(buf, mem_size);
    if (!ctx) {
        return false;
    }

    ggml_tensor * t_hidden_in = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden_in_dim, 1);
    ggml_set_input(t_hidden_in);

    ggml_tensor * t_tok = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_set_input(t_tok);

    ggml_tensor * t_embd = ggml_get_rows(ctx.get(), rt.tok_embd, t_tok);
    t_embd = ggml_cast(ctx.get(), t_embd, GGML_TYPE_F32);

    ggml_tensor * t_hidden = t_hidden_in;
    if (hidden_in_dim != hp.hidden_size) {
        t_hidden = ggml_mul_mat(ctx.get(), tensors.fc_w, t_hidden_in);
    }

    ggml_tensor * t_hidden_norm = ggml_rms_norm(ctx.get(), t_hidden, hp.rms_norm_eps);
    ggml_tensor * t_hidden_norm_w = ggml_cast(ctx.get(), tensors.hidden_norm_w, GGML_TYPE_F32);
    t_hidden_norm = ggml_mul(ctx.get(), t_hidden_norm, t_hidden_norm_w);

    ggml_tensor * t_embd_norm = ggml_rms_norm(ctx.get(), t_embd, hp.rms_norm_eps);
    ggml_tensor * t_input_norm_w = ggml_cast(ctx.get(), tensors.input_norm_w, GGML_TYPE_F32);
    t_embd_norm = ggml_mul(ctx.get(), t_embd_norm, t_input_norm_w);

    ggml_tensor * t_cat = ggml_concat(ctx.get(), t_embd_norm, t_hidden_norm, 0);

    ggml_tensor * t_q = ggml_mul_mat(ctx.get(), tensors.attn_q_w, t_cat);
    if (tensors.attn_q_b) {
        ggml_tensor * t_q_b = ggml_cast(ctx.get(), tensors.attn_q_b, GGML_TYPE_F32);
        t_q = ggml_add(ctx.get(), t_q, t_q_b);
    }
    ggml_tensor * t_k = ggml_mul_mat(ctx.get(), tensors.attn_k_w, t_cat);
    if (tensors.attn_k_b) {
        ggml_tensor * t_k_b = ggml_cast(ctx.get(), tensors.attn_k_b, GGML_TYPE_F32);
        t_k = ggml_add(ctx.get(), t_k, t_k_b);
    }
    ggml_tensor * t_v = ggml_mul_mat(ctx.get(), tensors.attn_v_w, t_cat);
    if (tensors.attn_v_b) {
        ggml_tensor * t_v_b = ggml_cast(ctx.get(), tensors.attn_v_b, GGML_TYPE_F32);
        t_v = ggml_add(ctx.get(), t_v, t_v_b);
    }

    t_q = ggml_reshape_3d(ctx.get(), t_q, hp.head_dim, hp.num_heads, 1);
    t_k = ggml_reshape_3d(ctx.get(), t_k, hp.head_dim, hp.num_kv_heads, 1);
    t_v = ggml_reshape_3d(ctx.get(), t_v, hp.head_dim, hp.num_kv_heads, 1);

    ggml_tensor * t_pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_set_input(t_pos);

    t_q = ggml_rope_ext(
            ctx.get(), t_q, t_pos, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);

    t_k = ggml_rope_ext(
            ctx.get(), t_k, t_pos, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);

    ggml_tensor * t_k_total = t_k;
    ggml_tensor * t_v_total = t_v;
    ggml_tensor * t_k_past_input = nullptr;
    ggml_tensor * t_v_past_input = nullptr;

    if (state.past_len > 0) {
        t_k_past_input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, state.past_len);
        t_v_past_input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, state.past_len);
        ggml_set_input(t_k_past_input);
        ggml_set_input(t_v_past_input);

        t_k_total = ggml_concat(ctx.get(), t_k_past_input, t_k, 2);
        t_v_total = ggml_concat(ctx.get(), t_v_past_input, t_v, 2);
    }

    if (hp.num_kv_heads != hp.num_heads) {
        const int32_t n_rep = hp.num_heads / hp.num_kv_heads;

        // Match HF `repeat_kv` semantics: repeat each KV head `n_rep` times consecutively.
        // Our tensors are [head_dim, n_kv, seq]. We reshape/permute to make the repeat dimension
        // "inner" before merging back to [head_dim, n_head, seq].
        ggml_tensor * k4 = ggml_reshape_4d(ctx.get(), t_k_total, hp.head_dim, hp.num_kv_heads, t_k_total->ne[2], 1);
        ggml_tensor * v4 = ggml_reshape_4d(ctx.get(), t_v_total, hp.head_dim, hp.num_kv_heads, t_v_total->ne[2], 1);

        // Note: ggml_permute uses "destination axis for each source axis" (see ggml_permute impl).
        // We want [head_dim, 1, n_kv, seq] from [head_dim, n_kv, seq, 1].
        k4 = ggml_permute(ctx.get(), k4, 0, 2, 3, 1); // [head_dim, 1, n_kv, seq]
        v4 = ggml_permute(ctx.get(), v4, 0, 2, 3, 1); // [head_dim, 1, n_kv, seq]

        if (std::getenv("CASCADE_EAGLE_DEBUG")) {
            fprintf(stderr,
                    "eagle3 repeat_kv: k4 ne=[%lld,%lld,%lld,%lld] -> [%d,%d,%d,%lld]\n",
                    (long long) k4->ne[0], (long long) k4->ne[1], (long long) k4->ne[2], (long long) k4->ne[3],
                    hp.head_dim, n_rep, hp.num_kv_heads, (long long) t_k_total->ne[2]);
        }

        k4 = ggml_repeat_4d(ctx.get(), k4, hp.head_dim, n_rep, hp.num_kv_heads, t_k_total->ne[2]); // [head_dim, n_rep, n_kv, seq]
        v4 = ggml_repeat_4d(ctx.get(), v4, hp.head_dim, n_rep, hp.num_kv_heads, t_v_total->ne[2]); // [head_dim, n_rep, n_kv, seq]

        k4 = ggml_cont(ctx.get(), k4);
        v4 = ggml_cont(ctx.get(), v4);

        t_k_total = ggml_reshape_3d(ctx.get(), k4, hp.head_dim, hp.num_heads, t_k_total->ne[2]);
        t_v_total = ggml_reshape_3d(ctx.get(), v4, hp.head_dim, hp.num_heads, t_v_total->ne[2]);
    }

    ggml_tensor * qv = ggml_view_4d(ctx.get(), t_q, t_q->ne[0], t_q->ne[1], t_q->ne[2], 1, t_q->nb[1], t_q->nb[2], t_q->nb[3], 0);
    ggml_tensor * kv = ggml_view_4d(ctx.get(), t_k_total, t_k_total->ne[0], t_k_total->ne[1], t_k_total->ne[2], 1,
                                    t_k_total->nb[1], t_k_total->nb[2], t_k_total->nb[3], 0);
    ggml_tensor * vv = ggml_view_4d(ctx.get(), t_v_total, t_v_total->ne[0], t_v_total->ne[1], t_v_total->ne[2], 1,
                                    t_v_total->nb[1], t_v_total->nb[2], t_v_total->nb[3], 0);

    qv = ggml_permute(ctx.get(), qv, 0, 2, 1, 3);
    kv = ggml_permute(ctx.get(), kv, 0, 2, 1, 3);
    vv = ggml_permute(ctx.get(), vv, 0, 2, 1, 3);

    ggml_tensor * kq = ggml_mul_mat(ctx.get(), kv, qv);
    const float kq_scale = 1.0f / std::sqrt(float(hp.head_dim));
    kq = ggml_scale(ctx.get(), kq, kq_scale);
    kq = ggml_cont(ctx.get(), kq);
    kq = ggml_soft_max(ctx.get(), kq);

    ggml_tensor * vv_t = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), vv));
    ggml_tensor * kqv = ggml_mul_mat(ctx.get(), vv_t, kq);
    ggml_tensor * attn_out = ggml_permute(ctx.get(), kqv, 0, 2, 1, 3);
    attn_out = ggml_cont_2d(ctx.get(), attn_out, attn_out->ne[0]*attn_out->ne[1], attn_out->ne[2]*attn_out->ne[3]);

    ggml_tensor * t_attn = ggml_mul_mat(ctx.get(), tensors.attn_o_w, attn_out);
    if (tensors.attn_o_b) {
        ggml_tensor * t_attn_b = ggml_cast(ctx.get(), tensors.attn_o_b, GGML_TYPE_F32);
        t_attn = ggml_add(ctx.get(), t_attn, t_attn_b);
    }

    ggml_tensor * t_resid = hp.norm_before_residual ? t_hidden_norm : t_hidden;
    ggml_tensor * t_hidden_attn = ggml_add(ctx.get(), t_attn, t_resid);

    ggml_tensor * t_post = ggml_rms_norm(ctx.get(), t_hidden_attn, hp.rms_norm_eps);
    ggml_tensor * t_post_norm_w = ggml_cast(ctx.get(), tensors.post_norm_w, GGML_TYPE_F32);
    t_post = ggml_mul(ctx.get(), t_post, t_post_norm_w);

    ggml_tensor * t_gate = ggml_mul_mat(ctx.get(), tensors.ffn_gate_w, t_post);
    if (tensors.ffn_gate_b) {
        ggml_tensor * t_gate_b = ggml_cast(ctx.get(), tensors.ffn_gate_b, GGML_TYPE_F32);
        t_gate = ggml_add(ctx.get(), t_gate, t_gate_b);
    }
    ggml_tensor * t_up = ggml_mul_mat(ctx.get(), tensors.ffn_up_w, t_post);
    if (tensors.ffn_up_b) {
        ggml_tensor * t_up_b = ggml_cast(ctx.get(), tensors.ffn_up_b, GGML_TYPE_F32);
        t_up = ggml_add(ctx.get(), t_up, t_up_b);
    }
    ggml_tensor * t_act = ggml_mul(ctx.get(), ggml_silu(ctx.get(), t_gate), t_up);
    ggml_tensor * t_ffn = ggml_mul_mat(ctx.get(), tensors.ffn_down_w, t_act);
    if (tensors.ffn_down_b) {
        ggml_tensor * t_down_b = ggml_cast(ctx.get(), tensors.ffn_down_b, GGML_TYPE_F32);
        t_ffn = ggml_add(ctx.get(), t_ffn, t_down_b);
    }

    ggml_tensor * t_hidden_out = ggml_add(ctx.get(), t_hidden_attn, t_ffn);
    t_hidden_out = ggml_cont(ctx.get(), t_hidden_out);

    ggml_tensor * t_logits = nullptr;
    if (logits_out) {
        ggml_tensor * t_norm = ggml_rms_norm(ctx.get(), t_hidden_out, hp.rms_norm_eps);
        ggml_tensor * t_norm_w = ggml_cast(ctx.get(), tensors.norm_w, GGML_TYPE_F32);
        t_norm = ggml_mul(ctx.get(), t_norm, t_norm_w);
        t_logits = ggml_mul_mat(ctx.get(), tensors.lm_head_w, t_norm);
    }

    ggml_tensor * t_embd_cont       = nullptr;
    ggml_tensor * t_embd_norm_cont  = nullptr;
    ggml_tensor * t_hidden_cont     = nullptr;
    ggml_tensor * t_hidden_norm_cont = nullptr;
    ggml_tensor * t_cat_cont        = nullptr;
    ggml_tensor * t_q_cont          = nullptr;
    ggml_tensor * t_k_cont          = nullptr;
    ggml_tensor * t_v_cont          = nullptr;
    if (dbg) {
        if (dbg->embd)        { t_embd_cont        = ggml_cont(ctx.get(), t_embd); }
        if (dbg->embd_norm)   { t_embd_norm_cont   = ggml_cont(ctx.get(), t_embd_norm); }
        if (dbg->hidden_proj) { t_hidden_cont      = ggml_cont(ctx.get(), t_hidden); }
        if (dbg->hidden_norm) { t_hidden_norm_cont = ggml_cont(ctx.get(), t_hidden_norm); }
        if (dbg->cat)         { t_cat_cont         = ggml_cont(ctx.get(), t_cat); }
        if (dbg->q)           { t_q_cont           = ggml_cont(ctx.get(), t_q); }
        if (dbg->k)           { t_k_cont           = ggml_cont(ctx.get(), t_k); }
        if (dbg->v)           { t_v_cont           = ggml_cont(ctx.get(), t_v); }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    if (t_logits) {
        ggml_build_forward_expand(gf, t_logits);
    } else {
        ggml_build_forward_expand(gf, t_hidden_out);
    }
    if (t_embd_cont)        { ggml_build_forward_expand(gf, t_embd_cont); }
    if (t_embd_norm_cont)   { ggml_build_forward_expand(gf, t_embd_norm_cont); }
    if (t_hidden_cont)      { ggml_build_forward_expand(gf, t_hidden_cont); }
    if (t_hidden_norm_cont) { ggml_build_forward_expand(gf, t_hidden_norm_cont); }
    if (t_cat_cont)         { ggml_build_forward_expand(gf, t_cat_cont); }
    if (t_q_cont)           { ggml_build_forward_expand(gf, t_q_cont); }
    if (t_k_cont)           { ggml_build_forward_expand(gf, t_k_cont); }
    if (t_v_cont)           { ggml_build_forward_expand(gf, t_v_cont); }
    const float * hidden_src = hidden_in ? hidden_in : (state.hidden.empty() ? nullptr : state.hidden.data());
    if (!hidden_src) {
        return false;
    }
    std::memcpy(t_hidden_in->data, hidden_src, hidden_in_dim * sizeof(float));
    reinterpret_cast<int32_t *>(t_tok->data)[0] = input_id;
    reinterpret_cast<int32_t *>(t_pos->data)[0] = state.past_len;
    if (t_k_past_input) {
        std::memcpy(t_k_past_input->data, state.k.data(), state.k.size() * sizeof(float));
    }
    if (t_v_past_input) {
        std::memcpy(t_v_past_input->data, state.v.data(), state.v.size() * sizeof(float));
    }

    ggml_graph_compute_with_ctx(ctx.get(), gf, std::max(1, rt.n_threads));

    state.hidden.resize(hp.hidden_size);
    std::memcpy(state.hidden.data(), t_hidden_out->data, hp.hidden_size * sizeof(float));

    state.k.resize((state.past_len + 1) * hp.head_dim * hp.num_kv_heads);
    state.v.resize((state.past_len + 1) * hp.head_dim * hp.num_kv_heads);
    {
        const size_t block = (size_t) hp.head_dim * hp.num_kv_heads;
        const float * k_src = reinterpret_cast<const float *>(t_k->data);
        const float * v_src = reinterpret_cast<const float *>(t_v->data);
        std::memcpy(state.k.data() + state.past_len * block, k_src, block * sizeof(float));
        std::memcpy(state.v.data() + state.past_len * block, v_src, block * sizeof(float));
    }

    if (logits_out && t_logits) {
        logits_out->resize(hp.draft_vocab_size);
        std::memcpy(logits_out->data(), t_logits->data, hp.draft_vocab_size * sizeof(float));
    }

    state.dev.reset();
    state.past_len += 1;

    if (dbg) {
        if (dbg->embd && t_embd_cont) {
            const float * src = reinterpret_cast<const float *>(t_embd_cont->data);
            dbg->embd->insert(dbg->embd->end(), src, src + hp.hidden_size);
        }
        if (dbg->embd_norm && t_embd_norm_cont) {
            const float * src = reinterpret_cast<const float *>(t_embd_norm_cont->data);
            dbg->embd_norm->insert(dbg->embd_norm->end(), src, src + hp.hidden_size);
        }
        if (dbg->hidden_proj && t_hidden_cont) {
            const float * src = reinterpret_cast<const float *>(t_hidden_cont->data);
            dbg->hidden_proj->insert(dbg->hidden_proj->end(), src, src + hp.hidden_size);
        }
        if (dbg->hidden_norm && t_hidden_norm_cont) {
            const float * src = reinterpret_cast<const float *>(t_hidden_norm_cont->data);
            dbg->hidden_norm->insert(dbg->hidden_norm->end(), src, src + hp.hidden_size);
        }
        if (dbg->cat && t_cat_cont) {
            const float * src = reinterpret_cast<const float *>(t_cat_cont->data);
            dbg->cat->insert(dbg->cat->end(), src, src + (size_t) hp.hidden_size * 2);
        }
        if (dbg->q && t_q_cont) {
            const float * src = reinterpret_cast<const float *>(t_q_cont->data);
            dbg->q->insert(dbg->q->end(), src, src + (size_t) hp.head_dim * hp.num_heads);
        }
        if (dbg->k && t_k_cont) {
            const float * src = reinterpret_cast<const float *>(t_k_cont->data);
            dbg->k->insert(dbg->k->end(), src, src + (size_t) hp.head_dim * hp.num_kv_heads);
        }
        if (dbg->v && t_v_cont) {
            const float * src = reinterpret_cast<const float *>(t_v_cont->data);
            dbg->v->insert(dbg->v->end(), src, src + (size_t) hp.head_dim * hp.num_kv_heads);
        }
    }

    return true;
}
