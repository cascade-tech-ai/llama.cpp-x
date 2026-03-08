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

#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

struct llama_eagle3_state::device_state {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor *           t_hidden = nullptr;
    ggml_tensor *           t_k = nullptr;
    ggml_tensor *           t_v = nullptr;
    int32_t                 past_len = 0;
    int32_t                 kv_capacity = 0;
};

namespace {

#if defined(GGML_USE_CUDA)
struct ggml_cuda_profiler_scope {
    ggml_backend_t backend = nullptr;
    bool active = false;

    ggml_cuda_profiler_scope(ggml_backend_t backend, const char * name) : backend(backend) {
        if (backend && ggml_backend_is_cuda(backend)) {
            ggml_backend_cuda_nvtx_push(name);
            active = true;
        }
    }

    ~ggml_cuda_profiler_scope() {
        if (active) {
            ggml_backend_cuda_nvtx_pop();
        }
    }
};
#endif

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

uint64_t make_step_batch_graph_key(int32_t past_len, int32_t hidden_in_dim, int32_t n_beams, bool with_logits) {
    return (uint64_t(uint16_t(past_len & 0xffff)) << 48) |
           (uint64_t(uint16_t(hidden_in_dim & 0xffff)) << 32) |
           (uint64_t(uint16_t(n_beams & 0xffff)) << 1) |
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

const llama_eagle3_tensors & get_host_tensors(const llama_eagle3_model & model) {
    return model.tensors;
}

bool alloc_state_device(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t kv_capacity,
        std::shared_ptr<llama_eagle3_state::device_state> & out) {
    if (!rt.buft_compute || kv_capacity < 0) {
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
    if (kv_capacity > 0) {
        t_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, kv_capacity);
        t_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, kv_capacity);
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
    dev->past_len = 0;
    dev->kv_capacity = kv_capacity;
    out = std::move(dev);
    return true;
}

int32_t choose_kv_capacity(int32_t required_len, int32_t reserve_kv, int32_t current_capacity) {
    const int32_t min_capacity = required_len + std::max(0, reserve_kv);
    if (current_capacity >= min_capacity) {
        return current_capacity;
    }
    if (current_capacity <= 0) {
        return min_capacity + 64;
    }

    int32_t grown = current_capacity + std::max<int32_t>(64, current_capacity / 4);
    if (grown < min_capacity) {
        grown = min_capacity;
    }
    return grown;
}

bool tensor_copy_3d_prefix_async(
        ggml_backend_t backend,
        ggml_tensor * src,
        ggml_tensor * dst,
        int64_t n0,
        int64_t n1,
        int32_t len) {
    if (len <= 0) {
        return true;
    }
    if (!backend || !src || !dst) {
        return false;
    }

    const int64_t src_len = src->ne[2];
    const int64_t dst_len = dst->ne[2];
    if (src->ne[0] != n0 || src->ne[1] != n1 || dst->ne[0] != n0 || dst->ne[1] != n1 ||
        src_len < len || dst_len < len) {
        return false;
    }

    if (src_len == len && dst_len == len) {
        ggml_backend_tensor_copy_async(backend, backend, src, dst);
        return true;
    }

#if defined(GGML_USE_CUDA)
    if (ggml_backend_is_cuda(backend)) {
        return ggml_backend_cuda_tensor_copy_3d_prefix_async(backend, src, dst, len);
    }
#endif

    const size_t n_bytes = (size_t) n0 * (size_t) n1 * (size_t) len * sizeof(float);
    std::memcpy(dst->data, src->data, n_bytes);
    return true;
}

bool tensor_copy_bytes_async(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        ggml_tensor * src,
        size_t src_offset,
        ggml_tensor * dst,
        size_t dst_offset,
        size_t size) {
    if (!backend_src || !backend_dst || !src || !dst) {
        return false;
    }
    if (src_offset + size > ggml_nbytes(src) || dst_offset + size > ggml_nbytes(dst)) {
        return false;
    }
    if (size == 0) {
        return true;
    }

#if defined(GGML_USE_CUDA)
    if (ggml_backend_is_cuda(backend_src) && ggml_backend_is_cuda(backend_dst)) {
        return ggml_backend_cuda_tensor_copy_bytes_between_async(backend_src, backend_dst, src, src_offset, dst, dst_offset, size);
    }
#endif

    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (backend_src == backend_dst && buf_src && buf_dst && ggml_backend_buffer_is_host(buf_src) && ggml_backend_buffer_is_host(buf_dst)) {
        std::memcpy(
                (char *) dst->data + dst_offset,
                (const char *) src->data + src_offset,
                size);
        return true;
    }

    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);

    std::vector<uint8_t> tmp(size);
    ggml_backend_tensor_get(src, tmp.data(), src_offset, size);
    ggml_backend_tensor_set(dst, tmp.data(), dst_offset, size);
    return true;
}

bool build_hidden_concat_from_capture_host(
        const llama_eagle3_model & model,
        const std::vector<const ggml_tensor *> & hidden_capture,
        size_t token_idx,
        std::vector<float> & hidden_concat_out) {
    const size_t n_layers = hidden_capture.size();
    const size_t hidden_size = (size_t) model.hparams.target_hidden_size;
    hidden_concat_out.resize(n_layers * hidden_size);

    for (size_t i = 0; i < n_layers; ++i) {
        const ggml_tensor * src = hidden_capture[i];
        if (!src || src->type != GGML_TYPE_F32 || src->ne[0] != (int64_t) hidden_size || token_idx >= (size_t) src->ne[1]) {
            return false;
        }

        const size_t n_bytes = hidden_size * sizeof(float);
        const size_t src_offset = token_idx * (size_t) src->nb[1];
        float * dst = hidden_concat_out.data() + i * hidden_size;

        ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
        if (buf_src && ggml_backend_buffer_is_host(buf_src)) {
            std::memcpy(dst, (const char *) src->data + src_offset, n_bytes);
        } else {
            ggml_backend_tensor_get(src, dst, src_offset, n_bytes);
        }
    }

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
    if (graph.ctx && graph.buf_compute && graph.gf &&
        graph.t_hidden && graph.t_probs && graph.t_topk_idx && graph.t_topk_prob &&
        graph.k == k) {
        return true;
    }

    const auto & hp = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);

    // Must cover: norm + matmul + softmax + top_k + gathers and reshapes.
    auto ctx = make_ctx_no_alloc(/* max_nodes = */ 512);
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

    // Gather top-k probabilities without transferring the full probability vector to the host.
    ggml_tensor * t_probs_flat = ggml_reshape_2d(ctx.get(), t_probs, 1, hp.draft_vocab_size);
    ggml_tensor * t_topk_idx_flat = ggml_reshape_1d(ctx.get(), t_topk_idx, k);
    t_topk_idx_flat = ggml_cont(ctx.get(), t_topk_idx_flat);
    ggml_tensor * t_topk_prob = ggml_get_rows(ctx.get(), t_probs_flat, t_topk_idx_flat); // [1, k]
    t_topk_prob = ggml_reshape_2d(ctx.get(), t_topk_prob, k, 1);
    t_topk_prob = ggml_cont(ctx.get(), t_topk_prob);

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, t_topk_idx);
    ggml_build_forward_expand(gf, t_topk_prob);

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
    graph.t_topk_prob = t_topk_prob;
    return true;
}

bool build_topk_batch_graph(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t n_beams,
        int32_t k,
        llama_eagle3_topk_batch_graph & graph) {
    if (graph.ctx && graph.buf_compute && graph.gf &&
        graph.t_hidden && graph.t_probs && graph.t_topk_idx && graph.t_topk_prob &&
        graph.n_beams == n_beams && graph.k == k &&
        (int32_t) graph.t_hidden_cols.size() == n_beams) {
        return true;
    }

    const auto & hp = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);

    // Must cover: batched norm + matmul + softmax + top_k + prob gather + views for filling.
    auto ctx = make_ctx_no_alloc(/* max_nodes = */ 1024);
    if (!ctx) {
        return false;
    }

    ggml_tensor * t_hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hp.hidden_size, n_beams);
    ggml_set_input(t_hidden);

    // Views into t_hidden for convenient per-beam filling without host staging.
    // Must be created before backend buffer allocation so the backend can assign buffer/data pointers.
    std::vector<ggml_tensor *> t_hidden_cols;
    t_hidden_cols.reserve((size_t) n_beams);
    for (int32_t ib = 0; ib < n_beams; ++ib) {
        ggml_tensor * view = ggml_view_2d(
                ctx.get(),
                t_hidden,
                hp.hidden_size,
                1,
                t_hidden->nb[1],
                (size_t) ib * t_hidden->nb[1]);
        t_hidden_cols.push_back(view);
    }

    ggml_tensor * t_norm = ggml_rms_norm(ctx.get(), t_hidden, hp.rms_norm_eps);
    ggml_tensor * t_norm_w = ggml_cast(ctx.get(), tensors.norm_w, GGML_TYPE_F32);
    t_norm = ggml_mul(ctx.get(), t_norm, t_norm_w);

    ggml_tensor * t_logits = ggml_mul_mat(ctx.get(), tensors.lm_head_w, t_norm);
    ggml_tensor * t_probs = ggml_soft_max(ctx.get(), t_logits);
    t_probs = ggml_cont(ctx.get(), t_probs);
    ggml_tensor * t_topk_idx = ggml_top_k(ctx.get(), t_probs, k);
    t_topk_idx = ggml_cont(ctx.get(), t_topk_idx);

    // Gather top-k probabilities efficiently:
    // 1) flatten probs as [1, vocab*n_beams]
    // 2) convert per-beam indices to linear indices by adding beam offsets
    // 3) ggml_get_rows() to fetch only the k*n_beams scalar probs.
    ggml_tensor * t_probs_flat = ggml_reshape_2d(ctx.get(), t_probs, 1, hp.draft_vocab_size * n_beams);

    ggml_tensor * t_beam = ggml_arange(ctx.get(), 0.0f, (float) n_beams, 1.0f); // [n_beams]
    t_beam = ggml_scale(ctx.get(), t_beam, (float) hp.draft_vocab_size);         // [n_beams]
    t_beam = ggml_reshape_2d(ctx.get(), t_beam, 1, n_beams);                     // [1, n_beams]

    ggml_tensor * t_topk_idx_f = ggml_cast(ctx.get(), t_topk_idx, GGML_TYPE_F32);
    ggml_tensor * t_beam_rep   = ggml_repeat(ctx.get(), t_beam, t_topk_idx_f); // [k, n_beams]
    ggml_tensor * t_linear_f   = ggml_add(ctx.get(), t_topk_idx_f, t_beam_rep);
    ggml_tensor * t_linear     = ggml_cast(ctx.get(), t_linear_f, GGML_TYPE_I32);
    t_linear = ggml_cont(ctx.get(), t_linear);
    ggml_tensor * t_linear_flat = ggml_reshape_1d(ctx.get(), t_linear, k * n_beams);

    ggml_tensor * t_topk_prob = ggml_get_rows(ctx.get(), t_probs_flat, t_linear_flat); // [1, k*n_beams]
    t_topk_prob = ggml_reshape_2d(ctx.get(), t_topk_prob, k, n_beams);
    t_topk_prob = ggml_cont(ctx.get(), t_topk_prob);

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, t_topk_idx);
    ggml_build_forward_expand(gf, t_topk_prob);

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
    graph.t_topk_prob = t_topk_prob;
    graph.t_hidden_cols = std::move(t_hidden_cols);
    return true;
}

bool build_select_batch_graph(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t n_beams,
        int32_t k,
        int32_t n_select,
        llama_eagle3_select_batch_graph & graph) {
    if (graph.ctx && graph.buf_compute && graph.gf &&
        graph.t_hidden && graph.t_beam_logprob &&
        graph.t_selected_linear && graph.t_selected_draft && graph.t_selected_logprob &&
        graph.n_beams == n_beams && graph.k == k && graph.n_select == n_select &&
        (int32_t) graph.t_hidden_cols.size() == n_beams) {
        return true;
    }

    const auto & hp = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);

    auto ctx = make_ctx_no_alloc(/* max_nodes = */ 1536);
    if (!ctx) {
        return false;
    }

    ggml_tensor * t_hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hp.hidden_size, n_beams);
    ggml_set_input(t_hidden);

    std::vector<ggml_tensor *> t_hidden_cols;
    t_hidden_cols.reserve((size_t) n_beams);
    for (int32_t ib = 0; ib < n_beams; ++ib) {
        ggml_tensor * view = ggml_view_2d(
                ctx.get(),
                t_hidden,
                hp.hidden_size,
                1,
                t_hidden->nb[1],
                (size_t) ib * t_hidden->nb[1]);
        t_hidden_cols.push_back(view);
    }

    ggml_tensor * t_beam_logprob = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_beams);
    ggml_set_input(t_beam_logprob);

    ggml_tensor * t_norm = ggml_rms_norm(ctx.get(), t_hidden, hp.rms_norm_eps);
    ggml_tensor * t_norm_w = ggml_cast(ctx.get(), tensors.norm_w, GGML_TYPE_F32);
    t_norm = ggml_mul(ctx.get(), t_norm, t_norm_w);

    ggml_tensor * t_logits = ggml_mul_mat(ctx.get(), tensors.lm_head_w, t_norm);
    ggml_tensor * t_probs = ggml_soft_max(ctx.get(), t_logits);
    t_probs = ggml_cont(ctx.get(), t_probs);

    ggml_tensor * t_topk_idx = ggml_top_k(ctx.get(), t_probs, k);
    t_topk_idx = ggml_cont(ctx.get(), t_topk_idx);

    ggml_tensor * t_probs_flat = ggml_reshape_2d(ctx.get(), t_probs, 1, hp.draft_vocab_size * n_beams);

    ggml_tensor * t_beam = ggml_arange(ctx.get(), 0.0f, (float) n_beams, 1.0f);
    t_beam = ggml_scale(ctx.get(), t_beam, (float) hp.draft_vocab_size);
    t_beam = ggml_reshape_2d(ctx.get(), t_beam, 1, n_beams);

    ggml_tensor * t_topk_idx_f = ggml_cast(ctx.get(), t_topk_idx, GGML_TYPE_F32);
    ggml_tensor * t_beam_rep   = ggml_repeat(ctx.get(), t_beam, t_topk_idx_f);
    ggml_tensor * t_linear_f   = ggml_add(ctx.get(), t_topk_idx_f, t_beam_rep);
    ggml_tensor * t_linear     = ggml_cast(ctx.get(), t_linear_f, GGML_TYPE_I32);
    t_linear = ggml_cont(ctx.get(), t_linear);
    ggml_tensor * t_linear_flat = ggml_reshape_1d(ctx.get(), t_linear, k * n_beams);

    ggml_tensor * t_topk_prob = ggml_get_rows(ctx.get(), t_probs_flat, t_linear_flat); // [1, k*n_beams]
    t_topk_prob = ggml_reshape_2d(ctx.get(), t_topk_prob, k, n_beams);
    t_topk_prob = ggml_cont(ctx.get(), t_topk_prob);

    ggml_tensor * t_logprob = ggml_clamp(ctx.get(), t_topk_prob, 1e-12f, 1.0f);
    t_logprob = ggml_log(ctx.get(), t_logprob);
    t_logprob = ggml_cont(ctx.get(), t_logprob);

    ggml_tensor * t_beam_lp = ggml_reshape_2d(ctx.get(), t_beam_logprob, 1, n_beams);
    t_beam_lp = ggml_repeat(ctx.get(), t_beam_lp, t_logprob);
    ggml_tensor * t_total = ggml_add(ctx.get(), t_logprob, t_beam_lp);
    t_total = ggml_cont(ctx.get(), t_total);

    ggml_tensor * t_total_flat = ggml_reshape_2d(ctx.get(), t_total, k * n_beams, 1);
    ggml_tensor * t_selected_linear = ggml_top_k(ctx.get(), t_total_flat, n_select);
    t_selected_linear = ggml_cont(ctx.get(), t_selected_linear);
    ggml_tensor * t_selected_linear_flat = ggml_reshape_1d(ctx.get(), t_selected_linear, n_select);
    t_selected_linear_flat = ggml_cont(ctx.get(), t_selected_linear_flat);
    ggml_tensor * t_selected_linear_f = ggml_cast(ctx.get(), t_selected_linear_flat, GGML_TYPE_F32);
    ggml_tensor * t_selected_parent_f = ggml_scale(ctx.get(), t_selected_linear_f, 1.0f / float(k));
    ggml_tensor * t_selected_parent = ggml_cast(ctx.get(), t_selected_parent_f, GGML_TYPE_I32);
    t_selected_parent = ggml_cont(ctx.get(), t_selected_parent);

    ggml_tensor * t_total_rows = ggml_reshape_2d(ctx.get(), t_total, 1, k * n_beams);
    ggml_tensor * t_selected_logprob = ggml_get_rows(ctx.get(), t_total_rows, t_selected_linear_flat);
    t_selected_logprob = ggml_reshape_2d(ctx.get(), t_selected_logprob, n_select, 1);
    t_selected_logprob = ggml_cont(ctx.get(), t_selected_logprob);

    ggml_tensor * t_topk_idx_rows = ggml_reshape_2d(ctx.get(), t_topk_idx_f, 1, k * n_beams);
    ggml_tensor * t_selected_draft_f = ggml_get_rows(ctx.get(), t_topk_idx_rows, t_selected_linear_flat);
    t_selected_draft_f = ggml_reshape_2d(ctx.get(), t_selected_draft_f, n_select, 1);
    ggml_tensor * t_selected_draft = ggml_cast(ctx.get(), t_selected_draft_f, GGML_TYPE_I32);
    t_selected_draft = ggml_cont(ctx.get(), t_selected_draft);
    ggml_tensor * t_selected_draft_flat = ggml_reshape_1d(ctx.get(), t_selected_draft, n_select);
    t_selected_draft_flat = ggml_cont(ctx.get(), t_selected_draft_flat);

    ggml_tensor * t_d2t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, hp.draft_vocab_size);
    ggml_set_name(t_d2t, "eagle3.d2t");
    ggml_tensor * t_selected_offset = ggml_get_rows(ctx.get(), t_d2t, t_selected_draft_flat);
    t_selected_offset = ggml_reshape_1d(ctx.get(), t_selected_offset, n_select);
    ggml_tensor * t_selected_base = ggml_add(ctx.get(), t_selected_draft_flat, t_selected_offset);
    t_selected_base = ggml_cont(ctx.get(), t_selected_base);

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, t_selected_linear);
    ggml_build_forward_expand(gf, t_selected_parent);
    ggml_build_forward_expand(gf, t_selected_draft);
    ggml_build_forward_expand(gf, t_selected_base);
    ggml_build_forward_expand(gf, t_selected_logprob);

    ggml_backend_buffer_ptr buf_compute;
    if (rt.buft_compute) {
        buf_compute.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    }
    if (!buf_compute) {
        return false;
    }

    ggml_backend_tensor_set(t_d2t, model.d2t.data(), 0, (size_t) hp.draft_vocab_size * sizeof(int32_t));

    graph.ctx = std::move(ctx);
    graph.buf_compute = std::move(buf_compute);
    graph.gf = gf;
    graph.k = k;
    graph.n_beams = n_beams;
    graph.n_select = n_select;
    graph.t_hidden = t_hidden;
    graph.t_beam_logprob = t_beam_logprob;
    graph.t_d2t = t_d2t;
    graph.t_selected_linear = t_selected_linear;
    graph.t_selected_parent = t_selected_parent;
    graph.t_selected_draft = t_selected_draft;
    graph.t_selected_base = t_selected_base;
    graph.t_selected_logprob = t_selected_logprob;
    graph.t_hidden_cols = std::move(t_hidden_cols);
    return true;
}

struct llama_eagle3_step_ops {
    ggml_tensor * t_hidden_in = nullptr;
    ggml_tensor * t_tok = nullptr;
    ggml_tensor * t_pos = nullptr;
    ggml_tensor * t_k_past_input = nullptr;
    ggml_tensor * t_v_past_input = nullptr;

    ggml_tensor * t_hidden_out = nullptr;
    ggml_tensor * t_k_curr = nullptr;
    ggml_tensor * t_v_curr = nullptr;
    ggml_tensor * t_k_total = nullptr;
    ggml_tensor * t_v_total = nullptr;
    ggml_tensor * t_logits = nullptr;
};

bool build_step_ops(
        ggml_context * ctx,
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t past_len,
        int32_t hidden_in_dim,
        bool with_logits,
        llama_eagle3_step_ops & ops) {
    if (!ctx || !rt.tok_embd) {
        return false;
    }

    const auto & hp = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);

    ggml_tensor * t_hidden_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_in_dim, 1);
    ggml_set_input(t_hidden_in);

    ggml_tensor * t_tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(t_tok);

    ggml_tensor * t_embd = ggml_get_rows(ctx, rt.tok_embd, t_tok);
    t_embd = ggml_cast(ctx, t_embd, GGML_TYPE_F32);

    ggml_tensor * t_hidden = t_hidden_in;
    if (hidden_in_dim != hp.hidden_size) {
        t_hidden = ggml_mul_mat(ctx, tensors.fc_w, t_hidden_in);
    }

    ggml_tensor * t_hidden_norm = ggml_rms_norm(ctx, t_hidden, hp.rms_norm_eps);
    ggml_tensor * t_hidden_norm_w = ggml_cast(ctx, tensors.hidden_norm_w, GGML_TYPE_F32);
    t_hidden_norm = ggml_mul(ctx, t_hidden_norm, t_hidden_norm_w);

    ggml_tensor * t_embd_norm = ggml_rms_norm(ctx, t_embd, hp.rms_norm_eps);
    ggml_tensor * t_input_norm_w = ggml_cast(ctx, tensors.input_norm_w, GGML_TYPE_F32);
    t_embd_norm = ggml_mul(ctx, t_embd_norm, t_input_norm_w);

    ggml_tensor * t_cat = ggml_concat(ctx, t_embd_norm, t_hidden_norm, 0);

    ggml_tensor * t_q = ggml_mul_mat(ctx, tensors.attn_q_w, t_cat);
    if (tensors.attn_q_b) {
        ggml_tensor * t_q_b = ggml_cast(ctx, tensors.attn_q_b, GGML_TYPE_F32);
        t_q = ggml_add(ctx, t_q, t_q_b);
    }
    ggml_tensor * t_k = ggml_mul_mat(ctx, tensors.attn_k_w, t_cat);
    if (tensors.attn_k_b) {
        ggml_tensor * t_k_b = ggml_cast(ctx, tensors.attn_k_b, GGML_TYPE_F32);
        t_k = ggml_add(ctx, t_k, t_k_b);
    }
    ggml_tensor * t_v = ggml_mul_mat(ctx, tensors.attn_v_w, t_cat);
    if (tensors.attn_v_b) {
        ggml_tensor * t_v_b = ggml_cast(ctx, tensors.attn_v_b, GGML_TYPE_F32);
        t_v = ggml_add(ctx, t_v, t_v_b);
    }

    t_q = ggml_reshape_3d(ctx, t_q, hp.head_dim, hp.num_heads, 1);
    t_k = ggml_reshape_3d(ctx, t_k, hp.head_dim, hp.num_kv_heads, 1);
    t_v = ggml_reshape_3d(ctx, t_v, hp.head_dim, hp.num_kv_heads, 1);

    ggml_tensor * t_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(t_pos);

    t_q = ggml_rope_ext(
            ctx, t_q, t_pos, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);

    t_k = ggml_rope_ext(
            ctx, t_k, t_pos, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);

    ggml_tensor * t_k_total = t_k;
    ggml_tensor * t_v_total = t_v;
    ggml_tensor * t_k_past_input = nullptr;
    ggml_tensor * t_v_past_input = nullptr;

    if (past_len > 0) {
        t_k_past_input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, past_len);
        t_v_past_input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, past_len);
        ggml_set_input(t_k_past_input);
        ggml_set_input(t_v_past_input);

        t_k_total = ggml_concat(ctx, t_k_past_input, t_k, 2);
        t_v_total = ggml_concat(ctx, t_v_past_input, t_v, 2);
    }

    ggml_tensor * t_k_attn = t_k_total;
    ggml_tensor * t_v_attn = t_v_total;

    if (hp.num_kv_heads != hp.num_heads) {
        const int32_t n_rep = hp.num_heads / hp.num_kv_heads;
        ggml_tensor * k4 = ggml_reshape_4d(ctx, t_k_attn, hp.head_dim, hp.num_kv_heads, t_k_attn->ne[2], 1);
        ggml_tensor * v4 = ggml_reshape_4d(ctx, t_v_attn, hp.head_dim, hp.num_kv_heads, t_v_attn->ne[2], 1);

        k4 = ggml_permute(ctx, k4, 0, 2, 3, 1);
        v4 = ggml_permute(ctx, v4, 0, 2, 3, 1);

        k4 = ggml_repeat_4d(ctx, k4, hp.head_dim, n_rep, hp.num_kv_heads, t_k_attn->ne[2]);
        v4 = ggml_repeat_4d(ctx, v4, hp.head_dim, n_rep, hp.num_kv_heads, t_v_attn->ne[2]);

        k4 = ggml_cont(ctx, k4);
        v4 = ggml_cont(ctx, v4);

        t_k_attn = ggml_reshape_3d(ctx, k4, hp.head_dim, hp.num_heads, t_k_attn->ne[2]);
        t_v_attn = ggml_reshape_3d(ctx, v4, hp.head_dim, hp.num_heads, t_v_attn->ne[2]);
    }

    ggml_tensor * qv = ggml_view_4d(ctx, t_q, t_q->ne[0], t_q->ne[1], t_q->ne[2], 1, t_q->nb[1], t_q->nb[2], t_q->nb[3], 0);
    ggml_tensor * kv = ggml_view_4d(ctx, t_k_attn, t_k_attn->ne[0], t_k_attn->ne[1], t_k_attn->ne[2], 1,
                                    t_k_attn->nb[1], t_k_attn->nb[2], t_k_attn->nb[3], 0);
    ggml_tensor * vv = ggml_view_4d(ctx, t_v_attn, t_v_attn->ne[0], t_v_attn->ne[1], t_v_attn->ne[2], 1,
                                    t_v_attn->nb[1], t_v_attn->nb[2], t_v_attn->nb[3], 0);

    qv = ggml_permute(ctx, qv, 0, 2, 1, 3);
    kv = ggml_permute(ctx, kv, 0, 2, 1, 3);
    vv = ggml_permute(ctx, vv, 0, 2, 1, 3);

    ggml_tensor * kq = ggml_mul_mat(ctx, kv, qv);
    const float kq_scale = 1.0f / std::sqrt(float(hp.head_dim));
    kq = ggml_scale(ctx, kq, kq_scale);
    kq = ggml_cont(ctx, kq);
    kq = ggml_soft_max(ctx, kq);

    ggml_tensor * vv_t = ggml_cont(ctx, ggml_transpose(ctx, vv));
    ggml_tensor * kqv = ggml_mul_mat(ctx, vv_t, kq);
    ggml_tensor * attn_out = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    attn_out = ggml_cont_2d(ctx, attn_out, attn_out->ne[0]*attn_out->ne[1], attn_out->ne[2]*attn_out->ne[3]);

    ggml_tensor * t_attn = ggml_mul_mat(ctx, tensors.attn_o_w, attn_out);
    if (tensors.attn_o_b) {
        ggml_tensor * t_attn_b = ggml_cast(ctx, tensors.attn_o_b, GGML_TYPE_F32);
        t_attn = ggml_add(ctx, t_attn, t_attn_b);
    }

    ggml_tensor * t_resid = hp.norm_before_residual ? t_hidden_norm : t_hidden;
    ggml_tensor * t_hidden_attn = ggml_add(ctx, t_attn, t_resid);

    ggml_tensor * t_post = ggml_rms_norm(ctx, t_hidden_attn, hp.rms_norm_eps);
    ggml_tensor * t_post_norm_w = ggml_cast(ctx, tensors.post_norm_w, GGML_TYPE_F32);
    t_post = ggml_mul(ctx, t_post, t_post_norm_w);

    ggml_tensor * t_gate = ggml_mul_mat(ctx, tensors.ffn_gate_w, t_post);
    if (tensors.ffn_gate_b) {
        ggml_tensor * t_gate_b = ggml_cast(ctx, tensors.ffn_gate_b, GGML_TYPE_F32);
        t_gate = ggml_add(ctx, t_gate, t_gate_b);
    }
    ggml_tensor * t_up = ggml_mul_mat(ctx, tensors.ffn_up_w, t_post);
    if (tensors.ffn_up_b) {
        ggml_tensor * t_up_b = ggml_cast(ctx, tensors.ffn_up_b, GGML_TYPE_F32);
        t_up = ggml_add(ctx, t_up, t_up_b);
    }
    ggml_tensor * t_act = ggml_mul(ctx, ggml_silu(ctx, t_gate), t_up);
    ggml_tensor * t_ffn = ggml_mul_mat(ctx, tensors.ffn_down_w, t_act);
    if (tensors.ffn_down_b) {
        ggml_tensor * t_down_b = ggml_cast(ctx, tensors.ffn_down_b, GGML_TYPE_F32);
        t_ffn = ggml_add(ctx, t_ffn, t_down_b);
    }

    ggml_tensor * t_hidden_out = ggml_add(ctx, t_hidden_attn, t_ffn);
    t_hidden_out = ggml_cont(ctx, t_hidden_out);
    ggml_tensor * t_k_total_out = ggml_cont(ctx, t_k_total);
    ggml_tensor * t_v_total_out = ggml_cont(ctx, t_v_total);

    ggml_tensor * t_logits = nullptr;
    if (with_logits) {
        ggml_tensor * t_norm = ggml_rms_norm(ctx, t_hidden_out, hp.rms_norm_eps);
        ggml_tensor * t_norm_w = ggml_cast(ctx, tensors.norm_w, GGML_TYPE_F32);
        t_norm = ggml_mul(ctx, t_norm, t_norm_w);
        t_logits = ggml_mul_mat(ctx, tensors.lm_head_w, t_norm);
    }

    ops.t_hidden_in = t_hidden_in;
    ops.t_tok = t_tok;
    ops.t_pos = t_pos;
    ops.t_k_past_input = t_k_past_input;
    ops.t_v_past_input = t_v_past_input;
    ops.t_hidden_out = t_hidden_out;
    ops.t_k_curr = t_k;
    ops.t_v_curr = t_v;
    ops.t_k_total = t_k_total_out;
    ops.t_v_total = t_v_total_out;
    ops.t_logits = t_logits;
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

bool build_step_batch_graph(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t past_len,
        int32_t hidden_in_dim,
        int32_t n_beams,
        bool with_logits,
        llama_eagle3_step_batch_graph & graph) {
    if (graph.ctx && graph.buf_compute && graph.gf &&
        graph.n_beams == n_beams &&
        (int32_t) graph.t_hidden_in.size() == n_beams &&
        (int32_t) graph.t_hidden_out.size() == n_beams &&
        (int32_t) graph.t_k_total.size() == n_beams &&
        (int32_t) graph.t_v_total.size() == n_beams &&
        (int32_t) graph.t_tok.size() == n_beams &&
        (int32_t) graph.t_pos.size() == n_beams) {
        return true;
    }

    if (!rt.tok_embd || n_beams <= 0) {
        return false;
    }

    const auto & hp = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);

    // This graph is executed frequently for small n_beams. The per-beam implementation
    // (duplicating the full block N times) tends to fall back to GEMV-heavy execution on CUDA.
    // Instead, we batch the expensive matmuls (embed norms, Q/K/V projections, O proj, FFN) across beams
    // and keep the attention compute per-beam (small for typical draft depths).
    auto ctx = make_ctx_no_alloc(/* max_nodes = */ (size_t) 3072 * (size_t) n_beams + 1024);
    if (!ctx) {
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());

    // Batched inputs: [*, n_beams]. Per-beam views are stored into graph.* vectors for the caller.
    ggml_tensor * t_hidden_in_b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden_in_dim, n_beams);
    ggml_set_input(t_hidden_in_b);

    ggml_tensor * t_tok_b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_beams);
    ggml_set_input(t_tok_b);

    ggml_tensor * t_pos_b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_beams);
    ggml_set_input(t_pos_b);

    std::vector<ggml_tensor *> t_hidden_in_cols;
    std::vector<ggml_tensor *> t_tok_elems;
    std::vector<ggml_tensor *> t_pos_elems;
    t_hidden_in_cols.reserve((size_t) n_beams);
    t_tok_elems.reserve((size_t) n_beams);
    t_pos_elems.reserve((size_t) n_beams);

    for (int32_t ib = 0; ib < n_beams; ++ib) {
        t_hidden_in_cols.push_back(ggml_view_2d(
                ctx.get(), t_hidden_in_b, hidden_in_dim, 1, t_hidden_in_b->nb[1], (size_t) ib * t_hidden_in_b->nb[1]));
        t_tok_elems.push_back(ggml_view_1d(ctx.get(), t_tok_b, 1, (size_t) ib * sizeof(int32_t)));
        t_pos_elems.push_back(ggml_view_1d(ctx.get(), t_pos_b, 1, (size_t) ib * sizeof(int32_t)));
    }

    // Token embeddings for all beams.
    ggml_tensor * t_embd = ggml_get_rows(ctx.get(), rt.tok_embd, t_tok_b); // [hidden, n_beams]
    t_embd = ggml_cast(ctx.get(), t_embd, GGML_TYPE_F32);

    // Teacher/prev hidden for all beams.
    ggml_tensor * t_hidden = t_hidden_in_b;
    if (hidden_in_dim != hp.hidden_size) {
        t_hidden = ggml_mul_mat(ctx.get(), tensors.fc_w, t_hidden_in_b); // [hidden, n_beams]
    }

    ggml_tensor * t_hidden_norm = ggml_rms_norm(ctx.get(), t_hidden, hp.rms_norm_eps);
    ggml_tensor * t_hidden_norm_w = ggml_cast(ctx.get(), tensors.hidden_norm_w, GGML_TYPE_F32); // [hidden]
    t_hidden_norm_w = ggml_repeat(ctx.get(), t_hidden_norm_w, t_hidden_norm);                   // [hidden, n_beams]
    t_hidden_norm = ggml_mul(ctx.get(), t_hidden_norm, t_hidden_norm_w);

    ggml_tensor * t_embd_norm = ggml_rms_norm(ctx.get(), t_embd, hp.rms_norm_eps);
    ggml_tensor * t_input_norm_w = ggml_cast(ctx.get(), tensors.input_norm_w, GGML_TYPE_F32); // [hidden]
    t_input_norm_w = ggml_repeat(ctx.get(), t_input_norm_w, t_embd_norm);                     // [hidden, n_beams]
    t_embd_norm = ggml_mul(ctx.get(), t_embd_norm, t_input_norm_w);

    ggml_tensor * t_cat = ggml_concat(ctx.get(), t_embd_norm, t_hidden_norm, 0); // [hidden*2, n_beams]

    ggml_tensor * t_q = ggml_mul_mat(ctx.get(), tensors.attn_q_w, t_cat); // [hidden, n_beams]
    if (tensors.attn_q_b) {
        ggml_tensor * t_q_b = ggml_cast(ctx.get(), tensors.attn_q_b, GGML_TYPE_F32); // [hidden]
        t_q_b = ggml_repeat(ctx.get(), t_q_b, t_q);
        t_q = ggml_add(ctx.get(), t_q, t_q_b);
    }

    ggml_tensor * t_k = ggml_mul_mat(ctx.get(), tensors.attn_k_w, t_cat); // [head_dim*n_kv, n_beams]
    if (tensors.attn_k_b) {
        ggml_tensor * t_k_b = ggml_cast(ctx.get(), tensors.attn_k_b, GGML_TYPE_F32); // [head_dim*n_kv]
        t_k_b = ggml_repeat(ctx.get(), t_k_b, t_k);
        t_k = ggml_add(ctx.get(), t_k, t_k_b);
    }

    ggml_tensor * t_v = ggml_mul_mat(ctx.get(), tensors.attn_v_w, t_cat); // [head_dim*n_kv, n_beams]
    if (tensors.attn_v_b) {
        ggml_tensor * t_v_b = ggml_cast(ctx.get(), tensors.attn_v_b, GGML_TYPE_F32); // [head_dim*n_kv]
        t_v_b = ggml_repeat(ctx.get(), t_v_b, t_v);
        t_v = ggml_add(ctx.get(), t_v, t_v_b);
    }

    // Reshape into [head_dim, n_head, n_beams] / [head_dim, n_kv, n_beams] and apply RoPE for the current token.
    t_q = ggml_reshape_3d(ctx.get(), t_q, hp.head_dim, hp.num_heads, n_beams);
    t_k = ggml_reshape_3d(ctx.get(), t_k, hp.head_dim, hp.num_kv_heads, n_beams);
    t_v = ggml_reshape_3d(ctx.get(), t_v, hp.head_dim, hp.num_kv_heads, n_beams);

    t_q = ggml_rope_ext(
            ctx.get(), t_q, t_pos_b, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);

    t_k = ggml_rope_ext(
            ctx.get(), t_k, t_pos_b, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);

    // Per-beam attention + KV concat, then batch the output projection + FFN.
    std::vector<ggml_tensor *> t_k_past_in((size_t) n_beams, nullptr);
    std::vector<ggml_tensor *> t_v_past_in((size_t) n_beams, nullptr);
    std::vector<ggml_tensor *> t_k_total_out((size_t) n_beams, nullptr);
    std::vector<ggml_tensor *> t_v_total_out((size_t) n_beams, nullptr);
    std::vector<ggml_tensor *> t_attn_out_cols;
    t_attn_out_cols.reserve((size_t) n_beams);

    for (int32_t ib = 0; ib < n_beams; ++ib) {
        // Slice current q/k/v for this beam: [*, *, 1].
        ggml_tensor * q_i = ggml_view_3d(ctx.get(), t_q, hp.head_dim, hp.num_heads, 1, t_q->nb[1], t_q->nb[2], (size_t) ib * t_q->nb[2]);
        ggml_tensor * k_i = ggml_view_3d(ctx.get(), t_k, hp.head_dim, hp.num_kv_heads, 1, t_k->nb[1], t_k->nb[2], (size_t) ib * t_k->nb[2]);
        ggml_tensor * v_i = ggml_view_3d(ctx.get(), t_v, hp.head_dim, hp.num_kv_heads, 1, t_v->nb[1], t_v->nb[2], (size_t) ib * t_v->nb[2]);

        ggml_tensor * k_total = k_i;
        ggml_tensor * v_total = v_i;

        if (past_len > 0) {
            ggml_tensor * k_past = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, past_len);
            ggml_tensor * v_past = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, past_len);
            ggml_set_input(k_past);
            ggml_set_input(v_past);
            t_k_past_in[(size_t) ib] = k_past;
            t_v_past_in[(size_t) ib] = v_past;

            k_total = ggml_concat(ctx.get(), k_past, k_i, 2); // [head_dim, n_kv, past+1]
            v_total = ggml_concat(ctx.get(), v_past, v_i, 2);
        }

        ggml_tensor * k_attn = k_total;
        ggml_tensor * v_attn = v_total;

        if (hp.num_kv_heads != hp.num_heads) {
            const int32_t n_rep = hp.num_heads / hp.num_kv_heads;
            ggml_tensor * k4 = ggml_reshape_4d(ctx.get(), k_attn, hp.head_dim, hp.num_kv_heads, k_attn->ne[2], 1);
            ggml_tensor * v4 = ggml_reshape_4d(ctx.get(), v_attn, hp.head_dim, hp.num_kv_heads, v_attn->ne[2], 1);

            k4 = ggml_permute(ctx.get(), k4, 0, 2, 3, 1);
            v4 = ggml_permute(ctx.get(), v4, 0, 2, 3, 1);

            k4 = ggml_repeat_4d(ctx.get(), k4, hp.head_dim, n_rep, hp.num_kv_heads, k_attn->ne[2]); // [head_dim, n_rep, n_kv, seq]
            v4 = ggml_repeat_4d(ctx.get(), v4, hp.head_dim, n_rep, hp.num_kv_heads, v_attn->ne[2]);

            k4 = ggml_cont(ctx.get(), k4);
            v4 = ggml_cont(ctx.get(), v4);

            k_attn = ggml_reshape_3d(ctx.get(), k4, hp.head_dim, hp.num_heads, k_attn->ne[2]);
            v_attn = ggml_reshape_3d(ctx.get(), v4, hp.head_dim, hp.num_heads, v_attn->ne[2]);
        }

        ggml_tensor * qv = ggml_view_4d(ctx.get(), q_i, q_i->ne[0], q_i->ne[1], q_i->ne[2], 1, q_i->nb[1], q_i->nb[2], q_i->nb[3], 0);
        ggml_tensor * kv = ggml_view_4d(ctx.get(), k_attn, k_attn->ne[0], k_attn->ne[1], k_attn->ne[2], 1,
                                        k_attn->nb[1], k_attn->nb[2], k_attn->nb[3], 0);
        ggml_tensor * vv = ggml_view_4d(ctx.get(), v_attn, v_attn->ne[0], v_attn->ne[1], v_attn->ne[2], 1,
                                        v_attn->nb[1], v_attn->nb[2], v_attn->nb[3], 0);

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
        attn_out = ggml_cont_2d(ctx.get(), attn_out, attn_out->ne[0]*attn_out->ne[1], attn_out->ne[2]*attn_out->ne[3]); // [hidden, 1]

        t_attn_out_cols.push_back(attn_out);

        // Materialize KV totals for the caller (stored in KV-head space, not repeated-to-heads).
        t_k_total_out[(size_t) ib] = ggml_cont(ctx.get(), k_total);
        t_v_total_out[(size_t) ib] = ggml_cont(ctx.get(), v_total);
    }

    ggml_tensor * t_attn_out_b = t_attn_out_cols.empty() ? nullptr : t_attn_out_cols[0];
    for (int32_t ib = 1; ib < n_beams; ++ib) {
        t_attn_out_b = ggml_concat(ctx.get(), t_attn_out_b, t_attn_out_cols[(size_t) ib], 1);
    }
    t_attn_out_b = ggml_cont(ctx.get(), t_attn_out_b); // [hidden, n_beams]

    ggml_tensor * t_attn = ggml_mul_mat(ctx.get(), tensors.attn_o_w, t_attn_out_b); // [hidden, n_beams]
    if (tensors.attn_o_b) {
        ggml_tensor * t_attn_b = ggml_cast(ctx.get(), tensors.attn_o_b, GGML_TYPE_F32);
        t_attn_b = ggml_repeat(ctx.get(), t_attn_b, t_attn);
        t_attn = ggml_add(ctx.get(), t_attn, t_attn_b);
    }

    ggml_tensor * t_resid = hp.norm_before_residual ? t_hidden_norm : t_hidden;
    ggml_tensor * t_hidden_attn = ggml_add(ctx.get(), t_attn, t_resid); // [hidden, n_beams]

    ggml_tensor * t_post = ggml_rms_norm(ctx.get(), t_hidden_attn, hp.rms_norm_eps);
    ggml_tensor * t_post_norm_w = ggml_cast(ctx.get(), tensors.post_norm_w, GGML_TYPE_F32);
    t_post_norm_w = ggml_repeat(ctx.get(), t_post_norm_w, t_post);
    t_post = ggml_mul(ctx.get(), t_post, t_post_norm_w);

    ggml_tensor * t_gate = ggml_mul_mat(ctx.get(), tensors.ffn_gate_w, t_post);
    if (tensors.ffn_gate_b) {
        ggml_tensor * t_gate_b = ggml_cast(ctx.get(), tensors.ffn_gate_b, GGML_TYPE_F32);
        t_gate_b = ggml_repeat(ctx.get(), t_gate_b, t_gate);
        t_gate = ggml_add(ctx.get(), t_gate, t_gate_b);
    }

    ggml_tensor * t_up = ggml_mul_mat(ctx.get(), tensors.ffn_up_w, t_post);
    if (tensors.ffn_up_b) {
        ggml_tensor * t_up_b = ggml_cast(ctx.get(), tensors.ffn_up_b, GGML_TYPE_F32);
        t_up_b = ggml_repeat(ctx.get(), t_up_b, t_up);
        t_up = ggml_add(ctx.get(), t_up, t_up_b);
    }

    ggml_tensor * t_act = ggml_mul(ctx.get(), ggml_silu(ctx.get(), t_gate), t_up);
    ggml_tensor * t_ffn = ggml_mul_mat(ctx.get(), tensors.ffn_down_w, t_act);
    if (tensors.ffn_down_b) {
        ggml_tensor * t_down_b = ggml_cast(ctx.get(), tensors.ffn_down_b, GGML_TYPE_F32);
        t_down_b = ggml_repeat(ctx.get(), t_down_b, t_ffn);
        t_ffn = ggml_add(ctx.get(), t_ffn, t_down_b);
    }

    ggml_tensor * t_hidden_out_b = ggml_add(ctx.get(), t_hidden_attn, t_ffn);
    t_hidden_out_b = ggml_cont(ctx.get(), t_hidden_out_b); // [hidden, n_beams]

    // Per-beam views for outputs (caller API expects [hidden,1] tensors).
    std::vector<ggml_tensor *> t_hidden_out_views;
    t_hidden_out_views.reserve((size_t) n_beams);
    for (int32_t ib = 0; ib < n_beams; ++ib) {
        t_hidden_out_views.push_back(ggml_view_2d(
                ctx.get(), t_hidden_out_b, hp.hidden_size, 1, t_hidden_out_b->nb[1], (size_t) ib * t_hidden_out_b->nb[1]));
    }

    // Optional per-beam logits (rare for step_batch; used by tests/tools).
    std::vector<ggml_tensor *> t_logits_views((size_t) n_beams, nullptr);
    if (with_logits) {
        ggml_tensor * t_norm = ggml_rms_norm(ctx.get(), t_hidden_out_b, hp.rms_norm_eps);
        ggml_tensor * t_norm_w = ggml_cast(ctx.get(), tensors.norm_w, GGML_TYPE_F32);
        t_norm_w = ggml_repeat(ctx.get(), t_norm_w, t_norm);
        t_norm = ggml_mul(ctx.get(), t_norm, t_norm_w);
        ggml_tensor * t_logits_b = ggml_mul_mat(ctx.get(), tensors.lm_head_w, t_norm); // [vocab, n_beams]
        for (int32_t ib = 0; ib < n_beams; ++ib) {
            // Column view: [vocab, 1]
            t_logits_views[(size_t) ib] = ggml_view_2d(
                    ctx.get(), t_logits_b, hp.draft_vocab_size, 1, t_logits_b->nb[1], (size_t) ib * t_logits_b->nb[1]);
        }
    }

    for (int32_t ib = 0; ib < n_beams; ++ib) {
        ggml_build_forward_expand(gf, t_hidden_out_views[(size_t) ib]);
        ggml_build_forward_expand(gf, t_k_total_out[(size_t) ib]);
        ggml_build_forward_expand(gf, t_v_total_out[(size_t) ib]);
        if (t_logits_views[(size_t) ib]) {
            ggml_build_forward_expand(gf, t_logits_views[(size_t) ib]);
        }
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
    graph.n_beams = n_beams;
    graph.t_hidden_in_b = t_hidden_in_b;
    graph.t_tok_b       = t_tok_b;
    graph.t_pos_b       = t_pos_b;
    graph.t_hidden_in.resize((size_t) n_beams);
    graph.t_tok.resize((size_t) n_beams);
    graph.t_pos.resize((size_t) n_beams);
    graph.t_k_past_input.resize((size_t) n_beams);
    graph.t_v_past_input.resize((size_t) n_beams);
    graph.t_hidden_out.resize((size_t) n_beams);
    graph.t_k_total.resize((size_t) n_beams);
    graph.t_v_total.resize((size_t) n_beams);
    graph.t_logits.resize((size_t) n_beams);

    for (int32_t ib = 0; ib < n_beams; ++ib) {
        graph.t_hidden_in[(size_t) ib] = t_hidden_in_cols[(size_t) ib];
        graph.t_tok[(size_t) ib] = t_tok_elems[(size_t) ib];
        graph.t_pos[(size_t) ib] = t_pos_elems[(size_t) ib];
        graph.t_k_past_input[(size_t) ib] = t_k_past_in[(size_t) ib];
        graph.t_v_past_input[(size_t) ib] = t_v_past_in[(size_t) ib];
        graph.t_hidden_out[(size_t) ib] = t_hidden_out_views[(size_t) ib];
        graph.t_k_total[(size_t) ib] = t_k_total_out[(size_t) ib];
        graph.t_v_total[(size_t) ib] = t_v_total_out[(size_t) ib];
        graph.t_logits[(size_t) ib] = t_logits_views[(size_t) ib];
    }

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
    rt.target_backend = ctx_impl->primary_backend();
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

    if (rt.backend_compute && rt.buft_compute) {
        if (build_logits_graph(model, rt, rt.logits_graph)) {
            ggml_backend_tensor_set_async(rt.backend_compute.get(), rt.logits_graph.t_hidden, hidden, 0, hp.hidden_size * sizeof(float));
            const ggml_status status = ggml_backend_graph_compute_async(rt.backend_compute.get(), rt.logits_graph.gf);
            if (status == GGML_STATUS_SUCCESS) {
                logits_out.resize(hp.draft_vocab_size);
                ggml_backend_tensor_get_async(rt.backend_compute.get(), rt.logits_graph.t_logits, logits_out.data(), 0, hp.draft_vocab_size * sizeof(float));
                ggml_backend_synchronize(rt.backend_compute.get());
                return true;
            }
        }
    }

    const auto & tensors = get_host_tensors(model);

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
            ggml_backend_tensor_set_async(rt.backend_compute.get(), rt.topk_graph.t_hidden, hidden, 0, hp.hidden_size * sizeof(float));
            const ggml_status status = ggml_backend_graph_compute_async(rt.backend_compute.get(), rt.topk_graph.gf);
            if (status == GGML_STATUS_SUCCESS) {
                topk_idx_out.resize(k);
                topk_prob_out.resize(k);
                ggml_backend_tensor_get_async(rt.backend_compute.get(), rt.topk_graph.t_topk_idx, topk_idx_out.data(), 0, (size_t) k * sizeof(int32_t));
                ggml_backend_tensor_get_async(rt.backend_compute.get(), rt.topk_graph.t_topk_prob, topk_prob_out.data(), 0, (size_t) k * sizeof(float));
                ggml_backend_synchronize(rt.backend_compute.get());
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
            ggml_backend_tensor_copy_async(rt.backend_compute.get(), rt.backend_compute.get(), state.dev->t_hidden, rt.topk_graph.t_hidden);
            const ggml_status status = ggml_backend_graph_compute_async(rt.backend_compute.get(), rt.topk_graph.gf);
            if (status == GGML_STATUS_SUCCESS) {
                topk_idx_out.resize(k);
                topk_prob_out.resize(k);
                ggml_backend_tensor_get_async(rt.backend_compute.get(), rt.topk_graph.t_topk_idx, topk_idx_out.data(), 0, (size_t) k * sizeof(int32_t));
                ggml_backend_tensor_get_async(rt.backend_compute.get(), rt.topk_graph.t_topk_prob, topk_prob_out.data(), 0, (size_t) k * sizeof(float));
                ggml_backend_synchronize(rt.backend_compute.get());
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
#if defined(GGML_USE_CUDA)
        ggml_cuda_profiler_scope zone_total(rt.backend_compute.get(), "eagle3/topk_state_batch");
#endif
        if (build_topk_batch_graph(model, rt, n_beams, k, rt.topk_batch_graph)) {
            auto & graph = rt.topk_batch_graph;
            if ((int32_t) graph.t_hidden_cols.size() != n_beams) {
                return false;
            }

            {
#if defined(GGML_USE_CUDA)
                ggml_cuda_profiler_scope zone_fill(rt.backend_compute.get(), "eagle3/topk_state_batch/fill_hidden");
#endif
                for (int32_t ib = 0; ib < n_beams; ++ib) {
                    const llama_eagle3_state * st = states[ib];
                    if (st == nullptr) {
                        return false;
                    }

                    ggml_tensor * dst = graph.t_hidden_cols[(size_t) ib];
                    if (!dst) {
                        return false;
                    }

                    if (!st->hidden.empty()) {
                        ggml_backend_tensor_set_async(rt.backend_compute.get(), dst, st->hidden.data(), 0, (size_t) hp.hidden_size * sizeof(float));
                    } else if (st->dev && st->dev->t_hidden) {
                        ggml_backend_tensor_copy_async(rt.backend_compute.get(), rt.backend_compute.get(), st->dev->t_hidden, dst);
                    } else {
                        return false;
                    }
                }
            }

            ggml_status status = GGML_STATUS_FAILED;
            {
#if defined(GGML_USE_CUDA)
                ggml_cuda_profiler_scope zone_graph(rt.backend_compute.get(), "eagle3/topk_state_batch/graph");
#endif
                status = ggml_backend_graph_compute_async(rt.backend_compute.get(), graph.gf);
            }
            if (status == GGML_STATUS_SUCCESS) {
                topk_idx_out.resize((size_t) n_beams * k);
                topk_prob_out.resize((size_t) n_beams * k);

                {
#if defined(GGML_USE_CUDA)
                    ggml_cuda_profiler_scope zone_get(rt.backend_compute.get(), "eagle3/topk_state_batch/get_topk");
#endif
                    ggml_backend_tensor_get_async(
                            rt.backend_compute.get(),
                            graph.t_topk_idx, topk_idx_out.data(), 0, topk_idx_out.size() * sizeof(int32_t));
                    ggml_backend_tensor_get_async(
                            rt.backend_compute.get(),
                            graph.t_topk_prob, topk_prob_out.data(), 0, topk_prob_out.size() * sizeof(float));
                }
                ggml_backend_synchronize(rt.backend_compute.get());
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

bool llama_eagle3_select_state_batch(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const std::vector<const llama_eagle3_state *> & states,
        const std::vector<float> & beam_logprob,
        int32_t k,
        std::vector<int32_t> & selected_linear_out,
        std::vector<int32_t> & selected_draft_idx_out,
        std::vector<float> & selected_logprob_out) {
    const auto & hp = model.hparams;
    const int32_t n_beams = (int32_t) states.size();
    if (n_beams <= 0 || (int32_t) beam_logprob.size() != n_beams) {
        return false;
    }

    k = std::min(k, hp.draft_vocab_size);
    if (k <= 0) {
        return false;
    }

    const int32_t n_select = n_beams * k;
    if (n_select <= 0) {
        return false;
    }

    if (rt.backend_compute && rt.buft_compute) {
#if defined(GGML_USE_CUDA)
        ggml_cuda_profiler_scope zone_total(rt.backend_compute.get(), "eagle3/select_state_batch");
#endif
        if (build_select_batch_graph(model, rt, n_beams, k, n_select, rt.select_batch_graph)) {
            auto & graph = rt.select_batch_graph;
            if ((int32_t) graph.t_hidden_cols.size() != n_beams) {
                return false;
            }

            for (int32_t ib = 0; ib < n_beams; ++ib) {
                const llama_eagle3_state * st = states[(size_t) ib];
                if (!st) {
                    return false;
                }

                ggml_tensor * dst = graph.t_hidden_cols[(size_t) ib];
                if (!dst) {
                    return false;
                }

                if (!st->hidden.empty()) {
                    ggml_backend_tensor_set_async(rt.backend_compute.get(), dst, st->hidden.data(), 0, (size_t) hp.hidden_size * sizeof(float));
                } else if (st->dev && st->dev->t_hidden) {
                    ggml_backend_tensor_copy_async(rt.backend_compute.get(), rt.backend_compute.get(), st->dev->t_hidden, dst);
                } else {
                    return false;
                }
            }

            ggml_backend_tensor_set_async(
                    rt.backend_compute.get(),
                    graph.t_beam_logprob,
                    beam_logprob.data(),
                    0,
                    beam_logprob.size() * sizeof(float));

            const ggml_status status = ggml_backend_graph_compute_async(rt.backend_compute.get(), graph.gf);
            if (status == GGML_STATUS_SUCCESS) {
                selected_linear_out.resize((size_t) n_select);
                selected_draft_idx_out.resize((size_t) n_select);
                selected_logprob_out.resize((size_t) n_select);

                ggml_backend_tensor_get_async(
                        rt.backend_compute.get(),
                        graph.t_selected_linear,
                        selected_linear_out.data(),
                        0,
                        selected_linear_out.size() * sizeof(int32_t));
                ggml_backend_tensor_get_async(
                        rt.backend_compute.get(),
                        graph.t_selected_draft,
                        selected_draft_idx_out.data(),
                        0,
                        selected_draft_idx_out.size() * sizeof(int32_t));
                ggml_backend_tensor_get_async(
                        rt.backend_compute.get(),
                        graph.t_selected_logprob,
                        selected_logprob_out.data(),
                        0,
                        selected_logprob_out.size() * sizeof(float));
                ggml_backend_synchronize(rt.backend_compute.get());
                return true;
            }
        }
    }

    std::vector<int32_t> topk_idx;
    std::vector<float> topk_prob;
    if (!llama_eagle3_topk_state_batch(model, rt, states, k, topk_idx, topk_prob)) {
        return false;
    }

    struct candidate {
        int32_t linear = 0;
        int32_t draft_idx = 0;
        float logprob = -INFINITY;
    };
    std::vector<candidate> all;
    all.reserve((size_t) n_select);
    for (int32_t ib = 0; ib < n_beams; ++ib) {
        const size_t off = (size_t) ib * k;
        for (int32_t j = 0; j < k; ++j) {
            const int32_t draft_idx = topk_idx[off + (size_t) j];
            const float prob = topk_prob[off + (size_t) j];
            all.push_back({
                    /* linear    = */ ib * k + j,
                    /* draft_idx = */ draft_idx,
                    /* logprob   = */ beam_logprob[(size_t) ib] + std::log(std::max(prob, 1e-12f)),
            });
        }
    }

    std::sort(all.begin(), all.end(),
        [](const candidate & a, const candidate & b) {
            return a.logprob > b.logprob;
        });

    selected_linear_out.resize((size_t) n_select);
    selected_draft_idx_out.resize((size_t) n_select);
    selected_logprob_out.resize((size_t) n_select);
    for (int32_t i = 0; i < n_select; ++i) {
        selected_linear_out[(size_t) i] = all[(size_t) i].linear;
        selected_draft_idx_out[(size_t) i] = all[(size_t) i].draft_idx;
        selected_logprob_out[(size_t) i] = all[(size_t) i].logprob;
    }
    return true;
}

bool llama_eagle3_select_state_batch_device(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const std::vector<const llama_eagle3_state *> & states,
        const std::vector<float> & beam_logprob,
        int32_t k,
        llama_eagle3_select_batch_device_result & out) {
    out = {};

    const auto & hp = model.hparams;
    const int32_t n_beams = (int32_t) states.size();
    if (n_beams <= 0 || (int32_t) beam_logprob.size() != n_beams) {
        return false;
    }

    k = std::min(k, hp.draft_vocab_size);
    if (k <= 0) {
        return false;
    }

    const int32_t n_select = n_beams * k;
    if (n_select <= 0) {
        return false;
    }

    if (!(rt.backend_compute && rt.buft_compute)) {
        return false;
    }

    if (!build_select_batch_graph(model, rt, n_beams, k, n_select, rt.select_batch_graph)) {
        return false;
    }

    auto & graph = rt.select_batch_graph;
    if ((int32_t) graph.t_hidden_cols.size() != n_beams) {
        return false;
    }

    for (int32_t ib = 0; ib < n_beams; ++ib) {
        const llama_eagle3_state * st = states[(size_t) ib];
        if (!st) {
            return false;
        }

        ggml_tensor * dst = graph.t_hidden_cols[(size_t) ib];
        if (!dst) {
            return false;
        }

        if (!st->hidden.empty()) {
            ggml_backend_tensor_set_async(rt.backend_compute.get(), dst, st->hidden.data(), 0, (size_t) hp.hidden_size * sizeof(float));
        } else if (st->dev && st->dev->t_hidden) {
            ggml_backend_tensor_copy_async(rt.backend_compute.get(), rt.backend_compute.get(), st->dev->t_hidden, dst);
        } else {
            return false;
        }
    }

    ggml_backend_tensor_set_async(
            rt.backend_compute.get(),
            graph.t_beam_logprob,
            beam_logprob.data(),
            0,
            beam_logprob.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute_async(rt.backend_compute.get(), graph.gf);
    if (status != GGML_STATUS_SUCCESS) {
        return false;
    }

    out.backend = rt.backend_compute.get();
    out.t_selected_linear = graph.t_selected_linear;
    out.t_selected_parent = graph.t_selected_parent;
    out.t_selected_draft = graph.t_selected_draft;
    out.t_selected_base = graph.t_selected_base;
    out.t_selected_logprob = graph.t_selected_logprob;
    out.n_beams = n_beams;
    out.k = k;
    out.n_select = n_select;
    return true;
}

bool llama_eagle3_select_state_slots_device(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const std::vector<const llama_eagle3_state *> & states,
        const std::vector<uint8_t> & active_mask,
        const std::vector<float> & beam_logprob,
        int32_t k,
        llama_eagle3_select_batch_device_result & out) {
    out = {};

    const auto & hp = model.hparams;
    const int32_t n_beams = (int32_t) states.size();
    if (n_beams <= 0 || (int32_t) active_mask.size() != n_beams || (int32_t) beam_logprob.size() != n_beams) {
        return false;
    }

    k = std::min(k, hp.draft_vocab_size);
    if (k <= 0) {
        return false;
    }

    const int32_t n_select = n_beams * k;
    if (n_select <= 0) {
        return false;
    }

    if (!(rt.backend_compute && rt.buft_compute)) {
        return false;
    }

    if (!build_select_batch_graph(model, rt, n_beams, k, n_select, rt.select_batch_graph)) {
        return false;
    }

    auto & graph = rt.select_batch_graph;
    if ((int32_t) graph.t_hidden_cols.size() != n_beams) {
        return false;
    }

    std::vector<float> zero_hidden((size_t) hp.hidden_size, 0.0f);
    std::vector<float> masked_logprob = beam_logprob;
    const float inactive_logprob = -1e30f;

    for (int32_t ib = 0; ib < n_beams; ++ib) {
        const bool active = active_mask[(size_t) ib] != 0;
        const llama_eagle3_state * st = states[(size_t) ib];
        ggml_tensor * dst = graph.t_hidden_cols[(size_t) ib];
        if (!dst) {
            return false;
        }

        if (active && st && st->dev && st->dev->t_hidden) {
            ggml_backend_tensor_copy_async(rt.backend_compute.get(), rt.backend_compute.get(), st->dev->t_hidden, dst);
        } else if (active && st && !st->hidden.empty()) {
            ggml_backend_tensor_set_async(rt.backend_compute.get(), dst, st->hidden.data(), 0, (size_t) hp.hidden_size * sizeof(float));
        } else {
            masked_logprob[(size_t) ib] = inactive_logprob;
            ggml_backend_tensor_set_async(rt.backend_compute.get(), dst, zero_hidden.data(), 0, (size_t) hp.hidden_size * sizeof(float));
        }
    }

    ggml_backend_tensor_set_async(
            rt.backend_compute.get(),
            graph.t_beam_logprob,
            masked_logprob.data(),
            0,
            masked_logprob.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute_async(rt.backend_compute.get(), graph.gf);
    if (status != GGML_STATUS_SUCCESS) {
        return false;
    }

    out.backend = rt.backend_compute.get();
    out.t_selected_linear = graph.t_selected_linear;
    out.t_selected_parent = graph.t_selected_parent;
    out.t_selected_draft = graph.t_selected_draft;
    out.t_selected_base = graph.t_selected_base;
    out.t_selected_logprob = graph.t_selected_logprob;
    out.n_beams = n_beams;
    out.k = k;
    out.n_select = n_select;
    return true;
}

bool llama_eagle3_select_state_slots(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const std::vector<const llama_eagle3_state *> & states,
        const std::vector<uint8_t> & active_mask,
        const std::vector<float> & beam_logprob,
        int32_t k,
        std::vector<int32_t> & selected_linear_out,
        std::vector<int32_t> & selected_draft_idx_out,
        std::vector<float> & selected_logprob_out) {
    const auto & hp = model.hparams;
    const int32_t n_beams = (int32_t) states.size();
    if (n_beams <= 0 || (int32_t) active_mask.size() != n_beams || (int32_t) beam_logprob.size() != n_beams) {
        return false;
    }

    k = std::min(k, hp.draft_vocab_size);
    if (k <= 0) {
        return false;
    }

    const int32_t n_select = n_beams * k;
    if (n_select <= 0) {
        return false;
    }

    if (rt.backend_compute && rt.buft_compute) {
        llama_eagle3_select_batch_device_result device_out;
        if (llama_eagle3_select_state_slots_device(model, rt, states, active_mask, beam_logprob, k, device_out)) {
            selected_linear_out.resize((size_t) n_select);
            selected_draft_idx_out.resize((size_t) n_select);
            selected_logprob_out.resize((size_t) n_select);

            ggml_backend_tensor_get_async(
                    rt.backend_compute.get(),
                    const_cast<ggml_tensor *>(device_out.t_selected_linear),
                    selected_linear_out.data(),
                    0,
                    selected_linear_out.size() * sizeof(int32_t));
            ggml_backend_tensor_get_async(
                    rt.backend_compute.get(),
                    const_cast<ggml_tensor *>(device_out.t_selected_draft),
                    selected_draft_idx_out.data(),
                    0,
                    selected_draft_idx_out.size() * sizeof(int32_t));
            ggml_backend_tensor_get_async(
                    rt.backend_compute.get(),
                    const_cast<ggml_tensor *>(device_out.t_selected_logprob),
                    selected_logprob_out.data(),
                    0,
                    selected_logprob_out.size() * sizeof(float));
            ggml_backend_synchronize(rt.backend_compute.get());
            return true;
        }
    }

    std::vector<const llama_eagle3_state *> active_states;
    std::vector<float> active_logprob;
    std::vector<int32_t> active_to_slot;
    active_states.reserve((size_t) n_beams);
    active_logprob.reserve((size_t) n_beams);
    active_to_slot.reserve((size_t) n_beams);

    for (int32_t ib = 0; ib < n_beams; ++ib) {
        if (active_mask[(size_t) ib] == 0) {
            continue;
        }
        const llama_eagle3_state * st = states[(size_t) ib];
        if (!st || !llama_eagle3_state_has_hidden(*st)) {
            continue;
        }
        active_to_slot.push_back(ib);
        active_states.push_back(st);
        active_logprob.push_back(beam_logprob[(size_t) ib]);
    }

    if (active_states.empty()) {
        selected_linear_out.assign((size_t) n_select, -1);
        selected_draft_idx_out.assign((size_t) n_select, -1);
        selected_logprob_out.assign((size_t) n_select, -INFINITY);
        return true;
    }

    std::vector<int32_t> active_linear;
    std::vector<int32_t> active_draft_idx;
    std::vector<float> active_selected_logprob;
    if (!llama_eagle3_select_state_batch(model, rt, active_states, active_logprob, k, active_linear, active_draft_idx, active_selected_logprob)) {
        return false;
    }

    selected_linear_out.assign((size_t) n_select, -1);
    selected_draft_idx_out.assign((size_t) n_select, -1);
    selected_logprob_out.assign((size_t) n_select, -INFINITY);

    const size_t n_active_select = active_linear.size();
    for (size_t i = 0; i < n_active_select && i < (size_t) n_select; ++i) {
        const int32_t active_linear_i = active_linear[i];
        const int32_t active_parent = active_linear_i / k;
        const int32_t local_rank = active_linear_i % k;
        if (active_parent < 0 || active_parent >= (int32_t) active_to_slot.size()) {
            continue;
        }
        selected_linear_out[i] = active_to_slot[(size_t) active_parent] * k + local_rank;
        selected_draft_idx_out[i] = active_draft_idx[i];
        selected_logprob_out[i] = active_selected_logprob[i];
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
    const auto & hp = model.hparams;
    if (!state.hidden.empty()) {
        hidden_out = state.hidden;
        return hidden_out.size() == (size_t) hp.hidden_size;
    }

    if (state.dev && state.dev->t_hidden) {
        if (rt.backend_compute) {
            ggml_backend_synchronize(rt.backend_compute.get());
        }
        hidden_out.resize(hp.hidden_size);
        ggml_backend_tensor_get(state.dev->t_hidden, hidden_out.data(), 0, (size_t) hp.hidden_size * sizeof(float));
        state.hidden = hidden_out;
        return true;
    }

    return false;
}

bool llama_eagle3_step_batch(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const std::vector<llama_eagle3_state *> & states,
        int32_t hidden_in_dim,
        const std::vector<llama_token> & input_ids) {
    if (states.empty()) {
        return false;
    }

    std::vector<const llama_eagle3_state *> parent_states;
    parent_states.reserve(states.size());
    for (const auto * st : states) {
        parent_states.push_back(st);
    }

    return llama_eagle3_step_batch_from_parents(
            model,
            rt,
            parent_states,
            hidden_in_dim,
            input_ids,
            states,
            /* reserve_kv = */ 0);
}

bool llama_eagle3_step_batch_from_parents(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const std::vector<const llama_eagle3_state *> & parent_states,
        int32_t hidden_in_dim,
        const std::vector<llama_token> & input_ids,
        const std::vector<llama_eagle3_state *> & out_states,
        int32_t reserve_kv) {
    if (parent_states.empty() || parent_states.size() != input_ids.size() || parent_states.size() != out_states.size()) {
        return false;
    }

    const auto & hp = model.hparams;
    const int32_t n_beams = (int32_t) parent_states.size();
    const bool with_logits = false;

    if (rt.backend_compute && rt.buft_compute) {
#if defined(GGML_USE_CUDA)
        ggml_cuda_profiler_scope zone_total(rt.backend_compute.get(), "eagle3/step_batch");
#endif
        int32_t past_len = parent_states[0] ? parent_states[0]->past_len : -1;
        if (past_len < 0) {
            return false;
        }
        for (const auto * st : parent_states) {
            if (!st || st->past_len != past_len) {
                return false;
            }
        }

        const uint64_t key = make_step_batch_graph_key(past_len, hidden_in_dim, n_beams, with_logits);
        if (rt.step_batch_graph_key != key) {
            rt.step_batch_graph = {};
            if (build_step_batch_graph(model, rt, past_len, hidden_in_dim, n_beams, with_logits, rt.step_batch_graph)) {
                rt.step_batch_graph_key = key;
            }
        }

        if (rt.step_batch_graph_key == key && rt.step_batch_graph.gf) {
            auto & graph = rt.step_batch_graph;
            const size_t block = (size_t) hp.head_dim * hp.num_kv_heads;
            const size_t past_bytes = (size_t) past_len * block * sizeof(float);

            {
#if defined(GGML_USE_CUDA)
                ggml_cuda_profiler_scope zone_set_inputs(rt.backend_compute.get(), "eagle3/step_batch/set_inputs");
#endif
                // These are tiny and were previously uploaded per-beam via 1-element views, causing lots
                // of H2D copies (and stream syncs on the synchronous buffer path). Upload batched once.
                if (graph.t_tok_b) {
                    ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_tok_b, input_ids.data(), 0, (size_t) n_beams * sizeof(llama_token));
                }
                if (graph.t_pos_b) {
                    std::vector<int32_t> pos_host((size_t) n_beams, past_len);
                    ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_pos_b, pos_host.data(), 0, pos_host.size() * sizeof(int32_t));
                }

                for (int32_t ib = 0; ib < n_beams; ++ib) {
                    const llama_eagle3_state * st = parent_states[(size_t) ib];
                    if (!st || !out_states[(size_t) ib]) {
                        return false;
                    }

                    if (st->dev && st->dev->t_hidden) {
                        ggml_backend_tensor_copy_async(rt.backend_compute.get(), rt.backend_compute.get(), st->dev->t_hidden, graph.t_hidden_in[(size_t) ib]);
                    } else if (!st->hidden.empty()) {
                        ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_hidden_in[(size_t) ib], st->hidden.data(), 0, (size_t) hidden_in_dim * sizeof(float));
                    } else {
                        return false;
                    }

                    if (past_len > 0 && graph.t_k_past_input[(size_t) ib] && graph.t_v_past_input[(size_t) ib]) {
                        if (st->dev && st->dev->t_k && st->dev->t_v && st->dev->kv_capacity >= past_len) {
                            if (!tensor_copy_3d_prefix_async(
                                        rt.backend_compute.get(),
                                        st->dev->t_k,
                                        graph.t_k_past_input[(size_t) ib],
                                        hp.head_dim,
                                        hp.num_kv_heads,
                                        past_len)) {
                                return false;
                            }
                            if (!tensor_copy_3d_prefix_async(
                                        rt.backend_compute.get(),
                                        st->dev->t_v,
                                        graph.t_v_past_input[(size_t) ib],
                                        hp.head_dim,
                                        hp.num_kv_heads,
                                        past_len)) {
                                return false;
                            }
                        } else if (st->k.size() * sizeof(float) >= past_bytes && st->v.size() * sizeof(float) >= past_bytes) {
                            ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_k_past_input[(size_t) ib], st->k.data(), 0, past_bytes);
                            ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_v_past_input[(size_t) ib], st->v.data(), 0, past_bytes);
                        } else {
                            return false;
                        }
                    }
                }
            }

            ggml_status status = GGML_STATUS_FAILED;
            {
#if defined(GGML_USE_CUDA)
                ggml_cuda_profiler_scope zone_graph(rt.backend_compute.get(), "eagle3/step_batch/graph");
#endif
                status = ggml_backend_graph_compute_async(rt.backend_compute.get(), graph.gf);
            }
            if (status == GGML_STATUS_SUCCESS) {
                {
#if defined(GGML_USE_CUDA)
                    ggml_cuda_profiler_scope zone_copy_out(rt.backend_compute.get(), "eagle3/step_batch/copy_outputs");
#endif
                    const int32_t required_len = past_len + 1;
                    for (int32_t ib = 0; ib < n_beams; ++ib) {
                        llama_eagle3_state * out_st = out_states[(size_t) ib];
                        if (!out_st) {
                            return false;
                        }

                        const bool can_reuse = out_st->dev &&
                                out_st->dev.use_count() == 1 &&
                                out_st->dev->t_hidden &&
                                out_st->dev->kv_capacity >= required_len + std::max(0, reserve_kv);

                        std::shared_ptr<llama_eagle3_state::device_state> next_dev = can_reuse ? out_st->dev : nullptr;
                        if (!next_dev) {
                            const int32_t cur_capacity = out_st->dev ? out_st->dev->kv_capacity : 0;
                            const int32_t desired_capacity = choose_kv_capacity(required_len, reserve_kv, cur_capacity);
                            if (!alloc_state_device(model, rt, desired_capacity, next_dev)) {
                                return false;
                            }
                        }

                        ggml_backend_tensor_copy_async(rt.backend_compute.get(), rt.backend_compute.get(), graph.t_hidden_out[(size_t) ib], next_dev->t_hidden);
                        if (next_dev->t_k && next_dev->t_v && graph.t_k_total[(size_t) ib] && graph.t_v_total[(size_t) ib]) {
                            if (!tensor_copy_3d_prefix_async(
                                        rt.backend_compute.get(),
                                        graph.t_k_total[(size_t) ib],
                                        next_dev->t_k,
                                        hp.head_dim,
                                        hp.num_kv_heads,
                                        required_len)) {
                                return false;
                            }
                            if (!tensor_copy_3d_prefix_async(
                                        rt.backend_compute.get(),
                                        graph.t_v_total[(size_t) ib],
                                        next_dev->t_v,
                                        hp.head_dim,
                                        hp.num_kv_heads,
                                        required_len)) {
                                return false;
                            }
                        }

                        next_dev->past_len = required_len;
                        out_st->dev = std::move(next_dev);
                        out_st->hidden.clear();
                        out_st->k.clear();
                        out_st->v.clear();
                        out_st->past_len = required_len;
                    }
                }
                return true;
            }
        }
    }

    for (size_t i = 0; i < parent_states.size(); ++i) {
        const llama_eagle3_state * parent = parent_states[i];
        llama_eagle3_state * out_st = out_states[i];
        if (!parent || !out_st) {
            return false;
        }
        llama_eagle3_state tmp = *parent;
        const float * hidden_step = tmp.hidden.empty() ? nullptr : tmp.hidden.data();
        if (!llama_eagle3_step(model, rt, tmp, hidden_step, hidden_in_dim, input_ids[i], nullptr, nullptr)) {
            return false;
        }
        *out_st = std::move(tmp);
    }

    return true;
}

bool llama_eagle3_step_from_hidden_capture(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        llama_eagle3_state & state,
        const std::vector<const ggml_tensor *> & hidden_capture,
        size_t token_idx,
        llama_token input_id,
        std::vector<float> * logits_out,
        llama_eagle3_step_debug * dbg) {
    const auto & hp = model.hparams;
    const int32_t hidden_in_dim = hp.hidden_concat * hp.target_hidden_size;
    if ((int32_t) hidden_capture.size() != hp.hidden_concat) {
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
            const int32_t required_len = state.past_len + 1;

            for (int32_t il = 0; il < hp.hidden_concat; ++il) {
                ggml_tensor * src = const_cast<ggml_tensor *>(hidden_capture[(size_t) il]);
                if (!src || src->type != GGML_TYPE_F32 || src->ne[0] != hp.target_hidden_size || token_idx >= (size_t) src->ne[1]) {
                    return false;
                }

                const size_t n_bytes = (size_t) hp.target_hidden_size * sizeof(float);
                const size_t src_offset = token_idx * (size_t) src->nb[1];
                const size_t dst_offset = (size_t) il * n_bytes;
                if (!tensor_copy_bytes_async(rt.target_backend ? rt.target_backend : rt.backend_compute.get(), rt.backend_compute.get(), src, src_offset, graph.t_hidden_in, dst_offset, n_bytes)) {
                    return false;
                }
            }

            ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_tok, &input_id, 0, sizeof(input_id));
            ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_pos, &state.past_len, 0, sizeof(state.past_len));

            if (graph.t_k_past_input && state.past_len > 0) {
                if (state.dev && state.dev->t_k && state.dev->kv_capacity >= state.past_len) {
                    if (!tensor_copy_3d_prefix_async(
                                rt.backend_compute.get(),
                                state.dev->t_k,
                                graph.t_k_past_input,
                                hp.head_dim,
                                hp.num_kv_heads,
                                state.past_len)) {
                        return false;
                    }
                } else if (state.k.size() * sizeof(float) >= past_bytes) {
                    ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_k_past_input, state.k.data(), 0, past_bytes);
                } else {
                    return false;
                }
            }
            if (graph.t_v_past_input && state.past_len > 0) {
                if (state.dev && state.dev->t_v && state.dev->kv_capacity >= state.past_len) {
                    if (!tensor_copy_3d_prefix_async(
                                rt.backend_compute.get(),
                                state.dev->t_v,
                                graph.t_v_past_input,
                                hp.head_dim,
                                hp.num_kv_heads,
                                state.past_len)) {
                        return false;
                    }
                } else if (state.v.size() * sizeof(float) >= past_bytes) {
                    ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_v_past_input, state.v.data(), 0, past_bytes);
                } else {
                    return false;
                }
            }

            const ggml_status status = ggml_backend_graph_compute_async(rt.backend_compute.get(), graph.gf);
            if (status == GGML_STATUS_SUCCESS) {
                const bool can_reuse = state.dev &&
                        state.dev.use_count() == 1 &&
                        state.dev->t_hidden &&
                        state.dev->kv_capacity >= required_len;

                std::shared_ptr<llama_eagle3_state::device_state> next_dev = can_reuse ? state.dev : nullptr;
                if (!next_dev) {
                    const int32_t cur_capacity = state.dev ? state.dev->kv_capacity : 0;
                    const int32_t desired_capacity = choose_kv_capacity(required_len, /* reserve_kv = */ 0, cur_capacity);
                    if (!alloc_state_device(model, rt, desired_capacity, next_dev)) {
                        return false;
                    }
                }

                ggml_backend_tensor_copy_async(rt.backend_compute.get(), rt.backend_compute.get(), graph.t_hidden_out, next_dev->t_hidden);
                if (next_dev->t_k && next_dev->t_v && graph.t_k_total && graph.t_v_total) {
                    if (!tensor_copy_3d_prefix_async(
                                rt.backend_compute.get(),
                                graph.t_k_total,
                                next_dev->t_k,
                                hp.head_dim,
                                hp.num_kv_heads,
                                required_len)) {
                        return false;
                    }
                    if (!tensor_copy_3d_prefix_async(
                                rt.backend_compute.get(),
                                graph.t_v_total,
                                next_dev->t_v,
                                hp.head_dim,
                                hp.num_kv_heads,
                                required_len)) {
                        return false;
                    }
                }

                next_dev->past_len = required_len;
                state.dev = std::move(next_dev);
                state.hidden.clear();
                state.k.clear();
                state.v.clear();

                if (with_logits && graph.t_logits) {
                    logits_out->resize(hp.draft_vocab_size);
                    ggml_backend_tensor_get_async(rt.backend_compute.get(), graph.t_logits, logits_out->data(), 0, hp.draft_vocab_size * sizeof(float));
                    ggml_backend_synchronize(rt.backend_compute.get());
                }

                state.past_len = required_len;
                return true;
            }
        }
    }

    std::vector<float> hidden_concat;
    if (!build_hidden_concat_from_capture_host(model, hidden_capture, token_idx, hidden_concat)) {
        return false;
    }

    return llama_eagle3_step(model, rt, state, hidden_concat.data(), hidden_in_dim, input_id, logits_out, dbg);
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
            const int32_t required_len = state.past_len + 1;

            if (hidden_in) {
                ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_hidden_in, hidden_in, 0, hidden_in_dim * sizeof(float));
            } else if (state.dev && state.dev->t_hidden) {
                ggml_backend_tensor_copy_async(rt.backend_compute.get(), rt.backend_compute.get(), state.dev->t_hidden, graph.t_hidden_in);
            } else {
                ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_hidden_in, state.hidden.data(), 0, hidden_in_dim * sizeof(float));
            }
            ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_tok, &input_id, 0, sizeof(input_id));
            ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_pos, &state.past_len, 0, sizeof(state.past_len));

            if (graph.t_k_past_input && state.past_len > 0) {
                if (state.dev && state.dev->t_k && state.dev->kv_capacity >= state.past_len) {
                    if (!tensor_copy_3d_prefix_async(
                                rt.backend_compute.get(),
                                state.dev->t_k,
                                graph.t_k_past_input,
                                hp.head_dim,
                                hp.num_kv_heads,
                                state.past_len)) {
                        return false;
                    }
                } else if (state.k.size() * sizeof(float) >= past_bytes) {
                    ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_k_past_input, state.k.data(), 0, past_bytes);
                } else {
                    return false;
                }
            }
            if (graph.t_v_past_input && state.past_len > 0) {
                if (state.dev && state.dev->t_v && state.dev->kv_capacity >= state.past_len) {
                    if (!tensor_copy_3d_prefix_async(
                                rt.backend_compute.get(),
                                state.dev->t_v,
                                graph.t_v_past_input,
                                hp.head_dim,
                                hp.num_kv_heads,
                                state.past_len)) {
                        return false;
                    }
                } else if (state.v.size() * sizeof(float) >= past_bytes) {
                    ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_v_past_input, state.v.data(), 0, past_bytes);
                } else {
                    return false;
                }
            }

            const ggml_status status = ggml_backend_graph_compute_async(rt.backend_compute.get(), graph.gf);
            if (status == GGML_STATUS_SUCCESS) {
                const bool can_reuse = state.dev &&
                        state.dev.use_count() == 1 &&
                        state.dev->t_hidden &&
                        state.dev->kv_capacity >= required_len;

                std::shared_ptr<llama_eagle3_state::device_state> next_dev = can_reuse ? state.dev : nullptr;
                if (!next_dev) {
                    const int32_t cur_capacity = state.dev ? state.dev->kv_capacity : 0;
                    const int32_t desired_capacity = choose_kv_capacity(required_len, /* reserve_kv = */ 0, cur_capacity);
                    if (!alloc_state_device(model, rt, desired_capacity, next_dev)) {
                        return false;
                    }
                }

                ggml_backend_tensor_copy_async(rt.backend_compute.get(), rt.backend_compute.get(), graph.t_hidden_out, next_dev->t_hidden);
                if (next_dev->t_k && next_dev->t_v && graph.t_k_total && graph.t_v_total) {
                    if (!tensor_copy_3d_prefix_async(
                                rt.backend_compute.get(),
                                graph.t_k_total,
                                next_dev->t_k,
                                hp.head_dim,
                                hp.num_kv_heads,
                                required_len)) {
                        return false;
                    }
                    if (!tensor_copy_3d_prefix_async(
                                rt.backend_compute.get(),
                                graph.t_v_total,
                                next_dev->t_v,
                                hp.head_dim,
                                hp.num_kv_heads,
                                required_len)) {
                        return false;
                    }
                }

                next_dev->past_len = required_len;
                state.dev = std::move(next_dev);
                state.hidden.clear();
                state.k.clear();
                state.v.clear();

                if (with_logits && graph.t_logits) {
                    logits_out->resize(hp.draft_vocab_size);
                    ggml_backend_tensor_get_async(rt.backend_compute.get(), graph.t_logits, logits_out->data(), 0, hp.draft_vocab_size * sizeof(float));
                    ggml_backend_synchronize(rt.backend_compute.get());
                }

                state.past_len = required_len;
                return true;
            }
        }
    }

    const auto & tensors = get_host_tensors(model);

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
