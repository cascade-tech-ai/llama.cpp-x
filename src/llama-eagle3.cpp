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
    int32_t                 rollout_kv_start = 0; // 0 = full KV from position 0; >0 = partial KV starting at this position
    std::shared_ptr<llama_eagle3_rollout_batch> rollout_batch;
    int32_t                 rollout_slot = -1;
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

void log_eagle_flash_attn_once(
        const llama_eagle3_runtime & rt,
        bool & logged_flag,
        const char * graph_name,
        const char * status) {
    if (logged_flag) {
        return;
    }

    logged_flag = true;

    const char * backend_name = rt.backend_compute ? ggml_backend_name(rt.backend_compute.get()) : "none";
    LLAMA_LOG_INFO("%s: eagle %s attention %s (backend = %s)\n", __func__, graph_name, status, backend_name);
}

ggml_tensor * build_eagle_attn_output(
        ggml_context * ctx,
        const llama_eagle3_hparams & hp,
        const llama_eagle3_runtime & rt,
        ggml_tensor * t_q,
        ggml_tensor * t_k_total,
        ggml_tensor * t_v_total,
        ggml_tensor * t_mask,
        const char * graph_name,
        bool & logged_flag) {
    ggml_tensor * qv = ggml_view_4d(ctx, t_q, t_q->ne[0], t_q->ne[1], t_q->ne[2], t_q->ne[3],
                                    t_q->nb[1], t_q->nb[2], t_q->nb[3], 0);
    ggml_tensor * kv = ggml_view_4d(ctx, t_k_total, t_k_total->ne[0], t_k_total->ne[1], t_k_total->ne[2], t_k_total->ne[3],
                                    t_k_total->nb[1], t_k_total->nb[2], t_k_total->nb[3], 0);
    ggml_tensor * vv = ggml_view_4d(ctx, t_v_total, t_v_total->ne[0], t_v_total->ne[1], t_v_total->ne[2], t_v_total->ne[3],
                                    t_v_total->nb[1], t_v_total->nb[2], t_v_total->nb[3], 0);

    qv = ggml_permute(ctx, qv, 0, 2, 1, 3);
    kv = ggml_permute(ctx, kv, 0, 2, 1, 3);
    vv = ggml_permute(ctx, vv, 0, 2, 1, 3);

    const float kq_scale = 1.0f / std::sqrt(float(hp.head_dim));

    if (rt.flash_attn && rt.backend_compute) {
        ggml_tensor * kv_fa = kv;
        ggml_tensor * vv_fa = vv;

        if (kv_fa->type == GGML_TYPE_F32) {
            kv_fa = ggml_cast(ctx, kv_fa, GGML_TYPE_F16);
        }

        if (vv_fa->type == GGML_TYPE_F32) {
            vv_fa = ggml_cast(ctx, vv_fa, GGML_TYPE_F16);
        }

        ggml_tensor * fattn = ggml_flash_attn_ext(ctx, qv, kv_fa, vv_fa, t_mask, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fattn, GGML_PREC_F32);

        if (ggml_backend_supports_op(rt.backend_compute.get(), fattn)) {
            log_eagle_flash_attn_once(rt, logged_flag, graph_name,
                    t_mask ? "using ggml_flash_attn_ext (with mask)" : "using ggml_flash_attn_ext");
            return ggml_reshape_2d(ctx, fattn, fattn->ne[0] * fattn->ne[1], fattn->ne[2] * fattn->ne[3]);
        }

        log_eagle_flash_attn_once(rt, logged_flag, graph_name, "requested ggml_flash_attn_ext but fell back to generic attention");
    } else {
        log_eagle_flash_attn_once(rt, logged_flag, graph_name, "using generic attention");
    }

    ggml_tensor * t_k_attn = t_k_total;
    ggml_tensor * t_v_attn = t_v_total;

    if (hp.num_kv_heads != hp.num_heads) {
        const int32_t n_rep = hp.num_heads / hp.num_kv_heads;
        const int64_t batch = t_k_attn->ne[3];
        const int64_t seq = t_k_attn->ne[2];
        const int64_t seq_total = seq * batch;

        ggml_tensor * k_flat = ggml_reshape_3d(ctx, ggml_cont(ctx, t_k_attn), hp.head_dim, hp.num_kv_heads, seq_total);
        ggml_tensor * v_flat = ggml_reshape_3d(ctx, ggml_cont(ctx, t_v_attn), hp.head_dim, hp.num_kv_heads, seq_total);

        ggml_tensor * k4 = ggml_reshape_4d(ctx, k_flat, hp.head_dim, hp.num_kv_heads, seq_total, 1);
        ggml_tensor * v4 = ggml_reshape_4d(ctx, v_flat, hp.head_dim, hp.num_kv_heads, seq_total, 1);

        k4 = ggml_permute(ctx, k4, 0, 2, 3, 1);
        v4 = ggml_permute(ctx, v4, 0, 2, 3, 1);

        k4 = ggml_repeat_4d(ctx, k4, hp.head_dim, n_rep, hp.num_kv_heads, seq_total);
        v4 = ggml_repeat_4d(ctx, v4, hp.head_dim, n_rep, hp.num_kv_heads, seq_total);

        k4 = ggml_cont(ctx, k4);
        v4 = ggml_cont(ctx, v4);

        k_flat = ggml_reshape_3d(ctx, k4, hp.head_dim, hp.num_heads, seq_total);
        v_flat = ggml_reshape_3d(ctx, v4, hp.head_dim, hp.num_heads, seq_total);

        t_k_attn = batch > 1
                ? ggml_reshape_4d(ctx, k_flat, hp.head_dim, hp.num_heads, seq, batch)
                : ggml_reshape_3d(ctx, k_flat, hp.head_dim, hp.num_heads, seq);
        t_v_attn = batch > 1
                ? ggml_reshape_4d(ctx, v_flat, hp.head_dim, hp.num_heads, seq, batch)
                : ggml_reshape_3d(ctx, v_flat, hp.head_dim, hp.num_heads, seq);
    }

    kv = ggml_view_4d(ctx, t_k_attn, t_k_attn->ne[0], t_k_attn->ne[1], t_k_attn->ne[2], t_k_attn->ne[3],
                      t_k_attn->nb[1], t_k_attn->nb[2], t_k_attn->nb[3], 0);
    vv = ggml_view_4d(ctx, t_v_attn, t_v_attn->ne[0], t_v_attn->ne[1], t_v_attn->ne[2], t_v_attn->ne[3],
                      t_v_attn->nb[1], t_v_attn->nb[2], t_v_attn->nb[3], 0);

    kv = ggml_permute(ctx, kv, 0, 2, 1, 3);
    vv = ggml_permute(ctx, vv, 0, 2, 1, 3);

    ggml_tensor * kq = ggml_mul_mat(ctx, kv, qv);
    kq = ggml_soft_max_ext(ctx, kq, t_mask, kq_scale, 0.0f);

    ggml_tensor * vv_t = ggml_cont(ctx, ggml_transpose(ctx, vv));
    ggml_tensor * kqv = ggml_mul_mat(ctx, vv_t, kq);
    ggml_tensor * attn_out = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    return ggml_cont_2d(ctx, attn_out, attn_out->ne[0]*attn_out->ne[1], attn_out->ne[2]*attn_out->ne[3]);
}

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

uint64_t make_step_multi_graph_key(int32_t past_len, int32_t hidden_in_dim, int32_t n_tokens, bool with_logits) {
    return (uint64_t(uint16_t(past_len & 0xffff)) << 48) |
           (uint64_t(uint16_t(hidden_in_dim & 0xffff)) << 32) |
           (uint64_t(uint16_t(n_tokens & 0xffff)) << 1) |
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

bool tensor_copy_3d_prefix_async(
        ggml_backend_t backend,
        ggml_tensor * src,
        ggml_tensor * dst,
        int64_t n0,
        int64_t n1,
        int32_t len);

bool tensor_copy_bytes_async(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        ggml_tensor * src,
        size_t src_offset,
        ggml_tensor * dst,
        size_t dst_offset,
        size_t size);

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

bool llama_eagle3_rollout_batch_ensure_impl(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t n_beams,
        int32_t kv_capacity,
        llama_eagle3_rollout_batch & batch) {
    if (!rt.buft_compute || n_beams <= 0 || kv_capacity < 0) {
        return false;
    }

    if (batch.ctx && batch.buf &&
        batch.n_beams == n_beams &&
        batch.kv_capacity == kv_capacity &&
        batch.t_hidden &&
        batch.t_k &&
        batch.t_v &&
        batch.t_mask) {
        return true;
    }

    const auto & hp = model.hparams;
    auto ctx = make_ctx_no_alloc(/* max_nodes = */ 32);
    if (!ctx) {
        return false;
    }

    ggml_tensor * t_hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hp.hidden_size, n_beams);
    ggml_tensor * t_k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, kv_capacity, n_beams);
    ggml_tensor * t_v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, kv_capacity, n_beams);
    ggml_tensor * t_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, kv_capacity, 1, 1, n_beams);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    if (!buf) {
        return false;
    }

    batch = {};
    batch.ctx = std::move(ctx);
    batch.buf = std::move(buf);
    batch.t_hidden = t_hidden;
    batch.t_k = t_k;
    batch.t_v = t_v;
    batch.t_mask = t_mask;
    batch.n_beams = n_beams;
    batch.kv_capacity = kv_capacity;
    return true;
}

bool llama_eagle3_rollout_batch_bind_slot_impl(
        const llama_eagle3_model & model,
        std::shared_ptr<llama_eagle3_rollout_batch> batch,
        int32_t slot,
        llama_eagle3_state & state,
        int32_t past_len) {
    if (!batch || slot < 0 || slot >= batch->n_beams) {
        return false;
    }

    const auto & hp = model.hparams;
    auto ctx = make_ctx_no_alloc(/* max_nodes = */ 16);
    if (!ctx) {
        return false;
    }

    ggml_tensor * t_hidden = ggml_view_2d(
            ctx.get(),
            batch->t_hidden,
            hp.hidden_size,
            1,
            batch->t_hidden->nb[1],
            (size_t) slot * batch->t_hidden->nb[1]);

    ggml_tensor * t_k = ggml_view_3d(
            ctx.get(),
            batch->t_k,
            hp.head_dim,
            hp.num_kv_heads,
            batch->kv_capacity,
            batch->t_k->nb[1],
            batch->t_k->nb[2],
            (size_t) slot * batch->t_k->nb[3]);

    ggml_tensor * t_v = ggml_view_3d(
            ctx.get(),
            batch->t_v,
            hp.head_dim,
            hp.num_kv_heads,
            batch->kv_capacity,
            batch->t_v->nb[1],
            batch->t_v->nb[2],
            (size_t) slot * batch->t_v->nb[3]);

    auto dev = std::make_shared<llama_eagle3_state::device_state>();
    dev->ctx = std::move(ctx);
    dev->t_hidden = t_hidden;
    dev->t_k = t_k;
    dev->t_v = t_v;
    dev->past_len = past_len;
    dev->kv_capacity = batch->kv_capacity;
    dev->rollout_batch = std::move(batch);
    dev->rollout_slot = slot;

    state.dev = std::move(dev);
    state.past_len = past_len;
    return true;
}

bool llama_eagle3_rollout_batch_copy_state_to_slot_impl(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const llama_eagle3_state & src,
        std::shared_ptr<llama_eagle3_rollout_batch> batch,
        int32_t slot,
        llama_eagle3_state & dst_state) {
    if (!batch || !rt.backend_compute || slot < 0 || slot >= batch->n_beams) {
        return false;
    }

    llama_eagle3_state dst;
    if (!llama_eagle3_rollout_batch_bind_slot_impl(model, batch, slot, dst, src.past_len)) {
        return false;
    }

    const auto & hp = model.hparams;
    const size_t hidden_bytes = (size_t) hp.hidden_size * sizeof(float);
    const size_t kv_bytes = (size_t) src.past_len * hp.head_dim * hp.num_kv_heads * sizeof(float);

    if (src.dev && src.dev->t_hidden) {
        if (!tensor_copy_bytes_async(
                    rt.backend_compute.get(),
                    rt.backend_compute.get(),
                    src.dev->t_hidden,
                    0,
                    dst.dev->t_hidden,
                    0,
                    hidden_bytes)) {
            return false;
        }
    } else if (!src.hidden.empty()) {
        ggml_backend_tensor_set_async(rt.backend_compute.get(), dst.dev->t_hidden, src.hidden.data(), 0, hidden_bytes);
    } else {
        return false;
    }

    if (src.past_len > 0) {
        if (src.dev && src.dev->t_k && src.dev->t_v && src.dev->kv_capacity >= src.past_len) {
            if (!tensor_copy_3d_prefix_async(
                        rt.backend_compute.get(),
                        src.dev->t_k,
                        dst.dev->t_k,
                        hp.head_dim,
                        hp.num_kv_heads,
                        src.past_len)) {
                return false;
            }
            if (!tensor_copy_3d_prefix_async(
                        rt.backend_compute.get(),
                        src.dev->t_v,
                        dst.dev->t_v,
                        hp.head_dim,
                        hp.num_kv_heads,
                        src.past_len)) {
                return false;
            }
        } else if (src.k.size() * sizeof(float) >= kv_bytes && src.v.size() * sizeof(float) >= kv_bytes) {
            ggml_backend_tensor_set_async(rt.backend_compute.get(), dst.dev->t_k, src.k.data(), 0, kv_bytes);
            ggml_backend_tensor_set_async(rt.backend_compute.get(), dst.dev->t_v, src.v.data(), 0, kv_bytes);
        } else {
            return false;
        }
    }

    dst_state = std::move(dst);
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

#if defined(GGML_USE_CUDA)
    if (ggml_backend_is_cuda(backend)) {
        return ggml_backend_cuda_tensor_copy_3d_prefix_async(backend, src, dst, len);
    }
#endif

    if (src_len == len && dst_len == len) {
        ggml_backend_tensor_copy_async(backend, backend, src, dst);
        return true;
    }

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

    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

#if defined(GGML_USE_CUDA)
    // Only use CUDA async copy when both tensors are actually on CUDA buffers.
    // The caller-provided backend hints may be wrong when the target model has
    // mixed-device layers (e.g. hybrid models with some layers on CPU).
    if (ggml_backend_is_cuda(backend_src) && ggml_backend_is_cuda(backend_dst) &&
        buf_src && !ggml_backend_buffer_is_host(buf_src) &&
        buf_dst && !ggml_backend_buffer_is_host(buf_dst)) {
        return ggml_backend_cuda_tensor_copy_bytes_between_async(backend_src, backend_dst, src, src_offset, dst, dst_offset, size);
    }
#endif
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
        float prob_threshold,
        llama_eagle3_select_batch_graph & graph) {
    if (graph.ctx && graph.buf_compute && graph.gf &&
        graph.t_hidden && graph.t_beam_logprob &&
        graph.t_selected_linear && graph.t_selected_draft && graph.t_selected_logprob &&
        graph.n_beams == n_beams && graph.k == k && graph.n_select == n_select &&
        graph.prob_threshold == prob_threshold &&
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

    ggml_tensor * t_topk_idx = ggml_top_k_threshold(ctx.get(), t_probs, k, prob_threshold);
    t_topk_idx = ggml_cont(ctx.get(), t_topk_idx);

    ggml_tensor * t_probs_zero = ggml_view_2d(ctx.get(), t_probs, 1, n_beams, t_probs->nb[1], 0);
    t_probs_zero = ggml_scale(ctx.get(), t_probs_zero, 0.0f);
    ggml_tensor * t_probs_ext = ggml_concat(ctx.get(), t_probs, t_probs_zero, 0);
    t_probs_ext = ggml_cont(ctx.get(), t_probs_ext);

    ggml_tensor * t_probs_flat = ggml_reshape_2d(ctx.get(), t_probs_ext, 1, (hp.draft_vocab_size + 1) * n_beams);

    ggml_tensor * t_beam = ggml_arange(ctx.get(), 0.0f, (float) n_beams, 1.0f);
    t_beam = ggml_scale(ctx.get(), t_beam, (float) (hp.draft_vocab_size + 1));
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

    ggml_tensor * t_total_rows = ggml_reshape_2d(ctx.get(), t_total, 1, k * n_beams);
    ggml_tensor * t_selected_logprob = ggml_get_rows(ctx.get(), t_total_rows, t_selected_linear_flat);
    t_selected_logprob = ggml_reshape_2d(ctx.get(), t_selected_logprob, n_select, 1);
    t_selected_logprob = ggml_cont(ctx.get(), t_selected_logprob);

    ggml_tensor * t_topk_idx_rows = ggml_reshape_2d(ctx.get(), t_topk_idx_f, 1, k * n_beams);
    ggml_tensor * t_selected_draft_f = ggml_get_rows(ctx.get(), t_topk_idx_rows, t_selected_linear_flat);
    t_selected_draft_f = ggml_reshape_2d(ctx.get(), t_selected_draft_f, n_select, 1);
    ggml_tensor * t_selected_draft = ggml_cast(ctx.get(), t_selected_draft_f, GGML_TYPE_I32);
    t_selected_draft = ggml_cont(ctx.get(), t_selected_draft);

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, t_selected_linear);
    ggml_build_forward_expand(gf, t_selected_draft);
    ggml_build_forward_expand(gf, t_selected_logprob);

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
    graph.n_select = n_select;
    graph.prob_threshold = prob_threshold;
    graph.t_hidden = t_hidden;
    graph.t_beam_logprob = t_beam_logprob;
    graph.t_selected_linear = t_selected_linear;
    graph.t_selected_draft = t_selected_draft;
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

    ggml_tensor * attn_out = build_eagle_attn_output(
            ctx.get(), hp, rt, t_q, t_k_total, t_v_total, nullptr, "step", rt.flash_attn_logged_step);

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

bool build_step_multi_graph(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t past_len,
        int32_t hidden_in_dim,
        int32_t n_tokens,
        bool with_logits,
        llama_eagle3_step_multi_graph & graph) {
    if (graph.ctx && graph.buf_compute && graph.gf && graph.n_tokens == n_tokens &&
        graph.t_hidden_in && graph.t_hidden_out && graph.t_k_total && graph.t_v_total) {
        return true;
    }

    if (!rt.tok_embd || n_tokens <= 0) {
        return false;
    }

    const auto & hp = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);

    auto ctx = make_ctx_no_alloc(/* max_nodes = */ (size_t) 2048 + (size_t) 32 * (size_t) n_tokens);
    if (!ctx) {
        return false;
    }

    ggml_tensor * t_hidden_in = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden_in_dim, n_tokens);
    ggml_set_input(t_hidden_in);

    ggml_tensor * t_tok = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_tokens);
    ggml_set_input(t_tok);

    ggml_tensor * t_pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_tokens);
    ggml_set_input(t_pos);

    ggml_tensor * t_embd = ggml_get_rows(ctx.get(), rt.tok_embd, t_tok);
    t_embd = ggml_cast(ctx.get(), t_embd, GGML_TYPE_F32);

    ggml_tensor * t_hidden = t_hidden_in;
    if (hidden_in_dim != hp.hidden_size) {
        t_hidden = ggml_mul_mat(ctx.get(), tensors.fc_w, t_hidden_in);
    }

    ggml_tensor * t_hidden_norm = ggml_rms_norm(ctx.get(), t_hidden, hp.rms_norm_eps);
    ggml_tensor * t_hidden_norm_w = ggml_cast(ctx.get(), tensors.hidden_norm_w, GGML_TYPE_F32);
    t_hidden_norm_w = ggml_repeat(ctx.get(), t_hidden_norm_w, t_hidden_norm);
    t_hidden_norm = ggml_mul(ctx.get(), t_hidden_norm, t_hidden_norm_w);

    ggml_tensor * t_embd_norm = ggml_rms_norm(ctx.get(), t_embd, hp.rms_norm_eps);
    ggml_tensor * t_input_norm_w = ggml_cast(ctx.get(), tensors.input_norm_w, GGML_TYPE_F32);
    t_input_norm_w = ggml_repeat(ctx.get(), t_input_norm_w, t_embd_norm);
    t_embd_norm = ggml_mul(ctx.get(), t_embd_norm, t_input_norm_w);

    ggml_tensor * t_cat = ggml_concat(ctx.get(), t_embd_norm, t_hidden_norm, 0);

    ggml_tensor * t_q = ggml_mul_mat(ctx.get(), tensors.attn_q_w, t_cat);
    if (tensors.attn_q_b) {
        ggml_tensor * t_q_b = ggml_cast(ctx.get(), tensors.attn_q_b, GGML_TYPE_F32);
        t_q_b = ggml_repeat(ctx.get(), t_q_b, t_q);
        t_q = ggml_add(ctx.get(), t_q, t_q_b);
    }

    ggml_tensor * t_k = ggml_mul_mat(ctx.get(), tensors.attn_k_w, t_cat);
    if (tensors.attn_k_b) {
        ggml_tensor * t_k_b = ggml_cast(ctx.get(), tensors.attn_k_b, GGML_TYPE_F32);
        t_k_b = ggml_repeat(ctx.get(), t_k_b, t_k);
        t_k = ggml_add(ctx.get(), t_k, t_k_b);
    }

    ggml_tensor * t_v = ggml_mul_mat(ctx.get(), tensors.attn_v_w, t_cat);
    if (tensors.attn_v_b) {
        ggml_tensor * t_v_b = ggml_cast(ctx.get(), tensors.attn_v_b, GGML_TYPE_F32);
        t_v_b = ggml_repeat(ctx.get(), t_v_b, t_v);
        t_v = ggml_add(ctx.get(), t_v, t_v_b);
    }

    t_q = ggml_reshape_3d(ctx.get(), t_q, hp.head_dim, hp.num_heads, n_tokens);
    t_k = ggml_reshape_3d(ctx.get(), t_k, hp.head_dim, hp.num_kv_heads, n_tokens);
    t_v = ggml_reshape_3d(ctx.get(), t_v, hp.head_dim, hp.num_kv_heads, n_tokens);

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

    ggml_tensor * t_mask = nullptr;
    if (n_tokens > 1) {
        t_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, past_len + n_tokens, n_tokens, 1, 1);
    }

    ggml_tensor * t_attn_out = build_eagle_attn_output(
            ctx.get(), hp, rt, t_q, t_k_total, t_v_total, t_mask, "step_multi", rt.flash_attn_logged_step);

    ggml_tensor * t_attn = ggml_mul_mat(ctx.get(), tensors.attn_o_w, t_attn_out);
    if (tensors.attn_o_b) {
        ggml_tensor * t_attn_b = ggml_cast(ctx.get(), tensors.attn_o_b, GGML_TYPE_F32);
        t_attn_b = ggml_repeat(ctx.get(), t_attn_b, t_attn);
        t_attn = ggml_add(ctx.get(), t_attn, t_attn_b);
    }

    ggml_tensor * t_resid = hp.norm_before_residual ? t_hidden_norm : t_hidden;
    ggml_tensor * t_hidden_attn = ggml_add(ctx.get(), t_attn, t_resid);

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

    ggml_tensor * t_hidden_out = ggml_add(ctx.get(), t_hidden_attn, t_ffn);
    t_hidden_out = ggml_cont(ctx.get(), t_hidden_out);
    ggml_tensor * t_k_total_out = ggml_cont(ctx.get(), t_k_total);
    ggml_tensor * t_v_total_out = ggml_cont(ctx.get(), t_v_total);

    ggml_tensor * t_logits = nullptr;
    if (with_logits) {
        ggml_tensor * t_hidden_last = ggml_view_2d(
                ctx.get(), t_hidden_out, hp.hidden_size, 1, t_hidden_out->nb[1], (size_t) (n_tokens - 1) * t_hidden_out->nb[1]);
        t_hidden_last = ggml_cont(ctx.get(), t_hidden_last);
        ggml_tensor * t_norm = ggml_rms_norm(ctx.get(), t_hidden_last, hp.rms_norm_eps);
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

    if (t_mask) {
        std::vector<ggml_fp16_t> mask((size_t) (past_len + n_tokens) * (size_t) n_tokens, ggml_fp32_to_fp16(0.0f));
        for (int32_t iq = 0; iq < n_tokens; ++iq) {
            const int32_t max_key = past_len + iq;
            for (int32_t ik = max_key + 1; ik < past_len + n_tokens; ++ik) {
                mask[(size_t) iq * (size_t) (past_len + n_tokens) + (size_t) ik] = ggml_fp32_to_fp16(-INFINITY);
            }
        }
        ggml_backend_tensor_set(t_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    graph.ctx = std::move(ctx);
    graph.buf_compute = std::move(buf_compute);
    graph.gf = gf;
    graph.n_tokens = n_tokens;
    graph.t_hidden_in = t_hidden_in;
    graph.t_tok = t_tok;
    graph.t_pos = t_pos;
    graph.t_k_past_input = t_k_past_input;
    graph.t_v_past_input = t_v_past_input;
    graph.t_hidden_out = t_hidden_out;
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

    // Apply RoPE while positions still live on the sequence axis, then reinterpret beams as a true
    // batch axis for one batched attention call.
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

    ggml_tensor * t_q_b = ggml_reshape_4d(ctx.get(), t_q, hp.head_dim, hp.num_heads, 1, n_beams);
    ggml_tensor * t_k_b = ggml_reshape_4d(ctx.get(), t_k, hp.head_dim, hp.num_kv_heads, 1, n_beams);
    ggml_tensor * t_v_b = ggml_reshape_4d(ctx.get(), t_v, hp.head_dim, hp.num_kv_heads, 1, n_beams);

    std::vector<ggml_tensor *> t_k_past_in((size_t) n_beams, nullptr);
    std::vector<ggml_tensor *> t_v_past_in((size_t) n_beams, nullptr);
    ggml_tensor * t_k_past_b = nullptr;
    ggml_tensor * t_v_past_b = nullptr;
    if (past_len > 0) {
        t_k_past_b = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, past_len, n_beams);
        t_v_past_b = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, past_len, n_beams);
        ggml_set_input(t_k_past_b);
        ggml_set_input(t_v_past_b);

        for (int32_t ib = 0; ib < n_beams; ++ib) {
            t_k_past_in[(size_t) ib] = ggml_view_3d(
                    ctx.get(),
                    t_k_past_b,
                    hp.head_dim,
                    hp.num_kv_heads,
                    past_len,
                    t_k_past_b->nb[1],
                    t_k_past_b->nb[2],
                    (size_t) ib * t_k_past_b->nb[3]);
            t_v_past_in[(size_t) ib] = ggml_view_3d(
                    ctx.get(),
                    t_v_past_b,
                    hp.head_dim,
                    hp.num_kv_heads,
                    past_len,
                    t_v_past_b->nb[1],
                    t_v_past_b->nb[2],
                    (size_t) ib * t_v_past_b->nb[3]);
        }
    }

    ggml_tensor * t_k_total_b = past_len > 0 ? ggml_concat(ctx.get(), t_k_past_b, t_k_b, 2) : t_k_b;
    ggml_tensor * t_v_total_b = past_len > 0 ? ggml_concat(ctx.get(), t_v_past_b, t_v_b, 2) : t_v_b;

    ggml_tensor * t_attn_out_b = build_eagle_attn_output(
            ctx.get(), hp, rt, t_q_b, t_k_total_b, t_v_total_b, nullptr, "step_batch", rt.flash_attn_logged_step_batch);
    t_attn_out_b = ggml_cont(ctx.get(), t_attn_out_b); // [hidden, n_beams]
    t_k_total_b = ggml_cont(ctx.get(), t_k_total_b);
    t_v_total_b = ggml_cont(ctx.get(), t_v_total_b);

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
    std::vector<ggml_tensor *> t_k_total_out;
    std::vector<ggml_tensor *> t_v_total_out;
    t_hidden_out_views.reserve((size_t) n_beams);
    t_k_total_out.reserve((size_t) n_beams);
    t_v_total_out.reserve((size_t) n_beams);
    for (int32_t ib = 0; ib < n_beams; ++ib) {
        t_hidden_out_views.push_back(ggml_view_2d(
                ctx.get(), t_hidden_out_b, hp.hidden_size, 1, t_hidden_out_b->nb[1], (size_t) ib * t_hidden_out_b->nb[1]));
        t_k_total_out.push_back(ggml_view_3d(
                ctx.get(),
                t_k_total_b,
                hp.head_dim,
                hp.num_kv_heads,
                past_len + 1,
                t_k_total_b->nb[1],
                t_k_total_b->nb[2],
                (size_t) ib * t_k_total_b->nb[3]));
        t_v_total_out.push_back(ggml_view_3d(
                ctx.get(),
                t_v_total_b,
                hp.head_dim,
                hp.num_kv_heads,
                past_len + 1,
                t_v_total_b->nb[1],
                t_v_total_b->nb[2],
                (size_t) ib * t_v_total_b->nb[3]));
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
    graph.t_k_past_input_b = t_k_past_b;
    graph.t_v_past_input_b = t_v_past_b;
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

// ---------------------------------------------------------------------------
// Optimized prefill: KV-only bulk pass + single root pass, cached on runtime
// ---------------------------------------------------------------------------

struct prefill_kv_cache {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor *           t_k = nullptr; // [head_dim, n_kv_heads, capacity]
    ggml_tensor *           t_v = nullptr; // [head_dim, n_kv_heads, capacity]
    int32_t                 capacity = 0;
};

// KV-only graph: K/V projections + RoPE(K) + set_rows.  Per-layer hidden inputs
// to allow bulk copies (2 copies instead of 2*n_tokens).
struct kv_only_graph {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf_compute;
    ggml_cgraph *           gf = nullptr;
    int32_t                 n_tokens = 0;

    std::vector<ggml_tensor *> t_hidden_layers; // [hidden_concat], each [target_hidden_size, n_tokens]
    ggml_tensor * t_tok    = nullptr; // [n_tokens]
    ggml_tensor * t_pos    = nullptr; // [n_tokens]
    ggml_tensor * t_k_idxs = nullptr; // [n_tokens]
    ggml_tensor * t_v_idxs = nullptr; // [n_tokens]
};

// Full single-token root graph: produces hidden output for speculative rollout.
// Also uses per-layer hidden inputs.
struct root_graph {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf_compute;
    ggml_cgraph *           gf = nullptr;
    int32_t                 kv_capacity = 0;

    std::vector<ggml_tensor *> t_hidden_layers; // [hidden_concat], each [target_hidden_size, 1]
    ggml_tensor * t_tok        = nullptr; // [1]
    ggml_tensor * t_pos        = nullptr; // [1]
    ggml_tensor * t_k_idxs     = nullptr; // [1]
    ggml_tensor * t_v_idxs     = nullptr; // [1]
    ggml_tensor * t_mask       = nullptr; // [kv_capacity, 1, 1, 1]
    ggml_tensor * t_hidden_out = nullptr; // [hidden_size, 1]
};

bool alloc_prefill_cache(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t capacity,
        prefill_kv_cache & cache) {
    if (!rt.buft_compute || capacity <= 0) {
        return false;
    }
    if (cache.capacity >= capacity && cache.t_k && cache.t_v) {
        return true; // reuse existing
    }
    const auto & hp = model.hparams;
    auto ctx = make_ctx_no_alloc(8);
    if (!ctx) {
        return false;
    }
    ggml_tensor * t_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, capacity);
    ggml_tensor * t_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hp.head_dim, hp.num_kv_heads, capacity);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    if (!buf) {
        return false;
    }
    cache = {};
    cache.ctx      = std::move(ctx);
    cache.buf      = std::move(buf);
    cache.t_k      = t_k;
    cache.t_v      = t_v;
    cache.capacity = capacity;
    return true;
}

// Helper: build the common prefix of KV-only and root graphs (embedding + norms + concat).
// Returns t_cat. Populates per-layer hidden input tensors in `hidden_layers_out`.
ggml_tensor * build_prefill_common(
        ggml_context * ctx,
        const llama_eagle3_hparams & hp,
        const llama_eagle3_runtime & rt,
        const llama_eagle3_tensors & tensors,
        int32_t n_tokens,
        std::vector<ggml_tensor *> & hidden_layers_out,
        ggml_tensor *& t_tok_out,
        ggml_tensor *& t_pos_out,
        ggml_tensor *& t_k_idxs_out,
        ggml_tensor *& t_v_idxs_out) {

    // Per-layer hidden inputs (bulk-copyable from capture tensors)
    hidden_layers_out.resize((size_t) hp.hidden_concat);
    for (int32_t il = 0; il < hp.hidden_concat; ++il) {
        hidden_layers_out[(size_t) il] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.target_hidden_size, n_tokens);
        ggml_set_input(hidden_layers_out[(size_t) il]);
    }

    t_tok_out = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(t_tok_out);
    t_pos_out = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(t_pos_out);
    t_k_idxs_out = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(t_k_idxs_out);
    t_v_idxs_out = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(t_v_idxs_out);

    // Embedding
    ggml_tensor * t_embd = ggml_get_rows(ctx, rt.tok_embd, t_tok_out);
    t_embd = ggml_cast(ctx, t_embd, GGML_TYPE_F32);

    // Concat per-layer hidden inputs, then FC project
    ggml_tensor * t_hidden_in = hidden_layers_out[0];
    for (int32_t il = 1; il < hp.hidden_concat; ++il) {
        t_hidden_in = ggml_concat(ctx, t_hidden_in, hidden_layers_out[(size_t) il], 0);
    }

    ggml_tensor * t_hidden = t_hidden_in;
    const int32_t hidden_in_dim = hp.hidden_concat * hp.target_hidden_size;
    if (hidden_in_dim != hp.hidden_size) {
        t_hidden = ggml_mul_mat(ctx, tensors.fc_w, t_hidden_in);
    }

    // Hidden norm
    ggml_tensor * t_hidden_norm = ggml_rms_norm(ctx, t_hidden, hp.rms_norm_eps);
    ggml_tensor * t_hidden_norm_w = ggml_cast(ctx, tensors.hidden_norm_w, GGML_TYPE_F32);
    t_hidden_norm_w = ggml_repeat(ctx, t_hidden_norm_w, t_hidden_norm);
    t_hidden_norm = ggml_mul(ctx, t_hidden_norm, t_hidden_norm_w);

    // Embedding norm
    ggml_tensor * t_embd_norm = ggml_rms_norm(ctx, t_embd, hp.rms_norm_eps);
    ggml_tensor * t_input_norm_w = ggml_cast(ctx, tensors.input_norm_w, GGML_TYPE_F32);
    t_input_norm_w = ggml_repeat(ctx, t_input_norm_w, t_embd_norm);
    t_embd_norm = ggml_mul(ctx, t_embd_norm, t_input_norm_w);

    return ggml_concat(ctx, t_embd_norm, t_hidden_norm, 0);
}

bool build_kv_only_graph_impl(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const prefill_kv_cache & cache,
        int32_t n_tokens,
        kv_only_graph & graph) {
    if (!rt.tok_embd || n_tokens <= 0 || cache.capacity <= 0) {
        return false;
    }
    const auto & hp      = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);
    auto ctx = make_ctx_no_alloc((size_t) 512 + (size_t) 16 * (size_t) n_tokens);
    if (!ctx) {
        return false;
    }

    std::vector<ggml_tensor *> hidden_layers;
    ggml_tensor * t_tok = nullptr, * t_pos = nullptr, * t_k_idxs = nullptr, * t_v_idxs = nullptr;
    ggml_tensor * t_cat = build_prefill_common(ctx.get(), hp, rt, tensors, n_tokens,
            hidden_layers, t_tok, t_pos, t_k_idxs, t_v_idxs);

    // K and V projections only
    ggml_tensor * t_k = ggml_mul_mat(ctx.get(), tensors.attn_k_w, t_cat);
    if (tensors.attn_k_b) {
        ggml_tensor * t_k_b = ggml_cast(ctx.get(), tensors.attn_k_b, GGML_TYPE_F32);
        t_k_b = ggml_repeat(ctx.get(), t_k_b, t_k);
        t_k = ggml_add(ctx.get(), t_k, t_k_b);
    }
    ggml_tensor * t_v = ggml_mul_mat(ctx.get(), tensors.attn_v_w, t_cat);
    if (tensors.attn_v_b) {
        ggml_tensor * t_v_b = ggml_cast(ctx.get(), tensors.attn_v_b, GGML_TYPE_F32);
        t_v_b = ggml_repeat(ctx.get(), t_v_b, t_v);
        t_v = ggml_add(ctx.get(), t_v, t_v_b);
    }

    t_k = ggml_reshape_3d(ctx.get(), t_k, hp.head_dim, hp.num_kv_heads, n_tokens);
    t_v = ggml_reshape_3d(ctx.get(), t_v, hp.head_dim, hp.num_kv_heads, n_tokens);

    // RoPE on K only
    t_k = ggml_rope_ext(ctx.get(), t_k, t_pos, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);

    // set_rows into cache
    ggml_tensor * t_k_flat = ggml_view_2d(ctx.get(), t_k, hp.head_dim * hp.num_kv_heads, n_tokens, t_k->nb[2], 0);
    ggml_tensor * t_v_flat = ggml_view_2d(ctx.get(), t_v, hp.head_dim * hp.num_kv_heads, n_tokens, t_v->nb[2], 0);
    ggml_tensor * t_k_c2d = ggml_reshape_2d(ctx.get(), cache.t_k, hp.head_dim * hp.num_kv_heads, cache.capacity);
    ggml_tensor * t_v_c2d = ggml_reshape_2d(ctx.get(), cache.t_v, hp.head_dim * hp.num_kv_heads, cache.capacity);
    ggml_tensor * t_k_w = ggml_set_rows(ctx.get(), t_k_c2d, t_k_flat, t_k_idxs);
    ggml_tensor * t_v_w = ggml_set_rows(ctx.get(), t_v_c2d, t_v_flat, t_v_idxs);

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, t_k_w);
    ggml_build_forward_expand(gf, t_v_w);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    if (!buf) {
        return false;
    }

    graph = {};
    graph.ctx            = std::move(ctx);
    graph.buf_compute    = std::move(buf);
    graph.gf             = gf;
    graph.n_tokens       = n_tokens;
    graph.t_hidden_layers = std::move(hidden_layers);
    graph.t_tok          = t_tok;
    graph.t_pos          = t_pos;
    graph.t_k_idxs       = t_k_idxs;
    graph.t_v_idxs       = t_v_idxs;
    return true;
}

bool build_root_graph_impl(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const prefill_kv_cache & cache,
        root_graph & graph) {
    if (!rt.tok_embd || cache.capacity <= 0) {
        return false;
    }
    const auto & hp      = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);
    auto ctx = make_ctx_no_alloc(2048);
    if (!ctx) {
        return false;
    }

    std::vector<ggml_tensor *> hidden_layers;
    ggml_tensor * t_tok = nullptr, * t_pos = nullptr, * t_k_idxs = nullptr, * t_v_idxs = nullptr;
    ggml_tensor * t_cat = build_prefill_common(ctx.get(), hp, rt, tensors, 1,
            hidden_layers, t_tok, t_pos, t_k_idxs, t_v_idxs);

    // Full Q, K, V
    ggml_tensor * t_q = ggml_mul_mat(ctx.get(), tensors.attn_q_w, t_cat);
    if (tensors.attn_q_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_q_b, GGML_TYPE_F32);
        t_q = ggml_add(ctx.get(), t_q, b);
    }
    ggml_tensor * t_k = ggml_mul_mat(ctx.get(), tensors.attn_k_w, t_cat);
    if (tensors.attn_k_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_k_b, GGML_TYPE_F32);
        t_k = ggml_add(ctx.get(), t_k, b);
    }
    ggml_tensor * t_v = ggml_mul_mat(ctx.get(), tensors.attn_v_w, t_cat);
    if (tensors.attn_v_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_v_b, GGML_TYPE_F32);
        t_v = ggml_add(ctx.get(), t_v, b);
    }

    t_q = ggml_reshape_3d(ctx.get(), t_q, hp.head_dim, hp.num_heads, 1);
    t_k = ggml_reshape_3d(ctx.get(), t_k, hp.head_dim, hp.num_kv_heads, 1);
    t_v = ggml_reshape_3d(ctx.get(), t_v, hp.head_dim, hp.num_kv_heads, 1);

    // RoPE on Q and K
    t_q = ggml_rope_ext(ctx.get(), t_q, t_pos, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);
    t_k = ggml_rope_ext(ctx.get(), t_k, t_pos, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);

    // Write root K/V into cache
    ggml_tensor * t_k_flat = ggml_view_2d(ctx.get(), t_k, hp.head_dim * hp.num_kv_heads, 1, t_k->nb[2], 0);
    ggml_tensor * t_v_flat = ggml_view_2d(ctx.get(), t_v, hp.head_dim * hp.num_kv_heads, 1, t_v->nb[2], 0);
    ggml_tensor * t_k_c2d = ggml_reshape_2d(ctx.get(), cache.t_k, hp.head_dim * hp.num_kv_heads, cache.capacity);
    ggml_tensor * t_v_c2d = ggml_reshape_2d(ctx.get(), cache.t_v, hp.head_dim * hp.num_kv_heads, cache.capacity);
    ggml_tensor * t_k_w = ggml_set_rows(ctx.get(), t_k_c2d, t_k_flat, t_k_idxs);
    ggml_tensor * t_v_w = ggml_set_rows(ctx.get(), t_v_c2d, t_v_flat, t_v_idxs);

    // Attention over full KV cache
    ggml_tensor * t_k_all = ggml_reshape_3d(ctx.get(), t_k_w, hp.head_dim, hp.num_kv_heads, cache.capacity);
    ggml_tensor * t_v_all = ggml_reshape_3d(ctx.get(), t_v_w, hp.head_dim, hp.num_kv_heads, cache.capacity);

    ggml_tensor * t_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, cache.capacity, 1, 1, 1);
    ggml_set_input(t_mask);

    ggml_tensor * t_attn_out = build_eagle_attn_output(
            ctx.get(), hp, rt, t_q, t_k_all, t_v_all, t_mask,
            "prefill_root", rt.flash_attn_logged_step);

    // O projection + residual
    ggml_tensor * t_attn = ggml_mul_mat(ctx.get(), tensors.attn_o_w, t_attn_out);
    if (tensors.attn_o_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_o_b, GGML_TYPE_F32);
        t_attn = ggml_add(ctx.get(), t_attn, b);
    }

    // Need t_hidden for residual — reconstruct from concat inputs
    ggml_tensor * t_hidden_in = hidden_layers[0];
    for (int32_t il = 1; il < hp.hidden_concat; ++il) {
        t_hidden_in = ggml_concat(ctx.get(), t_hidden_in, hidden_layers[(size_t) il], 0);
    }
    ggml_tensor * t_hidden = t_hidden_in;
    const int32_t hid = hp.hidden_concat * hp.target_hidden_size;
    if (hid != hp.hidden_size) {
        t_hidden = ggml_mul_mat(ctx.get(), tensors.fc_w, t_hidden_in);
    }

    ggml_tensor * t_resid = t_hidden;
    if (hp.norm_before_residual) {
        t_resid = ggml_rms_norm(ctx.get(), t_hidden, hp.rms_norm_eps);
        ggml_tensor * w = ggml_cast(ctx.get(), tensors.hidden_norm_w, GGML_TYPE_F32);
        t_resid = ggml_mul(ctx.get(), t_resid, w);
    }
    ggml_tensor * t_hidden_attn = ggml_add(ctx.get(), t_attn, t_resid);

    // Post-norm + FFN
    ggml_tensor * t_post = ggml_rms_norm(ctx.get(), t_hidden_attn, hp.rms_norm_eps);
    ggml_tensor * t_post_w = ggml_cast(ctx.get(), tensors.post_norm_w, GGML_TYPE_F32);
    t_post = ggml_mul(ctx.get(), t_post, t_post_w);

    ggml_tensor * t_gate = ggml_mul_mat(ctx.get(), tensors.ffn_gate_w, t_post);
    if (tensors.ffn_gate_b) { t_gate = ggml_add(ctx.get(), t_gate, ggml_cast(ctx.get(), tensors.ffn_gate_b, GGML_TYPE_F32)); }
    ggml_tensor * t_up = ggml_mul_mat(ctx.get(), tensors.ffn_up_w, t_post);
    if (tensors.ffn_up_b) { t_up = ggml_add(ctx.get(), t_up, ggml_cast(ctx.get(), tensors.ffn_up_b, GGML_TYPE_F32)); }
    ggml_tensor * t_ffn = ggml_mul_mat(ctx.get(), tensors.ffn_down_w,
            ggml_mul(ctx.get(), ggml_silu(ctx.get(), t_gate), t_up));
    if (tensors.ffn_down_b) { t_ffn = ggml_add(ctx.get(), t_ffn, ggml_cast(ctx.get(), tensors.ffn_down_b, GGML_TYPE_F32)); }

    ggml_tensor * t_hidden_out = ggml_cont(ctx.get(), ggml_add(ctx.get(), t_hidden_attn, t_ffn));

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, t_hidden_out);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    if (!buf) {
        return false;
    }

    graph = {};
    graph.ctx             = std::move(ctx);
    graph.buf_compute     = std::move(buf);
    graph.gf              = gf;
    graph.kv_capacity     = cache.capacity;
    graph.t_hidden_layers = std::move(hidden_layers);
    graph.t_tok           = t_tok;
    graph.t_pos           = t_pos;
    graph.t_k_idxs        = t_k_idxs;
    graph.t_v_idxs        = t_v_idxs;
    graph.t_mask          = t_mask;
    graph.t_hidden_out    = t_hidden_out;
    return true;
}

// ---------------------------------------------------------------------------
// Rollout step graph: fixed-capacity KV with set_rows + mask, cached on runtime
// ---------------------------------------------------------------------------

struct rollout_step_graph {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf_compute;
    ggml_cgraph *           gf = nullptr;
    int32_t                 n_beams = 0;
    int32_t                 kv_capacity = 0;

    // Batched inputs
    ggml_tensor * t_hidden_in_b = nullptr; // [hidden_in_dim, n_beams]
    ggml_tensor * t_tok_b       = nullptr; // [n_beams]
    ggml_tensor * t_pos_b       = nullptr; // [n_beams]
    ggml_tensor * t_k_idxs_b    = nullptr; // [n_beams]  (4D index = beam*kv_capacity + pos)
    ggml_tensor * t_v_idxs_b    = nullptr; // [n_beams]
    ggml_tensor * t_mask        = nullptr; // [kv_capacity, 1, 1, 1]

    // External KV (from rollout batch, not owned by this graph)
    // These are set during graph build and reference the batch's KV tensors.

    // Outputs
    ggml_tensor * t_hidden_out_b = nullptr; // [hidden_size, n_beams]
};

bool build_rollout_step_graph_impl(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const llama_eagle3_rollout_batch & batch,
        int32_t hidden_in_dim,
        rollout_step_graph & graph) {
    if (!rt.tok_embd || batch.n_beams <= 0 || batch.kv_capacity <= 0) {
        return false;
    }

    const auto & hp      = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);
    const int32_t n_beams = batch.n_beams;
    const int32_t kv_cap  = batch.kv_capacity;
    auto ctx = make_ctx_no_alloc((size_t) 3072 * (size_t) n_beams + 1024);
    if (!ctx) {
        return false;
    }

    // -- inputs ---------------------------------------------------------------
    ggml_tensor * t_hidden_in_b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden_in_dim, n_beams);
    ggml_set_input(t_hidden_in_b);

    ggml_tensor * t_tok_b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_beams);
    ggml_set_input(t_tok_b);

    ggml_tensor * t_pos_b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_beams);
    ggml_set_input(t_pos_b);

    ggml_tensor * t_k_idxs_b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_beams);
    ggml_set_input(t_k_idxs_b);

    ggml_tensor * t_v_idxs_b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_beams);
    ggml_set_input(t_v_idxs_b);

    ggml_tensor * t_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, kv_cap, 1, 1, 1);
    ggml_set_input(t_mask);

    // -- embedding + norms ----------------------------------------------------
    ggml_tensor * t_embd = ggml_get_rows(ctx.get(), rt.tok_embd, t_tok_b);
    t_embd = ggml_cast(ctx.get(), t_embd, GGML_TYPE_F32);

    ggml_tensor * t_hidden = t_hidden_in_b;
    if (hidden_in_dim != hp.hidden_size) {
        t_hidden = ggml_mul_mat(ctx.get(), tensors.fc_w, t_hidden_in_b);
    }

    ggml_tensor * t_hidden_norm = ggml_rms_norm(ctx.get(), t_hidden, hp.rms_norm_eps);
    ggml_tensor * t_hnw = ggml_cast(ctx.get(), tensors.hidden_norm_w, GGML_TYPE_F32);
    t_hnw = ggml_repeat(ctx.get(), t_hnw, t_hidden_norm);
    t_hidden_norm = ggml_mul(ctx.get(), t_hidden_norm, t_hnw);

    ggml_tensor * t_embd_norm = ggml_rms_norm(ctx.get(), t_embd, hp.rms_norm_eps);
    ggml_tensor * t_inw = ggml_cast(ctx.get(), tensors.input_norm_w, GGML_TYPE_F32);
    t_inw = ggml_repeat(ctx.get(), t_inw, t_embd_norm);
    t_embd_norm = ggml_mul(ctx.get(), t_embd_norm, t_inw);

    ggml_tensor * t_cat = ggml_concat(ctx.get(), t_embd_norm, t_hidden_norm, 0);

    // -- Q, K, V projections --------------------------------------------------
    ggml_tensor * t_q = ggml_mul_mat(ctx.get(), tensors.attn_q_w, t_cat);
    if (tensors.attn_q_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_q_b, GGML_TYPE_F32);
        b = ggml_repeat(ctx.get(), b, t_q);
        t_q = ggml_add(ctx.get(), t_q, b);
    }
    ggml_tensor * t_k = ggml_mul_mat(ctx.get(), tensors.attn_k_w, t_cat);
    if (tensors.attn_k_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_k_b, GGML_TYPE_F32);
        b = ggml_repeat(ctx.get(), b, t_k);
        t_k = ggml_add(ctx.get(), t_k, b);
    }
    ggml_tensor * t_v = ggml_mul_mat(ctx.get(), tensors.attn_v_w, t_cat);
    if (tensors.attn_v_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_v_b, GGML_TYPE_F32);
        b = ggml_repeat(ctx.get(), b, t_v);
        t_v = ggml_add(ctx.get(), t_v, b);
    }

    // Reshape: [proj_dim, n_beams] → [head_dim, n_heads/n_kv_heads, n_beams]
    // RoPE expects ne[2] = n_seq (= n_beams here, one token per beam) and pos [n_seq]
    t_q = ggml_reshape_3d(ctx.get(), t_q, hp.head_dim, hp.num_heads,    n_beams);
    t_k = ggml_reshape_3d(ctx.get(), t_k, hp.head_dim, hp.num_kv_heads, n_beams);
    t_v = ggml_reshape_3d(ctx.get(), t_v, hp.head_dim, hp.num_kv_heads, n_beams);

    // -- RoPE -----------------------------------------------------------------
    t_q = ggml_rope_ext(ctx.get(), t_q, t_pos_b, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);
    t_k = ggml_rope_ext(ctx.get(), t_k, t_pos_b, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);

    // Expand to 4D for batched attention: [head_dim, n_heads, 1, n_beams]
    t_q = ggml_reshape_4d(ctx.get(), t_q, hp.head_dim, hp.num_heads,    1, n_beams);
    t_k = ggml_reshape_4d(ctx.get(), t_k, hp.head_dim, hp.num_kv_heads, 1, n_beams);
    t_v = ggml_reshape_4d(ctx.get(), t_v, hp.head_dim, hp.num_kv_heads, 1, n_beams);

    // -- write new K/V into rollout batch via set_rows ------------------------
    // Flatten new K/V: [head_dim, n_kv_heads, 1, n_beams] → [kv_row_size, n_beams]
    const int32_t kv_row_size = hp.head_dim * hp.num_kv_heads;
    ggml_tensor * t_k_flat = ggml_reshape_2d(ctx.get(), t_k, kv_row_size, n_beams);
    ggml_tensor * t_v_flat = ggml_reshape_2d(ctx.get(), t_v, kv_row_size, n_beams);

    // Flatten batch KV: [head_dim, n_kv_heads, kv_capacity, n_beams] → [kv_row_size, kv_capacity * n_beams]
    ggml_tensor * t_bk_2d = ggml_reshape_2d(ctx.get(), batch.t_k, kv_row_size, (int64_t) kv_cap * n_beams);
    ggml_tensor * t_bv_2d = ggml_reshape_2d(ctx.get(), batch.t_v, kv_row_size, (int64_t) kv_cap * n_beams);

    // set_rows: index for beam b at position p = b * kv_capacity + p
    ggml_tensor * t_k_write = ggml_set_rows(ctx.get(), t_bk_2d, t_k_flat, t_k_idxs_b);
    ggml_tensor * t_v_write = ggml_set_rows(ctx.get(), t_bv_2d, t_v_flat, t_v_idxs_b);

    // Reshape back to 4D for attention
    ggml_tensor * t_k_all = ggml_reshape_4d(ctx.get(), t_k_write, hp.head_dim, hp.num_kv_heads, kv_cap, n_beams);
    ggml_tensor * t_v_all = ggml_reshape_4d(ctx.get(), t_v_write, hp.head_dim, hp.num_kv_heads, kv_cap, n_beams);

    // -- attention (batched across beams, mask broadcast) ----------------------
    ggml_tensor * t_attn_out = build_eagle_attn_output(
            ctx.get(), hp, rt, t_q, t_k_all, t_v_all, t_mask,
            "rollout_step", rt.flash_attn_logged_step_batch);

    // -- O projection + residual + FFN ----------------------------------------
    ggml_tensor * t_attn = ggml_mul_mat(ctx.get(), tensors.attn_o_w, t_attn_out);
    if (tensors.attn_o_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_o_b, GGML_TYPE_F32);
        b = ggml_repeat(ctx.get(), b, t_attn);
        t_attn = ggml_add(ctx.get(), t_attn, b);
    }

    ggml_tensor * t_resid = hp.norm_before_residual ? t_hidden_norm : t_hidden;
    ggml_tensor * t_ha = ggml_add(ctx.get(), t_attn, t_resid);

    ggml_tensor * t_post = ggml_rms_norm(ctx.get(), t_ha, hp.rms_norm_eps);
    ggml_tensor * t_pw = ggml_cast(ctx.get(), tensors.post_norm_w, GGML_TYPE_F32);
    t_pw = ggml_repeat(ctx.get(), t_pw, t_post);
    t_post = ggml_mul(ctx.get(), t_post, t_pw);

    ggml_tensor * t_gate = ggml_mul_mat(ctx.get(), tensors.ffn_gate_w, t_post);
    if (tensors.ffn_gate_b) { t_gate = ggml_add(ctx.get(), t_gate, ggml_repeat(ctx.get(), ggml_cast(ctx.get(), tensors.ffn_gate_b, GGML_TYPE_F32), t_gate)); }
    ggml_tensor * t_up = ggml_mul_mat(ctx.get(), tensors.ffn_up_w, t_post);
    if (tensors.ffn_up_b) { t_up = ggml_add(ctx.get(), t_up, ggml_repeat(ctx.get(), ggml_cast(ctx.get(), tensors.ffn_up_b, GGML_TYPE_F32), t_up)); }
    ggml_tensor * t_ffn = ggml_mul_mat(ctx.get(), tensors.ffn_down_w,
            ggml_mul(ctx.get(), ggml_silu(ctx.get(), t_gate), t_up));
    if (tensors.ffn_down_b) { t_ffn = ggml_add(ctx.get(), t_ffn, ggml_repeat(ctx.get(), ggml_cast(ctx.get(), tensors.ffn_down_b, GGML_TYPE_F32), t_ffn)); }

    ggml_tensor * t_hidden_out_b = ggml_cont(ctx.get(), ggml_add(ctx.get(), t_ha, t_ffn));

    // -- build and allocate ---------------------------------------------------
    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, t_hidden_out_b);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    if (!buf) {
        return false;
    }

    graph = {};
    graph.ctx           = std::move(ctx);
    graph.buf_compute   = std::move(buf);
    graph.gf            = gf;
    graph.n_beams       = n_beams;
    graph.kv_capacity   = kv_cap;
    graph.t_hidden_in_b = t_hidden_in_b;
    graph.t_tok_b       = t_tok_b;
    graph.t_pos_b       = t_pos_b;
    graph.t_k_idxs_b    = t_k_idxs_b;
    graph.t_v_idxs_b    = t_v_idxs_b;
    graph.t_mask         = t_mask;
    graph.t_hidden_out_b = t_hidden_out_b;
    return true;
}

// Fork buffer: small staging area for copying rollout rows during beam fork.
// Layout: [kv_row_size, max_depth * n_beams] for K and V separately.
struct rollout_fork_buffer {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor * t_k = nullptr;
    ggml_tensor * t_v = nullptr;
    int32_t n_beams = 0;
    int32_t max_depth = 0;
};

bool alloc_fork_buffer(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t max_depth,
        int32_t n_beams,
        rollout_fork_buffer & out) {
    if (!rt.buft_compute || max_depth <= 0 || n_beams <= 0) {
        return false;
    }
    const auto & hp = model.hparams;
    const int32_t kv_row_size = hp.head_dim * hp.num_kv_heads;
    const int32_t total_rows = max_depth * n_beams;

    auto ctx = make_ctx_no_alloc(/* max_nodes = */ 16);
    if (!ctx) { return false; }

    ggml_tensor * t_k = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kv_row_size, total_rows);
    ggml_tensor * t_v = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kv_row_size, total_rows);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    if (!buf) { return false; }

    out = {};
    out.ctx = std::move(ctx);
    out.buf = std::move(buf);
    out.t_k = t_k;
    out.t_v = t_v;
    out.n_beams = n_beams;
    out.max_depth = max_depth;
    return true;
}

// =====================================================================
// GPU-fused rollout: SELECT + BRIDGE + STEP in a single ggml graph
// =====================================================================

// GPU-resident d2t lookup table (uploaded once at runtime init).
struct fused_d2t_gpu {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor * t_base_id_table = nullptr;  // [1, draft_vocab_size] F32
    // Precomputed: t_base_id_table[i] = float(i + d2t[i])
};

// Rollout journal: GPU storage for post-rollout reconstruction.
// Indexed as: depth * n_beams + beam.
struct fused_journal {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor * t_parents  = nullptr;  // [n_beams * max_depth] I32
    ggml_tensor * t_tokens   = nullptr;  // [n_beams * max_depth] I32
    ggml_tensor * t_logprobs = nullptr;  // [n_beams * max_depth] F32
    ggml_tensor * t_hidden   = nullptr;  // [hidden_size, n_beams * max_depth] F32
    int32_t n_beams   = 0;
    int32_t max_depth = 0;
};

// Fused SELECT+BRIDGE+STEP graph.
struct fused_depth_graph_data {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf_compute;
    ggml_cgraph * gf = nullptr;
    int32_t n_beams       = 0;
    int32_t k             = 0;
    int32_t kv_capacity   = 0;
    int32_t max_fork_rows = 0;  // max_depth - 1
    float   prob_threshold = 0.0f;

    // Inputs (updated per depth via async set).
    ggml_tensor * t_hidden_in       = nullptr;  // [hidden_size, n_beams]
    ggml_tensor * t_beam_logprob    = nullptr;  // [n_beams]
    ggml_tensor * t_pos_b           = nullptr;  // [n_beams] I32
    ggml_tensor * t_mask            = nullptr;  // [kv_capacity, 1, 1, 1] F16
    ggml_tensor * t_kv_write_idxs   = nullptr;  // [n_beams] I32
    ggml_tensor * t_fork_row_offsets = nullptr; // [max_fork_rows * n_beams] F32 (constant within rollout)
    ggml_tensor * t_fork_dst_idxs   = nullptr;  // [max_fork_rows * n_beams] I32 (constant within rollout)

    // Outputs (read between dispatches for journal + next depth inputs).
    ggml_tensor * t_hidden_out       = nullptr; // [hidden_size, n_beams]
    ggml_tensor * t_selected_logprob = nullptr; // [n_beams]
    ggml_tensor * t_parent           = nullptr; // [n_beams] I32
    ggml_tensor * t_base_id          = nullptr; // [n_beams] I32
};

bool upload_fused_d2t(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        fused_d2t_gpu & out) {
    if (!rt.buft_compute) { return false; }

    const auto & hp = model.hparams;
    const int32_t dvs = hp.draft_vocab_size;
    if (dvs <= 0 || (int32_t) model.d2t.size() != dvs) { return false; }

    // Already uploaded?  Table has dvs+1 rows: one extra for the sentinel index
    // that ggml_top_k_threshold emits for empty slots (sentinel = draft_vocab_size).
    if (out.t_base_id_table && ggml_nelements(out.t_base_id_table) == dvs + 1) {
        return true;
    }

    auto ctx = make_ctx_no_alloc(/* max_nodes = */ 4);
    if (!ctx) { return false; }

    ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, dvs + 1);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    if (!buf) { return false; }

    // Precompute base_id_table[i] = float(i + d2t[i]).
    // Entry [dvs] is the sentinel — map to token 0 (safe; sentinels should
    // never win beam selection due to their extremely low logprob).
    std::vector<float> host(dvs + 1);
    for (int32_t i = 0; i < dvs; ++i) {
        host[i] = (float)(i + model.d2t[i]);
    }
    host[dvs] = 0.0f;
    ggml_backend_tensor_set(t, host.data(), 0, (dvs + 1) * sizeof(float));

    out.ctx = std::move(ctx);
    out.buf = std::move(buf);
    out.t_base_id_table = t;
    return true;
}

bool alloc_fused_journal(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t n_beams,
        int32_t max_depth,
        fused_journal & out) {
    if (!rt.buft_compute || n_beams <= 0 || max_depth <= 0) { return false; }

    const auto & hp = model.hparams;
    const int32_t total = n_beams * max_depth;

    // Already allocated with sufficient size?
    if (out.t_parents && out.n_beams >= n_beams && out.max_depth >= max_depth) {
        return true;
    }

    auto ctx = make_ctx_no_alloc(/* max_nodes = */ 16);
    if (!ctx) { return false; }

    ggml_tensor * t_parents  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, total);
    ggml_tensor * t_tokens   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, total);
    ggml_tensor * t_logprobs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, total);
    ggml_tensor * t_hidden   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hp.hidden_size, total);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    if (!buf) { return false; }

    out.ctx       = std::move(ctx);
    out.buf       = std::move(buf);
    out.t_parents  = t_parents;
    out.t_tokens   = t_tokens;
    out.t_logprobs = t_logprobs;
    out.t_hidden   = t_hidden;
    out.n_beams    = n_beams;
    out.max_depth  = max_depth;
    return true;
}

bool build_fused_depth_graph(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const llama_eagle3_rollout_batch & batch,
        const fused_d2t_gpu & d2t,
        int32_t n_beams,
        int32_t k,
        int32_t max_depth,
        float prob_threshold,
        fused_depth_graph_data & graph) {
    if (!rt.tok_embd || !d2t.t_base_id_table || batch.n_beams < n_beams ||
        batch.kv_capacity <= 0 || n_beams <= 0 || k <= 0) {
        return false;
    }

    // Check if already built with matching params.
    if (graph.ctx && graph.buf_compute && graph.gf &&
        graph.n_beams == n_beams && graph.k == k &&
        graph.kv_capacity == batch.kv_capacity &&
        graph.max_fork_rows == std::max(0, max_depth - 1) &&
        graph.prob_threshold == prob_threshold) {
        return true;
    }

    const auto & hp = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);
    const int32_t kv_cap = batch.kv_capacity;
    const int32_t kv_row_size = hp.head_dim * hp.num_kv_heads;
    const int32_t max_fork_rows = std::max(0, max_depth - 1);
    const int32_t total_fork = max_fork_rows * n_beams;
    const int32_t hidden_in_dim = hp.hidden_size; // rollout always uses head's hidden_size

    // Large node budget: SELECT (~40) + BRIDGE (~30) + STEP (~40) + overhead.
    auto ctx = make_ctx_no_alloc(/* max_nodes = */ 2048);
    if (!ctx) { return false; }

    // ===================== INPUTS =====================
    ggml_tensor * t_hidden_in = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hp.hidden_size, n_beams);
    ggml_set_name(t_hidden_in, "fused_hidden_in");
    ggml_set_input(t_hidden_in);

    ggml_tensor * t_beam_logprob = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_beams);
    ggml_set_name(t_beam_logprob, "fused_beam_logprob");
    ggml_set_input(t_beam_logprob);

    ggml_tensor * t_pos_b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_beams);
    ggml_set_name(t_pos_b, "fused_pos");
    ggml_set_input(t_pos_b);

    ggml_tensor * t_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, kv_cap, 1, 1, 1);
    ggml_set_name(t_mask, "fused_mask");
    ggml_set_input(t_mask);

    ggml_tensor * t_kv_write_idxs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_beams);
    ggml_set_name(t_kv_write_idxs, "fused_kv_write_idxs");
    ggml_set_input(t_kv_write_idxs);

    ggml_tensor * t_fork_row_offsets = nullptr;
    ggml_tensor * t_fork_dst_idxs   = nullptr;
    if (total_fork > 0) {
        t_fork_row_offsets = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, total_fork);
        ggml_set_name(t_fork_row_offsets, "fused_fork_row_offsets");
        ggml_set_input(t_fork_row_offsets);

        t_fork_dst_idxs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, total_fork);
        ggml_set_name(t_fork_dst_idxs, "fused_fork_dst_idxs");
        ggml_set_input(t_fork_dst_idxs);
    }

    // ===================== SELECT =====================
    // RMS norm + logits.
    ggml_tensor * t_norm = ggml_rms_norm(ctx.get(), t_hidden_in, hp.rms_norm_eps);
    ggml_tensor * t_norm_w = ggml_cast(ctx.get(), tensors.norm_w, GGML_TYPE_F32);
    t_norm = ggml_mul(ctx.get(), t_norm, t_norm_w);

    ggml_tensor * t_logits = ggml_mul_mat(ctx.get(), tensors.lm_head_w, t_norm);
    ggml_tensor * t_probs = ggml_soft_max(ctx.get(), t_logits);
    t_probs = ggml_cont(ctx.get(), t_probs);

    // Top-k per beam with threshold filtering.
    ggml_tensor * t_topk_idx = ggml_top_k_threshold(ctx.get(), t_probs, k, prob_threshold);
    t_topk_idx = ggml_cont(ctx.get(), t_topk_idx);

    // Extend probs with zero column for padding index.
    ggml_tensor * t_probs_zero = ggml_view_2d(ctx.get(), t_probs, 1, n_beams, t_probs->nb[1], 0);
    t_probs_zero = ggml_scale(ctx.get(), t_probs_zero, 0.0f);
    ggml_tensor * t_probs_ext = ggml_concat(ctx.get(), t_probs, t_probs_zero, 0);
    t_probs_ext = ggml_cont(ctx.get(), t_probs_ext);
    ggml_tensor * t_probs_flat = ggml_reshape_2d(ctx.get(), t_probs_ext, 1, (int64_t)(hp.draft_vocab_size + 1) * n_beams);

    // Linear indexing: beam_idx * (vocab+1) + draft_token_idx.
    ggml_tensor * t_beam_off = ggml_arange(ctx.get(), 0.0f, (float) n_beams, 1.0f);
    t_beam_off = ggml_scale(ctx.get(), t_beam_off, (float)(hp.draft_vocab_size + 1));
    t_beam_off = ggml_reshape_2d(ctx.get(), t_beam_off, 1, n_beams);

    ggml_tensor * t_topk_idx_f = ggml_cast(ctx.get(), t_topk_idx, GGML_TYPE_F32);
    ggml_tensor * t_beam_rep   = ggml_repeat(ctx.get(), t_beam_off, t_topk_idx_f);
    ggml_tensor * t_linear_f   = ggml_add(ctx.get(), t_topk_idx_f, t_beam_rep);
    ggml_set_name(t_linear_f, "eagle3_beam_off");
    ggml_tensor * t_linear     = ggml_cast(ctx.get(), t_linear_f, GGML_TYPE_I32);
    t_linear = ggml_cont(ctx.get(), t_linear);
    ggml_tensor * t_linear_flat = ggml_reshape_1d(ctx.get(), t_linear, (int64_t) k * n_beams);

    // Extract probabilities and compute log-probs.
    ggml_tensor * t_topk_prob = ggml_get_rows(ctx.get(), t_probs_flat, t_linear_flat);
    t_topk_prob = ggml_reshape_2d(ctx.get(), t_topk_prob, k, n_beams);
    t_topk_prob = ggml_cont(ctx.get(), t_topk_prob);

    ggml_tensor * t_logprob_cand = ggml_clamp(ctx.get(), t_topk_prob, 1e-12f, 1.0f);
    t_logprob_cand = ggml_log(ctx.get(), t_logprob_cand);
    t_logprob_cand = ggml_cont(ctx.get(), t_logprob_cand);

    // Beam scoring: add accumulated beam logprob.
    ggml_tensor * t_beam_lp = ggml_reshape_2d(ctx.get(), t_beam_logprob, 1, n_beams);
    t_beam_lp = ggml_repeat(ctx.get(), t_beam_lp, t_logprob_cand);
    ggml_tensor * t_total = ggml_add(ctx.get(), t_logprob_cand, t_beam_lp);
    ggml_set_name(t_total, "eagle3_lp_accum");
    t_total = ggml_cont(ctx.get(), t_total);

    // Select top n_beams from flattened [k * n_beams] candidates.
    ggml_tensor * t_total_flat = ggml_reshape_2d(ctx.get(), t_total, (int64_t) k * n_beams, 1);
    ggml_tensor * t_selected_linear = ggml_top_k(ctx.get(), t_total_flat, n_beams);
    t_selected_linear = ggml_cont(ctx.get(), t_selected_linear);
    ggml_tensor * t_sel_flat = ggml_reshape_1d(ctx.get(), t_selected_linear, n_beams);
    t_sel_flat = ggml_cont(ctx.get(), t_sel_flat);

    // Extract selected logprobs.
    ggml_tensor * t_total_rows = ggml_reshape_2d(ctx.get(), t_total, 1, (int64_t) k * n_beams);
    ggml_tensor * t_selected_logprob = ggml_get_rows(ctx.get(), t_total_rows, t_sel_flat);
    t_selected_logprob = ggml_reshape_1d(ctx.get(), t_selected_logprob, n_beams);
    t_selected_logprob = ggml_cont(ctx.get(), t_selected_logprob);

    // Extract selected draft indices.
    ggml_tensor * t_topk_idx_rows = ggml_reshape_2d(ctx.get(), t_topk_idx_f, 1, (int64_t) k * n_beams);
    ggml_tensor * t_selected_draft_f = ggml_get_rows(ctx.get(), t_topk_idx_rows, t_sel_flat);
    t_selected_draft_f = ggml_reshape_1d(ctx.get(), t_selected_draft_f, n_beams);
    ggml_tensor * t_selected_draft = ggml_cast(ctx.get(), t_selected_draft_f, GGML_TYPE_I32);
    t_selected_draft = ggml_cont(ctx.get(), t_selected_draft);

    // ===================== BRIDGE =====================

    // 2a. Parent beam index = floor(selected_linear / k).
    ggml_tensor * t_sel_f = ggml_cast(ctx.get(), t_sel_flat, GGML_TYPE_F32);
    ggml_tensor * t_parent_f = ggml_floor(ctx.get(), ggml_scale(ctx.get(), t_sel_f, 1.0f / (float) k));
    ggml_tensor * t_parent = ggml_cast(ctx.get(), t_parent_f, GGML_TYPE_I32);
    t_parent = ggml_cont(ctx.get(), t_parent);

    // 2b. Token mapping via GPU d2t table.
    ggml_tensor * t_base_id_f = ggml_get_rows(ctx.get(), d2t.t_base_id_table, t_selected_draft);
    t_base_id_f = ggml_reshape_1d(ctx.get(), t_base_id_f, n_beams);
    ggml_tensor * t_base_id = ggml_cast(ctx.get(), t_base_id_f, GGML_TYPE_I32);
    t_base_id = ggml_cont(ctx.get(), t_base_id);

    // 2c. Gather parent hidden states for STEP input.
    ggml_tensor * t_hidden_gathered = ggml_get_rows(ctx.get(), t_hidden_in, t_parent);

    // 2d. KV fork: two-phase copy through fork buffer.
    // Flatten batch KV: [head_dim, n_kv_heads, kv_cap, n_beams] → [kv_row_size, kv_cap * n_beams].
    ggml_tensor * t_bk_2d = ggml_reshape_2d(ctx.get(), batch.t_k, kv_row_size, (int64_t) kv_cap * n_beams);
    ggml_tensor * t_bv_2d = ggml_reshape_2d(ctx.get(), batch.t_v, kv_row_size, (int64_t) kv_cap * n_beams);

    ggml_tensor * t_bk_forked = t_bk_2d;
    ggml_tensor * t_bv_forked = t_bv_2d;

    if (total_fork > 0) {
        // Compute source indices: parent[beam] * kv_cap + fork_row_offset.
        ggml_tensor * t_parent_f_2d = ggml_reshape_2d(ctx.get(), t_parent_f, 1, n_beams);
        ggml_tensor * t_fork_offsets_2d = ggml_reshape_2d(ctx.get(), t_fork_row_offsets, max_fork_rows, n_beams);
        ggml_tensor * t_parent_rep = ggml_repeat(ctx.get(), t_parent_f_2d, t_fork_offsets_2d);
        ggml_tensor * t_parent_fork_flat = ggml_cont(ctx.get(), ggml_reshape_1d(ctx.get(), t_parent_rep, total_fork));

        ggml_tensor * t_src_idx_f = ggml_add(ctx.get(),
                ggml_scale(ctx.get(), t_parent_fork_flat, (float) kv_cap),
                t_fork_row_offsets);
        ggml_tensor * t_src_idx = ggml_cast(ctx.get(), t_src_idx_f, GGML_TYPE_I32);
        t_src_idx = ggml_cont(ctx.get(), t_src_idx);

        // Phase A: gather parent rows into fork staging.
        ggml_tensor * t_fork_k = ggml_get_rows(ctx.get(), t_bk_2d, t_src_idx);
        ggml_tensor * t_fork_v = ggml_get_rows(ctx.get(), t_bv_2d, t_src_idx);

        // Phase B: scatter from fork staging to child slots.
        t_bk_forked = ggml_set_rows(ctx.get(), t_bk_2d, t_fork_k, t_fork_dst_idxs);
        t_bv_forked = ggml_set_rows(ctx.get(), t_bv_2d, t_fork_v, t_fork_dst_idxs);
    }

    // ===================== STEP =====================

    // Token embedding from base_id.
    ggml_tensor * t_embd = ggml_get_rows(ctx.get(), rt.tok_embd, t_base_id);
    t_embd = ggml_cast(ctx.get(), t_embd, GGML_TYPE_F32);

    // Hidden processing (hidden_in_dim == hidden_size during rollout, no fc_w needed).
    ggml_tensor * t_hidden = t_hidden_gathered;
    if (hidden_in_dim != hp.hidden_size && tensors.fc_w) {
        t_hidden = ggml_mul_mat(ctx.get(), tensors.fc_w, t_hidden_gathered);
    }

    // Input/hidden norms.
    ggml_tensor * t_hidden_norm = ggml_rms_norm(ctx.get(), t_hidden, hp.rms_norm_eps);
    ggml_tensor * t_hnw = ggml_cast(ctx.get(), tensors.hidden_norm_w, GGML_TYPE_F32);
    t_hnw = ggml_repeat(ctx.get(), t_hnw, t_hidden_norm);
    t_hidden_norm = ggml_mul(ctx.get(), t_hidden_norm, t_hnw);

    ggml_tensor * t_embd_norm = ggml_rms_norm(ctx.get(), t_embd, hp.rms_norm_eps);
    ggml_tensor * t_inw = ggml_cast(ctx.get(), tensors.input_norm_w, GGML_TYPE_F32);
    t_inw = ggml_repeat(ctx.get(), t_inw, t_embd_norm);
    t_embd_norm = ggml_mul(ctx.get(), t_embd_norm, t_inw);

    // Concat [embd_norm, hidden_norm].
    ggml_tensor * t_cat = ggml_concat(ctx.get(), t_embd_norm, t_hidden_norm, 0);

    // QKV projections.
    ggml_tensor * t_q = ggml_mul_mat(ctx.get(), tensors.attn_q_w, t_cat);
    if (tensors.attn_q_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_q_b, GGML_TYPE_F32);
        b = ggml_repeat(ctx.get(), b, t_q);
        t_q = ggml_add(ctx.get(), t_q, b);
    }
    ggml_tensor * t_k_proj = ggml_mul_mat(ctx.get(), tensors.attn_k_w, t_cat);
    if (tensors.attn_k_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_k_b, GGML_TYPE_F32);
        b = ggml_repeat(ctx.get(), b, t_k_proj);
        t_k_proj = ggml_add(ctx.get(), t_k_proj, b);
    }
    ggml_tensor * t_v_proj = ggml_mul_mat(ctx.get(), tensors.attn_v_w, t_cat);
    if (tensors.attn_v_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_v_b, GGML_TYPE_F32);
        b = ggml_repeat(ctx.get(), b, t_v_proj);
        t_v_proj = ggml_add(ctx.get(), t_v_proj, b);
    }

    // Reshape for RoPE: [head_dim, n_heads/n_kv, n_beams] — must be 3D so
    // ne[2] == n_beams matches t_pos_b ne[0] == n_beams for ggml_rope_ext.
    t_q = ggml_reshape_3d(ctx.get(), t_q, hp.head_dim, hp.num_heads,    n_beams);
    t_k_proj = ggml_reshape_3d(ctx.get(), t_k_proj, hp.head_dim, hp.num_kv_heads, n_beams);
    t_v_proj = ggml_reshape_3d(ctx.get(), t_v_proj, hp.head_dim, hp.num_kv_heads, n_beams);

    // RoPE.
    t_q = ggml_rope_ext(ctx.get(), t_q, t_pos_b, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);
    t_k_proj = ggml_rope_ext(ctx.get(), t_k_proj, t_pos_b, rt.rope_factors,
            hp.head_dim, rt.rope_type, rt.n_ctx_orig,
            rt.rope_freq_base, rt.rope_freq_scale,
            rt.yarn_ext_factor, rt.yarn_attn_factor,
            rt.yarn_beta_fast, rt.yarn_beta_slow);

    // Expand back to 4D for batched attention: [head_dim, n_heads, 1, n_beams].
    t_q = ggml_reshape_4d(ctx.get(), t_q, hp.head_dim, hp.num_heads, 1, n_beams);
    t_k_proj = ggml_reshape_4d(ctx.get(), t_k_proj, hp.head_dim, hp.num_kv_heads, 1, n_beams);
    t_v_proj = ggml_reshape_4d(ctx.get(), t_v_proj, hp.head_dim, hp.num_kv_heads, 1, n_beams);

    // Write new KV row to batch (via set_rows on forked batch).
    ggml_tensor * t_k_flat = ggml_reshape_2d(ctx.get(), t_k_proj, kv_row_size, n_beams);
    ggml_tensor * t_v_flat = ggml_reshape_2d(ctx.get(), t_v_proj, kv_row_size, n_beams);

    ggml_tensor * t_bk_written = ggml_set_rows(ctx.get(), t_bk_forked, t_k_flat, t_kv_write_idxs);
    ggml_tensor * t_bv_written = ggml_set_rows(ctx.get(), t_bv_forked, t_v_flat, t_kv_write_idxs);

    // Reshape for attention.
    ggml_tensor * t_k_all = ggml_reshape_4d(ctx.get(), t_bk_written, hp.head_dim, hp.num_kv_heads, kv_cap, n_beams);
    ggml_tensor * t_v_all = ggml_reshape_4d(ctx.get(), t_bv_written, hp.head_dim, hp.num_kv_heads, kv_cap, n_beams);

    // Attention.
    bool fa_logged = false;
    ggml_tensor * t_attn_out = build_eagle_attn_output(
            ctx.get(), hp, rt, t_q, t_k_all, t_v_all, t_mask,
            "fused_rollout", fa_logged);

    // Output projection + residual.
    ggml_tensor * t_attn = ggml_mul_mat(ctx.get(), tensors.attn_o_w, t_attn_out);
    if (tensors.attn_o_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_o_b, GGML_TYPE_F32);
        b = ggml_repeat(ctx.get(), b, t_attn);
        t_attn = ggml_add(ctx.get(), t_attn, b);
    }
    ggml_tensor * t_resid = hp.norm_before_residual ? t_hidden_norm : t_hidden;
    ggml_tensor * t_ha = ggml_add(ctx.get(), t_attn, t_resid);
    ggml_set_name(t_ha, "eagle3_resid_attn");

    // FFN: post-norm → gate/up → silu → down → residual.
    ggml_tensor * t_post = ggml_rms_norm(ctx.get(), t_ha, hp.rms_norm_eps);
    ggml_tensor * t_pw = ggml_cast(ctx.get(), tensors.post_norm_w, GGML_TYPE_F32);
    t_pw = ggml_repeat(ctx.get(), t_pw, t_post);
    t_post = ggml_mul(ctx.get(), t_post, t_pw);

    ggml_tensor * t_gate = ggml_mul_mat(ctx.get(), tensors.ffn_gate_w, t_post);
    if (tensors.ffn_gate_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.ffn_gate_b, GGML_TYPE_F32);
        b = ggml_repeat(ctx.get(), b, t_gate);
        t_gate = ggml_add(ctx.get(), t_gate, b);
    }
    ggml_tensor * t_up = ggml_mul_mat(ctx.get(), tensors.ffn_up_w, t_post);
    if (tensors.ffn_up_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.ffn_up_b, GGML_TYPE_F32);
        b = ggml_repeat(ctx.get(), b, t_up);
        t_up = ggml_add(ctx.get(), t_up, b);
    }
    ggml_tensor * t_ffn_out = ggml_mul(ctx.get(), ggml_silu(ctx.get(), t_gate), t_up);
    t_ffn_out = ggml_mul_mat(ctx.get(), tensors.ffn_down_w, t_ffn_out);
    if (tensors.ffn_down_b) {
        ggml_tensor * b = ggml_cast(ctx.get(), tensors.ffn_down_b, GGML_TYPE_F32);
        b = ggml_repeat(ctx.get(), b, t_ffn_out);
        t_ffn_out = ggml_add(ctx.get(), t_ffn_out, b);
    }

    // Hidden output.
    ggml_tensor * t_resid_ffn = ggml_add(ctx.get(), t_ha, t_ffn_out);
    ggml_set_name(t_resid_ffn, "eagle3_resid_ffn");
    ggml_tensor * t_hidden_out = ggml_cont(ctx.get(), t_resid_ffn);
    ggml_set_name(t_hidden_out, "fused_hidden_out");

    // ===================== BUILD GRAPH =====================
    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 2048, /* grads = */ false);
    ggml_build_forward_expand(gf, t_hidden_out);
    ggml_build_forward_expand(gf, t_selected_logprob);
    ggml_build_forward_expand(gf, t_parent);
    ggml_build_forward_expand(gf, t_base_id);
    // Ensure KV writes are in the graph (they may not be reachable from outputs).
    ggml_build_forward_expand(gf, t_bk_written);
    ggml_build_forward_expand(gf, t_bv_written);

    // Allocate compute buffer.
    ggml_backend_buffer_ptr buf_compute;
    if (rt.buft_compute) {
        buf_compute.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    }
    if (!buf_compute) { return false; }

    graph.ctx              = std::move(ctx);
    graph.buf_compute      = std::move(buf_compute);
    graph.gf               = gf;
    graph.n_beams          = n_beams;
    graph.k                = k;
    graph.kv_capacity      = kv_cap;
    graph.max_fork_rows    = max_fork_rows;
    graph.prob_threshold   = prob_threshold;
    graph.t_hidden_in       = t_hidden_in;
    graph.t_beam_logprob    = t_beam_logprob;
    graph.t_pos_b           = t_pos_b;
    graph.t_mask            = t_mask;
    graph.t_kv_write_idxs   = t_kv_write_idxs;
    graph.t_fork_row_offsets = t_fork_row_offsets;
    graph.t_fork_dst_idxs   = t_fork_dst_idxs;
    graph.t_hidden_out       = t_hidden_out;
    graph.t_selected_logprob = t_selected_logprob;
    graph.t_parent           = t_parent;
    graph.t_base_id          = t_base_id;
    return true;
}

// =====================================================================
// Mega-graph: chains all max_depth fused depths into a single dispatch.
// Eliminates inter-depth async writes (pos, mask, kv_write_idxs) and
// inter-depth GPU→GPU copies (hidden_out→hidden_in, logprob forwarding).
// =====================================================================

struct fused_mega_graph_data {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf_compute;
    ggml_cgraph * gf = nullptr;
    int32_t n_beams       = 0;
    int32_t k             = 0;
    int32_t kv_capacity   = 0;
    int32_t max_depth     = 0;
    int32_t max_fork_rows = 0;
    float   prob_threshold = 0.0f;

    // Inputs (set once before single dispatch).
    ggml_tensor * t_hidden_in        = nullptr;  // [hidden_size, n_beams]
    ggml_tensor * t_beam_logprob     = nullptr;  // [n_beams]
    ggml_tensor * t_pos_all          = nullptr;  // [n_beams, max_depth] I32
    ggml_tensor * t_mask_all         = nullptr;  // [kv_cap, max_depth] F16
    ggml_tensor * t_kv_write_idxs_all = nullptr; // [n_beams, max_depth] I32
    ggml_tensor * t_fork_row_offsets = nullptr;  // [max_fork_rows * n_beams] F32
    ggml_tensor * t_fork_dst_idxs   = nullptr;   // [max_fork_rows * n_beams] I32

    // Per-depth outputs (marked ggml_set_output, read after sync).
    std::vector<ggml_tensor *> out_parents;   // [n_beams] I32 each
    std::vector<ggml_tensor *> out_tokens;    // [n_beams] I32 each
    std::vector<ggml_tensor *> out_logprobs;  // [n_beams] F32 each
    std::vector<ggml_tensor *> out_hidden;    // [hidden_size, n_beams] F32 each
};

bool build_fused_mega_graph(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const llama_eagle3_rollout_batch & batch,
        const fused_d2t_gpu & d2t,
        int32_t n_beams,
        int32_t k,
        int32_t max_depth,
        float prob_threshold,
        fused_mega_graph_data & mg) {
    if (!rt.tok_embd || !d2t.t_base_id_table || batch.n_beams < n_beams ||
        batch.kv_capacity <= 0 || n_beams <= 0 || k <= 0 || max_depth <= 0) {
        return false;
    }

    // Check if already built with matching params.
    if (mg.ctx && mg.buf_compute && mg.gf &&
        mg.n_beams == n_beams && mg.k == k &&
        mg.kv_capacity == batch.kv_capacity &&
        mg.max_depth == max_depth &&
        mg.max_fork_rows == std::max(0, max_depth - 1) &&
        mg.prob_threshold == prob_threshold) {
        return true;
    }

    const auto & hp = model.hparams;
    const auto & tensors = get_runtime_tensors(model, rt);
    const int32_t kv_cap = batch.kv_capacity;
    const int32_t kv_row_size = hp.head_dim * hp.num_kv_heads;
    const int32_t max_fork_rows = std::max(0, max_depth - 1);
    const int32_t total_fork = max_fork_rows * n_beams;

    // Node budget: ~120 ops/depth × max_depth + overhead.
    const int32_t max_nodes = std::max(4096, max_depth * 200);
    auto ctx = make_ctx_no_alloc(max_nodes);
    if (!ctx) { return false; }

    // ===================== GLOBAL INPUTS =====================

    ggml_tensor * t_hidden_in = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hp.hidden_size, n_beams);
    ggml_set_name(t_hidden_in, "mega_hidden_in");
    ggml_set_input(t_hidden_in);

    ggml_tensor * t_beam_logprob = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_beams);
    ggml_set_name(t_beam_logprob, "mega_beam_logprob");
    ggml_set_input(t_beam_logprob);

    // Per-depth inputs packed as 2D arrays: slice d via view.
    ggml_tensor * t_pos_all = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_beams, max_depth);
    ggml_set_name(t_pos_all, "mega_pos_all");
    ggml_set_input(t_pos_all);

    ggml_tensor * t_mask_all = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, kv_cap, max_depth);
    ggml_set_name(t_mask_all, "mega_mask_all");
    ggml_set_input(t_mask_all);

    ggml_tensor * t_kv_idxs_all = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_beams, max_depth);
    ggml_set_name(t_kv_idxs_all, "mega_kv_idxs_all");
    ggml_set_input(t_kv_idxs_all);

    ggml_tensor * t_fork_row_offsets = nullptr;
    ggml_tensor * t_fork_dst_idxs   = nullptr;
    if (total_fork > 0) {
        t_fork_row_offsets = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, total_fork);
        ggml_set_name(t_fork_row_offsets, "mega_fork_offsets");
        ggml_set_input(t_fork_row_offsets);

        t_fork_dst_idxs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, total_fork);
        ggml_set_name(t_fork_dst_idxs, "mega_fork_dst");
        ggml_set_input(t_fork_dst_idxs);
    }

    // ===================== KV CHAIN START =====================

    ggml_tensor * cur_bk = ggml_reshape_2d(ctx.get(), batch.t_k, kv_row_size, (int64_t) kv_cap * n_beams);
    ggml_tensor * cur_bv = ggml_reshape_2d(ctx.get(), batch.t_v, kv_row_size, (int64_t) kv_cap * n_beams);

    ggml_tensor * cur_hidden = t_hidden_in;
    ggml_tensor * cur_logprob = t_beam_logprob;

    // Per-depth output vectors.
    std::vector<ggml_tensor *> v_parents(max_depth);
    std::vector<ggml_tensor *> v_tokens(max_depth);
    std::vector<ggml_tensor *> v_logprobs(max_depth);
    std::vector<ggml_tensor *> v_hidden(max_depth);

    bool fa_logged = false;

    for (int32_t d = 0; d < max_depth; ++d) {
        // ---- Extract per-depth inputs via views ----
        ggml_tensor * t_pos_d = ggml_view_1d(ctx.get(), t_pos_all, n_beams,
                (size_t) d * n_beams * sizeof(int32_t));
        ggml_tensor * t_mask_slice = ggml_view_1d(ctx.get(), t_mask_all, kv_cap,
                (size_t) d * kv_cap * sizeof(ggml_fp16_t));
        ggml_tensor * t_mask_d = ggml_reshape_4d(ctx.get(), t_mask_slice, kv_cap, 1, 1, 1);
        ggml_tensor * t_kv_idxs_d = ggml_view_1d(ctx.get(), t_kv_idxs_all, n_beams,
                (size_t) d * n_beams * sizeof(int32_t));

        // ==================== SELECT ====================

        ggml_tensor * t_norm = ggml_rms_norm(ctx.get(), cur_hidden, hp.rms_norm_eps);
        ggml_tensor * t_norm_w = ggml_cast(ctx.get(), tensors.norm_w, GGML_TYPE_F32);
        t_norm = ggml_mul(ctx.get(), t_norm, t_norm_w);

        ggml_tensor * t_logits = ggml_mul_mat(ctx.get(), tensors.lm_head_w, t_norm);
        ggml_tensor * t_probs = ggml_soft_max(ctx.get(), t_logits);
        t_probs = ggml_cont(ctx.get(), t_probs);

        ggml_tensor * t_topk_idx = ggml_top_k_threshold(ctx.get(), t_probs, k, prob_threshold);
        t_topk_idx = ggml_cont(ctx.get(), t_topk_idx);

        ggml_tensor * t_probs_zero = ggml_view_2d(ctx.get(), t_probs, 1, n_beams, t_probs->nb[1], 0);
        t_probs_zero = ggml_scale(ctx.get(), t_probs_zero, 0.0f);
        ggml_tensor * t_probs_ext = ggml_concat(ctx.get(), t_probs, t_probs_zero, 0);
        t_probs_ext = ggml_cont(ctx.get(), t_probs_ext);
        ggml_tensor * t_probs_flat = ggml_reshape_2d(ctx.get(), t_probs_ext, 1,
                (int64_t)(hp.draft_vocab_size + 1) * n_beams);

        ggml_tensor * t_beam_off = ggml_arange(ctx.get(), 0.0f, (float) n_beams, 1.0f);
        t_beam_off = ggml_scale(ctx.get(), t_beam_off, (float)(hp.draft_vocab_size + 1));
        t_beam_off = ggml_reshape_2d(ctx.get(), t_beam_off, 1, n_beams);

        ggml_tensor * t_topk_idx_f = ggml_cast(ctx.get(), t_topk_idx, GGML_TYPE_F32);
        ggml_tensor * t_beam_rep   = ggml_repeat(ctx.get(), t_beam_off, t_topk_idx_f);
        ggml_tensor * t_linear_f   = ggml_add(ctx.get(), t_topk_idx_f, t_beam_rep);
        ggml_set_name(t_linear_f, "eagle3_beam_off");
        ggml_tensor * t_linear     = ggml_cast(ctx.get(), t_linear_f, GGML_TYPE_I32);
        t_linear = ggml_cont(ctx.get(), t_linear);
        ggml_tensor * t_linear_flat = ggml_reshape_1d(ctx.get(), t_linear, (int64_t) k * n_beams);

        ggml_tensor * t_topk_prob = ggml_get_rows(ctx.get(), t_probs_flat, t_linear_flat);
        t_topk_prob = ggml_reshape_2d(ctx.get(), t_topk_prob, k, n_beams);
        t_topk_prob = ggml_cont(ctx.get(), t_topk_prob);

        ggml_tensor * t_logprob_cand = ggml_clamp(ctx.get(), t_topk_prob, 1e-12f, 1.0f);
        t_logprob_cand = ggml_log(ctx.get(), t_logprob_cand);
        t_logprob_cand = ggml_cont(ctx.get(), t_logprob_cand);

        ggml_tensor * t_beam_lp = ggml_reshape_2d(ctx.get(), cur_logprob, 1, n_beams);
        t_beam_lp = ggml_repeat(ctx.get(), t_beam_lp, t_logprob_cand);
        ggml_tensor * t_total = ggml_add(ctx.get(), t_logprob_cand, t_beam_lp);
        ggml_set_name(t_total, "eagle3_lp_accum");
        t_total = ggml_cont(ctx.get(), t_total);

        ggml_tensor * t_total_flat = ggml_reshape_2d(ctx.get(), t_total, (int64_t) k * n_beams, 1);
        ggml_tensor * t_selected_linear = ggml_top_k(ctx.get(), t_total_flat, n_beams);
        t_selected_linear = ggml_cont(ctx.get(), t_selected_linear);
        ggml_tensor * t_sel_flat = ggml_reshape_1d(ctx.get(), t_selected_linear, n_beams);
        t_sel_flat = ggml_cont(ctx.get(), t_sel_flat);

        ggml_tensor * t_total_rows = ggml_reshape_2d(ctx.get(), t_total, 1, (int64_t) k * n_beams);
        ggml_tensor * t_selected_logprob = ggml_get_rows(ctx.get(), t_total_rows, t_sel_flat);
        t_selected_logprob = ggml_reshape_1d(ctx.get(), t_selected_logprob, n_beams);
        t_selected_logprob = ggml_cont(ctx.get(), t_selected_logprob);

        ggml_tensor * t_topk_idx_rows = ggml_reshape_2d(ctx.get(), t_topk_idx_f, 1, (int64_t) k * n_beams);
        ggml_tensor * t_selected_draft_f = ggml_get_rows(ctx.get(), t_topk_idx_rows, t_sel_flat);
        t_selected_draft_f = ggml_reshape_1d(ctx.get(), t_selected_draft_f, n_beams);
        ggml_tensor * t_selected_draft = ggml_cast(ctx.get(), t_selected_draft_f, GGML_TYPE_I32);
        t_selected_draft = ggml_cont(ctx.get(), t_selected_draft);

        // ==================== BRIDGE ====================

        ggml_tensor * t_sel_f = ggml_cast(ctx.get(), t_sel_flat, GGML_TYPE_F32);
        ggml_tensor * t_parent_f = ggml_floor(ctx.get(), ggml_scale(ctx.get(), t_sel_f, 1.0f / (float) k));
        ggml_tensor * t_parent = ggml_cast(ctx.get(), t_parent_f, GGML_TYPE_I32);
        t_parent = ggml_cont(ctx.get(), t_parent);

        ggml_tensor * t_base_id_f = ggml_get_rows(ctx.get(), d2t.t_base_id_table, t_selected_draft);
        t_base_id_f = ggml_reshape_1d(ctx.get(), t_base_id_f, n_beams);
        ggml_tensor * t_base_id = ggml_cast(ctx.get(), t_base_id_f, GGML_TYPE_I32);
        t_base_id = ggml_cont(ctx.get(), t_base_id);

        ggml_tensor * t_hidden_gathered = ggml_get_rows(ctx.get(), cur_hidden, t_parent);

        // KV fork.
        ggml_tensor * t_bk_forked = cur_bk;
        ggml_tensor * t_bv_forked = cur_bv;

        if (total_fork > 0) {
            ggml_tensor * t_parent_f_2d = ggml_reshape_2d(ctx.get(), t_parent_f, 1, n_beams);
            ggml_tensor * t_fork_offsets_2d = ggml_reshape_2d(ctx.get(), t_fork_row_offsets, max_fork_rows, n_beams);
            ggml_tensor * t_parent_rep = ggml_repeat(ctx.get(), t_parent_f_2d, t_fork_offsets_2d);
            ggml_tensor * t_parent_fork_flat = ggml_cont(ctx.get(),
                    ggml_reshape_1d(ctx.get(), t_parent_rep, total_fork));

            ggml_tensor * t_src_idx_f = ggml_add(ctx.get(),
                    ggml_scale(ctx.get(), t_parent_fork_flat, (float) kv_cap),
                    t_fork_row_offsets);
            ggml_tensor * t_src_idx = ggml_cast(ctx.get(), t_src_idx_f, GGML_TYPE_I32);
            t_src_idx = ggml_cont(ctx.get(), t_src_idx);

            ggml_tensor * t_fork_k = ggml_get_rows(ctx.get(), cur_bk, t_src_idx);
            ggml_tensor * t_fork_v = ggml_get_rows(ctx.get(), cur_bv, t_src_idx);

            t_bk_forked = ggml_set_rows(ctx.get(), cur_bk, t_fork_k, t_fork_dst_idxs);
            t_bv_forked = ggml_set_rows(ctx.get(), cur_bv, t_fork_v, t_fork_dst_idxs);
        }

        // ==================== STEP ====================

        ggml_tensor * t_embd = ggml_get_rows(ctx.get(), rt.tok_embd, t_base_id);
        t_embd = ggml_cast(ctx.get(), t_embd, GGML_TYPE_F32);

        // Hidden processing (hidden_in_dim == hidden_size during rollout, no fc_w needed).
        ggml_tensor * t_hidden = t_hidden_gathered;

        ggml_tensor * t_hidden_norm = ggml_rms_norm(ctx.get(), t_hidden, hp.rms_norm_eps);
        ggml_tensor * t_hnw = ggml_cast(ctx.get(), tensors.hidden_norm_w, GGML_TYPE_F32);
        t_hnw = ggml_repeat(ctx.get(), t_hnw, t_hidden_norm);
        t_hidden_norm = ggml_mul(ctx.get(), t_hidden_norm, t_hnw);

        ggml_tensor * t_embd_norm = ggml_rms_norm(ctx.get(), t_embd, hp.rms_norm_eps);
        ggml_tensor * t_inw = ggml_cast(ctx.get(), tensors.input_norm_w, GGML_TYPE_F32);
        t_inw = ggml_repeat(ctx.get(), t_inw, t_embd_norm);
        t_embd_norm = ggml_mul(ctx.get(), t_embd_norm, t_inw);

        ggml_tensor * t_cat = ggml_concat(ctx.get(), t_embd_norm, t_hidden_norm, 0);

        ggml_tensor * t_q = ggml_mul_mat(ctx.get(), tensors.attn_q_w, t_cat);
        if (tensors.attn_q_b) {
            ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_q_b, GGML_TYPE_F32);
            b = ggml_repeat(ctx.get(), b, t_q);
            t_q = ggml_add(ctx.get(), t_q, b);
        }
        ggml_tensor * t_k_proj = ggml_mul_mat(ctx.get(), tensors.attn_k_w, t_cat);
        if (tensors.attn_k_b) {
            ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_k_b, GGML_TYPE_F32);
            b = ggml_repeat(ctx.get(), b, t_k_proj);
            t_k_proj = ggml_add(ctx.get(), t_k_proj, b);
        }
        ggml_tensor * t_v_proj = ggml_mul_mat(ctx.get(), tensors.attn_v_w, t_cat);
        if (tensors.attn_v_b) {
            ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_v_b, GGML_TYPE_F32);
            b = ggml_repeat(ctx.get(), b, t_v_proj);
            t_v_proj = ggml_add(ctx.get(), t_v_proj, b);
        }

        // RoPE: 3D for ne[2] == n_beams matching t_pos_d.
        t_q = ggml_reshape_3d(ctx.get(), t_q, hp.head_dim, hp.num_heads, n_beams);
        t_k_proj = ggml_reshape_3d(ctx.get(), t_k_proj, hp.head_dim, hp.num_kv_heads, n_beams);
        t_v_proj = ggml_reshape_3d(ctx.get(), t_v_proj, hp.head_dim, hp.num_kv_heads, n_beams);

        t_q = ggml_rope_ext(ctx.get(), t_q, t_pos_d, rt.rope_factors,
                hp.head_dim, rt.rope_type, rt.n_ctx_orig,
                rt.rope_freq_base, rt.rope_freq_scale,
                rt.yarn_ext_factor, rt.yarn_attn_factor,
                rt.yarn_beta_fast, rt.yarn_beta_slow);
        t_k_proj = ggml_rope_ext(ctx.get(), t_k_proj, t_pos_d, rt.rope_factors,
                hp.head_dim, rt.rope_type, rt.n_ctx_orig,
                rt.rope_freq_base, rt.rope_freq_scale,
                rt.yarn_ext_factor, rt.yarn_attn_factor,
                rt.yarn_beta_fast, rt.yarn_beta_slow);

        // Back to 4D for batched attention.
        t_q = ggml_reshape_4d(ctx.get(), t_q, hp.head_dim, hp.num_heads, 1, n_beams);
        t_k_proj = ggml_reshape_4d(ctx.get(), t_k_proj, hp.head_dim, hp.num_kv_heads, 1, n_beams);
        t_v_proj = ggml_reshape_4d(ctx.get(), t_v_proj, hp.head_dim, hp.num_kv_heads, 1, n_beams);

        // Write new KV row to batch.
        ggml_tensor * t_k_flat = ggml_reshape_2d(ctx.get(), t_k_proj, kv_row_size, n_beams);
        ggml_tensor * t_v_flat = ggml_reshape_2d(ctx.get(), t_v_proj, kv_row_size, n_beams);

        ggml_tensor * t_bk_written = ggml_set_rows(ctx.get(), t_bk_forked, t_k_flat, t_kv_idxs_d);
        ggml_tensor * t_bv_written = ggml_set_rows(ctx.get(), t_bv_forked, t_v_flat, t_kv_idxs_d);

        // Attention over full KV.
        ggml_tensor * t_k_all = ggml_reshape_4d(ctx.get(), t_bk_written,
                hp.head_dim, hp.num_kv_heads, kv_cap, n_beams);
        ggml_tensor * t_v_all = ggml_reshape_4d(ctx.get(), t_bv_written,
                hp.head_dim, hp.num_kv_heads, kv_cap, n_beams);

        ggml_tensor * t_attn_out = build_eagle_attn_output(
                ctx.get(), hp, rt, t_q, t_k_all, t_v_all, t_mask_d,
                "mega_rollout", fa_logged);

        ggml_tensor * t_attn = ggml_mul_mat(ctx.get(), tensors.attn_o_w, t_attn_out);
        if (tensors.attn_o_b) {
            ggml_tensor * b = ggml_cast(ctx.get(), tensors.attn_o_b, GGML_TYPE_F32);
            b = ggml_repeat(ctx.get(), b, t_attn);
            t_attn = ggml_add(ctx.get(), t_attn, b);
        }
        ggml_tensor * t_resid = hp.norm_before_residual ? t_hidden_norm : t_hidden;
        ggml_tensor * t_ha = ggml_add(ctx.get(), t_attn, t_resid);
        ggml_set_name(t_ha, "eagle3_resid_attn");

        // FFN.
        ggml_tensor * t_post = ggml_rms_norm(ctx.get(), t_ha, hp.rms_norm_eps);
        ggml_tensor * t_pw = ggml_cast(ctx.get(), tensors.post_norm_w, GGML_TYPE_F32);
        t_pw = ggml_repeat(ctx.get(), t_pw, t_post);
        t_post = ggml_mul(ctx.get(), t_post, t_pw);

        ggml_tensor * t_gate = ggml_mul_mat(ctx.get(), tensors.ffn_gate_w, t_post);
        if (tensors.ffn_gate_b) {
            ggml_tensor * b = ggml_cast(ctx.get(), tensors.ffn_gate_b, GGML_TYPE_F32);
            b = ggml_repeat(ctx.get(), b, t_gate);
            t_gate = ggml_add(ctx.get(), t_gate, b);
        }
        ggml_tensor * t_up = ggml_mul_mat(ctx.get(), tensors.ffn_up_w, t_post);
        if (tensors.ffn_up_b) {
            ggml_tensor * b = ggml_cast(ctx.get(), tensors.ffn_up_b, GGML_TYPE_F32);
            b = ggml_repeat(ctx.get(), b, t_up);
            t_up = ggml_add(ctx.get(), t_up, b);
        }
        ggml_tensor * t_ffn_out = ggml_mul(ctx.get(), ggml_silu(ctx.get(), t_gate), t_up);
        t_ffn_out = ggml_mul_mat(ctx.get(), tensors.ffn_down_w, t_ffn_out);
        if (tensors.ffn_down_b) {
            ggml_tensor * b = ggml_cast(ctx.get(), tensors.ffn_down_b, GGML_TYPE_F32);
            b = ggml_repeat(ctx.get(), b, t_ffn_out);
            t_ffn_out = ggml_add(ctx.get(), t_ffn_out, b);
        }

        ggml_tensor * t_resid_ffn = ggml_add(ctx.get(), t_ha, t_ffn_out);
        ggml_set_name(t_resid_ffn, "eagle3_resid_ffn");
        ggml_tensor * t_hidden_out = ggml_cont(ctx.get(), t_resid_ffn);

        // ---- Mark per-depth outputs ----
        ggml_set_output(t_parent);
        ggml_set_output(t_base_id);
        ggml_set_output(t_selected_logprob);
        ggml_set_output(t_hidden_out);

        v_parents[d]  = t_parent;
        v_tokens[d]   = t_base_id;
        v_logprobs[d] = t_selected_logprob;
        v_hidden[d]   = t_hidden_out;

        // ---- Forward to next depth ----
        cur_hidden  = t_hidden_out;
        cur_logprob = t_selected_logprob;
        cur_bk      = t_bk_written;
        cur_bv      = t_bv_written;
    }

    // ===================== BUILD GRAPH =====================

    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), max_nodes, /* grads = */ false);

    // Expand all per-depth outputs (pulls in all dependencies).
    for (int32_t d = 0; d < max_depth; ++d) {
        ggml_build_forward_expand(gf, v_parents[d]);
        ggml_build_forward_expand(gf, v_tokens[d]);
        ggml_build_forward_expand(gf, v_logprobs[d]);
        ggml_build_forward_expand(gf, v_hidden[d]);
    }
    // Ensure final KV writes are in the graph.
    ggml_build_forward_expand(gf, cur_bk);
    ggml_build_forward_expand(gf, cur_bv);

    // Allocate compute buffer.
    ggml_backend_buffer_ptr buf_compute;
    if (rt.buft_compute) {
        buf_compute.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), rt.buft_compute));
    }
    if (!buf_compute) { return false; }

    mg.ctx               = std::move(ctx);
    mg.buf_compute       = std::move(buf_compute);
    mg.gf                = gf;
    mg.n_beams           = n_beams;
    mg.k                 = k;
    mg.kv_capacity       = kv_cap;
    mg.max_depth         = max_depth;
    mg.max_fork_rows     = max_fork_rows;
    mg.prob_threshold    = prob_threshold;
    mg.t_hidden_in        = t_hidden_in;
    mg.t_beam_logprob     = t_beam_logprob;
    mg.t_pos_all          = t_pos_all;
    mg.t_mask_all         = t_mask_all;
    mg.t_kv_write_idxs_all = t_kv_idxs_all;
    mg.t_fork_row_offsets = t_fork_row_offsets;
    mg.t_fork_dst_idxs   = t_fork_dst_idxs;
    mg.out_parents        = std::move(v_parents);
    mg.out_tokens         = std::move(v_tokens);
    mg.out_logprobs       = std::move(v_logprobs);
    mg.out_hidden         = std::move(v_hidden);
    return true;
}

} // namespace

// The cached prefill state, stored on the runtime and reused across calls.
struct llama_eagle3_runtime::prefill_cached_state {
    prefill_kv_cache cache;
    kv_only_graph    kv_graph;
    root_graph       r_graph;
    bool             warmed_up = false;
};

// Cached rollout state: fixed-capacity KV batch + graph + fork buffer,
// reused across depths within a rollout and across rollouts between target passes.
struct llama_eagle3_runtime::rollout_cached_state {
    std::shared_ptr<llama_eagle3_rollout_batch> batch;
    rollout_step_graph graph;
    rollout_fork_buffer fork_buf;
    bool warmed_up = false;
    int32_t rollout_start = -1;     // position where current rollout begins (-1 = not populated)
    int32_t prefix_len = 0;         // number of prefix rows valid in all beam slots
    int32_t n_beams_populated = 0;  // how many beam slots have the valid prefix
    int32_t max_depth = 0;          // max rollout depth (fork buffer sizing)

    // GPU-fused rollout path.
    fused_d2t_gpu          d2t_gpu;
    fused_journal          journal;
    fused_depth_graph_data fused_graph;
    fused_mega_graph_data  mega_graph;
    bool fused_warmed_up = false;
    bool mega_warmed_up  = false;
};

bool llama_eagle3_rollout_batch_ensure(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t n_beams,
        int32_t kv_capacity,
        llama_eagle3_rollout_batch & batch) {
    return llama_eagle3_rollout_batch_ensure_impl(model, rt, n_beams, kv_capacity, batch);
}

bool llama_eagle3_rollout_batch_bind_slot(
        const llama_eagle3_model & model,
        std::shared_ptr<llama_eagle3_rollout_batch> batch,
        int32_t slot,
        llama_eagle3_state & state,
        int32_t past_len) {
    return llama_eagle3_rollout_batch_bind_slot_impl(model, std::move(batch), slot, state, past_len);
}

bool llama_eagle3_rollout_batch_copy_state_to_slot(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const llama_eagle3_state & src,
        std::shared_ptr<llama_eagle3_rollout_batch> batch,
        int32_t slot,
        llama_eagle3_state & dst_state) {
    return llama_eagle3_rollout_batch_copy_state_to_slot_impl(model, rt, src, std::move(batch), slot, dst_state);
}

bool llama_eagle3_rollout_populate_prefix(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const llama_eagle3_state & root_state,
        int32_t n_beams,
        int32_t max_depth) {
    if (!rt.backend_compute || !rt.buft_compute || n_beams <= 0 || max_depth <= 0) {
        return false;
    }
    if (!root_state.dev || !root_state.dev->t_k || !root_state.dev->t_v) {
        return false;
    }
    if (root_state.dev->rollout_kv_start > 0) {
        // Can't populate from a partial-KV state; caller must materialize first.
        return false;
    }

    const auto & hp = model.hparams;
    const int32_t past_len = root_state.past_len;
    if (past_len <= 0) {
        return false;
    }

    // Ensure rollout cached state exists.
    if (!rt.rollout_state) {
        rt.rollout_state = std::make_shared<llama_eagle3_runtime::rollout_cached_state>();
    }
    auto & rs = *rt.rollout_state;

    // Ensure batch capacity.
    int32_t needed_cap = past_len + max_depth;
    if (needed_cap <= 256)       needed_cap = 256;
    else if (needed_cap <= 512)  needed_cap = 512;
    else if (needed_cap <= 1024) needed_cap = 1024;
    else needed_cap = ((needed_cap + 511) / 512) * 512;

    if (!rs.batch || rs.batch->n_beams < n_beams || rs.batch->kv_capacity < needed_cap) {
        rs.batch = std::make_shared<llama_eagle3_rollout_batch>();
        if (!llama_eagle3_rollout_batch_ensure_impl(model, rt, n_beams, needed_cap, *rs.batch)) {
            return false;
        }
        rs.graph = {};
        rs.warmed_up = false;
    }

    // NOTE: we intentionally do NOT build or warm up the rollout graph here.
    // The graph is built lazily in step_batch_from_parents with the caller's
    // hidden_in_dim, which may differ from hidden_concat * target_hidden_size
    // (during rollout, the input is the eagle head's output hidden_size, not
    // the full concatenated teacher hidden).

    // Ensure fork buffer.
    if (rs.fork_buf.n_beams < n_beams || rs.fork_buf.max_depth < max_depth) {
        if (!alloc_fork_buffer(model, rt, max_depth, n_beams, rs.fork_buf)) {
            return false;
        }
    }

    // Copy root_state's KV to ALL beam slots.
    const size_t kv_row_bytes = (size_t) hp.head_dim * hp.num_kv_heads * sizeof(float);
    const size_t beam_kv_bytes = (size_t) rs.batch->kv_capacity * kv_row_bytes;
    const size_t prefix_bytes = (size_t) past_len * kv_row_bytes;

    for (int32_t ib = 0; ib < n_beams; ++ib) {
        const size_t dst_offset = (size_t) ib * beam_kv_bytes;
        if (!tensor_copy_bytes_async(
                    rt.backend_compute.get(), rt.backend_compute.get(),
                    root_state.dev->t_k, 0,
                    rs.batch->t_k, dst_offset, prefix_bytes)) {
            return false;
        }
        if (!tensor_copy_bytes_async(
                    rt.backend_compute.get(), rt.backend_compute.get(),
                    root_state.dev->t_v, 0,
                    rs.batch->t_v, dst_offset, prefix_bytes)) {
            return false;
        }
    }

    rs.rollout_start = past_len;
    rs.prefix_len = past_len;
    rs.n_beams_populated = n_beams;
    rs.max_depth = max_depth;
    return true;
}

bool llama_eagle3_rollout_materialize_state(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const llama_eagle3_state & partial_state,
        llama_eagle3_state & full_state_out) {
    if (!partial_state.dev) {
        return false;
    }
    if (partial_state.dev->rollout_kv_start <= 0) {
        // Already full KV; just copy.
        full_state_out = partial_state;
        return true;
    }
    if (!rt.rollout_state || !rt.rollout_state->batch) {
        return false;
    }

    const auto & hp = model.hparams;
    const auto & batch = *rt.rollout_state->batch;
    const int32_t past_len = partial_state.past_len;
    const int32_t rollout_start = partial_state.dev->rollout_kv_start;
    const int32_t n_rollout_rows = past_len - rollout_start;

    if (rollout_start > rt.rollout_state->prefix_len || n_rollout_rows < 0) {
        return false;
    }

    // Allocate full individual state.
    std::shared_ptr<llama_eagle3_state::device_state> full_dev;
    const int32_t full_cap = choose_kv_capacity(past_len, 0, 0);
    if (!alloc_state_device(model, rt, full_cap, full_dev)) {
        return false;
    }

    const size_t kv_row_bytes = (size_t) hp.head_dim * hp.num_kv_heads * sizeof(float);

    // Copy prefix from batch slot 0 (all slots have the same prefix).
    if (rollout_start > 0) {
        const size_t prefix_bytes = (size_t) rollout_start * kv_row_bytes;
        if (!tensor_copy_bytes_async(
                    rt.backend_compute.get(), rt.backend_compute.get(),
                    batch.t_k, 0, full_dev->t_k, 0, prefix_bytes)) {
            return false;
        }
        if (!tensor_copy_bytes_async(
                    rt.backend_compute.get(), rt.backend_compute.get(),
                    batch.t_v, 0, full_dev->t_v, 0, prefix_bytes)) {
            return false;
        }
    }

    // Copy rollout rows from partial state.
    if (n_rollout_rows > 0 && partial_state.dev->t_k && partial_state.dev->t_v) {
        const size_t rollout_bytes = (size_t) n_rollout_rows * kv_row_bytes;
        const size_t dst_offset = (size_t) rollout_start * kv_row_bytes;
        if (!tensor_copy_bytes_async(
                    rt.backend_compute.get(), rt.backend_compute.get(),
                    partial_state.dev->t_k, 0, full_dev->t_k, dst_offset, rollout_bytes)) {
            return false;
        }
        if (!tensor_copy_bytes_async(
                    rt.backend_compute.get(), rt.backend_compute.get(),
                    partial_state.dev->t_v, 0, full_dev->t_v, dst_offset, rollout_bytes)) {
            return false;
        }
    }

    // Copy hidden.
    if (partial_state.dev->t_hidden) {
        if (!tensor_copy_bytes_async(
                    rt.backend_compute.get(), rt.backend_compute.get(),
                    partial_state.dev->t_hidden, 0,
                    full_dev->t_hidden, 0,
                    (size_t) hp.hidden_size * sizeof(float))) {
            return false;
        }
    }

    full_dev->past_len = past_len;
    full_dev->rollout_kv_start = 0;
    full_state_out.dev = std::move(full_dev);
    full_state_out.past_len = past_len;
    full_state_out.rollout_only = false;
    full_state_out.hidden.clear();
    full_state_out.k.clear();
    full_state_out.v.clear();
    return true;
}

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
    rt.flash_attn     = cparams.flash_attn;
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

    // Pre-build and warm up prefill graphs so the first prefill call doesn't pay
    // ~20ms of CUDA JIT / cuBLAS workspace setup.
    if (model && rt.backend_compute && rt.buft_compute) {
        const int32_t init_cap = 256; // initial capacity; grows automatically if needed
        auto ps = std::make_shared<llama_eagle3_runtime::prefill_cached_state>();

        if (alloc_prefill_cache(*model, rt, init_cap, ps->cache) &&
            build_kv_only_graph_impl(*model, rt, ps->cache, init_cap, ps->kv_graph) &&
            build_root_graph_impl(*model, rt, ps->cache, ps->r_graph)) {
            // Warm up: execute each graph once with dummy data to trigger CUDA setup.
            // Zero-init index tensors so set_rows doesn't OOB on garbage indices.
            auto zero_tensor = [&](ggml_tensor * t) {
                if (!t) return;
                const size_t nbytes = ggml_nbytes(t);
                std::vector<uint8_t> zeros(nbytes, 0);
                ggml_backend_tensor_set(t, zeros.data(), 0, nbytes);
            };
            zero_tensor(ps->kv_graph.t_k_idxs);
            zero_tensor(ps->kv_graph.t_v_idxs);
            zero_tensor(ps->r_graph.t_k_idxs);
            zero_tensor(ps->r_graph.t_v_idxs);
            ggml_backend_graph_compute_async(rt.backend_compute.get(), ps->kv_graph.gf);
            ggml_backend_graph_compute_async(rt.backend_compute.get(), ps->r_graph.gf);
            ggml_backend_synchronize(rt.backend_compute.get());
            ps->warmed_up = true;
            rt.prefill_state = std::move(ps);
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
            if (!tensor_copy_bytes_async(
                        rt.backend_compute.get(),
                        rt.backend_compute.get(),
                        state.dev->t_hidden,
                        0,
                        rt.topk_graph.t_hidden,
                        0,
                        (size_t) hp.hidden_size * sizeof(float))) {
                return false;
            }
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
                        if (!tensor_copy_bytes_async(
                                    rt.backend_compute.get(),
                                    rt.backend_compute.get(),
                                    st->dev->t_hidden,
                                    0,
                                    dst,
                                    0,
                                    (size_t) hp.hidden_size * sizeof(float))) {
                            return false;
                        }
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
        float prob_threshold,
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
        if (build_select_batch_graph(model, rt, n_beams, k, n_select, prob_threshold, rt.select_batch_graph)) {
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
                    if (!tensor_copy_bytes_async(
                                rt.backend_compute.get(),
                                rt.backend_compute.get(),
                                st->dev->t_hidden,
                                0,
                                dst,
                                0,
                                (size_t) hp.hidden_size * sizeof(float))) {
                        return false;
                    }
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
        float prob_threshold,
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

    if (!build_select_batch_graph(model, rt, n_beams, k, n_select, prob_threshold, rt.select_batch_graph)) {
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
            if (!tensor_copy_bytes_async(
                        rt.backend_compute.get(),
                        rt.backend_compute.get(),
                        st->dev->t_hidden,
                        0,
                        dst,
                        0,
                        (size_t) hp.hidden_size * sizeof(float))) {
                return false;
            }
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
    out.t_selected_draft = graph.t_selected_draft;
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
        float prob_threshold,
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

    if (!build_select_batch_graph(model, rt, n_beams, k, n_select, prob_threshold, rt.select_batch_graph)) {
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
            if (!tensor_copy_bytes_async(
                        rt.backend_compute.get(),
                        rt.backend_compute.get(),
                        st->dev->t_hidden,
                        0,
                        dst,
                        0,
                        (size_t) hp.hidden_size * sizeof(float))) {
                return false;
            }
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
    out.t_selected_draft = graph.t_selected_draft;
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
        float prob_threshold,
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
        if (llama_eagle3_select_state_slots_device(model, rt, states, active_mask, beam_logprob, k, prob_threshold, device_out)) {
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
    if (!llama_eagle3_select_state_batch(model, rt, active_states, active_logprob, k, prob_threshold, active_linear, active_draft_idx, active_selected_logprob)) {
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

        // --- Cached rollout graph path (fixed-capacity KV, no graph rebuild) ---
        {
            // Ensure rollout cached state exists.
            if (!rt.rollout_state) {
                rt.rollout_state = std::make_shared<llama_eagle3_runtime::rollout_cached_state>();
            }
            auto & rs = *rt.rollout_state;

            // Ensure rollout batch has enough capacity and beams.
            const int32_t required_len = past_len + 1;
            int32_t needed_cap = required_len + std::max(0, reserve_kv);
            if (needed_cap <= 256)  needed_cap = 256;
            else if (needed_cap <= 512)  needed_cap = 512;
            else if (needed_cap <= 1024) needed_cap = 1024;
            else needed_cap = ((needed_cap + 511) / 512) * 512;

            const int32_t needed_beams = n_beams; // could pad to beam_width for reuse

            if (!rs.batch || rs.batch->n_beams < needed_beams || rs.batch->kv_capacity < needed_cap) {
                rs.batch = std::make_shared<llama_eagle3_rollout_batch>();
                if (!llama_eagle3_rollout_batch_ensure_impl(model, rt, needed_beams, needed_cap, *rs.batch)) {
                    goto fallback_old_path;
                }
                rs.graph = {};
                rs.warmed_up = false;
            }

            // Build graph if needed (keyed on n_beams + kv_capacity).
            if (rs.graph.n_beams != rs.batch->n_beams || rs.graph.kv_capacity != rs.batch->kv_capacity) {
                if (!build_rollout_step_graph_impl(model, rt, *rs.batch, hidden_in_dim, rs.graph)) {
                    goto fallback_old_path;
                }
                rs.warmed_up = false;
            }

            // Warm up once.
            if (!rs.warmed_up) {
                ggml_backend_graph_compute_async(rt.backend_compute.get(), rs.graph.gf);
                ggml_backend_synchronize(rt.backend_compute.get());
                rs.warmed_up = true;
            }

            auto & graph = rs.graph;
            auto & batch = *rs.batch;
            const int32_t kv_cap = batch.kv_capacity;
            const size_t kv_row_bytes = (size_t) hp.head_dim * hp.num_kv_heads * sizeof(float);
            const size_t beam_kv_bytes = (size_t) kv_cap * kv_row_bytes;

            // Determine whether we can use the fork path (O(depth) copies instead of O(past_len)).
            // The fork path requires that populate_prefix has been called for this rollout.
            const bool use_fork_path = (rs.rollout_start >= 0 &&
                                         rs.prefix_len > 0 &&
                                         rs.n_beams_populated >= n_beams &&
                                         past_len >= rs.rollout_start &&
                                         rs.max_depth > 0 &&
                                         rs.fork_buf.t_k);
            const int32_t n_rollout_rows = use_fork_path ? (past_len - rs.rollout_start) : 0;

            // -- Copy parent hidden + KV into batch slots ---------------------
            for (int32_t ib = 0; ib < n_beams; ++ib) {
                const llama_eagle3_state * st = parent_states[(size_t) ib];
                if (!st || !out_states[(size_t) ib]) {
                    goto fallback_old_path;
                }

                // Hidden input (same for both paths)
                if (st->dev && st->dev->t_hidden) {
                    if (!tensor_copy_bytes_async(
                                rt.backend_compute.get(), rt.backend_compute.get(),
                                st->dev->t_hidden, 0,
                                graph.t_hidden_in_b, (size_t) ib * (size_t) graph.t_hidden_in_b->nb[1],
                                (size_t) hidden_in_dim * sizeof(float))) {
                        goto fallback_old_path;
                    }
                } else if (!st->hidden.empty()) {
                    ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_hidden_in_b,
                            st->hidden.data(), (size_t) ib * (size_t) graph.t_hidden_in_b->nb[1],
                            (size_t) hidden_in_dim * sizeof(float));
                } else {
                    goto fallback_old_path;
                }

                // KV copy-in
                if (use_fork_path) {
                    // Fork path: copy only rollout rows (O(depth)) from parent → batch slot.
                    // The prefix (0..rollout_start-1) is already in the batch from populate_prefix.
                    if (n_rollout_rows > 0 && st->dev && st->dev->t_k && st->dev->t_v) {
                        const size_t copy_bytes = (size_t) n_rollout_rows * kv_row_bytes;
                        // Source offset depends on whether parent has full or rollout-only KV.
                        const size_t src_offset = (st->dev->rollout_kv_start > 0)
                            ? 0  // rollout-only: rows start at offset 0 in individual tensor
                            : (size_t) rs.rollout_start * kv_row_bytes;  // full KV: rows at rollout_start
                        const size_t dst_offset = (size_t) ib * beam_kv_bytes + (size_t) rs.rollout_start * kv_row_bytes;
                        if (!tensor_copy_bytes_async(
                                    rt.backend_compute.get(), rt.backend_compute.get(),
                                    st->dev->t_k, src_offset, batch.t_k, dst_offset, copy_bytes)) {
                            goto fallback_old_path;
                        }
                        if (!tensor_copy_bytes_async(
                                    rt.backend_compute.get(), rt.backend_compute.get(),
                                    st->dev->t_v, src_offset, batch.t_v, dst_offset, copy_bytes)) {
                            goto fallback_old_path;
                        }
                    }
                    // n_rollout_rows == 0 at depth 0: prefix already in batch, nothing to copy.
                } else {
                    // Full-copy path: copy all past_len rows from parent → batch slot.
                    if (past_len > 0) {
                        if (st->dev && st->dev->t_k && st->dev->t_v && st->dev->kv_capacity >= past_len) {
                            const size_t copy_bytes = (size_t) past_len * kv_row_bytes;
                            const size_t dst_offset = (size_t) ib * beam_kv_bytes;
                            if (!tensor_copy_bytes_async(
                                        rt.backend_compute.get(), rt.backend_compute.get(),
                                        st->dev->t_k, 0, batch.t_k, dst_offset, copy_bytes)) {
                                goto fallback_old_path;
                            }
                            if (!tensor_copy_bytes_async(
                                        rt.backend_compute.get(), rt.backend_compute.get(),
                                        st->dev->t_v, 0, batch.t_v, dst_offset, copy_bytes)) {
                                goto fallback_old_path;
                            }
                        } else {
                            goto fallback_old_path;
                        }
                    }
                }
            }

            // Token ids
            ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_tok_b,
                    input_ids.data(), 0, (size_t) n_beams * sizeof(llama_token));

            // Positions
            std::vector<int32_t> pos_host((size_t) n_beams, past_len);
            ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_pos_b,
                    pos_host.data(), 0, (size_t) n_beams * sizeof(int32_t));

            // KV write indices: beam b writes to row b * kv_capacity + past_len
            std::vector<int32_t> kv_idxs((size_t) n_beams);
            for (int32_t ib = 0; ib < n_beams; ++ib) {
                kv_idxs[(size_t) ib] = ib * kv_cap + past_len;
            }
            ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_k_idxs_b,
                    kv_idxs.data(), 0, (size_t) n_beams * sizeof(int32_t));
            ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_v_idxs_b,
                    kv_idxs.data(), 0, (size_t) n_beams * sizeof(int32_t));

            // Mask: 0 for positions [0, past_len], -inf for rest (broadcast across beams)
            {
                std::vector<ggml_fp16_t> mask_data((size_t) kv_cap, ggml_fp32_to_fp16(-INFINITY));
                for (int32_t i = 0; i <= past_len; ++i) {
                    mask_data[(size_t) i] = ggml_fp32_to_fp16(0.0f);
                }
                ggml_backend_tensor_set(graph.t_mask, mask_data.data(), 0,
                        (size_t) kv_cap * sizeof(ggml_fp16_t));
            }

            // -- Execute graph ------------------------------------------------
            const ggml_status status = ggml_backend_graph_compute_async(
                    rt.backend_compute.get(), graph.gf);
            if (status != GGML_STATUS_SUCCESS) {
                goto fallback_old_path;
            }

            // -- Extract outputs: hidden + KV per beam -------------------------
            for (int32_t ib = 0; ib < n_beams; ++ib) {
                llama_eagle3_state * out_st = out_states[(size_t) ib];
                if (!out_st) {
                    return false;
                }

                if (use_fork_path) {
                    // Fork path: allocate small state, copy only rollout rows (O(depth+1)).
                    const int32_t n_out_rollout = n_rollout_rows + 1;

                    const bool can_reuse = out_st->dev &&
                            out_st->dev.use_count() == 1 &&
                            out_st->dev->t_hidden &&
                            out_st->dev->kv_capacity >= n_out_rollout;

                    std::shared_ptr<llama_eagle3_state::device_state> next_dev = can_reuse ? out_st->dev : nullptr;
                    if (!next_dev) {
                        if (!alloc_state_device(model, rt, rs.max_depth, next_dev)) {
                            return false;
                        }
                    }

                    // Copy hidden output
                    const size_t h_offset = (size_t) ib * (size_t) graph.t_hidden_out_b->nb[1];
                    if (!tensor_copy_bytes_async(
                                rt.backend_compute.get(), rt.backend_compute.get(),
                                graph.t_hidden_out_b, h_offset,
                                next_dev->t_hidden, 0,
                                (size_t) hp.hidden_size * sizeof(float))) {
                        return false;
                    }

                    // Copy rollout rows (rollout_start..rollout_start+n_out_rollout-1) from batch → individual
                    if (next_dev->t_k && next_dev->t_v && n_out_rollout > 0) {
                        const size_t src_offset = (size_t) ib * beam_kv_bytes + (size_t) rs.rollout_start * kv_row_bytes;
                        const size_t copy_bytes = (size_t) n_out_rollout * kv_row_bytes;
                        if (!tensor_copy_bytes_async(
                                    rt.backend_compute.get(), rt.backend_compute.get(),
                                    batch.t_k, src_offset, next_dev->t_k, 0, copy_bytes)) {
                            return false;
                        }
                        if (!tensor_copy_bytes_async(
                                    rt.backend_compute.get(), rt.backend_compute.get(),
                                    batch.t_v, src_offset, next_dev->t_v, 0, copy_bytes)) {
                            return false;
                        }
                    }

                    next_dev->past_len = required_len;
                    next_dev->rollout_kv_start = rs.rollout_start;
                    out_st->dev = std::move(next_dev);
                    out_st->hidden.clear();
                    out_st->k.clear();
                    out_st->v.clear();
                    out_st->past_len = required_len;
                    out_st->rollout_only = true;
                } else {
                    // Full-copy path: copy all required_len rows.
                    const bool can_reuse = out_st->dev &&
                            out_st->dev.use_count() == 1 &&
                            out_st->dev->t_hidden &&
                            out_st->dev->kv_capacity >= required_len + std::max(0, reserve_kv);

                    std::shared_ptr<llama_eagle3_state::device_state> next_dev = can_reuse ? out_st->dev : nullptr;
                    if (!next_dev) {
                        const int32_t cur_cap = out_st->dev ? out_st->dev->kv_capacity : 0;
                        const int32_t desired = choose_kv_capacity(required_len, reserve_kv, cur_cap);
                        if (!alloc_state_device(model, rt, desired, next_dev)) {
                            return false;
                        }
                    }

                    // Copy hidden output
                    const size_t h_offset = (size_t) ib * (size_t) graph.t_hidden_out_b->nb[1];
                    if (!tensor_copy_bytes_async(
                                rt.backend_compute.get(), rt.backend_compute.get(),
                                graph.t_hidden_out_b, h_offset,
                                next_dev->t_hidden, 0,
                                (size_t) hp.hidden_size * sizeof(float))) {
                        return false;
                    }

                    // Copy full KV from batch slot → child state
                    if (next_dev->t_k && next_dev->t_v) {
                        const size_t src_offset = (size_t) ib * beam_kv_bytes;
                        const size_t copy_bytes = (size_t) required_len * kv_row_bytes;
                        if (!tensor_copy_bytes_async(
                                    rt.backend_compute.get(), rt.backend_compute.get(),
                                    batch.t_k, src_offset,
                                    next_dev->t_k, 0, copy_bytes)) {
                            return false;
                        }
                        if (!tensor_copy_bytes_async(
                                    rt.backend_compute.get(), rt.backend_compute.get(),
                                    batch.t_v, src_offset,
                                    next_dev->t_v, 0, copy_bytes)) {
                            return false;
                        }
                    }

                    next_dev->past_len = required_len;
                    next_dev->rollout_kv_start = 0;
                    out_st->dev = std::move(next_dev);
                    out_st->hidden.clear();
                    out_st->k.clear();
                    out_st->v.clear();
                    out_st->past_len = required_len;
                    out_st->rollout_only = false;
                }
            }
            return true;
        }
        fallback_old_path:

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
                        if (!tensor_copy_bytes_async(
                                    rt.backend_compute.get(),
                                    rt.backend_compute.get(),
                                    st->dev->t_hidden,
                                    0,
                                    graph.t_hidden_in[(size_t) ib],
                                    0,
                                    (size_t) hidden_in_dim * sizeof(float))) {
                            return false;
                        }
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

                        if (!tensor_copy_bytes_async(
                                    rt.backend_compute.get(),
                                    rt.backend_compute.get(),
                                    graph.t_hidden_out[(size_t) ib],
                                    0,
                                    next_dev->t_hidden,
                                    0,
                                    (size_t) hp.hidden_size * sizeof(float))) {
                            return false;
                        }
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

                if (!tensor_copy_bytes_async(
                            rt.backend_compute.get(),
                            rt.backend_compute.get(),
                            graph.t_hidden_out,
                            0,
                            next_dev->t_hidden,
                            0,
                            (size_t) hp.hidden_size * sizeof(float))) {
                    return false;
                }
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

bool llama_eagle3_step_multi_from_hidden_capture(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const llama_eagle3_state & base_state,
        const float * first_hidden_in,
        int32_t hidden_in_dim,
        const std::vector<const ggml_tensor *> & hidden_capture,
        size_t token_idx_start,
        const std::vector<llama_token> & input_ids,
        int32_t split_after,
        llama_eagle3_state * split_state_out,
        llama_eagle3_state & final_state_out,
        std::vector<float> * logits_out) {
    const auto & hp = model.hparams;
    const bool with_logits = logits_out != nullptr;
    const int32_t n_tokens = (int32_t) input_ids.size();
    const int32_t n_capture_tokens = n_tokens - (first_hidden_in ? 1 : 0);

    if (n_tokens == 0) {
        if (split_after != 0) {
            return false;
        }
        final_state_out = base_state;
        return true;
    }

    if (n_capture_tokens < 0) {
        return false;
    }

    if (split_after < 0 || split_after > n_tokens) {
        return false;
    }

    if (n_capture_tokens > 0 && (int32_t) hidden_capture.size() != hp.hidden_concat) {
        return false;
    }

    if (rt.backend_compute && rt.buft_compute) {
        const uint64_t key = make_step_multi_graph_key(base_state.past_len, hidden_in_dim, n_tokens, with_logits);
        if (rt.step_multi_graph_key != key) {
            rt.step_multi_graph = {};
            if (build_step_multi_graph(model, rt, base_state.past_len, hidden_in_dim, n_tokens, with_logits, rt.step_multi_graph)) {
                rt.step_multi_graph_key = key;
            }
        }

        if (rt.step_multi_graph_key == key && rt.step_multi_graph.gf) {
            auto & graph = rt.step_multi_graph;
            const size_t block = (size_t) hp.head_dim * hp.num_kv_heads;
            const size_t past_bytes = (size_t) base_state.past_len * block * sizeof(float);
            const size_t hidden_step_bytes = (size_t) hidden_in_dim * sizeof(float);

            if (first_hidden_in) {
                ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_hidden_in, first_hidden_in, 0, hidden_step_bytes);
            }

            for (int32_t it = 0; it < n_capture_tokens; ++it) {
                const size_t dst_col = (size_t) (it + (first_hidden_in ? 1 : 0));
                for (int32_t il = 0; il < hp.hidden_concat; ++il) {
                    ggml_tensor * src = const_cast<ggml_tensor *>(hidden_capture[(size_t) il]);
                    if (!src || src->type != GGML_TYPE_F32 || src->ne[0] != hp.target_hidden_size ||
                        token_idx_start + (size_t) it >= (size_t) src->ne[1]) {
                        return false;
                    }

                    const size_t n_bytes = (size_t) hp.target_hidden_size * sizeof(float);
                    const size_t src_offset = (token_idx_start + (size_t) it) * (size_t) src->nb[1];
                    const size_t dst_offset = dst_col * (size_t) graph.t_hidden_in->nb[1] + (size_t) il * n_bytes;
                    if (!tensor_copy_bytes_async(
                                rt.target_backend ? rt.target_backend : rt.backend_compute.get(),
                                rt.backend_compute.get(),
                                src,
                                src_offset,
                                graph.t_hidden_in,
                                dst_offset,
                                n_bytes)) {
                        return false;
                    }
                }
            }

            ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_tok, input_ids.data(), 0, (size_t) n_tokens * sizeof(llama_token));

            std::vector<int32_t> pos((size_t) n_tokens);
            for (int32_t i = 0; i < n_tokens; ++i) {
                pos[(size_t) i] = base_state.past_len + i;
            }
            ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_pos, pos.data(), 0, pos.size() * sizeof(int32_t));

            if (graph.t_k_past_input && base_state.past_len > 0) {
                if (base_state.dev && base_state.dev->t_k && base_state.dev->kv_capacity >= base_state.past_len) {
                    if (!tensor_copy_3d_prefix_async(
                                rt.backend_compute.get(),
                                base_state.dev->t_k,
                                graph.t_k_past_input,
                                hp.head_dim,
                                hp.num_kv_heads,
                                base_state.past_len)) {
                        return false;
                    }
                } else if (base_state.k.size() * sizeof(float) >= past_bytes) {
                    ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_k_past_input, base_state.k.data(), 0, past_bytes);
                } else {
                    return false;
                }
            }

            if (graph.t_v_past_input && base_state.past_len > 0) {
                if (base_state.dev && base_state.dev->t_v && base_state.dev->kv_capacity >= base_state.past_len) {
                    if (!tensor_copy_3d_prefix_async(
                                rt.backend_compute.get(),
                                base_state.dev->t_v,
                                graph.t_v_past_input,
                                hp.head_dim,
                                hp.num_kv_heads,
                                base_state.past_len)) {
                        return false;
                    }
                } else if (base_state.v.size() * sizeof(float) >= past_bytes) {
                    ggml_backend_tensor_set_async(rt.backend_compute.get(), graph.t_v_past_input, base_state.v.data(), 0, past_bytes);
                } else {
                    return false;
                }
            }

            const ggml_status status = ggml_backend_graph_compute_async(rt.backend_compute.get(), graph.gf);
            if (status == GGML_STATUS_SUCCESS) {
                const auto materialize_state = [&](llama_eagle3_state & out_state, int32_t n_done) -> bool {
                    if (n_done <= 0) {
                        out_state = base_state;
                        return true;
                    }

                    const int32_t required_len = base_state.past_len + n_done;
                    const bool can_reuse = out_state.dev &&
                            out_state.dev.use_count() == 1 &&
                            out_state.dev->t_hidden &&
                            out_state.dev->kv_capacity >= required_len;

                    std::shared_ptr<llama_eagle3_state::device_state> next_dev = can_reuse ? out_state.dev : nullptr;
                    if (!next_dev) {
                        const int32_t cur_capacity = out_state.dev ? out_state.dev->kv_capacity : 0;
                        const int32_t desired_capacity = choose_kv_capacity(required_len, /* reserve_kv = */ 0, cur_capacity);
                        if (!alloc_state_device(model, rt, desired_capacity, next_dev)) {
                            return false;
                        }
                    }

                    const size_t hidden_offset = (size_t) (n_done - 1) * (size_t) graph.t_hidden_out->nb[1];
                    if (!tensor_copy_bytes_async(
                                rt.backend_compute.get(),
                                rt.backend_compute.get(),
                                graph.t_hidden_out,
                                hidden_offset,
                                next_dev->t_hidden,
                                0,
                                (size_t) hp.hidden_size * sizeof(float))) {
                        return false;
                    }

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
                    out_state.dev = std::move(next_dev);
                    out_state.hidden.clear();
                    out_state.k.clear();
                    out_state.v.clear();
                    out_state.past_len = required_len;
                    return true;
                };

                if (split_after > 0 && split_state_out) {
                    if (!materialize_state(*split_state_out, split_after)) {
                        return false;
                    }
                }
                if (!materialize_state(final_state_out, n_tokens)) {
                    return false;
                }

                if (with_logits && graph.t_logits) {
                    logits_out->resize(hp.draft_vocab_size);
                    ggml_backend_tensor_get_async(rt.backend_compute.get(), graph.t_logits, logits_out->data(), 0, (size_t) hp.draft_vocab_size * sizeof(float));
                    ggml_backend_synchronize(rt.backend_compute.get());
                }

                return true;
            }
        }
    }

    std::vector<float> hidden_concat;
    llama_eagle3_state cur = base_state;
    for (int32_t i = 0; i < n_tokens; ++i) {
        const float * hidden_in = nullptr;
        if (first_hidden_in && i == 0) {
            hidden_in = first_hidden_in;
        } else {
            const size_t capture_idx = token_idx_start + (size_t) i - (first_hidden_in ? 1u : 0u);
            if (!build_hidden_concat_from_capture_host(model, hidden_capture, capture_idx, hidden_concat)) {
                return false;
            }
            hidden_in = hidden_concat.data();
        }

        std::vector<float> * step_logits_out = (with_logits && i == n_tokens - 1) ? logits_out : nullptr;
        if (!llama_eagle3_step(model, rt, cur, hidden_in, hidden_in_dim, input_ids[(size_t) i], step_logits_out, nullptr)) {
            return false;
        }

        if (split_after > 0 && split_state_out && i + 1 == split_after) {
            *split_state_out = cur;
        }
    }

    final_state_out = cur;
    return true;
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
                if (!tensor_copy_bytes_async(
                            rt.backend_compute.get(),
                            rt.backend_compute.get(),
                            state.dev->t_hidden,
                            0,
                            graph.t_hidden_in,
                            0,
                            (size_t) hidden_in_dim * sizeof(float))) {
                    return false;
                }
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

                if (!tensor_copy_bytes_async(
                            rt.backend_compute.get(),
                            rt.backend_compute.get(),
                            graph.t_hidden_out,
                            0,
                            next_dev->t_hidden,
                            0,
                            (size_t) hp.hidden_size * sizeof(float))) {
                    return false;
                }
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

// ---------------------------------------------------------------------------
// llama_eagle3_fused_rollout — GPU-fused SELECT+STEP beam search
// ---------------------------------------------------------------------------

bool llama_eagle3_fused_rollout_available(const llama_eagle3_runtime & rt) {
    if (!rt.backend_compute || !rt.buft_compute) {
        return false;
    }
    // Env override to disable fused path for correctness debugging.
    static const bool disabled = [] {
        const char * env = std::getenv("CASCADE_EAGLE_NO_FUSED");
        return env && (std::string(env) == "1" || std::string(env) == "true");
    }();
    return !disabled;
}

bool llama_eagle3_fused_rollout(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        const llama_eagle3_state & root_state,
        int32_t n_beams,
        int32_t max_depth,
        int32_t k,
        float prob_threshold,
        llama_eagle3_fused_rollout_result & result) {
    result = {};

    if (!llama_eagle3_fused_rollout_available(rt)) {
        return false;
    }
    if (!rt.rollout_state || !rt.rollout_state->batch) {
        return false;
    }
    if (n_beams <= 0 || max_depth <= 0 || k <= 0) {
        return false;
    }

    auto & rs = *rt.rollout_state;
    if (rs.rollout_start < 0 || rs.prefix_len <= 0 || rs.n_beams_populated < n_beams) {
        return false;
    }
    if (!root_state.dev || !root_state.dev->t_hidden) {
        return false;
    }

    const auto & hp = model.hparams;
    k = std::min(k, hp.draft_vocab_size);
    const int32_t rollout_start = rs.rollout_start;
    const int32_t kv_cap = rs.batch->kv_capacity;
    const int32_t max_fork_rows = std::max(0, max_depth - 1);

    // Ensure d2t table is on GPU.
    if (!upload_fused_d2t(model, rt, rs.d2t_gpu)) {
        return false;
    }

    // ================ TRY MEGA-GRAPH PATH ================
    // Single dispatch for all depths — eliminates inter-depth async writes.
    // NOTE: Currently opt-in only. The 840-node graph incurs ggml scheduler
    //       overhead that outweighs the inter-depth upload savings on small
    //       models. Phase 3 (CUDA graph capture) should fix this.
    static const bool use_mega_env = [] {
        const char * env = std::getenv("CASCADE_EAGLE_MEGA");
        return env && (std::string(env) == "1" || std::string(env) == "true");
    }();

    if (use_mega_env && build_fused_mega_graph(model, rt, *rs.batch, rs.d2t_gpu,
                                            n_beams, k, max_depth, prob_threshold, rs.mega_graph)) {
        auto & mg = rs.mega_graph;

        // Warm up once.
        if (!rs.mega_warmed_up) {
            ggml_backend_graph_compute_async(rt.backend_compute.get(), mg.gf);
            ggml_backend_synchronize(rt.backend_compute.get());
            rs.mega_warmed_up = true;
        }

        // ---- Set initial inputs (hidden, logprob, fork) ----
        {
            const size_t h_bytes = (size_t) hp.hidden_size * sizeof(float);
            tensor_copy_bytes_async(
                    rt.backend_compute.get(), rt.backend_compute.get(),
                    root_state.dev->t_hidden, 0,
                    mg.t_hidden_in, 0, h_bytes);
            if (n_beams > 1) {
                std::vector<float> zeros((size_t) hp.hidden_size * (size_t)(n_beams - 1), 0.0f);
                ggml_backend_tensor_set_async(rt.backend_compute.get(), mg.t_hidden_in,
                        zeros.data(), h_bytes, zeros.size() * sizeof(float));
            }
        }
        {
            std::vector<float> lp(n_beams, -1e30f);
            lp[0] = 0.0f;
            ggml_backend_tensor_set_async(rt.backend_compute.get(), mg.t_beam_logprob,
                    lp.data(), 0, lp.size() * sizeof(float));
        }
        if (max_fork_rows > 0 && mg.t_fork_row_offsets && mg.t_fork_dst_idxs) {
            const int32_t total_fork = max_fork_rows * n_beams;
            std::vector<float>   fork_offsets(total_fork);
            std::vector<int32_t> fork_dst(total_fork);
            for (int32_t b = 0; b < n_beams; ++b) {
                for (int32_t r = 0; r < max_fork_rows; ++r) {
                    const int32_t idx = b * max_fork_rows + r;
                    fork_offsets[idx] = (float)(rollout_start + r);
                    fork_dst[idx] = b * kv_cap + rollout_start + r;
                }
            }
            ggml_backend_tensor_set_async(rt.backend_compute.get(), mg.t_fork_row_offsets,
                    fork_offsets.data(), 0, total_fork * sizeof(float));
            ggml_backend_tensor_set_async(rt.backend_compute.get(), mg.t_fork_dst_idxs,
                    fork_dst.data(), 0, total_fork * sizeof(int32_t));
        }

        // ---- Upload all per-depth inputs at once ----
        {
            std::vector<int32_t> pos_all((size_t) n_beams * max_depth);
            std::vector<int32_t> kv_idxs_all((size_t) n_beams * max_depth);
            std::vector<ggml_fp16_t> mask_all((size_t) kv_cap * max_depth, ggml_fp32_to_fp16(-INFINITY));

            for (int32_t d = 0; d < max_depth; ++d) {
                const int32_t pos = rollout_start + d;

                // pos_all: all beams at depth d have the same position.
                for (int32_t b = 0; b < n_beams; ++b) {
                    pos_all[(size_t) d * n_beams + b] = pos;
                }

                // kv_write_idxs: each beam writes to its slot at the current position.
                for (int32_t b = 0; b < n_beams; ++b) {
                    kv_idxs_all[(size_t) d * n_beams + b] = b * kv_cap + pos;
                }

                // mask: positions [0..pos] are visible (0.0f), rest is -inf.
                const size_t mask_offset = (size_t) d * kv_cap;
                for (int32_t i = 0; i <= pos; ++i) {
                    mask_all[mask_offset + i] = ggml_fp32_to_fp16(0.0f);
                }
            }

            ggml_backend_tensor_set_async(rt.backend_compute.get(), mg.t_pos_all,
                    pos_all.data(), 0, pos_all.size() * sizeof(int32_t));
            ggml_backend_tensor_set_async(rt.backend_compute.get(), mg.t_kv_write_idxs_all,
                    kv_idxs_all.data(), 0, kv_idxs_all.size() * sizeof(int32_t));
            ggml_backend_tensor_set_async(rt.backend_compute.get(), mg.t_mask_all,
                    mask_all.data(), 0, mask_all.size() * sizeof(ggml_fp16_t));
        }

        // ---- Single dispatch ----
        const ggml_status status = ggml_backend_graph_compute_async(rt.backend_compute.get(), mg.gf);
        if (status != GGML_STATUS_SUCCESS) {
            return false;
        }

        // ---- Sync + download from per-depth output tensors ----
        ggml_backend_synchronize(rt.backend_compute.get());

        const int32_t total = n_beams * max_depth;
        result.n_beams      = n_beams;
        result.actual_depth = max_depth;
        result.hidden_size  = hp.hidden_size;

        result.parents.resize(total);
        result.tokens.resize(total);
        result.logprobs.resize(total);

        const size_t beam_i32 = (size_t) n_beams * sizeof(int32_t);
        const size_t beam_f32 = (size_t) n_beams * sizeof(float);
        for (int32_t d = 0; d < max_depth; ++d) {
            const size_t off = (size_t) d * n_beams;
            ggml_backend_tensor_get(mg.out_parents[d],  result.parents.data()  + off, 0, beam_i32);
            ggml_backend_tensor_get(mg.out_tokens[d],   result.tokens.data()   + off, 0, beam_i32);
            ggml_backend_tensor_get(mg.out_logprobs[d], result.logprobs.data() + off, 0, beam_f32);
        }

        // ---- Build device states (same KV logic, hidden from output tensors) ----
        goto build_device_states;
    }

    // ================ FALLBACK: PER-DEPTH FUSED GRAPH ================
    {
        // Ensure journal is allocated.
        if (!alloc_fused_journal(model, rt, n_beams, max_depth, rs.journal)) {
            return false;
        }

        // Build or reuse fused graph.
        if (!build_fused_depth_graph(model, rt, *rs.batch, rs.d2t_gpu,
                                     n_beams, k, max_depth, prob_threshold, rs.fused_graph)) {
            return false;
        }
        auto & fg = rs.fused_graph;

        // Warm up once.
        if (!rs.fused_warmed_up) {
            ggml_backend_graph_compute_async(rt.backend_compute.get(), fg.gf);
            ggml_backend_synchronize(rt.backend_compute.get());
            rs.fused_warmed_up = true;
        }

        // ---- Set initial inputs ----
        {
            const size_t h_bytes = (size_t) hp.hidden_size * sizeof(float);
            tensor_copy_bytes_async(
                    rt.backend_compute.get(), rt.backend_compute.get(),
                    root_state.dev->t_hidden, 0,
                    fg.t_hidden_in, 0, h_bytes);
            if (n_beams > 1) {
                std::vector<float> zeros((size_t) hp.hidden_size * (size_t)(n_beams - 1), 0.0f);
                ggml_backend_tensor_set_async(rt.backend_compute.get(), fg.t_hidden_in,
                        zeros.data(), h_bytes, zeros.size() * sizeof(float));
            }
        }
        {
            std::vector<float> lp(n_beams, -1e30f);
            lp[0] = 0.0f;
            ggml_backend_tensor_set_async(rt.backend_compute.get(), fg.t_beam_logprob,
                    lp.data(), 0, lp.size() * sizeof(float));
        }
        if (max_fork_rows > 0 && fg.t_fork_row_offsets && fg.t_fork_dst_idxs) {
            const int32_t total_fork = max_fork_rows * n_beams;
            std::vector<float>   fork_offsets(total_fork);
            std::vector<int32_t> fork_dst(total_fork);
            for (int32_t b = 0; b < n_beams; ++b) {
                for (int32_t r = 0; r < max_fork_rows; ++r) {
                    const int32_t idx = b * max_fork_rows + r;
                    fork_offsets[idx] = (float)(rollout_start + r);
                    fork_dst[idx] = b * kv_cap + rollout_start + r;
                }
            }
            ggml_backend_tensor_set_async(rt.backend_compute.get(), fg.t_fork_row_offsets,
                    fork_offsets.data(), 0, total_fork * sizeof(float));
            ggml_backend_tensor_set_async(rt.backend_compute.get(), fg.t_fork_dst_idxs,
                    fork_dst.data(), 0, total_fork * sizeof(int32_t));
        }

        // ---- Depth loop ----
        std::vector<int32_t> pos_host(n_beams);
        std::vector<int32_t> kv_idxs(n_beams);
        std::vector<ggml_fp16_t> mask_data(kv_cap, ggml_fp32_to_fp16(-INFINITY));
        for (int32_t i = 0; i < rollout_start; ++i) {
            mask_data[i] = ggml_fp32_to_fp16(0.0f);
        }

        for (int32_t depth = 0; depth < max_depth; ++depth) {
            const int32_t pos = rollout_start + depth;

            std::fill(pos_host.begin(), pos_host.end(), pos);
            ggml_backend_tensor_set_async(rt.backend_compute.get(), fg.t_pos_b,
                    pos_host.data(), 0, n_beams * sizeof(int32_t));

            for (int32_t b = 0; b < n_beams; ++b) {
                kv_idxs[b] = b * kv_cap + pos;
            }
            ggml_backend_tensor_set_async(rt.backend_compute.get(), fg.t_kv_write_idxs,
                    kv_idxs.data(), 0, n_beams * sizeof(int32_t));

            mask_data[pos] = ggml_fp32_to_fp16(0.0f);
            ggml_backend_tensor_set_async(rt.backend_compute.get(), fg.t_mask,
                    mask_data.data(), 0, kv_cap * sizeof(ggml_fp16_t));

            const ggml_status status = ggml_backend_graph_compute_async(rt.backend_compute.get(), fg.gf);
            if (status != GGML_STATUS_SUCCESS) {
                return false;
            }

            const size_t journal_offset = (size_t) depth * (size_t) n_beams;
            {
                const size_t i32_bytes = (size_t) n_beams * sizeof(int32_t);
                const size_t f32_bytes = (size_t) n_beams * sizeof(float);

                tensor_copy_bytes_async(rt.backend_compute.get(), rt.backend_compute.get(),
                        fg.t_parent, 0,
                        rs.journal.t_parents, journal_offset * sizeof(int32_t), i32_bytes);
                tensor_copy_bytes_async(rt.backend_compute.get(), rt.backend_compute.get(),
                        fg.t_base_id, 0,
                        rs.journal.t_tokens, journal_offset * sizeof(int32_t), i32_bytes);
                tensor_copy_bytes_async(rt.backend_compute.get(), rt.backend_compute.get(),
                        fg.t_selected_logprob, 0,
                        rs.journal.t_logprobs, journal_offset * sizeof(float), f32_bytes);

                const size_t h_bytes = (size_t) hp.hidden_size * (size_t) n_beams * sizeof(float);
                const size_t h_offset = (size_t) depth * h_bytes;
                tensor_copy_bytes_async(rt.backend_compute.get(), rt.backend_compute.get(),
                        fg.t_hidden_out, 0,
                        rs.journal.t_hidden, h_offset, h_bytes);
            }

            if (depth + 1 < max_depth) {
                const size_t h_bytes = (size_t) hp.hidden_size * (size_t) n_beams * sizeof(float);
                tensor_copy_bytes_async(rt.backend_compute.get(), rt.backend_compute.get(),
                        fg.t_hidden_out, 0,
                        fg.t_hidden_in, 0, h_bytes);
                tensor_copy_bytes_async(rt.backend_compute.get(), rt.backend_compute.get(),
                        fg.t_selected_logprob, 0,
                        fg.t_beam_logprob, 0, (size_t) n_beams * sizeof(float));
            }
        }

        ggml_backend_synchronize(rt.backend_compute.get());

        const int32_t total = n_beams * max_depth;
        result.n_beams      = n_beams;
        result.actual_depth = max_depth;
        result.hidden_size  = hp.hidden_size;

        result.parents.resize(total);
        result.tokens.resize(total);
        result.logprobs.resize(total);

        ggml_backend_tensor_get(rs.journal.t_parents,  result.parents.data(),  0, total * sizeof(int32_t));
        ggml_backend_tensor_get(rs.journal.t_tokens,   result.tokens.data(),   0, total * sizeof(int32_t));
        ggml_backend_tensor_get(rs.journal.t_logprobs, result.logprobs.data(), 0, total * sizeof(float));
    }

    // ================ BUILD DEVICE STATES ================
    //
    // Each batch slot b contains the correct KV for beam b's path at any depth:
    //   positions 0..rollout_start-1: prefix (from rollout_populate_prefix)
    //   positions rollout_start..rollout_start+d: forked parent KV + new KV from STEP
    // So we can directly use slot=b for every valid (depth d, beam b) entry.
    build_device_states:

    {
    const bool use_mega = use_mega_env && rs.mega_graph.gf;
    const auto & batch = *rs.batch;
    const size_t kv_row_bytes = (size_t) hp.head_dim * hp.num_kv_heads * sizeof(float);
    const size_t beam_kv_bytes = (size_t) kv_cap * kv_row_bytes;
    const int32_t total = n_beams * max_depth;

    result.states.resize(total);

    for (int32_t d = 0; d < max_depth; ++d) {
        for (int32_t b = 0; b < n_beams; ++b) {
            const size_t idx = (size_t) d * n_beams + b;
            if (result.logprobs[idx] < -1e20f ||
                result.tokens[idx] < 0 ||
                result.tokens[idx] >= hp.vocab_size) {
                continue;
            }

            const int32_t n_rollout = d + 1;

            llama_eagle3_state & st = result.states[idx];
            st.past_len = rollout_start + n_rollout;

            // Download hidden from per-depth output tensor (mega) or journal (fallback).
            st.hidden.resize(hp.hidden_size);
            if (use_mega) {
                ggml_backend_tensor_get(rs.mega_graph.out_hidden[d], st.hidden.data(),
                        (size_t) b * hp.hidden_size * sizeof(float),
                        (size_t) hp.hidden_size * sizeof(float));
            } else {
                ggml_backend_tensor_get(rs.journal.t_hidden, st.hidden.data(),
                        idx * (size_t) hp.hidden_size * sizeof(float),
                        (size_t) hp.hidden_size * sizeof(float));
            }

            // Allocate device state and copy KV from batch slot b.
            std::shared_ptr<llama_eagle3_state::device_state> dev;
            if (alloc_state_device(model, rt, n_rollout, dev)) {
                // Upload hidden to device.
                ggml_backend_tensor_set_async(rt.backend_compute.get(), dev->t_hidden,
                        st.hidden.data(), 0,
                        (size_t) hp.hidden_size * sizeof(float));

                // Copy KV rows from batch slot b.
                const size_t src_offset = (size_t) b * beam_kv_bytes
                    + (size_t) rollout_start * kv_row_bytes;
                const size_t copy_bytes = (size_t) n_rollout * kv_row_bytes;
                tensor_copy_bytes_async(
                        rt.backend_compute.get(), rt.backend_compute.get(),
                        batch.t_k, src_offset, dev->t_k, 0, copy_bytes);
                tensor_copy_bytes_async(
                        rt.backend_compute.get(), rt.backend_compute.get(),
                        batch.t_v, src_offset, dev->t_v, 0, copy_bytes);

                dev->past_len = rollout_start + n_rollout;
                dev->rollout_kv_start = rollout_start;
                st.dev = std::move(dev);
                st.rollout_only = true;
            }
        }
    }
    } // build_device_states scope

    return true;
}

// ---------------------------------------------------------------------------
// llama_eagle3_prefill_chunked — cached KV-only + root (public API)
// ---------------------------------------------------------------------------
bool llama_eagle3_prefill_chunked(
        const llama_eagle3_model & model,
        const llama_eagle3_runtime & rt,
        int32_t hidden_in_dim,
        const std::vector<const ggml_tensor *> & hidden_capture,
        const llama_token * input_ids,
        int32_t total_tokens,
        int32_t chunk_size,
        llama_eagle3_state & final_state_out) {
    GGML_UNUSED(chunk_size); // single-pass now; chunk_size ignored
    GGML_UNUSED(hidden_in_dim); // derived from model hparams now
    if (total_tokens <= 0) {
        final_state_out = {};
        return true;
    }
    if (!rt.backend_compute || !rt.buft_compute) {
        LLAMA_LOG_WARN("%s: no compute backend\n", __func__);
        return false;
    }

    const auto & hp = model.hparams;
    const bool do_profile = std::getenv("CASCADE_EAGLE_PROFILE") != nullptr;
    const int32_t kv_only_count = total_tokens - 1;

    int64_t t0 = do_profile ? ggml_time_us() : 0;

    // --- ensure cached prefill state exists and is large enough ---------------
    if (!rt.prefill_state) {
        rt.prefill_state = std::make_shared<llama_eagle3_runtime::prefill_cached_state>();
    }
    auto & ps = *rt.prefill_state;

    // (Re)allocate cache if needed. Round up to avoid frequent rebuilds.
    const int32_t old_capacity = ps.cache.capacity;
    {
        int32_t needed = total_tokens;
        if (needed <= 256) needed = 256;
        else if (needed <= 512) needed = 512;
        else if (needed <= 1024) needed = 1024;
        else needed = ((needed + 511) / 512) * 512;
        if (!alloc_prefill_cache(model, rt, needed, ps.cache)) {
            LLAMA_LOG_WARN("%s: alloc_prefill_cache failed (needed=%d)\n", __func__, needed);
            return false;
        }
    }
    const bool cache_reallocated = (ps.cache.capacity != old_capacity);

    // (Re)build KV-only graph if it doesn't fit or if cache was reallocated
    // (the graph bakes in cache tensor pointers, which become stale after realloc).
    if (kv_only_count > 0 && (cache_reallocated || ps.kv_graph.n_tokens < kv_only_count)) {
        if (!build_kv_only_graph_impl(model, rt, ps.cache, ps.cache.capacity, ps.kv_graph)) {
            LLAMA_LOG_WARN("%s: build_kv_only_graph_impl failed\n", __func__);
            return false;
        }
        ps.warmed_up = false; // need to re-warm after rebuild
    }

    // (Re)build root graph if cache capacity changed.
    if (cache_reallocated || ps.r_graph.kv_capacity != ps.cache.capacity) {
        if (!build_root_graph_impl(model, rt, ps.cache, ps.r_graph)) {
            LLAMA_LOG_WARN("%s: build_root_graph_impl failed\n", __func__);
            return false;
        }
        ps.warmed_up = false;
    }

    // Warm up: run each graph once with dummy data to trigger CUDA JIT/cuBLAS setup.
    // Zero-init index tensors so set_rows doesn't OOB on garbage indices.
    if (!ps.warmed_up) {
        auto zero_tensor = [&](ggml_tensor * t) {
            if (!t) return;
            const size_t nbytes = ggml_nbytes(t);
            std::vector<uint8_t> zeros(nbytes, 0);
            ggml_backend_tensor_set(t, zeros.data(), 0, nbytes);
        };
        if (kv_only_count > 0 && ps.kv_graph.gf) {
            zero_tensor(ps.kv_graph.t_k_idxs);
            zero_tensor(ps.kv_graph.t_v_idxs);
            ggml_backend_graph_compute_async(rt.backend_compute.get(), ps.kv_graph.gf);
        }
        if (ps.r_graph.gf) {
            zero_tensor(ps.r_graph.t_k_idxs);
            zero_tensor(ps.r_graph.t_v_idxs);
            ggml_backend_graph_compute_async(rt.backend_compute.get(), ps.r_graph.gf);
        }
        ggml_backend_synchronize(rt.backend_compute.get());
        ps.warmed_up = true;
    }

    int64_t t_setup = do_profile ? ggml_time_us() : 0;

    // --- Helper: bulk-copy per-layer hidden captures -------------------------
    auto fill_hidden_bulk = [&](const std::vector<ggml_tensor *> & layers,
                                int32_t pos, int32_t n_tokens) -> bool {
        for (int32_t il = 0; il < hp.hidden_concat; ++il) {
            ggml_tensor * src = const_cast<ggml_tensor *>(hidden_capture[(size_t) il]);
            if (!src || src->type != GGML_TYPE_F32 ||
                src->ne[0] != hp.target_hidden_size ||
                (size_t)(pos + n_tokens) > (size_t) src->ne[1]) {
                return false;
            }
            // Single contiguous copy: capture[il][pos : pos+n_tokens] → layer input
            const size_t n_bytes    = (size_t) hp.target_hidden_size * (size_t) n_tokens * sizeof(float);
            const size_t src_offset = (size_t) pos * (size_t) src->nb[1];
            if (!tensor_copy_bytes_async(
                        rt.target_backend ? rt.target_backend : rt.backend_compute.get(),
                        rt.backend_compute.get(),
                        src, src_offset,
                        layers[(size_t) il], 0,
                        n_bytes)) {
                return false;
            }
        }
        return true;
    };

    auto fill_positions = [&](ggml_tensor * t_tok_g, ggml_tensor * t_pos_g,
                              ggml_tensor * t_k_idxs_g, ggml_tensor * t_v_idxs_g,
                              int32_t pos, int32_t n_tokens) {
        ggml_backend_tensor_set_async(rt.backend_compute.get(), t_tok_g,
                input_ids + pos, 0, (size_t) n_tokens * sizeof(llama_token));
        std::vector<int32_t> positions((size_t) n_tokens);
        for (int32_t i = 0; i < n_tokens; ++i) {
            positions[(size_t) i] = pos + i;
        }
        ggml_backend_tensor_set_async(rt.backend_compute.get(), t_pos_g,
                positions.data(), 0, (size_t) n_tokens * sizeof(int32_t));
        ggml_backend_tensor_set_async(rt.backend_compute.get(), t_k_idxs_g,
                positions.data(), 0, (size_t) n_tokens * sizeof(int32_t));
        ggml_backend_tensor_set_async(rt.backend_compute.get(), t_v_idxs_g,
                positions.data(), 0, (size_t) n_tokens * sizeof(int32_t));
    };

    // --- KV-only: single pass for all tokens except root ---------------------
    // The KV graph is built for ps.cache.capacity tokens. We fill real data for
    // [0, kv_only_count) and pad the rest with dummy indices pointing to the root
    // position (which the root pass overwrites, so garbage there is harmless).
    int64_t t_kv = 0;
    if (kv_only_count > 0) {
        auto & kv = ps.kv_graph;
        const int32_t graph_n = kv.n_tokens; // = ps.cache.capacity

        // Bulk copy per-layer hidden captures for real tokens
        if (!fill_hidden_bulk(kv.t_hidden_layers, 0, kv_only_count)) {
            LLAMA_LOG_WARN("%s: fill_hidden_bulk(kv) failed (kv_only_count=%d, n_layers=%d, target_hidden=%d)\n",
                           __func__, kv_only_count, hp.hidden_concat, hp.target_hidden_size);
            for (int32_t il = 0; il < hp.hidden_concat; ++il) {
                const ggml_tensor * src = hidden_capture[(size_t) il];
                LLAMA_LOG_WARN("%s:   layer %d: src=%p type=%d ne=[%lld,%lld]\n",
                               __func__, il, (const void*)src,
                               src ? src->type : -1,
                               src ? (long long)src->ne[0] : -1,
                               src ? (long long)src->ne[1] : -1);
            }
            return false;
        }

        // Token ids: real for [0,kv_only_count), pad with token 0 for the rest
        {
            std::vector<llama_token> toks((size_t) graph_n, input_ids[0]);
            std::memcpy(toks.data(), input_ids, (size_t) kv_only_count * sizeof(llama_token));
            ggml_backend_tensor_set_async(rt.backend_compute.get(), kv.t_tok,
                    toks.data(), 0, (size_t) graph_n * sizeof(llama_token));
        }

        // Positions and KV indices: real for [0,kv_only_count), extra slots
        // point to kv_only_count (root position, overwritten by root pass).
        {
            std::vector<int32_t> positions((size_t) graph_n, kv_only_count);
            for (int32_t i = 0; i < kv_only_count; ++i) {
                positions[(size_t) i] = i;
            }
            ggml_backend_tensor_set_async(rt.backend_compute.get(), kv.t_pos,
                    positions.data(), 0, (size_t) graph_n * sizeof(int32_t));
            ggml_backend_tensor_set_async(rt.backend_compute.get(), kv.t_k_idxs,
                    positions.data(), 0, (size_t) graph_n * sizeof(int32_t));
            ggml_backend_tensor_set_async(rt.backend_compute.get(), kv.t_v_idxs,
                    positions.data(), 0, (size_t) graph_n * sizeof(int32_t));
        }

        if (do_profile) { t_kv = ggml_time_us(); }

        if (ggml_backend_graph_compute_async(rt.backend_compute.get(), kv.gf) != GGML_STATUS_SUCCESS) {
            return false;
        }
    }

    // --- Root: single full-layer pass for the last token ---------------------
    {
        auto & rg = ps.r_graph;
        const int32_t root_pos = kv_only_count;
        if (!fill_hidden_bulk(rg.t_hidden_layers, root_pos, 1)) {
            return false;
        }
        fill_positions(rg.t_tok, rg.t_pos, rg.t_k_idxs, rg.t_v_idxs, root_pos, 1);

        // Mask: 0 for valid positions [0, total_tokens), -inf for padding beyond.
        const int32_t cap = ps.cache.capacity;
        std::vector<ggml_fp16_t> mask_data((size_t) cap, ggml_fp32_to_fp16(-INFINITY));
        for (int32_t i = 0; i < total_tokens; ++i) {
            mask_data[(size_t) i] = ggml_fp32_to_fp16(0.0f);
        }
        ggml_backend_tensor_set(rg.t_mask, mask_data.data(), 0, (size_t) cap * sizeof(ggml_fp16_t));

        int64_t t_root = do_profile ? ggml_time_us() : 0;

        if (ggml_backend_graph_compute_async(rt.backend_compute.get(), rg.gf) != GGML_STATUS_SUCCESS) {
            return false;
        }

        if (do_profile) {
            ggml_backend_synchronize(rt.backend_compute.get());
            int64_t t_done = ggml_time_us();
            fprintf(stderr, "eagle3 prefill: tokens=%d | "
                    "setup=%.1f kv_fill=%.1f kv_compute=%.1f root_fill=%.1f root_compute=%.1f | total=%.1fms\n",
                    total_tokens,
                    (t_setup - t0) / 1000.0,
                    kv_only_count > 0 ? (t_kv - t_setup) / 1000.0 : 0.0,
                    kv_only_count > 0 ? (t_root - t_kv) / 1000.0 : 0.0,
                    (t_root - (kv_only_count > 0 ? t_kv : t_setup)) / 1000.0,
                    (t_done - t_root) / 1000.0,
                    (t_done - t0) / 1000.0);
        }
    }

    // --- Materialize final state ---------------------------------------------
    const int32_t required_len = total_tokens;
    std::shared_ptr<llama_eagle3_state::device_state> next_dev;
    const int32_t desired_capacity = choose_kv_capacity(required_len, 0, 0);
    if (!alloc_state_device(model, rt, desired_capacity, next_dev)) {
        return false;
    }

    if (!tensor_copy_bytes_async(
                rt.backend_compute.get(), rt.backend_compute.get(),
                ps.r_graph.t_hidden_out, 0,
                next_dev->t_hidden, 0,
                (size_t) hp.hidden_size * sizeof(float))) {
        return false;
    }

    if (next_dev->t_k && next_dev->t_v) {
        if (!tensor_copy_3d_prefix_async(
                    rt.backend_compute.get(), ps.cache.t_k, next_dev->t_k,
                    hp.head_dim, hp.num_kv_heads, required_len)) {
            return false;
        }
        if (!tensor_copy_3d_prefix_async(
                    rt.backend_compute.get(), ps.cache.t_v, next_dev->t_v,
                    hp.head_dim, hp.num_kv_heads, required_len)) {
            return false;
        }
    }

    next_dev->past_len = required_len;
    final_state_out.dev      = std::move(next_dev);
    final_state_out.hidden.clear();
    final_state_out.k.clear();
    final_state_out.v.clear();
    final_state_out.past_len = required_len;

    ggml_backend_synchronize(rt.backend_compute.get());
    return true;
}
