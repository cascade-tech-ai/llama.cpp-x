#include "argsort.cuh"
#include "top-k.cuh"
#include <cfloat>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env);

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env);
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

static __global__ void top_k_serial_row(float * row, int * out, int ncols, int k) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    for (int j = 0; j < k; ++j) {
        float best = -FLT_MAX;
        int best_i = 0;
        for (int i = 0; i < ncols; ++i) {
            const float v = row[i];
            if (v > best) {
                best = v;
                best_i = i;
            }
        }
        out[j] = best_i;
        row[best_i] = -FLT_MAX;
    }
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

#ifdef CUB_TOP_K_AVAILABLE
    for (int64_t i = 0; i < nrows; ++i) {
        top_k_cub(pool,
                  src0_d + i * ncols,
                  dst_d + i * k,
                  (int) ncols,
                  (int) k,
                  stream);
    }
    return;
#else
    ggml_cuda_pool_alloc<float> tmp_src_alloc(pool, ncols * nrows);
    float *                     tmp_src = tmp_src_alloc.get();
    CUDA_CHECK(cudaMemcpyAsync(tmp_src, src0_d, ncols * nrows * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    for (int i = 0; i < nrows; ++i) {
        top_k_serial_row<<<1, 1, 0, stream>>>(tmp_src + i * ncols, dst_d + i * k, ncols, k);
    }
#endif
}

namespace {

constexpr int GGML_CUDA_THRESHOLD_TOP_K_BLOCK_SIZE = 128;
constexpr int GGML_CUDA_THRESHOLD_TOP_K_TARGET_TILE = 8192;
constexpr int GGML_CUDA_THRESHOLD_TOP_K_MAX_BLOCKS_PER_ROW = 16;

template<int MAX_K>
static __device__ __forceinline__ void ggml_cuda_threshold_top_k_init(
        float (&vals)[MAX_K],
        int   (&idxs)[MAX_K],
        const int sentinel) {
    for (int i = 0; i < MAX_K; ++i) {
        vals[i] = -FLT_MAX;
        idxs[i] = sentinel;
    }
}

template<int MAX_K>
static __device__ __forceinline__ void ggml_cuda_threshold_top_k_insert(
        float (&vals)[MAX_K],
        int   (&idxs)[MAX_K],
        const int k,
        const float v,
        const int idx) {
    if (v <= vals[k - 1]) {
        return;
    }

    int pos = k - 1;
    while (pos > 0 && v > vals[pos - 1]) {
        vals[pos] = vals[pos - 1];
        idxs[pos] = idxs[pos - 1];
        --pos;
    }

    vals[pos] = v;
    idxs[pos] = idx;
}

template<int MAX_K>
static __device__ __forceinline__ void ggml_cuda_threshold_top_k_merge(
        float (&dst_vals)[MAX_K],
        int   (&dst_idxs)[MAX_K],
        const float * src_vals,
        const int   * src_idxs,
        const int k,
        const int sentinel) {
    for (int i = 0; i < k; ++i) {
        if (src_idxs[i] == sentinel) {
            continue;
        }
        ggml_cuda_threshold_top_k_insert<MAX_K>(dst_vals, dst_idxs, k, src_vals[i], src_idxs[i]);
    }
}

template<int MAX_K, int BLOCK_SIZE>
static __global__ void ggml_cuda_threshold_top_k_stage1(
        const float * src,
        int * block_out,
        const int ncols,
        const int k,
        const float threshold,
        const int blocks_per_row,
        const int sentinel) {
    const int tile = blockIdx.x;
    const int row  = blockIdx.y;
    const int tid  = threadIdx.x;

    const int64_t start = ((int64_t) tile * ncols) / blocks_per_row;
    const int64_t end   = ((int64_t) (tile + 1) * ncols) / blocks_per_row;
    const float * row_src = src + (int64_t) row * ncols;

    float local_vals[MAX_K];
    int   local_idxs[MAX_K];
    ggml_cuda_threshold_top_k_init<MAX_K>(local_vals, local_idxs, sentinel);

    for (int64_t col = start + tid; col < end; col += BLOCK_SIZE) {
        const float v = row_src[col];
        if (v >= threshold) {
            ggml_cuda_threshold_top_k_insert<MAX_K>(local_vals, local_idxs, k, v, (int) col);
        }
    }

    __shared__ float shared_vals[BLOCK_SIZE][MAX_K];
    __shared__ int   shared_idxs[BLOCK_SIZE][MAX_K];

    for (int i = 0; i < MAX_K; ++i) {
        shared_vals[tid][i] = local_vals[i];
        shared_idxs[tid][i] = local_idxs[i];
    }
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ggml_cuda_threshold_top_k_merge<MAX_K>(
                    shared_vals[tid],
                    shared_idxs[tid],
                    shared_vals[tid + stride],
                    shared_idxs[tid + stride],
                    k,
                    sentinel);
        }
        __syncthreads();
    }

    if (tid == 0) {
        int * out = block_out + ((row * blocks_per_row + tile) * k);
        for (int i = 0; i < k; ++i) {
            out[i] = shared_idxs[0][i];
        }
    }
}

template<int MAX_K, int BLOCK_SIZE>
static __global__ void ggml_cuda_threshold_top_k_stage2(
        const float * src,
        const int * block_in,
        int * dst,
        const int ncols,
        const int k,
        const int blocks_per_row,
        const int sentinel) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float * row_src = src + (int64_t) row * ncols;
    const int * row_in = block_in + (row * blocks_per_row * k);

    float local_vals[MAX_K];
    int   local_idxs[MAX_K];
    ggml_cuda_threshold_top_k_init<MAX_K>(local_vals, local_idxs, sentinel);

    const int n_candidates = blocks_per_row * k;
    for (int i = tid; i < n_candidates; i += BLOCK_SIZE) {
        const int idx = row_in[i];
        if (idx == sentinel) {
            continue;
        }
        ggml_cuda_threshold_top_k_insert<MAX_K>(local_vals, local_idxs, k, row_src[idx], idx);
    }

    __shared__ float shared_vals[BLOCK_SIZE][MAX_K];
    __shared__ int   shared_idxs[BLOCK_SIZE][MAX_K];

    for (int i = 0; i < MAX_K; ++i) {
        shared_vals[tid][i] = local_vals[i];
        shared_idxs[tid][i] = local_idxs[i];
    }
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ggml_cuda_threshold_top_k_merge<MAX_K>(
                    shared_vals[tid],
                    shared_idxs[tid],
                    shared_vals[tid + stride],
                    shared_idxs[tid + stride],
                    k,
                    sentinel);
        }
        __syncthreads();
    }

    if (tid == 0) {
        int * row_dst = dst + (int64_t) row * k;
        for (int i = 0; i < k; ++i) {
            row_dst[i] = shared_idxs[0][i];
        }
    }
}

static int ggml_cuda_threshold_top_k_blocks_per_row(const int64_t ncols) {
    const int blocks = (int) ((ncols + GGML_CUDA_THRESHOLD_TOP_K_TARGET_TILE - 1) / GGML_CUDA_THRESHOLD_TOP_K_TARGET_TILE);
    return std::max(1, std::min(GGML_CUDA_THRESHOLD_TOP_K_MAX_BLOCKS_PER_ROW, blocks));
}

} // namespace

void ggml_cuda_op_top_k_threshold(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);
    const int64_t k = dst->ne[0];
    const float threshold = ggml_get_op_params_f32(dst, 0);
    const int sentinel = (int) ncols;

    GGML_ASSERT(k > 0);

    const int blocks_per_row = ggml_cuda_threshold_top_k_blocks_per_row(ncols);
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<int> block_out_alloc(pool, (size_t) nrows * blocks_per_row * k);
    int * block_out = block_out_alloc.get();

    const dim3 grid_stage1(blocks_per_row, nrows, 1);
    const dim3 grid_stage2(nrows, 1, 1);

#define GGML_CUDA_THRESHOLD_TOP_K_DISPATCH(MAX_K) \
    do { \
        ggml_cuda_threshold_top_k_stage1<MAX_K, GGML_CUDA_THRESHOLD_TOP_K_BLOCK_SIZE> \
            <<<grid_stage1, GGML_CUDA_THRESHOLD_TOP_K_BLOCK_SIZE, 0, stream>>>( \
                    src0_d, \
                    block_out, \
                    (int) ncols, \
                    (int) k, \
                    threshold, \
                    blocks_per_row, \
                    sentinel); \
        ggml_cuda_threshold_top_k_stage2<MAX_K, GGML_CUDA_THRESHOLD_TOP_K_BLOCK_SIZE> \
            <<<grid_stage2, GGML_CUDA_THRESHOLD_TOP_K_BLOCK_SIZE, 0, stream>>>( \
                    src0_d, \
                    block_out, \
                    dst_d, \
                    (int) ncols, \
                    (int) k, \
                    blocks_per_row, \
                    sentinel); \
    } while (0)

    if (k <= 8) {
        GGML_CUDA_THRESHOLD_TOP_K_DISPATCH(8);
    } else if (k <= 16) {
        GGML_CUDA_THRESHOLD_TOP_K_DISPATCH(16);
    } else if (k <= 32) {
        GGML_CUDA_THRESHOLD_TOP_K_DISPATCH(32);
    } else {
        GGML_ABORT("ggml_cuda_op_top_k_threshold: unsupported k=%d", (int) k);
    }

#undef GGML_CUDA_THRESHOLD_TOP_K_DISPATCH
}
