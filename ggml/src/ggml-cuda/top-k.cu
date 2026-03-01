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
