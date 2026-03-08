#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// ---------------------------------------------------------------------------
// Optional CUDA profiling helpers (experimental; may change).
//
// - Set GGML_CUDA_NVTX=1 to enable NVTX ranges (Nsight Systems/Compute).
// - Set GGML_CUDA_TIMING=1 to enable CUDA-event timing for zones.
//
// Notes:
// - Timing zones will synchronize on the CUDA stream to report elapsed time.
//   Use only for profiling runs (it can change perf characteristics).
// ---------------------------------------------------------------------------

typedef struct ggml_backend_cuda_profiler_zone {
    void * ev_start; // cudaEvent_t (opaque)
    void * ev_end;   // cudaEvent_t (opaque)
    int    nvtx_pushed;
} ggml_backend_cuda_profiler_zone;

#ifdef GGML_USE_CUDA

// NVTX helpers (no-ops unless ggml-cuda was built with NVTX support and
// GGML_CUDA_NVTX=1 is set).
GGML_BACKEND_API void ggml_backend_cuda_nvtx_push(const char * name);
GGML_BACKEND_API void ggml_backend_cuda_nvtx_pop(void);

// Begin/end a profiling zone on the backend's CUDA stream.
// When GGML_CUDA_TIMING=1, returns elapsed GPU time in milliseconds from *_end().
GGML_BACKEND_API void  ggml_backend_cuda_profiler_zone_begin(ggml_backend_t backend, ggml_backend_cuda_profiler_zone * zone, const char * name);
GGML_BACKEND_API float ggml_backend_cuda_profiler_zone_end  (ggml_backend_t backend, ggml_backend_cuda_profiler_zone * zone, const char * name);

// Copy the prefix [0:n2) of a 3D tensor asynchronously on the backend stream.
// src and dst must match on type and ne[0], ne[1], and have ne[2] >= n2.
GGML_BACKEND_API bool ggml_backend_cuda_tensor_copy_3d_prefix_async(
        ggml_backend_t backend,
        const struct ggml_tensor * src,
        struct ggml_tensor * dst,
        int64_t n2);

// Copy a contiguous 2D tensor [ne0, src->ne1] into dst at token row offset dst_i1.
// src and dst must match on type and ne[0], and dst must have ne[1] >= dst_i1 + src->ne[1].
GGML_BACKEND_API bool ggml_backend_cuda_tensor_copy_2d_async(
        ggml_backend_t backend,
        const struct ggml_tensor * src,
        struct ggml_tensor * dst,
        int64_t dst_i1);

// Copy an arbitrary contiguous byte range between backend-resident tensors.
GGML_BACKEND_API bool ggml_backend_cuda_tensor_copy_bytes_async(
        ggml_backend_t backend,
        const struct ggml_tensor * src,
        size_t src_offset,
        struct ggml_tensor * dst,
        size_t dst_offset,
        size_t size);

// Copy an arbitrary contiguous byte range between CUDA tensors, preserving ordering across backend streams.
GGML_BACKEND_API bool ggml_backend_cuda_tensor_copy_bytes_between_async(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        const struct ggml_tensor * src,
        size_t src_offset,
        struct ggml_tensor * dst,
        size_t dst_offset,
        size_t size);

#endif // GGML_USE_CUDA

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
