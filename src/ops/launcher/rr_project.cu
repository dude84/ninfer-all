// Implements: include/ninfer/ops/rr_project.h
// One block per column (BF16 or FP32): an FP32 dot product over the hidden dimension, then the rank-1 update in place.
#include "ninfer/ops/rr_project.h"

#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kBlock = 256;

__device__ float rr_load(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ float rr_load(const float v) { return v; }
__device__ void rr_store(__nv_bfloat16& out, float v) { out = __float2bfloat16(v); }
__device__ void rr_store(float& out, float v) { out = v; }

template <class T>
__global__ void rr_project_kernel(T* x, std::int64_t pitch, int hidden, const float* u, float strength,
                                  const float* gate) {
    if (gate != nullptr) {
        strength *= *gate;
        if (strength == 0.0f) { return; }
    }
    T* col    = x + static_cast<std::int64_t>(blockIdx.x) * pitch;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += kBlock) { sum += rr_load(col[i]) * u[i]; }
    __shared__ float warp_sums[kBlock / 32];
    for (int off = 16; off > 0; off >>= 1) { sum += __shfl_down_sync(0xffffffffu, sum, off); }
    if ((threadIdx.x & 31) == 0) { warp_sums[threadIdx.x >> 5] = sum; }
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < kBlock / 32 ? warp_sums[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) { sum += __shfl_down_sync(0xffffffffu, sum, off); }
        if (threadIdx.x == 0) { warp_sums[0] = sum; }
    }
    __syncthreads();
    const float scale = strength * warp_sums[0];
    for (int i = threadIdx.x; i < hidden; i += kBlock) { rr_store(col[i], rr_load(col[i]) - scale * u[i]); }
}

template <class T>
void launch(Tensor& x, const float* u, float strength, cudaStream_t stream, const float* gate) {
    const std::int64_t pitch = x.nb[1] / static_cast<std::int64_t>(sizeof(T));
    rr_project_kernel<T><<<x.ne[1], kBlock, 0, stream>>>(static_cast<T*>(x.data), pitch, x.ne[0], u, strength, gate);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void rr_project(Tensor& x, const float* u, float strength, cudaStream_t stream, const float* gate) {
    const bool bf16 = x.dtype == DType::BF16;
    if ((!bf16 && x.dtype != DType::FP32) || u == nullptr || x.ne[0] <= 0 || x.ne[1] <= 0 ||
        x.nb[1] % static_cast<std::int64_t>(bf16 ? sizeof(__nv_bfloat16) : sizeof(float)) != 0) {
        throw std::invalid_argument("rr_project: x must be BF16 or FP32 [hidden, columns] with a direction");
    }
    if (strength == 0.0f) { return; }
    if (bf16) {
        launch<__nv_bfloat16>(x, u, strength, stream, gate);
    } else {
        launch<float>(x, u, strength, stream, gate);
    }
}

} // namespace ninfer::ops
