// Implements: include/ninfer/ops/rr_project.h
// One block per column: an FP32 dot product over the hidden dimension, then the rank-1 update in place.
#include "ninfer/ops/rr_project.h"

#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kBlock = 256;

__global__ void rr_project_kernel(__nv_bfloat16* x, std::int64_t pitch, int hidden, const float* u, float strength) {
    __nv_bfloat16* col = x + static_cast<std::int64_t>(blockIdx.x) * pitch;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += kBlock) { sum += __bfloat162float(col[i]) * u[i]; }
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
    for (int i = threadIdx.x; i < hidden; i += kBlock) {
        col[i] = __float2bfloat16(__bfloat162float(col[i]) - scale * u[i]);
    }
}

} // namespace

void rr_project(Tensor& x, const float* u, float strength, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || u == nullptr || x.ne[0] <= 0 || x.ne[1] <= 0 ||
        x.nb[1] % static_cast<std::int64_t>(sizeof(__nv_bfloat16)) != 0) {
        throw std::invalid_argument("rr_project: x must be BF16 [hidden, columns] with a direction");
    }
    if (x.ne[1] == 0 || strength == 0.0f) { return; }
    const std::int64_t pitch = x.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
    rr_project_kernel<<<x.ne[1], kBlock, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), pitch, x.ne[0], u, strength);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
