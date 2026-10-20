// R7 diagnostic reference routes (see nvfp4_reference.h). Plain FP32 loops; not a production route.

#include "ops/linear/nvfp4/nvfp4_reference.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::ops::detail {
namespace {

constexpr int kRowsPerBlock = 8; // one warp per output row

__device__ __forceinline__ float reference_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// K16M128x4 block-scale byte of (row, 16-value group), as core/weight_view.cpp's
// weight_scale_offset states it.
__device__ __forceinline__ std::int64_t reference_scale_offset(int row, int group, int k) {
    const int inner = row % 128;
    return (static_cast<std::int64_t>(row / 128) * (k / 64) + group / 4) * 512 +
           (inner % 32) * 16 + (inner / 32) * 4 + group % 4;
}

// One row's dot product with token t's activation, FP32 throughout, divisor not applied.
__device__ float reference_row_dot(const std::uint8_t* codes, const std::uint8_t* scales,
                                   const __nv_bfloat16* x, int row, int t, int k, int lane) {
    const std::uint8_t* row_codes = codes + static_cast<std::int64_t>(row) * (k / 2);
    const __nv_bfloat16* column   = x + static_cast<std::int64_t>(t) * k;
    float acc                     = 0.0F;
    for (int group = lane; group < k / 16; group += 32) {
        const float scale = decode_nvfp4_e4m3(scales[reference_scale_offset(row, group, k)]);
        float sum         = 0.0F;
#pragma unroll
        for (int byte = 0; byte < 8; ++byte) {
            const float2 pair = decode_nvfp4_e2m1x2(row_codes[group * 8 + byte]);
            const int index   = group * 16 + byte * 2;
            sum               = fmaf(pair.x, __bfloat162float(column[index]), sum);
            sum               = fmaf(pair.y, __bfloat162float(column[index + 1]), sum);
        }
        acc = fmaf(scale, sum, acc);
    }
    return reference_warp_sum(acc);
}

struct ReferenceSegments {
    __nv_bfloat16* data[3];
    int end[3]; // exclusive cumulative row bound of each segment
    int count;
};

__global__ void nvfp4_reference_linear_kernel(const __nv_bfloat16* x, const std::uint8_t* codes,
                                              const std::uint8_t* scales, float inverse_divisor,
                                              int n, int k, int tokens,
                                              ReferenceSegments segments) {
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * kRowsPerBlock + (threadIdx.x >> 5);
    if (row >= n) return;
    int segment = 0, begin = 0;
    while (segment + 1 < segments.count && row >= segments.end[segment]) {
        begin = segments.end[segment];
        ++segment;
    }
    const int rows = segments.end[segment] - begin;
    for (int t = 0; t < tokens; ++t) {
        const float value = reference_row_dot(codes, scales, x, row, t, k, lane) * inverse_divisor;
        if (lane == 0) {
            segments.data[segment][static_cast<std::int64_t>(t) * rows + (row - begin)] =
                __float2bfloat16_rn(value);
        }
    }
}

__global__ void nvfp4_reference_swiglu_kernel(const __nv_bfloat16* x, const std::uint8_t* codes,
                                              const std::uint8_t* scales, float inverse_divisor,
                                              int half_rows, int k, int tokens,
                                              __nv_bfloat16* out) {
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * kRowsPerBlock + (threadIdx.x >> 5);
    if (row >= half_rows) return;
    for (int t = 0; t < tokens; ++t) {
        const float gate = reference_row_dot(codes, scales, x, row, t, k, lane) * inverse_divisor;
        const float up =
            reference_row_dot(codes, scales, x, row + half_rows, t, k, lane) * inverse_divisor;
        if (lane == 0) {
            const float silu = gate / (1.0F + expf(-gate));
            out[static_cast<std::int64_t>(t) * half_rows + row] = __float2bfloat16_rn(silu * up);
        }
    }
}

void require_reference_operands(const Tensor& x, const Weight& weight) {
    if (weight.qtype != QType::NVFP4 || weight.layout != QuantLayout::BlockScaleK16M128x4 ||
        x.dtype != DType::BF16 || x.ne[0] != weight.k || x.ne[1] <= 0 || !x.is_contiguous() ||
        weight.n % 128 != 0 || weight.k % 64 != 0 || !(weight.weight_scale_divisor > 0.0F)) {
        throw std::invalid_argument("nvfp4 reference: invalid operands");
    }
}

} // namespace

bool nvfp4_reference_site(const char* site) {
    static const std::string sites = [] {
        const char* value = std::getenv("NINFER_R7_REFERENCE");
        return value == nullptr ? std::string() : "," + std::string(value) + ",";
    }();
    if (sites.empty()) return false;
    return sites.find(",all,") != std::string::npos ||
           sites.find("," + std::string(site) + ",") != std::string::npos;
}

const char* nvfp4_reference_linear_site(std::int32_t n, std::int32_t k) {
    if (n == 5120 && k == 25600) return "feature";
    if (n == 5120 && k == 4096) return "attn_out";
    if (n == 5120 && k == 17408) return "down";
    return nullptr;
}

void nvfp4_reference_linear(const Tensor& x, const Weight& weight, std::span<Tensor* const> outputs,
                            cudaStream_t stream) {
    require_reference_operands(x, weight);
    if (outputs.empty() || outputs.size() > 3) {
        throw std::invalid_argument("nvfp4 reference: one to three outputs");
    }
    ReferenceSegments segments{};
    segments.count = static_cast<int>(outputs.size());
    int rows       = 0;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const Tensor& out = *outputs[i];
        if (out.dtype != DType::BF16 || out.ne[1] != x.ne[1] || !out.is_contiguous()) {
            throw std::invalid_argument("nvfp4 reference: invalid output");
        }
        rows += out.ne[0];
        segments.data[i] = static_cast<__nv_bfloat16*>(out.data);
        segments.end[i]  = rows;
    }
    if (rows != weight.n) throw std::invalid_argument("nvfp4 reference: outputs do not cover N");
    const int blocks = (weight.n + kRowsPerBlock - 1) / kRowsPerBlock;
    nvfp4_reference_linear_kernel<<<blocks, kRowsPerBlock * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), 1.0F / weight.weight_scale_divisor,
        weight.n, weight.k, x.ne[1], segments);
    CUDA_CHECK(cudaGetLastError());
}

void nvfp4_reference_swiglu(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    require_reference_operands(x, weight);
    const int half_rows = weight.n / 2;
    if (out.dtype != DType::BF16 || out.ne[0] != half_rows || out.ne[1] != x.ne[1] ||
        !out.is_contiguous()) {
        throw std::invalid_argument("nvfp4 reference swiglu: invalid output");
    }
    const int blocks = (half_rows + kRowsPerBlock - 1) / kRowsPerBlock;
    nvfp4_reference_swiglu_kernel<<<blocks, kRowsPerBlock * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), 1.0F / weight.weight_scale_divisor,
        half_rows, weight.k, x.ne[1], static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
