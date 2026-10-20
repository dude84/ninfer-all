#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::ops::detail {

// R7 diagnostic: naive FP32 reference routes for the NVFP4-encoded DFlash2 drafter's weight-only
// projections, selected per site by NINFER_R7_REFERENCE (a comma list of qkv, feature, attn_out,
// down, gate_up, context, or "all"; unset = off, the production routes). Each reference decodes
// E2M1 x E4M3 / divisor and accumulates every product in FP32 with one plain loop per output, so
// it shares no schedule, staging or MMA with the production kernels. It exists to bisect a
// drafter acceptance gap against the ninfer-5090 port on the GPU; it is slow and not a route.
[[nodiscard]] bool nvfp4_reference_site(const char* site);

// The site a weight-only Linear of this shape belongs to (feature, attn_out, down), or nullptr.
[[nodiscard]] const char* nvfp4_reference_linear_site(std::int32_t n, std::int32_t k);

// out_s[r,t] = sum_k decode(W)[begin_s + r, k] * x[k,t]; outputs are ordered row segments whose
// row counts sum to W's rows (one output for a plain Linear, q/k/v for the drafter's bank).
void nvfp4_reference_linear(const Tensor& x, const Weight& weight, std::span<Tensor* const> outputs,
                            cudaStream_t stream);

// out[r,t] = silu(gate[r,t]) * up[r,t], gate rows [0,N/2) and up rows [N/2,N) of W.
void nvfp4_reference_swiglu(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
