#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * RR, refusal removal (llm-lab dev branch r1-runtime-uncensor): removes one direction from every column of a residual stream,
 *
 *   x[:, t] <- x[:, t] - strength * (x[:, t] . u) * u,
 *
 * with `x` BF16 or FP32 [hidden, columns] (columns may be strided by nb[1]; FP32 for Flash-Next's hyper-connection
 * stack, viewed as [hidden, streams * tokens]), `u` a device pointer to `hidden` FP32 values (a unit vector for full
 * removal at strength 1) and the dot product accumulated in FP32. In place; no workspace.
 */
void rr_project(Tensor& x, const float* u, float strength, cudaStream_t stream);

} // namespace ninfer::ops
