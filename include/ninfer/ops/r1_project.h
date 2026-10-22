#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * R1 (llm-lab dev branch r1-runtime-uncensor): removes one direction from every column of a residual stream,
 *
 *   x[:, t] <- x[:, t] - alpha * (x[:, t] . u) * u,
 *
 * with `x` BF16 [hidden, columns] (columns may be strided by nb[1]), `u` a device pointer to `hidden` FP32 values
 * (a unit vector for full removal at alpha 1) and the dot product accumulated in FP32. In place; no workspace.
 */
void r1_project(Tensor& x, const float* u, float alpha, cudaStream_t stream);

} // namespace ninfer::ops
