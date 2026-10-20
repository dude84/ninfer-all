// The fused NVFP4 A16 SwiGLU routes above the small-T family (T > 16), for weight-only gate/up
// parents such as an NVFP4-encoded DFlash2 drafter's at a verify width times a batch above 16.
// Both arms apply the nonlinearity to FP32 accumulators; the ninfer-5090 port records that
// routing these extents through a materialized BF16 projection instead misses the A16 tail bound
// (8.1e-3 against 6.3e-3 at 17/32/64/129). The A16 MMA schedules are the ones the registered
// [34816,5120] Linear selects at the same extents.
//
// Ported from the ninfer-5090 port (v1.2.1), src/ops/linear/swiglu/nvfp4/shapes/n34816_k5120/
// nvfp4_linear_swiglu_matrix.cu: same kernels, schedules and row-major epilogue.

#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan_unified.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh"
#include "ops/linear_swiglu/row_major_mma_epilogue.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail::unified {
namespace {

using Geometry              = Nvfp4N34816K5120;
constexpr int kIntermediate = Geometry::kOutputRows / 2;

template <class Schedule>
void launch_sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>;
    launch_nvfp4_a16_sliced_k_mma<S>(
        nvfp4_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
        Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<8>{});
}

template <class Schedule>
void launch_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>;
    launch_nvfp4_a16_mma<S>(nvfp4_a16_operands(x, weight),
                            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
                            SwiGluRowMajorMmaEpilogue{}, stream, SwiGluRowMajorMmaRows<S>{});
}

} // namespace

void nvfp4_linear_swiglu_matrix_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream) {
    if (x.ne[1] <= 32) {
        return launch_sliced<Nvfp4SlicedInstance<32, 8, 2>>(x, weight, out, stream);
    }
    if (x.ne[1] <= 64) {
        return launch_mma<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, out, stream);
    }
    if (x.ne[1] <= 96) {
        return launch_mma<Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, out, stream);
    }
    launch_mma<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out, stream);
}

} // namespace ninfer::ops::detail::unified
