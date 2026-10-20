// NVFP4-encoded DFlash2 drafter attention input projection: one fused [6144,5120] BlockScale
// parent writes q [4096,T], k [1024,T] and v [1024,T] directly, weight-only A16. This fork's
// three-output drafter projection was Q8-only, so an artifact whose draft module is NVFP4 (the
// ninfer-5090 port's `_nvfp4_draft` converter step, draft projections `nvfp4_maxabs`, A16Only)
// was refused at startup with "QKV input projection: native form requires Q8".
//
// Ported from the ninfer-5090 port (v1.2.1), src/ops/linear/split/nvfp4/shapes/n6144_k5120/
// dflash2.cu: the same unified A16 kernels (nvfp4_a16_gemv / nvfp4_a16_simt, which are
// line-for-line the same in both trees), the same schedules per token count, the same
// segmented output and the same 32-token chunk walk, so the projection rounds exactly as the
// port's does.

#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail::unified {
namespace {

using Geometry = Nvfp4Geometry<6144, 5120>;
using Output   = LinearBf16SegmentedOutput<4096, 1024, 1024>;

constexpr int kFirstSmallT = 2;
constexpr int kLastSmallT  = 32;

// The port's measured low-T warp mapping for the drafter's exact-token family (the same mapping
// the four-output attention route uses; see nvfp4_attn_input_small_t_unified.cu).
template <int ActiveTokens>
struct Nvfp4DFlash2AttentionSmallTSchedule {
    static_assert(ActiveTokens >= kFirstSmallT);
    static_assert(ActiveTokens <= kLastSmallT);
    static constexpr int kWarpsPerCta       = ActiveTokens >= 17 ? 4 : (ActiveTokens >= 8 ? 16 : 8);
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SimtActivationAccess::SharedPhase
                                                  : Nvfp4SimtActivationAccess::TokenPacked;
    using Type =
        Nvfp4A16SimtSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                             Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                             Nvfp4SimtBlockOrder::RowsContiguous, 1>;
};

using Slices = std::array<Tensor, 3>;

Output segmented(const Slices& outputs) {
    return Output{static_cast<__nv_bfloat16*>(outputs[0].data),
                  static_cast<__nv_bfloat16*>(outputs[1].data),
                  static_cast<__nv_bfloat16*>(outputs[2].data)};
}

void launch_decode(const Tensor& x, const Weight& weight, const Slices& outputs,
                   cudaStream_t stream) {
    using Schedule =
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
    launch_nvfp4_a16_gemv<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, weight), segmented(outputs), LinearIdentityEpilogue{}, stream);
}

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, const Slices& outputs,
                  cudaStream_t stream) {
    using Schedule = typename Nvfp4DFlash2AttentionSmallTSchedule<ActiveTokens>::Type;
    launch_nvfp4_a16_simt<
        Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows, ActiveTokens, true>>(
        nvfp4_a16_operands(x, weight), segmented(outputs), LinearIdentityEpilogue{}, stream);
}

using Launch = void (*)(const Tensor&, const Weight&, const Slices&, cudaStream_t);

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<kFirstSmallT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kLastSmallT - kFirstSmallT + 1>{});

// Extents above the exact family walk in 32-token chunks, each chunk through its own selection.
void launch_dflash2_qkv(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                        cudaStream_t stream) {
    constexpr std::int32_t kChunk = kLastSmallT;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        const Tensor x_slice      = x.slice(1, token_begin, active);
        const Slices slices{q.slice(1, token_begin, active), k.slice(1, token_begin, active),
                            v.slice(1, token_begin, active)};
        if (active == 1) {
            launch_decode(x_slice, weight, slices, stream);
        } else {
            kLaunchers[static_cast<std::size_t>(active - kFirstSmallT)](x_slice, weight, slices,
                                                                        stream);
        }
    }
}

} // namespace
} // namespace ninfer::ops::detail::unified

namespace ninfer::ops::detail {

void nvfp4_dflash2_attn_input_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                     Tensor& v, cudaStream_t stream) {
    unified::launch_dflash2_qkv(x, weight, q, k, v, stream);
}

} // namespace ninfer::ops::detail
