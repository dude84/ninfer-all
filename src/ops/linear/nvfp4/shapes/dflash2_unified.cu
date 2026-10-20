// DFlash2 drafter matrices of an NVFP4-encoded draft module (the ninfer-5090 port's converter
// step `_nvfp4_draft`: draft projections `nvfp4_maxabs`, A16Only): weight-only A16 routes -- a
// GEMV at one token, an exact SIMT family from 5 to 28 tokens, and a 32-token chunk walk for
// wider extents. The draft never quantizes activations, so no A4 route is registered.
//
// Ported from the ninfer-5090 port (v1.2.1), src/ops/linear/projection/nvfp4/shapes/dflash2.cu
// and src/ops/linear/kernels/nvfp4/nvfp4_dflash2_chunks.cuh. The kernels are the unified A16
// ones both trees share (nvfp4_a16_gemv.cuh / nvfp4_a16_simt.cuh); the schedule per token count is
// the port's, so these projections round exactly as the port's do.
//
// Registered shapes: the feature projection [5120,25600] (five target layers' hidden states into
// the draft width) and the attention output [5120,4096]. The drafter's down projection
// [5120,17408] is the dense shape already registered, and its q/k/v parent has its own
// three-output route (attn_input_proj/nvfp4/nvfp4_dflash2_attn_input_unified.cu). Its convolution
// kernel projections and selector projection stay BF16 in that converter's skip list.

#include "ops/linear/nvfp4/nvfp4_shapes_unified.h"
#include "ops/linear/nvfp4/nvfp4_launch_unified.cuh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail::unified {
namespace {

using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
template <int Tokens>
using Exact = Nvfp4A16SimtSchedule<(Tokens >= 8 && Tokens <= 16) ? 16 : 4, 1, 2, 16, Tokens, 1,
                                   Nvfp4SimtActivationAccess::TokenPacked, Nvfp4ScaleAccess::Direct,
                                   Nvfp4CodeCache::Default, 1, Nvfp4SimtBlockOrder::RowsContiguous,
                                   1>;
using C2  = Nvfp4A16SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                 Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                 Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C4  = Nvfp4A16SimtSchedule<4, 1, 2, 16, 4, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                 Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                 Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C32 = Nvfp4A16SimtSchedule<4, 1, 2, 16, 16, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                 Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                 Nvfp4SimtBlockOrder::TokenTilesContiguous, 3>;
using FullChunk = Nvfp4A16SimtSchedule<4, 1, 2, 16, 32, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                       Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                       Nvfp4SimtBlockOrder::RowsContiguous, 1>;

constexpr int kExactFirst = 5;
constexpr int kExactLast  = 28;
constexpr int kChunk      = 32;

// One exact-token launcher per token count in [First, Last], each specialized on its own count.
template <class Geometry, std::size_t... Indices>
constexpr auto exact_launchers(std::index_sequence<Indices...>) {
    return std::array<Nvfp4Launch, sizeof...(Indices)>{
        &nvfp4_linear_a16_simt<Geometry, kExactFirst + static_cast<int>(Indices),
                               Exact<kExactFirst + static_cast<int>(Indices)>, true>...};
}

template <class Geometry>
Nvfp4Launch select_exact(std::int32_t tokens) {
    static constexpr auto launchers =
        exact_launchers<Geometry>(std::make_index_sequence<kExactLast - kExactFirst + 1>{});
    return launchers.at(static_cast<std::size_t>(tokens - kExactFirst));
}

template <class Geometry>
Nvfp4Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens == 32) return nvfp4_linear_a16_simt<Geometry, 32, FullChunk, true>;
    if (tokens >= kExactFirst && tokens <= kExactLast) return select_exact<Geometry>(tokens);
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 4) return nvfp4_linear_a16_simt<Geometry, 4, C4, false>;
    if (tokens <= 32) return nvfp4_linear_a16_simt<Geometry, 32, C32, false>;
    throw std::logic_error("nvfp4 DFlash2 A16 chunk exceeds shape capacity");
}

// Wide extents walk in fixed chunks, each chunk dispatched through the shape's own selection.
template <class Geometry>
void launch_chunks(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    for (std::int32_t offset = 0; offset < x.ne[1]; offset += kChunk) {
        const std::int32_t count = std::min(kChunk, x.ne[1] - offset);
        const Tensor input       = x.slice(1, offset, count);
        Tensor output            = out.slice(1, offset, count);
        select_a16<Geometry>(count)(input, weight, output, stream);
    }
}

bool never_a4(std::int32_t, std::int32_t) { return false; }

template <class Geometry>
constexpr Nvfp4LinearShape make_dflash2_shape() {
    return {Geometry::kOutputRows, Geometry::kInputRows, launch_chunks<Geometry>, nullptr,
            never_a4};
}

using Feature = Nvfp4Geometry<5120, 25600>;
using AttnOut = Nvfp4Geometry<5120, 4096>;

} // namespace

const Nvfp4LinearShape kNvfp4DFlash2Feature = make_dflash2_shape<Feature>();
const Nvfp4LinearShape kNvfp4DFlash2AttnOut = make_dflash2_shape<AttnOut>();

bool nvfp4_dflash2_shape(std::int32_t n, std::int32_t k) {
    return (n == Feature::kOutputRows && k == Feature::kInputRows) ||
           (n == AttnOut::kOutputRows && k == AttnOut::kInputRows);
}

} // namespace ninfer::ops::detail::unified
