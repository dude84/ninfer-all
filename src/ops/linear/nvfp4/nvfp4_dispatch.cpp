#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/nvfp4/nvfp4_dispatch_unified.h"
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_shapes_unified.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
const std::array kShapes{&kNvfp4N14336K5120, &kNvfp4N16384K5120, &kNvfp4N34816K5120,
                         &kNvfp4N5120K6144, &kNvfp4N5120K17408};

const Nvfp4LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("nvfp4 linear: unsupported policy");
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("nvfp4 linear: unsupported shape");
}

// Calls that take the unified templates at every width. The DFlash2 drafter shapes exist only
// there. And a weight-only parent (a policy without A4, as the NVFP4-encoded DFlash2 drafter's
// projections are stored) runs the A16 routes the ninfer-5090 port runs, at every width, so its
// arithmetic does not depend on this line's measured width bands. sm_8x builds read every NVFP4
// policy as A16 and keep their legacy bands.
bool unified_at_every_width(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (unified::nvfp4_dflash2_shape(n, k)) return true;
#if defined(NINFER_SM8X_COMPAT)
    (void)policy;
    return false;
#else
    return kUnifiedLinearRoutes && valid_linear_policy(policy) && !allows_a4(policy);
#endif
}
} // namespace

static std::size_t nvfp4_linear_workspace_capacity_bytes_own(std::int32_t n, std::int32_t k,
                                                             LinearPolicy policy,
                                                             std::int32_t min_tokens,
                                                             std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(n, k, policy);
    return allows_a4(policy) && shape.uses_a4(min_tokens, max_tokens)
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, k)
               : 0;
}

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (unified_at_every_width(n, k, policy)) {
        return unified::nvfp4_linear_workspace_capacity_bytes(n, k, policy, min_tokens,
                                                              max_tokens);
    }
    return capacity_by_table(
        min_tokens, max_tokens,
        [](std::int32_t width) { return linear_route_table(LinearRouteFamily::Nvfp4, width); },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) {
            return table == LinearRouteTable::Unified
                       ? unified::nvfp4_linear_workspace_capacity_bytes(n, k, policy, first, last)
                       : nvfp4_linear_workspace_capacity_bytes_own(n, k, policy, first, last);
        });
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    if (x.ne[1] > 0 && (unified_at_every_width(weight.n, weight.k, policy) ||
                        linear_route_table(LinearRouteFamily::Nvfp4, x.ne[1]) ==
                            LinearRouteTable::Unified)) {
        unified::nvfp4_dispatch(x, weight, out, policy, workspace, stream);
        return;
    }
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("nvfp4 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    if (!allows_a4(policy) || !shape.uses_a4(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("nvfp4 A4 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    shape.a4(x, weight, out, scratch, stream);
}
} // namespace ninfer::ops::detail
