#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level1/passthrough.hpp"
#include "registry.hpp"
#include "detail/compose.hpp"

namespace axp::level2 {

// Matmul pattern selection (warp vs warpgroup based on ExecGroup + capability).
template<class Recipe, class Shape, class ATile, class BTile, class AccFrag,
         class ASubj, class BSubj, class AccSubj, class ExecGroup, class WgmmaSubj,
         class CapT = axp::target_cap>
using Matmul = registry::Select<registry::MatmulPattern<
    Recipe, Shape, ATile, BTile, AccFrag, ASubj, BSubj, AccSubj, ExecGroup, WgmmaSubj>, CapT>;

// Explicit warp-level MMA.
template<class Recipe, class Shape, class ATile, class BTile, class AccFrag,
         class ASubj, class BSubj, class AccSubj, class CapT = axp::target_cap>
using MatmulWarp = registry::Select<registry::MatmulWarpPattern<
    Recipe, Shape, ATile, BTile, AccFrag, ASubj, BSubj, AccSubj>, CapT>;

// Explicit warpgroup-level WGMMA (descriptor-based).
template<class Recipe, class Shape, class ADesc, class BDesc, class AccFrag,
         class ADescSubj, class BDescSubj, class AccSubj, class WgmmaSubj, class ExecGroup,
         class CapT = axp::target_cap>
using MatmulWarpgroup = registry::Select<registry::MatmulWarpgroupPattern<
    Recipe, Shape, ADesc, BDesc, AccFrag, ADescSubj, BDescSubj, AccSubj, WgmmaSubj, ExecGroup>, CapT>;

} // namespace axp::level2

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level2::registry {

template<class Recipe, class Shape, class ATile, class BTile, class AccFrag,
         class ASubj, class BSubj, class AccSubj, class ExecGroup, class WgmmaSubj, class Cap>
struct resolve_impl<
    MatmulPattern<Recipe, Shape, ATile, BTile, AccFrag, ASubj, BSubj, AccSubj, ExecGroup, WgmmaSubj>,
    Cap,
    std::enable_if_t<std::is_same_v<ExecGroup, iro::exec::warp>>> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::low::MMA<Recipe, Shape, ATile, BTile, AccFrag, ASubj, BSubj, AccSubj>,
        Cap
    >;
};

template<class Recipe, class Shape, class ADesc, class BDesc, class AccFrag,
         class ADescSubj, class BDescSubj, class AccSubj, class ExecGroup, class WgmmaSubj, class Cap>
struct resolve_impl<
    MatmulPattern<Recipe, Shape, ADesc, BDesc, AccFrag, ADescSubj, BDescSubj, AccSubj, ExecGroup, WgmmaSubj>,
    Cap,
    std::enable_if_t<iro::exec::is_warpgroup_v<ExecGroup> && Cap::has_wgmma>> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::low::WarpgroupMma<Recipe, Shape, ADesc, BDesc, AccFrag, ADescSubj, BDescSubj, AccSubj, WgmmaSubj>,
        Cap
    >;
};

template<class Recipe, class Shape, class ATile, class BTile, class AccFrag,
         class ASubj, class BSubj, class AccSubj, class Cap>
struct resolve_impl<MatmulWarpPattern<Recipe, Shape, ATile, BTile, AccFrag, ASubj, BSubj, AccSubj>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::low::MMA<Recipe, Shape, ATile, BTile, AccFrag, ASubj, BSubj, AccSubj>,
        Cap
    >;
};

template<class Recipe, class Shape, class ADesc, class BDesc, class AccFrag,
         class ADescSubj, class BDescSubj, class AccSubj, class WgmmaSubj, class ExecGroup, class Cap>
struct resolve_impl<MatmulWarpgroupPattern<Recipe, Shape, ADesc, BDesc, AccFrag, ADescSubj, BDescSubj, AccSubj, WgmmaSubj, ExecGroup>, Cap,
                    std::enable_if_t<iro::exec::is_warpgroup_v<ExecGroup> && Cap::has_wgmma>> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::low::WarpgroupMma<Recipe, Shape, ADesc, BDesc, AccFrag, ADescSubj, BDescSubj, AccSubj, WgmmaSubj>,
        Cap
    >;
};

} // namespace axp::level2::registry
#endif // AXP_LIBRARY_BUILD
