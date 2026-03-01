#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level0/memory.hpp"
#include "registry.hpp"
#include "detail/compose.hpp"

namespace axp::level1 {

template<class Recipe, class InTile, class IndexPayload, class OutPayload,
         class InSubj, class IndexSubj, class OutSubj, class ExecGroup,
         class CachePolicy = axp::cache::ca,
         class CapT = axp::target_cap>
using GatherGlobal = registry::Select<
    registry::GatherGlobalPattern<Recipe, InTile, IndexPayload, OutPayload,
                                  InSubj, IndexSubj, OutSubj, ExecGroup, CachePolicy>, CapT
>;

template<class Recipe, class InPayload, class IndexPayload, class OutTile,
         class InSubj, class IndexSubj, class OutSubj, class ExecGroup,
         class CachePolicy = axp::cache::wb,
         class CapT = axp::target_cap>
using ScatterGlobal = registry::Select<
    registry::ScatterGlobalPattern<Recipe, InPayload, IndexPayload, OutTile,
                                   InSubj, IndexSubj, OutSubj, ExecGroup, CachePolicy>, CapT
>;

} // namespace axp::level1

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level1::registry {

template<class Recipe, class InTile, class IndexPayload, class OutPayload,
         class InSubj, class IndexSubj, class OutSubj, class ExecGroup, class CachePolicy,
         class Cap>
struct resolve_impl<GatherGlobalPattern<Recipe, InTile, IndexPayload, OutPayload,
                                   InSubj, IndexSubj, OutSubj, ExecGroup, CachePolicy>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::GatherGlobal<Recipe, InTile, IndexPayload, OutPayload,
                                  InSubj, IndexSubj, OutSubj, ExecGroup, CachePolicy>,
        Cap
    >;
};

template<class Recipe, class InPayload, class IndexPayload, class OutTile,
         class InSubj, class IndexSubj, class OutSubj, class ExecGroup, class CachePolicy,
         class Cap>
struct resolve_impl<ScatterGlobalPattern<Recipe, InPayload, IndexPayload, OutTile,
                                    InSubj, IndexSubj, OutSubj, ExecGroup, CachePolicy>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::ScatterGlobal<Recipe, InPayload, IndexPayload, OutTile,
                                   InSubj, IndexSubj, OutSubj, ExecGroup, CachePolicy>,
        Cap
    >;
};

} // namespace axp::level1::registry
#endif // AXP_LIBRARY_BUILD
