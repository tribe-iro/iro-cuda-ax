#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level0/atomic.hpp"
#include "registry.hpp"
#include "detail/compose.hpp"

namespace axp::level1 {

template<class Recipe, class Subject, class ExecGroup, class ScopeT,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block,
         class CapT = axp::target_cap>
using MarkAtomicDone = registry::Select<
    registry::MarkAtomicDonePattern<Recipe, Subject, ExecGroup, ScopeT, OrderT, Lifetime>, CapT>;

template<class Recipe, class Subject, class ExecGroup, class ScopeT,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block,
         class CapT = axp::target_cap>
using RequireAtomicDone = registry::Select<
    registry::RequireAtomicDonePattern<Recipe, Subject, ExecGroup, ScopeT, OrderT, Lifetime>, CapT>;

template<class Recipe, class TilePayload, class Subject, class ExecGroup, class ScopeT,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block,
         class CapT = axp::target_cap>
using MarkAtomicDoneFromTile = registry::Select<
    registry::MarkAtomicDoneFromTilePattern<Recipe, TilePayload, Subject, ExecGroup, ScopeT, OrderT, Lifetime>, CapT>;

} // namespace axp::level1

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level1::registry {

template<class Recipe, class Subject, class ExecGroup, class ScopeT, class OrderT, class Lifetime, class Cap>
struct resolve_impl<MarkAtomicDonePattern<Recipe, Subject, ExecGroup, ScopeT, OrderT, Lifetime>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::MarkAtomicDone<Recipe, Subject, ExecGroup, ScopeT, OrderT, Lifetime>,
        Cap
    >;
};

template<class Recipe, class Subject, class ExecGroup, class ScopeT, class OrderT, class Lifetime, class Cap>
struct resolve_impl<RequireAtomicDonePattern<Recipe, Subject, ExecGroup, ScopeT, OrderT, Lifetime>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::RequireAtomicDone<Recipe, Subject, ExecGroup, ScopeT, OrderT, Lifetime>,
        Cap
    >;
};

template<class Recipe, class TilePayload, class Subject, class ExecGroup, class ScopeT, class OrderT, class Lifetime, class Cap>
struct resolve_impl<MarkAtomicDoneFromTilePattern<Recipe, TilePayload, Subject, ExecGroup, ScopeT, OrderT, Lifetime>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::MarkAtomicDoneFromTile<Recipe, TilePayload, Subject, ExecGroup, ScopeT, OrderT, Lifetime>,
        Cap
    >;
};

} // namespace axp::level1::registry
#endif // AXP_LIBRARY_BUILD

