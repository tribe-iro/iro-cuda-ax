#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level1/atomic.hpp"
#include "registry.hpp"
#include "detail/compose.hpp"

namespace axp::level2 {

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

} // namespace axp::level2

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level2::registry {

template<class Recipe, class Subject, class ExecGroup, class ScopeT, class OrderT, class Lifetime, class Cap>
struct resolve_impl<MarkAtomicDonePattern<Recipe, Subject, ExecGroup, ScopeT, OrderT, Lifetime>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::MarkAtomicDone<Recipe, Subject, ExecGroup, ScopeT, OrderT, Lifetime, Cap>,
        Cap
    >;
};

template<class Recipe, class Subject, class ExecGroup, class ScopeT, class OrderT, class Lifetime, class Cap>
struct resolve_impl<RequireAtomicDonePattern<Recipe, Subject, ExecGroup, ScopeT, OrderT, Lifetime>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::RequireAtomicDone<Recipe, Subject, ExecGroup, ScopeT, OrderT, Lifetime, Cap>,
        Cap
    >;
};

template<class Recipe, class TilePayload, class Subject, class ExecGroup, class ScopeT, class OrderT, class Lifetime, class Cap>
struct resolve_impl<MarkAtomicDoneFromTilePattern<Recipe, TilePayload, Subject, ExecGroup, ScopeT, OrderT, Lifetime>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::MarkAtomicDoneFromTile<Recipe, TilePayload, Subject, ExecGroup, ScopeT, OrderT, Lifetime, Cap>,
        Cap
    >;
};

} // namespace axp::level2::registry
#endif // AXP_LIBRARY_BUILD

