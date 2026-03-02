#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level1/epoch.hpp"
#include "registry.hpp"
#include "detail/compose.hpp"

namespace axp::level2 {

template<class Recipe, class Subject, class EpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block,
         class CapT = axp::target_cap>
using InitEpoch = registry::Select<
    registry::InitEpochPattern<Recipe, Subject, EpochTag, ExecGroup, Lifetime>, CapT>;

template<class Recipe, class Subject, class PrevEpochTag, class NextEpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block,
         class CapT = axp::target_cap>
using AdvanceEpochToken = registry::Select<
    registry::AdvanceEpochTokenPattern<Recipe, Subject, PrevEpochTag, NextEpochTag, ExecGroup, Lifetime>, CapT>;

template<class Recipe, class Subject, class EpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block,
         class CapT = axp::target_cap>
using RequireEpoch = registry::Select<
    registry::RequireEpochPattern<Recipe, Subject, EpochTag, ExecGroup, Lifetime>, CapT>;

} // namespace axp::level2

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level2::registry {

template<class Recipe, class Subject, class EpochTag, class ExecGroup, class Lifetime, class Cap>
struct resolve_impl<InitEpochPattern<Recipe, Subject, EpochTag, ExecGroup, Lifetime>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::InitEpoch<Recipe, Subject, EpochTag, ExecGroup, Lifetime, Cap>,
        Cap
    >;
};

template<class Recipe, class Subject, class PrevEpochTag, class NextEpochTag, class ExecGroup, class Lifetime, class Cap>
struct resolve_impl<AdvanceEpochTokenPattern<Recipe, Subject, PrevEpochTag, NextEpochTag, ExecGroup, Lifetime>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::AdvanceEpochToken<Recipe, Subject, PrevEpochTag, NextEpochTag, ExecGroup, Lifetime, Cap>,
        Cap
    >;
};

template<class Recipe, class Subject, class EpochTag, class ExecGroup, class Lifetime, class Cap>
struct resolve_impl<RequireEpochPattern<Recipe, Subject, EpochTag, ExecGroup, Lifetime>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::RequireEpoch<Recipe, Subject, EpochTag, ExecGroup, Lifetime, Cap>,
        Cap
    >;
};

} // namespace axp::level2::registry
#endif // AXP_LIBRARY_BUILD

