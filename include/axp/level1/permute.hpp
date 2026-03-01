#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>
#include "../level0/communication.hpp"
#include "../level0/fragment.hpp"
#include "registry.hpp"
#include "detail/compose.hpp"

namespace axp::level1 {

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class Pattern,
         int BlockThreads = 0, class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using Permute = registry::Select<
    registry::PermutePattern<Recipe, Payload, InSubj, OutSubj, ExecGroup, Pattern, BlockThreads, InExtra, OutExtra>, CapT
>;

} // namespace axp::level1

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level1::registry {

template<class Recipe, class FragPayload, class InSubj, class OutSubj, class ExecGroup, class Pattern,
         class InExtra, class OutExtra, class Cap>
struct resolve_impl<
    PermutePattern<Recipe, FragPayload, InSubj, OutSubj, ExecGroup, Pattern, 0, InExtra, OutExtra>,
    Cap,
    std::enable_if_t<
        iro::contract::FragmentPayload<FragPayload> &&
        (std::is_same_v<ExecGroup, iro::exec::warp> || iro::exec::is_warpgroup_v<ExecGroup>)
    >
> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::FragmentPermute<Recipe, FragPayload, InSubj, OutSubj, ExecGroup, Pattern, InExtra, OutExtra>,
        Cap
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class Pattern, int BlockThreads,
         class InExtra, class OutExtra, class Cap>
struct resolve_impl<
    PermutePattern<Recipe, Payload, InSubj, OutSubj, iro::exec::block, Pattern, BlockThreads, InExtra, OutExtra>,
    Cap,
    std::enable_if_t<
        (BlockThreads > 0) &&
        (iro::contract::FragmentPayload<Payload> ||
         iro::contract::ScalarPayload<Payload> ||
         iro::contract::VectorPayload<Payload>)
    >
> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::PermuteCross<Recipe, Payload, InSubj, OutSubj, iro::exec::block, Pattern, BlockThreads, InExtra, OutExtra>,
        Cap
    >;
};

} // namespace axp::level1::registry
#endif // AXP_LIBRARY_BUILD
