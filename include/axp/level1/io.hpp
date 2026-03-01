#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level0/stage.hpp"
#include "../level0/memory.hpp"
#include "detail/compose.hpp"

namespace axp::level1 {

// Async vector load: Issue + Wait (pipeline)
template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class CapT = axp::target_cap>
struct VecLoadImpl {
    using Issue = axp::level0::AsyncCopy<
        Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots
    >;
    using Wait = axp::level0::WaitSlot<
        Recipe, OutTile, SlotSubj, ExecGroup, Lifetime
    >;

    using obligations = iro::util::type_list<Issue, Wait>;
    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<Issue, 0>, detail::in_port_t<Wait, 0>>
    >;
    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class CapT = axp::target_cap>
using VecLoad = typename VecLoadImpl<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, CapT>::type;

// Vector store: register tile -> global tile
template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CachePolicy, class InDist, class OutDist, class CapT = axp::target_cap>
struct VecStoreImpl {
    using Store = axp::level0::StGlobal<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, CachePolicy, InDist, OutDist>;
    using obligations = iro::util::type_list<Store>;
    using edges = iro::util::type_list<>;
    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CachePolicy = axp::cache::wb,
         class InDist = iro::contract::no_dist, class OutDist = iro::contract::no_dist,
         class CapT = axp::target_cap>
using VecStore = typename VecStoreImpl<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, CachePolicy, InDist, OutDist, CapT>::type;

} // namespace axp::level1
