#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level1/passthrough.hpp"
#include "detail/compose.hpp"

namespace axp::level2::pipeline {

namespace detail {

template<class Issue, class Wait, int OutIndex, int InIndex, bool Enable>
struct edge_if;

template<class Issue, class Wait, int OutIndex, int InIndex>
struct edge_if<Issue, Wait, OutIndex, InIndex, true> {
    using type = iro::util::type_list<
        iro::compose::Edge<
            iro::compose::out_port_ref<Issue, OutIndex>,
            iro::compose::in_port_ref<Wait, InIndex>
        >
    >;
};

template<class Issue, class Wait, int OutIndex, int InIndex>
struct edge_if<Issue, Wait, OutIndex, InIndex, false> {
    using type = iro::util::type_list<>;
};

template<class Issue, class Wait,
         int SlotOut, int SlotIn,
         int BarrierOut, int BarrierIn,
         bool HasBarrier>
struct stage_edges {
    static constexpr bool has_slot = (SlotOut >= 0);
    using slot_edge = typename edge_if<Issue, Wait, SlotOut, SlotIn, has_slot>::type;
    using barrier_edge = typename edge_if<Issue, Wait, BarrierOut, BarrierIn, HasBarrier>::type;
    using type = iro::util::concat_t<slot_edge, barrier_edge>;
};

} // namespace detail

template<class Issue, class Wait,
         int SlotOut = 0, int SlotIn = 0,
         int BarrierOut = 1, int BarrierIn = 1,
         bool HasBarrier = false,
         class CapT = axp::target_cap>
struct StageImpl {
    using obligations = iro::util::type_list<Issue, Wait>;
    using edges = typename detail::stage_edges<Issue, Wait, SlotOut, SlotIn, BarrierOut, BarrierIn, HasBarrier>::type;
    using type = axp::level2::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

// Pipeline stage: connect issue -> wait with explicit slot/barrier wiring.
template<class Issue, class Wait,
         int SlotOut = 0, int SlotIn = 0,
         int BarrierOut = 1, int BarrierIn = 1,
         bool HasBarrier = false,
         class CapT = axp::target_cap>
using Stage = typename StageImpl<Issue, Wait, SlotOut, SlotIn, BarrierOut, BarrierIn, HasBarrier, CapT>::type;

// Rotate software pipeline slots (producer/consumer index).
template<class Recipe, class IndexPayload, class InSubj, class OutSubj, class ExecGroup, int Stages,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using RotateSlots = axp::level2::detail::as_composition_t<
    axp::level1::low::PipelineAdvance<Recipe, IndexPayload, InSubj, OutSubj, ExecGroup, Stages, InExtra, OutExtra>,
    CapT
>;

// Commit consumption (ready -> used).
template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime, long long Bytes,
         class CapT = axp::target_cap>
using Commit = axp::level2::detail::as_composition_t<
    axp::level1::low::MarkConsumed<Recipe, SlotSubj, ExecGroup, Lifetime, Bytes>,
    CapT
>;

// Retire slot (used -> free).
template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime, class CapT = axp::target_cap>
using Retire = axp::level2::detail::as_composition_t<
    axp::level1::low::ReleaseSlot<Recipe, SlotSubj, ExecGroup, Lifetime>,
    CapT
>;

} // namespace axp::level2::pipeline
