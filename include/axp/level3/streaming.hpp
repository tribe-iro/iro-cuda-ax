#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>
#include "../bundles/checklist.hpp"
#include "../level2/passthrough.hpp"
#include "../level2/scan.hpp"
#include "detail/compose.hpp"
#include "detail/reg_pressure.hpp"
#include "registry.hpp"

namespace axp::level3::streaming {

namespace detail {

template<class Obligation, int I>
using in_port_t = axp::level3::detail::in_port_t<Obligation, I>;
template<class Obligation, int I>
using out_port_t = axp::level3::detail::out_port_t<Obligation, I>;
using axp::level3::detail::reg_pressure_const;

struct scan_add_tag {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.streaming.scan_add");
};

template<class Payload>
consteval int payload_count() {
    if constexpr (iro::contract::ScalarPayload<Payload>) {
        return 1;
    } else if constexpr (iro::contract::VectorPayload<Payload>) {
        return Payload::lanes;
    } else {
        return 0;
    }
}

template<class Payload>
consteval bool integral_index_payload() {
    static_assert(iro::contract::ScalarPayload<Payload> || iro::contract::VectorPayload<Payload>,
                  "StreamingMicrobatchTile: IndexPayload must be Scalar/Vector");
    static_assert(std::is_integral_v<typename Payload::elem::storage_t>,
                  "StreamingMicrobatchTile: IndexPayload element must be integral");
    static_assert(sizeof(typename Payload::elem::storage_t) <= 4,
                  "StreamingMicrobatchTile: IndexPayload element must be <= 4 bytes");
    return true;
}

} // namespace detail

// Event-driven microbatch tile:
// ingest(value/index/state) -> process(block scan) -> emit(atomic update + completion event).
template<class Recipe, class ValuePayload, class IndexPayload, class StateTile,
         class ValueSubj, class IndexSubj, class StateSubj, class AtomicOutSubj,
         class DependEventTag, class PhaseTag, class DoneEventTag, class ExecGroup, class CapT>
struct StreamingMicrobatchTileImpl {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>,
                  "StreamingMicrobatchTile: ExecGroup must be block");
    static_assert(std::is_same_v<typename Recipe::in, typename Recipe::out>,
                  "StreamingMicrobatchTile: Recipe::in and Recipe::out must match");
    static_assert(iro::contract::ScalarPayload<ValuePayload> || iro::contract::VectorPayload<ValuePayload>,
                  "StreamingMicrobatchTile: ValuePayload must be Scalar/Vector");
    static_assert(detail::integral_index_payload<IndexPayload>(),
                  "StreamingMicrobatchTile: invalid IndexPayload");
    static_assert(detail::payload_count<ValuePayload>() == detail::payload_count<IndexPayload>(),
                  "StreamingMicrobatchTile: ValuePayload/IndexPayload lane count mismatch");
    static_assert(iro::contract::TilePayload<StateTile>, "StreamingMicrobatchTile: StateTile must be Tile");
    static_assert(StateTile::shape::rank == 1, "StreamingMicrobatchTile: StateTile must be rank-1");
    static_assert(std::is_same_v<typename StateTile::space, iro::contract::space::global>,
                  "StreamingMicrobatchTile: StateTile must be global");
    static_assert(std::is_same_v<typename StateTile::elem, typename Recipe::out>,
                  "StreamingMicrobatchTile: StateTile elem must match Recipe::out");
    static_assert(std::is_same_v<typename ValuePayload::elem, typename Recipe::in>,
                  "StreamingMicrobatchTile: ValuePayload elem must match Recipe::in");
    static_assert(axp::bundle::check::subject_follows_derivation_policy<ValueSubj>() &&
                  axp::bundle::check::subject_follows_derivation_policy<IndexSubj>() &&
                  axp::bundle::check::subject_follows_derivation_policy<StateSubj>() &&
                  axp::bundle::check::subject_follows_derivation_policy<AtomicOutSubj>(),
                  "StreamingMicrobatchTile: subjects must follow canonical derivation policy");

    using RegPressure = detail::reg_pressure_const<10>;

    using DependOn = axp::level2::low::DependOnEvent<
        Recipe, StateSubj, DependEventTag, PhaseTag, ExecGroup, iro::token::lifetime::block
    >;

    using DependGate = axp::level2::low::DependOnEventGate<
        Recipe, ValuePayload, ValueSubj, StateSubj, DependEventTag, PhaseTag, ExecGroup, iro::token::lifetime::block
    >;

    using StateIn = axp::level2::low::TileBoundaryIn<
        Recipe, StateTile, StateSubj, ExecGroup, iro::token::lifetime::block
    >;

    using Scan = axp::level2::BlockScan<
        Recipe, ValuePayload, ValueSubj, ExecGroup,
        detail::scan_add_tag, axp::level2::proto::scan::scan::inclusive,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;

    using AtomicUpdate = axp::level2::low::AtomicAdd<
        Recipe, ValuePayload, IndexPayload, ValuePayload, StateTile,
        ValueSubj, IndexSubj, AtomicOutSubj, StateSubj, ExecGroup
    >;

    using AtomicDone = axp::level2::low::MarkAtomicDoneFromTile<
        Recipe, StateTile, StateSubj, ExecGroup,
        iro::scope::device, iro::memory_order::release, iro::token::lifetime::block
    >;

    using DoneEvent = axp::level2::low::EventFromAtomicDone<
        Recipe, StateSubj, iro::scope::device, DoneEventTag, ExecGroup,
        iro::memory_order::release, iro::token::lifetime::block
    >;

    using StateOut = axp::level2::low::TileBoundaryOut<
        Recipe, StateTile, StateSubj, ExecGroup, iro::token::lifetime::block
    >;

    using obligations = iro::util::type_list<
        RegPressure, DependOn, DependGate, StateIn, Scan, AtomicUpdate, AtomicDone, DoneEvent, StateOut>;
    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<DependOn, 0>, detail::in_port_t<DependGate, 0>>,
        iro::compose::Edge<detail::out_port_t<DependGate, 0>, detail::in_port_t<Scan, 0>>,
        iro::compose::Edge<detail::out_port_t<StateIn, 0>, detail::in_port_t<AtomicUpdate, 0>>,
        iro::compose::Edge<detail::out_port_t<Scan, 0>, detail::in_port_t<AtomicUpdate, 1>>,
        iro::compose::Edge<detail::out_port_t<AtomicUpdate, 1>, detail::in_port_t<AtomicDone, 0>>,
        iro::compose::Edge<detail::out_port_t<AtomicDone, 0>, detail::in_port_t<DoneEvent, 0>>,
        iro::compose::Edge<detail::out_port_t<AtomicUpdate, 1>, detail::in_port_t<StateOut, 0>>
    >;
    using type = axp::level3::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

} // namespace axp::level3::streaming

namespace axp::level3 {

template<class Recipe, class ValuePayload, class IndexPayload, class StateTile,
         class ValueSubj, class IndexSubj, class StateSubj, class AtomicOutSubj,
         class DependEventTag, class PhaseTag, class DoneEventTag,
         class ExecGroup = iro::exec::block>
struct StreamingMicrobatchTileConfig {
    using recipe = Recipe;
    using value_payload = ValuePayload;
    using index_payload = IndexPayload;
    using state_tile = StateTile;
    using value_subj = ValueSubj;
    using index_subj = IndexSubj;
    using state_subj = StateSubj;
    using atomic_out_subj = AtomicOutSubj;
    using depend_event_tag = DependEventTag;
    using phase_tag = PhaseTag;
    using done_event_tag = DoneEventTag;
    using exec_group = ExecGroup;
};

template<class Config, class CapT = axp::target_cap>
using StreamingMicrobatchTile = registry::Select<registry::StreamingMicrobatchTilePattern<
    typename Config::recipe,
    typename Config::value_payload,
    typename Config::index_payload,
    typename Config::state_tile,
    typename Config::value_subj,
    typename Config::index_subj,
    typename Config::state_subj,
    typename Config::atomic_out_subj,
    typename Config::depend_event_tag,
    typename Config::phase_tag,
    typename Config::done_event_tag,
    typename Config::exec_group>, CapT>;

} // namespace axp::level3

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level3::registry {

template<class Recipe, class ValuePayload, class IndexPayload, class StateTile,
         class ValueSubj, class IndexSubj, class StateSubj, class AtomicOutSubj,
         class DependEventTag, class PhaseTag, class DoneEventTag, class ExecGroup, class Cap>
struct resolve_impl<StreamingMicrobatchTilePattern<
    Recipe, ValuePayload, IndexPayload, StateTile,
    ValueSubj, IndexSubj, StateSubj, AtomicOutSubj,
    DependEventTag, PhaseTag, DoneEventTag, ExecGroup>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level3::streaming::StreamingMicrobatchTileImpl<
        Recipe, ValuePayload, IndexPayload, StateTile,
        ValueSubj, IndexSubj, StateSubj, AtomicOutSubj,
        DependEventTag, PhaseTag, DoneEventTag, ExecGroup, Cap
    >::type;
};

} // namespace axp::level3::registry
#endif // AXP_LIBRARY_BUILD
