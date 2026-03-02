#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level2/passthrough.hpp"
#include "detail/compose.hpp"
#include "detail/reg_pressure.hpp"
#include "registry.hpp"

namespace axp::level3::histogram {

namespace detail {
template<class Obligation, int I>
using in_port_t = axp::level3::detail::in_port_t<Obligation, I>;
template<class Obligation, int I>
using out_port_t = axp::level3::detail::out_port_t<Obligation, I>;
using axp::level3::detail::reg_pressure_const;

struct flush_event_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.histogram.flush_event"); };
} // namespace detail

// Histogram tile (shared privatization + global atomic reduce).
template<class Recipe, class ValuePayload, class IndexPayload, class SharedTile, class OutTile,
         class ValueSubj, class IndexSubj, class SharedSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class CapT>
struct HistogramTileImpl {
    static_assert(std::is_same_v<typename SharedTile::elem, typename OutTile::elem>,
                  "HistogramTile: SharedTile elem must match OutTile elem");
    static_assert(SharedTile::shape::rank == 1 && OutTile::shape::rank == 1,
                  "HistogramTile: rank-1 tiles required");
    static_assert(SharedTile::shape::size == OutTile::shape::size,
                  "HistogramTile: shared/global tile size mismatch");
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>,
                  "HistogramTile: ExecGroup must be block");

    using Zero = axp::level2::low::TileZero<
        Recipe, SharedTile, SharedSubj, ExecGroup
    >;

    using LocalAtomic = axp::level2::low::AtomicAdd<
        Recipe, ValuePayload, IndexPayload, ValuePayload, SharedTile,
        ValueSubj, IndexSubj, OutValSubj, SharedSubj, ExecGroup
    >;

    using Flush = axp::level2::low::ReduceSharedToGlobalAtomicAdd<
        Recipe, SharedTile, OutTile, SharedSubj, OutSubj, ExecGroup
    >;

    using AtomicDone = axp::level2::low::MarkAtomicDoneFromTile<
        Recipe, OutTile, OutSubj, ExecGroup, iro::scope::device, iro::memory_order::release, iro::token::lifetime::block
    >;

    using FlushEvent = axp::level2::low::EventFromAtomicDone<
        Recipe, OutSubj, iro::scope::device, detail::flush_event_tag, ExecGroup,
        iro::memory_order::release, iro::token::lifetime::block
    >;

    using TileOutIn = axp::level2::low::TileBoundaryIn<
        Recipe, OutTile, OutSubj, ExecGroup, iro::token::lifetime::block
    >;
    using RegPressure = detail::reg_pressure_const<12>;

    using TileOut = axp::level2::low::TileBoundaryOut<
        Recipe, OutTile, OutSubj, iro::exec::block, iro::token::lifetime::block
    >;

    using obligations = iro::util::type_list<RegPressure, TileOutIn, Zero, LocalAtomic, Flush, AtomicDone, FlushEvent, TileOut>;
    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<Zero, 0>, detail::in_port_t<LocalAtomic, 0>>,
        iro::compose::Edge<detail::out_port_t<LocalAtomic, 1>, detail::in_port_t<Flush, 0>>,
        iro::compose::Edge<detail::out_port_t<TileOutIn, 0>, detail::in_port_t<Flush, 1>>,
        iro::compose::Edge<detail::out_port_t<Flush, 0>, detail::in_port_t<AtomicDone, 0>>,
        iro::compose::Edge<detail::out_port_t<AtomicDone, 0>, detail::in_port_t<FlushEvent, 0>>,
        iro::compose::Edge<detail::out_port_t<Flush, 0>, detail::in_port_t<TileOut, 0>>
    >;
    using type = axp::level3::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

} // namespace axp::level3::histogram

namespace axp::level3 {

template<class Recipe, class ValuePayload, class IndexPayload, class SharedTile, class OutTile,
         class ValueSubj, class IndexSubj, class SharedSubj, class OutValSubj, class OutSubj,
         class ExecGroup = iro::exec::block>
struct HistogramTileConfig {
    using recipe = Recipe;
    using value_payload = ValuePayload;
    using index_payload = IndexPayload;
    using shared_tile = SharedTile;
    using out_tile = OutTile;
    using value_subj = ValueSubj;
    using index_subj = IndexSubj;
    using shared_subj = SharedSubj;
    using out_val_subj = OutValSubj;
    using out_subj = OutSubj;
    using exec_group = ExecGroup;
};

template<class Config, class CapT = axp::target_cap>
using HistogramTile = registry::Select<registry::HistogramTilePattern<
    typename Config::recipe,
    typename Config::value_payload, typename Config::index_payload,
    typename Config::shared_tile, typename Config::out_tile,
    typename Config::value_subj, typename Config::index_subj, typename Config::shared_subj,
    typename Config::out_val_subj, typename Config::out_subj, typename Config::exec_group>, CapT>;

} // namespace axp::level3

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level3::registry {

template<class Recipe, class ValuePayload, class IndexPayload, class SharedTile, class OutTile,
         class ValueSubj, class IndexSubj, class SharedSubj, class OutValSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<HistogramTilePattern<
    Recipe, ValuePayload, IndexPayload, SharedTile, OutTile,
    ValueSubj, IndexSubj, SharedSubj, OutValSubj, OutSubj, ExecGroup>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level3::histogram::HistogramTileImpl<
        Recipe, ValuePayload, IndexPayload, SharedTile, OutTile,
        ValueSubj, IndexSubj, SharedSubj, OutValSubj, OutSubj, ExecGroup, Cap
    >::type;
};

} // namespace axp::level3::registry
#endif // AXP_LIBRARY_BUILD
