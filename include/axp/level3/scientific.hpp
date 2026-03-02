#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>
#include "../bundles/checklist.hpp"
#include "../level2/passthrough.hpp"
#include "../level2/gather.hpp"
#include "../level2/scan.hpp"
#include "detail/compose.hpp"
#include "detail/reg_pressure.hpp"
#include "registry.hpp"

namespace axp::level3::scientific {

namespace detail {

template<class Obligation, int I>
using in_port_t = axp::level3::detail::in_port_t<Obligation, I>;
template<class Obligation, int I>
using out_port_t = axp::level3::detail::out_port_t<Obligation, I>;
using axp::level3::detail::reg_pressure_const;

struct scan_add_tag {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.scientific.scan_add");
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
                  "Scientific pattern: IndexPayload must be Scalar/Vector");
    static_assert(std::is_integral_v<typename Payload::elem::storage_t>,
                  "Scientific pattern: IndexPayload element must be integral");
    static_assert(sizeof(typename Payload::elem::storage_t) <= 4,
                  "Scientific pattern: IndexPayload element must be <= 4 bytes");
    return true;
}

} // namespace detail

// Sparse gather + segmented scan + scatter (irregular HPC flow).
template<class Recipe, class InTile, class OutTile, class GatherPayload, class IndexPayload,
         class InSubj, class IndexSubj, class GatherSubj, class OutSubj, class EmitEventTag,
         int SegmentWidth, class ExecGroup, class CapT>
struct SparseSegmentedTileImpl {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "SparseSegmentedTile: ExecGroup must be warp");
    static_assert(std::is_same_v<typename Recipe::in, typename Recipe::out>,
                  "SparseSegmentedTile: Recipe::in and Recipe::out must match");
    static_assert(iro::contract::TilePayload<InTile> && iro::contract::TilePayload<OutTile>,
                  "SparseSegmentedTile: InTile/OutTile must be tiles");
    static_assert(InTile::shape::rank == 1 && OutTile::shape::rank == 1,
                  "SparseSegmentedTile: rank-1 tiles required");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::global> &&
                  std::is_same_v<typename OutTile::space, iro::contract::space::global>,
                  "SparseSegmentedTile: InTile/OutTile must be global");
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in> &&
                  std::is_same_v<typename OutTile::elem, typename Recipe::out>,
                  "SparseSegmentedTile: tile element/recipe mismatch");
    static_assert(iro::contract::ScalarPayload<GatherPayload> || iro::contract::VectorPayload<GatherPayload>,
                  "SparseSegmentedTile: GatherPayload must be Scalar/Vector");
    static_assert(std::is_same_v<typename GatherPayload::elem, typename Recipe::out>,
                  "SparseSegmentedTile: GatherPayload elem must match Recipe::out");
    static_assert(detail::integral_index_payload<IndexPayload>(),
                  "SparseSegmentedTile: invalid IndexPayload");
    static_assert(detail::payload_count<GatherPayload>() == detail::payload_count<IndexPayload>(),
                  "SparseSegmentedTile: GatherPayload/IndexPayload lane count mismatch");
    static_assert(SegmentWidth > 0 && SegmentWidth <= 32 && ((SegmentWidth & (SegmentWidth - 1)) == 0),
                  "SparseSegmentedTile: SegmentWidth must be power-of-two in [1,32]");
    static_assert(axp::bundle::check::subject_follows_derivation_policy<InSubj>() &&
                  axp::bundle::check::subject_follows_derivation_policy<IndexSubj>() &&
                  axp::bundle::check::subject_follows_derivation_policy<GatherSubj>() &&
                  axp::bundle::check::subject_follows_derivation_policy<OutSubj>(),
                  "SparseSegmentedTile: subjects must follow canonical derivation policy");

    using RegPressure = detail::reg_pressure_const<9>;

    using Gather = axp::level2::GatherGlobal<
        Recipe, InTile, IndexPayload, GatherPayload,
        InSubj, IndexSubj, GatherSubj, ExecGroup, axp::cache::ca, CapT
    >;

    using SegScan = axp::level2::WarpSegmentedScan<
        Recipe, GatherPayload, GatherSubj, ExecGroup,
        detail::scan_add_tag, axp::level2::proto::scan::scan::inclusive, SegmentWidth,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;

    using Scatter = axp::level2::ScatterGlobal<
        Recipe, GatherPayload, IndexPayload, OutTile,
        GatherSubj, IndexSubj, OutSubj, ExecGroup, axp::cache::wb, CapT
    >;

    using Emit = axp::level2::low::EmitEventAfter<
        Recipe, OutTile, OutSubj, OutSubj, EmitEventTag, ExecGroup, iro::token::lifetime::warp
    >;

    using OutBoundary = axp::level2::low::TileBoundaryOut<
        Recipe, OutTile, OutSubj, ExecGroup, iro::token::lifetime::warp
    >;

    using obligations = iro::util::type_list<RegPressure, Gather, SegScan, Scatter, OutBoundary, Emit>;
    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<Gather, 0>, detail::in_port_t<SegScan, 0>>,
        iro::compose::Edge<detail::out_port_t<SegScan, 0>, detail::in_port_t<Scatter, 0>>,
        iro::compose::Edge<detail::out_port_t<Scatter, 0>, detail::in_port_t<OutBoundary, 0>>,
        iro::compose::Edge<detail::out_port_t<OutBoundary, 0>, detail::in_port_t<Emit, 0>>
    >;
    using type = axp::level3::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

// Swizzle-centric tile transform (layout plane as explicit protocol edge).
template<class Recipe, class InTile, class OutTile, class TileSubj,
         class SwizzleAtom, class EmitEventTag, class ExecGroup, class CapT>
struct SwizzleTileImpl {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>,
                  "SwizzleTile: ExecGroup must be block");
    static_assert(iro::contract::TilePayload<InTile> && iro::contract::TilePayload<OutTile>,
                  "SwizzleTile: InTile/OutTile must be tiles");
    static_assert(InTile::shape::rank == 2 && OutTile::shape::rank == 2,
                  "SwizzleTile: rank-2 tiles required");
    static_assert(InTile::shape::template dim<0>() == OutTile::shape::template dim<0>() &&
                  InTile::shape::template dim<1>() == OutTile::shape::template dim<1>(),
                  "SwizzleTile: tile shape mismatch");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::shared> &&
                  std::is_same_v<typename OutTile::space, iro::contract::space::shared>,
                  "SwizzleTile: InTile/OutTile must be shared");
    static_assert(std::is_same_v<typename InTile::elem, typename OutTile::elem> &&
                  std::is_same_v<typename InTile::elem, typename Recipe::in> &&
                  std::is_same_v<typename OutTile::elem, typename Recipe::out>,
                  "SwizzleTile: recipe/tile element mismatch");
    static_assert(std::is_same_v<typename Recipe::in, typename Recipe::out>,
                  "SwizzleTile: Recipe::in and Recipe::out must match");
    static_assert(axp::bundle::check::subject_follows_derivation_policy<TileSubj>(),
                  "SwizzleTile: TileSubj must follow canonical derivation policy");

    using RegPressure = detail::reg_pressure_const<6>;

    using TileIn = axp::level2::low::TileBoundaryIn<
        Recipe, InTile, TileSubj, ExecGroup, iro::token::lifetime::block
    >;

    using RequiredTokens = iro::token::bundle_list<
        axp::level2::proto::view::ViewReadableSync<TileSubj, ExecGroup, iro::token::lifetime::block>
    >;
    using ProvidedTokens = iro::token::bundle_list<
        axp::level2::proto::view::ViewProducedSync<TileSubj, ExecGroup, iro::token::lifetime::block>
    >;

    using Swizzle = typename axp::level2::low::Swizzle<
        Recipe, InTile, OutTile, TileSubj, ExecGroup,
        RequiredTokens, ProvidedTokens, SwizzleAtom
    >::obligation;

    using TileOut = axp::level2::low::TileBoundaryOut<
        Recipe, OutTile, TileSubj, ExecGroup, iro::token::lifetime::block
    >;

    using Emit = axp::level2::low::EmitEventAfter<
        Recipe, OutTile, TileSubj, TileSubj, EmitEventTag, ExecGroup, iro::token::lifetime::block
    >;

    using obligations = iro::util::type_list<RegPressure, TileIn, Swizzle, TileOut, Emit>;
    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<TileIn, 0>, detail::in_port_t<Swizzle, 0>>,
        iro::compose::Edge<detail::out_port_t<Swizzle, 0>, detail::in_port_t<TileOut, 0>>,
        iro::compose::Edge<detail::out_port_t<TileOut, 0>, detail::in_port_t<Emit, 0>>
    >;
    using type = axp::level3::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

} // namespace axp::level3::scientific

namespace axp::level3 {

template<class Recipe, class InTile, class OutTile, class GatherPayload, class IndexPayload,
         class InSubj, class IndexSubj, class GatherSubj, class OutSubj, class EmitEventTag,
         int SegmentWidth, class ExecGroup = iro::exec::warp>
struct ScientificSparseSegmentedTileConfig {
    using recipe = Recipe;
    using in_tile = InTile;
    using out_tile = OutTile;
    using gather_payload = GatherPayload;
    using index_payload = IndexPayload;
    using in_subj = InSubj;
    using index_subj = IndexSubj;
    using gather_subj = GatherSubj;
    using out_subj = OutSubj;
    using emit_event_tag = EmitEventTag;
    static constexpr int segment_width = SegmentWidth;
    using exec_group = ExecGroup;
};

template<class Recipe, class InTile, class OutTile, class TileSubj,
         class SwizzleAtom, class EmitEventTag, class ExecGroup = iro::exec::block>
struct ScientificSwizzleTileConfig {
    using recipe = Recipe;
    using in_tile = InTile;
    using out_tile = OutTile;
    using tile_subj = TileSubj;
    using swizzle_atom = SwizzleAtom;
    using emit_event_tag = EmitEventTag;
    using exec_group = ExecGroup;
};

template<class Config, class CapT = axp::target_cap>
using ScientificSparseSegmentedTile = registry::Select<registry::ScientificSparseSegmentedTilePattern<
    typename Config::recipe, typename Config::in_tile, typename Config::out_tile,
    typename Config::gather_payload, typename Config::index_payload,
    typename Config::in_subj, typename Config::index_subj,
    typename Config::gather_subj, typename Config::out_subj,
    typename Config::emit_event_tag, Config::segment_width, typename Config::exec_group>, CapT>;

template<class Config, class CapT = axp::target_cap>
using ScientificSwizzleTile = registry::Select<registry::ScientificSwizzleTilePattern<
    typename Config::recipe, typename Config::in_tile, typename Config::out_tile,
    typename Config::tile_subj, typename Config::swizzle_atom,
    typename Config::emit_event_tag, typename Config::exec_group>, CapT>;

} // namespace axp::level3

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level3::registry {

template<class Recipe, class InTile, class OutTile, class GatherPayload, class IndexPayload,
         class InSubj, class IndexSubj, class GatherSubj, class OutSubj, class EmitEventTag,
         int SegmentWidth, class ExecGroup, class Cap>
struct resolve_impl<ScientificSparseSegmentedTilePattern<
    Recipe, InTile, OutTile, GatherPayload, IndexPayload,
    InSubj, IndexSubj, GatherSubj, OutSubj, EmitEventTag, SegmentWidth, ExecGroup>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level3::scientific::SparseSegmentedTileImpl<
        Recipe, InTile, OutTile, GatherPayload, IndexPayload,
        InSubj, IndexSubj, GatherSubj, OutSubj, EmitEventTag, SegmentWidth, ExecGroup, Cap
    >::type;
};

template<class Recipe, class InTile, class OutTile, class TileSubj,
         class SwizzleAtom, class EmitEventTag, class ExecGroup, class Cap>
struct resolve_impl<ScientificSwizzleTilePattern<
    Recipe, InTile, OutTile, TileSubj, SwizzleAtom, EmitEventTag, ExecGroup>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level3::scientific::SwizzleTileImpl<
        Recipe, InTile, OutTile, TileSubj, SwizzleAtom, EmitEventTag, ExecGroup, Cap
    >::type;
};

} // namespace axp::level3::registry
#endif // AXP_LIBRARY_BUILD
