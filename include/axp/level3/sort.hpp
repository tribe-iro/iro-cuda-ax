#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level2/passthrough.hpp"
#include "../level2/sort.hpp"
#include "detail/compose.hpp"
#include "detail/reg_pressure.hpp"
#include "registry.hpp"

namespace axp::level3::sort {

namespace detail {
struct merge_rev_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.sort.merge_rev"); };
using merge_rev_subj = iro::contract::subject::indexed<merge_rev_tag, 0>;
template<class Obligation, int I>
using in_port_t = axp::level3::detail::in_port_t<Obligation, I>;
template<class Obligation, int I>
using out_port_t = axp::level3::detail::out_port_t<Obligation, I>;
using axp::level3::detail::reg_pressure_const;
using axp::level3::detail::reg_pressure_obligation;

template<int N>
struct reverse_second_half {
    static_assert(N > 0, "reverse_second_half requires positive size");
    static constexpr auto id = iro::util::mix_u64(
        iro::util::fnv1a_64_cstr("axp.level3.sort.reverse_second_half"),
        static_cast<iro::util::u64>(N));
    static constexpr int map(int i, int) {
        const int half = N / 2;
        return (i < half) ? i : (N - 1 - (i - half));
    }
};
} // namespace detail

// Sort tile (fragment payload, lane-local).
template<class Recipe, int TileElems, class InSubj, class OutSubj, class CapT>
struct SortTileImpl {
    static_assert(TileElems > 1, "SortTile: TileElems must be > 1");
    static_assert((TileElems & (TileElems - 1)) == 0, "SortTile: TileElems must be power of two");
    static_assert(std::is_same_v<typename Recipe::in, typename Recipe::out>,
                  "SortTile: Recipe::in must match Recipe::out");
    using ExecGroup = iro::exec::warp;

    using Frag = iro::contract::FragmentDesc<
        iro::contract::Shape<TileElems>,
        typename Recipe::in,
        iro::dist::reg_owned,
        TileElems
    >;
    static constexpr int kBaseRegs = 8;
    using RegPressure = detail::reg_pressure_obligation<kBaseRegs, Frag>;

    using Sort = axp::level2::BitonicSort<
        Recipe, Frag, InSubj, OutSubj, ExecGroup, CapT
    >;

    using obligations = iro::util::type_list<RegPressure, Sort>;
    using edges = iro::util::type_list<>;
    using type = axp::level3::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

// Merge tile (fragment payload, assumes two ascending halves).
template<class Recipe, int TileElems, class InSubj, class OutSubj, class CapT>
struct MergeTileImpl {
    static_assert(TileElems > 1, "MergeTile: TileElems must be > 1");
    static_assert((TileElems & (TileElems - 1)) == 0, "MergeTile: TileElems must be power of two");
    static_assert((TileElems % 2) == 0, "MergeTile: TileElems must be even");
    static_assert(std::is_same_v<typename Recipe::in, typename Recipe::out>,
                  "MergeTile: Recipe::in must match Recipe::out");
    using ExecGroup = iro::exec::warp;

    using Frag = iro::contract::FragmentDesc<
        iro::contract::Shape<TileElems>,
        typename Recipe::in,
        iro::dist::reg_owned,
        TileElems
    >;
    static constexpr int kBaseRegs = 10;
    using RegPressure = detail::reg_pressure_obligation<kBaseRegs, Frag>;

    struct Reverse : axp::level2::low::FragmentPermute<
        Recipe, Frag, InSubj, detail::merge_rev_subj, ExecGroup, detail::reverse_second_half<TileElems>
    > {};

    using Merge = axp::level2::BitonicMerge<
        Recipe, Frag, detail::merge_rev_subj, OutSubj, ExecGroup, CapT
    >;

    using obligations = iro::util::type_list<RegPressure, Reverse, Merge>;
    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<Reverse, 0>, detail::in_port_t<Merge, 0>>
    >;
    using type = axp::level3::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

// Warp-wide merge tile (scalar per-lane, two sorted halves).
template<class Recipe, class InSubj, class OutSubj, class CapT>
struct MergeWarpTileImpl {
    static_assert(std::is_same_v<typename Recipe::in, typename Recipe::out>,
                  "MergeWarpTile: Recipe::in must match Recipe::out");
    using ExecGroup = iro::exec::warp;

    using Payload = iro::contract::ScalarDesc<typename Recipe::in, iro::dist::lane>;
    using RegPressure = detail::reg_pressure_const<8>;

    using Reverse = axp::level2::low::WarpReverseSecondHalf<
        Recipe, Payload, InSubj, detail::merge_rev_subj, ExecGroup
    >;

    using Merge = axp::level2::BitonicMergeCross<
        Recipe, Payload, detail::merge_rev_subj, OutSubj, ExecGroup, CapT
    >;

    using obligations = iro::util::type_list<RegPressure, Reverse, Merge>;
    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<Reverse, 0>, detail::in_port_t<Merge, 0>>
    >;
    using type = axp::level3::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

} // namespace axp::level3::sort

namespace axp::level3 {

template<class Recipe, int TileElems, class InSubj, class OutSubj>
struct SortTileConfig {
    using recipe = Recipe;
    static constexpr int tile_elems = TileElems;
    using in_subj = InSubj;
    using out_subj = OutSubj;
};

template<class Recipe, int TileElems, class InSubj, class OutSubj>
struct MergeTileConfig {
    using recipe = Recipe;
    static constexpr int tile_elems = TileElems;
    using in_subj = InSubj;
    using out_subj = OutSubj;
};

template<class Recipe, class InSubj, class OutSubj>
struct MergeWarpTileConfig {
    using recipe = Recipe;
    using in_subj = InSubj;
    using out_subj = OutSubj;
};

template<class Config, class CapT = axp::target_cap>
using SortTile = registry::Select<registry::SortTilePattern<
    typename Config::recipe,
    Config::tile_elems,
    typename Config::in_subj,
    typename Config::out_subj>, CapT>;

template<class Config, class CapT = axp::target_cap>
using MergeTile = registry::Select<registry::MergeTilePattern<
    typename Config::recipe,
    Config::tile_elems,
    typename Config::in_subj,
    typename Config::out_subj>, CapT>;

template<class Config, class CapT = axp::target_cap>
using MergeWarpTile = registry::Select<registry::MergeWarpTilePattern<
    typename Config::recipe,
    typename Config::in_subj,
    typename Config::out_subj>, CapT>;

} // namespace axp::level3

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level3::registry {

template<class Recipe, int TileElems, class InSubj, class OutSubj, class Cap>
struct resolve_impl<SortTilePattern<Recipe, TileElems, InSubj, OutSubj>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level3::sort::SortTileImpl<
        Recipe, TileElems, InSubj, OutSubj, Cap
    >::type;
};

template<class Recipe, int TileElems, class InSubj, class OutSubj, class Cap>
struct resolve_impl<MergeTilePattern<Recipe, TileElems, InSubj, OutSubj>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level3::sort::MergeTileImpl<
        Recipe, TileElems, InSubj, OutSubj, Cap
    >::type;
};

template<class Recipe, class InSubj, class OutSubj, class Cap>
struct resolve_impl<MergeWarpTilePattern<Recipe, InSubj, OutSubj>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level3::sort::MergeWarpTileImpl<
        Recipe, InSubj, OutSubj, Cap
    >::type;
};

} // namespace axp::level3::registry
#endif // AXP_LIBRARY_BUILD
