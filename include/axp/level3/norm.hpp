#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level0/ownership.hpp"
#include "../level0/memory.hpp"
#include "../level2/norm.hpp"
#include "detail/compose.hpp"
#include "detail/reg_pressure.hpp"
#include "registry.hpp"

namespace axp::level3::norm {

namespace detail {
struct norm_in_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.norm.in_frag"); };
struct norm_out_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.norm.out_frag"); };
using norm_in_frag_subj = iro::contract::subject::indexed<norm_in_frag_tag, 0>;
using norm_out_frag_subj = iro::contract::subject::indexed<norm_out_frag_tag, 0>;
template<class Obligation, int I>
using in_port_t = axp::level3::detail::in_port_t<Obligation, I>;
template<class Obligation, int I>
using out_port_t = axp::level3::detail::out_port_t<Obligation, I>;
using axp::level3::detail::reg_pressure_obligation;
} // namespace detail

// LayerNorm tile (warp-level, 16x16 WMMA fragment). Outputs shared tile.
template<
    class Recipe,
    int TileRows, int TileCols,
    class InSubj, class OutSubj,
    class GammaSubj, class BetaSubj,
    class EpsSubj,
    class CapT>
struct LayerNormTileImpl {
    static_assert(TileRows == 16 && TileCols == 16,
                  "LayerNormTile: WMMA 16x16 only (TileRows=16, TileCols=16)");
    static_assert(std::is_same_v<typename Recipe::in, typename Recipe::out>,
                  "LayerNormTile: Recipe::in must match Recipe::out");
    static_assert(std::is_same_v<typename Recipe::acc, iro::elem::f32>,
                  "LayerNormTile: Recipe::acc must be f32");
    using ExecGroup = iro::exec::warp;

    using Frag = iro::contract::FragmentDesc<
        iro::contract::Shape<TileRows, TileCols>,
        typename Recipe::in,
        iro::dist::warp_row_major,
        TileRows * TileCols
    >;
    static constexpr int kBaseRegs = 16;
    using RegPressure = detail::reg_pressure_obligation<kBaseRegs, Frag>;

    using GammaFrag = iro::contract::FragmentDesc<
        iro::contract::Shape<TileRows, TileCols>,
        typename Recipe::in,
        iro::dist::warp_row_major,
        TileRows * TileCols
    >;

    using BetaFrag = iro::contract::FragmentDesc<
        iro::contract::Shape<TileRows, TileCols>,
        typename Recipe::in,
        iro::dist::warp_row_major,
        TileRows * TileCols
    >;

    using EpsPayload = iro::contract::ScalarDesc<typename Recipe::acc, iro::dist::replicated>;

    using InTile = iro::contract::Tile<
        iro::contract::Shape<TileRows, TileCols>,
        typename Recipe::in,
        iro::contract::layout::RowMajor<TileCols>,
        iro::contract::space::shared,
        iro::contract::Align<16>
    >;

    using OutTile = iro::contract::Tile<
        iro::contract::Shape<TileRows, TileCols>,
        typename Recipe::out,
        iro::contract::layout::RowMajor<TileCols>,
        iro::contract::space::shared,
        iro::contract::Align<16>
    >;

    using TileIn = axp::level0::TileBoundaryIn<
        Recipe, InTile, InSubj, ExecGroup, iro::token::lifetime::block
    >;
    using TileOut = axp::level0::TileBoundaryOut<
        Recipe, OutTile, OutSubj, iro::exec::block, iro::token::lifetime::block
    >;

    using Load = axp::level0::SharedTileToFragment<
        Recipe, InTile, Frag, InSubj, detail::norm_in_frag_subj, ExecGroup, iro::token::lifetime::block
    >;

    using Norm = axp::level2::registry::Select<axp::level2::registry::LayerNormFragPattern<
        Recipe, Frag, GammaFrag, BetaFrag, EpsPayload,
        detail::norm_in_frag_subj, GammaSubj, BetaSubj, EpsSubj,
        detail::norm_out_frag_subj, ExecGroup>, CapT>;

    using StoreRecipe = iro::recipe::Precision<
        typename Recipe::out,
        typename Recipe::out,
        typename Recipe::out,
        Recipe::vec_bytes,
        typename Recipe::math
    >;

    using Store = axp::level0::FragmentToSharedTile<
        StoreRecipe,
        Frag,
        OutTile,
        detail::norm_out_frag_subj,
        OutSubj,
        ExecGroup,
        iro::token::lifetime::warp
    >;

    using Fence = axp::level0::TileFence<
        Recipe,
        OutTile,
        OutSubj,
        iro::exec::block
    >;

    using obligations = iro::util::type_list<RegPressure, TileIn, Load, Norm, Store, Fence, TileOut>;

    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<TileIn, 0>, detail::in_port_t<Load, 0>>,
        iro::compose::Edge<detail::out_port_t<Load, 0>, detail::in_port_t<Norm, 0>>,
        iro::compose::Edge<detail::out_port_t<Norm, 0>, detail::in_port_t<Store, 0>>,
        iro::compose::Edge<detail::out_port_t<Store, 0>, detail::in_port_t<Fence, 0>>,
        iro::compose::Edge<detail::out_port_t<Fence, 0>, detail::in_port_t<TileOut, 0>>
    >;

    using type = axp::level3::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

// RMSNorm tile (warp-level, 16x16 WMMA fragment). Outputs shared tile.
template<
    class Recipe,
    int TileRows, int TileCols,
    class InSubj, class OutSubj,
    class WeightSubj,
    class EpsSubj,
    class CapT>
struct RMSNormTileImpl {
    static_assert(TileRows == 16 && TileCols == 16,
                  "RMSNormTile: WMMA 16x16 only (TileRows=16, TileCols=16)");
    static_assert(std::is_same_v<typename Recipe::in, typename Recipe::out>,
                  "RMSNormTile: Recipe::in must match Recipe::out");
    static_assert(std::is_same_v<typename Recipe::acc, iro::elem::f32>,
                  "RMSNormTile: Recipe::acc must be f32");
    using ExecGroup = iro::exec::warp;

    using Frag = iro::contract::FragmentDesc<
        iro::contract::Shape<TileRows, TileCols>,
        typename Recipe::in,
        iro::dist::warp_row_major,
        TileRows * TileCols
    >;

    using WeightFrag = iro::contract::FragmentDesc<
        iro::contract::Shape<TileRows, TileCols>,
        typename Recipe::in,
        iro::dist::warp_row_major,
        TileRows * TileCols
    >;
    static constexpr int kBaseRegs = 16;
    using RegPressure = detail::reg_pressure_obligation<kBaseRegs, Frag, WeightFrag>;

    using EpsPayload = iro::contract::ScalarDesc<typename Recipe::acc, iro::dist::replicated>;

    using InTile = iro::contract::Tile<
        iro::contract::Shape<TileRows, TileCols>,
        typename Recipe::in,
        iro::contract::layout::RowMajor<TileCols>,
        iro::contract::space::shared,
        iro::contract::Align<16>
    >;

    using OutTile = iro::contract::Tile<
        iro::contract::Shape<TileRows, TileCols>,
        typename Recipe::out,
        iro::contract::layout::RowMajor<TileCols>,
        iro::contract::space::shared,
        iro::contract::Align<16>
    >;

    using TileIn = axp::level0::TileBoundaryIn<
        Recipe, InTile, InSubj, ExecGroup, iro::token::lifetime::block
    >;
    using TileOut = axp::level0::TileBoundaryOut<
        Recipe, OutTile, OutSubj, iro::exec::block, iro::token::lifetime::block
    >;

    using Load = axp::level0::SharedTileToFragment<
        Recipe, InTile, Frag, InSubj, detail::norm_in_frag_subj, ExecGroup, iro::token::lifetime::block
    >;

    using Norm = axp::level2::registry::Select<axp::level2::registry::RMSNormFragPattern<
        Recipe, Frag, WeightFrag, EpsPayload,
        detail::norm_in_frag_subj, WeightSubj, EpsSubj,
        detail::norm_out_frag_subj, ExecGroup>, CapT>;

    using StoreRecipe = iro::recipe::Precision<
        typename Recipe::out,
        typename Recipe::out,
        typename Recipe::out,
        Recipe::vec_bytes,
        typename Recipe::math
    >;

    using Store = axp::level0::FragmentToSharedTile<
        StoreRecipe,
        Frag,
        OutTile,
        detail::norm_out_frag_subj,
        OutSubj,
        ExecGroup,
        iro::token::lifetime::warp
    >;

    using Fence = axp::level0::TileFence<
        Recipe,
        OutTile,
        OutSubj,
        iro::exec::block
    >;

    using obligations = iro::util::type_list<RegPressure, TileIn, Load, Norm, Store, Fence, TileOut>;

    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<TileIn, 0>, detail::in_port_t<Load, 0>>,
        iro::compose::Edge<detail::out_port_t<Load, 0>, detail::in_port_t<Norm, 0>>,
        iro::compose::Edge<detail::out_port_t<Norm, 0>, detail::in_port_t<Store, 0>>,
        iro::compose::Edge<detail::out_port_t<Store, 0>, detail::in_port_t<Fence, 0>>,
        iro::compose::Edge<detail::out_port_t<Fence, 0>, detail::in_port_t<TileOut, 0>>
    >;

    using type = axp::level3::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

} // namespace axp::level3::norm

namespace axp::level3 {

template<class Recipe, int TileRows, int TileCols,
         class InSubj, class OutSubj,
         class GammaSubj, class BetaSubj, class EpsSubj>
struct LayerNormTileConfig {
    using recipe = Recipe;
    static constexpr int tile_rows = TileRows;
    static constexpr int tile_cols = TileCols;
    using in_subj = InSubj;
    using out_subj = OutSubj;
    using gamma_subj = GammaSubj;
    using beta_subj = BetaSubj;
    using eps_subj = EpsSubj;
};

template<class Recipe, int TileRows, int TileCols,
         class InSubj, class OutSubj,
         class WeightSubj, class EpsSubj>
struct RMSNormTileConfig {
    using recipe = Recipe;
    static constexpr int tile_rows = TileRows;
    static constexpr int tile_cols = TileCols;
    using in_subj = InSubj;
    using out_subj = OutSubj;
    using weight_subj = WeightSubj;
    using eps_subj = EpsSubj;
};

template<class Config, class CapT = axp::target_cap>
using LayerNormTile = registry::Select<registry::LayerNormTilePattern<
    typename Config::recipe,
    Config::tile_rows, Config::tile_cols,
    typename Config::in_subj, typename Config::out_subj,
    typename Config::gamma_subj, typename Config::beta_subj, typename Config::eps_subj>, CapT>;

template<class Config, class CapT = axp::target_cap>
using RMSNormTile = registry::Select<registry::RMSNormTilePattern<
    typename Config::recipe,
    Config::tile_rows, Config::tile_cols,
    typename Config::in_subj, typename Config::out_subj,
    typename Config::weight_subj, typename Config::eps_subj>, CapT>;

} // namespace axp::level3

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level3::registry {

template<class Recipe, int TileRows, int TileCols,
         class InSubj, class OutSubj,
         class GammaSubj, class BetaSubj, class EpsSubj, class Cap>
struct resolve_impl<LayerNormTilePattern<Recipe, TileRows, TileCols,
                                        InSubj, OutSubj, GammaSubj, BetaSubj, EpsSubj>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level3::norm::LayerNormTileImpl<
        Recipe, TileRows, TileCols, InSubj, OutSubj, GammaSubj, BetaSubj, EpsSubj, Cap
    >::type;
};

template<class Recipe, int TileRows, int TileCols,
         class InSubj, class OutSubj,
         class WeightSubj, class EpsSubj, class Cap>
struct resolve_impl<RMSNormTilePattern<Recipe, TileRows, TileCols,
                                      InSubj, OutSubj, WeightSubj, EpsSubj>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level3::norm::RMSNormTileImpl<
        Recipe, TileRows, TileCols, InSubj, OutSubj, WeightSubj, EpsSubj, Cap
    >::type;
};

} // namespace axp::level3::registry
#endif // AXP_LIBRARY_BUILD
