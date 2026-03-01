#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level0/ownership.hpp"
#include "../level0/memory.hpp"
#include "../level2/row.hpp"
#include "detail/compose.hpp"
#include "detail/reg_pressure.hpp"
#include "registry.hpp"

namespace axp::level3::softmax {

namespace detail {
struct softmax_in_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.softmax.in_frag"); };
struct softmax_out_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.softmax.out_frag"); };
using softmax_in_frag_subj = iro::contract::subject::indexed<softmax_in_frag_tag, 0>;
using softmax_out_frag_subj = iro::contract::subject::indexed<softmax_out_frag_tag, 0>;
template<class Obligation, int I>
using in_port_t = axp::level3::detail::in_port_t<Obligation, I>;
template<class Obligation, int I>
using out_port_t = axp::level3::detail::out_port_t<Obligation, I>;
using axp::level3::detail::reg_pressure_obligation;
} // namespace detail

// Warp-level row softmax tile (shared in/out).
template<class Recipe, int ElementsPerThread, class InSubj, class OutSubj, class CapT>
struct SoftmaxRowTileImpl {
    static_assert(ElementsPerThread > 0, "SoftmaxRowTile: ElementsPerThread must be positive");
    using ExecGroup = iro::exec::warp;
    static constexpr int kRow = 32 * ElementsPerThread;

    using Frag = iro::contract::FragmentDesc<
        iro::contract::Shape<ElementsPerThread>,
        typename Recipe::in,
        iro::dist::striped<32>,
        ElementsPerThread
    >;
    static constexpr int kBaseRegs = 12;
    using RegPressure = detail::reg_pressure_obligation<kBaseRegs, Frag>;

    using InTile = iro::contract::Tile<
        iro::contract::Shape<kRow>,
        typename Recipe::in,
        iro::contract::layout::Contiguous,
        iro::contract::space::shared,
        iro::contract::Align<16>
    >;

    using OutTile = iro::contract::Tile<
        iro::contract::Shape<kRow>,
        typename Recipe::out,
        iro::contract::layout::Contiguous,
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
        Recipe, InTile, Frag, InSubj, detail::softmax_in_frag_subj, ExecGroup, iro::token::lifetime::block
    >;

    using Softmax = axp::level2::WarpSoftmax<
        Recipe, Frag, detail::softmax_in_frag_subj, detail::softmax_out_frag_subj, ExecGroup, CapT
    >;

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
        detail::softmax_out_frag_subj,
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

    using obligations = iro::util::type_list<RegPressure, TileIn, Load, Softmax, Store, Fence, TileOut>;
    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<TileIn, 0>, detail::in_port_t<Load, 0>>,
        iro::compose::Edge<detail::out_port_t<Load, 0>, detail::in_port_t<Softmax, 0>>,
        iro::compose::Edge<detail::out_port_t<Softmax, 0>, detail::in_port_t<Store, 0>>,
        iro::compose::Edge<detail::out_port_t<Store, 0>, detail::in_port_t<Fence, 0>>,
        iro::compose::Edge<detail::out_port_t<Fence, 0>, detail::in_port_t<TileOut, 0>>
    >;

    using type = axp::level3::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

} // namespace axp::level3::softmax

namespace axp::level3 {

template<class Recipe, int ElementsPerThread, class InSubj, class OutSubj>
struct SoftmaxRowTileConfig {
    using recipe = Recipe;
    static constexpr int elements_per_thread = ElementsPerThread;
    using in_subj = InSubj;
    using out_subj = OutSubj;
};

template<class Config, class CapT = axp::target_cap>
using SoftmaxRowTile = registry::Select<registry::SoftmaxRowTilePattern<
    typename Config::recipe,
    Config::elements_per_thread,
    typename Config::in_subj,
    typename Config::out_subj>, CapT>;

} // namespace axp::level3

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level3::registry {

template<class Recipe, int ElementsPerThread, class InSubj, class OutSubj, class Cap>
struct resolve_impl<SoftmaxRowTilePattern<Recipe, ElementsPerThread, InSubj, OutSubj>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level3::softmax::SoftmaxRowTileImpl<
        Recipe, ElementsPerThread, InSubj, OutSubj, Cap
    >::type;
};

} // namespace axp::level3::registry
#endif // AXP_LIBRARY_BUILD
