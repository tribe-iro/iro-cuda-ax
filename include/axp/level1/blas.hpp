#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level0/compute.hpp"
#include "../level0/memory.hpp"
#include "../level0/convert.hpp"
#include "../protocol/ownership/dist_tags.hpp"
#include "reduction.hpp"
#include "detail/compose.hpp"

namespace axp::level1 {

// Scale: y = a * x (fragment-level)
template<class Recipe, class Frag, class ASubj, class XSubj, class OutSubj, class ExecGroup,
         class CapT = axp::target_cap>
struct ScaleImpl {
    using Mul = axp::level0::Mul<Recipe, Frag, ASubj, XSubj, OutSubj, ExecGroup>;
    using obligations = iro::util::type_list<Mul>;
    using edges = iro::util::type_list<>;
    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Frag, class ASubj, class XSubj, class OutSubj, class ExecGroup,
         class CapT = axp::target_cap>
using Scale = typename ScaleImpl<Recipe, Frag, ASubj, XSubj, OutSubj, ExecGroup, CapT>::type;

// Axpy: y = a * x + y (fragment-level)
template<class Recipe, class Frag, class ASubj, class XSubj, class YSubj, class OutSubj, class ExecGroup,
         class CapT = axp::target_cap>
struct AxpyImpl {
    using Fma = axp::level0::Fma<Recipe, Frag, ASubj, XSubj, YSubj, OutSubj, ExecGroup>;
    using obligations = iro::util::type_list<Fma>;
    using edges = iro::util::type_list<>;
    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Frag, class ASubj, class XSubj, class YSubj, class OutSubj, class ExecGroup,
         class CapT = axp::target_cap>
using Axpy = typename AxpyImpl<Recipe, Frag, ASubj, XSubj, YSubj, OutSubj, ExecGroup, CapT>::type;

// Dot: sum(x*y) (fragment-level)
template<class Recipe, class Frag, class XSubj, class YSubj, class AccSubj, class ExecGroup,
         class CapT = axp::target_cap>
struct DotImpl {
    using Mul = axp::level0::Mul<Recipe, Frag, XSubj, YSubj, AccSubj, ExecGroup>;
    using Warp = detail::warp_reduce_impl<Recipe, Frag, AccSubj, ExecGroup, axp::level0::Add, CapT>;

    using obligations = iro::util::concat_t<
        iro::util::type_list<Mul>,
        typename Warp::obligations
    >;

    using edges = iro::util::concat_t<
        typename Warp::edges,
        iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<Mul, 0>, detail::in_port_t<typename Warp::S16, 0>>
        >
    >;

    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Frag, class XSubj, class YSubj, class AccSubj, class ExecGroup,
         class CapT = axp::target_cap>
using Dot = typename DotImpl<Recipe, Frag, XSubj, YSubj, AccSubj, ExecGroup, CapT>::type;

// Nrm2: sqrt(sum(x*x)) (fragment-level)
template<class Recipe, class Frag, class XSubj, class SumSubj, class RsqrtSubj, class OutSubj, class ExecGroup,
         class CapT = axp::target_cap>
struct Nrm2Impl {
    using Mul0 = axp::level0::Mul<Recipe, Frag, XSubj, XSubj, SumSubj, ExecGroup>;
    using Warp = detail::warp_reduce_impl<Recipe, Frag, SumSubj, ExecGroup, axp::level0::Add, CapT>;
    using Rsqrt = axp::level0::Rsqrt<Recipe, Frag, SumSubj, RsqrtSubj, ExecGroup>;
    using Mul1 = axp::level0::Mul<Recipe, Frag, SumSubj, RsqrtSubj, OutSubj, ExecGroup>;

    using obligations = iro::util::concat_t<
        iro::util::type_list<Mul0>,
        iro::util::concat_t<
            typename Warp::obligations,
            iro::util::type_list<Rsqrt, Mul1>
        >
    >;

    using edges = iro::util::concat_t<
        typename Warp::edges,
        iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<Mul0, 0>, detail::in_port_t<typename Warp::S16, 0>>,
            iro::compose::Edge<detail::out_port_t<typename Warp::O1, 0>, detail::in_port_t<Rsqrt, 0>>,
            iro::compose::Edge<detail::out_port_t<typename Warp::O1, 0>, detail::in_port_t<Mul1, 0>>,
            iro::compose::Edge<detail::out_port_t<Rsqrt, 0>, detail::in_port_t<Mul1, 1>>
        >
    >;

    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Frag, class XSubj, class SumSubj, class RsqrtSubj, class OutSubj, class ExecGroup,
         class CapT = axp::target_cap>
using Nrm2 = typename Nrm2Impl<Recipe, Frag, XSubj, SumSubj, RsqrtSubj, OutSubj, ExecGroup, CapT>::type;

// Asum: sum(abs(x)) (fragment-level)
template<class Recipe, class Frag, class XSubj, class AccSubj, class ExecGroup, class CapT = axp::target_cap>
struct AsumImpl {
    using Abs = axp::level0::Abs<Recipe, Frag, XSubj, AccSubj, ExecGroup>;
    using Warp = detail::warp_reduce_impl<Recipe, Frag, AccSubj, ExecGroup, axp::level0::Add, CapT>;

    using obligations = iro::util::concat_t<
        iro::util::type_list<Abs>,
        typename Warp::obligations
    >;

    using edges = iro::util::concat_t<
        typename Warp::edges,
        iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<Abs, 0>, detail::in_port_t<typename Warp::S16, 0>>
        >
    >;

    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Frag, class XSubj, class AccSubj, class ExecGroup, class CapT = axp::target_cap>
using Asum = typename AsumImpl<Recipe, Frag, XSubj, AccSubj, ExecGroup, CapT>::type;

// Tile copy: global -> reg -> global
// Requires explicit reg dist on intermediate tile.
template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CapT = axp::target_cap>
struct CopyImpl {
    using RegTile = iro::contract::Tile<
        typename InTile::shape,
        typename InTile::elem,
        typename InTile::layout,
        iro::contract::space::reg,
        typename InTile::align
    >;

    using Ld = axp::level0::LdGlobal<
        Recipe,
        InTile,
        RegTile,
        InSubj,
        OutSubj,
        ExecGroup,
        axp::cache::ca,
        iro::contract::no_dist,
        axp::dist::reg_tile
    >;

    using St = axp::level0::StGlobal<
        Recipe,
        RegTile,
        OutTile,
        OutSubj,
        OutSubj,
        ExecGroup,
        axp::cache::wb,
        axp::dist::reg_tile,
        iro::contract::no_dist
    >;

    using obligations = iro::util::type_list<Ld, St>;
    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<Ld, 0>, detail::in_port_t<St, 0>>
    >;
    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CapT = axp::target_cap>
using Copy = typename CopyImpl<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, CapT>::type;

// Cast (tile-level): composition of L0 CastTile

template<class RecipeIn, class RecipeOut, class InTile, class OutTile, class InSubj, class OutSubj,
         class ExecGroup, int VecBytes, class CapT = axp::target_cap>
struct CastImpl {
    using InDist = std::conditional_t<
        std::is_same_v<typename InTile::space, iro::contract::space::reg>,
        axp::dist::reg_tile,
        std::conditional_t<
            std::is_same_v<typename InTile::space, iro::contract::space::tmem>,
            axp::dist::tmem_tile,
            iro::contract::no_dist
        >
    >;
    using OutDist = std::conditional_t<
        std::is_same_v<typename OutTile::space, iro::contract::space::reg>,
        axp::dist::reg_tile,
        std::conditional_t<
            std::is_same_v<typename OutTile::space, iro::contract::space::tmem>,
            axp::dist::tmem_tile,
            iro::contract::no_dist
        >
    >;
    using Cast = axp::level0::CastTile<
        RecipeIn, RecipeOut, InTile, OutTile, InSubj, OutSubj, ExecGroup, VecBytes, InDist, OutDist
    >;

    using obligations = iro::util::type_list<Cast>;
    using edges = iro::util::type_list<>;
    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class RecipeIn, class RecipeOut, class InTile, class OutTile, class InSubj, class OutSubj,
         class ExecGroup, int VecBytes, class CapT = axp::target_cap>
using Cast = typename CastImpl<RecipeIn, RecipeOut, InTile, OutTile, InSubj, OutSubj, ExecGroup, VecBytes, CapT>::type;

} // namespace axp::level1
