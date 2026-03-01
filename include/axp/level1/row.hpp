#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level0/fragment.hpp"
#include "../level0/compute.hpp"
#include "../protocol/reduction/contracts.hpp"
#include "reduction.hpp"
#include "registry.hpp"
#include "detail/compose.hpp"

namespace axp::level1 {

namespace detail {

template<class Recipe, class Frag>
struct row_vec_legal : std::bool_constant<
    (Frag::count % 2 == 0) &&
    (std::is_same_v<typename Frag::elem, iro::elem::f16> ||
     std::is_same_v<typename Frag::elem, iro::elem::bf16>) &&
    std::is_same_v<typename Recipe::acc, typename Frag::elem>
> {};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup,
         class OpTag, template<class, class, class, class, class, class, class, class> class Op,
         class CapT = axp::target_cap>
struct row_reduce_impl {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "RowReduce: ExecGroup must be warp");
    using AccRecipe = iro::recipe::Accumulate<Recipe>;
    using ScalarAcc = iro::contract::ScalarDesc<typename AccRecipe::in, typename Frag::dist>;

    using Reduce = axp::level0::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, InSubj, OutSubj, ExecGroup, OpTag
    >;
    using Warp = axp::level1::detail::warp_reduce_impl<AccRecipe, ScalarAcc, OutSubj, ExecGroup, Op, CapT>;

    using obligations = iro::util::concat_t<
        iro::util::type_list<Reduce>,
        typename Warp::obligations
    >;

    using edges = iro::util::concat_t<
        typename Warp::edges,
        iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<Reduce, 0>, detail::in_port_t<typename Warp::S16, 0>>
        >
    >;

    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup,
         class OpTag, template<class, class, class, class, class, class, class, class> class Op,
         class CapT = axp::target_cap>
struct row_reduce_vec_impl {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "RowReduceVec: ExecGroup must be warp");
    using AccRecipe = iro::recipe::Accumulate<Recipe>;
    using ScalarAcc = iro::contract::ScalarDesc<typename AccRecipe::in, typename Frag::dist>;

    using Reduce = axp::level0::FragmentReduceAccVec<
        Recipe, Frag, ScalarAcc, InSubj, OutSubj, ExecGroup, OpTag
    >;
    using Warp = axp::level1::detail::warp_reduce_impl<AccRecipe, ScalarAcc, OutSubj, ExecGroup, Op, CapT>;

    using obligations = iro::util::concat_t<
        iro::util::type_list<Reduce>,
        typename Warp::obligations
    >;

    using edges = iro::util::concat_t<
        typename Warp::edges,
        iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<Reduce, 0>, detail::in_port_t<typename Warp::S16, 0>>
        >
    >;

    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

} // namespace detail

// RowSum: reduce fragment elements -> scalar and then warp-reduce across lanes
template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
using RowSum = registry::Select<registry::RowSumPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, CapT>;

// RowSumVec: explicit vectorized row reduction (f16/bf16, even fragment count)
template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
using RowSumVec = registry::Select<registry::RowSumVecPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, CapT>;

// RowMax: reduce fragment elements -> scalar and then warp-reduce across lanes
template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
using RowMax = registry::Select<registry::RowMaxPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, CapT>;

// RowMaxVec: explicit vectorized row reduction (f16/bf16, even fragment count)
template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
using RowMaxVec = registry::Select<registry::RowMaxVecPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, CapT>;

} // namespace axp::level1

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level1::registry {

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<RowSumPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, Cap> {
    static constexpr bool supported = true;
    using base = typename axp::level1::detail::row_reduce_impl<
        Recipe, Frag, InSubj, OutSubj, ExecGroup,
        axp::protocol::reduction::op_add, axp::level0::Add, Cap
    >::type;
    using type = base;
};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<
    RowSumVecPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>,
    Cap,
    std::enable_if_t<detail::row_vec_legal<Recipe, Frag>::value>> {
    static constexpr bool supported = true;
    using type = typename axp::level1::detail::row_reduce_vec_impl<
        Recipe, Frag, InSubj, OutSubj, ExecGroup,
        axp::protocol::reduction::op_add, axp::level0::Add, Cap
    >::type;
};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<RowMaxPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, Cap> {
    static constexpr bool supported = true;
    using base = typename axp::level1::detail::row_reduce_impl<
        Recipe, Frag, InSubj, OutSubj, ExecGroup,
        axp::protocol::reduction::op_max, axp::level0::Max, Cap
    >::type;
    using type = base;
};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<
    RowMaxVecPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>,
    Cap,
    std::enable_if_t<detail::row_vec_legal<Recipe, Frag>::value>> {
    static constexpr bool supported = true;
    using type = typename axp::level1::detail::row_reduce_vec_impl<
        Recipe, Frag, InSubj, OutSubj, ExecGroup,
        axp::protocol::reduction::op_max, axp::level0::Max, Cap
    >::type;
};

} // namespace axp::level1::registry
#endif // AXP_LIBRARY_BUILD
