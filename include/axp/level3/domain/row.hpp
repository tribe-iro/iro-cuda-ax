#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../../level2/passthrough.hpp"
#include "../../level2/communication.hpp"
#include "../../level2/reduction.hpp"
#include "registry.hpp"
#include "../detail/compose.hpp"

namespace axp::level3::domain {

namespace detail {

struct softmax_max_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.domain.softmax.max"); };
struct softmax_max_b_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.domain.softmax.max_b"); };
struct softmax_sum_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.domain.softmax.sum"); };
struct softmax_sum_b_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.domain.softmax.sum_b"); };
struct softmax_inv_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.domain.softmax.inv"); };
struct softmax_tmp_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.domain.softmax.tmp"); };
struct softmax_inv_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.domain.softmax.inv_frag"); };
struct softmax_max_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.domain.softmax.max_frag"); };
struct softmax_sum_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.domain.softmax.sum_frag"); };
struct softmax_neg_inf_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.domain.softmax.neg_inf_frag"); };
struct softmax_masked_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.domain.softmax.masked"); };

using softmax_max_subj = iro::contract::subject::indexed<softmax_max_tag, 0>;
using softmax_max_b_subj = iro::contract::subject::indexed<softmax_max_b_tag, 0>;
using softmax_sum_subj = iro::contract::subject::indexed<softmax_sum_tag, 0>;
using softmax_sum_b_subj = iro::contract::subject::indexed<softmax_sum_b_tag, 0>;
using softmax_inv_subj = iro::contract::subject::indexed<softmax_inv_tag, 0>;
using softmax_tmp_subj = iro::contract::subject::indexed<softmax_tmp_tag, 0>;
using softmax_inv_frag_subj = iro::contract::subject::indexed<softmax_inv_frag_tag, 0>;
using softmax_max_frag_subj = iro::contract::subject::indexed<softmax_max_frag_tag, 0>;
using softmax_sum_frag_subj = iro::contract::subject::indexed<softmax_sum_frag_tag, 0>;
using softmax_neg_inf_frag_subj = iro::contract::subject::indexed<softmax_neg_inf_frag_tag, 0>;
using softmax_masked_subj = iro::contract::subject::indexed<softmax_masked_tag, 0>;

template<class Recipe, class Frag>
struct softmax_vec_legal : std::bool_constant<
    (Frag::count % 2 == 0) &&
    (std::is_same_v<typename Frag::elem, iro::elem::f16> ||
     std::is_same_v<typename Frag::elem, iro::elem::bf16>) &&
    std::is_same_v<typename Recipe::acc, typename Frag::elem>
> {};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
struct warp_softmax_impl {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "WarpSoftmax: ExecGroup must be warp");
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::in>,
                  "WarpSoftmax: fragment elem must match Recipe::in");

    using ScalarAcc = iro::contract::ScalarDesc<typename Recipe::acc, typename Frag::dist>;
    using AccRecipe = iro::recipe::Accumulate<Recipe>;

    using ReduceMaxFrag = axp::level2::low::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, InSubj, softmax_max_subj, ExecGroup, axp::level2::proto::reduction::op_max
    >;
    using ReduceMaxWarp = axp::level2::WarpReduce<
        AccRecipe, ScalarAcc, softmax_max_subj, ExecGroup, axp::level2::low::Max, CapT
    >;
    using BroadcastMax = axp::level2::BroadcastLane0<
        AccRecipe, ScalarAcc, softmax_max_subj, softmax_max_b_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using FragMax = axp::level2::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, softmax_max_b_subj, softmax_tmp_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Sub = axp::level2::low::Sub<Recipe, Frag, InSubj, softmax_tmp_subj, softmax_tmp_subj, ExecGroup>;
    using Exp = axp::level2::low::Exp<Recipe, Frag, softmax_tmp_subj, softmax_tmp_subj, ExecGroup>;

    using ReduceSumFrag = axp::level2::low::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, softmax_tmp_subj, softmax_sum_subj, ExecGroup, axp::level2::proto::reduction::op_add
    >;
    using ReduceSumWarp = axp::level2::WarpReduce<
        AccRecipe, ScalarAcc, softmax_sum_subj, ExecGroup, axp::level2::low::Add, CapT
    >;
    using BroadcastSum = axp::level2::BroadcastLane0<
        AccRecipe, ScalarAcc, softmax_sum_subj, softmax_sum_b_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using InvSum = axp::level2::low::Rcp<AccRecipe, ScalarAcc, softmax_sum_b_subj, softmax_inv_subj, ExecGroup>;
    using FragInv = axp::level2::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, softmax_inv_subj, softmax_inv_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Mul = axp::level2::low::Mul<Recipe, Frag, softmax_tmp_subj, softmax_inv_frag_subj, OutSubj, ExecGroup>;

    using obligations = iro::util::type_list<
        ReduceMaxFrag,
        ReduceMaxWarp,
        BroadcastMax,
        FragMax,
        Sub,
        Exp,
        ReduceSumFrag,
        ReduceSumWarp,
        BroadcastSum,
        InvSum,
        FragInv,
        Mul
    >;

    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<ReduceMaxFrag, 0>, detail::in_port_t<ReduceMaxWarp, 0>>,
        iro::compose::Edge<detail::out_port_t<ReduceMaxWarp, 0>, detail::in_port_t<BroadcastMax, 0>>,
        iro::compose::Edge<detail::out_port_t<BroadcastMax, 0>, detail::in_port_t<FragMax, 0>>,
        iro::compose::Edge<detail::out_port_t<FragMax, 0>, detail::in_port_t<Sub, 1>>,
        iro::compose::Edge<detail::out_port_t<Sub, 0>, detail::in_port_t<Exp, 0>>,
        iro::compose::Edge<detail::out_port_t<Exp, 0>, detail::in_port_t<ReduceSumFrag, 0>>,
        iro::compose::Edge<detail::out_port_t<ReduceSumFrag, 0>, detail::in_port_t<ReduceSumWarp, 0>>,
        iro::compose::Edge<detail::out_port_t<ReduceSumWarp, 0>, detail::in_port_t<BroadcastSum, 0>>,
        iro::compose::Edge<detail::out_port_t<BroadcastSum, 0>, detail::in_port_t<InvSum, 0>>,
        iro::compose::Edge<detail::out_port_t<InvSum, 0>, detail::in_port_t<FragInv, 0>>,
        iro::compose::Edge<detail::out_port_t<Exp, 0>, detail::in_port_t<Mul, 0>>,
        iro::compose::Edge<detail::out_port_t<FragInv, 0>, detail::in_port_t<Mul, 1>>
    >;

    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
struct warp_softmax_vec_impl {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "WarpSoftmaxVec: ExecGroup must be warp");
    static_assert(softmax_vec_legal<Recipe, Frag>::value,
                  "WarpSoftmaxVec: requires f16/bf16 with even fragment count and Recipe::acc == Frag::elem");
    static_assert(std::is_same_v<typename Recipe::in, typename Recipe::acc>,
                  "WarpSoftmaxVec: requires Recipe::in == Recipe::acc");
    static_assert(std::is_same_v<typename Recipe::acc, typename Recipe::out>,
                  "WarpSoftmaxVec: requires Recipe::acc == Recipe::out");

    using ScalarAcc = iro::contract::ScalarDesc<typename Recipe::acc, typename Frag::dist>;
    using AccRecipe = iro::recipe::Accumulate<Recipe>;

    using ReduceMaxFrag = axp::level2::low::FragmentReduceAccVec<
        Recipe, Frag, ScalarAcc, InSubj, softmax_max_subj, ExecGroup, axp::level2::proto::reduction::op_max
    >;
    using ReduceMaxWarp = axp::level2::WarpReduce<
        AccRecipe, ScalarAcc, softmax_max_subj, ExecGroup, axp::level2::low::Max, CapT
    >;
    using BroadcastMax = axp::level2::BroadcastLane0<
        AccRecipe, ScalarAcc, softmax_max_subj, softmax_max_b_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using FragMax = axp::level2::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, softmax_max_b_subj, softmax_tmp_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Sub = axp::level2::low::Sub<Recipe, Frag, InSubj, softmax_tmp_subj, softmax_tmp_subj, ExecGroup>;
    using Exp = axp::level2::low::Exp<Recipe, Frag, softmax_tmp_subj, softmax_tmp_subj, ExecGroup>;

    using ReduceSumFrag = axp::level2::low::FragmentReduceAccVec<
        Recipe, Frag, ScalarAcc, softmax_tmp_subj, softmax_sum_subj, ExecGroup, axp::level2::proto::reduction::op_add
    >;
    using ReduceSumWarp = axp::level2::WarpReduce<
        AccRecipe, ScalarAcc, softmax_sum_subj, ExecGroup, axp::level2::low::Add, CapT
    >;
    using BroadcastSum = axp::level2::BroadcastLane0<
        AccRecipe, ScalarAcc, softmax_sum_subj, softmax_sum_b_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using InvSum = axp::level2::low::Rcp<AccRecipe, ScalarAcc, softmax_sum_b_subj, softmax_inv_subj, ExecGroup>;
    using FragInv = axp::level2::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, softmax_inv_subj, softmax_inv_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Mul = axp::level2::low::Mul<Recipe, Frag, softmax_tmp_subj, softmax_inv_frag_subj, OutSubj, ExecGroup>;

    using obligations = iro::util::type_list<
        ReduceMaxFrag,
        ReduceMaxWarp,
        BroadcastMax,
        FragMax,
        Sub,
        Exp,
        ReduceSumFrag,
        ReduceSumWarp,
        BroadcastSum,
        InvSum,
        FragInv,
        Mul
    >;

    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<ReduceMaxFrag, 0>, detail::in_port_t<ReduceMaxWarp, 0>>,
        iro::compose::Edge<detail::out_port_t<ReduceMaxWarp, 0>, detail::in_port_t<BroadcastMax, 0>>,
        iro::compose::Edge<detail::out_port_t<BroadcastMax, 0>, detail::in_port_t<FragMax, 0>>,
        iro::compose::Edge<detail::out_port_t<FragMax, 0>, detail::in_port_t<Sub, 1>>,
        iro::compose::Edge<detail::out_port_t<Sub, 0>, detail::in_port_t<Exp, 0>>,
        iro::compose::Edge<detail::out_port_t<Exp, 0>, detail::in_port_t<ReduceSumFrag, 0>>,
        iro::compose::Edge<detail::out_port_t<ReduceSumFrag, 0>, detail::in_port_t<ReduceSumWarp, 0>>,
        iro::compose::Edge<detail::out_port_t<ReduceSumWarp, 0>, detail::in_port_t<BroadcastSum, 0>>,
        iro::compose::Edge<detail::out_port_t<BroadcastSum, 0>, detail::in_port_t<InvSum, 0>>,
        iro::compose::Edge<detail::out_port_t<InvSum, 0>, detail::in_port_t<FragInv, 0>>,
        iro::compose::Edge<detail::out_port_t<Exp, 0>, detail::in_port_t<Mul, 0>>,
        iro::compose::Edge<detail::out_port_t<FragInv, 0>, detail::in_port_t<Mul, 1>>
    >;

    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Frag, class MaskPayload, class MaskSubj, class NegInfSubj,
         class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
struct warp_softmax_masked_impl {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "WarpSoftmaxMasked: ExecGroup must be warp");
    static_assert(iro::contract::MaskPayload<MaskPayload>, "WarpSoftmaxMasked: MaskPayload required");

    using ScalarAcc = iro::contract::ScalarDesc<typename Recipe::acc, typename Frag::dist>;
    using AccRecipe = iro::recipe::Accumulate<Recipe>;

    using NegInfFrag = axp::level2::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, NegInfSubj, softmax_neg_inf_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Masked = axp::level2::low::Select<
        Recipe, Frag, MaskPayload, InSubj, softmax_neg_inf_frag_subj, MaskSubj, softmax_masked_subj, ExecGroup
    >;

    using ReduceMaxFrag = axp::level2::low::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, softmax_masked_subj, softmax_max_subj, ExecGroup, axp::level2::proto::reduction::op_max
    >;
    using ReduceMaxWarp = axp::level2::WarpReduce<
        AccRecipe, ScalarAcc, softmax_max_subj, ExecGroup, axp::level2::low::Max, CapT
    >;
    using BroadcastMax = axp::level2::BroadcastLane0<
        AccRecipe, ScalarAcc, softmax_max_subj, softmax_max_b_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using FragMax = axp::level2::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, softmax_max_b_subj, softmax_tmp_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Sub = axp::level2::low::Sub<Recipe, Frag, softmax_masked_subj, softmax_tmp_subj, softmax_tmp_subj, ExecGroup>;
    using Exp = axp::level2::low::Exp<Recipe, Frag, softmax_tmp_subj, softmax_tmp_subj, ExecGroup>;

    using ReduceSumFrag = axp::level2::low::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, softmax_tmp_subj, softmax_sum_subj, ExecGroup, axp::level2::proto::reduction::op_add
    >;
    using ReduceSumWarp = axp::level2::WarpReduce<
        AccRecipe, ScalarAcc, softmax_sum_subj, ExecGroup, axp::level2::low::Add, CapT
    >;
    using BroadcastSum = axp::level2::BroadcastLane0<
        AccRecipe, ScalarAcc, softmax_sum_subj, softmax_sum_b_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using InvSum = axp::level2::low::Rcp<AccRecipe, ScalarAcc, softmax_sum_b_subj, softmax_inv_subj, ExecGroup>;
    using FragInv = axp::level2::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, softmax_inv_subj, softmax_inv_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Mul = axp::level2::low::Mul<Recipe, Frag, softmax_tmp_subj, softmax_inv_frag_subj, OutSubj, ExecGroup>;

    using obligations = iro::util::type_list<
        NegInfFrag,
        Masked,
        ReduceMaxFrag,
        ReduceMaxWarp,
        BroadcastMax,
        FragMax,
        Sub,
        Exp,
        ReduceSumFrag,
        ReduceSumWarp,
        BroadcastSum,
        InvSum,
        FragInv,
        Mul
    >;

    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<NegInfFrag, 0>, detail::in_port_t<Masked, 1>>,
        iro::compose::Edge<detail::out_port_t<Masked, 0>, detail::in_port_t<ReduceMaxFrag, 0>>,
        iro::compose::Edge<detail::out_port_t<ReduceMaxFrag, 0>, detail::in_port_t<ReduceMaxWarp, 0>>,
        iro::compose::Edge<detail::out_port_t<ReduceMaxWarp, 0>, detail::in_port_t<BroadcastMax, 0>>,
        iro::compose::Edge<detail::out_port_t<BroadcastMax, 0>, detail::in_port_t<FragMax, 0>>,
        iro::compose::Edge<detail::out_port_t<FragMax, 0>, detail::in_port_t<Sub, 1>>,
        iro::compose::Edge<detail::out_port_t<Masked, 0>, detail::in_port_t<Sub, 0>>,
        iro::compose::Edge<detail::out_port_t<Sub, 0>, detail::in_port_t<Exp, 0>>,
        iro::compose::Edge<detail::out_port_t<Exp, 0>, detail::in_port_t<ReduceSumFrag, 0>>,
        iro::compose::Edge<detail::out_port_t<ReduceSumFrag, 0>, detail::in_port_t<ReduceSumWarp, 0>>,
        iro::compose::Edge<detail::out_port_t<ReduceSumWarp, 0>, detail::in_port_t<BroadcastSum, 0>>,
        iro::compose::Edge<detail::out_port_t<BroadcastSum, 0>, detail::in_port_t<InvSum, 0>>,
        iro::compose::Edge<detail::out_port_t<InvSum, 0>, detail::in_port_t<FragInv, 0>>,
        iro::compose::Edge<detail::out_port_t<Exp, 0>, detail::in_port_t<Mul, 0>>,
        iro::compose::Edge<detail::out_port_t<FragInv, 0>, detail::in_port_t<Mul, 1>>
    >;

    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Frag, class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
struct block_softmax_impl {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "BlockSoftmaxFrag: ExecGroup must be warp");

    using ScalarAcc = iro::contract::ScalarDesc<typename Recipe::acc, typename Frag::dist>;
    using ScalarFrag = iro::contract::FragmentDesc<iro::contract::Shape<1>, typename Recipe::acc, typename Frag::dist, 1>;
    using AccRecipe = iro::recipe::Accumulate<Recipe>;

    using ReduceMaxFrag = axp::level2::low::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, InSubj, softmax_max_subj, ExecGroup, axp::level2::proto::reduction::op_max
    >;
    using ScalarMaxFrag = axp::level2::FragmentBroadcast<
        AccRecipe, ScalarFrag, ScalarAcc, softmax_max_subj, softmax_max_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using BlockMax = axp::level2::BlockReduce<
        AccRecipe, ScalarFrag, SmemTile, softmax_max_frag_subj, SmemSubjMax, ExecGroup, axp::level2::low::Max, CapT
    >;
    using BlockMaxB = axp::level2::BroadcastLane0<
        AccRecipe, ScalarFrag, softmax_max_frag_subj, softmax_max_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using MaxScalar = axp::level2::low::FragmentExtract<
        AccRecipe, ScalarFrag, ScalarAcc, softmax_max_frag_subj, softmax_max_b_subj, ExecGroup, 0
    >;
    using FragMax = axp::level2::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, softmax_max_b_subj, softmax_tmp_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Sub = axp::level2::low::Sub<Recipe, Frag, InSubj, softmax_tmp_subj, softmax_tmp_subj, ExecGroup>;
    using Exp = axp::level2::low::Exp<Recipe, Frag, softmax_tmp_subj, softmax_tmp_subj, ExecGroup>;

    using ReduceSumFrag = axp::level2::low::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, softmax_tmp_subj, softmax_sum_subj, ExecGroup, axp::level2::proto::reduction::op_add
    >;
    using ScalarSumFrag = axp::level2::FragmentBroadcast<
        AccRecipe, ScalarFrag, ScalarAcc, softmax_sum_subj, softmax_sum_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using BlockSum = axp::level2::BlockReduce<
        AccRecipe, ScalarFrag, SmemTile, softmax_sum_frag_subj, SmemSubjSum, ExecGroup, axp::level2::low::Add, CapT
    >;
    using BlockSumB = axp::level2::BroadcastLane0<
        AccRecipe, ScalarFrag, softmax_sum_frag_subj, softmax_sum_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using SumScalar = axp::level2::low::FragmentExtract<
        AccRecipe, ScalarFrag, ScalarAcc, softmax_sum_frag_subj, softmax_sum_b_subj, ExecGroup, 0
    >;
    using InvSum = axp::level2::low::Rcp<AccRecipe, ScalarAcc, softmax_sum_b_subj, softmax_inv_subj, ExecGroup>;
    using FragInv = axp::level2::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, softmax_inv_subj, softmax_inv_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Mul = axp::level2::low::Mul<Recipe, Frag, softmax_tmp_subj, softmax_inv_frag_subj, OutSubj, ExecGroup>;

    using obligations = iro::util::type_list<
        ReduceMaxFrag,
        ScalarMaxFrag,
        BlockMax,
        BlockMaxB,
        MaxScalar,
        FragMax,
        Sub,
        Exp,
        ReduceSumFrag,
        ScalarSumFrag,
        BlockSum,
        BlockSumB,
        SumScalar,
        InvSum,
        FragInv,
        Mul
    >;

    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<ReduceMaxFrag, 0>, detail::in_port_t<ScalarMaxFrag, 0>>,
        iro::compose::Edge<detail::out_port_t<ScalarMaxFrag, 0>, detail::in_port_t<BlockMax, 0>>,
        iro::compose::Edge<detail::out_port_t<BlockMax, 0>, detail::in_port_t<BlockMaxB, 0>>,
        iro::compose::Edge<detail::out_port_t<BlockMaxB, 0>, detail::in_port_t<MaxScalar, 0>>,
        iro::compose::Edge<detail::out_port_t<MaxScalar, 0>, detail::in_port_t<FragMax, 0>>,
        iro::compose::Edge<detail::out_port_t<FragMax, 0>, detail::in_port_t<Sub, 1>>,
        iro::compose::Edge<detail::out_port_t<Sub, 0>, detail::in_port_t<Exp, 0>>,
        iro::compose::Edge<detail::out_port_t<Exp, 0>, detail::in_port_t<ReduceSumFrag, 0>>,
        iro::compose::Edge<detail::out_port_t<ReduceSumFrag, 0>, detail::in_port_t<ScalarSumFrag, 0>>,
        iro::compose::Edge<detail::out_port_t<ScalarSumFrag, 0>, detail::in_port_t<BlockSum, 0>>,
        iro::compose::Edge<detail::out_port_t<BlockSum, 0>, detail::in_port_t<BlockSumB, 0>>,
        iro::compose::Edge<detail::out_port_t<BlockSumB, 0>, detail::in_port_t<SumScalar, 0>>,
        iro::compose::Edge<detail::out_port_t<SumScalar, 0>, detail::in_port_t<InvSum, 0>>,
        iro::compose::Edge<detail::out_port_t<InvSum, 0>, detail::in_port_t<FragInv, 0>>,
        iro::compose::Edge<detail::out_port_t<Exp, 0>, detail::in_port_t<Mul, 0>>,
        iro::compose::Edge<detail::out_port_t<FragInv, 0>, detail::in_port_t<Mul, 1>>
    >;

    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Frag, class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
struct block_softmax_vec_impl {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "BlockSoftmaxFragVec: ExecGroup must be warp");
    static_assert(softmax_vec_legal<Recipe, Frag>::value,
                  "BlockSoftmaxFragVec: requires f16/bf16 with even fragment count and Recipe::acc == Frag::elem");

    using ScalarAcc = iro::contract::ScalarDesc<typename Recipe::acc, typename Frag::dist>;
    using ScalarFrag = iro::contract::FragmentDesc<iro::contract::Shape<1>, typename Recipe::acc, typename Frag::dist, 1>;
    using AccRecipe = iro::recipe::Accumulate<Recipe>;

    using ReduceMaxFrag = axp::level2::low::FragmentReduceAccVec<
        Recipe, Frag, ScalarAcc, InSubj, softmax_max_subj, ExecGroup, axp::level2::proto::reduction::op_max
    >;
    using ScalarMaxFrag = axp::level2::FragmentBroadcast<
        AccRecipe, ScalarFrag, ScalarAcc, softmax_max_subj, softmax_max_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using BlockMax = axp::level2::BlockReduce<
        AccRecipe, ScalarFrag, SmemTile, softmax_max_frag_subj, SmemSubjMax, ExecGroup, axp::level2::low::Max, CapT
    >;
    using BlockMaxB = axp::level2::BroadcastLane0<
        AccRecipe, ScalarFrag, softmax_max_frag_subj, softmax_max_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using MaxScalar = axp::level2::low::FragmentExtract<
        AccRecipe, ScalarFrag, ScalarAcc, softmax_max_frag_subj, softmax_max_b_subj, ExecGroup, 0
    >;
    using FragMax = axp::level2::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, softmax_max_b_subj, softmax_tmp_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Sub = axp::level2::low::Sub<Recipe, Frag, InSubj, softmax_tmp_subj, softmax_tmp_subj, ExecGroup>;
    using Exp = axp::level2::low::Exp<Recipe, Frag, softmax_tmp_subj, softmax_tmp_subj, ExecGroup>;

    using ReduceSumFrag = axp::level2::low::FragmentReduceAccVec<
        Recipe, Frag, ScalarAcc, softmax_tmp_subj, softmax_sum_subj, ExecGroup, axp::level2::proto::reduction::op_add
    >;
    using ScalarSumFrag = axp::level2::FragmentBroadcast<
        AccRecipe, ScalarFrag, ScalarAcc, softmax_sum_subj, softmax_sum_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using BlockSum = axp::level2::BlockReduce<
        AccRecipe, ScalarFrag, SmemTile, softmax_sum_frag_subj, SmemSubjSum, ExecGroup, axp::level2::low::Add, CapT
    >;
    using BlockSumB = axp::level2::BroadcastLane0<
        AccRecipe, ScalarFrag, softmax_sum_frag_subj, softmax_sum_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using SumScalar = axp::level2::low::FragmentExtract<
        AccRecipe, ScalarFrag, ScalarAcc, softmax_sum_frag_subj, softmax_sum_b_subj, ExecGroup, 0
    >;
    using InvSum = axp::level2::low::Rcp<AccRecipe, ScalarAcc, softmax_sum_b_subj, softmax_inv_subj, ExecGroup>;
    using FragInv = axp::level2::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, softmax_inv_subj, softmax_inv_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Mul = axp::level2::low::Mul<Recipe, Frag, softmax_tmp_subj, softmax_inv_frag_subj, OutSubj, ExecGroup>;

    using obligations = iro::util::type_list<
        ReduceMaxFrag,
        ScalarMaxFrag,
        BlockMax,
        BlockMaxB,
        MaxScalar,
        FragMax,
        Sub,
        Exp,
        ReduceSumFrag,
        ScalarSumFrag,
        BlockSum,
        BlockSumB,
        SumScalar,
        InvSum,
        FragInv,
        Mul
    >;

    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<ReduceMaxFrag, 0>, detail::in_port_t<ScalarMaxFrag, 0>>,
        iro::compose::Edge<detail::out_port_t<ScalarMaxFrag, 0>, detail::in_port_t<BlockMax, 0>>,
        iro::compose::Edge<detail::out_port_t<BlockMax, 0>, detail::in_port_t<BlockMaxB, 0>>,
        iro::compose::Edge<detail::out_port_t<BlockMaxB, 0>, detail::in_port_t<MaxScalar, 0>>,
        iro::compose::Edge<detail::out_port_t<MaxScalar, 0>, detail::in_port_t<FragMax, 0>>,
        iro::compose::Edge<detail::out_port_t<FragMax, 0>, detail::in_port_t<Sub, 1>>,
        iro::compose::Edge<detail::out_port_t<Sub, 0>, detail::in_port_t<Exp, 0>>,
        iro::compose::Edge<detail::out_port_t<Exp, 0>, detail::in_port_t<ReduceSumFrag, 0>>,
        iro::compose::Edge<detail::out_port_t<ReduceSumFrag, 0>, detail::in_port_t<ScalarSumFrag, 0>>,
        iro::compose::Edge<detail::out_port_t<ScalarSumFrag, 0>, detail::in_port_t<BlockSum, 0>>,
        iro::compose::Edge<detail::out_port_t<BlockSum, 0>, detail::in_port_t<BlockSumB, 0>>,
        iro::compose::Edge<detail::out_port_t<BlockSumB, 0>, detail::in_port_t<SumScalar, 0>>,
        iro::compose::Edge<detail::out_port_t<SumScalar, 0>, detail::in_port_t<InvSum, 0>>,
        iro::compose::Edge<detail::out_port_t<InvSum, 0>, detail::in_port_t<FragInv, 0>>,
        iro::compose::Edge<detail::out_port_t<Exp, 0>, detail::in_port_t<Mul, 0>>,
        iro::compose::Edge<detail::out_port_t<FragInv, 0>, detail::in_port_t<Mul, 1>>
    >;

    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Frag, class MaskPayload, class MaskSubj, class NegInfSubj,
         class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
struct block_softmax_masked_impl {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "BlockSoftmaxFragMasked: ExecGroup must be warp");
    static_assert(iro::contract::MaskPayload<MaskPayload>, "BlockSoftmaxFragMasked: MaskPayload required");

    using ScalarAcc = iro::contract::ScalarDesc<typename Recipe::acc, typename Frag::dist>;
    using ScalarFrag = iro::contract::FragmentDesc<iro::contract::Shape<1>, typename Recipe::acc, typename Frag::dist, 1>;
    using AccRecipe = iro::recipe::Accumulate<Recipe>;

    using NegInfFrag = axp::level2::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, NegInfSubj, softmax_neg_inf_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Masked = axp::level2::low::Select<
        Recipe, Frag, MaskPayload, InSubj, softmax_neg_inf_frag_subj, MaskSubj, softmax_masked_subj, ExecGroup
    >;

    using ReduceMaxFrag = axp::level2::low::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, softmax_masked_subj, softmax_max_subj, ExecGroup, axp::level2::proto::reduction::op_max
    >;
    using ScalarMaxFrag = axp::level2::FragmentBroadcast<
        AccRecipe, ScalarFrag, ScalarAcc, softmax_max_subj, softmax_max_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using BlockMax = axp::level2::BlockReduce<
        AccRecipe, ScalarFrag, SmemTile, softmax_max_frag_subj, SmemSubjMax, ExecGroup, axp::level2::low::Max, CapT
    >;
    using BlockMaxB = axp::level2::BroadcastLane0<
        AccRecipe, ScalarFrag, softmax_max_frag_subj, softmax_max_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using MaxScalar = axp::level2::low::FragmentExtract<
        AccRecipe, ScalarFrag, ScalarAcc, softmax_max_frag_subj, softmax_max_b_subj, ExecGroup, 0
    >;
    using FragMax = axp::level2::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, softmax_max_b_subj, softmax_tmp_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Sub = axp::level2::low::Sub<Recipe, Frag, softmax_masked_subj, softmax_tmp_subj, softmax_tmp_subj, ExecGroup>;
    using Exp = axp::level2::low::Exp<Recipe, Frag, softmax_tmp_subj, softmax_tmp_subj, ExecGroup>;

    using ReduceSumFrag = axp::level2::low::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, softmax_tmp_subj, softmax_sum_subj, ExecGroup, axp::level2::proto::reduction::op_add
    >;
    using ScalarSumFrag = axp::level2::FragmentBroadcast<
        AccRecipe, ScalarFrag, ScalarAcc, softmax_sum_subj, softmax_sum_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using BlockSum = axp::level2::BlockReduce<
        AccRecipe, ScalarFrag, SmemTile, softmax_sum_frag_subj, SmemSubjSum, ExecGroup, axp::level2::low::Add, CapT
    >;
    using BlockSumB = axp::level2::BroadcastLane0<
        AccRecipe, ScalarFrag, softmax_sum_frag_subj, softmax_sum_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using SumScalar = axp::level2::low::FragmentExtract<
        AccRecipe, ScalarFrag, ScalarAcc, softmax_sum_frag_subj, softmax_sum_b_subj, ExecGroup, 0
    >;
    using InvSum = axp::level2::low::Rcp<AccRecipe, ScalarAcc, softmax_sum_b_subj, softmax_inv_subj, ExecGroup>;
    using FragInv = axp::level2::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, softmax_inv_subj, softmax_inv_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Mul = axp::level2::low::Mul<Recipe, Frag, softmax_tmp_subj, softmax_inv_frag_subj, OutSubj, ExecGroup>;

    using obligations = iro::util::type_list<
        NegInfFrag,
        Masked,
        ReduceMaxFrag,
        ScalarMaxFrag,
        BlockMax,
        BlockMaxB,
        MaxScalar,
        FragMax,
        Sub,
        Exp,
        ReduceSumFrag,
        ScalarSumFrag,
        BlockSum,
        BlockSumB,
        SumScalar,
        InvSum,
        FragInv,
        Mul
    >;

    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<NegInfFrag, 0>, detail::in_port_t<Masked, 1>>,
        iro::compose::Edge<detail::out_port_t<Masked, 0>, detail::in_port_t<ReduceMaxFrag, 0>>,
        iro::compose::Edge<detail::out_port_t<ReduceMaxFrag, 0>, detail::in_port_t<ScalarMaxFrag, 0>>,
        iro::compose::Edge<detail::out_port_t<ScalarMaxFrag, 0>, detail::in_port_t<BlockMax, 0>>,
        iro::compose::Edge<detail::out_port_t<BlockMax, 0>, detail::in_port_t<BlockMaxB, 0>>,
        iro::compose::Edge<detail::out_port_t<BlockMaxB, 0>, detail::in_port_t<MaxScalar, 0>>,
        iro::compose::Edge<detail::out_port_t<MaxScalar, 0>, detail::in_port_t<FragMax, 0>>,
        iro::compose::Edge<detail::out_port_t<FragMax, 0>, detail::in_port_t<Sub, 1>>,
        iro::compose::Edge<detail::out_port_t<Masked, 0>, detail::in_port_t<Sub, 0>>,
        iro::compose::Edge<detail::out_port_t<Sub, 0>, detail::in_port_t<Exp, 0>>,
        iro::compose::Edge<detail::out_port_t<Exp, 0>, detail::in_port_t<ReduceSumFrag, 0>>,
        iro::compose::Edge<detail::out_port_t<ReduceSumFrag, 0>, detail::in_port_t<ScalarSumFrag, 0>>,
        iro::compose::Edge<detail::out_port_t<ScalarSumFrag, 0>, detail::in_port_t<BlockSum, 0>>,
        iro::compose::Edge<detail::out_port_t<BlockSum, 0>, detail::in_port_t<BlockSumB, 0>>,
        iro::compose::Edge<detail::out_port_t<BlockSumB, 0>, detail::in_port_t<SumScalar, 0>>,
        iro::compose::Edge<detail::out_port_t<SumScalar, 0>, detail::in_port_t<InvSum, 0>>,
        iro::compose::Edge<detail::out_port_t<InvSum, 0>, detail::in_port_t<FragInv, 0>>,
        iro::compose::Edge<detail::out_port_t<Exp, 0>, detail::in_port_t<Mul, 0>>,
        iro::compose::Edge<detail::out_port_t<FragInv, 0>, detail::in_port_t<Mul, 1>>
    >;

    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<bool UseVec, class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class CapT>
struct warp_softmax_dispatch;

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class CapT>
struct warp_softmax_dispatch<true, Recipe, Frag, InSubj, OutSubj, ExecGroup, CapT> {
    using type = typename warp_softmax_vec_impl<Recipe, Frag, InSubj, OutSubj, ExecGroup, CapT>::type;
};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class CapT>
struct warp_softmax_dispatch<false, Recipe, Frag, InSubj, OutSubj, ExecGroup, CapT> {
    using type = typename warp_softmax_impl<Recipe, Frag, InSubj, OutSubj, ExecGroup, CapT>::type;
};

template<bool UseVec, class Recipe, class Frag, class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup, class CapT>
struct block_softmax_dispatch;

template<class Recipe, class Frag, class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup, class CapT>
struct block_softmax_dispatch<true, Recipe, Frag, SmemTile, SmemSubjMax, SmemSubjSum,
                              InSubj, OutSubj, ExecGroup, CapT> {
    using type = typename block_softmax_vec_impl<
        Recipe, Frag, SmemTile, SmemSubjMax, SmemSubjSum, InSubj, OutSubj, ExecGroup, CapT
    >::type;
};

template<class Recipe, class Frag, class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup, class CapT>
struct block_softmax_dispatch<false, Recipe, Frag, SmemTile, SmemSubjMax, SmemSubjSum,
                               InSubj, OutSubj, ExecGroup, CapT> {
    using type = typename block_softmax_impl<
        Recipe, Frag, SmemTile, SmemSubjMax, SmemSubjSum, InSubj, OutSubj, ExecGroup, CapT
    >::type;
};

} // namespace detail

// Warp-level row softmax for fragment payloads.
template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup = iro::exec::warp,
         class CapT = axp::target_cap>
using WarpSoftmax = registry::Select<registry::WarpSoftmaxPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, CapT>;

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup = iro::exec::warp,
         class CapT = axp::target_cap>
using WarpSoftmaxVec = registry::Select<registry::WarpSoftmaxVecPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, CapT>;

template<class Recipe, class Frag, class MaskPayload, class MaskSubj, class NegInfSubj,
         class InSubj, class OutSubj, class ExecGroup = iro::exec::warp,
         class CapT = axp::target_cap>
using WarpSoftmaxMasked = registry::Select<
    registry::WarpSoftmaxMaskedPattern<Recipe, Frag, MaskPayload, MaskSubj, NegInfSubj, InSubj, OutSubj, ExecGroup>, CapT>;

// Warp-level row reductions (no shared memory handoff).
template<class Recipe, class Frag, class Subj, class ExecGroup = iro::exec::warp,
         class CapT = axp::target_cap>
using RowMaxWarp = axp::level2::WarpReduce<Recipe, Frag, Subj, ExecGroup, axp::level2::low::Max, CapT>;

template<class Recipe, class Frag, class Subj, class ExecGroup = iro::exec::warp,
         class CapT = axp::target_cap>
using RowSumWarp = axp::level2::WarpReduce<Recipe, Frag, Subj, ExecGroup, axp::level2::low::Add, CapT>;

// Block-level row reductions (two-phase via shared memory).
template<class Recipe, class Frag, class SmemTile, class Subj, class SmemSubj, class ExecGroup = iro::exec::warp,
         class CapT = axp::target_cap>
using RowMax = axp::level2::BlockReduce<Recipe, Frag, SmemTile, Subj, SmemSubj, ExecGroup, axp::level2::low::Max, CapT>;

template<class Recipe, class Frag, class SmemTile, class Subj, class SmemSubj, class ExecGroup = iro::exec::warp,
         class CapT = axp::target_cap>
using RowSum = axp::level2::BlockReduce<Recipe, Frag, SmemTile, Subj, SmemSubj, ExecGroup, axp::level2::low::Add, CapT>;

template<class Recipe, class Frag, class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup = iro::exec::warp,
         class CapT = axp::target_cap>
using BlockSoftmaxFrag = registry::Select<registry::BlockSoftmaxFragPattern<
    Recipe, Frag, SmemTile, SmemSubjMax, SmemSubjSum, InSubj, OutSubj, ExecGroup>, CapT>;

template<class Recipe, class Frag, class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup = iro::exec::warp,
         class CapT = axp::target_cap>
using BlockSoftmaxFragVec = registry::Select<registry::BlockSoftmaxFragVecPattern<
    Recipe, Frag, SmemTile, SmemSubjMax, SmemSubjSum, InSubj, OutSubj, ExecGroup>, CapT>;

template<class Recipe, class Frag, class MaskPayload, class MaskSubj, class NegInfSubj,
         class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup = iro::exec::warp,
         class CapT = axp::target_cap>
using BlockSoftmaxFragMasked = registry::Select<registry::BlockSoftmaxFragMaskedPattern<
    Recipe, Frag, MaskPayload, MaskSubj, NegInfSubj,
    SmemTile, SmemSubjMax, SmemSubjSum, InSubj, OutSubj, ExecGroup>, CapT>;

} // namespace axp::level3::domain

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level3::domain::registry {

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<WarpSoftmaxPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, Cap,
                    std::enable_if_t<std::is_same_v<ExecGroup, iro::exec::warp>>> {
    static constexpr bool supported = true;
    using type = typename axp::level3::domain::detail::warp_softmax_dispatch<
        axp::level3::domain::detail::softmax_vec_legal<Recipe, Frag>::value,
        Recipe, Frag, InSubj, OutSubj, ExecGroup, Cap
    >::type;
};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<WarpSoftmaxVecPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, Cap,
                    std::enable_if_t<std::is_same_v<ExecGroup, iro::exec::warp> &&
                                     axp::level3::domain::detail::softmax_vec_legal<Recipe, Frag>::value>> {
    static constexpr bool supported = true;
    using type = typename axp::level3::domain::detail::warp_softmax_vec_impl<
        Recipe, Frag, InSubj, OutSubj, ExecGroup, Cap
    >::type;
};

template<class Recipe, class Frag, class MaskPayload, class MaskSubj, class NegInfSubj,
         class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<WarpSoftmaxMaskedPattern<Recipe, Frag, MaskPayload, MaskSubj, NegInfSubj, InSubj, OutSubj, ExecGroup>, Cap,
                    std::enable_if_t<std::is_same_v<ExecGroup, iro::exec::warp>>> {
    static constexpr bool supported = true;
    using type = typename axp::level3::domain::detail::warp_softmax_masked_impl<
        Recipe, Frag, MaskPayload, MaskSubj, NegInfSubj, InSubj, OutSubj, ExecGroup, Cap
    >::type;
};

template<class Recipe, class Frag, class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<BlockSoftmaxFragPattern<Recipe, Frag, SmemTile, SmemSubjMax, SmemSubjSum, InSubj, OutSubj, ExecGroup>, Cap,
                    std::enable_if_t<std::is_same_v<ExecGroup, iro::exec::warp>>> {
    static constexpr bool supported = true;
    using type = typename axp::level3::domain::detail::block_softmax_dispatch<
        axp::level3::domain::detail::softmax_vec_legal<Recipe, Frag>::value,
        Recipe, Frag, SmemTile, SmemSubjMax, SmemSubjSum, InSubj, OutSubj, ExecGroup, Cap
    >::type;
};

template<class Recipe, class Frag, class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<BlockSoftmaxFragVecPattern<Recipe, Frag, SmemTile, SmemSubjMax, SmemSubjSum, InSubj, OutSubj, ExecGroup>, Cap,
                    std::enable_if_t<std::is_same_v<ExecGroup, iro::exec::warp> &&
                                     axp::level3::domain::detail::softmax_vec_legal<Recipe, Frag>::value>> {
    static constexpr bool supported = true;
    using type = typename axp::level3::domain::detail::block_softmax_vec_impl<
        Recipe, Frag, SmemTile, SmemSubjMax, SmemSubjSum, InSubj, OutSubj, ExecGroup, Cap
    >::type;
};

template<class Recipe, class Frag, class MaskPayload, class MaskSubj, class NegInfSubj,
         class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<BlockSoftmaxFragMaskedPattern<
                        Recipe, Frag, MaskPayload, MaskSubj, NegInfSubj,
                        SmemTile, SmemSubjMax, SmemSubjSum, InSubj, OutSubj, ExecGroup>,
                    Cap,
                    std::enable_if_t<std::is_same_v<ExecGroup, iro::exec::warp>>> {
    static constexpr bool supported = true;
    using type = typename axp::level3::domain::detail::block_softmax_masked_impl<
        Recipe, Frag, MaskPayload, MaskSubj, NegInfSubj,
        SmemTile, SmemSubjMax, SmemSubjSum, InSubj, OutSubj, ExecGroup, Cap
    >::type;
};

} // namespace axp::level3::domain::registry
#endif // AXP_LIBRARY_BUILD
