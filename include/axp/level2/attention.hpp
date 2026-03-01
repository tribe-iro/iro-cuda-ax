#pragma once

#include <iro_cuda_ax_core.hpp>
#include <limits>
#include <axp/concepts.hpp>
#include <axp/detail/resources.hpp>
#include <axp/detail/math.hpp>
#include <axp/state.hpp>
#include <axp/bundles/token_bundles.hpp>
#include <axp/naming/subjects.hpp>
#include "../level0/compute.hpp"
#include "../level0/convert.hpp"
#include "../level0/fragment.hpp"
#include "../level0/memory.hpp"
#include "../level0/ownership.hpp"
#include "../level0/specialize.hpp"
#include "../level0/stage.hpp"
#include "../level1/communication.hpp"
#include "../level1/reduction.hpp"
#include "../level2/matmul.hpp"
#include "../level2/wgmma.hpp"
#include "../level2/staging.hpp"
#include "../swizzle.hpp"
#include "../intent.hpp"
#include "../kits/intent.hpp"
#include "../level1/mask.hpp"
#include "registry.hpp"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace axp::level2::attention {

using SoftmaxStateF32 = axp::state::SoftmaxStateF32;

namespace detail {
template<class ExecGroup>
struct exec_lifetime;

template<>
struct exec_lifetime<iro::exec::warp> { using type = iro::token::lifetime::warp; };

template<int Warps>
struct exec_lifetime<iro::exec::warpgroup_t<Warps>> { using type = iro::token::lifetime::warpgroup; };

template<class Subject, class ExecGroup, class Lifetime>
using value_tokens = axp::bundle::ValueLive<Subject, ExecGroup, Lifetime>;

template<class Subject, class ExecGroup, class Lifetime>
using warp_reduce_out_tokens = axp::bundle::ValueLane0<Subject, ExecGroup, Lifetime>;

template<class Issue, class Wait, bool HasTma, bool Streaming>
struct stage_issue_wait_edges;

template<class Issue, class Wait, bool HasTma>
struct stage_issue_wait_edges<Issue, Wait, HasTma, true> {
    using type = iro::util::type_list<
        iro::compose::Edge<iro::compose::out_port_ref<Issue, 0>, iro::compose::in_port_ref<Wait, 0>>
    >;
};

template<class Issue, class Wait>
struct stage_issue_wait_edges<Issue, Wait, false, false> {
    using type = iro::util::type_list<
        iro::compose::Edge<iro::compose::out_port_ref<Issue, 0>, iro::compose::in_port_ref<Wait, 0>>
    >;
};

template<class Issue, class Wait>
struct stage_issue_wait_edges<Issue, Wait, true, false> {
    using type = iro::util::type_list<
        iro::compose::Edge<iro::compose::out_port_ref<Issue, 0>, iro::compose::in_port_ref<Wait, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<Issue, 1>, iro::compose::in_port_ref<Wait, 1>>
    >;
};

template<class Issue, class Wait, bool HasTma, bool Streaming>
using stage_issue_wait_edges_t = typename stage_issue_wait_edges<Issue, Wait, HasTma, Streaming>::type;

struct softmax_state_max_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.softmax_state.max"); };
struct softmax_state_max_b_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.softmax_state.max_b"); };
struct softmax_state_sum_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.softmax_state.sum"); };
struct softmax_state_sum_b_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.softmax_state.sum_b"); };
struct softmax_state_inv_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.softmax_state.inv"); };
struct softmax_state_tmp_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.softmax_state.tmp"); };
struct softmax_state_inv_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.softmax_state.inv_frag"); };
struct softmax_mask_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.softmax.mask"); };
struct softmax_neg_inf_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.softmax.neg_inf"); };
struct softmax_masked_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.softmax.masked"); };
struct softmax_neg_inf_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.softmax.neg_inf_frag"); };

using softmax_state_max_subj = iro::contract::subject::indexed<softmax_state_max_tag, 0>;
using softmax_state_max_b_subj = iro::contract::subject::indexed<softmax_state_max_b_tag, 0>;
using softmax_state_sum_subj = iro::contract::subject::indexed<softmax_state_sum_tag, 0>;
using softmax_state_sum_b_subj = iro::contract::subject::indexed<softmax_state_sum_b_tag, 0>;
using softmax_state_inv_subj = iro::contract::subject::indexed<softmax_state_inv_tag, 0>;
using softmax_state_tmp_subj = iro::contract::subject::indexed<softmax_state_tmp_tag, 0>;
using softmax_state_inv_frag_subj = iro::contract::subject::indexed<softmax_state_inv_frag_tag, 0>;
using softmax_mask_subj = iro::contract::subject::indexed<softmax_mask_tag, 0>;
using softmax_neg_inf_subj = iro::contract::subject::indexed<softmax_neg_inf_tag, 0>;
using softmax_masked_subj = iro::contract::subject::indexed<softmax_masked_tag, 0>;
using softmax_neg_inf_frag_subj = iro::contract::subject::indexed<softmax_neg_inf_frag_tag, 0>;
} // namespace detail

template<class ExecGroup>
struct is_supported_exec : std::false_type {};
template<>
struct is_supported_exec<iro::exec::warp> : std::true_type {};
template<int Warps>
struct is_supported_exec<iro::exec::warpgroup_t<Warps>> : std::true_type {};

// Combine two softmax states (m, l) into one.
template<class Recipe, class ExecGroup, class ASubj, class BSubj, class OutSubj>
    requires axp::concepts::RecipeAccF32<Recipe> &&
             (axp::concepts::WarpExec<ExecGroup> || axp::concepts::WarpgroupExec<ExecGroup>)
struct CombineSoftmaxStateF32 {
    static_assert(is_supported_exec<ExecGroup>::value, "CombineSoftmaxStateF32: ExecGroup must be warp or warpgroup");
    static_assert(std::is_same_v<typename Recipe::acc, iro::elem::f32>, "CombineSoftmaxStateF32: Recipe::acc must be f32");
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;
    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SoftmaxStateF32,
            ASubj,
            ExecGroup,
            detail::value_tokens<ASubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            SoftmaxStateF32,
            BSubj,
            ExecGroup,
            detail::value_tokens<BSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            SoftmaxStateF32,
            OutSubj,
            ExecGroup,
            detail::value_tokens<OutSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

template<class Recipe, class ExecGroup, class ASubj, class BSubj, class OutSubj>
struct CombineSoftmaxStateF32Realization
    : iro::contract::Realization<
        CombineSoftmaxStateF32<Recipe, ExecGroup, ASubj, BSubj, OutSubj>,
        iro::util::fnv1a_64_cstr("axp.attention.softmax_state_combine")> {
    __device__ __forceinline__ static SoftmaxStateF32 execute(SoftmaxStateF32 a, SoftmaxStateF32 b) {
#ifdef __CUDA_ARCH__
        float m_new = a.m > b.m ? a.m : b.m;
        float l_new = a.l * axp::detail::math::expf_recipe<Recipe>(a.m - m_new) +
                      b.l * axp::detail::math::expf_recipe<Recipe>(b.m - m_new);
        return SoftmaxStateF32{m_new, l_new};
#else
        return a.m > b.m ? a : b;
#endif
    }
};

// Warp-level reduction of softmax state (lane 0 holds the combined state).
template<class Recipe, class InSubj, class OutSubj>
    requires axp::concepts::RecipeAccF32<Recipe>
struct WarpReduceSoftmaxStateF32 {
    using ExecGroup = iro::exec::warp;
    static_assert(std::is_same_v<typename Recipe::acc, iro::elem::f32>, "WarpReduceSoftmaxStateF32: Recipe::acc must be f32");
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;
    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SoftmaxStateF32,
            InSubj,
            ExecGroup,
            detail::value_tokens<InSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            SoftmaxStateF32,
            OutSubj,
            ExecGroup,
            detail::warp_reduce_out_tokens<OutSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

template<class Recipe, class InSubj, class OutSubj>
struct WarpReduceSoftmaxStateF32Realization
    : iro::contract::Realization<
        WarpReduceSoftmaxStateF32<Recipe, InSubj, OutSubj>,
        iro::util::fnv1a_64_cstr("axp.attention.softmax_state_warp_reduce")> {
    __device__ __forceinline__ static SoftmaxStateF32 execute(SoftmaxStateF32 s) {
#ifdef __CUDA_ARCH__
        const unsigned mask = __activemask();
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float m_other = __shfl_down_sync(mask, s.m, offset);
            float l_other = __shfl_down_sync(mask, s.l, offset);
            float m_new = s.m > m_other ? s.m : m_other;
            float l_new = s.l * axp::detail::math::expf_recipe<Recipe>(s.m - m_new) +
                          l_other * axp::detail::math::expf_recipe<Recipe>(m_other - m_new);
            s.m = m_new;
            s.l = l_new;
        }
        return s;
#else
        return s;
#endif
    }
};

// Make softmax state from max/sum scalars (lane-replicated).
template<class Recipe, class ScalarPayload, class MaxSubj, class SumSubj, class OutSubj, class ExecGroup>
    requires axp::concepts::RecipeAccF32<Recipe>
struct SoftmaxStateFromScalarsF32 {
    static_assert(is_supported_exec<ExecGroup>::value, "SoftmaxStateFromScalarsF32: ExecGroup must be warp or warpgroup");
    static_assert(iro::contract::ScalarPayload<ScalarPayload>, "SoftmaxStateFromScalarsF32: ScalarPayload required");
    static_assert(std::is_same_v<typename ScalarPayload::elem, iro::elem::f32>,
                  "SoftmaxStateFromScalarsF32: ScalarPayload elem must be f32");
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            ScalarPayload,
            MaxSubj,
            ExecGroup,
            detail::value_tokens<MaxSubj, ExecGroup, lifetime>,
            typename ScalarPayload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            ScalarPayload,
            SumSubj,
            ExecGroup,
            detail::value_tokens<SumSubj, ExecGroup, lifetime>,
            typename ScalarPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            SoftmaxStateF32,
            OutSubj,
            ExecGroup,
            detail::value_tokens<OutSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

template<class Recipe, class ScalarPayload, class MaxSubj, class SumSubj, class OutSubj, class ExecGroup>
struct SoftmaxStateFromScalarsF32Realization
    : iro::contract::Realization<
        SoftmaxStateFromScalarsF32<Recipe, ScalarPayload, MaxSubj, SumSubj, OutSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.attention.softmax_state_from_scalars")> {
    using scalar_t = typename ScalarPayload::elem::storage_t;
    __device__ __forceinline__ static SoftmaxStateF32 execute(const scalar_t* max_in, const scalar_t* sum_in) {
#ifdef __CUDA_ARCH__
        return SoftmaxStateF32{static_cast<float>(max_in[0]), static_cast<float>(sum_in[0])};
#else
        (void)max_in; (void)sum_in;
        return SoftmaxStateF32{0.0f, 1.0f};
#endif
    }
};

// Copy softmax state (explicit fork).
template<class Recipe, class InSubj, class OutSubj, class ExecGroup>
    requires axp::concepts::RecipeAccF32<Recipe>
struct SoftmaxStateCopyF32 {
    static_assert(is_supported_exec<ExecGroup>::value, "SoftmaxStateCopyF32: ExecGroup must be warp or warpgroup");
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SoftmaxStateF32,
            InSubj,
            ExecGroup,
            detail::value_tokens<InSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            SoftmaxStateF32,
            OutSubj,
            ExecGroup,
            detail::value_tokens<OutSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

template<class Recipe, class InSubj, class OutSubj, class ExecGroup>
struct SoftmaxStateCopyF32Realization
    : iro::contract::Realization<
        SoftmaxStateCopyF32<Recipe, InSubj, OutSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.attention.softmax_state_copy")> {
    __device__ __forceinline__ static SoftmaxStateF32 execute(SoftmaxStateF32 s) {
#ifdef __CUDA_ARCH__
        return s;
#else
        return s;
#endif
    }
};

// Compute scale for tile-normalized weights (exp(m_tile - m_new) * l_tile / l_new).
template<class Recipe, class TileStateSubj, class NewStateSubj, class OutSubj, class ExecGroup>
    requires axp::concepts::RecipeAccF32<Recipe>
struct SoftmaxStateScaleF32 {
    static_assert(is_supported_exec<ExecGroup>::value, "SoftmaxStateScaleF32: ExecGroup must be warp or warpgroup");
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;
    using ScalarOut = iro::contract::ScalarDesc<iro::elem::f32, iro::dist::replicated>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SoftmaxStateF32,
            TileStateSubj,
            ExecGroup,
            detail::value_tokens<TileStateSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            SoftmaxStateF32,
            NewStateSubj,
            ExecGroup,
            detail::value_tokens<NewStateSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            ScalarOut,
            OutSubj,
            ExecGroup,
            detail::value_tokens<OutSubj, ExecGroup, lifetime>,
            typename ScalarOut::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

template<class Recipe, class TileStateSubj, class NewStateSubj, class OutSubj, class ExecGroup>
struct SoftmaxStateScaleF32Realization
    : iro::contract::Realization<
        SoftmaxStateScaleF32<Recipe, TileStateSubj, NewStateSubj, OutSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.attention.softmax_state_scale")> {
    __device__ __forceinline__ static float execute(SoftmaxStateF32 tile_state, SoftmaxStateF32 new_state) {
#ifdef __CUDA_ARCH__
        float scale = axp::detail::math::expf_recipe<Recipe>(tile_state.m - new_state.m) * (tile_state.l / new_state.l);
        return scale;
#else
        (void)tile_state; (void)new_state;
        return 0.0f;
#endif
    }
};

// Warp softmax that also emits the tile softmax state (m, l).
template<class Recipe, class Frag, class InSubj, class OutSubj, class StateSubj, class ExecGroup,
         class CapT = axp::target_cap>
    requires axp::concepts::RecipeAccF32<Recipe>
struct WarpSoftmaxStateF32 {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "WarpSoftmaxStateF32: ExecGroup must be warp");
    static_assert(std::is_same_v<typename Recipe::in, typename Recipe::acc>,
                  "WarpSoftmaxStateF32: requires Recipe::in == Recipe::acc");
    static_assert(std::is_same_v<typename Recipe::acc, typename Recipe::out>,
                  "WarpSoftmaxStateF32: requires Recipe::acc == Recipe::out");
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::in>,
                  "WarpSoftmaxStateF32: fragment elem must match Recipe::in");
    static_assert(iro::contract::FragmentPayload<Frag>, "WarpSoftmaxStateF32: Frag payload required");

    using Exec = iro::exec::warp;
    using ScalarAcc = iro::contract::ScalarDesc<typename Recipe::acc, typename Frag::dist>;
    using AccRecipe = iro::recipe::Accumulate<Recipe>;

    using ReduceMaxFrag = axp::level0::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, InSubj, detail::softmax_state_max_subj, Exec, axp::protocol::reduction::op_max
    >;
    using ReduceMaxWarp = axp::level1::WarpReduce<
        AccRecipe, ScalarAcc, detail::softmax_state_max_subj, Exec, axp::level0::Max, CapT
    >;
    using BroadcastMax = axp::level1::BroadcastLane0<
        AccRecipe, ScalarAcc, detail::softmax_state_max_subj, detail::softmax_state_max_b_subj, Exec,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using FragMax = axp::level1::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, detail::softmax_state_max_b_subj, detail::softmax_state_tmp_subj, Exec,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Sub = axp::level0::Sub<Recipe, Frag, InSubj, detail::softmax_state_tmp_subj, detail::softmax_state_tmp_subj, Exec>;
    using Exp = axp::level0::Exp<Recipe, Frag, detail::softmax_state_tmp_subj, detail::softmax_state_tmp_subj, Exec>;

    using ReduceSumFrag = axp::level0::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, detail::softmax_state_tmp_subj, detail::softmax_state_sum_subj, Exec, axp::protocol::reduction::op_add
    >;
    using ReduceSumWarp = axp::level1::WarpReduce<
        AccRecipe, ScalarAcc, detail::softmax_state_sum_subj, Exec, axp::level0::Add, CapT
    >;
    using BroadcastSum = axp::level1::BroadcastLane0<
        AccRecipe, ScalarAcc, detail::softmax_state_sum_subj, detail::softmax_state_sum_b_subj, Exec,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using InvSum = axp::level0::Rcp<AccRecipe, ScalarAcc, detail::softmax_state_sum_b_subj, detail::softmax_state_inv_subj, Exec>;
    using FragInv = axp::level1::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, detail::softmax_state_inv_subj, detail::softmax_state_inv_frag_subj, Exec,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Mul = axp::level0::Mul<Recipe, Frag, detail::softmax_state_tmp_subj, detail::softmax_state_inv_frag_subj, OutSubj, Exec>;

    using MakeState = SoftmaxStateFromScalarsF32<
        AccRecipe, ScalarAcc, detail::softmax_state_max_b_subj, detail::softmax_state_sum_b_subj, StateSubj, Exec
    >;

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
        Mul,
        MakeState
    >;

    using edges = iro::util::type_list<
        iro::compose::Edge<iro::compose::out_port_ref<ReduceMaxFrag, 0>, iro::compose::in_port_ref<ReduceMaxWarp, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<ReduceMaxWarp, 0>, iro::compose::in_port_ref<BroadcastMax, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastMax, 0>, iro::compose::in_port_ref<FragMax, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<FragMax, 0>, iro::compose::in_port_ref<Sub, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<Sub, 0>, iro::compose::in_port_ref<Exp, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<Exp, 0>, iro::compose::in_port_ref<ReduceSumFrag, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<ReduceSumFrag, 0>, iro::compose::in_port_ref<ReduceSumWarp, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<ReduceSumWarp, 0>, iro::compose::in_port_ref<BroadcastSum, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastSum, 0>, iro::compose::in_port_ref<InvSum, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<InvSum, 0>, iro::compose::in_port_ref<FragInv, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<Exp, 0>, iro::compose::in_port_ref<Mul, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<FragInv, 0>, iro::compose::in_port_ref<Mul, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastMax, 0>, iro::compose::in_port_ref<MakeState, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastSum, 0>, iro::compose::in_port_ref<MakeState, 1>>
    >;

    using type = axp::level2::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

// Warp softmax with mask (uses mask to set -inf before softmax, emits tile state).
template<class Recipe, class Frag, class MaskPayload, class MaskSubj, class NegInfSubj,
         class InSubj, class OutSubj, class StateSubj, class ExecGroup,
         class CapT = axp::target_cap>
    requires axp::concepts::RecipeAccF32<Recipe>
struct WarpSoftmaxStateMaskedF32 {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "WarpSoftmaxStateMaskedF32: ExecGroup must be warp");
    static_assert(std::is_same_v<typename Recipe::in, typename Recipe::acc>,
                  "WarpSoftmaxStateMaskedF32: requires Recipe::in == Recipe::acc");
    static_assert(std::is_same_v<typename Recipe::acc, typename Recipe::out>,
                  "WarpSoftmaxStateMaskedF32: requires Recipe::acc == Recipe::out");
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::in>,
                  "WarpSoftmaxStateMaskedF32: fragment elem must match Recipe::in");
    static_assert(iro::contract::FragmentPayload<Frag>, "WarpSoftmaxStateMaskedF32: Frag payload required");
    static_assert(iro::contract::MaskPayload<MaskPayload>, "WarpSoftmaxStateMaskedF32: MaskPayload required");

    using Exec = iro::exec::warp;
    using ScalarAcc = iro::contract::ScalarDesc<typename Recipe::acc, typename Frag::dist>;
    using AccRecipe = iro::recipe::Accumulate<Recipe>;

    using NegInfFrag = axp::level1::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, NegInfSubj, detail::softmax_neg_inf_frag_subj, Exec,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Masked = axp::level0::Select<
        Recipe, Frag, MaskPayload, InSubj, detail::softmax_neg_inf_frag_subj, MaskSubj, detail::softmax_masked_subj, Exec
    >;

    using ReduceMaxFrag = axp::level0::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, detail::softmax_masked_subj, detail::softmax_state_max_subj, Exec, axp::protocol::reduction::op_max
    >;
    using ReduceMaxWarp = axp::level1::WarpReduce<
        AccRecipe, ScalarAcc, detail::softmax_state_max_subj, Exec, axp::level0::Max, CapT
    >;
    using BroadcastMax = axp::level1::BroadcastLane0<
        AccRecipe, ScalarAcc, detail::softmax_state_max_subj, detail::softmax_state_max_b_subj, Exec,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using FragMax = axp::level1::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, detail::softmax_state_max_b_subj, detail::softmax_state_tmp_subj, Exec,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Sub = axp::level0::Sub<Recipe, Frag, detail::softmax_masked_subj, detail::softmax_state_tmp_subj, detail::softmax_state_tmp_subj, Exec>;
    using Exp = axp::level0::Exp<Recipe, Frag, detail::softmax_state_tmp_subj, detail::softmax_state_tmp_subj, Exec>;

    using ReduceSumFrag = axp::level0::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, detail::softmax_state_tmp_subj, detail::softmax_state_sum_subj, Exec, axp::protocol::reduction::op_add
    >;
    using ReduceSumWarp = axp::level1::WarpReduce<
        AccRecipe, ScalarAcc, detail::softmax_state_sum_subj, Exec, axp::level0::Add, CapT
    >;
    using BroadcastSum = axp::level1::BroadcastLane0<
        AccRecipe, ScalarAcc, detail::softmax_state_sum_subj, detail::softmax_state_sum_b_subj, Exec,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using InvSum = axp::level0::Rcp<AccRecipe, ScalarAcc, detail::softmax_state_sum_b_subj, detail::softmax_state_inv_subj, Exec>;
    using FragInv = axp::level1::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, detail::softmax_state_inv_subj, detail::softmax_state_inv_frag_subj, Exec,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Mul = axp::level0::Mul<Recipe, Frag, detail::softmax_state_tmp_subj, detail::softmax_state_inv_frag_subj, OutSubj, Exec>;

    using MakeState = SoftmaxStateFromScalarsF32<
        AccRecipe, ScalarAcc, detail::softmax_state_max_b_subj, detail::softmax_state_sum_b_subj, StateSubj, Exec
    >;

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
        Mul,
        MakeState
    >;

    using edges = iro::util::type_list<
        iro::compose::Edge<iro::compose::out_port_ref<NegInfFrag, 0>, iro::compose::in_port_ref<Masked, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<Masked, 0>, iro::compose::in_port_ref<ReduceMaxFrag, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<ReduceMaxFrag, 0>, iro::compose::in_port_ref<ReduceMaxWarp, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<ReduceMaxWarp, 0>, iro::compose::in_port_ref<BroadcastMax, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastMax, 0>, iro::compose::in_port_ref<FragMax, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<FragMax, 0>, iro::compose::in_port_ref<Sub, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<Sub, 0>, iro::compose::in_port_ref<Exp, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<Exp, 0>, iro::compose::in_port_ref<ReduceSumFrag, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<ReduceSumFrag, 0>, iro::compose::in_port_ref<ReduceSumWarp, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<ReduceSumWarp, 0>, iro::compose::in_port_ref<BroadcastSum, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastSum, 0>, iro::compose::in_port_ref<InvSum, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<InvSum, 0>, iro::compose::in_port_ref<FragInv, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<Exp, 0>, iro::compose::in_port_ref<Mul, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<FragInv, 0>, iro::compose::in_port_ref<Mul, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastMax, 0>, iro::compose::in_port_ref<MakeState, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastSum, 0>, iro::compose::in_port_ref<MakeState, 1>>
    >;

    using type = axp::level2::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

// Warpgroup softmax that emits the tile softmax state (m, l).
template<class Recipe, class Frag, class InSubj, class OutSubj, class StateSubj, class ExecGroup,
         class CapT = axp::target_cap>
    requires axp::concepts::RecipeAccF32<Recipe>
struct WarpgroupSoftmaxStateF32 {
    static_assert(iro::exec::is_warpgroup_v<ExecGroup>,
                  "WarpgroupSoftmaxStateF32: ExecGroup must be warpgroup");
    static_assert(std::is_same_v<typename Recipe::in, typename Recipe::acc>,
                  "WarpgroupSoftmaxStateF32: requires Recipe::in == Recipe::acc");
    static_assert(std::is_same_v<typename Recipe::acc, typename Recipe::out>,
                  "WarpgroupSoftmaxStateF32: requires Recipe::acc == Recipe::out");
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::in>,
                  "WarpgroupSoftmaxStateF32: fragment elem must match Recipe::in");
    static_assert(iro::contract::FragmentPayload<Frag>, "WarpgroupSoftmaxStateF32: Frag payload required");

    using Exec = ExecGroup;
    using ScalarAcc = iro::contract::ScalarDesc<typename Recipe::acc, typename Frag::dist>;
    using AccRecipe = iro::recipe::Accumulate<Recipe>;
    static constexpr int kBarrierMax = 1;
    static constexpr int kBarrierSum = 2;

    using ReduceMaxFrag = axp::level0::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, InSubj, detail::softmax_state_max_subj, Exec, axp::protocol::reduction::op_max
    >;
    using ReduceMaxGroup = axp::level1::WarpgroupReduce<
        AccRecipe, ScalarAcc, detail::softmax_state_max_subj, Exec, axp::protocol::reduction::op_max, kBarrierMax,
        iro::exec::warpgroup_warps<Exec>::value, CapT
    >;
    using BroadcastMax = axp::level1::WarpgroupBroadcastLane0<
        AccRecipe, ScalarAcc, detail::softmax_state_max_subj, detail::softmax_state_max_b_subj, Exec, kBarrierMax,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using FragMax = axp::level1::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, detail::softmax_state_max_b_subj, detail::softmax_state_tmp_subj, Exec,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Sub = axp::level0::Sub<Recipe, Frag, InSubj, detail::softmax_state_tmp_subj, detail::softmax_state_tmp_subj, Exec>;
    using Exp = axp::level0::Exp<Recipe, Frag, detail::softmax_state_tmp_subj, detail::softmax_state_tmp_subj, Exec>;

    using ReduceSumFrag = axp::level0::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, detail::softmax_state_tmp_subj, detail::softmax_state_sum_subj, Exec, axp::protocol::reduction::op_add
    >;
    using ReduceSumGroup = axp::level1::WarpgroupReduce<
        AccRecipe, ScalarAcc, detail::softmax_state_sum_subj, Exec, axp::protocol::reduction::op_add, kBarrierSum,
        iro::exec::warpgroup_warps<Exec>::value, CapT
    >;
    using BroadcastSum = axp::level1::WarpgroupBroadcastLane0<
        AccRecipe, ScalarAcc, detail::softmax_state_sum_subj, detail::softmax_state_sum_b_subj, Exec, kBarrierSum,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using InvSum = axp::level0::Rcp<AccRecipe, ScalarAcc, detail::softmax_state_sum_b_subj, detail::softmax_state_inv_subj, Exec>;
    using FragInv = axp::level1::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, detail::softmax_state_inv_subj, detail::softmax_state_inv_frag_subj, Exec,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Mul = axp::level0::Mul<Recipe, Frag, detail::softmax_state_tmp_subj, detail::softmax_state_inv_frag_subj, OutSubj, Exec>;

    using MakeState = SoftmaxStateFromScalarsF32<
        AccRecipe, ScalarAcc, detail::softmax_state_max_b_subj, detail::softmax_state_sum_b_subj, StateSubj, Exec
    >;

    using obligations = iro::util::type_list<
        ReduceMaxFrag,
        ReduceMaxGroup,
        BroadcastMax,
        FragMax,
        Sub,
        Exp,
        ReduceSumFrag,
        ReduceSumGroup,
        BroadcastSum,
        InvSum,
        FragInv,
        Mul,
        MakeState
    >;

    using edges = iro::util::type_list<
        iro::compose::Edge<iro::compose::out_port_ref<ReduceMaxFrag, 0>, iro::compose::in_port_ref<ReduceMaxGroup, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<ReduceMaxGroup, 0>, iro::compose::in_port_ref<BroadcastMax, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastMax, 0>, iro::compose::in_port_ref<FragMax, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<FragMax, 0>, iro::compose::in_port_ref<Sub, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<Sub, 0>, iro::compose::in_port_ref<Exp, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<Exp, 0>, iro::compose::in_port_ref<ReduceSumFrag, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<ReduceSumFrag, 0>, iro::compose::in_port_ref<ReduceSumGroup, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<ReduceSumGroup, 0>, iro::compose::in_port_ref<BroadcastSum, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastSum, 0>, iro::compose::in_port_ref<InvSum, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<InvSum, 0>, iro::compose::in_port_ref<FragInv, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<Exp, 0>, iro::compose::in_port_ref<Mul, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<FragInv, 0>, iro::compose::in_port_ref<Mul, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastMax, 0>, iro::compose::in_port_ref<MakeState, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastSum, 0>, iro::compose::in_port_ref<MakeState, 1>>
    >;

    using type = axp::level2::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

// Warpgroup softmax with mask (uses mask to set -inf before softmax, emits tile state).
template<class Recipe, class Frag, class MaskPayload, class MaskSubj, class NegInfSubj,
         class InSubj, class OutSubj, class StateSubj, class ExecGroup,
         class CapT = axp::target_cap>
    requires axp::concepts::RecipeAccF32<Recipe>
struct WarpgroupSoftmaxStateMaskedF32 {
    static_assert(iro::exec::is_warpgroup_v<ExecGroup>,
                  "WarpgroupSoftmaxStateMaskedF32: ExecGroup must be warpgroup");
    static_assert(std::is_same_v<typename Recipe::in, typename Recipe::acc>,
                  "WarpgroupSoftmaxStateMaskedF32: requires Recipe::in == Recipe::acc");
    static_assert(std::is_same_v<typename Recipe::acc, typename Recipe::out>,
                  "WarpgroupSoftmaxStateMaskedF32: requires Recipe::acc == Recipe::out");
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::in>,
                  "WarpgroupSoftmaxStateMaskedF32: fragment elem must match Recipe::in");
    static_assert(iro::contract::FragmentPayload<Frag>, "WarpgroupSoftmaxStateMaskedF32: Frag payload required");
    static_assert(iro::contract::MaskPayload<MaskPayload>, "WarpgroupSoftmaxStateMaskedF32: MaskPayload required");

    using Exec = ExecGroup;
    using ScalarAcc = iro::contract::ScalarDesc<typename Recipe::acc, typename Frag::dist>;
    using AccRecipe = iro::recipe::Accumulate<Recipe>;
    static constexpr int kBarrierMax = 1;
    static constexpr int kBarrierSum = 2;

    using NegInfFrag = axp::level1::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, NegInfSubj, detail::softmax_neg_inf_frag_subj, Exec,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Masked = axp::level0::Select<
        Recipe, Frag, MaskPayload, InSubj, detail::softmax_neg_inf_frag_subj, MaskSubj, detail::softmax_masked_subj, Exec
    >;

    using ReduceMaxFrag = axp::level0::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, detail::softmax_masked_subj, detail::softmax_state_max_subj, Exec, axp::protocol::reduction::op_max
    >;
    using ReduceMaxGroup = axp::level1::WarpgroupReduce<
        AccRecipe, ScalarAcc, detail::softmax_state_max_subj, Exec, axp::protocol::reduction::op_max, kBarrierMax,
        iro::exec::warpgroup_warps<Exec>::value, CapT
    >;
    using BroadcastMax = axp::level1::WarpgroupBroadcastLane0<
        AccRecipe, ScalarAcc, detail::softmax_state_max_subj, detail::softmax_state_max_b_subj, Exec, kBarrierMax,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using FragMax = axp::level1::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, detail::softmax_state_max_b_subj, detail::softmax_state_tmp_subj, Exec,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Sub = axp::level0::Sub<Recipe, Frag, detail::softmax_masked_subj, detail::softmax_state_tmp_subj, detail::softmax_state_tmp_subj, Exec>;
    using Exp = axp::level0::Exp<Recipe, Frag, detail::softmax_state_tmp_subj, detail::softmax_state_tmp_subj, Exec>;

    using ReduceSumFrag = axp::level0::FragmentReduceAcc<
        Recipe, Frag, ScalarAcc, detail::softmax_state_tmp_subj, detail::softmax_state_sum_subj, Exec, axp::protocol::reduction::op_add
    >;
    using ReduceSumGroup = axp::level1::WarpgroupReduce<
        AccRecipe, ScalarAcc, detail::softmax_state_sum_subj, Exec, axp::protocol::reduction::op_add, kBarrierSum,
        iro::exec::warpgroup_warps<Exec>::value, CapT
    >;
    using BroadcastSum = axp::level1::WarpgroupBroadcastLane0<
        AccRecipe, ScalarAcc, detail::softmax_state_sum_subj, detail::softmax_state_sum_b_subj, Exec, kBarrierSum,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using InvSum = axp::level0::Rcp<AccRecipe, ScalarAcc, detail::softmax_state_sum_b_subj, detail::softmax_state_inv_subj, Exec>;
    using FragInv = axp::level1::FragmentBroadcast<
        Recipe, Frag, ScalarAcc, detail::softmax_state_inv_subj, detail::softmax_state_inv_frag_subj, Exec,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Mul = axp::level0::Mul<Recipe, Frag, detail::softmax_state_tmp_subj, detail::softmax_state_inv_frag_subj, OutSubj, Exec>;

    using MakeState = SoftmaxStateFromScalarsF32<
        AccRecipe, ScalarAcc, detail::softmax_state_max_b_subj, detail::softmax_state_sum_b_subj, StateSubj, Exec
    >;

    using obligations = iro::util::type_list<
        NegInfFrag,
        Masked,
        ReduceMaxFrag,
        ReduceMaxGroup,
        BroadcastMax,
        FragMax,
        Sub,
        Exp,
        ReduceSumFrag,
        ReduceSumGroup,
        BroadcastSum,
        InvSum,
        FragInv,
        Mul,
        MakeState
    >;

    using edges = iro::util::type_list<
        iro::compose::Edge<iro::compose::out_port_ref<NegInfFrag, 0>, iro::compose::in_port_ref<Masked, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<Masked, 0>, iro::compose::in_port_ref<ReduceMaxFrag, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<ReduceMaxFrag, 0>, iro::compose::in_port_ref<ReduceMaxGroup, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<ReduceMaxGroup, 0>, iro::compose::in_port_ref<BroadcastMax, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastMax, 0>, iro::compose::in_port_ref<FragMax, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<FragMax, 0>, iro::compose::in_port_ref<Sub, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<Sub, 0>, iro::compose::in_port_ref<Exp, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<Exp, 0>, iro::compose::in_port_ref<ReduceSumFrag, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<ReduceSumFrag, 0>, iro::compose::in_port_ref<ReduceSumGroup, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<ReduceSumGroup, 0>, iro::compose::in_port_ref<BroadcastSum, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastSum, 0>, iro::compose::in_port_ref<InvSum, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<InvSum, 0>, iro::compose::in_port_ref<FragInv, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<Exp, 0>, iro::compose::in_port_ref<Mul, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<FragInv, 0>, iro::compose::in_port_ref<Mul, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastMax, 0>, iro::compose::in_port_ref<MakeState, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<BroadcastSum, 0>, iro::compose::in_port_ref<MakeState, 1>>
    >;

    using type = axp::level2::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

// Rescale accumulator fragment when softmax state changes across tiles.
template<class Recipe, class AccFrag, class AccSubj, class OldStateSubj, class NewStateSubj, class ExecGroup>
    requires axp::concepts::RecipeAccF32<Recipe> &&
             (axp::concepts::WarpExec<ExecGroup> || axp::concepts::WarpgroupExec<ExecGroup>)
struct RescaleAccumulatorF32 {
    static_assert(is_supported_exec<ExecGroup>::value, "RescaleAccumulatorF32: ExecGroup must be warp or warpgroup");
    static_assert(iro::contract::FragmentPayload<AccFrag>, "RescaleAccumulatorF32 expects FragmentDesc payload");
    static_assert(std::is_same_v<typename AccFrag::elem, iro::elem::f32>, "RescaleAccumulatorF32 requires f32 acc");
    static_assert(std::is_same_v<typename Recipe::acc, iro::elem::f32>, "RescaleAccumulatorF32: Recipe::acc must be f32");
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;
    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            AccFrag,
            AccSubj,
            ExecGroup,
            detail::value_tokens<AccSubj, ExecGroup, lifetime>,
            typename AccFrag::dist,
            Recipe
        >,
        iro::contract::InputPort<
            SoftmaxStateF32,
            OldStateSubj,
            ExecGroup,
            detail::value_tokens<OldStateSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            SoftmaxStateF32,
            NewStateSubj,
            ExecGroup,
            detail::value_tokens<NewStateSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            AccFrag,
            AccSubj,
            ExecGroup,
            detail::value_tokens<AccSubj, ExecGroup, lifetime>,
            typename AccFrag::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

template<class Recipe, class AccFrag, class AccSubj, class OldStateSubj, class NewStateSubj, class ExecGroup>
struct RescaleAccumulatorF32Realization
    : iro::contract::Realization<
        RescaleAccumulatorF32<Recipe, AccFrag, AccSubj, OldStateSubj, NewStateSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.attention.softmax_rescale")> {
    __device__ __forceinline__ static void execute(float* acc, SoftmaxStateF32 old_state, SoftmaxStateF32 new_state) {
#ifdef __CUDA_ARCH__
        float scale = axp::detail::math::expf_recipe<Recipe>(old_state.m - new_state.m) * (old_state.l / new_state.l);
        constexpr int kElems = static_cast<int>(AccFrag::count);
        #pragma unroll
        for (int i = 0; i < kElems; ++i) {
            acc[i] *= scale;
        }
#else
        (void)acc; (void)old_state; (void)new_state;
#endif
    }
};

namespace detail {

struct online_state_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.online.state"); };
using online_state_subj = iro::contract::subject::indexed<online_state_tag, 0>;

template<class Recipe, class AccFrag, class AccSubj, class OldStateSubj, class NewStateSubj, class OutStateSubj,
         class ExecGroup, class CapT = axp::target_cap>
struct online_softmax_update_impl {
    using Combine = CombineSoftmaxStateF32<Recipe, ExecGroup, OldStateSubj, NewStateSubj, OutStateSubj>;
    using Rescale = RescaleAccumulatorF32<Recipe, AccFrag, AccSubj, OldStateSubj, OutStateSubj, ExecGroup>;

    using obligations = iro::util::type_list<Combine, Rescale>;
    using edges = iro::util::type_list<
        iro::compose::Edge<iro::compose::out_port_ref<Combine, 0>, iro::compose::in_port_ref<Rescale, 2>>
    >;
    using type = axp::level2::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

} // namespace detail

template<class Recipe, class ExecGroup, class ASubj, class BSubj, class OutSubj>
using SoftmaxStateCombine = CombineSoftmaxStateF32<Recipe, ExecGroup, ASubj, BSubj, OutSubj>;

template<class Recipe, class Frag, class InSubj, class OutSubj, class StateSubj, class ExecGroup,
         class CapT = axp::target_cap>
using WarpSoftmaxState = registry::Select<
    registry::WarpSoftmaxStatePattern<Recipe, Frag, InSubj, OutSubj, StateSubj, ExecGroup>, CapT>;

template<class Recipe, class Frag, class MaskPayload, class MaskSubj, class NegInfSubj,
         class InSubj, class OutSubj, class StateSubj, class ExecGroup,
         class CapT = axp::target_cap>
using WarpSoftmaxStateMasked = registry::Select<
    registry::WarpSoftmaxStateMaskedPattern<
        Recipe, Frag, MaskPayload, MaskSubj, NegInfSubj, InSubj, OutSubj, StateSubj, ExecGroup>, CapT>;

template<class Recipe, class InSubj, class OutSubj>
using WarpReduceSoftmaxState = WarpReduceSoftmaxStateF32<Recipe, InSubj, OutSubj>;

template<class Recipe, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
using SoftmaxStateCopy = registry::Select<
    registry::SoftmaxStateCopyPattern<Recipe, InSubj, OutSubj, ExecGroup>, CapT>;

template<class Recipe, class TileStateSubj, class NewStateSubj, class OutSubj, class ExecGroup,
         class CapT = axp::target_cap>
using SoftmaxStateScale = registry::Select<
    registry::SoftmaxStateScalePattern<Recipe, TileStateSubj, NewStateSubj, OutSubj, ExecGroup>, CapT>;

template<class Recipe, class AccFrag, class AccSubj, class OldStateSubj, class NewStateSubj, class ExecGroup>
using RescaleAccumulator = RescaleAccumulatorF32<Recipe, AccFrag, AccSubj, OldStateSubj, NewStateSubj, ExecGroup>;

// Tile-skip hook (pipeline-level predicate, no-op for now).
template<class Recipe, class PredSubj, class ExecGroup>
struct TileSkipHook {
    static_assert(is_supported_exec<ExecGroup>::value, "TileSkipHook: ExecGroup must be warp or warpgroup");
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;
    using PredPayload = iro::contract::ScalarDesc<iro::elem::u8, iro::dist::replicated>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            PredPayload,
            PredSubj,
            ExecGroup,
            detail::value_tokens<PredSubj, ExecGroup, lifetime>,
            typename PredPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            PredPayload,
            PredSubj,
            ExecGroup,
            detail::value_tokens<PredSubj, ExecGroup, lifetime>,
            typename PredPayload::dist,
            Recipe
        >
    >;
    using resources = iro::util::type_list<>;
};

template<class Recipe, class PredSubj, class ExecGroup>
struct TileSkipHookRealization
    : iro::contract::Realization<
        TileSkipHook<Recipe, PredSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.attention.tile_skip_hook")> {
    using PredPayload = typename TileSkipHook<Recipe, PredSubj, ExecGroup>::PredPayload;
    using pred_t = typename PredPayload::elem::storage_t;
    __device__ __forceinline__ static void execute(const pred_t* pred, pred_t* out) {
#ifdef __CUDA_ARCH__
        out[0] = pred[0];
#else
        (void)pred;
        (void)out;
#endif
    }
};

// Warpgroup attention tile using WGMMA (SM90).
template<class Recipe, int TileM, int TileN, int HeadDim, int Stages,
         class QSubj, class KSubj, class VSubj,
         class AccSubj, class OldStateSubj, class OutStateSubj,
         class MemoryPatternQ, class MemoryPatternK, class MemoryPatternV,
         class LoadModeQ, class LoadModeK, class LoadModeV,
         class Schedule,
         class TileSkip,
         class QTma, class KTma, class VTma, class CapT>
struct AttentionWgmmaImpl {
    static_assert(std::is_same_v<typename Recipe::acc, iro::elem::f32>,
                  "AttentionWgmma: Recipe::acc must be f32");
    static_assert(Stages >= 2 && Stages <= 4, "AttentionWgmma: Stages must be 2-4");
    using ElemA = iro::verify::recipe_in_a_t<Recipe>;
    using ElemB = iro::verify::recipe_in_b_t<Recipe>;
    static_assert(std::is_same_v<ElemA, ElemB>, "AttentionWgmma: Q/K/V elem must match");
    static_assert(std::is_same_v<ElemA, iro::elem::f16> || std::is_same_v<ElemA, iro::elem::bf16>,
                  "AttentionWgmma: only f16/bf16 inputs supported");
    static_assert(std::is_same_v<TileSkip, axp::intent::tile_skip::None> ||
                  std::is_same_v<TileSkip, axp::intent::tile_skip::Causal>,
                  "AttentionWgmma: TileSkip must be intent::tile_skip::None or Causal");
    static constexpr int kWgmmaK = 16;
    static_assert(TileN % kWgmmaK == 0, "AttentionWgmma: TileN must be a multiple of 16");
    static constexpr int kSlices = TileN / kWgmmaK;
    static_assert(kSlices > 0, "AttentionWgmma: TileN must be positive");
    static_assert(kSlices <= 8, "AttentionWgmma: TileN requires more than 8 WGMMA groups");

    using ExecGroup = iro::exec::warpgroup_t<CapT::warpgroup_warps>;
    using AccRecipe = iro::recipe::Accumulate<Recipe>;
    using ScheduleT = axp::kit::detail::select_schedule_t<Schedule, CapT>;
    static constexpr bool kProducerConsumer = std::is_same_v<ScheduleT, axp::intent::schedule::ProducerConsumer>;
    static constexpr bool kBulkSchedule = std::is_same_v<ScheduleT, axp::intent::schedule::BulkSynchronous>;
    template<class Op>
    using ProducerOp = std::conditional_t<
        kProducerConsumer,
        axp::level0::SpecializedOp<axp::level0::role::producer, Op>,
        Op
    >;
    template<class Op>
    using ConsumerOp = std::conditional_t<
        kProducerConsumer,
        axp::level0::SpecializedOp<axp::level0::role::consumer, Op>,
        Op
    >;
    using ScheduleReq = axp::level0::RequireWarpgroupCount<2, CapT::warpgroup_warps>;
    using ScheduleObligations = std::conditional_t<
        kProducerConsumer,
        iro::util::type_list<ScheduleReq>,
        iro::util::type_list<>
    >;
    static constexpr bool kTileSkip = !std::is_same_v<TileSkip, axp::intent::tile_skip::None>;
    using tile_skip_subj = axp::subject::TileSkip;
    template<class Tma, class = void>
    struct tma_coord_traits {
        static constexpr bool valid = false;
        static constexpr bool has_coord1 = false;
        using Coord0Payload = void;
        using Coord0Subj = void;
        using Coord1Payload = void;
        using Coord1Subj = void;
    };
    template<class Tma>
    struct tma_coord_traits<Tma, std::enable_if_t<axp::level2::staging::tma_traits<Tma>::valid>> {
        static constexpr bool valid = true;
        static constexpr bool has_coord1 = axp::level2::staging::tma_traits<Tma>::has_coord1;
        using Coord0Payload = typename axp::level2::staging::tma_traits<Tma>::Coord0Payload;
        using Coord0Subj = typename axp::level2::staging::tma_traits<Tma>::Coord0Subj;
        using Coord1Payload = typename axp::level2::staging::tma_traits<Tma>::Coord1Payload;
        using Coord1Subj = typename axp::level2::staging::tma_traits<Tma>::Coord1Subj;
    };
    template<class Tma>
    struct tma_coord_traits<Tma, std::enable_if_t<axp::level2::staging::tma_multicast_traits<Tma>::valid>> {
        static constexpr bool valid = true;
        static constexpr bool has_coord1 = axp::level2::staging::tma_multicast_traits<Tma>::has_coord1;
        using Coord0Payload = typename axp::level2::staging::tma_multicast_traits<Tma>::Coord0Payload;
        using Coord0Subj = typename axp::level2::staging::tma_multicast_traits<Tma>::Coord0Subj;
        using Coord1Payload = typename axp::level2::staging::tma_multicast_traits<Tma>::Coord1Payload;
        using Coord1Subj = typename axp::level2::staging::tma_multicast_traits<Tma>::Coord1Subj;
    };
    using ScalePayload = iro::contract::ScalarDesc<iro::elem::f32, iro::dist::replicated>;
    using SwizzleAtomQ = std::conditional_t<
        std::is_same_v<LoadModeQ, axp::intent::load_mode::AsyncPrefetch>,
        axp::kit::detail::select_swizzle_t<MemoryPatternQ, ElemA, HeadDim, CapT, true>,
        axp::swizzle::None>;
    using SwizzleAtomK = std::conditional_t<
        std::is_same_v<LoadModeK, axp::intent::load_mode::AsyncPrefetch>,
        axp::kit::detail::select_swizzle_t<MemoryPatternK, ElemB, HeadDim, CapT, false>,
        axp::swizzle::None>;
    using SwizzleAtomV = std::conditional_t<
        std::is_same_v<LoadModeV, axp::intent::load_mode::AsyncPrefetch>,
        axp::kit::detail::select_swizzle_t<MemoryPatternV, ElemB, TileN, CapT, false>,
        axp::swizzle::None>;
    using StageSwizzleQ = std::conditional_t<std::is_same_v<SwizzleAtomQ, axp::swizzle::None>, void, SwizzleAtomQ>;
    using StageSwizzleK = std::conditional_t<std::is_same_v<SwizzleAtomK, axp::swizzle::None>, void, SwizzleAtomK>;
    using StageSwizzleV = std::conditional_t<std::is_same_v<SwizzleAtomV, axp::swizzle::None>, void, SwizzleAtomV>;
    static constexpr bool kSwizzleQ = !std::is_same_v<SwizzleAtomQ, axp::swizzle::None>;
    static constexpr bool kSwizzleK = !std::is_same_v<SwizzleAtomK, axp::swizzle::None>;
    static constexpr bool kSwizzleV = !std::is_same_v<SwizzleAtomV, axp::swizzle::None>;
    static constexpr int kQSmemAlign = kSwizzleQ ? (1 << (SwizzleAtomQ::B_bits + SwizzleAtomQ::S_bits)) : 16;
    static constexpr int kKSmemAlign = kSwizzleK ? (1 << (SwizzleAtomK::B_bits + SwizzleAtomK::S_bits)) : 16;
    static constexpr int kVSmemAlign = kSwizzleV ? (1 << (SwizzleAtomV::B_bits + SwizzleAtomV::S_bits)) : 16;

    struct wgmma_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.wgmma"); };
    struct qk_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.qk_frag"); };
    struct qk_raw_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.qk_raw"); };
    struct weights_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.weights_frag"); };
    struct tile_state_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.tile_state"); };
    struct combined_state_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.combined_state"); };
    struct scale_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.scale"); };
    struct scale_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.scale_frag"); };
    struct weights_scaled_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.weights_scaled"); };
    struct weights_f32_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.weights_f32"); };
    struct weights_f16_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.weights_f16"); };
    struct pv_raw_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.pv_raw"); };
    struct pv_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.pv_frag"); };
    struct q_desc_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.q_desc"); };
    struct k_desc_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.k_desc"); };
    struct p_desc_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.p_desc"); };
    struct v_desc_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.v_desc"); };
    struct q_pipe_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.q_pipe"); };
    struct k_pipe_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.k_pipe"); };
    struct v_pipe_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.attn.v_pipe"); };

    using wgmma_subj = iro::contract::subject::indexed<wgmma_tag, 0>;
    using qk_frag_subj = iro::contract::subject::indexed<qk_frag_tag, 0>;
    using qk_raw_subj = iro::contract::subject::indexed<qk_raw_tag, 0>;
    using weights_frag_subj = iro::contract::subject::indexed<weights_frag_tag, 0>;
    using tile_state_subj = iro::contract::subject::indexed<tile_state_tag, 0>;
    using combined_state_subj = iro::contract::subject::indexed<combined_state_tag, 0>;
    using scale_subj = iro::contract::subject::indexed<scale_tag, 0>;
    using scale_frag_subj = iro::contract::subject::indexed<scale_frag_tag, 0>;
    using weights_scaled_subj = iro::contract::subject::indexed<weights_scaled_tag, 0>;
    using weights_f32_subj = iro::contract::subject::indexed<weights_f32_tag, 0>;
    using weights_f16_subj = iro::contract::subject::indexed<weights_f16_tag, 0>;
    using q_desc_subj = iro::contract::subject::indexed<q_desc_tag, 0>;
    using k_desc_subj = iro::contract::subject::indexed<k_desc_tag, 0>;
    template<int I>
    using pv_raw_subj = iro::contract::subject::indexed<pv_raw_tag, I>;
    template<int I>
    using pv_frag_subj = iro::contract::subject::indexed<pv_frag_tag, I>;
    template<int I>
    using p_desc_subj = iro::contract::subject::indexed<p_desc_tag, I>;
    template<int I>
    using v_desc_subj = iro::contract::subject::indexed<v_desc_tag, I>;

    using QTileG = iro::contract::Tile<
        iro::contract::Shape<TileM, HeadDim>,
        ElemA,
        iro::contract::layout::RowMajor<HeadDim>,
        iro::contract::space::global,
        iro::contract::Align<16>
    >;

    using KTileG = iro::contract::Tile<
        iro::contract::Shape<HeadDim, TileN>,
        ElemB,
        iro::contract::layout::ColMajor<HeadDim>,
        iro::contract::space::global,
        iro::contract::Align<16>
    >;

    using VTileG = iro::contract::Tile<
        iro::contract::Shape<TileN, HeadDim>,
        ElemB,
        iro::contract::layout::ColMajor<TileN>,
        iro::contract::space::global,
        iro::contract::Align<16>
    >;

    using AutoQTma = axp::kit::detail::select_tma_t<LoadModeQ, CapT, QTileG, QSubj, q_pipe_tag>;
    using AutoKTma = axp::kit::detail::select_tma_t<LoadModeK, CapT, KTileG, KSubj, k_pipe_tag>;
    using AutoVTma = axp::kit::detail::select_tma_t<LoadModeV, CapT, VTileG, VSubj, v_pipe_tag>;
    using QTmaT = std::conditional_t<!std::is_void_v<QTma>, QTma, AutoQTma>;
    using KTmaT = std::conditional_t<!std::is_void_v<KTma>, KTma, AutoKTma>;
    using VTmaT = std::conditional_t<!std::is_void_v<VTma>, VTma, AutoVTma>;
    static constexpr bool kHasTmaQ =
        axp::level2::staging::tma_traits<QTmaT>::valid || axp::level2::staging::tma_multicast_traits<QTmaT>::valid;
    static constexpr bool kHasTmaK =
        axp::level2::staging::tma_traits<KTmaT>::valid || axp::level2::staging::tma_multicast_traits<KTmaT>::valid;
    static constexpr bool kHasTmaV =
        axp::level2::staging::tma_traits<VTmaT>::valid || axp::level2::staging::tma_multicast_traits<VTmaT>::valid;
    static constexpr bool kStreamingQ = axp::level2::staging::streaming_traits<QTmaT>::valid;
    static constexpr bool kStreamingK = axp::level2::staging::streaming_traits<KTmaT>::valid;
    static constexpr bool kStreamingV = axp::level2::staging::streaming_traits<VTmaT>::valid;
    using QTmaCoords = tma_coord_traits<QTmaT>;
    using KTmaCoords = tma_coord_traits<KTmaT>;
    static_assert(!kTileSkip || (QTmaCoords::valid && KTmaCoords::valid),
                  "AttentionWgmma: Causal tile skip requires TMA coords for Q/K");
    using QCoordPayload = typename QTmaCoords::Coord0Payload;
    using QCoordSubj = typename QTmaCoords::Coord0Subj;
    using KCoordPayload = std::conditional_t<
        KTmaCoords::has_coord1,
        typename KTmaCoords::Coord1Payload,
        typename KTmaCoords::Coord0Payload
    >;
    using KCoordSubj = std::conditional_t<
        KTmaCoords::has_coord1,
        typename KTmaCoords::Coord1Subj,
        typename KTmaCoords::Coord0Subj
    >;
    static_assert(!kSwizzleQ || kHasTmaQ,
                  "AttentionWgmma: SwizzleAtom requires TMA staging for Q");
    static_assert(!kSwizzleK || kHasTmaK,
                  "AttentionWgmma: SwizzleAtom requires TMA staging for K");
    static_assert(!kSwizzleV || kHasTmaV,
                  "AttentionWgmma: SwizzleAtom requires TMA staging for V");

    using QTileSLayout = std::conditional_t<
        kSwizzleQ,
        iro::contract::layout::Swizzled<HeadDim, SwizzleAtomQ::B, SwizzleAtomQ::S>,
        iro::contract::layout::RowMajor<HeadDim>
    >;
    using QTileS = iro::contract::Tile<
        iro::contract::Shape<TileM, HeadDim>,
        ElemA,
        QTileSLayout,
        iro::contract::space::shared,
        iro::contract::Align<kQSmemAlign>
    >;

    using KTileSLayout = std::conditional_t<
        kSwizzleK,
        iro::contract::layout::SwizzledColMajor<HeadDim, SwizzleAtomK::B, SwizzleAtomK::S>,
        iro::contract::layout::ColMajor<HeadDim>
    >;
    using KTileS = iro::contract::Tile<
        iro::contract::Shape<HeadDim, TileN>,
        ElemB,
        KTileSLayout,
        iro::contract::space::shared,
        iro::contract::Align<kKSmemAlign>
    >;

    using VTileSLayout = std::conditional_t<
        kSwizzleV,
        iro::contract::layout::SwizzledColMajor<TileN, SwizzleAtomV::B, SwizzleAtomV::S>,
        iro::contract::layout::ColMajor<TileN>
    >;
    using VTileS = iro::contract::Tile<
        iro::contract::Shape<TileN, HeadDim>,
        ElemB,
        VTileSLayout,
        iro::contract::space::shared,
        iro::contract::Align<kVSmemAlign>
    >;

    using PTileF32 = iro::contract::Tile<
        iro::contract::Shape<TileM, TileN>,
        iro::elem::f32,
        iro::contract::layout::RowMajor<TileN>,
        iro::contract::space::shared,
        iro::contract::Align<16>
    >;

    using PTileF16 = iro::contract::Tile<
        iro::contract::Shape<TileM, TileN>,
        ElemA,
        iro::contract::layout::RowMajor<TileN>,
        iro::contract::space::shared,
        iro::contract::Align<16>
    >;

    using PTileSlice = iro::contract::Tile<
        iro::contract::Shape<TileM, kWgmmaK>,
        ElemA,
        typename PTileF16::layout,
        iro::contract::space::shared,
        typename PTileF16::align
    >;

    using VTileSlice = iro::contract::Tile<
        iro::contract::Shape<kWgmmaK, HeadDim>,
        ElemB,
        VTileSLayout,
        iro::contract::space::shared,
        typename VTileS::align
    >;

    using PipeQ = iro::contract::res::smem_pipeline<
        q_pipe_tag, Stages, QTileS::bytes, QTileS::align::bytes
    >;
    using PipeK = iro::contract::res::smem_pipeline<
        k_pipe_tag, Stages, KTileS::bytes, KTileS::align::bytes
    >;
    using PipeV = iro::contract::res::smem_pipeline<
        v_pipe_tag, Stages, VTileS::bytes, VTileS::align::bytes
    >;

    using SlotQ = iro::contract::res::slot_subject<PipeQ, 0>;
    using SlotK = iro::contract::res::slot_subject<PipeK, 0>;
    using SlotV = iro::contract::res::slot_subject<PipeV, 0>;

    using TileInQ = axp::level0::TileBoundaryIn<
        Recipe, QTileG, QSubj, iro::exec::block, iro::token::lifetime::block
    >;
    using TileInK = axp::level0::TileBoundaryIn<
        Recipe, KTileG, KSubj, iro::exec::block, iro::token::lifetime::block
    >;
    using TileInV = axp::level0::TileBoundaryIn<
        Recipe, VTileG, VSubj, iro::exec::block, iro::token::lifetime::block
    >;

    using QStageIssueExec = axp::level2::staging::tma_issue_exec_group_t<QTmaT>;
    using KStageIssueExec = axp::level2::staging::tma_issue_exec_group_t<KTmaT>;
    using VStageIssueExec = axp::level2::staging::tma_issue_exec_group_t<VTmaT>;

    using StageQ = axp::level2::staging::StageGmemToSmem<
        Recipe, QTileG, QTileS, QSubj, q_pipe_tag, SlotQ,
        iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleQ, QTmaT,
        QStageIssueExec, CapT
    >;
    using StageK = axp::level2::staging::StageGmemToSmem<
        Recipe, KTileG, KTileS, KSubj, k_pipe_tag, SlotK,
        iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleK, KTmaT,
        KStageIssueExec, CapT
    >;
    using StageV = axp::level2::staging::StageGmemToSmem<
        Recipe, VTileG, VTileS, VSubj, v_pipe_tag, SlotV,
        iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleV, VTmaT,
        VStageIssueExec, CapT
    >;

    struct IssueQImpl : StageQ::Issue {};
    struct IssueKImpl : StageK::Issue {};
    struct IssueVImpl : StageV::Issue {};
    struct WaitQImpl : StageQ::Wait {};
    struct WaitKImpl : StageK::Wait {};
    struct WaitVImpl : StageV::Wait {};
    struct MarkQImpl : StageQ::Mark {};
    struct MarkKImpl : StageK::Mark {};
    struct MarkVImpl : StageV::Mark {};
    struct ReleaseQImpl : StageQ::Release {};
    struct ReleaseKImpl : StageK::Release {};
    struct ReleaseVImpl : StageV::Release {};
    using IssueQ = ProducerOp<IssueQImpl>;
    using IssueK = ProducerOp<IssueKImpl>;
    using IssueV = ProducerOp<IssueVImpl>;
    using WaitQ = ConsumerOp<WaitQImpl>;
    using WaitK = ConsumerOp<WaitKImpl>;
    using WaitV = ConsumerOp<WaitVImpl>;
    using MarkQ = ConsumerOp<MarkQImpl>;
    using MarkK = ConsumerOp<MarkKImpl>;
    using MarkV = ConsumerOp<MarkVImpl>;
    using ReleaseQ = ConsumerOp<ReleaseQImpl>;
    using ReleaseK = ConsumerOp<ReleaseKImpl>;
    using ReleaseV = ConsumerOp<ReleaseVImpl>;

    using FenceQImpl = axp::level0::TileFence<
        Recipe, QTileS, SlotQ, iro::exec::block
    >;
    using FenceKImpl = axp::level0::TileFence<
        Recipe, KTileS, SlotK, iro::exec::block
    >;
    using FenceVImpl = axp::level0::TileFence<
        Recipe, VTileS, SlotV, iro::exec::block
    >;
    using FenceQ = ConsumerOp<FenceQImpl>;
    using FenceK = ConsumerOp<FenceKImpl>;
    using FenceV = ConsumerOp<FenceVImpl>;

    using TileSkipHookOp = TileSkipHook<Recipe, tile_skip_subj, ExecGroup>;
    template<bool Enable, class = void>
    struct tile_skip_hook {
        using obligations = iro::util::type_list<>;
        using edges = iro::util::type_list<>;
    };
    template<bool Enable>
    struct tile_skip_hook<Enable, std::enable_if_t<Enable>> {
        using obligations = iro::util::type_list<TileSkipHookOp>;
        using edges = iro::util::type_list<>;
    };

    template<bool Enable, class = void>
    struct bulk_fence_q {
        using edges = iro::util::type_list<>;
        using obligations = iro::util::type_list<>;
        using source = WaitQ;
    };

    template<bool Enable>
    struct bulk_fence_q<Enable, std::enable_if_t<Enable>> {
        using edges = iro::util::type_list<
            iro::compose::Edge<iro::compose::out_port_ref<WaitQ, 0>, iro::compose::in_port_ref<FenceQ, 0>>
        >;
        using obligations = iro::util::type_list<FenceQ>;
        using source = FenceQ;
    };

    template<bool Enable, class = void>
    struct bulk_fence_k {
        using edges = iro::util::type_list<>;
        using obligations = iro::util::type_list<>;
        using source = WaitK;
    };

    template<bool Enable>
    struct bulk_fence_k<Enable, std::enable_if_t<Enable>> {
        using edges = iro::util::type_list<
            iro::compose::Edge<iro::compose::out_port_ref<WaitK, 0>, iro::compose::in_port_ref<FenceK, 0>>
        >;
        using obligations = iro::util::type_list<FenceK>;
        using source = FenceK;
    };

    template<bool Enable, class = void>
    struct bulk_fence_v {
        using edges = iro::util::type_list<>;
        using obligations = iro::util::type_list<>;
        using source = WaitV;
    };

    template<bool Enable>
    struct bulk_fence_v<Enable, std::enable_if_t<Enable>> {
        using edges = iro::util::type_list<
            iro::compose::Edge<iro::compose::out_port_ref<WaitV, 0>, iro::compose::in_port_ref<FenceV, 0>>
        >;
        using obligations = iro::util::type_list<FenceV>;
        using source = FenceV;
    };

    using QDescSource = typename bulk_fence_q<kBulkSchedule>::source;
    using KDescSource = typename bulk_fence_k<kBulkSchedule>::source;
    using VDescSource = typename bulk_fence_v<kBulkSchedule>::source;


    using QKShape = axp::protocol::compute::MmaShape<
        TileM, TileN, HeadDim,
        ElemA, ElemB, typename Recipe::acc,
        typename QTileS::layout, typename KTileS::layout
    >;

    using PVShape = axp::protocol::compute::MmaShape<
        TileM, HeadDim, kWgmmaK,
        ElemA, ElemB, typename Recipe::acc,
        typename PTileSlice::layout, typename VTileSlice::layout
    >;

    using QKFrag = iro::contract::FragmentDesc<
        iro::contract::Shape<TileM, TileN>,
        typename Recipe::acc,
        iro::dist::accumulator,
        TileN / 2
    >;

    using OFrag = iro::contract::FragmentDesc<
        iro::contract::Shape<TileM, HeadDim>,
        typename Recipe::acc,
        iro::dist::accumulator,
        HeadDim / 2
    >;
    using MaskPayload = iro::contract::MaskDesc<QKFrag::count, typename QKFrag::dist>;
    using PredPayload = typename TileSkipHookOp::PredPayload;
    using SoftmaxScalar = iro::contract::ScalarDesc<typename Recipe::acc, typename QKFrag::dist>;

    struct HoldQ : axp::level0::SlotAfter<
        Recipe, SlotQ, iro::exec::block, iro::token::lifetime::block, QTileS::bytes,
        QKFrag, qk_frag_subj, ExecGroup, typename QKFrag::dist
    > {};
    struct HoldK : axp::level0::SlotAfter<
        Recipe, SlotK, iro::exec::block, iro::token::lifetime::block, KTileS::bytes,
        QKFrag, qk_frag_subj, ExecGroup, typename QKFrag::dist
    > {};
    struct HoldV : axp::level0::SlotAfter<
        Recipe, SlotV, iro::exec::block, iro::token::lifetime::block, VTileS::bytes,
        OFrag, AccSubj, ExecGroup, typename OFrag::dist
    > {};

    static constexpr int cast_vec_bytes = Recipe::vec_bytes;
    static constexpr int cast_out_vec_bytes = (cast_vec_bytes * PTileF16::elem::bytes) / PTileF32::elem::bytes;
    static_assert(cast_vec_bytes == 8 || cast_vec_bytes == 16, "AttentionWgmma: Recipe::vec_bytes must be 8 or 16");
    static_assert(cast_out_vec_bytes == 4 || cast_out_vec_bytes == 8 || cast_out_vec_bytes == 16,
                  "AttentionWgmma: cast output vec bytes must be 4/8/16");

    template<class R, class Enable = void>
    struct weight_recipe_impl;

    template<class R>
    struct weight_recipe_impl<R, std::void_t<typename R::scale>> {
        using type = iro::recipe::Precision<
            ElemA,
            typename R::acc,
            typename R::out,
            cast_out_vec_bytes,
            typename R::math,
            typename R::fp8_policy,
            typename R::scale,
            R::scale_vec
        >;
    };

    template<class R>
    struct weight_recipe_impl<R, std::void_t<typename R::scale_a>> {
        using type = iro::recipe::Precision<
            ElemA,
            typename R::acc,
            typename R::out,
            cast_out_vec_bytes,
            typename R::math,
            typename R::fp8_policy,
            typename R::scale_a,
            R::scale_vec_a
        >;
    };

    using WeightRecipe = typename weight_recipe_impl<Recipe>::type;

    using FenceHandleImpl = axp::level2::wgmma::Fence<
        Recipe, wgmma_subj, ExecGroup, iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using FenceHandle = ConsumerOp<FenceHandleImpl>;

    using WgmmaSwizzleQ = SwizzleAtomQ;
    using WgmmaSwizzleK = SwizzleAtomK;
    using WgmmaSwizzleP = axp::swizzle::None;
    using WgmmaSwizzleV = SwizzleAtomV;

    using QDesc = axp::level0::WgmmaSmemDesc<QTileS, SlotQ, WgmmaSwizzleQ>;
    using KDesc = axp::level0::WgmmaSmemDesc<KTileS, SlotK, WgmmaSwizzleK>;
    using PDesc = axp::level0::WgmmaSmemDesc<PTileSlice, weights_f16_subj, WgmmaSwizzleP>;
    using VDesc = axp::level0::WgmmaSmemDesc<VTileSlice, SlotV, WgmmaSwizzleV>;

    struct MakeDescQImpl : axp::level0::MakeDesc<
        Recipe, QTileS, SlotQ, q_desc_subj, ExecGroup, iro::token::lifetime::block, WgmmaSwizzleQ
    > {};
    struct MakeDescKImpl : axp::level0::MakeDesc<
        Recipe, KTileS, SlotK, k_desc_subj, ExecGroup, iro::token::lifetime::block, WgmmaSwizzleK
    > {};
    using MakeDescQ = ConsumerOp<MakeDescQImpl>;
    using MakeDescK = ConsumerOp<MakeDescKImpl>;

    struct QKImpl : axp::level2::Matmul<
        Recipe, QKShape, QDesc, KDesc, QKFrag,
        q_desc_subj, k_desc_subj, qk_raw_subj, ExecGroup, wgmma_subj, CapT
    > {};
    using QK = ConsumerOp<QKImpl>;
    struct CommitQKImpl : axp::level2::wgmma::CommitGroup<
        AccRecipe, wgmma_subj, ExecGroup, 0, iro::util::type_list<>, iro::util::type_list<>, CapT
    > {};
    struct WaitQKImpl : axp::level2::wgmma::WaitGroup<
        AccRecipe, wgmma_subj, ExecGroup, 0, iro::util::type_list<>, iro::util::type_list<>, CapT
    > {};
    struct WaitAccQKImpl : axp::level2::wgmma::WaitAcc<
        AccRecipe, QKFrag, qk_raw_subj, qk_frag_subj, wgmma_subj, ExecGroup, 0,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    > {};
    using CommitQK = ConsumerOp<CommitQKImpl>;
    using WaitQK = ConsumerOp<WaitQKImpl>;
    using WaitAccQK = ConsumerOp<WaitAccQKImpl>;

    using SoftmaxImpl = std::conditional_t<
        kTileSkip,
        axp::level2::attention::WarpSoftmaxStateMasked<
            AccRecipe, QKFrag, MaskPayload, detail::softmax_mask_subj, detail::softmax_neg_inf_subj,
            qk_frag_subj, weights_frag_subj, tile_state_subj, ExecGroup, CapT
        >,
        axp::level2::attention::WarpSoftmaxState<
            AccRecipe, QKFrag, qk_frag_subj, weights_frag_subj, tile_state_subj, ExecGroup, CapT
        >
    >;
    using CombineImpl = axp::level2::attention::SoftmaxStateCombine<
        AccRecipe, ExecGroup, OldStateSubj, tile_state_subj, combined_state_subj
    >;
    using RescaleImpl = axp::level2::attention::RescaleAccumulator<
        AccRecipe, OFrag, AccSubj, OldStateSubj, combined_state_subj, ExecGroup
    >;
    using StateCopyImpl = axp::level2::attention::SoftmaxStateCopy<
        AccRecipe, combined_state_subj, OutStateSubj, ExecGroup, CapT
    >;
    using ScaleImpl = axp::level2::attention::SoftmaxStateScale<
        AccRecipe, tile_state_subj, combined_state_subj, scale_subj, ExecGroup, CapT
    >;
    using ScaleFragImpl = axp::level1::FragmentBroadcast<
        AccRecipe, QKFrag, ScalePayload, scale_subj, scale_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using ScaleWeightsImpl = axp::level0::Mul<
        AccRecipe, QKFrag, weights_frag_subj, scale_frag_subj, weights_scaled_subj, ExecGroup
    >;
    using Softmax = ConsumerOp<SoftmaxImpl>;
    using Combine = ConsumerOp<CombineImpl>;
    using Rescale = ConsumerOp<RescaleImpl>;
    using StateCopy = ConsumerOp<StateCopyImpl>;
    using Scale = ConsumerOp<ScaleImpl>;
    using ScaleFrag = ConsumerOp<ScaleFragImpl>;
    using ScaleWeights = ConsumerOp<ScaleWeightsImpl>;

    template<bool Enable, class = void>
    struct tile_skip_mask {
        using obligations = iro::util::type_list<>;
        using edges = iro::util::type_list<>;
    };
    template<bool Enable>
    struct tile_skip_mask<Enable, std::enable_if_t<Enable>> {
        struct neg_inf_pattern {
            static constexpr typename SoftmaxScalar::elem::storage_t value =
                -std::numeric_limits<typename SoftmaxScalar::elem::storage_t>::infinity();
        };
        struct CausalMaskImpl : axp::level1::CausalMask<
            AccRecipe, MaskPayload, PredPayload,
            QCoordPayload, KCoordPayload,
            QCoordSubj, KCoordSubj,
            detail::softmax_mask_subj, tile_skip_subj,
            ExecGroup, TileM, TileN, CapT
        > {};
        struct NegInfConstImpl : axp::level0::ScalarConst<
            AccRecipe, SoftmaxScalar, detail::softmax_neg_inf_subj, ExecGroup, neg_inf_pattern
        > {};
        using CausalMask = ConsumerOp<CausalMaskImpl>;
        using NegInfConst = ConsumerOp<NegInfConstImpl>;
        using obligations = iro::util::type_list<CausalMask, NegInfConst>;
        using edges = iro::util::type_list<
            iro::compose::Edge<iro::compose::out_port_ref<NegInfConst, 0>, iro::compose::in_port_ref<Softmax, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<CausalMask, 0>, iro::compose::in_port_ref<Softmax, 2>>,
            iro::compose::Edge<iro::compose::out_port_ref<CausalMask, 1>, iro::compose::in_port_ref<TileSkipHookOp, 0>>
        >;
    };

    using softmax_input_edge_t = std::conditional_t<
        kTileSkip,
        iro::compose::Edge<
            iro::compose::out_port_ref<WaitAccQK, 0>,
            iro::compose::in_port_ref<Softmax, 1>
        >,
        iro::compose::Edge<
            iro::compose::out_port_ref<WaitAccQK, 0>,
            iro::compose::in_port_ref<Softmax, 0>
        >
    >;

    using StoreWeightsImpl = axp::level0::FragmentToSharedTile<
        AccRecipe, QKFrag, PTileF32, weights_scaled_subj, weights_f32_subj, ExecGroup, iro::token::lifetime::warpgroup
    >;
    using WeightsFenceImpl = axp::level0::TileFence<
        AccRecipe, PTileF32, weights_f32_subj, ExecGroup
    >;
    using CastWeightsImpl = axp::level0::CastTile<
        AccRecipe, WeightRecipe, PTileF32, PTileF16,
        weights_f32_subj, weights_f16_subj, ExecGroup, cast_vec_bytes
    >;
    using StoreWeights = ConsumerOp<StoreWeightsImpl>;
    using WeightsFence = ConsumerOp<WeightsFenceImpl>;
    using CastWeights = ConsumerOp<CastWeightsImpl>;

    template<int SliceIdx>
    struct pv_slice {
        static_assert(SliceIdx >= 0 && SliceIdx < kSlices, "AttentionWgmma: PV slice index out of range");
        using p_desc_subj_t = p_desc_subj<SliceIdx>;
        using v_desc_subj_t = v_desc_subj<SliceIdx>;
        using pv_raw_subj_t = pv_raw_subj<SliceIdx>;
        using pv_frag_subj_t = pv_frag_subj<SliceIdx>;

        struct MakeDescPImpl : axp::level0::MakeDescSliceReady<
            Recipe, PTileF16, PTileSlice, weights_f16_subj, p_desc_subj_t,
            ExecGroup, iro::token::lifetime::warpgroup, WgmmaSwizzleP,
            0, SliceIdx * kWgmmaK
        > {};
        struct MakeDescVImpl : axp::level0::MakeDescSlice<
            Recipe, VTileS, VTileSlice, SlotV, v_desc_subj_t,
            ExecGroup, iro::token::lifetime::block, WgmmaSwizzleV,
            SliceIdx * kWgmmaK, 0
        > {};
        using MakeDescP = ConsumerOp<MakeDescPImpl>;
        using MakeDescV = ConsumerOp<MakeDescVImpl>;

        struct PVImpl : axp::level2::Matmul<
            Recipe, PVShape, PDesc, VDesc, OFrag,
            p_desc_subj_t, v_desc_subj_t, pv_raw_subj_t, ExecGroup, wgmma_subj, CapT
        > {};
        using PV = ConsumerOp<PVImpl>;
        struct CommitPVImpl : axp::level2::wgmma::CommitGroup<
            AccRecipe, wgmma_subj, ExecGroup, SliceIdx, iro::util::type_list<>, iro::util::type_list<>, CapT
        > {};
        struct WaitPVImpl : axp::level2::wgmma::WaitGroup<
            AccRecipe, wgmma_subj, ExecGroup, SliceIdx, iro::util::type_list<>, iro::util::type_list<>, CapT
        > {};
        struct WaitAccPVImpl : axp::level2::wgmma::WaitAcc<
            AccRecipe, OFrag, pv_raw_subj_t, pv_frag_subj_t, wgmma_subj, ExecGroup, SliceIdx,
            iro::util::type_list<>, iro::util::type_list<>, CapT
        > {};
        using CommitPV = ConsumerOp<CommitPVImpl>;
        using WaitPV = ConsumerOp<WaitPVImpl>;
        using WaitAccPV = ConsumerOp<WaitAccPVImpl>;

        using AddImpl = axp::level0::Add<
            AccRecipe, OFrag, AccSubj, pv_frag_subj_t, AccSubj, ExecGroup
        >;
        using Add = ConsumerOp<AddImpl>;

        using obligations = iro::util::type_list<
            MakeDescP,
            MakeDescV,
            PV,
            CommitPV,
            WaitPV,
            WaitAccPV,
            Add
        >;

        using edges = iro::util::type_list<
            iro::compose::Edge<iro::compose::out_port_ref<CastWeights, 0>, iro::compose::in_port_ref<MakeDescP, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<VDescSource, 0>, iro::compose::in_port_ref<MakeDescV, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<FenceHandle, 0>, iro::compose::in_port_ref<PV, 2>>,
            iro::compose::Edge<iro::compose::out_port_ref<MakeDescP, 0>, iro::compose::in_port_ref<PV, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<MakeDescV, 0>, iro::compose::in_port_ref<PV, 1>>,
            iro::compose::Edge<iro::compose::out_port_ref<PV, 1>, iro::compose::in_port_ref<CommitPV, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<CommitPV, 0>, iro::compose::in_port_ref<WaitPV, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<PV, 0>, iro::compose::in_port_ref<WaitAccPV, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<WaitPV, 0>, iro::compose::in_port_ref<WaitAccPV, 1>>,
            iro::compose::Edge<iro::compose::out_port_ref<WaitAccPV, 0>, iro::compose::in_port_ref<Add, 1>>
        >;
    };

    template<int SliceIdx, class = void>
    struct pv_slices;

    template<int SliceIdx>
    struct pv_slices<SliceIdx, std::enable_if_t<(SliceIdx > 0)>> {
        using prev = pv_slices<SliceIdx - 1>;
        using slice = pv_slice<SliceIdx>;
        using obligations = iro::util::concat_t<typename prev::obligations, typename slice::obligations>;
        using edges = iro::util::concat_t<
            typename prev::edges,
            typename slice::edges,
            iro::util::type_list<
                iro::compose::Edge<
                    iro::compose::out_port_ref<typename pv_slice<SliceIdx - 1>::Add, 0>,
                    iro::compose::in_port_ref<typename pv_slice<SliceIdx>::Add, 0>
                >
            >
        >;
    };

    template<int SliceIdx>
    struct pv_slices<SliceIdx, std::enable_if_t<(SliceIdx == 0)>> {
        using slice = pv_slice<0>;
        using obligations = typename slice::obligations;
        using edges = typename slice::edges;
    };

    using pv_ops = pv_slices<kSlices - 1>;
    using AddFirst = typename pv_slice<0>::Add;
    using AddLast = typename pv_slice<kSlices - 1>::Add;

    template<bool Enable, class = void>
    struct boundary_tilein_q {
        using edges = iro::util::type_list<>;
        using obligations = iro::util::type_list<>;
    };

    template<bool Enable>
    struct boundary_tilein_q<Enable, std::enable_if_t<Enable>> {
        using edges = iro::util::type_list<
            iro::compose::Edge<iro::compose::out_port_ref<TileInQ, 0>, iro::compose::in_port_ref<IssueQ, 0>>
        >;
        using obligations = iro::util::type_list<TileInQ>;
    };

    template<bool Enable, class = void>
    struct boundary_tilein_k {
        using edges = iro::util::type_list<>;
        using obligations = iro::util::type_list<>;
    };

    template<bool Enable>
    struct boundary_tilein_k<Enable, std::enable_if_t<Enable>> {
        using edges = iro::util::type_list<
            iro::compose::Edge<iro::compose::out_port_ref<TileInK, 0>, iro::compose::in_port_ref<IssueK, 0>>
        >;
        using obligations = iro::util::type_list<TileInK>;
    };

    template<bool Enable, class = void>
    struct boundary_tilein_v {
        using edges = iro::util::type_list<>;
        using obligations = iro::util::type_list<>;
    };

    template<bool Enable>
    struct boundary_tilein_v<Enable, std::enable_if_t<Enable>> {
        using edges = iro::util::type_list<
            iro::compose::Edge<iro::compose::out_port_ref<TileInV, 0>, iro::compose::in_port_ref<IssueV, 0>>
        >;
        using obligations = iro::util::type_list<TileInV>;
    };

    using boundary_edges = iro::util::concat_t<
        iro::util::concat_t<
            typename boundary_tilein_q<!kHasTmaQ>::edges,
            typename boundary_tilein_k<!kHasTmaK>::edges
        >,
        typename boundary_tilein_v<!kHasTmaV>::edges
    >;

    using boundary_obligations = iro::util::concat_t<
        iro::util::concat_t<
            typename boundary_tilein_q<!kHasTmaQ>::obligations,
            typename boundary_tilein_k<!kHasTmaK>::obligations
        >,
        typename boundary_tilein_v<!kHasTmaV>::obligations
    >;

    using core_obligations = iro::util::concat_t<
        ScheduleObligations,
        typename tile_skip_hook<kTileSkip>::obligations,
        typename tile_skip_mask<kTileSkip>::obligations,
        iro::util::type_list<
            IssueQ,
            IssueK,
            IssueV,
            WaitQ,
            WaitK,
            WaitV,
            FenceHandle,
            MakeDescQ,
            MakeDescK,
            QK,
            CommitQK,
            WaitQK,
            WaitAccQK,
            Softmax,
            Combine,
            Rescale,
            StateCopy,
            Scale,
            ScaleFrag,
            ScaleWeights,
            StoreWeights,
            WeightsFence,
            CastWeights
        >,
        typename bulk_fence_q<kBulkSchedule>::obligations,
        typename bulk_fence_k<kBulkSchedule>::obligations,
        typename bulk_fence_v<kBulkSchedule>::obligations,
        typename pv_ops::obligations,
        iro::util::type_list<
            HoldQ,
            HoldK,
            HoldV,
            MarkQ,
            MarkK,
            MarkV,
            ReleaseQ,
            ReleaseK,
            ReleaseV
        >
    >;
    using obligations = iro::util::concat_t<core_obligations, boundary_obligations>;

    using stage_q_edges = detail::stage_issue_wait_edges_t<IssueQ, WaitQ, kHasTmaQ, kStreamingQ>;
    using stage_k_edges = detail::stage_issue_wait_edges_t<IssueK, WaitK, kHasTmaK, kStreamingK>;
    using stage_v_edges = detail::stage_issue_wait_edges_t<IssueV, WaitV, kHasTmaV, kStreamingV>;

    using edges = iro::util::concat_t<
        typename tile_skip_hook<kTileSkip>::edges,
        typename tile_skip_mask<kTileSkip>::edges,
        boundary_edges,
        stage_q_edges,
        stage_k_edges,
        stage_v_edges,
        typename bulk_fence_q<kBulkSchedule>::edges,
        typename bulk_fence_k<kBulkSchedule>::edges,
        typename bulk_fence_v<kBulkSchedule>::edges,
        typename pv_ops::edges,
        iro::util::type_list<
            iro::compose::Edge<iro::compose::out_port_ref<QDescSource, 0>, iro::compose::in_port_ref<MakeDescQ, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<KDescSource, 0>, iro::compose::in_port_ref<MakeDescK, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<WaitQ, 1>, iro::compose::in_port_ref<HoldQ, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<WaitK, 1>, iro::compose::in_port_ref<HoldK, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<WaitV, 1>, iro::compose::in_port_ref<HoldV, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<WaitAccQK, 0>, iro::compose::in_port_ref<HoldQ, 1>>,
            iro::compose::Edge<iro::compose::out_port_ref<WaitAccQK, 0>, iro::compose::in_port_ref<HoldK, 1>>,
            iro::compose::Edge<iro::compose::out_port_ref<AddLast, 0>, iro::compose::in_port_ref<HoldV, 1>>,
            iro::compose::Edge<iro::compose::out_port_ref<HoldQ, 0>, iro::compose::in_port_ref<MarkQ, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<HoldK, 0>, iro::compose::in_port_ref<MarkK, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<HoldV, 0>, iro::compose::in_port_ref<MarkV, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<MarkQ, 0>, iro::compose::in_port_ref<ReleaseQ, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<MarkK, 0>, iro::compose::in_port_ref<ReleaseK, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<MarkV, 0>, iro::compose::in_port_ref<ReleaseV, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<FenceHandle, 0>, iro::compose::in_port_ref<QK, 2>>,
            iro::compose::Edge<iro::compose::out_port_ref<MakeDescQ, 0>, iro::compose::in_port_ref<QK, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<MakeDescK, 0>, iro::compose::in_port_ref<QK, 1>>,
            iro::compose::Edge<iro::compose::out_port_ref<QK, 1>, iro::compose::in_port_ref<CommitQK, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<CommitQK, 0>, iro::compose::in_port_ref<WaitQK, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<QK, 0>, iro::compose::in_port_ref<WaitAccQK, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<WaitQK, 0>, iro::compose::in_port_ref<WaitAccQK, 1>>,
            softmax_input_edge_t,
            iro::compose::Edge<iro::compose::out_port_ref<Softmax, 1>, iro::compose::in_port_ref<Combine, 1>>,
            iro::compose::Edge<iro::compose::out_port_ref<Softmax, 1>, iro::compose::in_port_ref<Scale, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<Combine, 0>, iro::compose::in_port_ref<Rescale, 2>>,
            iro::compose::Edge<iro::compose::out_port_ref<Combine, 0>, iro::compose::in_port_ref<Scale, 1>>,
            iro::compose::Edge<iro::compose::out_port_ref<Combine, 0>, iro::compose::in_port_ref<StateCopy, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<Scale, 0>, iro::compose::in_port_ref<ScaleFrag, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<ScaleFrag, 0>, iro::compose::in_port_ref<ScaleWeights, 1>>,
            iro::compose::Edge<iro::compose::out_port_ref<Softmax, 0>, iro::compose::in_port_ref<ScaleWeights, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<ScaleWeights, 0>, iro::compose::in_port_ref<StoreWeights, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<StoreWeights, 0>, iro::compose::in_port_ref<WeightsFence, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<WeightsFence, 0>, iro::compose::in_port_ref<CastWeights, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<Rescale, 0>, iro::compose::in_port_ref<AddFirst, 0>>
        >
    >;

    using type = axp::level2::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, int TileM, int TileN, int HeadDim, int Stages,
         class QSubj, class KSubj, class VSubj,
         class AccSubj, class OldStateSubj, class OutStateSubj,
         class MemoryPatternQ = axp::intent::memory_pattern::Optimized,
         class MemoryPatternK = axp::intent::memory_pattern::Optimized,
         class MemoryPatternV = axp::intent::memory_pattern::Optimized,
         class LoadModeQ = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeK = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeV = axp::intent::load_mode::AsyncPrefetch,
         class Schedule = axp::intent::schedule::Pipelined,
         class TileSkip = axp::intent::tile_skip::None,
         class QTma = void, class KTma = void, class VTma = void,
         class CapT = axp::target_cap>
using AttentionWgmma = registry::Select<registry::AttentionWgmmaPattern<
    Recipe, TileM, TileN, HeadDim, Stages,
    QSubj, KSubj, VSubj, AccSubj, OldStateSubj, OutStateSubj,
    MemoryPatternQ, MemoryPatternK, MemoryPatternV,
    LoadModeQ, LoadModeK, LoadModeV,
    Schedule,
    TileSkip,
    QTma, KTma, VTma>, CapT>;

} // namespace axp::level2::attention

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level2::registry {

template<class Recipe, class ExecGroup, class ASubj, class BSubj, class OutSubj, class Cap>
struct resolve_impl<SoftmaxStateCombinePattern<Recipe, ExecGroup, ASubj, BSubj, OutSubj>, Cap,
                    std::enable_if_t<axp::level2::attention::is_supported_exec<ExecGroup>::value>> {
    static constexpr bool supported = true;
    using type = axp::level2::attention::CombineSoftmaxStateF32<Recipe, ExecGroup, ASubj, BSubj, OutSubj>;
};

template<class Recipe, class Frag, class InSubj, class OutSubj, class StateSubj, class ExecGroup, class Cap>
struct resolve_impl<WarpSoftmaxStatePattern<Recipe, Frag, InSubj, OutSubj, StateSubj, ExecGroup>, Cap,
                    std::enable_if_t<std::is_same_v<ExecGroup, iro::exec::warp>>> {
    static constexpr bool supported = true;
    using type = typename axp::level2::attention::WarpSoftmaxStateF32<
        Recipe, Frag, InSubj, OutSubj, StateSubj, ExecGroup, Cap
    >::type;
};

template<class Recipe, class Frag, class InSubj, class OutSubj, class StateSubj, class ExecGroup, class Cap>
struct resolve_impl<WarpSoftmaxStatePattern<Recipe, Frag, InSubj, OutSubj, StateSubj, ExecGroup>, Cap,
                    std::enable_if_t<iro::exec::is_warpgroup_v<ExecGroup>>> {
    static constexpr bool supported = true;
    using type = typename axp::level2::attention::WarpgroupSoftmaxStateF32<
        Recipe, Frag, InSubj, OutSubj, StateSubj, ExecGroup, Cap
    >::type;
};

template<class Recipe, class Frag, class MaskPayload, class MaskSubj, class NegInfSubj,
         class InSubj, class OutSubj, class StateSubj, class ExecGroup, class Cap>
struct resolve_impl<WarpSoftmaxStateMaskedPattern<
    Recipe, Frag, MaskPayload, MaskSubj, NegInfSubj, InSubj, OutSubj, StateSubj, ExecGroup>, Cap,
    std::enable_if_t<std::is_same_v<ExecGroup, iro::exec::warp>>> {
    static constexpr bool supported = true;
    using type = typename axp::level2::attention::WarpSoftmaxStateMaskedF32<
        Recipe, Frag, MaskPayload, MaskSubj, NegInfSubj, InSubj, OutSubj, StateSubj, ExecGroup, Cap
    >::type;
};

template<class Recipe, class Frag, class MaskPayload, class MaskSubj, class NegInfSubj,
         class InSubj, class OutSubj, class StateSubj, class ExecGroup, class Cap>
struct resolve_impl<WarpSoftmaxStateMaskedPattern<
    Recipe, Frag, MaskPayload, MaskSubj, NegInfSubj, InSubj, OutSubj, StateSubj, ExecGroup>, Cap,
    std::enable_if_t<iro::exec::is_warpgroup_v<ExecGroup>>> {
    static constexpr bool supported = true;
    using type = typename axp::level2::attention::WarpgroupSoftmaxStateMaskedF32<
        Recipe, Frag, MaskPayload, MaskSubj, NegInfSubj, InSubj, OutSubj, StateSubj, ExecGroup, Cap
    >::type;
};

template<class Recipe, class InSubj, class OutSubj, class Cap>
struct resolve_impl<WarpReduceSoftmaxStatePattern<Recipe, InSubj, OutSubj>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::attention::WarpReduceSoftmaxStateF32<Recipe, InSubj, OutSubj>;
};

template<class Recipe, class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<SoftmaxStateCopyPattern<Recipe, InSubj, OutSubj, ExecGroup>, Cap,
                    std::enable_if_t<axp::level2::attention::is_supported_exec<ExecGroup>::value>> {
    static constexpr bool supported = true;
    using type = axp::level2::attention::SoftmaxStateCopyF32<Recipe, InSubj, OutSubj, ExecGroup>;
};

template<class Recipe, class TileStateSubj, class NewStateSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<SoftmaxStateScalePattern<Recipe, TileStateSubj, NewStateSubj, OutSubj, ExecGroup>, Cap,
                    std::enable_if_t<axp::level2::attention::is_supported_exec<ExecGroup>::value>> {
    static constexpr bool supported = true;
    using type = axp::level2::attention::SoftmaxStateScaleF32<Recipe, TileStateSubj, NewStateSubj, OutSubj, ExecGroup>;
};

template<class Recipe, class AccFrag, class AccSubj, class OldStateSubj, class NewStateSubj, class ExecGroup, class Cap>
struct resolve_impl<RescaleAccumulatorPattern<Recipe, AccFrag, AccSubj, OldStateSubj, NewStateSubj, ExecGroup>, Cap,
                    std::enable_if_t<axp::level2::attention::is_supported_exec<ExecGroup>::value>> {
    static constexpr bool supported = true;
    using type = axp::level2::attention::RescaleAccumulatorF32<Recipe, AccFrag, AccSubj, OldStateSubj, NewStateSubj, ExecGroup>;
};

template<class Recipe, class AccFrag, class AccSubj, class OldStateSubj, class NewStateSubj, class OutStateSubj,
         class ExecGroup, class Cap>
struct resolve_impl<OnlineSoftmaxUpdatePattern<Recipe, AccFrag, AccSubj, OldStateSubj, NewStateSubj, OutStateSubj, ExecGroup>, Cap,
                    std::enable_if_t<axp::level2::attention::is_supported_exec<ExecGroup>::value>> {
    static constexpr bool supported = true;
    using type = typename axp::level2::attention::detail::online_softmax_update_impl<
        Recipe, AccFrag, AccSubj, OldStateSubj, NewStateSubj, OutStateSubj, ExecGroup, Cap
    >::type;
};

template<class Recipe, int TileM, int TileN, int HeadDim, int Stages,
         class QSubj, class KSubj, class VSubj,
         class AccSubj, class OldStateSubj, class OutStateSubj,
         class MemoryPatternQ, class MemoryPatternK, class MemoryPatternV,
         class LoadModeQ, class LoadModeK, class LoadModeV,
         class Schedule,
         class TileSkip,
         class QTma, class KTma, class VTma, class Cap>
struct resolve_impl<AttentionWgmmaPattern<Recipe, TileM, TileN, HeadDim, Stages,
                                          QSubj, KSubj, VSubj, AccSubj, OldStateSubj, OutStateSubj,
                                          MemoryPatternQ, MemoryPatternK, MemoryPatternV,
                                          LoadModeQ, LoadModeK, LoadModeV, Schedule,
                                          TileSkip,
                                          QTma, KTma, VTma>,
                    Cap,
                    std::enable_if_t<Cap::has_wgmma>> {
    static constexpr bool supported = true;
    using type = typename axp::level2::attention::AttentionWgmmaImpl<
        Recipe, TileM, TileN, HeadDim, Stages,
        QSubj, KSubj, VSubj,
        AccSubj, OldStateSubj, OutStateSubj,
        MemoryPatternQ, MemoryPatternK, MemoryPatternV,
        LoadModeQ, LoadModeK, LoadModeV,
        Schedule,
        TileSkip,
        QTma, KTma, VTma, Cap
    >::type;
};

} // namespace axp::level2::registry
#endif // AXP_LIBRARY_BUILD
