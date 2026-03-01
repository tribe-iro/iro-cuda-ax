#pragma once

#include <iro_cuda_ax_core.hpp>
#include <axp/concepts.hpp>
#include <axp/detail/resources.hpp>
#include <axp/detail/math.hpp>
#include <axp/state.hpp>
#include <axp/bundles/token_bundles.hpp>
#include "../level0/compute.hpp"
#include "../level0/fragment.hpp"
#include "../level1/communication.hpp"
#include "detail/compose.hpp"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace axp::level2::norm {

using WelfordStateF32 = axp::state::WelfordStateF32;

namespace detail {
template<class ExecGroup>
struct exec_lifetime;

template<>
struct exec_lifetime<iro::exec::warp> { using type = iro::token::lifetime::warp; };

template<int Warps>
struct exec_lifetime<iro::exec::warpgroup_t<Warps>> { using type = iro::token::lifetime::warpgroup; };

template<class ExecGroup>
struct is_supported_exec : std::false_type {};
template<>
struct is_supported_exec<iro::exec::warp> : std::true_type {};
template<int Warps>
struct is_supported_exec<iro::exec::warpgroup_t<Warps>> : std::true_type {};

template<class Subject, class ExecGroup, class Lifetime>
using value_tokens = axp::bundle::ValueLive<Subject, ExecGroup, Lifetime>;

template<class Subject, class ExecGroup, class Lifetime>
using warp_reduce_out_tokens = axp::bundle::ValueLane0<Subject, ExecGroup, Lifetime>;

__device__ __forceinline__ WelfordStateF32 welford_combine(WelfordStateF32 a, WelfordStateF32 b) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;
    const int count = a.count + b.count;
    const float delta = b.mean - a.mean;
    const float mean = a.mean + delta * (static_cast<float>(b.count) / static_cast<float>(count));
    const float m2 = a.m2 + b.m2 + delta * delta *
        (static_cast<float>(a.count) * static_cast<float>(b.count) / static_cast<float>(count));
    return WelfordStateF32{mean, m2, count};
}

template<class ElemT>
struct elem_traits;

template<>
struct elem_traits<iro::elem::f32> {
    using storage_t = float;
    __device__ __forceinline__ static float to_f32(storage_t x) { return x; }
};

#ifdef __CUDACC__
template<>
struct elem_traits<iro::elem::f16> {
    using storage_t = __half;
    __device__ __forceinline__ static float to_f32(storage_t x) { return __half2float(x); }
};

template<>
struct elem_traits<iro::elem::bf16> {
    using storage_t = __nv_bfloat16;
    __device__ __forceinline__ static float to_f32(storage_t x) { return __bfloat162float(x); }
};
#endif
} // namespace detail

template<class Recipe, class ExecGroup, class ASubj, class BSubj, class OutSubj>
    requires axp::concepts::RecipeAccF32<Recipe> &&
             (axp::concepts::WarpExec<ExecGroup> || axp::concepts::WarpgroupExec<ExecGroup>)
struct CombineWelfordStateF32 {
    static_assert(detail::is_supported_exec<ExecGroup>::value, "CombineWelfordStateF32: exec must be warp or warpgroup");
    static_assert(std::is_same_v<typename Recipe::acc, iro::elem::f32>, "CombineWelfordStateF32: Recipe::acc must be f32");
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;
    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            WelfordStateF32,
            ASubj,
            ExecGroup,
            detail::value_tokens<ASubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            WelfordStateF32,
            BSubj,
            ExecGroup,
            detail::value_tokens<BSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            WelfordStateF32,
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
struct CombineWelfordStateF32Realization
    : iro::contract::Realization<
        CombineWelfordStateF32<Recipe, ExecGroup, ASubj, BSubj, OutSubj>,
        iro::util::fnv1a_64_cstr("axp.layernorm.welford_state_combine")> {
    __device__ __forceinline__ static WelfordStateF32 execute(WelfordStateF32 a, WelfordStateF32 b) {
#ifdef __CUDA_ARCH__
        return detail::welford_combine(a, b);
#else
        return a.count == 0 ? b : a;
#endif
    }
};

template<class Recipe, class InSubj, class OutSubj>
    requires axp::concepts::RecipeAccF32<Recipe>
struct WarpReduceWelfordStateF32 {
    using ExecGroup = iro::exec::warp;
    static_assert(std::is_same_v<typename Recipe::acc, iro::elem::f32>, "WarpReduceWelfordStateF32: Recipe::acc must be f32");
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;
    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            WelfordStateF32,
            InSubj,
            ExecGroup,
            detail::value_tokens<InSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            WelfordStateF32,
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
struct WarpReduceWelfordStateF32Realization
    : iro::contract::Realization<
        WarpReduceWelfordStateF32<Recipe, InSubj, OutSubj>,
        iro::util::fnv1a_64_cstr("axp.layernorm.welford_state_warp_reduce")> {
    __device__ __forceinline__ static WelfordStateF32 execute(WelfordStateF32 s) {
#ifdef __CUDA_ARCH__
        const unsigned mask = __activemask();
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            WelfordStateF32 other;
            other.mean = __shfl_down_sync(mask, s.mean, offset);
            other.m2 = __shfl_down_sync(mask, s.m2, offset);
            other.count = __shfl_down_sync(mask, s.count, offset);
            s = detail::welford_combine(s, other);
        }
        return s;
#else
        return s;
#endif
    }
};

// Per-thread fragment -> Welford state (warp scope)
template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup>
    requires axp::concepts::RecipeAccF32<Recipe>
struct FragmentWelfordStateF32 {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "FragmentWelfordStateF32: ExecGroup must be warp");
    static_assert(iro::contract::FragmentPayload<Frag>, "FragmentWelfordStateF32 requires Fragment payload");
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;
    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Frag,
            InSubj,
            ExecGroup,
            detail::value_tokens<InSubj, ExecGroup, lifetime>,
            typename Frag::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            WelfordStateF32,
            OutSubj,
            ExecGroup,
            detail::value_tokens<OutSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup>
struct FragmentWelfordStateF32Realization
    : iro::contract::Realization<
        FragmentWelfordStateF32<Recipe, Frag, InSubj, OutSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.level2.welford.fragment_state")> {
    using storage_t = typename detail::elem_traits<typename Frag::elem>::storage_t;
    __device__ __forceinline__ static WelfordStateF32 execute(const storage_t* in) {
#ifdef __CUDA_ARCH__
        float mean = 0.0f;
        float m2 = 0.0f;
        int count = 0;
        constexpr int kElems = static_cast<int>(Frag::count);
        #pragma unroll
        for (int i = 0; i < kElems; ++i) {
            float x = detail::elem_traits<typename Frag::elem>::to_f32(in[i]);
            ++count;
            float delta = x - mean;
            mean += delta / static_cast<float>(count);
            float delta2 = x - mean;
            m2 += delta * delta2;
        }
        return WelfordStateF32{mean, m2, count};
#else
        (void)in;
        return WelfordStateF32{0.0f, 0.0f, 0};
#endif
    }
};

// Warp all-reduce Welford state (replicate to all lanes)
template<class Recipe, class InSubj, class OutSubj>
    requires axp::concepts::RecipeAccF32<Recipe>
struct WarpAllReduceWelfordStateF32 {
    using ExecGroup = iro::exec::warp;
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;
    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            WelfordStateF32,
            InSubj,
            ExecGroup,
            detail::value_tokens<InSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            WelfordStateF32,
            OutSubj,
            ExecGroup,
            detail::value_tokens<OutSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

template<class Recipe, class InSubj, class OutSubj>
struct WarpAllReduceWelfordStateF32Realization
    : iro::contract::Realization<
        WarpAllReduceWelfordStateF32<Recipe, InSubj, OutSubj>,
        iro::util::fnv1a_64_cstr("axp.level2.welford.warp_all_reduce")> {
    __device__ __forceinline__ static WelfordStateF32 execute(WelfordStateF32 s) {
#ifdef __CUDA_ARCH__
        const unsigned mask = __activemask();
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            WelfordStateF32 other;
            other.mean = __shfl_down_sync(mask, s.mean, offset);
            other.m2 = __shfl_down_sync(mask, s.m2, offset);
            other.count = __shfl_down_sync(mask, s.count, offset);
            s = detail::welford_combine(s, other);
        }
        s.mean = __shfl_sync(mask, s.mean, 0);
        s.m2 = __shfl_sync(mask, s.m2, 0);
        s.count = __shfl_sync(mask, s.count, 0);
        return s;
#else
        return s;
#endif
    }
};

// Welford state -> mean + inv_std (scalar outputs)
template<class Recipe, class EpsPayload, class StateSubj, class EpsSubj, class MeanSubj, class InvStdSubj, class ExecGroup>
    requires axp::concepts::RecipeAccF32<Recipe>
struct WelfordStateMeanInvStdF32 {
    static_assert(iro::contract::ScalarPayload<EpsPayload>, "WelfordStateMeanInvStdF32: EpsPayload must be scalar");
    static_assert(std::is_same_v<typename EpsPayload::elem, iro::elem::f32>,
                  "WelfordStateMeanInvStdF32: EpsPayload::elem must be f32");
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;
    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            WelfordStateF32,
            StateSubj,
            ExecGroup,
            detail::value_tokens<StateSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            EpsPayload,
            EpsSubj,
            ExecGroup,
            detail::value_tokens<EpsSubj, ExecGroup, lifetime>,
            typename EpsPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            EpsPayload,
            MeanSubj,
            ExecGroup,
            detail::value_tokens<MeanSubj, ExecGroup, lifetime>,
            typename EpsPayload::dist,
            Recipe
        >,
        iro::contract::OutputPort<
            EpsPayload,
            InvStdSubj,
            ExecGroup,
            detail::value_tokens<InvStdSubj, ExecGroup, lifetime>,
            typename EpsPayload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

template<class Recipe, class EpsPayload, class StateSubj, class EpsSubj, class MeanSubj, class InvStdSubj, class ExecGroup>
struct WelfordStateMeanInvStdF32Realization
    : iro::contract::Realization<
        WelfordStateMeanInvStdF32<Recipe, EpsPayload, StateSubj, EpsSubj, MeanSubj, InvStdSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.level2.welford.mean_inv_std")> {
    using eps_t = typename EpsPayload::elem::storage_t;
    __device__ __forceinline__ static void execute(WelfordStateF32 state, const eps_t* eps,
                                                   eps_t* mean_out, eps_t* inv_std_out) {
#ifdef __CUDA_ARCH__
        const int count = state.count > 0 ? state.count : 1;
        const float var = state.m2 / static_cast<float>(count);
        const float mean = state.mean;
        const float inv_std = axp::detail::math::rsqrtf_recipe<Recipe>(var + static_cast<float>(eps[0]));
        mean_out[0] = static_cast<eps_t>(mean);
        inv_std_out[0] = static_cast<eps_t>(inv_std);
#else
        (void)state; (void)eps; (void)mean_out; (void)inv_std_out;
#endif
    }
};

// Welford state -> inv_rms (scalar output)
template<class Recipe, class EpsPayload, class StateSubj, class EpsSubj, class InvRmsSubj, class ExecGroup>
    requires axp::concepts::RecipeAccF32<Recipe>
struct WelfordStateInvRmsF32 {
    static_assert(iro::contract::ScalarPayload<EpsPayload>, "WelfordStateInvRmsF32: EpsPayload must be scalar");
    static_assert(std::is_same_v<typename EpsPayload::elem, iro::elem::f32>,
                  "WelfordStateInvRmsF32: EpsPayload::elem must be f32");
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;
    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            WelfordStateF32,
            StateSubj,
            ExecGroup,
            detail::value_tokens<StateSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            EpsPayload,
            EpsSubj,
            ExecGroup,
            detail::value_tokens<EpsSubj, ExecGroup, lifetime>,
            typename EpsPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            EpsPayload,
            InvRmsSubj,
            ExecGroup,
            detail::value_tokens<InvRmsSubj, ExecGroup, lifetime>,
            typename EpsPayload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

template<class Recipe, class EpsPayload, class StateSubj, class EpsSubj, class InvRmsSubj, class ExecGroup>
struct WelfordStateInvRmsF32Realization
    : iro::contract::Realization<
        WelfordStateInvRmsF32<Recipe, EpsPayload, StateSubj, EpsSubj, InvRmsSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.level2.welford.inv_rms")> {
    using eps_t = typename EpsPayload::elem::storage_t;
    __device__ __forceinline__ static void execute(WelfordStateF32 state, const eps_t* eps, eps_t* inv_rms_out) {
#ifdef __CUDA_ARCH__
        const int count = state.count > 0 ? state.count : 1;
        const float var = state.m2 / static_cast<float>(count);
        const float mean = state.mean;
        const float mean_sq = var + mean * mean;
        inv_rms_out[0] = static_cast<eps_t>(
            axp::detail::math::rsqrtf_recipe<Recipe>(mean_sq + static_cast<float>(eps[0]))
        );
#else
        (void)state; (void)eps; (void)inv_rms_out;
#endif
    }
};

// Welford state -> mean (scalar output)
template<class Recipe, class OutPayload, class StateSubj, class OutSubj, class ExecGroup>
    requires axp::concepts::RecipeAccF32<Recipe>
struct WelfordStateMeanF32 {
    static_assert(iro::contract::ScalarPayload<OutPayload>, "WelfordStateMeanF32: OutPayload must be scalar");
    static_assert(std::is_same_v<typename OutPayload::elem, iro::elem::f32>,
                  "WelfordStateMeanF32: OutPayload::elem must be f32");
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            WelfordStateF32,
            StateSubj,
            ExecGroup,
            detail::value_tokens<StateSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutPayload,
            OutSubj,
            ExecGroup,
            detail::value_tokens<OutSubj, ExecGroup, lifetime>,
            typename OutPayload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

template<class Recipe, class OutPayload, class StateSubj, class OutSubj, class ExecGroup>
struct WelfordStateMeanF32Realization
    : iro::contract::Realization<
        WelfordStateMeanF32<Recipe, OutPayload, StateSubj, OutSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.level2.welford.mean")> {
    using out_t = typename OutPayload::elem::storage_t;
    __device__ __forceinline__ static void execute(WelfordStateF32 state, out_t* mean_out) {
#ifdef __CUDA_ARCH__
        const float mean = state.mean;
        mean_out[0] = static_cast<out_t>(mean);
#else
        (void)state; (void)mean_out;
#endif
    }
};

// Welford state -> variance (scalar output)
template<class Recipe, class OutPayload, class StateSubj, class OutSubj, class ExecGroup>
    requires axp::concepts::RecipeAccF32<Recipe>
struct WelfordStateVarF32 {
    static_assert(iro::contract::ScalarPayload<OutPayload>, "WelfordStateVarF32: OutPayload must be scalar");
    static_assert(std::is_same_v<typename OutPayload::elem, iro::elem::f32>,
                  "WelfordStateVarF32: OutPayload::elem must be f32");
    using lifetime = typename detail::exec_lifetime<ExecGroup>::type;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            WelfordStateF32,
            StateSubj,
            ExecGroup,
            detail::value_tokens<StateSubj, ExecGroup, lifetime>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutPayload,
            OutSubj,
            ExecGroup,
            detail::value_tokens<OutSubj, ExecGroup, lifetime>,
            typename OutPayload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

template<class Recipe, class OutPayload, class StateSubj, class OutSubj, class ExecGroup>
struct WelfordStateVarF32Realization
    : iro::contract::Realization<
        WelfordStateVarF32<Recipe, OutPayload, StateSubj, OutSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.level2.welford.var")> {
    using out_t = typename OutPayload::elem::storage_t;
    __device__ __forceinline__ static void execute(WelfordStateF32 state, out_t* var_out) {
#ifdef __CUDA_ARCH__
        const int count = state.count > 0 ? state.count : 1;
        const float var = state.m2 / static_cast<float>(count);
        var_out[0] = static_cast<out_t>(var);
#else
        (void)state; (void)var_out;
#endif
    }
};

namespace detail {

struct norm_state_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.norm.state"); };
struct norm_mean_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.norm.mean"); };
struct norm_invstd_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.norm.invstd"); };
struct norm_invrms_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.norm.invrms"); };
struct norm_mean_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.norm.mean_frag"); };
struct norm_inv_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.norm.inv_frag"); };

using norm_state_subj = iro::contract::subject::indexed<norm_state_tag, 0>;
using norm_mean_subj = iro::contract::subject::indexed<norm_mean_tag, 0>;
using norm_invstd_subj = iro::contract::subject::indexed<norm_invstd_tag, 0>;
using norm_invrms_subj = iro::contract::subject::indexed<norm_invrms_tag, 0>;
using norm_mean_frag_subj = iro::contract::subject::indexed<norm_mean_frag_tag, 0>;
using norm_inv_frag_subj = iro::contract::subject::indexed<norm_inv_frag_tag, 0>;

template<class Recipe, class Frag, class GammaFrag, class BetaFrag, class EpsPayload,
         class InSubj, class GammaSubj, class BetaSubj, class EpsSubj, class OutSubj, class ExecGroup,
         class CapT = axp::target_cap>
struct layernorm_frag_impl {
    using State = FragmentWelfordStateF32<Recipe, Frag, InSubj, norm_state_subj, ExecGroup>;
    using AllReduce = WarpAllReduceWelfordStateF32<Recipe, norm_state_subj, norm_state_subj>;
    using MeanInv = WelfordStateMeanInvStdF32<Recipe, EpsPayload, norm_state_subj, EpsSubj, norm_mean_subj, norm_invstd_subj, ExecGroup>;
    using MeanFrag = axp::level1::FragmentBroadcast<
        Recipe, Frag, EpsPayload, norm_mean_subj, norm_mean_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using InvFrag = axp::level1::FragmentBroadcast<
        Recipe, Frag, EpsPayload, norm_invstd_subj, norm_inv_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Sub = axp::level0::Sub<Recipe, Frag, InSubj, norm_mean_frag_subj, norm_mean_frag_subj, ExecGroup>;
    using Mul = axp::level0::Mul<Recipe, Frag, norm_mean_frag_subj, norm_inv_frag_subj, norm_inv_frag_subj, ExecGroup>;
    using Scale = axp::level0::Mul<Recipe, Frag, norm_inv_frag_subj, GammaSubj, norm_inv_frag_subj, ExecGroup>;
    using Shift = axp::level0::Add<Recipe, Frag, norm_inv_frag_subj, BetaSubj, OutSubj, ExecGroup>;

    using obligations = iro::util::type_list<
        State, AllReduce, MeanInv, MeanFrag, InvFrag, Sub, Mul, Scale, Shift
    >;

    using edges = iro::util::type_list<
        iro::compose::Edge<iro::compose::out_port_ref<State, 0>, iro::compose::in_port_ref<AllReduce, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<AllReduce, 0>, iro::compose::in_port_ref<MeanInv, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<MeanInv, 0>, iro::compose::in_port_ref<MeanFrag, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<MeanInv, 1>, iro::compose::in_port_ref<InvFrag, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<MeanFrag, 0>, iro::compose::in_port_ref<Sub, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<Sub, 0>, iro::compose::in_port_ref<Mul, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<InvFrag, 0>, iro::compose::in_port_ref<Mul, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<Mul, 0>, iro::compose::in_port_ref<Scale, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<Scale, 0>, iro::compose::in_port_ref<Shift, 0>>
    >;

    using type = axp::level2::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Frag, class WeightFrag, class EpsPayload,
         class InSubj, class WeightSubj, class EpsSubj, class OutSubj, class ExecGroup,
         class CapT = axp::target_cap>
struct rmsnorm_frag_impl {
    using State = FragmentWelfordStateF32<Recipe, Frag, InSubj, norm_state_subj, ExecGroup>;
    using AllReduce = WarpAllReduceWelfordStateF32<Recipe, norm_state_subj, norm_state_subj>;
    using InvRms = WelfordStateInvRmsF32<Recipe, EpsPayload, norm_state_subj, EpsSubj, norm_invrms_subj, ExecGroup>;
    using InvFrag = axp::level1::FragmentBroadcast<
        Recipe, Frag, EpsPayload, norm_invrms_subj, norm_inv_frag_subj, ExecGroup,
        iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using Mul = axp::level0::Mul<Recipe, Frag, InSubj, norm_inv_frag_subj, norm_inv_frag_subj, ExecGroup>;
    using Scale = axp::level0::Mul<Recipe, Frag, norm_inv_frag_subj, WeightSubj, OutSubj, ExecGroup>;

    using obligations = iro::util::type_list<
        State, AllReduce, InvRms, InvFrag, Mul, Scale
    >;

    using edges = iro::util::type_list<
        iro::compose::Edge<iro::compose::out_port_ref<State, 0>, iro::compose::in_port_ref<AllReduce, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<AllReduce, 0>, iro::compose::in_port_ref<InvRms, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<InvRms, 0>, iro::compose::in_port_ref<InvFrag, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<InvFrag, 0>, iro::compose::in_port_ref<Mul, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<Mul, 0>, iro::compose::in_port_ref<Scale, 0>>
    >;

    using type = axp::level2::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

} // namespace detail
template<class Recipe, class ExecGroup, class ASubj, class BSubj, class OutSubj>
using WelfordStep = CombineWelfordStateF32<Recipe, ExecGroup, ASubj, BSubj, OutSubj>;

// RowMean/RowVariance: Welford over fragment, warp all-reduce, then extract mean/var.
template<class Recipe, class Frag, class InSubj, class OutSubj,
         class ExecGroup = iro::exec::warp, class CapT = axp::target_cap>
using RowMean = registry::Select<registry::RowMeanPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, CapT>;

template<class Recipe, class Frag, class InSubj, class OutSubj,
         class ExecGroup = iro::exec::warp, class CapT = axp::target_cap>
using RowVariance = registry::Select<registry::RowVariancePattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, CapT>;

} // namespace axp::level2::norm

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level2::registry {

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<FragmentWelfordPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, Cap,
                    std::enable_if_t<std::is_same_v<ExecGroup, iro::exec::warp>>> {
    static constexpr bool supported = true;
    using type = axp::level2::norm::FragmentWelfordStateF32<Recipe, Frag, InSubj, OutSubj, ExecGroup>;
};

template<class Recipe, class InSubj, class OutSubj, class Cap>
struct resolve_impl<WarpAllReduceWelfordPattern<Recipe, InSubj, OutSubj>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::norm::WarpAllReduceWelfordStateF32<Recipe, InSubj, OutSubj>;
};

template<class Recipe, class Frag, class GammaFrag, class BetaFrag, class EpsPayload,
         class InSubj, class GammaSubj, class BetaSubj, class EpsSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<LayerNormFragPattern<Recipe, Frag, GammaFrag, BetaFrag, EpsPayload,
                                        InSubj, GammaSubj, BetaSubj, EpsSubj, OutSubj, ExecGroup>,
                    Cap,
                    std::enable_if_t<std::is_same_v<ExecGroup, iro::exec::warp>>> {
    static constexpr bool supported = true;
    using type = typename axp::level2::norm::detail::layernorm_frag_impl<
        Recipe, Frag, GammaFrag, BetaFrag, EpsPayload, InSubj, GammaSubj, BetaSubj, EpsSubj, OutSubj, ExecGroup, Cap
    >::type;
};

template<class Recipe, class Frag, class WeightFrag, class EpsPayload,
         class InSubj, class WeightSubj, class EpsSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<RMSNormFragPattern<Recipe, Frag, WeightFrag, EpsPayload,
                                      InSubj, WeightSubj, EpsSubj, OutSubj, ExecGroup>,
                    Cap,
                    std::enable_if_t<std::is_same_v<ExecGroup, iro::exec::warp>>> {
    static constexpr bool supported = true;
    using type = typename axp::level2::norm::detail::rmsnorm_frag_impl<
        Recipe, Frag, WeightFrag, EpsPayload, InSubj, WeightSubj, EpsSubj, OutSubj, ExecGroup, Cap
    >::type;
};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<RowMeanPattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, Cap,
                    std::enable_if_t<std::is_same_v<ExecGroup, iro::exec::warp>>> {
    static constexpr bool supported = true;
    using OutPayload = iro::contract::ScalarDesc<iro::elem::f32, typename Frag::dist>;
    using State = axp::level2::norm::FragmentWelfordStateF32<Recipe, Frag, InSubj, axp::level2::norm::detail::norm_state_subj, ExecGroup>;
    using AllReduce = axp::level2::norm::WarpAllReduceWelfordStateF32<Recipe, axp::level2::norm::detail::norm_state_subj,
                                                                     axp::level2::norm::detail::norm_state_subj>;
    using Mean = axp::level2::norm::WelfordStateMeanF32<Recipe, OutPayload,
                                                       axp::level2::norm::detail::norm_state_subj, OutSubj, ExecGroup>;
    using obligations = iro::util::type_list<State, AllReduce, Mean>;
    using edges = iro::util::type_list<
        iro::compose::Edge<iro::compose::out_port_ref<State, 0>, iro::compose::in_port_ref<AllReduce, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<AllReduce, 0>, iro::compose::in_port_ref<Mean, 0>>
    >;
    using type = axp::level2::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, Cap>;
};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<RowVariancePattern<Recipe, Frag, InSubj, OutSubj, ExecGroup>, Cap,
                    std::enable_if_t<std::is_same_v<ExecGroup, iro::exec::warp>>> {
    static constexpr bool supported = true;
    using OutPayload = iro::contract::ScalarDesc<iro::elem::f32, typename Frag::dist>;
    using State = axp::level2::norm::FragmentWelfordStateF32<Recipe, Frag, InSubj, axp::level2::norm::detail::norm_state_subj, ExecGroup>;
    using AllReduce = axp::level2::norm::WarpAllReduceWelfordStateF32<Recipe, axp::level2::norm::detail::norm_state_subj,
                                                                     axp::level2::norm::detail::norm_state_subj>;
    using Var = axp::level2::norm::WelfordStateVarF32<Recipe, OutPayload,
                                                     axp::level2::norm::detail::norm_state_subj, OutSubj, ExecGroup>;
    using obligations = iro::util::type_list<State, AllReduce, Var>;
    using edges = iro::util::type_list<
        iro::compose::Edge<iro::compose::out_port_ref<State, 0>, iro::compose::in_port_ref<AllReduce, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<AllReduce, 0>, iro::compose::in_port_ref<Var, 0>>
    >;
    using type = axp::level2::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, Cap>;
};

} // namespace axp::level2::registry
#endif // AXP_LIBRARY_BUILD
