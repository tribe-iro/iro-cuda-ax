#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level0/compute.hpp"
#include "../level2/registry.hpp"
#include "detail/compose.hpp"

namespace axp::level2::epilogue {

namespace detail {

template<int ElemBytes, int AlignBytes, int MaxVecBytes>
consteval int select_vec_bytes() {
    if constexpr ((MaxVecBytes >= 16) && (AlignBytes >= 16) && ((16 % ElemBytes) == 0)) {
        return 16;
    } else if constexpr ((MaxVecBytes >= 8) && (AlignBytes >= 8) && ((8 % ElemBytes) == 0)) {
        return 8;
    } else {
        return 4;
    }
}

template<class Recipe, class Elem, class Dist>
struct vec_payload_selector {
    static constexpr int vec_bytes = select_vec_bytes<Elem::bytes, Elem::align, Recipe::vec_bytes>();
    static constexpr int lanes = vec_bytes / Elem::bytes;
    using type = iro::contract::VectorDesc<Elem, lanes, Dist>;
};

template<class Tag>
struct subj_tagged {
    using type = iro::contract::subject::indexed<Tag, 0>;
};

template<class Obligation, int I>
using in_port_t = axp::level2::detail::in_port_t<Obligation, I>;

template<class Obligation, int I>
using out_port_t = axp::level2::detail::out_port_t<Obligation, I>;

using axp::level2::detail::make_composition_t;

} // namespace detail

template<class Recipe, class Elem, class Dist, class InSubj, class BiasSubj, class OutSubj, class ExecGroup,
         template<class, class, class, class, class, class, class> class ActOp,
         class CapT = axp::target_cap>
struct FusedBiasActivationVecImpl {
    using VecPayload = typename detail::vec_payload_selector<Recipe, Elem, Dist>::type;
    struct tmp_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.epilogue.bias_act.tmp"); };
    using tmp_subj = typename detail::subj_tagged<tmp_tag>::type;

    using Add = axp::level0::Add<Recipe, VecPayload, InSubj, BiasSubj, tmp_subj, ExecGroup>;
    using Act = ActOp<Recipe, VecPayload, tmp_subj, OutSubj, ExecGroup, iro::util::type_list<>, iro::util::type_list<>>;

    using obligations = iro::util::type_list<Add, Act>;
    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<Add, 0>, detail::in_port_t<Act, 0>>
    >;
    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Elem, class Dist, class AccSubj, class CSubj,
         class AlphaSubj, class BetaSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
struct LinearCombinationVecImpl {
    using VecPayload = typename detail::vec_payload_selector<Recipe, Elem, Dist>::type;
    struct acc_scaled_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.epilogue.acc_scaled"); };
    struct c_scaled_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level2.epilogue.c_scaled"); };
    using acc_scaled_subj = typename detail::subj_tagged<acc_scaled_tag>::type;
    using c_scaled_subj = typename detail::subj_tagged<c_scaled_tag>::type;

    using MulAcc = axp::level0::Mul<Recipe, VecPayload, AccSubj, AlphaSubj, acc_scaled_subj, ExecGroup>;
    using MulC = axp::level0::Mul<Recipe, VecPayload, CSubj, BetaSubj, c_scaled_subj, ExecGroup>;
    using Add = axp::level0::Add<Recipe, VecPayload, acc_scaled_subj, c_scaled_subj, OutSubj, ExecGroup>;

    using obligations = iro::util::type_list<MulAcc, MulC, Add>;
    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<MulAcc, 0>, detail::in_port_t<Add, 0>>,
        iro::compose::Edge<detail::out_port_t<MulC, 0>, detail::in_port_t<Add, 1>>
    >;
    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Elem, class Dist, class InSubj, class BiasSubj, class OutSubj, class ExecGroup,
         template<class, class, class, class, class, class, class> class ActOp,
         class CapT = axp::target_cap>
using FusedBiasActivationVec = registry::Select<registry::FusedBiasActivationVecPattern<
    Recipe, Elem, Dist, InSubj, BiasSubj, OutSubj, ExecGroup, ActOp>, CapT>;

template<class Recipe, class Elem, class Dist, class AccSubj, class CSubj,
         class AlphaSubj, class BetaSubj, class OutSubj, class ExecGroup,
         class CapT = axp::target_cap>
using LinearCombinationVec = registry::Select<registry::LinearCombinationVecPattern<
    Recipe, Elem, Dist, AccSubj, CSubj, AlphaSubj, BetaSubj, OutSubj, ExecGroup>, CapT>;

} // namespace axp::level2::epilogue

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level2::registry {

template<class Recipe, class Elem, class Dist, class InSubj, class BiasSubj, class OutSubj, class ExecGroup,
         template<class, class, class, class, class, class, class> class ActOp, class Cap>
struct resolve_impl<FusedBiasActivationVecPattern<Recipe, Elem, Dist, InSubj, BiasSubj, OutSubj, ExecGroup, ActOp>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level2::epilogue::FusedBiasActivationVecImpl<
        Recipe, Elem, Dist, InSubj, BiasSubj, OutSubj, ExecGroup, ActOp, Cap
    >::type;
};

template<class Recipe, class Elem, class Dist, class AccSubj, class CSubj,
         class AlphaSubj, class BetaSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<LinearCombinationVecPattern<Recipe, Elem, Dist, AccSubj, CSubj, AlphaSubj, BetaSubj, OutSubj, ExecGroup>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level2::epilogue::LinearCombinationVecImpl<
        Recipe, Elem, Dist, AccSubj, CSubj, AlphaSubj, BetaSubj, OutSubj, ExecGroup, Cap
    >::type;
};

} // namespace axp::level2::registry
#endif // AXP_LIBRARY_BUILD
