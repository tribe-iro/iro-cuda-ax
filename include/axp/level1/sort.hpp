#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level0/communication.hpp"
#include "../level0/compute.hpp"
#include "../level0/fragment.hpp"
#include "registry.hpp"
#include "detail/compose.hpp"

namespace axp::level1 {

// Bitonic sort (fragment payload, lane-local).
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
using BitonicSort = registry::Select<registry::BitonicSortPattern<Recipe, Payload, InSubj, OutSubj, ExecGroup>, CapT>;

// Bitonic merge (fragment payload, assumes bitonic sequence).
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
using BitonicMerge = registry::Select<registry::BitonicMergePattern<Recipe, Payload, InSubj, OutSubj, ExecGroup>, CapT>;

// Bitonic merge across warp lanes (scalar payload, per-lane values).
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
using BitonicMergeCross = registry::Select<registry::BitonicMergeCrossPattern<Recipe, Payload, InSubj, OutSubj, ExecGroup>, CapT>;

} // namespace axp::level1

namespace axp::level1::detail {

template<int N>
consteval bool is_pow2() {
    return (N > 0) && ((N & (N - 1)) == 0);
}

template<class Payload>
consteval int payload_count() {
    if constexpr (iro::contract::FragmentPayload<Payload>) {
        return static_cast<int>(Payload::count);
    } else if constexpr (iro::contract::VectorPayload<Payload>) {
        return static_cast<int>(Payload::lanes);
    } else if constexpr (iro::contract::ScalarPayload<Payload>) {
        return 1;
    } else {
        return 0;
    }
}

template<int K, int J>
struct step_desc {
    static constexpr int k = K;
    static constexpr int j = J;
};

template<int N, int K, int J, bool Done = (K > N)>
struct bitonic_sort_steps_impl;

template<int N, int K, int J>
struct bitonic_sort_steps_impl<N, K, J, true> {
    using type = iro::util::type_list<>;
};

template<int N, int K, int J>
struct bitonic_sort_steps_impl<N, K, J, false> {
    using next = std::conditional_t<
        (J > 1),
        bitonic_sort_steps_impl<N, K, J / 2>,
        bitonic_sort_steps_impl<N, K * 2, (K * 2) / 2>
    >;
    using type = iro::util::concat_t<
        iro::util::type_list<step_desc<K, J>>,
        typename next::type
    >;
};

template<int N>
using bitonic_sort_steps = typename bitonic_sort_steps_impl<N, 2, 1>::type;

template<int N, int J, bool Done = (J == 0)>
struct bitonic_merge_steps_impl;

template<int N, int J>
struct bitonic_merge_steps_impl<N, J, true> {
    using type = iro::util::type_list<>;
};

template<int N, int J>
struct bitonic_merge_steps_impl<N, J, false> {
    using next = bitonic_merge_steps_impl<N, J / 2>;
    using type = iro::util::concat_t<
        iro::util::type_list<step_desc<N, J>>,
        typename next::type
    >;
};

template<int N>
using bitonic_merge_steps = typename bitonic_merge_steps_impl<N, N / 2>::type;

template<int N, int J>
struct partner_pattern {
    static constexpr auto id = iro::util::mix_u64(
        iro::util::fnv1a_64_cstr("axp.level1.sort.partner"),
        iro::util::mix_u64(static_cast<iro::util::u64>(N), static_cast<iro::util::u64>(J))
    );
    static constexpr int map(int i, int) {
        return i ^ J;
    }
};

template<int Width, int K, int J>
struct select_min_mask {
    static constexpr int width = Width;
    static constexpr bool select_min(int i) {
        const bool ascending = ((i & K) == 0);
        const bool jbit_zero = ((i & J) == 0);
        return ascending ? jbit_zero : !jbit_zero;
    }
    static constexpr uint32_t word(int word_index) {
        uint32_t mask = 0u;
        const int base = word_index * 32;
        for (int i = 0; i < 32; ++i) {
            const int idx = base + i;
            if (idx < width && select_min(idx)) {
                mask |= (1u << i);
            }
        }
        return mask;
    }
};

template<int Index>
struct step_tag {
    static constexpr auto id = iro::util::mix_u64(
        iro::util::fnv1a_64_cstr("axp.level1.sort.step"),
        static_cast<iro::util::u64>(Index));
};

template<int Index>
using step_subj = iro::contract::subject::indexed<step_tag<Index>, 0>;

template<int Index, int Slot>
struct step_internal_tag {
    static constexpr auto id = iro::util::mix_u64(
        iro::util::fnv1a_64_cstr("axp.level1.sort.step_internal"),
        iro::util::mix_u64(static_cast<iro::util::u64>(Index), static_cast<iro::util::u64>(Slot)));
};

template<int Index, int Slot>
using step_internal_subj = iro::contract::subject::indexed<step_internal_tag<Index, Slot>, 0>;

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, int J, int K, int Index>
struct bitonic_step_impl {
    static_assert(is_pow2<payload_count<Payload>()>(), "BitonicStep requires power-of-two payload count");
    static constexpr int kCount = payload_count<Payload>();
    using MaskPayload = iro::contract::MaskDesc<kCount, iro::dist::replicated>;

    struct Permute : axp::level0::FragmentPermute<
        Recipe, Payload, InSubj, step_internal_subj<Index, 0>, ExecGroup, partner_pattern<kCount, J>> {};
    struct Min : axp::level0::Min<
        Recipe, Payload, InSubj, step_internal_subj<Index, 0>, step_internal_subj<Index, 1>, ExecGroup> {};
    struct Max : axp::level0::Max<
        Recipe, Payload, InSubj, step_internal_subj<Index, 0>, step_internal_subj<Index, 2>, ExecGroup> {};
    struct Mask : axp::level0::MaskConst<
        Recipe, MaskPayload, step_internal_subj<Index, 3>, ExecGroup, select_min_mask<kCount, K, J>> {};
    struct Select : axp::level0::Select<
        Recipe, Payload, MaskPayload,
        step_internal_subj<Index, 1>,
        step_internal_subj<Index, 2>,
        step_internal_subj<Index, 3>,
        OutSubj,
        ExecGroup> {};

    using obligations = iro::util::type_list<Permute, Min, Max, Mask, Select>;
    using edges = iro::util::type_list<
        iro::compose::Edge<iro::compose::out_port_ref<Permute, 0>, iro::compose::in_port_ref<Min, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<Permute, 0>, iro::compose::in_port_ref<Max, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<Min, 0>, iro::compose::in_port_ref<Select, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<Max, 0>, iro::compose::in_port_ref<Select, 1>>,
        iro::compose::Edge<iro::compose::out_port_ref<Mask, 0>, iro::compose::in_port_ref<Select, 2>>
    >;
    using select_obligation = Select;
    using permute_obligation = Permute;
    using min_obligation = Min;
    using max_obligation = Max;
};

template<class StepList>
struct step_obligations;

template<>
struct step_obligations<iro::util::type_list<>> {
    using type = iro::util::type_list<>;
};

template<class Step, class... Rest>
struct step_obligations<iro::util::type_list<Step, Rest...>> {
    using type = iro::util::concat_t<typename Step::obligations, typename step_obligations<iro::util::type_list<Rest...>>::type>;
};

template<class StepList>
struct step_edges_internal;

template<>
struct step_edges_internal<iro::util::type_list<>> {
    using type = iro::util::type_list<>;
};

template<class Step, class... Rest>
struct step_edges_internal<iro::util::type_list<Step, Rest...>> {
    using type = iro::util::concat_t<typename Step::edges, typename step_edges_internal<iro::util::type_list<Rest...>>::type>;
};

template<class StepList>
struct step_edges_link;

template<>
struct step_edges_link<iro::util::type_list<>> {
    using type = iro::util::type_list<>;
};

template<class Step0>
struct step_edges_link<iro::util::type_list<Step0>> {
    using type = iro::util::type_list<>;
};

template<class Step0, class Step1, class... Rest>
struct step_edges_link<iro::util::type_list<Step0, Step1, Rest...>> {
    using type = iro::util::concat_t<
        iro::util::type_list<
            iro::compose::Edge<iro::compose::out_port_ref<typename Step0::select_obligation, 0>,
                               iro::compose::in_port_ref<typename Step1::permute_obligation, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<typename Step0::select_obligation, 0>,
                               iro::compose::in_port_ref<typename Step1::min_obligation, 0>>,
            iro::compose::Edge<iro::compose::out_port_ref<typename Step0::select_obligation, 0>,
                               iro::compose::in_port_ref<typename Step1::max_obligation, 0>>
        >,
        typename step_edges_link<iro::util::type_list<Step1, Rest...>>::type
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class Steps, int Index, int Total>
struct make_steps;

template<int Index, class InSubj>
struct step_in_subj;

template<class InSubj>
struct step_in_subj<0, InSubj> {
    using type = InSubj;
};

template<int Index, class InSubj>
struct step_in_subj {
    using type = step_subj<Index - 1>;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, int Index, int Total>
struct make_steps<Recipe, Payload, InSubj, OutSubj, ExecGroup, iro::util::type_list<>, Index, Total> {
    using type = iro::util::type_list<>;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class StepDesc0, class... Rest, int Index, int Total>
struct make_steps<Recipe, Payload, InSubj, OutSubj, ExecGroup,
                  iro::util::type_list<StepDesc0, Rest...>, Index, Total> {
    static constexpr bool is_last = (Index + 1 == Total);
    using step_in = typename step_in_subj<Index, InSubj>::type;
    using step_out = std::conditional_t<is_last, OutSubj, step_subj<Index>>;
    using step = bitonic_step_impl<Recipe, Payload, step_in, step_out, ExecGroup, StepDesc0::j, StepDesc0::k, Index>;
    using rest = typename make_steps<Recipe, Payload, InSubj, OutSubj, ExecGroup,
                                     iro::util::type_list<Rest...>, Index + 1, Total>::type;
    using type = iro::util::concat_t<iro::util::type_list<step>, rest>;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class StepList,
         class CapT = axp::target_cap>
struct bitonic_sort_impl_base {
    static constexpr int kCount = payload_count<Payload>();
    static_assert(kCount > 1, "BitonicSort requires payload count > 1");
    static_assert(is_pow2<kCount>(), "BitonicSort requires power-of-two payload count");
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp> || iro::exec::is_warpgroup_v<ExecGroup>,
                  "BitonicSort requires warp/warpgroup ExecGroup");
    using steps = typename make_steps<Recipe, Payload, InSubj, OutSubj, ExecGroup,
                                      StepList, 0, iro::util::size_v<StepList>>::type;
    using obligations = typename step_obligations<steps>::type;
    using edges = iro::util::concat_t<
        typename step_edges_internal<steps>::type,
        typename step_edges_link<steps>::type
    >;
    using type = axp::level1::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
struct bitonic_sort_impl
    : bitonic_sort_impl_base<Recipe, Payload, InSubj, OutSubj, ExecGroup, bitonic_sort_steps<payload_count<Payload>()>, CapT> {};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
struct bitonic_merge_impl
    : bitonic_sort_impl_base<Recipe, Payload, InSubj, OutSubj, ExecGroup, bitonic_merge_steps<payload_count<Payload>()>, CapT> {};

template<class StepList>
struct cross_edges_link;

template<>
struct cross_edges_link<iro::util::type_list<>> {
    using type = iro::util::type_list<>;
};

template<class Step0>
struct cross_edges_link<iro::util::type_list<Step0>> {
    using type = iro::util::type_list<>;
};

template<class Step0, class Step1, class... Rest>
struct cross_edges_link<iro::util::type_list<Step0, Step1, Rest...>> {
    using type = iro::util::concat_t<
        iro::util::type_list<
            iro::compose::Edge<iro::compose::out_port_ref<Step0, 0>,
                               iro::compose::in_port_ref<Step1, 0>>
        >,
        typename cross_edges_link<iro::util::type_list<Step1, Rest...>>::type
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class Steps, int Index, int Total>
struct make_cross_steps;

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, int Index, int Total>
struct make_cross_steps<Recipe, Payload, InSubj, OutSubj, ExecGroup, iro::util::type_list<>, Index, Total> {
    using type = iro::util::type_list<>;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class StepDesc0, class... Rest, int Index, int Total>
struct make_cross_steps<Recipe, Payload, InSubj, OutSubj, ExecGroup,
                        iro::util::type_list<StepDesc0, Rest...>, Index, Total> {
    static constexpr bool is_last = (Index + 1 == Total);
    using step_in = typename step_in_subj<Index, InSubj>::type;
    using step_out = std::conditional_t<is_last, OutSubj, step_subj<Index>>;
    using step = axp::level0::WarpBitonicStep<
        Recipe, Payload, step_in, step_out, ExecGroup, StepDesc0::k, StepDesc0::j
    >;
    using rest = typename make_cross_steps<Recipe, Payload, InSubj, OutSubj, ExecGroup,
                                           iro::util::type_list<Rest...>, Index + 1, Total>::type;
    using type = iro::util::concat_t<iro::util::type_list<step>, rest>;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class StepList,
         class CapT = axp::target_cap>
struct bitonic_merge_cross_impl_base {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "BitonicMergeCross requires warp ExecGroup");
    static_assert(iro::contract::ScalarPayload<Payload>,
                  "BitonicMergeCross requires scalar payload");
    static_assert(std::is_same_v<typename Payload::dist, iro::dist::lane>,
                  "BitonicMergeCross requires dist::lane payload");
    using steps = typename make_cross_steps<Recipe, Payload, InSubj, OutSubj, ExecGroup,
                                            StepList, 0, iro::util::size_v<StepList>>::type;
    using obligations = steps;
    using edges = typename cross_edges_link<steps>::type;
    using type = axp::level1::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
struct bitonic_merge_cross_impl
    : bitonic_merge_cross_impl_base<Recipe, Payload, InSubj, OutSubj, ExecGroup, bitonic_merge_steps<32>, CapT> {};

} // namespace axp::level1::detail

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level1::registry {

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<BitonicSortPattern<Recipe, Payload, InSubj, OutSubj, ExecGroup>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level1::detail::bitonic_sort_impl<Recipe, Payload, InSubj, OutSubj, ExecGroup, Cap>::type;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<BitonicMergePattern<Recipe, Payload, InSubj, OutSubj, ExecGroup>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level1::detail::bitonic_merge_impl<Recipe, Payload, InSubj, OutSubj, ExecGroup, Cap>::type;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class Cap>
struct resolve_impl<BitonicMergeCrossPattern<Recipe, Payload, InSubj, OutSubj, ExecGroup>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level1::detail::bitonic_merge_cross_impl<Recipe, Payload, InSubj, OutSubj, ExecGroup, Cap>::type;
};

} // namespace axp::level1::registry
#endif // AXP_LIBRARY_BUILD
