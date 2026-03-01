#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level0/communication.hpp"
#include "../level0/compute.hpp"
#include "../level0/fragment.hpp"
#include "../level0/scan.hpp"
#include "detail/compose.hpp"
#include "registry.hpp"

namespace axp::level1 {

namespace detail {

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         int SrcLane, class InExtra, class OutExtra, class CapT>
struct broadcast_src_impl {
    using type = as_composition_t<
        axp::level0::Broadcast<Recipe, Payload, InSubj, OutSubj, ExecGroup, SrcLane, InExtra, OutExtra>,
        CapT
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra, class CapT>
struct broadcast_src_impl<Recipe, Payload, InSubj, OutSubj, ExecGroup, 0, InExtra, OutExtra, CapT> {
    using type = as_composition_t<
        axp::level0::BroadcastLane0<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        CapT
    >;
};

} // namespace detail

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         int SrcLane = 0, class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using BroadcastSrc = typename detail::broadcast_src_impl<
    Recipe, Payload, InSubj, OutSubj, ExecGroup, SrcLane, InExtra, OutExtra, CapT
>::type;

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         int SrcLane, class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using Broadcast = BroadcastSrc<Recipe, Payload, InSubj, OutSubj, ExecGroup, SrcLane, InExtra, OutExtra, CapT>;

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using BroadcastLane0 = BroadcastSrc<Recipe, Payload, InSubj, OutSubj, ExecGroup, 0, InExtra, OutExtra, CapT>;

template<class Recipe, class FragPayload, class ScalarPayload, class ScalarSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using FragmentBroadcast = detail::as_composition_t<
    axp::level0::FragmentBroadcast<Recipe, FragPayload, ScalarPayload, ScalarSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
    CapT
>;

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, int BarrierId = 1,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using WarpgroupBroadcastLane0 = detail::as_composition_t<
    axp::level0::WarpgroupBroadcastLane0<Recipe, Payload, InSubj, OutSubj, ExecGroup, BarrierId, InExtra, OutExtra>,
    CapT
>;

template<class Recipe, class InPayload, class InSubj, class PrefixSubj, class CountSubj, class ExecGroup,
         class CapT = axp::target_cap>
using CountBits = registry::Select<
    registry::CountBitsPattern<Recipe, InPayload, InSubj, PrefixSubj, CountSubj, ExecGroup>, CapT
>;

} // namespace axp::level1

namespace axp::level1::detail {

struct count_bits_mask_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level1.count_bits.mask"); };
using count_bits_mask_subj = iro::contract::subject::indexed<count_bits_mask_tag, 0>;

template<class Recipe, class InPayload, class InSubj, class PrefixSubj, class CountSubj, class ExecGroup,
         class CapT = axp::target_cap>
struct count_bits_impl {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "CountBits: ExecGroup must be warp");
    static_assert(iro::contract::ScalarPayload<InPayload>,
                  "CountBits: InPayload must be ScalarPayload");
    static_assert(std::is_same_v<typename InPayload::elem, iro::elem::u32>,
                  "CountBits: InPayload elem must be u32");
    using MaskPayload = iro::contract::ScalarDesc<iro::elem::u32, iro::dist::replicated>;
    using Alias = axp::level0::Alias<
        Recipe, InPayload, InSubj, PrefixSubj, ExecGroup
    >;
    using Vote = axp::level0::Vote<
        Recipe, InPayload, MaskPayload, PrefixSubj, count_bits_mask_subj, ExecGroup, axp::level0::vote::ballot
    >;
    using Popc = axp::level0::Popc<
        Recipe, MaskPayload, count_bits_mask_subj, CountSubj, ExecGroup
    >;
    using Prefix = axp::level0::WarpScan<
        Recipe, InPayload, PrefixSubj, ExecGroup, axp::protocol::reduction::op_add, axp::level0::scan::exclusive
    >;
    using obligations = iro::util::type_list<Alias, Vote, Popc, Prefix>;
    using edges = iro::util::type_list<
        iro::compose::Edge<iro::compose::out_port_ref<Alias, 0>, iro::compose::in_port_ref<Vote, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<Alias, 0>, iro::compose::in_port_ref<Prefix, 0>>,
        iro::compose::Edge<iro::compose::out_port_ref<Vote, 0>, iro::compose::in_port_ref<Popc, 0>>
    >;
    using type = axp::level1::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

} // namespace axp::level1::detail

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level1::registry {

template<class Recipe, class InPayload, class InSubj, class PrefixSubj, class CountSubj, class ExecGroup, class Cap>
struct resolve_impl<CountBitsPattern<Recipe, InPayload, InSubj, PrefixSubj, CountSubj, ExecGroup>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level1::detail::count_bits_impl<
        Recipe, InPayload, InSubj, PrefixSubj, CountSubj, ExecGroup, Cap
    >::type;
};

} // namespace axp::level1::registry
#endif // AXP_LIBRARY_BUILD
