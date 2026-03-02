#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level1/communication.hpp"

namespace axp::level2 {

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         int SrcLane = 0, class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using BroadcastSrc =
    axp::level1::BroadcastSrc<Recipe, Payload, InSubj, OutSubj, ExecGroup, SrcLane, InExtra, OutExtra, CapT>;

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         int SrcLane, class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using Broadcast =
    axp::level1::Broadcast<Recipe, Payload, InSubj, OutSubj, ExecGroup, SrcLane, InExtra, OutExtra, CapT>;

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using BroadcastLane0 =
    axp::level1::BroadcastLane0<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra, CapT>;

template<class Recipe, class FragPayload, class ScalarPayload, class ScalarSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using FragmentBroadcast =
    axp::level1::FragmentBroadcast<Recipe, FragPayload, ScalarPayload, ScalarSubj, OutSubj, ExecGroup, InExtra, OutExtra, CapT>;

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, int BarrierId = 1,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using WarpgroupBroadcastLane0 =
    axp::level1::WarpgroupBroadcastLane0<Recipe, Payload, InSubj, OutSubj, ExecGroup, BarrierId, InExtra, OutExtra, CapT>;

template<class Recipe, class InPayload, class InSubj, class PrefixSubj, class CountSubj, class ExecGroup,
         class CapT = axp::target_cap>
using CountBits = axp::level1::CountBits<Recipe, InPayload, InSubj, PrefixSubj, CountSubj, ExecGroup, CapT>;

} // namespace axp::level2
