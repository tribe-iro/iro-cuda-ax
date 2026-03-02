#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level1/scan.hpp"

namespace axp::level2 {

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using WarpScan = axp::level1::WarpScan<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra, CapT>;

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode, int SegmentWidth,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using WarpSegmentedScan =
    axp::level1::WarpSegmentedScan<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, SegmentWidth, InExtra, OutExtra, CapT>;

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         int BarrierId = 1, int WarpgroupCount = 1,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using WarpgroupScan = axp::level1::WarpgroupScan<
    Recipe, Payload, Subj, ExecGroup, OpTag, Mode, BarrierId, WarpgroupCount, InExtra, OutExtra, CapT>;

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using BlockScan = axp::level1::BlockScan<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra, CapT>;

template<class Recipe, class Payload, class CarryPayload, class Subj, class CarryInSubj, class CarryOutSubj,
         class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using ChainedScan = axp::level1::ChainedScan<
    Recipe, Payload, CarryPayload, Subj, CarryInSubj, CarryOutSubj, ExecGroup, OpTag, Mode, InExtra, OutExtra, CapT>;

} // namespace axp::level2
