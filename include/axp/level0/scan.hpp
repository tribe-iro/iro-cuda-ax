#pragma once

#include "../protocol/scan/contracts.hpp"

namespace axp::level0 {

namespace scan {
using inclusive = axp::protocol::scan::scan::inclusive;
using exclusive = axp::protocol::scan::scan::exclusive;
} // namespace scan

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
using WarpScan = axp::protocol::scan::WarpScan<
    Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra>;

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode, int SegmentWidth,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
using WarpSegmentedScan = axp::protocol::scan::WarpSegmentedScan<
    Recipe, Payload, Subj, ExecGroup, OpTag, Mode, SegmentWidth, InExtra, OutExtra>;

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         int BarrierId = 1, int WarpgroupCount = 1,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
using WarpgroupScan = axp::protocol::scan::WarpgroupScan<
    Recipe, Payload, Subj, ExecGroup, OpTag, Mode, BarrierId, WarpgroupCount, InExtra, OutExtra>;

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
using BlockScan = axp::protocol::scan::BlockScan<
    Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra>;

template<class Recipe, class Payload, class CarryPayload, class Subj, class CarryInSubj, class CarryOutSubj,
         class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
using ChainedScan = axp::protocol::scan::ChainedScan<
    Recipe, Payload, CarryPayload, Subj, CarryInSubj, CarryOutSubj, ExecGroup, OpTag, Mode, InExtra, OutExtra>;

} // namespace axp::level0

namespace iro::contract {

template<class... Args>
struct is_fused_atom<axp::protocol::scan::WarpScan<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::scan::BlockScan<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::scan::ChainedScan<Args...>> : std::true_type {};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode, int SegmentWidth,
         class InExtra, class OutExtra>
struct is_fused_atom<axp::protocol::scan::WarpSegmentedScan<
    Recipe, Payload, Subj, ExecGroup, OpTag, Mode, SegmentWidth, InExtra, OutExtra>> : std::true_type {};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         int BarrierId, int WarpgroupCount, class InExtra, class OutExtra>
struct is_fused_atom<axp::protocol::scan::WarpgroupScan<
    Recipe, Payload, Subj, ExecGroup, OpTag, Mode, BarrierId, WarpgroupCount, InExtra, OutExtra>> : std::true_type {};

} // namespace iro::contract
