#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level0/scan.hpp"
#include "registry.hpp"
#include "detail/compose.hpp"

namespace axp::level1 {

// Warp-level scan (value -> value)
template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using WarpScan = registry::Select<
    registry::WarpScanPattern<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra>, CapT>;

// Warp-level segmented scan (value -> value) within fixed-width segments
template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode, int SegmentWidth,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using WarpSegmentedScan = registry::Select<
    registry::WarpSegmentedScanPattern<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, SegmentWidth, InExtra, OutExtra>, CapT>;

// Warpgroup-level scan (value -> value)
template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         int BarrierId = 1, int WarpgroupCount = 1,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using WarpgroupScan = registry::Select<
    registry::WarpgroupScanPattern<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, BarrierId, WarpgroupCount, InExtra, OutExtra>, CapT>;

// Block-level scan (value -> value)
template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using BlockScan = registry::Select<
    registry::BlockScanPattern<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra>, CapT>;

// Block-level chained scan with explicit carry (scalar payloads only)
template<class Recipe, class Payload, class CarryPayload, class Subj, class CarryInSubj, class CarryOutSubj,
         class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using ChainedScan = registry::Select<
    registry::ChainedScanPattern<Recipe, Payload, CarryPayload, Subj, CarryInSubj, CarryOutSubj,
                                 ExecGroup, OpTag, Mode, InExtra, OutExtra>, CapT
>;

} // namespace axp::level1

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level1::registry {

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra, class OutExtra, class Cap>
struct resolve_impl<WarpScanPattern<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::WarpScan<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra>,
        Cap
    >;
};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode, int SegmentWidth,
         class InExtra, class OutExtra, class Cap>
struct resolve_impl<WarpSegmentedScanPattern<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, SegmentWidth, InExtra, OutExtra>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::WarpSegmentedScan<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, SegmentWidth, InExtra, OutExtra>,
        Cap
    >;
};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         int BarrierId, int WarpgroupCount, class InExtra, class OutExtra, class Cap>
struct resolve_impl<WarpgroupScanPattern<
    Recipe, Payload, Subj, ExecGroup, OpTag, Mode, BarrierId, WarpgroupCount, InExtra, OutExtra>, Cap> {
    static constexpr bool supported = true;
    static_assert(iro::exec::warpgroup_warps<ExecGroup>::value <= Cap::warpgroup_warps,
                  "WarpgroupScan: ExecGroup warps exceed target capability");
    using type = axp::level1::detail::as_composition_t<
        axp::level0::WarpgroupScan<
            Recipe, Payload, Subj, ExecGroup, OpTag, Mode, BarrierId, WarpgroupCount, InExtra, OutExtra>,
        Cap
    >;
};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra, class OutExtra, class Cap>
struct resolve_impl<BlockScanPattern<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::BlockScan<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra>,
        Cap
    >;
};

template<class Recipe, class Payload, class CarryPayload, class Subj, class CarryInSubj, class CarryOutSubj,
         class ExecGroup, class OpTag, class Mode, class InExtra, class OutExtra, class Cap>
struct resolve_impl<ChainedScanPattern<Recipe, Payload, CarryPayload, Subj, CarryInSubj, CarryOutSubj,
                                  ExecGroup, OpTag, Mode, InExtra, OutExtra>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::ChainedScan<Recipe, Payload, CarryPayload, Subj, CarryInSubj, CarryOutSubj,
                                 ExecGroup, OpTag, Mode, InExtra, OutExtra>,
        Cap
    >;
};

} // namespace axp::level1::registry
#endif // AXP_LIBRARY_BUILD
