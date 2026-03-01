#pragma once

#include <iro_cuda_ax_core.hpp>
#include "detail/tokens.hpp"
#include "../protocol/reduction/contracts.hpp"
#include "../detail/resources.hpp"

namespace axp::level0 {

namespace scan {
struct inclusive {};
struct exclusive {};
} // namespace scan

namespace detail {
template<class ExecGroup>
struct is_supported_exec_scan_warp : std::false_type {};
template<> struct is_supported_exec_scan_warp<iro::exec::warp> : std::true_type {};

template<class ExecGroup>
struct is_supported_exec_scan_block : std::false_type {};
template<> struct is_supported_exec_scan_block<iro::exec::block> : std::true_type {};

template<class ExecGroup>
struct is_supported_exec_scan_warpgroup : std::false_type {};
template<int Warps> struct is_supported_exec_scan_warpgroup<iro::exec::warpgroup_t<Warps>> : std::true_type {};
} // namespace detail

// Warp-level scan (value -> value)
template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct WarpScan {
    static_assert(detail::is_supported_exec_scan_warp<ExecGroup>::value,
                  "WarpScan requires warp exec group");
    static_assert(detail::is_value_payload<Payload>::value,
                  "WarpScan: Payload must be Fragment/Scalar/Vector");
    static_assert(iro::util::HasId<OpTag>, "WarpScan: OpTag must have id");
    static_assert(std::is_same_v<Mode, scan::inclusive> || std::is_same_v<Mode, scan::exclusive>,
                  "WarpScan: Mode must be inclusive/exclusive");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "WarpScan: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            Subj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, Subj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            Subj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, Subj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Warp-level segmented scan (value -> value) within fixed-width segments
template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode, int SegmentWidth,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct WarpSegmentedScan {
    static_assert(detail::is_supported_exec_scan_warp<ExecGroup>::value,
                  "WarpSegmentedScan requires warp exec group");
    static_assert(detail::is_value_payload<Payload>::value,
                  "WarpSegmentedScan: Payload must be Fragment/Scalar/Vector");
    static_assert(iro::util::HasId<OpTag>, "WarpSegmentedScan: OpTag must have id");
    static_assert(std::is_same_v<Mode, scan::inclusive> || std::is_same_v<Mode, scan::exclusive>,
                  "WarpSegmentedScan: Mode must be inclusive/exclusive");
    static_assert(SegmentWidth > 0 && SegmentWidth <= 32, "WarpSegmentedScan: SegmentWidth must be 1..32");
    static_assert((SegmentWidth & (SegmentWidth - 1)) == 0, "WarpSegmentedScan: SegmentWidth must be power of two");
    static_assert(32 % SegmentWidth == 0, "WarpSegmentedScan: SegmentWidth must divide 32");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "WarpSegmentedScan: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            Subj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, Subj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            Subj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, Subj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Warpgroup-level scan (value -> value)
template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         int BarrierId = 1, int WarpgroupCount = 1,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct WarpgroupScan {
    static_assert(detail::is_supported_exec_scan_warpgroup<ExecGroup>::value,
                  "WarpgroupScan requires warpgroup exec group");
    static_assert(detail::is_value_payload<Payload>::value,
                  "WarpgroupScan: Payload must be Fragment/Scalar/Vector");
    static_assert(iro::util::HasId<OpTag>, "WarpgroupScan: OpTag must have id");
    static_assert(std::is_same_v<Mode, scan::inclusive> || std::is_same_v<Mode, scan::exclusive>,
                  "WarpgroupScan: Mode must be inclusive/exclusive");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "WarpgroupScan: Recipe must be explicit");
    static_assert(BarrierId >= 1 && BarrierId <= 8, "WarpgroupScan: BarrierId must be 1..8");
    static_assert(WarpgroupCount >= 1, "WarpgroupScan: WarpgroupCount must be >= 1");
    static_assert(BarrierId + WarpgroupCount - 1 <= 8,
                  "WarpgroupScan: BarrierId + WarpgroupCount - 1 must be <= 8");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            Subj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, Subj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            Subj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, Subj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::concat_t<
        axp::detail::warpgroup_layout_resources_t<ExecGroup>,
        iro::util::type_list<
            iro::contract::res::warpgroup_count<WarpgroupCount>,
            iro::contract::res::warpgroup_barrier<Subj, BarrierId>
        >
    >;
};

// Block-level scan (value -> value)
template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct BlockScan {
    static_assert(detail::is_supported_exec_scan_block<ExecGroup>::value,
                  "BlockScan requires block exec group");
    static_assert(detail::is_value_payload<Payload>::value,
                  "BlockScan: Payload must be Fragment/Scalar/Vector");
    static_assert(iro::util::HasId<OpTag>, "BlockScan: OpTag must have id");
    static_assert(std::is_same_v<Mode, scan::inclusive> || std::is_same_v<Mode, scan::exclusive>,
                  "BlockScan: Mode must be inclusive/exclusive");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "BlockScan: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            Subj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, Subj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            Subj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, Subj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Block-level chained scan with explicit carry (scalar payloads only)
template<class Recipe, class Payload, class CarryPayload, class Subj, class CarryInSubj, class CarryOutSubj,
         class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct ChainedScan {
    static_assert(detail::is_supported_exec_scan_block<ExecGroup>::value,
                  "ChainedScan requires block exec group");
    static_assert(iro::contract::ScalarPayload<Payload>, "ChainedScan: Payload must be Scalar");
    static_assert(iro::contract::ScalarPayload<CarryPayload>, "ChainedScan: CarryPayload must be Scalar");
    static_assert(std::is_same_v<typename Payload::elem, typename CarryPayload::elem>,
                  "ChainedScan: Payload and CarryPayload elems must match");
    static_assert(iro::util::HasId<OpTag>, "ChainedScan: OpTag must have id");
    static_assert(std::is_same_v<Mode, scan::inclusive> || std::is_same_v<Mode, scan::exclusive>,
                  "ChainedScan: Mode must be inclusive/exclusive");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "ChainedScan: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            Subj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, Subj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            CarryPayload,
            CarryInSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<CarryPayload, CarryInSubj, ExecGroup>, InExtra>,
            typename CarryPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            Subj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, Subj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >,
        iro::contract::OutputPort<
            CarryPayload,
            CarryOutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<CarryPayload, CarryOutSubj, ExecGroup>, OutExtra>,
            typename CarryPayload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

} // namespace axp::level0
