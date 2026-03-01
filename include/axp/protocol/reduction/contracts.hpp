#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>

namespace axp::protocol::reduction {

namespace detail {
template<class Payload>
struct is_value_payload : std::bool_constant<
    iro::contract::FragmentPayload<Payload> ||
    iro::contract::ScalarPayload<Payload> ||
    iro::contract::VectorPayload<Payload>
> {};
} // namespace detail

// Warp-reduce operation tags (for fast intrinsic mapping)
struct op_add { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.reduction.op_add"); };
struct op_mul { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.reduction.op_mul"); };
struct op_max { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.reduction.op_max"); };
struct op_min { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.reduction.op_min"); };
struct op_and { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.reduction.op_and"); };
struct op_or  { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.reduction.op_or"); };
struct op_xor { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.reduction.op_xor"); };

// Warp reduction (fragment -> fragment, lane0 valid)
template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag>
struct WarpReduce {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>, "WarpReduce requires warp exec group");
    static_assert(iro::contract::FragmentPayload<Frag>, "WarpReduce expects Fragment payload");
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::in>, "WarpReduce: Frag elem != Recipe::in");
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::out>, "WarpReduce: Frag elem != Recipe::out");
    static_assert(iro::util::HasId<OpTag>, "WarpReduce requires OpTag with id");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Frag,
            Subj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<Subj, iro::token::lifetime::warp>,
                iro::token::lanes_valid<Subj, 32>
            >,
            typename Frag::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Frag,
            Subj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<Subj, iro::token::lifetime::warp>,
                iro::token::lanes_valid<Subj, 1>
            >,
            typename Frag::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Warp all-reduce (fragment -> fragment, all lanes valid)
template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag>
struct WarpAllReduce {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>, "WarpAllReduce requires warp exec group");
    static_assert(iro::contract::FragmentPayload<Frag>, "WarpAllReduce expects Fragment payload");
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::in>, "WarpAllReduce: Frag elem != Recipe::in");
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::out>, "WarpAllReduce: Frag elem != Recipe::out");
    static_assert(iro::util::HasId<OpTag>, "WarpAllReduce requires OpTag with id");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Frag,
            Subj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<Subj, iro::token::lifetime::warp>,
                iro::token::lanes_valid<Subj, 32>
            >,
            typename Frag::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Frag,
            Subj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<Subj, iro::token::lifetime::warp>,
                iro::token::lanes_valid<Subj, 32>
            >,
            typename Frag::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Warp segmented all-reduce (fragment -> fragment, segment width lanes valid)
template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag, int SegmentWidth>
struct WarpSegmentedReduce {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>, "WarpSegmentedReduce requires warp exec group");
    static_assert(iro::contract::FragmentPayload<Frag>, "WarpSegmentedReduce expects Fragment payload");
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::in>, "WarpSegmentedReduce: Frag elem != Recipe::in");
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::out>, "WarpSegmentedReduce: Frag elem != Recipe::out");
    static_assert(iro::util::HasId<OpTag>, "WarpSegmentedReduce requires OpTag with id");
    static_assert(SegmentWidth > 0 && SegmentWidth <= 32, "WarpSegmentedReduce: SegmentWidth must be 1..32");
    static_assert((SegmentWidth & (SegmentWidth - 1)) == 0, "WarpSegmentedReduce: SegmentWidth must be power of two");
    static_assert(32 % SegmentWidth == 0, "WarpSegmentedReduce: SegmentWidth must divide 32");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Frag,
            Subj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<Subj, iro::token::lifetime::warp>,
                iro::token::lanes_valid<Subj, 32>
            >,
            typename Frag::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Frag,
            Subj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<Subj, iro::token::lifetime::warp>,
                iro::token::lanes_valid<Subj, SegmentWidth>
            >,
            typename Frag::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Warp shuffle-tree reduction with explicit mask (value -> value, lane0 valid)
template<class Recipe, class Payload, class MaskPayload, class Subj, class MaskSubj, class ExecGroup, class OpTag>
struct ShuffleReduceTree {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>, "ShuffleReduceTree requires warp exec group");
    static_assert(detail::is_value_payload<Payload>::value,
                  "ShuffleReduceTree expects Fragment/Scalar/Vector payload");
    static_assert(iro::contract::MaskPayload<MaskPayload>, "ShuffleReduceTree requires Mask payload");
    static_assert(MaskPayload::width <= 32, "ShuffleReduceTree supports width <= 32");
    static_assert((MaskPayload::width & (MaskPayload::width - 1)) == 0,
                  "ShuffleReduceTree requires power-of-two mask width");
    static_assert(std::is_same_v<typename Payload::elem, typename Recipe::in>, "ShuffleReduceTree: elem != Recipe::in");
    static_assert(std::is_same_v<typename Payload::elem, typename Recipe::out>, "ShuffleReduceTree: elem != Recipe::out");
    static_assert(iro::util::HasId<OpTag>, "ShuffleReduceTree requires OpTag with id");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            Subj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<Subj, iro::token::lifetime::warp>
            >,
            typename Payload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            MaskPayload,
            MaskSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<MaskSubj, iro::token::lifetime::warp>,
                iro::token::mask_at<MaskSubj, iro::scope::warp>
            >,
            typename MaskPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            Subj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<Subj, iro::token::lifetime::warp>,
                iro::token::lanes_valid<Subj, 1>
            >,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Warpgroup reduction (value -> value, lane0 valid per warpgroup).
// Uses explicit ExecGroup=warpgroup and op tag.
template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag,
         int BarrierId = 1, int WarpgroupCount = 1>
struct WarpgroupReduce {
    static_assert(iro::exec::is_warpgroup_v<ExecGroup>, "WarpgroupReduce requires warpgroup exec group");
    static_assert(detail::is_value_payload<Payload>::value, "WarpgroupReduce expects Fragment/Scalar/Vector payload");
    static_assert(std::is_same_v<typename Payload::elem, typename Recipe::in>, "WarpgroupReduce: elem != Recipe::in");
    static_assert(std::is_same_v<typename Payload::elem, typename Recipe::out>, "WarpgroupReduce: elem != Recipe::out");
    static_assert(iro::util::HasId<OpTag>, "WarpgroupReduce requires OpTag with id");
    static_assert(BarrierId >= 1 && BarrierId <= 8, "WarpgroupReduce: BarrierId must be 1..8");
    static_assert(WarpgroupCount >= 1, "WarpgroupReduce: WarpgroupCount must be >= 1");
    static_assert(BarrierId + WarpgroupCount - 1 <= 8,
                  "WarpgroupReduce: BarrierId + WarpgroupCount - 1 must be <= 8");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            Subj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<Subj, iro::token::lifetime::warpgroup>,
                iro::token::lanes_valid<Subj, 32>,
                iro::token::warps_valid<Subj, iro::exec::warpgroup_warps<ExecGroup>::value>,
                iro::token::warpgroup_participates<Subj, iro::exec::warpgroup_warps<ExecGroup>::value>
            >,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            Subj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<Subj, iro::token::lifetime::warpgroup>,
                iro::token::lanes_valid<Subj, 1>,
                iro::token::warps_valid<Subj, iro::exec::warpgroup_warps<ExecGroup>::value>,
                iro::token::warpgroup_participates<Subj, iro::exec::warpgroup_warps<ExecGroup>::value>
            >,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::block_threads_multiple_of<iro::exec::warpgroup_warps<ExecGroup>::value * 32>,
        iro::contract::res::warpgroup_layout<iro::exec::warpgroup_warps<ExecGroup>::value, 32>,
        iro::contract::res::warpgroup_linear_x,
        iro::contract::res::warpgroup_count<WarpgroupCount>,
        iro::contract::res::warpgroup_barrier<Subj, BarrierId>
    >;
};

// Block reduction (global -> global) for associative ops

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup>
struct BlockReduce {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BlockReduce requires block exec group");
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>, "BlockReduce: InTile elem != Recipe::in");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::out>, "BlockReduce: OutTile elem != Recipe::out");
    static_assert(InTile::shape::rank == 1, "BlockReduce expects rank-1 input tile");
    static_assert(OutTile::shape::rank == 1, "BlockReduce expects rank-1 output tile");
    static_assert(OutTile::shape::size == 1, "BlockReduce expects scalar output tile");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<InSubj, iro::scope::device>,
                iro::token::alive<InSubj, iro::token::lifetime::block>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<OutSubj, iro::scope::device>,
                iro::token::alive<OutSubj, iro::token::lifetime::block>,
                iro::token::sync_at<OutSubj, iro::scope::block>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

} // namespace axp::protocol::reduction
