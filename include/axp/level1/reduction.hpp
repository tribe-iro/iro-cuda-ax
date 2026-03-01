#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level0/communication.hpp"
#include "../level0/compute.hpp"
#include "../level0/ownership.hpp"
#include "../level0/memory.hpp"
#include "../level0/reduction.hpp"
#include "registry.hpp"
#include "detail/compose.hpp"

namespace axp::level1 {

namespace detail {

template<class Recipe, class Frag, class Subj, class ExecGroup,
         template<class, class, class, class, class, class, class, class> class Op,
         class CapT = axp::target_cap, class Tag = void>
struct warp_reduce_impl {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "WarpReduce: ExecGroup must be warp");
    static_assert(std::is_same_v<Tag, void> || iro::util::HasId<Tag>,
                  "WarpReduce: Tag must provide id");

    using tag_tokens = std::conditional_t<
        std::is_same_v<Tag, void>,
        iro::util::type_list<>,
        iro::util::type_list<iro::token::version<Tag, 0>>
    >;

    using S16 = axp::level0::Shuffle<
        Recipe, Frag, Subj, Subj, ExecGroup, axp::level0::shuffle::down, 16,
        iro::util::type_list<>, tag_tokens
    >;
    struct O16 : Op<Recipe, Frag, Subj, Subj, Subj, ExecGroup,
                    iro::util::type_list<>, tag_tokens> {};
    using S8 = axp::level0::Shuffle<
        Recipe, Frag, Subj, Subj, ExecGroup, axp::level0::shuffle::down, 8,
        iro::util::type_list<>, tag_tokens
    >;
    struct O8 : Op<Recipe, Frag, Subj, Subj, Subj, ExecGroup,
                   iro::util::type_list<>, tag_tokens> {};
    using S4 = axp::level0::Shuffle<
        Recipe, Frag, Subj, Subj, ExecGroup, axp::level0::shuffle::down, 4,
        iro::util::type_list<>, tag_tokens
    >;
    struct O4 : Op<Recipe, Frag, Subj, Subj, Subj, ExecGroup,
                   iro::util::type_list<>, tag_tokens> {};
    using S2 = axp::level0::Shuffle<
        Recipe, Frag, Subj, Subj, ExecGroup, axp::level0::shuffle::down, 2,
        iro::util::type_list<>, tag_tokens
    >;
    struct O2 : Op<Recipe, Frag, Subj, Subj, Subj, ExecGroup,
                   iro::util::type_list<>, tag_tokens> {};
    using S1 = axp::level0::Shuffle<
        Recipe, Frag, Subj, Subj, ExecGroup, axp::level0::shuffle::down, 1,
        iro::util::type_list<>, tag_tokens
    >;
    struct O1 : Op<
        Recipe,
        Frag,
        Subj,
        Subj,
        Subj,
        ExecGroup,
        iro::util::type_list<>,
        iro::util::concat_t<tag_tokens, iro::util::type_list<iro::token::lanes_valid<Subj, 1>>>
    > {};

    using obligations = iro::util::type_list<S16, O16, S8, O8, S4, O4, S2, O2, S1, O1>;

    using edges = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<S16, 0>, detail::in_port_t<O16, 1>>,
        iro::compose::Edge<detail::out_port_t<O16, 0>, detail::in_port_t<S8, 0>>,
        iro::compose::Edge<detail::out_port_t<S8, 0>, detail::in_port_t<O8, 1>>,
        iro::compose::Edge<detail::out_port_t<O8, 0>, detail::in_port_t<S4, 0>>,
        iro::compose::Edge<detail::out_port_t<S4, 0>, detail::in_port_t<O4, 1>>,
        iro::compose::Edge<detail::out_port_t<O4, 0>, detail::in_port_t<S2, 0>>,
        iro::compose::Edge<detail::out_port_t<S2, 0>, detail::in_port_t<O2, 1>>,
        iro::compose::Edge<detail::out_port_t<O2, 0>, detail::in_port_t<S1, 0>>,
        iro::compose::Edge<detail::out_port_t<S1, 0>, detail::in_port_t<O1, 1>>
    >;

    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

// Block-level reduction (two-phase warp reduction with shared handoff).
template<class Recipe, class Frag, class SmemTile, class Subj, class SmemSubj, class ExecGroup,
         template<class, class, class, class, class, class, class, class> class Op = axp::level0::Add,
         class CapT = axp::target_cap>
struct block_reduce_impl {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "BlockReduce: ExecGroup must be warp for warp-level steps");
    static_assert(std::is_same_v<typename SmemTile::space, iro::contract::space::shared>,
                  "BlockReduce: SmemTile must be shared space");

    using WarpStage = detail::warp_reduce_impl<Recipe, Frag, Subj, ExecGroup, Op, CapT>;

    using FragToTile = axp::level0::FragmentToSharedTile<
        Recipe,
        Frag,
        SmemTile,
        Subj,
        SmemSubj,
        ExecGroup,
        iro::token::lifetime::warp
    >;

    using Fence = axp::level0::TileFence<
        Recipe,
        SmemTile,
        SmemSubj,
        iro::exec::block
    >;

    using TileToFrag = axp::level0::SharedTileToFragment<
        Recipe,
        SmemTile,
        Frag,
        SmemSubj,
        Subj,
        ExecGroup,
        iro::token::lifetime::block,
        iro::util::type_list<
            iro::token::sync_at<SmemSubj, iro::scope::block>
        >
    >;

    struct stage2_tag {
        static constexpr auto id =
            iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.level1.block_reduce.stage2"), Subj::id);
    };
    using WarpStage2 = detail::warp_reduce_impl<Recipe, Frag, Subj, ExecGroup, Op, CapT, stage2_tag>;

    using obligations = iro::util::concat_t<
        typename WarpStage::obligations,
        iro::util::concat_t<
            iro::util::type_list<FragToTile, Fence, TileToFrag>,
            typename WarpStage2::obligations
        >
    >;

    using edges = iro::util::concat_t<
        typename WarpStage::edges,
        iro::util::concat_t<
            typename WarpStage2::edges,
            iro::util::type_list<
                iro::compose::Edge<detail::out_port_t<typename WarpStage::O1, 0>, detail::in_port_t<FragToTile, 0>>,
                iro::compose::Edge<detail::out_port_t<FragToTile, 0>, detail::in_port_t<Fence, 0>>,
                iro::compose::Edge<detail::out_port_t<Fence, 0>, detail::in_port_t<TileToFrag, 0>>,
                iro::compose::Edge<detail::out_port_t<TileToFrag, 0>, detail::in_port_t<typename WarpStage2::S16, 0>>
            >
        >
    >;

    using type = detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

} // namespace detail

// Warp-level reduction (one value per lane, reduced to lane 0).
template<class Recipe, class Frag, class Subj, class ExecGroup,
         template<class, class, class, class, class, class, class, class> class Op = axp::level0::Add,
         class CapT = axp::target_cap>
using WarpReduce = registry::Select<registry::WarpReducePattern<Recipe, Frag, Subj, ExecGroup, Op>, CapT>;

template<class Recipe, class Frag, class SmemTile, class Subj, class SmemSubj, class ExecGroup,
         template<class, class, class, class, class, class, class, class> class Op = axp::level0::Add,
         class CapT = axp::target_cap>
using BlockReduce = registry::Select<registry::BlockReducePattern<Recipe, Frag, SmemTile, Subj, SmemSubj, ExecGroup, Op>, CapT>;

// Warp all-reduce (fragment -> fragment, all lanes valid)
template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag, class CapT = axp::target_cap>
using WarpAllReduce = registry::Select<registry::WarpAllReducePattern<Recipe, Frag, Subj, ExecGroup, OpTag>, CapT>;

// Warp segmented reduce (fragment -> fragment, segment width lanes valid)
template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag, int SegmentWidth, class CapT = axp::target_cap>
using WarpSegmentedReduce = registry::Select<registry::WarpSegmentedReducePattern<Recipe, Frag, Subj, ExecGroup, OpTag, SegmentWidth>, CapT>;

// Warpgroup reduction (fragment -> fragment, lane0 valid)
template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag,
         int BarrierId = 1, int WarpgroupCount = 1, class CapT = axp::target_cap>
using WarpgroupReduce =
    registry::Select<registry::WarpgroupReducePattern<Recipe, Frag, Subj, ExecGroup, OpTag, BarrierId, WarpgroupCount>, CapT>;

// Shuffle-reduce tree (fragment -> fragment, lane0 valid)
template<class Recipe, class Payload, class MaskPayload, class Subj, class MaskSubj, class ExecGroup, class OpTag,
         class CapT = axp::target_cap>
using ShuffleReduceTree =
    registry::Select<registry::ShuffleReduceTreePattern<Recipe, Payload, MaskPayload, Subj, MaskSubj, ExecGroup, OpTag>, CapT>;

} // namespace axp::level1

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level1::registry {

template<class Recipe, class Frag, class Subj, class ExecGroup,
         template<class, class, class, class, class, class, class, class> class Op, class Cap>
struct resolve_impl<WarpReducePattern<Recipe, Frag, Subj, ExecGroup, Op>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level1::detail::warp_reduce_impl<Recipe, Frag, Subj, ExecGroup, Op, Cap>::type;
};

template<class Recipe, class Frag, class SmemTile, class Subj, class SmemSubj, class ExecGroup,
         template<class, class, class, class, class, class, class, class> class Op, class Cap>
struct resolve_impl<BlockReducePattern<Recipe, Frag, SmemTile, Subj, SmemSubj, ExecGroup, Op>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level1::detail::block_reduce_impl<Recipe, Frag, SmemTile, Subj, SmemSubj, ExecGroup, Op, Cap>::type;
};

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag, class Cap>
struct resolve_impl<WarpAllReducePattern<Recipe, Frag, Subj, ExecGroup, OpTag>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::fused::WarpAllReduce<Recipe, Frag, Subj, ExecGroup, OpTag>
    , Cap>;
};

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag, int SegmentWidth, class Cap>
struct resolve_impl<WarpSegmentedReducePattern<Recipe, Frag, Subj, ExecGroup, OpTag, SegmentWidth>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::fused::WarpSegmentedReduce<Recipe, Frag, Subj, ExecGroup, OpTag, SegmentWidth>
    , Cap>;
};

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag,
         int BarrierId, int WarpgroupCount, class Cap>
struct resolve_impl<WarpgroupReducePattern<Recipe, Frag, Subj, ExecGroup, OpTag, BarrierId, WarpgroupCount>, Cap> {
    static constexpr bool supported = true;
    static_assert(iro::exec::warpgroup_warps<ExecGroup>::value <= Cap::warpgroup_warps,
                  "WarpgroupReduce: ExecGroup warps exceed target capability");
    using type = axp::level1::detail::as_composition_t<
        axp::level0::fused::WarpgroupReduce<Recipe, Frag, Subj, ExecGroup, OpTag, BarrierId, WarpgroupCount>
    , Cap>;
};

template<class Recipe, class Payload, class MaskPayload, class Subj, class MaskSubj, class ExecGroup, class OpTag, class Cap>
struct resolve_impl<ShuffleReduceTreePattern<Recipe, Payload, MaskPayload, Subj, MaskSubj, ExecGroup, OpTag>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::fused::ShuffleReduceTree<Recipe, Payload, MaskPayload, Subj, MaskSubj, ExecGroup, OpTag>
    , Cap>;
};

} // namespace axp::level1::registry
#endif // AXP_LIBRARY_BUILD
