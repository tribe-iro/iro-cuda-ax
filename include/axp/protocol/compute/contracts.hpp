#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>
#include "../../detail/resources.hpp"
#include "../../detail/participation_tokens.hpp"
#include "../scale/contracts.hpp"
#include "../ownership/dist_tags.hpp"
#include "bundles.hpp"
#include "detail/mma_shapes.hpp"

namespace axp::protocol::compute {

// MMA shape tag (explicit contract-level shape)
// ElemA/ElemB may differ for mixed-precision MMA (e.g., FP8).
template<int M, int N, int K, class ElemA, class ElemB, class Acc, class LayoutA = void, class LayoutB = void>
struct MmaShape {
    static_assert(M > 0 && N > 0 && K > 0, "MmaShape: dimensions must be positive");
    static_assert(detail::is_valid_mma_shape_v<M, N, K, ElemA, ElemB, Acc>,
                  "MmaShape: unsupported shape or element type");
    static constexpr int m = M;
    static constexpr int n = N;
    static constexpr int k = K;
    using elem_a = ElemA;
    using elem_b = ElemB;
    using acc = Acc;
    using layout_a = LayoutA;
    using layout_b = LayoutB;
    static constexpr iro::util::u64 id = iro::util::mix_u64(
        iro::util::fnv1a_64_cstr("axp.compute.mma_shape"),
        iro::util::mix_u64((iro::util::u64)M,
            iro::util::mix_u64((iro::util::u64)N,
                iro::util::mix_u64((iro::util::u64)K,
                    iro::util::mix_u64(ElemA::id,
                        iro::util::mix_u64(ElemB::id, Acc::id))))));
};

namespace detail {
template<class Shape>
struct is_mma_shape : std::false_type {};

template<int M, int N, int K, class ElemA, class ElemB, class Acc, class LayoutA, class LayoutB>
struct is_mma_shape<MmaShape<M, N, K, ElemA, ElemB, Acc, LayoutA, LayoutB>> : std::true_type {};

template<class Shape, class ATile, class BTile, class AccFrag>
consteval bool mma_shape_matches() {
    static_assert(is_mma_shape<Shape>::value, "MMA contract requires MmaShape");
    static_assert(ATile::shape::rank == 2 && BTile::shape::rank == 2 && AccFrag::shape::rank == 2,
                  "MMA contract requires rank-2 tiles/fragments");
    static_assert(ATile::shape::template dim<0>() == Shape::m, "MMA shape mismatch: ATile M");
    static_assert(ATile::shape::template dim<1>() == Shape::k, "MMA shape mismatch: ATile K");
    static_assert(BTile::shape::template dim<0>() == Shape::k, "MMA shape mismatch: BTile K");
    static_assert(BTile::shape::template dim<1>() == Shape::n, "MMA shape mismatch: BTile N");
    static_assert(AccFrag::shape::template dim<0>() == Shape::m, "MMA shape mismatch: AccFrag M");
    static_assert(AccFrag::shape::template dim<1>() == Shape::n, "MMA shape mismatch: AccFrag N");
    static_assert(std::is_same_v<typename ATile::elem, typename Shape::elem_a>,
                  "MMA shape mismatch: ATile elem");
    static_assert(std::is_same_v<typename BTile::elem, typename Shape::elem_b>,
                  "MMA shape mismatch: BTile elem");
    static_assert(std::is_same_v<typename AccFrag::elem, typename Shape::acc>,
                  "MMA shape mismatch: AccFrag elem");
    if constexpr (!std::is_same_v<typename Shape::layout_a, void>) {
        static_assert(std::is_same_v<typename ATile::layout, typename Shape::layout_a>,
                      "MMA shape mismatch: ATile layout");
    }
    if constexpr (!std::is_same_v<typename Shape::layout_b, void>) {
        static_assert(std::is_same_v<typename BTile::layout, typename Shape::layout_b>,
                      "MMA shape mismatch: BTile layout");
    }
    return true;
}

template<class Subject, class ExecGroup>
using participation_tokens = axp::detail::participation_tokens<Subject, ExecGroup>;
} // namespace detail

// Warp MMA from shared memory tiles
// A,B tiles must be visible at warp scope
// Accumulator fragment is produced in registers

template<class Recipe, class Shape, class ATile, class BTile, class AccFrag, class ASubj, class BSubj, class AccSubj>
struct WarpMmaFromSmem {
    using ExecGroup = iro::exec::warp;
    static_assert(detail::mma_shape_matches<Shape, ATile, BTile, AccFrag>(),
                  "WarpMmaFromSmem: shape mismatch");
    static_assert(std::is_same_v<typename ATile::elem, iro::verify::recipe_in_a_t<Recipe>>,
                  "WarpMmaFromSmem: ATile elem != Recipe::in_a");
    static_assert(std::is_same_v<typename BTile::elem, iro::verify::recipe_in_b_t<Recipe>>,
                  "WarpMmaFromSmem: BTile elem != Recipe::in_b");
    static_assert(std::is_same_v<typename AccFrag::elem, typename Recipe::acc>, "WarpMmaFromSmem: AccFrag elem != Recipe::acc");
    static_assert(std::is_same_v<typename ATile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<typename BTile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<typename AccFrag::dist, iro::dist::accumulator>,
                  "WarpMmaFromSmem requires accumulator distribution");
    static_assert(AccFrag::count == (Shape::n / 2),
                  "WarpMmaFromSmem requires AccFrag::count == N/2");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            ATile,
            ASubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<ASubj, iro::scope::warp>,
                iro::token::alive<ASubj, iro::token::lifetime::warp>,
                iro::token::slot_state<ASubj, iro::token::state::ready>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            BTile,
            BSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<BSubj, iro::scope::warp>,
                iro::token::alive<BSubj, iro::token::lifetime::warp>,
                iro::token::slot_state<BSubj, iro::token::state::ready>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            AccFrag,
            AccSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<iro::token::alive<AccSubj, iro::token::lifetime::warp>>,
                detail::participation_tokens<AccSubj, ExecGroup>
            >,
            typename AccFrag::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Warp MMA from shared tiles without slot_state (for computed/shared tiles).
// Requires only visibility and lifetime tokens.
template<class Recipe, class Shape, class ATile, class BTile, class AccFrag, class ASubj, class BSubj, class AccSubj>
struct WarpMmaFromShared {
    using ExecGroup = iro::exec::warp;
    static_assert(detail::mma_shape_matches<Shape, ATile, BTile, AccFrag>(),
                  "WarpMmaFromShared: shape mismatch");
    static_assert(std::is_same_v<typename ATile::elem, iro::verify::recipe_in_a_t<Recipe>>,
                  "WarpMmaFromShared: ATile elem != Recipe::in_a");
    static_assert(std::is_same_v<typename BTile::elem, iro::verify::recipe_in_b_t<Recipe>>,
                  "WarpMmaFromShared: BTile elem != Recipe::in_b");
    static_assert(std::is_same_v<typename AccFrag::elem, typename Recipe::acc>, "WarpMmaFromShared: AccFrag elem != Recipe::acc");
    static_assert(std::is_same_v<typename ATile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<typename BTile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<typename AccFrag::dist, iro::dist::accumulator>,
                  "WarpMmaFromShared requires accumulator distribution");
    static_assert(AccFrag::count == (Shape::n / 2),
                  "WarpMmaFromShared requires AccFrag::count == N/2");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            ATile,
            ASubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<ASubj, iro::scope::warp>,
                iro::token::alive<ASubj, iro::token::lifetime::warp>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            BTile,
            BSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<BSubj, iro::scope::warp>,
                iro::token::alive<BSubj, iro::token::lifetime::warp>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            AccFrag,
            AccSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<iro::token::alive<AccSubj, iro::token::lifetime::warp>>,
                detail::participation_tokens<AccSubj, ExecGroup>
            >,
            typename AccFrag::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Warpgroup MMA from SMEM descriptors (SM90)
template<class Recipe, class Shape, class ADesc, class BDesc, class AccFrag,
         class ADescSubj, class BDescSubj, class AccSubj, class WgmmaSubj>
struct WarpgroupMmaFromDesc {
    using ExecGroup = iro::exec::warpgroup;
    static_assert(detail::mma_shape_matches<Shape, typename ADesc::tile, typename BDesc::tile, AccFrag>(),
                  "WarpgroupMmaFromDesc: shape mismatch");
    static_assert(std::is_same_v<typename ADesc::tile::elem, iro::verify::recipe_in_a_t<Recipe>>,
                  "WarpgroupMmaFromDesc: A elem != Recipe::in_a");
    static_assert(std::is_same_v<typename BDesc::tile::elem, iro::verify::recipe_in_b_t<Recipe>>,
                  "WarpgroupMmaFromDesc: B elem != Recipe::in_b");
    static_assert(std::is_same_v<typename AccFrag::elem, typename Recipe::acc>, "WarpgroupMmaFromDesc: AccFrag elem != Recipe::acc");
    static_assert(std::is_same_v<typename AccFrag::dist, iro::dist::accumulator>,
                  "WarpgroupMmaFromDesc requires accumulator distribution");
    static_assert(AccFrag::count == (Shape::n / 2),
                  "WarpgroupMmaFromDesc requires AccFrag::count == N/2");
    static_assert(std::is_same_v<typename ADesc::tile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<typename BDesc::tile::space, iro::contract::space::shared>);
    static_assert(iro::util::HasId<WgmmaSubj>, "WarpgroupMmaFromDesc: WgmmaSubj must have id");
    using ASubj = typename ADesc::subject;
    using BSubj = typename BDesc::subject;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            ADesc,
            ADescSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<ADescSubj, iro::token::lifetime::warpgroup>,
                iro::token::visible_at<ASubj, iro::scope::warpgroup>,
                iro::token::alive<ASubj, iro::token::lifetime::warpgroup>,
                iro::token::slot_state<ASubj, iro::token::state::ready>,
                iro::token::sync_at<ASubj, iro::scope::warpgroup>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            BDesc,
            BDescSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<BDescSubj, iro::token::lifetime::warpgroup>,
                iro::token::visible_at<BSubj, iro::scope::warpgroup>,
                iro::token::alive<BSubj, iro::token::lifetime::warpgroup>,
                iro::token::slot_state<BSubj, iro::token::state::ready>,
                iro::token::sync_at<BSubj, iro::scope::warpgroup>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            axp::protocol::compute::WgmmaHandle,
            WgmmaSubj,
            ExecGroup,
            iro::util::type_list<
                axp::protocol::compute::wgmma_fenced<WgmmaSubj>,
                iro::token::alive<WgmmaSubj, iro::token::lifetime::warpgroup>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            AccFrag,
            AccSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<iro::token::alive<AccSubj, iro::token::lifetime::warpgroup>>,
                detail::participation_tokens<AccSubj, ExecGroup>
            >,
            typename AccFrag::dist,
            Recipe
        >,
        iro::contract::OutputPort<
            axp::protocol::compute::WgmmaHandle,
            WgmmaSubj,
            ExecGroup,
            iro::util::type_list<
                axp::protocol::compute::wgmma_issued<WgmmaSubj>,
                iro::token::alive<WgmmaSubj, iro::token::lifetime::warpgroup>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

} // namespace axp::protocol::compute

#if defined(AXP_ENABLE_SM100)
// Blackwell tcgen05 MMA (CTA-group)
// A/B in shared, accumulator in TMEM.
namespace axp::protocol::compute {

template<class Recipe, class ATile, class BTile, class AccTile, class ASubj, class BSubj, class AccSubj,
         class ExecGroup, class AccDist,
         class ScaleASubj = iro::contract::subject::global,
         class ScaleBSubj = iro::contract::subject::global>
struct Tcgen05Mma {
    static_assert(std::is_same_v<ExecGroup, iro::exec::cta_group1> ||
                  std::is_same_v<ExecGroup, iro::exec::cta_group2>,
                  "Tcgen05Mma requires CTA-group exec");
    static_assert(std::is_same_v<typename ATile::elem, iro::verify::recipe_in_a_t<Recipe>>,
                  "Tcgen05Mma: ATile elem != Recipe::in_a");
    static_assert(std::is_same_v<typename BTile::elem, iro::verify::recipe_in_b_t<Recipe>>,
                  "Tcgen05Mma: BTile elem != Recipe::in_b");
    static_assert(std::is_same_v<typename AccTile::elem, typename Recipe::acc>, "Tcgen05Mma: AccTile elem != Recipe::acc");
    static_assert(std::is_same_v<typename ATile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<typename BTile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<typename AccTile::space, iro::contract::space::tmem>);
    static_assert(iro::util::HasId<AccDist>, "Tcgen05Mma: AccDist must have id");

    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;
    using in_tokens = iro::util::type_list<
        iro::token::visible_at<ASubj, scope_t>,
        iro::token::alive<ASubj, iro::token::lifetime::cluster>,
        iro::token::slot_state<ASubj, iro::token::state::ready>,
        iro::token::sync_at<ASubj, scope_t>
    >;
    using in_tokens_b = iro::util::type_list<
        iro::token::visible_at<BSubj, scope_t>,
        iro::token::alive<BSubj, iro::token::lifetime::cluster>,
        iro::token::slot_state<BSubj, iro::token::state::ready>,
        iro::token::sync_at<BSubj, scope_t>
    >;

    using out_tokens = iro::util::type_list<
        iro::token::visible_at<AccSubj, scope_t>,
        iro::token::alive<AccSubj, iro::token::lifetime::cluster>,
        iro::token::sync_at<AccSubj, scope_t>
    >;

    template<bool HasScale, class ScaleElem, int ScaleVec, class ScaleSubj, class Dummy = void>
    struct scale_input_list {
        using type = iro::util::type_list<>;
    };
    template<class ScaleElem, int ScaleVec, class ScaleSubj, class Dummy>
    struct scale_input_list<true, ScaleElem, ScaleVec, ScaleSubj, Dummy> {
        using ScaleTile = axp::protocol::scale::ScaleTile<ScaleElem, ScaleVec>;
        using type = iro::util::type_list<
            iro::contract::InputPort<
                ScaleTile,
                ScaleSubj,
                ExecGroup,
                iro::util::type_list<
                    iro::token::visible_at<ScaleSubj, scope_t>,
                    iro::token::alive<ScaleSubj, iro::token::lifetime::cluster>
                >,
                iro::contract::no_dist,
                Recipe
            >
        >;
    };

    using base_inputs = iro::util::type_list<
        iro::contract::InputPort<
            ATile,
            ASubj,
            ExecGroup,
            in_tokens,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            BTile,
            BSubj,
            ExecGroup,
            in_tokens_b,
            iro::contract::no_dist,
            Recipe
        >
    >;

    static constexpr bool has_scale_a = iro::verify::recipe_has_scale_a_v<Recipe>;
    static constexpr bool has_scale_b = iro::recipe::is_precision_ab_v<Recipe> &&
                                        iro::verify::recipe_has_scale_b_v<Recipe>;

    using scale_inputs_a = typename scale_input_list<
        has_scale_a,
        iro::verify::recipe_scale_a_t<Recipe>,
        iro::verify::recipe_scale_vec_a_v<Recipe>,
        ScaleASubj
    >::type;

    using scale_inputs_b = typename scale_input_list<
        has_scale_b,
        iro::verify::recipe_scale_b_t<Recipe>,
        iro::verify::recipe_scale_vec_b_v<Recipe>,
        ScaleBSubj
    >::type;

    using scale_inputs = typename iro::util::concat<scale_inputs_a, scale_inputs_b>::type;

    using inputs = typename iro::util::concat<base_inputs, scale_inputs>::type;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            AccTile,
            AccSubj,
            ExecGroup,
            out_tokens,
            AccDist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

} // namespace axp::protocol::compute
#endif
