#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>
#include "../../detail/resources.hpp"
#include "../../detail/participation_tokens.hpp"

namespace axp::protocol::convert {

namespace detail {

template<class Subject, class ExecGroup>
using participation_tokens = axp::detail::participation_tokens<Subject, ExecGroup>;

template<class Elem>
inline constexpr bool is_fp8_elem =
    iro::recipe::is_fp8_elem_v<Elem>;

template<class R>
consteval bool fp8_policy_ok() {
    if constexpr (iro::recipe::has_fp8_policy<R>::value) {
        return iro::recipe::implemented_fp8_policy_v<typename R::fp8_policy>;
    } else {
        return false;
    }
}
} // namespace detail

// Explicit tile cast (no hidden effects).
// Converts element type while preserving shape/layout/space.
template<class RecipeIn, class RecipeOut, class InTile, class OutTile,
         class InSubj, class OutSubj, class ExecGroup, int VecBytes,
         class InDist = iro::contract::no_dist, class OutDist = iro::contract::no_dist>
struct CastTile {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp> ||
                  iro::exec::is_warpgroup_v<ExecGroup> ||
                  std::is_same_v<ExecGroup, iro::exec::block> ||
                  std::is_same_v<ExecGroup, iro::exec::cta_group1> ||
                  std::is_same_v<ExecGroup, iro::exec::cta_group2>,
                  "CastTile: ExecGroup must be warp/warpgroup/block/cta_group");
    static_assert(std::is_same_v<typename InTile::elem, typename RecipeIn::out>,
                  "CastTile: InTile elem must match RecipeIn::out");
    static_assert(std::is_same_v<typename OutTile::elem, typename RecipeOut::in>,
                  "CastTile: OutTile elem must match RecipeOut::in");
    static_assert(std::is_same_v<typename InTile::space, typename OutTile::space>,
                  "CastTile: spaces must match");
    static_assert(std::is_same_v<typename InTile::layout, typename OutTile::layout>,
                  "CastTile: layouts must match");
    static_assert(InTile::shape::rank == OutTile::shape::rank, "CastTile: rank mismatch");
    static_assert(InTile::shape::size == OutTile::shape::size, "CastTile: size mismatch");
    static_assert(VecBytes == 4 || VecBytes == 8 || VecBytes == 16,
                  "CastTile: VecBytes must be 4, 8, or 16");
    static_assert(RecipeIn::vec_bytes == VecBytes, "CastTile: RecipeIn vec_bytes mismatch");
    static_assert(VecBytes % InTile::elem::bytes == 0, "CastTile: VecBytes must divide input elem bytes");
    static_assert(!detail::is_fp8_elem<typename InTile::elem> ||
                  (iro::recipe::has_fp8_policy<RecipeIn>::value && detail::fp8_policy_ok<RecipeIn>()),
                  "CastTile: FP8 input requires explicit, supported fp8_policy in RecipeIn");
    static_assert(!detail::is_fp8_elem<typename OutTile::elem> ||
                  (iro::recipe::has_fp8_policy<RecipeOut>::value && detail::fp8_policy_ok<RecipeOut>()),
                  "CastTile: FP8 output requires explicit, supported fp8_policy in RecipeOut");
    static_assert(iro::util::HasId<InDist>, "CastTile: InDist must have id");
    static_assert(iro::util::HasId<OutDist>, "CastTile: OutDist must have id");
    static_assert(!std::is_same_v<typename InTile::space, iro::contract::space::reg> ||
                  !std::is_same_v<InDist, iro::contract::no_dist>,
                  "CastTile: reg InTile requires explicit InDist");
    static_assert(!std::is_same_v<typename OutTile::space, iro::contract::space::reg> ||
                  !std::is_same_v<OutDist, iro::contract::no_dist>,
                  "CastTile: reg OutTile requires explicit OutDist");
    static_assert(!std::is_same_v<typename InTile::space, iro::contract::space::tmem> ||
                  !std::is_same_v<InDist, iro::contract::no_dist>,
                  "CastTile: tmem InTile requires explicit InDist");
    static_assert(!std::is_same_v<typename OutTile::space, iro::contract::space::tmem> ||
                  !std::is_same_v<OutDist, iro::contract::no_dist>,
                  "CastTile: tmem OutTile requires explicit OutDist");

    static constexpr int in_bytes = InTile::elem::bytes;
    static constexpr int out_bytes = OutTile::elem::bytes;
    static constexpr int out_vec_bytes = (VecBytes * out_bytes) / in_bytes;
    static_assert((VecBytes * out_bytes) % in_bytes == 0, "CastTile: output vector bytes must be integral");
    static_assert(RecipeOut::vec_bytes == out_vec_bytes, "CastTile: RecipeOut vec_bytes mismatch");
    static_assert(out_vec_bytes == 4 || out_vec_bytes == 8 || out_vec_bytes == 16,
                  "CastTile: output vector bytes must be 4, 8, or 16");
    static_assert(InTile::align::bytes >= VecBytes, "CastTile: input alignment too small");
    static_assert(OutTile::align::bytes >= out_vec_bytes, "CastTile: output alignment too small");
    static_assert((InTile::bytes % VecBytes) == 0, "CastTile: input bytes must be a multiple of VecBytes");

    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;
    using lifetime_t = std::conditional_t<
        std::is_same_v<ExecGroup, iro::exec::warp>,
        iro::token::lifetime::warp,
        std::conditional_t<
            iro::exec::is_warpgroup_v<ExecGroup>,
            iro::token::lifetime::warpgroup,
            std::conditional_t<
                std::is_same_v<ExecGroup, iro::exec::block>,
                iro::token::lifetime::block,
                iro::token::lifetime::cluster
            >
        >
    >;
    using base_out_tokens = iro::util::concat_t<
        iro::util::type_list<
            iro::token::visible_at<OutSubj, scope_t>,
            iro::token::alive<OutSubj, lifetime_t>
        >,
        detail::participation_tokens<OutSubj, ExecGroup>
    >;
    using sync_out_tokens = iro::util::concat_t<
        iro::util::type_list<
            iro::token::visible_at<OutSubj, scope_t>,
            iro::token::alive<OutSubj, lifetime_t>,
            iro::token::sync_at<OutSubj, scope_t>
        >,
        detail::participation_tokens<OutSubj, ExecGroup>
    >;
    using out_tokens = std::conditional_t<(scope_t::level >= iro::scope::warpgroup::level),
                                          sync_out_tokens, base_out_tokens>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<InSubj, scope_t>,
                    iro::token::alive<InSubj, lifetime_t>
                >,
                detail::participation_tokens<InSubj, ExecGroup>
            >,
            InDist,
            RecipeIn
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            out_tokens,
            OutDist,
            RecipeOut
        >
    >;

    using resources = iro::util::type_list<>;
};

// Explicit fragment cast (register-to-register).
template<class RecipeIn, class RecipeOut, class InFrag, class OutFrag,
         class InSubj, class OutSubj, class ExecGroup>
struct CastFragment {
    static_assert(std::is_same_v<typename InFrag::elem, typename RecipeIn::acc>,
                  "CastFragment: InFrag elem must match RecipeIn::acc");
    static_assert(std::is_same_v<typename OutFrag::elem, typename RecipeOut::acc>,
                  "CastFragment: OutFrag elem must match RecipeOut::acc");
    static_assert(std::is_same_v<typename InFrag::dist, typename OutFrag::dist>,
                  "CastFragment: dist must match");
    static_assert(InFrag::shape::rank == OutFrag::shape::rank, "CastFragment: rank mismatch");
    static_assert(InFrag::shape::size == OutFrag::shape::size, "CastFragment: size mismatch");
    static_assert(InFrag::count == OutFrag::count, "CastFragment: count mismatch");
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp> ||
                  iro::exec::is_warpgroup_v<ExecGroup>,
                  "CastFragment: ExecGroup must be warp or warpgroup");
    static_assert(!detail::is_fp8_elem<typename InFrag::elem> ||
                  (iro::recipe::has_fp8_policy<RecipeIn>::value && detail::fp8_policy_ok<RecipeIn>()),
                  "CastFragment: FP8 input requires explicit, supported fp8_policy in RecipeIn");
    static_assert(!detail::is_fp8_elem<typename OutFrag::elem> ||
                  (iro::recipe::has_fp8_policy<RecipeOut>::value && detail::fp8_policy_ok<RecipeOut>()),
                  "CastFragment: FP8 output requires explicit, supported fp8_policy in RecipeOut");

    using lifetime = std::conditional_t<
        std::is_same_v<ExecGroup, iro::exec::warp>,
        iro::token::lifetime::warp,
        iro::token::lifetime::warpgroup
    >;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InFrag,
            InSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<iro::token::alive<InSubj, lifetime>>,
                detail::participation_tokens<InSubj, ExecGroup>
            >,
            typename InFrag::dist,
            RecipeIn
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutFrag,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<iro::token::alive<OutSubj, lifetime>>,
                detail::participation_tokens<OutSubj, ExecGroup>
            >,
            typename OutFrag::dist,
            RecipeOut
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Fragment -> Vector slice (register-to-register).
template<class Recipe, class Frag, class Vec,
         class FragSubj, class VecSubj, class ExecGroup, int Offset>
struct FragmentToVectorSlice {
    static_assert(iro::contract::FragmentPayload<Frag>, "FragmentToVectorSlice requires Fragment payload");
    static_assert(iro::contract::VectorPayload<Vec>, "FragmentToVectorSlice requires Vector payload");
    static_assert(std::is_same_v<typename Frag::elem, typename Vec::elem>,
                  "FragmentToVectorSlice: element type mismatch");
    static_assert(std::is_same_v<typename Frag::dist, typename Vec::dist>,
                  "FragmentToVectorSlice: dist mismatch");
    static_assert(Offset >= 0, "FragmentToVectorSlice: Offset must be non-negative");
    static_assert((Offset + static_cast<int>(Vec::lanes)) <= static_cast<int>(Frag::count),
                  "FragmentToVectorSlice: slice exceeds fragment bounds");
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp> ||
                  iro::exec::is_warpgroup_v<ExecGroup>,
                  "FragmentToVectorSlice: ExecGroup must be warp or warpgroup");

    using lifetime = std::conditional_t<
        std::is_same_v<ExecGroup, iro::exec::warp>,
        iro::token::lifetime::warp,
        iro::token::lifetime::warpgroup
    >;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Frag,
            FragSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<iro::token::alive<FragSubj, lifetime>>,
                detail::participation_tokens<FragSubj, ExecGroup>
            >,
            typename Frag::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Vec,
            VecSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<iro::token::alive<VecSubj, lifetime>>,
                detail::participation_tokens<VecSubj, ExecGroup>
            >,
            typename Vec::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Vector slice -> Fragment (register-to-register, inserts slice).
template<class Recipe, class Frag, class Vec,
         class FragInSubj, class VecSubj, class FragOutSubj,
         class ExecGroup, int Offset>
struct VectorSliceToFragment {
    static_assert(iro::contract::FragmentPayload<Frag>, "VectorSliceToFragment requires Fragment payload");
    static_assert(iro::contract::VectorPayload<Vec>, "VectorSliceToFragment requires Vector payload");
    static_assert(std::is_same_v<typename Frag::elem, typename Vec::elem>,
                  "VectorSliceToFragment: element type mismatch");
    static_assert(std::is_same_v<typename Frag::dist, typename Vec::dist>,
                  "VectorSliceToFragment: dist mismatch");
    static_assert(Offset >= 0, "VectorSliceToFragment: Offset must be non-negative");
    static_assert((Offset + static_cast<int>(Vec::lanes)) <= static_cast<int>(Frag::count),
                  "VectorSliceToFragment: slice exceeds fragment bounds");
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp> ||
                  iro::exec::is_warpgroup_v<ExecGroup>,
                  "VectorSliceToFragment: ExecGroup must be warp or warpgroup");

    using lifetime = std::conditional_t<
        std::is_same_v<ExecGroup, iro::exec::warp>,
        iro::token::lifetime::warp,
        iro::token::lifetime::warpgroup
    >;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Frag,
            FragInSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<iro::token::alive<FragInSubj, lifetime>>,
                detail::participation_tokens<FragInSubj, ExecGroup>
            >,
            typename Frag::dist,
            Recipe
        >,
        iro::contract::InputPort<
            Vec,
            VecSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<iro::token::alive<VecSubj, lifetime>>,
                detail::participation_tokens<VecSubj, ExecGroup>
            >,
            typename Vec::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Frag,
            FragOutSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<iro::token::alive<FragOutSubj, lifetime>>,
                detail::participation_tokens<FragOutSubj, ExecGroup>
            >,
            typename Frag::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

} // namespace axp::protocol::convert
