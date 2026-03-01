#pragma once

#include <iro_cuda_ax_core.hpp>
#include "compute.hpp"
#include "communication.hpp"
#include "ownership.hpp"
#include "convert.hpp"
#include "detail/tokens.hpp"
#include "../detail/resources.hpp"
#include "../protocol/reduction/contracts.hpp"

namespace axp::level0 {

// Explicit matrix load/store for WMMA fragments (shared tile <-> fragment).
// These are dedicated fragment-typed ops and must not be conflated with generic swizzled tile copies.
template<class... Args>
using LdMatrix = SharedTileToFragment<Args...>;

template<class... Args>
using StMatrix = FragmentToSharedTile<Args...>;

namespace permute {
struct identity {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.permute.identity");
    static constexpr int map(int i, int) { return i; }
};
struct reverse {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.permute.reverse");
    static constexpr int map(int i, int n) { return (n - 1) - i; }
};
template<int Shift>
struct rotate_left {
    static constexpr auto id = iro::util::mix_u64(
        iro::util::fnv1a_64_cstr("axp.permute.rotate_left"),
        static_cast<iro::util::u64>(Shift));
    static constexpr int map(int i, int n) { return (i + Shift) % n; }
};
template<int Shift>
struct rotate_right {
    static constexpr auto id = iro::util::mix_u64(
        iro::util::fnv1a_64_cstr("axp.permute.rotate_right"),
        static_cast<iro::util::u64>(Shift));
    static constexpr int map(int i, int n) { return (i - Shift + n) % n; }
};
} // namespace permute

namespace detail {

template<class Payload>
consteval bool require_fragment() {
    static_assert(iro::contract::FragmentPayload<Payload>,
                  "Fragment op requires Fragment payload");
    return true;
}

template<class FragPayload, class ScalarPayload>
consteval bool require_scalar_match() {
    static_assert(iro::contract::ScalarPayload<ScalarPayload>,
                  "Fragment op requires Scalar payload");
    using frag_t = typename FragPayload::elem::storage_t;
    using scalar_t = typename ScalarPayload::elem::storage_t;
    static_assert(std::is_same_v<typename FragPayload::elem, typename ScalarPayload::elem> ||
                  std::is_constructible_v<frag_t, scalar_t>,
                  "Fragment op requires scalar elem convertible to frag elem");
    return true;
}

template<class ExecGroup>
struct is_supported_exec_frag : std::false_type {};

template<> struct is_supported_exec_frag<iro::exec::warp> : std::true_type {};
template<int Warps> struct is_supported_exec_frag<iro::exec::warpgroup_t<Warps>> : std::true_type {};

template<class Pattern, int N>
consteval bool require_pattern() {
    static_assert(N > 0, "Permute requires positive element count");
    static_assert(iro::util::HasId<Pattern>,
                  "Permute Pattern must provide id for determinism");
    static_assert(Pattern::map(0, N) >= 0 && Pattern::map(0, N) < N,
                  "Permute Pattern::map must return index in [0, N)");
    return true;
}

template<class FragPayload, class Recipe>
consteval bool require_frag_recipe() {
    static_assert(std::is_same_v<typename FragPayload::elem, typename Recipe::in>,
                  "Fragment op requires Recipe::in matching fragment elem");
    static_assert(std::is_same_v<typename FragPayload::elem, typename Recipe::out>,
                  "Fragment op requires Recipe::out matching fragment elem");
    return true;
}

} // namespace detail

// Fragment scale by scalar (frag + scalar -> frag)
template<class Recipe, class FragPayload, class ScalarPayload,
         class FragSubj, class ScalarSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct FragmentScale {
    static_assert(detail::require_fragment<FragPayload>());
    static_assert(detail::require_scalar_match<FragPayload, ScalarPayload>());
    static_assert(detail::require_frag_recipe<FragPayload, Recipe>());
    static_assert(detail::is_supported_exec_frag<ExecGroup>::value,
                  "FragmentScale: ExecGroup must be warp/warpgroup");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "FragmentScale: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            FragPayload,
            FragSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<FragPayload, FragSubj, ExecGroup>, InExtra>,
            typename FragPayload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            ScalarPayload,
            ScalarSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<ScalarPayload, ScalarSubj, ExecGroup>, InExtra>,
            typename ScalarPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            FragPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<FragPayload, OutSubj, ExecGroup>, OutExtra>,
            typename FragPayload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Fragment clamp by scalar min/max (frag + min + max -> frag)
template<class Recipe, class FragPayload, class MinPayload, class MaxPayload,
         class FragSubj, class MinSubj, class MaxSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct FragmentClamp {
    static_assert(detail::require_fragment<FragPayload>());
    static_assert(detail::require_scalar_match<FragPayload, MinPayload>());
    static_assert(detail::require_scalar_match<FragPayload, MaxPayload>());
    static_assert(detail::require_frag_recipe<FragPayload, Recipe>());
    static_assert(detail::is_supported_exec_frag<ExecGroup>::value,
                  "FragmentClamp: ExecGroup must be warp/warpgroup");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "FragmentClamp: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            FragPayload,
            FragSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<FragPayload, FragSubj, ExecGroup>, InExtra>,
            typename FragPayload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            MinPayload,
            MinSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<MinPayload, MinSubj, ExecGroup>, InExtra>,
            typename MinPayload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            MaxPayload,
            MaxSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<MaxPayload, MaxSubj, ExecGroup>, InExtra>,
            typename MaxPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            FragPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<FragPayload, OutSubj, ExecGroup>, OutExtra>,
            typename FragPayload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Fragment reduce (frag -> scalar) with explicit op tag
template<class Recipe, class FragPayload, class ScalarPayload,
         class FragSubj, class OutSubj, class ExecGroup, class OpTag,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct FragmentReduce {
    static_assert(detail::require_fragment<FragPayload>());
    static_assert(detail::require_scalar_match<FragPayload, ScalarPayload>());
    static_assert(detail::require_frag_recipe<FragPayload, Recipe>());
    static_assert(detail::is_supported_exec_frag<ExecGroup>::value,
                  "FragmentReduce: ExecGroup must be warp/warpgroup");
    static_assert(iro::util::HasId<OpTag>, "FragmentReduce: OpTag must have id");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "FragmentReduce: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            FragPayload,
            FragSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<FragPayload, FragSubj, ExecGroup>, InExtra>,
            typename FragPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            ScalarPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<ScalarPayload, OutSubj, ExecGroup>, OutExtra>,
            typename ScalarPayload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Fragment reduce (frag -> scalar) producing Recipe::acc
template<class Recipe, class FragPayload, class ScalarPayload,
         class FragSubj, class OutSubj, class ExecGroup, class OpTag,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct FragmentReduceAcc {
    static_assert(detail::require_fragment<FragPayload>());
    static_assert(iro::contract::ScalarPayload<ScalarPayload>,
                  "FragmentReduceAcc: ScalarPayload required");
    static_assert(std::is_same_v<typename ScalarPayload::elem, typename Recipe::acc>,
                  "FragmentReduceAcc: Scalar elem must match Recipe::acc");
    static_assert(detail::require_frag_recipe<FragPayload, Recipe>());
    static_assert(detail::is_supported_exec_frag<ExecGroup>::value,
                  "FragmentReduceAcc: ExecGroup must be warp/warpgroup");
    static_assert(iro::util::HasId<OpTag>, "FragmentReduceAcc: OpTag must have id");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "FragmentReduceAcc: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            FragPayload,
            FragSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<FragPayload, FragSubj, ExecGroup>, InExtra>,
            typename FragPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            ScalarPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<ScalarPayload, OutSubj, ExecGroup>, OutExtra>,
            typename ScalarPayload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Fragment reduce (frag -> scalar) with explicit vectorization contract.
// This is a semantic atom that guarantees a vectorized realization path.
template<class Recipe, class FragPayload, class ScalarPayload,
         class FragSubj, class OutSubj, class ExecGroup, class OpTag,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct FragmentReduceAccVec
    : FragmentReduceAcc<Recipe, FragPayload, ScalarPayload,
                        FragSubj, OutSubj, ExecGroup, OpTag, InExtra, OutExtra> {
    using elem_t = typename FragPayload::elem;
    static_assert(std::is_same_v<elem_t, iro::elem::f16> || std::is_same_v<elem_t, iro::elem::bf16>,
                  "FragmentReduceAccVec: only f16/bf16 fragments are vectorized");
    static_assert((FragPayload::count % 2) == 0,
                  "FragmentReduceAccVec: fragment element count must be even");
    static_assert(std::is_same_v<typename Recipe::acc, elem_t>,
                  "FragmentReduceAccVec: Recipe::acc must match fragment element type");
    static_assert(
        std::is_same_v<OpTag, axp::protocol::reduction::op_add> ||
        std::is_same_v<OpTag, axp::protocol::reduction::op_mul> ||
        std::is_same_v<OpTag, axp::protocol::reduction::op_max> ||
        std::is_same_v<OpTag, axp::protocol::reduction::op_min>,
        "FragmentReduceAccVec: OpTag must be add/mul/max/min");
};

// Fragment permute (frag -> frag) using explicit compile-time pattern
template<class Recipe, class FragPayload, class InSubj, class OutSubj, class ExecGroup, class Pattern,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct FragmentPermute {
    static_assert(detail::require_fragment<FragPayload>());
    static_assert(detail::require_frag_recipe<FragPayload, Recipe>());
    static_assert(detail::is_supported_exec_frag<ExecGroup>::value,
                  "FragmentPermute: ExecGroup must be warp/warpgroup");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "FragmentPermute: Recipe must be explicit");
    static_assert(detail::require_pattern<Pattern, static_cast<int>(FragPayload::count)>());

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            FragPayload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<FragPayload, InSubj, ExecGroup>, InExtra>,
            typename FragPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            FragPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<FragPayload, OutSubj, ExecGroup>, OutExtra>,
            typename FragPayload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Fragment transpose (frag -> frag) for rank-2 shapes
template<class Recipe, class InFrag, class OutFrag,
         class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct FragmentTranspose {
    static_assert(detail::require_fragment<InFrag>());
    static_assert(detail::require_fragment<OutFrag>());
    static_assert(std::is_same_v<typename InFrag::elem, typename OutFrag::elem>,
                  "FragmentTranspose: element types must match");
    static_assert(std::is_same_v<typename InFrag::dist, typename OutFrag::dist>,
                  "FragmentTranspose: distribution must match");
    static_assert(detail::is_supported_exec_frag<ExecGroup>::value,
                  "FragmentTranspose: ExecGroup must be warp/warpgroup");
    static_assert(InFrag::shape::rank == 2 && OutFrag::shape::rank == 2,
                  "FragmentTranspose: rank-2 shapes required");
    static_assert(InFrag::shape::template dim<0>() == OutFrag::shape::template dim<1>(),
                  "FragmentTranspose: shape mismatch (M)");
    static_assert(InFrag::shape::template dim<1>() == OutFrag::shape::template dim<0>(),
                  "FragmentTranspose: shape mismatch (N)");
    static_assert(InFrag::count == InFrag::shape::size, "FragmentTranspose: full fragment required");
    static_assert(OutFrag::count == OutFrag::shape::size, "FragmentTranspose: full fragment required");
    static_assert(InFrag::count == OutFrag::count, "FragmentTranspose: count mismatch");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "FragmentTranspose: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InFrag,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InFrag, InSubj, ExecGroup>, InExtra>,
            typename InFrag::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutFrag,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutFrag, OutSubj, ExecGroup>, OutExtra>,
            typename OutFrag::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Fragment extract (frag -> scalar) at compile-time index
template<class Recipe, class FragPayload, class ScalarPayload,
         class FragSubj, class OutSubj, class ExecGroup, int Index,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct FragmentExtract {
    static_assert(detail::require_fragment<FragPayload>());
    static_assert(detail::require_scalar_match<FragPayload, ScalarPayload>());
    static_assert(detail::is_supported_exec_frag<ExecGroup>::value,
                  "FragmentExtract: ExecGroup must be warp/warpgroup");
    static_assert(Index >= 0 && Index < static_cast<int>(FragPayload::count),
                  "FragmentExtract: Index out of range");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "FragmentExtract: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            FragPayload,
            FragSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<FragPayload, FragSubj, ExecGroup>, InExtra>,
            typename FragPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            ScalarPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<ScalarPayload, OutSubj, ExecGroup>, OutExtra>,
            typename ScalarPayload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Fragment insert (frag + scalar -> frag) at compile-time index
template<class Recipe, class FragPayload, class ScalarPayload,
         class FragSubj, class ScalarSubj, class OutSubj, class ExecGroup, int Index,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct FragmentInsert {
    static_assert(detail::require_fragment<FragPayload>());
    static_assert(detail::require_scalar_match<FragPayload, ScalarPayload>());
    static_assert(detail::is_supported_exec_frag<ExecGroup>::value,
                  "FragmentInsert: ExecGroup must be warp/warpgroup");
    static_assert(Index >= 0 && Index < static_cast<int>(FragPayload::count),
                  "FragmentInsert: Index out of range");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "FragmentInsert: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            FragPayload,
            FragSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<FragPayload, FragSubj, ExecGroup>, InExtra>,
            typename FragPayload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            ScalarPayload,
            ScalarSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<ScalarPayload, ScalarSubj, ExecGroup>, InExtra>,
            typename ScalarPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            FragPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<FragPayload, OutSubj, ExecGroup>, OutExtra>,
            typename FragPayload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Fragment slice (frag -> frag) [Start, Start+Count)
template<class Recipe, class InFrag, class OutFrag,
         class InSubj, class OutSubj, class ExecGroup, int Start, int Count,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct FragmentSlice {
    static_assert(detail::require_fragment<InFrag>());
    static_assert(detail::require_fragment<OutFrag>());
    static_assert(std::is_same_v<typename InFrag::elem, typename OutFrag::elem>,
                  "FragmentSlice: element types must match");
    static_assert(std::is_same_v<typename InFrag::dist, typename OutFrag::dist>,
                  "FragmentSlice: distribution must match");
    static_assert(detail::is_supported_exec_frag<ExecGroup>::value,
                  "FragmentSlice: ExecGroup must be warp/warpgroup");
    static_assert(Start >= 0 && Count > 0,
                  "FragmentSlice: invalid range");
    static_assert(Start + Count <= static_cast<int>(InFrag::count),
                  "FragmentSlice: out of range");
    static_assert(static_cast<int>(OutFrag::count) == Count,
                  "FragmentSlice: OutFrag size must equal Count");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "FragmentSlice: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InFrag,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InFrag, InSubj, ExecGroup>, InExtra>,
            typename InFrag::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutFrag,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutFrag, OutSubj, ExecGroup>, OutExtra>,
            typename OutFrag::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Fragment broadcast (scalar -> frag)
template<class Recipe, class FragPayload, class ScalarPayload,
         class ScalarSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct FragmentBroadcast {
    static_assert(detail::require_fragment<FragPayload>());
    static_assert(detail::require_scalar_match<FragPayload, ScalarPayload>());
    static_assert(detail::is_supported_exec_frag<ExecGroup>::value,
                  "FragmentBroadcast: ExecGroup must be warp/warpgroup");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "FragmentBroadcast: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            ScalarPayload,
            ScalarSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<ScalarPayload, ScalarSubj, ExecGroup>, InExtra>,
            typename ScalarPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            FragPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<FragPayload, OutSubj, ExecGroup>, OutExtra>,
            typename FragPayload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Fragment pack/unpack are explicit aliases to fragment casts.
// "Pack" and "unpack" describe cast direction (narrow->wide vs wide->narrow),
// not bit-level packing or layout changes.
template<class... Args>
using FragmentPack = axp::level0::CastFragment<Args...>;

template<class... Args>
using FragmentUnpack = axp::level0::CastFragment<Args...>;

} // namespace axp::level0
