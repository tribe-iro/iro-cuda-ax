#pragma once

#include <iro_cuda_ax_core.hpp>
#include "detail/tokens.hpp"
#include "../protocol/reduction/contracts.hpp"
#include "../detail/resources.hpp"

namespace axp::level0 {

namespace shuffle {
struct down {};
struct up {};
struct xor_ {};
} // namespace shuffle

namespace vote {
struct ballot {};
struct any {};
struct all {};
} // namespace vote

namespace match {
struct any {};
struct all {};
} // namespace match

namespace detail {

template<class ExecGroup>
struct is_supported_exec_warp : std::false_type {};

template<> struct is_supported_exec_warp<iro::exec::warp> : std::true_type {};

template<class Pattern, int N>
consteval bool require_cross_pattern() {
    static_assert(N > 0, "PermuteCross requires positive element count");
    static_assert(iro::util::HasId<Pattern>,
                  "PermuteCross Pattern must provide id for determinism");
    static_assert(Pattern::map(0, N) >= 0 && Pattern::map(0, N) < N,
                  "PermuteCross Pattern::map must return index in [0, N)");
    return true;
}

} // namespace detail

// Warp shuffle (value -> value)
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class Mode, int Delta, class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Shuffle {
    static_assert(detail::is_supported_exec_warp<ExecGroup>::value,
                  "Shuffle: ExecGroup must be warp");
    static_assert(detail::is_value_payload<Payload>::value, "Shuffle: Payload must be Fragment/Scalar/Vector");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>, "Shuffle: Recipe must be explicit");
    static_assert(std::is_same_v<Mode, shuffle::down> || std::is_same_v<Mode, shuffle::up> ||
                  std::is_same_v<Mode, shuffle::xor_>, "Shuffle: Mode must be down/up/xor_");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, InSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, OutSubj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
    static constexpr int delta = Delta;
};

// Warp shuffle with explicit mask (value -> value)
template<class Recipe, class Payload, class MaskPayload, class InSubj, class MaskSubj, class OutSubj, class ExecGroup,
         class Mode, int Delta, class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct ShuffleSync {
    static_assert(detail::is_supported_exec_warp<ExecGroup>::value,
                  "ShuffleSync: ExecGroup must be warp");
    static_assert(detail::is_value_payload<Payload>::value, "ShuffleSync: Payload must be Fragment/Scalar/Vector");
    static_assert(iro::contract::MaskPayload<MaskPayload>, "ShuffleSync: MaskPayload required");
    static_assert(MaskPayload::width <= 32, "ShuffleSync: width > 32 not supported");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>, "ShuffleSync: Recipe must be explicit");
    static_assert(std::is_same_v<Mode, shuffle::down> || std::is_same_v<Mode, shuffle::up> ||
                  std::is_same_v<Mode, shuffle::xor_>, "ShuffleSync: Mode must be down/up/xor_");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, InSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            MaskPayload,
            MaskSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<MaskPayload, MaskSubj, ExecGroup>, InExtra>,
            typename MaskPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, OutSubj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
    static constexpr int delta = Delta;
};

// Warp broadcast from a single source lane (value -> value)
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         int SrcLane, class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Broadcast {
    static_assert(detail::is_supported_exec_warp<ExecGroup>::value,
                  "Broadcast: ExecGroup must be warp");
    static_assert(detail::is_value_payload<Payload>::value, "Broadcast: Payload must be Fragment/Scalar/Vector");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>, "Broadcast: Recipe must be explicit");
    static_assert(SrcLane >= 0 && SrcLane < 32, "Broadcast: SrcLane must be in [0,31]");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, InSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, OutSubj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
    static constexpr int src_lane = SrcLane;
};

// Warp broadcast from lane 0 (lane0 valid -> all lanes valid)
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct BroadcastLane0 {
    static_assert(detail::is_supported_exec_warp<ExecGroup>::value,
                  "BroadcastLane0: ExecGroup must be warp");
    static_assert(detail::is_value_payload<Payload>::value, "BroadcastLane0: Payload must be Fragment/Scalar/Vector");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>, "BroadcastLane0: Recipe must be explicit");

    using lifetime = detail::lifetime_for_exec_t<ExecGroup>;
    using in_base = iro::util::type_list<
        iro::token::alive<InSubj, lifetime>,
        iro::token::lanes_valid<InSubj, 1>
    >;
    using out_base = iro::util::type_list<
        iro::token::alive<OutSubj, lifetime>,
        iro::token::lanes_valid<OutSubj, 32>
    >;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<in_base, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<out_base, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Warpgroup broadcast from lane 0 (lane0 valid -> all lanes valid)
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, int BarrierId = 1,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct WarpgroupBroadcastLane0 {
    static_assert(iro::exec::is_warpgroup_v<ExecGroup>,
                  "WarpgroupBroadcastLane0: ExecGroup must be warpgroup");
    static_assert(detail::is_value_payload<Payload>::value,
                  "WarpgroupBroadcastLane0: Payload must be Fragment/Scalar/Vector");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "WarpgroupBroadcastLane0: Recipe must be explicit");
    static_assert(BarrierId >= 1 && BarrierId <= 8,
                  "WarpgroupBroadcastLane0: BarrierId must be 1..8");

    using lifetime = detail::lifetime_for_exec_t<ExecGroup>;
    using in_base = iro::util::type_list<
        iro::token::alive<InSubj, lifetime>,
        iro::token::lanes_valid<InSubj, 1>,
        iro::token::warps_valid<InSubj, iro::exec::warpgroup_warps<ExecGroup>::value>,
        iro::token::warpgroup_participates<InSubj, iro::exec::warpgroup_warps<ExecGroup>::value>
    >;
    using out_base = iro::util::type_list<
        iro::token::alive<OutSubj, lifetime>,
        iro::token::lanes_valid<OutSubj, 32>,
        iro::token::warps_valid<OutSubj, iro::exec::warpgroup_warps<ExecGroup>::value>,
        iro::token::warpgroup_participates<OutSubj, iro::exec::warpgroup_warps<ExecGroup>::value>
    >;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<in_base, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<out_base, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::concat_t<
        axp::detail::warpgroup_layout_resources_t<ExecGroup>,
        iro::util::type_list<iro::contract::res::warpgroup_barrier<InSubj, BarrierId>>
    >;
};

// Warp vote/ballot (value -> value)
template<class Recipe, class InPayload, class OutPayload, class InSubj, class OutSubj, class ExecGroup, class Kind,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Vote {
    static_assert(detail::is_supported_exec_warp<ExecGroup>::value,
                  "Vote: ExecGroup must be warp");
    static_assert(detail::is_value_payload<InPayload>::value, "Vote: InPayload must be Fragment/Scalar/Vector");
    static_assert(detail::is_value_payload<OutPayload>::value, "Vote: OutPayload must be Fragment/Scalar/Vector");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>, "Vote: Recipe must be explicit");
    static_assert(std::is_same_v<Kind, vote::ballot> || std::is_same_v<Kind, vote::any> ||
                  std::is_same_v<Kind, vote::all>, "Vote: Kind must be ballot/any/all");
    static_assert(!std::is_same_v<Kind, vote::ballot> ||
                  std::is_same_v<typename OutPayload::elem, iro::elem::u32>,
                  "Vote(ballot): OutPayload elem must be u32");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InPayload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InPayload, InSubj, ExecGroup>, InExtra>,
            typename InPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutPayload, OutSubj, ExecGroup>, OutExtra>,
            typename OutPayload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Warp redux.sync (value -> value) with explicit mask
template<class Recipe, class Payload, class MaskPayload, class InSubj, class MaskSubj, class OutSubj,
         class ExecGroup, class OpTag, class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct ReduxSync {
    static_assert(detail::is_supported_exec_warp<ExecGroup>::value,
                  "ReduxSync: ExecGroup must be warp");
    static_assert(iro::contract::ScalarPayload<Payload>, "ReduxSync: Scalar payload required");
    static_assert(iro::contract::MaskPayload<MaskPayload>, "ReduxSync: MaskPayload required");
    static_assert(MaskPayload::width <= 32, "ReduxSync: width > 32 not supported");
    static_assert(iro::util::HasId<OpTag>, "ReduxSync: OpTag must have id");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>, "ReduxSync: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, InSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            MaskPayload,
            MaskSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<MaskPayload, MaskSubj, ExecGroup>, InExtra>,
            typename MaskPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, OutSubj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Warp match (value -> mask)
template<class Recipe, class InPayload, class MaskPayload, class InSubj, class OutSubj, class ExecGroup, class Kind,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Match {
    static_assert(detail::is_supported_exec_warp<ExecGroup>::value,
                  "Match: ExecGroup must be warp");
    static_assert(iro::contract::ScalarPayload<InPayload>, "Match: InPayload must be Scalar");
    static_assert(sizeof(typename InPayload::elem::storage_t) <= 4,
                  "Match: InPayload storage type must be <= 4 bytes");
    static_assert(iro::contract::MaskPayload<MaskPayload>, "Match: MaskPayload required");
    static_assert(std::is_same_v<Kind, match::any> || std::is_same_v<Kind, match::all>,
                  "Match: Kind must be any/all");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>, "Match: Recipe must be explicit");
    static_assert(MaskPayload::width <= 32, "Match: width > 32 not supported");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InPayload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InPayload, InSubj, ExecGroup>, InExtra>,
            typename InPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            MaskPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<MaskPayload, OutSubj, ExecGroup>, OutExtra>,
            typename MaskPayload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Elect one lane (mask payload), no inputs
template<class Recipe, class MaskPayload, class OutSubj, class ExecGroup,
         class OutExtra = iro::util::type_list<>>
struct ElectOne {
    static_assert(detail::is_supported_exec_warp<ExecGroup>::value,
                  "ElectOne: ExecGroup must be warp");
    static_assert(iro::contract::MaskPayload<MaskPayload>,
                  "ElectOne: MaskPayload required");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>, "ElectOne: Recipe must be explicit");
    static_assert(MaskPayload::width <= 32, "ElectOne: width > 32 not supported");

    using inputs = iro::util::type_list<>;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            MaskPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<MaskPayload, OutSubj, ExecGroup>, OutExtra>,
            typename MaskPayload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Cross-warp permutation via shared memory (block scope)
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class Pattern, int BlockThreads,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct PermuteCross {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "PermuteCross requires block exec group");
    static_assert(detail::is_value_payload<Payload>::value, "PermuteCross: Payload must be Fragment/Scalar/Vector");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>, "PermuteCross: Recipe must be explicit");
    static_assert(BlockThreads > 0, "PermuteCross: BlockThreads must be positive");
    static_assert(detail::require_cross_pattern<Pattern, BlockThreads>());

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, InSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, OutSubj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::block_threads<BlockThreads>
    >;
};

// Warp bitonic compare-swap step across lanes (scalar payload).
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, int K, int J,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct WarpBitonicStep {
    static_assert(detail::is_supported_exec_warp<ExecGroup>::value,
                  "WarpBitonicStep: ExecGroup must be warp");
    static_assert(iro::contract::ScalarPayload<Payload>,
                  "WarpBitonicStep: Scalar payload required");
    static_assert((K & (K - 1)) == 0 && K > 0 && K <= 32,
                  "WarpBitonicStep: K must be power-of-two <= 32");
    static_assert((J & (J - 1)) == 0 && J > 0 && J < K,
                  "WarpBitonicStep: J must be power-of-two < K");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "WarpBitonicStep: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, InSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, OutSubj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Reverse the second half of a warp (lane-local scalar).
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct WarpReverseSecondHalf {
    static_assert(detail::is_supported_exec_warp<ExecGroup>::value,
                  "WarpReverseSecondHalf: ExecGroup must be warp");
    static_assert(iro::contract::ScalarPayload<Payload>,
                  "WarpReverseSecondHalf: Scalar payload required");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "WarpReverseSecondHalf: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, InSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, OutSubj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

} // namespace axp::level0
