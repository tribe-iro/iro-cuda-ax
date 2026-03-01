#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>
#include "../../bundles/token_bundles.hpp"

namespace axp::protocol::mask {

namespace detail {

template<class ExecGroup>
struct exec_lifetime;

template<>
struct exec_lifetime<iro::exec::warp> { using type = iro::token::lifetime::warp; };

template<int Warps>
struct exec_lifetime<iro::exec::warpgroup_t<Warps>> { using type = iro::token::lifetime::warpgroup; };

template<class ExecGroup>
using exec_lifetime_t = typename exec_lifetime<ExecGroup>::type;

} // namespace detail

// Mask fragment payload (opaque)
template<class ShapeT, class DistT>
using MaskFrag = iro::contract::FragmentDesc<ShapeT, iro::elem::u8, DistT>;

// Generate ragged mask (e.g., attention)
template<class Recipe, class MaskFragT, class MaskSubj, class ExecGroup>
struct MaskGen {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>, "MaskGen requires warp exec group");
    using inputs = iro::util::type_list<>;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            MaskFragT,
            MaskSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<MaskSubj, iro::token::lifetime::warp>,
                iro::token::lanes_valid<MaskSubj, 32>
            >,
            typename MaskFragT::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Apply mask to an input fragment
template<class Recipe, class InFrag, class MaskFragT, class OutFrag, class InSubj, class MaskSubj, class OutSubj, class ExecGroup>
struct MaskApply {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>, "MaskApply requires warp exec group");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InFrag,
            InSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<InSubj, iro::token::lifetime::warp>
            >,
            typename InFrag::dist,
            Recipe
        >,
        iro::contract::InputPort<
            MaskFragT,
            MaskSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<MaskSubj, iro::token::lifetime::warp>,
                iro::token::lanes_valid<MaskSubj, 32>
            >,
            typename MaskFragT::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutFrag,
            OutSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<OutSubj, iro::token::lifetime::warp>
            >,
            typename OutFrag::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Causal mask + tile-skip predicate (attention)
// Inputs: Q row coord, K col coord (block-uniform scalar coords).
// Outputs: per-lane fragment mask and tile-skip predicate.
template<class Recipe, class MaskPayload, class PredPayload,
         class QCoordPayload, class KCoordPayload,
         class QCoordSubj, class KCoordSubj,
         class MaskSubj, class PredSubj,
         class ExecGroup, int TileM, int TileN>
struct CausalMaskPred {
    static_assert(iro::contract::MaskPayload<MaskPayload>, "CausalMaskPred: MaskPayload required");
    static_assert(iro::contract::ScalarPayload<PredPayload>, "CausalMaskPred: PredPayload required");
    static_assert(std::is_same_v<typename PredPayload::elem, iro::elem::u8>,
                  "CausalMaskPred: PredPayload elem must be u8");
    static_assert(iro::contract::ScalarPayload<QCoordPayload>, "CausalMaskPred: QCoordPayload must be scalar");
    static_assert(iro::contract::ScalarPayload<KCoordPayload>, "CausalMaskPred: KCoordPayload must be scalar");
    static_assert(
        std::is_same_v<typename QCoordPayload::elem, iro::elem::i32> ||
        std::is_same_v<typename QCoordPayload::elem, iro::elem::u32>,
        "CausalMaskPred: QCoordPayload elem must be i32/u32");
    static_assert(
        std::is_same_v<typename KCoordPayload::elem, iro::elem::i32> ||
        std::is_same_v<typename KCoordPayload::elem, iro::elem::u32>,
        "CausalMaskPred: KCoordPayload elem must be i32/u32");
    static_assert(TileM > 0 && TileN > 0, "CausalMaskPred: tile dims must be positive");
    static_assert(
        std::is_same_v<ExecGroup, iro::exec::warp> || iro::exec::is_warpgroup_v<ExecGroup>,
        "CausalMaskPred: ExecGroup must be warp or warpgroup");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            QCoordPayload,
            QCoordSubj,
            ExecGroup,
            axp::bundle::ValueLive<QCoordSubj, ExecGroup, iro::token::lifetime::instruction>,
            typename QCoordPayload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            KCoordPayload,
            KCoordSubj,
            ExecGroup,
            axp::bundle::ValueLive<KCoordSubj, ExecGroup, iro::token::lifetime::instruction>,
            typename KCoordPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            MaskPayload,
            MaskSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<MaskSubj, detail::exec_lifetime_t<ExecGroup>>,
                iro::token::mask_at<MaskSubj, iro::scope::min_scope_for_t<ExecGroup>>
            >,
            typename MaskPayload::dist,
            Recipe
        >,
        iro::contract::OutputPort<
            PredPayload,
            PredSubj,
            ExecGroup,
            axp::bundle::ValueLive<PredSubj, ExecGroup, detail::exec_lifetime_t<ExecGroup>>,
            typename PredPayload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

} // namespace axp::protocol::mask
