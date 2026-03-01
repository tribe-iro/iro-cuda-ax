#pragma once

#include "bundles.hpp"

namespace axp::protocol::stage {

// ---------------------------------------------------------------------------
// CpAsync atoms (SM80+ pipeline)
// These are explicit pipeline steps: issue -> commit -> wait.
// ---------------------------------------------------------------------------

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class SwizzleAtom = void>
struct CpAsyncIssue {
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>, "CpAsyncIssue: InTile elem != Recipe::in");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::in>, "CpAsyncIssue: OutTile elem != Recipe::in");
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "CpAsyncIssue requires block exec group");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::global>);
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::shared>);
    static_assert(InTile::align::bytes >= 16, "CpAsyncIssue requires 16B gmem alignment");
    static_assert(OutTile::align::bytes >= 16, "CpAsyncIssue requires 16B smem alignment");
    static_assert(OutTile::bytes % 16 == 0, "CpAsyncIssue requires 16B granularity");
    static_assert(Slots >= 2 && Slots <= 4, "CpAsyncIssue requires 2-4 pipeline slots (SOTA)");

    static_assert(std::is_void_v<SwizzleAtom> ||
                  detail::swizzle_layout_ok<OutTile, SwizzleAtom>::value,
                  "CpAsyncIssue: SwizzleAtom requires Swizzled layout");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            axp::bundle::TileInTokens<InSubj, ExecGroup, iro::token::lifetime::block>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::slot_state<SlotSubj, iro::token::state::free>,
                iro::token::alive<SlotSubj, Lifetime>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerIssued<SlotSubj, Lifetime, OutTile::bytes>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::smem_pipeline<
            PipeTag,
            Slots,
            OutTile::bytes,
            OutTile::align::bytes
        >
    >;
};

template<class Recipe, class OutTile, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots>
struct CpAsyncCommit {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "CpAsyncCommit requires block exec group");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerIssued<SlotSubj, Lifetime, OutTile::bytes>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerCommittedTx<SlotSubj, Lifetime, OutTile::bytes>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::smem_pipeline<
            PipeTag,
            Slots,
            OutTile::bytes,
            OutTile::align::bytes
        >
    >;
};

template<class Recipe, class OutTile, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, int Prior = 0>
struct CpAsyncWait {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "CpAsyncWait requires block exec group");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerCommittedTx<SlotSubj, Lifetime, OutTile::bytes>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerCommitted<SlotSubj, ExecGroup, Lifetime, OutTile::bytes>>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::OutputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerCommitted<SlotSubj, ExecGroup, Lifetime, OutTile::bytes>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::smem_pipeline<
            PipeTag,
            Slots,
            OutTile::bytes,
            OutTile::align::bytes
        >
    >;
};

} // namespace axp::protocol::stage
