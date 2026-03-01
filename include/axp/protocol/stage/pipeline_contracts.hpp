#pragma once

#include "bundles.hpp"
#include "../../bundles/token_bundles.hpp"
#include "../../detail/resources.hpp"

namespace axp::protocol::stage {

namespace sync = axp::protocol::sync;

namespace detail {
template<class OutTile, class SwizzleAtom, class = void>
struct swizzle_layout_ok : std::false_type {};

template<class OutTile, class SwizzleAtom>
struct swizzle_layout_ok<OutTile, SwizzleAtom, std::void_t<decltype(SwizzleAtom::B), decltype(SwizzleAtom::S)>>
    : std::bool_constant<
        std::is_same_v<typename OutTile::layout,
            iro::contract::layout::Swizzled<
                OutTile::shape::template dim<1>(),
                SwizzleAtom::B,
                SwizzleAtom::S>> ||
        std::is_same_v<typename OutTile::layout,
            iro::contract::layout::SwizzledColMajor<
                OutTile::shape::template dim<0>(),
                SwizzleAtom::B,
                SwizzleAtom::S>>> {};
} // namespace detail

// Issue async transfer into a slot (produces filling state).
template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class SwizzleAtom = void>
struct IssueGmemToSmemSlot {
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>, "IssueGmemToSmemSlot: InTile elem != Recipe::in");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::in>, "IssueGmemToSmemSlot: OutTile elem != Recipe::in");
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "IssueGmemToSmemSlot requires block exec group");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::global>);
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::shared>);
    static_assert(InTile::align::bytes >= 16, "Issue requires 16B gmem alignment");
    static_assert(OutTile::align::bytes >= 16, "Issue requires 16B smem alignment");
    static_assert(OutTile::bytes % 16 == 0, "Issue requires 16B granularity");
    static_assert(Slots >= 2 && Slots <= 4, "Issue requires 2-4 pipeline slots (SOTA)");

    // Swizzle requires explicit Swizzled layout (row-major or col-major).
    static_assert(std::is_void_v<SwizzleAtom> ||
                  detail::swizzle_layout_ok<OutTile, SwizzleAtom>::value,
                  "IssueGmemToSmemSlot: SwizzleAtom requires Swizzled layout");

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

// Direct (synchronous) transfer into a slot (produces ready state).
template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class SwizzleAtom = void>
struct DirectGmemToSmemSlot {
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>, "DirectGmemToSmemSlot: InTile elem != Recipe::in");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::in>, "DirectGmemToSmemSlot: OutTile elem != Recipe::in");
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "DirectGmemToSmemSlot requires block exec group");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::global>);
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::shared>);
    static_assert(InTile::align::bytes >= 16, "DirectGmemToSmemSlot requires 16B gmem alignment");
    static_assert(OutTile::align::bytes >= 16, "DirectGmemToSmemSlot requires 16B smem alignment");
    static_assert(OutTile::bytes % 16 == 0, "DirectGmemToSmemSlot requires 16B granularity");
    static_assert(Slots >= 2 && Slots <= 4, "DirectGmemToSmemSlot requires 2-4 pipeline slots (SOTA)");

    // Swizzle requires explicit Swizzled layout (row-major or col-major).
    static_assert(std::is_void_v<SwizzleAtom> ||
                  detail::swizzle_layout_ok<OutTile, SwizzleAtom>::value,
                  "DirectGmemToSmemSlot: SwizzleAtom requires Swizzled layout");

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

// Wait for slot to become ready (transitions filling -> ready).
template<class Recipe, class OutTile, class SlotSubj, class ExecGroup, class Lifetime>
struct WaitSmemSlot {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "WaitSmemSlot requires block exec group");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::in>, "WaitSmemSlot: OutTile elem != Recipe::in");
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

    using resources = iro::util::type_list<>;
};

// Commit slot after async TMA copy (barrier wait + ready tokens).
template<class Recipe, class OutTile, class SlotSubj, class BarrierSubj, class ExecGroup, class Lifetime>
struct CommitSmemSlot {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "CommitSmemSlot requires block exec group");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::in>, "CommitSmemSlot: OutTile elem != Recipe::in");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerIssued<SlotSubj, Lifetime, OutTile::bytes>>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            sync::BarrierHandle,
            BarrierSubj,
            ExecGroup,
            iro::token::bundle_list<sync::BarrierArrived<BarrierSubj, ExecGroup>>,
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

    using resources = iro::util::type_list<>;
};

// Pass-through for already-ready slot (producer committed -> producer committed).
template<class Recipe, class OutTile, class SlotSubj, class ExecGroup, class Lifetime>
struct ReadySmemSlot {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "ReadySmemSlot requires block exec group");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::in>, "ReadySmemSlot: OutTile elem != Recipe::in");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerCommitted<SlotSubj, ExecGroup, Lifetime, OutTile::bytes>>,
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

    using resources = iro::util::type_list<>;
};

// Release slot after consumption (used -> free).
template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime>
struct ReleaseSmemSlot {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "ReleaseSmemSlot requires block exec group");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::slot_state<SlotSubj, iro::token::state::used>,
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
            iro::token::bundle_list<SlotReleased<SlotSubj, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Mark a ready slot as consumed (ready -> used).
template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime, long long Bytes>
struct MarkConsumed {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp> ||
                  iro::exec::is_warpgroup_v<ExecGroup> ||
                  std::is_same_v<ExecGroup, iro::exec::block>,
                  "MarkConsumed requires warp/warpgroup/block exec group");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerCommitted<SlotSubj, ExecGroup, Lifetime, Bytes>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ConsumerConsumed<SlotSubj, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Store shared slot to global and pass through slot handle (ordering).
template<class Recipe, class InTile, class OutTile, class SlotSubj, class OutSubj,
         class ExecGroup, class Lifetime>
struct StoreSmemToGmemSlot {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "StoreSmemToGmemSlot requires block exec group");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::shared>,
                  "StoreSmemToGmemSlot requires shared-memory source");
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::global>,
                  "StoreSmemToGmemSlot requires global-memory destination");
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>, "StoreSmemToGmemSlot: InTile elem != Recipe::in");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::out>, "StoreSmemToGmemSlot: OutTile elem != Recipe::out");
    static_assert(InTile::align::bytes >= 16, "StoreSmemToGmemSlot requires 16B smem alignment");
    static_assert(OutTile::align::bytes >= 16, "StoreSmemToGmemSlot requires 16B gmem alignment");
    static_assert(InTile::bytes % 16 == 0, "StoreSmemToGmemSlot requires 16B granularity");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerCommitted<SlotSubj, ExecGroup, Lifetime, InTile::bytes>>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerCommitted<SlotSubj, ExecGroup, Lifetime, InTile::bytes>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            axp::bundle::TileOutTokens<OutSubj, ExecGroup, Lifetime>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::OutputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerCommitted<SlotSubj, ExecGroup, Lifetime, InTile::bytes>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Commit slot after async TMA store (barrier wait, pass-through ready tokens).
template<class Recipe, class SlotSubj, class BarrierSubj, class ExecGroup, class Lifetime, long long Bytes>
struct CommitSmemStoreSlot {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "CommitSmemStoreSlot requires block exec group");
    static_assert(Bytes > 0, "CommitSmemStoreSlot requires positive byte count");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerCommitted<SlotSubj, ExecGroup, Lifetime, Bytes>>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            sync::BarrierHandle,
            BarrierSubj,
            ExecGroup,
            iro::token::bundle_list<sync::BarrierArrived<BarrierSubj, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerCommitted<SlotSubj, ExecGroup, Lifetime, Bytes>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Pass-through slot handle (ordering without side-effects).
template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime, long long Bytes>
struct PassSlot {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "PassSlot requires block exec group");
    static_assert(Bytes > 0, "PassSlot requires positive byte count");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerCommitted<SlotSubj, ExecGroup, Lifetime, Bytes>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            SlotHandle,
            SlotSubj,
            ExecGroup,
            iro::token::bundle_list<ProducerCommitted<SlotSubj, ExecGroup, Lifetime, Bytes>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Order slot consumption after a dependent payload (tokenized ordering only).
template<class Recipe, class SlotSubj, class SlotExecGroup, class Lifetime, long long Bytes,
         class DepPayload, class DepSubj, class DepExecGroup,
         class DepDist = iro::contract::no_dist,
         class DepTokens = iro::util::type_list<>>
struct SlotAfter {
    static_assert(Bytes > 0, "SlotAfter requires positive byte count");
    static_assert(std::is_same_v<SlotExecGroup, iro::exec::warp> ||
                  iro::exec::is_warpgroup_v<SlotExecGroup> ||
                  std::is_same_v<SlotExecGroup, iro::exec::block>,
                  "SlotAfter requires warp/warpgroup/block slot exec group");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SlotHandle,
            SlotSubj,
            SlotExecGroup,
            iro::token::bundle_list<ProducerCommitted<SlotSubj, SlotExecGroup, Lifetime, Bytes>>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            DepPayload,
            DepSubj,
            DepExecGroup,
            DepTokens,
            DepDist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            SlotHandle,
            SlotSubj,
            SlotExecGroup,
            iro::token::bundle_list<ProducerCommitted<SlotSubj, SlotExecGroup, Lifetime, Bytes>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};
}
