#pragma once

#include "../protocol/stage/pipeline_contracts.hpp"
#include "../protocol/stage/async_contracts.hpp"
#include "detail/tokens.hpp"

namespace axp::level0 {

// Stage atoms (aliases to protocol layer)
template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class SwizzleAtom = void>
using AsyncCopy = axp::protocol::stage::IssueGmemToSmemSlot<
    Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom
>;

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class SwizzleAtom = void>
using DirectCopy = axp::protocol::stage::DirectGmemToSmemSlot<
    Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom
>;

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class SwizzleAtom = void>
using CpAsyncIssue = axp::protocol::stage::CpAsyncIssue<
    Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom
>;

template<class Recipe, class OutTile, class PipeTag, class SlotSubj, class ExecGroup, class Lifetime, int Slots>
using CpAsyncCommit = axp::protocol::stage::CpAsyncCommit<
    Recipe, OutTile, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots
>;

template<class Recipe, class OutTile, class PipeTag, class SlotSubj, class ExecGroup, class Lifetime, int Slots, int Prior = 0>
using CpAsyncWait = axp::protocol::stage::CpAsyncWait<
    Recipe, OutTile, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, Prior
>;

template<class Recipe, class OutTile, class SlotSubj, class ExecGroup, class Lifetime>
using WaitSlot = axp::protocol::stage::WaitSmemSlot<Recipe, OutTile, SlotSubj, ExecGroup, Lifetime>;

template<class Recipe, class OutTile, class SlotSubj, class ExecGroup, class Lifetime>
using ReadySlot = axp::protocol::stage::ReadySmemSlot<Recipe, OutTile, SlotSubj, ExecGroup, Lifetime>;

template<class Recipe, class OutTile, class SlotSubj, class BarrierSubj, class ExecGroup, class Lifetime>
using CommitSlot = axp::protocol::stage::CommitSmemSlot<Recipe, OutTile, SlotSubj, BarrierSubj, ExecGroup, Lifetime>;

template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime>
using ReleaseSlot = axp::protocol::stage::ReleaseSmemSlot<Recipe, SlotSubj, ExecGroup, Lifetime>;

template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime, long long Bytes>
using MarkConsumed = axp::protocol::stage::MarkConsumed<Recipe, SlotSubj, ExecGroup, Lifetime, Bytes>;

template<class Recipe, class InTile, class OutTile, class SlotSubj, class OutSubj,
         class ExecGroup, class Lifetime>
using StoreSmemToGmemSlot = axp::protocol::stage::StoreSmemToGmemSlot<
    Recipe, InTile, OutTile, SlotSubj, OutSubj, ExecGroup, Lifetime
>;

template<class Recipe, class SlotSubj, class BarrierSubj, class ExecGroup, class Lifetime, long long Bytes>
using CommitStoreSlot = axp::protocol::stage::CommitSmemStoreSlot<
    Recipe, SlotSubj, BarrierSubj, ExecGroup, Lifetime, Bytes
>;

template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime, long long Bytes>
using PassSlot = axp::protocol::stage::PassSlot<
    Recipe, SlotSubj, ExecGroup, Lifetime, Bytes
>;

template<class Recipe, class SlotSubj, class SlotExecGroup, class Lifetime, long long Bytes,
         class DepPayload, class DepSubj, class DepExecGroup,
         class DepDist = iro::contract::no_dist,
         class DepTokens = iro::util::type_list<>>
using SlotAfter = axp::protocol::stage::SlotAfter<
    Recipe, SlotSubj, SlotExecGroup, Lifetime, Bytes,
    DepPayload, DepSubj, DepExecGroup, DepDist, DepTokens
>;

// Software pipeline index atoms (ping-pong control)
template<class Recipe, class IndexPayload, class InSubj, class OutSubj, class ExecGroup, int Stages,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct PipelineAdvance {
    static_assert(iro::contract::ScalarPayload<IndexPayload>, "PipelineAdvance: Scalar payload required");
    static_assert(std::is_same_v<typename IndexPayload::elem, iro::elem::u32>,
                  "PipelineAdvance: IndexPayload elem must be u32");
    static_assert(Stages > 0, "PipelineAdvance: Stages must be positive");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "PipelineAdvance: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            IndexPayload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<IndexPayload, InSubj, ExecGroup>, InExtra>,
            typename IndexPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            IndexPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<IndexPayload, OutSubj, ExecGroup>, OutExtra>,
            typename IndexPayload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

template<class Recipe, class IndexPayload, class InSubj, class OutSubj, class ExecGroup, int Stages,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct PipelineProduce
    : PipelineAdvance<Recipe, IndexPayload, InSubj, OutSubj, ExecGroup, Stages, InExtra, OutExtra> {};

template<class Recipe, class IndexPayload, class InSubj, class OutSubj, class ExecGroup, int Stages,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct PipelineConsume
    : PipelineAdvance<Recipe, IndexPayload, InSubj, OutSubj, ExecGroup, Stages, InExtra, OutExtra> {};

} // namespace axp::level0

namespace iro::contract {

template<class... Args>
struct is_fused_atom<axp::protocol::stage::IssueGmemToSmemSlot<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::stage::DirectGmemToSmemSlot<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::stage::CpAsyncIssue<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::stage::CpAsyncCommit<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::stage::CpAsyncWait<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::stage::SlotAfter<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::stage::WaitSmemSlot<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::stage::CommitSmemSlot<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::stage::ReleaseSmemSlot<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::stage::MarkConsumed<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::stage::StoreSmemToGmemSlot<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::stage::CommitSmemStoreSlot<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::stage::PassSlot<Args...>> : std::true_type {};

} // namespace iro::contract
