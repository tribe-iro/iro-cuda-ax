#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>
#include "../protocol/sync/bundles.hpp"
#include "../protocol/stage/tokens.hpp"
#include "../protocol/order/bundles.hpp"
#include "../protocol/atomic/bundles.hpp"
#include "../detail/participation_tokens.hpp"

namespace axp::bundle {

namespace detail {

template<class Subject, class ExecGroup>
using tile_participation_tokens = axp::detail::participation_tokens<Subject, ExecGroup>;

template<class Subject, class ExecGroup, class Lifetime>
using tile_in_base = iro::util::concat_t<
    iro::util::type_list<
        iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
        iro::token::alive<Subject, Lifetime>
    >,
    tile_participation_tokens<Subject, ExecGroup>
>;

template<class Subject, class ExecGroup, class Lifetime>
using tile_in_sync = iro::util::concat_t<
    iro::util::type_list<
        iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
        iro::token::alive<Subject, Lifetime>,
        iro::token::sync_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>
    >,
    tile_participation_tokens<Subject, ExecGroup>
>;

template<class Subject, class ExecGroup, class Lifetime>
using tile_in_tokens = std::conditional_t<
    (iro::scope::min_scope_for_t<ExecGroup>::level >= iro::scope::warpgroup::level),
    tile_in_sync<Subject, ExecGroup, Lifetime>,
    tile_in_base<Subject, ExecGroup, Lifetime>
>;

template<class Subject, class ExecGroup, class Lifetime>
using tile_out_base = iro::util::concat_t<
    iro::util::type_list<
        iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
        iro::token::alive<Subject, Lifetime>
    >,
    tile_participation_tokens<Subject, ExecGroup>
>;

template<class Subject, class ExecGroup, class Lifetime>
using tile_out_sync = iro::util::concat_t<
    iro::util::type_list<
        iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
        iro::token::alive<Subject, Lifetime>,
        iro::token::sync_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>
    >,
    tile_participation_tokens<Subject, ExecGroup>
>;

template<class Subject, class ExecGroup, class Lifetime>
using tile_out_tokens = std::conditional_t<
    (iro::scope::min_scope_for_t<ExecGroup>::level >= iro::scope::warpgroup::level),
    tile_out_sync<Subject, ExecGroup, Lifetime>,
    tile_out_base<Subject, ExecGroup, Lifetime>
>;

template<class Subject, class ExecGroup, class Lifetime>
using value_live_tokens = iro::util::concat_t<
    iro::util::type_list<
        iro::token::alive<Subject, Lifetime>
    >,
    tile_participation_tokens<Subject, ExecGroup>
>;

template<class Subject, class ExecGroup, class Lifetime>
using value_lane0_tokens = iro::util::concat_t<
    iro::util::type_list<
        iro::token::alive<Subject, Lifetime>,
        iro::token::lanes_valid<Subject, 1>
    >,
    std::conditional_t<
        iro::exec::is_warpgroup_v<ExecGroup>,
        iro::util::type_list<
            iro::token::warps_valid<Subject, iro::exec::warpgroup_warps<ExecGroup>::value>,
            iro::token::warpgroup_participates<Subject, iro::exec::warpgroup_warps<ExecGroup>::value>
        >,
        iro::util::type_list<>
    >
>;

} // namespace detail

template<class Subject, class ExecGroup, class Lifetime>
using TileInTokens = detail::tile_in_tokens<Subject, ExecGroup, Lifetime>;

template<class Subject, class ExecGroup, class Lifetime>
using TileOutTokens = detail::tile_out_tokens<Subject, ExecGroup, Lifetime>;

template<class Subject, class ExecGroup, class Lifetime = iro::token::lifetime::warp>
using ValueLive = detail::value_live_tokens<Subject, ExecGroup, Lifetime>;

template<class Subject, class ExecGroup, class Lifetime = iro::token::lifetime::warp>
using ValueLane0 = detail::value_lane0_tokens<Subject, ExecGroup, Lifetime>;

template<class SlotSubj, class ExecGroup, class Lifetime>
using SmemReady = iro::token::bundle<
    iro::token::slot_state<SlotSubj, iro::token::state::ready>,
    iro::token::visible_at<SlotSubj, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<SlotSubj, Lifetime>,
    iro::token::sync_at<SlotSubj, iro::scope::min_scope_for_t<ExecGroup>>
>;

template<class SlotSubj, class Lifetime>
using SmemFilling = iro::token::bundle<
    iro::token::slot_state<SlotSubj, iro::token::state::filling>,
    iro::token::alive<SlotSubj, Lifetime>
>;

template<class SlotSubj, class Lifetime, long long Bytes>
using SmemFillingTx = iro::token::bundle<
    iro::token::slot_state<SlotSubj, iro::token::state::filling>,
    iro::token::alive<SlotSubj, Lifetime>,
    axp::protocol::sync::tx_bytes<SlotSubj, Bytes>
>;

template<class SlotSubj, class Lifetime, long long Bytes>
using SmemCommittedTx = iro::token::bundle<
    iro::token::slot_state<SlotSubj, iro::token::state::filling>,
    iro::token::alive<SlotSubj, Lifetime>,
    axp::protocol::sync::tx_bytes<SlotSubj, Bytes>,
    axp::protocol::stage::cp_async_committed<SlotSubj>
>;

template<class SlotSubj, class Lifetime>
using SmemConsumed = iro::token::bundle<
    iro::token::slot_state<SlotSubj, iro::token::state::used>,
    iro::token::alive<SlotSubj, Lifetime>
>;

template<class SlotSubj, class Lifetime>
using SmemReleased = iro::token::bundle<
    iro::token::slot_state<SlotSubj, iro::token::state::free>,
    iro::token::alive<SlotSubj, Lifetime>
>;

template<class SlotSubj, class ExecGroup, class Lifetime, long long Bytes>
using SmemReadyTx = iro::token::bundle<
    iro::token::slot_state<SlotSubj, iro::token::state::ready>,
    iro::token::visible_at<SlotSubj, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<SlotSubj, Lifetime>,
    iro::token::sync_at<SlotSubj, iro::scope::min_scope_for_t<ExecGroup>>,
    axp::protocol::sync::tx_bytes<SlotSubj, Bytes>
>;

template<class Subject, class ExecGroup, class Lifetime = iro::token::lifetime::warp>
using FragmentLive = iro::token::bundle<
    iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<Subject, Lifetime>
>;

template<class Subject, class Lifetime = iro::token::lifetime::block>
using GmemVisible = iro::token::bundle<
    iro::token::visible_at<Subject, iro::scope::device>,
    iro::token::alive<Subject, Lifetime>
>;

template<class Subject>
using LeaderIssued = iro::token::bundle<
    iro::token::issued_by_lane0<Subject>
>;

template<class Subject, class ExecGroup, class Lifetime = iro::token::lifetime::block>
using BarrierReady = axp::protocol::sync::BarrierReady<Subject, ExecGroup, Lifetime>;

template<class Subject, class ExecGroup, class Lifetime = iro::token::lifetime::block>
using BarrierArrived = axp::protocol::sync::BarrierArrived<Subject, ExecGroup, Lifetime>;

template<class Subject, class EventTag, class ExecGroup, class Lifetime = iro::token::lifetime::block>
using EventPublished = axp::protocol::order::EventPublished<Subject, EventTag, ExecGroup, Lifetime>;

template<class Subject, class EventTag, class PhaseTag, class ExecGroup, class Lifetime = iro::token::lifetime::block>
using EventPhaseReady = axp::protocol::order::EventPhaseReady<Subject, EventTag, PhaseTag, ExecGroup, Lifetime>;

template<class Subject, class EpochTag, class ExecGroup, class Lifetime = iro::token::lifetime::block>
using EpochStamped = axp::protocol::order::EpochStamped<Subject, EpochTag, ExecGroup, Lifetime>;

template<class Subject, class ScopeT, class OrderT = iro::memory_order::seq_cst, class Lifetime = iro::token::lifetime::block>
using AtomicDone = axp::protocol::atomic::AtomicDone<Subject, ScopeT, OrderT, Lifetime>;

} // namespace axp::bundle
