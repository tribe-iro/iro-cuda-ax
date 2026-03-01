#pragma once

#include "../../realize/l0.hpp"
#include "../../realize/sm90.hpp"
#include "../../level0/specialize.hpp"

namespace axp::level2::attention {
template<class Recipe, class PredSubj, class ExecGroup>
struct TileSkipHook;
template<class Recipe, class PredSubj, class ExecGroup>
struct TileSkipHookRealization;
} // namespace axp::level2::attention

namespace axp::kit::sm90 {

namespace detail {
template<class>
inline constexpr bool always_false_v = false;

template<class Role>
__device__ __forceinline__ bool role_active() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const int warp_id = static_cast<int>(threadIdx.x) >> 5;
    const int warpgroup_id = warp_id / iro::cap::sm90::warpgroup_warps;
    if constexpr (std::is_same_v<Role, axp::level0::role::producer>) {
        return warpgroup_id == 0;
    } else if constexpr (std::is_same_v<Role, axp::level0::role::consumer>) {
        return warpgroup_id == 1;
    } else {
        return true;
    }
#else
    return true;
#endif
}

template<class Role, class Op, class Realization>
struct SpecializedOpRealization
    : iro::contract::Realization<
        axp::level0::SpecializedOp<Role, Op>,
        iro::util::mix_u64(
            iro::util::fnv1a_64_cstr("axp.realize.sm90.specialized"),
            iro::util::mix_u64(Role::id, Op::id))> {
    template<class... Args>
    __device__ static void execute(Args... args) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        if (role_active<Role>()) {
            Realization::execute(args...);
        }
#else
        (void)sizeof...(args);
#endif
    }
};
} // namespace detail

template<class Obligation>
struct registry_for {
    static_assert(detail::always_false_v<Obligation>,
                  "axp::kit::sm90::registry_for: unsupported obligation");
};

template<class Obligation>
using registry_for_t = typename registry_for<Obligation>::type;

template<class Obligation>
using bind_t = iro::bind::lookup_realization_t<Obligation, iro::cap::sm90, registry_for_t<Obligation>>;

template<class Role, class Op>
struct registry_for<axp::level0::SpecializedOp<Role, Op>> {
    using Bound = bind_t<Op>;
    using Real = typename Bound::realization;
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            detail::SpecializedOpRealization<Role, Op, Real>,
            iro::cap::sm90>
    >;
};

template<int WarpgroupCount, int WarpsPerGroup>
struct registry_for<axp::level0::RequireWarpgroupCount<WarpgroupCount, WarpsPerGroup>> {
    struct Realization
        : iro::contract::Realization<
            axp::level0::RequireWarpgroupCount<WarpgroupCount, WarpsPerGroup>,
            iro::util::mix_u64(
                iro::util::fnv1a_64_cstr("axp.realize.sm90.require_warpgroup_count"),
                iro::util::mix_u64(static_cast<iro::util::u64>(WarpgroupCount),
                                   static_cast<iro::util::u64>(WarpsPerGroup)))> {
        __device__ static void execute() {}
    };
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<Realization, iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class SwizzleAtom>
struct registry_for<axp::protocol::stage::IssueGmemToSmemSlot<
    Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::IssueGmemToSmemSlot<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class SwizzleAtom>
struct registry_for<axp::protocol::stage::DirectGmemToSmemSlot<
    Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::DirectGmemToSmemSlot<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class SwizzleAtom>
struct registry_for<axp::protocol::stage::CpAsyncIssue<
    Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::CpAsyncIssue<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class OutTile, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots>
struct registry_for<axp::protocol::stage::CpAsyncCommit<
    Recipe, OutTile, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::CpAsyncCommit<Recipe, OutTile, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class OutTile, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, int Prior>
struct registry_for<axp::protocol::stage::CpAsyncWait<
    Recipe, OutTile, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, Prior>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::CpAsyncWait<Recipe, OutTile, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, Prior>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class OutTile, class SlotSubj, class ExecGroup, class Lifetime>
struct registry_for<axp::protocol::stage::WaitSmemSlot<
    Recipe, OutTile, SlotSubj, ExecGroup, Lifetime>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::WaitSmemSlot<Recipe, OutTile, SlotSubj, ExecGroup, Lifetime>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class OutTile, class SlotSubj, class ExecGroup, class Lifetime>
struct registry_for<axp::protocol::stage::ReadySmemSlot<
    Recipe, OutTile, SlotSubj, ExecGroup, Lifetime>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::ReadySmemSlot<Recipe, OutTile, SlotSubj, ExecGroup, Lifetime>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class OutTile, class SlotSubj, class BarrierSubj, class ExecGroup, class Lifetime>
struct registry_for<axp::protocol::stage::CommitSmemSlot<
    Recipe, OutTile, SlotSubj, BarrierSubj, ExecGroup, Lifetime>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::CommitSmemSlot<Recipe, OutTile, SlotSubj, BarrierSubj, ExecGroup, Lifetime>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime>
struct registry_for<axp::protocol::stage::ReleaseSmemSlot<
    Recipe, SlotSubj, ExecGroup, Lifetime>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::ReleaseSmemSlot<Recipe, SlotSubj, ExecGroup, Lifetime>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime, long long Bytes>
struct registry_for<axp::protocol::stage::MarkConsumed<
    Recipe, SlotSubj, ExecGroup, Lifetime, Bytes>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::MarkConsumed<Recipe, SlotSubj, ExecGroup, Lifetime, Bytes>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class OutTile, class SlotSubj, class OutSubj,
         class ExecGroup, class Lifetime>
struct registry_for<axp::protocol::stage::StoreSmemToGmemSlot<
    Recipe, InTile, OutTile, SlotSubj, OutSubj, ExecGroup, Lifetime>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::StoreSmemToGmemSlot<Recipe, InTile, OutTile, SlotSubj, OutSubj, ExecGroup, Lifetime>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class SlotSubj, class BarrierSubj, class ExecGroup, class Lifetime, long long Bytes>
struct registry_for<axp::protocol::stage::CommitSmemStoreSlot<
    Recipe, SlotSubj, BarrierSubj, ExecGroup, Lifetime, Bytes>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::CommitSmemStoreSlot<Recipe, SlotSubj, BarrierSubj, ExecGroup, Lifetime, Bytes>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime, long long Bytes>
struct registry_for<axp::protocol::stage::PassSlot<
    Recipe, SlotSubj, ExecGroup, Lifetime, Bytes>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::PassSlot<Recipe, SlotSubj, ExecGroup, Lifetime, Bytes>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class SlotSubj, class SlotExecGroup, class Lifetime, long long Bytes,
         class DepPayload, class DepSubj, class DepExecGroup,
         class DepDist, class DepTokens>
struct registry_for<axp::protocol::stage::SlotAfter<
    Recipe, SlotSubj, SlotExecGroup, Lifetime, Bytes,
    DepPayload, DepSubj, DepExecGroup, DepDist, DepTokens>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::SlotAfter<
                Recipe, SlotSubj, SlotExecGroup, Lifetime, Bytes,
                DepPayload, DepSubj, DepExecGroup, DepDist, DepTokens>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Shape, class ATile, class BTile, class AccFrag, class ASubj, class BSubj, class AccSubj>
struct registry_for<axp::protocol::compute::WarpMmaFromSmem<
    Recipe, Shape, ATile, BTile, AccFrag, ASubj, BSubj, AccSubj>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::WarpMmaFromSmem<Recipe, Shape, ATile, BTile, AccFrag, ASubj, BSubj, AccSubj>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Shape, class ATile, class BTile, class AccFrag, class ASubj, class BSubj, class AccSubj>
struct registry_for<axp::protocol::compute::WarpMmaFromShared<
    Recipe, Shape, ATile, BTile, AccFrag, ASubj, BSubj, AccSubj>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::WarpMmaFromShared<Recipe, Shape, ATile, BTile, AccFrag, ASubj, BSubj, AccSubj>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct registry_for<axp::protocol::tma::BulkTmaCopy2D<
    Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
    Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj, ExecGroup, Lifetime, Bytes>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::BulkTmaCopy2D<Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
                                              Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj,
                                              ExecGroup, Lifetime, Bytes>,
            iro::cap::sm90>
    >;
};

#if defined(AXP_CUDA_HAS_THREAD_SCOPE_CLUSTER)
template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class MaskPayload, class MaskSubj,
         class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct registry_for<axp::protocol::tma::BulkTmaCopyMulticast2D<
    Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
    MaskPayload, MaskSubj,
    Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj, ExecGroup, Lifetime, Bytes>> {
    using type = std::conditional_t<
        std::is_same_v<ExecGroup, iro::exec::cluster>,
        iro::util::type_list<
            iro::registry::RealizationEntry<
                axp::realize::sm90::BulkTmaCopyMulticast2D<Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
                                                           MaskPayload, MaskSubj,
                                                           Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj,
                                                           ExecGroup, Lifetime, Bytes>,
                iro::cap::sm90>
        >,
        iro::util::type_list<>
    >;
};
#endif

template<class Recipe, class MapHandle, class MapSubj, class ExecGroup, class Lifetime>
struct registry_for<axp::protocol::tma::HostMakeTensorMap<
    Recipe, MapHandle, MapSubj, ExecGroup, Lifetime>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::HostMakeTensorMap<Recipe, MapHandle, MapSubj, ExecGroup, Lifetime>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class Coord0Payload, class Coord0Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct registry_for<axp::protocol::tma::BulkTmaCopy1D<
    Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
    Coord0Payload, Coord0Subj, ExecGroup, Lifetime, Bytes>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::BulkTmaCopy1D<Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
                                              Coord0Payload, Coord0Subj, ExecGroup, Lifetime, Bytes>,
            iro::cap::sm90>
    >;
};

#if defined(AXP_CUDA_HAS_THREAD_SCOPE_CLUSTER)
template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class MaskPayload, class MaskSubj,
         class Coord0Payload, class Coord0Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct registry_for<axp::protocol::tma::BulkTmaCopyMulticast1D<
    Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
    MaskPayload, MaskSubj,
    Coord0Payload, Coord0Subj, ExecGroup, Lifetime, Bytes>> {
    using type = std::conditional_t<
        std::is_same_v<ExecGroup, iro::exec::cluster>,
        iro::util::type_list<
            iro::registry::RealizationEntry<
                axp::realize::sm90::BulkTmaCopyMulticast1D<Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
                                                           MaskPayload, MaskSubj,
                                                           Coord0Payload, Coord0Subj,
                                                           ExecGroup, Lifetime, Bytes>,
                iro::cap::sm90>
        >,
        iro::util::type_list<>
    >;
};
#endif

template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct registry_for<axp::protocol::tma::BulkTmaStore2D<
    Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
    Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj, ExecGroup, Lifetime, Bytes>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::BulkTmaStore2D<Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
                                               Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj,
                                               ExecGroup, Lifetime, Bytes>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class Coord0Payload, class Coord0Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct registry_for<axp::protocol::tma::BulkTmaStore1D<
    Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
    Coord0Payload, Coord0Subj, ExecGroup, Lifetime, Bytes>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::BulkTmaStore1D<Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
                                               Coord0Payload, Coord0Subj, ExecGroup, Lifetime, Bytes>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Shape, class ADesc, class BDesc, class AccFrag,
         class ADescSubj, class BDescSubj, class AccSubj, class WgmmaSubj>
struct registry_for<axp::protocol::compute::WarpgroupMmaFromDesc<
    Recipe, Shape, ADesc, BDesc, AccFrag, ADescSubj, BDescSubj, AccSubj, WgmmaSubj>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::WarpgroupMmaFromDesc<Recipe, Shape, ADesc, BDesc, AccFrag, ADescSubj, BDescSubj, AccSubj, WgmmaSubj>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup>
struct registry_for<axp::protocol::reduction::BlockReduce<
    Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::BlockReduce<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag>
struct registry_for<axp::protocol::reduction::WarpReduce<
    Recipe, Frag, Subj, ExecGroup, OpTag>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::WarpReduce<Recipe, Frag, Subj, ExecGroup, OpTag>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag>
struct registry_for<axp::protocol::reduction::WarpAllReduce<
    Recipe, Frag, Subj, ExecGroup, OpTag>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::WarpAllReduce<Recipe, Frag, Subj, ExecGroup, OpTag>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag, int SegmentWidth>
struct registry_for<axp::protocol::reduction::WarpSegmentedReduce<
    Recipe, Frag, Subj, ExecGroup, OpTag, SegmentWidth>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::WarpSegmentedReduce<Recipe, Frag, Subj, ExecGroup, OpTag, SegmentWidth>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class MaskPayload, class Subj, class MaskSubj, class ExecGroup, class OpTag>
struct registry_for<axp::protocol::reduction::ShuffleReduceTree<
    Recipe, Payload, MaskPayload, Subj, MaskSubj, ExecGroup, OpTag>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::ShuffleReduceTree<Recipe, Payload, MaskPayload, Subj, MaskSubj, ExecGroup, OpTag>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag,
         int BarrierId, int WarpgroupCount>
struct registry_for<axp::protocol::reduction::WarpgroupReduce<
    Recipe, Payload, Subj, ExecGroup, OpTag, BarrierId, WarpgroupCount>> {
    static_assert(iro::exec::warpgroup_warps<ExecGroup>::value <= iro::cap::sm90::warpgroup_warps,
                  "WarpgroupReduce: ExecGroup warps exceed sm90 capability");
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::WarpgroupReduce<Recipe, Payload, Subj, ExecGroup, OpTag, BarrierId, WarpgroupCount>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class MaskFragT, class MaskSubj, class ExecGroup>
struct registry_for<axp::protocol::mask::MaskGen<
    Recipe, MaskFragT, MaskSubj, ExecGroup>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::MaskGen<Recipe, MaskFragT, MaskSubj, ExecGroup>,
            iro::cap::sm90>
    >;
};

// -----------------------------------------------------------------------------
// L0 compute (scalar/vector/fragment + mask ops)
// -----------------------------------------------------------------------------

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Alias<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Alias<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Exp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Exp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Log<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Log<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Tanh<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Tanh<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Rsqrt<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Rsqrt<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Abs<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Abs<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Neg<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Neg<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Rcp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Rcp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Sqrt<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Sqrt<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Sigmoid<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Sigmoid<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::SiLU<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::SiLU<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Gelu<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Gelu<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Popc<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Popc<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Add<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Add<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Sub<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Sub<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Mul<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Mul<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Div<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Div<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Max<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Max<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Min<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Min<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class MinSubj, class MaxSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Clamp<Recipe, Payload, InSubj, MinSubj, MaxSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Clamp<Recipe, Payload, InSubj, MinSubj, MaxSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class ASubj, class BSubj, class CSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Fma<Recipe, Payload, ASubj, BSubj, CSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Fma<Recipe, Payload, ASubj, BSubj, CSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class FragPayload, class ScalarPayload, class FragSubj, class ScalarSubj, class OutSubj,
         class ExecGroup, class InExtra, class OutExtra>
struct registry_for<axp::level0::FragmentScale<
    Recipe, FragPayload, ScalarPayload, FragSubj, ScalarSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::FragmentScale<Recipe, FragPayload, ScalarPayload, FragSubj, ScalarSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class FragPayload, class MinPayload, class MaxPayload,
         class FragSubj, class MinSubj, class MaxSubj, class OutSubj,
         class ExecGroup, class InExtra, class OutExtra>
struct registry_for<axp::level0::FragmentClamp<
    Recipe, FragPayload, MinPayload, MaxPayload, FragSubj, MinSubj, MaxSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::FragmentClamp<Recipe, FragPayload, MinPayload, MaxPayload, FragSubj, MinSubj, MaxSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class FragPayload, class ScalarPayload,
         class FragSubj, class OutSubj, class ExecGroup, class OpTag,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::FragmentReduce<
    Recipe, FragPayload, ScalarPayload, FragSubj, OutSubj, ExecGroup, OpTag, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::FragmentReduce<Recipe, FragPayload, ScalarPayload, FragSubj, OutSubj, ExecGroup, OpTag, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class FragPayload, class ScalarPayload,
         class FragSubj, class OutSubj, class ExecGroup, class OpTag,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::FragmentReduceAcc<
    Recipe, FragPayload, ScalarPayload, FragSubj, OutSubj, ExecGroup, OpTag, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::FragmentReduceAcc<Recipe, FragPayload, ScalarPayload, FragSubj, OutSubj, ExecGroup, OpTag, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class FragPayload, class InSubj, class OutSubj, class ExecGroup, class Pattern,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::FragmentPermute<
    Recipe, FragPayload, InSubj, OutSubj, ExecGroup, Pattern, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::FragmentPermute<Recipe, FragPayload, InSubj, OutSubj, ExecGroup, Pattern, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InFrag, class OutFrag,
         class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::FragmentTranspose<
    Recipe, InFrag, OutFrag, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::FragmentTranspose<Recipe, InFrag, OutFrag, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class FragPayload, class ScalarPayload, class FragSubj, class OutSubj,
         class ExecGroup, int Index, class InExtra, class OutExtra>
struct registry_for<axp::level0::FragmentExtract<
    Recipe, FragPayload, ScalarPayload, FragSubj, OutSubj, ExecGroup, Index, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::FragmentExtract<Recipe, FragPayload, ScalarPayload, FragSubj, OutSubj, ExecGroup, Index, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class FragPayload, class ScalarPayload, class FragSubj, class ScalarSubj, class OutSubj,
         class ExecGroup, int Index, class InExtra, class OutExtra>
struct registry_for<axp::level0::FragmentInsert<
    Recipe, FragPayload, ScalarPayload, FragSubj, ScalarSubj, OutSubj, ExecGroup, Index, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::FragmentInsert<Recipe, FragPayload, ScalarPayload, FragSubj, ScalarSubj, OutSubj, ExecGroup, Index, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InFrag, class OutFrag, class InSubj, class OutSubj, class ExecGroup,
         int Start, int Count, class InExtra, class OutExtra>
struct registry_for<axp::level0::FragmentSlice<
    Recipe, InFrag, OutFrag, InSubj, OutSubj, ExecGroup, Start, Count, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::FragmentSlice<Recipe, InFrag, OutFrag, InSubj, OutSubj, ExecGroup, Start, Count, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class FragPayload, class ScalarPayload, class ScalarSubj, class OutSubj,
         class ExecGroup, class InExtra, class OutExtra>
struct registry_for<axp::level0::FragmentBroadcast<
    Recipe, FragPayload, ScalarPayload, ScalarSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::FragmentBroadcast<Recipe, FragPayload, ScalarPayload, ScalarSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Subject, class ExecGroup, class InExtra, class OutExtra>
struct registry_for<axp::level0::WgmmaFence<Recipe, Subject, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::WgmmaFence<Recipe, Subject, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Subject, class ExecGroup, int Group, class InExtra, class OutExtra>
struct registry_for<axp::level0::WgmmaCommitGroup<Recipe, Subject, ExecGroup, Group, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::WgmmaCommitGroup<Recipe, Subject, ExecGroup, Group, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Subject, class ExecGroup, int Group, class InExtra, class OutExtra>
struct registry_for<axp::level0::WgmmaWaitGroup<Recipe, Subject, ExecGroup, Group, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::WgmmaWaitGroup<Recipe, Subject, ExecGroup, Group, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class AccFrag, class InSubj, class OutSubj, class WgmmaSubj, class ExecGroup, int Group,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::WgmmaWaitAcc<
    Recipe, AccFrag, InSubj, OutSubj, WgmmaSubj, ExecGroup, Group, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::WgmmaWaitAcc<Recipe, AccFrag, InSubj, OutSubj, WgmmaSubj, ExecGroup, Group, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CachePolicy, class InDist, class OutDist, class InExtra, class OutExtra>
struct registry_for<axp::level0::LdGlobal<
    Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, CachePolicy, InDist, OutDist, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::LdGlobal<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, CachePolicy, InDist, OutDist, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CachePolicy, class InDist, class OutDist, class InExtra, class OutExtra>
struct registry_for<axp::level0::StGlobal<
    Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, CachePolicy, InDist, OutDist, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::StGlobal<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, CachePolicy, InDist, OutDist, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class IndexPayload, class OutPayload,
         class InSubj, class IndexSubj, class OutSubj, class ExecGroup,
         class CachePolicy, class InExtra, class OutExtra>
struct registry_for<axp::level0::GatherGlobal<
    Recipe, InTile, IndexPayload, OutPayload, InSubj, IndexSubj, OutSubj, ExecGroup, CachePolicy, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::GatherGlobal<Recipe, InTile, IndexPayload, OutPayload,
                                           InSubj, IndexSubj, OutSubj, ExecGroup, CachePolicy, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InPayload, class IndexPayload, class OutTile,
         class InSubj, class IndexSubj, class OutSubj, class ExecGroup,
         class CachePolicy, class InExtra, class OutExtra>
struct registry_for<axp::level0::ScatterGlobal<
    Recipe, InPayload, IndexPayload, OutTile, InSubj, IndexSubj, OutSubj, ExecGroup, CachePolicy, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::ScatterGlobal<Recipe, InPayload, IndexPayload, OutTile,
                                            InSubj, IndexSubj, OutSubj, ExecGroup, CachePolicy, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::AtomicAdd<
    Recipe, InPayload, IndexPayload, OutPayload, OutTile,
    InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::AtomicAdd<Recipe, InPayload, IndexPayload, OutPayload, OutTile,
                                        InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::AtomicMin<
    Recipe, InPayload, IndexPayload, OutPayload, OutTile,
    InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::AtomicMin<Recipe, InPayload, IndexPayload, OutPayload, OutTile,
                                        InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::AtomicMax<
    Recipe, InPayload, IndexPayload, OutPayload, OutTile,
    InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::AtomicMax<Recipe, InPayload, IndexPayload, OutPayload, OutTile,
                                        InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::AtomicAnd<
    Recipe, InPayload, IndexPayload, OutPayload, OutTile,
    InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::AtomicAnd<Recipe, InPayload, IndexPayload, OutPayload, OutTile,
                                        InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::AtomicOr<
    Recipe, InPayload, IndexPayload, OutPayload, OutTile,
    InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::AtomicOr<Recipe, InPayload, IndexPayload, OutPayload, OutTile,
                                       InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::AtomicXor<
    Recipe, InPayload, IndexPayload, OutPayload, OutTile,
    InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::AtomicXor<Recipe, InPayload, IndexPayload, OutPayload, OutTile,
                                        InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class ComparePayload, class ValuePayload, class IndexPayload, class OutPayload, class OutTile,
         class CompareSubj, class ValueSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::AtomicCAS<
    Recipe, ComparePayload, ValuePayload, IndexPayload, OutPayload, OutTile,
    CompareSubj, ValueSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::AtomicCAS<Recipe, ComparePayload, ValuePayload, IndexPayload, OutPayload, OutTile,
                                        CompareSubj, ValueSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class InSubj, class ExecGroup,
         class CachePolicy, class InDist, class InExtra>
struct registry_for<axp::level0::PrefetchGlobal<
    Recipe, InTile, InSubj, ExecGroup, CachePolicy, InDist, InExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::PrefetchGlobal<Recipe, InTile, InSubj, ExecGroup, CachePolicy, InDist, InExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class InDist, class OutDist, class InExtra, class OutExtra>
struct registry_for<axp::level0::LdShared<
    Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, InDist, OutDist, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::LdShared<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, InDist, OutDist, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class InDist, class OutDist, class InExtra, class OutExtra>
struct registry_for<axp::level0::StShared<
    Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, InDist, OutDist, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::StShared<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, InDist, OutDist, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Tile, class ScaleTile, class TileSubj, class ScaleSubj, class ExecGroup,
         class TileInExtra, class ScaleInExtra, class OutExtra>
struct registry_for<axp::level0::ScaleSharedTile<
    Recipe, Tile, ScaleTile, TileSubj, ScaleSubj, ExecGroup, TileInExtra, ScaleInExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::ScaleSharedTile<
                Recipe, Tile, ScaleTile, TileSubj, ScaleSubj, ExecGroup, TileInExtra, ScaleInExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Tile, class OutSubj, class ExecGroup, class OutExtra>
struct registry_for<axp::level0::TileZero<
    Recipe, Tile, OutSubj, ExecGroup, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::TileZero<Recipe, Tile, OutSubj, ExecGroup, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::ReduceSharedToGlobalAtomicAdd<
    Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::ReduceSharedToGlobalAtomicAdd<
                Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup, class SwizzleAtom,
         class InDist, class OutDist, class InExtra, class OutExtra>
struct registry_for<axp::level0::SwizzledLdShared<
    Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, SwizzleAtom, InDist, OutDist, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::SwizzledLdShared<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, SwizzleAtom, InDist, OutDist, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup, class SwizzleAtom,
         class InDist, class OutDist, class InExtra, class OutExtra>
struct registry_for<axp::level0::SwizzledStShared<
    Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, SwizzleAtom, InDist, OutDist, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::SwizzledStShared<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, SwizzleAtom, InDist, OutDist, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class MaskPayload, class ASubj, class BSubj, class MaskSubj,
         class OutSubj, class ExecGroup, class InExtra, class OutExtra>
struct registry_for<axp::level0::Select<
    Recipe, Payload, MaskPayload, ASubj, BSubj, MaskSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Select<Recipe, Payload, MaskPayload, ASubj, BSubj, MaskSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class ScalarPayload, class OutSubj, class ExecGroup, class Pattern, class OutExtra>
struct registry_for<axp::level0::ScalarConst<
    Recipe, ScalarPayload, OutSubj, ExecGroup, Pattern, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::ScalarConst<Recipe, ScalarPayload, OutSubj, ExecGroup, Pattern, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class MaskPayload, class OutSubj, class ExecGroup, class Pattern, class OutExtra>
struct registry_for<axp::level0::MaskConst<
    Recipe, MaskPayload, OutSubj, ExecGroup, Pattern, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::MaskConst<Recipe, MaskPayload, OutSubj, ExecGroup, Pattern, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class MaskPayload, class PredPayload,
         class QCoordPayload, class KCoordPayload,
         class QCoordSubj, class KCoordSubj,
         class MaskSubj, class PredSubj,
         class ExecGroup, int TileM, int TileN>
struct registry_for<axp::level0::CausalMaskPred<
    Recipe, MaskPayload, PredPayload,
    QCoordPayload, KCoordPayload,
    QCoordSubj, KCoordSubj,
    MaskSubj, PredSubj,
    ExecGroup, TileM, TileN>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::CausalMaskPred<
                Recipe, MaskPayload, PredPayload,
                QCoordPayload, KCoordPayload,
                QCoordSubj, KCoordSubj,
                MaskSubj, PredSubj,
                ExecGroup, TileM, TileN>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class MaskPayload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::MaskNot<
    Recipe, MaskPayload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::MaskNot<Recipe, MaskPayload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class MaskPayload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::MaskAnd<
    Recipe, MaskPayload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::MaskAnd<Recipe, MaskPayload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class PredSubj, class ExecGroup>
struct registry_for<axp::level2::attention::TileSkipHook<Recipe, PredSubj, ExecGroup>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::level2::attention::TileSkipHookRealization<Recipe, PredSubj, ExecGroup>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class MaskPayload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::MaskOr<
    Recipe, MaskPayload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::MaskOr<Recipe, MaskPayload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class MaskPayload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::MaskXor<
    Recipe, MaskPayload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::MaskXor<Recipe, MaskPayload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

// -----------------------------------------------------------------------------
// L0 communication (shuffle/vote)
// -----------------------------------------------------------------------------

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class Mode, int Delta, class InExtra, class OutExtra>
struct registry_for<axp::level0::Shuffle<
    Recipe, Payload, InSubj, OutSubj, ExecGroup, Mode, Delta, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Shuffle<Recipe, Payload, InSubj, OutSubj, ExecGroup, Mode, Delta, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class MaskPayload, class InSubj, class MaskSubj, class OutSubj, class ExecGroup,
         class Mode, int Delta, class InExtra, class OutExtra>
struct registry_for<axp::level0::ShuffleSync<
    Recipe, Payload, MaskPayload, InSubj, MaskSubj, OutSubj, ExecGroup, Mode, Delta, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::ShuffleSync<Recipe, Payload, MaskPayload, InSubj, MaskSubj, OutSubj, ExecGroup, Mode, Delta, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, int K, int J,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::WarpBitonicStep<
    Recipe, Payload, InSubj, OutSubj, ExecGroup, K, J, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::WarpBitonicStep<Recipe, Payload, InSubj, OutSubj, ExecGroup, K, J, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::WarpReverseSecondHalf<
    Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::WarpReverseSecondHalf<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, int SrcLane,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Broadcast<
    Recipe, Payload, InSubj, OutSubj, ExecGroup, SrcLane, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Broadcast<Recipe, Payload, InSubj, OutSubj, ExecGroup, SrcLane, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class InExtra, class OutExtra>
struct registry_for<axp::level0::BroadcastLane0<
    Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::BroadcastLane0<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, int BarrierId,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::WarpgroupBroadcastLane0<
    Recipe, Payload, InSubj, OutSubj, ExecGroup, BarrierId, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::WarpgroupBroadcastLane0<
                Recipe, Payload, InSubj, OutSubj, ExecGroup, BarrierId, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InPayload, class OutPayload, class InSubj, class OutSubj, class ExecGroup, class Kind,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Vote<
    Recipe, InPayload, OutPayload, InSubj, OutSubj, ExecGroup, Kind, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Vote<Recipe, InPayload, OutPayload, InSubj, OutSubj, ExecGroup, Kind, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class MaskPayload, class InSubj, class MaskSubj, class OutSubj,
         class ExecGroup, class OpTag, class InExtra, class OutExtra>
struct registry_for<axp::level0::ReduxSync<
    Recipe, Payload, MaskPayload, InSubj, MaskSubj, OutSubj, ExecGroup, OpTag, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::ReduxSync<Recipe, Payload, MaskPayload, InSubj, MaskSubj, OutSubj, ExecGroup, OpTag, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InPayload, class MaskPayload, class InSubj, class OutSubj, class ExecGroup, class Kind,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::Match<
    Recipe, InPayload, MaskPayload, InSubj, OutSubj, ExecGroup, Kind, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::Match<Recipe, InPayload, MaskPayload, InSubj, OutSubj, ExecGroup, Kind, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class MaskPayload, class OutSubj, class ExecGroup, class OutExtra>
struct registry_for<axp::level0::ElectOne<
    Recipe, MaskPayload, OutSubj, ExecGroup, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::ElectOne<Recipe, MaskPayload, OutSubj, ExecGroup, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class Pattern, int BlockThreads,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::PermuteCross<
    Recipe, Payload, InSubj, OutSubj, ExecGroup, Pattern, BlockThreads, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::PermuteCross<Recipe, Payload, InSubj, OutSubj, ExecGroup, Pattern, BlockThreads, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::WarpScan<
    Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::WarpScan<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode, int SegmentWidth,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::WarpSegmentedScan<
    Recipe, Payload, Subj, ExecGroup, OpTag, Mode, SegmentWidth, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::WarpSegmentedScan<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, SegmentWidth, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         int BarrierId, int WarpgroupCount, class InExtra, class OutExtra>
struct registry_for<axp::level0::WarpgroupScan<
    Recipe, Payload, Subj, ExecGroup, OpTag, Mode, BarrierId, WarpgroupCount, InExtra, OutExtra>> {
    static_assert(iro::exec::warpgroup_warps<ExecGroup>::value <= iro::cap::sm90::warpgroup_warps,
                  "WarpgroupScan: ExecGroup warps exceed sm90 capability");
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::WarpgroupScan<
                Recipe, Payload, Subj, ExecGroup, OpTag, Mode, BarrierId, WarpgroupCount, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::BlockScan<
    Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::BlockScan<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Payload, class CarryPayload, class Subj, class CarryInSubj, class CarryOutSubj,
         class ExecGroup, class OpTag, class Mode, class InExtra, class OutExtra>
struct registry_for<axp::level0::ChainedScan<
    Recipe, Payload, CarryPayload, Subj, CarryInSubj, CarryOutSubj, ExecGroup, OpTag, Mode, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::ChainedScan<Recipe, Payload, CarryPayload, Subj, CarryInSubj, CarryOutSubj,
                                          ExecGroup, OpTag, Mode, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class IndexPayload, class InSubj, class OutSubj, class ExecGroup, int Stages,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::PipelineAdvance<
    Recipe, IndexPayload, InSubj, OutSubj, ExecGroup, Stages, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::PipelineAdvance<Recipe, IndexPayload, InSubj, OutSubj, ExecGroup, Stages, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class IndexPayload, class InSubj, class OutSubj, class ExecGroup, int Stages,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::PipelineProduce<
    Recipe, IndexPayload, InSubj, OutSubj, ExecGroup, Stages, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::PipelineProduce<Recipe, IndexPayload, InSubj, OutSubj, ExecGroup, Stages, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class IndexPayload, class InSubj, class OutSubj, class ExecGroup, int Stages,
         class InExtra, class OutExtra>
struct registry_for<axp::level0::PipelineConsume<
    Recipe, IndexPayload, InSubj, OutSubj, ExecGroup, Stages, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::PipelineConsume<Recipe, IndexPayload, InSubj, OutSubj, ExecGroup, Stages, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

// -----------------------------------------------------------------------------
// L0 memory (tile copy)
// -----------------------------------------------------------------------------

template<class Recipe, class Tile, class Subj, class ExecGroup, class InExtra, class OutExtra>
struct registry_for<axp::level0::TileFence<
    Recipe, Tile, Subj, ExecGroup, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::l0::TileFence<Recipe, Tile, Subj, ExecGroup, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class RecipeIn, class RecipeOut, class InTile, class OutTile,
         class InSubj, class OutSubj, class ExecGroup, int VecBytes, class InDist, class OutDist>
struct registry_for<axp::protocol::convert::CastTile<
    RecipeIn, RecipeOut, InTile, OutTile, InSubj, OutSubj, ExecGroup, VecBytes, InDist, OutDist>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::CastTile<RecipeIn, RecipeOut, InTile, OutTile, InSubj, OutSubj, ExecGroup, VecBytes, InDist, OutDist>,
            iro::cap::sm90>
    >;
};

template<class RecipeIn, class RecipeOut, class InFrag, class OutFrag,
         class InSubj, class OutSubj, class ExecGroup>
struct registry_for<axp::protocol::convert::CastFragment<
    RecipeIn, RecipeOut, InFrag, OutFrag, InSubj, OutSubj, ExecGroup>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::CastFragment<RecipeIn, RecipeOut, InFrag, OutFrag, InSubj, OutSubj, ExecGroup>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Frag, class Vec, class FragSubj, class VecSubj, class ExecGroup, int Offset>
struct registry_for<axp::protocol::convert::FragmentToVectorSlice<
    Recipe, Frag, Vec, FragSubj, VecSubj, ExecGroup, Offset>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::FragmentToVectorSlice<Recipe, Frag, Vec, FragSubj, VecSubj, ExecGroup, Offset>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Frag, class Vec, class FragInSubj, class VecSubj, class FragOutSubj,
         class ExecGroup, int Offset>
struct registry_for<axp::protocol::convert::VectorSliceToFragment<
    Recipe, Frag, Vec, FragInSubj, VecSubj, FragOutSubj, ExecGroup, Offset>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::VectorSliceToFragment<
                Recipe, Frag, Vec, FragInSubj, VecSubj, FragOutSubj, ExecGroup, Offset>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Subject, class ExecGroup>
struct registry_for<axp::protocol::sync::SyncPoint<
    Recipe, Subject, ExecGroup>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::SyncPoint<Recipe, Subject, ExecGroup>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Subject, class ExecGroup>
struct registry_for<axp::protocol::sync::SyncWarp<
    Recipe, Subject, ExecGroup>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::SyncWarp<Recipe, Subject, ExecGroup>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Subject, class ExecGroup>
struct registry_for<axp::protocol::sync::SyncThreads<
    Recipe, Subject, ExecGroup>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::SyncThreads<Recipe, Subject, ExecGroup>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Subject, class ExecGroup, class ScopeT, class OrderT>
struct registry_for<axp::protocol::sync::Fence<
    Recipe, Subject, ExecGroup, ScopeT, OrderT>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::Fence<Recipe, Subject, ExecGroup, ScopeT, OrderT>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Subject, class ExecGroup, int Expected>
struct registry_for<axp::protocol::sync::BarrierInit<
    Recipe, Subject, ExecGroup, Expected>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::BarrierInit<Recipe, Subject, ExecGroup, Expected>,
            iro::cap::sm90>
    >;
};

#if defined(AXP_CUDA_HAS_THREAD_SCOPE_CLUSTER)
template<class Recipe, class Subject, class ExecGroup, int Expected>
struct registry_for<axp::protocol::sync::ClusterBarrierInit<
    Recipe, Subject, ExecGroup, Expected>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::ClusterBarrierInit<Recipe, Subject, ExecGroup, Expected>,
            iro::cap::sm90>
    >;
};
#endif

template<class Recipe, class Subject, class ExecGroup>
struct registry_for<axp::protocol::sync::BarrierArrive<
    Recipe, Subject, ExecGroup>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::BarrierArrive<Recipe, Subject, ExecGroup>,
            iro::cap::sm90>
    >;
};

#if defined(AXP_CUDA_HAS_THREAD_SCOPE_CLUSTER)
template<class Recipe, class Subject, class ExecGroup>
struct registry_for<axp::protocol::sync::ClusterBarrierArrive<
    Recipe, Subject, ExecGroup>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::ClusterBarrierArrive<Recipe, Subject, ExecGroup>,
            iro::cap::sm90>
    >;
};
#endif

template<class Recipe, class Subject, class ExecGroup, long long Bytes>
struct registry_for<axp::protocol::sync::BarrierExpectTx<
    Recipe, Subject, ExecGroup, Bytes>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::BarrierExpectTx<Recipe, Subject, ExecGroup, Bytes>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Subject, class ExecGroup, long long Bytes>
struct registry_for<axp::protocol::sync::BarrierArriveTx<
    Recipe, Subject, ExecGroup, Bytes>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::BarrierArriveTx<Recipe, Subject, ExecGroup, Bytes>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Subject, class ExecGroup>
struct registry_for<axp::protocol::sync::BarrierWait<
    Recipe, Subject, ExecGroup>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::BarrierWait<Recipe, Subject, ExecGroup>,
            iro::cap::sm90>
    >;
};

#if defined(AXP_CUDA_HAS_THREAD_SCOPE_CLUSTER)
template<class Recipe, class Subject, class ExecGroup>
struct registry_for<axp::protocol::sync::ClusterBarrierWait<
    Recipe, Subject, ExecGroup>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::ClusterBarrierWait<Recipe, Subject, ExecGroup>,
            iro::cap::sm90>
    >;
};
#endif

template<class Recipe, class Subject, class ExecGroup, class FlagPayload, class FlagSubj, class OutExtra>
struct registry_for<axp::protocol::sync::BarrierTryWait<
    Recipe, Subject, ExecGroup, FlagPayload, FlagSubj, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::BarrierTryWait<Recipe, Subject, ExecGroup, FlagPayload, FlagSubj, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Subject, class ExecGroup, int Expected>
struct registry_for<axp::protocol::sync::BarrierInvalidate<
    Recipe, Subject, ExecGroup, Expected>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::BarrierInvalidate<Recipe, Subject, ExecGroup, Expected>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class InTile, class Frag, class InSubj, class FragSubj, class ExecGroup, class Lifetime,
         class InExtra, class OutExtra>
struct registry_for<axp::protocol::ownership::SharedTileToFragment<
    Recipe, InTile, Frag, InSubj, FragSubj, ExecGroup, Lifetime, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::SharedTileToFragment<Recipe, InTile, Frag, InSubj, FragSubj, ExecGroup, Lifetime, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Frag, class OutTile, class FragSubj, class OutSubj, class ExecGroup, class Lifetime,
         class InExtra, class OutExtra>
struct registry_for<axp::protocol::ownership::FragmentToSharedTile<
    Recipe, Frag, OutTile, FragSubj, OutSubj, ExecGroup, Lifetime, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::FragmentToSharedTile<Recipe, Frag, OutTile, FragSubj, OutSubj, ExecGroup, Lifetime, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Tile, class Subj, class ExecGroup, class Lifetime,
         class InExtra, class OutExtra>
struct registry_for<axp::protocol::ownership::TileBoundaryIn<
    Recipe, Tile, Subj, ExecGroup, Lifetime, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::TileBoundaryIn<Recipe, Tile, Subj, ExecGroup, Lifetime, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class Tile, class Subj, class ExecGroup, class Lifetime,
         class InExtra, class OutExtra>
struct registry_for<axp::protocol::ownership::TileBoundaryOut<
    Recipe, Tile, Subj, ExecGroup, Lifetime, InExtra, OutExtra>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::TileBoundaryOut<Recipe, Tile, Subj, ExecGroup, Lifetime, InExtra, OutExtra>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class SmemTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime, class SwizzleAtom>
struct registry_for<axp::protocol::ownership::UseWgmmaSmemDesc<
    Recipe, SmemTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::UseWgmmaSmemDesc<Recipe, SmemTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class SmemTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime, class SwizzleAtom>
struct registry_for<axp::protocol::ownership::MakeWgmmaSmemDesc<
    Recipe, SmemTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::MakeWgmmaSmemDesc<Recipe, SmemTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class SmemTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime, class SwizzleAtom>
struct registry_for<axp::protocol::ownership::MakeWgmmaSmemDescReady<
    Recipe, SmemTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::MakeWgmmaSmemDescReady<Recipe, SmemTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class SmemTile, class DescTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime,
         class SwizzleAtom, int RowOffset, int ColOffset>
struct registry_for<axp::protocol::ownership::MakeWgmmaSmemDescSlice<
    Recipe, SmemTile, DescTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom, RowOffset, ColOffset>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::MakeWgmmaSmemDescSlice<
                Recipe, SmemTile, DescTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom, RowOffset, ColOffset>,
            iro::cap::sm90>
    >;
};

template<class Recipe, class SmemTile, class DescTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime,
         class SwizzleAtom, int RowOffset, int ColOffset>
struct registry_for<axp::protocol::ownership::MakeWgmmaSmemDescSliceReady<
    Recipe, SmemTile, DescTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom, RowOffset, ColOffset>> {
    using type = iro::util::type_list<
        iro::registry::RealizationEntry<
            axp::realize::sm90::MakeWgmmaSmemDescSliceReady<
                Recipe, SmemTile, DescTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom, RowOffset, ColOffset>,
            iro::cap::sm90>
    >;
};

template<class... Lists>
struct registry_concat;

template<>
struct registry_concat<> {
    using type = iro::util::type_list<>;
};

template<class List>
struct registry_concat<List> {
    using type = List;
};

template<class L1, class L2, class... Rest>
struct registry_concat<L1, L2, Rest...> {
    using type = typename registry_concat<typename iro::util::concat<L1, L2>::type, Rest...>::type;
};

template<class... Obligations>
using registry_list_t = typename registry_concat<registry_for_t<Obligations>...>::type;

template<class... Obligations>
using registry = registry_list_t<Obligations...>;

} // namespace axp::kit::sm90
