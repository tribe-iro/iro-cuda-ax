#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level1/passthrough.hpp"
#include "detail/compose.hpp"
#include "registry.hpp"

namespace axp::level2::staging {

template<class MapHandleT, class BarrierSubjT,
         class Coord0PayloadT, class Coord0SubjT,
         class Coord1PayloadT = void, class Coord1SubjT = void>
struct TmaConfig {
    using MapHandle = MapHandleT;
    using BarrierSubj = BarrierSubjT;
    using Coord0Payload = Coord0PayloadT;
    using Coord0Subj = Coord0SubjT;
    using Coord1Payload = Coord1PayloadT;
    using Coord1Subj = Coord1SubjT;
};

template<class MapHandleT, class BarrierSubjT,
         class MaskPayloadT, class MaskSubjT,
         class Coord0PayloadT, class Coord0SubjT,
         class Coord1PayloadT = void, class Coord1SubjT = void>
struct TmaMulticastConfig {
    using MapHandle = MapHandleT;
    using BarrierSubj = BarrierSubjT;
    using MaskPayload = MaskPayloadT;
    using MaskSubj = MaskSubjT;
    using Coord0Payload = Coord0PayloadT;
    using Coord0Subj = Coord0SubjT;
    using Coord1Payload = Coord1PayloadT;
    using Coord1Subj = Coord1SubjT;
    using ExecGroup = iro::exec::cluster;
};

template<class T, class = void>
struct tma_multicast_traits {
    static constexpr bool valid = false;
};

template<class T>
struct tma_multicast_traits<T, std::void_t<typename T::MapHandle, typename T::BarrierSubj,
                                           typename T::MaskPayload, typename T::MaskSubj,
                                           typename T::Coord0Payload, typename T::Coord0Subj>> {
    static constexpr bool valid = true;
    using MapHandle = typename T::MapHandle;
    using BarrierSubj = typename T::BarrierSubj;
    using MaskPayload = typename T::MaskPayload;
    using MaskSubj = typename T::MaskSubj;
    using Coord0Payload = typename T::Coord0Payload;
    using Coord0Subj = typename T::Coord0Subj;
    using Coord1Payload = typename T::Coord1Payload;
    using Coord1Subj = typename T::Coord1Subj;
    static constexpr bool has_coord1 = !std::is_void_v<Coord1Payload>;
    using ExecGroup = typename T::ExecGroup;
};

template<class T, class = void>
struct tma_traits {
    static constexpr bool valid = false;
};

template<class T>
struct tma_traits<T, std::void_t<typename T::MapHandle, typename T::BarrierSubj,
                                 typename T::Coord0Payload, typename T::Coord0Subj>> {
    static constexpr bool valid = !tma_multicast_traits<T>::valid;
    using MapHandle = typename T::MapHandle;
    using BarrierSubj = typename T::BarrierSubj;
    using Coord0Payload = typename T::Coord0Payload;
    using Coord0Subj = typename T::Coord0Subj;
    using Coord1Payload = typename T::Coord1Payload;
    using Coord1Subj = typename T::Coord1Subj;
    static constexpr bool has_coord1 = !std::is_void_v<Coord1Payload>;
};

struct streaming_tag {};

template<class T>
struct streaming_traits {
    static constexpr bool valid = false;
};

template<>
struct streaming_traits<streaming_tag> {
    static constexpr bool valid = true;
};

template<class Tma>
using tma_issue_exec_group_t = std::conditional_t<tma_multicast_traits<Tma>::valid,
                                                  iro::exec::cluster,
                                                  iro::exec::block>;

template<bool HasCoord1>
struct tma_copy_selector;

template<>
struct tma_copy_selector<true> {
    template<class Recipe, class MapHandle, class OutTile, class MapSubj, class SlotSubj, class BarrierSubj,
             class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
             class ExecGroup, class Lifetime>
    using type = axp::level1::proto::tma::BulkTmaCopy2D<
        Recipe, MapHandle, OutTile, MapSubj, SlotSubj, BarrierSubj,
        Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj, ExecGroup, Lifetime, OutTile::bytes
    >;
};

template<>
struct tma_copy_selector<false> {
    template<class Recipe, class MapHandle, class OutTile, class MapSubj, class SlotSubj, class BarrierSubj,
             class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
             class ExecGroup, class Lifetime>
    using type = axp::level1::proto::tma::BulkTmaCopy1D<
        Recipe, MapHandle, OutTile, MapSubj, SlotSubj, BarrierSubj,
        Coord0Payload, Coord0Subj, ExecGroup, Lifetime, OutTile::bytes
    >;
};

template<bool HasCoord1>
struct tma_store_selector;

template<>
struct tma_store_selector<true> {
    template<class Recipe, class MapHandle, class InTile, class MapSubj, class SlotSubj, class BarrierSubj,
             class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
             class ExecGroup, class Lifetime>
    using type = axp::level1::proto::tma::BulkTmaStore2D<
        Recipe, MapHandle, InTile, MapSubj, SlotSubj, BarrierSubj,
        Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj, ExecGroup, Lifetime, InTile::bytes
    >;
};

template<>
struct tma_store_selector<false> {
    template<class Recipe, class MapHandle, class InTile, class MapSubj, class SlotSubj, class BarrierSubj,
             class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
             class ExecGroup, class Lifetime>
    using type = axp::level1::proto::tma::BulkTmaStore1D<
        Recipe, MapHandle, InTile, MapSubj, SlotSubj, BarrierSubj,
        Coord0Payload, Coord0Subj, ExecGroup, Lifetime, InTile::bytes
    >;
};

template<bool HasCoord1>
struct tma_multicast_copy_selector;

template<>
struct tma_multicast_copy_selector<true> {
    template<class Recipe, class MapHandle, class OutTile, class MapSubj, class SlotSubj, class BarrierSubj,
             class MaskPayload, class MaskSubj,
             class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
             class ExecGroup, class Lifetime>
    using type = axp::level1::proto::tma::BulkTmaCopyMulticast2D<
        Recipe, MapHandle, OutTile, MapSubj, SlotSubj, BarrierSubj,
        MaskPayload, MaskSubj,
        Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj,
        ExecGroup, Lifetime, OutTile::bytes
    >;
};

template<>
struct tma_multicast_copy_selector<false> {
    template<class Recipe, class MapHandle, class OutTile, class MapSubj, class SlotSubj, class BarrierSubj,
             class MaskPayload, class MaskSubj,
             class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
             class ExecGroup, class Lifetime>
    using type = axp::level1::proto::tma::BulkTmaCopyMulticast1D<
        Recipe, MapHandle, OutTile, MapSubj, SlotSubj, BarrierSubj,
        MaskPayload, MaskSubj,
        Coord0Payload, Coord0Subj, ExecGroup, Lifetime, OutTile::bytes
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class MarkExecGroup, class Lifetime, int Slots, class SwizzleAtom = void, class Tma = void,
         class IssueExecGroup = iro::exec::block, class CapT = axp::target_cap>
struct StageGmemToSmemImpl {
    static_assert(std::is_same_v<IssueExecGroup, iro::exec::block>,
                  "StageGmemToSmemImpl: IssueExecGroup must be block for non-TMA staging");
    using Issue = axp::level1::low::AsyncCopy<
        Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj,
        iro::exec::block, Lifetime, Slots, SwizzleAtom
    >;
    using Wait = axp::level1::low::WaitSlot<
        Recipe, OutTile, SlotSubj, iro::exec::block, Lifetime
    >;
    using Mark = axp::level1::low::MarkConsumed<
        Recipe, SlotSubj, MarkExecGroup, Lifetime, OutTile::bytes
    >;
    using Release = axp::level1::low::ReleaseSlot<
        Recipe, SlotSubj, iro::exec::block, Lifetime
    >;

    using producer_obligations = iro::util::type_list<Issue, Wait>;
    using producer_edges = iro::util::type_list<
        iro::compose::Edge<
            axp::level2::detail::out_port_t<Issue, 0>,
            axp::level2::detail::in_port_t<Wait, 0>
        >
    >;
    using Producer = axp::level2::detail::make_composition_t<producer_obligations, producer_edges,
                                                            iro::profile::BudgetMax, CapT>;

    using consumer_obligations = iro::util::type_list<Mark, Release>;
    using consumer_edges = iro::util::type_list<
        iro::compose::Edge<
            axp::level2::detail::out_port_t<Mark, 0>,
            axp::level2::detail::in_port_t<Release, 0>
        >
    >;
    using Consumer = axp::level2::detail::make_composition_t<consumer_obligations, consumer_edges,
                                                            iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class MarkExecGroup, class Lifetime, int Slots, class SwizzleAtom, class Tma,
         class IssueExecGroup, class CapT = axp::target_cap>
struct StageGmemToSmemStreamingImpl {
    static_assert(std::is_same_v<IssueExecGroup, iro::exec::block>,
                  "StageGmemToSmemStreamingImpl: IssueExecGroup must be block");
    static_assert(std::is_void_v<SwizzleAtom>,
                  "StageGmemToSmemStreamingImpl: swizzle requires async prefetch (cp.async/TMA)");
    using Issue = axp::level1::low::DirectCopy<
        Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj,
        iro::exec::block, Lifetime, Slots, SwizzleAtom
    >;
    using Wait = axp::level1::low::ReadySlot<
        Recipe, OutTile, SlotSubj, iro::exec::block, Lifetime
    >;
    using Mark = axp::level1::low::MarkConsumed<
        Recipe, SlotSubj, MarkExecGroup, Lifetime, OutTile::bytes
    >;
    using Release = axp::level1::low::ReleaseSlot<
        Recipe, SlotSubj, iro::exec::block, Lifetime
    >;

    using producer_obligations = iro::util::type_list<Issue, Wait>;
    using producer_edges = iro::util::type_list<
        iro::compose::Edge<
            axp::level2::detail::out_port_t<Issue, 0>,
            axp::level2::detail::in_port_t<Wait, 0>
        >
    >;
    using Producer = axp::level2::detail::make_composition_t<producer_obligations, producer_edges,
                                                            iro::profile::BudgetMax, CapT>;

    using consumer_obligations = iro::util::type_list<Mark, Release>;
    using consumer_edges = iro::util::type_list<
        iro::compose::Edge<
            axp::level2::detail::out_port_t<Mark, 0>,
            axp::level2::detail::in_port_t<Release, 0>
        >
    >;
    using Consumer = axp::level2::detail::make_composition_t<consumer_obligations, consumer_edges,
                                                            iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class MarkExecGroup, class Lifetime, int Slots, class SwizzleAtom, class Tma,
         class IssueExecGroup, class CapT = axp::target_cap>
struct StageGmemToSmemTmaImpl {
    using traits = tma_traits<Tma>;
    static_assert(traits::valid, "StageGmemToSmemTmaImpl requires TmaConfig");
    using MapHandle = typename traits::MapHandle;
    using BarrierSubj = typename traits::BarrierSubj;
    using Coord0Payload = typename traits::Coord0Payload;
    using Coord0Subj = typename traits::Coord0Subj;
    using Coord1Payload = typename traits::Coord1Payload;
    using Coord1Subj = typename traits::Coord1Subj;
    using MapSubj = typename MapHandle::subject;
    static_assert(std::is_same_v<typename MapHandle::tile, InTile>,
                  "StageGmemToSmemTmaImpl: MapHandle tile must match InTile");

    using Issue = typename tma_copy_selector<traits::has_coord1>::template type<
        Recipe, MapHandle, OutTile, MapSubj, SlotSubj, BarrierSubj,
        Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj, IssueExecGroup, Lifetime
    >;

    using Wait = axp::level1::low::CommitSlot<
        Recipe, OutTile, SlotSubj, BarrierSubj, IssueExecGroup, Lifetime
    >;
    using Mark = axp::level1::low::MarkConsumed<
        Recipe, SlotSubj, MarkExecGroup, Lifetime, OutTile::bytes
    >;
    using Release = axp::level1::low::ReleaseSlot<
        Recipe, SlotSubj, IssueExecGroup, Lifetime
    >;

    using producer_obligations = iro::util::type_list<Issue, Wait>;
    using producer_edges = iro::util::type_list<
        iro::compose::Edge<
            axp::level2::detail::out_port_t<Issue, 0>,
            axp::level2::detail::in_port_t<Wait, 0>
        >,
        iro::compose::Edge<
            axp::level2::detail::out_port_t<Issue, 1>,
            axp::level2::detail::in_port_t<Wait, 1>
        >
    >;
    using Producer = axp::level2::detail::make_composition_t<producer_obligations, producer_edges,
                                                            iro::profile::BudgetMax, CapT>;

    using consumer_obligations = iro::util::type_list<Mark, Release>;
    using consumer_edges = iro::util::type_list<
        iro::compose::Edge<
            axp::level2::detail::out_port_t<Mark, 0>,
            axp::level2::detail::in_port_t<Release, 0>
        >
    >;
    using Consumer = axp::level2::detail::make_composition_t<consumer_obligations, consumer_edges,
                                                            iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class MarkExecGroup, class Lifetime, int Slots, class SwizzleAtom, class Tma,
         class IssueExecGroup, class CapT = axp::target_cap>
struct StageGmemToSmemMulticastTmaImpl {
    using traits = tma_multicast_traits<Tma>;
    static_assert(traits::valid, "StageGmemToSmemMulticastTmaImpl requires TmaMulticastConfig");
    static_assert(std::is_same_v<IssueExecGroup, iro::exec::cluster>,
                  "StageGmemToSmemMulticastTmaImpl: IssueExecGroup must be cluster");
    using MapHandle = typename traits::MapHandle;
    using BarrierSubj = typename traits::BarrierSubj;
    using MaskPayload = typename traits::MaskPayload;
    using MaskSubj = typename traits::MaskSubj;
    using Coord0Payload = typename traits::Coord0Payload;
    using Coord0Subj = typename traits::Coord0Subj;
    using Coord1Payload = typename traits::Coord1Payload;
    using Coord1Subj = typename traits::Coord1Subj;
    using MapSubj = typename MapHandle::subject;
    static_assert(std::is_same_v<typename MapHandle::tile, InTile>,
                  "StageGmemToSmemMulticastTmaImpl: MapHandle tile must match InTile");

    using Issue = typename tma_multicast_copy_selector<traits::has_coord1>::template type<
        Recipe, MapHandle, OutTile, MapSubj, SlotSubj, BarrierSubj,
        MaskPayload, MaskSubj,
        Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj, IssueExecGroup, Lifetime
    >;

    using Wait = axp::level1::low::CommitSlot<
        Recipe, OutTile, SlotSubj, BarrierSubj, iro::exec::block, Lifetime
    >;
    using Mark = axp::level1::low::MarkConsumed<
        Recipe, SlotSubj, MarkExecGroup, Lifetime, OutTile::bytes
    >;
    using Release = axp::level1::low::ReleaseSlot<
        Recipe, SlotSubj, iro::exec::block, Lifetime
    >;

    using producer_obligations = iro::util::type_list<Issue, Wait>;
    using producer_edges = iro::util::type_list<
        iro::compose::Edge<
            axp::level2::detail::out_port_t<Issue, 0>,
            axp::level2::detail::in_port_t<Wait, 0>
        >,
        iro::compose::Edge<
            axp::level2::detail::out_port_t<Issue, 1>,
            axp::level2::detail::in_port_t<Wait, 1>
        >
    >;
    using Producer = axp::level2::detail::make_composition_t<producer_obligations, producer_edges,
                                                            iro::profile::BudgetMax, CapT>;

    using consumer_obligations = iro::util::type_list<Mark, Release>;
    using consumer_edges = iro::util::type_list<
        iro::compose::Edge<
            axp::level2::detail::out_port_t<Mark, 0>,
            axp::level2::detail::in_port_t<Release, 0>
        >
    >;
    using Consumer = axp::level2::detail::make_composition_t<consumer_obligations, consumer_edges,
                                                            iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class InTile, class OutTile, class SlotSubj, class OutSubj,
         class MarkExecGroup, class Lifetime, class Tma = void, class CapT = axp::target_cap>
struct StageSmemToGmemImpl {
    using Issue = axp::level1::low::StoreSmemToGmemSlot<
        Recipe, InTile, OutTile, SlotSubj, OutSubj, iro::exec::block, Lifetime
    >;
    using Wait = axp::level1::low::PassSlot<
        Recipe, SlotSubj, iro::exec::block, Lifetime, InTile::bytes
    >;
    using Mark = axp::level1::low::MarkConsumed<
        Recipe, SlotSubj, MarkExecGroup, Lifetime, InTile::bytes
    >;
    using Release = axp::level1::low::ReleaseSlot<
        Recipe, SlotSubj, iro::exec::block, Lifetime
    >;

    using producer_obligations = iro::util::type_list<Issue, Wait>;
    using producer_edges = iro::util::type_list<
        iro::compose::Edge<
            axp::level2::detail::out_port_t<Issue, 1>,
            axp::level2::detail::in_port_t<Wait, 0>
        >
    >;
    using Producer = axp::level2::detail::make_composition_t<producer_obligations, producer_edges,
                                                            iro::profile::BudgetMax, CapT>;

    using consumer_obligations = iro::util::type_list<Mark, Release>;
    using consumer_edges = iro::util::type_list<
        iro::compose::Edge<
            axp::level2::detail::out_port_t<Mark, 0>,
            axp::level2::detail::in_port_t<Release, 0>
        >
    >;
    using Consumer = axp::level2::detail::make_composition_t<consumer_obligations, consumer_edges,
                                                            iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class InTile, class OutTile, class SlotSubj, class OutSubj,
         class MarkExecGroup, class Lifetime, class Tma, class CapT = axp::target_cap>
struct StageSmemToGmemTmaImpl {
    using traits = tma_traits<Tma>;
    static_assert(traits::valid, "StageSmemToGmemTmaImpl requires TmaConfig");
    using MapHandle = typename traits::MapHandle;
    using BarrierSubj = typename traits::BarrierSubj;
    using Coord0Payload = typename traits::Coord0Payload;
    using Coord0Subj = typename traits::Coord0Subj;
    using Coord1Payload = typename traits::Coord1Payload;
    using Coord1Subj = typename traits::Coord1Subj;
    using MapSubj = typename MapHandle::subject;
    static_assert(std::is_same_v<typename MapHandle::tile, OutTile>,
                  "StageSmemToGmemTmaImpl: MapHandle tile must match OutTile");

    using Issue = typename tma_store_selector<traits::has_coord1>::template type<
        Recipe, MapHandle, InTile, MapSubj, SlotSubj, BarrierSubj,
        Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj, iro::exec::block, Lifetime
    >;

    using Wait = axp::level1::low::CommitStoreSlot<
        Recipe, SlotSubj, BarrierSubj, iro::exec::block, Lifetime, InTile::bytes
    >;
    using Mark = axp::level1::low::MarkConsumed<
        Recipe, SlotSubj, MarkExecGroup, Lifetime, InTile::bytes
    >;
    using Release = axp::level1::low::ReleaseSlot<
        Recipe, SlotSubj, iro::exec::block, Lifetime
    >;

    using producer_obligations = iro::util::type_list<Issue, Wait>;
    using producer_edges = iro::util::type_list<
        iro::compose::Edge<
            axp::level2::detail::out_port_t<Issue, 0>,
            axp::level2::detail::in_port_t<Wait, 1>
        >
    >;
    using Producer = axp::level2::detail::make_composition_t<producer_obligations, producer_edges,
                                                            iro::profile::BudgetMax, CapT>;

    using consumer_obligations = iro::util::type_list<Mark, Release>;
    using consumer_edges = iro::util::type_list<
        iro::compose::Edge<
            axp::level2::detail::out_port_t<Mark, 0>,
            axp::level2::detail::in_port_t<Release, 0>
        >
    >;
    using Consumer = axp::level2::detail::make_composition_t<consumer_obligations, consumer_edges,
                                                            iro::profile::BudgetMax, CapT>;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class MarkExecGroup, class Lifetime, int Slots, class SwizzleAtom = void, class Tma = void,
         class IssueExecGroup = iro::exec::block, class CapT = axp::target_cap>
using StageGmemToSmem = registry::Select<registry::StageGmemToSmemPattern<
    Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj,
    MarkExecGroup, Lifetime, Slots, SwizzleAtom, Tma, IssueExecGroup>, CapT>;

template<class Recipe, class InTile, class OutTile, class SlotSubj, class OutSubj,
         class MarkExecGroup, class Lifetime, class Tma = void, class CapT = axp::target_cap>
using StageSmemToGmem = registry::Select<registry::StageSmemToGmemPattern<
    Recipe, InTile, OutTile, SlotSubj, OutSubj, MarkExecGroup, Lifetime, Tma>, CapT>;

} // namespace axp::level2::staging

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level2::registry {

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class MarkExecGroup, class Lifetime, int Slots, class SwizzleAtom, class Tma, class IssueExecGroup,
         class Cap>
struct resolve_impl<
    StageGmemToSmemPattern<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj,
                           MarkExecGroup, Lifetime, Slots, SwizzleAtom, Tma, IssueExecGroup>,
    Cap,
    std::enable_if_t<axp::level2::staging::streaming_traits<Tma>::valid>> {
    static constexpr bool supported = true;
    using type = axp::level2::staging::StageGmemToSmemStreamingImpl<
        Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, MarkExecGroup, Lifetime, Slots,
        SwizzleAtom, Tma, IssueExecGroup, Cap
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class MarkExecGroup, class Lifetime, int Slots, class SwizzleAtom, class Tma, class IssueExecGroup,
         class Cap>
struct resolve_impl<
    StageGmemToSmemPattern<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj,
                           MarkExecGroup, Lifetime, Slots, SwizzleAtom, Tma, IssueExecGroup>,
    Cap,
    std::enable_if_t<!axp::level2::staging::tma_traits<Tma>::valid &&
                     !axp::level2::staging::tma_multicast_traits<Tma>::valid &&
                     !axp::level2::staging::streaming_traits<Tma>::valid>> {
    static constexpr bool supported = true;
    using type = axp::level2::staging::StageGmemToSmemImpl<
        Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, MarkExecGroup, Lifetime, Slots,
        SwizzleAtom, Tma, IssueExecGroup, Cap
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class MarkExecGroup, class Lifetime, int Slots, class SwizzleAtom, class Tma, class IssueExecGroup,
         class Cap>
struct resolve_impl<
    StageGmemToSmemPattern<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj,
                           MarkExecGroup, Lifetime, Slots, SwizzleAtom, Tma, IssueExecGroup>,
    Cap,
    std::enable_if_t<axp::level2::staging::tma_multicast_traits<Tma>::valid>> {
    static constexpr bool supported = true;
    using type = axp::level2::staging::StageGmemToSmemMulticastTmaImpl<
        Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, MarkExecGroup, Lifetime, Slots,
        SwizzleAtom, Tma, IssueExecGroup, Cap
    >;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class MarkExecGroup, class Lifetime, int Slots, class SwizzleAtom, class Tma, class IssueExecGroup,
         class Cap>
struct resolve_impl<
    StageGmemToSmemPattern<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj,
                           MarkExecGroup, Lifetime, Slots, SwizzleAtom, Tma, IssueExecGroup>,
    Cap,
    std::enable_if_t<axp::level2::staging::tma_traits<Tma>::valid>> {
    static constexpr bool supported = true;
    using type = axp::level2::staging::StageGmemToSmemTmaImpl<
        Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, MarkExecGroup, Lifetime, Slots,
        SwizzleAtom, Tma, IssueExecGroup, Cap
    >;
};

template<class Recipe, class InTile, class OutTile, class SlotSubj, class OutSubj,
         class MarkExecGroup, class Lifetime, class Tma, class Cap>
struct resolve_impl<
    StageSmemToGmemPattern<Recipe, InTile, OutTile, SlotSubj, OutSubj, MarkExecGroup, Lifetime, Tma>,
    Cap,
    std::enable_if_t<!axp::level2::staging::tma_traits<Tma>::valid>> {
    static constexpr bool supported = true;
    using type = axp::level2::staging::StageSmemToGmemImpl<
        Recipe, InTile, OutTile, SlotSubj, OutSubj, MarkExecGroup, Lifetime, Tma, Cap
    >;
};

template<class Recipe, class InTile, class OutTile, class SlotSubj, class OutSubj,
         class MarkExecGroup, class Lifetime, class Tma, class Cap>
struct resolve_impl<
    StageSmemToGmemPattern<Recipe, InTile, OutTile, SlotSubj, OutSubj, MarkExecGroup, Lifetime, Tma>,
    Cap,
    std::enable_if_t<axp::level2::staging::tma_traits<Tma>::valid>> {
    static constexpr bool supported = true;
    using type = axp::level2::staging::StageSmemToGmemTmaImpl<
        Recipe, InTile, OutTile, SlotSubj, OutSubj, MarkExecGroup, Lifetime, Tma, Cap
    >;
};

} // namespace axp::level2::registry
#endif // AXP_LIBRARY_BUILD
