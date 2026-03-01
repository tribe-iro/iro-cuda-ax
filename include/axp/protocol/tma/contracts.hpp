#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>
#include "../sync/bundles.hpp"
#include "../stage/bundles.hpp"
#include "../../bundles/token_bundles.hpp"

namespace axp::protocol::tma {

namespace detail {
template<class Payload>
consteval bool coord_payload_ok_block() {
    static_assert(iro::contract::ScalarPayload<Payload>, "TMA coord must be Scalar payload");
    static_assert(std::is_same_v<typename Payload::elem, iro::elem::i32> ||
                  std::is_same_v<typename Payload::elem, iro::elem::u32>,
                  "TMA coord must use i32/u32 element type");
    static_assert(std::is_same_v<typename Payload::dist, iro::dist::uniform<iro::scope::block>>,
                  "TMA coord must be block-uniform");
    return true;
}

template<class Payload>
consteval bool coord_payload_ok_cluster() {
    static_assert(iro::contract::ScalarPayload<Payload>, "TMA coord must be Scalar payload");
    static_assert(std::is_same_v<typename Payload::elem, iro::elem::i32> ||
                  std::is_same_v<typename Payload::elem, iro::elem::u32>,
                  "TMA coord must use i32/u32 element type");
    static_assert(std::is_same_v<typename Payload::dist, iro::dist::uniform<iro::scope::cluster>>,
                  "TMA coord must be cluster-uniform");
    return true;
}

template<class Payload>
consteval bool mask_payload_ok_block() {
    static_assert(iro::contract::ScalarPayload<Payload>, "TMA multicast mask must be Scalar payload");
    static_assert(std::is_same_v<typename Payload::elem, iro::elem::u32> ||
                  std::is_same_v<typename Payload::elem, iro::elem::i32>,
                  "TMA multicast mask must use i32/u32 element type");
    static_assert(std::is_same_v<typename Payload::dist, iro::dist::uniform<iro::scope::block>>,
                  "TMA multicast mask must be block-uniform");
    return true;
}

template<class Payload>
consteval bool mask_payload_ok_cluster() {
    static_assert(iro::contract::ScalarPayload<Payload>, "TMA multicast mask must be Scalar payload");
    static_assert(std::is_same_v<typename Payload::elem, iro::elem::u32> ||
                  std::is_same_v<typename Payload::elem, iro::elem::i32>,
                  "TMA multicast mask must use i32/u32 element type");
    static_assert(std::is_same_v<typename Payload::dist, iro::dist::uniform<iro::scope::cluster>>,
                  "TMA multicast mask must be cluster-uniform");
    return true;
}
} // namespace detail

// TensorMap handle (opaque descriptor)
template<class GlobalTile, class MapSubj>
struct TensorMapHandle {
    static_assert(std::is_same_v<typename GlobalTile::space, iro::contract::space::global>,
                  "TensorMapHandle requires global-space tile");
    static_assert(iro::util::HasId<MapSubj>, "TensorMapHandle requires subject id");
    static constexpr iro::util::u64 id = iro::util::mix_u64(
        iro::util::fnv1a_64_cstr("axp.tma.tensor_map"),
        iro::util::mix_u64(GlobalTile::id, MapSubj::id));
    using tile = GlobalTile;
    using subject = MapSubj;
};

// Host-side tensor map creation (explicit, no hidden effects)
template<class Recipe, class MapHandle, class MapSubj, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct HostMakeTensorMap {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>,
                  "HostMakeTensorMap requires block exec group");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "HostMakeTensorMap requires explicit Recipe");
    static_assert(std::is_same_v<typename MapHandle::subject, MapSubj>,
                  "HostMakeTensorMap: MapSubj mismatch");

    using inputs = iro::util::type_list<>;
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            MapHandle,
            MapSubj,
            ExecGroup,
            iro::token::bundle_list<axp::bundle::GmemVisible<MapSubj, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using resources = iro::util::type_list<>;
};

// Bulk TMA copy (2D) global -> shared
template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct BulkTmaCopy2D {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BulkTmaCopy2D requires block exec group");
    static_assert(std::is_same_v<typename SmemTile::space, iro::contract::space::shared>,
                  "BulkTmaCopy2D requires shared-memory destination");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "BulkTmaCopy2D requires explicit Recipe");
    static_assert(Bytes > 0, "BulkTmaCopy2D requires positive byte count");
    static_assert(detail::coord_payload_ok_block<Coord0Payload>());
    static_assert(detail::coord_payload_ok_block<Coord1Payload>());

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            MapHandle,
            MapSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<MapSubj, iro::scope::device>,
                iro::token::alive<MapSubj, iro::token::lifetime::block>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            sync::BarrierHandle,
            BarrierSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::token::bundle_list<sync::BarrierReady<BarrierSubj, ExecGroup>>,
                iro::token::bundle_list<axp::bundle::LeaderIssued<BarrierSubj>>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            Coord0Payload,
            Coord0Subj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<Coord0Subj, iro::scope::block>,
                    iro::token::alive<Coord0Subj, iro::token::lifetime::instruction>
                >,
                iro::token::bundle_list<axp::bundle::LeaderIssued<Coord0Subj>>
            >,
            typename Coord0Payload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            Coord1Payload,
            Coord1Subj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<Coord1Subj, iro::scope::block>,
                    iro::token::alive<Coord1Subj, iro::token::lifetime::instruction>
                >,
                iro::token::bundle_list<axp::bundle::LeaderIssued<Coord1Subj>>
            >,
            typename Coord1Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            stage::SlotHandle,
            SmemSubj,
            ExecGroup,
            iro::token::bundle_list<stage::ProducerIssued<SmemSubj, Lifetime, Bytes>>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::OutputPort<
            sync::BarrierHandle,
            BarrierSubj,
            ExecGroup,
            iro::token::bundle_list<sync::BarrierArrived<BarrierSubj, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<BarrierSubj>
    >;
};

// Bulk TMA copy (1D) global -> shared
template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class Coord0Payload, class Coord0Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct BulkTmaCopy1D {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BulkTmaCopy1D requires block exec group");
    static_assert(std::is_same_v<typename SmemTile::space, iro::contract::space::shared>,
                  "BulkTmaCopy1D requires shared-memory destination");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "BulkTmaCopy1D requires explicit Recipe");
    static_assert(Bytes > 0, "BulkTmaCopy1D requires positive byte count");
    static_assert(detail::coord_payload_ok_block<Coord0Payload>());

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            MapHandle,
            MapSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<MapSubj, iro::scope::device>,
                iro::token::alive<MapSubj, iro::token::lifetime::block>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            sync::BarrierHandle,
            BarrierSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::token::bundle_list<sync::BarrierReady<BarrierSubj, ExecGroup>>,
                iro::token::bundle_list<axp::bundle::LeaderIssued<BarrierSubj>>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            Coord0Payload,
            Coord0Subj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<Coord0Subj, iro::scope::block>,
                    iro::token::alive<Coord0Subj, iro::token::lifetime::instruction>
                >,
                iro::token::bundle_list<axp::bundle::LeaderIssued<Coord0Subj>>
            >,
            typename Coord0Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            stage::SlotHandle,
            SmemSubj,
            ExecGroup,
            iro::token::bundle_list<stage::ProducerIssued<SmemSubj, Lifetime, Bytes>>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::OutputPort<
            sync::BarrierHandle,
            BarrierSubj,
            ExecGroup,
            iro::token::bundle_list<sync::BarrierArrived<BarrierSubj, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<BarrierSubj>
    >;
};

// Bulk TMA multicast copy (1D) global -> shared (cluster multicast)
template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class MaskPayload, class MaskSubj,
         class Coord0Payload, class Coord0Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct BulkTmaCopyMulticast1D {
    static_assert(std::is_same_v<ExecGroup, iro::exec::cluster>,
                  "BulkTmaCopyMulticast1D requires cluster exec group");
    static_assert(std::is_same_v<typename SmemTile::space, iro::contract::space::shared>,
                  "BulkTmaCopyMulticast1D requires shared-memory destination");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "BulkTmaCopyMulticast1D requires explicit Recipe");
    static_assert(Bytes > 0, "BulkTmaCopyMulticast1D requires positive byte count");
    static_assert(detail::coord_payload_ok_cluster<Coord0Payload>());
    static_assert(detail::mask_payload_ok_cluster<MaskPayload>());

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            MapHandle,
            MapSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<MapSubj, iro::scope::device>,
                iro::token::alive<MapSubj, iro::token::lifetime::block>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            sync::BarrierHandle,
            BarrierSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::token::bundle_list<sync::BarrierReady<BarrierSubj, ExecGroup>>,
                iro::token::bundle_list<axp::bundle::LeaderIssued<BarrierSubj>>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            MaskPayload,
            MaskSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<MaskSubj, iro::scope::cluster>,
                    iro::token::alive<MaskSubj, iro::token::lifetime::instruction>
                >,
                iro::token::bundle_list<axp::bundle::LeaderIssued<MaskSubj>>
            >,
            typename MaskPayload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            Coord0Payload,
            Coord0Subj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<Coord0Subj, iro::scope::cluster>,
                    iro::token::alive<Coord0Subj, iro::token::lifetime::instruction>
                >,
                iro::token::bundle_list<axp::bundle::LeaderIssued<Coord0Subj>>
            >,
            typename Coord0Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            stage::SlotHandle,
            SmemSubj,
            ExecGroup,
            iro::token::bundle_list<stage::ProducerIssued<SmemSubj, Lifetime, Bytes>>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::OutputPort<
            sync::BarrierHandle,
            BarrierSubj,
            ExecGroup,
            iro::token::bundle_list<sync::BarrierArrived<BarrierSubj, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<BarrierSubj>
    >;
};

// Bulk TMA multicast copy (2D) global -> shared (cluster multicast)
template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class MaskPayload, class MaskSubj,
         class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct BulkTmaCopyMulticast2D {
    static_assert(std::is_same_v<ExecGroup, iro::exec::cluster>,
                  "BulkTmaCopyMulticast2D requires cluster exec group");
    static_assert(std::is_same_v<typename SmemTile::space, iro::contract::space::shared>,
                  "BulkTmaCopyMulticast2D requires shared-memory destination");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "BulkTmaCopyMulticast2D requires explicit Recipe");
    static_assert(Bytes > 0, "BulkTmaCopyMulticast2D requires positive byte count");
    static_assert(detail::coord_payload_ok_cluster<Coord0Payload>());
    static_assert(detail::coord_payload_ok_cluster<Coord1Payload>());
    static_assert(detail::mask_payload_ok_cluster<MaskPayload>());

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            MapHandle,
            MapSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<MapSubj, iro::scope::device>,
                iro::token::alive<MapSubj, iro::token::lifetime::block>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            sync::BarrierHandle,
            BarrierSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::token::bundle_list<sync::BarrierReady<BarrierSubj, ExecGroup>>,
                iro::token::bundle_list<axp::bundle::LeaderIssued<BarrierSubj>>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            MaskPayload,
            MaskSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<MaskSubj, iro::scope::cluster>,
                    iro::token::alive<MaskSubj, iro::token::lifetime::instruction>
                >,
                iro::token::bundle_list<axp::bundle::LeaderIssued<MaskSubj>>
            >,
            typename MaskPayload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            Coord0Payload,
            Coord0Subj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<Coord0Subj, iro::scope::cluster>,
                    iro::token::alive<Coord0Subj, iro::token::lifetime::instruction>
                >,
                iro::token::bundle_list<axp::bundle::LeaderIssued<Coord0Subj>>
            >,
            typename Coord0Payload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            Coord1Payload,
            Coord1Subj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<Coord1Subj, iro::scope::cluster>,
                    iro::token::alive<Coord1Subj, iro::token::lifetime::instruction>
                >,
                iro::token::bundle_list<axp::bundle::LeaderIssued<Coord1Subj>>
            >,
            typename Coord1Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            stage::SlotHandle,
            SmemSubj,
            ExecGroup,
            iro::token::bundle_list<stage::ProducerIssued<SmemSubj, Lifetime, Bytes>>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::OutputPort<
            sync::BarrierHandle,
            BarrierSubj,
            ExecGroup,
            iro::token::bundle_list<sync::BarrierArrived<BarrierSubj, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<BarrierSubj>
    >;
};

// Multicast copy selector (1D vs 2D)
template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class MaskPayload, class MaskSubj,
         class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
         class ExecGroup, class Lifetime, long long Bytes, class Enable = void>
struct BulkTmaCopyMulticast;

template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class MaskPayload, class MaskSubj,
         class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct BulkTmaCopyMulticast<Recipe, MapHandle, SmemTile,
                            MapSubj, SmemSubj, BarrierSubj,
                            MaskPayload, MaskSubj,
                            Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj,
                            ExecGroup, Lifetime, Bytes,
                            std::enable_if_t<!std::is_void_v<Coord1Payload>>> {
    using type = BulkTmaCopyMulticast2D<
        Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
        MaskPayload, MaskSubj,
        Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj,
        ExecGroup, Lifetime, Bytes
    >;
};

template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class MaskPayload, class MaskSubj,
         class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct BulkTmaCopyMulticast<Recipe, MapHandle, SmemTile,
                            MapSubj, SmemSubj, BarrierSubj,
                            MaskPayload, MaskSubj,
                            Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj,
                            ExecGroup, Lifetime, Bytes,
                            std::enable_if_t<std::is_void_v<Coord1Payload>>> {
    using type = BulkTmaCopyMulticast1D<
        Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
        MaskPayload, MaskSubj,
        Coord0Payload, Coord0Subj,
        ExecGroup, Lifetime, Bytes
    >;
};

// Bulk TMA store (2D) shared -> global
template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct BulkTmaStore2D {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BulkTmaStore2D requires block exec group");
    static_assert(std::is_same_v<typename SmemTile::space, iro::contract::space::shared>,
                  "BulkTmaStore2D requires shared-memory source");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "BulkTmaStore2D requires explicit Recipe");
    static_assert(Bytes > 0, "BulkTmaStore2D requires positive byte count");
    static_assert(detail::coord_payload_ok_block<Coord0Payload>());
    static_assert(detail::coord_payload_ok_block<Coord1Payload>());

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            MapHandle,
            MapSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<MapSubj, iro::scope::device>,
                iro::token::alive<MapSubj, iro::token::lifetime::block>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            SmemTile,
            SmemSubj,
            ExecGroup,
            iro::token::bundle_list<axp::bundle::SmemReady<SmemSubj, ExecGroup, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            sync::BarrierHandle,
            BarrierSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::token::bundle_list<sync::BarrierReady<BarrierSubj, ExecGroup>>,
                iro::token::bundle_list<axp::bundle::LeaderIssued<BarrierSubj>>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            Coord0Payload,
            Coord0Subj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<Coord0Subj, iro::scope::block>,
                    iro::token::alive<Coord0Subj, iro::token::lifetime::instruction>
                >,
                iro::token::bundle_list<axp::bundle::LeaderIssued<Coord0Subj>>
            >,
            typename Coord0Payload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            Coord1Payload,
            Coord1Subj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<Coord1Subj, iro::scope::block>,
                    iro::token::alive<Coord1Subj, iro::token::lifetime::instruction>
                >,
                iro::token::bundle_list<axp::bundle::LeaderIssued<Coord1Subj>>
            >,
            typename Coord1Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            sync::BarrierHandle,
            BarrierSubj,
            ExecGroup,
            iro::token::bundle_list<sync::BarrierArrived<BarrierSubj, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<BarrierSubj>
    >;
};

template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class Coord0Payload, class Coord0Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct BulkTmaStore1D {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BulkTmaStore1D requires block exec group");
    static_assert(std::is_same_v<typename SmemTile::space, iro::contract::space::shared>,
                  "BulkTmaStore1D requires shared-memory source");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "BulkTmaStore1D requires explicit Recipe");
    static_assert(Bytes > 0, "BulkTmaStore1D requires positive byte count");
    static_assert(detail::coord_payload_ok_block<Coord0Payload>());

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            MapHandle,
            MapSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<MapSubj, iro::scope::device>,
                iro::token::alive<MapSubj, iro::token::lifetime::block>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            SmemTile,
            SmemSubj,
            ExecGroup,
            iro::token::bundle_list<axp::bundle::SmemReady<SmemSubj, ExecGroup, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            sync::BarrierHandle,
            BarrierSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::token::bundle_list<sync::BarrierReady<BarrierSubj, ExecGroup>>,
                iro::token::bundle_list<axp::bundle::LeaderIssued<BarrierSubj>>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            Coord0Payload,
            Coord0Subj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<Coord0Subj, iro::scope::block>,
                    iro::token::alive<Coord0Subj, iro::token::lifetime::instruction>
                >,
                iro::token::bundle_list<axp::bundle::LeaderIssued<Coord0Subj>>
            >,
            typename Coord0Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            sync::BarrierHandle,
            BarrierSubj,
            ExecGroup,
            iro::token::bundle_list<sync::BarrierArrived<BarrierSubj, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<BarrierSubj>
    >;
};

} // namespace axp::protocol::tma
