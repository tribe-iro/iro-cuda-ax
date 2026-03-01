#pragma once

#include "../intent.hpp"
#include "../swizzle.hpp"
#include "../naming/subjects.hpp"
#include "../protocol/tma/contracts.hpp"
#include "../level2/staging.hpp"

namespace axp::kit::detail {

struct tma_map_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.kit.tma.map"); };
struct tma_barrier_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.kit.tma.barrier"); };

template<class Subj, class Tag>
using derived_subj = iro::contract::subject::pair<Subj, iro::contract::subject::indexed<Tag, 0>>;

template<class Subj>
using tma_map_subj = derived_subj<Subj, tma_map_tag>;

template<class Subj, class PipeTag>
using tma_barrier_subj = derived_subj<Subj, PipeTag>;

template<class Subj>
using tma_coord0_subj = iro::contract::subject::pair<Subj, axp::subject::Coord0>;

template<class Subj>
using tma_coord1_subj = iro::contract::subject::pair<Subj, axp::subject::Coord1>;

using coord_payload_block = iro::contract::ScalarDesc<iro::elem::i32, iro::dist::uniform<iro::scope::block>>;

template<class GlobalTile, class Subj, class PipeTag>
struct default_tma_config {
    using MapSubj = tma_map_subj<Subj>;
    using MapHandle = axp::protocol::tma::TensorMapHandle<GlobalTile, MapSubj>;
    using BarrierSubj = tma_barrier_subj<Subj, PipeTag>;
    using Coord0Payload = coord_payload_block;
    using Coord1Payload = coord_payload_block;
    using Coord0Subj = tma_coord0_subj<Subj>;
    using Coord1Subj = tma_coord1_subj<Subj>;
    using type = axp::level2::staging::TmaConfig<
        MapHandle, BarrierSubj,
        Coord0Payload, Coord0Subj,
        Coord1Payload, Coord1Subj
    >;
};

template<class Pattern, class Elem, int TileK, class Cap, bool RowMajor>
struct select_swizzle {
    using type = axp::swizzle::None;
};

template<class Elem, int TileK, class Cap>
struct select_swizzle<axp::intent::memory_pattern::Optimized, Elem, TileK, Cap, true> {
    static constexpr int row_bytes = Elem::bytes * TileK;
    using type = std::conditional_t<
        Cap::has_tma && row_bytes >= 128, axp::swizzle::B128,
        std::conditional_t<
            Cap::has_tma && row_bytes >= 64, axp::swizzle::B64,
            std::conditional_t<
                Cap::has_tma && row_bytes >= 32, axp::swizzle::B32,
                axp::swizzle::None
            >
        >
    >;
};

template<class Elem, int TileK, class Cap>
struct select_swizzle<axp::intent::memory_pattern::Optimized, Elem, TileK, Cap, false> {
    static constexpr int row_bytes = Elem::bytes * TileK;
    using type = std::conditional_t<
        Cap::has_tma && row_bytes >= 128, axp::swizzle::B128,
        std::conditional_t<
            Cap::has_tma && row_bytes >= 64, axp::swizzle::B64,
            std::conditional_t<
                Cap::has_tma && row_bytes >= 32, axp::swizzle::B32,
                axp::swizzle::None
            >
        >
    >;
};

template<class LoadMode, class Cap, class GlobalTile, class Subj, class PipeTag>
struct select_tma {
    using type = void;
};

template<class Cap, class GlobalTile, class Subj, class PipeTag>
struct select_tma<axp::intent::load_mode::AsyncPrefetch, Cap, GlobalTile, Subj, PipeTag> {
    using type = std::conditional_t<
        Cap::has_tma,
        typename default_tma_config<GlobalTile, Subj, PipeTag>::type,
        void
    >;
};

template<class Cap, class GlobalTile, class Subj, class PipeTag>
struct select_tma<axp::intent::load_mode::Streaming, Cap, GlobalTile, Subj, PipeTag> {
    using type = axp::level2::staging::streaming_tag;
};

} // namespace axp::kit::detail

namespace axp::kit::expert {

template<class Pattern, class Elem, int TileK, class Cap, bool RowMajor>
struct swizzle_override {
    using type = void;
};

template<class LoadMode, class Cap, class GlobalTile, class Subj, class PipeTag>
struct tma_override {
    using type = void;
};

template<class Schedule, class Cap>
struct schedule_override {
    using type = void;
};

} // namespace axp::kit::expert

namespace axp::kit::detail {

template<class Pattern, class Elem, int TileK, class Cap, bool RowMajor>
using swizzle_override_t = typename axp::kit::expert::swizzle_override<Pattern, Elem, TileK, Cap, RowMajor>::type;

template<class LoadMode, class Cap, class GlobalTile, class Subj, class PipeTag>
using tma_override_t = typename axp::kit::expert::tma_override<LoadMode, Cap, GlobalTile, Subj, PipeTag>::type;

template<class Schedule, class Cap>
using schedule_override_t = typename axp::kit::expert::schedule_override<Schedule, Cap>::type;

template<class Pattern, class Elem, int TileK, class Cap, bool RowMajor>
struct select_swizzle_with_override {
    using override_t = swizzle_override_t<Pattern, Elem, TileK, Cap, RowMajor>;
    using base_t = typename select_swizzle<Pattern, Elem, TileK, Cap, RowMajor>::type;
    using type = std::conditional_t<!std::is_void_v<override_t>, override_t, base_t>;
};

template<class LoadMode, class Cap, class GlobalTile, class Subj, class PipeTag>
struct select_tma_with_override {
    using override_t = tma_override_t<LoadMode, Cap, GlobalTile, Subj, PipeTag>;
    using base_t = typename select_tma<LoadMode, Cap, GlobalTile, Subj, PipeTag>::type;
    using type = std::conditional_t<!std::is_void_v<override_t>, override_t, base_t>;
};

template<class Schedule, class Cap>
struct select_schedule;

template<class Schedule, class Cap>
struct select_schedule_with_override {
    using override_t = schedule_override_t<Schedule, Cap>;
    using base_t = typename select_schedule<Schedule, Cap>::type;
    using type = std::conditional_t<!std::is_void_v<override_t>, override_t, base_t>;
};

template<class Cap>
struct select_schedule<axp::intent::schedule::ProducerConsumer, Cap> {
    static_assert(Cap::has_wgmma, "ProducerConsumer schedule requires WGMMA-capable target");
    using type = axp::intent::schedule::ProducerConsumer;
};

template<class Schedule, class Cap>
struct select_schedule {
    using type = Schedule;
};

template<class Pattern, class Elem, int TileK, class Cap, bool RowMajor>
using select_swizzle_t = typename select_swizzle_with_override<Pattern, Elem, TileK, Cap, RowMajor>::type;

template<class LoadMode, class Cap, class GlobalTile, class Subj, class PipeTag>
using select_tma_t = typename select_tma_with_override<LoadMode, Cap, GlobalTile, Subj, PipeTag>::type;

template<class Schedule, class Cap>
using select_schedule_t = typename select_schedule_with_override<Schedule, Cap>::type;

} // namespace axp::kit::detail
