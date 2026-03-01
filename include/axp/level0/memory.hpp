#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>
#include "detail/tokens.hpp"
#include "../detail/resources.hpp"
#include "memory_cache.hpp"

namespace axp::level0 {

namespace detail {

template<class Tile>
consteval bool require_tile() {
    static_assert(iro::contract::TilePayload<Tile>, "L0 memory ops require Tile payloads");
    return true;
}

template<class InTile, class OutTile>
consteval bool require_same_tile_shape() {
    static_assert(InTile::shape::rank == OutTile::shape::rank, "L0 memory op: rank mismatch");
    static_assert(InTile::shape::size == OutTile::shape::size, "L0 memory op: size mismatch");
    static_assert(std::is_same_v<typename InTile::layout, typename OutTile::layout>,
                  "L0 memory op: layout mismatch");
    static_assert(std::is_same_v<typename InTile::elem, typename OutTile::elem>,
                  "L0 memory op: elem mismatch");
    return true;
}

template<class ExecGroup>
using warpgroup_layout_res = axp::detail::warpgroup_layout_resources_t<ExecGroup>;

template<class Payload>
struct is_index_payload : std::bool_constant<
    iro::contract::ScalarPayload<Payload> || iro::contract::VectorPayload<Payload>
> {};

template<class Payload>
struct is_atomic_payload : std::bool_constant<
    iro::contract::ScalarPayload<Payload> || iro::contract::VectorPayload<Payload>
> {};

template<class Payload>
consteval int value_count();

template<class Payload>
consteval bool index_payload_compatible() {
    static_assert(is_index_payload<Payload>::value, "Index payload must be Scalar or Vector");
    static_assert(std::is_integral_v<typename Payload::elem::storage_t>,
                  "Index payload elem must be integral");
    static_assert(sizeof(typename Payload::elem::storage_t) <= 4,
                  "Index payload elem must be <= 4 bytes");
    return true;
}

template<class Payload>
consteval bool atomic_payload_compatible() {
    static_assert(is_atomic_payload<Payload>::value, "Atomic payload must be Scalar or Vector");
    static_assert(Payload::elem::bytes == 2 || Payload::elem::bytes == 4,
                  "Atomic payload elem must be 2 or 4 bytes");
    return true;
}

template<class Payload>
consteval bool atomic_add_compatible() {
    static_assert(atomic_payload_compatible<Payload>(), "AtomicAdd: invalid payload");
    using E = typename Payload::elem;
    static_assert(std::is_same_v<E, iro::elem::i32> || std::is_same_v<E, iro::elem::u32> ||
                  std::is_same_v<E, iro::elem::f32> || std::is_same_v<E, iro::elem::f16> ||
                  std::is_same_v<E, iro::elem::bf16>,
                  "AtomicAdd supports i32/u32/f32/f16/bf16");
    return true;
}

template<class Payload>
consteval bool atomic_minmax_compatible() {
    static_assert(atomic_payload_compatible<Payload>(), "AtomicMin/Max: invalid payload");
    using E = typename Payload::elem;
    static_assert(std::is_same_v<E, iro::elem::i32> || std::is_same_v<E, iro::elem::u32> ||
                  std::is_same_v<E, iro::elem::f32> || std::is_same_v<E, iro::elem::f16>,
                  "AtomicMin/Max supports i32/u32/f32/f16");
    return true;
}

template<class Payload>
consteval bool atomic_bitwise_compatible() {
    static_assert(atomic_payload_compatible<Payload>(), "AtomicAnd/Or/Xor: invalid payload");
    using T = typename Payload::elem::storage_t;
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>,
                  "AtomicAnd/Or/Xor supports i32/u32");
    return true;
}

template<class Payload>
consteval bool atomic_cas_compatible() {
    static_assert(atomic_payload_compatible<Payload>(), "AtomicCAS: invalid payload");
    using T = typename Payload::elem::storage_t;
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float>,
                  "AtomicCAS supports i32/u32/f32");
    return true;
}

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct atomic_global_base {
    static_assert(detail::is_supported_exec_memory<ExecGroup>::value,
                  "AtomicGlobal requires lane/warp/warpgroup/block exec group");
    static_assert(detail::atomic_payload_compatible<InPayload>(), "AtomicGlobal: invalid InPayload");
    static_assert(detail::atomic_payload_compatible<OutPayload>(), "AtomicGlobal: invalid OutPayload");
    static_assert(detail::index_payload_compatible<IndexPayload>(), "AtomicGlobal: invalid IndexPayload");
    static_assert(detail::value_count<InPayload>() == detail::value_count<IndexPayload>(),
                  "AtomicGlobal: InPayload and IndexPayload must have same lane count");
    static_assert(detail::value_count<OutPayload>() == detail::value_count<IndexPayload>(),
                  "AtomicGlobal: OutPayload and IndexPayload must have same lane count");
    static_assert(detail::require_tile<OutTile>(), "AtomicGlobal: OutTile must be TilePayload");
    static_assert(OutTile::shape::rank == 1, "AtomicGlobal: OutTile must be rank-1");
    static_assert(OutTile::layout::rank == 1, "AtomicGlobal: OutTile layout must be rank-1");
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::global> ||
                  std::is_same_v<typename OutTile::space, iro::contract::space::shared>,
                  "AtomicGlobal: OutTile space must be global or shared");
    static_assert(std::is_same_v<typename InPayload::elem, typename OutTile::elem>,
                  "AtomicGlobal: InPayload elem must match OutTile elem");
    static_assert(std::is_same_v<typename OutPayload::elem, typename OutTile::elem>,
                  "AtomicGlobal: OutPayload elem must match OutTile elem");
    static_assert(std::is_same_v<typename InPayload::elem, typename Recipe::in>,
                  "AtomicGlobal: InPayload elem must match Recipe::in");
    static_assert(std::is_same_v<typename OutPayload::elem, typename Recipe::out>,
                  "AtomicGlobal: OutPayload elem must match Recipe::out");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "AtomicGlobal: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<OutTile, OutSubj, ExecGroup>, InExtra>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            InPayload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InPayload, InSubj, ExecGroup>, InExtra>,
            typename InPayload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            IndexPayload,
            IndexSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<IndexPayload, IndexSubj, ExecGroup>, InExtra>,
            typename IndexPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutPayload,
            OutValSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutPayload, OutValSubj, ExecGroup>, OutExtra>,
            typename OutPayload::dist,
            Recipe
        >,
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutTile, OutSubj, ExecGroup>, OutExtra>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = detail::warpgroup_layout_res<ExecGroup>;
};

template<class Payload>
consteval int value_count() {
    return detail::payload_count<Payload>();
}

} // namespace detail

// Global -> register tile load
// Note: explicit dist required for reg tiles
// Recipe must be explicit and match element type

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CachePolicy = axp::cache::ca,
         class InDist = iro::contract::no_dist, class OutDist = iro::contract::no_dist,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct LdGlobal {
    static_assert(detail::is_supported_exec_memory<ExecGroup>::value,
                  "LdGlobal requires lane/warp/warpgroup/block exec group");
    static_assert(detail::require_tile<InTile>(), "LdGlobal: InTile must be TilePayload");
    static_assert(detail::require_tile<OutTile>(), "LdGlobal: OutTile must be TilePayload");
    static_assert(detail::require_same_tile_shape<InTile, OutTile>(), "LdGlobal: tile mismatch");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::global>,
                  "LdGlobal: InTile space must be global");
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::reg>,
                  "LdGlobal: OutTile space must be reg");
    static_assert(!std::is_same_v<OutDist, iro::contract::no_dist>,
                  "LdGlobal: reg tiles require explicit OutDist");
    static_assert(iro::util::HasId<CachePolicy>, "LdGlobal: CachePolicy must have id");
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>,
                  "LdGlobal: InTile elem must match Recipe::in");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::out>,
                  "LdGlobal: OutTile elem must match Recipe::out");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "LdGlobal: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InTile, InSubj, ExecGroup>, InExtra>,
            InDist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutTile, OutSubj, ExecGroup>, OutExtra>,
            OutDist,
            Recipe
        >
    >;

    using resources = iro::util::concat_t<
        detail::warpgroup_layout_res<ExecGroup>,
        iro::util::type_list<iro::contract::res::cache_policy<CachePolicy>>
    >;
};

// Register tile -> global store

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CachePolicy = axp::cache::wb,
         class InDist = iro::contract::no_dist, class OutDist = iro::contract::no_dist,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct StGlobal {
    static_assert(detail::is_supported_exec_memory<ExecGroup>::value,
                  "StGlobal requires lane/warp/warpgroup/block exec group");
    static_assert(detail::require_tile<InTile>(), "StGlobal: InTile must be TilePayload");
    static_assert(detail::require_tile<OutTile>(), "StGlobal: OutTile must be TilePayload");
    static_assert(detail::require_same_tile_shape<InTile, OutTile>(), "StGlobal: tile mismatch");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::reg>,
                  "StGlobal: InTile space must be reg");
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::global>,
                  "StGlobal: OutTile space must be global");
    static_assert(!std::is_same_v<InDist, iro::contract::no_dist>,
                  "StGlobal: reg tiles require explicit InDist");
    static_assert(iro::util::HasId<CachePolicy>, "StGlobal: CachePolicy must have id");
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>,
                  "StGlobal: InTile elem must match Recipe::in");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::out>,
                  "StGlobal: OutTile elem must match Recipe::out");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "StGlobal: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InTile, InSubj, ExecGroup>, InExtra>,
            InDist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutTile, OutSubj, ExecGroup>, OutExtra>,
            OutDist,
            Recipe
        >
    >;

    using resources = iro::util::concat_t<
        detail::warpgroup_layout_res<ExecGroup>,
        iro::util::type_list<iro::contract::res::cache_policy<CachePolicy>>
    >;
};

// Shared -> register tile load

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class InDist = iro::contract::no_dist, class OutDist = iro::contract::no_dist,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct LdShared {
    static_assert(detail::is_supported_exec_memory<ExecGroup>::value,
                  "LdShared requires lane/warp/warpgroup/block exec group");
    static_assert(detail::require_tile<InTile>(), "LdShared: InTile must be TilePayload");
    static_assert(detail::require_tile<OutTile>(), "LdShared: OutTile must be TilePayload");
    static_assert(detail::require_same_tile_shape<InTile, OutTile>(), "LdShared: tile mismatch");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::shared>,
                  "LdShared: InTile space must be shared");
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::reg>,
                  "LdShared: OutTile space must be reg");
    static_assert(!std::is_same_v<OutDist, iro::contract::no_dist>,
                  "LdShared: reg tiles require explicit OutDist");
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>,
                  "LdShared: InTile elem must match Recipe::in");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::out>,
                  "LdShared: OutTile elem must match Recipe::out");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "LdShared: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InTile, InSubj, ExecGroup>, InExtra>,
            InDist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutTile, OutSubj, ExecGroup>, OutExtra>,
            OutDist,
            Recipe
        >
    >;

    using resources = detail::warpgroup_layout_res<ExecGroup>;
};

// Register -> shared tile store

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class InDist = iro::contract::no_dist, class OutDist = iro::contract::no_dist,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct StShared {
    static_assert(detail::is_supported_exec_memory<ExecGroup>::value,
                  "StShared requires lane/warp/warpgroup/block exec group");
    static_assert(detail::require_tile<InTile>(), "StShared: InTile must be TilePayload");
    static_assert(detail::require_tile<OutTile>(), "StShared: OutTile must be TilePayload");
    static_assert(detail::require_same_tile_shape<InTile, OutTile>(), "StShared: tile mismatch");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::reg>,
                  "StShared: InTile space must be reg");
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::shared>,
                  "StShared: OutTile space must be shared");
    static_assert(!std::is_same_v<InDist, iro::contract::no_dist>,
                  "StShared: reg tiles require explicit InDist");
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>,
                  "StShared: InTile elem must match Recipe::in");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::out>,
                  "StShared: OutTile elem must match Recipe::out");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "StShared: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InTile, InSubj, ExecGroup>, InExtra>,
            InDist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutTile, OutSubj, ExecGroup>, OutExtra>,
            OutDist,
            Recipe
        >
    >;

    using resources = detail::warpgroup_layout_res<ExecGroup>;
};

// Shared/global/register tile zero-fill (explicit initialization)
template<class Recipe, class Tile, class OutSubj, class ExecGroup,
         class OutExtra = iro::util::type_list<>>
struct TileZero {
    static_assert(detail::is_supported_exec_memory<ExecGroup>::value,
                  "TileZero requires lane/warp/warpgroup/block exec group");
    static_assert(detail::require_tile<Tile>(), "TileZero: Tile must be TilePayload");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "TileZero: Recipe must be explicit");

    using inputs = iro::util::type_list<>;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Tile,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Tile, OutSubj, ExecGroup>, OutExtra>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = detail::warpgroup_layout_res<ExecGroup>;
};

// Reduce shared tile into global tile using atomic add (rank-1 only)
template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct ReduceSharedToGlobalAtomicAdd {
    static_assert(detail::is_supported_exec_memory<ExecGroup>::value,
                  "ReduceSharedToGlobalAtomicAdd requires lane/warp/warpgroup/block exec group");
    static_assert(detail::require_tile<InTile>(), "ReduceSharedToGlobalAtomicAdd: InTile must be TilePayload");
    static_assert(detail::require_tile<OutTile>(), "ReduceSharedToGlobalAtomicAdd: OutTile must be TilePayload");
    static_assert(InTile::shape::rank == 1 && OutTile::shape::rank == 1,
                  "ReduceSharedToGlobalAtomicAdd: rank-1 tiles required");
    static_assert(InTile::shape::size == OutTile::shape::size,
                  "ReduceSharedToGlobalAtomicAdd: tile size mismatch");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::shared>,
                  "ReduceSharedToGlobalAtomicAdd: InTile must be shared");
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::global>,
                  "ReduceSharedToGlobalAtomicAdd: OutTile must be global");
    static_assert(std::is_same_v<typename InTile::elem, typename OutTile::elem>,
                  "ReduceSharedToGlobalAtomicAdd: elem mismatch");
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>,
                  "ReduceSharedToGlobalAtomicAdd: elem must match Recipe::in");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::out>,
                  "ReduceSharedToGlobalAtomicAdd: elem must match Recipe::out");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "ReduceSharedToGlobalAtomicAdd: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InTile, InSubj, ExecGroup>, InExtra>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<OutTile, OutSubj, ExecGroup>, InExtra>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutTile, OutSubj, ExecGroup>, OutExtra>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = detail::warpgroup_layout_res<ExecGroup>;
};

// Global gather (indexed) into scalar/vector payload
template<class Recipe, class InTile, class IndexPayload, class OutPayload,
         class InSubj, class IndexSubj, class OutSubj, class ExecGroup,
         class CachePolicy = axp::cache::ca,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct GatherGlobal {
    static_assert(detail::is_supported_exec_memory<ExecGroup>::value,
                  "GatherGlobal requires lane/warp/warpgroup/block exec group");
    static_assert(detail::require_tile<InTile>(), "GatherGlobal: InTile must be TilePayload");
    static_assert(detail::index_payload_compatible<IndexPayload>(), "GatherGlobal: invalid IndexPayload");
    static_assert(detail::is_index_payload<OutPayload>::value, "GatherGlobal: OutPayload must be Scalar or Vector");
    static_assert(InTile::shape::rank == 1, "GatherGlobal: InTile must be rank-1");
    static_assert(InTile::layout::rank == 1, "GatherGlobal: InTile layout must be rank-1");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::global>,
                  "GatherGlobal: InTile space must be global");
    static_assert(detail::value_count<OutPayload>() == detail::value_count<IndexPayload>(),
                  "GatherGlobal: OutPayload and IndexPayload must have same lane count");
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>,
                  "GatherGlobal: InTile elem must match Recipe::in");
    static_assert(std::is_same_v<typename OutPayload::elem, typename Recipe::out>,
                  "GatherGlobal: OutPayload elem must match Recipe::out");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "GatherGlobal: Recipe must be explicit");
    static_assert(iro::util::HasId<CachePolicy>, "GatherGlobal: CachePolicy must have id");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InTile, InSubj, ExecGroup>, InExtra>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            IndexPayload,
            IndexSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<IndexPayload, IndexSubj, ExecGroup>, InExtra>,
            typename IndexPayload::dist,
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

    using resources = iro::util::concat_t<
        detail::warpgroup_layout_res<ExecGroup>,
        iro::util::type_list<iro::contract::res::cache_policy<CachePolicy>>
    >;
};

// Global scatter (indexed) from scalar/vector payload
template<class Recipe, class InPayload, class IndexPayload, class OutTile,
         class InSubj, class IndexSubj, class OutSubj, class ExecGroup,
         class CachePolicy = axp::cache::wb,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct ScatterGlobal {
    static_assert(detail::is_supported_exec_memory<ExecGroup>::value,
                  "ScatterGlobal requires lane/warp/warpgroup/block exec group");
    static_assert(detail::is_index_payload<InPayload>::value, "ScatterGlobal: InPayload must be Scalar or Vector");
    static_assert(detail::index_payload_compatible<IndexPayload>(), "ScatterGlobal: invalid IndexPayload");
    static_assert(detail::require_tile<OutTile>(), "ScatterGlobal: OutTile must be TilePayload");
    static_assert(OutTile::shape::rank == 1, "ScatterGlobal: OutTile must be rank-1");
    static_assert(OutTile::layout::rank == 1, "ScatterGlobal: OutTile layout must be rank-1");
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::global>,
                  "ScatterGlobal: OutTile space must be global");
    static_assert(detail::value_count<InPayload>() == detail::value_count<IndexPayload>(),
                  "ScatterGlobal: InPayload and IndexPayload must have same lane count");
    static_assert(std::is_same_v<typename InPayload::elem, typename Recipe::in>,
                  "ScatterGlobal: InPayload elem must match Recipe::in");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::out>,
                  "ScatterGlobal: OutTile elem must match Recipe::out");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "ScatterGlobal: Recipe must be explicit");
    static_assert(iro::util::HasId<CachePolicy>, "ScatterGlobal: CachePolicy must have id");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InPayload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InPayload, InSubj, ExecGroup>, InExtra>,
            typename InPayload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            IndexPayload,
            IndexSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<IndexPayload, IndexSubj, ExecGroup>, InExtra>,
            typename IndexPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutTile, OutSubj, ExecGroup>, OutExtra>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::concat_t<
        detail::warpgroup_layout_res<ExecGroup>,
        iro::util::type_list<iro::contract::res::cache_policy<CachePolicy>>
    >;
};

// Global atomics (indexed) from scalar/vector payload
template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct AtomicAdd : detail::atomic_global_base<
    Recipe, InPayload, IndexPayload, OutPayload, OutTile,
    InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra> {
    static_assert(detail::atomic_add_compatible<InPayload>(), "AtomicAdd: invalid payload");
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct AtomicMin : detail::atomic_global_base<
    Recipe, InPayload, IndexPayload, OutPayload, OutTile,
    InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra> {
    static_assert(detail::atomic_minmax_compatible<InPayload>(), "AtomicMin: invalid payload");
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct AtomicMax : detail::atomic_global_base<
    Recipe, InPayload, IndexPayload, OutPayload, OutTile,
    InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra> {
    static_assert(detail::atomic_minmax_compatible<InPayload>(), "AtomicMax: invalid payload");
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct AtomicAnd : detail::atomic_global_base<
    Recipe, InPayload, IndexPayload, OutPayload, OutTile,
    InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra> {
    static_assert(detail::atomic_bitwise_compatible<InPayload>(), "AtomicAnd: invalid payload");
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct AtomicOr : detail::atomic_global_base<
    Recipe, InPayload, IndexPayload, OutPayload, OutTile,
    InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra> {
    static_assert(detail::atomic_bitwise_compatible<InPayload>(), "AtomicOr: invalid payload");
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct AtomicXor : detail::atomic_global_base<
    Recipe, InPayload, IndexPayload, OutPayload, OutTile,
    InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra> {
    static_assert(detail::atomic_bitwise_compatible<InPayload>(), "AtomicXor: invalid payload");
};

template<class Recipe, class ComparePayload, class ValuePayload, class IndexPayload, class OutPayload, class OutTile,
         class CompareSubj, class ValueSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct AtomicCAS {
    static_assert(detail::is_supported_exec_memory<ExecGroup>::value,
                  "AtomicCAS requires lane/warp/warpgroup/block exec group");
    static_assert(detail::atomic_cas_compatible<ComparePayload>(), "AtomicCAS: invalid ComparePayload");
    static_assert(detail::atomic_cas_compatible<ValuePayload>(), "AtomicCAS: invalid ValuePayload");
    static_assert(detail::atomic_cas_compatible<OutPayload>(), "AtomicCAS: invalid OutPayload");
    static_assert(detail::index_payload_compatible<IndexPayload>(), "AtomicCAS: invalid IndexPayload");
    static_assert(detail::value_count<ComparePayload>() == detail::value_count<ValuePayload>(),
                  "AtomicCAS: ComparePayload and ValuePayload must have same lane count");
    static_assert(detail::value_count<ComparePayload>() == detail::value_count<IndexPayload>(),
                  "AtomicCAS: ComparePayload and IndexPayload must have same lane count");
    static_assert(detail::value_count<OutPayload>() == detail::value_count<IndexPayload>(),
                  "AtomicCAS: OutPayload and IndexPayload must have same lane count");
    static_assert(detail::require_tile<OutTile>(), "AtomicCAS: OutTile must be TilePayload");
    static_assert(OutTile::shape::rank == 1, "AtomicCAS: OutTile must be rank-1");
    static_assert(OutTile::layout::rank == 1, "AtomicCAS: OutTile layout must be rank-1");
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::global>,
                  "AtomicCAS: OutTile space must be global");
    static_assert(std::is_same_v<typename ComparePayload::elem, typename OutTile::elem>,
                  "AtomicCAS: ComparePayload elem must match OutTile elem");
    static_assert(std::is_same_v<typename ValuePayload::elem, typename OutTile::elem>,
                  "AtomicCAS: ValuePayload elem must match OutTile elem");
    static_assert(std::is_same_v<typename OutPayload::elem, typename OutTile::elem>,
                  "AtomicCAS: OutPayload elem must match OutTile elem");
    static_assert(std::is_same_v<typename ComparePayload::elem, typename Recipe::in>,
                  "AtomicCAS: ComparePayload elem must match Recipe::in");
    static_assert(std::is_same_v<typename OutPayload::elem, typename Recipe::out>,
                  "AtomicCAS: OutPayload elem must match Recipe::out");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "AtomicCAS: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            ComparePayload,
            CompareSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<ComparePayload, CompareSubj, ExecGroup>, InExtra>,
            typename ComparePayload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            ValuePayload,
            ValueSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<ValuePayload, ValueSubj, ExecGroup>, InExtra>,
            typename ValuePayload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            IndexPayload,
            IndexSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<IndexPayload, IndexSubj, ExecGroup>, InExtra>,
            typename IndexPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutPayload,
            OutValSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutPayload, OutValSubj, ExecGroup>, OutExtra>,
            typename OutPayload::dist,
            Recipe
        >,
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutTile, OutSubj, ExecGroup>, OutExtra>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = detail::warpgroup_layout_res<ExecGroup>;
};

// Global prefetch (no outputs)
template<class Recipe, class InTile, class InSubj, class ExecGroup,
         class CachePolicy = axp::cache::cg,
         class InDist = iro::contract::no_dist, class InExtra = iro::util::type_list<>>
struct PrefetchGlobal {
    static_assert(detail::is_supported_exec_memory<ExecGroup>::value,
                  "PrefetchGlobal requires lane/warp/warpgroup/block exec group");
    static_assert(detail::require_tile<InTile>(), "PrefetchGlobal: InTile must be TilePayload");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::global>,
                  "PrefetchGlobal: InTile space must be global");
    static_assert(iro::util::HasId<CachePolicy>, "PrefetchGlobal: CachePolicy must have id");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InTile, InSubj, ExecGroup>, InExtra>,
            InDist,
            Recipe
        >
    >;
    using outputs = iro::util::type_list<>;
    using resources = iro::util::concat_t<
        detail::warpgroup_layout_res<ExecGroup>,
        iro::util::type_list<iro::contract::res::cache_policy<CachePolicy>>
    >;
};

// Shared-memory tile fence (block sync for a tile subject)
template<class Recipe, class Tile, class Subj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct TileFence {
    static_assert(detail::is_supported_exec_memory<ExecGroup>::value,
                  "TileFence requires lane/warp/warpgroup/block exec group");
    static_assert(detail::require_tile<Tile>(), "TileFence: Tile must be TilePayload");
    static_assert(std::is_same_v<typename Tile::space, iro::contract::space::shared>,
                  "TileFence: Tile must be shared space");
    static_assert(std::is_same_v<ExecGroup, iro::exec::block> ||
                  iro::exec::is_warpgroup_v<ExecGroup>,
                  "TileFence: ExecGroup must be block or warpgroup");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "TileFence: Recipe must be explicit");

    // Fence accepts minimally-visible shared data (warp scope) and upgrades to block/warpgroup sync.
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Tile,
            Subj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<Subj, iro::scope::warp>,
                    iro::token::alive<Subj, iro::token::lifetime::warp>
                >,
                detail::participation_tokens<Subj, ExecGroup>,
                InExtra
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Tile,
            Subj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::concat_t<
                    detail::out_tokens<Tile, Subj, ExecGroup>,
                    iro::util::type_list<iro::token::lanes_valid<Subj, 32>>
                >,
                OutExtra
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::concat_t<
        detail::warpgroup_layout_res<ExecGroup>,
        std::conditional_t<
            iro::exec::is_warpgroup_v<ExecGroup>,
            iro::util::type_list<iro::contract::res::warpgroup_barrier<Subj, 1>>,
            iro::util::type_list<>
        >
    >;
};

// Swizzled shared-memory load (Swizzled -> RowMajor)
template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup, class SwizzleAtom,
         class InDist = iro::contract::no_dist, class OutDist = iro::contract::no_dist,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct SwizzledLdShared {
    static_assert(detail::is_supported_exec_memory<ExecGroup>::value,
                  "SwizzledLdShared requires lane/warp/warpgroup/block exec group");
    static_assert(detail::require_tile<InTile>(), "SwizzledLdShared: InTile must be TilePayload");
    static_assert(detail::require_tile<OutTile>(), "SwizzledLdShared: OutTile must be TilePayload");
    static_assert(InTile::shape::rank == 2 && OutTile::shape::rank == 2, "SwizzledLdShared: rank-2 tiles required");
    static_assert(InTile::shape::template dim<0>() == OutTile::shape::template dim<0>() &&
                  InTile::shape::template dim<1>() == OutTile::shape::template dim<1>(),
                  "SwizzledLdShared: shape mismatch");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::shared>,
                  "SwizzledLdShared: InTile must be shared");
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::reg>,
                  "SwizzledLdShared: OutTile must be reg");
    static_assert(!std::is_same_v<OutDist, iro::contract::no_dist>,
                  "SwizzledLdShared: reg tiles require explicit OutDist");
    static_assert(std::is_same_v<typename InTile::elem, typename OutTile::elem>,
                  "SwizzledLdShared: elem mismatch");
    static_assert(std::is_same_v<typename InTile::layout,
                  iro::contract::layout::Swizzled<
                      InTile::shape::template dim<1>(),
                      SwizzleAtom::B,
                      SwizzleAtom::S>>,
                  "SwizzledLdShared: InTile layout must be Swizzled<RowMajorCols,B,S>");
    static_assert(std::is_same_v<typename OutTile::layout,
                  iro::contract::layout::RowMajor<OutTile::shape::template dim<1>()>>,
                  "SwizzledLdShared: OutTile layout must be RowMajor");
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>,
                  "SwizzledLdShared: InTile elem must match Recipe::in");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::out>,
                  "SwizzledLdShared: OutTile elem must match Recipe::out");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "SwizzledLdShared: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InTile, InSubj, ExecGroup>, InExtra>,
            InDist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutTile, OutSubj, ExecGroup>, OutExtra>,
            OutDist,
            Recipe
        >
    >;

    using resources = detail::warpgroup_layout_res<ExecGroup>;
};

// Swizzled shared-memory store (RowMajor -> Swizzled)
template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup, class SwizzleAtom,
         class InDist = iro::contract::no_dist, class OutDist = iro::contract::no_dist,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct SwizzledStShared {
    static_assert(detail::is_supported_exec_memory<ExecGroup>::value,
                  "SwizzledStShared requires lane/warp/warpgroup/block exec group");
    static_assert(detail::require_tile<InTile>(), "SwizzledStShared: InTile must be TilePayload");
    static_assert(detail::require_tile<OutTile>(), "SwizzledStShared: OutTile must be TilePayload");
    static_assert(InTile::shape::rank == 2 && OutTile::shape::rank == 2, "SwizzledStShared: rank-2 tiles required");
    static_assert(InTile::shape::template dim<0>() == OutTile::shape::template dim<0>() &&
                  InTile::shape::template dim<1>() == OutTile::shape::template dim<1>(),
                  "SwizzledStShared: shape mismatch");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::reg>,
                  "SwizzledStShared: InTile must be reg");
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::shared>,
                  "SwizzledStShared: OutTile must be shared");
    static_assert(!std::is_same_v<InDist, iro::contract::no_dist>,
                  "SwizzledStShared: reg tiles require explicit InDist");
    static_assert(std::is_same_v<typename InTile::elem, typename OutTile::elem>,
                  "SwizzledStShared: elem mismatch");
    static_assert(std::is_same_v<typename OutTile::layout,
                  iro::contract::layout::Swizzled<
                      OutTile::shape::template dim<1>(),
                      SwizzleAtom::B,
                      SwizzleAtom::S>>,
                  "SwizzledStShared: OutTile layout must be Swizzled<RowMajorCols,B,S>");
    static_assert(std::is_same_v<typename InTile::layout,
                  iro::contract::layout::RowMajor<InTile::shape::template dim<1>()>>,
                  "SwizzledStShared: InTile layout must be RowMajor");
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>,
                  "SwizzledStShared: InTile elem must match Recipe::in");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::out>,
                  "SwizzledStShared: OutTile elem must match Recipe::out");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "SwizzledStShared: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<InTile, InSubj, ExecGroup>, InExtra>,
            InDist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<OutTile, OutSubj, ExecGroup>, OutExtra>,
            OutDist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

} // namespace axp::level0
