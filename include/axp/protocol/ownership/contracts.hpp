#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>
#include "../../detail/resources.hpp"
#include "../../detail/participation_tokens.hpp"
#include "dist_tags.hpp"
#include "bundles.hpp"
#include "../../bundles/token_bundles.hpp"
#include "../stage/resources.hpp"

namespace axp::protocol::ownership {

template<class ShapeT, class ElemT, class DistT>
using Fragment = iro::contract::FragmentDesc<ShapeT, ElemT, DistT>;

namespace detail {
template<class Subject, class ExecGroup>
using participation_tokens = axp::detail::participation_tokens<Subject, ExecGroup>;
} // namespace detail

// ---------------------------------------------------------------------------
// Tile boundary adapters (explicit TileIn/TileOut token enforcement)
// ---------------------------------------------------------------------------

template<class Recipe, class Tile, class Subj, class ExecGroup, class Lifetime,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct TileBoundaryIn {
    static_assert(iro::contract::TilePayload<Tile>, "TileBoundaryIn: Tile must be TilePayload");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "TileBoundaryIn: Recipe must be explicit");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Tile,
            Subj,
            ExecGroup,
            iro::util::concat_t<
                axp::bundle::TileInTokens<Subj, ExecGroup, Lifetime>,
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
                axp::bundle::TileInTokens<Subj, ExecGroup, Lifetime>,
                OutExtra
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

template<class Recipe, class Tile, class Subj, class ExecGroup, class Lifetime,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct TileBoundaryOut {
    static_assert(iro::contract::TilePayload<Tile>, "TileBoundaryOut: Tile must be TilePayload");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "TileBoundaryOut: Recipe must be explicit");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Tile,
            Subj,
            ExecGroup,
            iro::util::concat_t<
                axp::bundle::TileOutTokens<Subj, ExecGroup, Lifetime>,
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
                axp::bundle::TileOutTokens<Subj, ExecGroup, Lifetime>,
                OutExtra
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Shared Tile -> Fragment adapter (no slot_state required)
template<class Recipe, class InTile, class Frag, class InSubj, class FragSubj, class ExecGroup, class Lifetime,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct SharedTileToFragment {
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>, "SharedTileToFragment: InTile elem != Recipe::in");
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::in>, "SharedTileToFragment: Frag elem != Recipe::in");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "SharedTileToFragment requires warp exec group");
    using out_lifetime = iro::token::lifetime::warp;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<InSubj, iro::scope::min_scope_for_t<ExecGroup>>,
                    iro::token::alive<InSubj, Lifetime>,
                    iro::token::lanes_valid<InSubj, 32>
                >,
                InExtra
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Frag,
            FragSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::alive<FragSubj, out_lifetime>,
                    iro::token::lanes_valid<FragSubj, 32>
                >,
                OutExtra
            >,
            typename Frag::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Fragment -> Shared Tile adapter (no slot_state required)
template<class Recipe, class Frag, class OutTile, class FragSubj, class OutSubj, class ExecGroup, class Lifetime,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct FragmentToSharedTile {
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::acc>, "FragmentToSharedTile: Frag elem != Recipe::acc");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::out>, "FragmentToSharedTile: OutTile elem != Recipe::out");
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp> || iro::exec::is_warpgroup_v<ExecGroup>,
                  "FragmentToSharedTile requires warp or warpgroup exec group");
    // Shared stores provide visibility at the executing group scope; TileFence upgrades to block/warpgroup sync.
    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;
    using out_lifetime = std::conditional_t<
        iro::exec::is_warpgroup_v<ExecGroup>,
        iro::token::lifetime::warpgroup,
        iro::token::lifetime::warp
    >;
    using out_tokens = iro::util::type_list<
        iro::token::visible_at<OutSubj, scope_t>,
        iro::token::alive<OutSubj, out_lifetime>
    >;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Frag,
            FragSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::concat_t<
                    iro::util::type_list<iro::token::alive<FragSubj, Lifetime>>,
                    detail::participation_tokens<FragSubj, ExecGroup>
                >,
                InExtra
            >,
            typename Frag::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::concat_t<out_tokens, detail::participation_tokens<OutSubj, ExecGroup>>,
                OutExtra
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Tile -> Fragment adapter (shared-memory tile to register fragment)
template<class Recipe, class InTile, class Frag, class InSubj, class FragSubj, class ExecGroup, class Lifetime>
struct TileToFragment {
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::in>, "TileToFragment: InTile elem != Recipe::in");
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::in>, "TileToFragment: Frag elem != Recipe::in");
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "TileToFragment requires warp exec group");
    using out_lifetime = iro::token::lifetime::warp;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<InSubj, iro::scope::min_scope_for_t<ExecGroup>>,
                iro::token::alive<InSubj, Lifetime>,
                iro::token::slot_state<InSubj, iro::token::state::ready>,
                iro::token::lanes_valid<InSubj, 32>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Frag,
            FragSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<FragSubj, out_lifetime>,
                iro::token::lanes_valid<FragSubj, 32>
            >,
            typename Frag::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Fragment -> Tile adapter (register fragment to shared-memory tile)
template<class Recipe, class Frag, class OutTile, class FragSubj, class OutSubj, class ExecGroup, class Lifetime>
struct FragmentToTile {
    static_assert(std::is_same_v<typename Frag::elem, typename Recipe::acc>, "FragmentToTile: Frag elem != Recipe::acc");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::out>, "FragmentToTile: OutTile elem != Recipe::out");
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "FragmentToTile requires warp exec group");
    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;
    using base_tokens = iro::util::type_list<
        iro::token::visible_at<OutSubj, scope_t>,
        iro::token::alive<OutSubj, Lifetime>,
        iro::token::lanes_valid<OutSubj, 32>
    >;
    using sync_tokens = iro::util::type_list<
        iro::token::visible_at<OutSubj, scope_t>,
        iro::token::alive<OutSubj, Lifetime>,
        iro::token::lanes_valid<OutSubj, 32>,
        iro::token::sync_at<OutSubj, scope_t>
    >;
    using out_tokens = std::conditional_t<(scope_t::level >= iro::scope::warpgroup::level), sync_tokens, base_tokens>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Frag,
            FragSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<FragSubj, Lifetime>,
                iro::token::lanes_valid<FragSubj, 32>
            >,
            typename Frag::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            out_tokens,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// WGMMA SMEM descriptor handle (opaque)
template<class SmemTile, class SmemSubj, class SwizzleAtom>
struct WgmmaSmemDesc {
    static_assert(std::is_same_v<typename SmemTile::space, iro::contract::space::shared>);
    static_assert(iro::util::HasId<SwizzleAtom>);
    static_assert(iro::util::HasId<SmemSubj>);
    static constexpr iro::util::u64 id = iro::util::mix_u64(
        iro::util::fnv1a_64_cstr("axp.wgmma.smem_desc"),
        iro::util::mix_u64(SmemTile::id,
            iro::util::mix_u64(SmemSubj::id, SwizzleAtom::id)));
    using tile = SmemTile;
    using subject = SmemSubj;
    using swizzle = SwizzleAtom;
};

// Reuse an existing WGMMA SMEM descriptor while enforcing current slot readiness.
template<class Recipe, class SmemTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime, class SwizzleAtom>
struct UseWgmmaSmemDesc {
    static_assert(std::is_same_v<typename SmemTile::elem, typename Recipe::in>, "UseWgmmaSmemDesc: SmemTile elem != Recipe::in");
    static_assert(std::is_same_v<typename SmemTile::space, iro::contract::space::shared>);
    static_assert(iro::exec::is_warpgroup_v<ExecGroup>,
                  "UseWgmmaSmemDesc requires warpgroup exec group");
    static_assert(iro::util::HasId<SwizzleAtom>, "UseWgmmaSmemDesc: SwizzleAtom must have id");
    using out_lifetime = iro::token::lifetime::warpgroup;
    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;
    using DescPayload = WgmmaSmemDesc<SmemTile, SmemSubj, SwizzleAtom>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SmemTile,
            SmemSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<SmemSubj, scope_t>,
                iro::token::alive<SmemSubj, Lifetime>,
                iro::token::slot_state<SmemSubj, iro::token::state::ready>,
                iro::token::sync_at<SmemSubj, scope_t>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            DescPayload,
            DescSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<DescSubj, out_lifetime>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            DescPayload,
            DescSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<DescSubj, out_lifetime>,
                iro::token::visible_at<SmemSubj, scope_t>,
                iro::token::alive<SmemSubj, Lifetime>,
                iro::token::slot_state<SmemSubj, iro::token::state::ready>,
                iro::token::sync_at<SmemSubj, scope_t>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Create WGMMA SMEM descriptor from a shared-memory tile
template<class Recipe, class SmemTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime, class SwizzleAtom>
struct MakeWgmmaSmemDesc {
    static_assert(std::is_same_v<typename SmemTile::elem, typename Recipe::in>, "MakeWgmmaSmemDesc: SmemTile elem != Recipe::in");
    static_assert(std::is_same_v<typename SmemTile::space, iro::contract::space::shared>);
    static_assert(iro::exec::is_warpgroup_v<ExecGroup>,
                  "MakeWgmmaSmemDesc requires warpgroup exec group");
    static_assert(iro::util::HasId<SwizzleAtom>, "MakeWgmmaSmemDesc: SwizzleAtom must have id");
    static_assert(
        (SmemTile::align::bytes >= (1 << (SwizzleAtom::B_bits + SwizzleAtom::S_bits))),
        "MakeWgmmaSmemDesc requires smem alignment matching swizzle atom");
    using out_lifetime = std::conditional_t<
        iro::exec::is_warpgroup_v<ExecGroup>,
        iro::token::lifetime::warpgroup,
        iro::token::lifetime::warp
    >;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SmemTile,
            SmemSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<SmemSubj, iro::scope::min_scope_for_t<ExecGroup>>,
                iro::token::alive<SmemSubj, Lifetime>,
                iro::token::slot_state<SmemSubj, iro::token::state::ready>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            WgmmaSmemDesc<SmemTile, SmemSubj, SwizzleAtom>,
            DescSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<DescSubj, out_lifetime>,
                iro::token::visible_at<SmemSubj, iro::scope::min_scope_for_t<ExecGroup>>,
                iro::token::alive<SmemSubj, Lifetime>,
                iro::token::slot_state<SmemSubj, iro::token::state::ready>,
                iro::token::sync_at<SmemSubj, iro::scope::min_scope_for_t<ExecGroup>>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Create WGMMA SMEM descriptor from a shared-memory tile slice (offset by row/col).
template<class Recipe, class SmemTile, class DescTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime,
         class SwizzleAtom, int RowOffset, int ColOffset>
struct MakeWgmmaSmemDescSlice {
    static_assert(std::is_same_v<typename SmemTile::elem, typename Recipe::in>,
                  "MakeWgmmaSmemDescSlice: SmemTile elem != Recipe::in");
    static_assert(std::is_same_v<typename DescTile::elem, typename SmemTile::elem>,
                  "MakeWgmmaSmemDescSlice: DescTile elem mismatch");
    static_assert(std::is_same_v<typename SmemTile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<typename DescTile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<typename SmemTile::layout, typename DescTile::layout>,
                  "MakeWgmmaSmemDescSlice: SmemTile/DescTile layout mismatch");
    static_assert(SmemTile::align::bytes >= DescTile::align::bytes,
                  "MakeWgmmaSmemDescSlice: SmemTile alignment must subsume DescTile");
    static_assert(iro::exec::is_warpgroup_v<ExecGroup>,
                  "MakeWgmmaSmemDescSlice requires warpgroup exec group");
    static_assert(iro::util::HasId<SwizzleAtom>, "MakeWgmmaSmemDescSlice: SwizzleAtom must have id");
    static_assert(RowOffset >= 0 && ColOffset >= 0, "MakeWgmmaSmemDescSlice: offsets must be non-negative");
    static_assert(
        (SmemTile::align::bytes >= (1 << (SwizzleAtom::B_bits + SwizzleAtom::S_bits))),
        "MakeWgmmaSmemDescSlice requires smem alignment matching swizzle atom");

    static constexpr uint64_t kElemBytes = static_cast<uint64_t>(sizeof(typename SmemTile::elem::storage_t));
    static constexpr long long kOffsetElems = SmemTile::layout::offset(RowOffset, ColOffset);
    static_assert(kOffsetElems >= 0, "MakeWgmmaSmemDescSlice: layout offset must be non-negative");
    static constexpr uint64_t kOffsetBytes = static_cast<uint64_t>(kOffsetElems) * kElemBytes;
    static_assert((kOffsetBytes % 16u) == 0u, "MakeWgmmaSmemDescSlice: offset must be 16B aligned");

    static constexpr bool kHasSwizzle = !std::is_same_v<SwizzleAtom, axp::swizzle::None>;
    static constexpr uint64_t kSwizzleAlign = kHasSwizzle
        ? (1ull << (SwizzleAtom::B_bits + SwizzleAtom::S_bits))
        : 16ull;
    static constexpr uint64_t kPtrOffsetBytes = kHasSwizzle
        ? (kOffsetBytes / kSwizzleAlign) * kSwizzleAlign
        : kOffsetBytes;
    static constexpr uint32_t kBaseOffsetUnits = kHasSwizzle
        ? static_cast<uint32_t>((kOffsetBytes - kPtrOffsetBytes) / 16u)
        : 0u;
    static_assert(!kHasSwizzle || (kBaseOffsetUnits <= 7u),
                  "MakeWgmmaSmemDescSlice: base offset exceeds 3-bit field");

    using out_lifetime = std::conditional_t<
        iro::exec::is_warpgroup_v<ExecGroup>,
        iro::token::lifetime::warpgroup,
        iro::token::lifetime::warp
    >;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SmemTile,
            SmemSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<SmemSubj, iro::scope::min_scope_for_t<ExecGroup>>,
                iro::token::alive<SmemSubj, Lifetime>,
                iro::token::slot_state<SmemSubj, iro::token::state::ready>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            WgmmaSmemDesc<DescTile, SmemSubj, SwizzleAtom>,
            DescSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<DescSubj, out_lifetime>,
                iro::token::visible_at<SmemSubj, iro::scope::min_scope_for_t<ExecGroup>>,
                iro::token::alive<SmemSubj, Lifetime>,
                iro::token::slot_state<SmemSubj, iro::token::state::ready>,
                iro::token::sync_at<SmemSubj, iro::scope::min_scope_for_t<ExecGroup>>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Create WGMMA SMEM descriptor from a ready shared-memory tile (no slot_state required).
template<class Recipe, class SmemTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime, class SwizzleAtom>
struct MakeWgmmaSmemDescReady {
    static_assert(std::is_same_v<typename SmemTile::elem, typename Recipe::in>, "MakeWgmmaSmemDescReady: SmemTile elem != Recipe::in");
    static_assert(std::is_same_v<typename SmemTile::space, iro::contract::space::shared>);
    static_assert(iro::exec::is_warpgroup_v<ExecGroup>,
                  "MakeWgmmaSmemDescReady requires warpgroup exec group");
    static_assert(iro::util::HasId<SwizzleAtom>, "MakeWgmmaSmemDescReady: SwizzleAtom must have id");
    static_assert(
        (SmemTile::align::bytes >= (1 << (SwizzleAtom::B_bits + SwizzleAtom::S_bits))),
        "MakeWgmmaSmemDescReady requires smem alignment matching swizzle atom");
    using out_lifetime = iro::token::lifetime::warpgroup;
    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SmemTile,
            SmemSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<SmemSubj, scope_t>,
                iro::token::alive<SmemSubj, Lifetime>,
                iro::token::sync_at<SmemSubj, scope_t>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            WgmmaSmemDesc<SmemTile, SmemSubj, SwizzleAtom>,
            DescSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<DescSubj, out_lifetime>,
                iro::token::visible_at<SmemSubj, scope_t>,
                iro::token::alive<SmemSubj, Lifetime>,
                iro::token::slot_state<SmemSubj, iro::token::state::ready>,
                iro::token::sync_at<SmemSubj, scope_t>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Create WGMMA SMEM descriptor from a ready shared-memory tile slice (no slot_state required).
template<class Recipe, class SmemTile, class DescTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime,
         class SwizzleAtom, int RowOffset, int ColOffset>
struct MakeWgmmaSmemDescSliceReady {
    static_assert(std::is_same_v<typename SmemTile::elem, typename Recipe::in>,
                  "MakeWgmmaSmemDescSliceReady: SmemTile elem != Recipe::in");
    static_assert(std::is_same_v<typename DescTile::elem, typename SmemTile::elem>,
                  "MakeWgmmaSmemDescSliceReady: DescTile elem mismatch");
    static_assert(std::is_same_v<typename SmemTile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<typename DescTile::space, iro::contract::space::shared>);
    static_assert(std::is_same_v<typename SmemTile::layout, typename DescTile::layout>,
                  "MakeWgmmaSmemDescSliceReady: SmemTile/DescTile layout mismatch");
    static_assert(SmemTile::align::bytes >= DescTile::align::bytes,
                  "MakeWgmmaSmemDescSliceReady: SmemTile alignment must subsume DescTile");
    static_assert(iro::exec::is_warpgroup_v<ExecGroup>,
                  "MakeWgmmaSmemDescSliceReady requires warpgroup exec group");
    static_assert(iro::util::HasId<SwizzleAtom>, "MakeWgmmaSmemDescSliceReady: SwizzleAtom must have id");
    static_assert(RowOffset >= 0 && ColOffset >= 0, "MakeWgmmaSmemDescSliceReady: offsets must be non-negative");
    static_assert(
        (SmemTile::align::bytes >= (1 << (SwizzleAtom::B_bits + SwizzleAtom::S_bits))),
        "MakeWgmmaSmemDescSliceReady requires smem alignment matching swizzle atom");

    static constexpr uint64_t kElemBytes = static_cast<uint64_t>(sizeof(typename SmemTile::elem::storage_t));
    static constexpr long long kOffsetElems = SmemTile::layout::offset(RowOffset, ColOffset);
    static_assert(kOffsetElems >= 0, "MakeWgmmaSmemDescSliceReady: layout offset must be non-negative");
    static constexpr uint64_t kOffsetBytes = static_cast<uint64_t>(kOffsetElems) * kElemBytes;
    static_assert((kOffsetBytes % 16u) == 0u, "MakeWgmmaSmemDescSliceReady: offset must be 16B aligned");

    static constexpr bool kHasSwizzle = !std::is_same_v<SwizzleAtom, axp::swizzle::None>;
    static constexpr uint64_t kSwizzleAlign = kHasSwizzle
        ? (1ull << (SwizzleAtom::B_bits + SwizzleAtom::S_bits))
        : 16ull;
    static constexpr uint64_t kPtrOffsetBytes = kHasSwizzle
        ? (kOffsetBytes / kSwizzleAlign) * kSwizzleAlign
        : kOffsetBytes;
    static constexpr uint32_t kBaseOffsetUnits = kHasSwizzle
        ? static_cast<uint32_t>((kOffsetBytes - kPtrOffsetBytes) / 16u)
        : 0u;
    static_assert(!kHasSwizzle || (kBaseOffsetUnits <= 7u),
                  "MakeWgmmaSmemDescSliceReady: base offset exceeds 3-bit field");

    using out_lifetime = iro::token::lifetime::warpgroup;
    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SmemTile,
            SmemSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<SmemSubj, scope_t>,
                iro::token::alive<SmemSubj, Lifetime>,
                iro::token::sync_at<SmemSubj, scope_t>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            WgmmaSmemDesc<DescTile, SmemSubj, SwizzleAtom>,
            DescSubj,
            ExecGroup,
            iro::util::type_list<
                iro::token::alive<DescSubj, out_lifetime>,
                iro::token::visible_at<SmemSubj, scope_t>,
                iro::token::alive<SmemSubj, Lifetime>,
                iro::token::slot_state<SmemSubj, iro::token::state::ready>,
                iro::token::sync_at<SmemSubj, scope_t>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

} // namespace axp::protocol::ownership
