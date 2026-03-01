#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../target.hpp"
#include "../swizzle.hpp"
#include "../intent.hpp"
#include "../l4.hpp"

namespace axp::level3::registry {

template<class>
inline constexpr bool always_false_v = false;

// Gemm tile patterns
template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj,
         class WgmmaSubj,
         class MemoryPatternA, class MemoryPatternB,
         class LoadModeA, class LoadModeB,
         class Schedule,
         class ScaleASubj = iro::contract::subject::global,
         class ScaleBSubj = iro::contract::subject::global,
         class ATma = void, class BTma = void>
struct GemmTilePattern {};

// Attention tile patterns
template<class Recipe, int TileQ, int TileK, int TileV, int HeadDim, int Stages, int SlotIdx,
         class QSubj, class KSubj, class VSubj,
         class AccSubj, class OldStateSubj, class OutStateSubj,
         class MemoryPatternQ, class MemoryPatternK, class MemoryPatternV,
         class LoadModeQ, class LoadModeK, class LoadModeV,
         class Schedule,
         class TileSkip = axp::intent::tile_skip::None,
         class QTma = void, class KTma = void, class VTma = void>
struct AttentionTilePattern {};

// Norm tile patterns
template<class Recipe, int TileRows, int TileCols,
         class InSubj, class OutSubj, class GammaSubj, class BetaSubj, class EpsSubj>
struct LayerNormTilePattern {};

template<class Recipe, int TileRows, int TileCols,
         class InSubj, class OutSubj, class WeightSubj, class EpsSubj>
struct RMSNormTilePattern {};

// Softmax tile patterns
template<class Recipe, int ElementsPerThread, class InSubj, class OutSubj>
struct SoftmaxRowTilePattern {};

// Elementwise tile patterns
template<class Recipe, int TileRows, int TileCols, class InSubj, class OutSubj>
struct ElementwiseTilePattern {};

// Histogram tile patterns
template<class Recipe, class ValuePayload, class IndexPayload, class SharedTile, class OutTile,
         class ValueSubj, class IndexSubj, class SharedSubj, class OutValSubj, class OutSubj, class ExecGroup>
struct HistogramTilePattern {};

// Sort/merge tile patterns
template<class Recipe, int TileElems, class InSubj, class OutSubj>
struct SortTilePattern {};

template<class Recipe, int TileElems, class InSubj, class OutSubj>
struct MergeTilePattern {};

// Warp-wide merge tile (scalar per-lane)
template<class Recipe, class InSubj, class OutSubj>
struct MergeWarpTilePattern {};

// Fused GEMM pattern.
template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj,
         class WgmmaSubj,
         class EpiloguePolicy,
         class MemoryPatternA, class MemoryPatternB,
         class LoadModeA, class LoadModeB,
         class Schedule,
         class ScaleASubj = iro::contract::subject::global,
         class ScaleBSubj = iro::contract::subject::global,
         class ATma = void, class BTma = void>
struct GemmTileFusedPattern {};

// Registry plumbing
template<class Pattern, class Cap = axp::target_cap, class = void>
struct resolve_impl {
    static constexpr bool supported = false;
};

// Canonical L4 patterns lower to L3 patterns here.
template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj, class WgmmaSubj,
         class MemoryPatternA, class MemoryPatternB, class LoadModeA, class LoadModeB,
         class Schedule, class ScaleASubj, class ScaleBSubj, class Cap>
struct resolve_impl<
    axp::l4::GemmPattern<
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
        ASubj, BSubj, CSubj, WgmmaSubj,
        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB,
        Schedule, ScaleASubj, ScaleBSubj>,
    Cap
> : resolve_impl<
    GemmTilePattern<
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
        ASubj, BSubj, CSubj, WgmmaSubj,
        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB,
        Schedule, ScaleASubj, ScaleBSubj>,
    Cap
> {};

template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class EpiloguePolicy, class ASubj, class BSubj, class CSubj, class WgmmaSubj,
         class MemoryPatternA, class MemoryPatternB, class LoadModeA, class LoadModeB,
         class Schedule, class ScaleASubj, class ScaleBSubj, class Cap>
struct resolve_impl<
    axp::l4::GemmFusedPattern<
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles, EpiloguePolicy,
        ASubj, BSubj, CSubj, WgmmaSubj,
        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB,
        Schedule, ScaleASubj, ScaleBSubj>,
    Cap
> : resolve_impl<
    GemmTileFusedPattern<
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
        ASubj, BSubj, CSubj, WgmmaSubj, EpiloguePolicy,
        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB,
        Schedule, ScaleASubj, ScaleBSubj>,
    Cap
> {};

template<class Recipe, int TileQ, int TileK, int TileV, int HeadDim, int Stages, int SlotIdx,
         class QSubj, class KSubj, class VSubj,
         class AccSubj, class OldStateSubj, class OutStateSubj,
         class MemoryPatternQ, class MemoryPatternK, class MemoryPatternV,
         class LoadModeQ, class LoadModeK, class LoadModeV,
         class Schedule, class TileSkip, class Cap>
struct resolve_impl<
    axp::l4::AttentionPattern<
        Recipe, TileQ, TileK, TileV, HeadDim, Stages, SlotIdx,
        QSubj, KSubj, VSubj, AccSubj, OldStateSubj, OutStateSubj,
        MemoryPatternQ, MemoryPatternK, MemoryPatternV,
        LoadModeQ, LoadModeK, LoadModeV, Schedule, TileSkip>,
    Cap
> : resolve_impl<
    AttentionTilePattern<
        Recipe, TileQ, TileK, TileV, HeadDim, Stages, SlotIdx,
        QSubj, KSubj, VSubj, AccSubj, OldStateSubj, OutStateSubj,
        MemoryPatternQ, MemoryPatternK, MemoryPatternV,
        LoadModeQ, LoadModeK, LoadModeV, Schedule, TileSkip>,
    Cap
> {};

template<class Recipe, int ElementsPerThread, class InSubj, class OutSubj, class Cap>
struct resolve_impl<
    axp::l4::SoftmaxRowPattern<Recipe, ElementsPerThread, InSubj, OutSubj>,
    Cap
> : resolve_impl<
    SoftmaxRowTilePattern<Recipe, ElementsPerThread, InSubj, OutSubj>,
    Cap
> {};

template<class Recipe, int TileRows, int TileCols, class InSubj, class OutSubj, class Cap>
struct resolve_impl<
    axp::l4::ElementwisePattern<Recipe, TileRows, TileCols, InSubj, OutSubj>,
    Cap
> : resolve_impl<
    ElementwiseTilePattern<Recipe, TileRows, TileCols, InSubj, OutSubj>,
    Cap
> {};

template<class Recipe, int TileRows, int TileCols,
         class InSubj, class OutSubj, class GammaSubj, class BetaSubj, class EpsSubj, class Cap>
struct resolve_impl<
    axp::l4::LayerNormPattern<Recipe, TileRows, TileCols, InSubj, OutSubj, GammaSubj, BetaSubj, EpsSubj>,
    Cap
> : resolve_impl<
    LayerNormTilePattern<Recipe, TileRows, TileCols, InSubj, OutSubj, GammaSubj, BetaSubj, EpsSubj>,
    Cap
> {};

template<class Recipe, int TileRows, int TileCols,
         class InSubj, class OutSubj, class WeightSubj, class EpsSubj, class Cap>
struct resolve_impl<
    axp::l4::RMSNormPattern<Recipe, TileRows, TileCols, InSubj, OutSubj, WeightSubj, EpsSubj>,
    Cap
> : resolve_impl<
    RMSNormTilePattern<Recipe, TileRows, TileCols, InSubj, OutSubj, WeightSubj, EpsSubj>,
    Cap
> {};

template<class Recipe, class ValuePayload, class IndexPayload, class SharedTile, class OutTile,
         class ValueSubj, class IndexSubj, class SharedSubj, class OutValSubj, class OutSubj,
         class ExecGroup, class Cap>
struct resolve_impl<
    axp::l4::HistogramPattern<
        Recipe, ValuePayload, IndexPayload, SharedTile, OutTile,
        ValueSubj, IndexSubj, SharedSubj, OutValSubj, OutSubj, ExecGroup>,
    Cap
> : resolve_impl<
    HistogramTilePattern<
        Recipe, ValuePayload, IndexPayload, SharedTile, OutTile,
        ValueSubj, IndexSubj, SharedSubj, OutValSubj, OutSubj, ExecGroup>,
    Cap
> {};

template<class Recipe, int TileElems, class InSubj, class OutSubj, class Cap>
struct resolve_impl<
    axp::l4::SortPattern<Recipe, TileElems, InSubj, OutSubj>,
    Cap
> : resolve_impl<
    SortTilePattern<Recipe, TileElems, InSubj, OutSubj>,
    Cap
> {};

template<class Pattern, class Cap = axp::target_cap>
struct resolve {
    static_assert(resolve_impl<Pattern, Cap>::supported,
                  "axp::level3::registry::resolve: unsupported pattern");
    using type = typename resolve_impl<Pattern, Cap>::type;
};

template<class Pattern, class Cap = axp::target_cap>
using Select = typename resolve<Pattern, Cap>::type;

template<class Pattern, class Cap = axp::target_cap>
struct supports : std::bool_constant<resolve_impl<Pattern, Cap>::supported> {};

} // namespace axp::level3::registry
