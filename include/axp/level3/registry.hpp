#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../target.hpp"
#include "../swizzle.hpp"
#include "../intent.hpp"

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

// Optional fused patterns (enabled by including level3/fused.hpp)
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

template<class Recipe, int TileQ, int TileK, int TileV, int HeadDim, int Stages, int SlotIdx,
         class QSubj, class KSubj, class VSubj,
         class AccSubj, class OldStateSubj, class OutStateSubj,
         class MemoryPatternQ, class MemoryPatternK, class MemoryPatternV,
         class LoadModeQ, class LoadModeK, class LoadModeV,
         class Schedule,
         class QTma = void, class KTma = void, class VTma = void>
struct AttentionTileFusedPattern {};

template<class Recipe, int TileRows, int TileCols,
         class InSubj, class OutSubj, class GammaSubj, class BetaSubj, class EpsSubj>
struct LayerNormTileFusedPattern {};

template<class Recipe, int TileRows, int TileCols,
         class InSubj, class OutSubj, class WeightSubj, class EpsSubj>
struct RMSNormTileFusedPattern {};

// Registry plumbing
template<class Pattern, class Cap = axp::target_cap, class = void>
struct resolve_impl {
    static constexpr bool supported = false;
};

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
