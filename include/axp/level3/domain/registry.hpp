#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../../target.hpp"
#include "../../intent.hpp"

namespace axp::level3::domain::registry {

template<class>
inline constexpr bool always_false_v = false;

// Row/softmax patterns (fragment-level)
template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup>
struct WarpSoftmaxPattern {};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup>
struct WarpSoftmaxVecPattern {};

template<class Recipe, class Frag, class MaskPayload, class MaskSubj, class NegInfSubj,
         class InSubj, class OutSubj, class ExecGroup>
struct WarpSoftmaxMaskedPattern {};

template<class Recipe, class Frag, class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup>
struct BlockSoftmaxFragPattern {};

template<class Recipe, class Frag, class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup>
struct BlockSoftmaxFragVecPattern {};

template<class Recipe, class Frag, class MaskPayload, class MaskSubj, class NegInfSubj,
         class SmemTile, class SmemSubjMax, class SmemSubjSum,
         class InSubj, class OutSubj, class ExecGroup>
struct BlockSoftmaxFragMaskedPattern {};

// Fused block softmax patterns (tile-level, optional)
template<class Recipe, int BlockThreads, int ElementsPerThread,
         class InTile, class OutTile, class InSubj, class OutSubj>
struct BlockSoftmaxFusedPattern {};

template<class Recipe, int BlockThreads, int ElementsPerThread, int VecBytes,
         class InTile, class OutTile, class InSubj, class OutSubj>
struct BlockSoftmaxVecFusedPattern {};

// Norm patterns (fragment-level)
template<class Recipe, class Frag, class GammaFrag, class BetaFrag, class EpsPayload,
         class InSubj, class GammaSubj, class BetaSubj, class EpsSubj, class OutSubj, class ExecGroup>
struct LayerNormFragPattern {};

template<class Recipe, class Frag, class WeightFrag, class EpsPayload,
         class InSubj, class WeightSubj, class EpsSubj, class OutSubj, class ExecGroup>
struct RMSNormFragPattern {};

// Row mean/variance patterns (fragment-level, Welford-based)
template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup>
struct RowMeanPattern {};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup>
struct RowVariancePattern {};

// Fused block norm patterns (tile-level, optional)
template<class Recipe, int BlockThreads, int ElementsPerThread,
         class InTile, class OutTile, class GammaTile, class BetaTile,
         class InSubj, class OutSubj, class GammaSubj, class BetaSubj>
struct LayerNormRowFusedPattern {};

template<class Recipe, int BlockThreads, int ElementsPerThread,
         class InTile, class OutTile, class WeightTile,
         class InSubj, class OutSubj, class WeightSubj>
struct RMSNormRowFusedPattern {};

// Welford patterns
template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup>
struct FragmentWelfordPattern {};

template<class Recipe, class InSubj, class OutSubj>
struct WarpAllReduceWelfordPattern {};

// Attention softmax state patterns
template<class Recipe, class ExecGroup, class ASubj, class BSubj, class OutSubj>
struct SoftmaxStateCombinePattern {};

template<class Recipe, class Frag, class InSubj, class OutSubj, class StateSubj, class ExecGroup>
struct WarpSoftmaxStatePattern {};

template<class Recipe, class Frag, class MaskPayload, class MaskSubj, class NegInfSubj,
         class InSubj, class OutSubj, class StateSubj, class ExecGroup>
struct WarpSoftmaxStateMaskedPattern {};

template<class Recipe, class InSubj, class OutSubj>
struct WarpReduceSoftmaxStatePattern {};

template<class Recipe, class InSubj, class OutSubj, class ExecGroup>
struct SoftmaxStateCopyPattern {};

template<class Recipe, class TileStateSubj, class NewStateSubj, class OutSubj, class ExecGroup>
struct SoftmaxStateScalePattern {};

template<class Recipe, class AccFrag, class AccSubj, class OldStateSubj, class NewStateSubj, class ExecGroup>
struct RescaleAccumulatorPattern {};

template<class Recipe, class AccFrag, class AccSubj, class OldStateSubj, class NewStateSubj,
         class OutStateSubj, class ExecGroup>
struct OnlineSoftmaxUpdatePattern {};

// Warpgroup attention (WGMMA) patterns
template<class Recipe, int TileM, int TileN, int HeadDim, int Stages,
         class QSubj, class KSubj, class VSubj,
         class AccSubj, class OldStateSubj, class OutStateSubj,
         class MemoryPatternQ, class MemoryPatternK, class MemoryPatternV,
         class LoadModeQ, class LoadModeK, class LoadModeV,
         class Schedule,
         class TileSkip = axp::intent::tile_skip::None,
         class QTma = void, class KTma = void, class VTma = void>
struct AttentionWgmmaPattern {};

// Registry plumbing
template<class Pattern, class Cap = axp::target_cap, class = void>
struct resolve_impl {
    static constexpr bool supported = false;
};

template<class Pattern, class Cap = axp::target_cap>
struct resolve {
    static_assert(resolve_impl<Pattern, Cap>::supported,
                  "axp::level3::domain::registry::resolve: unsupported pattern");
    using type = typename resolve_impl<Pattern, Cap>::type;
};

template<class Pattern, class Cap = axp::target_cap>
using Select = typename resolve<Pattern, Cap>::type;

template<class Pattern, class Cap = axp::target_cap>
struct supports : std::bool_constant<resolve_impl<Pattern, Cap>::supported> {};

} // namespace axp::level3::domain::registry
