#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/l4/lowering_presets.hpp is library-only; use axp/l4.hpp in application code."
#endif

#include "../level3/registry.hpp"
#include "lowering.hpp"

namespace axp::l4::lowering::detail {

template<class Pattern>
struct map_pattern;

template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj, class WgmmaSubj,
         class MemoryPatternA, class MemoryPatternB, class LoadModeA, class LoadModeB,
         class Schedule, class ScaleASubj, class ScaleBSubj>
struct map_pattern<
    axp::l4::GemmPattern<
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
        ASubj, BSubj, CSubj, WgmmaSubj,
        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB,
        Schedule, ScaleASubj, ScaleBSubj
    >
> {
    using type = axp::level3::registry::GemmTilePattern<
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
        ASubj, BSubj, CSubj, WgmmaSubj,
        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB,
        Schedule, ScaleASubj, ScaleBSubj
    >;
};

template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class EpiloguePolicy, class ASubj, class BSubj, class CSubj, class WgmmaSubj,
         class MemoryPatternA, class MemoryPatternB, class LoadModeA, class LoadModeB,
         class Schedule, class ScaleASubj, class ScaleBSubj>
struct map_pattern<
    axp::l4::GemmFusedPattern<
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles, EpiloguePolicy,
        ASubj, BSubj, CSubj, WgmmaSubj,
        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB,
        Schedule, ScaleASubj, ScaleBSubj
    >
> {
    using type = axp::level3::registry::GemmTileFusedPattern<
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
        ASubj, BSubj, CSubj, WgmmaSubj, EpiloguePolicy,
        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB,
        Schedule, ScaleASubj, ScaleBSubj
    >;
};

template<class Recipe, int TileQ, int TileK, int TileV, int HeadDim, int Stages, int SlotIdx,
         class QSubj, class KSubj, class VSubj,
         class AccSubj, class OldStateSubj, class OutStateSubj,
         class MemoryPatternQ, class MemoryPatternK, class MemoryPatternV,
         class LoadModeQ, class LoadModeK, class LoadModeV, class Schedule, class TileSkip>
struct map_pattern<
    axp::l4::AttentionPattern<
        Recipe, TileQ, TileK, TileV, HeadDim, Stages, SlotIdx,
        QSubj, KSubj, VSubj, AccSubj, OldStateSubj, OutStateSubj,
        MemoryPatternQ, MemoryPatternK, MemoryPatternV,
        LoadModeQ, LoadModeK, LoadModeV, Schedule, TileSkip
    >
> {
    using type = axp::level3::registry::AttentionTilePattern<
        Recipe, TileQ, TileK, TileV, HeadDim, Stages, SlotIdx,
        QSubj, KSubj, VSubj,
        AccSubj, OldStateSubj, OutStateSubj,
        MemoryPatternQ, MemoryPatternK, MemoryPatternV,
        LoadModeQ, LoadModeK, LoadModeV, Schedule, TileSkip
    >;
};

template<class Recipe, int ElementsPerThread, class InSubj, class OutSubj>
struct map_pattern<axp::l4::SoftmaxRowPattern<Recipe, ElementsPerThread, InSubj, OutSubj>> {
    using type = axp::level3::registry::SoftmaxRowTilePattern<Recipe, ElementsPerThread, InSubj, OutSubj>;
};

template<class Recipe, int TileRows, int TileCols, class InSubj, class OutSubj>
struct map_pattern<axp::l4::ElementwisePattern<Recipe, TileRows, TileCols, InSubj, OutSubj>> {
    using type = axp::level3::registry::ElementwiseTilePattern<Recipe, TileRows, TileCols, InSubj, OutSubj>;
};

template<class Recipe, int TileRows, int TileCols,
         class InSubj, class OutSubj, class GammaSubj, class BetaSubj, class EpsSubj>
struct map_pattern<axp::l4::LayerNormPattern<Recipe, TileRows, TileCols, InSubj, OutSubj, GammaSubj, BetaSubj, EpsSubj>> {
    using type = axp::level3::registry::LayerNormTilePattern<
        Recipe, TileRows, TileCols, InSubj, OutSubj, GammaSubj, BetaSubj, EpsSubj
    >;
};

template<class Recipe, int TileRows, int TileCols,
         class InSubj, class OutSubj, class WeightSubj, class EpsSubj>
struct map_pattern<axp::l4::RMSNormPattern<Recipe, TileRows, TileCols, InSubj, OutSubj, WeightSubj, EpsSubj>> {
    using type = axp::level3::registry::RMSNormTilePattern<
        Recipe, TileRows, TileCols, InSubj, OutSubj, WeightSubj, EpsSubj
    >;
};

template<class Recipe, class ValuePayload, class IndexPayload, class SharedTile, class OutTile,
         class ValueSubj, class IndexSubj, class SharedSubj, class OutValSubj, class OutSubj, class ExecGroup>
struct map_pattern<
    axp::l4::HistogramPattern<
        Recipe, ValuePayload, IndexPayload, SharedTile, OutTile,
        ValueSubj, IndexSubj, SharedSubj, OutValSubj, OutSubj, ExecGroup
    >
> {
    using type = axp::level3::registry::HistogramTilePattern<
        Recipe, ValuePayload, IndexPayload, SharedTile, OutTile,
        ValueSubj, IndexSubj, SharedSubj, OutValSubj, OutSubj, ExecGroup
    >;
};

template<class Recipe, int TileElems, class InSubj, class OutSubj>
struct map_pattern<axp::l4::SortPattern<Recipe, TileElems, InSubj, OutSubj>> {
    using type = axp::level3::registry::SortTilePattern<Recipe, TileElems, InSubj, OutSubj>;
};

template<class Recipe, class ValuePayload, class IndexPayload, class StateTile,
         class ValueSubj, class IndexSubj, class StateSubj, class AtomicOutSubj,
         class DependEventTag, class PhaseTag, class DoneEventTag, class ExecGroup>
struct map_pattern<
    axp::l4::StreamingMicrobatchPattern<
        Recipe, ValuePayload, IndexPayload, StateTile,
        ValueSubj, IndexSubj, StateSubj, AtomicOutSubj,
        DependEventTag, PhaseTag, DoneEventTag, ExecGroup
    >
> {
    using type = axp::level3::registry::StreamingMicrobatchTilePattern<
        Recipe, ValuePayload, IndexPayload, StateTile,
        ValueSubj, IndexSubj, StateSubj, AtomicOutSubj,
        DependEventTag, PhaseTag, DoneEventTag, ExecGroup
    >;
};

template<class Recipe, class InTile, class OutTile, class GatherPayload, class IndexPayload,
         class InSubj, class IndexSubj, class GatherSubj, class OutSubj, class EmitEventTag,
         int SegmentWidth, class ExecGroup>
struct map_pattern<
    axp::l4::ScientificSparseSegmentedPattern<
        Recipe, InTile, OutTile, GatherPayload, IndexPayload,
        InSubj, IndexSubj, GatherSubj, OutSubj, EmitEventTag, SegmentWidth, ExecGroup
    >
> {
    using type = axp::level3::registry::ScientificSparseSegmentedTilePattern<
        Recipe, InTile, OutTile, GatherPayload, IndexPayload,
        InSubj, IndexSubj, GatherSubj, OutSubj, EmitEventTag, SegmentWidth, ExecGroup
    >;
};

template<class Recipe, class InTile, class OutTile, class TileSubj,
         class SwizzleAtom, class EmitEventTag, class ExecGroup>
struct map_pattern<
    axp::l4::ScientificSwizzlePattern<Recipe, InTile, OutTile, TileSubj, SwizzleAtom, EmitEventTag, ExecGroup>
> {
    using type = axp::level3::registry::ScientificSwizzleTilePattern<
        Recipe, InTile, OutTile, TileSubj, SwizzleAtom, EmitEventTag, ExecGroup
    >;
};

} // namespace axp::l4::lowering::detail

namespace axp::l4::lowering {

template<class Pattern>
struct is_canonical_preset_pattern : std::false_type {};

template<class Pattern>
inline constexpr bool is_canonical_preset_pattern_v = is_canonical_preset_pattern<Pattern>::value;

template<>
struct to_l3_pattern<axp::l4::preset::Gemm16x16x16> : detail::map_pattern<axp::l4::preset::Gemm16x16x16> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::Gemm16x16x16> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::Gemm64x64x16> : detail::map_pattern<axp::l4::preset::Gemm64x64x16> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::Gemm64x64x16> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::Gemm64x64x16BiasSiLU> : detail::map_pattern<axp::l4::preset::Gemm64x64x16BiasSiLU> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::Gemm64x64x16BiasSiLU> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::Attention16x16> : detail::map_pattern<axp::l4::preset::Attention16x16> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::Attention16x16> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::Attention64x64> : detail::map_pattern<axp::l4::preset::Attention64x64> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::Attention64x64> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::Attention64x64x128Decode> : detail::map_pattern<axp::l4::preset::Attention64x64x128Decode> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::Attention64x64x128Decode> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::Attention64x64x128Prefill> : detail::map_pattern<axp::l4::preset::Attention64x64x128Prefill> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::Attention64x64x128Prefill> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::SoftmaxRow4> : detail::map_pattern<axp::l4::preset::SoftmaxRow4> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::SoftmaxRow4> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::LayerNorm16x16> : detail::map_pattern<axp::l4::preset::LayerNorm16x16> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::LayerNorm16x16> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::RMSNorm16x16> : detail::map_pattern<axp::l4::preset::RMSNorm16x16> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::RMSNorm16x16> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::Histogram256> : detail::map_pattern<axp::l4::preset::Histogram256> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::Histogram256> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::Sort16> : detail::map_pattern<axp::l4::preset::Sort16> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::Sort16> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::VectorizedElementwise16x16>
    : detail::map_pattern<axp::l4::preset::VectorizedElementwise16x16> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::VectorizedElementwise16x16> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::StreamingMicrobatch1024>
    : detail::map_pattern<axp::l4::preset::StreamingMicrobatch1024> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::StreamingMicrobatch1024> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::ScientificSparseSegmented1024>
    : detail::map_pattern<axp::l4::preset::ScientificSparseSegmented1024> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::ScientificSparseSegmented1024> : std::true_type {};

template<>
struct to_l3_pattern<axp::l4::preset::ScientificSwizzle16x16>
    : detail::map_pattern<axp::l4::preset::ScientificSwizzle16x16> {};
template<>
struct is_canonical_preset_pattern<axp::l4::preset::ScientificSwizzle16x16> : std::true_type {};

} // namespace axp::l4::lowering
