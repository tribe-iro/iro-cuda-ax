#pragma once

#include "prelude.hpp"

namespace axp::l4 {

// L4 intent patterns (portable descriptions only).
template<class Recipe,
         int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj = axp::subject::MatrixA,
         class BSubj = axp::subject::MatrixB,
         class CSubj = axp::subject::MatrixC,
         class WgmmaSubj = axp::subject::Accumulator,
         class MemoryPatternA = axp::intent::memory_pattern::Optimized,
         class MemoryPatternB = axp::intent::memory_pattern::Optimized,
         class LoadModeA = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeB = axp::intent::load_mode::AsyncPrefetch,
         class Schedule = axp::intent::schedule::Pipelined,
         class ScaleASubj = iro::contract::subject::global,
         class ScaleBSubj = iro::contract::subject::global>
struct GemmPattern {};

template<class Recipe,
         int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class EpiloguePolicy,
         class ASubj = axp::subject::MatrixA,
         class BSubj = axp::subject::MatrixB,
         class CSubj = axp::subject::MatrixC,
         class WgmmaSubj = axp::subject::Accumulator,
         class MemoryPatternA = axp::intent::memory_pattern::Optimized,
         class MemoryPatternB = axp::intent::memory_pattern::Optimized,
         class LoadModeA = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeB = axp::intent::load_mode::AsyncPrefetch,
         class Schedule = axp::intent::schedule::Pipelined,
         class ScaleASubj = iro::contract::subject::global,
         class ScaleBSubj = iro::contract::subject::global>
struct GemmFusedPattern {};

template<class Recipe,
         int TileQ, int TileK, int TileV, int HeadDim, int Stages, int SlotIdx,
         class QSubj = axp::subject::AttentionQ,
         class KSubj = axp::subject::AttentionK,
         class VSubj = axp::subject::AttentionV,
         class AccSubj = axp::subject::Accumulator,
         class OldStateSubj = axp::subject::AttentionS,
         class OutStateSubj = axp::subject::indexed<axp::tag::S, 1>,
         class MemoryPatternQ = axp::intent::memory_pattern::Optimized,
         class MemoryPatternK = axp::intent::memory_pattern::Optimized,
         class MemoryPatternV = axp::intent::memory_pattern::Optimized,
         class LoadModeQ = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeK = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeV = axp::intent::load_mode::AsyncPrefetch,
         class Schedule = axp::intent::schedule::Pipelined,
         class TileSkip = axp::intent::tile_skip::None>
struct AttentionPattern {};

template<class Recipe, int ElementsPerThread,
         class InSubj = axp::subject::MatrixA,
         class OutSubj = axp::subject::Output>
struct SoftmaxRowPattern {};

template<class Recipe, int TileRows, int TileCols,
         class InSubj = axp::subject::MatrixA,
         class OutSubj = axp::subject::Output>
struct ElementwisePattern {};

template<class Recipe, int TileRows, int TileCols,
         class InSubj = axp::subject::MatrixA,
         class OutSubj = axp::subject::Output,
         class GammaSubj = axp::subject::indexed<axp::tag::B, 1>,
         class BetaSubj = axp::subject::indexed<axp::tag::C, 1>,
         class EpsSubj = axp::subject::indexed<axp::tag::S, 2>>
struct LayerNormPattern {};

template<class Recipe, int TileRows, int TileCols,
         class InSubj = axp::subject::MatrixA,
         class OutSubj = axp::subject::Output,
         class WeightSubj = axp::subject::indexed<axp::tag::B, 2>,
         class EpsSubj = axp::subject::indexed<axp::tag::S, 2>>
struct RMSNormPattern {};

template<class Recipe, class ValuePayload, class IndexPayload, class SharedTile, class OutTile,
         class ValueSubj, class IndexSubj, class SharedSubj, class OutValSubj, class OutSubj,
         class ExecGroup = iro::exec::block>
struct HistogramPattern {};

template<class Recipe, int TileElems, class InSubj, class OutSubj>
struct SortPattern {};

namespace preset {

// Common subjects for presets.
using GemmA = axp::subject::MatrixA;
using GemmB = axp::subject::MatrixB;
using GemmC = axp::subject::MatrixC;
using GemmAcc = axp::subject::Accumulator;
struct GemmBiasSiLUTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.gemm.bias_silu"); };
using GemmBiasSiLU = axp::subject::indexed<GemmBiasSiLUTag, 0>;
struct GemmSiLUEpilogueTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.gemm.epilogue.silu"); };

using AttnQ = axp::subject::AttentionQ;
using AttnK = axp::subject::AttentionK;
using AttnV = axp::subject::AttentionV;
using AttnAcc = axp::subject::indexed<axp::tag::Acc, 1>;
using AttnStateOld = axp::subject::indexed<axp::tag::S, 0>;
using AttnStateNew = axp::subject::indexed<axp::tag::S, 1>;
struct AttnQDecodeTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.attention.decode.q"); };
struct AttnKDecodeTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.attention.decode.k"); };
struct AttnVDecodeTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.attention.decode.v"); };
using AttnQDecode = axp::subject::indexed<AttnQDecodeTag, 0>;
using AttnKDecode = axp::subject::indexed<AttnKDecodeTag, 0>;
using AttnVDecode = axp::subject::indexed<AttnVDecodeTag, 0>;
struct AttnQPrefillTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.attention.prefill.q"); };
struct AttnKPrefillTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.attention.prefill.k"); };
struct AttnVPrefillTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.attention.prefill.v"); };
using AttnQPrefill = axp::subject::indexed<AttnQPrefillTag, 0>;
using AttnKPrefill = axp::subject::indexed<AttnKPrefillTag, 0>;
using AttnVPrefill = axp::subject::indexed<AttnVPrefillTag, 0>;

using NormIn = axp::subject::MatrixA;
using NormOut = axp::subject::Output;
using NormGamma = axp::subject::indexed<axp::tag::B, 1>;
using NormBeta = axp::subject::indexed<axp::tag::C, 1>;
using NormWeight = axp::subject::indexed<axp::tag::B, 2>;
using NormEps = axp::subject::indexed<axp::tag::S, 2>;

struct HistValueTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.hist.value"); };
struct HistIndexTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.hist.index"); };
struct HistSharedTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.hist.shared"); };
struct HistOutValTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.hist.out_val"); };
struct HistOutTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.hist.out"); };
using HistValueSubj = axp::subject::indexed<HistValueTag, 0>;
using HistIndexSubj = axp::subject::indexed<HistIndexTag, 0>;
using HistSharedSubj = axp::subject::indexed<HistSharedTag, 0>;
using HistOutValSubj = axp::subject::indexed<HistOutValTag, 0>;
using HistOutSubj = axp::subject::indexed<HistOutTag, 0>;

struct SortInTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.sort.in"); };
struct SortOutTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.sort.out"); };
using SortInSubj = axp::subject::indexed<SortInTag, 0>;
using SortOutSubj = axp::subject::indexed<SortOutTag, 0>;
struct ElemwiseInTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.elementwise.in"); };
struct ElemwiseOutTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.elementwise.out"); };
using ElemwiseInSubj = axp::subject::indexed<ElemwiseInTag, 0>;
using ElemwiseOutSubj = axp::subject::indexed<ElemwiseOutTag, 0>;

using Gemm16x16x16 = axp::l4::GemmPattern<
    axp::recipe::F16AccF32Fast,
    16, 16, 16,
    2, 2,
    GemmA, GemmB, GemmC,
    GemmAcc,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::Optimized,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::schedule::Pipelined
>;

using Gemm64x64x16 = axp::l4::GemmPattern<
    axp::recipe::F16AccF32Fast,
    64, 64, 16,
    2, 2,
    GemmA, GemmB, GemmC,
    GemmAcc,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::Optimized,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::schedule::Pipelined
>;

using Gemm64x64x16BiasSiLU = axp::l4::GemmFusedPattern<
    axp::recipe::F16AccF32Fast,
    64, 64, 16,
    2, 2,
    GemmSiLUEpilogueTag,
    GemmA, GemmB, GemmC,
    GemmAcc,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::Optimized,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::schedule::Pipelined
>;

using Attention16x16 = axp::l4::AttentionPattern<
    axp::recipe::F16AccF32Fast,
    16, 16, 16, 16,
    2, 0,
    AttnQ, AttnK, AttnV,
    AttnAcc, AttnStateOld, AttnStateNew,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::Optimized,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::schedule::Pipelined,
    axp::intent::tile_skip::None
>;

using Attention64x64 = axp::l4::AttentionPattern<
    axp::recipe::F16AccF32Fast,
    64, 64, 64, 16,
    2, 0,
    AttnQ, AttnK, AttnV,
    AttnAcc, AttnStateOld, AttnStateNew,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::Optimized,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::schedule::Pipelined,
    axp::intent::tile_skip::None
>;

using Attention64x64x128Decode = axp::l4::AttentionPattern<
    axp::recipe::F16AccF32Fast,
    64, 64, 64, 128,
    2, 0,
    AttnQDecode, AttnKDecode, AttnVDecode,
    AttnAcc, AttnStateOld, AttnStateNew,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::Optimized,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::schedule::Pipelined,
    axp::intent::tile_skip::Causal
>;

using Attention64x64x128Prefill = axp::l4::AttentionPattern<
    axp::recipe::F16AccF32Fast,
    64, 64, 64, 128,
    2, 0,
    AttnQPrefill, AttnKPrefill, AttnVPrefill,
    AttnAcc, AttnStateOld, AttnStateNew,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::Optimized,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::schedule::Pipelined,
    axp::intent::tile_skip::Causal
>;

using SoftmaxRow4 = axp::l4::SoftmaxRowPattern<
    axp::recipe::BF16AccF32Fast,
    4,
    NormIn, NormOut
>;

using LayerNorm16x16 = axp::l4::LayerNormPattern<
    axp::recipe::F16AccF32Fast,
    16, 16,
    NormIn, NormOut,
    NormGamma, NormBeta, NormEps
>;

using RMSNorm16x16 = axp::l4::RMSNormPattern<
    axp::recipe::F16AccF32Fast,
    16, 16,
    NormIn, NormOut,
    NormWeight, NormEps
>;

using HistSharedTile = iro::contract::Tile<
    iro::contract::Shape<256>,
    iro::elem::f32,
    iro::contract::layout::Contiguous,
    iro::contract::space::shared,
    iro::contract::Align<16>
>;
using HistOutTile = iro::contract::Tile<
    iro::contract::Shape<256>,
    iro::elem::f32,
    iro::contract::layout::Contiguous,
    iro::contract::space::global,
    iro::contract::Align<16>
>;
using HistValuePayload = iro::contract::ScalarDesc<iro::elem::f32, iro::dist::lane>;
using HistIndexPayload = iro::contract::ScalarDesc<iro::elem::u32, iro::dist::lane>;
using Histogram256 = axp::l4::HistogramPattern<
    axp::recipe::F32Exact,
    HistValuePayload,
    HistIndexPayload,
    HistSharedTile,
    HistOutTile,
    HistValueSubj,
    HistIndexSubj,
    HistSharedSubj,
    HistOutValSubj,
    HistOutSubj,
    iro::exec::block
>;

using Sort16 = axp::l4::SortPattern<
    axp::recipe::F32Exact,
    16,
    SortInSubj,
    SortOutSubj
>;

using VectorizedElementwise16x16 = axp::l4::ElementwisePattern<
    axp::recipe::F32Exact,
    16, 16,
    ElemwiseInSubj,
    ElemwiseOutSubj
>;

} // namespace preset

namespace manifest {

// Explicit deterministic tie-break key per public L4 preset.
template<class Pattern>
struct tie_break_key;

template<class Pattern, class = void>
struct has_tie_break_key : std::false_type {};

template<class Pattern>
struct has_tie_break_key<Pattern, std::void_t<decltype(tie_break_key<Pattern>::value)>> : std::true_type {};

template<class Pattern>
inline constexpr bool has_tie_break_key_v = has_tie_break_key<Pattern>::value;

template<class Pattern, class Cap>
struct enabled : std::false_type {};

template<class Pattern, class Cap>
inline constexpr bool enabled_v = enabled<Pattern, Cap>::value;
// Specializations for tie_break_key and enabled<Pattern, Cap> are generated in:
// axp/l4/graph_registry_index.hpp

} // namespace manifest

} // namespace axp::l4
