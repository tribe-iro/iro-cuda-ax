#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/l3_presets.hpp is library-only; use axp/l4.hpp in application code."
#endif

#include <iro_cuda_ax_core.hpp>
#include "recipes.hpp"
#include "intent.hpp"
#include "swizzle.hpp"
#include "naming/subjects.hpp"
#include "level0/compute.hpp"
#include "level3/attention.hpp"
#include "level3/elementwise.hpp"
#include "level3/gemm.hpp"
#include "level3/histogram.hpp"
#include "level3/norm.hpp"
#include "level3/registry.hpp"
#include "level3/softmax.hpp"
#include "level3/sort.hpp"

namespace axp::preset {

// Common subjects for presets.
using GemmA = axp::subject::MatrixA;
using GemmB = axp::subject::MatrixB;
using GemmC = axp::subject::MatrixC;
using GemmAcc = axp::subject::Accumulator;
struct GemmBiasSiLUTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.gemm.bias_silu"); };
using GemmBiasSiLU = axp::subject::indexed<GemmBiasSiLUTag, 0>;

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

// Preset patterns (intent defaults).
using Gemm16x16x16 = axp::level3::registry::GemmTilePattern<
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

using Gemm64x64x16 = axp::level3::registry::GemmTilePattern<
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

using Gemm64x64x16BiasSiLU = axp::level3::registry::GemmTileFusedPattern<
    axp::recipe::F16AccF32Fast,
    64, 64, 16,
    2, 2,
    GemmA, GemmB, GemmC,
    GemmAcc,
    axp::level3::gemm::epilogue::BiasActivationVec<GemmBiasSiLUTag, axp::level0::SiLU>,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::Optimized,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::schedule::Pipelined
>;

using Attention16x16 = axp::level3::registry::AttentionTilePattern<
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

using Attention64x64x128Decode = axp::level3::registry::AttentionTilePattern<
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

using Attention64x64x128Prefill = axp::level3::registry::AttentionTilePattern<
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

using Attention64x64 = axp::level3::registry::AttentionTilePattern<
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

using SoftmaxRow4 = axp::level3::registry::SoftmaxRowTilePattern<
    axp::recipe::BF16AccF32Fast,
    4,
    NormIn, NormOut
>;

using LayerNorm16x16 = axp::level3::registry::LayerNormTilePattern<
    axp::recipe::F16AccF32Fast,
    16, 16,
    NormIn, NormOut,
    NormGamma, NormBeta, NormEps
>;

using RMSNorm16x16 = axp::level3::registry::RMSNormTilePattern<
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
using Histogram256 = axp::level3::registry::HistogramTilePattern<
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

using Sort16 = axp::level3::registry::SortTilePattern<
    axp::recipe::F32Exact,
    16,
    SortInSubj,
    SortOutSubj
>;

using VectorizedElementwise16x16 = axp::level3::registry::ElementwiseTilePattern<
    axp::recipe::F32Exact,
    16, 16,
    ElemwiseInSubj,
    ElemwiseOutSubj
>;

#if defined(AXP_ENABLE_SM89)
extern template struct axp::level3::registry::resolve_impl<
    Gemm16x16x16,
    iro::cap::sm89>;
extern template struct axp::level3::registry::resolve_impl<
    Attention16x16,
    iro::cap::sm89>;
extern template struct axp::level3::registry::resolve_impl<
    SoftmaxRow4,
    iro::cap::sm89>;
extern template struct axp::level3::registry::resolve_impl<
    LayerNorm16x16,
    iro::cap::sm89>;
extern template struct axp::level3::registry::resolve_impl<
    RMSNorm16x16,
    iro::cap::sm89>;
extern template struct axp::level3::registry::resolve_impl<
    Histogram256,
    iro::cap::sm89>;
extern template struct axp::level3::registry::resolve_impl<
    Sort16,
    iro::cap::sm89>;
extern template struct axp::level3::registry::resolve_impl<
    VectorizedElementwise16x16,
    iro::cap::sm89>;
#endif

#if defined(AXP_ENABLE_SM90)
extern template struct axp::level3::registry::resolve_impl<
    Gemm64x64x16,
    iro::cap::sm90>;
extern template struct axp::level3::registry::resolve_impl<
    Gemm64x64x16BiasSiLU,
    iro::cap::sm90>;
extern template struct axp::level3::registry::resolve_impl<
    Attention64x64,
    iro::cap::sm90>;
extern template struct axp::level3::registry::resolve_impl<
    Attention64x64x128Decode,
    iro::cap::sm90>;
extern template struct axp::level3::registry::resolve_impl<
    Attention64x64x128Prefill,
    iro::cap::sm90>;
extern template struct axp::level3::registry::resolve_impl<
    SoftmaxRow4,
    iro::cap::sm90>;
extern template struct axp::level3::registry::resolve_impl<
    LayerNorm16x16,
    iro::cap::sm90>;
extern template struct axp::level3::registry::resolve_impl<
    RMSNorm16x16,
    iro::cap::sm90>;
extern template struct axp::level3::registry::resolve_impl<
    Histogram256,
    iro::cap::sm90>;
extern template struct axp::level3::registry::resolve_impl<
    Sort16,
    iro::cap::sm90>;
extern template struct axp::level3::registry::resolve_impl<
    VectorizedElementwise16x16,
    iro::cap::sm90>;
#endif

} // namespace axp::preset
