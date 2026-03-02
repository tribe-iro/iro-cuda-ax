#pragma once

namespace preset {

using GemmA = axp::subject::MatrixA;
using GemmB = axp::subject::MatrixB;
using GemmC = axp::subject::MatrixC;
using GemmAcc = axp::subject::Accumulator;
struct GemmBiasSiLUTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.gemm.bias_silu"); };
using GemmBiasSiLU = axp::subject::wire<GemmBiasSiLUTag, 0>;
struct GemmSiLUEpilogueTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.gemm.epilogue.silu"); };

using AttnQ = axp::subject::AttentionQ;
using AttnK = axp::subject::AttentionK;
using AttnV = axp::subject::AttentionV;
using AttnAcc = axp::subject::wire<axp::tag::Acc, 1>;
using AttnStateOld = axp::subject::wire<axp::tag::S, 0>;
using AttnStateNew = axp::subject::wire<axp::tag::S, 1>;
struct AttnQDecodeTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.attention.decode.q"); };
struct AttnKDecodeTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.attention.decode.k"); };
struct AttnVDecodeTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.attention.decode.v"); };
using AttnQDecode = axp::subject::wire<AttnQDecodeTag, 0>;
using AttnKDecode = axp::subject::wire<AttnKDecodeTag, 0>;
using AttnVDecode = axp::subject::wire<AttnVDecodeTag, 0>;
struct AttnQPrefillTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.attention.prefill.q"); };
struct AttnKPrefillTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.attention.prefill.k"); };
struct AttnVPrefillTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.attention.prefill.v"); };
using AttnQPrefill = axp::subject::wire<AttnQPrefillTag, 0>;
using AttnKPrefill = axp::subject::wire<AttnKPrefillTag, 0>;
using AttnVPrefill = axp::subject::wire<AttnVPrefillTag, 0>;

using NormIn = axp::subject::MatrixA;
using NormOut = axp::subject::Output;
using NormGamma = axp::subject::wire<axp::tag::B, 1>;
using NormBeta = axp::subject::wire<axp::tag::C, 1>;
using NormWeight = axp::subject::wire<axp::tag::B, 2>;
using NormEps = axp::subject::wire<axp::tag::S, 2>;

struct HistValueTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.hist.value"); };
struct HistIndexTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.hist.index"); };
struct HistSharedTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.hist.shared"); };
struct HistOutValTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.hist.out_val"); };
struct HistOutTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.hist.out"); };
using HistValueSubj = axp::subject::wire<HistValueTag, 0>;
using HistIndexSubj = axp::subject::wire<HistIndexTag, 0>;
using HistSharedSubj = axp::subject::wire<HistSharedTag, 0>;
using HistOutValSubj = axp::subject::wire<HistOutValTag, 0>;
using HistOutSubj = axp::subject::wire<HistOutTag, 0>;

struct SortInTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.sort.in"); };
struct SortOutTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.sort.out"); };
using SortInSubj = axp::subject::wire<SortInTag, 0>;
using SortOutSubj = axp::subject::wire<SortOutTag, 0>;

struct ElemwiseInTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.elementwise.in"); };
struct ElemwiseOutTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.elementwise.out"); };
using ElemwiseInSubj = axp::subject::wire<ElemwiseInTag, 0>;
using ElemwiseOutSubj = axp::subject::wire<ElemwiseOutTag, 0>;

struct StreamValueTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.streaming.value"); };
struct StreamIndexTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.streaming.index"); };
struct StreamStateTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.streaming.state"); };
struct StreamAtomicOutTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.streaming.atomic_out"); };
struct StreamDependEventTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.streaming.depend_event"); };
struct StreamPhaseProcessTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.streaming.phase_process"); };
struct StreamDoneEventTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.streaming.done_event"); };
using StreamValueSubj = axp::subject::wire<StreamValueTag, 0>;
using StreamIndexSubj = axp::subject::wire<StreamIndexTag, 0>;
using StreamStateSubj = axp::subject::wire<StreamStateTag, 0>;
using StreamAtomicOutSubj = axp::subject::wire<StreamAtomicOutTag, 0>;
using StreamValuePayload = iro::contract::ScalarDesc<iro::elem::f32, iro::dist::lane>;
using StreamIndexPayload = iro::contract::ScalarDesc<iro::elem::u32, iro::dist::lane>;
using StreamStateTile = iro::contract::Tile<
    iro::contract::Shape<1024>,
    iro::elem::f32,
    iro::contract::layout::Contiguous,
    iro::contract::space::global,
    iro::contract::Align<16>
>;

struct SciSparseInTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.scientific.sparse.in"); };
struct SciSparseIndexTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.scientific.sparse.index"); };
struct SciSparseGatherTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.scientific.sparse.gather"); };
struct SciSparseOutTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.scientific.sparse.out"); };
struct SciSparseEmitTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.scientific.sparse.emit"); };
using SciSparseInSubj = axp::subject::wire<SciSparseInTag, 0>;
using SciSparseIndexSubj = axp::subject::wire<SciSparseIndexTag, 0>;
using SciSparseGatherSubj = axp::subject::wire<SciSparseGatherTag, 0>;
using SciSparseOutSubj = axp::subject::wire<SciSparseOutTag, 0>;
using SciSparseTile = iro::contract::Tile<
    iro::contract::Shape<1024>,
    iro::elem::f32,
    iro::contract::layout::Contiguous,
    iro::contract::space::global,
    iro::contract::Align<16>
>;
using SciSparseGatherPayload = iro::contract::ScalarDesc<iro::elem::f32, iro::dist::lane>;
using SciSparseIndexPayload = iro::contract::ScalarDesc<iro::elem::u32, iro::dist::lane>;

struct SciSwizzleTileTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.scientific.swizzle.tile"); };
struct SciSwizzleEmitTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.preset.scientific.swizzle.emit"); };
using SciSwizzleTileSubj = axp::subject::wire<SciSwizzleTileTag, 0>;
struct SciSwizzleAtom128 {
    static constexpr int M = 3;
    static constexpr int B = 4;
    static constexpr int S = 3;
};
using SciSwizzleInTile = iro::contract::Tile<
    iro::contract::Shape<16, 16>,
    iro::elem::f32,
    iro::contract::layout::RowMajor<16>,
    iro::contract::space::shared,
    iro::contract::Align<16>
>;
using SciSwizzleOutTile = iro::contract::Tile<
    iro::contract::Shape<16, 16>,
    iro::elem::f32,
    iro::contract::layout::Swizzled<16, SciSwizzleAtom128::B, SciSwizzleAtom128::S>,
    iro::contract::space::shared,
    iro::contract::Align<16>
>;

} // namespace preset
