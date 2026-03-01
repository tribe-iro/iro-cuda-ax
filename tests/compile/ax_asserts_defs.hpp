#pragma once

#include "iro_cuda_ax_core.hpp"
#include <axp/primitives.hpp>

namespace axp_compile_test {

using RecipeF16 = iro::recipe::Precision<iro::elem::f16, iro::elem::f32, iro::elem::f16, 16, iro::recipe::Exact>;
using RecipeF32 = iro::recipe::Precision<iro::elem::f32, iro::elem::f32, iro::elem::f32, 4, iro::recipe::Exact>;
using RecipeBF16 = iro::recipe::Precision<iro::elem::bf16, iro::elem::f32, iro::elem::bf16, 16, iro::recipe::Exact>;
using RecipeF16Acc = iro::recipe::Precision<iro::elem::f16, iro::elem::f16, iro::elem::f16, 16, iro::recipe::Exact>;
using RecipeE4M3 = iro::recipe::Precision<iro::elem::e4m3, iro::elem::f32, iro::elem::e4m3, 8, iro::recipe::Exact,
                                          iro::recipe::fp8_native>;
using RecipeE5M2 = iro::recipe::Precision<iro::elem::e5m2, iro::elem::f32, iro::elem::e5m2, 8, iro::recipe::Exact,
                                          iro::recipe::fp8_native>;
using RecipeE4M3Acc = iro::recipe::Precision<iro::elem::e4m3, iro::elem::e4m3, iro::elem::e4m3, 8, iro::recipe::Exact,
                                             iro::recipe::fp8_native>;
using RecipeE5M2Acc = iro::recipe::Precision<iro::elem::e5m2, iro::elem::e5m2, iro::elem::e5m2, 8, iro::recipe::Exact,
                                             iro::recipe::fp8_native>;

using RecipeF16Fast = iro::recipe::Precision<iro::elem::f16, iro::elem::f32, iro::elem::f16, 16, iro::recipe::Fast>;
using RecipeF16Approx = iro::recipe::Precision<iro::elem::f16, iro::elem::f32, iro::elem::f16, 16, iro::recipe::ApproxExp>;

using InTileG = iro::contract::Tile<
    iro::contract::Shape<64, 64>,
    iro::elem::f16,
    iro::contract::layout::RowMajor<64>,
    iro::contract::space::global,
    iro::contract::Align<16>
>;
using GemmATileG = iro::contract::Tile<
    iro::contract::Shape<64, 16>,
    iro::elem::f16,
    iro::contract::layout::RowMajor<16>,
    iro::contract::space::global,
    iro::contract::Align<16>
>;
using GemmBTileG = iro::contract::Tile<
    iro::contract::Shape<16, 64>,
    iro::elem::f16,
    iro::contract::layout::ColMajor<16>,
    iro::contract::space::global,
    iro::contract::Align<16>
>;
using InTileFP8G = iro::contract::Tile<
    iro::contract::Shape<64, 64>,
    iro::elem::e4m3,
    iro::contract::layout::RowMajor<64>,
    iro::contract::space::global,
    iro::contract::Align<16>
>;
using OutTileS = iro::contract::Tile<
    iro::contract::Shape<64, 64>,
    iro::elem::f16,
    iro::contract::layout::RowMajor<64>,
    iro::contract::space::shared,
    iro::contract::Align<16>
>;
using OutTileG = iro::contract::Tile<
    iro::contract::Shape<64, 64>,
    iro::elem::f16,
    iro::contract::layout::RowMajor<64>,
    iro::contract::space::global,
    iro::contract::Align<16>
>;
using AtomicOutTile = iro::contract::Tile<
    iro::contract::Shape<256>,
    iro::elem::f16,
    iro::contract::layout::Contiguous,
    iro::contract::space::global,
    iro::contract::Align<2>
>;
using OutTileFP8G = iro::contract::Tile<
    iro::contract::Shape<64, 64>,
    iro::elem::e5m2,
    iro::contract::layout::RowMajor<64>,
    iro::contract::space::global,
    iro::contract::Align<16>
>;
using SmemTileF16 = iro::contract::Tile<
    iro::contract::Shape<64, 64>,
    iro::elem::f16,
    iro::contract::layout::RowMajor<64>,
    iro::contract::space::shared,
    iro::contract::Align<16>
>;
using SmemTileF16_128 = iro::contract::Tile<
    iro::contract::Shape<64, 64>,
    iro::elem::f16,
    iro::contract::layout::RowMajor<64>,
    iro::contract::space::shared,
    iro::contract::Align<128>
>;
using SmemTileF32 = iro::contract::Tile<
    iro::contract::Shape<64, 64>,
    iro::elem::f32,
    iro::contract::layout::RowMajor<64>,
    iro::contract::space::shared,
    iro::contract::Align<16>
>;
using RegTileF16 = iro::contract::Tile<
    iro::contract::Shape<64, 64>,
    iro::elem::f16,
    iro::contract::layout::RowMajor<64>,
    iro::contract::space::reg,
    iro::contract::Align<16>
>;
using TmemTileF32 = iro::contract::Tile<
    iro::contract::Shape<64, 64>,
    iro::elem::f32,
    iro::contract::layout::RowMajor<64>,
    iro::contract::space::tmem,
    iro::contract::Align<16>
>;

using ASubj = axp::subject::indexed<axp::tag::A, 0>;
using BSubj = axp::subject::indexed<axp::tag::B, 0>;
using CSubj = axp::subject::indexed<axp::tag::C, 0>;
using OSubj = axp::subject::indexed<axp::tag::O, 0>;
using WgmmaSubj = axp::subject::indexed<axp::tag::Acc, 7>;
using QSubj = axp::subject::indexed<axp::tag::Q, 0>;
using KSubj = axp::subject::indexed<axp::tag::K, 0>;
using VSubj = axp::subject::indexed<axp::tag::V, 0>;
using AttnAccSubj = axp::subject::indexed<axp::tag::Acc, 4>;
using AttnStateOldSubj = axp::subject::indexed<axp::tag::S, 0>;
using AttnStateNewSubj = axp::subject::indexed<axp::tag::S, 1>;
using GammaSubj = axp::subject::indexed<axp::tag::B, 1>;
using BetaSubj = axp::subject::indexed<axp::tag::C, 1>;
using WeightSubj = axp::subject::indexed<axp::tag::B, 2>;
using EpsSubj = axp::subject::indexed<axp::tag::S, 2>;
using HistValueSubj = axp::subject::indexed<axp::tag::A, 2>;
using HistIndexSubj = axp::subject::indexed<axp::tag::B, 3>;
using HistSharedSubj = axp::subject::indexed<axp::tag::C, 2>;
using HistOutValSubj = axp::subject::indexed<axp::tag::D, 4>;
using HistOutSubj = axp::subject::indexed<axp::tag::O, 2>;
using SortInSubj = axp::subject::indexed<axp::tag::A, 3>;
using SortOutSubj = axp::subject::indexed<axp::tag::O, 3>;
using SlotSubj = axp::subject::indexed<axp::tag::SlotA0, 0>;
using AtomicInSubj = axp::subject::indexed<axp::tag::A, 5>;
using AtomicIndexSubj = axp::subject::indexed<axp::tag::B, 5>;
using AtomicOutValSubj = axp::subject::indexed<axp::tag::C, 5>;
using AtomicOutSubj = axp::subject::indexed<axp::tag::O, 5>;
struct MapTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.test.map"); };
using MapSubj = axp::subject::indexed<MapTag, 0>;
struct GemmMapTagA { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.test.gemm.map.a"); };
struct GemmMapTagB { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.test.gemm.map.b"); };
using GemmMapSubjA = axp::subject::indexed<GemmMapTagA, 0>;
using GemmMapSubjB = axp::subject::indexed<GemmMapTagB, 0>;
using BarSubj = axp::subject::indexed<axp::tag::PipeA, 9>;
struct Coord0Tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.test.coord0"); };
struct Coord1Tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.test.coord1"); };
using Coord0Subj = axp::subject::indexed<Coord0Tag, 0>;
using Coord1Subj = axp::subject::indexed<Coord1Tag, 0>;
using ClusterBarSubj = axp::subject::indexed<axp::tag::PipeB, 7>;
using ClusterBarSubjA = axp::subject::indexed<axp::tag::PipeB, 8>;
using ClusterBarSubjB = axp::subject::indexed<axp::tag::PipeB, 9>;

using RecipeU32 = iro::recipe::Precision<iro::elem::u32, iro::elem::u32, iro::elem::u32, 4, iro::recipe::Exact>;

using FragF32 = iro::contract::FragmentDesc<
    iro::contract::Shape<16>,
    iro::elem::f32,
    iro::dist::accumulator
>;
using FragF16 = iro::contract::FragmentDesc<
    iro::contract::Shape<16>,
    iro::elem::f16,
    iro::dist::accumulator
>;
using FragF16_1 = iro::contract::FragmentDesc<
    iro::contract::Shape<1>,
    iro::elem::f16,
    iro::dist::lane
>;
using FragE4M3 = iro::contract::FragmentDesc<
    iro::contract::Shape<1>,
    iro::elem::e4m3,
    iro::dist::lane
>;
using FragE5M2 = iro::contract::FragmentDesc<
    iro::contract::Shape<1>,
    iro::elem::e5m2,
    iro::dist::lane
>;
using FragF32_4 = iro::contract::FragmentDesc<
    iro::contract::Shape<4>,
    iro::elem::f32,
    iro::dist::accumulator
>;
using FragU32 = iro::contract::FragmentDesc<
    iro::contract::Shape<1>,
    iro::elem::u32,
    iro::dist::accumulator
>;
using ScalarF32 = iro::contract::ScalarDesc<
    iro::elem::f32,
    iro::dist::lane
>;
using ScalarU32 = iro::contract::ScalarDesc<
    iro::elem::u32,
    iro::dist::replicated
>;
using CoordI32 = iro::contract::ScalarDesc<
    iro::elem::i32,
    iro::dist::uniform<iro::scope::block>
>;
using CoordI32Cluster = iro::contract::ScalarDesc<
    iro::elem::i32,
    iro::dist::uniform<iro::scope::cluster>
>;
using MaskU32Cluster = iro::contract::ScalarDesc<
    iro::elem::u32,
    iro::dist::uniform<iro::scope::cluster>
>;
using AtomicF16 = iro::contract::ScalarDesc<iro::elem::f16, iro::dist::lane>;
using AtomicIdx = iro::contract::ScalarDesc<iro::elem::u32, iro::dist::lane>;
using VectorF32 = iro::contract::VectorDesc<
    iro::elem::f32, 4,
    iro::dist::lane
>;
using MaskW32 = iro::contract::MaskDesc<
    32, iro::dist::mask<iro::scope::warp>
>;

using FSubjA = axp::subject::indexed<axp::tag::Acc, 0>;
using FSubjB = axp::subject::indexed<axp::tag::Acc, 1>;
using FSubjC = axp::subject::indexed<axp::tag::Acc, 2>;
using FSubjO = axp::subject::indexed<axp::tag::Acc, 3>;
using SSubjA = axp::subject::indexed<axp::tag::D, 0>;
using SSubjB = axp::subject::indexed<axp::tag::D, 1>;
using SSubjC = axp::subject::indexed<axp::tag::D, 3>;
using SSubjO = axp::subject::indexed<axp::tag::D, 2>;
using VSubjA = axp::subject::indexed<axp::tag::Mask, 0>;
using MaskSubj = axp::subject::indexed<axp::tag::Mask, 2>;
using VSubjB = axp::subject::indexed<axp::tag::Mask, 1>;
using VSubjO = axp::subject::indexed<axp::tag::Mask, 2>;

struct ScanAdd { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.test.scan_add"); };

} // namespace axp_compile_test
