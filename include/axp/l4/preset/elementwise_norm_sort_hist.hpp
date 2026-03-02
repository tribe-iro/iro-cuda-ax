#pragma once

#include "common.hpp"

namespace preset {

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
