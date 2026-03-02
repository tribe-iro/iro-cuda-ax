#pragma once

#include "common.hpp"

namespace preset {

using ScientificSparseSegmented1024 = axp::l4::ScientificSparseSegmentedPattern<
    axp::recipe::F32Exact,
    SciSparseTile,
    SciSparseTile,
    SciSparseGatherPayload,
    SciSparseIndexPayload,
    SciSparseInSubj,
    SciSparseIndexSubj,
    SciSparseGatherSubj,
    SciSparseOutSubj,
    SciSparseEmitTag,
    8,
    iro::exec::warp
>;

using ScientificSwizzle16x16 = axp::l4::ScientificSwizzlePattern<
    axp::recipe::F32Exact,
    SciSwizzleInTile,
    SciSwizzleOutTile,
    SciSwizzleTileSubj,
    SciSwizzleAtom128,
    SciSwizzleEmitTag,
    iro::exec::block
>;

} // namespace preset
