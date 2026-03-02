#pragma once

#include "common.hpp"

namespace preset {

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

} // namespace preset
