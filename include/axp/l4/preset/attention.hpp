#pragma once

#include "common.hpp"

namespace preset {

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

} // namespace preset
