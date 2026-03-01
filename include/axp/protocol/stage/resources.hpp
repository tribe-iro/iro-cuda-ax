#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../../naming/tags.hpp"

namespace axp::protocol::stage {
// ============================================================================
// Canonical SMEM pipeline shapes (SOTA-aligned, minimal set)
// ============================================================================

template<class Tag>
using Pipe2_128x64_F16 = iro::contract::res::smem_pipeline<
    Tag,
    /*Slots=*/2,
    /*BytesPerSlot=*/128LL * 64LL * 2LL,
    /*AlignBytes=*/128
>;

template<class Tag>
using Pipe2_128x128_F16 = iro::contract::res::smem_pipeline<
    Tag,
    /*Slots=*/2,
    /*BytesPerSlot=*/128LL * 128LL * 2LL,
    /*AlignBytes=*/128
>;

template<class Tag>
using Pipe2_64x64_F16 = iro::contract::res::smem_pipeline<
    Tag,
    /*Slots=*/2,
    /*BytesPerSlot=*/64LL * 64LL * 2LL,
    /*AlignBytes=*/128
>;

template<class Tag>
using Pipe3_128x64_F16 = iro::contract::res::smem_pipeline<
    Tag,
    /*Slots=*/3,
    /*BytesPerSlot=*/128LL * 64LL * 2LL,
    /*AlignBytes=*/128
>;

template<class Tag>
using PipeQ_64x128_F16 = iro::contract::res::smem_pipeline<
    Tag,
    /*Slots=*/2,
    /*BytesPerSlot=*/64LL * 128LL * 2LL,
    /*AlignBytes=*/128
>;

template<class Tag>
using PipeKV_64x128_F16 = iro::contract::res::smem_pipeline<
    Tag,
    /*Slots=*/2,
    /*BytesPerSlot=*/64LL * 128LL * 2LL,
    /*AlignBytes=*/128
>;

template<class Tag>
using Pipe2_64x32_F16_SM89 = iro::contract::res::smem_pipeline<
    Tag,
    /*Slots=*/2,
    /*BytesPerSlot=*/64LL * 32LL * 2LL,
    /*AlignBytes=*/16
>;

// ============================================================================
// Swizzle atoms (layout metadata for smem)
// M = mask bits, B = base bits, S = shift bits
// ============================================================================

template<int MaskBits, int BaseBits, int ShiftBits>
struct SwizzleAtom {
    static_assert(((MaskBits == 0) && (BaseBits == 0) && (ShiftBits == 0)) ||
                  (MaskBits > 0 && BaseBits >= 0 && ShiftBits >= 0),
                  "SwizzleAtom: invalid parameters");
    static constexpr int M_bits = MaskBits;
    static constexpr int B_bits = BaseBits;
    static constexpr int S_bits = ShiftBits;
    static constexpr int M = M_bits;
    static constexpr int B = B_bits;
    static constexpr int S = S_bits;
    static constexpr auto id = iro::util::mix_u64(
        iro::util::fnv1a_64_cstr("axp.swizzle.atom"),
        iro::util::mix_u64((iro::util::u64)MaskBits,
            iro::util::mix_u64((iro::util::u64)BaseBits, (iro::util::u64)ShiftBits)));
    static constexpr __host__ __device__ int apply(int idx) {
        return idx ^ (((idx >> BaseBits) & ((1 << MaskBits) - 1)) << ShiftBits);
    }
};

using SwizzleAtom_128B = SwizzleAtom<3, 4, 3>;
using SwizzleAtom_64B  = SwizzleAtom<2, 3, 3>;
using SwizzleAtom_32B  = SwizzleAtom<1, 2, 3>;
using SwizzleAtom_None = SwizzleAtom<0, 0, 0>;
}
