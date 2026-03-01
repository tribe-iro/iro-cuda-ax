#pragma once

#include <iro_cuda_ax_core.hpp>
#include "common.hpp"
#include "../detail/type_traits.hpp"
#include "../protocol/reduction/contracts.hpp"
#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace axp::realize::common::reduction {

template<class Frag, class OpTag>
__device__ __forceinline__ void warp_reduce(const typename Frag::elem::storage_t* in,
                                            typename Frag::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
    constexpr int N = static_cast<int>(Frag::count);
    const uint32_t mask = __activemask();
    if constexpr (std::is_same_v<typename Frag::elem, iro::elem::f32>) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            float v = in[i];
            if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
                v = __reduce_add_sync(mask, v);
            } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
                v = __reduce_max_sync(mask, v);
            } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
                v = __reduce_min_sync(mask, v);
            } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
                for (int offset = 16; offset > 0; offset >>= 1) {
                    const float o = __shfl_down_sync(mask, v, offset);
                    v = v * o;
                }
            } else {
                static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for WarpReduce");
            }
            out[i] = v;
        }
    } else if constexpr (std::is_same_v<typename Frag::elem, iro::elem::f16>) {
        static_assert((N % 2) == 0, "WarpReduce f16 requires even element count");
        const __half2* in2 = reinterpret_cast<const __half2*>(in);
        __half2* out2 = reinterpret_cast<__half2*>(out);
        #pragma unroll
        for (int i = 0; i < N / 2; ++i) {
            __half2 v = in2[i];
            for (int offset = 16; offset > 0; offset >>= 1) {
                const __half2 o = detail::shfl_down_vec(v, offset, mask);
                if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
                    v = __hadd2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
                    v = __hmax2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
                    v = __hmin2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
                    v = __hmul2(v, o);
                } else {
                    static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for WarpReduce");
                }
            }
            out2[i] = v;
        }
    } else if constexpr (std::is_same_v<typename Frag::elem, iro::elem::bf16>) {
        static_assert((N % 2) == 0, "WarpReduce bf16 requires even element count");
        const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(in);
        __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
        #pragma unroll
        for (int i = 0; i < N / 2; ++i) {
            __nv_bfloat162 v = in2[i];
            for (int offset = 16; offset > 0; offset >>= 1) {
                const __nv_bfloat162 o = detail::shfl_down_vec(v, offset, mask);
                if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
                    v = __hadd2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
                    v = __hmax2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
                    v = __hmin2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
                    v = __hmul2(v, o);
                } else {
                    static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for WarpReduce");
                }
            }
            out2[i] = v;
        }
    } else if constexpr (std::is_same_v<typename Frag::elem, iro::elem::i32> ||
                         std::is_same_v<typename Frag::elem, iro::elem::u32>) {
        using storage_t = typename Frag::elem::storage_t;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            storage_t v = in[i];
            for (int offset = 16; offset > 0; offset >>= 1) {
                const storage_t o = __shfl_down_sync(mask, v, offset);
                if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
                    v = v + o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
                    v = v * o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
                    v = v > o ? v : o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
                    v = v < o ? v : o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_and>) {
                    v = v & o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_or>) {
                    v = v | o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_xor>) {
                    v = v ^ o;
                } else {
                    static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for WarpReduce");
                }
            }
            out[i] = v;
        }
    } else {
        static_assert(axp::detail::always_false_v<typename Frag::elem>,
                      "WarpReduce supports f32/f16/bf16/i32/u32 only");
    }
#else
    (void)in; (void)out;
#endif
}

template<class Frag, class OpTag>
__device__ __forceinline__ void warp_all_reduce(const typename Frag::elem::storage_t* in,
                                                typename Frag::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
    constexpr int N = static_cast<int>(Frag::count);
    const uint32_t mask = __activemask();
    if constexpr (std::is_same_v<typename Frag::elem, iro::elem::f32>) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            float v = in[i];
            if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
                v = __reduce_add_sync(mask, v);
            } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
                v = __reduce_max_sync(mask, v);
            } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
                v = __reduce_min_sync(mask, v);
            } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
                for (int offset = 16; offset > 0; offset >>= 1) {
                    const float o = __shfl_down_sync(mask, v, offset);
                    v = v * o;
                }
            } else {
                static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for WarpAllReduce");
            }
            v = __shfl_sync(mask, v, 0);
            out[i] = v;
        }
    } else if constexpr (std::is_same_v<typename Frag::elem, iro::elem::f16>) {
        static_assert((N % 2) == 0, "WarpAllReduce f16 requires even element count");
        const __half2* in2 = reinterpret_cast<const __half2*>(in);
        __half2* out2 = reinterpret_cast<__half2*>(out);
        #pragma unroll
        for (int i = 0; i < N / 2; ++i) {
            __half2 v = in2[i];
            for (int offset = 16; offset > 0; offset >>= 1) {
                const __half2 o = detail::shfl_down_vec(v, offset, mask);
                if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
                    v = __hadd2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
                    v = __hmax2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
                    v = __hmin2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
                    v = __hmul2(v, o);
                } else {
                    static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for WarpAllReduce");
                }
            }
            v = detail::shfl_sync_vec(v, 0, mask);
            out2[i] = v;
        }
    } else if constexpr (std::is_same_v<typename Frag::elem, iro::elem::bf16>) {
        static_assert((N % 2) == 0, "WarpAllReduce bf16 requires even element count");
        const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(in);
        __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
        #pragma unroll
        for (int i = 0; i < N / 2; ++i) {
            __nv_bfloat162 v = in2[i];
            for (int offset = 16; offset > 0; offset >>= 1) {
                const __nv_bfloat162 o = detail::shfl_down_vec(v, offset, mask);
                if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
                    v = __hadd2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
                    v = __hmax2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
                    v = __hmin2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
                    v = __hmul2(v, o);
                } else {
                    static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for WarpAllReduce");
                }
            }
            v = detail::shfl_sync_vec(v, 0, mask);
            out2[i] = v;
        }
    } else if constexpr (std::is_same_v<typename Frag::elem, iro::elem::i32> ||
                         std::is_same_v<typename Frag::elem, iro::elem::u32>) {
        using storage_t = typename Frag::elem::storage_t;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            storage_t v = in[i];
            for (int offset = 16; offset > 0; offset >>= 1) {
                const storage_t o = __shfl_down_sync(mask, v, offset);
                if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
                    v = v + o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
                    v = v * o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
                    v = v > o ? v : o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
                    v = v < o ? v : o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_and>) {
                    v = v & o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_or>) {
                    v = v | o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_xor>) {
                    v = v ^ o;
                } else {
                    static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for WarpAllReduce");
                }
            }
            v = __shfl_sync(mask, v, 0);
            out[i] = v;
        }
    } else {
        static_assert(axp::detail::always_false_v<typename Frag::elem>,
                      "WarpAllReduce supports f32/f16/bf16/i32/u32 only");
    }
#else
    (void)in; (void)out;
#endif
}

template<class Frag, class OpTag, int SegmentWidth>
__device__ __forceinline__ void warp_segmented_reduce(const typename Frag::elem::storage_t* in,
                                                      typename Frag::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
    constexpr int N = static_cast<int>(Frag::count);
    const int lane = threadIdx.x & 31;
    const int base = lane & ~(SegmentWidth - 1);
    uint32_t seg_mask = 0xffffffffu;
    if constexpr (SegmentWidth != 32) {
        seg_mask = ((1u << SegmentWidth) - 1u) << base;
    }
    if constexpr (std::is_same_v<typename Frag::elem, iro::elem::f32>) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            float v = in[i];
            for (int offset = SegmentWidth / 2; offset > 0; offset >>= 1) {
                const float o = __shfl_down_sync(seg_mask, v, offset);
                if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
                    v = v + o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
                    v = v * o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
                    v = v > o ? v : o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
                    v = v < o ? v : o;
                } else {
                    static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for WarpSegmentedReduce");
                }
            }
            v = __shfl_sync(seg_mask, v, base);
            out[i] = v;
        }
    } else if constexpr (std::is_same_v<typename Frag::elem, iro::elem::f16>) {
        static_assert((N % 2) == 0, "WarpSegmentedReduce f16 requires even element count");
        const __half2* in2 = reinterpret_cast<const __half2*>(in);
        __half2* out2 = reinterpret_cast<__half2*>(out);
        #pragma unroll
        for (int i = 0; i < N / 2; ++i) {
            __half2 v = in2[i];
            for (int offset = SegmentWidth / 2; offset > 0; offset >>= 1) {
                const __half2 o = detail::shfl_down_vec(v, offset, seg_mask);
                if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
                    v = __hadd2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
                    v = __hmax2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
                    v = __hmin2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
                    v = __hmul2(v, o);
                } else {
                    static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for WarpSegmentedReduce");
                }
            }
            v = detail::shfl_sync_vec(v, base, seg_mask);
            out2[i] = v;
        }
    } else if constexpr (std::is_same_v<typename Frag::elem, iro::elem::bf16>) {
        static_assert((N % 2) == 0, "WarpSegmentedReduce bf16 requires even element count");
        const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(in);
        __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
        #pragma unroll
        for (int i = 0; i < N / 2; ++i) {
            __nv_bfloat162 v = in2[i];
            for (int offset = SegmentWidth / 2; offset > 0; offset >>= 1) {
                const __nv_bfloat162 o = detail::shfl_down_vec(v, offset, seg_mask);
                if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
                    v = __hadd2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
                    v = __hmax2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
                    v = __hmin2(v, o);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
                    v = __hmul2(v, o);
                } else {
                    static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for WarpSegmentedReduce");
                }
            }
            v = detail::shfl_sync_vec(v, base, seg_mask);
            out2[i] = v;
        }
    } else if constexpr (std::is_same_v<typename Frag::elem, iro::elem::i32> ||
                         std::is_same_v<typename Frag::elem, iro::elem::u32>) {
        using storage_t = typename Frag::elem::storage_t;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            storage_t v = in[i];
            for (int offset = SegmentWidth / 2; offset > 0; offset >>= 1) {
                const storage_t o = __shfl_down_sync(seg_mask, v, offset);
                if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
                    v = v + o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
                    v = v * o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
                    v = v > o ? v : o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
                    v = v < o ? v : o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_and>) {
                    v = v & o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_or>) {
                    v = v | o;
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_xor>) {
                    v = v ^ o;
                } else {
                    static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for WarpSegmentedReduce");
                }
            }
            v = __shfl_sync(seg_mask, v, base);
            out[i] = v;
        }
    } else {
        static_assert(axp::detail::always_false_v<typename Frag::elem>,
                      "WarpSegmentedReduce supports f32/f16/bf16/i32/u32 only");
    }
#else
    (void)in; (void)out;
#endif
}

} // namespace axp::realize::common::reduction
