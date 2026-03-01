#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/realize/sm89.hpp is internal substrate; use axp/l4.hpp in application code."
#endif

#include <iro_cuda_ax_core.hpp>
#include <cstdint>
#include <type_traits>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#endif
#include "../detail/type_traits.hpp"
#include "common.hpp"
#include "common_reduction.hpp"
#include "../protocol/stage/pipeline_contracts.hpp"
#include "../protocol/compute/contracts.hpp"
#include "../protocol/reduction/contracts.hpp"
#include "../protocol/mask/contracts.hpp"
#include "../protocol/sync/contracts.hpp"
#include "../protocol/view/contracts.hpp"
#include "../protocol/ownership/contracts.hpp"
#include "../protocol/ownership/bundles.hpp"
#include "../protocol/convert/contracts.hpp"
#include "../detail/conversion.hpp"
#include "../level0/memory_cache.hpp"

namespace axp::realize::sm89 {

namespace detail {
using axp::detail::is_f32;
using axp::detail::is_f16;
using axp::detail::is_bf16;
using axp::detail::is_fp8_e4m3_like;
using axp::detail::is_fp8_e5m2_like;
using axp::detail::is_supported_elem_v;
using axp::detail::to_f32_dispatch;
using axp::detail::from_f32_dispatch;
using axp::detail::from_f32_recipe;
using axp::detail::from_f32_half;
using axp::detail::always_false_v;
using namespace axp::realize::common::detail;

#ifdef __CUDACC__
__device__ __forceinline__ bool is_aligned_16(const void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) & 0xFu) == 0u;
}

__device__ __forceinline__ void copy_16_bytes(char* dst, const char* src) {
    if (is_aligned_16(dst) && is_aligned_16(src)) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
        return;
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        dst[i] = src[i];
    }
}

__device__ __forceinline__ void cp_async_cg_16(char* smem_dst, const char* gmem_src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned smem_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_src));
#else
    (void)smem_dst;
    (void)gmem_src;
#endif
}

template<class PipeT>
__device__ __forceinline__ void stage_copy_gmem_to_smem_16b(
    PipeT* pipe, char* smem_b, const char* gmem_b, std::uint32_t bytes) {
    for (std::uint32_t off = static_cast<std::uint32_t>(threadIdx.x) * 16u;
         off < bytes;
         off += static_cast<std::uint32_t>(blockDim.x) * 16u) {
        char* dst = smem_b + off;
        const char* src = gmem_b + off;
        if (is_aligned_16(dst) && is_aligned_16(src)) {
            cp_async_cg_16(dst, src);
        } else {
            cuda::memcpy_async(*pipe, dst, src, 16u);
        }
    }
}
#endif
} // namespace detail

// Stage realizations (SM89 - cp.async pipeline)

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class SwizzleAtom>
struct CpAsyncIssue
    : iro::contract::Realization<
        axp::protocol::stage::CpAsyncIssue<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.stage.cp_async.issue")> {
    __device__ static void execute(const typename InTile::elem::storage_t* gmem,
                                   typename OutTile::elem::storage_t* smem,
                                   uint64_t base,
                                   cuda::pipeline<cuda::thread_scope_block>* pipe) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM89 CpAsyncIssue requires block exec group");
        static_assert(std::is_void_v<SwizzleAtom>,
                      "SM89 CpAsyncIssue does not support swizzled layouts; use explicit SwizzledStShared");
        constexpr uint32_t bytes = static_cast<uint32_t>(OutTile::bytes);
        static_assert(bytes % 16u == 0u, "SM89 CpAsyncIssue requires 16B granularity");
        char* smem_b = reinterpret_cast<char*>(smem);
        const char* gmem_b = reinterpret_cast<const char*>(gmem + base);
        detail::stage_copy_gmem_to_smem_16b(pipe, smem_b, gmem_b, bytes);
#else
        (void)gmem; (void)smem; (void)base; (void)pipe;
#endif
    }
};

template<class Recipe, class OutTile, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots>
struct CpAsyncCommit
    : iro::contract::Realization<
        axp::protocol::stage::CpAsyncCommit<Recipe, OutTile, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.stage.cp_async.commit")> {
    __device__ static void execute(cuda::pipeline<cuda::thread_scope_block>* pipe) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM89 CpAsyncCommit requires block exec group");
        pipe->commit();
#else
        (void)pipe;
#endif
    }
};

template<class Recipe, class OutTile, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, int Prior>
struct CpAsyncWait
    : iro::contract::Realization<
        axp::protocol::stage::CpAsyncWait<Recipe, OutTile, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, Prior>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.stage.cp_async.wait")> {
    __device__ static void execute(cuda::pipeline<cuda::thread_scope_block>* pipe) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM89 CpAsyncWait requires block exec group");
        pipe->wait_prior<Prior>();
#else
        (void)pipe;
#endif
    }
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class SwizzleAtom>
struct IssueGmemToSmemSlot
    : iro::contract::Realization<
        axp::protocol::stage::IssueGmemToSmemSlot<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.stage.issue")> {
    __device__ static void execute(const typename InTile::elem::storage_t* gmem,
                                   typename OutTile::elem::storage_t* smem,
                                   uint64_t base,
                                   cuda::pipeline<cuda::thread_scope_block>* pipe) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM89 issue requires block exec group");
        static_assert(std::is_void_v<SwizzleAtom>,
                      "SM89 IssueGmemToSmemSlot does not support swizzled layouts; use explicit SwizzledStShared");
        constexpr uint32_t bytes = static_cast<uint32_t>(OutTile::bytes);
        static_assert(bytes % 16u == 0u, "SM89 issue requires 16B granularity");
        char* smem_b = reinterpret_cast<char*>(smem);
        const char* gmem_b = reinterpret_cast<const char*>(gmem + base);
        detail::stage_copy_gmem_to_smem_16b(pipe, smem_b, gmem_b, bytes);
        pipe->commit();
#else
        (void)gmem; (void)smem; (void)base; (void)pipe;
#endif
    }
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class SwizzleAtom>
struct DirectGmemToSmemSlot
    : iro::contract::Realization<
        axp::protocol::stage::DirectGmemToSmemSlot<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.stage.direct")> {
    __device__ static void execute(const typename InTile::elem::storage_t* gmem,
                                   typename OutTile::elem::storage_t* smem,
                                   uint64_t base,
                                   cuda::pipeline<cuda::thread_scope_block>* pipe) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM89 direct requires block exec group");
        static_assert(std::is_void_v<SwizzleAtom>,
                      "SM89 DirectGmemToSmemSlot does not support swizzled layouts; use TMA");
        constexpr uint32_t bytes = static_cast<uint32_t>(OutTile::bytes);
        static_assert(bytes % 16u == 0u, "SM89 direct requires 16B granularity");
        char* smem_b = reinterpret_cast<char*>(smem);
        const char* gmem_b = reinterpret_cast<const char*>(gmem + base);
        for (uint32_t off = static_cast<uint32_t>(threadIdx.x) * 16u; off < bytes; off += static_cast<uint32_t>(blockDim.x) * 16u) {
            detail::copy_16_bytes(smem_b + off, gmem_b + off);
        }
        __syncthreads();
#else
        (void)gmem; (void)smem; (void)base; (void)pipe;
#endif
    }
};

template<class Recipe, class OutTile, class SlotSubj, class ExecGroup, class Lifetime>
struct WaitSmemSlot
    : iro::contract::Realization<
        axp::protocol::stage::WaitSmemSlot<Recipe, OutTile, SlotSubj, ExecGroup, Lifetime>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.stage.wait")> {
    __device__ static void execute(cuda::pipeline<cuda::thread_scope_block>* pipe) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM89 wait requires block exec group");
        pipe->wait_prior<0>();
#else
        (void)pipe;
#endif
    }
};

template<class Recipe, class OutTile, class SlotSubj, class ExecGroup, class Lifetime>
struct ReadySmemSlot
    : iro::contract::Realization<
        axp::protocol::stage::ReadySmemSlot<Recipe, OutTile, SlotSubj, ExecGroup, Lifetime>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.stage.ready")> {
    __device__ static void execute() {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM89 ready requires block exec group");
#endif
    }
};

template<class Recipe, class OutTile, class SlotSubj, class BarrierSubj, class ExecGroup, class Lifetime>
struct CommitSmemSlot
    : iro::contract::Realization<
        axp::protocol::stage::CommitSmemSlot<Recipe, OutTile, SlotSubj, BarrierSubj, ExecGroup, Lifetime>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.stage.commit")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_in) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM89 commit requires block exec group");
        bar->wait(*token_in);
#else
        (void)bar; (void)token_in;
#endif
    }
};

template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime>
struct ReleaseSmemSlot
    : iro::contract::Realization<
        axp::protocol::stage::ReleaseSmemSlot<Recipe, SlotSubj, ExecGroup, Lifetime>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.stage.release")> {
    __device__ static void execute() {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM89 release requires block exec group");
#endif
    }
};

template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime, long long Bytes>
struct MarkConsumed
    : iro::contract::Realization<
        axp::protocol::stage::MarkConsumed<Recipe, SlotSubj, ExecGroup, Lifetime, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.stage.mark_consumed")> {
    __device__ static void execute() {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::warp> ||
                      iro::exec::is_warpgroup_v<ExecGroup> ||
                      std::is_same_v<ExecGroup, iro::exec::block>,
                      "SM89 mark-consumed requires warp/warpgroup/block");
#endif
    }
};

template<class Recipe, class InTile, class OutTile, class SlotSubj, class OutSubj,
         class ExecGroup, class Lifetime>
struct StoreSmemToGmemSlot
    : iro::contract::Realization<
        axp::protocol::stage::StoreSmemToGmemSlot<Recipe, InTile, OutTile, SlotSubj, OutSubj, ExecGroup, Lifetime>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.stage.store_smem_to_gmem")> {
    __device__ static void execute(const typename InTile::elem::storage_t* smem,
                                   typename OutTile::elem::storage_t* gmem) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM89 store requires block exec group");
        constexpr uint32_t bytes = static_cast<uint32_t>(OutTile::bytes);
        static_assert(bytes % 16u == 0u, "SM89 store requires 16B granularity");
        const char* smem_b = reinterpret_cast<const char*>(smem);
        char* gmem_b = reinterpret_cast<char*>(gmem);
        for (uint32_t off = static_cast<uint32_t>(threadIdx.x) * 16u; off < bytes; off += static_cast<uint32_t>(blockDim.x) * 16u) {
            detail::copy_16_bytes(gmem_b + off, smem_b + off);
        }
#else
        (void)smem; (void)gmem;
#endif
    }
};

template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime, long long Bytes>
struct PassSlot
    : iro::contract::Realization<
        axp::protocol::stage::PassSlot<Recipe, SlotSubj, ExecGroup, Lifetime, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.stage.pass_slot")> {
    __device__ static void execute() {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM89 pass-slot requires block exec group");
#endif
    }
};

template<class Recipe, class SlotSubj, class SlotExecGroup, class Lifetime, long long Bytes,
         class DepPayload, class DepSubj, class DepExecGroup,
         class DepDist, class DepTokens>
struct SlotAfter
    : iro::contract::Realization<
        axp::protocol::stage::SlotAfter<
            Recipe, SlotSubj, SlotExecGroup, Lifetime, Bytes,
            DepPayload, DepSubj, DepExecGroup, DepDist, DepTokens>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.stage.slot_after")> {
    using storage_t = typename DepPayload::elem::storage_t;
    __device__ static void execute(const storage_t* /*dep*/) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<SlotExecGroup, iro::exec::warp> ||
                      iro::exec::is_warpgroup_v<SlotExecGroup> ||
                      std::is_same_v<SlotExecGroup, iro::exec::block>,
                      "SM89 slot-after requires warp/warpgroup/block exec group");
#endif
    }
};

// Compute realizations (SM89 - warp MMA)

template<class Recipe, class Shape, class ATile, class BTile, class AccFrag, class ASubj, class BSubj, class AccSubj>
struct WarpMmaFromSmem
    : iro::contract::Realization<
        axp::protocol::compute::WarpMmaFromSmem<Recipe, Shape, ATile, BTile, AccFrag, ASubj, BSubj, AccSubj>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.compute.warp_mma")> {
    __device__ static void execute(const typename ATile::elem::storage_t* a,
                                   const typename BTile::elem::storage_t* b,
                                   float* acc,
                                   int lda, int ldb) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<typename ATile::space, iro::contract::space::shared>);
        static_assert(std::is_same_v<typename BTile::space, iro::contract::space::shared>);
        static_assert(std::is_same_v<typename AccFrag::elem, iro::elem::f32>);
        static_assert(ATile::shape::rank == 2 && BTile::shape::rank == 2);
        static_assert(std::is_same_v<typename ATile::layout, iro::contract::layout::RowMajor<ATile::shape::template dim<1>()>>,
                      "SM89 WarpMmaFromSmem requires row-major A");
        static_assert(std::is_same_v<typename BTile::layout, iro::contract::layout::ColMajor<BTile::shape::template dim<0>()>>,
                      "SM89 WarpMmaFromSmem requires col-major B");

        using namespace nvcuda::wmma;
        if constexpr (Shape::m == 16 && Shape::n == 16 && Shape::k == 16 &&
                      std::is_same_v<typename ATile::elem, iro::elem::f16> &&
                      std::is_same_v<typename BTile::elem, iro::elem::f16>) {
            fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
            fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
            fragment<accumulator, 16, 16, 16, float> c_frag;
            static_assert(AccFrag::count == decltype(c_frag)::num_elements,
                          "WarpMmaFromSmem: AccFrag count mismatch (f16)");
            fill_fragment(c_frag, 0.0f);
            load_matrix_sync(a_frag, reinterpret_cast<const __half*>(a), lda);
            load_matrix_sync(b_frag, reinterpret_cast<const __half*>(b), ldb);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
            #pragma unroll
            for (int i = 0; i < c_frag.num_elements; ++i) {
                acc[i] = c_frag.x[i];
            }
        } else if constexpr (Shape::m == 16 && Shape::n == 16 && Shape::k == 16 &&
                             std::is_same_v<typename ATile::elem, iro::elem::bf16> &&
                             std::is_same_v<typename BTile::elem, iro::elem::bf16>) {
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
            fragment<accumulator, 16, 16, 16, float> c_frag;
            static_assert(AccFrag::count == decltype(c_frag)::num_elements,
                          "WarpMmaFromSmem: AccFrag count mismatch (bf16)");
            fill_fragment(c_frag, 0.0f);
            load_matrix_sync(a_frag, reinterpret_cast<const __nv_bfloat16*>(a), lda);
            load_matrix_sync(b_frag, reinterpret_cast<const __nv_bfloat16*>(b), ldb);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
            #pragma unroll
            for (int i = 0; i < c_frag.num_elements; ++i) {
                acc[i] = c_frag.x[i];
            }
        } else if constexpr (Shape::m == 16 && Shape::n == 16 && Shape::k == 8 &&
                             std::is_same_v<typename ATile::elem, iro::elem::tf32> &&
                             std::is_same_v<typename BTile::elem, iro::elem::tf32>) {
            fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
            fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> b_frag;
            fragment<accumulator, 16, 16, 8, float> c_frag;
            static_assert(AccFrag::count == decltype(c_frag)::num_elements,
                          "WarpMmaFromSmem: AccFrag count mismatch (tf32)");
            fill_fragment(c_frag, 0.0f);
            load_matrix_sync(a_frag, reinterpret_cast<const float*>(a), lda);
            load_matrix_sync(b_frag, reinterpret_cast<const float*>(b), ldb);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
            #pragma unroll
            for (int i = 0; i < c_frag.num_elements; ++i) {
                acc[i] = c_frag.x[i];
            }
        } else {
            static_assert(axp::detail::always_false_v<Shape>,
                          "SM89 WarpMmaFromSmem: unsupported shape/elem");
        }
#else
        (void)a; (void)b; (void)acc; (void)lda; (void)ldb;
#endif
    }
};

template<class Recipe, class Shape, class ATile, class BTile, class AccFrag, class ASubj, class BSubj, class AccSubj>
struct WarpMmaFromShared
    : iro::contract::Realization<
        axp::protocol::compute::WarpMmaFromShared<Recipe, Shape, ATile, BTile, AccFrag, ASubj, BSubj, AccSubj>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.compute.warp_mma_shared")> {
    __device__ static void execute(const typename ATile::elem::storage_t* a,
                                   const typename BTile::elem::storage_t* b,
                                   float* acc,
                                   int lda, int ldb) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<typename ATile::space, iro::contract::space::shared>);
        static_assert(std::is_same_v<typename BTile::space, iro::contract::space::shared>);
        static_assert(std::is_same_v<typename AccFrag::elem, iro::elem::f32>);
        static_assert(ATile::shape::rank == 2 && BTile::shape::rank == 2);
        static_assert(std::is_same_v<typename ATile::layout, iro::contract::layout::RowMajor<ATile::shape::template dim<1>()>>,
                      "SM89 WarpMmaFromShared requires row-major A");
        static_assert(std::is_same_v<typename BTile::layout, iro::contract::layout::ColMajor<BTile::shape::template dim<0>()>>,
                      "SM89 WarpMmaFromShared requires col-major B");

        using namespace nvcuda::wmma;
        if constexpr (Shape::m == 16 && Shape::n == 16 && Shape::k == 16 &&
                      std::is_same_v<typename ATile::elem, iro::elem::f16> &&
                      std::is_same_v<typename BTile::elem, iro::elem::f16>) {
            fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
            fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
            fragment<accumulator, 16, 16, 16, float> c_frag;
            static_assert(AccFrag::count == decltype(c_frag)::num_elements,
                          "WarpMmaFromShared: AccFrag count mismatch (f16)");
            fill_fragment(c_frag, 0.0f);
            load_matrix_sync(a_frag, reinterpret_cast<const __half*>(a), lda);
            load_matrix_sync(b_frag, reinterpret_cast<const __half*>(b), ldb);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
            #pragma unroll
            for (int i = 0; i < c_frag.num_elements; ++i) {
                acc[i] = c_frag.x[i];
            }
        } else if constexpr (Shape::m == 16 && Shape::n == 16 && Shape::k == 16 &&
                             std::is_same_v<typename ATile::elem, iro::elem::bf16> &&
                             std::is_same_v<typename BTile::elem, iro::elem::bf16>) {
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
            fragment<accumulator, 16, 16, 16, float> c_frag;
            static_assert(AccFrag::count == decltype(c_frag)::num_elements,
                          "WarpMmaFromShared: AccFrag count mismatch (bf16)");
            fill_fragment(c_frag, 0.0f);
            load_matrix_sync(a_frag, reinterpret_cast<const __nv_bfloat16*>(a), lda);
            load_matrix_sync(b_frag, reinterpret_cast<const __nv_bfloat16*>(b), ldb);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
            #pragma unroll
            for (int i = 0; i < c_frag.num_elements; ++i) {
                acc[i] = c_frag.x[i];
            }
        } else if constexpr (Shape::m == 16 && Shape::n == 16 && Shape::k == 8 &&
                             std::is_same_v<typename ATile::elem, iro::elem::tf32> &&
                             std::is_same_v<typename BTile::elem, iro::elem::tf32>) {
            fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
            fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> b_frag;
            fragment<accumulator, 16, 16, 8, float> c_frag;
            static_assert(AccFrag::count == decltype(c_frag)::num_elements,
                          "WarpMmaFromShared: AccFrag count mismatch (tf32)");
            fill_fragment(c_frag, 0.0f);
            load_matrix_sync(a_frag, reinterpret_cast<const float*>(a), lda);
            load_matrix_sync(b_frag, reinterpret_cast<const float*>(b), ldb);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
            #pragma unroll
            for (int i = 0; i < c_frag.num_elements; ++i) {
                acc[i] = c_frag.x[i];
            }
        } else {
            static_assert(axp::detail::always_false_v<Shape>,
                          "SM89 WarpMmaFromShared: unsupported shape/elem");
        }
#else
        (void)a; (void)b; (void)acc; (void)lda; (void)ldb;
#endif
    }
};

// Reduction realization (SM89)

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup>
struct BlockReduce
    : iro::contract::Realization<
        axp::protocol::reduction::BlockReduce<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.reduction.block")> {
    __device__ static void execute(const float* input, float* output) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<typename InTile::elem, iro::elem::f32>);
        static_assert(std::is_same_v<typename OutTile::elem, iro::elem::f32>);
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;
        float val = input[idx];

        // Warp-level reduce
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        // Block-level reduce
        extern __shared__ float smem[];
        int warp = tid >> 5;
        int lane = tid & 31;
        if (lane == 0) {
            smem[warp] = val;
        }
        __syncthreads();

        if (warp == 0) {
            float sum = smem[lane];
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane == 0) {
                output[blockIdx.x] = sum;
            }
        }
#else
        (void)input; (void)output;
#endif
    }
};

// Warp reduction (SM89 fast path using warp reduce intrinsics)
template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag>
struct WarpReduce
    : iro::contract::Realization<
        axp::protocol::reduction::WarpReduce<Recipe, Frag, Subj, ExecGroup, OpTag>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.reduction.warp")> {
    __device__ static void execute(const typename Frag::elem::storage_t* in,
                                   typename Frag::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::warp>, "WarpReduce requires warp exec group");
        axp::realize::common::reduction::warp_reduce<Frag, OpTag>(in, out);
#else
        (void)in; (void)out;
#endif
    }
};

// Warp all-reduce (SM89)
template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag>
struct WarpAllReduce
    : iro::contract::Realization<
        axp::protocol::reduction::WarpAllReduce<Recipe, Frag, Subj, ExecGroup, OpTag>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.reduction.warp_all")> {
    __device__ static void execute(const typename Frag::elem::storage_t* in,
                                   typename Frag::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::warp>, "WarpAllReduce requires warp exec group");
        axp::realize::common::reduction::warp_all_reduce<Frag, OpTag>(in, out);
#else
        (void)in; (void)out;
#endif
    }
};

// Warp segmented all-reduce (SM89)
template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag, int SegmentWidth>
struct WarpSegmentedReduce
    : iro::contract::Realization<
        axp::protocol::reduction::WarpSegmentedReduce<Recipe, Frag, Subj, ExecGroup, OpTag, SegmentWidth>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.reduction.warp_segmented")> {
    __device__ static void execute(const typename Frag::elem::storage_t* in,
                                   typename Frag::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::warp>, "WarpSegmentedReduce requires warp exec group");
        axp::realize::common::reduction::warp_segmented_reduce<Frag, OpTag, SegmentWidth>(in, out);
#else
        (void)in; (void)out;
#endif
    }
};

// Mask realizations

template<class Recipe, class MaskFragT, class MaskSubj, class ExecGroup>
struct MaskGen
    : iro::contract::Realization<
        axp::protocol::mask::MaskGen<Recipe, MaskFragT, MaskSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.mask.gen")> {
    __device__ static void execute(uint32_t* out_mask, int n, int offset) {
#ifdef __CUDA_ARCH__
        int lane = threadIdx.x & 31;
        bool pred = (lane + offset) < n;
        uint32_t mask = __ballot_sync(__activemask(), pred);
        if (lane == 0) {
            out_mask[0] = mask;
        }
#else
        (void)out_mask; (void)n; (void)offset;
#endif
    }
};

// L0 memory realizations (SM89, vectorized global/shared)

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CachePolicy, class InDist, class OutDist, class InExtra, class OutExtra>
struct LdGlobal
    : iro::contract::Realization<
        axp::level0::LdGlobal<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, CachePolicy, InDist, OutDist, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.l0.ld_global")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in,
                                   typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
        detail::tile_copy_global<Recipe, InTile, OutTile, ExecGroup, CachePolicy>(in, out);
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CachePolicy, class InDist, class OutDist, class InExtra, class OutExtra>
struct StGlobal
    : iro::contract::Realization<
        axp::level0::StGlobal<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, CachePolicy, InDist, OutDist, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.l0.st_global")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in,
                                   typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
        detail::tile_copy_store_global<Recipe, InTile, OutTile, ExecGroup, CachePolicy>(in, out);
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class InTile, class InSubj, class ExecGroup,
         class CachePolicy, class InDist, class InExtra>
struct PrefetchGlobal
    : iro::contract::Realization<
        axp::level0::PrefetchGlobal<Recipe, InTile, InSubj, ExecGroup, CachePolicy, InDist, InExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.l0.prefetch_global")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in) {
#ifdef __CUDA_ARCH__
        detail::tile_prefetch_global<Recipe, InTile, ExecGroup, CachePolicy>(in);
#else
        (void)in;
#endif
    }
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class InDist, class OutDist, class InExtra, class OutExtra>
struct LdShared
    : iro::contract::Realization<
        axp::level0::LdShared<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, InDist, OutDist, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.l0.ld_shared")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in,
                                   typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
        detail::tile_copy_shared<Recipe, InTile, OutTile, ExecGroup>(in, out);
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class InDist, class OutDist, class InExtra, class OutExtra>
struct StShared
    : iro::contract::Realization<
        axp::level0::StShared<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, InDist, OutDist, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.l0.st_shared")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in,
                                   typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
        detail::tile_store_shared<Recipe, InTile, OutTile, ExecGroup>(in, out);
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup, class SwizzleAtom,
         class InDist, class OutDist, class InExtra, class OutExtra>
struct SwizzledLdShared
    : iro::contract::Realization<
        axp::level0::SwizzledLdShared<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, SwizzleAtom, InDist, OutDist, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.l0.swizzled_ld_shared")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in,
                                   typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
        detail::tile_copy_swizzled_to_row<Recipe, InTile, OutTile, ExecGroup, SwizzleAtom>(in, out);
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup, class SwizzleAtom,
         class InDist, class OutDist, class InExtra, class OutExtra>
struct SwizzledStShared
    : iro::contract::Realization<
        axp::level0::SwizzledStShared<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, SwizzleAtom, InDist, OutDist, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.l0.swizzled_st_shared")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in,
                                   typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
        detail::tile_copy_row_to_swizzled<Recipe, InTile, OutTile, ExecGroup, SwizzleAtom>(in, out);
#else
        (void)in; (void)out;
#endif
    }
};

// Convert realizations (SM89)

template<class RecipeIn, class RecipeOut, class InTile, class OutTile,
         class InSubj, class OutSubj, class ExecGroup, int VecBytes,
         class InDist, class OutDist>
struct CastTile
    : iro::contract::Realization<
        axp::protocol::convert::CastTile<RecipeIn, RecipeOut, InTile, OutTile, InSubj, OutSubj, ExecGroup, VecBytes, InDist, OutDist>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.convert.tile")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in,
                                   typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
        using InT = typename InTile::elem::storage_t;
        using OutT = typename OutTile::elem::storage_t;
        using OutElem = typename OutTile::elem;
        static_assert(detail::is_supported_elem_v<InT>, "CastTile: unsupported input element type");
        static_assert(detail::is_supported_elem_v<OutT>, "CastTile: unsupported output element type");
        static_assert(VecBytes % sizeof(InT) == 0, "CastTile: VecBytes must divide input element size");
        constexpr int kVecElems = VecBytes / static_cast<int>(sizeof(InT));
        static_assert(kVecElems > 0, "CastTile: invalid vector element count");
        static_assert(InTile::shape::size % kVecElems == 0, "CastTile: tile size must be vector aligned");

        int tid = 0;
        int stride = 0;
        if constexpr (std::is_same_v<ExecGroup, iro::exec::warp>) {
            tid = threadIdx.x & 31;
            stride = 32;
        } else if constexpr (iro::exec::is_warpgroup_v<ExecGroup>) {
            constexpr int warps = iro::exec::warpgroup_warps<ExecGroup>::value;
            constexpr int lanes = warps * 32;
            tid = threadIdx.x % lanes;
            stride = lanes;
        } else {
            static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "CastTile supports warp/warpgroup or block exec group");
            tid = threadIdx.x;
            stride = blockDim.x;
        }

        for (int idx = tid * kVecElems; idx < InTile::shape::size; idx += stride * kVecElems) {
            InT tmp_in[kVecElems];
            // Vectorized loads
            if constexpr (detail::is_fp8_e4m3_like<InT>::value || detail::is_fp8_e5m2_like<InT>::value) {
                if constexpr (kVecElems == 4) {
                    union { uint32_t u; InT v[4]; } u{};
                    u.u = reinterpret_cast<const uint32_t*>(in + idx)[0];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) tmp_in[j] = u.v[j];
                } else if constexpr (kVecElems == 8) {
                    union { uint64_t u; InT v[8]; } u{};
                    u.u = reinterpret_cast<const uint64_t*>(in + idx)[0];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) tmp_in[j] = u.v[j];
                } else {
                    union { uint4 u; InT v[16]; } u{};
                    u.u = reinterpret_cast<const uint4*>(in + idx)[0];
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) tmp_in[j] = u.v[j];
                }
            } else if constexpr (detail::is_f32<InT>::value) {
                if constexpr (kVecElems == 1) {
                    tmp_in[0] = in[idx];
                } else if constexpr (kVecElems == 2) {
                    float2 v = reinterpret_cast<const float2*>(in + idx)[0];
                    tmp_in[0] = v.x; tmp_in[1] = v.y;
                } else {
                    float4 v = reinterpret_cast<const float4*>(in + idx)[0];
                    tmp_in[0] = v.x; tmp_in[1] = v.y; tmp_in[2] = v.z; tmp_in[3] = v.w;
                }
            } else if constexpr (detail::is_f16<InT>::value) {
                static_assert((kVecElems % 2) == 0, "CastTile: half input requires even elements");
                const __half2* p = reinterpret_cast<const __half2*>(in + idx);
                #pragma unroll
                for (int j = 0; j < kVecElems; j += 2) {
                    __half2 v = p[j / 2];
                    tmp_in[j] = __low2half(v);
                    tmp_in[j + 1] = __high2half(v);
                }
            } else {
                static_assert((kVecElems % 2) == 0, "CastTile: bf16 input requires even elements");
                const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(in + idx);
                #pragma unroll
                for (int j = 0; j < kVecElems; j += 2) {
                    __nv_bfloat162 v = p[j / 2];
                    tmp_in[j] = __low2bfloat16(v);
                    tmp_in[j + 1] = __high2bfloat16(v);
                }
            }

            // Convert and store
            if constexpr (detail::is_fp8_e4m3_like<OutT>::value || detail::is_fp8_e5m2_like<OutT>::value) {
                if constexpr (kVecElems == 4) {
                    union { uint32_t u; OutT v[4]; } u{};
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        u.v[j] = detail::from_f32_recipe<OutElem, RecipeOut>(detail::to_f32_dispatch(tmp_in[j]));
                    }
                    reinterpret_cast<uint32_t*>(out + idx)[0] = u.u;
                } else if constexpr (kVecElems == 8) {
                    union { uint64_t u; OutT v[8]; } u{};
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        u.v[j] = detail::from_f32_recipe<OutElem, RecipeOut>(detail::to_f32_dispatch(tmp_in[j]));
                    }
                    reinterpret_cast<uint64_t*>(out + idx)[0] = u.u;
                } else {
                    union { uint4 u; OutT v[16]; } u{};
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        u.v[j] = detail::from_f32_recipe<OutElem, RecipeOut>(detail::to_f32_dispatch(tmp_in[j]));
                    }
                    reinterpret_cast<uint4*>(out + idx)[0] = u.u;
                }
            } else if constexpr (detail::is_f32<OutT>::value) {
                if constexpr (kVecElems == 1) {
                    out[idx] = detail::from_f32_dispatch<OutT>(detail::to_f32_dispatch(tmp_in[0]));
                } else if constexpr (kVecElems == 2) {
                    float2 v;
                    v.x = detail::to_f32_dispatch(tmp_in[0]);
                    v.y = detail::to_f32_dispatch(tmp_in[1]);
                    reinterpret_cast<float2*>(out + idx)[0] = v;
                } else {
                    float4 v;
                    v.x = detail::to_f32_dispatch(tmp_in[0]);
                    v.y = detail::to_f32_dispatch(tmp_in[1]);
                    v.z = detail::to_f32_dispatch(tmp_in[2]);
                    v.w = detail::to_f32_dispatch(tmp_in[3]);
                    reinterpret_cast<float4*>(out + idx)[0] = v;
                }
            } else if constexpr (detail::is_f16<OutT>::value) {
                static_assert((kVecElems % 2) == 0, "CastTile: half output requires even elements");
                __half2* p = reinterpret_cast<__half2*>(out + idx);
                #pragma unroll
                for (int j = 0; j < kVecElems; j += 2) {
                    __half a = detail::from_f32_dispatch<__half>(detail::to_f32_dispatch(tmp_in[j]));
                    __half b = detail::from_f32_dispatch<__half>(detail::to_f32_dispatch(tmp_in[j + 1]));
                    p[j / 2] = __halves2half2(a, b);
                }
            } else {
                static_assert((kVecElems % 2) == 0, "CastTile: bf16 output requires even elements");
                __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(out + idx);
                #pragma unroll
                for (int j = 0; j < kVecElems; j += 2) {
                    __nv_bfloat16 a = detail::from_f32_dispatch<__nv_bfloat16>(detail::to_f32_dispatch(tmp_in[j]));
                    __nv_bfloat16 b = detail::from_f32_dispatch<__nv_bfloat16>(detail::to_f32_dispatch(tmp_in[j + 1]));
                    p[j / 2] = __halves2bfloat162(a, b);
                }
            }
        }
#else
        (void)in; (void)out;
#endif
    }
};

template<class RecipeIn, class RecipeOut, class InFrag, class OutFrag,
         class InSubj, class OutSubj, class ExecGroup>
struct CastFragment
    : iro::contract::Realization<
        axp::protocol::convert::CastFragment<RecipeIn, RecipeOut, InFrag, OutFrag, InSubj, OutSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.convert.frag")> {
    __device__ static void execute(const typename InFrag::elem::storage_t* in,
                                   typename OutFrag::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
        using InT = typename InFrag::elem::storage_t;
        using OutT = typename OutFrag::elem::storage_t;
        using OutElem = typename OutFrag::elem;
        static_assert(detail::is_supported_elem_v<InT>, "CastFragment: unsupported input element type");
        static_assert(detail::is_supported_elem_v<OutT>, "CastFragment: unsupported output element type");
        #pragma unroll
        for (int i = 0; i < InFrag::count; ++i) {
            out[i] = detail::from_f32_recipe<OutElem, RecipeOut>(detail::to_f32_dispatch(in[i]));
        }
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class Frag, class Vec, class FragSubj, class VecSubj, class ExecGroup, int Offset>
struct FragmentToVectorSlice
    : iro::contract::Realization<
        axp::protocol::convert::FragmentToVectorSlice<Recipe, Frag, Vec, FragSubj, VecSubj, ExecGroup, Offset>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.convert.frag_to_vec_slice")> {
    __device__ static void execute(const typename Frag::elem::storage_t* frag,
                                   typename Vec::elem::storage_t* vec) {
#ifdef __CUDA_ARCH__
        #pragma unroll
        for (int i = 0; i < static_cast<int>(Vec::lanes); ++i) {
            vec[i] = frag[Offset + i];
        }
#else
        (void)frag; (void)vec;
#endif
    }
};

template<class Recipe, class Frag, class Vec, class FragInSubj, class VecSubj, class FragOutSubj,
         class ExecGroup, int Offset>
struct VectorSliceToFragment
    : iro::contract::Realization<
        axp::protocol::convert::VectorSliceToFragment<
            Recipe, Frag, Vec, FragInSubj, VecSubj, FragOutSubj, ExecGroup, Offset>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.convert.vec_slice_to_frag")> {
    __device__ static void execute(const typename Frag::elem::storage_t* frag_in,
                                   const typename Vec::elem::storage_t* vec,
                                   typename Frag::elem::storage_t* frag_out) {
#ifdef __CUDA_ARCH__
        #pragma unroll
        for (int i = 0; i < static_cast<int>(Frag::count); ++i) {
            frag_out[i] = frag_in[i];
        }
        #pragma unroll
        for (int i = 0; i < static_cast<int>(Vec::lanes); ++i) {
            frag_out[Offset + i] = vec[i];
        }
#else
        (void)frag_in; (void)vec; (void)frag_out;
#endif
    }
};

// Sync realization (SM89)

template<class Recipe, class Subject, class ExecGroup, int Expected>
struct BarrierInit
    : iro::contract::Realization<
        axp::protocol::sync::BarrierInit<Recipe, Subject, ExecGroup, Expected>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.sync.barrier_init")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierInit requires block exec group");
        static_assert(Expected > 0, "BarrierInit requires positive expected count");
        if (threadIdx.x == 0) {
            init(bar, Expected);
        }
        __syncthreads();
#else
        (void)bar;
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup>
struct SyncPoint
    : iro::contract::Realization<
        axp::protocol::sync::SyncPoint<Recipe, Subject, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.sync.point")> {
    __device__ static void execute() {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                      "SM89 SyncPoint supports warp only; block uses barriers");
        __syncwarp();
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup>
struct SyncWarp
    : iro::contract::Realization<
        axp::protocol::sync::SyncWarp<Recipe, Subject, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.sync.warp")> {
    __device__ static void execute() {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::warp>, "SyncWarp requires warp exec group");
        __syncwarp();
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup>
struct SyncThreads
    : iro::contract::Realization<
        axp::protocol::sync::SyncThreads<Recipe, Subject, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.sync.threads")> {
    __device__ static void execute() {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SyncThreads requires block exec group");
        __syncthreads();
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup, class ScopeT, class OrderT>
struct Fence
    : iro::contract::Realization<
        axp::protocol::sync::Fence<Recipe, Subject, ExecGroup, ScopeT, OrderT>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.sync.fence")> {
    __device__ static void execute() {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ScopeT, iro::scope::block> ||
                      std::is_same_v<ScopeT, iro::scope::cluster> ||
                      std::is_same_v<ScopeT, iro::scope::device>,
                      "Fence supports block/cluster/device scope");
        if constexpr (std::is_same_v<ScopeT, iro::scope::block>) {
            __threadfence_block();
        } else {
            __threadfence();
        }
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup>
struct BarrierArrive
    : iro::contract::Realization<
        axp::protocol::sync::BarrierArrive<Recipe, Subject, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.sync.barrier_arrive")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_out) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierArrive v1 supports block exec only");
        *token_out = bar->arrive();
#else
        (void)bar; (void)token_out;
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup, long long Bytes>
struct BarrierExpectTx
    : iro::contract::Realization<
        axp::protocol::sync::BarrierExpectTx<Recipe, Subject, ExecGroup, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.sync.barrier_expect_tx")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar) {
#ifdef __CUDA_ARCH__
        static_assert(detail::always_false_v<Subject>, "BarrierExpectTx requires SM90 (mbarrier)");
#else
        (void)bar;
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup, long long Bytes>
struct BarrierArriveTx
    : iro::contract::Realization<
        axp::protocol::sync::BarrierArriveTx<Recipe, Subject, ExecGroup, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.sync.barrier_arrive_tx")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_out) {
#ifdef __CUDA_ARCH__
        static_assert(detail::always_false_v<Subject>, "BarrierArriveTx requires SM90 (mbarrier)");
#else
        (void)bar; (void)token_out;
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup>
struct BarrierWait
    : iro::contract::Realization<
        axp::protocol::sync::BarrierWait<Recipe, Subject, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.sync.barrier_wait")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_in) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierWait v1 supports block exec only");
        bar->wait(*token_in);
#else
        (void)bar; (void)token_in;
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup, class FlagPayload, class FlagSubj, class OutExtra>
struct BarrierTryWait
    : iro::contract::Realization<
        axp::protocol::sync::BarrierTryWait<Recipe, Subject, ExecGroup, FlagPayload, FlagSubj, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.sync.barrier_try_wait")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_in,
                                   typename FlagPayload::elem::storage_t* out_flag) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierTryWait requires block exec group");
        const bool done = bar->try_wait(*token_in);
        out_flag[0] = done ? 1u : 0u;
#else
        (void)bar; (void)token_in; (void)out_flag;
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup, int Expected>
struct BarrierInvalidate
    : iro::contract::Realization<
        axp::protocol::sync::BarrierInvalidate<Recipe, Subject, ExecGroup, Expected>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.sync.barrier_invalidate")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierInvalidate requires block exec group");
        if (threadIdx.x == 0) {
            init(bar, Expected);
        }
        __syncthreads();
#else
        (void)bar;
#endif
    }
};

// Ownership realizations (generic, non-MMA fragments)

template<class Recipe, class InTile, class Frag, class InSubj, class FragSubj, class ExecGroup, class Lifetime,
         class InExtra, class OutExtra>
struct SharedTileToFragment
    : iro::contract::Realization<
        axp::protocol::ownership::SharedTileToFragment<Recipe, InTile, Frag, InSubj, FragSubj, ExecGroup, Lifetime, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.ownership.shared_tile_to_frag")> {
    __device__ static void execute(const typename InTile::elem::storage_t* smem,
                                   typename Frag::elem::storage_t* frag,
                                   int ld) {
#ifdef __CUDA_ARCH__
        static_assert(iro::schema::ElemHasStorage<typename InTile::elem>);
        static_assert(iro::schema::ElemHasStorage<typename Frag::elem>);
        static_assert(InTile::shape::rank == 2, "SharedTileToFragment requires rank-2 tile");
        // Optimized WMMA path for 16x16 f16 tiles
        if constexpr (std::is_same_v<ExecGroup, iro::exec::warp> &&
                      std::is_same_v<typename InTile::elem, iro::elem::f16> &&
                      std::is_same_v<typename Frag::elem, iro::elem::f16> &&
                      (InTile::shape::template dim<0>() == 16) &&
                      (InTile::shape::template dim<1>() == 16)) {
            if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_row_major> &&
                          std::is_same_v<typename InTile::layout, iro::contract::layout::RowMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_a, 16, 16, 16, __half, row_major> f{};
                load_matrix_sync(f, reinterpret_cast<const __half*>(smem), ld);
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    frag[i] = f.x[i];
                }
                return;
            } else if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_col_major> &&
                                 std::is_same_v<typename InTile::layout, iro::contract::layout::ColMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_b, 16, 16, 16, __half, col_major> f{};
                load_matrix_sync(f, reinterpret_cast<const __half*>(smem), ld);
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    frag[i] = f.x[i];
                }
                return;
            }
        } else if constexpr (std::is_same_v<ExecGroup, iro::exec::warp> &&
                             std::is_same_v<typename InTile::elem, iro::elem::bf16> &&
                             std::is_same_v<typename Frag::elem, iro::elem::bf16> &&
                             (InTile::shape::template dim<0>() == 16) &&
                             (InTile::shape::template dim<1>() == 16)) {
            if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_row_major> &&
                          std::is_same_v<typename InTile::layout, iro::contract::layout::RowMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> f{};
                load_matrix_sync(f, reinterpret_cast<const __nv_bfloat16*>(smem), ld);
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    frag[i] = f.x[i];
                }
                return;
            } else if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_col_major> &&
                                 std::is_same_v<typename InTile::layout, iro::contract::layout::ColMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> f{};
                load_matrix_sync(f, reinterpret_cast<const __nv_bfloat16*>(smem), ld);
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    frag[i] = f.x[i];
                }
                return;
            }
        } else if constexpr (std::is_same_v<ExecGroup, iro::exec::warp> &&
                             std::is_same_v<typename InTile::elem, iro::elem::tf32> &&
                             std::is_same_v<typename Frag::elem, iro::elem::tf32> &&
                             (InTile::shape::template dim<0>() == 16) &&
                             (InTile::shape::template dim<1>() == 16)) {
            if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_row_major> &&
                          std::is_same_v<typename InTile::layout, iro::contract::layout::RowMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> f{};
                load_matrix_sync(f, reinterpret_cast<const float*>(smem), ld);
                static_assert(Frag::count == f.num_elements, "TF32 fragment count mismatch");
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    frag[i] = f.x[i];
                }
                return;
            } else if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_col_major> &&
                                 std::is_same_v<typename InTile::layout, iro::contract::layout::ColMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> f{};
                load_matrix_sync(f, reinterpret_cast<const float*>(smem), ld);
                static_assert(Frag::count == f.num_elements, "TF32 fragment count mismatch");
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    frag[i] = f.x[i];
                }
                return;
            }
        }
        static_assert(detail::always_false_v<InTile>,
                      "SM89 SharedTileToFragment supports WMMA 16x16 f16/bf16/tf32 (warp, row/col major).");
#else
        (void)smem; (void)frag; (void)ld;
#endif
    }
};

template<class Recipe, class InTile, class Frag, class InSubj, class FragSubj, class ExecGroup, class Lifetime>
struct TileToFragment
    : iro::contract::Realization<
        axp::protocol::ownership::TileToFragment<Recipe, InTile, Frag, InSubj, FragSubj, ExecGroup, Lifetime>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.ownership.tile_to_frag")> {
    __device__ static void execute(const typename InTile::elem::storage_t* smem,
                                   typename Frag::elem::storage_t* frag,
                                   int ld) {
#ifdef __CUDA_ARCH__
        static_assert(iro::schema::ElemHasStorage<typename InTile::elem>);
        static_assert(iro::schema::ElemHasStorage<typename Frag::elem>);
        static_assert(InTile::shape::rank == 2, "TileToFragment requires rank-2 tile");
        // Optimized WMMA path for 16x16 f16 tiles
        if constexpr (std::is_same_v<ExecGroup, iro::exec::warp> &&
                      std::is_same_v<typename InTile::elem, iro::elem::f16> &&
                      std::is_same_v<typename Frag::elem, iro::elem::f16> &&
                      (InTile::shape::template dim<0>() == 16) &&
                      (InTile::shape::template dim<1>() == 16)) {
            if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_row_major> &&
                          std::is_same_v<typename InTile::layout, iro::contract::layout::RowMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_a, 16, 16, 16, __half, row_major> f{};
                load_matrix_sync(f, reinterpret_cast<const __half*>(smem), ld);
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    frag[i] = f.x[i];
                }
                return;
            } else if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_col_major> &&
                                 std::is_same_v<typename InTile::layout, iro::contract::layout::ColMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_b, 16, 16, 16, __half, col_major> f{};
                load_matrix_sync(f, reinterpret_cast<const __half*>(smem), ld);
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    frag[i] = f.x[i];
                }
                return;
            }
        } else if constexpr (std::is_same_v<ExecGroup, iro::exec::warp> &&
                             std::is_same_v<typename InTile::elem, iro::elem::bf16> &&
                             std::is_same_v<typename Frag::elem, iro::elem::bf16> &&
                             (InTile::shape::template dim<0>() == 16) &&
                             (InTile::shape::template dim<1>() == 16)) {
            if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_row_major> &&
                          std::is_same_v<typename InTile::layout, iro::contract::layout::RowMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> f{};
                load_matrix_sync(f, reinterpret_cast<const __nv_bfloat16*>(smem), ld);
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    frag[i] = f.x[i];
                }
                return;
            } else if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_col_major> &&
                                 std::is_same_v<typename InTile::layout, iro::contract::layout::ColMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> f{};
                load_matrix_sync(f, reinterpret_cast<const __nv_bfloat16*>(smem), ld);
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    frag[i] = f.x[i];
                }
                return;
            }
        } else if constexpr (std::is_same_v<ExecGroup, iro::exec::warp> &&
                             std::is_same_v<typename InTile::elem, iro::elem::tf32> &&
                             std::is_same_v<typename Frag::elem, iro::elem::tf32> &&
                             (InTile::shape::template dim<0>() == 16) &&
                             (InTile::shape::template dim<1>() == 16)) {
            if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_row_major> &&
                          std::is_same_v<typename InTile::layout, iro::contract::layout::RowMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> f{};
                load_matrix_sync(f, reinterpret_cast<const float*>(smem), ld);
                static_assert(Frag::count == f.num_elements, "TF32 fragment count mismatch");
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    frag[i] = f.x[i];
                }
                return;
            } else if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_col_major> &&
                                 std::is_same_v<typename InTile::layout, iro::contract::layout::ColMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> f{};
                load_matrix_sync(f, reinterpret_cast<const float*>(smem), ld);
                static_assert(Frag::count == f.num_elements, "TF32 fragment count mismatch");
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    frag[i] = f.x[i];
                }
                return;
            }
        }
        static_assert(detail::always_false_v<InTile>,
                      "SM89 TileToFragment supports WMMA 16x16 f16/bf16/tf32 (warp, row/col major).");
#else
        (void)smem; (void)frag; (void)ld;
#endif
    }
};

template<class Recipe, class Frag, class OutTile, class FragSubj, class OutSubj, class ExecGroup, class Lifetime,
         class InExtra, class OutExtra>
struct FragmentToSharedTile
    : iro::contract::Realization<
        axp::protocol::ownership::FragmentToSharedTile<Recipe, Frag, OutTile, FragSubj, OutSubj, ExecGroup, Lifetime, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.ownership.frag_to_shared")> {
    __device__ static void execute(const typename Frag::elem::storage_t* frag,
                                   typename OutTile::elem::storage_t* smem,
                                   int ld) {
#ifdef __CUDA_ARCH__
        static_assert(iro::schema::ElemHasStorage<typename Frag::elem>);
        static_assert(iro::schema::ElemHasStorage<typename OutTile::elem>);
        static_assert(OutTile::shape::rank == 2, "FragmentToSharedTile requires rank-2 tile");
        // Optimized WMMA path for 16x16 f16 tiles
        if constexpr (std::is_same_v<ExecGroup, iro::exec::warp> &&
                      std::is_same_v<typename Frag::elem, iro::elem::f16> &&
                      std::is_same_v<typename OutTile::elem, iro::elem::f16> &&
                      (OutTile::shape::template dim<0>() == 16) &&
                      (OutTile::shape::template dim<1>() == 16)) {
            if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_row_major> &&
                          std::is_same_v<typename OutTile::layout, iro::contract::layout::RowMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_a, 16, 16, 16, __half, row_major> f{};
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    f.x[i] = frag[i];
                }
                store_matrix_sync(reinterpret_cast<__half*>(smem), f, ld, mem_row_major);
                return;
            } else if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_col_major> &&
                                 std::is_same_v<typename OutTile::layout, iro::contract::layout::ColMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_b, 16, 16, 16, __half, col_major> f{};
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    f.x[i] = frag[i];
                }
                store_matrix_sync(reinterpret_cast<__half*>(smem), f, ld, mem_col_major);
                return;
            }
        } else if constexpr (std::is_same_v<ExecGroup, iro::exec::warp> &&
                             std::is_same_v<typename Frag::elem, iro::elem::bf16> &&
                             std::is_same_v<typename OutTile::elem, iro::elem::bf16> &&
                             (OutTile::shape::template dim<0>() == 16) &&
                             (OutTile::shape::template dim<1>() == 16)) {
            if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_row_major> &&
                          std::is_same_v<typename OutTile::layout, iro::contract::layout::RowMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> f{};
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    f.x[i] = frag[i];
                }
                store_matrix_sync(reinterpret_cast<__nv_bfloat16*>(smem), f, ld, mem_row_major);
                return;
            } else if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_col_major> &&
                                 std::is_same_v<typename OutTile::layout, iro::contract::layout::ColMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> f{};
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    f.x[i] = frag[i];
                }
                store_matrix_sync(reinterpret_cast<__nv_bfloat16*>(smem), f, ld, mem_col_major);
                return;
            }
        } else if constexpr (std::is_same_v<ExecGroup, iro::exec::warp> &&
                             std::is_same_v<typename Frag::elem, iro::elem::tf32> &&
                             std::is_same_v<typename OutTile::elem, iro::elem::tf32> &&
                             (OutTile::shape::template dim<0>() == 16) &&
                             (OutTile::shape::template dim<1>() == 16)) {
            if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_row_major> &&
                          std::is_same_v<typename OutTile::layout, iro::contract::layout::RowMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> f{};
                static_assert(Frag::count == f.num_elements, "TF32 fragment count mismatch");
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    f.x[i] = frag[i];
                }
                store_matrix_sync(reinterpret_cast<float*>(smem), f, ld, mem_row_major);
                return;
            } else if constexpr (std::is_same_v<typename Frag::dist, iro::dist::warp_col_major> &&
                                 std::is_same_v<typename OutTile::layout, iro::contract::layout::ColMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> f{};
                static_assert(Frag::count == f.num_elements, "TF32 fragment count mismatch");
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    f.x[i] = frag[i];
                }
                store_matrix_sync(reinterpret_cast<float*>(smem), f, ld, mem_col_major);
                return;
            }
        } else if constexpr (std::is_same_v<ExecGroup, iro::exec::warp> &&
                             std::is_same_v<typename Frag::elem, iro::elem::f32> &&
                             std::is_same_v<typename Frag::dist, iro::dist::accumulator> &&
                             std::is_same_v<typename OutTile::elem, iro::elem::f32> &&
                             (OutTile::shape::template dim<0>() == 16) &&
                             (OutTile::shape::template dim<1>() == 16)) {
            if constexpr (std::is_same_v<typename OutTile::layout, iro::contract::layout::RowMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<accumulator, 16, 16, 16, float> f{};
                static_assert(Frag::count == f.num_elements, "FragmentToSharedTile: AccFrag count mismatch");
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    f.x[i] = frag[i];
                }
                store_matrix_sync(reinterpret_cast<float*>(smem), f, ld, mem_row_major);
                return;
            } else if constexpr (std::is_same_v<typename OutTile::layout, iro::contract::layout::ColMajor<16>>) {
                using namespace nvcuda::wmma;
                fragment<accumulator, 16, 16, 16, float> f{};
                static_assert(Frag::count == f.num_elements, "FragmentToSharedTile: AccFrag count mismatch");
                #pragma unroll
                for (int i = 0; i < f.num_elements; ++i) {
                    f.x[i] = frag[i];
                }
                store_matrix_sync(reinterpret_cast<float*>(smem), f, ld, mem_col_major);
                return;
            }
        }
        static_assert(detail::always_false_v<OutTile>,
                      "SM89 FragmentToSharedTile supports WMMA 16x16 f16/bf16/tf32 and f32 accumulator (warp, row/col major).");
#else
        (void)frag; (void)smem; (void)ld;
#endif
    }
};

template<class Recipe, class Frag, class OutTile, class FragSubj, class OutSubj, class ExecGroup, class Lifetime>
struct FragmentToTile
    : iro::contract::Realization<
        axp::protocol::ownership::FragmentToTile<Recipe, Frag, OutTile, FragSubj, OutSubj, ExecGroup, Lifetime>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.ownership.frag_to_tile")> {
    __device__ static void execute(const typename Frag::elem::storage_t* frag,
                                   typename OutTile::elem::storage_t* smem,
                                   int ld) {
#ifdef __CUDA_ARCH__
        static_assert(iro::schema::ElemHasStorage<typename Frag::elem>);
        static_assert(iro::schema::ElemHasStorage<typename OutTile::elem>);
        static_assert(OutTile::shape::rank == 2, "FragmentToTile requires rank-2 tile");
        // Optimized WMMA accumulator store for 16x16 f32 fragments
        if constexpr (std::is_same_v<ExecGroup, iro::exec::warp> &&
                      std::is_same_v<typename Frag::elem, iro::elem::f32> &&
                      std::is_same_v<typename Frag::dist, iro::dist::accumulator> &&
                      (OutTile::shape::template dim<0>() == 16) &&
                      (OutTile::shape::template dim<1>() == 16) &&
                      std::is_same_v<typename OutTile::layout, iro::contract::layout::RowMajor<16>> &&
                      std::is_same_v<typename OutTile::elem, iro::elem::f32>) {
            using namespace nvcuda::wmma;
            fragment<accumulator, 16, 16, 16, float> f{};
            #pragma unroll
            for (int i = 0; i < f.num_elements; ++i) {
                f.x[i] = frag[i];
            }
            store_matrix_sync(reinterpret_cast<float*>(smem), f, ld, mem_row_major);
            return;
        }
        static_assert(detail::always_false_v<OutTile>,
                      "SM89 FragmentToTile supports only WMMA 16x16 f32 accumulator (warp, row-major).");
#else
        (void)frag; (void)smem; (void)ld;
#endif
    }
};

template<class Recipe, class Tile, class Subj, class ExecGroup, class Lifetime,
         class InExtra, class OutExtra>
struct TileBoundaryIn
    : iro::contract::Realization<
        axp::protocol::ownership::TileBoundaryIn<Recipe, Tile, Subj, ExecGroup, Lifetime, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.ownership.tile_boundary_in")> {
    template<class... Args>
    __device__ static void execute(Args&&...) {
#ifdef __CUDA_ARCH__
        static_assert(iro::contract::TilePayload<Tile>, "TileBoundaryIn: Tile must be TilePayload");
#endif
    }
};

template<class Recipe, class Tile, class Subj, class ExecGroup, class Lifetime,
         class InExtra, class OutExtra>
struct TileBoundaryOut
    : iro::contract::Realization<
        axp::protocol::ownership::TileBoundaryOut<Recipe, Tile, Subj, ExecGroup, Lifetime, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm89.ownership.tile_boundary_out")> {
    template<class... Args>
    __device__ static void execute(Args&&...) {
#ifdef __CUDA_ARCH__
        static_assert(iro::contract::TilePayload<Tile>, "TileBoundaryOut: Tile must be TilePayload");
#endif
    }
};

} // namespace axp::realize::sm89
