#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/realize/sm90.hpp is internal substrate; use axp/l4.hpp in application code."
#endif

#include <iro_cuda_ax_core.hpp>
#include <type_traits>
#ifdef __CUDACC__
#ifndef __cccl_lib_experimental_ctk12_cp_async_exposure
#define __cccl_lib_experimental_ctk12_cp_async_exposure
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/atomic>
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
#if defined(AXP_ENABLE_WGMMA) && defined(__CUDACC__)
#include "detail/wgmma_generated.hpp"
#endif
#include "../protocol/stage/pipeline_contracts.hpp"
#include "../protocol/stage/async_contracts.hpp"
#include "../protocol/compute/contracts.hpp"
#include "../protocol/compute/bundles.hpp"
#include "../protocol/reduction/contracts.hpp"
#include "../protocol/mask/contracts.hpp"
#include "../protocol/sync/contracts.hpp"
#include "../protocol/view/contracts.hpp"
#include "../protocol/ownership/contracts.hpp"
#include "../protocol/stage/resources.hpp"
#include "../protocol/convert/contracts.hpp"
#include "../protocol/tma/contracts.hpp"
#include "../detail/conversion.hpp"
#include "../level0/memory_cache.hpp"

namespace axp::realize::sm90 {

namespace detail {

using axp::detail::is_f32;
using axp::detail::is_f16;
using axp::detail::is_bf16;

#ifdef __CUDACC__
template<class Elem>
__host__ __device__ constexpr CUtensorMapDataType tensor_map_dtype() {
    if constexpr (std::is_same_v<Elem, iro::elem::f16>) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    } else if constexpr (std::is_same_v<Elem, iro::elem::bf16>) {
        return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    } else if constexpr (std::is_same_v<Elem, iro::elem::f32>) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    } else if constexpr (std::is_same_v<Elem, iro::elem::f64>) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
    } else if constexpr (std::is_same_v<Elem, iro::elem::u8>) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    } else if constexpr (std::is_same_v<Elem, iro::elem::u16>) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT16;
    } else if constexpr (std::is_same_v<Elem, iro::elem::u32>) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    } else if constexpr (std::is_same_v<Elem, iro::elem::u64>) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT64;
    } else if constexpr (std::is_same_v<Elem, iro::elem::i32>) {
        return CU_TENSOR_MAP_DATA_TYPE_INT32;
    } else if constexpr (std::is_same_v<Elem, iro::elem::i64>) {
        return CU_TENSOR_MAP_DATA_TYPE_INT64;
    } else {
        static_assert(!std::is_same_v<Elem, Elem>, "Unsupported tensor map element type");
    }
}
#endif
using axp::detail::is_fp8_e4m3_like;
using axp::detail::is_fp8_e5m2_like;
using axp::detail::is_supported_elem_v;
using axp::detail::to_f32_dispatch;
using axp::detail::from_f32_dispatch;

#if defined(__CUDACC__) && defined(AXP_CUDA_HAS_THREAD_SCOPE_CLUSTER)
template<class = void>
inline constexpr cuda::thread_scope cuda_cluster_scope = cuda::thread_scope::thread_scope_cluster;
#endif
using axp::detail::from_f32_recipe;
using axp::detail::from_f32_half;
using axp::detail::always_false_v;
using namespace axp::realize::common::detail;
} // namespace detail

// Stage realizations (SM90 - TMA + mbarrier)

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class SwizzleAtom>
struct CpAsyncIssue
    : iro::contract::Realization<
        axp::protocol::stage::CpAsyncIssue<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.stage.cp_async.issue")> {
    __device__ static void execute(const typename InTile::elem::storage_t* gmem,
                                   typename OutTile::elem::storage_t* smem,
                                   uint64_t base,
                                   cuda::pipeline<cuda::thread_scope_block>* pipe) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM90 CpAsyncIssue requires block exec group");
        static_assert(std::is_void_v<SwizzleAtom>,
                      "SM90 CpAsyncIssue does not support swizzled layouts; use explicit SwizzledStShared");
        constexpr uint32_t bytes = static_cast<uint32_t>(OutTile::bytes);
        static_assert(bytes % 16u == 0u, "SM90 CpAsyncIssue requires 16B granularity");
        char* smem_b = reinterpret_cast<char*>(smem);
        const char* gmem_b = reinterpret_cast<const char*>(gmem + base);
        for (uint32_t off = static_cast<uint32_t>(threadIdx.x) * 16u; off < bytes; off += static_cast<uint32_t>(blockDim.x) * 16u) {
            cuda::memcpy_async(*pipe, smem_b + off, gmem_b + off, 16u);
        }
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.stage.cp_async.commit")> {
    __device__ static void execute(cuda::pipeline<cuda::thread_scope_block>* pipe) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM90 CpAsyncCommit requires block exec group");
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.stage.cp_async.wait")> {
    __device__ static void execute(cuda::pipeline<cuda::thread_scope_block>* pipe) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM90 CpAsyncWait requires block exec group");
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.stage.issue")> {
    __device__ static void execute(const typename InTile::elem::storage_t* gmem,
                                   typename OutTile::elem::storage_t* smem,
                                   uint64_t base,
                                   cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM90 issue requires block exec group");
        static_assert(std::is_void_v<SwizzleAtom>,
                      "SM90 IssueGmemToSmemSlot pointer path does not support swizzled layouts; use tensor map");
        constexpr uint32_t bytes = static_cast<uint32_t>(OutTile::bytes);
        static_assert(bytes % 16u == 0u, "SM90 issue requires 16B granularity");
        if (threadIdx.x == 0) {
            cuda::device::barrier_expect_tx(*bar, bytes);
            cuda::device::experimental::cp_async_bulk_global_to_shared(
                reinterpret_cast<void*>(smem),
                reinterpret_cast<const void*>(gmem + base),
                bytes,
                *bar);
            cuda::device::experimental::cp_async_bulk_commit_group();
            *token_out = cuda::device::barrier_arrive_tx(*bar, 1, bytes);
        }
#else
        (void)gmem; (void)smem; (void)base; (void)bar; (void)token_out;
#endif
    }

#ifdef __CUDACC__
    __device__ static void execute(const CUtensorMap* map,
                                   typename OutTile::elem::storage_t* smem,
                                   int c0,
                                   cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        constexpr uint32_t bytes = static_cast<uint32_t>(OutTile::bytes);
        if (threadIdx.x == 0) {
            cuda::device::barrier_expect_tx(*bar, bytes);
            cuda::device::experimental::cp_async_bulk_tensor_1d_global_to_shared(
                reinterpret_cast<void*>(smem),
                map,
                c0,
                *bar);
            cuda::device::experimental::cp_async_bulk_commit_group();
            *token_out = cuda::device::barrier_arrive_tx(*bar, 1, bytes);
        }
#else
        (void)map; (void)smem; (void)c0; (void)bar; (void)token_out;
#endif
    }

    __device__ static void execute(const CUtensorMap* map,
                                   typename OutTile::elem::storage_t* smem,
                                   int c0,
                                   int c1,
                                   cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        constexpr uint32_t bytes = static_cast<uint32_t>(OutTile::bytes);
        if (threadIdx.x == 0) {
            cuda::device::barrier_expect_tx(*bar, bytes);
            cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<void*>(smem),
                map,
                c0,
                c1,
                *bar);
            cuda::device::experimental::cp_async_bulk_commit_group();
            *token_out = cuda::device::barrier_arrive_tx(*bar, 1, bytes);
        }
#else
        (void)map; (void)smem; (void)c0; (void)c1; (void)bar; (void)token_out;
#endif
    }
#endif
};

template<class Recipe, class OutTile, class SlotSubj, class ExecGroup, class Lifetime>
struct WaitSmemSlot
    : iro::contract::Realization<
        axp::protocol::stage::WaitSmemSlot<Recipe, OutTile, SlotSubj, ExecGroup, Lifetime>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.stage.wait")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_in) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM90 wait requires block exec group");
        bar->wait(*token_in);
#else
        (void)bar; (void)token_in;
#endif
    }
};

template<class Recipe, class OutTile, class SlotSubj, class ExecGroup, class Lifetime>
struct ReadySmemSlot
    : iro::contract::Realization<
        axp::protocol::stage::ReadySmemSlot<Recipe, OutTile, SlotSubj, ExecGroup, Lifetime>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.stage.ready")> {
    __device__ static void execute() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM90 ready requires block exec group");
#endif
    }
};

template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class ExecGroup, class Lifetime, int Slots, class SwizzleAtom>
struct DirectGmemToSmemSlot
    : iro::contract::Realization<
        axp::protocol::stage::DirectGmemToSmemSlot<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.stage.direct")> {
    __device__ static void execute(const typename InTile::elem::storage_t* gmem,
                                   typename OutTile::elem::storage_t* smem,
                                   uint64_t base,
                                   cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM90 direct requires block exec group");
        static_assert(std::is_void_v<SwizzleAtom>,
                      "SM90 DirectGmemToSmemSlot does not support swizzled layouts; use TMA");
        constexpr uint32_t bytes = static_cast<uint32_t>(OutTile::bytes);
        static_assert(bytes % 16u == 0u, "SM90 direct requires 16B granularity");
        char* smem_b = reinterpret_cast<char*>(smem);
        const char* gmem_b = reinterpret_cast<const char*>(gmem + base);
        for (uint32_t off = static_cast<uint32_t>(threadIdx.x) * 16u; off < bytes; off += static_cast<uint32_t>(blockDim.x) * 16u) {
            *reinterpret_cast<uint4*>(smem_b + off) = *reinterpret_cast<const uint4*>(gmem_b + off);
        }
        __syncthreads();
#else
        (void)gmem; (void)smem; (void)base; (void)bar; (void)token_out;
#endif
    }
};

template<class Recipe, class OutTile, class SlotSubj, class BarrierSubj, class ExecGroup, class Lifetime>
struct CommitSmemSlot
    : iro::contract::Realization<
        axp::protocol::stage::CommitSmemSlot<Recipe, OutTile, SlotSubj, BarrierSubj, ExecGroup, Lifetime>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.stage.commit")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_in) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM90 commit requires block exec group");
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.stage.release")> {
    __device__ static void execute() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM90 release requires block exec group");
#endif
    }
};

template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime, long long Bytes>
struct MarkConsumed
    : iro::contract::Realization<
        axp::protocol::stage::MarkConsumed<Recipe, SlotSubj, ExecGroup, Lifetime, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.stage.mark_consumed")> {
    __device__ static void execute() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::warp> ||
                      iro::exec::is_warpgroup_v<ExecGroup> ||
                      std::is_same_v<ExecGroup, iro::exec::block>,
                      "SM90 mark-consumed requires warp/warpgroup/block");
#endif
    }
};

template<class Recipe, class InTile, class OutTile, class SlotSubj, class OutSubj,
         class ExecGroup, class Lifetime>
struct StoreSmemToGmemSlot
    : iro::contract::Realization<
        axp::protocol::stage::StoreSmemToGmemSlot<Recipe, InTile, OutTile, SlotSubj, OutSubj, ExecGroup, Lifetime>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.stage.store_smem_to_gmem")> {
    __device__ static void execute(const typename InTile::elem::storage_t* smem,
                                   typename OutTile::elem::storage_t* gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM90 store requires block exec group");
        constexpr uint32_t bytes = static_cast<uint32_t>(OutTile::bytes);
        static_assert(bytes % 16u == 0u, "SM90 store requires 16B granularity");
        const char* smem_b = reinterpret_cast<const char*>(smem);
        char* gmem_b = reinterpret_cast<char*>(gmem);
        for (uint32_t off = static_cast<uint32_t>(threadIdx.x) * 16u; off < bytes; off += static_cast<uint32_t>(blockDim.x) * 16u) {
            *reinterpret_cast<int4*>(gmem_b + off) = *reinterpret_cast<const int4*>(smem_b + off);
        }
#else
        (void)smem; (void)gmem;
#endif
    }
};

template<class Recipe, class SlotSubj, class BarrierSubj, class ExecGroup, class Lifetime, long long Bytes>
struct CommitSmemStoreSlot
    : iro::contract::Realization<
        axp::protocol::stage::CommitSmemStoreSlot<Recipe, SlotSubj, BarrierSubj, ExecGroup, Lifetime, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.stage.commit_store")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_in) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM90 store-commit requires block exec group");
        bar->wait(*token_in);
#else
        (void)bar; (void)token_in;
#endif
    }
};

template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime, long long Bytes>
struct PassSlot
    : iro::contract::Realization<
        axp::protocol::stage::PassSlot<Recipe, SlotSubj, ExecGroup, Lifetime, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.stage.pass_slot")> {
    __device__ static void execute() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SM90 pass-slot requires block exec group");
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.stage.slot_after")> {
    using storage_t = typename DepPayload::elem::storage_t;
    __device__ static void execute(const storage_t* /*dep*/) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<SlotExecGroup, iro::exec::warp> ||
                      iro::exec::is_warpgroup_v<SlotExecGroup> ||
                      std::is_same_v<SlotExecGroup, iro::exec::block>,
                      "SM90 slot-after requires warp/warpgroup/block exec group");
#endif
    }
};

// TMA realizations (SM90)

template<class Recipe, class MapHandle, class MapSubj, class ExecGroup, class Lifetime>
struct HostMakeTensorMap
    : iro::contract::Realization<
        axp::protocol::tma::HostMakeTensorMap<Recipe, MapHandle, MapSubj, ExecGroup, Lifetime>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.tma.make_host")> {
    __host__ static CUresult execute(CUtensorMap* out,
                                     void* global_ptr,
                                     const cuuint64_t* global_dim,
                                     const cuuint64_t* global_strides,
                                     const cuuint32_t* box_dim,
                                     const cuuint32_t* element_strides,
                                     CUtensorMapInterleave interleave,
                                     CUtensorMapSwizzle swizzle,
                                     CUtensorMapL2promotion l2_promotion,
                                     CUtensorMapFloatOOBfill oob_fill) {
#if defined(__CUDACC__)
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "HostMakeTensorMap requires block exec group");
        constexpr cuuint32_t rank = static_cast<cuuint32_t>(MapHandle::tile::shape::rank);
        static_assert(rank >= 1 && rank <= 5, "HostMakeTensorMap: rank must be 1..5");
        const CUtensorMapDataType dtype = detail::tensor_map_dtype<typename MapHandle::tile::elem>();
        return cuTensorMapEncodeTiled(
            out, dtype, rank, global_ptr, global_dim, global_strides,
            box_dim, element_strides, interleave, swizzle, l2_promotion, oob_fill);
#else
        (void)out; (void)global_ptr; (void)global_dim; (void)global_strides;
        (void)box_dim; (void)element_strides; (void)interleave; (void)swizzle;
        (void)l2_promotion; (void)oob_fill;
        return CUDA_ERROR_NOT_SUPPORTED;
#endif
    }
};

template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct BulkTmaCopy2D
    : iro::contract::Realization<
        axp::protocol::tma::BulkTmaCopy2D<Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
                                          Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj,
                                          ExecGroup, Lifetime, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.tma.copy2d")> {
    using coord0_t = typename Coord0Payload::elem::storage_t;
    using coord1_t = typename Coord1Payload::elem::storage_t;
    __device__ static void execute(const CUtensorMap* map,
                                   typename SmemTile::elem::storage_t* smem,
                                   const coord0_t* c0, const coord1_t* c1,
                                   cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BulkTmaCopy2D requires block exec group");
        static_assert(Bytes > 0, "BulkTmaCopy2D requires positive byte count");
        static_assert(SmemTile::align::bytes >= 16, "BulkTmaCopy2D requires 16B-aligned smem tile");
        if (threadIdx.x == 0) {
            const int c0v = static_cast<int>(c0[0]);
            const int c1v = static_cast<int>(c1[0]);
            cuda::device::barrier_expect_tx(*bar, static_cast<uint32_t>(Bytes));
            cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<void*>(smem), map, c0v, c1v, *bar);
            cuda::device::experimental::cp_async_bulk_commit_group();
            *token_out = cuda::device::barrier_arrive_tx(*bar, 1, static_cast<uint32_t>(Bytes));
        }
        __syncthreads();
#else
        (void)map; (void)smem; (void)c0; (void)c1; (void)bar; (void)token_out;
#endif
    }
};

template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class Coord0Payload, class Coord0Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct BulkTmaCopy1D
    : iro::contract::Realization<
        axp::protocol::tma::BulkTmaCopy1D<Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
                                          Coord0Payload, Coord0Subj, ExecGroup, Lifetime, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.tma.copy1d")> {
    using coord0_t = typename Coord0Payload::elem::storage_t;
    __device__ static void execute(const CUtensorMap* map,
                                   typename SmemTile::elem::storage_t* smem,
                                   const coord0_t* c0,
                                   cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BulkTmaCopy1D requires block exec group");
        static_assert(Bytes > 0, "BulkTmaCopy1D requires positive byte count");
        static_assert(SmemTile::align::bytes >= 16, "BulkTmaCopy1D requires 16B-aligned smem tile");
        if (threadIdx.x == 0) {
            const int c0v = static_cast<int>(c0[0]);
            cuda::device::barrier_expect_tx(*bar, static_cast<uint32_t>(Bytes));
            cuda::device::experimental::cp_async_bulk_tensor_1d_global_to_shared(
                reinterpret_cast<void*>(smem), map, c0v, *bar);
            cuda::device::experimental::cp_async_bulk_commit_group();
            *token_out = cuda::device::barrier_arrive_tx(*bar, 1, static_cast<uint32_t>(Bytes));
        }
        __syncthreads();
#else
        (void)map; (void)smem; (void)c0; (void)bar; (void)token_out;
#endif
    }
};

#if defined(AXP_CUDA_HAS_THREAD_SCOPE_CLUSTER)
template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class MaskPayload, class MaskSubj,
         class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct BulkTmaCopyMulticast2D
    : iro::contract::Realization<
        axp::protocol::tma::BulkTmaCopyMulticast2D<Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
                                                   MaskPayload, MaskSubj,
                                                   Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj,
                                                   ExecGroup, Lifetime, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.tma.copy_multicast_2d")> {
    using coord0_t = typename Coord0Payload::elem::storage_t;
    using coord1_t = typename Coord1Payload::elem::storage_t;
    using mask_t = typename MaskPayload::elem::storage_t;
    __device__ static void execute(const CUtensorMap* map,
                                   typename SmemTile::elem::storage_t* smem,
                                   const mask_t* mask,
                                   const coord0_t* c0, const coord1_t* c1,
                                   cuda::barrier<detail::cuda_cluster_scope<>>* bar,
                                   cuda::barrier<detail::cuda_cluster_scope<>>::arrival_token* token_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::cluster>,
                      "BulkTmaCopyMulticast2D requires cluster exec group");
        static_assert(Bytes > 0, "BulkTmaCopyMulticast2D requires positive byte count");
        static_assert(SmemTile::align::bytes >= 16, "BulkTmaCopyMulticast2D requires 16B-aligned smem tile");
        if (threadIdx.x == 0) {
            const int c0v = static_cast<int>(c0[0]);
            const int c1v = static_cast<int>(c1[0]);
            const uint32_t maskv = static_cast<uint32_t>(mask[0]);
            cuda::device::barrier_expect_tx(*bar, static_cast<uint32_t>(Bytes));
            cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared_cluster(
                reinterpret_cast<void*>(smem), map, maskv, c0v, c1v, *bar);
            cuda::device::experimental::cp_async_bulk_commit_group();
            *token_out = cuda::device::barrier_arrive_tx(*bar, 1, static_cast<uint32_t>(Bytes));
        }
        __syncthreads();
#else
        (void)map; (void)smem; (void)mask; (void)c0; (void)c1; (void)bar; (void)token_out;
#endif
    }
};
#endif

#if defined(AXP_CUDA_HAS_THREAD_SCOPE_CLUSTER)
template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class MaskPayload, class MaskSubj,
         class Coord0Payload, class Coord0Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct BulkTmaCopyMulticast1D
    : iro::contract::Realization<
        axp::protocol::tma::BulkTmaCopyMulticast1D<Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
                                                   MaskPayload, MaskSubj,
                                                   Coord0Payload, Coord0Subj, ExecGroup, Lifetime, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.tma.copy_multicast_1d")> {
    using coord0_t = typename Coord0Payload::elem::storage_t;
    using mask_t = typename MaskPayload::elem::storage_t;
    __device__ static void execute(const CUtensorMap* map,
                                   typename SmemTile::elem::storage_t* smem,
                                   const mask_t* mask,
                                   const coord0_t* c0,
                                   cuda::barrier<detail::cuda_cluster_scope<>>* bar,
                                   cuda::barrier<detail::cuda_cluster_scope<>>::arrival_token* token_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::cluster>,
                      "BulkTmaCopyMulticast1D requires cluster exec group");
        static_assert(Bytes > 0, "BulkTmaCopyMulticast1D requires positive byte count");
        static_assert(SmemTile::align::bytes >= 16, "BulkTmaCopyMulticast1D requires 16B-aligned smem tile");
        if (threadIdx.x == 0) {
            const int c0v = static_cast<int>(c0[0]);
            const uint32_t maskv = static_cast<uint32_t>(mask[0]);
            cuda::device::barrier_expect_tx(*bar, static_cast<uint32_t>(Bytes));
            cuda::device::experimental::cp_async_bulk_tensor_1d_global_to_shared_cluster(
                reinterpret_cast<void*>(smem), map, maskv, c0v, *bar);
            cuda::device::experimental::cp_async_bulk_commit_group();
            *token_out = cuda::device::barrier_arrive_tx(*bar, 1, static_cast<uint32_t>(Bytes));
        }
        __syncthreads();
#else
        (void)map; (void)smem; (void)mask; (void)c0; (void)bar; (void)token_out;
#endif
    }
};
#endif

template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class Coord0Payload, class Coord1Payload, class Coord0Subj, class Coord1Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct BulkTmaStore2D
    : iro::contract::Realization<
        axp::protocol::tma::BulkTmaStore2D<Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
                                           Coord0Payload, Coord1Payload, Coord0Subj, Coord1Subj,
                                           ExecGroup, Lifetime, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.tma.store2d")> {
    using coord0_t = typename Coord0Payload::elem::storage_t;
    using coord1_t = typename Coord1Payload::elem::storage_t;
    __device__ static void execute(const CUtensorMap* map,
                                   const typename SmemTile::elem::storage_t* smem,
                                   const coord0_t* c0, const coord1_t* c1,
                                   cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BulkTmaStore2D requires block exec group");
        static_assert(Bytes > 0, "BulkTmaStore2D requires positive byte count");
        static_assert(SmemTile::align::bytes >= 16, "BulkTmaStore2D requires 16B-aligned smem tile");
        if (threadIdx.x == 0) {
            const int c0v = static_cast<int>(c0[0]);
            const int c1v = static_cast<int>(c1[0]);
            cuda::device::barrier_expect_tx(*bar, static_cast<uint32_t>(Bytes));
            cuda::device::experimental::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<void*>(const_cast<typename SmemTile::elem::storage_t*>(smem)), map, c0v, c1v, *bar);
            cuda::device::experimental::cp_async_bulk_commit_group();
            *token_out = cuda::device::barrier_arrive_tx(*bar, 1, static_cast<uint32_t>(Bytes));
        }
        __syncthreads();
#else
        (void)map; (void)smem; (void)c0; (void)c1; (void)bar; (void)token_out;
#endif
    }
};

template<class Recipe, class MapHandle, class SmemTile,
         class MapSubj, class SmemSubj, class BarrierSubj,
         class Coord0Payload, class Coord0Subj,
         class ExecGroup, class Lifetime, long long Bytes>
struct BulkTmaStore1D
    : iro::contract::Realization<
        axp::protocol::tma::BulkTmaStore1D<Recipe, MapHandle, SmemTile, MapSubj, SmemSubj, BarrierSubj,
                                           Coord0Payload, Coord0Subj, ExecGroup, Lifetime, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.tma.store1d")> {
    using coord0_t = typename Coord0Payload::elem::storage_t;
    __device__ static void execute(const CUtensorMap* map,
                                   const typename SmemTile::elem::storage_t* smem,
                                   const coord0_t* c0,
                                   cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BulkTmaStore1D requires block exec group");
        static_assert(Bytes > 0, "BulkTmaStore1D requires positive byte count");
        static_assert(SmemTile::align::bytes >= 16, "BulkTmaStore1D requires 16B-aligned smem tile");
        if (threadIdx.x == 0) {
            const int c0v = static_cast<int>(c0[0]);
            cuda::device::barrier_expect_tx(*bar, static_cast<uint32_t>(Bytes));
            cuda::device::experimental::cp_async_bulk_tensor_1d_shared_to_global(
                reinterpret_cast<void*>(const_cast<typename SmemTile::elem::storage_t*>(smem)), map, c0v, *bar);
            cuda::device::experimental::cp_async_bulk_commit_group();
            *token_out = cuda::device::barrier_arrive_tx(*bar, 1, static_cast<uint32_t>(Bytes));
        }
        __syncthreads();
#else
        (void)map; (void)smem; (void)c0; (void)bar; (void)token_out;
#endif
    }
};

// Compute realizations (SM90 - warpgroup MMA via descriptors)

template<class Recipe, class Shape, class ADesc, class BDesc, class AccFrag,
         class ADescSubj, class BDescSubj, class AccSubj, class WgmmaSubj>
struct WarpgroupMmaFromDesc
    : iro::contract::Realization<
        axp::protocol::compute::WarpgroupMmaFromDesc<Recipe, Shape, ADesc, BDesc, AccFrag, ADescSubj, BDescSubj, AccSubj, WgmmaSubj>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.compute.wg_mma")> {
    __device__ static void execute(uint64_t desc_a, uint64_t desc_b,
                                   const axp::protocol::compute::WgmmaHandle* /*in_handle*/,
                                   float* acc,
                                   axp::protocol::compute::WgmmaHandle* /*out_handle*/) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<typename ADesc::tile::space, iro::contract::space::shared>);
        static_assert(std::is_same_v<typename BDesc::tile::space, iro::contract::space::shared>);
        static_assert(std::is_same_v<typename AccFrag::elem, iro::elem::f32>);
        static_assert(ADesc::tile::shape::rank == 2 && BDesc::tile::shape::rank == 2);
        static_assert(Shape::m == 64, "SM90 WarpgroupMmaFromDesc requires M=64");
        static_assert(ADesc::tile::shape::template dim<0>() == Shape::m &&
                      ADesc::tile::shape::template dim<1>() == Shape::k);
        static_assert(BDesc::tile::shape::template dim<0>() == Shape::k &&
                      BDesc::tile::shape::template dim<1>() == Shape::n);

        static_assert(AccFrag::count == (Shape::n / 2),
                      "WarpgroupMmaFromDesc: AccFrag count must be N/2 for f32 accum");
        detail::wgmma_mma_async<Shape>(desc_a, desc_b, acc);
#else
        (void)desc_a; (void)desc_b; (void)acc;
#endif
    }
};

// Reduction realization (SM90)

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup>
struct BlockReduce
    : iro::contract::Realization<
        axp::protocol::reduction::BlockReduce<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.reduction.block")> {
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

// Mask realizations

template<class Recipe, class MaskFragT, class MaskSubj, class ExecGroup>
struct MaskGen
    : iro::contract::Realization<
        axp::protocol::mask::MaskGen<Recipe, MaskFragT, MaskSubj, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.mask.gen")> {
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

// L0 memory realizations (SM90, vectorized global/shared)

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CachePolicy, class InDist, class OutDist, class InExtra, class OutExtra>
struct LdGlobal
    : iro::contract::Realization<
        axp::level0::LdGlobal<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, CachePolicy, InDist, OutDist, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.l0.ld_global")> {
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.l0.st_global")> {
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.l0.prefetch_global")> {
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.l0.ld_shared")> {
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.l0.st_shared")> {
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.l0.swizzled_ld_shared")> {
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.l0.swizzled_st_shared")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in,
                                   typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
        detail::tile_copy_row_to_swizzled<Recipe, InTile, OutTile, ExecGroup, SwizzleAtom>(in, out);
#else
        (void)in; (void)out;
#endif
    }
};

// Convert realizations (SM90)

template<class RecipeIn, class RecipeOut, class InTile, class OutTile,
         class InSubj, class OutSubj, class ExecGroup, int VecBytes,
         class InDist, class OutDist>
struct CastTile
    : iro::contract::Realization<
        axp::protocol::convert::CastTile<RecipeIn, RecipeOut, InTile, OutTile, InSubj, OutSubj, ExecGroup, VecBytes, InDist, OutDist>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.convert.tile")> {
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.convert.frag")> {
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.convert.frag_to_vec_slice")> {
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.convert.vec_slice_to_frag")> {
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

// Sync realization (SM90)

template<class Recipe, class Subject, class ExecGroup, int Expected>
struct BarrierInit
    : iro::contract::Realization<
        axp::protocol::sync::BarrierInit<Recipe, Subject, ExecGroup, Expected>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.sync.barrier_init")> {
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

#if defined(AXP_CUDA_HAS_THREAD_SCOPE_CLUSTER)
template<class Recipe, class Subject, class ExecGroup, int Expected>
struct ClusterBarrierInit
    : iro::contract::Realization<
        axp::protocol::sync::ClusterBarrierInit<Recipe, Subject, ExecGroup, Expected>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.sync.cluster_barrier_init")> {
    __device__ static void execute(cuda::barrier<detail::cuda_cluster_scope<>>* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::cluster>, "ClusterBarrierInit requires cluster exec group");
        static_assert(Expected > 0, "ClusterBarrierInit requires positive expected count");
        if (threadIdx.x == 0) {
            init(bar, Expected);
        }
        __syncthreads();
#else
        (void)bar;
#endif
    }
};
#endif

template<class Recipe, class Subject, class ExecGroup>
struct SyncPoint
    : iro::contract::Realization<
        axp::protocol::sync::SyncPoint<Recipe, Subject, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.sync.point")> {
    __device__ static void execute() {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                      "SM90 SyncPoint supports warp only; warpgroup sync requires an explicit barrier primitive");
        __syncwarp();
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup>
struct SyncWarp
    : iro::contract::Realization<
        axp::protocol::sync::SyncWarp<Recipe, Subject, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.sync.warp")> {
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.sync.threads")> {
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.sync.fence")> {
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.sync.barrier_arrive")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierArrive v1 supports block exec only");
        *token_out = bar->arrive();
#else
        (void)bar; (void)token_out;
#endif
    }
};

#if defined(AXP_CUDA_HAS_THREAD_SCOPE_CLUSTER)
template<class Recipe, class Subject, class ExecGroup>
struct ClusterBarrierArrive
    : iro::contract::Realization<
        axp::protocol::sync::ClusterBarrierArrive<Recipe, Subject, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.sync.cluster_barrier_arrive")> {
    __device__ static void execute(cuda::barrier<detail::cuda_cluster_scope<>>* bar,
                                   cuda::barrier<detail::cuda_cluster_scope<>>::arrival_token* token_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::cluster>, "ClusterBarrierArrive requires cluster exec group");
        *token_out = bar->arrive();
#else
        (void)bar; (void)token_out;
#endif
    }
};
#endif

template<class Recipe, class Subject, class ExecGroup, long long Bytes>
struct BarrierExpectTx
    : iro::contract::Realization<
        axp::protocol::sync::BarrierExpectTx<Recipe, Subject, ExecGroup, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.sync.barrier_expect_tx")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierExpectTx requires block exec group");
        static_assert(Bytes > 0, "BarrierExpectTx requires positive byte count");
        cuda::device::barrier_expect_tx(*bar, static_cast<uint32_t>(Bytes));
#else
        (void)bar;
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup, long long Bytes>
struct BarrierArriveTx
    : iro::contract::Realization<
        axp::protocol::sync::BarrierArriveTx<Recipe, Subject, ExecGroup, Bytes>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.sync.barrier_arrive_tx")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierArriveTx requires block exec group");
        static_assert(Bytes > 0, "BarrierArriveTx requires positive byte count");
        *token_out = cuda::device::barrier_arrive_tx(*bar, 1, static_cast<uint32_t>(Bytes));
#else
        (void)bar; (void)token_out;
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup>
struct BarrierWait
    : iro::contract::Realization<
        axp::protocol::sync::BarrierWait<Recipe, Subject, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.sync.barrier_wait")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_in) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierWait v1 supports block exec only");
        bar->wait(*token_in);
#else
        (void)bar; (void)token_in;
#endif
    }
};

#if defined(AXP_CUDA_HAS_THREAD_SCOPE_CLUSTER)
template<class Recipe, class Subject, class ExecGroup>
struct ClusterBarrierWait
    : iro::contract::Realization<
        axp::protocol::sync::ClusterBarrierWait<Recipe, Subject, ExecGroup>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.sync.cluster_barrier_wait")> {
    __device__ static void execute(cuda::barrier<detail::cuda_cluster_scope<>>* bar,
                                   cuda::barrier<detail::cuda_cluster_scope<>>::arrival_token* token_in) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(std::is_same_v<ExecGroup, iro::exec::cluster>, "ClusterBarrierWait requires cluster exec group");
        bar->wait(*token_in);
#else
        (void)bar; (void)token_in;
#endif
    }
};
#endif

template<class Recipe, class Subject, class ExecGroup, class FlagPayload, class FlagSubj, class OutExtra>
struct BarrierTryWait
    : iro::contract::Realization<
        axp::protocol::sync::BarrierTryWait<Recipe, Subject, ExecGroup, FlagPayload, FlagSubj, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.sync.barrier_try_wait")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar,
                                   cuda::barrier<cuda::thread_scope_block>::arrival_token* token_in,
                                   typename FlagPayload::elem::storage_t* out_flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.sync.barrier_invalidate")> {
    __device__ static void execute(cuda::barrier<cuda::thread_scope_block>* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
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

// Warp reduction (SM90 fast path using warp reduce intrinsics)
template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag>
struct WarpReduce
    : iro::contract::Realization<
        axp::protocol::reduction::WarpReduce<Recipe, Frag, Subj, ExecGroup, OpTag>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.reduction.warp")> {
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

// Warp all-reduce (SM90)
template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag>
struct WarpAllReduce
    : iro::contract::Realization<
        axp::protocol::reduction::WarpAllReduce<Recipe, Frag, Subj, ExecGroup, OpTag>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.reduction.warp_all")> {
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

// Warp segmented all-reduce (SM90)
template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag, int SegmentWidth>
struct WarpSegmentedReduce
    : iro::contract::Realization<
        axp::protocol::reduction::WarpSegmentedReduce<Recipe, Frag, Subj, ExecGroup, OpTag, SegmentWidth>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.reduction.warp_segmented")> {
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

// WGMMA control atoms (SM90)

template<class Recipe, class Subject, class ExecGroup, class InExtra, class OutExtra>
struct WgmmaFence
    : iro::contract::Realization<
        axp::level0::WgmmaFence<Recipe, Subject, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.wgmma.fence")> {
    __device__ static void execute(const axp::protocol::compute::WgmmaHandle* /*in*/,
                                   axp::protocol::compute::WgmmaHandle* /*out*/) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(iro::exec::is_warpgroup_v<ExecGroup>, "WgmmaFence requires warpgroup exec");
        detail::wgmma_fence();
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup, int Group, class InExtra, class OutExtra>
struct WgmmaCommitGroup
    : iro::contract::Realization<
        axp::level0::WgmmaCommitGroup<Recipe, Subject, ExecGroup, Group, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.wgmma.commit_group")> {
    __device__ static void execute(const axp::protocol::compute::WgmmaHandle* /*in*/,
                                   axp::protocol::compute::WgmmaHandle* /*out*/) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(iro::exec::is_warpgroup_v<ExecGroup>, "WgmmaCommitGroup requires warpgroup exec");
        detail::wgmma_commit_group();
#endif
    }
};

template<class Recipe, class Subject, class ExecGroup, int Group, class InExtra, class OutExtra>
struct WgmmaWaitGroup
    : iro::contract::Realization<
        axp::level0::WgmmaWaitGroup<Recipe, Subject, ExecGroup, Group, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.wgmma.wait_group")> {
    __device__ static void execute(const axp::protocol::compute::WgmmaHandle* /*in*/,
                                   axp::protocol::compute::WgmmaHandle* /*out*/) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        static_assert(iro::exec::is_warpgroup_v<ExecGroup>, "WgmmaWaitGroup requires warpgroup exec");
        detail::wgmma_wait_group<Group>();
#endif
    }
};

template<class Recipe, class AccFrag, class InSubj, class OutSubj, class WgmmaSubj, class ExecGroup, int Group,
         class InExtra, class OutExtra>
struct WgmmaWaitAcc
    : iro::contract::Realization<
        axp::level0::WgmmaWaitAcc<Recipe, AccFrag, InSubj, OutSubj, WgmmaSubj, ExecGroup, Group, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.wgmma.wait_acc")> {
    using storage_t = typename AccFrag::elem::storage_t;
    __device__ static void execute(const storage_t* acc_in,
                                   const axp::protocol::compute::WgmmaHandle* /*handle*/,
                                   storage_t* acc_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        constexpr int kElems = static_cast<int>(AccFrag::count);
        if (acc_out != acc_in) {
            #pragma unroll
            for (int i = 0; i < kElems; ++i) {
                acc_out[i] = acc_in[i];
            }
        }
#else
        (void)acc_in; (void)acc_out;
#endif
    }
};

// WGMMA descriptor creation (SM90)

namespace detail {
template<class Layout>
struct wgmma_layout_info {
    static constexpr bool valid = false;
    static constexpr int lbo_elems = 0;
};

template<int RowMajorCols>
struct wgmma_layout_info<iro::contract::layout::RowMajor<RowMajorCols>> {
    static constexpr bool valid = true;
    static constexpr int lbo_elems = RowMajorCols;
};

template<int ColMajorRows>
struct wgmma_layout_info<iro::contract::layout::ColMajor<ColMajorRows>> {
    static constexpr bool valid = true;
    static constexpr int lbo_elems = ColMajorRows;
};

// Allow WGMMA descriptor creation from swizzled row-major layouts.
template<int RowMajorCols, int B, int S>
struct wgmma_layout_info<iro::contract::layout::Swizzled<RowMajorCols, B, S>> {
    static constexpr bool valid = true;
    static constexpr int lbo_elems = RowMajorCols;
};

template<int ColMajorRows, int B, int S>
struct wgmma_layout_info<iro::contract::layout::SwizzledColMajor<ColMajorRows, B, S>> {
    static constexpr bool valid = true;
    static constexpr int lbo_elems = ColMajorRows;
};
} // namespace detail

template<class Recipe, class SmemTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime, class SwizzleAtom>
struct UseWgmmaSmemDesc
    : iro::contract::Realization<
        axp::protocol::ownership::UseWgmmaSmemDesc<Recipe, SmemTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.ownership.wgmma_desc_use")> {
    __device__ static void execute(const typename SmemTile::elem::storage_t* /*smem*/,
                                   uint64_t desc_in,
                                   uint64_t* desc_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        *desc_out = desc_in;
#else
        (void)desc_in; (void)desc_out;
#endif
    }
};

template<class Recipe, class SmemTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime, class SwizzleAtom>
struct MakeWgmmaSmemDesc
    : iro::contract::Realization<
        axp::protocol::ownership::MakeWgmmaSmemDesc<Recipe, SmemTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.ownership.wgmma_desc")> {
    static_assert(SmemTile::shape::rank == 2, "WGMMA desc requires rank-2 tile");
    static_assert(SmemTile::align::bytes >= 16, "WGMMA desc requires >=16B smem alignment");
    static_assert((SmemTile::align::bytes % 16) == 0, "WGMMA desc requires 16B-aligned smem base");
    using layout_info = detail::wgmma_layout_info<typename SmemTile::layout>;
    static_assert(layout_info::valid, "WGMMA desc requires RowMajor/ColMajor layout");
    static constexpr uint64_t kElemBytes = static_cast<uint64_t>(sizeof(typename SmemTile::elem::storage_t));
    static constexpr uint64_t kMaxUnits = 0x3FFFu;
    static constexpr uint64_t kLboBytes = static_cast<uint64_t>(layout_info::lbo_elems) * kElemBytes;
    static constexpr uint64_t kSboBytes = kLboBytes * 16u;
    static_assert((kLboBytes % 16u) == 0u, "WGMMA desc LBO must be 16B aligned");
    static_assert((kSboBytes % 16u) == 0u, "WGMMA desc SBO must be 16B aligned");
    static_assert((kLboBytes >> 4) <= kMaxUnits, "WGMMA desc LBO exceeds 14-bit field");
    static_assert((kSboBytes >> 4) <= kMaxUnits, "WGMMA desc SBO exceeds 14-bit field");

    __device__ static void execute(const typename SmemTile::elem::storage_t* smem,
                                   uint64_t* desc_out,
                                   int lbo_elems,
                                   int sbo_elems,
                                   int base_offset) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        const uint32_t lbo_bytes = static_cast<uint32_t>(lbo_elems * sizeof(typename SmemTile::elem::storage_t));
        const uint32_t sbo_bytes = static_cast<uint32_t>(sbo_elems * sizeof(typename SmemTile::elem::storage_t));
        detail::WgmmaSwizzle swz = detail::WgmmaSwizzle::none;
        if constexpr (std::is_same_v<SwizzleAtom, axp::protocol::stage::SwizzleAtom_128B>) {
            swz = detail::WgmmaSwizzle::swizzle_128b;
        } else if constexpr (std::is_same_v<SwizzleAtom, axp::protocol::stage::SwizzleAtom_64B>) {
            swz = detail::WgmmaSwizzle::swizzle_64b;
        } else if constexpr (std::is_same_v<SwizzleAtom, axp::protocol::stage::SwizzleAtom_32B>) {
            swz = detail::WgmmaSwizzle::swizzle_32b;
        } else {
            swz = detail::WgmmaSwizzle::none;
        }
        *desc_out = detail::make_wgmma_smem_desc(smem, lbo_bytes, sbo_bytes, swz, static_cast<uint32_t>(base_offset));
#else
        (void)smem; (void)desc_out; (void)lbo_elems; (void)sbo_elems; (void)base_offset;
#endif
    }
};

template<class Recipe, class SmemTile, class DescTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime,
         class SwizzleAtom, int RowOffset, int ColOffset>
struct MakeWgmmaSmemDescSlice
    : iro::contract::Realization<
        axp::protocol::ownership::MakeWgmmaSmemDescSlice<
            Recipe, SmemTile, DescTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom, RowOffset, ColOffset>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.ownership.wgmma_desc_slice")> {
    static_assert(SmemTile::shape::rank == 2, "WGMMA desc requires rank-2 tile");
    static_assert(DescTile::shape::rank == 2, "WGMMA desc requires rank-2 tile");
    static_assert(SmemTile::align::bytes >= 16, "WGMMA desc requires >=16B smem alignment");
    static_assert((SmemTile::align::bytes % 16) == 0, "WGMMA desc requires 16B-aligned smem base");
    using layout_info = detail::wgmma_layout_info<typename DescTile::layout>;
    static_assert(layout_info::valid, "WGMMA desc requires RowMajor/ColMajor layout");
    static constexpr uint64_t kElemBytes = static_cast<uint64_t>(sizeof(typename SmemTile::elem::storage_t));
    static constexpr uint64_t kMaxUnits = 0x3FFFu;
    static constexpr uint64_t kLboBytes = static_cast<uint64_t>(layout_info::lbo_elems) * kElemBytes;
    static constexpr uint64_t kSboBytes = kLboBytes * 16u;
    static_assert((kLboBytes % 16u) == 0u, "WGMMA desc LBO must be 16B aligned");
    static_assert((kSboBytes % 16u) == 0u, "WGMMA desc SBO must be 16B aligned");
    static_assert((kLboBytes >> 4) <= kMaxUnits, "WGMMA desc LBO exceeds 14-bit field");
    static_assert((kSboBytes >> 4) <= kMaxUnits, "WGMMA desc SBO exceeds 14-bit field");

    using contract_t = axp::protocol::ownership::MakeWgmmaSmemDescSlice<
        Recipe, SmemTile, DescTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom, RowOffset, ColOffset>;
    static constexpr uint64_t kPtrOffsetBytes = contract_t::kPtrOffsetBytes;
    static constexpr uint32_t kBaseOffsetUnits = contract_t::kBaseOffsetUnits;

    __device__ static void execute(const typename SmemTile::elem::storage_t* smem,
                                   uint64_t* desc_out,
                                   int lbo_elems,
                                   int sbo_elems,
                                   int base_offset) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        const uint32_t lbo_bytes = static_cast<uint32_t>(lbo_elems * sizeof(typename SmemTile::elem::storage_t));
        const uint32_t sbo_bytes = static_cast<uint32_t>(sbo_elems * sizeof(typename SmemTile::elem::storage_t));
        detail::WgmmaSwizzle swz = detail::WgmmaSwizzle::none;
        if constexpr (std::is_same_v<SwizzleAtom, axp::protocol::stage::SwizzleAtom_128B>) {
            swz = detail::WgmmaSwizzle::swizzle_128b;
        } else if constexpr (std::is_same_v<SwizzleAtom, axp::protocol::stage::SwizzleAtom_64B>) {
            swz = detail::WgmmaSwizzle::swizzle_64b;
        } else if constexpr (std::is_same_v<SwizzleAtom, axp::protocol::stage::SwizzleAtom_32B>) {
            swz = detail::WgmmaSwizzle::swizzle_32b;
        } else {
            swz = detail::WgmmaSwizzle::none;
        }
        const auto* smem_bytes = reinterpret_cast<const char*>(smem) + kPtrOffsetBytes;
        const auto* smem_ptr = reinterpret_cast<const typename SmemTile::elem::storage_t*>(smem_bytes);
        const uint32_t base = static_cast<uint32_t>(base_offset) + kBaseOffsetUnits;
        *desc_out = detail::make_wgmma_smem_desc(smem_ptr, lbo_bytes, sbo_bytes, swz, base);
#else
        (void)smem; (void)desc_out; (void)lbo_elems; (void)sbo_elems; (void)base_offset;
#endif
    }
};

template<class Recipe, class SmemTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime, class SwizzleAtom>
struct MakeWgmmaSmemDescReady
    : iro::contract::Realization<
        axp::protocol::ownership::MakeWgmmaSmemDescReady<Recipe, SmemTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.ownership.wgmma_desc_ready")> {
    static_assert(SmemTile::shape::rank == 2, "WGMMA desc requires rank-2 tile");
    static_assert(SmemTile::align::bytes >= 16, "WGMMA desc requires >=16B smem alignment");
    static_assert((SmemTile::align::bytes % 16) == 0, "WGMMA desc requires 16B-aligned smem base");
    using layout_info = detail::wgmma_layout_info<typename SmemTile::layout>;
    static_assert(layout_info::valid, "WGMMA desc requires RowMajor/ColMajor layout");
    static constexpr uint64_t kElemBytes = static_cast<uint64_t>(sizeof(typename SmemTile::elem::storage_t));
    static constexpr uint64_t kMaxUnits = 0x3FFFu;
    static constexpr uint64_t kLboBytes = static_cast<uint64_t>(layout_info::lbo_elems) * kElemBytes;
    static constexpr uint64_t kSboBytes = kLboBytes * 16u;
    static_assert((kLboBytes % 16u) == 0u, "WGMMA desc LBO must be 16B aligned");
    static_assert((kSboBytes % 16u) == 0u, "WGMMA desc SBO must be 16B aligned");
    static_assert((kLboBytes >> 4) <= kMaxUnits, "WGMMA desc LBO exceeds 14-bit field");
    static_assert((kSboBytes >> 4) <= kMaxUnits, "WGMMA desc SBO exceeds 14-bit field");

    __device__ static void execute(const typename SmemTile::elem::storage_t* smem,
                                   uint64_t* desc_out,
                                   int lbo_elems,
                                   int sbo_elems,
                                   int base_offset) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        const uint32_t lbo_bytes = static_cast<uint32_t>(lbo_elems * sizeof(typename SmemTile::elem::storage_t));
        const uint32_t sbo_bytes = static_cast<uint32_t>(sbo_elems * sizeof(typename SmemTile::elem::storage_t));
        detail::WgmmaSwizzle swz = detail::WgmmaSwizzle::none;
        if constexpr (std::is_same_v<SwizzleAtom, axp::protocol::stage::SwizzleAtom_128B>) {
            swz = detail::WgmmaSwizzle::swizzle_128b;
        } else if constexpr (std::is_same_v<SwizzleAtom, axp::protocol::stage::SwizzleAtom_64B>) {
            swz = detail::WgmmaSwizzle::swizzle_64b;
        } else if constexpr (std::is_same_v<SwizzleAtom, axp::protocol::stage::SwizzleAtom_32B>) {
            swz = detail::WgmmaSwizzle::swizzle_32b;
        } else {
            swz = detail::WgmmaSwizzle::none;
        }
        *desc_out = detail::make_wgmma_smem_desc(smem, lbo_bytes, sbo_bytes, swz, static_cast<uint32_t>(base_offset));
#else
        (void)smem; (void)desc_out; (void)lbo_elems; (void)sbo_elems; (void)base_offset;
#endif
    }
};

template<class Recipe, class SmemTile, class DescTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime,
         class SwizzleAtom, int RowOffset, int ColOffset>
struct MakeWgmmaSmemDescSliceReady
    : iro::contract::Realization<
        axp::protocol::ownership::MakeWgmmaSmemDescSliceReady<
            Recipe, SmemTile, DescTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom, RowOffset, ColOffset>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.ownership.wgmma_desc_slice_ready")> {
    static_assert(SmemTile::shape::rank == 2, "WGMMA desc requires rank-2 tile");
    static_assert(DescTile::shape::rank == 2, "WGMMA desc requires rank-2 tile");
    static_assert(SmemTile::align::bytes >= 16, "WGMMA desc requires >=16B smem alignment");
    static_assert((SmemTile::align::bytes % 16) == 0, "WGMMA desc requires 16B-aligned smem base");
    using layout_info = detail::wgmma_layout_info<typename DescTile::layout>;
    static_assert(layout_info::valid, "WGMMA desc requires RowMajor/ColMajor layout");
    static constexpr uint64_t kElemBytes = static_cast<uint64_t>(sizeof(typename SmemTile::elem::storage_t));
    static constexpr uint64_t kMaxUnits = 0x3FFFu;
    static constexpr uint64_t kLboBytes = static_cast<uint64_t>(layout_info::lbo_elems) * kElemBytes;
    static constexpr uint64_t kSboBytes = kLboBytes * 16u;
    static_assert((kLboBytes % 16u) == 0u, "WGMMA desc LBO must be 16B aligned");
    static_assert((kSboBytes % 16u) == 0u, "WGMMA desc SBO must be 16B aligned");
    static_assert((kLboBytes >> 4) <= kMaxUnits, "WGMMA desc LBO exceeds 14-bit field");
    static_assert((kSboBytes >> 4) <= kMaxUnits, "WGMMA desc SBO exceeds 14-bit field");

    using contract_t = axp::protocol::ownership::MakeWgmmaSmemDescSliceReady<
        Recipe, SmemTile, DescTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom, RowOffset, ColOffset>;
    static constexpr uint64_t kPtrOffsetBytes = contract_t::kPtrOffsetBytes;
    static constexpr uint32_t kBaseOffsetUnits = contract_t::kBaseOffsetUnits;

    __device__ static void execute(const typename SmemTile::elem::storage_t* smem,
                                   uint64_t* desc_out,
                                   int lbo_elems,
                                   int sbo_elems,
                                   int base_offset) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        const uint32_t lbo_bytes = static_cast<uint32_t>(lbo_elems * sizeof(typename SmemTile::elem::storage_t));
        const uint32_t sbo_bytes = static_cast<uint32_t>(sbo_elems * sizeof(typename SmemTile::elem::storage_t));
        detail::WgmmaSwizzle swz = detail::WgmmaSwizzle::none;
        if constexpr (std::is_same_v<SwizzleAtom, axp::protocol::stage::SwizzleAtom_128B>) {
            swz = detail::WgmmaSwizzle::swizzle_128b;
        } else if constexpr (std::is_same_v<SwizzleAtom, axp::protocol::stage::SwizzleAtom_64B>) {
            swz = detail::WgmmaSwizzle::swizzle_64b;
        } else if constexpr (std::is_same_v<SwizzleAtom, axp::protocol::stage::SwizzleAtom_32B>) {
            swz = detail::WgmmaSwizzle::swizzle_32b;
        } else {
            swz = detail::WgmmaSwizzle::none;
        }
        const auto* smem_bytes = reinterpret_cast<const char*>(smem) + kPtrOffsetBytes;
        const auto* smem_ptr = reinterpret_cast<const typename SmemTile::elem::storage_t*>(smem_bytes);
        const uint32_t base = static_cast<uint32_t>(base_offset) + kBaseOffsetUnits;
        *desc_out = detail::make_wgmma_smem_desc(smem_ptr, lbo_bytes, sbo_bytes, swz, base);
#else
        (void)smem; (void)desc_out; (void)lbo_elems; (void)sbo_elems; (void)base_offset;
#endif
    }
};

// Shared Tile <-> Fragment (warp-level WMMA fallback)
template<class Recipe, class InTile, class Frag, class InSubj, class FragSubj, class ExecGroup, class Lifetime,
         class InExtra, class OutExtra>
struct SharedTileToFragment
    : iro::contract::Realization<
        axp::protocol::ownership::SharedTileToFragment<Recipe, InTile, Frag, InSubj, FragSubj, ExecGroup, Lifetime, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.ownership.shared_tile_to_frag")> {
    __device__ static void execute(const typename InTile::elem::storage_t* smem,
                                   typename Frag::elem::storage_t* frag,
                                   int ld) {
#ifdef __CUDA_ARCH__
        static_assert(iro::schema::ElemHasStorage<typename InTile::elem>);
        static_assert(iro::schema::ElemHasStorage<typename Frag::elem>);
        static_assert(InTile::shape::rank == 2, "SharedTileToFragment requires rank-2 tile");
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
                      "SM90 SharedTileToFragment supports WMMA 16x16 f16/bf16/tf32 (warp, row/col major).");
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.ownership.frag_to_shared")> {
    __device__ static void execute(const typename Frag::elem::storage_t* frag,
                                   typename OutTile::elem::storage_t* smem,
                                   int ld) {
#ifdef __CUDA_ARCH__
        static_assert(iro::schema::ElemHasStorage<typename Frag::elem>);
        static_assert(iro::schema::ElemHasStorage<typename OutTile::elem>);
        static_assert(OutTile::shape::rank == 2, "FragmentToSharedTile requires rank-2 tile");
        if constexpr (iro::exec::is_warpgroup_v<ExecGroup> &&
                      std::is_same_v<typename Frag::elem, iro::elem::f32> &&
                      std::is_same_v<typename Frag::dist, iro::dist::accumulator> &&
                      std::is_same_v<typename OutTile::elem, iro::elem::f32> &&
                      std::is_same_v<typename OutTile::layout,
                          iro::contract::layout::RowMajor<OutTile::shape::template dim<1>()>>) {
            constexpr int kM = OutTile::shape::template dim<0>();
            constexpr int kN = OutTile::shape::template dim<1>();
            static_assert(kM == 64, "FragmentToSharedTile (warpgroup): requires M=64");
            static_assert((kN % 8) == 0, "FragmentToSharedTile (warpgroup): requires N multiple of 8");
            static_assert(Frag::count == (kN / 2),
                          "FragmentToSharedTile (warpgroup): Frag::count must be N/2");
            const int tid = threadIdx.x;
            const int warp = tid >> 5;
            const int lane = tid & 31;
            if (warp < iro::exec::warpgroup_warps<ExecGroup>::value) {
                const int row_base = warp * 16;
                const int row_in = lane >> 2; // 0..7
                const int col_pair = lane & 3; // 0..3
                #pragma unroll
                for (int v = 0; v < Frag::count; ++v) {
                    const int block = v >> 2; // each block spans 8 columns
                    const int v_in = v & 3;
                    const int row = row_base + row_in + ((v_in >> 1) * 8);
                    const int col = block * 8 + (col_pair * 2) + (v_in & 1);
                    smem[row * ld + col] = frag[v];
                }
            }
            return;
        } else if constexpr (std::is_same_v<ExecGroup, iro::exec::warp> &&
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
                      "SM90 FragmentToSharedTile supports WMMA 16x16 f16/bf16/tf32 and f32 accumulator (warp, row/col major).");
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.ownership.tile_boundary_in")> {
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.ownership.tile_boundary_out")> {
    template<class... Args>
    __device__ static void execute(Args&&...) {
#ifdef __CUDA_ARCH__
        static_assert(iro::contract::TilePayload<Tile>, "TileBoundaryOut: Tile must be TilePayload");
#endif
    }
};

// Warp MMA from shared memory tiles (SM90 supports WMMA for warp-level ops).
template<class Recipe, class Shape, class ATile, class BTile, class AccFrag, class ASubj, class BSubj, class AccSubj>
struct WarpMmaFromSmem
    : iro::contract::Realization<
        axp::protocol::compute::WarpMmaFromSmem<Recipe, Shape, ATile, BTile, AccFrag, ASubj, BSubj, AccSubj>,
        iro::util::fnv1a_64_cstr("axp.realize.sm90.compute.warp_mma")> {
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
                      "SM90 WarpMmaFromSmem requires row-major A");
        static_assert(std::is_same_v<typename BTile::layout, iro::contract::layout::ColMajor<BTile::shape::template dim<0>()>>,
                      "SM90 WarpMmaFromSmem requires col-major B");

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
                          "SM90 WarpMmaFromSmem: unsupported shape/elem");
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
        iro::util::fnv1a_64_cstr("axp.realize.sm90.compute.warp_mma_shared")> {
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
                      "SM90 WarpMmaFromShared requires row-major A");
        static_assert(std::is_same_v<typename BTile::layout, iro::contract::layout::ColMajor<BTile::shape::template dim<0>()>>,
                      "SM90 WarpMmaFromShared requires col-major B");

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
                          "SM90 WarpMmaFromShared: unsupported shape/elem");
        }
#else
        (void)a; (void)b; (void)acc; (void)lda; (void)ldb;
#endif
    }
};

} // namespace axp::realize::sm90
