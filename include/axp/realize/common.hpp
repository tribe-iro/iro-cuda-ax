#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#include "../level0/memory_cache.hpp"

namespace axp::realize::common {

namespace detail {

template<class T>
__device__ __forceinline__ uint32_t bitcast_u32(T v) {
    static_assert(sizeof(T) == 4, "bitcast_u32 expects 32-bit type");
    union {
        T t;
        uint32_t u;
    } x;
    x.t = v;
    return x.u;
}

template<class T>
__device__ __forceinline__ T bitcast_from_u32(uint32_t u) {
    static_assert(sizeof(T) == 4, "bitcast_from_u32 expects 32-bit type");
    union {
        T t;
        uint32_t u;
    } x;
    x.u = u;
    return x.t;
}

template<class VecT>
__device__ __forceinline__ VecT shfl_down_vec(VecT v, int offset, uint32_t mask) {
    const uint32_t bits = bitcast_u32(v);
    const uint32_t sh = __shfl_down_sync(mask, bits, offset);
    return bitcast_from_u32<VecT>(sh);
}

template<class VecT>
__device__ __forceinline__ VecT shfl_sync_vec(VecT v, int src_lane, uint32_t mask) {
    const uint32_t bits = bitcast_u32(v);
    const uint32_t sh = __shfl_sync(mask, bits, src_lane);
    return bitcast_from_u32<VecT>(sh);
}

template<int VecBytes>
struct vec_type;
template<> struct vec_type<4> { using type = uint32_t; };
template<> struct vec_type<8> { using type = uint2; };
template<> struct vec_type<16> { using type = uint4; };
template<int VecBytes>
using vec_type_t = typename vec_type<VecBytes>::type;

template<int VecBytes>
__device__ __forceinline__ vec_type_t<VecBytes> ld_global_ca(const vec_type_t<VecBytes>* src) {
#ifdef __CUDA_ARCH__
    vec_type_t<VecBytes> out{};
    if constexpr (VecBytes == 16) {
        asm("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
            : "l"(src));
    } else if constexpr (VecBytes == 8) {
        asm("ld.global.ca.v2.u32 {%0,%1}, [%2];"
            : "=r"(out.x), "=r"(out.y)
            : "l"(src));
    } else {
        asm("ld.global.ca.u32 %0, [%1];"
            : "=r"(out)
            : "l"(src));
    }
    return out;
#else
    (void)src;
    return vec_type_t<VecBytes>{};
#endif
}

template<int VecBytes>
__device__ __forceinline__ vec_type_t<VecBytes> ld_global_cg(const vec_type_t<VecBytes>* src) {
#ifdef __CUDA_ARCH__
    vec_type_t<VecBytes> out{};
    if constexpr (VecBytes == 16) {
        asm("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
            : "l"(src));
    } else if constexpr (VecBytes == 8) {
        asm("ld.global.cg.v2.u32 {%0,%1}, [%2];"
            : "=r"(out.x), "=r"(out.y)
            : "l"(src));
    } else {
        asm("ld.global.cg.u32 %0, [%1];"
            : "=r"(out)
            : "l"(src));
    }
    return out;
#else
    (void)src;
    return vec_type_t<VecBytes>{};
#endif
}

template<int VecBytes>
__device__ __forceinline__ vec_type_t<VecBytes> ld_global_cs(const vec_type_t<VecBytes>* src) {
#ifdef __CUDA_ARCH__
    vec_type_t<VecBytes> out{};
    if constexpr (VecBytes == 16) {
        asm("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
            : "l"(src));
    } else if constexpr (VecBytes == 8) {
        asm("ld.global.cs.v2.u32 {%0,%1}, [%2];"
            : "=r"(out.x), "=r"(out.y)
            : "l"(src));
    } else {
        asm("ld.global.cs.u32 %0, [%1];"
            : "=r"(out)
            : "l"(src));
    }
    return out;
#else
    (void)src;
    return vec_type_t<VecBytes>{};
#endif
}

template<int VecBytes>
__device__ __forceinline__ vec_type_t<VecBytes> ld_global_lu(const vec_type_t<VecBytes>* src) {
#ifdef __CUDA_ARCH__
    vec_type_t<VecBytes> out{};
    if constexpr (VecBytes == 16) {
        asm("ld.global.LU.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
            : "l"(src));
    } else if constexpr (VecBytes == 8) {
        asm("ld.global.LU.v2.u32 {%0,%1}, [%2];"
            : "=r"(out.x), "=r"(out.y)
            : "l"(src));
    } else {
        asm("ld.global.LU.u32 %0, [%1];"
            : "=r"(out)
            : "l"(src));
    }
    return out;
#else
    (void)src;
    return vec_type_t<VecBytes>{};
#endif
}

template<int VecBytes>
__device__ __forceinline__ vec_type_t<VecBytes> ld_global_cv(const vec_type_t<VecBytes>* src) {
#ifdef __CUDA_ARCH__
    vec_type_t<VecBytes> out{};
    if constexpr (VecBytes == 16) {
        asm("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
            : "l"(src));
    } else if constexpr (VecBytes == 8) {
        asm("ld.global.cv.v2.u32 {%0,%1}, [%2];"
            : "=r"(out.x), "=r"(out.y)
            : "l"(src));
    } else {
        asm("ld.global.cv.u32 %0, [%1];"
            : "=r"(out)
            : "l"(src));
    }
    return out;
#else
    (void)src;
    return vec_type_t<VecBytes>{};
#endif
}

template<class CachePolicy, int VecBytes>
__device__ __forceinline__ vec_type_t<VecBytes> ld_global_cache(const vec_type_t<VecBytes>* src) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<CachePolicy, axp::cache::cg>) {
        return ld_global_cg<VecBytes>(src);
    } else if constexpr (std::is_same_v<CachePolicy, axp::cache::cs>) {
        return ld_global_cs<VecBytes>(src);
    } else if constexpr (std::is_same_v<CachePolicy, axp::cache::lu>) {
        return ld_global_lu<VecBytes>(src);
    } else if constexpr (std::is_same_v<CachePolicy, axp::cache::cv>) {
        return ld_global_cv<VecBytes>(src);
    } else {
        return ld_global_ca<VecBytes>(src);
    }
#else
    (void)src;
    return vec_type_t<VecBytes>{};
#endif
}

template<int VecBytes>
__device__ __forceinline__ void st_global_wb(vec_type_t<VecBytes>* dst, vec_type_t<VecBytes> v) {
#ifdef __CUDA_ARCH__
    if constexpr (VecBytes == 16) {
        asm("st.global.wb.v4.u32 [%0], {%1,%2,%3,%4};"
            :
            : "l"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
    } else if constexpr (VecBytes == 8) {
        asm("st.global.wb.v2.u32 [%0], {%1,%2};"
            :
            : "l"(dst), "r"(v.x), "r"(v.y));
    } else {
        asm("st.global.wb.u32 [%0], %1;"
            :
            : "l"(dst), "r"(v));
    }
#else
    (void)dst; (void)v;
#endif
}

template<int VecBytes>
__device__ __forceinline__ void st_global_wt(vec_type_t<VecBytes>* dst, vec_type_t<VecBytes> v) {
#ifdef __CUDA_ARCH__
    if constexpr (VecBytes == 16) {
        asm("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"
            :
            : "l"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
    } else if constexpr (VecBytes == 8) {
        asm("st.global.wt.v2.u32 [%0], {%1,%2};"
            :
            : "l"(dst), "r"(v.x), "r"(v.y));
    } else {
        asm("st.global.wt.u32 [%0], %1;"
            :
            : "l"(dst), "r"(v));
    }
#else
    (void)dst; (void)v;
#endif
}

template<int VecBytes>
__device__ __forceinline__ void st_global_cs(vec_type_t<VecBytes>* dst, vec_type_t<VecBytes> v) {
#ifdef __CUDA_ARCH__
    if constexpr (VecBytes == 16) {
        asm("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
            :
            : "l"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
    } else if constexpr (VecBytes == 8) {
        asm("st.global.cs.v2.u32 [%0], {%1,%2};"
            :
            : "l"(dst), "r"(v.x), "r"(v.y));
    } else {
        asm("st.global.cs.u32 [%0], %1;"
            :
            : "l"(dst), "r"(v));
    }
#else
    (void)dst; (void)v;
#endif
}

template<class CachePolicy, int VecBytes>
__device__ __forceinline__ void st_global_cache(vec_type_t<VecBytes>* dst, vec_type_t<VecBytes> v) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<CachePolicy, axp::cache::wt>) {
        st_global_wt<VecBytes>(dst, v);
    } else if constexpr (std::is_same_v<CachePolicy, axp::cache::cs_store>) {
        st_global_cs<VecBytes>(dst, v);
    } else {
        st_global_wb<VecBytes>(dst, v);
    }
#else
    (void)dst; (void)v;
#endif
}

template<class CachePolicy>
__device__ __forceinline__ void prefetch_global(const void* ptr) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<CachePolicy, axp::cache::ca> || std::is_same_v<CachePolicy, axp::cache::lu>) {
        asm("prefetch.global.L1 [%0];" :: "l"(ptr));
    } else {
        asm("prefetch.global.L2 [%0];" :: "l"(ptr));
    }
#else
    (void)ptr;
#endif
}

template<int VecBytes>
__device__ __forceinline__ vec_type_t<VecBytes> ld_shared_vec(const vec_type_t<VecBytes>* src) {
#ifdef __CUDA_ARCH__
    vec_type_t<VecBytes> out{};
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    if constexpr (VecBytes == 16) {
        asm("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
            : "r"(addr));
    } else if constexpr (VecBytes == 8) {
        asm("ld.shared.v2.u32 {%0,%1}, [%2];"
            : "=r"(out.x), "=r"(out.y)
            : "r"(addr));
    } else {
        asm("ld.shared.u32 %0, [%1];"
            : "=r"(out)
            : "r"(addr));
    }
    return out;
#else
    (void)src;
    return vec_type_t<VecBytes>{};
#endif
}

template<int VecBytes>
__device__ __forceinline__ void st_shared_vec(vec_type_t<VecBytes>* dst, vec_type_t<VecBytes> v) {
#ifdef __CUDA_ARCH__
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    if constexpr (VecBytes == 16) {
        asm("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
            :
            : "r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
    } else if constexpr (VecBytes == 8) {
        asm("st.shared.v2.u32 [%0], {%1,%2};"
            :
            : "r"(addr), "r"(v.x), "r"(v.y));
    } else {
        asm("st.shared.u32 [%0], %1;"
            :
            : "r"(addr), "r"(v));
    }
#else
    (void)dst; (void)v;
#endif
}

template<class ExecGroup>
__device__ __forceinline__ void exec_group_stride(int& tid, int& stride) {
    if constexpr (std::is_same_v<ExecGroup, iro::exec::lane>) {
        tid = 0;
        stride = 1;
    } else if constexpr (std::is_same_v<ExecGroup, iro::exec::warp>) {
        tid = threadIdx.x & 31;
        stride = 32;
    } else if constexpr (iro::exec::is_warpgroup_v<ExecGroup>) {
        constexpr int lanes = iro::exec::warpgroup_warps<ExecGroup>::value * 32;
        tid = threadIdx.x % lanes;
        stride = lanes;
    } else {
        tid = threadIdx.x;
        stride = blockDim.x;
    }
}

template<class Recipe, class InTile, class OutTile, class ExecGroup, class CachePolicy>
__device__ __forceinline__ void tile_copy_global(const typename InTile::elem::storage_t* in,
                                                 typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
    constexpr int VecBytes = Recipe::vec_bytes;
    static_assert(VecBytes == 4 || VecBytes == 8 || VecBytes == 16);
    static_assert(InTile::align::bytes >= VecBytes, "tile_copy_global: input alignment too small");
    static_assert(OutTile::align::bytes >= VecBytes, "tile_copy_global: output alignment too small");
    constexpr int elems_per_vec = VecBytes / static_cast<int>(sizeof(typename InTile::elem::storage_t));
    static_assert((InTile::shape::size % elems_per_vec) == 0, "tile_copy_global: tile size must be vector aligned");
    int tid = 0, stride = 0;
    exec_group_stride<ExecGroup>(tid, stride);
    using VecT = vec_type_t<VecBytes>;
    for (int idx = tid * elems_per_vec; idx < InTile::shape::size; idx += stride * elems_per_vec) {
        const VecT* src = reinterpret_cast<const VecT*>(in + idx);
        VecT* dst = reinterpret_cast<VecT*>(out + idx);
        *dst = ld_global_cache<CachePolicy, VecBytes>(src);
    }
#else
    (void)in; (void)out;
#endif
}

template<class Recipe, class InTile, class ExecGroup, class CachePolicy>
__device__ __forceinline__ void tile_prefetch_global(const typename InTile::elem::storage_t* in) {
#ifdef __CUDA_ARCH__
    constexpr int VecBytes = Recipe::vec_bytes;
    static_assert(VecBytes == 4 || VecBytes == 8 || VecBytes == 16);
    static_assert(InTile::align::bytes >= VecBytes, "tile_prefetch_global: input alignment too small");
    constexpr int elems_per_vec = VecBytes / static_cast<int>(sizeof(typename InTile::elem::storage_t));
    static_assert((InTile::shape::size % elems_per_vec) == 0, "tile_prefetch_global: tile size must be vector aligned");
    int tid = 0, stride = 0;
    exec_group_stride<ExecGroup>(tid, stride);
    for (int idx = tid * elems_per_vec; idx < InTile::shape::size; idx += stride * elems_per_vec) {
        const void* addr = static_cast<const void*>(in + idx);
        prefetch_global<CachePolicy>(addr);
    }
#else
    (void)in;
#endif
}

template<class Recipe, class InTile, class OutTile, class ExecGroup, class CachePolicy>
__device__ __forceinline__ void tile_copy_store_global(const typename InTile::elem::storage_t* in,
                                                       typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
    constexpr int VecBytes = Recipe::vec_bytes;
    static_assert(VecBytes == 4 || VecBytes == 8 || VecBytes == 16);
    static_assert(InTile::align::bytes >= VecBytes, "tile_copy_store_global: input alignment too small");
    static_assert(OutTile::align::bytes >= VecBytes, "tile_copy_store_global: output alignment too small");
    constexpr int elems_per_vec = VecBytes / static_cast<int>(sizeof(typename InTile::elem::storage_t));
    static_assert((InTile::shape::size % elems_per_vec) == 0, "tile_copy_store_global: tile size must be vector aligned");
    int tid = 0, stride = 0;
    exec_group_stride<ExecGroup>(tid, stride);
    using VecT = vec_type_t<VecBytes>;
    for (int idx = tid * elems_per_vec; idx < InTile::shape::size; idx += stride * elems_per_vec) {
        const VecT* src = reinterpret_cast<const VecT*>(in + idx);
        VecT* dst = reinterpret_cast<VecT*>(out + idx);
        st_global_cache<CachePolicy, VecBytes>(dst, *src);
    }
#else
    (void)in; (void)out;
#endif
}

template<class Recipe, class InTile, class OutTile, class ExecGroup>
__device__ __forceinline__ void tile_copy_shared(const typename InTile::elem::storage_t* in,
                                                 typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
    constexpr int VecBytes = Recipe::vec_bytes;
    static_assert(VecBytes == 4 || VecBytes == 8 || VecBytes == 16);
    static_assert(InTile::align::bytes >= VecBytes, "tile_copy_shared: input alignment too small");
    static_assert(OutTile::align::bytes >= VecBytes, "tile_copy_shared: output alignment too small");
    constexpr int elems_per_vec = VecBytes / static_cast<int>(sizeof(typename InTile::elem::storage_t));
    static_assert((InTile::shape::size % elems_per_vec) == 0, "tile_copy_shared: tile size must be vector aligned");
    int tid = 0, stride = 0;
    exec_group_stride<ExecGroup>(tid, stride);
    using VecT = vec_type_t<VecBytes>;
    for (int idx = tid * elems_per_vec; idx < InTile::shape::size; idx += stride * elems_per_vec) {
        const VecT* src = reinterpret_cast<const VecT*>(in + idx);
        VecT* dst = reinterpret_cast<VecT*>(out + idx);
        *dst = ld_shared_vec<VecBytes>(src);
    }
#else
    (void)in; (void)out;
#endif
}

template<class Recipe, class InTile, class OutTile, class ExecGroup>
__device__ __forceinline__ void tile_store_shared(const typename InTile::elem::storage_t* in,
                                                  typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
    constexpr int VecBytes = Recipe::vec_bytes;
    static_assert(VecBytes == 4 || VecBytes == 8 || VecBytes == 16);
    static_assert(InTile::align::bytes >= VecBytes, "tile_store_shared: input alignment too small");
    static_assert(OutTile::align::bytes >= VecBytes, "tile_store_shared: output alignment too small");
    constexpr int elems_per_vec = VecBytes / static_cast<int>(sizeof(typename InTile::elem::storage_t));
    static_assert((InTile::shape::size % elems_per_vec) == 0, "tile_store_shared: tile size must be vector aligned");
    int tid = 0, stride = 0;
    exec_group_stride<ExecGroup>(tid, stride);
    using VecT = vec_type_t<VecBytes>;
    for (int idx = tid * elems_per_vec; idx < InTile::shape::size; idx += stride * elems_per_vec) {
        const VecT* src = reinterpret_cast<const VecT*>(in + idx);
        VecT* dst = reinterpret_cast<VecT*>(out + idx);
        st_shared_vec<VecBytes>(dst, *src);
    }
#else
    (void)in; (void)out;
#endif
}

template<class SwizzleAtom>
__device__ __forceinline__ int swizzle_mask_for_row(int r) {
    return (r >> SwizzleAtom::S_bits) & ((1 << SwizzleAtom::B_bits) - 1);
}

template<class Recipe, class InTile, class OutTile, class ExecGroup, class SwizzleAtom>
__device__ __forceinline__ void tile_copy_swizzled_to_row(const typename InTile::elem::storage_t* in,
                                                          typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
    constexpr int VecBytes = Recipe::vec_bytes;
    static_assert(VecBytes == 4 || VecBytes == 8 || VecBytes == 16);
    using storage_t = typename InTile::elem::storage_t;
    constexpr int ElemBytes = static_cast<int>(sizeof(storage_t));
    static_assert(VecBytes % ElemBytes == 0, "tile_copy_swizzled_to_row: VecBytes must divide element size");
    constexpr int VecElems = VecBytes / ElemBytes;
    constexpr int Rows = InTile::shape::template dim<0>();
    constexpr int Cols = InTile::shape::template dim<1>();
    int tid = 0, stride = 0;
    exec_group_stride<ExecGroup>(tid, stride);

    if constexpr ((Cols % VecElems) == 0 && (InTile::align::bytes >= VecBytes)) {
        constexpr int VecCols = Cols / VecElems;
        using VecT = vec_type_t<VecBytes>;
        for (int v_idx = tid; v_idx < Rows * VecCols; v_idx += stride) {
            const int r = v_idx / VecCols;
            const int sc = (v_idx - r * VecCols) * VecElems;
            const int mask = swizzle_mask_for_row<SwizzleAtom>(r);
            const VecT* src = reinterpret_cast<const VecT*>(in + r * Cols + sc);
            union {
                VecT v;
                storage_t e[VecElems];
            } pack{};
            pack.v = ld_shared_vec<VecBytes>(src);
            #pragma unroll
            for (int i = 0; i < VecElems; ++i) {
                const int c = (sc + i) ^ mask;
                const long long out_idx = OutTile::layout::offset(r, c);
                out[out_idx] = pack.e[i];
            }
        }
    } else {
        for (int idx = tid; idx < Rows * Cols; idx += stride) {
            const int r = idx / Cols;
            const int c = idx - r * Cols;
            const int mask = swizzle_mask_for_row<SwizzleAtom>(r);
            const int sc = c ^ mask;
            const long long in_idx = static_cast<long long>(r) * Cols + sc;
            const long long out_idx = OutTile::layout::offset(r, c);
            out[out_idx] = in[in_idx];
        }
    }
#else
    (void)in; (void)out;
#endif
}

template<class Recipe, class InTile, class OutTile, class ExecGroup, class SwizzleAtom>
__device__ __forceinline__ void tile_copy_row_to_swizzled(const typename InTile::elem::storage_t* in,
                                                          typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
    constexpr int VecBytes = Recipe::vec_bytes;
    static_assert(VecBytes == 4 || VecBytes == 8 || VecBytes == 16);
    using storage_t = typename InTile::elem::storage_t;
    constexpr int ElemBytes = static_cast<int>(sizeof(storage_t));
    static_assert(VecBytes % ElemBytes == 0, "tile_copy_row_to_swizzled: VecBytes must divide element size");
    constexpr int VecElems = VecBytes / ElemBytes;
    constexpr int Rows = InTile::shape::template dim<0>();
    constexpr int Cols = InTile::shape::template dim<1>();
    int tid = 0, stride = 0;
    exec_group_stride<ExecGroup>(tid, stride);

    if constexpr ((Cols % VecElems) == 0 && (OutTile::align::bytes >= VecBytes)) {
        constexpr int VecCols = Cols / VecElems;
        using VecT = vec_type_t<VecBytes>;
        for (int v_idx = tid; v_idx < Rows * VecCols; v_idx += stride) {
            const int r = v_idx / VecCols;
            const int sc = (v_idx - r * VecCols) * VecElems;
            const int mask = swizzle_mask_for_row<SwizzleAtom>(r);
            union {
                VecT v;
                storage_t e[VecElems];
            } pack{};
            #pragma unroll
            for (int i = 0; i < VecElems; ++i) {
                const int c = (sc + i) ^ mask;
                const long long in_idx = InTile::layout::offset(r, c);
                pack.e[i] = in[in_idx];
            }
            VecT* dst = reinterpret_cast<VecT*>(out + r * Cols + sc);
            st_shared_vec<VecBytes>(dst, pack.v);
        }
    } else {
        for (int idx = tid; idx < Rows * Cols; idx += stride) {
            const int r = idx / Cols;
            const int c = idx - r * Cols;
            const long long in_idx = InTile::layout::offset(r, c);
            const long long out_idx = OutTile::layout::offset(r, c);
            out[out_idx] = in[in_idx];
        }
    }
#else
    (void)in; (void)out;
#endif
}

} // namespace detail

} // namespace axp::realize::common
