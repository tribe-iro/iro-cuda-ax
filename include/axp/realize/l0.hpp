#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/realize/l0.hpp is internal substrate; use axp/l4.hpp in application code."
#endif

#include <iro_cuda_ax_core.hpp>
#include <cstdint>
#include <type_traits>
#include <limits>
#include <utility>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif
#include "../detail/type_traits.hpp"
#include "../detail/conversion.hpp"
#include "../target.hpp"
#include "../level0/compute.hpp"
#include "../level0/communication.hpp"
#include "../level0/fragment.hpp"
#include "../level0/memory.hpp"
#include "../level0/mask.hpp"
#include "../level0/scan.hpp"
#include "../level0/scale.hpp"
#include "../protocol/reduction/contracts.hpp"

namespace axp::realize::l0 {

namespace detail {

using axp::detail::is_f32;
using axp::detail::is_f16;
using axp::detail::is_bf16;
using axp::detail::is_fp8_e4m3_like;
using axp::detail::is_fp8_e5m2_like;
using axp::detail::to_f32_dispatch;
using axp::detail::from_f32_dispatch;
using axp::detail::from_f32_recipe;

template<class Payload>
struct value_traits;

template<class Elem, class Recipe>
struct acc_traits;

template<class Elem, class Recipe>
__device__ __forceinline__ typename acc_traits<Elem, Recipe>::acc_t
to_acc(typename Elem::storage_t v);

template<class Elem, class Recipe>
__device__ __forceinline__ typename Elem::storage_t
from_acc(typename acc_traits<Elem, Recipe>::acc_t v);

template<class S, class E, class D, int C>
struct value_traits<iro::contract::FragmentDesc<S, E, D, C>> {
    using elem = E;
    using storage_t = typename E::storage_t;
    static constexpr int count = C;
};

template<class E, class D>
struct value_traits<iro::contract::ScalarDesc<E, D>> {
    using elem = E;
    using storage_t = typename E::storage_t;
    static constexpr int count = 1;
};

template<class E, int N, class D>
struct value_traits<iro::contract::VectorDesc<E, N, D>> {
    using elem = E;
    using storage_t = typename E::storage_t;
    static constexpr int count = N;
};

template<int W, class D>
struct value_traits<iro::contract::MaskDesc<W, D>> {
    using elem = iro::elem::u32;
    using storage_t = typename elem::storage_t;
    static constexpr int count = iro::contract::MaskDesc<W, D>::words;
};

template<class Recipe>
__device__ __forceinline__ float expf_recipe(float x) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<typename Recipe::math, iro::recipe::ApproxExp> ||
                  std::is_same_v<typename Recipe::math, iro::recipe::Fast>) {
        return __expf(x);
    } else {
        return expf(x);
    }
#else
    (void)x;
    return 0.0f;
#endif
}

template<class Recipe>
__device__ __forceinline__ float logf_recipe(float x) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<typename Recipe::math, iro::recipe::ApproxExp> ||
                  std::is_same_v<typename Recipe::math, iro::recipe::Fast>) {
        return __logf(x);
    } else {
        return logf(x);
    }
#else
    (void)x;
    return 0.0f;
#endif
}

template<class Recipe>
__device__ __forceinline__ float tanhf_recipe(float x) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<typename Recipe::math, iro::recipe::ApproxExp> ||
                  std::is_same_v<typename Recipe::math, iro::recipe::Fast>) {
        return __tanhf(x);
    } else {
        return tanhf(x);
    }
#else
    (void)x;
    return 0.0f;
#endif
}

template<class Recipe>
__device__ __forceinline__ float divf_recipe(float a, float b) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<typename Recipe::math, iro::recipe::ApproxExp> ||
                  std::is_same_v<typename Recipe::math, iro::recipe::Fast>) {
        return __fdividef(a, b);
    } else {
        return a / b;
    }
#else
    (void)a; (void)b;
    return 0.0f;
#endif
}

template<class Recipe>
__device__ __forceinline__ float rcpf_recipe(float x) {
#ifdef __CUDA_ARCH__
    return divf_recipe<Recipe>(1.0f, x);
#else
    (void)x;
    return 0.0f;
#endif
}

template<class Recipe>
__device__ __forceinline__ float sqrtf_recipe(float x) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<typename Recipe::math, iro::recipe::ApproxExp> ||
                  std::is_same_v<typename Recipe::math, iro::recipe::Fast>) {
        return __fsqrt_rn(x);
    } else {
        return sqrtf(x);
    }
#else
    (void)x;
    return 0.0f;
#endif
}

template<class Recipe>
__device__ __forceinline__ float rsqrtf_recipe(float x) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<typename Recipe::math, iro::recipe::ApproxExp> ||
                  std::is_same_v<typename Recipe::math, iro::recipe::Fast>) {
        return rsqrtf(x);
    } else {
        return 1.0f / sqrtf(x);
    }
#else
    (void)x;
    return 0.0f;
#endif
}

template<class Recipe>
__device__ __forceinline__ float sigmoidf_recipe(float x) {
#ifdef __CUDA_ARCH__
    const float e = expf_recipe<Recipe>(-x);
    return divf_recipe<Recipe>(1.0f, 1.0f + e);
#else
    (void)x;
    return 0.0f;
#endif
}

template<class Recipe>
__device__ __forceinline__ float geluf_recipe(float x) {
#ifdef __CUDA_ARCH__
    constexpr float k0 = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float k1 = 0.044715f;
    const float x3 = x * x * x;
    const float inner = k0 * (x + k1 * x3);
    const float t = tanhf_recipe<Recipe>(inner);
    return 0.5f * x * (1.0f + t);
#else
    (void)x;
    return 0.0f;
#endif
}

template<class T>
__device__ __forceinline__ T atomic_add(T* addr, T val) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int32_t>) {
        return atomicAdd(reinterpret_cast<int*>(addr), val);
    } else if constexpr (std::is_same_v<T, unsigned int> || std::is_same_v<T, uint32_t>) {
        return atomicAdd(reinterpret_cast<unsigned int*>(addr), val);
    } else if constexpr (std::is_same_v<T, float>) {
        return atomicAdd(reinterpret_cast<float*>(addr), val);
    } else if constexpr (std::is_same_v<T, __half>) {
#if __CUDA_ARCH__ >= 600
        return atomicAdd(reinterpret_cast<__half*>(addr), val);
#else
        static_assert(axp::detail::always_false_v<T>, "atomic_add(__half) requires sm60+");
        return T{};
#endif
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
#if __CUDA_ARCH__ >= 800
        return atomicAdd(reinterpret_cast<__nv_bfloat16*>(addr), val);
#else
        static_assert(axp::detail::always_false_v<T>, "atomic_add(__nv_bfloat16) requires sm80+");
        return T{};
#endif
    } else {
        static_assert(axp::detail::always_false_v<T>, "atomic_add supports i32/u32/f32/f16/bf16 only");
        return T{};
    }
#else
    (void)addr; (void)val;
    return T{};
#endif
}

template<class T>
__device__ __forceinline__ T atomic_cas(T* addr, T compare, T val) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int32_t>) {
        return atomicCAS(reinterpret_cast<int*>(addr), compare, val);
    } else if constexpr (std::is_same_v<T, unsigned int> || std::is_same_v<T, uint32_t>) {
        return atomicCAS(reinterpret_cast<unsigned int*>(addr), compare, val);
    } else if constexpr (std::is_same_v<T, float>) {
        const int old = atomicCAS(reinterpret_cast<int*>(addr),
                                  __float_as_int(compare),
                                  __float_as_int(val));
        return __int_as_float(old);
    } else {
        static_assert(axp::detail::always_false_v<T>, "atomic_cas supports i32/u32/f32 only");
        return T{};
    }
#else
    (void)addr; (void)compare; (void)val;
    return T{};
#endif
}

__device__ __forceinline__ float atomic_min_float(float* addr, float val) {
#ifdef __CUDA_ARCH__
    int* addr_i = reinterpret_cast<int*>(addr);
    int old = *addr_i;
    int assumed = 0;
    do {
        assumed = old;
        const float old_f = __int_as_float(assumed);
        const float new_f = fminf(val, old_f);
        const int new_i = __float_as_int(new_f);
        old = atomicCAS(addr_i, assumed, new_i);
    } while (assumed != old);
    return __int_as_float(old);
#else
    (void)addr; (void)val;
    return 0.0f;
#endif
}

__device__ __forceinline__ float atomic_max_float(float* addr, float val) {
#ifdef __CUDA_ARCH__
    int* addr_i = reinterpret_cast<int*>(addr);
    int old = *addr_i;
    int assumed = 0;
    do {
        assumed = old;
        const float old_f = __int_as_float(assumed);
        const float new_f = fmaxf(val, old_f);
        const int new_i = __float_as_int(new_f);
        old = atomicCAS(addr_i, assumed, new_i);
    } while (assumed != old);
    return __int_as_float(old);
#else
    (void)addr; (void)val;
    return 0.0f;
#endif
}

__device__ __forceinline__ uint16_t half_bits(__half v) {
#ifdef __CUDA_ARCH__
    return reinterpret_cast<const __half_raw&>(v).x;
#else
    (void)v;
    return 0;
#endif
}

__device__ __forceinline__ __half half_from_bits(uint16_t x) {
#ifdef __CUDA_ARCH__
    __half_raw raw;
    raw.x = x;
    return reinterpret_cast<const __half&>(raw);
#else
    (void)x;
    return __half{};
#endif
}

__device__ __forceinline__ __half atomic_min_half(__half* addr, __half val) {
#ifdef __CUDA_ARCH__
    const uintptr_t addr_i = reinterpret_cast<uintptr_t>(addr);
    const uintptr_t base_addr = addr_i & ~uintptr_t(0x3);
    unsigned int* base = reinterpret_cast<unsigned int*>(base_addr);
    const bool upper = (addr_i & 0x2) != 0;
    unsigned int old = *base;
    unsigned int assumed = 0;
    do {
        assumed = old;
        const uint16_t old_bits = upper
            ? static_cast<uint16_t>(assumed >> 16)
            : static_cast<uint16_t>(assumed & 0xFFFFu);
        const __half old_half = half_from_bits(old_bits);
        const float old_f = __half2float(old_half);
        const float val_f = __half2float(val);
        const float new_f = fminf(val_f, old_f);
        const __half new_half = __float2half_rn(new_f);
        const uint16_t new_bits = half_bits(new_half);
        const unsigned int new_word = upper
            ? ((assumed & 0x0000FFFFu) | (static_cast<unsigned int>(new_bits) << 16))
            : ((assumed & 0xFFFF0000u) | static_cast<unsigned int>(new_bits));
        old = atomicCAS(base, assumed, new_word);
    } while (assumed != old);
    const uint16_t old_bits = upper
        ? static_cast<uint16_t>(old >> 16)
        : static_cast<uint16_t>(old & 0xFFFFu);
    return half_from_bits(old_bits);
#else
    (void)addr; (void)val;
    return __half{};
#endif
}

__device__ __forceinline__ __half atomic_max_half(__half* addr, __half val) {
#ifdef __CUDA_ARCH__
    const uintptr_t addr_i = reinterpret_cast<uintptr_t>(addr);
    const uintptr_t base_addr = addr_i & ~uintptr_t(0x3);
    unsigned int* base = reinterpret_cast<unsigned int*>(base_addr);
    const bool upper = (addr_i & 0x2) != 0;
    unsigned int old = *base;
    unsigned int assumed = 0;
    do {
        assumed = old;
        const uint16_t old_bits = upper
            ? static_cast<uint16_t>(assumed >> 16)
            : static_cast<uint16_t>(assumed & 0xFFFFu);
        const __half old_half = half_from_bits(old_bits);
        const float old_f = __half2float(old_half);
        const float val_f = __half2float(val);
        const float new_f = fmaxf(val_f, old_f);
        const __half new_half = __float2half_rn(new_f);
        const uint16_t new_bits = half_bits(new_half);
        const unsigned int new_word = upper
            ? ((assumed & 0x0000FFFFu) | (static_cast<unsigned int>(new_bits) << 16))
            : ((assumed & 0xFFFF0000u) | static_cast<unsigned int>(new_bits));
        old = atomicCAS(base, assumed, new_word);
    } while (assumed != old);
    const uint16_t old_bits = upper
        ? static_cast<uint16_t>(old >> 16)
        : static_cast<uint16_t>(old & 0xFFFFu);
    return half_from_bits(old_bits);
#else
    (void)addr; (void)val;
    return __half{};
#endif
}

template<class T>
__device__ __forceinline__ T atomic_min(T* addr, T val) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int32_t>) {
        return atomicMin(reinterpret_cast<int*>(addr), val);
    } else if constexpr (std::is_same_v<T, unsigned int> || std::is_same_v<T, uint32_t>) {
        return atomicMin(reinterpret_cast<unsigned int*>(addr), val);
    } else if constexpr (std::is_same_v<T, float>) {
        return atomic_min_float(reinterpret_cast<float*>(addr), val);
    } else if constexpr (std::is_same_v<T, __half>) {
#if __CUDA_ARCH__ >= 900
        static_assert(axp::target_cap::has_f16_atomics, "atomic_min(__half) requires cap::has_f16_atomics");
        return atomic_min_half(reinterpret_cast<__half*>(addr), val);
#else
        static_assert(axp::detail::always_false_v<T>, "atomic_min(__half) requires sm90+");
        return T{};
#endif
    } else {
        static_assert(axp::detail::always_false_v<T>, "atomic_min supports i32/u32/f32/f16 only");
        return T{};
    }
#else
    (void)addr; (void)val;
    return T{};
#endif
}

template<class T>
__device__ __forceinline__ T atomic_max(T* addr, T val) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int32_t>) {
        return atomicMax(reinterpret_cast<int*>(addr), val);
    } else if constexpr (std::is_same_v<T, unsigned int> || std::is_same_v<T, uint32_t>) {
        return atomicMax(reinterpret_cast<unsigned int*>(addr), val);
    } else if constexpr (std::is_same_v<T, float>) {
        return atomic_max_float(reinterpret_cast<float*>(addr), val);
    } else if constexpr (std::is_same_v<T, __half>) {
#if __CUDA_ARCH__ >= 900
        static_assert(axp::target_cap::has_f16_atomics, "atomic_max(__half) requires cap::has_f16_atomics");
        return atomic_max_half(reinterpret_cast<__half*>(addr), val);
#else
        static_assert(axp::detail::always_false_v<T>, "atomic_max(__half) requires sm90+");
        return T{};
#endif
    } else {
        static_assert(axp::detail::always_false_v<T>, "atomic_max supports i32/u32/f32/f16 only");
        return T{};
    }
#else
    (void)addr; (void)val;
    return T{};
#endif
}

template<class T>
__device__ __forceinline__ T atomic_and(T* addr, T val) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int32_t>) {
        return atomicAnd(reinterpret_cast<int*>(addr), val);
    } else if constexpr (std::is_same_v<T, unsigned int> || std::is_same_v<T, uint32_t>) {
        return atomicAnd(reinterpret_cast<unsigned int*>(addr), val);
    } else {
        static_assert(axp::detail::always_false_v<T>, "atomic_and supports i32/u32 only");
        return T{};
    }
#else
    (void)addr; (void)val;
    return T{};
#endif
}

template<class T>
__device__ __forceinline__ T atomic_or(T* addr, T val) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int32_t>) {
        return atomicOr(reinterpret_cast<int*>(addr), val);
    } else if constexpr (std::is_same_v<T, unsigned int> || std::is_same_v<T, uint32_t>) {
        return atomicOr(reinterpret_cast<unsigned int*>(addr), val);
    } else {
        static_assert(axp::detail::always_false_v<T>, "atomic_or supports i32/u32 only");
        return T{};
    }
#else
    (void)addr; (void)val;
    return T{};
#endif
}

template<class T>
__device__ __forceinline__ T atomic_xor(T* addr, T val) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int32_t>) {
        return atomicXor(reinterpret_cast<int*>(addr), val);
    } else if constexpr (std::is_same_v<T, unsigned int> || std::is_same_v<T, uint32_t>) {
        return atomicXor(reinterpret_cast<unsigned int*>(addr), val);
    } else {
        static_assert(axp::detail::always_false_v<T>, "atomic_xor supports i32/u32 only");
        return T{};
    }
#else
    (void)addr; (void)val;
    return T{};
#endif
}

template<class T>
__device__ __forceinline__ T add_op(T a, T b) { return a + b; }

#ifdef __CUDACC__
template<>
__device__ __forceinline__ __half add_op(__half a, __half b) { return __hadd(a, b); }
#endif

template<class T>
__device__ __forceinline__ T mul_op(T a, T b) { return a * b; }

#ifdef __CUDACC__
template<>
__device__ __forceinline__ __half mul_op(__half a, __half b) { return __hmul(a, b); }
#endif

template<class T>
__device__ __forceinline__ T fma_op(T a, T b, T c) { return a * b + c; }

#ifdef __CUDACC__
template<>
__device__ __forceinline__ float fma_op(float a, float b, float c) { return fmaf(a, b, c); }
template<>
__device__ __forceinline__ __half fma_op(__half a, __half b, __half c) { return __hfma(a, b, c); }
#endif

template<class T>
__device__ __forceinline__ T max_op(T a, T b) { return a > b ? a : b; }

template<class T>
__device__ __forceinline__ T min_op(T a, T b) { return a < b ? a : b; }

template<class T>
__device__ __forceinline__ T abs_op(T a) { return a < T(0) ? -a : a; }

template<class T>
__device__ __forceinline__ T sub_op(T a, T b) { return a - b; }

template<class T>
__device__ __forceinline__ T div_op(T a, T b) { return a / b; }

template<class T>
__device__ __forceinline__ T neg_op(T a) { return -a; }

template<class T>
__device__ __forceinline__ bool is_zero(T a) { return a == T(0); }

template<class Elem>
__device__ __forceinline__ Elem exp_elem(float x) { return from_f32_dispatch<typename Elem::storage_t>(x); }

template<class Payload, class Func>
__device__ __forceinline__ void apply_unary(const typename value_traits<Payload>::storage_t* in,
                                            typename value_traits<Payload>::storage_t* out,
                                            Func&& fn) {
#ifdef __CUDA_ARCH__
    constexpr int N = value_traits<Payload>::count;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        out[i] = fn(in[i]);
    }
#else
    (void)in; (void)out; (void)fn;
#endif
}

template<class Payload, class Func>
__device__ __forceinline__ void apply_binary(const typename value_traits<Payload>::storage_t* a,
                                             const typename value_traits<Payload>::storage_t* b,
                                             typename value_traits<Payload>::storage_t* out,
                                             Func&& fn) {
#ifdef __CUDA_ARCH__
    constexpr int N = value_traits<Payload>::count;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        out[i] = fn(a[i], b[i]);
    }
#else
    (void)a; (void)b; (void)out; (void)fn;
#endif
}

template<class Payload, class Func>
__device__ __forceinline__ void apply_fma(const typename value_traits<Payload>::storage_t* a,
                                          const typename value_traits<Payload>::storage_t* b,
                                          const typename value_traits<Payload>::storage_t* c,
                                          typename value_traits<Payload>::storage_t* out,
                                          Func&& fn) {
#ifdef __CUDA_ARCH__
    constexpr int N = value_traits<Payload>::count;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        out[i] = fn(a[i], b[i], c[i]);
    }
#else
    (void)a; (void)b; (void)c; (void)out; (void)fn;
#endif
}

template<class Payload, class Func>
__device__ __forceinline__ void apply_half2_unary(const typename value_traits<Payload>::storage_t* in,
                                                  typename value_traits<Payload>::storage_t* out,
                                                  Func&& fn) {
#ifdef __CUDA_ARCH__
    using storage_t = typename value_traits<Payload>::storage_t;
    static_assert(std::is_same_v<storage_t, __half>);
    constexpr int N = value_traits<Payload>::count;
    static_assert((N % 2) == 0);
    const __half2* in2 = reinterpret_cast<const __half2*>(in);
    __half2* out2 = reinterpret_cast<__half2*>(out);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        const float2 f = __half22float2(in2[i]);
        const float2 r = { fn(f.x), fn(f.y) };
        out2[i] = __floats2half2_rn(r.x, r.y);
    }
#else
    (void)in; (void)out; (void)fn;
#endif
}

template<class Payload, class Func>
__device__ __forceinline__ void apply_bf162_unary(const typename value_traits<Payload>::storage_t* in,
                                                  typename value_traits<Payload>::storage_t* out,
                                                  Func&& fn) {
#ifdef __CUDA_ARCH__
    using storage_t = typename value_traits<Payload>::storage_t;
    static_assert(std::is_same_v<storage_t, __nv_bfloat16>);
    constexpr int N = value_traits<Payload>::count;
    static_assert((N % 2) == 0);
    const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(in);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        const float2 f = __bfloat1622float2(in2[i]);
        const float2 r = { fn(f.x), fn(f.y) };
        out2[i] = __float22bfloat162_rn(r.x, r.y);
    }
#else
    (void)in; (void)out; (void)fn;
#endif
}

template<class Payload>
__device__ __forceinline__ void apply_bf162_unary_native(const typename value_traits<Payload>::storage_t* in,
                                                         typename value_traits<Payload>::storage_t* out,
                                                         __nv_bfloat162 (*fn)(__nv_bfloat162)) {
#ifdef __CUDA_ARCH__
    using storage_t = typename value_traits<Payload>::storage_t;
    static_assert(std::is_same_v<storage_t, __nv_bfloat16>);
    constexpr int N = value_traits<Payload>::count;
    static_assert((N % 2) == 0);
    const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(in);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        out2[i] = fn(in2[i]);
    }
#else
    (void)in; (void)out; (void)fn;
#endif
}

template<class Payload, class Func, std::enable_if_t<!std::is_pointer_v<std::decay_t<Func>>, int> = 0>
__device__ __forceinline__ void apply_bf162_unary_native(const typename value_traits<Payload>::storage_t* in,
                                                         typename value_traits<Payload>::storage_t* out,
                                                         Func&& fn) {
#ifdef __CUDA_ARCH__
    using storage_t = typename value_traits<Payload>::storage_t;
    static_assert(std::is_same_v<storage_t, __nv_bfloat16>);
    constexpr int N = value_traits<Payload>::count;
    static_assert((N % 2) == 0);
    const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(in);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        out2[i] = fn(in2[i]);
    }
#else
    (void)in; (void)out; (void)fn;
#endif
}

template<class Payload, class Func>
__device__ __forceinline__ void apply_bf162_binary(const typename value_traits<Payload>::storage_t* a,
                                                   const typename value_traits<Payload>::storage_t* b,
                                                   typename value_traits<Payload>::storage_t* out,
                                                   Func&& fn) {
#ifdef __CUDA_ARCH__
    using storage_t = typename value_traits<Payload>::storage_t;
    static_assert(std::is_same_v<storage_t, __nv_bfloat16>);
    constexpr int N = value_traits<Payload>::count;
    static_assert((N % 2) == 0);
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        const float2 fa = __bfloat1622float2(a2[i]);
        const float2 fb = __bfloat1622float2(b2[i]);
        const float2 r = { fn(fa.x, fb.x), fn(fa.y, fb.y) };
        out2[i] = __float22bfloat162_rn(r.x, r.y);
    }
#else
    (void)a; (void)b; (void)out; (void)fn;
#endif
}

template<class Payload>
__device__ __forceinline__ void apply_bf162_binary_native(const typename value_traits<Payload>::storage_t* a,
                                                          const typename value_traits<Payload>::storage_t* b,
                                                          typename value_traits<Payload>::storage_t* out,
                                                          __nv_bfloat162 (*fn)(__nv_bfloat162, __nv_bfloat162)) {
#ifdef __CUDA_ARCH__
    using storage_t = typename value_traits<Payload>::storage_t;
    static_assert(std::is_same_v<storage_t, __nv_bfloat16>);
    constexpr int N = value_traits<Payload>::count;
    static_assert((N % 2) == 0);
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        out2[i] = fn(a2[i], b2[i]);
    }
#else
    (void)a; (void)b; (void)out; (void)fn;
#endif
}

template<class Payload, class Func>
__device__ __forceinline__ void apply_bf162_fma(const typename value_traits<Payload>::storage_t* a,
                                                const typename value_traits<Payload>::storage_t* b,
                                                const typename value_traits<Payload>::storage_t* c,
                                                typename value_traits<Payload>::storage_t* out,
                                                Func&& fn) {
#ifdef __CUDA_ARCH__
    using storage_t = typename value_traits<Payload>::storage_t;
    static_assert(std::is_same_v<storage_t, __nv_bfloat16>);
    constexpr int N = value_traits<Payload>::count;
    static_assert((N % 2) == 0);
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b);
    const __nv_bfloat162* c2 = reinterpret_cast<const __nv_bfloat162*>(c);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        const float2 fa = __bfloat1622float2(a2[i]);
        const float2 fb = __bfloat1622float2(b2[i]);
        const float2 fc = __bfloat1622float2(c2[i]);
        const float2 r = { fn(fa.x, fb.x, fc.x), fn(fa.y, fb.y, fc.y) };
        out2[i] = __float22bfloat162_rn(r.x, r.y);
    }
#else
    (void)a; (void)b; (void)c; (void)out; (void)fn;
#endif
}

template<class Payload>
__device__ __forceinline__ void apply_bf162_fma_native(const typename value_traits<Payload>::storage_t* a,
                                                       const typename value_traits<Payload>::storage_t* b,
                                                       const typename value_traits<Payload>::storage_t* c,
                                                       typename value_traits<Payload>::storage_t* out,
                                                       __nv_bfloat162 (*fn)(__nv_bfloat162, __nv_bfloat162, __nv_bfloat162)) {
#ifdef __CUDA_ARCH__
    using storage_t = typename value_traits<Payload>::storage_t;
    static_assert(std::is_same_v<storage_t, __nv_bfloat16>);
    constexpr int N = value_traits<Payload>::count;
    static_assert((N % 2) == 0);
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b);
    const __nv_bfloat162* c2 = reinterpret_cast<const __nv_bfloat162*>(c);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        out2[i] = fn(a2[i], b2[i], c2[i]);
    }
#else
    (void)a; (void)b; (void)c; (void)out; (void)fn;
#endif
}

template<class Payload>
__device__ __forceinline__ void apply_half2_binary(const typename value_traits<Payload>::storage_t* a,
                                                   const typename value_traits<Payload>::storage_t* b,
                                                   typename value_traits<Payload>::storage_t* out,
                                                   __half2 (*fn)(__half2, __half2)) {
#ifdef __CUDA_ARCH__
    using storage_t = typename value_traits<Payload>::storage_t;
    static_assert(std::is_same_v<storage_t, __half>);
    constexpr int N = value_traits<Payload>::count;
    static_assert((N % 2) == 0);
    auto a2 = reinterpret_cast<const __half2*>(a);
    auto b2 = reinterpret_cast<const __half2*>(b);
    auto o2 = reinterpret_cast<__half2*>(out);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        o2[i] = fn(a2[i], b2[i]);
    }
#else
    (void)a; (void)b; (void)out; (void)fn;
#endif
}

template<class Payload, class Func, std::enable_if_t<!std::is_pointer_v<std::decay_t<Func>>, int> = 0>
__device__ __forceinline__ void apply_half2_binary(const typename value_traits<Payload>::storage_t* a,
                                                   const typename value_traits<Payload>::storage_t* b,
                                                   typename value_traits<Payload>::storage_t* out,
                                                   Func&& fn) {
#ifdef __CUDA_ARCH__
    using storage_t = typename value_traits<Payload>::storage_t;
    static_assert(std::is_same_v<storage_t, __half>);
    constexpr int N = value_traits<Payload>::count;
    static_assert((N % 2) == 0);
    const __half2* a2 = reinterpret_cast<const __half2*>(a);
    const __half2* b2 = reinterpret_cast<const __half2*>(b);
    __half2* out2 = reinterpret_cast<__half2*>(out);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        const float2 fa = __half22float2(a2[i]);
        const float2 fb = __half22float2(b2[i]);
        const float2 r = { fn(fa.x, fb.x), fn(fa.y, fb.y) };
        out2[i] = __floats2half2_rn(r.x, r.y);
    }
#else
    (void)a; (void)b; (void)out; (void)fn;
#endif
}

template<class Payload>
__device__ __forceinline__ void apply_half2_unary_native(const typename value_traits<Payload>::storage_t* in,
                                                         typename value_traits<Payload>::storage_t* out,
                                                         __half2 (*fn)(__half2)) {
#ifdef __CUDA_ARCH__
    using storage_t = typename value_traits<Payload>::storage_t;
    static_assert(std::is_same_v<storage_t, __half>);
    constexpr int N = value_traits<Payload>::count;
    static_assert((N % 2) == 0);
    const __half2* in2 = reinterpret_cast<const __half2*>(in);
    __half2* out2 = reinterpret_cast<__half2*>(out);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        out2[i] = fn(in2[i]);
    }
#else
    (void)in; (void)out; (void)fn;
#endif
}

template<class Payload, class Func, std::enable_if_t<!std::is_pointer_v<std::decay_t<Func>>, int> = 0>
__device__ __forceinline__ void apply_half2_unary_native(const typename value_traits<Payload>::storage_t* in,
                                                         typename value_traits<Payload>::storage_t* out,
                                                         Func&& fn) {
#ifdef __CUDA_ARCH__
    using storage_t = typename value_traits<Payload>::storage_t;
    static_assert(std::is_same_v<storage_t, __half>);
    constexpr int N = value_traits<Payload>::count;
    static_assert((N % 2) == 0);
    const __half2* in2 = reinterpret_cast<const __half2*>(in);
    __half2* out2 = reinterpret_cast<__half2*>(out);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        out2[i] = fn(in2[i]);
    }
#else
    (void)in; (void)out; (void)fn;
#endif
}

template<class Payload>
__device__ __forceinline__ void apply_half2_fma(const typename value_traits<Payload>::storage_t* a,
                                                const typename value_traits<Payload>::storage_t* b,
                                                const typename value_traits<Payload>::storage_t* c,
                                                typename value_traits<Payload>::storage_t* out,
                                                __half2 (*fn)(__half2, __half2, __half2)) {
#ifdef __CUDA_ARCH__
    using storage_t = typename value_traits<Payload>::storage_t;
    static_assert(std::is_same_v<storage_t, __half>);
    constexpr int N = value_traits<Payload>::count;
    static_assert((N % 2) == 0);
    auto a2 = reinterpret_cast<const __half2*>(a);
    auto b2 = reinterpret_cast<const __half2*>(b);
    auto c2 = reinterpret_cast<const __half2*>(c);
    auto o2 = reinterpret_cast<__half2*>(out);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        o2[i] = fn(a2[i], b2[i], c2[i]);
    }
#else
    (void)a; (void)b; (void)c; (void)out; (void)fn;
#endif
}

template<class Recipe, class Payload, class F32Func>
__device__ __forceinline__ void apply_unary_recipe(const typename value_traits<Payload>::storage_t* in,
                                                   typename value_traits<Payload>::storage_t* out,
                                                   F32Func&& fn) {
    using elem_t = typename value_traits<Payload>::elem;
    using acc_t = typename acc_traits<elem_t, Recipe>::acc_t;
    using storage_t = typename value_traits<Payload>::storage_t;
    constexpr int N = value_traits<Payload>::count;
    if constexpr (std::is_same_v<acc_t, storage_t>) {
        if constexpr (std::is_same_v<storage_t, float>) {
            apply_unary<Payload>(in, out, std::forward<F32Func>(fn));
        } else if constexpr (is_f16<storage_t>::value && (N % 2) == 0) {
            apply_half2_unary<Payload>(in, out, std::forward<F32Func>(fn));
        } else if constexpr (is_bf16<storage_t>::value && (N % 2) == 0) {
            apply_bf162_unary<Payload>(in, out, std::forward<F32Func>(fn));
        } else {
            apply_unary<Payload>(in, out, [=](storage_t x) {
                float fx = to_f32_dispatch(x);
                return from_acc<elem_t, Recipe>(static_cast<acc_t>(fn(fx)));
            });
        }
    } else if constexpr (is_f32<acc_t>::value) {
        if constexpr (is_f16<storage_t>::value && (N % 2) == 0) {
            apply_half2_unary<Payload>(in, out, std::forward<F32Func>(fn));
        } else if constexpr (is_bf16<storage_t>::value && (N % 2) == 0) {
            apply_bf162_unary<Payload>(in, out, std::forward<F32Func>(fn));
        } else {
            apply_unary<Payload>(in, out, [=](storage_t x) {
                float fx = to_f32_dispatch(x);
                return from_acc<elem_t, Recipe>(static_cast<acc_t>(fn(fx)));
            });
        }
    } else {
        static_assert(axp::detail::always_false_v<acc_t>,
                      "apply_unary_recipe: unsupported accumulator type");
    }
}

template<class Recipe, class Payload, class F32Func, class Half2Func, class Bf162Func>
__device__ __forceinline__ void apply_unary_recipe_native(const typename value_traits<Payload>::storage_t* in,
                                                          typename value_traits<Payload>::storage_t* out,
                                                          F32Func&& fn,
                                                          Half2Func&& fn_half2,
                                                          Bf162Func&& fn_bf162) {
    using elem_t = typename value_traits<Payload>::elem;
    using acc_t = typename acc_traits<elem_t, Recipe>::acc_t;
    using storage_t = typename value_traits<Payload>::storage_t;
    constexpr int N = value_traits<Payload>::count;
    if constexpr (std::is_same_v<acc_t, storage_t>) {
        if constexpr (is_f16<storage_t>::value && (N % 2) == 0) {
            apply_half2_unary_native<Payload>(in, out, std::forward<Half2Func>(fn_half2));
        } else if constexpr (is_bf16<storage_t>::value && (N % 2) == 0) {
            apply_bf162_unary_native<Payload>(in, out, std::forward<Bf162Func>(fn_bf162));
        } else {
            apply_unary_recipe<Recipe, Payload>(in, out, std::forward<F32Func>(fn));
        }
    } else {
        apply_unary_recipe<Recipe, Payload>(in, out, std::forward<F32Func>(fn));
    }
}

struct neg_half2_op {
    __device__ __forceinline__ __half2 operator()(__half2 v) const {
        return __hsub2(__float2half2_rn(0.0f), v);
    }
};

struct neg_bf162_op {
    __device__ __forceinline__ __nv_bfloat162 operator()(__nv_bfloat162 v) const {
        return __hsub2(__float2bfloat162_rn(0.0f), v);
    }
};

template<class Recipe, class Payload, class F32Func>
__device__ __forceinline__ void apply_binary_recipe(const typename value_traits<Payload>::storage_t* a,
                                                    const typename value_traits<Payload>::storage_t* b,
                                                    typename value_traits<Payload>::storage_t* out,
                                                    F32Func&& fn) {
    using elem_t = typename value_traits<Payload>::elem;
    using acc_t = typename acc_traits<elem_t, Recipe>::acc_t;
    using storage_t = typename value_traits<Payload>::storage_t;
    constexpr int N = value_traits<Payload>::count;
    if constexpr (std::is_same_v<acc_t, storage_t>) {
        if constexpr (std::is_same_v<storage_t, float>) {
            apply_binary<Payload>(a, b, out, std::forward<F32Func>(fn));
        } else if constexpr (is_f16<storage_t>::value && (N % 2) == 0) {
            apply_half2_binary<Payload>(a, b, out, std::forward<F32Func>(fn));
        } else if constexpr (is_bf16<storage_t>::value && (N % 2) == 0) {
            apply_bf162_binary<Payload>(a, b, out, std::forward<F32Func>(fn));
        } else {
            apply_binary<Payload>(a, b, out, [=](storage_t x, storage_t y) {
                float fx = to_f32_dispatch(x);
                float fy = to_f32_dispatch(y);
                return from_acc<elem_t, Recipe>(static_cast<acc_t>(fn(fx, fy)));
            });
        }
    } else if constexpr (is_f32<acc_t>::value) {
        if constexpr (is_f16<storage_t>::value && (N % 2) == 0) {
            apply_half2_binary<Payload>(a, b, out, std::forward<F32Func>(fn));
        } else if constexpr (is_bf16<storage_t>::value && (N % 2) == 0) {
            apply_bf162_binary<Payload>(a, b, out, std::forward<F32Func>(fn));
        } else {
            apply_binary<Payload>(a, b, out, [=](storage_t x, storage_t y) {
                float fx = to_f32_dispatch(x);
                float fy = to_f32_dispatch(y);
                return from_acc<elem_t, Recipe>(static_cast<acc_t>(fn(fx, fy)));
            });
        }
    } else {
        static_assert(axp::detail::always_false_v<acc_t>,
                      "apply_binary_recipe: unsupported accumulator type");
    }
}

template<class Recipe, class Payload, class F32Func, class Half2Func, class Bf162Func>
__device__ __forceinline__ void apply_binary_recipe_native(const typename value_traits<Payload>::storage_t* a,
                                                           const typename value_traits<Payload>::storage_t* b,
                                                           typename value_traits<Payload>::storage_t* out,
                                                           F32Func&& fn,
                                                           Half2Func&& fn_half2,
                                                           Bf162Func&& fn_bf162) {
    using elem_t = typename value_traits<Payload>::elem;
    using acc_t = typename acc_traits<elem_t, Recipe>::acc_t;
    using storage_t = typename value_traits<Payload>::storage_t;
    constexpr int N = value_traits<Payload>::count;
    if constexpr (std::is_same_v<acc_t, storage_t>) {
        if constexpr (is_f16<storage_t>::value && (N % 2) == 0) {
            apply_half2_binary<Payload>(a, b, out, std::forward<Half2Func>(fn_half2));
        } else if constexpr (is_bf16<storage_t>::value && (N % 2) == 0) {
            apply_bf162_binary_native<Payload>(a, b, out, std::forward<Bf162Func>(fn_bf162));
        } else {
            apply_binary_recipe<Recipe, Payload>(a, b, out, std::forward<F32Func>(fn));
        }
    } else {
        apply_binary_recipe<Recipe, Payload>(a, b, out, std::forward<F32Func>(fn));
    }
}

template<class Recipe, class Payload>
__device__ __forceinline__ void apply_fma_recipe(const typename value_traits<Payload>::storage_t* a,
                                                 const typename value_traits<Payload>::storage_t* b,
                                                 const typename value_traits<Payload>::storage_t* c,
                                                 typename value_traits<Payload>::storage_t* out) {
    using elem_t = typename value_traits<Payload>::elem;
    using acc_t = typename acc_traits<elem_t, Recipe>::acc_t;
    using storage_t = typename value_traits<Payload>::storage_t;
    constexpr int N = value_traits<Payload>::count;
    if constexpr (std::is_same_v<acc_t, storage_t>) {
        if constexpr (is_f16<storage_t>::value && (N % 2) == 0) {
            apply_half2_fma<Payload>(a, b, c, out, __hfma2);
        } else if constexpr (is_bf16<storage_t>::value && (N % 2) == 0) {
            apply_bf162_fma_native<Payload>(a, b, c, out, __hfma2);
        } else {
            apply_fma<Payload>(a, b, c, out, [](storage_t x, storage_t y, storage_t z) {
                return fma_op(x, y, z);
            });
        }
    } else if constexpr (is_f32<acc_t>::value) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const acc_t fa = to_acc<elem_t, Recipe>(a[i]);
            const acc_t fb = to_acc<elem_t, Recipe>(b[i]);
            const acc_t fc = to_acc<elem_t, Recipe>(c[i]);
            const acc_t r = static_cast<acc_t>(fa * fb + fc);
            out[i] = from_acc<elem_t, Recipe>(r);
        }
    } else {
        static_assert(axp::detail::always_false_v<acc_t>,
                      "apply_fma_recipe: unsupported accumulator type");
    }
}

template<class Payload>
__device__ __forceinline__ void mask_select(const typename value_traits<Payload>::storage_t* a,
                                            const typename value_traits<Payload>::storage_t* b,
                                            const uint32_t* mask_words,
                                            typename value_traits<Payload>::storage_t* out) {
#ifdef __CUDA_ARCH__
    constexpr int N = value_traits<Payload>::count;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        const uint32_t word = mask_words[i >> 5];
        const uint32_t bit = 1u << (i & 31);
        out[i] = (word & bit) ? a[i] : b[i];
    }
#else
    (void)a; (void)b; (void)mask_words; (void)out;
#endif
}

template<class T>
__device__ __forceinline__ uint32_t pack_u32(T v) {
    static_assert(sizeof(T) <= 4, "pack_u32 expects <= 4-byte types");
    union {
        T t;
        uint32_t u;
        uint16_t u16;
        uint8_t u8;
    } x;
    x.u = 0;
    x.t = v;
    return x.u;
}

template<class T>
__device__ __forceinline__ T unpack_u32(uint32_t u) {
    static_assert(sizeof(T) <= 4, "unpack_u32 expects <= 4-byte types");
    union {
        T t;
        uint32_t u;
        uint16_t u16;
        uint8_t u8;
    } x;
    x.u = u;
    return x.t;
}

template<class T>
__device__ __forceinline__ uint64_t pack_u64(T v) {
    static_assert(sizeof(T) == 8, "pack_u64 expects 8-byte types");
    union {
        T t;
        uint64_t u;
        uint32_t u32[2];
    } x;
    x.u = 0;
    x.t = v;
    return x.u;
}

template<class T>
__device__ __forceinline__ T unpack_u64(uint64_t u) {
    static_assert(sizeof(T) == 8, "unpack_u64 expects 8-byte types");
    union {
        T t;
        uint64_t u;
        uint32_t u32[2];
    } x;
    x.u = u;
    return x.t;
}

template<class Mode>
__device__ __forceinline__ uint32_t shfl_u32(uint32_t v, int delta) {
#ifdef __CUDA_ARCH__
    const uint32_t mask = __activemask();
    if constexpr (std::is_same_v<Mode, axp::level0::shuffle::down>) {
        return __shfl_down_sync(mask, v, delta);
    } else if constexpr (std::is_same_v<Mode, axp::level0::shuffle::up>) {
        return __shfl_up_sync(mask, v, delta);
    } else {
        return __shfl_xor_sync(mask, v, delta);
    }
#else
    (void)delta;
    return v;
#endif
}

template<class Mode>
__device__ __forceinline__ uint32_t shfl_u32_mask(uint32_t v, int delta, uint32_t mask) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<Mode, axp::level0::shuffle::down>) {
        return __shfl_down_sync(mask, v, delta);
    } else if constexpr (std::is_same_v<Mode, axp::level0::shuffle::up>) {
        return __shfl_up_sync(mask, v, delta);
    } else {
        return __shfl_xor_sync(mask, v, delta);
    }
#else
    (void)delta; (void)mask;
    return v;
#endif
}

template<class T, class Mode>
__device__ __forceinline__ T shfl_typed(T v, int delta) {
#ifdef __CUDA_ARCH__
    if constexpr (sizeof(T) <= 4) {
        uint32_t u = pack_u32(v);
        uint32_t r = shfl_u32<Mode>(u, delta);
        return unpack_u32<T>(r);
    } else {
        uint64_t u = pack_u64(v);
        uint32_t lo = static_cast<uint32_t>(u);
        uint32_t hi = static_cast<uint32_t>(u >> 32);
        lo = shfl_u32<Mode>(lo, delta);
        hi = shfl_u32<Mode>(hi, delta);
        uint64_t r = (static_cast<uint64_t>(hi) << 32) | lo;
        return unpack_u64<T>(r);
    }
#else
    (void)delta;
    return v;
#endif
}

template<class T, class Mode>
__device__ __forceinline__ T shfl_typed_mask(T v, int delta, uint32_t mask) {
#ifdef __CUDA_ARCH__
    if constexpr (sizeof(T) <= 4) {
        uint32_t u = pack_u32(v);
        uint32_t r = shfl_u32_mask<Mode>(u, delta, mask);
        return unpack_u32<T>(r);
    } else {
        uint64_t u = pack_u64(v);
        uint32_t lo = static_cast<uint32_t>(u);
        uint32_t hi = static_cast<uint32_t>(u >> 32);
        lo = shfl_u32_mask<Mode>(lo, delta, mask);
        hi = shfl_u32_mask<Mode>(hi, delta, mask);
        uint64_t r = (static_cast<uint64_t>(hi) << 32) | lo;
        return unpack_u64<T>(r);
    }
#else
    (void)delta; (void)mask;
    return v;
#endif
}

template<class T>
__device__ __forceinline__ T shfl_down_typed(T v, int delta, uint32_t mask) {
    return shfl_typed_mask<T, axp::level0::shuffle::down>(v, delta, mask);
}

template<int BaseId, int Warps>
__device__ __forceinline__ void warpgroup_sync() {
#ifdef __CUDA_ARCH__
    static_assert(BaseId >= 1 && BaseId <= 8, "warpgroup_sync BaseId must be 1..8");
    static_assert(Warps >= 4, "warpgroup_sync requires Warps >= 4");
    constexpr int threads = Warps * 32;
    const int warpgroup_id = threadIdx.x / threads;
    switch (warpgroup_id) {
        case 0: asm volatile("bar.sync %0, %1;" : : "n"(BaseId + 0), "n"(threads) : "memory"); break;
        case 1: asm volatile("bar.sync %0, %1;" : : "n"(BaseId + 1), "n"(threads) : "memory"); break;
        case 2: asm volatile("bar.sync %0, %1;" : : "n"(BaseId + 2), "n"(threads) : "memory"); break;
        case 3: asm volatile("bar.sync %0, %1;" : : "n"(BaseId + 3), "n"(threads) : "memory"); break;
        case 4: asm volatile("bar.sync %0, %1;" : : "n"(BaseId + 4), "n"(threads) : "memory"); break;
        case 5: asm volatile("bar.sync %0, %1;" : : "n"(BaseId + 5), "n"(threads) : "memory"); break;
        case 6: asm volatile("bar.sync %0, %1;" : : "n"(BaseId + 6), "n"(threads) : "memory"); break;
        case 7: asm volatile("bar.sync %0, %1;" : : "n"(BaseId + 7), "n"(threads) : "memory"); break;
        default: asm volatile("bar.sync %0, %1;" : : "n"(BaseId + 0), "n"(threads) : "memory"); break;
    }
#endif
}

template<class T>
__device__ __forceinline__ T shfl_idx_typed(T v, int src_lane, uint32_t mask) {
#ifdef __CUDA_ARCH__
    if constexpr (sizeof(T) <= 4) {
        uint32_t u = pack_u32(v);
        uint32_t r = __shfl_sync(mask, u, src_lane);
        return unpack_u32<T>(r);
    } else {
        uint64_t u = pack_u64(v);
        uint32_t lo = static_cast<uint32_t>(u);
        uint32_t hi = static_cast<uint32_t>(u >> 32);
        lo = __shfl_sync(mask, lo, src_lane);
        hi = __shfl_sync(mask, hi, src_lane);
        uint64_t r = (static_cast<uint64_t>(hi) << 32) | lo;
        return unpack_u64<T>(r);
    }
#else
    (void)src_lane; (void)mask;
    return v;
#endif
}

template<class T>
__device__ __forceinline__ bool is_nonzero(T v) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_integral_v<T>) {
        return v != 0;
        } else if constexpr (detail::is_f32<T>::value || detail::is_f16<T>::value ||
                         detail::is_bf16<T>::value || detail::is_fp8_e4m3_like<T>::value ||
                         detail::is_fp8_e5m2_like<T>::value) {
        return detail::to_f32_dispatch(v) != 0.0f;
    } else {
        return v != T(0);
    }
#else
    (void)v;
    return false;
#endif
}

template<class OpTag, class Elem, class Recipe>
__device__ __forceinline__ typename Elem::storage_t op_apply(typename Elem::storage_t a,
                                                             typename Elem::storage_t b) {
    using T = typename Elem::storage_t;
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
        if constexpr (std::is_integral_v<T>) {
            return a + b;
        } else if constexpr (axp::detail::is_f32<T>::value) {
            return a + b;
        } else {
            float fa = to_f32_dispatch(a);
            float fb = to_f32_dispatch(b);
            return from_f32_recipe<Elem, Recipe>(fa + fb);
        }
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
        if constexpr (std::is_integral_v<T>) {
            return a * b;
        } else if constexpr (axp::detail::is_f32<T>::value) {
            return a * b;
        } else {
            float fa = to_f32_dispatch(a);
            float fb = to_f32_dispatch(b);
            return from_f32_recipe<Elem, Recipe>(fa * fb);
        }
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
        if constexpr (std::is_integral_v<T>) {
            return a > b ? a : b;
        } else if constexpr (axp::detail::is_f32<T>::value) {
            return a > b ? a : b;
        } else {
            float fa = to_f32_dispatch(a);
            float fb = to_f32_dispatch(b);
            return from_f32_recipe<Elem, Recipe>(fa > fb ? fa : fb);
        }
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
        if constexpr (std::is_integral_v<T>) {
            return a < b ? a : b;
        } else if constexpr (axp::detail::is_f32<T>::value) {
            return a < b ? a : b;
        } else {
            float fa = to_f32_dispatch(a);
            float fb = to_f32_dispatch(b);
            return from_f32_recipe<Elem, Recipe>(fa < fb ? fa : fb);
        }
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_and>) {
        static_assert(std::is_integral_v<T>, "op_and requires integral type");
        return a & b;
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_or>) {
        static_assert(std::is_integral_v<T>, "op_or requires integral type");
        return a | b;
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_xor>) {
        static_assert(std::is_integral_v<T>, "op_xor requires integral type");
        return a ^ b;
    } else {
        static_assert(axp::detail::always_false_v<OpTag>, "Unsupported scan OpTag");
    }
#else
    (void)a; (void)b;
    return T{};
#endif
}

template<class OpTag, class VecT>
__device__ __forceinline__ VecT op_apply_vec(VecT a, VecT b) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
        return __hadd2(a, b);
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
        return __hmul2(a, b);
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
        return __hmax2(a, b);
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
        return __hmin2(a, b);
    } else {
        static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for vector op");
        return VecT{};
    }
#else
    (void)a; (void)b;
    return VecT{};
#endif
}

template<class OpTag, class VecT>
__device__ __forceinline__ VecT op_identity_vec() {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<VecT, __half2>) {
        if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
            return __floats2half2_rn(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
            return __floats2half2_rn(1.0f, 1.0f);
        } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
            return __floats2half2_rn(-CUDART_INF_F, -CUDART_INF_F);
        } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
            return __floats2half2_rn(CUDART_INF_F, CUDART_INF_F);
        } else {
            static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for vector identity");
            return __floats2half2_rn(0.0f, 0.0f);
        }
    } else if constexpr (std::is_same_v<VecT, __nv_bfloat162>) {
        if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
            return __float22bfloat162_rn(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
            return __float22bfloat162_rn(1.0f, 1.0f);
        } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
            return __float22bfloat162_rn(-CUDART_INF_F, -CUDART_INF_F);
        } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
            return __float22bfloat162_rn(CUDART_INF_F, CUDART_INF_F);
        } else {
            static_assert(axp::detail::always_false_v<OpTag>, "Unsupported OpTag for vector identity");
            return __float22bfloat162_rn(0.0f, 0.0f);
        }
    } else {
        static_assert(axp::detail::always_false_v<VecT>, "op_identity_vec requires half2 or bfloat162");
        return VecT{};
    }
#else
    return VecT{};
#endif
}

template<class OpTag, class Elem, class Recipe>
__device__ __forceinline__ typename Elem::storage_t op_identity() {
    using T = typename Elem::storage_t;
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
        if constexpr (std::is_integral_v<T>) {
            return T(0);
        } else if constexpr (axp::detail::is_f32<T>::value) {
            return 0.0f;
        } else {
            return from_f32_recipe<Elem, Recipe>(0.0f);
        }
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
        if constexpr (std::is_integral_v<T>) {
            return T(1);
        } else if constexpr (axp::detail::is_f32<T>::value) {
            return 1.0f;
        } else {
            return from_f32_recipe<Elem, Recipe>(1.0f);
        }
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
        if constexpr (std::is_integral_v<T>) {
            return std::numeric_limits<T>::lowest();
        } else if constexpr (axp::detail::is_f32<T>::value) {
            return -CUDART_INF_F;
        } else {
            return from_f32_recipe<Elem, Recipe>(-CUDART_INF_F);
        }
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
        if constexpr (std::is_integral_v<T>) {
            return std::numeric_limits<T>::max();
        } else if constexpr (axp::detail::is_f32<T>::value) {
            return CUDART_INF_F;
        } else {
            return from_f32_recipe<Elem, Recipe>(CUDART_INF_F);
        }
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_and>) {
        static_assert(std::is_integral_v<T>, "op_and requires integral type");
        return static_cast<T>(~T(0));
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_or> ||
                         std::is_same_v<OpTag, axp::protocol::reduction::op_xor>) {
        static_assert(std::is_integral_v<T>, "op_or/op_xor requires integral type");
        return T(0);
    } else {
        static_assert(axp::detail::always_false_v<OpTag>, "Unsupported scan OpTag");
    }
#else
    return T{};
#endif
}

template<class Elem, class Recipe>
struct acc_traits {
    using acc_elem = typename Recipe::acc;
    using acc_t = typename acc_elem::storage_t;
};

template<class Elem, class Recipe>
__device__ __forceinline__ typename acc_traits<Elem, Recipe>::acc_t
to_acc(typename Elem::storage_t v) {
    using acc_t = typename acc_traits<Elem, Recipe>::acc_t;
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<acc_t, typename Elem::storage_t>) {
        return v;
    } else if constexpr (axp::detail::is_f32<acc_t>::value) {
        return detail::to_f32_dispatch(v);
    } else if constexpr (std::is_integral_v<acc_t>) {
        return static_cast<acc_t>(v);
    } else {
        static_assert(axp::detail::always_false_v<acc_t>, "Unsupported accumulator type");
        return acc_t{};
    }
#else
    (void)v;
    return acc_t{};
#endif
}

template<class Elem, class Recipe>
__device__ __forceinline__ typename Elem::storage_t
from_acc(typename acc_traits<Elem, Recipe>::acc_t v) {
    using acc_t = typename acc_traits<Elem, Recipe>::acc_t;
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<acc_t, typename Elem::storage_t>) {
        return v;
    } else if constexpr (axp::detail::is_f32<acc_t>::value) {
        return detail::from_f32_recipe<Elem, Recipe>(static_cast<float>(v));
    } else if constexpr (std::is_integral_v<acc_t>) {
        return static_cast<typename Elem::storage_t>(v);
    } else {
        static_assert(axp::detail::always_false_v<acc_t>, "Unsupported accumulator type");
        return typename Elem::storage_t{};
    }
#else
    (void)v;
    return typename Elem::storage_t{};
#endif
}

template<class OpTag, class AccT>
__device__ __forceinline__ AccT op_apply_acc(AccT a, AccT b) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
        return a + b;
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
        return a * b;
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
        return a > b ? a : b;
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
        return a < b ? a : b;
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_and>) {
        static_assert(std::is_integral_v<AccT>, "op_and requires integral accumulator");
        return a & b;
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_or>) {
        static_assert(std::is_integral_v<AccT>, "op_or requires integral accumulator");
        return a | b;
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_xor>) {
        static_assert(std::is_integral_v<AccT>, "op_xor requires integral accumulator");
        return a ^ b;
    } else {
        static_assert(axp::detail::always_false_v<OpTag>, "Unsupported reduction OpTag");
        return AccT{};
    }
#else
    (void)a; (void)b;
    return AccT{};
#endif
}

template<class OpTag, class AccT>
__device__ __forceinline__ AccT op_identity_acc() {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
        return AccT(0);
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_mul>) {
        return AccT(1);
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
        if constexpr (axp::detail::is_f32<AccT>::value) {
            return static_cast<AccT>(-CUDART_INF_F);
        } else if constexpr (axp::detail::is_f16<AccT>::value || axp::detail::is_bf16<AccT>::value) {
            return axp::detail::from_f32_dispatch<AccT>(-CUDART_INF_F);
        } else {
            return std::numeric_limits<AccT>::lowest();
        }
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
        if constexpr (axp::detail::is_f32<AccT>::value) {
            return static_cast<AccT>(CUDART_INF_F);
        } else if constexpr (axp::detail::is_f16<AccT>::value || axp::detail::is_bf16<AccT>::value) {
            return axp::detail::from_f32_dispatch<AccT>(CUDART_INF_F);
        } else {
            return std::numeric_limits<AccT>::max();
        }
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_and>) {
        static_assert(std::is_integral_v<AccT>, "op_and requires integral accumulator");
        return ~AccT(0);
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_or>) {
        static_assert(std::is_integral_v<AccT>, "op_or requires integral accumulator");
        return AccT(0);
    } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_xor>) {
        static_assert(std::is_integral_v<AccT>, "op_xor requires integral accumulator");
        return AccT(0);
    } else {
        static_assert(axp::detail::always_false_v<OpTag>, "Unsupported reduction OpTag");
        return AccT{};
    }
#else
    return AccT{};
#endif
}

template<int VecBytes>
struct vec_type;

template<>
struct vec_type<4> { using type = uint32_t; };

template<>
struct vec_type<8> { using type = uint2; };

template<>
struct vec_type<16> { using type = uint4; };

template<int VecBytes>
using vec_type_t = typename vec_type<VecBytes>::type;

template<class VecT>
__device__ __forceinline__ VecT ldg_vec(const VecT* src) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<VecT, uint4>) {
        union { int4 i; VecT u; } conv;
        conv.i = __ldg(reinterpret_cast<const int4*>(src));
        return conv.u;
    } else if constexpr (std::is_same_v<VecT, uint2>) {
        union { int2 i; VecT u; } conv;
        conv.i = __ldg(reinterpret_cast<const int2*>(src));
        return conv.u;
    } else if constexpr (std::is_same_v<VecT, uint32_t>) {
        return __ldg(reinterpret_cast<const uint32_t*>(src));
    } else {
        return *src;
    }
#else
    (void)src;
    return VecT{};
#endif
}

template<class ExecGroup>
__device__ __forceinline__ void exec_group_lane_stride(int& tid, int& stride) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<ExecGroup, iro::exec::warp>) {
        tid = threadIdx.x & 31;
        stride = 32;
    } else if constexpr (iro::exec::is_warpgroup_v<ExecGroup>) {
        constexpr int lanes = iro::exec::warpgroup_warps<ExecGroup>::value * 32;
        tid = threadIdx.x % lanes;
        stride = lanes;
    } else if constexpr (std::is_same_v<ExecGroup, iro::exec::block>) {
        tid = threadIdx.x;
        stride = blockDim.x;
    } else {
        tid = 0;
        stride = 1;
    }
#else
    (void)tid; (void)stride;
#endif
}

template<class Layout>
struct is_row_major : std::false_type {};
template<int Cols>
struct is_row_major<iro::contract::layout::RowMajor<Cols>> : std::true_type {};

template<class Layout>
struct is_col_major : std::false_type {};
template<int Rows>
struct is_col_major<iro::contract::layout::ColMajor<Rows>> : std::true_type {};

template<class Recipe, class InTile, class OutTile, class ExecGroup>
__device__ __forceinline__ void tile_copy(const typename InTile::elem::storage_t* in,
                                          typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
    constexpr int VecBytes = Recipe::vec_bytes;
    static_assert(VecBytes == 4 || VecBytes == 8 || VecBytes == 16);
    static_assert(InTile::align::bytes >= VecBytes, "tile_copy: input alignment too small");
    static_assert(OutTile::align::bytes >= VecBytes, "tile_copy: output alignment too small");
    constexpr int elems_per_vec = VecBytes / static_cast<int>(sizeof(typename InTile::elem::storage_t));
    static_assert(elems_per_vec > 0, "tile_copy: invalid vector width");
    static_assert((InTile::shape::size % elems_per_vec) == 0, "tile_copy: tile size must be vector aligned");

    int tid = 0;
    int stride = 1;
    detail::exec_group_lane_stride<ExecGroup>(tid, stride);

    using VecT = vec_type_t<VecBytes>;
    for (int idx = tid * elems_per_vec; idx < InTile::shape::size; idx += stride * elems_per_vec) {
        const VecT* src = reinterpret_cast<const VecT*>(in + idx);
        VecT* dst = reinterpret_cast<VecT*>(out + idx);
        *dst = *src;
    }
#else
    (void)in; (void)out;
#endif
}

template<class Recipe, class Tile, class ScaleTile, class ExecGroup>
__device__ __forceinline__ void scale_shared_tile(const typename Tile::elem::storage_t* in,
                                                  const typename ScaleTile::elem::storage_t* scale,
                                                  typename Tile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
    constexpr int VecBytes = Recipe::vec_bytes;
    static_assert(VecBytes == 4 || VecBytes == 8 || VecBytes == 16);
    static_assert(Tile::align::bytes >= VecBytes, "ScaleSharedTile: tile alignment too small");
    constexpr int elems_per_vec = VecBytes / static_cast<int>(sizeof(typename Tile::elem::storage_t));
    static_assert(elems_per_vec > 0, "ScaleSharedTile: invalid vector width");
    static_assert((Tile::shape::size % elems_per_vec) == 0, "ScaleSharedTile: tile size must be vector aligned");
    constexpr int scale_vec = ScaleTile::shape::size;
    static_assert(scale_vec > 0, "ScaleSharedTile: scale vector must be positive");

    constexpr int k_dim = []() {
        if constexpr (Tile::shape::rank == 1) {
            return Tile::shape::size;
        } else if constexpr (is_row_major<typename Tile::layout>::value) {
            return Tile::shape::template dim<1>();
        } else if constexpr (is_col_major<typename Tile::layout>::value) {
            return Tile::shape::template dim<0>();
        } else {
            static_assert(axp::detail::always_false_v<typename Tile::layout>,
                          "ScaleSharedTile: layout must be RowMajor or ColMajor");
            return 1;
        }
    }();
    static_assert(k_dim > 0, "ScaleSharedTile: invalid K dimension");
    static_assert((k_dim % scale_vec) == 0, "ScaleSharedTile: scale vector must divide K dimension");
    constexpr int scale_stride = k_dim / scale_vec;

    int tid = 0;
    int stride = 1;
    detail::exec_group_lane_stride<ExecGroup>(tid, stride);

    using VecT = vec_type_t<VecBytes>;
    union Pack {
        VecT v;
        typename Tile::elem::storage_t e[elems_per_vec];
    };

    for (int idx = tid * elems_per_vec; idx < Tile::shape::size; idx += stride * elems_per_vec) {
        Pack p;
        p.v = *reinterpret_cast<const VecT*>(in + idx);
        #pragma unroll
        for (int i = 0; i < elems_per_vec; ++i) {
            const int k = (idx + i) % k_dim;
            const int s = k / scale_stride;
            const float scale_f = detail::to_f32_dispatch(scale[s]);
            const float val_f = detail::to_f32_dispatch(p.e[i]);
            p.e[i] = detail::from_f32_recipe<typename Tile::elem, Recipe>(val_f * scale_f);
        }
        *reinterpret_cast<VecT*>(out + idx) = p.v;
    }
#else
    (void)in; (void)scale; (void)out;
#endif
}

template<class Recipe, class InTile, class OutTile, class ExecGroup>
__device__ __forceinline__ void tile_copy_global(const typename InTile::elem::storage_t* in,
                                                 typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
    constexpr int VecBytes = Recipe::vec_bytes;
    static_assert(VecBytes == 4 || VecBytes == 8 || VecBytes == 16);
    static_assert(InTile::align::bytes >= VecBytes, "tile_copy_global: input alignment too small");
    static_assert(OutTile::align::bytes >= VecBytes, "tile_copy_global: output alignment too small");
    constexpr int elems_per_vec = VecBytes / static_cast<int>(sizeof(typename InTile::elem::storage_t));
    static_assert(elems_per_vec > 0, "tile_copy_global: invalid vector width");
    static_assert((InTile::shape::size % elems_per_vec) == 0, "tile_copy_global: tile size must be vector aligned");

    int tid = 0;
    int stride = 1;
    detail::exec_group_lane_stride<ExecGroup>(tid, stride);

    using VecT = vec_type_t<VecBytes>;
    for (int idx = tid * elems_per_vec; idx < InTile::shape::size; idx += stride * elems_per_vec) {
        const VecT* src = reinterpret_cast<const VecT*>(in + idx);
        VecT* dst = reinterpret_cast<VecT*>(out + idx);
        *dst = ldg_vec(src);
    }
#else
    (void)in; (void)out;
#endif
}

template<class InTile, class OutTile, class ExecGroup>
__device__ __forceinline__ void tile_copy_swizzled_to_row(const typename InTile::elem::storage_t* in,
                                                          typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
    static_assert(InTile::shape::rank == 2, "tile_copy_swizzled_to_row: rank-2 only");
    static_assert(OutTile::shape::rank == 2, "tile_copy_swizzled_to_row: rank-2 only");
    constexpr int Rows = InTile::shape::template dim<0>();
    constexpr int Cols = InTile::shape::template dim<1>();
    int tid = 0;
    int stride = 1;
    if constexpr (std::is_same_v<ExecGroup, iro::exec::lane>) {
        tid = 0; stride = 1;
    } else {
        detail::exec_group_lane_stride<ExecGroup>(tid, stride);
    }
    #pragma unroll
    for (int idx = tid; idx < Rows * Cols; idx += stride) {
        const int r = idx / Cols;
        const int c = idx - r * Cols;
        const long long in_idx = InTile::layout::offset(r, c);
        const long long out_idx = OutTile::layout::offset(r, c);
        out[out_idx] = in[in_idx];
    }
#else
    (void)in; (void)out;
#endif
}

template<class InTile, class OutTile, class ExecGroup>
__device__ __forceinline__ void tile_copy_row_to_swizzled(const typename InTile::elem::storage_t* in,
                                                          typename OutTile::elem::storage_t* out) {
#ifdef __CUDA_ARCH__
    static_assert(InTile::shape::rank == 2, "tile_copy_row_to_swizzled: rank-2 only");
    static_assert(OutTile::shape::rank == 2, "tile_copy_row_to_swizzled: rank-2 only");
    constexpr int Rows = InTile::shape::template dim<0>();
    constexpr int Cols = InTile::shape::template dim<1>();
    int tid = 0;
    int stride = 1;
    if constexpr (std::is_same_v<ExecGroup, iro::exec::lane>) {
        tid = 0; stride = 1;
    } else {
        detail::exec_group_lane_stride<ExecGroup>(tid, stride);
    }
    #pragma unroll
    for (int idx = tid; idx < Rows * Cols; idx += stride) {
        const int r = idx / Cols;
        const int c = idx - r * Cols;
        const long long in_idx = InTile::layout::offset(r, c);
        const long long out_idx = OutTile::layout::offset(r, c);
        out[out_idx] = in[in_idx];
    }
#else
    (void)in; (void)out;
#endif
}

} // namespace detail

// -----------------------------------------------------------------------------
// L0 Unary ops
// -----------------------------------------------------------------------------

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Alias
    : iro::contract::Realization<
        axp::level0::Alias<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.alias")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
        detail::apply_unary<Payload>(in, out, [](storage_t x) { return x; });
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Exp
    : iro::contract::Realization<
        axp::level0::Exp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.exp")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
        detail::apply_unary_recipe<Recipe, Payload>(in, out, [](float x) {
            return detail::expf_recipe<Recipe>(x);
        });
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Log
    : iro::contract::Realization<
        axp::level0::Log<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.log")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
        detail::apply_unary_recipe<Recipe, Payload>(in, out, [](float x) {
            return detail::logf_recipe<Recipe>(x);
        });
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Tanh
    : iro::contract::Realization<
        axp::level0::Tanh<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.tanh")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
        detail::apply_unary_recipe<Recipe, Payload>(in, out, [](float x) {
            return detail::tanhf_recipe<Recipe>(x);
        });
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Rsqrt
    : iro::contract::Realization<
        axp::level0::Rsqrt<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.rsqrt")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
        detail::apply_unary_recipe<Recipe, Payload>(in, out, [](float x) {
            return detail::rsqrtf_recipe<Recipe>(x);
        });
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Abs
    : iro::contract::Realization<
        axp::level0::Abs<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.abs")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
        detail::apply_unary_recipe_native<Recipe, Payload>(
            in, out,
            [](float x) { return fabsf(x); },
            __habs2,
            __habs2
        );
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Neg
    : iro::contract::Realization<
        axp::level0::Neg<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.neg")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
        detail::apply_unary_recipe_native<Recipe, Payload>(
            in, out,
            [](float x) { return -x; },
            detail::neg_half2_op{},
            detail::neg_bf162_op{}
        );
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Rcp
    : iro::contract::Realization<
        axp::level0::Rcp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.rcp")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
        detail::apply_unary_recipe<Recipe, Payload>(in, out, [](float x) {
            return detail::rcpf_recipe<Recipe>(x);
        });
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Sqrt
    : iro::contract::Realization<
        axp::level0::Sqrt<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.sqrt")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
        detail::apply_unary_recipe<Recipe, Payload>(in, out, [](float x) {
            return detail::sqrtf_recipe<Recipe>(x);
        });
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Sigmoid
    : iro::contract::Realization<
        axp::level0::Sigmoid<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.sigmoid")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
        detail::apply_unary_recipe<Recipe, Payload>(in, out, [](float x) {
            return detail::sigmoidf_recipe<Recipe>(x);
        });
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct SiLU
    : iro::contract::Realization<
        axp::level0::SiLU<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.silu")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
        detail::apply_unary_recipe<Recipe, Payload>(in, out, [](float x) {
            return x * detail::sigmoidf_recipe<Recipe>(x);
        });
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Gelu
    : iro::contract::Realization<
        axp::level0::Gelu<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.gelu")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
        detail::apply_unary_recipe<Recipe, Payload>(in, out, [](float x) {
            return detail::geluf_recipe<Recipe>(x);
        });
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Popc
    : iro::contract::Realization<
        axp::level0::Popc<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.popc")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<storage_t, uint32_t>, "Popc requires u32 storage");
        constexpr int N = detail::value_traits<Payload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            out[i] = static_cast<storage_t>(__popc(static_cast<unsigned int>(in[i])));
        }
#else
        (void)in; (void)out;
#endif
    }
};

// -----------------------------------------------------------------------------
// L0 Binary ops
// -----------------------------------------------------------------------------

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Add
    : iro::contract::Realization<
        axp::level0::Add<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.add")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* a, const storage_t* b, storage_t* out) {
        detail::apply_binary_recipe_native<Recipe, Payload>(
            a, b, out,
            [](float x, float y) { return x + y; },
            __hadd2,
            __hadd2
        );
    }
};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Sub
    : iro::contract::Realization<
        axp::level0::Sub<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.sub")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* a, const storage_t* b, storage_t* out) {
        detail::apply_binary_recipe_native<Recipe, Payload>(
            a, b, out,
            [](float x, float y) { return x - y; },
            __hsub2,
            __hsub2
        );
    }
};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Mul
    : iro::contract::Realization<
        axp::level0::Mul<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.mul")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* a, const storage_t* b, storage_t* out) {
        detail::apply_binary_recipe_native<Recipe, Payload>(
            a, b, out,
            [](float x, float y) { return x * y; },
            __hmul2,
            __hmul2
        );
    }
};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Div
    : iro::contract::Realization<
        axp::level0::Div<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.div")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* a, const storage_t* b, storage_t* out) {
        detail::apply_binary_recipe<Recipe, Payload>(a, b, out, [](float x, float y) {
            return detail::divf_recipe<Recipe>(x, y);
        });
    }
};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Max
    : iro::contract::Realization<
        axp::level0::Max<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.max")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* a, const storage_t* b, storage_t* out) {
        detail::apply_binary_recipe_native<Recipe, Payload>(
            a, b, out,
            [](float x, float y) { return fmaxf(x, y); },
            __hmax2,
            __hmax2
        );
    }
};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Min
    : iro::contract::Realization<
        axp::level0::Min<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.min")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* a, const storage_t* b, storage_t* out) {
        detail::apply_binary_recipe_native<Recipe, Payload>(
            a, b, out,
            [](float x, float y) { return fminf(x, y); },
            __hmin2,
            __hmin2
        );
    }
};

template<class Recipe, class Payload, class InSubj, class MinSubj, class MaxSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Clamp
    : iro::contract::Realization<
        axp::level0::Clamp<Recipe, Payload, InSubj, MinSubj, MaxSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.clamp")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in,
                                   const storage_t* minv,
                                   const storage_t* maxv,
                                   storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<Payload>::count;
        const storage_t min0 = minv[0];
        const storage_t max0 = maxv[0];
        if constexpr (detail::is_f16<storage_t>::value && (N % 2) == 0) {
            const __half2* in2 = reinterpret_cast<const __half2*>(in);
            const __half2* min2 = reinterpret_cast<const __half2*>(minv);
            const __half2* max2 = reinterpret_cast<const __half2*>(maxv);
            __half2* out2 = reinterpret_cast<__half2*>(out);
            const __half2 min_b = __half2half2(reinterpret_cast<const __half&>(min0));
            const __half2 max_b = __half2half2(reinterpret_cast<const __half&>(max0));
            #pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                const __half2 lo = (N == 1) ? min_b : min2[i];
                const __half2 hi = (N == 1) ? max_b : max2[i];
                out2[i] = __hmin2(__hmax2(in2[i], lo), hi);
            }
        } else if constexpr (detail::is_bf16<storage_t>::value && (N % 2) == 0) {
            const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(in);
            const __nv_bfloat162* min2 = reinterpret_cast<const __nv_bfloat162*>(minv);
            const __nv_bfloat162* max2 = reinterpret_cast<const __nv_bfloat162*>(maxv);
            __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
            const __nv_bfloat162 min_b = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16&>(min0));
            const __nv_bfloat162 max_b = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16&>(max0));
            #pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                const __nv_bfloat162 lo = (N == 1) ? min_b : min2[i];
                const __nv_bfloat162 hi = (N == 1) ? max_b : max2[i];
                out2[i] = __hmin2(__hmax2(in2[i], lo), hi);
            }
        } else {
            detail::apply_unary<Payload>(in, out, [=](storage_t x) {
                const storage_t lo = min0;
                const storage_t hi = max0;
                return detail::min_op(detail::max_op(x, lo), hi);
            });
        }
#else
        (void)in; (void)minv; (void)maxv; (void)out;
#endif
    }
};

// -----------------------------------------------------------------------------
// L0 FMA
// -----------------------------------------------------------------------------

template<class Recipe, class Payload, class ASubj, class BSubj, class CSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct Fma
    : iro::contract::Realization<
        axp::level0::Fma<Recipe, Payload, ASubj, BSubj, CSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.fma")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* a, const storage_t* b, const storage_t* c, storage_t* out) {
        detail::apply_fma_recipe<Recipe, Payload>(a, b, c, out);
    }
};

// -----------------------------------------------------------------------------
// L0 Fragment-specific ops (scale, clamp, permute)
// -----------------------------------------------------------------------------

template<class Recipe, class FragPayload, class ScalarPayload,
         class FragSubj, class ScalarSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct FragmentScale
    : iro::contract::Realization<
        axp::level0::FragmentScale<Recipe, FragPayload, ScalarPayload, FragSubj, ScalarSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.fragment_scale")> {
    using storage_t = typename detail::value_traits<FragPayload>::storage_t;
    using scalar_t = typename detail::value_traits<ScalarPayload>::storage_t;
    __device__ static void execute(const storage_t* in, const scalar_t* scalar, storage_t* out) {
#ifdef __CUDA_ARCH__
        const storage_t s = static_cast<storage_t>(scalar[0]);
        constexpr int N = detail::value_traits<FragPayload>::count;
        if constexpr (detail::is_f16<storage_t>::value && (N % 2) == 0) {
            const __half2* in2 = reinterpret_cast<const __half2*>(in);
            __half2* out2 = reinterpret_cast<__half2*>(out);
            const __half2 s2 = __half2half2(reinterpret_cast<const __half&>(s));
            #pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                out2[i] = __hmul2(in2[i], s2);
            }
        } else if constexpr (detail::is_bf16<storage_t>::value && (N % 2) == 0) {
            const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(in);
            __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
            const __nv_bfloat162 s2 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16&>(s));
            #pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                out2[i] = __hmul2(in2[i], s2);
            }
        } else {
            detail::apply_unary<FragPayload>(in, out, [=](storage_t x) { return detail::mul_op(x, s); });
        }
#else
        (void)in; (void)scalar; (void)out;
#endif
    }
};

template<class Recipe, class FragPayload, class MinPayload, class MaxPayload,
         class FragSubj, class MinSubj, class MaxSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct FragmentClamp
    : iro::contract::Realization<
        axp::level0::FragmentClamp<Recipe, FragPayload, MinPayload, MaxPayload, FragSubj, MinSubj, MaxSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.fragment_clamp")> {
    using storage_t = typename detail::value_traits<FragPayload>::storage_t;
    using scalar_t = typename detail::value_traits<MinPayload>::storage_t;
    __device__ static void execute(const storage_t* in, const scalar_t* minv,
                                   const scalar_t* maxv, storage_t* out) {
#ifdef __CUDA_ARCH__
        const storage_t lo = static_cast<storage_t>(minv[0]);
        const storage_t hi = static_cast<storage_t>(maxv[0]);
        constexpr int N = detail::value_traits<FragPayload>::count;
        if constexpr (detail::is_f16<storage_t>::value && (N % 2) == 0) {
            const __half2* in2 = reinterpret_cast<const __half2*>(in);
            __half2* out2 = reinterpret_cast<__half2*>(out);
            const __half2 lo2 = __half2half2(reinterpret_cast<const __half&>(lo));
            const __half2 hi2 = __half2half2(reinterpret_cast<const __half&>(hi));
            #pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                out2[i] = __hmin2(__hmax2(in2[i], lo2), hi2);
            }
        } else if constexpr (detail::is_bf16<storage_t>::value && (N % 2) == 0) {
            const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(in);
            __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
            const __nv_bfloat162 lo2 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16&>(lo));
            const __nv_bfloat162 hi2 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16&>(hi));
            #pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                out2[i] = __hmin2(__hmax2(in2[i], lo2), hi2);
            }
        } else {
            detail::apply_unary<FragPayload>(in, out, [=](storage_t x) {
                return detail::min_op(detail::max_op(x, lo), hi);
            });
        }
#else
        (void)in; (void)minv; (void)maxv; (void)out;
#endif
    }
};

template<class Recipe, class FragPayload, class ScalarPayload,
         class FragSubj, class OutSubj, class ExecGroup, class OpTag,
         class InExtra, class OutExtra>
struct FragmentReduce
    : iro::contract::Realization<
        axp::level0::FragmentReduce<Recipe, FragPayload, ScalarPayload, FragSubj, OutSubj, ExecGroup, OpTag, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.fragment_reduce")> {
    using storage_t = typename detail::value_traits<FragPayload>::storage_t;
    using scalar_t = typename detail::value_traits<ScalarPayload>::storage_t;
    using elem_t = typename detail::value_traits<FragPayload>::elem;
    __device__ static void execute(const storage_t* in, scalar_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<FragPayload>::count;
        using acc_t = typename detail::acc_traits<elem_t, Recipe>::acc_t;
        acc_t acc = detail::op_identity_acc<OpTag, acc_t>();
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            acc = detail::op_apply_acc<OpTag>(acc, detail::to_acc<elem_t, Recipe>(in[i]));
        }
        out[0] = static_cast<scalar_t>(detail::from_acc<elem_t, Recipe>(acc));
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class FragPayload, class ScalarPayload,
         class FragSubj, class OutSubj, class ExecGroup, class OpTag,
         class InExtra, class OutExtra>
struct FragmentReduceAcc
    : iro::contract::Realization<
        axp::level0::FragmentReduceAcc<Recipe, FragPayload, ScalarPayload, FragSubj, OutSubj, ExecGroup, OpTag, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.fragment_reduce_acc")> {
    using storage_t = typename detail::value_traits<FragPayload>::storage_t;
    using scalar_t = typename detail::value_traits<ScalarPayload>::storage_t;
    using elem_t = typename detail::value_traits<FragPayload>::elem;
    __device__ static void execute(const storage_t* in, scalar_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<FragPayload>::count;
        using acc_t = typename detail::acc_traits<elem_t, Recipe>::acc_t;
        constexpr bool vec_op_ok =
            std::is_same_v<OpTag, axp::protocol::reduction::op_add> ||
            std::is_same_v<OpTag, axp::protocol::reduction::op_mul> ||
            std::is_same_v<OpTag, axp::protocol::reduction::op_max> ||
            std::is_same_v<OpTag, axp::protocol::reduction::op_min>;
        constexpr bool vec_ok =
            (detail::is_f16<storage_t>::value || detail::is_bf16<storage_t>::value) &&
            (N % 2) == 0 &&
            std::is_same_v<acc_t, storage_t> &&
            vec_op_ok;
        if constexpr (vec_ok) {
            using vec_t = std::conditional_t<detail::is_f16<storage_t>::value, __half2, __nv_bfloat162>;
            constexpr int N2 = N / 2;
            const vec_t* in2 = reinterpret_cast<const vec_t*>(in);
            vec_t acc = detail::op_identity_vec<OpTag, vec_t>();
            #pragma unroll
            for (int i = 0; i < N2; ++i) {
                acc = detail::op_apply_vec<OpTag>(acc, in2[i]);
            }
            if constexpr (detail::is_f16<storage_t>::value) {
                const __half lo = __low2half(acc);
                const __half hi = __high2half(acc);
                const acc_t a = static_cast<acc_t>(lo);
                const acc_t b = static_cast<acc_t>(hi);
                out[0] = detail::op_apply_acc<OpTag>(a, b);
            } else {
                const __nv_bfloat16* parts = reinterpret_cast<const __nv_bfloat16*>(&acc);
                const acc_t a = static_cast<acc_t>(parts[0]);
                const acc_t b = static_cast<acc_t>(parts[1]);
                out[0] = detail::op_apply_acc<OpTag>(a, b);
            }
        } else {
            acc_t acc = detail::op_identity_acc<OpTag, acc_t>();
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                acc = detail::op_apply_acc<OpTag>(acc, detail::to_acc<elem_t, Recipe>(in[i]));
            }
            out[0] = static_cast<scalar_t>(acc);
        }
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class FragPayload, class ScalarPayload,
         class FragSubj, class OutSubj, class ExecGroup, class OpTag,
         class InExtra, class OutExtra>
struct FragmentReduceAccVec
    : iro::contract::Realization<
        axp::level0::FragmentReduceAccVec<Recipe, FragPayload, ScalarPayload, FragSubj, OutSubj, ExecGroup, OpTag, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.fragment_reduce_acc_vec")> {
    using base = FragmentReduceAcc<Recipe, FragPayload, ScalarPayload, FragSubj, OutSubj, ExecGroup, OpTag, InExtra, OutExtra>;
    using storage_t = typename base::storage_t;
    using scalar_t = typename base::scalar_t;
    __device__ static void execute(const storage_t* in, scalar_t* out) {
        base::execute(in, out);
    }
};

template<class Recipe, class FragPayload, class InSubj, class OutSubj, class ExecGroup, class Pattern,
         class InExtra, class OutExtra>
struct FragmentPermute
    : iro::contract::Realization<
        axp::level0::FragmentPermute<Recipe, FragPayload, InSubj, OutSubj, ExecGroup, Pattern, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.fragment_permute")> {
    using storage_t = typename detail::value_traits<FragPayload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<FragPayload>::count;
        storage_t tmp[N];
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            tmp[i] = in[i];
        }
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const int j = Pattern::map(i, N);
            out[i] = tmp[j];
        }
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class InFrag, class OutFrag,
         class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct FragmentTranspose
    : iro::contract::Realization<
        axp::level0::FragmentTranspose<Recipe, InFrag, OutFrag, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.fragment_transpose")> {
    using in_t = typename detail::value_traits<InFrag>::storage_t;
    using out_t = typename detail::value_traits<OutFrag>::storage_t;
    __device__ static void execute(const in_t* in, out_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int M = InFrag::shape::template dim<0>();
        constexpr int N = InFrag::shape::template dim<1>();
        #pragma unroll
        for (int i = 0; i < M; ++i) {
            #pragma unroll
            for (int j = 0; j < N; ++j) {
                const int in_idx = i * N + j;
                const int out_idx = j * M + i;
                out[out_idx] = static_cast<out_t>(in[in_idx]);
            }
        }
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class FragPayload, class ScalarPayload,
         class FragSubj, class OutSubj, class ExecGroup, int Index,
         class InExtra, class OutExtra>
struct FragmentExtract
    : iro::contract::Realization<
        axp::level0::FragmentExtract<Recipe, FragPayload, ScalarPayload, FragSubj, OutSubj, ExecGroup, Index, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.fragment_extract")> {
    using frag_t = typename detail::value_traits<FragPayload>::storage_t;
    using scalar_t = typename detail::value_traits<ScalarPayload>::storage_t;
    __device__ static void execute(const frag_t* in, scalar_t* out) {
#ifdef __CUDA_ARCH__
        out[0] = static_cast<scalar_t>(in[Index]);
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class FragPayload, class ScalarPayload,
         class FragSubj, class ScalarSubj, class OutSubj, class ExecGroup, int Index,
         class InExtra, class OutExtra>
struct FragmentInsert
    : iro::contract::Realization<
        axp::level0::FragmentInsert<Recipe, FragPayload, ScalarPayload, FragSubj, ScalarSubj, OutSubj, ExecGroup, Index, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.fragment_insert")> {
    using frag_t = typename detail::value_traits<FragPayload>::storage_t;
    using scalar_t = typename detail::value_traits<ScalarPayload>::storage_t;
    __device__ static void execute(const frag_t* in, const scalar_t* val, frag_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<FragPayload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            out[i] = in[i];
        }
        out[Index] = static_cast<frag_t>(val[0]);
#else
        (void)in; (void)val; (void)out;
#endif
    }
};

template<class Recipe, class InFrag, class OutFrag,
         class InSubj, class OutSubj, class ExecGroup, int Start, int Count,
         class InExtra, class OutExtra>
struct FragmentSlice
    : iro::contract::Realization<
        axp::level0::FragmentSlice<Recipe, InFrag, OutFrag, InSubj, OutSubj, ExecGroup, Start, Count, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.fragment_slice")> {
    using in_t = typename detail::value_traits<InFrag>::storage_t;
    using out_t = typename detail::value_traits<OutFrag>::storage_t;
    __device__ static void execute(const in_t* in, out_t* out) {
#ifdef __CUDA_ARCH__
        #pragma unroll
        for (int i = 0; i < Count; ++i) {
            out[i] = static_cast<out_t>(in[Start + i]);
        }
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class FragPayload, class ScalarPayload,
         class ScalarSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct FragmentBroadcast
    : iro::contract::Realization<
        axp::level0::FragmentBroadcast<Recipe, FragPayload, ScalarPayload, ScalarSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.fragment_broadcast")> {
    using frag_t = typename detail::value_traits<FragPayload>::storage_t;
    using scalar_t = typename detail::value_traits<ScalarPayload>::storage_t;
    __device__ static void execute(const scalar_t* in, frag_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<FragPayload>::count;
        const frag_t v = static_cast<frag_t>(in[0]);
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            out[i] = v;
        }
#else
        (void)in; (void)out;
#endif
    }
};

// -----------------------------------------------------------------------------
// L0 Select (mask)
// -----------------------------------------------------------------------------

template<class Recipe, class Payload, class MaskPayload, class ASubj, class BSubj, class MaskSubj, class OutSubj,
         class ExecGroup, class InExtra, class OutExtra>
struct Select
    : iro::contract::Realization<
        axp::level0::Select<Recipe, Payload, MaskPayload, ASubj, BSubj, MaskSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.select")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* a, const storage_t* b, const uint32_t* mask_words, storage_t* out) {
        detail::mask_select<Payload>(a, b, mask_words, out);
    }
};

// ---------------------------------------------------------------------------
// L0 Scalar const
// ---------------------------------------------------------------------------

template<class Recipe, class ScalarPayload, class OutSubj, class ExecGroup, class Pattern, class OutExtra>
struct ScalarConst
    : iro::contract::Realization<
        axp::level0::ScalarConst<Recipe, ScalarPayload, OutSubj, ExecGroup, Pattern, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.scalar_const")> {
    using storage_t = typename detail::value_traits<ScalarPayload>::storage_t;
    __device__ static void execute(storage_t* out) {
#ifdef __CUDA_ARCH__
        out[0] = static_cast<storage_t>(Pattern::value);
#else
        (void)out;
#endif
    }
};

// -----------------------------------------------------------------------------
// L0 Mask ops
// -----------------------------------------------------------------------------

template<class Recipe, class MaskPayload, class OutSubj, class ExecGroup, class Pattern, class OutExtra>
struct MaskConst
    : iro::contract::Realization<
        axp::level0::MaskConst<Recipe, MaskPayload, OutSubj, ExecGroup, Pattern, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.mask_const")> {
    using storage_t = typename detail::value_traits<MaskPayload>::storage_t;
    __device__ static void execute(storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int words = detail::value_traits<MaskPayload>::count;
        #pragma unroll
        for (int i = 0; i < words; ++i) {
            out[i] = static_cast<storage_t>(Pattern::word(i));
        }
#else
        (void)out;
#endif
    }
};

template<class Recipe, class MaskPayload, class PredPayload,
         class QCoordPayload, class KCoordPayload,
         class QCoordSubj, class KCoordSubj,
         class MaskSubj, class PredSubj,
         class ExecGroup, int TileM, int TileN>
struct CausalMaskPred
    : iro::contract::Realization<
        axp::level0::CausalMaskPred<
            Recipe, MaskPayload, PredPayload,
            QCoordPayload, KCoordPayload,
            QCoordSubj, KCoordSubj,
            MaskSubj, PredSubj,
            ExecGroup, TileM, TileN>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.causal_mask_pred")> {
    using mask_t = typename detail::value_traits<MaskPayload>::storage_t;
    using pred_t = typename detail::value_traits<PredPayload>::storage_t;
    using q_t = typename detail::value_traits<QCoordPayload>::storage_t;
    using k_t = typename detail::value_traits<KCoordPayload>::storage_t;

    __device__ static void execute(const q_t* q_coord, const k_t* k_coord,
                                   mask_t* mask_out, pred_t* pred_out) {
#ifdef __CUDA_ARCH__
        constexpr int mask_bits = MaskPayload::width;
        constexpr int mask_words = detail::value_traits<MaskPayload>::count;
        #pragma unroll
        for (int i = 0; i < mask_words; ++i) mask_out[i] = 0u;

        int lane = 0;
        if constexpr (std::is_same_v<ExecGroup, iro::exec::warp>) {
            lane = threadIdx.x & 31;
        } else if constexpr (iro::exec::is_warpgroup_v<ExecGroup>) {
            constexpr int warps = iro::exec::warpgroup_warps<ExecGroup>::value;
            constexpr int lanes = warps * 32;
            lane = threadIdx.x % lanes;
            static_assert(lanes % TileM == 0, "CausalMaskPred: TileM must divide warpgroup lanes");
        }

        constexpr int lane_groups = [] {
            if constexpr (std::is_same_v<ExecGroup, iro::exec::warp>) {
                return 32 / TileM;
            } else {
                return (iro::exec::warpgroup_warps<ExecGroup>::value * 32) / TileM;
            }
        }();
        static_assert((lane_groups * mask_bits) >= TileN,
                      "CausalMaskPred: mask width insufficient for TileN coverage");

        const int row = lane % TileM;
        const int col_base = (lane / TileM) * mask_bits;
        const int q_base = static_cast<int>(q_coord[0]);
        const int k_base = static_cast<int>(k_coord[0]);

        #pragma unroll
        for (int i = 0; i < mask_bits; ++i) {
            const int col = col_base + i;
            const int valid = (col < TileN) && ((k_base + col) <= (q_base + row));
            const int word = i >> 5;
            const int bit = i & 31;
            mask_out[word] |= (static_cast<uint32_t>(valid) << bit);
        }

        const int skip = (k_base > (q_base + TileM - 1));
        pred_out[0] = static_cast<pred_t>(skip ? 1 : 0);
#else
        (void)q_coord; (void)k_coord; (void)mask_out; (void)pred_out;
#endif
    }
};

template<class Recipe, class MaskPayload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct MaskNot
    : iro::contract::Realization<
        axp::level0::MaskNot<Recipe, MaskPayload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.mask_not")> {
    using storage_t = typename detail::value_traits<MaskPayload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<MaskPayload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) out[i] = ~in[i];
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class MaskPayload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct MaskAnd
    : iro::contract::Realization<
        axp::level0::MaskAnd<Recipe, MaskPayload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.mask_and")> {
    using storage_t = typename detail::value_traits<MaskPayload>::storage_t;
    __device__ static void execute(const storage_t* a, const storage_t* b, storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<MaskPayload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) out[i] = a[i] & b[i];
#else
        (void)a; (void)b; (void)out;
#endif
    }
};

template<class Recipe, class MaskPayload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct MaskOr
    : iro::contract::Realization<
        axp::level0::MaskOr<Recipe, MaskPayload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.mask_or")> {
    using storage_t = typename detail::value_traits<MaskPayload>::storage_t;
    __device__ static void execute(const storage_t* a, const storage_t* b, storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<MaskPayload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) out[i] = a[i] | b[i];
#else
        (void)a; (void)b; (void)out;
#endif
    }
};

template<class Recipe, class MaskPayload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct MaskXor
    : iro::contract::Realization<
        axp::level0::MaskXor<Recipe, MaskPayload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.mask_xor")> {
    using storage_t = typename detail::value_traits<MaskPayload>::storage_t;
    __device__ static void execute(const storage_t* a, const storage_t* b, storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<MaskPayload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) out[i] = a[i] ^ b[i];
#else
        (void)a; (void)b; (void)out;
#endif
    }
};

// -----------------------------------------------------------------------------
// L0 Shuffle/Vote
// -----------------------------------------------------------------------------

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class Mode, int Delta, class InExtra, class OutExtra>
struct Shuffle
    : iro::contract::Realization<
        axp::level0::Shuffle<Recipe, Payload, InSubj, OutSubj, ExecGroup, Mode, Delta, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.shuffle")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<Payload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            out[i] = detail::shfl_typed<storage_t, Mode>(in[i], Delta);
        }
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class Payload, class MaskPayload, class InSubj, class MaskSubj, class OutSubj, class ExecGroup,
         class Mode, int Delta, class InExtra, class OutExtra>
struct ShuffleSync
    : iro::contract::Realization<
        axp::level0::ShuffleSync<Recipe, Payload, MaskPayload, InSubj, MaskSubj, OutSubj, ExecGroup, Mode, Delta, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.shuffle_sync")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    using mask_t = typename detail::value_traits<MaskPayload>::storage_t;
    __device__ static void execute(const storage_t* in, const mask_t* mask_in, storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<Payload>::count;
        const uint32_t mask = static_cast<uint32_t>(mask_in[0]);
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            out[i] = detail::shfl_typed_mask<storage_t, Mode>(in[i], Delta, mask);
        }
#else
        (void)in; (void)mask_in; (void)out;
#endif
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, int K, int J,
         class InExtra, class OutExtra>
struct WarpBitonicStep
    : iro::contract::Realization<
        axp::level0::WarpBitonicStep<Recipe, Payload, InSubj, OutSubj, ExecGroup, K, J, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.warp_bitonic_step")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    using elem_t = typename detail::value_traits<Payload>::elem;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        const int lane = threadIdx.x & 31;
        const storage_t v = in[0];
        const storage_t other = detail::shfl_typed<storage_t, axp::level0::shuffle::xor_>(v, J);
        const bool ascending = ((lane & K) == 0);
        const bool jbit_zero = ((lane & J) == 0);
        const bool select_min = ascending ? jbit_zero : !jbit_zero;
        const storage_t minv = detail::op_apply<axp::protocol::reduction::op_min, elem_t, Recipe>(v, other);
        const storage_t maxv = detail::op_apply<axp::protocol::reduction::op_max, elem_t, Recipe>(v, other);
        out[0] = select_min ? minv : maxv;
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct WarpReverseSecondHalf
    : iro::contract::Realization<
        axp::level0::WarpReverseSecondHalf<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.warp_reverse_second_half")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        const int lane = threadIdx.x & 31;
        const storage_t v = in[0];
        const storage_t other = detail::shfl_typed<storage_t, axp::level0::shuffle::xor_>(v, 15);
        out[0] = (lane >= 16) ? other : v;
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         int SrcLane, class InExtra, class OutExtra>
struct Broadcast
    : iro::contract::Realization<
        axp::level0::Broadcast<Recipe, Payload, InSubj, OutSubj, ExecGroup, SrcLane, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.broadcast")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<Payload>::count;
        const uint32_t mask = __activemask();
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            out[i] = detail::shfl_idx_typed<storage_t>(in[i], SrcLane, mask);
        }
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class InExtra, class OutExtra>
struct BroadcastLane0
    : iro::contract::Realization<
        axp::level0::BroadcastLane0<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.broadcast_lane0")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<Payload>::count;
        const uint32_t mask = __activemask();
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            out[i] = detail::shfl_idx_typed<storage_t>(in[i], 0, mask);
        }
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, int BarrierId,
         class InExtra, class OutExtra>
struct WarpgroupBroadcastLane0
    : iro::contract::Realization<
        axp::level0::WarpgroupBroadcastLane0<Recipe, Payload, InSubj, OutSubj, ExecGroup, BarrierId, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.warpgroup_broadcast_lane0")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        static_assert(iro::exec::is_warpgroup_v<ExecGroup>, "WarpgroupBroadcastLane0 requires warpgroup exec");
        constexpr int N = detail::value_traits<Payload>::count;
        constexpr int warps = iro::exec::warpgroup_warps<ExecGroup>::value;
        constexpr int warpgroup_lanes = warps * 32;
        const int warpgroup_lane = threadIdx.x % warpgroup_lanes;
        const int warpgroup_id = threadIdx.x / warpgroup_lanes;
        __shared__ storage_t warpgroup_vals[32][N];
        if (warpgroup_lane == 0) {
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                warpgroup_vals[warpgroup_id][i] = in[i];
            }
        }
        detail::warpgroup_sync<BarrierId, warps>();
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            out[i] = warpgroup_vals[warpgroup_id][i];
        }
        detail::warpgroup_sync<BarrierId, warps>();
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class InPayload, class OutPayload, class InSubj, class OutSubj, class ExecGroup, class Kind,
         class InExtra, class OutExtra>
struct Vote
    : iro::contract::Realization<
        axp::level0::Vote<Recipe, InPayload, OutPayload, InSubj, OutSubj, ExecGroup, Kind, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.vote")> {
    using in_t = typename detail::value_traits<InPayload>::storage_t;
    using out_t = typename detail::value_traits<OutPayload>::storage_t;
    __device__ static void execute(const in_t* in, out_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<OutPayload>::count;
        const uint32_t lane_mask = __activemask();
        uint32_t word = 0;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const bool pred = detail::is_nonzero(in[i]);
            if constexpr (std::is_same_v<Kind, axp::level0::vote::ballot>) {
                word = __ballot_sync(lane_mask, pred);
                out[i] = static_cast<out_t>(word);
            } else if constexpr (std::is_same_v<Kind, axp::level0::vote::any>) {
                out[i] = static_cast<out_t>(__any_sync(lane_mask, pred));
            } else {
                out[i] = static_cast<out_t>(__all_sync(lane_mask, pred));
            }
        }
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class Payload, class MaskPayload, class InSubj, class MaskSubj, class OutSubj,
         class ExecGroup, class OpTag, class InExtra, class OutExtra>
struct ReduxSync
    : iro::contract::Realization<
        axp::level0::ReduxSync<Recipe, Payload, MaskPayload, InSubj, MaskSubj, OutSubj, ExecGroup, OpTag, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.redux_sync")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    using elem_t = typename detail::value_traits<Payload>::elem;
    using mask_t = typename detail::value_traits<MaskPayload>::storage_t;
    __device__ static void execute(const storage_t* in, const mask_t* mask_in, storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<Payload>::count;
        const uint32_t mask = static_cast<uint32_t>(mask_in[0]);
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            storage_t v = in[i];
            if constexpr (std::is_same_v<elem_t, iro::elem::f32>) {
                if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
                    v = __reduce_add_sync(mask, v);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
                    v = __reduce_max_sync(mask, v);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
                    v = __reduce_min_sync(mask, v);
                } else {
                    static_assert(detail::always_false_v<OpTag>, "ReduxSync: unsupported OpTag for f32");
                }
            } else if constexpr (std::is_same_v<elem_t, iro::elem::i32> ||
                                 std::is_same_v<elem_t, iro::elem::u32>) {
                if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_add>) {
                    v = __reduce_add_sync(mask, v);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_max>) {
                    v = __reduce_max_sync(mask, v);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_min>) {
                    v = __reduce_min_sync(mask, v);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_and>) {
                    v = __reduce_and_sync(mask, v);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_or>) {
                    v = __reduce_or_sync(mask, v);
                } else if constexpr (std::is_same_v<OpTag, axp::protocol::reduction::op_xor>) {
                    v = __reduce_xor_sync(mask, v);
                } else {
                    static_assert(detail::always_false_v<OpTag>, "ReduxSync: unsupported OpTag for int");
                }
            } else {
                static_assert(detail::always_false_v<elem_t>, "ReduxSync supports f32/i32/u32 only");
            }
            out[i] = v;
        }
#else
        (void)in; (void)mask_in; (void)out;
#endif
    }
};

template<class Recipe, class InPayload, class MaskPayload, class InSubj, class OutSubj, class ExecGroup, class Kind,
         class InExtra, class OutExtra>
struct Match
    : iro::contract::Realization<
        axp::level0::Match<Recipe, InPayload, MaskPayload, InSubj, OutSubj, ExecGroup, Kind, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.match")> {
    using in_t = typename detail::value_traits<InPayload>::storage_t;
    using out_t = typename detail::value_traits<MaskPayload>::storage_t;
    __device__ static void execute(const in_t* in, out_t* out) {
#ifdef __CUDA_ARCH__
        static_assert(sizeof(in_t) <= 4, "Match supports <=4-byte input types");
        const uint32_t mask = __activemask();
        const uint32_t value = detail::pack_u32(in[0]);
        uint32_t match_mask = 0;
        if constexpr (std::is_same_v<Kind, axp::level0::match::any>) {
            match_mask = __match_any_sync(mask, value);
        } else {
            int all_same = 0;
            match_mask = __match_all_sync(mask, value, &all_same);
            if (!all_same) {
                match_mask = 0u;
            }
        }
        out[0] = static_cast<out_t>(match_mask);
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class MaskPayload, class OutSubj, class ExecGroup, class OutExtra>
struct ElectOne
    : iro::contract::Realization<
        axp::level0::ElectOne<Recipe, MaskPayload, OutSubj, ExecGroup, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.elect_one")> {
    using out_t = typename detail::value_traits<MaskPayload>::storage_t;
    __device__ static void execute(out_t* out) {
#ifdef __CUDA_ARCH__
        const uint32_t mask = __activemask();
        const int leader = __ffs(mask) - 1;
        const uint32_t one = (leader >= 0) ? (1u << leader) : 0u;
        out[0] = static_cast<out_t>(one);
#else
        (void)out;
#endif
    }
};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class Pattern, int BlockThreads,
         class InExtra, class OutExtra>
struct PermuteCross
    : iro::contract::Realization<
        axp::level0::PermuteCross<Recipe, Payload, InSubj, OutSubj, ExecGroup, Pattern, BlockThreads, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.permute_cross")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<Payload>::count;
        __shared__ storage_t smem[BlockThreads * N];
        const int tid = threadIdx.x;
        if (tid < BlockThreads) {
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                smem[tid * N + i] = in[i];
            }
        }
        __syncthreads();
        if (tid < BlockThreads) {
            const int src = Pattern::map(tid, BlockThreads);
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                out[i] = smem[src * N + i];
            }
        }
        __syncthreads();
#else
        (void)in; (void)out;
#endif
    }
};

// -----------------------------------------------------------------------------
// L0 Scan (warp/block)
// -----------------------------------------------------------------------------

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra, class OutExtra>
struct WarpScan
    : iro::contract::Realization<
        axp::level0::WarpScan<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.scan.warp")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    using elem_t = typename detail::value_traits<Payload>::elem;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        const int lane = threadIdx.x & 31;
        constexpr int N = detail::value_traits<Payload>::count;
        using acc_t = typename detail::acc_traits<elem_t, Recipe>::acc_t;
        constexpr bool vec_ok =
            (detail::is_f16<storage_t>::value || detail::is_bf16<storage_t>::value) &&
            (N % 2 == 0) &&
            std::is_same_v<acc_t, storage_t>;
        if constexpr (vec_ok) {
            using vec_t = std::conditional_t<detail::is_f16<storage_t>::value, __half2, __nv_bfloat162>;
            constexpr int N2 = N / 2;
            const vec_t* in2 = reinterpret_cast<const vec_t*>(in);
            vec_t* out2 = reinterpret_cast<vec_t*>(out);
            #pragma unroll
            for (int i = 0; i < N2; ++i) {
                vec_t v = in2[i];
                for (int offset = 1; offset < 32; offset <<= 1) {
                    vec_t t = detail::shfl_typed<vec_t, axp::level0::shuffle::up>(v, offset);
                    if (lane >= offset) {
                        v = detail::op_apply_vec<OpTag>(v, t);
                    }
                }
                if constexpr (std::is_same_v<Mode, axp::level0::scan::exclusive>) {
                    vec_t prev = detail::shfl_typed<vec_t, axp::level0::shuffle::up>(v, 1);
                    v = (lane == 0) ? detail::op_identity_vec<OpTag, vec_t>() : prev;
                }
                out2[i] = v;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                acc_t v = detail::to_acc<elem_t, Recipe>(in[i]);
                for (int offset = 1; offset < 32; offset <<= 1) {
                    acc_t t = detail::shfl_typed<acc_t, axp::level0::shuffle::up>(v, offset);
                    if (lane >= offset) {
                        v = detail::op_apply_acc<OpTag>(v, t);
                    }
                }
                if constexpr (std::is_same_v<Mode, axp::level0::scan::exclusive>) {
                    acc_t prev = detail::shfl_typed<acc_t, axp::level0::shuffle::up>(v, 1);
                    v = (lane == 0) ? detail::op_identity_acc<OpTag, acc_t>() : prev;
                }
                out[i] = detail::from_acc<elem_t, Recipe>(v);
            }
        }
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode, int SegmentWidth,
         class InExtra, class OutExtra>
struct WarpSegmentedScan
    : iro::contract::Realization<
        axp::level0::WarpSegmentedScan<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, SegmentWidth, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.scan.warp_segmented")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    using elem_t = typename detail::value_traits<Payload>::elem;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        const int lane = threadIdx.x & 31;
        const int base = lane & ~(SegmentWidth - 1);
        const int pos = lane - base;
        uint32_t seg_mask = 0xffffffffu;
        if constexpr (SegmentWidth != 32) {
            seg_mask = ((1u << SegmentWidth) - 1u) << base;
        }
        constexpr int N = detail::value_traits<Payload>::count;
        using acc_t = typename detail::acc_traits<elem_t, Recipe>::acc_t;
        constexpr bool vec_ok =
            (detail::is_f16<storage_t>::value || detail::is_bf16<storage_t>::value) &&
            (N % 2 == 0) &&
            std::is_same_v<acc_t, storage_t>;
        if constexpr (vec_ok) {
            using vec_t = std::conditional_t<detail::is_f16<storage_t>::value, __half2, __nv_bfloat162>;
            constexpr int N2 = N / 2;
            const vec_t* in2 = reinterpret_cast<const vec_t*>(in);
            vec_t* out2 = reinterpret_cast<vec_t*>(out);
            #pragma unroll
            for (int i = 0; i < N2; ++i) {
                vec_t v = in2[i];
                for (int offset = 1; offset < SegmentWidth; offset <<= 1) {
                    vec_t t = detail::shfl_typed_mask<vec_t, axp::level0::shuffle::up>(v, offset, seg_mask);
                    if (pos >= offset) {
                        v = detail::op_apply_vec<OpTag>(v, t);
                    }
                }
                if constexpr (std::is_same_v<Mode, axp::level0::scan::exclusive>) {
                    vec_t prev = detail::shfl_typed_mask<vec_t, axp::level0::shuffle::up>(v, 1, seg_mask);
                    v = (pos == 0) ? detail::op_identity_vec<OpTag, vec_t>() : prev;
                }
                out2[i] = v;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                acc_t v = detail::to_acc<elem_t, Recipe>(in[i]);
                for (int offset = 1; offset < SegmentWidth; offset <<= 1) {
                    acc_t t = detail::shfl_typed_mask<acc_t, axp::level0::shuffle::up>(v, offset, seg_mask);
                    if (pos >= offset) {
                        v = detail::op_apply_acc<OpTag>(v, t);
                    }
                }
                if constexpr (std::is_same_v<Mode, axp::level0::scan::exclusive>) {
                    acc_t prev = detail::shfl_typed_mask<acc_t, axp::level0::shuffle::up>(v, 1, seg_mask);
                    v = (pos == 0) ? detail::op_identity_acc<OpTag, acc_t>() : prev;
                }
                out[i] = detail::from_acc<elem_t, Recipe>(v);
            }
        }
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         int BarrierId, int WarpgroupCount, class InExtra, class OutExtra>
struct WarpgroupScan
    : iro::contract::Realization<
        axp::level0::WarpgroupScan<
            Recipe, Payload, Subj, ExecGroup, OpTag, Mode, BarrierId, WarpgroupCount, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.scan.warpgroup")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    using elem_t = typename detail::value_traits<Payload>::elem;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        static_assert(iro::exec::is_warpgroup_v<ExecGroup>, "WarpgroupScan requires warpgroup exec");
        constexpr int N = detail::value_traits<Payload>::count;
        using acc_t = typename detail::acc_traits<elem_t, Recipe>::acc_t;
        constexpr int warps = iro::exec::warpgroup_warps<ExecGroup>::value;
        constexpr int warpgroup_lanes = warps * 32;
        const int lane = threadIdx.x & 31;
        const int warp_in_group = (threadIdx.x >> 5) % warps;
        const int warpgroup_id = threadIdx.x / warpgroup_lanes;
        const int warp_base = warpgroup_id * warps;
        constexpr bool vec_ok =
            (detail::is_f16<storage_t>::value || detail::is_bf16<storage_t>::value) &&
            (N % 2 == 0) &&
            std::is_same_v<acc_t, storage_t>;
        if constexpr (vec_ok) {
            using vec_t = std::conditional_t<detail::is_f16<storage_t>::value, __half2, __nv_bfloat162>;
            constexpr int N2 = N / 2;
            __shared__ vec_t warp_totals2[32];
            __shared__ vec_t warp_prefix2[32];
            const vec_t* in2 = reinterpret_cast<const vec_t*>(in);
            vec_t* out2 = reinterpret_cast<vec_t*>(out);
            #pragma unroll
            for (int i = 0; i < N2; ++i) {
                vec_t v = in2[i];
                for (int offset = 1; offset < 32; offset <<= 1) {
                    vec_t t = detail::shfl_typed<vec_t, axp::level0::shuffle::up>(v, offset);
                    if (lane >= offset) {
                        v = detail::op_apply_vec<OpTag>(v, t);
                    }
                }
                const vec_t warp_sum = detail::shfl_idx_typed<vec_t>(v, 31, __activemask());
                if constexpr (std::is_same_v<Mode, axp::level0::scan::exclusive>) {
                    vec_t prev = detail::shfl_typed<vec_t, axp::level0::shuffle::up>(v, 1);
                    v = (lane == 0) ? detail::op_identity_vec<OpTag, vec_t>() : prev;
                }
                if (lane == 31) {
                    warp_totals2[warp_base + warp_in_group] = warp_sum;
                }
                detail::warpgroup_sync<BarrierId, warps>();
                if (warp_in_group == 0) {
                    vec_t prefix = (lane < warps)
                        ? warp_totals2[warp_base + lane]
                        : detail::op_identity_vec<OpTag, vec_t>();
                    for (int offset = 1; offset < warps; offset <<= 1) {
                        vec_t t = detail::shfl_typed<vec_t, axp::level0::shuffle::up>(prefix, offset);
                        if (lane >= offset) {
                            prefix = detail::op_apply_vec<OpTag>(prefix, t);
                        }
                    }
                    if (lane < warps) {
                        warp_prefix2[warp_base + lane] = prefix;
                    }
                }
                detail::warpgroup_sync<BarrierId, warps>();
                vec_t add = (warp_in_group == 0)
                    ? detail::op_identity_vec<OpTag, vec_t>()
                    : warp_prefix2[warp_base + warp_in_group - 1];
                v = detail::op_apply_vec<OpTag>(add, v);
                out2[i] = v;
                detail::warpgroup_sync<BarrierId, warps>();
            }
        } else {
            __shared__ acc_t warp_totals[32];
            __shared__ acc_t warp_prefix[32];
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                acc_t v = detail::to_acc<elem_t, Recipe>(in[i]);
                for (int offset = 1; offset < 32; offset <<= 1) {
                    acc_t t = detail::shfl_typed<acc_t, axp::level0::shuffle::up>(v, offset);
                    if (lane >= offset) {
                        v = detail::op_apply_acc<OpTag>(v, t);
                    }
                }
                const acc_t warp_sum = detail::shfl_idx_typed<acc_t>(v, 31, __activemask());
                if constexpr (std::is_same_v<Mode, axp::level0::scan::exclusive>) {
                    acc_t prev = detail::shfl_typed<acc_t, axp::level0::shuffle::up>(v, 1);
                    v = (lane == 0) ? detail::op_identity_acc<OpTag, acc_t>() : prev;
                }
                if (lane == 31) {
                    warp_totals[warp_base + warp_in_group] = warp_sum;
                }
                detail::warpgroup_sync<BarrierId, warps>();
                if (warp_in_group == 0) {
                    acc_t prefix = (lane < warps)
                        ? warp_totals[warp_base + lane]
                        : detail::op_identity_acc<OpTag, acc_t>();
                    for (int offset = 1; offset < warps; offset <<= 1) {
                        acc_t t = detail::shfl_typed<acc_t, axp::level0::shuffle::up>(prefix, offset);
                        if (lane >= offset) {
                            prefix = detail::op_apply_acc<OpTag>(prefix, t);
                        }
                    }
                    if (lane < warps) {
                        warp_prefix[warp_base + lane] = prefix;
                    }
                }
                detail::warpgroup_sync<BarrierId, warps>();
                acc_t add = (warp_in_group == 0)
                    ? detail::op_identity_acc<OpTag, acc_t>()
                    : warp_prefix[warp_base + warp_in_group - 1];
                v = detail::op_apply_acc<OpTag>(add, v);
                out[i] = detail::from_acc<elem_t, Recipe>(v);
                detail::warpgroup_sync<BarrierId, warps>();
            }
        }
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra, class OutExtra>
struct BlockScan
    : iro::contract::Realization<
        axp::level0::BlockScan<Recipe, Payload, Subj, ExecGroup, OpTag, Mode, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.scan.block")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    using elem_t = typename detail::value_traits<Payload>::elem;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<Payload>::count;
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int num_warps = (blockDim.x + 31) >> 5;
        using acc_t = typename detail::acc_traits<elem_t, Recipe>::acc_t;
        constexpr bool vec_ok =
            (detail::is_f16<storage_t>::value || detail::is_bf16<storage_t>::value) &&
            (N % 2 == 0) &&
            std::is_same_v<acc_t, storage_t>;
        if constexpr (vec_ok) {
            using vec_t = std::conditional_t<detail::is_f16<storage_t>::value, __half2, __nv_bfloat162>;
            constexpr int N2 = N / 2;
            __shared__ vec_t warp_totals2[32];
            const vec_t* in2 = reinterpret_cast<const vec_t*>(in);
            vec_t* out2 = reinterpret_cast<vec_t*>(out);
            #pragma unroll
            for (int i = 0; i < N2; ++i) {
                vec_t v = in2[i];
                for (int offset = 1; offset < 32; offset <<= 1) {
                    vec_t t = detail::shfl_typed<vec_t, axp::level0::shuffle::up>(v, offset);
                    if (lane >= offset) {
                        v = detail::op_apply_vec<OpTag>(v, t);
                    }
                }
                if (lane == 31) {
                    warp_totals2[warp] = v;
                }
                __syncthreads();
                if (warp == 0) {
                    vec_t w = (lane < num_warps) ? warp_totals2[lane] : detail::op_identity_vec<OpTag, vec_t>();
                    for (int offset = 1; offset < 32; offset <<= 1) {
                        vec_t t = detail::shfl_typed<vec_t, axp::level0::shuffle::up>(w, offset);
                        if (lane >= offset) {
                            w = detail::op_apply_vec<OpTag>(w, t);
                        }
                    }
                    warp_totals2[lane] = w;
                }
                __syncthreads();
                vec_t base = (warp == 0) ? detail::op_identity_vec<OpTag, vec_t>() : warp_totals2[warp - 1];
                vec_t outv = detail::op_apply_vec<OpTag>(v, base);
                if constexpr (std::is_same_v<Mode, axp::level0::scan::exclusive>) {
                    vec_t prev = detail::shfl_typed<vec_t, axp::level0::shuffle::up>(outv, 1);
                    outv = (lane == 0) ? base : prev;
                }
                out2[i] = outv;
                __syncthreads();
            }
        } else {
            __shared__ acc_t warp_totals[32];
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                acc_t v = detail::to_acc<elem_t, Recipe>(in[i]);
                for (int offset = 1; offset < 32; offset <<= 1) {
                    acc_t t = detail::shfl_typed<acc_t, axp::level0::shuffle::up>(v, offset);
                    if (lane >= offset) {
                        v = detail::op_apply_acc<OpTag>(v, t);
                    }
                }
                if (lane == 31) {
                    warp_totals[warp] = v;
                }
                __syncthreads();
                if (warp == 0) {
                    acc_t w = (lane < num_warps) ? warp_totals[lane] : detail::op_identity_acc<OpTag, acc_t>();
                    for (int offset = 1; offset < 32; offset <<= 1) {
                        acc_t t = detail::shfl_typed<acc_t, axp::level0::shuffle::up>(w, offset);
                        if (lane >= offset) {
                            w = detail::op_apply_acc<OpTag>(w, t);
                        }
                    }
                    warp_totals[lane] = w;
                }
                __syncthreads();
                acc_t base = (warp == 0) ? detail::op_identity_acc<OpTag, acc_t>() : warp_totals[warp - 1];
                acc_t outv = detail::op_apply_acc<OpTag>(v, base);
                if constexpr (std::is_same_v<Mode, axp::level0::scan::exclusive>) {
                    acc_t prev = detail::shfl_typed<acc_t, axp::level0::shuffle::up>(outv, 1);
                    outv = (lane == 0) ? base : prev;
                }
                out[i] = detail::from_acc<elem_t, Recipe>(outv);
                __syncthreads();
            }
        }
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class Payload, class CarryPayload, class Subj, class CarryInSubj, class CarryOutSubj,
         class ExecGroup, class OpTag, class Mode, class InExtra, class OutExtra>
struct ChainedScan
    : iro::contract::Realization<
        axp::level0::ChainedScan<Recipe, Payload, CarryPayload, Subj, CarryInSubj, CarryOutSubj,
                                 ExecGroup, OpTag, Mode, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.scan.chained")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    using elem_t = typename detail::value_traits<Payload>::elem;
    __device__ static void execute(const storage_t* in, const storage_t* carry_in,
                                   storage_t* out, storage_t* carry_out) {
#ifdef __CUDA_ARCH__
        static_assert(detail::value_traits<Payload>::count == 1, "ChainedScan supports scalar payloads only");
        constexpr int N = detail::value_traits<Payload>::count;
        using acc_t = typename detail::acc_traits<elem_t, Recipe>::acc_t;
        __shared__ acc_t warp_totals[32];
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int num_warps = (blockDim.x + 31) >> 5;

        #pragma unroll
        for (int i = 0; i < N; ++i) {
            acc_t v = detail::to_acc<elem_t, Recipe>(in[i]);
            for (int offset = 1; offset < 32; offset <<= 1) {
                acc_t t = detail::shfl_typed<acc_t, axp::level0::shuffle::up>(v, offset);
                if (lane >= offset) {
                    v = detail::op_apply_acc<OpTag>(v, t);
                }
            }
            if (lane == 31) {
                warp_totals[warp] = v;
            }
            __syncthreads();
            if (warp == 0) {
                acc_t w = (lane < num_warps) ? warp_totals[lane] : detail::op_identity_acc<OpTag, acc_t>();
                for (int offset = 1; offset < 32; offset <<= 1) {
                    acc_t t = detail::shfl_typed<acc_t, axp::level0::shuffle::up>(w, offset);
                    if (lane >= offset) {
                        w = detail::op_apply_acc<OpTag>(w, t);
                    }
                }
                warp_totals[lane] = w;
            }
            __syncthreads();

            const acc_t carry = detail::to_acc<elem_t, Recipe>(carry_in[0]);
            const acc_t base = (warp == 0)
                ? carry
                : detail::op_apply_acc<OpTag>(carry, warp_totals[warp - 1]);
            acc_t outv = detail::op_apply_acc<OpTag>(base, v);
            if constexpr (std::is_same_v<Mode, axp::level0::scan::exclusive>) {
                acc_t prev = detail::shfl_typed<acc_t, axp::level0::shuffle::up>(outv, 1);
                outv = (lane == 0) ? base : prev;
            }
            out[i] = detail::from_acc<elem_t, Recipe>(outv);
            if (threadIdx.x == 0) {
                const acc_t total = (num_warps > 0)
                    ? warp_totals[num_warps - 1]
                    : detail::op_identity_acc<OpTag, acc_t>();
                carry_out[0] = detail::from_acc<elem_t, Recipe>(detail::op_apply_acc<OpTag>(carry, total));
            }
            __syncthreads();
        }
#else
        (void)in; (void)carry_in; (void)out; (void)carry_out;
#endif
    }
};
// -----------------------------------------------------------------------------
// L0 Memory (tile copy)
// -----------------------------------------------------------------------------

template<class Recipe, class Tile, class ScaleTile, class TileSubj, class ScaleSubj, class ExecGroup,
         class TileInExtra, class ScaleInExtra, class OutExtra>
struct ScaleSharedTile
    : iro::contract::Realization<
        axp::level0::ScaleSharedTile<Recipe, Tile, ScaleTile, TileSubj, ScaleSubj, ExecGroup,
                                     TileInExtra, ScaleInExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.scale_shared_tile")> {
    __device__ static void execute(const typename Tile::elem::storage_t* in,
                                   const typename ScaleTile::elem::storage_t* scale,
                                   typename Tile::elem::storage_t* out) {
        detail::scale_shared_tile<Recipe, Tile, ScaleTile, ExecGroup>(in, scale, out);
    }
};

template<class Recipe, class Tile, class OutSubj, class ExecGroup, class OutExtra>
struct TileZero
    : iro::contract::Realization<
        axp::level0::TileZero<Recipe, Tile, OutSubj, ExecGroup, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.tile_zero")> {
    using storage_t = typename Tile::elem::storage_t;
    __device__ static void execute(storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int VecBytes = Recipe::vec_bytes;
        static_assert(VecBytes == 4 || VecBytes == 8 || VecBytes == 16);
        static_assert(Tile::align::bytes >= VecBytes, "TileZero: output alignment too small");
        constexpr int elems_per_vec = VecBytes / static_cast<int>(sizeof(storage_t));
        static_assert(elems_per_vec > 0, "TileZero: invalid vector width");
        static_assert((Tile::shape::size % elems_per_vec) == 0, "TileZero: tile size must be vector aligned");

        int tid = 0;
        int stride = 1;
        detail::exec_group_lane_stride<ExecGroup>(tid, stride);

        using VecT = detail::vec_type_t<VecBytes>;
        const VecT zero{};
        for (int idx = tid * elems_per_vec; idx < Tile::shape::size; idx += stride * elems_per_vec) {
            VecT* dst = reinterpret_cast<VecT*>(out + idx);
            *dst = zero;
        }
        if constexpr (std::is_same_v<ExecGroup, iro::exec::block>) {
            __syncthreads();
        }
#else
        (void)out;
#endif
    }
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct ReduceSharedToGlobalAtomicAdd
    : iro::contract::Realization<
        axp::level0::ReduceSharedToGlobalAtomicAdd<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.reduce_shared_to_global_atomic_add")> {
    using in_t = typename InTile::elem::storage_t;
    using out_t = typename OutTile::elem::storage_t;
    __device__ static void execute(const in_t* in, const out_t* out_in, out_t* out) {
#ifdef __CUDA_ARCH__
        int tid = 0;
        int stride = 1;
        detail::exec_group_lane_stride<ExecGroup>(tid, stride);
        if constexpr (std::is_same_v<ExecGroup, iro::exec::block>) {
            __syncthreads();
        }
        for (int idx = tid; idx < InTile::shape::size; idx += stride) {
            const long long in_off = InTile::layout::offset(idx);
            const long long out_off = OutTile::layout::offset(idx);
            const out_t val = static_cast<out_t>(in[in_off]);
            if (val != static_cast<out_t>(0)) {
                detail::atomic_add(&out[out_off], val);
            }
        }
#else
        (void)in; (void)out_in; (void)out;
#endif
    }
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CachePolicy, class InDist, class OutDist, class InExtra, class OutExtra>
struct LdGlobal
    : iro::contract::Realization<
        axp::level0::LdGlobal<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, CachePolicy, InDist, OutDist, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.ld_global")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in,
                                   typename OutTile::elem::storage_t* out) {
        detail::tile_copy_global<Recipe, InTile, OutTile, ExecGroup>(in, out);
    }
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CachePolicy, class InDist, class OutDist, class InExtra, class OutExtra>
struct StGlobal
    : iro::contract::Realization<
        axp::level0::StGlobal<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, CachePolicy, InDist, OutDist, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.st_global")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in,
                                   typename OutTile::elem::storage_t* out) {
        detail::tile_copy<Recipe, InTile, OutTile, ExecGroup>(in, out);
    }
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class InDist, class OutDist, class InExtra, class OutExtra>
struct LdShared
    : iro::contract::Realization<
        axp::level0::LdShared<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, InDist, OutDist, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.ld_shared")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in,
                                   typename OutTile::elem::storage_t* out) {
        detail::tile_copy<Recipe, InTile, OutTile, ExecGroup>(in, out);
    }
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class InDist, class OutDist, class InExtra, class OutExtra>
struct StShared
    : iro::contract::Realization<
        axp::level0::StShared<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, InDist, OutDist, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.st_shared")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in,
                                   typename OutTile::elem::storage_t* out) {
        detail::tile_copy<Recipe, InTile, OutTile, ExecGroup>(in, out);
    }
};

template<class Recipe, class InTile, class IndexPayload, class OutPayload,
         class InSubj, class IndexSubj, class OutSubj, class ExecGroup,
         class CachePolicy, class InExtra, class OutExtra>
struct GatherGlobal
    : iro::contract::Realization<
        axp::level0::GatherGlobal<Recipe, InTile, IndexPayload, OutPayload, InSubj, IndexSubj, OutSubj,
                                  ExecGroup, CachePolicy, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.gather_global")> {
    using in_t = typename InTile::elem::storage_t;
    using idx_t = typename detail::value_traits<IndexPayload>::storage_t;
    using out_t = typename detail::value_traits<OutPayload>::storage_t;
    __device__ static void execute(const in_t* in, const idx_t* idx, out_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<OutPayload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const int index = static_cast<int>(idx[i]);
            const long long off = InTile::layout::offset(index);
            out[i] = in[off];
        }
#else
        (void)in; (void)idx; (void)out;
#endif
    }
};

template<class Recipe, class InPayload, class IndexPayload, class OutTile,
         class InSubj, class IndexSubj, class OutSubj, class ExecGroup,
         class CachePolicy, class InExtra, class OutExtra>
struct ScatterGlobal
    : iro::contract::Realization<
        axp::level0::ScatterGlobal<Recipe, InPayload, IndexPayload, OutTile, InSubj, IndexSubj, OutSubj,
                                   ExecGroup, CachePolicy, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.scatter_global")> {
    using in_t = typename detail::value_traits<InPayload>::storage_t;
    using idx_t = typename detail::value_traits<IndexPayload>::storage_t;
    using out_t = typename OutTile::elem::storage_t;
    __device__ static void execute(const in_t* in, const idx_t* idx, out_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<InPayload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const int index = static_cast<int>(idx[i]);
            const long long off = OutTile::layout::offset(index);
            out[off] = in[i];
        }
#else
        (void)in; (void)idx; (void)out;
#endif
    }
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct AtomicAdd
    : iro::contract::Realization<
        axp::level0::AtomicAdd<Recipe, InPayload, IndexPayload, OutPayload, OutTile,
                               InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.atomic_add")> {
    using in_t = typename detail::value_traits<InPayload>::storage_t;
    using idx_t = typename detail::value_traits<IndexPayload>::storage_t;
    using out_t = typename OutTile::elem::storage_t;
    using old_t = typename detail::value_traits<OutPayload>::storage_t;
    __device__ static void execute(const out_t* /*out_in*/, const in_t* in, const idx_t* idx,
                                   old_t* out_old, out_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<InPayload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const int index = static_cast<int>(idx[i]);
            const long long off = OutTile::layout::offset(index);
            out_old[i] = detail::atomic_add(&out[off], static_cast<out_t>(in[i]));
        }
#else
        (void)in; (void)idx; (void)out_old; (void)out;
#endif
    }
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct AtomicMin
    : iro::contract::Realization<
        axp::level0::AtomicMin<Recipe, InPayload, IndexPayload, OutPayload, OutTile,
                               InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.atomic_min")> {
    using in_t = typename detail::value_traits<InPayload>::storage_t;
    using idx_t = typename detail::value_traits<IndexPayload>::storage_t;
    using out_t = typename OutTile::elem::storage_t;
    using old_t = typename detail::value_traits<OutPayload>::storage_t;
    __device__ static void execute(const in_t* in, const idx_t* idx, old_t* out_old, out_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<InPayload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const int index = static_cast<int>(idx[i]);
            const long long off = OutTile::layout::offset(index);
            out_old[i] = detail::atomic_min(&out[off], static_cast<out_t>(in[i]));
        }
#else
        (void)in; (void)idx; (void)out_old; (void)out;
#endif
    }
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct AtomicMax
    : iro::contract::Realization<
        axp::level0::AtomicMax<Recipe, InPayload, IndexPayload, OutPayload, OutTile,
                               InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.atomic_max")> {
    using in_t = typename detail::value_traits<InPayload>::storage_t;
    using idx_t = typename detail::value_traits<IndexPayload>::storage_t;
    using out_t = typename OutTile::elem::storage_t;
    using old_t = typename detail::value_traits<OutPayload>::storage_t;
    __device__ static void execute(const in_t* in, const idx_t* idx, old_t* out_old, out_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<InPayload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const int index = static_cast<int>(idx[i]);
            const long long off = OutTile::layout::offset(index);
            out_old[i] = detail::atomic_max(&out[off], static_cast<out_t>(in[i]));
        }
#else
        (void)in; (void)idx; (void)out_old; (void)out;
#endif
    }
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct AtomicAnd
    : iro::contract::Realization<
        axp::level0::AtomicAnd<Recipe, InPayload, IndexPayload, OutPayload, OutTile,
                               InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.atomic_and")> {
    using in_t = typename detail::value_traits<InPayload>::storage_t;
    using idx_t = typename detail::value_traits<IndexPayload>::storage_t;
    using out_t = typename OutTile::elem::storage_t;
    using old_t = typename detail::value_traits<OutPayload>::storage_t;
    __device__ static void execute(const in_t* in, const idx_t* idx, old_t* out_old, out_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<InPayload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const int index = static_cast<int>(idx[i]);
            const long long off = OutTile::layout::offset(index);
            out_old[i] = detail::atomic_and(&out[off], static_cast<out_t>(in[i]));
        }
#else
        (void)in; (void)idx; (void)out_old; (void)out;
#endif
    }
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct AtomicOr
    : iro::contract::Realization<
        axp::level0::AtomicOr<Recipe, InPayload, IndexPayload, OutPayload, OutTile,
                              InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.atomic_or")> {
    using in_t = typename detail::value_traits<InPayload>::storage_t;
    using idx_t = typename detail::value_traits<IndexPayload>::storage_t;
    using out_t = typename OutTile::elem::storage_t;
    using old_t = typename detail::value_traits<OutPayload>::storage_t;
    __device__ static void execute(const in_t* in, const idx_t* idx, old_t* out_old, out_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<InPayload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const int index = static_cast<int>(idx[i]);
            const long long off = OutTile::layout::offset(index);
            out_old[i] = detail::atomic_or(&out[off], static_cast<out_t>(in[i]));
        }
#else
        (void)in; (void)idx; (void)out_old; (void)out;
#endif
    }
};

template<class Recipe, class InPayload, class IndexPayload, class OutPayload, class OutTile,
         class InSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct AtomicXor
    : iro::contract::Realization<
        axp::level0::AtomicXor<Recipe, InPayload, IndexPayload, OutPayload, OutTile,
                               InSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.atomic_xor")> {
    using in_t = typename detail::value_traits<InPayload>::storage_t;
    using idx_t = typename detail::value_traits<IndexPayload>::storage_t;
    using out_t = typename OutTile::elem::storage_t;
    using old_t = typename detail::value_traits<OutPayload>::storage_t;
    __device__ static void execute(const in_t* in, const idx_t* idx, old_t* out_old, out_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<InPayload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const int index = static_cast<int>(idx[i]);
            const long long off = OutTile::layout::offset(index);
            out_old[i] = detail::atomic_xor(&out[off], static_cast<out_t>(in[i]));
        }
#else
        (void)in; (void)idx; (void)out_old; (void)out;
#endif
    }
};

template<class Recipe, class ComparePayload, class ValuePayload, class IndexPayload, class OutPayload, class OutTile,
         class CompareSubj, class ValueSubj, class IndexSubj, class OutValSubj, class OutSubj, class ExecGroup,
         class InExtra, class OutExtra>
struct AtomicCAS
    : iro::contract::Realization<
        axp::level0::AtomicCAS<Recipe, ComparePayload, ValuePayload, IndexPayload, OutPayload, OutTile,
                               CompareSubj, ValueSubj, IndexSubj, OutValSubj, OutSubj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.atomic_cas")> {
    using cmp_t = typename detail::value_traits<ComparePayload>::storage_t;
    using val_t = typename detail::value_traits<ValuePayload>::storage_t;
    using idx_t = typename detail::value_traits<IndexPayload>::storage_t;
    using out_t = typename OutTile::elem::storage_t;
    using old_t = typename detail::value_traits<OutPayload>::storage_t;
    __device__ static void execute(const cmp_t* compare, const val_t* value, const idx_t* idx,
                                   old_t* out_old, out_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<ComparePayload>::count;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const int index = static_cast<int>(idx[i]);
            const long long off = OutTile::layout::offset(index);
            out_old[i] = detail::atomic_cas(&out[off],
                                            static_cast<out_t>(compare[i]),
                                            static_cast<out_t>(value[i]));
        }
#else
        (void)compare; (void)value; (void)idx; (void)out_old; (void)out;
#endif
    }
};

// Swizzled shared-memory load/store
template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class SwizzleAtom, class InDist, class OutDist, class InExtra, class OutExtra>
struct SwizzledLdShared
    : iro::contract::Realization<
        axp::level0::SwizzledLdShared<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, SwizzleAtom, InDist, OutDist, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.swizzled_ld_shared")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in,
                                   typename OutTile::elem::storage_t* out) {
        detail::tile_copy_swizzled_to_row<InTile, OutTile, ExecGroup>(in, out);
    }
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class SwizzleAtom, class InDist, class OutDist, class InExtra, class OutExtra>
struct SwizzledStShared
    : iro::contract::Realization<
        axp::level0::SwizzledStShared<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, SwizzleAtom, InDist, OutDist, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.swizzled_st_shared")> {
    __device__ static void execute(const typename InTile::elem::storage_t* in,
                                   typename OutTile::elem::storage_t* out) {
        detail::tile_copy_row_to_swizzled<InTile, OutTile, ExecGroup>(in, out);
    }
};

// Shared-memory tile fence (block sync)
template<class Recipe, class Tile, class Subj, class ExecGroup, class InExtra, class OutExtra>
struct TileFence
    : iro::contract::Realization<
        axp::level0::TileFence<Recipe, Tile, Subj, ExecGroup, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.tile_fence")> {
    __device__ static void execute(const typename Tile::elem::storage_t* /*in*/,
                                   typename Tile::elem::storage_t* /*out*/) {
#ifdef __CUDA_ARCH__
        static_assert(std::is_same_v<ExecGroup, iro::exec::block> ||
                      iro::exec::is_warpgroup_v<ExecGroup>,
                      "TileFence supports block or warpgroup exec only");
        if constexpr (std::is_same_v<ExecGroup, iro::exec::block>) {
            __syncthreads();
        } else {
            constexpr int warps = iro::exec::warpgroup_warps<ExecGroup>::value;
            detail::warpgroup_sync<1, warps>();
        }
#endif
    }
};

// -----------------------------------------------------------------------------
// L0 Pipeline control
// -----------------------------------------------------------------------------

template<class Recipe, class IndexPayload, class InSubj, class OutSubj, class ExecGroup, int Stages,
         class InExtra, class OutExtra>
struct PipelineAdvance
    : iro::contract::Realization<
        axp::level0::PipelineAdvance<Recipe, IndexPayload, InSubj, OutSubj, ExecGroup, Stages, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.pipeline_advance")> {
    using storage_t = typename detail::value_traits<IndexPayload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        const storage_t next = static_cast<storage_t>((in[0] + 1u) % static_cast<storage_t>(Stages));
        out[0] = next;
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class IndexPayload, class InSubj, class OutSubj, class ExecGroup, int Stages,
         class InExtra, class OutExtra>
struct PipelineProduce
    : iro::contract::Realization<
        axp::level0::PipelineProduce<Recipe, IndexPayload, InSubj, OutSubj, ExecGroup, Stages, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.pipeline_produce")> {
    using storage_t = typename detail::value_traits<IndexPayload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        const storage_t next = static_cast<storage_t>((in[0] + 1u) % static_cast<storage_t>(Stages));
        out[0] = next;
#else
        (void)in; (void)out;
#endif
    }
};

template<class Recipe, class IndexPayload, class InSubj, class OutSubj, class ExecGroup, int Stages,
         class InExtra, class OutExtra>
struct PipelineConsume
    : iro::contract::Realization<
        axp::level0::PipelineConsume<Recipe, IndexPayload, InSubj, OutSubj, ExecGroup, Stages, InExtra, OutExtra>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.pipeline_consume")> {
    using storage_t = typename detail::value_traits<IndexPayload>::storage_t;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        const storage_t next = static_cast<storage_t>((in[0] + 1u) % static_cast<storage_t>(Stages));
        out[0] = next;
#else
        (void)in; (void)out;
#endif
    }
};

// -----------------------------------------------------------------------------
// Warpgroup reduction (value payloads)
// -----------------------------------------------------------------------------

template<class Recipe, class Payload, class MaskPayload, class Subj, class MaskSubj, class ExecGroup, class OpTag>
struct ShuffleReduceTree
    : iro::contract::Realization<
        axp::protocol::reduction::ShuffleReduceTree<Recipe, Payload, MaskPayload, Subj, MaskSubj, ExecGroup, OpTag>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.reduce.shuffle_tree")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    using elem_t = typename detail::value_traits<Payload>::elem;
    using mask_t = typename detail::value_traits<MaskPayload>::storage_t;
    __device__ static void execute(const storage_t* in, const mask_t* mask_in, storage_t* out) {
#ifdef __CUDA_ARCH__
        constexpr int N = detail::value_traits<Payload>::count;
        const uint32_t mask = static_cast<uint32_t>(mask_in[0]);
        if (mask == 0u) {
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                out[i] = detail::op_identity<OpTag, elem_t, Recipe>();
            }
            return;
        }
        const int lane = threadIdx.x & 31;
        const int leader = __ffs(mask) - 1;
        constexpr int Width = MaskPayload::width;
        using acc_t = typename detail::acc_traits<elem_t, Recipe>::acc_t;
        constexpr bool vec_ok =
            (detail::is_f16<storage_t>::value || detail::is_bf16<storage_t>::value) &&
            (N % 2) == 0 &&
            std::is_same_v<acc_t, storage_t>;
        if constexpr (vec_ok) {
            using vec_t = std::conditional_t<detail::is_f16<storage_t>::value, __half2, __nv_bfloat162>;
            constexpr int N2 = N / 2;
            const vec_t* in2 = reinterpret_cast<const vec_t*>(in);
            vec_t* out2 = reinterpret_cast<vec_t*>(out);
            #pragma unroll
            for (int i = 0; i < N2; ++i) {
                vec_t v = in2[i];
                for (int offset = Width / 2; offset > 0; offset >>= 1) {
                    vec_t t = detail::shfl_typed_mask<vec_t, axp::level0::shuffle::down>(v, offset, mask);
                    v = detail::op_apply_vec<OpTag>(v, t);
                }
                if (lane == leader) {
                    out2[i] = v;
                } else {
                    out2[i] = detail::op_identity_vec<OpTag, vec_t>();
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                acc_t v = detail::to_acc<elem_t, Recipe>(in[i]);
                for (int offset = Width / 2; offset > 0; offset >>= 1) {
                    acc_t t = detail::shfl_typed_mask<acc_t, axp::level0::shuffle::down>(v, offset, mask);
                    v = detail::op_apply_acc<OpTag>(v, t);
                }
                if (lane == leader) {
                    out[i] = detail::from_acc<elem_t, Recipe>(v);
                } else {
                    out[i] = detail::from_acc<elem_t, Recipe>(detail::op_identity_acc<OpTag, acc_t>());
                }
            }
        }
#else
        (void)in; (void)mask_in; (void)out;
#endif
    }
};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag,
         int BarrierId, int WarpgroupCount>
struct WarpgroupReduce
    : iro::contract::Realization<
        axp::protocol::reduction::WarpgroupReduce<
            Recipe, Payload, Subj, ExecGroup, OpTag, BarrierId, WarpgroupCount>,
        iro::util::fnv1a_64_cstr("axp.realize.l0.reduce.warpgroup")> {
    using storage_t = typename detail::value_traits<Payload>::storage_t;
    using elem_t = typename detail::value_traits<Payload>::elem;
    __device__ static void execute(const storage_t* in, storage_t* out) {
#ifdef __CUDA_ARCH__
        static_assert(iro::exec::is_warpgroup_v<ExecGroup>, "WarpgroupReduce requires warpgroup exec");
        constexpr int N = detail::value_traits<Payload>::count;
        using acc_t = typename detail::acc_traits<elem_t, Recipe>::acc_t;
        constexpr int warps = iro::exec::warpgroup_warps<ExecGroup>::value;
        constexpr int warpgroup_lanes = warps * 32;
        const int lane = threadIdx.x & 31;
        const int warpgroup_lane = threadIdx.x % warpgroup_lanes;
        const int warp_in_group = warpgroup_lane >> 5;
        const int warpgroup_id = threadIdx.x / warpgroup_lanes;
        const int warp_idx = (warpgroup_id * warps) | warp_in_group;
        const uint32_t mask = __activemask();

        constexpr bool vec_ok =
            (detail::is_f16<storage_t>::value || detail::is_bf16<storage_t>::value) &&
            (N % 2) == 0 &&
            std::is_same_v<acc_t, storage_t>;
        if constexpr (vec_ok) {
            using vec_t = std::conditional_t<detail::is_f16<storage_t>::value, __half2, __nv_bfloat162>;
            constexpr int N2 = N / 2;
            __shared__ vec_t warp_vals2[32];
            const vec_t* in2 = reinterpret_cast<const vec_t*>(in);
            vec_t* out2 = reinterpret_cast<vec_t*>(out);
            #pragma unroll
            for (int i = 0; i < N2; ++i) {
                vec_t v = in2[i];
                for (int offset = 16; offset > 0; offset >>= 1) {
                    const vec_t t = detail::shfl_down_typed(v, offset, mask);
                    v = detail::op_apply_vec<OpTag>(v, t);
                }
                if (lane == 0) {
                    warp_vals2[warp_idx] = v;
                }
                detail::warpgroup_sync<BarrierId, warps>();
                if (warp_in_group == 0) {
                    vec_t acc = (lane < warps)
                        ? warp_vals2[(warpgroup_id * warps) + lane]
                        : detail::op_identity_vec<OpTag, vec_t>();
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        const vec_t t = detail::shfl_down_typed(acc, offset, mask);
                        acc = detail::op_apply_vec<OpTag>(acc, t);
                    }
                    if (lane == 0) {
                        out2[i] = acc;
                    }
                }
                detail::warpgroup_sync<BarrierId, warps>();
            }
        } else {
            __shared__ acc_t warp_vals[32];
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                acc_t v = detail::to_acc<elem_t, Recipe>(in[i]);
                for (int offset = 16; offset > 0; offset >>= 1) {
                    const acc_t t = detail::shfl_down_typed(v, offset, mask);
                    v = detail::op_apply_acc<OpTag>(v, t);
                }
                if (lane == 0) {
                    warp_vals[warp_idx] = v;
                }
                detail::warpgroup_sync<BarrierId, warps>();
                if (warp_in_group == 0) {
                    acc_t acc = (lane < warps)
                        ? warp_vals[(warpgroup_id * warps) + lane]
                        : detail::op_identity_acc<OpTag, acc_t>();
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        const acc_t t = detail::shfl_down_typed(acc, offset, mask);
                        acc = detail::op_apply_acc<OpTag>(acc, t);
                    }
                    if (lane == 0) {
                        out[i] = detail::from_acc<elem_t, Recipe>(acc);
                    }
                }
                detail::warpgroup_sync<BarrierId, warps>();
            }
        }
#else
        (void)in; (void)out;
#endif
    }
};

} // namespace axp::realize::l0
