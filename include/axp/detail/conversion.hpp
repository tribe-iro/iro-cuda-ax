#pragma once

#include <iro_cuda_ax_core.hpp>
#include "type_traits.hpp"
#include <cmath>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#endif

namespace axp::detail {

#ifdef __CUDACC__
__device__ __forceinline__ float to_f32(float x) { return x; }
__device__ __forceinline__ float to_f32(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float to_f32(__nv_fp8_e4m3 x) { return static_cast<float>(x); }
__device__ __forceinline__ float to_f32(__nv_fp8_e5m2 x) { return static_cast<float>(x); }

__device__ __forceinline__ float from_f32(float x) { return x; }
__device__ __forceinline__ __half from_f32_half(float x) { return __float2half_rn(x); }
__device__ __forceinline__ __nv_bfloat16 from_f32_bf16(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ __nv_fp8_e4m3 from_f32_e4m3(float x) { return __nv_fp8_e4m3(x); }
__device__ __forceinline__ __nv_fp8_e5m2 from_f32_e5m2(float x) { return __nv_fp8_e5m2(x); }
#endif

template<class ElemTag>
inline constexpr bool is_fp8_finite_only_v =
    std::is_same_v<ElemTag, iro::elem::e4m3fn> ||
    std::is_same_v<ElemTag, iro::elem::e5m2fnuz>;

template<class ElemTag>
__device__ __forceinline__ float fp8_max_finite() {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<ElemTag, iro::elem::e4m3> ||
                  std::is_same_v<ElemTag, iro::elem::e4m3fn>) {
        return 448.0f;
    } else if constexpr (std::is_same_v<ElemTag, iro::elem::e5m2> ||
                         std::is_same_v<ElemTag, iro::elem::e5m2fnuz>) {
        return 57344.0f;
    } else {
        return CUDART_INF_F;
    }
#else
    (void)sizeof(ElemTag);
    return 0.0f;
#endif
}

template<class OutT>
__device__ __forceinline__ OutT from_f32_dispatch(float x) {
#ifdef __CUDA_ARCH__
    if constexpr (is_f32<OutT>::value) {
        return from_f32(x);
    } else if constexpr (is_f16<OutT>::value) {
        return from_f32_half(x);
    } else if constexpr (is_bf16<OutT>::value) {
        return from_f32_bf16(x);
    } else if constexpr (is_fp8_e4m3_like<OutT>::value) {
        return from_f32_e4m3(x);
    } else if constexpr (is_fp8_e5m2_like<OutT>::value) {
        return from_f32_e5m2(x);
    } else {
        static_assert(always_false_v<OutT>, "from_f32_dispatch: unsupported type");
        return OutT{};
    }
#else
    (void)x;
    return OutT{};
#endif
}

template<class ElemTag, class Policy>
__device__ __forceinline__ typename ElemTag::storage_t from_f32_policy(float x) {
#ifdef __CUDA_ARCH__
    using out_t = typename ElemTag::storage_t;
    if constexpr (!iro::recipe::is_fp8_elem_v<ElemTag>) {
        return from_f32_dispatch<out_t>(x);
    } else if constexpr (std::is_same_v<Policy, iro::recipe::fp8_native>) {
        return from_f32_dispatch<out_t>(x);
    } else if constexpr (std::is_same_v<Policy, iro::recipe::fp8_saturate>) {
        float y = x;
        if constexpr (is_fp8_finite_only_v<ElemTag>) {
            if (isnan(y)) y = 0.0f;
        }
        const float maxv = fp8_max_finite<ElemTag>();
        y = fminf(fmaxf(y, -maxv), maxv);
        return from_f32_dispatch<out_t>(y);
    } else if constexpr (std::is_same_v<Policy, iro::recipe::fp8_nan_to_zero>) {
        float y = isnan(x) ? 0.0f : x;
        const float maxv = fp8_max_finite<ElemTag>();
        y = fminf(fmaxf(y, -maxv), maxv);
        return from_f32_dispatch<out_t>(y);
    } else {
        static_assert(always_false_v<Policy>, "from_f32_policy: unsupported fp8_policy");
        return out_t{};
    }
#else
    (void)x;
    return typename ElemTag::storage_t{};
#endif
}

template<class ElemTag, class Recipe>
__device__ __forceinline__ typename ElemTag::storage_t from_f32_recipe(float x) {
    return from_f32_policy<ElemTag, typename Recipe::fp8_policy>(x);
}

template<class InT>
__device__ __forceinline__ float to_f32_dispatch(InT x) {
#ifdef __CUDA_ARCH__
    if constexpr (is_f32<InT>::value) {
        return to_f32(x);
    } else if constexpr (is_f16<InT>::value) {
        return to_f32(x);
    } else if constexpr (is_bf16<InT>::value) {
        return to_f32(x);
    } else if constexpr (is_fp8_e4m3_like<InT>::value || is_fp8_e5m2_like<InT>::value) {
        return to_f32(x);
    } else {
        static_assert(always_false_v<InT>, "to_f32_dispatch: unsupported type");
        return 0.0f;
    }
#else
    (void)x;
    return 0.0f;
#endif
}

} // namespace axp::detail
