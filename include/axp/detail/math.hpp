#pragma once

#include <iro_cuda_ax_core.hpp>
#include <cmath>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace axp::detail::math {

template<class Recipe>
__device__ __forceinline__ float expf_recipe(float x) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<typename Recipe::math, iro::recipe::Exact>) {
        return expf(x);
    } else {
        return __expf(x);
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

} // namespace axp::detail::math
