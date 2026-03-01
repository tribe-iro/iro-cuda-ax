#pragma once

#include <type_traits>
#include <iro_cuda_ax_core.hpp>
#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#endif

namespace axp::detail {

template<class T> struct is_elem_f16 : std::false_type {};
template<> struct is_elem_f16<iro::elem::f16> : std::true_type {};
template<class T> struct is_elem_bf16 : std::false_type {};
template<> struct is_elem_bf16<iro::elem::bf16> : std::true_type {};
template<class T> struct is_elem_tf32 : std::false_type {};
template<> struct is_elem_tf32<iro::elem::tf32> : std::true_type {};
template<class T> struct is_elem_f32 : std::false_type {};
template<> struct is_elem_f32<iro::elem::f32> : std::true_type {};
template<class T> struct is_elem_fp8_e4m3 : std::false_type {};
template<> struct is_elem_fp8_e4m3<iro::elem::e4m3> : std::true_type {};
template<> struct is_elem_fp8_e4m3<iro::elem::e4m3fn> : std::true_type {};
template<class T> struct is_elem_fp8_e5m2 : std::false_type {};
template<> struct is_elem_fp8_e5m2<iro::elem::e5m2> : std::true_type {};
template<> struct is_elem_fp8_e5m2<iro::elem::e5m2fnuz> : std::true_type {};

template<class T> inline constexpr bool is_elem_f16_v = is_elem_f16<T>::value;
template<class T> inline constexpr bool is_elem_bf16_v = is_elem_bf16<T>::value;
template<class T> inline constexpr bool is_elem_tf32_v = is_elem_tf32<T>::value;
template<class T> inline constexpr bool is_elem_f32_v = is_elem_f32<T>::value;
template<class T> inline constexpr bool is_elem_fp8_e4m3_v = is_elem_fp8_e4m3<T>::value;
template<class T> inline constexpr bool is_elem_fp8_e5m2_v = is_elem_fp8_e5m2<T>::value;
template<class T> inline constexpr bool is_elem_fp8_v = is_elem_fp8_e4m3_v<T> || is_elem_fp8_e5m2_v<T>;

template<class T> struct is_f32 : std::false_type {};
template<> struct is_f32<float> : std::true_type {};

#ifdef __CUDACC__
template<class T> struct is_f16 : std::false_type {};
template<> struct is_f16<__half> : std::true_type {};
template<class T> struct is_bf16 : std::false_type {};
template<> struct is_bf16<__nv_bfloat16> : std::true_type {};
template<class T> struct is_fp8_e4m3 : std::false_type {};
template<> struct is_fp8_e4m3<__nv_fp8_e4m3> : std::true_type {};
template<class T> struct is_fp8_e4m3fn : is_fp8_e4m3<T> {};
template<class T> struct is_fp8_e5m2 : std::false_type {};
template<> struct is_fp8_e5m2<__nv_fp8_e5m2> : std::true_type {};
template<class T> struct is_fp8_e5m2fnuz : is_fp8_e5m2<T> {};

template<class T>
struct is_fp8_e4m3_like : std::bool_constant<is_fp8_e4m3<T>::value || is_fp8_e4m3fn<T>::value> {};

template<class T>
struct is_fp8_e5m2_like : std::bool_constant<is_fp8_e5m2<T>::value || is_fp8_e5m2fnuz<T>::value> {};
#endif

template<class T>
inline constexpr bool is_supported_elem_v =
    is_f32<T>::value
#ifdef __CUDACC__
    || is_f16<T>::value || is_bf16<T>::value ||
       is_fp8_e4m3<T>::value || is_fp8_e4m3fn<T>::value ||
       is_fp8_e5m2<T>::value || is_fp8_e5m2fnuz<T>::value
#endif
    ;

template<class T>
inline constexpr bool always_false_v = false;

} // namespace axp::detail
