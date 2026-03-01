#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../../../detail/type_traits.hpp"

namespace axp::protocol::compute::detail {

using axp::detail::is_elem_f16_v;
using axp::detail::is_elem_bf16_v;
using axp::detail::is_elem_tf32_v;
using axp::detail::is_elem_f32_v;
using axp::detail::is_elem_fp8_e4m3_v;
using axp::detail::is_elem_fp8_e5m2_v;
using axp::detail::is_elem_fp8_v;

template<class ElemA, class ElemB>
constexpr bool both_f16 = is_elem_f16_v<ElemA> && is_elem_f16_v<ElemB>;

template<class ElemA, class ElemB>
constexpr bool both_bf16 = is_elem_bf16_v<ElemA> && is_elem_bf16_v<ElemB>;

template<class ElemA, class ElemB>
constexpr bool both_tf32 = is_elem_tf32_v<ElemA> && is_elem_tf32_v<ElemB>;

template<class ElemA, class ElemB>
constexpr bool both_fp8_e4m3 = is_elem_fp8_e4m3_v<ElemA> && is_elem_fp8_e4m3_v<ElemB>;

template<class ElemA, class ElemB>
constexpr bool both_fp8_e5m2 = is_elem_fp8_e5m2_v<ElemA> && is_elem_fp8_e5m2_v<ElemB>;

template<class ElemA, class ElemB>
constexpr bool both_fp8 = is_elem_fp8_v<ElemA> && is_elem_fp8_v<ElemB>;

template<int M, int N, int K, class ElemA, class ElemB, class Acc>
constexpr bool is_wmma_shape_v =
    // WMMA F16/BF16: m16n16k16
    ((M == 16 && N == 16 && K == 16 &&
      (both_f16<ElemA, ElemB> || both_bf16<ElemA, ElemB>) &&
      is_elem_f32_v<Acc>) ||
     // WMMA TF32: m16n16k8
     (M == 16 && N == 16 && K == 8 &&
      both_tf32<ElemA, ElemB> &&
      is_elem_f32_v<Acc>));

template<int M, int N, int K, class ElemA, class ElemB, class Acc>
constexpr bool is_wgmma_shape_v =
    (M == 64 && (N % 8 == 0) && (N >= 8) && (N <= 256) &&
     (
      // WGMMA F16/BF16: k16
      ((K == 16) && (both_f16<ElemA, ElemB> || both_bf16<ElemA, ElemB>) && is_elem_f32_v<Acc>) ||
      // WGMMA TF32: k8
      ((K == 8) && both_tf32<ElemA, ElemB> && is_elem_f32_v<Acc>) ||
      // WGMMA FP8: k32 (allow mixed e4m3/e5m2)
      ((K == 32) && both_fp8<ElemA, ElemB> && is_elem_f32_v<Acc>)
     ));

template<int M, int N, int K, class ElemA, class ElemB, class Acc>
constexpr bool is_tcgen05_shape_v = false;

template<int M, int N, int K, class ElemA, class ElemB, class Acc>
constexpr bool is_valid_mma_shape_v =
    is_wmma_shape_v<M, N, K, ElemA, ElemB, Acc> ||
    is_wgmma_shape_v<M, N, K, ElemA, ElemB, Acc> ||
    is_tcgen05_shape_v<M, N, K, ElemA, ElemB, Acc>;

} // namespace axp::protocol::compute::detail
