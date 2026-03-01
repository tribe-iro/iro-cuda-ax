/**
 * @file ax_inst_presets.cu
 * @brief Explicit instantiations for public L3 preset configurations.
 */

#include <iro_rust_cuda_ffi.h>
#include "iro_cuda_ax_core.hpp"
#include <axp/primitives.hpp>

namespace axp_preset_inst {

using namespace axp::preset;

#if defined(AXP_ENABLE_SM89)
template struct axp::level3::registry::resolve_impl<Gemm16x16x16, iro::cap::sm89>;
template struct axp::level3::registry::resolve_impl<Attention16x16, iro::cap::sm89>;
template struct axp::level3::registry::resolve_impl<LayerNorm16x16, iro::cap::sm89>;
template struct axp::level3::registry::resolve_impl<RMSNorm16x16, iro::cap::sm89>;
template struct axp::level3::registry::resolve_impl<Histogram256, iro::cap::sm89>;
template struct axp::level3::registry::resolve_impl<Sort16, iro::cap::sm89>;
template struct axp::level3::registry::resolve_impl<VectorizedElementwise16x16, iro::cap::sm89>;
#endif

#if defined(AXP_ENABLE_SM90)
template struct axp::level3::registry::resolve_impl<Gemm64x64x16, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<Gemm64x64x16BiasSiLU, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<Attention64x64, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<Attention64x64x128Decode, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<Attention64x64x128Prefill, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<LayerNorm16x16, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<RMSNorm16x16, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<Histogram256, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<Sort16, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<VectorizedElementwise16x16, iro::cap::sm90>;
#endif

} // namespace axp_preset_inst
