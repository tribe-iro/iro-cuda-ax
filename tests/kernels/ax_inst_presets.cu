/**
 * @file ax_inst_presets.cu
 * @brief Explicit instantiations for public L4 preset configurations.
 */

#include "iro_cuda_ax_core.hpp"
#include <axp/primitives.hpp>

namespace axp_preset_inst {

using namespace axp::l4::preset;
template<class Pattern>
using L3Pattern = axp::l4::lowering::to_l3_pattern_t<Pattern>;

#if defined(AXP_ENABLE_SM89)
template struct axp::level3::registry::resolve_impl<L3Pattern<Gemm16x16x16>, iro::cap::sm89>;
template struct axp::level3::registry::resolve_impl<L3Pattern<Attention16x16>, iro::cap::sm89>;
template struct axp::level3::registry::resolve_impl<L3Pattern<SoftmaxRow4>, iro::cap::sm89>;
template struct axp::level3::registry::resolve_impl<L3Pattern<LayerNorm16x16>, iro::cap::sm89>;
template struct axp::level3::registry::resolve_impl<L3Pattern<RMSNorm16x16>, iro::cap::sm89>;
template struct axp::level3::registry::resolve_impl<L3Pattern<Histogram256>, iro::cap::sm89>;
template struct axp::level3::registry::resolve_impl<L3Pattern<Sort16>, iro::cap::sm89>;
template struct axp::level3::registry::resolve_impl<L3Pattern<VectorizedElementwise16x16>, iro::cap::sm89>;
#endif

#if defined(AXP_ENABLE_SM90)
template struct axp::level3::registry::resolve_impl<L3Pattern<Gemm16x16x16>, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<L3Pattern<Gemm64x64x16>, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<L3Pattern<Gemm64x64x16BiasSiLU>, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<L3Pattern<Attention16x16>, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<L3Pattern<Attention64x64>, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<L3Pattern<Attention64x64x128Decode>, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<L3Pattern<Attention64x64x128Prefill>, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<L3Pattern<SoftmaxRow4>, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<L3Pattern<LayerNorm16x16>, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<L3Pattern<RMSNorm16x16>, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<L3Pattern<Histogram256>, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<L3Pattern<Sort16>, iro::cap::sm90>;
template struct axp::level3::registry::resolve_impl<L3Pattern<VectorizedElementwise16x16>, iro::cap::sm90>;
#endif

} // namespace axp_preset_inst
