#pragma once

#include <iro_cuda_ax_core.hpp>

// Compile-time target capability selection.
// Device builds always use __CUDA_ARCH__. Host builds may define AXP_TARGET_SM.
#if defined(__CUDA_ARCH__)
#undef AXP_TARGET_SM
#define AXP_TARGET_SM __CUDA_ARCH__
#else
#ifndef AXP_TARGET_SM
#error "AXP_TARGET_SM must be defined for host compilation (expected 890, 900, or 1000)."
#endif
#endif

namespace axp {

#if AXP_TARGET_SM < 890
static_assert(AXP_TARGET_SM >= 890,
              "AXP_TARGET_SM must be one of: 890, 900, 1000 (SM89+ only).");
#endif

#if AXP_TARGET_SM >= 1000
#ifdef AXP_ENABLE_SM100
using target_cap = iro::cap::sm100;
#else
static_assert(AXP_TARGET_SM < 1000,
              "AXP target selects SM100 but AXP_ENABLE_SM100 is not defined. "
              "No cross-architecture fallback is allowed.");
#endif
#elif AXP_TARGET_SM >= 900
using target_cap = iro::cap::sm90;
#elif AXP_TARGET_SM >= 890
using target_cap = iro::cap::sm89;
#else
static_assert(AXP_TARGET_SM >= 890,
              "Unsupported AXP_TARGET_SM. Supported lanes: sm89/sm90/sm100.");
#endif

} // namespace axp
