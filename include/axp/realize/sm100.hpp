#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/realize/sm100.hpp is internal substrate; use axp/l4.hpp in application code."
#endif

#include <iro_cuda_ax_core.hpp>

namespace axp::realize::sm100 {

// Intentionally empty: SM100 realization entries must be added explicitly.
// Cross-architecture aliasing is forbidden by policy.

} // namespace axp::realize::sm100
