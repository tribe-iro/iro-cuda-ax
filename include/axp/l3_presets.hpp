#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/l3_presets.hpp is library-only; use axp/l4.hpp in application code."
#endif

#include "l4.hpp"

// Compatibility shim for pre-GA branch churn.
// Canonical preset surface is axp::l4::preset.
namespace axp {
namespace preset = l4::preset;
}
