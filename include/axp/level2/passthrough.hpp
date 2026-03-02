#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/level2/passthrough.hpp is internal substrate; use axp/l4.hpp in application code."
#endif

// Generated file: do not edit manually.
// Source: tools/gen_layer_adapters.cpp

#include "../level1/passthrough.hpp"

namespace axp::level2 {

// Canonical L2 pass-through views over L1 interfaces.
namespace low = axp::level1::low;
namespace proto = axp::level1::proto;

} // namespace axp::level2
