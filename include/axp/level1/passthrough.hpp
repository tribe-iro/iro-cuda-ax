#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/level1/passthrough.hpp is internal substrate; use axp/l4.hpp in application code."
#endif

// Generated file: do not edit manually.
// Source: tools/gen_layer_adapters.cpp

#include "../level0/index.hpp"
#include "../protocol/index.hpp"

namespace axp::level1 {

// Canonical L1 pass-through views over L0/protocol atoms.
namespace low = axp::level0;
namespace proto = axp::protocol;

} // namespace axp::level1
