#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/primitives.hpp is library-only; use axp/prelude.hpp or axp/l4.hpp in application code."
#endif

#include "prelude.hpp"
#include "l4.hpp"
#include "bundles/token_bundles.hpp"
#include "bundles/resource_bundles.hpp"
#include "state.hpp"
#include "swizzle.hpp"
#include "protocol/index.hpp"
#include "level0/index.hpp"
#include "level1/index.hpp"
#include "level2/index.hpp"
#include "level3/index.hpp"
#include "ir/index.hpp"
#include "kits/index.hpp"
#include "graph/index.hpp"
#include "l4/bind_key.hpp"
#include "l4/manifest_enable.hpp"
#include "l4/graph_registry_index.hpp"
#include "l4/resolve.hpp"
