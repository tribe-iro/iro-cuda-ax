#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/level2/index.hpp is internal substrate; use axp/l4.hpp in application code."
#endif

#include "passthrough.hpp"
#include "order.hpp"
#include "atomic.hpp"
#include "epoch.hpp"
#include "communication.hpp"
#include "reduction.hpp"
#include "scan.hpp"
#include "gather.hpp"
#include "sort.hpp"
#include "blas.hpp"
#include "mask.hpp"
#include "staging.hpp"
#include "pipeline.hpp"
#include "matmul.hpp"
#include "wgmma.hpp"
#include "scale.hpp"
#include "registry.hpp"
