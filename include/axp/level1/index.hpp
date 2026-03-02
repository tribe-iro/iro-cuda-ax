#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/level1/index.hpp is internal substrate; use axp/l4.hpp in application code."
#endif

#include "passthrough.hpp"
#include "reduction.hpp"
#include "scan.hpp"
#include "order.hpp"
#include "atomic.hpp"
#include "epoch.hpp"
#include "row.hpp"
#include "gather.hpp"
#include "io.hpp"
#include "blas.hpp"
#include "communication.hpp"
#include "permute.hpp"
#include "sort.hpp"
#include "mask.hpp"
#include "registry.hpp"
