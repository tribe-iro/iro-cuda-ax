#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/level3/index.hpp is internal substrate; use axp/l4.hpp in application code."
#endif

#include "registry.hpp"
#include "elementwise.hpp"
#include "gemm.hpp"
#include "attention.hpp"
#include "norm.hpp"
#include "softmax.hpp"
#include "histogram.hpp"
#include "sort.hpp"
#include "streaming.hpp"
#include "scientific.hpp"
