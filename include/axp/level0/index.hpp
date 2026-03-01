#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/level0/index.hpp is internal substrate; use axp/l4.hpp in application code."
#endif

#include <iro_cuda_ax_core.hpp>

#include "compute.hpp"
#include "memory.hpp"
#include "communication.hpp"
#include "reduction.hpp"
#include "scan.hpp"
#include "fragment.hpp"
#include "specialize.hpp"

#include "stage.hpp"
#include "sync.hpp"
#include "compute_alias.hpp"
#include "view.hpp"
#include "ownership.hpp"
#include "convert.hpp"
#include "mask.hpp"
#include "tma.hpp"
#include "scale.hpp"

#if defined(AXP_ENABLE_SM100) || defined(AXP_ENABLE_EXPERIMENTAL_TMEM)
#include "tmem.hpp"
#endif
