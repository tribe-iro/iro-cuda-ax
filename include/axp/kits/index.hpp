#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/kits/index.hpp is internal substrate; use axp/l4.hpp in application code."
#endif

#if defined(AXP_ENABLE_SM89)
#include "sm89/index.hpp"
#endif

#if defined(AXP_ENABLE_SM90)
#include "sm90/index.hpp"
#endif

#if defined(AXP_ENABLE_SM100)
#include "sm100/index.hpp"
#endif
