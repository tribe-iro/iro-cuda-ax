#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/level2/index.hpp is internal substrate; use axp/l4.hpp in application code."
#endif

#include "row.hpp"
#include "norm.hpp"
#include "attention.hpp"
#include "epilogue.hpp"
#include "matmul.hpp"
#include "wgmma.hpp"
#include "scale.hpp"
#include "staging.hpp"
#include "pipeline.hpp"
#include "registry.hpp"
