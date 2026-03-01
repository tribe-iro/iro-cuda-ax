#pragma once

#include "stage/resources.hpp"
#include "stage/bundles.hpp"
#include "stage/pipeline_contracts.hpp"
#include "stage/async_contracts.hpp"
#include "sync/contracts.hpp"
#include "sync/bundles.hpp"
#include "compute/contracts.hpp"
#include "compute/bundles.hpp"
#include "view/contracts.hpp"
#include "view/tileviews.hpp"
#include "ownership/bundles.hpp"
#include "ownership/contracts.hpp"
#include "tma/contracts.hpp"
#include "scale/contracts.hpp"
#include "reduction/contracts.hpp"
#include "mask/contracts.hpp"
#include "convert/contracts.hpp"

// Experimental / V2 surface (opt-in only)
#if defined(AXP_ENABLE_SM100) || defined(AXP_ENABLE_EXPERIMENTAL_TMEM)
#include "tmem/contracts.hpp"
#endif
