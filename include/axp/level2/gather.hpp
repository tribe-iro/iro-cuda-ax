#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level1/gather.hpp"

namespace axp::level2 {

template<class Recipe, class InTile, class IndexPayload, class OutPayload,
         class InSubj, class IndexSubj, class OutSubj, class ExecGroup,
         class CachePolicy = axp::cache::ca,
         class CapT = axp::target_cap>
using GatherGlobal = axp::level1::GatherGlobal<
    Recipe, InTile, IndexPayload, OutPayload, InSubj, IndexSubj, OutSubj, ExecGroup, CachePolicy, CapT>;

template<class Recipe, class InPayload, class IndexPayload, class OutTile,
         class InSubj, class IndexSubj, class OutSubj, class ExecGroup,
         class CachePolicy = axp::cache::wb,
         class CapT = axp::target_cap>
using ScatterGlobal = axp::level1::ScatterGlobal<
    Recipe, InPayload, IndexPayload, OutTile, InSubj, IndexSubj, OutSubj, ExecGroup, CachePolicy, CapT>;

} // namespace axp::level2
