#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level1/sort.hpp"

namespace axp::level2 {

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
using BitonicSort = axp::level1::BitonicSort<Recipe, Payload, InSubj, OutSubj, ExecGroup, CapT>;

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
using BitonicMerge = axp::level1::BitonicMerge<Recipe, Payload, InSubj, OutSubj, ExecGroup, CapT>;

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class CapT = axp::target_cap>
using BitonicMergeCross = axp::level1::BitonicMergeCross<Recipe, Payload, InSubj, OutSubj, ExecGroup, CapT>;

} // namespace axp::level2
