#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level1/blas.hpp"

namespace axp::level2 {

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CapT = axp::target_cap>
using Copy = axp::level1::Copy<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup, CapT>;

} // namespace axp::level2
