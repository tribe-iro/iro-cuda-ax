#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level1/mask.hpp"

namespace axp::level2 {

template<class Recipe, class MaskPayload, class PredPayload,
         class QCoordPayload, class KCoordPayload,
         class QCoordSubj, class KCoordSubj,
         class MaskSubj, class PredSubj,
         class ExecGroup, int TileM, int TileN,
         class CapT = axp::target_cap>
using CausalMask = axp::level1::CausalMask<
    Recipe, MaskPayload, PredPayload,
    QCoordPayload, KCoordPayload,
    QCoordSubj, KCoordSubj,
    MaskSubj, PredSubj,
    ExecGroup, TileM, TileN, CapT>;

} // namespace axp::level2
