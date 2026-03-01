#pragma once

#include "../protocol/mask/contracts.hpp"

namespace axp::level0 {

template<class... Args>
using MaskGen = axp::protocol::mask::MaskGen<Args...>;

template<class... Args>
using MaskApply = axp::protocol::mask::MaskApply<Args...>;

template<class Recipe, class MaskPayload, class PredPayload,
         class QCoordPayload, class KCoordPayload,
         class QCoordSubj, class KCoordSubj,
         class MaskSubj, class PredSubj,
         class ExecGroup, int TileM, int TileN>
using CausalMaskPred = axp::protocol::mask::CausalMaskPred<
    Recipe, MaskPayload, PredPayload,
    QCoordPayload, KCoordPayload,
    QCoordSubj, KCoordSubj,
    MaskSubj, PredSubj,
    ExecGroup, TileM, TileN>;

} // namespace axp::level0

namespace iro::contract {

template<class... Args>
struct is_fused_atom<axp::protocol::mask::MaskGen<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::mask::MaskApply<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::mask::CausalMaskPred<Args...>> : std::true_type {};

} // namespace iro::contract
