#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level0/mask.hpp"
#include "registry.hpp"
#include "detail/compose.hpp"

namespace axp::level1 {

// Causal mask + tile-skip predicate (thin L1 wrapper)
template<class Recipe, class MaskPayload, class PredPayload,
         class QCoordPayload, class KCoordPayload,
         class QCoordSubj, class KCoordSubj,
         class MaskSubj, class PredSubj,
         class ExecGroup, int TileM, int TileN,
         class CapT = axp::target_cap>
using CausalMask = registry::Select<
    registry::CausalMaskPattern<
        Recipe, MaskPayload, PredPayload,
        QCoordPayload, KCoordPayload,
        QCoordSubj, KCoordSubj,
        MaskSubj, PredSubj,
        ExecGroup, TileM, TileN>, CapT>;

} // namespace axp::level1

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level1::registry {

template<class Recipe, class MaskPayload, class PredPayload,
         class QCoordPayload, class KCoordPayload,
         class QCoordSubj, class KCoordSubj,
         class MaskSubj, class PredSubj,
         class ExecGroup, int TileM, int TileN,
         class Cap>
struct resolve_impl<CausalMaskPattern<
    Recipe, MaskPayload, PredPayload,
    QCoordPayload, KCoordPayload,
    QCoordSubj, KCoordSubj,
    MaskSubj, PredSubj,
    ExecGroup, TileM, TileN>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level1::detail::as_composition_t<
        axp::level0::CausalMaskPred<
            Recipe, MaskPayload, PredPayload,
            QCoordPayload, KCoordPayload,
            QCoordSubj, KCoordSubj,
            MaskSubj, PredSubj,
            ExecGroup, TileM, TileN>,
        Cap>;
};

} // namespace axp::level1::registry
#endif // AXP_LIBRARY_BUILD
