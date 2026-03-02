#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level1/passthrough.hpp"
#include "detail/compose.hpp"
#include "registry.hpp"
#include "../bundles/token_bundles.hpp"

namespace axp::level2::scale {

template<class Recipe, class Tile, class ScaleTile, class TileSubj, class ScaleSubj, class ExecGroup,
         class CapT = axp::target_cap>
using ScaleSharedTile = registry::Select<registry::ScaleSharedTilePattern<
    Recipe, Tile, ScaleTile, TileSubj, ScaleSubj, ExecGroup>, CapT>;

} // namespace axp::level2::scale

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level2::registry {

template<class Recipe, class Tile, class ScaleTile, class TileSubj, class ScaleSubj, class ExecGroup, class Cap>
struct resolve_impl<ScaleSharedTilePattern<Recipe, Tile, ScaleTile, TileSubj, ScaleSubj, ExecGroup>, Cap> {
    static constexpr bool supported = true;
    using TileReady = axp::bundle::SmemReadyTx<TileSubj, ExecGroup, iro::token::lifetime::block, Tile::bytes>;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::low::ScaleSharedTile<
            Recipe,
            Tile,
            ScaleTile,
            TileSubj,
            ScaleSubj,
            ExecGroup,
            iro::token::bundle_list<TileReady>,
            iro::util::type_list<>,
            iro::token::bundle_list<TileReady>
        >,
        Cap
    >;
};

} // namespace axp::level2::registry
#endif // AXP_LIBRARY_BUILD
