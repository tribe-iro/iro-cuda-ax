#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../../bundles/token_bundles.hpp"
#include "../../detail/resources.hpp"

namespace axp::protocol::scale {

// Scale vector tile (for block-scaled formats)
template<class ScaleElem, int N,
         class SpaceT = iro::contract::space::global,
         class AlignT = iro::contract::Align<16>>
using ScaleTile = iro::contract::Tile<
    iro::contract::Shape<N>,
    ScaleElem,
    iro::contract::layout::RowMajor<N>,
    SpaceT,
    AlignT>;

// Scale shared tile by scale vector (tile + scale -> tile).
template<class Recipe, class Tile, class ScaleTile, class TileSubj, class ScaleSubj, class ExecGroup,
         class TileInExtra = iro::util::type_list<>, class ScaleInExtra = iro::util::type_list<>,
         class OutExtra = iro::util::type_list<>>
struct ScaleSharedTile {
    static_assert(iro::contract::TilePayload<Tile>, "ScaleSharedTile: Tile must be TilePayload");
    static_assert(iro::contract::TilePayload<ScaleTile>, "ScaleSharedTile: ScaleTile must be TilePayload");
    static_assert(Tile::shape::rank == 1 || Tile::shape::rank == 2,
                  "ScaleSharedTile: Tile must be rank-1 or rank-2");
    static_assert(ScaleTile::shape::rank == 1, "ScaleSharedTile: ScaleTile must be rank-1");
    static_assert(std::is_same_v<typename Tile::space, iro::contract::space::shared>,
                  "ScaleSharedTile: Tile must be shared space");
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>,
                  "ScaleSharedTile: ExecGroup must be block");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "ScaleSharedTile: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Tile,
            TileSubj,
            ExecGroup,
            iro::util::concat_t<axp::bundle::TileInTokens<TileSubj, ExecGroup, iro::token::lifetime::block>, TileInExtra>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            ScaleTile,
            ScaleSubj,
            ExecGroup,
            iro::util::concat_t<axp::bundle::TileInTokens<ScaleSubj, ExecGroup, iro::token::lifetime::block>, ScaleInExtra>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Tile,
            TileSubj,
            ExecGroup,
            iro::util::concat_t<axp::bundle::TileOutTokens<TileSubj, ExecGroup, iro::token::lifetime::block>, OutExtra>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

} // namespace axp::protocol::scale
