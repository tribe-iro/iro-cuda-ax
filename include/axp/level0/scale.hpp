#pragma once

#include "../protocol/scale/contracts.hpp"

namespace axp::level0 {

template<class Recipe, class Tile, class ScaleTile, class TileSubj, class ScaleSubj, class ExecGroup,
         class TileInExtra = iro::util::type_list<>, class ScaleInExtra = iro::util::type_list<>,
         class OutExtra = iro::util::type_list<>>
struct ScaleSharedTile : axp::protocol::scale::ScaleSharedTile<
    Recipe, Tile, ScaleTile, TileSubj, ScaleSubj, ExecGroup, TileInExtra, ScaleInExtra, OutExtra
> {};

} // namespace axp::level0
