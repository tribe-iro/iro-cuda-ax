#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level2/blas.hpp"
#include "registry.hpp"

namespace axp::level3::elementwise {

template<class Recipe, int TileRows, int TileCols, class InSubj, class OutSubj, class CapT>
struct ElementwiseTileImpl {
    static_assert(std::is_same_v<typename Recipe::in, typename Recipe::out>,
                  "ElementwiseTile: Recipe::in and Recipe::out must match for copy-based realization");
    static_assert(TileRows > 0 && TileCols > 0, "ElementwiseTile: tile dimensions must be positive");

    using Tile = iro::contract::Tile<
        iro::contract::Shape<TileRows, TileCols>,
        typename Recipe::in,
        iro::contract::layout::RowMajor<TileCols>,
        iro::contract::space::global,
        iro::contract::Align<16>
    >;

    using type = axp::level2::Copy<Recipe, Tile, Tile, InSubj, OutSubj, iro::exec::block, CapT>;
};

} // namespace axp::level3::elementwise

namespace axp::level3 {

template<class Recipe, int TileRows, int TileCols, class InSubj, class OutSubj, class CapT = axp::target_cap>
using ElementwiseTile = registry::Select<registry::ElementwiseTilePattern<
    Recipe, TileRows, TileCols, InSubj, OutSubj>, CapT>;

} // namespace axp::level3

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level3::registry {

template<class Recipe, int TileRows, int TileCols, class InSubj, class OutSubj, class Cap>
struct resolve_impl<ElementwiseTilePattern<Recipe, TileRows, TileCols, InSubj, OutSubj>, Cap> {
    static constexpr bool supported = true;
    using type = typename axp::level3::elementwise::ElementwiseTileImpl<
        Recipe, TileRows, TileCols, InSubj, OutSubj, Cap
    >::type;
};

} // namespace axp::level3::registry
#endif // AXP_LIBRARY_BUILD
