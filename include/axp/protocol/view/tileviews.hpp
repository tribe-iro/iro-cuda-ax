#pragma once

#include "contracts.hpp"
#include "../stage/resources.hpp"

namespace axp::protocol::view {

template<class InTile>
using transposed_tile_t = iro::contract::Tile<
    iro::contract::Shape<InTile::shape::template dim<1>(), InTile::shape::template dim<0>()>,
    typename InTile::elem,
    std::conditional_t<
        detail::is_row_major_v<typename InTile::layout>,
        iro::contract::layout::ColMajor<InTile::shape::template dim<0>()>,
        iro::contract::layout::RowMajor<InTile::shape::template dim<1>()>
    >,
    typename InTile::space,
    typename InTile::align
>;

template<class InTile, class SwizzleAtom>
using swizzled_tile_t = iro::contract::Tile<
    typename InTile::shape,
    typename InTile::elem,
    iro::contract::layout::Swizzled<
        InTile::shape::template dim<1>(),
        SwizzleAtom::B,
        SwizzleAtom::S
    >,
    typename InTile::space,
    typename InTile::align
>;

template<class Recipe, class InTile, class SubjectT, class ExecGroupT, class RequiredTokens, class ProvidedTokens>
using TransposedView = TransposeView<
    Recipe,
    InTile,
    transposed_tile_t<InTile>,
    SubjectT,
    ExecGroupT,
    RequiredTokens,
    ProvidedTokens
>;

template<class Recipe, class InTile, class SubjectT, class ExecGroupT, class RequiredTokens, class ProvidedTokens>
using Swizzled128B = SwizzleView<
    Recipe,
    InTile,
    swizzled_tile_t<InTile, axp::protocol::stage::SwizzleAtom_128B>,
    SubjectT,
    ExecGroupT,
    RequiredTokens,
    ProvidedTokens,
    axp::protocol::stage::SwizzleAtom_128B
>;

template<class Recipe, class InTile, class SubjectT, class ExecGroupT, class RequiredTokens, class ProvidedTokens>
using Swizzled64B = SwizzleView<
    Recipe,
    InTile,
    swizzled_tile_t<InTile, axp::protocol::stage::SwizzleAtom_64B>,
    SubjectT,
    ExecGroupT,
    RequiredTokens,
    ProvidedTokens,
    axp::protocol::stage::SwizzleAtom_64B
>;
}
