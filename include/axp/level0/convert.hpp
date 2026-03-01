#pragma once

#include "../protocol/convert/contracts.hpp"

namespace axp::level0 {

template<class RecipeIn, class RecipeOut, class InTile, class OutTile, class InSubj, class OutSubj,
         class ExecGroup, int VecBytes, class InDist = iro::contract::no_dist, class OutDist = iro::contract::no_dist>
struct CastTile : axp::protocol::convert::CastTile<
    RecipeIn, RecipeOut, InTile, OutTile, InSubj, OutSubj, ExecGroup, VecBytes, InDist, OutDist
> {};

template<class RecipeIn, class RecipeOut, class InFrag, class OutFrag, class InSubj, class OutSubj, class ExecGroup>
struct CastFragment : axp::protocol::convert::CastFragment<
    RecipeIn, RecipeOut, InFrag, OutFrag, InSubj, OutSubj, ExecGroup
> {};

template<class Recipe, class Frag, class Vec, class FragSubj, class VecSubj, class ExecGroup, int Offset>
struct FragmentToVectorSlice : axp::protocol::convert::FragmentToVectorSlice<
    Recipe, Frag, Vec, FragSubj, VecSubj, ExecGroup, Offset
> {};

template<class Recipe, class Frag, class Vec, class FragInSubj, class VecSubj, class FragOutSubj,
         class ExecGroup, int Offset>
struct VectorSliceToFragment : axp::protocol::convert::VectorSliceToFragment<
    Recipe, Frag, Vec, FragInSubj, VecSubj, FragOutSubj, ExecGroup, Offset
> {};

} // namespace axp::level0
