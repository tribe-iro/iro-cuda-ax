#pragma once

#include "../../realize/sm89.hpp"
#include "registry.hpp"

namespace axp::kit::sm89 {

namespace stage {
  template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
           class ExecGroup, class Lifetime, int Slots, class SwizzleAtom = void>
  using IssueGmemToSmemSlot = axp::realize::sm89::IssueGmemToSmemSlot<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom>;
  template<class Recipe, class OutTile, class SlotSubj, class ExecGroup, class Lifetime>
  using WaitSmemSlot = axp::realize::sm89::WaitSmemSlot<Recipe, OutTile, SlotSubj, ExecGroup, Lifetime>;
template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime>
using ReleaseSmemSlot = axp::realize::sm89::ReleaseSmemSlot<Recipe, SlotSubj, ExecGroup, Lifetime>;
  template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime, long long Bytes>
  using MarkConsumed = axp::realize::sm89::MarkConsumed<Recipe, SlotSubj, ExecGroup, Lifetime, Bytes>;
}

namespace compute {
  template<class Recipe, class Shape, class ATile, class BTile, class AccFrag, class ASubj, class BSubj, class AccSubj>
  using WarpMma = axp::realize::sm89::WarpMmaFromSmem<Recipe, Shape, ATile, BTile, AccFrag, ASubj, BSubj, AccSubj>;
}

namespace reduction {
  template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup>
  using BlockReduce = axp::realize::sm89::BlockReduce<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup>;
}

namespace mask {
  template<class Recipe, class MaskFragT, class MaskSubj, class ExecGroup>
  using MaskGen = axp::realize::sm89::MaskGen<Recipe, MaskFragT, MaskSubj, ExecGroup>;
}

namespace ownership {
  template<class Recipe, class InTile, class Frag, class InSubj, class FragSubj, class ExecGroup, class Lifetime>
  using TileToFragment = axp::realize::sm89::TileToFragment<Recipe, InTile, Frag, InSubj, FragSubj, ExecGroup, Lifetime>;
  template<class Recipe, class Frag, class OutTile, class FragSubj, class OutSubj, class ExecGroup, class Lifetime>
  using FragmentToTile = axp::realize::sm89::FragmentToTile<Recipe, Frag, OutTile, FragSubj, OutSubj, ExecGroup, Lifetime>;
}

namespace sync {
  template<class Recipe, class Subject, class ExecGroup>
  using SyncPoint = axp::realize::sm89::SyncPoint<Recipe, Subject, ExecGroup>;
}

namespace convert {
  template<class RecipeIn, class RecipeOut, class InTile, class OutTile,
           class InSubj, class OutSubj, class ExecGroup, int VecBytes,
           class InDist = iro::contract::no_dist, class OutDist = iro::contract::no_dist>
  using CastTile = axp::realize::sm89::CastTile<RecipeIn, RecipeOut, InTile, OutTile, InSubj, OutSubj, ExecGroup, VecBytes, InDist, OutDist>;
  template<class RecipeIn, class RecipeOut, class InFrag, class OutFrag,
           class InSubj, class OutSubj, class ExecGroup>
  using CastFragment = axp::realize::sm89::CastFragment<RecipeIn, RecipeOut, InFrag, OutFrag, InSubj, OutSubj, ExecGroup>;
  template<class Recipe, class Frag, class Vec, class FragSubj, class VecSubj, class ExecGroup, int Offset>
  using FragmentToVectorSlice = axp::realize::sm89::FragmentToVectorSlice<Recipe, Frag, Vec, FragSubj, VecSubj, ExecGroup, Offset>;
  template<class Recipe, class Frag, class Vec, class FragInSubj, class VecSubj, class FragOutSubj,
           class ExecGroup, int Offset>
  using VectorSliceToFragment = axp::realize::sm89::VectorSliceToFragment<Recipe, Frag, Vec, FragInSubj, VecSubj, FragOutSubj, ExecGroup, Offset>;
}

template<class Obligation>
using bind = bind_t<Obligation>;

} // namespace axp::kit::sm89
