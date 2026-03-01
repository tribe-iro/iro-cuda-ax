#pragma once

#include "../../realize/sm90.hpp"
#include "registry.hpp"

namespace axp::kit::sm90 {

namespace stage {
  template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
           class ExecGroup, class Lifetime, int Slots, class SwizzleAtom = void>
  using IssueGmemToSmemSlot = axp::realize::sm90::IssueGmemToSmemSlot<Recipe, InTile, OutTile, InSubj, PipeTag, SlotSubj, ExecGroup, Lifetime, Slots, SwizzleAtom>;
  template<class Recipe, class OutTile, class SlotSubj, class ExecGroup, class Lifetime>
  using WaitSmemSlot = axp::realize::sm90::WaitSmemSlot<Recipe, OutTile, SlotSubj, ExecGroup, Lifetime>;
template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime>
using ReleaseSmemSlot = axp::realize::sm90::ReleaseSmemSlot<Recipe, SlotSubj, ExecGroup, Lifetime>;
  template<class Recipe, class SlotSubj, class ExecGroup, class Lifetime, long long Bytes>
  using MarkConsumed = axp::realize::sm90::MarkConsumed<Recipe, SlotSubj, ExecGroup, Lifetime, Bytes>;
}

namespace compute {
  template<class Recipe, class Shape, class ADesc, class BDesc, class AccFrag,
           class ADescSubj, class BDescSubj, class AccSubj, class WgmmaSubj>
  using WarpgroupMma = axp::realize::sm90::WarpgroupMmaFromDesc<Recipe, Shape, ADesc, BDesc, AccFrag, ADescSubj, BDescSubj, AccSubj, WgmmaSubj>;
}

namespace reduction {
  template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup>
  using BlockReduce = axp::realize::sm90::BlockReduce<Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup>;
}

namespace mask {
  template<class Recipe, class MaskFragT, class MaskSubj, class ExecGroup>
  using MaskGen = axp::realize::sm90::MaskGen<Recipe, MaskFragT, MaskSubj, ExecGroup>;
}

namespace ownership {
  template<class Recipe, class SmemTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime, class SwizzleAtom>
  using MakeWgmmaSmemDesc = axp::realize::sm90::MakeWgmmaSmemDesc<Recipe, SmemTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom>;
}

namespace sync {
  template<class Recipe, class Subject, class ExecGroup>
  using SyncPoint = axp::realize::sm90::SyncPoint<Recipe, Subject, ExecGroup>;
}

namespace convert {
  template<class RecipeIn, class RecipeOut, class InTile, class OutTile,
           class InSubj, class OutSubj, class ExecGroup, int VecBytes,
           class InDist = iro::contract::no_dist, class OutDist = iro::contract::no_dist>
  using CastTile = axp::realize::sm90::CastTile<RecipeIn, RecipeOut, InTile, OutTile, InSubj, OutSubj, ExecGroup, VecBytes, InDist, OutDist>;
  template<class RecipeIn, class RecipeOut, class InFrag, class OutFrag,
           class InSubj, class OutSubj, class ExecGroup>
  using CastFragment = axp::realize::sm90::CastFragment<RecipeIn, RecipeOut, InFrag, OutFrag, InSubj, OutSubj, ExecGroup>;
  template<class Recipe, class Frag, class Vec, class FragSubj, class VecSubj, class ExecGroup, int Offset>
  using FragmentToVectorSlice = axp::realize::sm90::FragmentToVectorSlice<Recipe, Frag, Vec, FragSubj, VecSubj, ExecGroup, Offset>;
  template<class Recipe, class Frag, class Vec, class FragInSubj, class VecSubj, class FragOutSubj,
           class ExecGroup, int Offset>
  using VectorSliceToFragment = axp::realize::sm90::VectorSliceToFragment<Recipe, Frag, Vec, FragInSubj, VecSubj, FragOutSubj, ExecGroup, Offset>;
}

template<class Obligation>
using bind = bind_t<Obligation>;

} // namespace axp::kit::sm90
