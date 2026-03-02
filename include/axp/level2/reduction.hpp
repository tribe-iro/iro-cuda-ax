#pragma once

#include <iro_cuda_ax_core.hpp>
#include "passthrough.hpp"
#include "../level1/reduction.hpp"

namespace axp::level2 {

template<class Recipe, class Frag, class Subj, class ExecGroup,
         template<class, class, class, class, class, class, class, class> class Op = axp::level2::low::Add,
         class CapT = axp::target_cap>
using WarpReduce = axp::level1::WarpReduce<Recipe, Frag, Subj, ExecGroup, Op, CapT>;

template<class Recipe, class Frag, class SmemTile, class Subj, class SmemSubj, class ExecGroup,
         template<class, class, class, class, class, class, class, class> class Op = axp::level2::low::Add,
         class CapT = axp::target_cap>
using BlockReduce = axp::level1::BlockReduce<Recipe, Frag, SmemTile, Subj, SmemSubj, ExecGroup, Op, CapT>;

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag, class CapT = axp::target_cap>
using WarpAllReduce = axp::level1::WarpAllReduce<Recipe, Frag, Subj, ExecGroup, OpTag, CapT>;

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag, int SegmentWidth, class CapT = axp::target_cap>
using WarpSegmentedReduce = axp::level1::WarpSegmentedReduce<Recipe, Frag, Subj, ExecGroup, OpTag, SegmentWidth, CapT>;

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag,
         int BarrierId = 1, int WarpgroupCount = 1, class CapT = axp::target_cap>
using WarpgroupReduce =
    axp::level1::WarpgroupReduce<Recipe, Frag, Subj, ExecGroup, OpTag, BarrierId, WarpgroupCount, CapT>;

template<class Recipe, class Payload, class MaskPayload, class Subj, class MaskSubj, class ExecGroup, class OpTag,
         class CapT = axp::target_cap>
using ShuffleReduceTree =
    axp::level1::ShuffleReduceTree<Recipe, Payload, MaskPayload, Subj, MaskSubj, ExecGroup, OpTag, CapT>;

} // namespace axp::level2
