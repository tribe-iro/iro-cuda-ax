#pragma once

#include "../protocol/reduction/contracts.hpp"

namespace axp::level0::fused {

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag>
using WarpReduce = axp::protocol::reduction::WarpReduce<Recipe, Frag, Subj, ExecGroup, OpTag>;

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag>
using WarpAllReduce = axp::protocol::reduction::WarpAllReduce<Recipe, Frag, Subj, ExecGroup, OpTag>;

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag, int SegmentWidth>
using WarpSegmentedReduce =
    axp::protocol::reduction::WarpSegmentedReduce<Recipe, Frag, Subj, ExecGroup, OpTag, SegmentWidth>;

template<class Recipe, class Payload, class MaskPayload, class Subj, class MaskSubj, class ExecGroup, class OpTag>
using ShuffleReduceTree =
    axp::protocol::reduction::ShuffleReduceTree<Recipe, Payload, MaskPayload, Subj, MaskSubj, ExecGroup, OpTag>;

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag,
         int BarrierId = 1, int WarpgroupCount = 1>
using WarpgroupReduce =
    axp::protocol::reduction::WarpgroupReduce<Recipe, Payload, Subj, ExecGroup, OpTag, BarrierId, WarpgroupCount>;

} // namespace axp::level0::fused

namespace iro::contract {

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag>
struct is_fused_atom<axp::protocol::reduction::WarpReduce<Recipe, Frag, Subj, ExecGroup, OpTag>>
    : std::true_type {};

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag>
struct is_fused_atom<axp::protocol::reduction::WarpAllReduce<Recipe, Frag, Subj, ExecGroup, OpTag>>
    : std::true_type {};

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag, int SegmentWidth>
struct is_fused_atom<axp::protocol::reduction::WarpSegmentedReduce<
    Recipe, Frag, Subj, ExecGroup, OpTag, SegmentWidth>> : std::true_type {};

template<class Recipe, class Payload, class MaskPayload, class Subj, class MaskSubj, class ExecGroup, class OpTag>
struct is_fused_atom<axp::protocol::reduction::ShuffleReduceTree<
    Recipe, Payload, MaskPayload, Subj, MaskSubj, ExecGroup, OpTag>> : std::true_type {};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, int BarrierId, int WarpgroupCount>
struct is_fused_atom<axp::protocol::reduction::WarpgroupReduce<
    Recipe, Payload, Subj, ExecGroup, OpTag, BarrierId, WarpgroupCount>> : std::true_type {};

} // namespace iro::contract
