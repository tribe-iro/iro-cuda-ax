#pragma once

#include <iro_cuda_ax_core.hpp>
#include "detail/compose.hpp"
#include "../level0/compute.hpp"

namespace axp::level2::wgmma {

// L2 wrappers for WGMMA control atoms (warpgroup exec)
template<class Recipe, class Subject, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using Fence = axp::level2::detail::as_composition_t<
    axp::level0::WgmmaFence<Recipe, Subject, ExecGroup, InExtra, OutExtra>,
    CapT
>;

template<class Recipe, class Subject, class ExecGroup, int Group,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using CommitGroup = axp::level2::detail::as_composition_t<
    axp::level0::WgmmaCommitGroup<Recipe, Subject, ExecGroup, Group, InExtra, OutExtra>,
    CapT
>;

template<class Recipe, class Subject, class ExecGroup, int Group,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using WaitGroup = axp::level2::detail::as_composition_t<
    axp::level0::WgmmaWaitGroup<Recipe, Subject, ExecGroup, Group, InExtra, OutExtra>,
    CapT
>;

template<class Recipe, class AccFrag, class InSubj, class OutSubj, class WgmmaSubj,
         class ExecGroup, int Group,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using WaitAcc = axp::level2::detail::as_composition_t<
    axp::level0::WgmmaWaitAcc<Recipe, AccFrag, InSubj, OutSubj, WgmmaSubj, ExecGroup, Group, InExtra, OutExtra>,
    CapT
>;

} // namespace axp::level2::wgmma
