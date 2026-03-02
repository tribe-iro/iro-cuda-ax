#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level1/passthrough.hpp"
#include "detail/compose.hpp"

namespace axp::level2::wgmma {

// L2 wrappers for WGMMA control atoms (warpgroup exec)
template<class Recipe, class Subject, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using Fence = axp::level2::detail::as_composition_t<
    axp::level1::low::WgmmaFence<Recipe, Subject, ExecGroup, InExtra, OutExtra>,
    CapT
>;

template<class Recipe, class Subject, class ExecGroup, int Group,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using CommitGroup = axp::level2::detail::as_composition_t<
    axp::level1::low::WgmmaCommitGroup<Recipe, Subject, ExecGroup, Group, InExtra, OutExtra>,
    CapT
>;

template<class Recipe, class Subject, class ExecGroup, int Group,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using WaitGroup = axp::level2::detail::as_composition_t<
    axp::level1::low::WgmmaWaitGroup<Recipe, Subject, ExecGroup, Group, InExtra, OutExtra>,
    CapT
>;

template<class Recipe, class AccFrag, class InSubj, class OutSubj, class WgmmaSubj,
         class ExecGroup, int Group,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using WaitAcc = axp::level2::detail::as_composition_t<
    axp::level1::low::WgmmaWaitAcc<Recipe, AccFrag, InSubj, OutSubj, WgmmaSubj, ExecGroup, Group, InExtra, OutExtra>,
    CapT
>;

} // namespace axp::level2::wgmma
