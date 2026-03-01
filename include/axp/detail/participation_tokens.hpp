#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::detail {

template<class Subject, class ExecGroup>
using participation_tokens = std::conditional_t<
    std::is_same_v<ExecGroup, iro::exec::warp>,
    iro::util::type_list<iro::token::lanes_valid<Subject, 32>>,
    std::conditional_t<
        iro::exec::is_warpgroup_v<ExecGroup>,
        iro::util::type_list<
            iro::token::lanes_valid<Subject, 32>,
            iro::token::warps_valid<Subject, iro::exec::warpgroup_warps<ExecGroup>::value>,
            iro::token::warpgroup_participates<Subject, iro::exec::warpgroup_warps<ExecGroup>::value>
        >,
        iro::util::type_list<>
    >
>;

} // namespace axp::detail
