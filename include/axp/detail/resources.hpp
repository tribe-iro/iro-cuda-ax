#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::detail {

template<class ExecGroup, bool IsWarpgroup = iro::exec::is_warpgroup_v<ExecGroup>>
struct warpgroup_layout_resources_impl {
    using type = iro::util::type_list<>;
};

template<class ExecGroup>
struct warpgroup_layout_resources_impl<ExecGroup, true> {
    using type = iro::util::type_list<
        iro::contract::res::block_threads_multiple_of<
            iro::exec::warpgroup_warps<ExecGroup>::value * 32>,
        iro::contract::res::warpgroup_layout<
            iro::exec::warpgroup_warps<ExecGroup>::value, 32>,
        iro::contract::res::warpgroup_linear_x
    >;
};

template<class ExecGroup>
struct warpgroup_layout_resources {
    using type = typename warpgroup_layout_resources_impl<ExecGroup>::type;
};

template<class ExecGroup>
using warpgroup_layout_resources_t = typename warpgroup_layout_resources<ExecGroup>::type;

} // namespace axp::detail
