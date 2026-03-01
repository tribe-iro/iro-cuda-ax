#pragma once

#include "../protocol/compute/contracts.hpp"

namespace axp::level0 {

template<class... Args>
using MMA = axp::protocol::compute::WarpMmaFromSmem<Args...>;

template<class... Args>
using WarpMma = axp::protocol::compute::WarpMmaFromSmem<Args...>;

template<class... Args>
using WarpMmaShared = axp::protocol::compute::WarpMmaFromShared<Args...>;

template<class... Args>
using WarpgroupMma = axp::protocol::compute::WarpgroupMmaFromDesc<Args...>;

#if defined(AXP_ENABLE_SM100)
template<class... Args>
using Tcgen05Mma = axp::protocol::compute::Tcgen05Mma<Args...>;
#endif

} // namespace axp::level0

namespace iro::contract {

template<class... Args>
struct is_fused_atom<axp::protocol::compute::WarpMmaFromSmem<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::compute::WarpMmaFromShared<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::compute::WarpgroupMmaFromDesc<Args...>> : std::true_type {};

#if defined(AXP_ENABLE_SM100)
template<class... Args>
struct is_fused_atom<axp::protocol::compute::Tcgen05Mma<Args...>> : std::true_type {};
#endif

} // namespace iro::contract
