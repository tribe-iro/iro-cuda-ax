#pragma once

#include "../protocol/tma/contracts.hpp"

namespace axp::level0 {

template<class... Args>
using TensorMapHandle = axp::protocol::tma::TensorMapHandle<Args...>;

template<class... Args>
using HostMakeTensorMap = axp::protocol::tma::HostMakeTensorMap<Args...>;

template<class... Args>
using BulkTmaCopy2D = axp::protocol::tma::BulkTmaCopy2D<Args...>;

template<class... Args>
using BulkTmaCopy1D = axp::protocol::tma::BulkTmaCopy1D<Args...>;

template<class... Args>
using BulkTmaCopyMulticast = typename axp::protocol::tma::BulkTmaCopyMulticast<Args...>::type;

template<class... Args>
using BulkTmaStore2D = axp::protocol::tma::BulkTmaStore2D<Args...>;

template<class... Args>
using BulkTmaStore1D = axp::protocol::tma::BulkTmaStore1D<Args...>;

} // namespace axp::level0

namespace iro::contract {

template<class... Args>
struct is_fused_atom<axp::protocol::tma::HostMakeTensorMap<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::tma::BulkTmaCopy2D<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::tma::BulkTmaCopy1D<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::tma::BulkTmaCopyMulticast1D<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::tma::BulkTmaCopyMulticast2D<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::tma::BulkTmaStore2D<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::tma::BulkTmaStore1D<Args...>> : std::true_type {};

} // namespace iro::contract
