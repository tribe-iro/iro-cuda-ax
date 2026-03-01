#pragma once

#include "../protocol/sync/contracts.hpp"

namespace axp::level0 {

template<class... Args>
using SyncPoint = axp::protocol::sync::SyncPoint<Args...>;

template<class... Args>
using SyncWarp = axp::protocol::sync::SyncWarp<Args...>;

template<class... Args>
using SyncThreads = axp::protocol::sync::SyncThreads<Args...>;

template<class... Args>
using Fence = axp::protocol::sync::Fence<Args...>;

template<class Recipe, class Subject, class ExecGroup, int Expected>
using BarrierInit = axp::protocol::sync::BarrierInit<Recipe, Subject, ExecGroup, Expected>;

template<class Recipe, class Subject, class ExecGroup, int Expected>
using ClusterBarrierInit = axp::protocol::sync::ClusterBarrierInit<Recipe, Subject, ExecGroup, Expected>;

template<class... Args>
using BarrierExpectTx = axp::protocol::sync::BarrierExpectTx<Args...>;

template<class... Args>
using BarrierArriveTx = axp::protocol::sync::BarrierArriveTx<Args...>;

template<class... Args>
using BarrierArrive = axp::protocol::sync::BarrierArrive<Args...>;

template<class... Args>
using ClusterBarrierArrive = axp::protocol::sync::ClusterBarrierArrive<Args...>;

template<class... Args>
using BarrierWait = axp::protocol::sync::BarrierWait<Args...>;

template<class... Args>
using ClusterBarrierWait = axp::protocol::sync::ClusterBarrierWait<Args...>;

template<class... Args>
using BarrierTryWait = axp::protocol::sync::BarrierTryWait<Args...>;

template<class... Args>
using BarrierInvalidate = axp::protocol::sync::BarrierInvalidate<Args...>;

} // namespace axp::level0

namespace iro::contract {

template<class... Args>
struct is_fused_atom<axp::protocol::sync::SyncPoint<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::sync::SyncWarp<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::sync::SyncThreads<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::sync::Fence<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::sync::BarrierInit<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::sync::ClusterBarrierInit<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::sync::BarrierExpectTx<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::sync::BarrierArriveTx<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::sync::BarrierArrive<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::sync::ClusterBarrierArrive<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::sync::BarrierWait<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::sync::ClusterBarrierWait<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::sync::BarrierTryWait<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::sync::BarrierInvalidate<Args...>> : std::true_type {};

} // namespace iro::contract
