#pragma once

#include "../protocol/atomic/contracts.hpp"

namespace axp::level0 {

template<class Recipe, class Subject, class ExecGroup, class ScopeT,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block>
using MarkAtomicDone = axp::protocol::atomic::MarkAtomicDone<
    Recipe, Subject, ExecGroup, ScopeT, OrderT, Lifetime>;

template<class Recipe, class Subject, class ExecGroup, class ScopeT,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block>
using RequireAtomicDone = axp::protocol::atomic::RequireAtomicDone<
    Recipe, Subject, ExecGroup, ScopeT, OrderT, Lifetime>;

template<class Recipe, class TilePayload, class Subject, class ExecGroup, class ScopeT,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block>
using MarkAtomicDoneFromTile = axp::protocol::atomic::MarkAtomicDoneFromTile<
    Recipe, TilePayload, Subject, ExecGroup, ScopeT, OrderT, Lifetime>;

} // namespace axp::level0

namespace iro::contract {

template<class... Args>
struct is_fused_atom<axp::protocol::atomic::MarkAtomicDone<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::atomic::RequireAtomicDone<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::atomic::MarkAtomicDoneFromTile<Args...>> : std::true_type {};

} // namespace iro::contract
