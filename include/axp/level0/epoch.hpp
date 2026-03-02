#pragma once

#include "../protocol/epoch/contracts.hpp"

namespace axp::level0 {

template<class Recipe, class Subject, class EpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
using InitEpoch = axp::protocol::epoch::InitEpoch<Recipe, Subject, EpochTag, ExecGroup, Lifetime>;

template<class Recipe, class Subject, class PrevEpochTag, class NextEpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
using AdvanceEpoch = axp::protocol::epoch::AdvanceEpoch<
    Recipe, Subject, PrevEpochTag, NextEpochTag, ExecGroup, Lifetime>;

template<class Recipe, class Subject, class EpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
using RequireEpoch = axp::protocol::epoch::RequireEpoch<Recipe, Subject, EpochTag, ExecGroup, Lifetime>;

} // namespace axp::level0

namespace iro::contract {

template<class... Args>
struct is_fused_atom<axp::protocol::epoch::InitEpoch<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::epoch::AdvanceEpoch<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::epoch::RequireEpoch<Args...>> : std::true_type {};

} // namespace iro::contract

