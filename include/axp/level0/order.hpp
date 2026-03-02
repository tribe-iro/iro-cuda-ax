#pragma once

#include "../protocol/order/contracts.hpp"

namespace axp::level0 {

template<class Recipe, class Subject, class EventTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
using PublishEvent = axp::protocol::order::PublishEvent<Recipe, Subject, EventTag, ExecGroup, Lifetime>;

template<class Recipe, class Payload, class PayloadSubj, class Subject, class EventTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
using EmitEventAfter = axp::protocol::order::EmitEventAfter<
    Recipe, Payload, PayloadSubj, Subject, EventTag, ExecGroup, Lifetime, InExtra, OutExtra>;

template<class Recipe, class Subject, class EventTag, class PhaseTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
using DependOnEvent = axp::protocol::order::DependOnEvent<
    Recipe, Subject, EventTag, PhaseTag, ExecGroup, Lifetime>;

template<class Recipe, class Payload, class PayloadSubj, class Subject, class EventTag, class PhaseTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
using DependOnEventGate = axp::protocol::order::DependOnEventGate<
    Recipe, Payload, PayloadSubj, Subject, EventTag, PhaseTag, ExecGroup, Lifetime, InExtra, OutExtra>;

template<class Recipe, class Subject, class PrevEpochTag, class NextEpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
using AdvanceEpoch = axp::protocol::order::AdvanceEpoch<
    Recipe, Subject, PrevEpochTag, NextEpochTag, ExecGroup, Lifetime>;

template<class Recipe, class Subject, class ScopeT, class EventTag, class ExecGroup,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block>
using EventFromAtomicDone = axp::protocol::order::EventFromAtomicDone<
    Recipe, Subject, ScopeT, EventTag, ExecGroup, OrderT, Lifetime>;

} // namespace axp::level0

namespace iro::contract {

template<class... Args>
struct is_fused_atom<axp::protocol::order::PublishEvent<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::order::EmitEventAfter<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::order::DependOnEvent<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::order::DependOnEventGate<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::order::AdvanceEpoch<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::order::EventFromAtomicDone<Args...>> : std::true_type {};

} // namespace iro::contract
