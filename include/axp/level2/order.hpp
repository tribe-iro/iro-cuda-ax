#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level1/order.hpp"
#include "registry.hpp"
#include "detail/compose.hpp"

namespace axp::level2 {

template<class Recipe, class Subject, class EventTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block,
         class CapT = axp::target_cap>
using PublishEvent = registry::Select<
    registry::PublishEventPattern<Recipe, Subject, EventTag, ExecGroup, Lifetime>, CapT>;

template<class Recipe, class Payload, class PayloadSubj, class Subject, class EventTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using EmitEventAfter = registry::Select<
    registry::EmitEventAfterPattern<
        Recipe, Payload, PayloadSubj, Subject, EventTag, ExecGroup, Lifetime, InExtra, OutExtra>, CapT>;

template<class Recipe, class Subject, class EventTag, class PhaseTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block,
         class CapT = axp::target_cap>
using DependOnEvent = registry::Select<
    registry::DependOnEventPattern<Recipe, Subject, EventTag, PhaseTag, ExecGroup, Lifetime>, CapT>;

template<class Recipe, class Payload, class PayloadSubj, class Subject, class EventTag, class PhaseTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>,
         class CapT = axp::target_cap>
using DependOnEventGate = registry::Select<
    registry::DependOnEventGatePattern<
        Recipe, Payload, PayloadSubj, Subject, EventTag, PhaseTag, ExecGroup, Lifetime, InExtra, OutExtra>, CapT>;

template<class Recipe, class Subject, class PrevEpochTag, class NextEpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block,
         class CapT = axp::target_cap>
using AdvanceEpoch = registry::Select<
    registry::AdvanceEpochPattern<Recipe, Subject, PrevEpochTag, NextEpochTag, ExecGroup, Lifetime>, CapT>;

template<class Recipe, class Subject, class ScopeT, class EventTag, class ExecGroup,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block,
         class CapT = axp::target_cap>
using EventFromAtomicDone = registry::Select<
    registry::EventFromAtomicDonePattern<Recipe, Subject, ScopeT, EventTag, ExecGroup, OrderT, Lifetime>, CapT>;

} // namespace axp::level2

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level2::registry {

template<class Recipe, class Subject, class EventTag, class ExecGroup, class Lifetime, class Cap>
struct resolve_impl<PublishEventPattern<Recipe, Subject, EventTag, ExecGroup, Lifetime>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::PublishEvent<Recipe, Subject, EventTag, ExecGroup, Lifetime, Cap>,
        Cap
    >;
};

template<class Recipe, class Payload, class PayloadSubj, class Subject, class EventTag, class ExecGroup,
         class Lifetime, class InExtra, class OutExtra, class Cap>
struct resolve_impl<
    EmitEventAfterPattern<Recipe, Payload, PayloadSubj, Subject, EventTag, ExecGroup, Lifetime, InExtra, OutExtra>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::EmitEventAfter<
            Recipe, Payload, PayloadSubj, Subject, EventTag, ExecGroup, Lifetime, InExtra, OutExtra, Cap>,
        Cap
    >;
};

template<class Recipe, class Subject, class EventTag, class PhaseTag, class ExecGroup, class Lifetime, class Cap>
struct resolve_impl<DependOnEventPattern<Recipe, Subject, EventTag, PhaseTag, ExecGroup, Lifetime>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::DependOnEvent<Recipe, Subject, EventTag, PhaseTag, ExecGroup, Lifetime, Cap>,
        Cap
    >;
};

template<class Recipe, class Payload, class PayloadSubj, class Subject, class EventTag, class PhaseTag, class ExecGroup,
         class Lifetime, class InExtra, class OutExtra, class Cap>
struct resolve_impl<
    DependOnEventGatePattern<
        Recipe, Payload, PayloadSubj, Subject, EventTag, PhaseTag, ExecGroup, Lifetime, InExtra, OutExtra>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::DependOnEventGate<
            Recipe, Payload, PayloadSubj, Subject, EventTag, PhaseTag, ExecGroup, Lifetime, InExtra, OutExtra, Cap>,
        Cap
    >;
};

template<class Recipe, class Subject, class PrevEpochTag, class NextEpochTag, class ExecGroup, class Lifetime, class Cap>
struct resolve_impl<AdvanceEpochPattern<Recipe, Subject, PrevEpochTag, NextEpochTag, ExecGroup, Lifetime>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::AdvanceEpoch<Recipe, Subject, PrevEpochTag, NextEpochTag, ExecGroup, Lifetime, Cap>,
        Cap
    >;
};

template<class Recipe, class Subject, class ScopeT, class EventTag, class ExecGroup, class OrderT, class Lifetime, class Cap>
struct resolve_impl<EventFromAtomicDonePattern<Recipe, Subject, ScopeT, EventTag, ExecGroup, OrderT, Lifetime>, Cap> {
    static constexpr bool supported = true;
    using type = axp::level2::detail::as_composition_t<
        axp::level1::EventFromAtomicDone<Recipe, Subject, ScopeT, EventTag, ExecGroup, OrderT, Lifetime, Cap>,
        Cap
    >;
};

} // namespace axp::level2::registry
#endif // AXP_LIBRARY_BUILD
