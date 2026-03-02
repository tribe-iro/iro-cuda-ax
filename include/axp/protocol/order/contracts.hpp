#pragma once

#include <iro_cuda_ax_core.hpp>
#include "bundles.hpp"
#include "../atomic/bundles.hpp"
#include "../token_policy.hpp"
#include "../../detail/participation_tokens.hpp"

namespace axp::protocol::order {

namespace detail {

template<class Payload>
struct is_supported_payload : std::bool_constant<
    iro::contract::TilePayload<Payload> ||
    iro::contract::FragmentPayload<Payload> ||
    iro::contract::ScalarPayload<Payload> ||
    iro::contract::VectorPayload<Payload> ||
    iro::contract::MaskPayload<Payload>
> {};

template<class Payload>
using payload_dist_t = std::conditional_t<
    iro::contract::TilePayload<Payload>,
    iro::contract::no_dist,
    typename Payload::dist
>;

template<class Subject, class ExecGroup, class Lifetime>
using value_tokens = iro::util::concat_t<
    iro::util::type_list<iro::token::alive<Subject, Lifetime>>,
    axp::detail::participation_tokens<Subject, ExecGroup>
>;

template<class Subject, class ExecGroup, class Lifetime>
using mask_tokens = iro::util::type_list<
    iro::token::alive<Subject, Lifetime>,
    iro::token::mask_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>
>;

template<class Subject, class ExecGroup, class Lifetime>
using tile_tokens_base = iro::util::concat_t<
    iro::util::type_list<
        iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
        iro::token::alive<Subject, Lifetime>
    >,
    axp::detail::participation_tokens<Subject, ExecGroup>
>;

template<class Subject, class ExecGroup, class Lifetime>
using tile_tokens_sync = iro::util::concat_t<
    iro::util::type_list<
        iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
        iro::token::alive<Subject, Lifetime>,
        iro::token::sync_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>
    >,
    axp::detail::participation_tokens<Subject, ExecGroup>
>;

template<class Subject, class ExecGroup, class Lifetime>
using tile_tokens = std::conditional_t<
    (iro::scope::min_scope_for_t<ExecGroup>::level >= iro::scope::warpgroup::level),
    tile_tokens_sync<Subject, ExecGroup, Lifetime>,
    tile_tokens_base<Subject, ExecGroup, Lifetime>
>;

template<class Payload, class Subject, class ExecGroup, class Lifetime>
using payload_tokens = std::conditional_t<
    iro::contract::TilePayload<Payload>,
    tile_tokens<Subject, ExecGroup, Lifetime>,
    std::conditional_t<
        iro::contract::MaskPayload<Payload>,
        mask_tokens<Subject, ExecGroup, Lifetime>,
        value_tokens<Subject, ExecGroup, Lifetime>
    >
>;

template<class Bundle, class Token>
struct bundle_contains : std::false_type {};

template<class Token, class... Ts>
struct bundle_contains<iro::token::bundle<Ts...>, Token>
    : std::bool_constant<iro::util::contains_v<iro::util::type_list<Ts...>, Token>> {};

} // namespace detail

// Publish a root/source ordering event token (handle-only output).
template<class Recipe, class Subject, class EventTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct PublishEvent {
    using emitted_token = event<Subject, EventTag>;
    static_assert(axp::protocol::token_policy::satisfiable_v<emitted_token, emitted_token>,
                  "PublishEvent: event token must be self-satisfiable under token policy");

    using inputs = iro::util::type_list<>;
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OrderHandle,
            Subject,
            ExecGroup,
            iro::util::concat_t<
                iro::token::bundle_list<EventPublished<Subject, EventTag, ExecGroup, Lifetime>>,
                iro::util::type_list<iro::token::issued_by_lane0<Subject>>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using resources = iro::util::type_list<>;
};

// Emit an event token only after an explicit predecessor payload is present.
template<class Recipe, class Payload, class PayloadSubj, class Subject, class EventTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct EmitEventAfter {
    static_assert(detail::is_supported_payload<Payload>::value,
                  "EmitEventAfter: Payload must be Tile/Fragment/Scalar/Vector/Mask");
    using emitted_token = event<Subject, EventTag>;
    static_assert(axp::protocol::token_policy::satisfiable_v<emitted_token, emitted_token>,
                  "EmitEventAfter: emitted event token must be self-satisfiable under token policy");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            PayloadSubj,
            ExecGroup,
            iro::util::concat_t<
                detail::payload_tokens<Payload, PayloadSubj, ExecGroup, Lifetime>,
                InExtra
            >,
            detail::payload_dist_t<Payload>,
            Recipe
        >
    >;
    static_assert(iro::util::size_v<inputs> == 1,
                  "EmitEventAfter: predecessor payload input is required");

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OrderHandle,
            Subject,
            ExecGroup,
            iro::util::concat_t<
                iro::token::bundle_list<EventPublished<Subject, EventTag, ExecGroup, Lifetime>>,
                iro::util::type_list<iro::token::issued_by_lane0<Subject>>
            >,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::OutputPort<
            Payload,
            PayloadSubj,
            ExecGroup,
            iro::util::concat_t<
                detail::payload_tokens<Payload, PayloadSubj, ExecGroup, Lifetime>,
                OutExtra
            >,
            detail::payload_dist_t<Payload>,
            Recipe
        >
    >;
    using resources = iro::util::type_list<>;
};

// Consume a published event and assert a phase-local happens-before edge on the order handle.
template<class Recipe, class Subject, class EventTag, class PhaseTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct DependOnEvent {
    using required_token = event<Subject, EventTag>;
    using produced_event_token = event<Subject, EventTag>;
    using produced_hb_token = happens_before<Subject, PhaseTag>;
    static_assert(axp::protocol::token_policy::satisfiable_v<required_token, produced_event_token>,
                  "DependOnEvent: produced event token must satisfy required event token");
    static_assert(!axp::protocol::token_policy::satisfiable_v<required_token, produced_hb_token>,
                  "DependOnEvent: event token must not be implicitly satisfied by happens-before token");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            OrderHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<EventPublished<Subject, EventTag, ExecGroup, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using output_bundle = EventPhaseReady<Subject, EventTag, PhaseTag, ExecGroup, Lifetime>;
    static_assert(detail::bundle_contains<output_bundle, produced_hb_token>::value,
                  "DependOnEvent: phase token must be produced and forwarded");

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OrderHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<output_bundle>,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using resources = iro::util::type_list<>;
};

// Gate a payload on a phase-qualified order handle and forward payload with explicit phase token.
template<class Recipe, class Payload, class PayloadSubj, class Subject, class EventTag, class PhaseTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct DependOnEventGate {
    static_assert(detail::is_supported_payload<Payload>::value,
                  "DependOnEventGate: Payload must be Tile/Fragment/Scalar/Vector/Mask");
    using required_token = happens_before<Subject, PhaseTag>;
    using published_token = event<Subject, EventTag>;
    using produced_hb_token = happens_before<Subject, PhaseTag>;
    static_assert(!axp::protocol::token_policy::satisfiable_v<published_token, produced_hb_token>,
                  "DependOnEventGate: event token must not be implicitly satisfied by happens-before token");

    using gate_out_tokens = iro::util::concat_t<
        detail::payload_tokens<Payload, PayloadSubj, ExecGroup, Lifetime>,
        iro::util::concat_t<
            iro::util::type_list<happens_before<Subject, PhaseTag>>,
            OutExtra
        >
    >;
    static_assert(iro::util::contains_v<gate_out_tokens, produced_hb_token>,
                  "DependOnEventGate: phase token must be produced and forwarded");
    static_assert(detail::bundle_contains<EventPhaseReady<Subject, EventTag, PhaseTag, ExecGroup, Lifetime>, required_token>::value,
                  "DependOnEventGate: input order handle must be phase-qualified");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            OrderHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<EventPhaseReady<Subject, EventTag, PhaseTag, ExecGroup, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::InputPort<
            Payload,
            PayloadSubj,
            ExecGroup,
            iro::util::concat_t<
                detail::payload_tokens<Payload, PayloadSubj, ExecGroup, Lifetime>,
                InExtra
            >,
            detail::payload_dist_t<Payload>,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            PayloadSubj,
            ExecGroup,
            gate_out_tokens,
            detail::payload_dist_t<Payload>,
            Recipe
        >
    >;
    using resources = iro::util::type_list<>;
};

// Advance an epoch marker for ring/rendezvous style protocols.
template<class Recipe, class Subject, class PrevEpochTag, class NextEpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct AdvanceEpoch {
    using required_token = epoch<Subject, PrevEpochTag>;
    using produced_token = epoch<Subject, NextEpochTag>;
    static_assert(PrevEpochTag::id != NextEpochTag::id,
                  "AdvanceEpoch: PrevEpochTag and NextEpochTag must differ");
    static_assert(!axp::protocol::token_policy::satisfiable_v<required_token, produced_token>,
                  "AdvanceEpoch: next epoch token must not implicitly satisfy previous epoch token");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            OrderHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<EpochStamped<Subject, PrevEpochTag, ExecGroup, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OrderHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<EpochStamped<Subject, NextEpochTag, ExecGroup, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using resources = iro::util::type_list<>;
};

// Emit an event token from an atomic completion token.
template<class Recipe, class Subject, class ScopeT, class EventTag, class ExecGroup,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block>
struct EventFromAtomicDone {
    using required_token = axp::protocol::atomic::atomic_done<Subject, ScopeT>;
    using produced_token = event<Subject, EventTag>;
    static_assert(!axp::protocol::token_policy::satisfiable_v<required_token, produced_token>,
                  "EventFromAtomicDone: atomic completion token must not implicitly satisfy event token");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            axp::protocol::atomic::AtomicHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<axp::protocol::atomic::AtomicDone<Subject, ScopeT, OrderT, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OrderHandle,
            Subject,
            ExecGroup,
            iro::util::concat_t<
                iro::token::bundle_list<EventPublished<Subject, EventTag, ExecGroup, Lifetime>>,
                iro::util::type_list<iro::token::issued_by_lane0<Subject>>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using resources = iro::util::type_list<>;
};

} // namespace axp::protocol::order
