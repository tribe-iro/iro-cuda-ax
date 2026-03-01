#pragma once

#include <iro_cuda_ax_core.hpp>
#include "codes.hpp"

namespace axp::diag {

template<Code C>
struct message;

template<>
struct message<Code::TOKEN_MISSING> {
    static constexpr const char* text =
        "TOKEN_MISSING: required token not produced by upstream composition";
};

template<>
struct message<Code::TOKEN_ILLEGAL_TRANSITION> {
    static constexpr const char* text =
        "TOKEN_ILLEGAL_TRANSITION: token state transition violates contract";
};

template<>
struct message<Code::CAPABILITY_UNSATISFIED> {
    static constexpr const char* text =
        "CAPABILITY_UNSATISFIED: selected path requires unavailable target capability";
};

template<>
struct message<Code::OBLIGATION_UNRESOLVED> {
    static constexpr const char* text =
        "OBLIGATION_UNRESOLVED: no deterministic realization satisfies obligation set";
};

template<Code C>
inline constexpr const char* message_v = message<C>::text;

template<Code C, class Producer, class Consumer, class ExpectedState, class ActualState, class TargetCap>
struct context {
    static constexpr Code code = C;
    using producer = Producer;
    using consumer = Consumer;
    using expected_state = ExpectedState;
    using actual_state = ActualState;
    using target_capability = TargetCap;
};

} // namespace axp::diag
