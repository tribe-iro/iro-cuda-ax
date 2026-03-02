#pragma once

#include <iro_cuda_ax_core.hpp>
#include "codes.hpp"

namespace axp::diag {

template<Code C>
struct message {
private:
    static constexpr const char* message_text() {
        switch (C) {
            case Code::MISSING_VISIBLE_AT:
                return "MISSING_VISIBLE_AT: producer did not provide required visible_at token";
            case Code::MISSING_ALIVE:
                return "MISSING_ALIVE: producer did not provide required alive token";
            case Code::MISSING_SYNC_AT:
                return "MISSING_SYNC_AT: required synchronization token is missing";
            case Code::SCOPE_INSUFFICIENT:
                return "SCOPE_INSUFFICIENT: provided token scope does not satisfy required scope";
            case Code::LIFETIME_INSUFFICIENT:
                return "LIFETIME_INSUFFICIENT: token lifetime is narrower than required";
            case Code::LANES_INSUFFICIENT:
                return "LANES_INSUFFICIENT: lane participation token does not satisfy requirement";
            case Code::SLOT_STATE_MISMATCH:
                return "SLOT_STATE_MISMATCH: slot state token does not match required transition";
            case Code::LEASE_MISMATCH:
                return "LEASE_MISMATCH: lease token state does not match expected ownership";
            case Code::TOKEN_LIST_NOT_CANONICAL:
                return "TOKEN_LIST_NOT_CANONICAL: token list is not canonicalized";
            case Code::PAYLOAD_INCOMPATIBLE:
                return "PAYLOAD_INCOMPATIBLE: payload contract is incompatible across edge";
            case Code::SHAPE_MISMATCH:
                return "SHAPE_MISMATCH: shape contract mismatch";
            case Code::ELEM_MISMATCH:
                return "ELEM_MISMATCH: element type contract mismatch";
            case Code::SPACE_MISMATCH:
                return "SPACE_MISMATCH: memory space contract mismatch";
            case Code::LAYOUT_MISMATCH:
                return "LAYOUT_MISMATCH: layout contract mismatch";
            case Code::ALIGNMENT_INSUFFICIENT:
                return "ALIGNMENT_INSUFFICIENT: alignment is weaker than required";
            case Code::DIST_MISMATCH:
                return "DIST_MISMATCH: distribution contract mismatch";
            case Code::RESOURCE_CONFLICT:
                return "RESOURCE_CONFLICT: resource requirements conflict";
            case Code::SMEM_OVERFLOW:
                return "SMEM_OVERFLOW: shared memory budget exceeded";
            case Code::REG_OVERFLOW:
                return "REG_OVERFLOW: register budget exceeded";
            case Code::BARRIER_OVERFLOW:
                return "BARRIER_OVERFLOW: barrier budget exceeded";
            case Code::PORT_SUBJECT_MISMATCH:
                return "PORT_SUBJECT_MISMATCH: connected ports use incompatible subjects";
            case Code::PORT_DIRECTION_ERROR:
                return "PORT_DIRECTION_ERROR: invalid port direction in composition edge";
            case Code::EDGE_INVALID:
                return "EDGE_INVALID: composition edge is invalid";
            case Code::CYCLE_DETECTED:
                return "CYCLE_DETECTED: composition contains a cycle";
            case Code::UNCONNECTED_INPUT:
                return "UNCONNECTED_INPUT: required input is not connected";
            case Code::CAP_UNSUPPORTED:
                return "CAP_UNSUPPORTED: selected target capability is unsupported";
            case Code::SM_VERSION_INSUFFICIENT:
                return "SM_VERSION_INSUFFICIENT: selected target SM version is insufficient";
            case Code::NO_REALIZATION_FOUND:
                return "NO_REALIZATION_FOUND: no realization satisfies obligation and capability";
            case Code::AMBIGUOUS_REALIZATION:
                return "AMBIGUOUS_REALIZATION: more than one realization satisfies selection";
        }
        return "UNKNOWN_DIAGNOSTIC";
    }

public:
    static constexpr const char* text = message_text();
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
