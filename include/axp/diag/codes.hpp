#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::diag {

enum class Code : std::uint32_t {
    // Token and protocol diagnostics.
    MISSING_VISIBLE_AT = 0xA401,
    MISSING_ALIVE = 0xA402,
    MISSING_SYNC_AT = 0xA403,
    SCOPE_INSUFFICIENT = 0xA404,
    LIFETIME_INSUFFICIENT = 0xA405,
    LANES_INSUFFICIENT = 0xA406,
    SLOT_STATE_MISMATCH = 0xA407,
    LEASE_MISMATCH = 0xA408,
    TOKEN_LIST_NOT_CANONICAL = 0xA409,

    // Payload diagnostics.
    PAYLOAD_INCOMPATIBLE = 0xA420,
    SHAPE_MISMATCH = 0xA421,
    ELEM_MISMATCH = 0xA422,
    SPACE_MISMATCH = 0xA423,
    LAYOUT_MISMATCH = 0xA424,
    ALIGNMENT_INSUFFICIENT = 0xA425,
    DIST_MISMATCH = 0xA426,

    // Resource diagnostics.
    RESOURCE_CONFLICT = 0xA430,
    SMEM_OVERFLOW = 0xA431,
    REG_OVERFLOW = 0xA432,
    BARRIER_OVERFLOW = 0xA433,

    // Port and composition diagnostics.
    PORT_SUBJECT_MISMATCH = 0xA440,
    PORT_DIRECTION_ERROR = 0xA441,
    EDGE_INVALID = 0xA450,
    CYCLE_DETECTED = 0xA451,
    UNCONNECTED_INPUT = 0xA452,

    // Capability and resolution diagnostics.
    CAP_UNSUPPORTED = 0xA460,
    SM_VERSION_INSUFFICIENT = 0xA461,
    NO_REALIZATION_FOUND = 0xA470,
    AMBIGUOUS_REALIZATION = 0xA471,
};

namespace detail {

constexpr const char* stable_id_token(Code code) {
    switch (code) {
        case Code::MISSING_VISIBLE_AT: return "axp.diag.MISSING_VISIBLE_AT";
        case Code::MISSING_ALIVE: return "axp.diag.MISSING_ALIVE";
        case Code::MISSING_SYNC_AT: return "axp.diag.MISSING_SYNC_AT";
        case Code::SCOPE_INSUFFICIENT: return "axp.diag.SCOPE_INSUFFICIENT";
        case Code::LIFETIME_INSUFFICIENT: return "axp.diag.LIFETIME_INSUFFICIENT";
        case Code::LANES_INSUFFICIENT: return "axp.diag.LANES_INSUFFICIENT";
        case Code::SLOT_STATE_MISMATCH: return "axp.diag.SLOT_STATE_MISMATCH";
        case Code::LEASE_MISMATCH: return "axp.diag.LEASE_MISMATCH";
        case Code::TOKEN_LIST_NOT_CANONICAL: return "axp.diag.TOKEN_LIST_NOT_CANONICAL";
        case Code::PAYLOAD_INCOMPATIBLE: return "axp.diag.PAYLOAD_INCOMPATIBLE";
        case Code::SHAPE_MISMATCH: return "axp.diag.SHAPE_MISMATCH";
        case Code::ELEM_MISMATCH: return "axp.diag.ELEM_MISMATCH";
        case Code::SPACE_MISMATCH: return "axp.diag.SPACE_MISMATCH";
        case Code::LAYOUT_MISMATCH: return "axp.diag.LAYOUT_MISMATCH";
        case Code::ALIGNMENT_INSUFFICIENT: return "axp.diag.ALIGNMENT_INSUFFICIENT";
        case Code::DIST_MISMATCH: return "axp.diag.DIST_MISMATCH";
        case Code::RESOURCE_CONFLICT: return "axp.diag.RESOURCE_CONFLICT";
        case Code::SMEM_OVERFLOW: return "axp.diag.SMEM_OVERFLOW";
        case Code::REG_OVERFLOW: return "axp.diag.REG_OVERFLOW";
        case Code::BARRIER_OVERFLOW: return "axp.diag.BARRIER_OVERFLOW";
        case Code::PORT_SUBJECT_MISMATCH: return "axp.diag.PORT_SUBJECT_MISMATCH";
        case Code::PORT_DIRECTION_ERROR: return "axp.diag.PORT_DIRECTION_ERROR";
        case Code::EDGE_INVALID: return "axp.diag.EDGE_INVALID";
        case Code::CYCLE_DETECTED: return "axp.diag.CYCLE_DETECTED";
        case Code::UNCONNECTED_INPUT: return "axp.diag.UNCONNECTED_INPUT";
        case Code::CAP_UNSUPPORTED: return "axp.diag.CAP_UNSUPPORTED";
        case Code::SM_VERSION_INSUFFICIENT: return "axp.diag.SM_VERSION_INSUFFICIENT";
        case Code::NO_REALIZATION_FOUND: return "axp.diag.NO_REALIZATION_FOUND";
        case Code::AMBIGUOUS_REALIZATION: return "axp.diag.AMBIGUOUS_REALIZATION";
    }
    return "axp.diag.UNKNOWN";
}

constexpr iro::diag::Code to_core(Code code) {
    switch (code) {
        case Code::MISSING_VISIBLE_AT: return iro::diag::Code::MISSING_VISIBLE_AT;
        case Code::MISSING_ALIVE: return iro::diag::Code::MISSING_ALIVE;
        case Code::MISSING_SYNC_AT: return iro::diag::Code::MISSING_SYNC_AT;
        case Code::SCOPE_INSUFFICIENT: return iro::diag::Code::SCOPE_INSUFFICIENT;
        case Code::LIFETIME_INSUFFICIENT: return iro::diag::Code::LIFETIME_INSUFFICIENT;
        case Code::LANES_INSUFFICIENT: return iro::diag::Code::LANES_INSUFFICIENT;
        case Code::SLOT_STATE_MISMATCH: return iro::diag::Code::SLOT_STATE_MISMATCH;
        case Code::LEASE_MISMATCH: return iro::diag::Code::LEASE_MISMATCH;
        case Code::TOKEN_LIST_NOT_CANONICAL: return iro::diag::Code::TOKEN_LIST_NOT_CANONICAL;
        case Code::PAYLOAD_INCOMPATIBLE: return iro::diag::Code::PAYLOAD_INCOMPATIBLE;
        case Code::SHAPE_MISMATCH: return iro::diag::Code::SHAPE_MISMATCH;
        case Code::ELEM_MISMATCH: return iro::diag::Code::ELEM_MISMATCH;
        case Code::SPACE_MISMATCH: return iro::diag::Code::SPACE_MISMATCH;
        case Code::LAYOUT_MISMATCH: return iro::diag::Code::LAYOUT_MISMATCH;
        case Code::ALIGNMENT_INSUFFICIENT: return iro::diag::Code::ALIGNMENT_INSUFFICIENT;
        case Code::DIST_MISMATCH: return iro::diag::Code::DIST_MISMATCH;
        case Code::RESOURCE_CONFLICT: return iro::diag::Code::RESOURCE_CONFLICT;
        case Code::SMEM_OVERFLOW: return iro::diag::Code::SMEM_OVERFLOW;
        case Code::REG_OVERFLOW: return iro::diag::Code::REG_OVERFLOW;
        case Code::BARRIER_OVERFLOW: return iro::diag::Code::BARRIER_OVERFLOW;
        case Code::PORT_SUBJECT_MISMATCH: return iro::diag::Code::PORT_SUBJECT_MISMATCH;
        case Code::PORT_DIRECTION_ERROR: return iro::diag::Code::PORT_DIRECTION_ERROR;
        case Code::EDGE_INVALID: return iro::diag::Code::EDGE_INVALID;
        case Code::CYCLE_DETECTED: return iro::diag::Code::CYCLE_DETECTED;
        case Code::UNCONNECTED_INPUT: return iro::diag::Code::UNCONNECTED_INPUT;
        case Code::CAP_UNSUPPORTED: return iro::diag::Code::CAP_UNSUPPORTED;
        case Code::SM_VERSION_INSUFFICIENT: return iro::diag::Code::SM_VERSION_INSUFFICIENT;
        case Code::NO_REALIZATION_FOUND: return iro::diag::Code::NO_REALIZATION_FOUND;
        case Code::AMBIGUOUS_REALIZATION: return iro::diag::Code::AMBIGUOUS_REALIZATION;
    }
    return iro::diag::Code::NO_REALIZATION_FOUND;
}

} // namespace detail

template<Code C>
struct stable_id {
    static constexpr iro::util::u64 value = iro::util::fnv1a_64_cstr(detail::stable_id_token(C));
};

template<Code C>
inline constexpr iro::util::u64 stable_id_v = stable_id<C>::value;

template<Code C>
struct to_core_code {
    static constexpr iro::diag::Code value = detail::to_core(C);
};

template<Code C>
using core_diagnostic_t = iro::diag::Diagnostic<to_core_code<C>::value, stable_id_v<C>>;

template<Code C>
using diagnostic_t = core_diagnostic_t<C>;

} // namespace axp::diag
