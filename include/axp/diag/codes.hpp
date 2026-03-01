#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::diag {

enum class Code : std::uint32_t {
    TOKEN_MISSING = 0xA100,
    TOKEN_ILLEGAL_TRANSITION = 0xA101,
    CAPABILITY_UNSATISFIED = 0xA200,
    OBLIGATION_UNRESOLVED = 0xA300,
};

template<Code C>
struct stable_id;

template<>
struct stable_id<Code::TOKEN_MISSING> {
    static constexpr iro::util::u64 value = iro::util::fnv1a_64_cstr("axp.diag.TOKEN_MISSING");
};

template<>
struct stable_id<Code::TOKEN_ILLEGAL_TRANSITION> {
    static constexpr iro::util::u64 value =
        iro::util::fnv1a_64_cstr("axp.diag.TOKEN_ILLEGAL_TRANSITION");
};

template<>
struct stable_id<Code::CAPABILITY_UNSATISFIED> {
    static constexpr iro::util::u64 value =
        iro::util::fnv1a_64_cstr("axp.diag.CAPABILITY_UNSATISFIED");
};

template<>
struct stable_id<Code::OBLIGATION_UNRESOLVED> {
    static constexpr iro::util::u64 value =
        iro::util::fnv1a_64_cstr("axp.diag.OBLIGATION_UNRESOLVED");
};

template<Code C>
inline constexpr iro::util::u64 stable_id_v = stable_id<C>::value;

template<Code C>
struct to_core_code;

template<>
struct to_core_code<Code::TOKEN_MISSING> {
    static constexpr iro::diag::Code value = iro::diag::Code::MISSING_SYNC_AT;
};

template<>
struct to_core_code<Code::TOKEN_ILLEGAL_TRANSITION> {
    static constexpr iro::diag::Code value = iro::diag::Code::SCOPE_INSUFFICIENT;
};

template<>
struct to_core_code<Code::CAPABILITY_UNSATISFIED> {
    static constexpr iro::diag::Code value = iro::diag::Code::CAP_UNSUPPORTED;
};

template<>
struct to_core_code<Code::OBLIGATION_UNRESOLVED> {
    static constexpr iro::diag::Code value = iro::diag::Code::NO_REALIZATION_FOUND;
};

template<Code C>
using core_diagnostic_t = iro::diag::Diagnostic<to_core_code<C>::value, stable_id_v<C>>;

template<Code C>
using diagnostic_t = core_diagnostic_t<C>;

using TokenMissing = diagnostic_t<Code::TOKEN_MISSING>;
using TokenIllegalTransition = diagnostic_t<Code::TOKEN_ILLEGAL_TRANSITION>;
using CapabilityUnsatisfied = diagnostic_t<Code::CAPABILITY_UNSATISFIED>;
using ObligationUnresolved = diagnostic_t<Code::OBLIGATION_UNRESOLVED>;

} // namespace axp::diag
