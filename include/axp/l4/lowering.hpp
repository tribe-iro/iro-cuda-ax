#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/l4/lowering.hpp is library-only; use axp/l4.hpp in application code."
#endif

#include <type_traits>

namespace axp::l4::lowering {

template<class>
inline constexpr bool always_false_v = false;

template<class L4Pattern>
struct to_l3_pattern {
    static_assert(always_false_v<L4Pattern>,
                  "axp::l4::lowering::to_l3_pattern: missing specialization for this pattern. "
                  "Expected canonical axp::l4::preset::* pattern specialization in "
                  "axp/l4/lowering_presets.hpp.");
};

template<class L4Pattern>
using to_l3_pattern_t = typename to_l3_pattern<L4Pattern>::type;

} // namespace axp::l4::lowering
