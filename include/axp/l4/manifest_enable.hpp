#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/l4/manifest_enable.hpp is library-only; use axp/l4.hpp in application code."
#endif

#include "../l4.hpp"

namespace axp::l4::manifest_enable {

template<class Pattern, class Cap>
using enabled = axp::l4::manifest::enabled<Pattern, Cap>;

template<class Pattern, class Cap>
inline constexpr bool enabled_v = axp::l4::manifest::enabled_v<Pattern, Cap>;

template<class Pattern>
using tie_break_key = axp::l4::manifest::tie_break_key<Pattern>;

template<class Pattern>
inline constexpr bool has_tie_break_key_v = axp::l4::manifest::has_tie_break_key_v<Pattern>;

} // namespace axp::l4::manifest_enable
