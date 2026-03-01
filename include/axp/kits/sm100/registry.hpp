#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::kit::sm100 {

template<class>
inline constexpr bool always_false_v = false;

template<class Obligation>
struct registry_for {
    static_assert(always_false_v<Obligation>,
                  "axp::kit::sm100::registry_for: no SM100 realizations are registered.");
};

template<class Obligation>
using registry_for_t = typename registry_for<Obligation>::type;

template<class Obligation>
using bind_t = iro::bind::lookup_realization_t<Obligation, iro::cap::sm100, registry_for_t<Obligation>>;

} // namespace axp::kit::sm100
