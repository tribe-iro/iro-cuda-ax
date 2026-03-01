#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/ir/concepts.hpp is library-only; use axp/l4.hpp in application code."
#endif

#include <type_traits>
#include <iro_cuda_ax_core.hpp>

namespace axp::ir {

template<class T>
concept ObligationLike = requires {
    typename T::inputs;
    typename T::outputs;
    typename T::resources;
    { T::id } -> std::convertible_to<iro::util::u64>;
};

template<class T>
concept RealizationLike = requires {
    typename T::obligation;
    { T::id } -> std::convertible_to<iro::util::u64>;
};

} // namespace axp::ir
