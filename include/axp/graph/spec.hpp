#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/graph/spec.hpp is library-only; use axp/l4.hpp in application code."
#endif

#include <concepts>
#include <type_traits>

#include <iro_cuda_ax_core.hpp>

#include "../ir/concepts.hpp"

namespace axp::graph {

template<class CompositionT>
struct GraphSpec {
    using composition = CompositionT;
};

template<class T, class = void>
struct composition_of {
    using type = T;
};

template<class T>
struct composition_of<T, std::void_t<typename T::composition>> {
    using type = typename T::composition;
};

template<class T>
using composition_of_t = typename composition_of<T>::type;

template<class T>
concept CompositionLike = requires {
    typename composition_of_t<T>::obligations;
    typename composition_of_t<T>::edges;
    typename composition_of_t<T>::resources;
    typename composition_of_t<T>::profile;
    typename composition_of_t<T>::cap;
};

template<class P>
concept SupportsGraphCapExpr =
    requires {
        { P::template supports<iro::cap::sm89> } -> std::convertible_to<bool>;
        { P::template supports<iro::cap::sm90> } -> std::convertible_to<bool>;
        { P::template supports<iro::cap::sm100> } -> std::convertible_to<bool>;
    };

template<class P>
concept GraphPart =
    axp::ir::ObligationLike<P> &&
    SupportsGraphCapExpr<P> &&
    requires {
        { P::part_id } -> std::convertible_to<iro::util::u64>;
        { P::version } -> std::convertible_to<unsigned>;
        typename P::numerics;
    };

} // namespace axp::graph
