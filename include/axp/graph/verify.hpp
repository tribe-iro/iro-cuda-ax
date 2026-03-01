#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/graph/verify.hpp is library-only; use axp/l4.hpp in application code."
#endif

#include <type_traits>

#include "spec.hpp"

namespace axp::graph {

namespace detail {

template<class EdgeList>
struct connected_edge_contracts_ok;

template<>
struct connected_edge_contracts_ok<iro::util::type_list<>> {
    static constexpr bool value = true;
};

template<class E0, class... Es>
struct connected_edge_contracts_ok<iro::util::type_list<E0, Es...>> {
    static constexpr bool head =
        iro::contract::verify::port_satisfies<typename E0::out_port, typename E0::in_port>() &&
        !std::is_same_v<typename E0::out_port::recipe, iro::recipe::no_recipe> &&
        !std::is_same_v<typename E0::in_port::recipe, iro::recipe::no_recipe> &&
        iro::verify::recipe_compatible<typename E0::out_port::recipe, typename E0::in_port::recipe>();
    static constexpr bool value = head && connected_edge_contracts_ok<iro::util::type_list<Es...>>::value;
};

} // namespace detail

template<class G>
consteval bool verify_structure() {
    static_assert(CompositionLike<G>,
                  "axp::graph::verify_structure: graph must provide an iro::compose::Composition shape");

    using C = composition_of_t<G>;
    using obligations = typename C::obligations;
    using edges = typename C::edges;
    using all_inputs = typename iro::compose::detail::inputs_of<obligations>::type;
    using all_outputs = typename iro::compose::detail::outputs_of<obligations>::type;

    constexpr bool edge_inputs_ok = iro::compose::detail::remove_edge_inputs<all_inputs, edges>::ok;
    constexpr bool edge_outputs_ok = iro::compose::detail::edges_outputs_exist<all_outputs, edges>::ok;
    constexpr bool acyclic = iro::compose::detail::acyclic<obligations, edges>::value;

    return edge_inputs_ok && edge_outputs_ok && acyclic;
}

template<class G>
consteval bool verify_contract_flow() {
    static_assert(CompositionLike<G>,
                  "axp::graph::verify_contract_flow: graph must provide an iro::compose::Composition shape");

    using C = composition_of_t<G>;
    using obligations = typename C::obligations;
    using edges = typename C::edges;
    using resources = typename C::resources;
    using expected_resources_raw = typename iro::compose::detail::resources_of<obligations>::type;
    using expected_resources = iro::verify::canonicalize_resource_list<expected_resources_raw>;
    using provided_resources = iro::verify::canonicalize_resource_list<resources>;

    constexpr bool connected_only_ok = detail::connected_edge_contracts_ok<edges>::value;
    constexpr bool resources_ok =
        iro::verify::resource_list_canonical<resources>() &&
        iro::verify::resources_ok_union<resources>() &&
        std::is_same_v<expected_resources, provided_resources>;
    constexpr bool budgets_ok =
        (iro::verify::detail::sum_smem<provided_resources>::value <= C::profile::max_smem) &&
        (iro::verify::detail::sum_smem<provided_resources>::value <= C::cap::max_smem_per_block) &&
        (iro::verify::detail::sum_barriers<provided_resources>::value <= C::profile::max_barriers) &&
        (iro::verify::detail::sum_regs<provided_resources>::value <= C::profile::max_regs) &&
        (iro::verify::detail::max_reg_pressure<provided_resources>::value <= C::profile::max_regs);

    return connected_only_ok && resources_ok && budgets_ok;
}

template<class G>
consteval bool verify() {
    return verify_structure<G>() && verify_contract_flow<G>();
}

} // namespace axp::graph
