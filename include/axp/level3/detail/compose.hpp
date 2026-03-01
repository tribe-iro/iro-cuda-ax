#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../../target.hpp"

namespace axp::level3::detail {

template<class Obligation, int I>
using in_port_t = iro::compose::in_port_ref<Obligation, I>;

template<class Obligation, int I>
using out_port_t = iro::compose::out_port_ref<Obligation, I>;

template<class Obligation, class CapT = axp::target_cap>
struct as_composition {
    using obligations = iro::util::type_list<Obligation>;
    using edges = iro::util::type_list<>;
    using resources = typename Obligation::resources;
    using type = iro::compose::Composition<
        obligations,
        edges,
        resources,
        iro::profile::BudgetMax,
        CapT
    >;
};

template<class Obligation, class CapT = axp::target_cap>
using as_composition_t = typename as_composition<Obligation, CapT>::type;

template<class Obligations, class Edges,
         class ProfileT = iro::profile::BudgetMax,
         class CapT = axp::target_cap>
using make_composition_t = iro::compose::CompositionAutoResources<Obligations, Edges, ProfileT, CapT>;

} // namespace axp::level3::detail
