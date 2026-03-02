#pragma once

#include <iro_cuda_ax_core.hpp>
#include "bundles.hpp"
#include "../token_policy.hpp"

namespace axp::protocol::atomic {

// Produce an explicit atomic completion marker token.
template<class Recipe, class Subject, class ExecGroup, class ScopeT,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block>
struct MarkAtomicDone {
    using produced_token = atomic_done<Subject, ScopeT>;
    static_assert(axp::protocol::token_policy::satisfiable_v<produced_token, produced_token>,
                  "MarkAtomicDone: atomic token must be self-satisfiable under token policy");

    using inputs = iro::util::type_list<>;
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            AtomicHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<AtomicDone<Subject, ScopeT, OrderT, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using resources = iro::util::type_list<>;
};

// Require and forward atomic completion semantics across composition boundaries.
template<class Recipe, class Subject, class ExecGroup, class ScopeT,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block>
struct RequireAtomicDone {
    using required_token = atomic_done<Subject, ScopeT>;
    using produced_token = atomic_done<Subject, ScopeT>;
    static_assert(axp::protocol::token_policy::satisfiable_v<required_token, produced_token>,
                  "RequireAtomicDone: produced token must satisfy required token");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            AtomicHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<AtomicDone<Subject, ScopeT, OrderT, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            AtomicHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<AtomicDone<Subject, ScopeT, OrderT, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using resources = iro::util::type_list<>;
};

// Derive atomic completion from a tile boundary in composition graphs.
template<class Recipe, class TilePayload, class Subject, class ExecGroup, class ScopeT,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block>
struct MarkAtomicDoneFromTile {
    static_assert(iro::contract::TilePayload<TilePayload>,
                  "MarkAtomicDoneFromTile: TilePayload is required");
    using produced_token = atomic_done<Subject, ScopeT>;
    static_assert(axp::protocol::token_policy::satisfiable_v<produced_token, produced_token>,
                  "MarkAtomicDoneFromTile: produced token must be self-satisfiable under token policy");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            TilePayload,
            Subject,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
                iro::token::alive<Subject, Lifetime>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            AtomicHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<AtomicDone<Subject, ScopeT, OrderT, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using resources = iro::util::type_list<>;
};

} // namespace axp::protocol::atomic
