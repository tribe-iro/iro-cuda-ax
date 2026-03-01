#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>
#include "bundles.hpp"
#include "../../detail/resources.hpp"

namespace axp::protocol::sync {

// Barrier initialization (block scope)
template<class Recipe, class Subject, class ExecGroup, int Expected>
struct BarrierInit {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierInit requires block exec group");
    static_assert(Expected > 0, "BarrierInit requires positive expected count");
    using inputs = iro::util::type_list<>;
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::util::concat_t<
                iro::token::bundle_list<BarrierReady<Subject, ExecGroup>>,
                iro::util::type_list<iro::token::issued_by_lane0<Subject>>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<Subject>
    >;
};

// Cluster barrier initialization (cluster scope)
template<class Recipe, class Subject, class ExecGroup, int Expected>
struct ClusterBarrierInit {
    static_assert(std::is_same_v<ExecGroup, iro::exec::cluster>, "ClusterBarrierInit requires cluster exec group");
    static_assert(Expected > 0, "ClusterBarrierInit requires positive expected count");
    using inputs = iro::util::type_list<>;
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::util::concat_t<
                iro::token::bundle_list<BarrierReady<Subject, ExecGroup>>,
                iro::util::type_list<iro::token::issued_by_lane0<Subject>>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<Subject>
    >;
};

// Explicit synchronization point (blocking, warp-only)
template<class Recipe, class Subject, class ExecGroup>
struct SyncPoint {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>,
                  "SyncPoint supports warp only; use explicit barriers for larger scopes");
    using req_lifetime = iro::token::lifetime::warp;
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::util::type_list<
                iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
                iro::token::alive<Subject, req_lifetime>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierArrived<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Warp-level sync (explicit, no barrier resource)
template<class Recipe, class Subject, class ExecGroup>
struct SyncWarp {
    static_assert(std::is_same_v<ExecGroup, iro::exec::warp>, "SyncWarp requires warp exec group");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SyncHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<SyncReady<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            SyncHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<SyncArrived<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Block-level sync (explicit, no barrier resource)
template<class Recipe, class Subject, class ExecGroup>
struct SyncThreads {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "SyncThreads requires block exec group");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            SyncHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<SyncReady<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            SyncHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<SyncArrived<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<>;
};

// Memory fence (explicit ordering)
template<class Recipe, class Subject, class ExecGroup, class ScopeT, class OrderT = iro::memory_order::seq_cst>
struct Fence {
    static_assert(iro::util::HasId<ScopeT>, "Fence requires scope with id");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>, "Fence requires explicit Recipe");
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            FenceHandle,
            Subject,
            ExecGroup,
            iro::util::type_list<
                iro::token::memory_order<Subject, OrderT, ScopeT>,
                iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
                iro::token::alive<Subject, iro::token::lifetime::block>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using inputs = iro::util::type_list<>;
    using resources = iro::util::type_list<>;
};

// Barrier expect transaction bytes (SM90 mbarrier)
template<class Recipe, class Subject, class ExecGroup, long long Bytes>
struct BarrierExpectTx {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierExpectTx requires block exec group");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierReady<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierTxExpected<Subject, Bytes>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<Subject>
    >;
};

// Barrier arrive with transaction bytes (SM90 mbarrier)
template<class Recipe, class Subject, class ExecGroup, long long Bytes>
struct BarrierArriveTx {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierArriveTx requires block exec group");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierTxExpected<Subject, Bytes>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierArrived<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<Subject>
    >;
};

// Barrier arrive (non-blocking)
template<class Recipe, class Subject, class ExecGroup>
struct BarrierArrive {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierArrive requires block exec group");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierReady<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierArrived<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<Subject>
    >;
};

// Cluster barrier arrive (non-blocking)
template<class Recipe, class Subject, class ExecGroup>
struct ClusterBarrierArrive {
    static_assert(std::is_same_v<ExecGroup, iro::exec::cluster>, "ClusterBarrierArrive requires cluster exec group");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierReady<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierArrived<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<Subject>
    >;
};

// Barrier wait (blocking)
template<class Recipe, class Subject, class ExecGroup>
struct BarrierWait {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierWait requires block exec group");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierArrived<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierArrived<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<Subject>
    >;
};

// Cluster barrier wait (blocking)
template<class Recipe, class Subject, class ExecGroup>
struct ClusterBarrierWait {
    static_assert(std::is_same_v<ExecGroup, iro::exec::cluster>, "ClusterBarrierWait requires cluster exec group");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierArrived<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierArrived<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<Subject>
    >;
};

// Barrier try-wait (non-blocking, returns predicate)
template<class Recipe, class Subject, class ExecGroup, class FlagPayload, class FlagSubj,
         class OutExtra = iro::util::type_list<>>
struct BarrierTryWait {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierTryWait requires block exec group");
    static_assert(iro::contract::ScalarPayload<FlagPayload>,
                  "BarrierTryWait requires scalar payload for flag");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierArrived<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierArrived<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >,
        iro::contract::OutputPort<
            FlagPayload,
            FlagSubj,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<FlagSubj, iro::scope::min_scope_for_t<ExecGroup>>,
                    iro::token::alive<FlagSubj, iro::token::lifetime::block>
                >,
                OutExtra
            >,
            typename FlagPayload::dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<Subject>
    >;
};

// Barrier invalidate/reset (reinitialize for reuse)
template<class Recipe, class Subject, class ExecGroup, int Expected>
struct BarrierInvalidate {
    static_assert(std::is_same_v<ExecGroup, iro::exec::block>, "BarrierInvalidate requires block exec group");
    static_assert(Expected > 0, "BarrierInvalidate requires positive expected count");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierArrived<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            BarrierHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<BarrierReady<Subject, ExecGroup>>,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using resources = iro::util::type_list<
        iro::contract::res::named_barrier<Subject>
    >;
};

} // namespace axp::protocol::sync
