#pragma once

#include <iro_cuda_ax_core.hpp>
#include "bundles.hpp"
#include "../token_policy.hpp"

namespace axp::protocol::epoch {

struct EpochHandle {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.epoch.handle");
};

template<class Recipe, class Subject, class EpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct InitEpoch {
    using produced_token = epoch<Subject, EpochTag>;
    static_assert(axp::protocol::token_policy::satisfiable_v<produced_token, produced_token>,
                  "InitEpoch: epoch token must be self-satisfiable under token policy");

    using inputs = iro::util::type_list<>;
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            EpochHandle,
            Subject,
            ExecGroup,
            iro::util::concat_t<
                iro::token::bundle_list<EpochStamped<Subject, EpochTag, ExecGroup, Lifetime>>,
                iro::util::type_list<iro::token::issued_by_lane0<Subject>>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using resources = iro::util::type_list<>;
};

template<class Recipe, class Subject, class PrevEpochTag, class NextEpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct AdvanceEpoch {
    using required_token = epoch<Subject, PrevEpochTag>;
    using produced_token = epoch<Subject, NextEpochTag>;
    static_assert(PrevEpochTag::id != NextEpochTag::id,
                  "AdvanceEpoch: PrevEpochTag and NextEpochTag must differ");
    static_assert(!axp::protocol::token_policy::satisfiable_v<required_token, produced_token>,
                  "AdvanceEpoch: next epoch token must not implicitly satisfy previous epoch token");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            EpochHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<EpochStamped<Subject, PrevEpochTag, ExecGroup, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            EpochHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<EpochStamped<Subject, NextEpochTag, ExecGroup, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using resources = iro::util::type_list<>;
};

template<class Recipe, class Subject, class EpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct RequireEpoch {
    using required_token = epoch<Subject, EpochTag>;
    using produced_token = epoch<Subject, EpochTag>;
    static_assert(axp::protocol::token_policy::satisfiable_v<required_token, produced_token>,
                  "RequireEpoch: produced token must satisfy required token");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            EpochHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<EpochStamped<Subject, EpochTag, ExecGroup, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            EpochHandle,
            Subject,
            ExecGroup,
            iro::token::bundle_list<EpochStamped<Subject, EpochTag, ExecGroup, Lifetime>>,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using resources = iro::util::type_list<>;
};

} // namespace axp::protocol::epoch
