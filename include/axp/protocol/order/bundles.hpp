#pragma once

#include <iro_cuda_ax_core.hpp>
#include "tokens.hpp"

namespace axp::protocol::order {

template<class Subject, class EventTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
using EventPublished = iro::token::bundle<
    event<Subject, EventTag>,
    iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<Subject, Lifetime>
>;

template<class Subject, class EventTag, class PhaseTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
using EventPhaseReady = iro::token::bundle<
    event<Subject, EventTag>,
    happens_before<Subject, PhaseTag>,
    iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<Subject, Lifetime>
>;

template<class Subject, class EpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
using EpochStamped = iro::token::bundle<
    epoch<Subject, EpochTag>,
    iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<Subject, Lifetime>
>;

} // namespace axp::protocol::order
