#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::protocol::view {

template<class Subject, class ExecGroup, class Lifetime = iro::token::lifetime::block>
using ViewReadable = iro::token::bundle<
    iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<Subject, Lifetime>
>;

template<class Subject, class ExecGroup, class Lifetime = iro::token::lifetime::block>
using ViewReadableSync = iro::token::bundle<
    iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::sync_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<Subject, Lifetime>
>;

template<class Subject, class ExecGroup, class Lifetime = iro::token::lifetime::block>
using ViewProduced = iro::token::bundle<
    iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<Subject, Lifetime>
>;

template<class Subject, class ExecGroup, class Lifetime = iro::token::lifetime::block>
using ViewProducedSync = iro::token::bundle<
    iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::sync_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<Subject, Lifetime>
>;

}
