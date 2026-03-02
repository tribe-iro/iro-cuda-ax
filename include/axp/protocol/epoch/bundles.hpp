#pragma once

#include <iro_cuda_ax_core.hpp>
#include "tokens.hpp"

namespace axp::protocol::epoch {

template<class Subject, class EpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
using EpochStamped = iro::token::bundle<
    epoch<Subject, EpochTag>,
    iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<Subject, Lifetime>
>;

} // namespace axp::protocol::epoch

