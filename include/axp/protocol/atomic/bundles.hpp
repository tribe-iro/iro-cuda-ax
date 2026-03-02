#pragma once

#include <iro_cuda_ax_core.hpp>
#include "tokens.hpp"

namespace axp::protocol::atomic {

template<class Subject, class ScopeT, class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block>
using AtomicDone = iro::token::bundle<
    atomic_done<Subject, ScopeT>,
    iro::token::memory_order<Subject, OrderT, ScopeT>,
    iro::token::visible_at<Subject, ScopeT>,
    iro::token::alive<Subject, Lifetime>
>;

} // namespace axp::protocol::atomic

