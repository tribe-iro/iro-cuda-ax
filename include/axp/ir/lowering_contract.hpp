#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/ir/lowering_contract.hpp is library-only; use axp/l4.hpp in application code."
#endif

#include "concepts.hpp"

namespace axp::ir {

template<class Backend, class Obligation, class Cap>
concept LoweringContract =
    ObligationLike<Obligation> &&
    requires {
        typename Backend::template resolve<Obligation, Cap>;
    };

} // namespace axp::ir
