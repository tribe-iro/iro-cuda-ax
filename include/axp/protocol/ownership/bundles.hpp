#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::protocol::ownership {

template<class FragSubj, class Lifetime>
using FragmentLive = iro::token::bundle<
    iro::token::alive<FragSubj, Lifetime>
>;

}
