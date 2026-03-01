#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::state {

struct SoftmaxStateF32 {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.attention.softmax_state_f32");
    float m;
    float l;
};

struct WelfordStateF32 {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.layernorm.welford_state_f32");
    float mean;
    float m2;
    int count;
};

} // namespace axp::state
