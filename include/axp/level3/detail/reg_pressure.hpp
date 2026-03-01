#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::level3::detail {

template<class Frag>
inline constexpr int frag_reg_count_v =
    (static_cast<int>(Frag::count) * static_cast<int>(Frag::elem::bytes) + 3) / 4;

template<int... Ns>
struct max_int;

template<>
struct max_int<> : std::integral_constant<int, 0> {};

template<int N>
struct max_int<N> : std::integral_constant<int, N> {};

template<int N0, int N1, int... Ns>
struct max_int<N0, N1, Ns...>
    : std::integral_constant<int, (N0 > max_int<N1, Ns...>::value ? N0 : max_int<N1, Ns...>::value)> {};

template<class... Frags>
inline constexpr int max_frag_reg_count_v = max_int<frag_reg_count_v<Frags>...>::value;

template<int BaseRegs, class... Frags>
struct reg_pressure_obligation {
    using reg_t = iro::contract::res::reg_pressure<BaseRegs + max_frag_reg_count_v<Frags...>>;
    using inputs = iro::util::type_list<>;
    using outputs = iro::util::type_list<>;
    using resources = iro::util::type_list<reg_t>;
    static constexpr auto id = reg_t::id;
};

template<int BaseRegs>
struct reg_pressure_const {
    using reg_t = iro::contract::res::reg_pressure<BaseRegs>;
    using inputs = iro::util::type_list<>;
    using outputs = iro::util::type_list<>;
    using resources = iro::util::type_list<reg_t>;
    static constexpr auto id = reg_t::id;
};

} // namespace axp::level3::detail
