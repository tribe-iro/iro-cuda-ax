#pragma once

#include <type_traits>
#include "protocol/compute/contracts.hpp"
#include "protocol/stage/resources.hpp"
#include "naming/tags.hpp"

namespace axp::swizzle {

template<int M, int B, int S>
using Atom = axp::protocol::stage::SwizzleAtom<M, B, S>;

using None = axp::protocol::stage::SwizzleAtom_None;
using B32 = axp::protocol::stage::SwizzleAtom_32B;
using B64 = axp::protocol::stage::SwizzleAtom_64B;
using B128 = axp::protocol::stage::SwizzleAtom_128B;

namespace detail {
template<class T>
inline constexpr bool always_false_v = false;
} // namespace detail

// Explicit, opt-in swizzle selection by MMA shape and operand tag.
// No implicit selection: unsupported shapes must be specified by user.
template<class Shape, class Operand>
struct for_mma {
    static_assert(detail::always_false_v<Shape>,
                  "swizzle::for_mma: no default swizzle for this shape; specify SwizzleAtom explicitly");
};

template<int M, int N, int K, class ElemA, class ElemB, class Acc, class LayoutA, class LayoutB, class Operand>
struct for_mma<axp::protocol::compute::MmaShape<M, N, K, ElemA, ElemB, Acc, LayoutA, LayoutB>, Operand> {
    static_assert(M == 64, "swizzle::for_mma: default swizzle only defined for WGMMA (M=64)");
    using type = std::conditional_t<(K == 32), B64, B128>;
};

template<class Shape, class Operand>
using for_mma_t = typename for_mma<Shape, Operand>::type;

} // namespace axp::swizzle
