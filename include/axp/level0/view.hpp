#pragma once

#include "../protocol/view/contracts.hpp"

namespace axp::level0 {

template<class... Args>
using TileView = axp::protocol::view::TileView<Args...>;

template<class... Args>
using Transpose = axp::protocol::view::TransposeView<Args...>;

template<class... Args>
using Swizzle = axp::protocol::view::SwizzleView<Args...>;

template<class... Args>
using Reshape = axp::protocol::view::ReshapeView<Args...>;

} // namespace axp::level0

namespace iro::contract {

template<class... Args>
struct is_fused_atom<axp::protocol::view::TileView<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::view::TransposeView<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::view::SwizzleView<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::view::ReshapeView<Args...>> : std::true_type {};

} // namespace iro::contract
