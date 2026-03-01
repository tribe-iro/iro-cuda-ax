#pragma once

#include "../protocol/tmem/contracts.hpp"

namespace axp::level0 {

template<class... Args>
using TileToTmem = axp::protocol::tmem::TileToTmem<Args...>;

template<class... Args>
using TmemToTile = axp::protocol::tmem::TmemToTile<Args...>;

} // namespace axp::level0

namespace iro::contract {

template<class... Args>
struct is_fused_atom<axp::protocol::tmem::TileToTmem<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::tmem::TmemToTile<Args...>> : std::true_type {};

} // namespace iro::contract
