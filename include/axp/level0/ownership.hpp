#pragma once

#include "../protocol/ownership/contracts.hpp"

namespace axp::level0 {

template<class... Args>
using TileToFragment = axp::protocol::ownership::TileToFragment<Args...>;

template<class... Args>
using FragmentToTile = axp::protocol::ownership::FragmentToTile<Args...>;

template<class... Args>
using SharedTileToFragment = axp::protocol::ownership::SharedTileToFragment<Args...>;

template<class... Args>
using FragmentToSharedTile = axp::protocol::ownership::FragmentToSharedTile<Args...>;

template<class... Args>
using TileBoundaryIn = axp::protocol::ownership::TileBoundaryIn<Args...>;

template<class... Args>
using TileBoundaryOut = axp::protocol::ownership::TileBoundaryOut<Args...>;

template<class... Args>
using WgmmaSmemDesc = axp::protocol::ownership::WgmmaSmemDesc<Args...>;

template<class... Args>
using UseWgmmaSmemDesc = axp::protocol::ownership::UseWgmmaSmemDesc<Args...>;

template<class... Args>
using MakeDesc = axp::protocol::ownership::MakeWgmmaSmemDesc<Args...>;

template<class... Args>
using MakeDescReady = axp::protocol::ownership::MakeWgmmaSmemDescReady<Args...>;

template<class Recipe, class SmemTile, class DescTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime,
         class SwizzleAtom, int RowOffset, int ColOffset>
using MakeDescSlice = axp::protocol::ownership::MakeWgmmaSmemDescSlice<
    Recipe, SmemTile, DescTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom, RowOffset, ColOffset>;

template<class Recipe, class SmemTile, class DescTile, class SmemSubj, class DescSubj, class ExecGroup, class Lifetime,
         class SwizzleAtom, int RowOffset, int ColOffset>
using MakeDescSliceReady = axp::protocol::ownership::MakeWgmmaSmemDescSliceReady<
    Recipe, SmemTile, DescTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom, RowOffset, ColOffset>;

} // namespace axp::level0

namespace iro::contract {

template<class... Args>
struct is_fused_atom<axp::protocol::ownership::TileToFragment<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::ownership::FragmentToTile<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::ownership::SharedTileToFragment<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::ownership::FragmentToSharedTile<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::ownership::TileBoundaryIn<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::ownership::TileBoundaryOut<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::ownership::UseWgmmaSmemDesc<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::ownership::MakeWgmmaSmemDesc<Args...>> : std::true_type {};

template<class... Args>
struct is_fused_atom<axp::protocol::ownership::MakeWgmmaSmemDescReady<Args...>> : std::true_type {};

template<class Recipe, class SmemTile, class DescTile, class SmemSubj, class DescSubj, class ExecGroup,
         class Lifetime, class SwizzleAtom, int RowOffset, int ColOffset>
struct is_fused_atom<axp::protocol::ownership::MakeWgmmaSmemDescSlice<
    Recipe, SmemTile, DescTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom, RowOffset, ColOffset>>
    : std::true_type {};

template<class Recipe, class SmemTile, class DescTile, class SmemSubj, class DescSubj, class ExecGroup,
         class Lifetime, class SwizzleAtom, int RowOffset, int ColOffset>
struct is_fused_atom<axp::protocol::ownership::MakeWgmmaSmemDescSliceReady<
    Recipe, SmemTile, DescTile, SmemSubj, DescSubj, ExecGroup, Lifetime, SwizzleAtom, RowOffset, ColOffset>>
    : std::true_type {};

} // namespace iro::contract
