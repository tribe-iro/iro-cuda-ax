#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::dist {

using warp_row_major = iro::dist::warp_row_major;
using warp_col_major = iro::dist::warp_col_major;
using warpgroup_row_major = iro::dist::warpgroup_row_major;
using warpgroup_col_major = iro::dist::warpgroup_col_major;
using accumulator = iro::dist::accumulator;
using reg_owned = iro::dist::reg_owned;
using tmem_row_major = iro::dist::tmem_row_major;
using tmem_col_major = iro::dist::tmem_col_major;
using lane = iro::dist::lane;
using replicated = iro::dist::replicated;

template<class ScopeT>
using uniform = iro::dist::uniform<ScopeT>;

template<class ScopeT, int Period>
using cyclic = iro::dist::cyclic<ScopeT, Period>;

template<int Lanes>
using striped = iro::dist::striped<Lanes>;

template<int Period>
using warp_cyclic = iro::dist::cyclic<iro::scope::warp, Period>;

template<int Period>
using block_cyclic = iro::dist::cyclic<iro::scope::block, Period>;

using warp_uniform = iro::dist::uniform<iro::scope::warp>;
using block_uniform = iro::dist::uniform<iro::scope::block>;
using warp_mask = iro::dist::mask<iro::scope::warp>;
using block_mask = iro::dist::mask<iro::scope::block>;

struct block {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.dist.block");
};

struct reg_tile {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.dist.reg_tile");
};

struct tmem_tile {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.dist.tmem_tile");
};

} // namespace axp::dist

namespace iro::schema {
template<>
struct is_dist<axp::dist::block> : std::true_type {};
template<>
struct is_dist<axp::dist::reg_tile> : std::true_type {};
template<>
struct is_dist<axp::dist::tmem_tile> : std::true_type {};
} // namespace iro::schema
