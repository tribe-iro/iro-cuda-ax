#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::level0::role {

struct producer {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.role.producer");
};

struct consumer {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.role.consumer");
};

} // namespace axp::level0::role

namespace axp::level0 {

// Role-tagged wrapper to enable warp specialization in higher layers.
template<class Role, class Op>
struct SpecializedOp : Op {
    static_assert(iro::util::HasId<Role>, "SpecializedOp: Role must have id");
    static_assert(iro::util::HasId<Op>, "SpecializedOp: Op must have id");
    using role = Role;
    static constexpr auto id =
        iro::util::mix_u64(
            iro::util::fnv1a_64_cstr("axp.level0.SpecializedOp"),
            iro::util::mix_u64(Role::id, Op::id));
};

// Resource-only obligation to require a minimum warpgroup count.
template<int WarpgroupCount, int WarpsPerGroup>
struct RequireWarpgroupCount {
    static_assert(WarpgroupCount >= 1, "RequireWarpgroupCount: WarpgroupCount must be >= 1");
    static_assert(WarpsPerGroup >= 1, "RequireWarpgroupCount: WarpsPerGroup must be >= 1");
    using inputs = iro::util::type_list<>;
    using outputs = iro::util::type_list<>;
    using resources = iro::util::type_list<
        iro::contract::res::warpgroup_count<WarpgroupCount>,
        iro::contract::res::block_threads_multiple_of<WarpgroupCount * WarpsPerGroup * 32>
    >;
    static constexpr auto id =
        iro::util::mix_u64(
            iro::util::fnv1a_64_cstr("axp.level0.RequireWarpgroupCount"),
            iro::util::mix_u64(static_cast<iro::util::u64>(WarpgroupCount),
                               static_cast<iro::util::u64>(WarpsPerGroup)));
};

} // namespace axp::level0
