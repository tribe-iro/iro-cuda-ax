#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::protocol::epoch {

template<class EpochTag>
struct kind_epoch {
    static_assert(iro::util::HasId<EpochTag>, "kind_epoch requires EpochTag with id");
    static constexpr auto id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.epoch.kind"), EpochTag::id);
};

template<class Subject, class EpochTag>
struct epoch {
    static_assert(iro::util::HasId<Subject>, "epoch requires Subject with id");
    static_assert(iro::util::HasId<EpochTag>, "epoch requires EpochTag with id");
    using kind = kind_epoch<EpochTag>;
    using subject = Subject;
    using tag = EpochTag;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.epoch"),
                           iro::util::mix_u64(Subject::id, EpochTag::id));
};

} // namespace axp::protocol::epoch

