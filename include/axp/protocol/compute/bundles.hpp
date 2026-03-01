#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::protocol::compute {

struct WgmmaHandle {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.compute.wgmma_handle");
};

struct kind_wgmma_fenced {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.token.kind_wgmma_fenced");
};

template<class Subject>
struct wgmma_fenced {
    using kind = kind_wgmma_fenced;
    using subject = Subject;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.wgmma_fenced"), Subject::id);
};

template<int Group>
struct kind_wgmma_committed {
    static_assert(Group >= 0, "kind_wgmma_committed group must be >= 0");
    static constexpr auto id = iro::util::mix_u64(
        iro::util::fnv1a_64_cstr("axp.token.kind_wgmma_committed"),
        static_cast<iro::util::u64>(Group));
};

struct kind_wgmma_issued {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.token.kind_wgmma_issued");
};

template<class Subject>
struct wgmma_issued {
    using kind = kind_wgmma_issued;
    using subject = Subject;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.wgmma_issued"), Subject::id);
};

template<class Subject, int Group>
struct wgmma_committed {
    static_assert(Group >= 0, "wgmma_committed group must be >= 0");
    using kind = kind_wgmma_committed<Group>;
    using subject = Subject;
    static constexpr int group = Group;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.wgmma_committed"),
            iro::util::mix_u64(Subject::id, static_cast<iro::util::u64>(Group)));
};

template<int Group>
struct kind_wgmma_waited {
    static_assert(Group >= 0, "kind_wgmma_waited group must be >= 0");
    static constexpr auto id = iro::util::mix_u64(
        iro::util::fnv1a_64_cstr("axp.token.kind_wgmma_waited"),
        static_cast<iro::util::u64>(Group));
};

template<class Subject, int Group>
struct wgmma_waited {
    static_assert(Group >= 0, "wgmma_waited group must be >= 0");
    using kind = kind_wgmma_waited<Group>;
    using subject = Subject;
    static constexpr int group = Group;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.wgmma_waited"),
            iro::util::mix_u64(Subject::id, static_cast<iro::util::u64>(Group)));
};

} // namespace axp::protocol::compute
