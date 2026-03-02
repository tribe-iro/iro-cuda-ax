#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::protocol::order {

struct OrderHandle {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.order.handle");
};

template<class EventTag>
struct kind_event {
    static_assert(iro::util::HasId<EventTag>, "kind_event requires EventTag with id");
    static constexpr auto id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.kind_event"), EventTag::id);
};

template<class Subject, class EventTag>
struct event {
    static_assert(iro::util::HasId<Subject>, "event requires Subject with id");
    static_assert(iro::util::HasId<EventTag>, "event requires EventTag with id");
    using kind = kind_event<EventTag>;
    using subject = Subject;
    using tag = EventTag;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.event"),
                           iro::util::mix_u64(Subject::id, EventTag::id));
};

template<class PhaseTag>
struct kind_happens_before {
    static_assert(iro::util::HasId<PhaseTag>, "kind_happens_before requires PhaseTag with id");
    static constexpr auto id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.kind_happens_before"), PhaseTag::id);
};

template<class Subject, class PhaseTag>
struct happens_before {
    static_assert(iro::util::HasId<Subject>, "happens_before requires Subject with id");
    static_assert(iro::util::HasId<PhaseTag>, "happens_before requires PhaseTag with id");
    using kind = kind_happens_before<PhaseTag>;
    using subject = Subject;
    using phase = PhaseTag;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.happens_before"),
                           iro::util::mix_u64(Subject::id, PhaseTag::id));
};

template<class EpochTag>
struct kind_epoch {
    static_assert(iro::util::HasId<EpochTag>, "kind_epoch requires EpochTag with id");
    static constexpr auto id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.kind_epoch"), EpochTag::id);
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

} // namespace axp::protocol::order

