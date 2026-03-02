#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::protocol::atomic {

struct AtomicHandle {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.atomic.handle");
};

template<class ScopeT>
struct kind_atomic_done {
    static_assert(iro::util::HasId<ScopeT>, "kind_atomic_done requires Scope with id");
    static constexpr auto id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.kind_atomic_done"), ScopeT::id);
};

template<class Subject, class ScopeT>
struct atomic_done {
    static_assert(iro::util::HasId<Subject>, "atomic_done requires Subject with id");
    static_assert(iro::util::HasId<ScopeT>, "atomic_done requires Scope with id");
    using kind = kind_atomic_done<ScopeT>;
    using subject = Subject;
    using scope = ScopeT;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.atomic_done"),
                           iro::util::mix_u64(Subject::id, ScopeT::id));
};

} // namespace axp::protocol::atomic

