#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::protocol::sync {

struct BarrierHandle {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.sync.barrier_handle");
};

struct SyncHandle {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.sync.handle");
};

struct FenceHandle {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.sync.fence_handle");
};

struct kind_tx_bytes {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.token.kind_tx_bytes");
};

template<class Subject, long long Bytes>
struct tx_bytes {
    static_assert(Bytes > 0, "tx_bytes requires positive byte count");
    using kind = kind_tx_bytes;
    using subject = Subject;
    static constexpr long long bytes = Bytes;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.tx_bytes"),
                           iro::util::mix_u64(Subject::id, (iro::util::u64)Bytes));
};

template<class Subject, class ExecGroup, class Lifetime = iro::token::lifetime::block>
using BarrierReady = iro::token::bundle<
    iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<Subject, Lifetime>
>;

template<class Subject, class ExecGroup, class Lifetime = iro::token::lifetime::block>
using BarrierArrived = iro::token::bundle<
    iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::sync_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<Subject, Lifetime>
>;

template<class Subject, long long Bytes, class Lifetime = iro::token::lifetime::block>
using BarrierTxExpected = iro::token::bundle<
    tx_bytes<Subject, Bytes>,
    iro::token::visible_at<Subject, iro::scope::block>,
    iro::token::alive<Subject, Lifetime>
>;

template<class Subject, class ExecGroup, class Lifetime = iro::token::lifetime::block>
using SyncReady = iro::token::bundle<
    iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<Subject, Lifetime>
>;

template<class Subject, class ExecGroup, class Lifetime = iro::token::lifetime::block>
using SyncArrived = iro::token::bundle<
    iro::token::visible_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::sync_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>,
    iro::token::alive<Subject, Lifetime>
>;
}
