#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::protocol::stage {

struct kind_cp_async_committed {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.token.kind_cp_async_committed");
};

template<class SlotSubj>
struct cp_async_committed {
    using kind = kind_cp_async_committed;
    using subject = SlotSubj;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.token.cp_async_committed"), SlotSubj::id);
};

} // namespace axp::protocol::stage
