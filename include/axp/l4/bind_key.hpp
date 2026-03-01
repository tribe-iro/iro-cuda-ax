#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/l4/bind_key.hpp is library-only; use axp/l4.hpp in application code."
#endif

#include <iro_cuda_ax_core.hpp>

namespace axp::l4::profile {

struct dev_fast {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("axp.l4.profile.dev_fast");
};

struct proof_full {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("axp.l4.profile.proof_full");
};

} // namespace axp::l4::profile

namespace axp::l4 {

template<iro::util::u64 GraphHash, class Cap, class ProfileT, iro::util::u64 RealizationKey>
struct BindKey {
    static constexpr iro::util::u64 graph_hash = GraphHash;
    static constexpr iro::util::u64 capability = Cap::id;
    static constexpr iro::util::u64 profile = ProfileT::id;
    static constexpr iro::util::u64 realization_key = RealizationKey;
    static constexpr iro::util::u64 value =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.l4.bind_key.v2"),
                           iro::util::mix_u64(graph_hash,
                                              iro::util::mix_u64(capability,
                                                                 iro::util::mix_u64(profile, realization_key))));
};

template<iro::util::u64 GraphHash, class Cap, class ProfileT, iro::util::u64 RealizationKey>
inline constexpr iro::util::u64 bind_key_v =
    BindKey<GraphHash, Cap, ProfileT, RealizationKey>::value;

} // namespace axp::l4
