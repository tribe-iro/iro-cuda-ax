#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/l4/resolve.hpp is library-only; use axp/l4.hpp in application code."
#endif

#include "../target.hpp"
#include "../graph/verify.hpp"
#include "../graph/hash.hpp"
#include "../l4.hpp"

#include "bind_key.hpp"
#include "manifest_enable.hpp"
#include "graph_registry_index.hpp"

namespace axp::l4 {

template<class G, class Cap = axp::target_cap, class ProfileT = axp::l4::profile::proof_full>
struct resolve {
    static_assert(axp::graph::verify<G>(),
                  "axp::l4::resolve: graph verification failed");

    static constexpr iro::util::u64 graph_hash = axp::graph::graph_hash_v<G>;
    using registry_entry = axp::l4::graph_registry::entry<graph_hash, Cap, ProfileT>;

    static_assert(registry_entry::enabled,
                  "axp::l4::resolve: no manifest-enabled graph row for (graph_hash, capability, profile)");
    static_assert(!std::is_same_v<typename registry_entry::pattern, void>,
                  "axp::l4::resolve: registry row missing pattern type");
    static_assert(axp::l4::manifest_enable::enabled_v<typename registry_entry::pattern, Cap>,
                  "axp::l4::resolve: pattern/capability pair is not manifest-enabled");

    using pattern = typename registry_entry::pattern;
    using graph = axp::level3::registry::Select<pattern, Cap>;
    static constexpr iro::util::u64 realization_key = registry_entry::realization_key;
    static constexpr iro::util::u64 bind_key =
        axp::l4::bind_key_v<graph_hash, Cap, ProfileT, realization_key>;
    using type = graph;
};

template<class G, class Cap = axp::target_cap, class ProfileT = axp::l4::profile::proof_full>
using Select = typename resolve<G, Cap, ProfileT>::type;

template<class G, class Cap = axp::target_cap, class ProfileT = axp::l4::profile::proof_full>
struct supports : std::bool_constant<axp::l4::graph_registry::enabled_v<axp::graph::graph_hash_v<G>, Cap, ProfileT>> {};

} // namespace axp::l4
