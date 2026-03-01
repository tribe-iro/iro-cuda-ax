// Generated file. DO NOT EDIT.
// Regenerate with: crates/iro-cuda-axkernels/tools/gen_registry_index.py

#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/l4/graph_registry_index.hpp is library-only; use axp/l4.hpp in application code."
#endif

#include "../l4.hpp"
#include "../l3_presets.hpp"
#include "bind_key.hpp"
#include "../graph/hash.hpp"
#include <type_traits>

namespace axp::l4::graph_registry::hashes {

inline constexpr iro::util::u64 attention_16x16 = 0x15583dd84da2d8e9ULL;
inline constexpr iro::util::u64 layernorm_16x16 = 0x24cfea1387e2539aULL;
inline constexpr iro::util::u64 gemm_64x64x16_bias_silu = 0x33608d15b66a61efULL;
inline constexpr iro::util::u64 fa2_prefill_attention_64x64_hd128 = 0x39ab933815bef6a8ULL;
inline constexpr iro::util::u64 attention_64x64 = 0x5049caf5e71d715bULL;
inline constexpr iro::util::u64 gemm_64x64x16 = 0x6a772f708674ef56ULL;
inline constexpr iro::util::u64 vectorized_elementwise_16x16 = 0x6b67a38f9488bc84ULL;
inline constexpr iro::util::u64 rmsnorm_16x16 = 0x88980e507c926e83ULL;
inline constexpr iro::util::u64 sort_16 = 0xa269e839e1669050ULL;
inline constexpr iro::util::u64 softmax_row4 = 0xb704344dd328af0fULL;
inline constexpr iro::util::u64 histogram_256 = 0xc30ae660583a0c50ULL;
inline constexpr iro::util::u64 gemm_16x16x16 = 0xc3f2b8930d5ffd1cULL;
inline constexpr iro::util::u64 fa2_decode_attention_64x64_hd128 = 0xc95bfaa16a0a3918ULL;

} // namespace axp::l4::graph_registry::hashes

namespace axp::l4::graph_registry {

template<iro::util::u64 GraphHash, class Cap, class ProfileT, class = void>
struct entry {
    static constexpr bool enabled = false;
    using pattern = void;
    static constexpr iro::util::u64 realization_key = 0;
};

template<iro::util::u64 GraphHash, class Cap, class ProfileT>
inline constexpr bool enabled_v = entry<GraphHash, Cap, ProfileT>::enabled;

} // namespace axp::l4::graph_registry

template<> struct axp::l4::manifest::tie_break_key<axp::preset::Attention16x16> { static constexpr auto value = iro::util::fnv1a_64_cstr("preset.attention.16x16"); };
template<> struct axp::l4::manifest::tie_break_key<axp::preset::Attention64x64> { static constexpr auto value = iro::util::fnv1a_64_cstr("preset.attention.64x64"); };
template<> struct axp::l4::manifest::tie_break_key<axp::preset::Attention64x64x128Decode> { static constexpr auto value = iro::util::fnv1a_64_cstr("preset.attention.64x64.hd128.decode"); };
template<> struct axp::l4::manifest::tie_break_key<axp::preset::Attention64x64x128Prefill> { static constexpr auto value = iro::util::fnv1a_64_cstr("preset.attention.64x64.hd128.prefill"); };
template<> struct axp::l4::manifest::tie_break_key<axp::preset::Gemm16x16x16> { static constexpr auto value = iro::util::fnv1a_64_cstr("preset.gemm.16x16x16"); };
template<> struct axp::l4::manifest::tie_break_key<axp::preset::Gemm64x64x16> { static constexpr auto value = iro::util::fnv1a_64_cstr("preset.gemm.64x64x16"); };
template<> struct axp::l4::manifest::tie_break_key<axp::preset::Gemm64x64x16BiasSiLU> { static constexpr auto value = iro::util::fnv1a_64_cstr("preset.gemm.64x64x16.bias_silu"); };
template<> struct axp::l4::manifest::tie_break_key<axp::preset::Histogram256> { static constexpr auto value = iro::util::fnv1a_64_cstr("preset.histogram.256"); };
template<> struct axp::l4::manifest::tie_break_key<axp::preset::LayerNorm16x16> { static constexpr auto value = iro::util::fnv1a_64_cstr("preset.layernorm.16x16"); };
template<> struct axp::l4::manifest::tie_break_key<axp::preset::RMSNorm16x16> { static constexpr auto value = iro::util::fnv1a_64_cstr("preset.rmsnorm.16x16"); };
template<> struct axp::l4::manifest::tie_break_key<axp::preset::SoftmaxRow4> { static constexpr auto value = iro::util::fnv1a_64_cstr("preset.softmax.row4"); };
template<> struct axp::l4::manifest::tie_break_key<axp::preset::Sort16> { static constexpr auto value = iro::util::fnv1a_64_cstr("preset.sort.16"); };
template<> struct axp::l4::manifest::tie_break_key<axp::preset::VectorizedElementwise16x16> { static constexpr auto value = iro::util::fnv1a_64_cstr("preset.elementwise.vec.16x16"); };

#if defined(AXP_ENABLE_SM89)
template<> struct axp::l4::manifest::enabled<axp::preset::Attention16x16, iro::cap::sm89> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM89)
template<> struct axp::l4::manifest::enabled<axp::preset::LayerNorm16x16, iro::cap::sm89> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM90)
template<> struct axp::l4::manifest::enabled<axp::preset::LayerNorm16x16, iro::cap::sm90> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM90)
template<> struct axp::l4::manifest::enabled<axp::preset::Gemm64x64x16BiasSiLU, iro::cap::sm90> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM90)
template<> struct axp::l4::manifest::enabled<axp::preset::Attention64x64x128Prefill, iro::cap::sm90> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM90)
template<> struct axp::l4::manifest::enabled<axp::preset::Attention64x64, iro::cap::sm90> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM90)
template<> struct axp::l4::manifest::enabled<axp::preset::Gemm64x64x16, iro::cap::sm90> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM89)
template<> struct axp::l4::manifest::enabled<axp::preset::VectorizedElementwise16x16, iro::cap::sm89> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM90)
template<> struct axp::l4::manifest::enabled<axp::preset::VectorizedElementwise16x16, iro::cap::sm90> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM89)
template<> struct axp::l4::manifest::enabled<axp::preset::RMSNorm16x16, iro::cap::sm89> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM90)
template<> struct axp::l4::manifest::enabled<axp::preset::RMSNorm16x16, iro::cap::sm90> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM89)
template<> struct axp::l4::manifest::enabled<axp::preset::Sort16, iro::cap::sm89> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM90)
template<> struct axp::l4::manifest::enabled<axp::preset::Sort16, iro::cap::sm90> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM89)
template<> struct axp::l4::manifest::enabled<axp::preset::SoftmaxRow4, iro::cap::sm89> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM90)
template<> struct axp::l4::manifest::enabled<axp::preset::SoftmaxRow4, iro::cap::sm90> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM89)
template<> struct axp::l4::manifest::enabled<axp::preset::Histogram256, iro::cap::sm89> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM90)
template<> struct axp::l4::manifest::enabled<axp::preset::Histogram256, iro::cap::sm90> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM89)
template<> struct axp::l4::manifest::enabled<axp::preset::Gemm16x16x16, iro::cap::sm89> : std::true_type {};
#endif
#if defined(AXP_ENABLE_SM90)
template<> struct axp::l4::manifest::enabled<axp::preset::Attention64x64x128Decode, iro::cap::sm90> : std::true_type {};
#endif

#define AXP_GRAPH_ENTRY(GRAPH_HASH, ENTRY, CAP, PROFILE)                                      \
template<>                                                                                    \
struct axp::l4::graph_registry::entry<GRAPH_HASH, CAP, PROFILE> {                           \
    static constexpr bool enabled = true;                                                     \
    using pattern = ENTRY;                                                                     \
    static constexpr iro::util::u64 realization_key =                                         \
        axp::l4::manifest::tie_break_key<pattern>::value;                                     \
}

#define AXP_GRAPH_HASH_OVERRIDE(GRAPH_HASH, ENTRY, CAP)                                       \
template<>                                                                                    \
struct axp::graph::graph_hash_override<axp::level3::registry::Select<ENTRY, CAP>> {         \
    static constexpr bool enabled = true;                                                     \
    static constexpr iro::util::u64 value = GRAPH_HASH;                                       \
}

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::attention_16x16, axp::preset::Attention16x16, iro::cap::sm89, axp::l4::profile::dev_fast);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::attention_16x16, axp::preset::Attention16x16, iro::cap::sm89, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::layernorm_16x16, axp::preset::LayerNorm16x16, iro::cap::sm89, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::layernorm_16x16, axp::preset::LayerNorm16x16, iro::cap::sm90, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::gemm_64x64x16_bias_silu, axp::preset::Gemm64x64x16BiasSiLU, iro::cap::sm90, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::fa2_prefill_attention_64x64_hd128, axp::preset::Attention64x64x128Prefill, iro::cap::sm90, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::attention_64x64, axp::preset::Attention64x64, iro::cap::sm90, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::gemm_64x64x16, axp::preset::Gemm64x64x16, iro::cap::sm90, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::vectorized_elementwise_16x16, axp::preset::VectorizedElementwise16x16, iro::cap::sm89, axp::l4::profile::dev_fast);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::vectorized_elementwise_16x16, axp::preset::VectorizedElementwise16x16, iro::cap::sm89, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::vectorized_elementwise_16x16, axp::preset::VectorizedElementwise16x16, iro::cap::sm90, axp::l4::profile::dev_fast);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::vectorized_elementwise_16x16, axp::preset::VectorizedElementwise16x16, iro::cap::sm90, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::rmsnorm_16x16, axp::preset::RMSNorm16x16, iro::cap::sm89, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::rmsnorm_16x16, axp::preset::RMSNorm16x16, iro::cap::sm90, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::sort_16, axp::preset::Sort16, iro::cap::sm89, axp::l4::profile::dev_fast);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::sort_16, axp::preset::Sort16, iro::cap::sm89, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::sort_16, axp::preset::Sort16, iro::cap::sm90, axp::l4::profile::dev_fast);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::sort_16, axp::preset::Sort16, iro::cap::sm90, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::softmax_row4, axp::preset::SoftmaxRow4, iro::cap::sm89, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::softmax_row4, axp::preset::SoftmaxRow4, iro::cap::sm90, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::histogram_256, axp::preset::Histogram256, iro::cap::sm89, axp::l4::profile::dev_fast);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::histogram_256, axp::preset::Histogram256, iro::cap::sm89, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::histogram_256, axp::preset::Histogram256, iro::cap::sm90, axp::l4::profile::dev_fast);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::histogram_256, axp::preset::Histogram256, iro::cap::sm90, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::gemm_16x16x16, axp::preset::Gemm16x16x16, iro::cap::sm89, axp::l4::profile::dev_fast);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::gemm_16x16x16, axp::preset::Gemm16x16x16, iro::cap::sm89, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_ENTRY(axp::l4::graph_registry::hashes::fa2_decode_attention_64x64_hd128, axp::preset::Attention64x64x128Decode, iro::cap::sm90, axp::l4::profile::proof_full);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::attention_16x16, axp::preset::Attention16x16, iro::cap::sm89);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::layernorm_16x16, axp::preset::LayerNorm16x16, iro::cap::sm89);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::layernorm_16x16, axp::preset::LayerNorm16x16, iro::cap::sm90);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::gemm_64x64x16_bias_silu, axp::preset::Gemm64x64x16BiasSiLU, iro::cap::sm90);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::fa2_prefill_attention_64x64_hd128, axp::preset::Attention64x64x128Prefill, iro::cap::sm90);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::attention_64x64, axp::preset::Attention64x64, iro::cap::sm90);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::gemm_64x64x16, axp::preset::Gemm64x64x16, iro::cap::sm90);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::vectorized_elementwise_16x16, axp::preset::VectorizedElementwise16x16, iro::cap::sm89);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::vectorized_elementwise_16x16, axp::preset::VectorizedElementwise16x16, iro::cap::sm90);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::rmsnorm_16x16, axp::preset::RMSNorm16x16, iro::cap::sm89);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::rmsnorm_16x16, axp::preset::RMSNorm16x16, iro::cap::sm90);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::sort_16, axp::preset::Sort16, iro::cap::sm89);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::sort_16, axp::preset::Sort16, iro::cap::sm90);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::softmax_row4, axp::preset::SoftmaxRow4, iro::cap::sm89);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::softmax_row4, axp::preset::SoftmaxRow4, iro::cap::sm90);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::histogram_256, axp::preset::Histogram256, iro::cap::sm89);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::histogram_256, axp::preset::Histogram256, iro::cap::sm90);
#endif

#if defined(AXP_ENABLE_SM89)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::gemm_16x16x16, axp::preset::Gemm16x16x16, iro::cap::sm89);
#endif

#if defined(AXP_ENABLE_SM90)
AXP_GRAPH_HASH_OVERRIDE(axp::l4::graph_registry::hashes::fa2_decode_attention_64x64_hd128, axp::preset::Attention64x64x128Decode, iro::cap::sm90);
#endif

#undef AXP_GRAPH_ENTRY
#undef AXP_GRAPH_HASH_OVERRIDE
