/**
 * @file ax_inst_l3_sm90.cu
 * @brief Explicit instantiations for L3 tiles (SM90).
 */

#include <iro_rust_cuda_ffi.h>
#include "iro_cuda_ax_core.hpp"
#include <axp/primitives.hpp>
#include "../compile/ax_asserts_defs.hpp"

namespace axp_compile_test {

template<class T>
struct ForceInstantiate {
    static constexpr int value = sizeof(T);
};

using L3GemmConfigSm90Small = axp::level3::GemmTileConfig<
    RecipeF16,
    16, 16, 16,
    2, 2,
    ASubj, BSubj, CSubj,
    WgmmaSubj
>;
using L3GemmSm90Small = axp::level3::GemmTile<L3GemmConfigSm90Small, iro::cap::sm90>;
static_assert(iro::util::size_v<typename L3GemmSm90Small::inputs> >= 2);
template struct ForceInstantiate<L3GemmSm90Small>;

using L3GemmConfigSm90 = axp::level3::GemmTileConfig<
    RecipeF16,
    64, 64, 16,
    2, 2,
    ASubj, BSubj, CSubj,
    WgmmaSubj
>;
using L3GemmSm90 = axp::level3::GemmTile<L3GemmConfigSm90, iro::cap::sm90>;
static_assert(iro::util::size_v<typename L3GemmSm90::inputs> >= 2);
using GemmExecSm90 = iro::exec::warpgroup_t<iro::cap::sm90::warpgroup_warps>;
using GemmReqSm90 = axp::detail::warpgroup_layout_resources_t<GemmExecSm90>;
static_assert(iro::contract::detail::resources_contain_all<L3GemmSm90::resources, GemmReqSm90>::value);
static_assert(iro::verify::resource_list_canonical<L3GemmSm90::resources>());
template struct ForceInstantiate<L3GemmSm90>;

using L3GemmSwizzleConfig = axp::level3::GemmTileConfig<
    RecipeF16,
    64, 64, 16,
    2, 2,
    ASubj, BSubj, CSubj,
    WgmmaSubj,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::None
>;
using L3GemmSwizzle = axp::level3::GemmTile<L3GemmSwizzleConfig, iro::cap::sm90>;
static_assert(iro::util::size_v<typename L3GemmSwizzle::inputs> >= 2);
template struct ForceInstantiate<L3GemmSwizzle>;

using GemmAMapHandle = axp::protocol::tma::TensorMapHandle<GemmATileG, GemmMapSubjA>;
using GemmBMapHandle = axp::protocol::tma::TensorMapHandle<GemmBTileG, GemmMapSubjB>;
using GemmATma = axp::level2::staging::TmaMulticastConfig<
    GemmAMapHandle, ClusterBarSubjA,
    MaskU32Cluster, MaskSubj,
    CoordI32Cluster, Coord0Subj,
    CoordI32Cluster, Coord1Subj
>;
using GemmBTma = axp::level2::staging::TmaMulticastConfig<
    GemmBMapHandle, ClusterBarSubjB,
    MaskU32Cluster, MaskSubj,
    CoordI32Cluster, Coord0Subj,
    CoordI32Cluster, Coord1Subj
>;
using L3GemmMulticastConfig = axp::level3::GemmTileConfigSm90Multicast<
    RecipeF16,
    64, 64, 16,
    2, 2,
    ASubj, BSubj, CSubj,
    WgmmaSubj,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::None,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::schedule::Pipelined,
    iro::contract::subject::global,
    iro::contract::subject::global,
    GemmATma, GemmBTma
>;
using L3GemmMulticast = axp::level3::GemmTile<L3GemmMulticastConfig, iro::cap::sm90>;
static_assert(iro::util::size_v<typename L3GemmMulticast::inputs> >= 2);
template struct ForceInstantiate<L3GemmMulticast>;

using L3AttentionConfigSm90Small = axp::level3::AttentionTileConfig<
    RecipeF16,
    16, 16, 16, 16,
    2, 0,
    QSubj, KSubj, VSubj,
    AttnAccSubj, AttnStateOldSubj, AttnStateNewSubj
>;
using L3AttentionSm90Small = axp::level3::AttentionTile<L3AttentionConfigSm90Small, iro::cap::sm90>;
static_assert(iro::util::size_v<typename L3AttentionSm90Small::inputs> >= 3);
template struct ForceInstantiate<L3AttentionSm90Small>;

using L3AttentionConfigSm90 = axp::level3::AttentionTileConfig<
    RecipeF16,
    64, 64, 64, 16,
    2, 0,
    QSubj, KSubj, VSubj,
    AttnAccSubj, AttnStateOldSubj, AttnStateNewSubj
>;
using L3AttentionSm90 = axp::level3::AttentionTile<L3AttentionConfigSm90, iro::cap::sm90>;
static_assert(iro::util::size_v<typename L3AttentionSm90::inputs> >= 3);
using AttnQKFragSm90 = iro::contract::FragmentDesc<
    iro::contract::Shape<64, 64>,
    RecipeF16::acc,
    iro::dist::accumulator,
    64 / 2
>;
using AttnOFragSm90 = iro::contract::FragmentDesc<
    iro::contract::Shape<64, 16>,
    RecipeF16::acc,
    iro::dist::accumulator,
    16 / 2
>;
using AttnRegPressureSm90 = axp::level3::detail::reg_pressure_obligation<
    64 + 2 * 8, AttnQKFragSm90, AttnOFragSm90>;
using AttnRegReqSm90 = iro::util::type_list<typename AttnRegPressureSm90::reg_t>;
static_assert(iro::contract::detail::resources_contain_all<L3AttentionSm90::resources, AttnRegReqSm90>::value);
template struct ForceInstantiate<L3AttentionSm90>;

using L3AttentionConfigSm90Causal = axp::level3::AttentionTileConfig<
    RecipeF16,
    64, 64, 64, 16,
    2, 0,
    QSubj, KSubj, VSubj,
    AttnAccSubj, AttnStateOldSubj, AttnStateNewSubj,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::Optimized,
    axp::intent::memory_pattern::Optimized,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::load_mode::AsyncPrefetch,
    axp::intent::schedule::Pipelined,
    axp::intent::tile_skip::Causal
>;
using L3AttentionSm90Causal = axp::level3::AttentionTile<L3AttentionConfigSm90Causal, iro::cap::sm90>;
static_assert(iro::util::size_v<typename L3AttentionSm90Causal::inputs> >= 3);
template struct ForceInstantiate<L3AttentionSm90Causal>;

using L3SoftmaxConfig = axp::level3::SoftmaxRowTileConfig<
    RecipeF16Acc,
    4,
    ASubj, OSubj
>;
using L3SoftmaxSm90 = axp::level3::SoftmaxRowTile<L3SoftmaxConfig, iro::cap::sm90>;
static_assert(iro::util::size_v<typename L3SoftmaxSm90::inputs> >= 1);
template struct ForceInstantiate<L3SoftmaxSm90>;

using L3LayerNormConfig = axp::level3::LayerNormTileConfig<
    RecipeF16,
    16, 16,
    ASubj, OSubj,
    GammaSubj, BetaSubj, EpsSubj
>;
using L3LayerNormSm90 = axp::level3::LayerNormTile<L3LayerNormConfig, iro::cap::sm90>;
static_assert(iro::util::size_v<typename L3LayerNormSm90::inputs> >= 1);
template struct ForceInstantiate<L3LayerNormSm90>;

using L3RMSNormConfig = axp::level3::RMSNormTileConfig<
    RecipeF16,
    16, 16,
    ASubj, OSubj,
    WeightSubj, EpsSubj
>;
using L3RMSNormSm90 = axp::level3::RMSNormTile<L3RMSNormConfig, iro::cap::sm90>;
static_assert(iro::util::size_v<typename L3RMSNormSm90::inputs> >= 1);
template struct ForceInstantiate<L3RMSNormSm90>;

using HistSharedTile = iro::contract::Tile<
    iro::contract::Shape<256>,
    iro::elem::f32,
    iro::contract::layout::Contiguous,
    iro::contract::space::shared,
    iro::contract::Align<16>
>;
using HistOutTile = iro::contract::Tile<
    iro::contract::Shape<256>,
    iro::elem::f32,
    iro::contract::layout::Contiguous,
    iro::contract::space::global,
    iro::contract::Align<16>
>;
using HistValuePayload = iro::contract::ScalarDesc<iro::elem::f32, iro::dist::lane>;
using HistIndexPayload = iro::contract::ScalarDesc<iro::elem::u32, iro::dist::lane>;
using L3HistogramConfig = axp::level3::HistogramTileConfig<
    RecipeF32,
    HistValuePayload,
    HistIndexPayload,
    HistSharedTile,
    HistOutTile,
    HistValueSubj,
    HistIndexSubj,
    HistSharedSubj,
    HistOutValSubj,
    HistOutSubj,
    iro::exec::block
>;
using L3HistogramSm90 = axp::level3::HistogramTile<L3HistogramConfig, iro::cap::sm90>;
static_assert(iro::util::size_v<typename L3HistogramSm90::inputs> >= 2);
template struct ForceInstantiate<L3HistogramSm90>;

using L3SortConfig = axp::level3::SortTileConfig<
    RecipeF32,
    16,
    SortInSubj,
    SortOutSubj
>;
using L3SortSm90 = axp::level3::SortTile<L3SortConfig, iro::cap::sm90>;
static_assert(iro::util::size_v<typename L3SortSm90::inputs> >= 1);
template struct ForceInstantiate<L3SortSm90>;

} // namespace axp_compile_test
