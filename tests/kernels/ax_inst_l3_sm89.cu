/**
 * @file ax_inst_l3_sm89.cu
 * @brief Explicit instantiations for L3 tiles (SM89).
 */

#include "iro_cuda_ax_core.hpp"
#include <axp/primitives.hpp>
#include "../compile/ax_asserts_defs.hpp"

namespace axp_compile_test {

template<class T>
struct ForceInstantiate {
    static constexpr int value = sizeof(T);
};

using L3GemmConfigSm89 = axp::level3::GemmTileConfig<
    RecipeF16,
    16, 16, 16,
    2, 2,
    ASubj, BSubj, CSubj,
    WgmmaSubj
>;
using L3GemmSm89 = axp::level3::GemmTile<L3GemmConfigSm89, iro::cap::sm89>;
static_assert(iro::util::size_v<typename L3GemmSm89::inputs> >= 2);
template struct ForceInstantiate<L3GemmSm89>;

using L3AttentionConfigSm89 = axp::level3::AttentionTileConfig<
    RecipeF16,
    16, 16, 16, 16,
    2, 0,
    QSubj, KSubj, VSubj,
    AttnAccSubj, AttnStateOldSubj, AttnStateNewSubj
>;
using L3AttentionSm89 = axp::level3::AttentionTile<L3AttentionConfigSm89, iro::cap::sm89>;
static_assert(iro::util::size_v<typename L3AttentionSm89::inputs> >= 3);
template struct ForceInstantiate<L3AttentionSm89>;

using L3SoftmaxConfig = axp::level3::SoftmaxRowTileConfig<
    RecipeF16Acc,
    4,
    ASubj, OSubj
>;
using L3SoftmaxSm89 = axp::level3::SoftmaxRowTile<L3SoftmaxConfig, iro::cap::sm89>;
static_assert(iro::util::size_v<typename L3SoftmaxSm89::inputs> >= 1);
template struct ForceInstantiate<L3SoftmaxSm89>;

using L3LayerNormConfig = axp::level3::LayerNormTileConfig<
    RecipeF16,
    16, 16,
    ASubj, OSubj,
    GammaSubj, BetaSubj, EpsSubj
>;
using L3LayerNormSm89 = axp::level3::LayerNormTile<L3LayerNormConfig, iro::cap::sm89>;
static_assert(iro::util::size_v<typename L3LayerNormSm89::inputs> >= 1);
template struct ForceInstantiate<L3LayerNormSm89>;

using L3RMSNormConfig = axp::level3::RMSNormTileConfig<
    RecipeF16,
    16, 16,
    ASubj, OSubj,
    WeightSubj, EpsSubj
>;
using L3RMSNormSm89 = axp::level3::RMSNormTile<L3RMSNormConfig, iro::cap::sm89>;
static_assert(iro::util::size_v<typename L3RMSNormSm89::inputs> >= 1);
template struct ForceInstantiate<L3RMSNormSm89>;

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
using L3HistogramSm89 = axp::level3::HistogramTile<L3HistogramConfig, iro::cap::sm89>;
static_assert(iro::util::size_v<typename L3HistogramSm89::inputs> >= 2);
template struct ForceInstantiate<L3HistogramSm89>;

using L3SortConfig = axp::level3::SortTileConfig<
    RecipeF32,
    16,
    SortInSubj,
    SortOutSubj
>;
using L3SortSm89 = axp::level3::SortTile<L3SortConfig, iro::cap::sm89>;
static_assert(iro::util::size_v<typename L3SortSm89::inputs> >= 1);
template struct ForceInstantiate<L3SortSm89>;

} // namespace axp_compile_test
