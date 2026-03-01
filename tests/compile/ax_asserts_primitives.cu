/**
 * @file ax_asserts_primitives.cu
 * @brief Compile-time verification for AX primitives and patterns.
 */

#include <iro_rust_cuda_ffi.h>
#include "iro_cuda_ax_core.hpp"
#include <axp/primitives.hpp>
#include <axp/protocol/audit/tokens.hpp>
#include "ax_asserts_defs.hpp"

namespace axp_compile_test {
static_assert(iro::recipe::Exact::id != iro::recipe::Fast::id);
static_assert(iro::recipe::Exact::id != iro::recipe::ApproxExp::id);
static_assert(iro::recipe::Fast::id != iro::recipe::ApproxExp::id);
static_assert(RecipeF16::id != RecipeF16Fast::id);
static_assert(RecipeF16::id != RecipeF16Approx::id);

using Issue = axp::protocol::stage::IssueGmemToSmemSlot<
    RecipeF16, InTileG, OutTileS, ASubj, axp::tag::PipeA, SlotSubj,
    iro::exec::block, iro::token::lifetime::block, 2>;
using Wait = axp::protocol::stage::WaitSmemSlot<
    RecipeF16, OutTileS, SlotSubj, iro::exec::block, iro::token::lifetime::block>;
using Release = axp::protocol::stage::ReleaseSmemSlot<
    RecipeF16, SlotSubj, iro::exec::block, iro::token::lifetime::block>;
using Mark = axp::protocol::stage::MarkConsumed<
    RecipeF16, SlotSubj, iro::exec::block, iro::token::lifetime::block, OutTileS::bytes>;

static_assert(iro::util::size_v<typename Issue::inputs> == 2);
static_assert(iro::util::size_v<typename Wait::outputs> == 2);
static_assert(iro::util::size_v<typename Release::outputs> == 1);
static_assert(iro::util::size_v<typename Mark::inputs> == 1);
static_assert(axp::protocol::audit::contract_tokens_complete<Issue>());
static_assert(axp::protocol::audit::contract_tokens_complete<Wait>());
static_assert(axp::protocol::audit::contract_tokens_complete<Release>());
static_assert(axp::protocol::audit::contract_tokens_complete<Mark>());

using Cast = axp::protocol::convert::CastTile<
    RecipeF16, RecipeBF16,
    InTileG,
    iro::contract::Tile<
        iro::contract::Shape<64, 64>,
        iro::elem::bf16,
        iro::contract::layout::RowMajor<64>,
        iro::contract::space::global,
        iro::contract::Align<16>
    >,
    ASubj, BSubj, iro::exec::block, 16,
    iro::contract::no_dist, iro::contract::no_dist>;
static_assert(iro::util::size_v<typename Cast::outputs> == 1);
static_assert(axp::protocol::audit::contract_tokens_complete<Cast>());

using CastFP8ToF16 = axp::protocol::convert::CastTile<
    RecipeE4M3, RecipeF16,
    InTileFP8G,
    OutTileG,
    ASubj, BSubj, iro::exec::block, 8,
    iro::contract::no_dist, iro::contract::no_dist>;
static_assert(axp::protocol::audit::contract_tokens_complete<CastFP8ToF16>());

using CastFrag = axp::protocol::convert::CastFragment<
    RecipeF16Acc, RecipeF32,
    FragF16, FragF32,
    ASubj, OSubj, iro::exec::warp
>;
static_assert(axp::protocol::audit::contract_tokens_complete<CastFrag>());

using CastFragFP8 = axp::protocol::convert::CastFragment<
    RecipeE4M3Acc, RecipeF16Acc,
    FragE4M3, FragF16_1,
    ASubj, OSubj, iro::exec::warp
>;
static_assert(axp::protocol::audit::contract_tokens_complete<CastFragFP8>());

using Reduce = axp::protocol::reduction::BlockReduce<
    RecipeF32,
    iro::contract::Tile<
        iro::contract::Shape<256>,
        iro::elem::f32,
        iro::contract::layout::RowMajor<256>,
        iro::contract::space::global,
        iro::contract::Align<16>
    >,
    iro::contract::Tile<
        iro::contract::Shape<1>,
        iro::elem::f32,
        iro::contract::layout::RowMajor<1>,
        iro::contract::space::global,
        iro::contract::Align<16>
    >,
    ASubj, OSubj, iro::exec::block
>;
static_assert(iro::util::size_v<typename Reduce::outputs> == 1);
static_assert(axp::protocol::audit::contract_tokens_complete<Reduce>());

using WarpgroupReduce = axp::protocol::reduction::WarpgroupReduce<
    RecipeF32, ScalarF32, SSubjA, iro::exec::warpgroup, axp::protocol::reduction::op_add
>;
static_assert(axp::protocol::audit::contract_tokens_complete<WarpgroupReduce>());

// Additional protocol audits
using MaskFragT = axp::protocol::mask::MaskFrag<
    iro::contract::Shape<32>,
    iro::dist::mask<iro::scope::warp>
>;
using MaskGen = axp::protocol::mask::MaskGen<
    RecipeF32, MaskFragT, VSubjA, iro::exec::warp>;
using MaskApply = axp::protocol::mask::MaskApply<
    RecipeF32, FragF32, MaskFragT, FragF32, FSubjA, VSubjA, FSubjO, iro::exec::warp>;
static_assert(axp::protocol::audit::contract_tokens_complete<MaskGen>());
static_assert(axp::protocol::audit::contract_tokens_complete<MaskApply>());

struct CausalMaskTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.test.causal_mask"); };
struct CausalPredTag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.test.causal_pred"); };
using CausalMaskSubj = axp::subject::indexed<CausalMaskTag, 0>;
using CausalPredSubj = axp::subject::indexed<CausalPredTag, 0>;
using CausalMaskPayload = iro::contract::MaskDesc<32, iro::dist::replicated>;
using CausalPredPayload = iro::contract::ScalarDesc<iro::elem::u8, iro::dist::replicated>;
using CoordPayload = iro::contract::ScalarDesc<iro::elem::i32, iro::dist::uniform<iro::scope::block>>;
using CausalMaskPred = axp::protocol::mask::CausalMaskPred<
    RecipeF32, CausalMaskPayload, CausalPredPayload,
    CoordPayload, CoordPayload,
    Coord0Subj, Coord1Subj,
    CausalMaskSubj, CausalPredSubj,
    iro::exec::warp, 16, 32>;
static_assert(axp::protocol::audit::contract_tokens_complete<CausalMaskPred>());

using SharedToFrag = axp::protocol::ownership::SharedTileToFragment<
    RecipeF16, SmemTileF16, FragF16, ASubj, FSubjA, iro::exec::warp, iro::token::lifetime::block>;
using FragToShared = axp::protocol::ownership::FragmentToSharedTile<
    RecipeF16Acc, FragF16, SmemTileF16, FSubjA, ASubj, iro::exec::warp, iro::token::lifetime::block>;
using TileToFrag = axp::protocol::ownership::TileToFragment<
    RecipeF16, SmemTileF16, FragF16, ASubj, FSubjA, iro::exec::warp, iro::token::lifetime::block>;
using FragToTile = axp::protocol::ownership::FragmentToTile<
    RecipeF16Acc, FragF16, SmemTileF16, FSubjA, ASubj, iro::exec::warp, iro::token::lifetime::block>;
using WgmmaDescA = axp::protocol::ownership::WgmmaSmemDesc<
    SmemTileF16_128, ASubj, axp::protocol::stage::SwizzleAtom_128B>;
using MakeWgmmaDesc = axp::protocol::ownership::MakeWgmmaSmemDesc<
    RecipeF16, SmemTileF16_128, ASubj, OSubj, iro::exec::warpgroup, iro::token::lifetime::warpgroup,
    axp::protocol::stage::SwizzleAtom_128B>;
static_assert(axp::protocol::audit::contract_tokens_complete<SharedToFrag>());
static_assert(axp::protocol::audit::contract_tokens_complete<FragToShared>());
static_assert(axp::protocol::audit::contract_tokens_complete<TileToFrag>());
static_assert(axp::protocol::audit::contract_tokens_complete<FragToTile>());
static_assert(axp::protocol::audit::contract_tokens_complete<MakeWgmmaDesc>());

#if defined(AXP_ENABLE_SM100) || defined(AXP_ENABLE_EXPERIMENTAL_TMEM)
using TmemTo = axp::protocol::tmem::TileToTmem<
    RecipeF32, SmemTileF32, TmemTileF32, ASubj, OSubj, iro::exec::warpgroup,
    iro::token::lifetime::warpgroup, axp::dist::reg_tile, axp::dist::tmem_tile>;
using TmemFrom = axp::protocol::tmem::TmemToTile<
    RecipeF32, TmemTileF32, SmemTileF32, ASubj, OSubj, iro::exec::warpgroup,
    iro::token::lifetime::warpgroup, axp::dist::tmem_tile, axp::dist::reg_tile>;
static_assert(axp::protocol::audit::contract_tokens_complete<TmemTo>());
static_assert(axp::protocol::audit::contract_tokens_complete<TmemFrom>());
#endif

using ViewReq = axp::bundle::TileInTokens<ASubj, iro::exec::block, iro::token::lifetime::block>;
using ViewProv = axp::bundle::TileOutTokens<ASubj, iro::exec::block, iro::token::lifetime::block>;
using TileView = typename axp::protocol::view::TileView<
    RecipeF16, SmemTileF16, SmemTileF16, ASubj, iro::exec::block, ViewReq, ViewProv>::obligation;
static_assert(axp::protocol::audit::contract_tokens_complete<TileView>());

using TransposeIn = iro::contract::Tile<
    iro::contract::Shape<16, 32>,
    iro::elem::f16,
    iro::contract::layout::RowMajor<32>,
    iro::contract::space::shared,
    iro::contract::Align<16>
>;
using TransposeOut = iro::contract::Tile<
    iro::contract::Shape<32, 16>,
    iro::elem::f16,
    iro::contract::layout::ColMajor<32>,
    iro::contract::space::shared,
    iro::contract::Align<16>
>;
using TransposeView = typename axp::protocol::view::TransposeView<
    RecipeF16, TransposeIn, TransposeOut, ASubj, iro::exec::block, ViewReq, ViewProv>::obligation;
static_assert(axp::protocol::audit::contract_tokens_complete<TransposeView>());

using SwzOutTile = iro::contract::Tile<
    iro::contract::Shape<64, 64>,
    iro::elem::f16,
    iro::contract::layout::Swizzled<64,
        axp::protocol::stage::SwizzleAtom_128B::B,
        axp::protocol::stage::SwizzleAtom_128B::S>,
    iro::contract::space::shared,
    iro::contract::Align<128>
>;
using SwizzleView = typename axp::protocol::view::SwizzleView<
    RecipeF16, SmemTileF16_128, SwzOutTile, ASubj, iro::exec::block, ViewReq, ViewProv,
    axp::protocol::stage::SwizzleAtom_128B>::obligation;
static_assert(axp::protocol::audit::contract_tokens_complete<SwizzleView>());

using ReshapeIn = iro::contract::Tile<
    iro::contract::Shape<256>,
    iro::elem::f16,
    iro::contract::layout::RowMajor<256>,
    iro::contract::space::shared,
    iro::contract::Align<16>
>;
using ReshapeOut = iro::contract::Tile<
    iro::contract::Shape<16, 16>,
    iro::elem::f16,
    iro::contract::layout::RowMajor<16>,
    iro::contract::space::shared,
    iro::contract::Align<16>
>;
using ReshapeView = typename axp::protocol::view::ReshapeView<
    RecipeF16, ReshapeIn, ReshapeOut, ASubj, iro::exec::block, ViewReq, ViewProv>::obligation;
static_assert(axp::protocol::audit::contract_tokens_complete<ReshapeView>());

using SyncPt = axp::protocol::sync::SyncPoint<
    RecipeF32, BarSubj, iro::exec::warp>;
static_assert(axp::protocol::audit::contract_tokens_complete<SyncPt>());
using SyncWarp = axp::protocol::sync::SyncWarp<
    RecipeF32, BarSubj, iro::exec::warp>;
using SyncThreads = axp::protocol::sync::SyncThreads<
    RecipeF32, BarSubj, iro::exec::block>;
using FenceBlock = axp::protocol::sync::Fence<
    RecipeF32, BarSubj, iro::exec::block, iro::scope::block>;
static_assert(axp::protocol::audit::contract_tokens_complete<SyncWarp>());
static_assert(axp::protocol::audit::contract_tokens_complete<SyncThreads>());
static_assert(axp::protocol::audit::contract_tokens_complete<FenceBlock>());

using HostMap = axp::protocol::tma::HostMakeTensorMap<
    RecipeF16, axp::protocol::tma::TensorMapHandle<InTileG, MapSubj>, MapSubj, iro::exec::block>;
using BulkCopy2D = axp::protocol::tma::BulkTmaCopy2D<
    RecipeF16, axp::protocol::tma::TensorMapHandle<InTileG, MapSubj>, SmemTileF16,
    MapSubj, ASubj, BarSubj, CoordI32, CoordI32, Coord0Subj, Coord1Subj,
    iro::exec::block, iro::token::lifetime::block, SmemTileF16::bytes>;
using BulkStore2D = axp::protocol::tma::BulkTmaStore2D<
    RecipeF16, axp::protocol::tma::TensorMapHandle<InTileG, MapSubj>, SmemTileF16,
    MapSubj, ASubj, BarSubj, CoordI32, CoordI32, Coord0Subj, Coord1Subj,
    iro::exec::block, iro::token::lifetime::block, SmemTileF16::bytes>;
static_assert(axp::protocol::audit::contract_tokens_complete<HostMap>());
static_assert(axp::protocol::audit::contract_tokens_complete<BulkCopy2D>());
static_assert(axp::protocol::audit::contract_tokens_complete<BulkStore2D>());

using CpAsyncIssue = axp::protocol::stage::CpAsyncIssue<
    RecipeF16, InTileG, SmemTileF16, ASubj, axp::tag::PipeA, SlotSubj, iro::exec::block,
    iro::token::lifetime::block, 2>;
using CpAsyncCommit = axp::protocol::stage::CpAsyncCommit<
    RecipeF16, SmemTileF16, axp::tag::PipeA, SlotSubj, iro::exec::block,
    iro::token::lifetime::block, 2>;
using CpAsyncWait = axp::protocol::stage::CpAsyncWait<
    RecipeF16, SmemTileF16, axp::tag::PipeA, SlotSubj, iro::exec::block,
    iro::token::lifetime::block, 2, 0>;
static_assert(axp::protocol::audit::contract_tokens_complete<CpAsyncIssue>());
static_assert(axp::protocol::audit::contract_tokens_complete<CpAsyncCommit>());
static_assert(axp::protocol::audit::contract_tokens_complete<CpAsyncWait>());

// -----------------------------------------------------------------------------
// L0 atom instantiations (compile-time contract validation)
// -----------------------------------------------------------------------------

static_assert(axp::level0::detail::is_supported_exec_scan_warp<iro::exec::warp>::value);
static_assert(!axp::level0::detail::is_supported_exec_scan_warp<iro::exec::block>::value);
static_assert(axp::level0::detail::is_supported_exec_scan_block<iro::exec::block>::value);
static_assert(!axp::level0::detail::is_supported_exec_scan_block<iro::exec::warp>::value);

static_assert(axp::level0::detail::is_supported_exec_memory<iro::exec::lane>::value);
static_assert(axp::level0::detail::is_supported_exec_memory<iro::exec::warp>::value);
static_assert(axp::level0::detail::is_supported_exec_memory<iro::exec::warpgroup>::value);
static_assert(axp::level0::detail::is_supported_exec_memory<iro::exec::block>::value);
static_assert(axp::level0::detail::is_supported_exec_memory<iro::exec::cluster>::value);
static_assert(!axp::level0::detail::is_supported_exec_memory<iro::exec::cta_group1>::value);
static_assert(!axp::level0::detail::is_supported_exec_memory<iro::exec::cta_group2>::value);

static_assert(axp::level0::detail::is_supported_exec_warp<iro::exec::warp>::value);
static_assert(!axp::level0::detail::is_supported_exec_warp<iro::exec::warpgroup>::value);
static_assert(!axp::level0::detail::is_supported_exec_warp<iro::exec::block>::value);

static_assert(axp::level0::detail::is_supported_exec_frag<iro::exec::warp>::value);
static_assert(axp::level0::detail::is_supported_exec_frag<iro::exec::warpgroup>::value);
static_assert(!axp::level0::detail::is_supported_exec_frag<iro::exec::block>::value);

static_assert(axp::level0::detail::exec_supported<FragF32, iro::exec::warp>());
static_assert(axp::level0::detail::exec_supported<FragF32, iro::exec::warpgroup>());
static_assert(!axp::level0::detail::exec_supported<FragF32, iro::exec::block>());
static_assert(axp::level0::detail::exec_supported<ScalarF32, iro::exec::lane>());
static_assert(axp::level0::detail::exec_supported<ScalarF32, iro::exec::block>());
static_assert(!axp::level0::detail::exec_supported<ScalarF32, iro::exec::cluster>());

using L0Fma = axp::level0::Fma<
    RecipeF32, FragF32, FSubjA, FSubjB, FSubjC, FSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0Fma::inputs> == 3);
static_assert(iro::util::size_v<typename L0Fma::outputs> == 1);

using L0Exp = axp::level0::Exp<
    RecipeF32, FragF32, FSubjA, FSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0Exp::inputs> == 1);
static_assert(iro::util::size_v<typename L0Exp::outputs> == 1);

using L0Log = axp::level0::Log<
    RecipeF32, ScalarF32, SSubjA, SSubjO, iro::exec::lane>;
static_assert(iro::util::size_v<typename L0Log::outputs> == 1);

using L0Tanh = axp::level0::Tanh<
    RecipeF32, ScalarF32, SSubjA, SSubjO, iro::exec::lane>;
static_assert(iro::util::size_v<typename L0Tanh::outputs> == 1);

using L0Rsqrt = axp::level0::Rsqrt<
    RecipeF32, ScalarF32, SSubjA, SSubjO, iro::exec::lane>;
static_assert(iro::util::size_v<typename L0Rsqrt::outputs> == 1);

using L0Abs = axp::level0::Abs<
    RecipeF32, ScalarF32, SSubjA, SSubjO, iro::exec::lane>;
static_assert(iro::util::size_v<typename L0Abs::outputs> == 1);

using L0Neg = axp::level0::Neg<
    RecipeF32, ScalarF32, SSubjA, SSubjO, iro::exec::lane>;
static_assert(iro::util::size_v<typename L0Neg::outputs> == 1);

using L0Rcp = axp::level0::Rcp<
    RecipeF32, ScalarF32, SSubjA, SSubjO, iro::exec::lane>;
static_assert(iro::util::size_v<typename L0Rcp::outputs> == 1);

using L0Sqrt = axp::level0::Sqrt<
    RecipeF32, ScalarF32, SSubjA, SSubjO, iro::exec::lane>;
static_assert(iro::util::size_v<typename L0Sqrt::outputs> == 1);

using L0Sigmoid = axp::level0::Sigmoid<
    RecipeF32, ScalarF32, SSubjA, SSubjO, iro::exec::lane>;
static_assert(iro::util::size_v<typename L0Sigmoid::outputs> == 1);

using L0Gelu = axp::level0::Gelu<
    RecipeF32, ScalarF32, SSubjA, SSubjO, iro::exec::lane>;
static_assert(iro::util::size_v<typename L0Gelu::outputs> == 1);

using L0Max = axp::level0::Max<
    RecipeF32, FragF32, FSubjA, FSubjB, FSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0Max::inputs> == 2);

using L0Min = axp::level0::Min<
    RecipeF32, VectorF32, SSubjA, SSubjB, SSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0Min::outputs> == 1);

using L0Add = axp::level0::Add<
    RecipeF32, VectorF32, SSubjA, SSubjB, SSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0Add::outputs> == 1);

using L0Sub = axp::level0::Sub<
    RecipeF32, VectorF32, SSubjA, SSubjB, SSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0Sub::outputs> == 1);

using L0Mul = axp::level0::Mul<
    RecipeF32, ScalarF32, SSubjA, SSubjB, SSubjO, iro::exec::lane>;
static_assert(iro::util::size_v<typename L0Mul::outputs> == 1);

using L0Div = axp::level0::Div<
    RecipeF32, ScalarF32, SSubjA, SSubjB, SSubjO, iro::exec::lane>;
static_assert(iro::util::size_v<typename L0Div::outputs> == 1);

using L0Clamp = axp::level0::Clamp<
    RecipeF32, ScalarF32, SSubjA, SSubjB, SSubjC, SSubjO, iro::exec::lane>;
static_assert(iro::util::size_v<typename L0Clamp::outputs> == 1);

using L0Select = axp::level0::Select<
    RecipeF32, ScalarF32, MaskW32, SSubjA, SSubjB, VSubjA, SSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0Select::inputs> == 3);

using L0MaskNot = axp::level0::MaskNot<
    RecipeU32, MaskW32, VSubjA, VSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0MaskNot::outputs> == 1);

using L0MaskAnd = axp::level0::MaskAnd<
    RecipeU32, MaskW32, VSubjA, VSubjB, VSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0MaskAnd::outputs> == 1);

using L0MaskOr = axp::level0::MaskOr<
    RecipeU32, MaskW32, VSubjA, VSubjB, VSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0MaskOr::outputs> == 1);

using L0MaskXor = axp::level0::MaskXor<
    RecipeU32, MaskW32, VSubjA, VSubjB, VSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0MaskXor::outputs> == 1);

using L0Shuffle = axp::level0::Shuffle<
    RecipeF32, FragF32, FSubjA, FSubjO, iro::exec::warp, axp::level0::shuffle::down, 16>;
static_assert(L0Shuffle::delta == 16);

using L0Vote = axp::level0::Vote<
    RecipeU32, FragU32, FragU32, VSubjA, VSubjO, iro::exec::warp, axp::level0::vote::ballot>;
static_assert(iro::util::size_v<typename L0Vote::outputs> == 1);

using L0Broadcast = axp::level0::Broadcast<
    RecipeF32, FragF32, FSubjA, FSubjO, iro::exec::warp, 0>;
static_assert(iro::util::size_v<typename L0Broadcast::outputs> == 1);

using L0Elect = axp::level0::ElectOne<
    RecipeU32, MaskW32, VSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0Elect::outputs> == 1);

using L0FragScale = axp::level0::FragmentScale<
    RecipeF32, FragF32, ScalarF32, FSubjA, SSubjA, FSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0FragScale::outputs> == 1);

using L0FragClamp = axp::level0::FragmentClamp<
    RecipeF32, FragF32, ScalarF32, ScalarF32, FSubjA, SSubjA, SSubjB, FSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0FragClamp::outputs> == 1);

using L0FragPermute = axp::level0::FragmentPermute<
    RecipeF32, FragF32, FSubjA, FSubjO, iro::exec::warp, axp::level0::permute::reverse>;
static_assert(iro::util::size_v<typename L0FragPermute::outputs> == 1);
using L0FragExtract = axp::level0::FragmentExtract<
    RecipeF32, FragF32, ScalarF32, FSubjA, SSubjO, iro::exec::warp, 0>;
using L0FragInsert = axp::level0::FragmentInsert<
    RecipeF32, FragF32, ScalarF32, FSubjA, SSubjA, FSubjO, iro::exec::warp, 1>;
using L0FragSlice = axp::level0::FragmentSlice<
    RecipeF32, FragF32, FragF32_4, FSubjA, FSubjO, iro::exec::warp, 0, 4>;
using L0FragBroadcast = axp::level0::FragmentBroadcast<
    RecipeF32, FragF32, ScalarF32, SSubjA, FSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L0FragExtract::outputs> == 1);
static_assert(iro::util::size_v<typename L0FragInsert::outputs> == 1);
static_assert(iro::util::size_v<typename L0FragSlice::outputs> == 1);
static_assert(iro::util::size_v<typename L0FragBroadcast::outputs> == 1);

using L0WarpScan = axp::level0::WarpScan<
    RecipeF32, FragF32, FSubjA, iro::exec::warp, ScanAdd, axp::level0::scan::inclusive>;
static_assert(iro::util::size_v<typename L0WarpScan::outputs> == 1);
using L0WarpScanMul = axp::level0::WarpScan<
    RecipeF32, FragF32, FSubjA, iro::exec::warp, axp::protocol::reduction::op_mul, axp::level0::scan::inclusive>;
static_assert(iro::util::size_v<typename L0WarpScanMul::outputs> == 1);
using L0WarpScanAnd = axp::level0::WarpScan<
    RecipeU32, FragU32, FSubjA, iro::exec::warp, axp::protocol::reduction::op_and, axp::level0::scan::inclusive>;
static_assert(iro::util::size_v<typename L0WarpScanAnd::outputs> == 1);

using L0BlockScan = axp::level0::BlockScan<
    RecipeF32, ScalarF32, SSubjA, iro::exec::block, ScanAdd, axp::level0::scan::exclusive>;
static_assert(iro::util::size_v<typename L0BlockScan::outputs> == 1);

using RegTile = RegTileF16;

using L0LdG = axp::level0::LdGlobal<
    RecipeF16, InTileG, RegTile, ASubj, OSubj, iro::exec::block, axp::cache::ca,
    iro::contract::no_dist, axp::dist::reg_tile>;
static_assert(iro::util::size_v<typename L0LdG::outputs> == 1);

using L0StG = axp::level0::StGlobal<
    RecipeF16, RegTile, InTileG, OSubj, ASubj, iro::exec::block, axp::cache::wb,
    axp::dist::reg_tile, iro::contract::no_dist>;
static_assert(iro::util::size_v<typename L0StG::outputs> == 1);

using L0PrefetchG = axp::level0::PrefetchGlobal<
    RecipeF16, InTileG, ASubj, iro::exec::block, axp::cache::cg>;
static_assert(iro::util::size_v<typename L0PrefetchG::inputs> == 1);

using L0LdS = axp::level0::LdShared<
    RecipeF16, OutTileS, RegTile, ASubj, OSubj, iro::exec::block,
    iro::contract::no_dist, axp::dist::reg_tile>;
static_assert(iro::util::size_v<typename L0LdS::outputs> == 1);

using L0StS = axp::level0::StShared<
    RecipeF16, RegTile, OutTileS, OSubj, ASubj, iro::exec::block,
    axp::dist::reg_tile, iro::contract::no_dist>;
static_assert(iro::util::size_v<typename L0StS::outputs> == 1);

using SwzTileS = iro::contract::Tile<
    iro::contract::Shape<64, 64>,
    iro::elem::f16,
    iro::contract::layout::Swizzled<
        64,
        axp::protocol::stage::SwizzleAtom_128B::B,
        axp::protocol::stage::SwizzleAtom_128B::S>,
    iro::contract::space::shared,
    iro::contract::Align<128>
>;

using L0SwzLd = axp::level0::SwizzledLdShared<
    RecipeF16, SwzTileS, RegTile, ASubj, OSubj, iro::exec::block,
    axp::protocol::stage::SwizzleAtom_128B, iro::contract::no_dist, axp::dist::reg_tile>;
static_assert(iro::util::size_v<typename L0SwzLd::outputs> == 1);

using L0SwzSt = axp::level0::SwizzledStShared<
    RecipeF16, RegTile, SwzTileS, OSubj, ASubj, iro::exec::block,
    axp::protocol::stage::SwizzleAtom_128B, axp::dist::reg_tile, iro::contract::no_dist>;
static_assert(iro::util::size_v<typename L0SwzSt::outputs> == 1);

using WSubj = axp::subject::indexed<axp::tag::Acc, 7>;
using L0WgmmaFence = axp::level0::WgmmaFence<
    RecipeF16, WSubj, iro::exec::warpgroup>;
using L0WgmmaCommit = axp::level0::WgmmaCommitGroup<
    RecipeF16, WSubj, iro::exec::warpgroup, 0>;
using L0WgmmaWait = axp::level0::WgmmaWaitGroup<
    RecipeF16, WSubj, iro::exec::warpgroup, 0>;
static_assert(iro::util::size_v<typename L0WgmmaFence::inputs> == 1);
static_assert(iro::util::size_v<typename L0WgmmaCommit::inputs> == 1);
static_assert(iro::util::size_v<typename L0WgmmaWait::inputs> == 1);

using ShapeWmma = axp::protocol::compute::MmaShape<16, 16, 16, iro::elem::f16, iro::elem::f16, iro::elem::f32>;
using ShapeWgmma = axp::protocol::compute::MmaShape<64, 16, 16, iro::elem::f16, iro::elem::f16, iro::elem::f32>;

using ATile16 = iro::contract::Tile<
    iro::contract::Shape<16, 16>,
    iro::elem::f16,
    iro::contract::layout::RowMajor<16>,
    iro::contract::space::shared,
    iro::contract::Align<16>
>;
using BTile16 = iro::contract::Tile<
    iro::contract::Shape<16, 16>,
    iro::elem::f16,
    iro::contract::layout::ColMajor<16>,
    iro::contract::space::shared,
    iro::contract::Align<16>
>;
using AccFrag16 = iro::contract::FragmentDesc<
    iro::contract::Shape<16, 16>,
    iro::elem::f32,
    iro::dist::accumulator,
    8
>;
using WarpMma = axp::protocol::compute::WarpMmaFromSmem<
    RecipeF16, ShapeWmma, ATile16, BTile16, AccFrag16, ASubj, BSubj, FSubjO>;
static_assert(axp::protocol::audit::contract_tokens_complete<WarpMma>());

using ATileWg = iro::contract::Tile<
    iro::contract::Shape<64, 16>,
    iro::elem::f16,
    iro::contract::layout::RowMajor<16>,
    iro::contract::space::shared,
    iro::contract::Align<128>
>;
using BTileWg = iro::contract::Tile<
    iro::contract::Shape<16, 16>,
    iro::elem::f16,
    iro::contract::layout::ColMajor<16>,
    iro::contract::space::shared,
    iro::contract::Align<128>
>;
using AccFragWg = iro::contract::FragmentDesc<
    iro::contract::Shape<64, 16>,
    iro::elem::f32,
    iro::dist::accumulator,
    8
>;
using DescASubj = axp::subject::indexed<axp::tag::A, 9>;
using DescBSubj = axp::subject::indexed<axp::tag::B, 9>;
using ADesc = axp::protocol::ownership::WgmmaSmemDesc<
    ATileWg, ASubj, axp::protocol::stage::SwizzleAtom_128B>;
using BDesc = axp::protocol::ownership::WgmmaSmemDesc<
    BTileWg, BSubj, axp::protocol::stage::SwizzleAtom_128B>;
using WarpgroupMma = axp::protocol::compute::WarpgroupMmaFromDesc<
    RecipeF16, ShapeWgmma, ADesc, BDesc, AccFragWg, DescASubj, DescBSubj, FSubjO, WSubj>;
static_assert(axp::protocol::audit::contract_tokens_complete<WarpgroupMma>());

#if defined(AXP_ENABLE_SM100)
using TcgenMma = axp::protocol::compute::Tcgen05Mma<
    RecipeF32, SmemTileF32, SmemTileF32, TmemTileF32, ASubj, BSubj, OSubj,
    iro::exec::cta_group1, axp::dist::tmem_tile>;
static_assert(axp::protocol::audit::contract_tokens_complete<TcgenMma>());
#endif

using BarInit = axp::protocol::sync::BarrierInit<
    RecipeF32, BarSubj, iro::exec::block, 1>;
using BarExpect = axp::protocol::sync::BarrierExpectTx<
    RecipeF32, BarSubj, iro::exec::block, 128>;
using BarArrive = axp::protocol::sync::BarrierArriveTx<
    RecipeF32, BarSubj, iro::exec::block, 128>;
using BarWait = axp::protocol::sync::BarrierWait<
    RecipeF32, BarSubj, iro::exec::block>;
using BarTry = axp::protocol::sync::BarrierTryWait<
    RecipeF32, BarSubj, iro::exec::block, ScalarU32, SSubjO>;
using BarInv = axp::protocol::sync::BarrierInvalidate<
    RecipeF32, BarSubj, iro::exec::block, 1>;
static_assert(axp::protocol::audit::contract_tokens_complete<BarInit>());
static_assert(axp::protocol::audit::contract_tokens_complete<BarExpect>());
static_assert(axp::protocol::audit::contract_tokens_complete<BarArrive>());
static_assert(axp::protocol::audit::contract_tokens_complete<BarWait>());
static_assert(axp::protocol::audit::contract_tokens_complete<BarTry>());
static_assert(axp::protocol::audit::contract_tokens_complete<BarInv>());

using BarE0 = iro::compose::Edge<
    iro::util::at_t<typename BarInit::outputs, 0>,
    iro::util::at_t<typename BarExpect::inputs, 0>
>;
using BarE1 = iro::compose::Edge<
    iro::util::at_t<typename BarExpect::outputs, 0>,
    iro::util::at_t<typename BarArrive::inputs, 0>
>;
using BarE2 = iro::compose::Edge<
    iro::util::at_t<typename BarArrive::outputs, 0>,
    iro::util::at_t<typename BarWait::inputs, 0>
>;
using BarE3 = iro::compose::Edge<
    iro::util::at_t<typename BarWait::outputs, 0>,
    iro::util::at_t<typename BarTry::inputs, 0>
>;
using BarE4 = iro::compose::Edge<
    iro::util::at_t<typename BarTry::outputs, 0>,
    iro::util::at_t<typename BarInv::inputs, 0>
>;

using BarrierChain = iro::compose::Composition<
    iro::util::type_list<BarInit, BarExpect, BarArrive, BarWait, BarTry, BarInv>,
    iro::util::type_list<BarE0, BarE1, BarE2, BarE3, BarE4>,
    iro::util::type_list<iro::contract::res::named_barrier<BarSubj>>,
    iro::profile::BudgetMax,
    axp::target_cap
>;
static_assert(std::is_same_v<BarrierChain, BarrierChain>);

// -----------------------------------------------------------------------------
// L1 pattern instantiations (compile-time contract validation)
// -----------------------------------------------------------------------------

using L1WarpReduce = axp::level1::WarpReduce<
    RecipeF32, FragF32, FSubjA, iro::exec::warp, axp::level0::Add>;
template<class T>
consteval void validate_warp_reduce() {
    if constexpr (requires { typename T::obligations; }) {
        static_assert(iro::util::size_v<typename T::obligations> == 10);
    } else {
        static_assert(iro::util::size_v<typename T::inputs> == 1);
        static_assert(iro::util::size_v<typename T::outputs> == 1);
    }
}
static_assert((validate_warp_reduce<L1WarpReduce>(), true));

using SmemTileReduce = iro::contract::Tile<
    iro::contract::Shape<32>,
    iro::elem::f32,
    iro::contract::layout::RowMajor<32>,
    iro::contract::space::shared,
    iro::contract::Align<16>
>;
using SmemSubj = axp::subject::indexed<axp::tag::S, 0>;

using L1BlockReduce = axp::level1::BlockReduce<
    RecipeF32, FragF32, SmemTileReduce, FSubjA, SmemSubj, iro::exec::warp, axp::level0::Add>;
static_assert(iro::util::size_v<typename L1BlockReduce::obligations> > 0);

using L1VecLoad = axp::level1::VecLoad<
    RecipeF16, InTileG, OutTileS, ASubj, axp::tag::PipeA, SlotSubj,
    iro::exec::block, iro::token::lifetime::block, 2>;
static_assert(iro::util::size_v<typename L1VecLoad::obligations> == 2);

using L1VecStore = axp::level1::VecStore<
    RecipeF16, RegTile, OutTileG, OSubj, ASubj, iro::exec::block,
    axp::cache::wb, axp::dist::reg_tile, iro::contract::no_dist>;
static_assert(iro::util::size_v<typename L1VecStore::obligations> == 1);

using L1Scale = axp::level1::Scale<
    RecipeF32, FragF32, FSubjA, FSubjB, FSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L1Scale::obligations> == 1);

using L1Axpy = axp::level1::Axpy<
    RecipeF32, FragF32, FSubjA, FSubjB, FSubjC, FSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L1Axpy::obligations> == 1);

using L1Dot = axp::level1::Dot<
    RecipeF32, FragF32, FSubjA, FSubjB, FSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L1Dot::obligations> > 1);

using L1Nrm2 = axp::level1::Nrm2<
    RecipeF32, FragF32, FSubjA, FSubjB, FSubjC, FSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L1Nrm2::obligations> > 1);

using L1Asum = axp::level1::Asum<
    RecipeF32, FragF32, FSubjA, FSubjO, iro::exec::warp>;
static_assert(iro::util::size_v<typename L1Asum::obligations> > 1);

using L1Copy = axp::level1::Copy<
    RecipeF16, InTileG, OutTileG, ASubj, OSubj, iro::exec::block>;
static_assert(iro::util::size_v<typename L1Copy::obligations> == 2);

using CastOutTileG = iro::contract::Tile<
    iro::contract::Shape<64, 64>,
    iro::elem::bf16,
    iro::contract::layout::RowMajor<64>,
    iro::contract::space::global,
    iro::contract::Align<16>
>;
using L1Cast = axp::level1::Cast<
    RecipeF16, RecipeBF16, InTileG, CastOutTileG, ASubj, BSubj, iro::exec::block, 16>;
static_assert(iro::util::size_v<typename L1Cast::obligations> == 1);

} // namespace axp_compile_test

// =============================================================================
// Verify resolution selection (§8)
// =============================================================================

namespace resolve_test {

// Simple obligation with no ports
using EmptyObligation = iro::contract::Obligation<
    iro::util::type_list<>,
    iro::util::type_list<>,
    iro::util::type_list<>
>;

// Realization for the obligation
using EmptyRealization = iro::contract::Realization<EmptyObligation, iro::util::fnv1a_64_cstr("empty.impl")>;
using EmptyEntry = iro::registry::RealizationEntry<EmptyRealization, iro::cap::sm90>;

// Registry
using TestRegistry = iro::util::type_list<EmptyEntry>;

// Selection should find the realization
static_assert(iro::bind::match_realization<EmptyObligation, iro::cap::sm90, TestRegistry>::found);
static_assert(!iro::bind::match_realization<EmptyObligation, iro::cap::sm90, TestRegistry>::ambiguous);
static_assert(iro::bind::match_realization<EmptyObligation, iro::cap::sm90, TestRegistry>::match_count == 1);

// L4 graph-hash dispatch must resolve deterministically without preset-only fallback.
using SortGraph = axp::level3::registry::Select<axp::preset::Sort16, iro::cap::sm89>;
using L4SortResolve = axp::l4::resolve<SortGraph, iro::cap::sm89, axp::l4::profile::proof_full>;
static_assert(axp::l4::graph_registry::enabled_v<
              axp::graph::graph_hash_v<SortGraph>, iro::cap::sm89, axp::l4::profile::proof_full>);
static_assert(L4SortResolve::realization_key == iro::util::fnv1a_64_cstr("preset.sort.16"));
static_assert(std::is_same_v<typename L4SortResolve::type, SortGraph>);
static_assert(!axp::l4::supports<SortGraph, iro::cap::sm100, axp::l4::profile::proof_full>::value);

}  // namespace resolve_test

// =============================================================================
// Verify storage types are available under nvcc
// =============================================================================

#ifdef __CUDACC__
static_assert(iro::schema::ElemHasStorage<iro::elem::f16>);
static_assert(iro::schema::ElemHasStorage<iro::elem::f32>);
static_assert(iro::schema::ElemHasStorage<iro::elem::bf16>);

static_assert(std::is_same_v<iro::elem::f16::storage_t, __half>);
static_assert(std::is_same_v<iro::elem::f32::storage_t, float>);
static_assert(std::is_same_v<iro::elem::bf16::storage_t, __nv_bfloat16>);
#endif

// =============================================================================
// Dummy function to ensure this TU is linked
// =============================================================================

extern "C" void irffi_ax_asserts_linked() {
    // This function exists only to ensure this TU is linked.
    // If all static_asserts pass, the build succeeds.
}
