#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../concepts.hpp"
#include "../level0/compute.hpp"
#include "../level0/convert.hpp"
#include "../level0/memory.hpp"
#include "../level0/ownership.hpp"
#include "../level0/specialize.hpp"
#include "../level0/sync.hpp"
#include "../level0/stage.hpp"
#include "../level2/matmul.hpp"
#include "../level2/epilogue.hpp"
#include "../level2/staging.hpp"
#include "../level2/scale.hpp"
#include "../level2/wgmma.hpp"
#include "../level0/fragment.hpp"
#include "../swizzle.hpp"
#include "../intent.hpp"
#include "../kits/intent.hpp"
#include "detail/compose.hpp"
#include "detail/reg_pressure.hpp"
#include "registry.hpp"

namespace axp::level3::gemm {

namespace detail {

struct acc_raw_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.gemm.acc_raw"); };
struct acc_accum_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.gemm.acc_accum"); };
struct acc_wait_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.gemm.acc_wait"); };
struct epi_vec_in_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.gemm.epi_vec_in"); };
struct epi_vec_out_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.gemm.epi_vec_out"); };
struct epi_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.gemm.epi_frag"); };
struct desc_a_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.gemm.desc_a"); };
struct desc_b_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.gemm.desc_b"); };
struct wgmma_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.gemm.wgmma"); };
struct pipe_a_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.gemm.pipe_a"); };
struct pipe_b_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.gemm.pipe_b"); };

template<int I>
using acc_raw_subj = iro::contract::subject::indexed<acc_raw_tag, I>;

template<int I>
using acc_accum_subj = iro::contract::subject::indexed<acc_accum_tag, I>;

template<int I>
using acc_wait_subj = iro::contract::subject::indexed<acc_wait_tag, I>;

template<int I>
using desc_a_subj = iro::contract::subject::indexed<desc_a_tag, I>;

template<int I>
using desc_b_subj = iro::contract::subject::indexed<desc_b_tag, I>;

template<int I>
using epi_vec_in_subj = iro::contract::subject::indexed<epi_vec_in_tag, I>;

template<int I>
using epi_vec_out_subj = iro::contract::subject::indexed<epi_vec_out_tag, I>;

template<int I>
using epi_frag_subj = iro::contract::subject::indexed<epi_frag_tag, I>;

template<class Obligation, int I>
using in_port_t = axp::level3::detail::in_port_t<Obligation, I>;

template<class Obligation, int I>
using out_port_t = axp::level3::detail::out_port_t<Obligation, I>;

using axp::level3::detail::frag_reg_count_v;
using axp::level3::detail::reg_pressure_obligation;

template<class Issue, class Wait, bool HasTma, bool Streaming>
struct stage_issue_wait_edges;

template<class Issue, class Wait, bool HasTma>
struct stage_issue_wait_edges<Issue, Wait, HasTma, true> {
    using type = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<Issue, 0>, detail::in_port_t<Wait, 0>>
    >;
};

template<class Issue, class Wait>
struct stage_issue_wait_edges<Issue, Wait, false, false> {
    using type = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<Issue, 0>, detail::in_port_t<Wait, 0>>
    >;
};

template<class Issue, class Wait>
struct stage_issue_wait_edges<Issue, Wait, true, false> {
    using type = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<Issue, 0>, detail::in_port_t<Wait, 0>>,
        iro::compose::Edge<detail::out_port_t<Issue, 1>, detail::in_port_t<Wait, 1>>
    >;
};

template<class Issue, class Wait, bool HasTma, bool Streaming>
using stage_issue_wait_edges_t = typename stage_issue_wait_edges<Issue, Wait, HasTma, Streaming>::type;

template<bool Enable, class Elem, int Vec>
struct scale_tile {
    using type = void;
};

template<class Elem, int Vec>
struct scale_tile<true, Elem, Vec> {
    using type = axp::protocol::scale::ScaleTile<Elem, Vec>;
};

template<bool Enable, class Recipe, class Tile, class ScaleTile, class TileSubj, class ScaleSubj, class CapT>
struct scale_op {
    using type = axp::level2::scale::ScaleSharedTile<
        Recipe, Tile, ScaleTile, TileSubj, ScaleSubj, iro::exec::block, CapT
    >;
    using obligations = iro::util::type_list<type>;
};

template<class Recipe, class Tile, class ScaleTile, class TileSubj, class ScaleSubj, class CapT>
struct scale_op<false, Recipe, Tile, ScaleTile, TileSubj, ScaleSubj, CapT> {
    using type = void;
    using obligations = iro::util::type_list<>;
};

template<class Wait, class Scale, class Downstream, bool Enable, int DownstreamInput>
struct wait_scale_edges;

template<class Wait, class Scale, class Downstream, int DownstreamInput>
struct wait_scale_edges<Wait, Scale, Downstream, false, DownstreamInput> {
    using type = iro::util::type_list<
        iro::compose::Edge<out_port_t<Wait, 0>, in_port_t<Downstream, DownstreamInput>>
    >;
};

template<class Wait, class Scale, class Downstream, int DownstreamInput>
struct wait_scale_edges<Wait, Scale, Downstream, true, DownstreamInput> {
    using type = iro::util::type_list<
        iro::compose::Edge<out_port_t<Wait, 0>, in_port_t<Scale, 0>>,
        iro::compose::Edge<out_port_t<Scale, 0>, in_port_t<Downstream, DownstreamInput>>
    >;
};

template<class Wait, class Scale, class Downstream, bool Enable, int DownstreamInput>
using wait_scale_edges_t = typename wait_scale_edges<Wait, Scale, Downstream, Enable, DownstreamInput>::type;

template<int ElemBytes, int AlignBytes, int MaxVecBytes>
consteval int select_vec_bytes() {
    if constexpr ((MaxVecBytes >= 16) && (AlignBytes >= 16) && ((16 % ElemBytes) == 0)) {
        return 16;
    } else if constexpr ((MaxVecBytes >= 8) && (AlignBytes >= 8) && ((8 % ElemBytes) == 0)) {
        return 8;
    } else {
        return 4;
    }
}

template<class Recipe, class Elem, class Dist>
struct vec_payload_selector {
    static constexpr int vec_bytes = select_vec_bytes<Elem::bytes, Elem::align, Recipe::vec_bytes>();
    static constexpr int lanes = vec_bytes / Elem::bytes;
    using type = iro::contract::VectorDesc<Elem, lanes, Dist>;
};

} // namespace detail

namespace epilogue {

struct None {
    static constexpr bool enabled = false;
};

template<class BiasTag, template<class, class, class, class, class, class, class> class ActOp>
struct BiasActivationVec {
    static constexpr bool enabled = true;
    using bias_tag = BiasTag;
    template<class Recipe, class VecPayload, class InSubj, class BiasSubj, class OutSubj, class ExecGroup,
             class CapT = axp::target_cap>
    using op = axp::level2::epilogue::FusedBiasActivationVec<
        Recipe, typename VecPayload::elem, typename VecPayload::dist,
        InSubj, BiasSubj, OutSubj, ExecGroup, ActOp, CapT
    >;
};

template<class CTag, class AlphaTag, class BetaTag>
struct LinearCombinationVec {
    static constexpr bool enabled = true;
    using c_tag = CTag;
    using alpha_tag = AlphaTag;
    using beta_tag = BetaTag;
    template<class Recipe, class VecPayload, class AccSubj, class CSubj,
             class AlphaSubj, class BetaSubj, class OutSubj, class ExecGroup,
             class CapT = axp::target_cap>
    using op = axp::level2::epilogue::LinearCombinationVec<
        Recipe, typename VecPayload::elem, typename VecPayload::dist,
        AccSubj, CSubj, AlphaSubj, BetaSubj, OutSubj, ExecGroup, CapT
    >;
};

} // namespace epilogue

namespace detail {

template<class Recipe, class Frag, class ExecGroup, class EpiloguePolicy, class FragSubj, class FragProducer, class CapT>
struct build_vec_epilogue;

template<class Recipe, class Frag, class ExecGroup, class FragSubj, class FragProducer, class CapT>
struct build_vec_epilogue<Recipe, Frag, ExecGroup, epilogue::None, FragSubj, FragProducer, CapT> {
    using obligations = iro::util::type_list<>;
    using edges = iro::util::type_list<>;
    using out_frag_subj = FragSubj;
    using out_frag_producer = FragProducer;
};

template<class Recipe, class Frag, class ExecGroup, class BiasTag,
         template<class, class, class, class, class, class, class> class ActOp,
         class FragSubj, class FragProducer, class CapT>
struct build_vec_epilogue<Recipe, Frag, ExecGroup, epilogue::BiasActivationVec<BiasTag, ActOp>, FragSubj, FragProducer, CapT> {
    static_assert(iro::util::HasId<BiasTag>, "GemmTile epilogue BiasTag must have id");
    using VecPayload = typename vec_payload_selector<Recipe, typename Frag::elem, typename Frag::dist>::type;
    static constexpr int kVecLanes = static_cast<int>(VecPayload::lanes);
    static constexpr int kSlices = static_cast<int>(Frag::count) / kVecLanes;
    static_assert(kVecLanes > 0, "GemmTile epilogue: vector lanes must be positive");
    static_assert((static_cast<int>(Frag::count) % kVecLanes) == 0,
                  "GemmTile epilogue: fragment count must be multiple of vector lanes");

    template<int I, int End, class InFragSubj, class InFragProducer>
    struct build_slices {
        static constexpr int kOffset = I * kVecLanes;
        using VecInSubj = epi_vec_in_subj<I>;
        using VecOutSubj = epi_vec_out_subj<I>;
        using BiasSubj = iro::contract::subject::indexed<BiasTag, I>;

        struct Load : axp::level0::FragmentToVectorSlice<
            Recipe, Frag, VecPayload, InFragSubj, VecInSubj, ExecGroup, kOffset> {};
        struct Epi : epilogue::BiasActivationVec<BiasTag, ActOp>::template op<
            Recipe, VecPayload, VecInSubj, BiasSubj, VecOutSubj, ExecGroup, CapT> {};
        struct Store : axp::level0::VectorSliceToFragment<
            Recipe, Frag, VecPayload, InFragSubj, VecOutSubj, epi_frag_subj<I>, ExecGroup, kOffset> {};

        using curr_obligations = iro::util::type_list<Load, Epi, Store>;
        using curr_edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<InFragProducer, 0>, detail::in_port_t<Load, 0>>,
            iro::compose::Edge<detail::out_port_t<Load, 0>, detail::in_port_t<Epi, 0>>,
            iro::compose::Edge<detail::out_port_t<InFragProducer, 0>, detail::in_port_t<Store, 0>>,
            iro::compose::Edge<detail::out_port_t<Epi, 0>, detail::in_port_t<Store, 1>>
        >;

        using next = build_slices<I + 1, End, epi_frag_subj<I>, Store>;
        using obligations = iro::util::concat_t<curr_obligations, typename next::obligations>;
        using edges = iro::util::concat_t<curr_edges, typename next::edges>;
        using out_frag_subj = typename next::out_frag_subj;
        using out_frag_producer = typename next::out_frag_producer;
    };

    template<int End, class InFragSubj, class InFragProducer>
    struct build_slices<End, End, InFragSubj, InFragProducer> {
        using obligations = iro::util::type_list<>;
        using edges = iro::util::type_list<>;
        using out_frag_subj = InFragSubj;
        using out_frag_producer = InFragProducer;
    };

    using build = build_slices<0, kSlices, FragSubj, FragProducer>;
    using obligations = typename build::obligations;
    using edges = typename build::edges;
    using out_frag_subj = typename build::out_frag_subj;
    using out_frag_producer = typename build::out_frag_producer;
};

template<class Recipe, class Frag, class ExecGroup, class CTag, class AlphaTag, class BetaTag,
         class FragSubj, class FragProducer, class CapT>
struct build_vec_epilogue<Recipe, Frag, ExecGroup, epilogue::LinearCombinationVec<CTag, AlphaTag, BetaTag>, FragSubj, FragProducer, CapT> {
    static_assert(iro::util::HasId<CTag>, "GemmTile epilogue CTag must have id");
    static_assert(iro::util::HasId<AlphaTag>, "GemmTile epilogue AlphaTag must have id");
    static_assert(iro::util::HasId<BetaTag>, "GemmTile epilogue BetaTag must have id");
    using VecPayload = typename vec_payload_selector<Recipe, typename Frag::elem, typename Frag::dist>::type;
    static constexpr int kVecLanes = static_cast<int>(VecPayload::lanes);
    static constexpr int kSlices = static_cast<int>(Frag::count) / kVecLanes;
    static_assert(kVecLanes > 0, "GemmTile epilogue: vector lanes must be positive");
    static_assert((static_cast<int>(Frag::count) % kVecLanes) == 0,
                  "GemmTile epilogue: fragment count must be multiple of vector lanes");

    template<int I, int End, class InFragSubj, class InFragProducer>
    struct build_slices {
        static constexpr int kOffset = I * kVecLanes;
        using VecInSubj = epi_vec_in_subj<I>;
        using VecOutSubj = epi_vec_out_subj<I>;
        using CSubj = iro::contract::subject::indexed<CTag, I>;
        using AlphaSubj = iro::contract::subject::indexed<AlphaTag, I>;
        using BetaSubj = iro::contract::subject::indexed<BetaTag, I>;

        struct Load : axp::level0::FragmentToVectorSlice<
            Recipe, Frag, VecPayload, InFragSubj, VecInSubj, ExecGroup, kOffset> {};
        struct Epi : epilogue::LinearCombinationVec<CTag, AlphaTag, BetaTag>::template op<
            Recipe, VecPayload, VecInSubj, CSubj, AlphaSubj, BetaSubj, VecOutSubj, ExecGroup, CapT> {};
        struct Store : axp::level0::VectorSliceToFragment<
            Recipe, Frag, VecPayload, InFragSubj, VecOutSubj, epi_frag_subj<I>, ExecGroup, kOffset> {};

        using curr_obligations = iro::util::type_list<Load, Epi, Store>;
        using curr_edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<InFragProducer, 0>, detail::in_port_t<Load, 0>>,
            iro::compose::Edge<detail::out_port_t<Load, 0>, detail::in_port_t<Epi, 0>>,
            iro::compose::Edge<detail::out_port_t<InFragProducer, 0>, detail::in_port_t<Store, 0>>,
            iro::compose::Edge<detail::out_port_t<Epi, 0>, detail::in_port_t<Store, 1>>
        >;

        using next = build_slices<I + 1, End, epi_frag_subj<I>, Store>;
        using obligations = iro::util::concat_t<curr_obligations, typename next::obligations>;
        using edges = iro::util::concat_t<curr_edges, typename next::edges>;
        using out_frag_subj = typename next::out_frag_subj;
        using out_frag_producer = typename next::out_frag_producer;
    };

    template<int End, class InFragSubj, class InFragProducer>
    struct build_slices<End, End, InFragSubj, InFragProducer> {
        using obligations = iro::util::type_list<>;
        using edges = iro::util::type_list<>;
        using out_frag_subj = InFragSubj;
        using out_frag_producer = InFragProducer;
    };

    using build = build_slices<0, kSlices, FragSubj, FragProducer>;
    using obligations = typename build::obligations;
    using edges = typename build::edges;
    using out_frag_subj = typename build::out_frag_subj;
    using out_frag_producer = typename build::out_frag_producer;
};

} // namespace detail

// ----------------------------------------------------------------------------
// Warp-level WMMA tile (explicit prologue/steady/drain pipeline)
// ----------------------------------------------------------------------------

template<
    class Recipe,
    int BlockM, int BlockN, int BlockK,
    int Stages, int KTiles,
    class ASubj, class BSubj, class CSubj,
    class ScaleASubj, class ScaleBSubj,
    class WgmmaSubj,
    class MemoryPatternA, class MemoryPatternB,
    class LoadModeA, class LoadModeB,
    class Schedule,
    class PipeATag, class PipeBTag,
    class ATma = void, class BTma = void,
    class CapT = axp::target_cap,
    class EpiloguePolicy = epilogue::None
>
struct GemmTileWarpImpl {
    static_assert(std::is_same_v<typename Recipe::acc, iro::elem::f32>, "GemmTile: Recipe::acc must be f32");
    static_assert(Stages >= 2 && Stages <= 4, "GemmTile: Stages must be 2-4");
    static_assert(KTiles > 0, "GemmTile: KTiles must be > 0");

    using ExecGroup = iro::exec::warp;
    using AccRecipe = iro::recipe::Accumulate<Recipe>;

    using ElemA = iro::verify::recipe_in_a_t<Recipe>;
    using ElemB = iro::verify::recipe_in_b_t<Recipe>;
    using ScaleElemA = iro::verify::recipe_scale_a_t<Recipe>;
    using ScaleElemB = iro::verify::recipe_scale_b_t<Recipe>;
    static constexpr int kScaleVecA = iro::verify::recipe_scale_vec_a_v<Recipe>;
    static constexpr int kScaleVecB = iro::verify::recipe_scale_vec_b_v<Recipe>;
    static constexpr bool kHasScaleA = iro::verify::recipe_has_scale_a_v<Recipe>;
    static constexpr bool kHasScaleB = iro::recipe::is_precision_ab_v<Recipe> &&
                                       iro::verify::recipe_has_scale_b_v<Recipe>;
    using ScaleTileA = typename detail::scale_tile<kHasScaleA, ScaleElemA, kScaleVecA>::type;
    using ScaleTileB = typename detail::scale_tile<kHasScaleB, ScaleElemB, kScaleVecB>::type;
    using ScheduleT = axp::kit::detail::select_schedule_t<Schedule, CapT>;
    static constexpr bool kBulkSchedule = std::is_same_v<ScheduleT, axp::intent::schedule::BulkSynchronous>;
    static constexpr bool kProducerConsumer = std::is_same_v<ScheduleT, axp::intent::schedule::ProducerConsumer>;
    template<class Op>
    using ProducerOp = std::conditional_t<
        kProducerConsumer,
        axp::level0::SpecializedOp<axp::level0::role::producer, Op>,
        Op
    >;
    template<class Op>
    using ConsumerOp = std::conditional_t<
        kProducerConsumer,
        axp::level0::SpecializedOp<axp::level0::role::consumer, Op>,
        Op
    >;
    template<class Op>
    using ConsumerMaybe = std::conditional_t<
        kProducerConsumer && !std::is_void_v<Op>,
        ConsumerOp<Op>,
        Op
    >;
    using ScheduleReq = axp::level0::RequireWarpgroupCount<2, CapT::warpgroup_warps>;
    using ScheduleObligations = std::conditional_t<
        kProducerConsumer,
        iro::util::type_list<ScheduleReq>,
        iro::util::type_list<>
    >;
    static_assert(!kProducerConsumer,
                  "GemmTileWarpImpl: ProducerConsumer schedule requires WGMMA (SM90+) path");
    using SwizzleAtomA = std::conditional_t<
        std::is_same_v<LoadModeA, axp::intent::load_mode::AsyncPrefetch>,
        axp::kit::detail::select_swizzle_t<MemoryPatternA, ElemA, BlockK, CapT, true>,
        axp::swizzle::None>;
    using SwizzleAtomB = std::conditional_t<
        std::is_same_v<LoadModeB, axp::intent::load_mode::AsyncPrefetch>,
        axp::kit::detail::select_swizzle_t<MemoryPatternB, ElemB, BlockK, CapT, false>,
        axp::swizzle::None>;
    using StageSwizzleA = std::conditional_t<std::is_same_v<SwizzleAtomA, axp::swizzle::None>, void, SwizzleAtomA>;
    using StageSwizzleB = std::conditional_t<std::is_same_v<SwizzleAtomB, axp::swizzle::None>, void, SwizzleAtomB>;
    static constexpr bool kSwizzleA = !std::is_same_v<SwizzleAtomA, axp::swizzle::None>;
    static constexpr bool kSwizzleB = !std::is_same_v<SwizzleAtomB, axp::swizzle::None>;
    static constexpr int kASmemAlign = kSwizzleA ? (1 << (SwizzleAtomA::B_bits + SwizzleAtomA::S_bits)) : 16;
    static constexpr int kBSmemAlign = kSwizzleB ? (1 << (SwizzleAtomB::B_bits + SwizzleAtomB::S_bits)) : 16;

    using ATileG = iro::contract::Tile<
        iro::contract::Shape<BlockM, BlockK>,
        ElemA,
        iro::contract::layout::RowMajor<BlockK>,
        iro::contract::space::global,
        iro::contract::Align<16>
    >;

    using BTileG = iro::contract::Tile<
        iro::contract::Shape<BlockK, BlockN>,
        ElemB,
        iro::contract::layout::ColMajor<BlockK>,
        iro::contract::space::global,
        iro::contract::Align<16>
    >;

    using ATileS = iro::contract::Tile<
        iro::contract::Shape<BlockM, BlockK>,
        ElemA,
        std::conditional_t<
            kSwizzleA,
            iro::contract::layout::Swizzled<BlockK, SwizzleAtomA::B, SwizzleAtomA::S>,
            iro::contract::layout::RowMajor<BlockK>
        >,
        iro::contract::space::shared,
        iro::contract::Align<kASmemAlign>
    >;

    using BTileSLayout = std::conditional_t<
        kSwizzleB,
        iro::contract::layout::SwizzledColMajor<BlockK, SwizzleAtomB::B, SwizzleAtomB::S>,
        iro::contract::layout::ColMajor<BlockK>
    >;
    using BTileS = iro::contract::Tile<
        iro::contract::Shape<BlockK, BlockN>,
        ElemB,
        BTileSLayout,
        iro::contract::space::shared,
        iro::contract::Align<kBSmemAlign>
    >;

    using CTileS = iro::contract::Tile<
        iro::contract::Shape<BlockM, BlockN>,
        typename Recipe::out,
        iro::contract::layout::RowMajor<BlockN>,
        iro::contract::space::shared,
        iro::contract::Align<16>
    >;

    using TileInA = axp::level0::TileBoundaryIn<
        Recipe, ATileG, ASubj, iro::exec::block, iro::token::lifetime::block
    >;
    using TileInB = axp::level0::TileBoundaryIn<
        Recipe, BTileG, BSubj, iro::exec::block, iro::token::lifetime::block
    >;
    using TileOut = axp::level0::TileBoundaryOut<
        Recipe, CTileS, CSubj, iro::exec::block, iro::token::lifetime::block
    >;

    using MmaShape = axp::protocol::compute::MmaShape<BlockM, BlockN, BlockK, ElemA, ElemB, typename Recipe::acc,
                                                     typename ATileS::layout, typename BTileS::layout>;

    static_assert(axp::protocol::compute::detail::is_wmma_shape_v<BlockM, BlockN, BlockK, ElemA, ElemB, typename Recipe::acc>,
                  "GemmTile: WMMA shape only (16x16x16 f16/bf16 or 16x16x8 tf32)");

    using AccFrag = iro::contract::FragmentDesc<
        iro::contract::Shape<BlockM, BlockN>,
        typename Recipe::acc,
        iro::dist::accumulator,
        BlockN / 2
    >;
    static constexpr int kAccRegs = detail::frag_reg_count_v<AccFrag>;
    static constexpr int kBaseRegs = 24 + Stages * 4;
    using RegPressure = detail::reg_pressure_obligation<kBaseRegs, AccFrag>;

    using PipeA = iro::contract::res::smem_pipeline<PipeATag, Stages, ATileS::bytes, ATileS::align::bytes>;
    using PipeB = iro::contract::res::smem_pipeline<PipeBTag, Stages, BTileS::bytes, BTileS::align::bytes>;

    using AutoATma = axp::kit::detail::select_tma_t<LoadModeA, CapT, ATileG, ASubj, PipeATag>;
    using AutoBTma = axp::kit::detail::select_tma_t<LoadModeB, CapT, BTileG, BSubj, PipeBTag>;
    using ATmaT = std::conditional_t<!std::is_void_v<ATma>, ATma, AutoATma>;
    using BTmaT = std::conditional_t<!std::is_void_v<BTma>, BTma, AutoBTma>;

    template<class Tma, class Enable = void>
    struct barrier_init {
        using init_type = void;
        using obligations = iro::util::type_list<>;
    };

    template<class Tma>
    struct barrier_init<Tma, std::enable_if_t<axp::level2::staging::tma_traits<Tma>::valid>> {
        using BarrierSubj = typename axp::level2::staging::tma_traits<Tma>::BarrierSubj;
        using init_type = axp::level0::BarrierInit<Recipe, BarrierSubj, iro::exec::block, 1>;
        using obligations = iro::util::type_list<init_type>;
    };

    template<class Tma>
    struct barrier_init<Tma, std::enable_if_t<axp::level2::staging::tma_multicast_traits<Tma>::valid>> {
        using BarrierSubj = typename axp::level2::staging::tma_multicast_traits<Tma>::BarrierSubj;
        using init_type = axp::level0::ClusterBarrierInit<Recipe, BarrierSubj, iro::exec::cluster, 1>;
        using obligations = iro::util::type_list<init_type>;
    };

    using ABarrierInitHelper = barrier_init<ATmaT>;
    using BBarrierInitHelper = barrier_init<BTmaT>;
    using ABarrierInit = typename ABarrierInitHelper::init_type;
    using BBarrierInit = typename BBarrierInitHelper::init_type;
    static constexpr bool kHasTmaA = !std::is_void_v<ABarrierInit>;
    static constexpr bool kHasTmaB = !std::is_void_v<BBarrierInit>;
    static constexpr bool kStreamingA = axp::level2::staging::streaming_traits<ATmaT>::valid;
    static constexpr bool kStreamingB = axp::level2::staging::streaming_traits<BTmaT>::valid;
    using AStageIssueExec = axp::level2::staging::tma_issue_exec_group_t<ATmaT>;
    using BStageIssueExec = axp::level2::staging::tma_issue_exec_group_t<BTmaT>;

    static constexpr int kPrefetchTiles = kBulkSchedule ? 0 : (KTiles < Stages ? KTiles : Stages);
    static constexpr int kSteadyTiles = kBulkSchedule ? KTiles : (KTiles > Stages ? (KTiles - Stages) : 0);
    static constexpr int kDrainStart = kBulkSchedule ? KTiles : kSteadyTiles;

    template<int I>
    struct slot_traits {
        static constexpr int slot = I % Stages;
        using SlotA = iro::contract::res::slot_subject<PipeA, slot>;
        using SlotB = iro::contract::res::slot_subject<PipeB, slot>;
    };

    template<int I>
    struct Prefetch {
        using SlotA = typename slot_traits<I>::SlotA;
        using SlotB = typename slot_traits<I>::SlotB;

        using StageA = axp::level2::staging::StageGmemToSmem<
        Recipe, ATileG, ATileS, ASubj, PipeATag, SlotA,
        iro::exec::block, iro::token::lifetime::block, Stages,
        std::conditional_t<std::is_same_v<SwizzleAtomA, axp::swizzle::None>, void, SwizzleAtomA>,
        ATmaT,
        AStageIssueExec, CapT
    >;
    using StageB = axp::level2::staging::StageGmemToSmem<
        Recipe, BTileG, BTileS, BSubj, PipeBTag, SlotB,
        iro::exec::block, iro::token::lifetime::block, Stages,
        std::conditional_t<std::is_same_v<SwizzleAtomB, axp::swizzle::None>, void, SwizzleAtomB>,
        BTmaT,
        BStageIssueExec, CapT
    >;

        struct IssueAImpl : StageA::Issue {};
        struct IssueBImpl : StageB::Issue {};
        struct WaitAImpl : StageA::Wait {};
        struct WaitBImpl : StageB::Wait {};
        using IssueA = ProducerOp<IssueAImpl>;
        using IssueB = ProducerOp<IssueBImpl>;
        using WaitA = ConsumerOp<WaitAImpl>;
        using WaitB = ConsumerOp<WaitBImpl>;

        using obligations = iro::util::type_list<IssueA, IssueB, WaitA, WaitB>;
        using edges = iro::util::concat_t<
            detail::stage_issue_wait_edges_t<IssueA, WaitA, kHasTmaA, kStreamingA>,
            detail::stage_issue_wait_edges_t<IssueB, WaitB, kHasTmaB, kStreamingB>
        >;
    };

    template<int I, bool HasPrev = (I > 0)>
    struct ComputeBase;

    template<int I>
    struct ComputeBase<I, false> {
        using SlotA = typename slot_traits<I>::SlotA;
        using SlotB = typename slot_traits<I>::SlotB;
        using StageA = axp::level2::staging::StageGmemToSmem<
            Recipe, ATileG, ATileS, ASubj, PipeATag, SlotA,
            iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleA, ATmaT,
            AStageIssueExec, CapT
        >;
        using StageB = axp::level2::staging::StageGmemToSmem<
            Recipe, BTileG, BTileS, BSubj, PipeBTag, SlotB,
            iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleB, BTmaT,
            BStageIssueExec, CapT
        >;

        struct MmaImpl : axp::level2::Matmul<
            Recipe, MmaShape, ATileS, BTileS, AccFrag,
            SlotA, SlotB, detail::acc_raw_subj<I>, ExecGroup, detail::wgmma_tag, CapT
        > {};
        using Mma = ConsumerOp<MmaImpl>;

        using ScaleAHelper = detail::scale_op<kHasScaleA, Recipe, ATileS, ScaleTileA, SlotA, ScaleASubj, CapT>;
        using ScaleBHelper = detail::scale_op<kHasScaleB, Recipe, BTileS, ScaleTileB, SlotB, ScaleBSubj, CapT>;
        using ScaleA = ConsumerMaybe<typename ScaleAHelper::type>;
        using ScaleB = ConsumerMaybe<typename ScaleBHelper::type>;
        using ScaleAObligations = std::conditional_t<kHasScaleA, iro::util::type_list<ScaleA>, iro::util::type_list<>>;
        using ScaleBObligations = std::conditional_t<kHasScaleB, iro::util::type_list<ScaleB>, iro::util::type_list<>>;

        struct MarkAImpl : StageA::Mark {};
        struct MarkBImpl : StageB::Mark {};
        struct ReleaseAImpl : StageA::Release {};
        struct ReleaseBImpl : StageB::Release {};
        using MarkA = ConsumerOp<MarkAImpl>;
        using MarkB = ConsumerOp<MarkBImpl>;
        using ReleaseA = ConsumerOp<ReleaseAImpl>;
        using ReleaseB = ConsumerOp<ReleaseBImpl>;
        struct HoldA : axp::level0::SlotAfter<
            Recipe, SlotA, iro::exec::block, iro::token::lifetime::block, ATileS::bytes,
            AccFrag, detail::acc_raw_subj<I>, ExecGroup, typename AccFrag::dist
        > {};
        struct HoldB : axp::level0::SlotAfter<
            Recipe, SlotB, iro::exec::block, iro::token::lifetime::block, BTileS::bytes,
            AccFrag, detail::acc_raw_subj<I>, ExecGroup, typename AccFrag::dist
        > {};

        using base_obligations = iro::util::type_list<Mma, HoldA, HoldB, MarkA, MarkB, ReleaseA, ReleaseB>;
        using obligations = iro::util::concat_t<
            iro::util::concat_t<ScaleAObligations, ScaleBObligations>,
            base_obligations
        >;
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<Mma, 0>, detail::in_port_t<HoldA, 1>>,
            iro::compose::Edge<detail::out_port_t<Mma, 0>, detail::in_port_t<HoldB, 1>>,
            iro::compose::Edge<detail::out_port_t<HoldA, 0>, detail::in_port_t<MarkA, 0>>,
            iro::compose::Edge<detail::out_port_t<HoldB, 0>, detail::in_port_t<MarkB, 0>>,
            iro::compose::Edge<detail::out_port_t<MarkA, 0>, detail::in_port_t<ReleaseA, 0>>,
            iro::compose::Edge<detail::out_port_t<MarkB, 0>, detail::in_port_t<ReleaseB, 0>>
        >;
        using accum_subj = detail::acc_raw_subj<I>;
        using accum_obligation = Mma;
    };

    template<int I>
    struct ComputeBase<I, true> {
        using SlotA = typename slot_traits<I>::SlotA;
        using SlotB = typename slot_traits<I>::SlotB;
        using StageA = axp::level2::staging::StageGmemToSmem<
            Recipe, ATileG, ATileS, ASubj, PipeATag, SlotA,
            iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleA, ATmaT,
            AStageIssueExec, CapT
        >;
        using StageB = axp::level2::staging::StageGmemToSmem<
            Recipe, BTileG, BTileS, BSubj, PipeBTag, SlotB,
            iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleB, BTmaT,
            BStageIssueExec, CapT
        >;

        struct MmaImpl : axp::level2::Matmul<
            Recipe, MmaShape, ATileS, BTileS, AccFrag,
            SlotA, SlotB, detail::acc_raw_subj<I>, ExecGroup, detail::wgmma_tag, CapT
        > {};
        using Mma = ConsumerOp<MmaImpl>;

        using ScaleAHelper = detail::scale_op<kHasScaleA, Recipe, ATileS, ScaleTileA, SlotA, ScaleASubj, CapT>;
        using ScaleBHelper = detail::scale_op<kHasScaleB, Recipe, BTileS, ScaleTileB, SlotB, ScaleBSubj, CapT>;
        using ScaleA = ConsumerMaybe<typename ScaleAHelper::type>;
        using ScaleB = ConsumerMaybe<typename ScaleBHelper::type>;
        using ScaleAObligations = std::conditional_t<kHasScaleA, iro::util::type_list<ScaleA>, iro::util::type_list<>>;
        using ScaleBObligations = std::conditional_t<kHasScaleB, iro::util::type_list<ScaleB>, iro::util::type_list<>>;

        using PrevAccum = std::conditional_t<
            (I == 1),
            detail::acc_raw_subj<0>,
            detail::acc_accum_subj<I - 1>
        >;

        struct AddImpl : axp::level0::Add<
            AccRecipe, AccFrag, PrevAccum, detail::acc_raw_subj<I>, detail::acc_accum_subj<I>, ExecGroup
        > {};
        using Add = ConsumerOp<AddImpl>;

        struct MarkAImpl : StageA::Mark {};
        struct MarkBImpl : StageB::Mark {};
        struct ReleaseAImpl : StageA::Release {};
        struct ReleaseBImpl : StageB::Release {};
        using MarkA = ConsumerOp<MarkAImpl>;
        using MarkB = ConsumerOp<MarkBImpl>;
        using ReleaseA = ConsumerOp<ReleaseAImpl>;
        using ReleaseB = ConsumerOp<ReleaseBImpl>;
        struct HoldA : axp::level0::SlotAfter<
            Recipe, SlotA, iro::exec::block, iro::token::lifetime::block, ATileS::bytes,
            AccFrag, detail::acc_raw_subj<I>, ExecGroup, typename AccFrag::dist
        > {};
        struct HoldB : axp::level0::SlotAfter<
            Recipe, SlotB, iro::exec::block, iro::token::lifetime::block, BTileS::bytes,
            AccFrag, detail::acc_raw_subj<I>, ExecGroup, typename AccFrag::dist
        > {};

        using base_obligations = iro::util::type_list<Mma, Add, HoldA, HoldB, MarkA, MarkB, ReleaseA, ReleaseB>;
        using obligations = iro::util::concat_t<
            iro::util::concat_t<ScaleAObligations, ScaleBObligations>,
            base_obligations
        >;
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<Mma, 0>, detail::in_port_t<Add, 1>>,
            iro::compose::Edge<detail::out_port_t<Mma, 0>, detail::in_port_t<HoldA, 1>>,
            iro::compose::Edge<detail::out_port_t<Mma, 0>, detail::in_port_t<HoldB, 1>>,
            iro::compose::Edge<detail::out_port_t<HoldA, 0>, detail::in_port_t<MarkA, 0>>,
            iro::compose::Edge<detail::out_port_t<HoldB, 0>, detail::in_port_t<MarkB, 0>>,
            iro::compose::Edge<detail::out_port_t<MarkA, 0>, detail::in_port_t<ReleaseA, 0>>,
            iro::compose::Edge<detail::out_port_t<MarkB, 0>, detail::in_port_t<ReleaseB, 0>>
        >;
        using accum_subj = detail::acc_accum_subj<I>;
        using accum_obligation = Add;
    };

    template<int I>
    struct ComputePrefetch : ComputeBase<I> {
        using PrefetchNext = Prefetch<I + Stages>;
        using obligations = iro::util::concat_t<typename ComputeBase<I>::obligations, typename PrefetchNext::obligations>;
        using edges = iro::util::concat_t<typename ComputeBase<I>::edges, typename PrefetchNext::edges>;
    };

    template<int I>
    struct ComputeDrain : ComputeBase<I> {
        using obligations = typename ComputeBase<I>::obligations;
        using edges = typename ComputeBase<I>::edges;
    };

    template<int I>
    struct ComputeBulk {
        using PrefetchCurr = Prefetch<I>;
        using ComputeCurr = ComputeBase<I>;
        using obligations = iro::util::concat_t<typename PrefetchCurr::obligations, typename ComputeCurr::obligations>;
        using edges = iro::util::concat_t<typename PrefetchCurr::edges, typename ComputeCurr::edges>;
    };

    template<int I, int End>
    struct build_prefetch {
        using curr = Prefetch<I>;
        using next = build_prefetch<I + 1, End>;
        using obligations = iro::util::concat_t<typename curr::obligations, typename next::obligations>;
        using edges = iro::util::concat_t<typename curr::edges, typename next::edges>;
    };

    template<int End>
    struct build_prefetch<End, End> {
        using obligations = iro::util::type_list<>;
        using edges = iro::util::type_list<>;
    };


    template<int I, int End>
    struct build_steady {
        using curr = std::conditional_t<kBulkSchedule, ComputeBulk<I>, ComputePrefetch<I>>;
        using next = build_steady<I + 1, End>;
        using obligations = iro::util::concat_t<typename curr::obligations, typename next::obligations>;
        using edges = iro::util::concat_t<typename curr::edges, typename next::edges>;
    };

    template<int End>
    struct build_steady<End, End> {
        using obligations = iro::util::type_list<>;
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_drain {
        using curr = ComputeDrain<I>;
        using next = build_drain<I + 1, End>;
        using obligations = iro::util::concat_t<typename curr::obligations, typename next::obligations>;
        using edges = iro::util::concat_t<typename curr::edges, typename next::edges>;
    };

    template<int End>
    struct build_drain<End, End> {
        using obligations = iro::util::type_list<>;
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_wait_edges {
        using wait_a_edges = detail::wait_scale_edges_t<
            typename Prefetch<I>::WaitA,
            typename ComputeBase<I>::ScaleA,
            typename ComputeBase<I>::Mma,
            kHasScaleA,
            0
        >;
        using wait_b_edges = detail::wait_scale_edges_t<
            typename Prefetch<I>::WaitB,
            typename ComputeBase<I>::ScaleB,
            typename ComputeBase<I>::Mma,
            kHasScaleB,
            1
        >;
        using curr = iro::util::concat_t<
            iro::util::concat_t<wait_a_edges, wait_b_edges>,
            iro::util::type_list<
                iro::compose::Edge<detail::out_port_t<typename Prefetch<I>::WaitA, 1>,
                                   detail::in_port_t<typename ComputeBase<I>::HoldA, 0>>,
                iro::compose::Edge<detail::out_port_t<typename Prefetch<I>::WaitB, 1>,
                                   detail::in_port_t<typename ComputeBase<I>::HoldB, 0>>
            >
        >;
        using next = build_wait_edges<I + 1, End>;
        using edges = iro::util::concat_t<curr, typename next::edges>;
    };

    template<int End>
    struct build_wait_edges<End, End> {
        using edges = iro::util::type_list<>;
    };

    template<int I, bool Enable>
    struct reuse_edge_a {
        using edges = iro::util::type_list<>;
    };

    template<int I>
    struct reuse_edge_a<I, true> {
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<typename ComputeBase<I>::ReleaseA, 0>,
                               detail::in_port_t<typename Prefetch<I + Stages>::IssueA, 1>>
        >;
    };

    template<int I, bool Enable>
    struct reuse_edge_b {
        using edges = iro::util::type_list<>;
    };

    template<int I>
    struct reuse_edge_b<I, true> {
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<typename ComputeBase<I>::ReleaseB, 0>,
                               detail::in_port_t<typename Prefetch<I + Stages>::IssueB, 1>>
        >;
    };

    template<int I, int End>
    struct build_reuse_edges {
        static constexpr bool has_next = (I + Stages < End);
        using curr = iro::util::concat_t<
            typename reuse_edge_a<I, has_next && !kHasTmaA>::edges,
            typename reuse_edge_b<I, has_next && !kHasTmaB>::edges
        >;
        using next = build_reuse_edges<I + 1, End>;
        using edges = iro::util::concat_t<curr, typename next::edges>;
    };

    template<int End>
    struct build_reuse_edges<End, End> {
        using edges = iro::util::type_list<>;
    };

    template<int I, int End, bool Enable, class = void>
    struct build_barrier_edges_a {
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_barrier_edges_a<I, End, true, std::enable_if_t<(I < End)>> {
        using curr = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<ABarrierInit, 0>,
                               detail::in_port_t<typename Prefetch<I>::IssueA, 1>>
        >;
        using next = build_barrier_edges_a<I + 1, End, true>;
        using edges = iro::util::concat_t<curr, typename next::edges>;
    };

    template<int End, bool Enable>
    struct build_barrier_edges_a<End, End, Enable, void> {
        using edges = iro::util::type_list<>;
    };

    template<int I, int End, bool Enable, class = void>
    struct build_barrier_edges_b {
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_barrier_edges_b<I, End, true, std::enable_if_t<(I < End)>> {
        using curr = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<BBarrierInit, 0>,
                               detail::in_port_t<typename Prefetch<I>::IssueB, 1>>
        >;
        using next = build_barrier_edges_b<I + 1, End, true>;
        using edges = iro::util::concat_t<curr, typename next::edges>;
    };

    template<int End, bool Enable>
    struct build_barrier_edges_b<End, End, Enable, void> {
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_accum_edges {
        using curr = iro::util::type_list<
            iro::compose::Edge<
                detail::out_port_t<typename ComputeBase<I - 1>::accum_obligation, 0>,
                detail::in_port_t<typename ComputeBase<I>::Add, 0>
            >
        >;
        using next = build_accum_edges<I + 1, End>;
        using edges = iro::util::concat_t<curr, typename next::edges>;
    };

    template<int End>
    struct build_accum_edges<0, End> {
        using next = build_accum_edges<1, End>;
        using edges = typename next::edges;
    };

    template<int End>
    struct build_accum_edges<End, End> {
        using edges = iro::util::type_list<>;
    };

    using Prologue = axp::level3::detail::make_composition_t<
        typename build_prefetch<0, kPrefetchTiles>::obligations,
        typename build_prefetch<0, kPrefetchTiles>::edges
    >;

    using Steady = axp::level3::detail::make_composition_t<
        typename build_steady<0, kSteadyTiles>::obligations,
        typename build_steady<0, kSteadyTiles>::edges
    >;

    using Drain = axp::level3::detail::make_composition_t<
        typename build_drain<kDrainStart, KTiles>::obligations,
        typename build_drain<kDrainStart, KTiles>::edges
    >;
    using FinalAccum = typename ComputeBase<KTiles - 1>::accum_subj;
    using FinalAccumObligation = typename ComputeBase<KTiles - 1>::accum_obligation;

    using EpilogueBuild = detail::build_vec_epilogue<
        Recipe, AccFrag, ExecGroup, EpiloguePolicy, FinalAccum, FinalAccumObligation, CapT
    >;
    using EpilogueAccum = typename EpilogueBuild::out_frag_subj;
    using EpilogueProducer = typename EpilogueBuild::out_frag_producer;

    using Store = axp::level0::FragmentToSharedTile<
        Recipe, AccFrag, CTileS, EpilogueAccum, CSubj, iro::exec::warp, iro::token::lifetime::warp
    >;

    using Fence = axp::level0::TileFence<
        Recipe, CTileS, CSubj, iro::exec::block
    >;

    using phase_obligations = iro::util::concat_t<
        iro::util::concat_t<typename Prologue::obligations, typename Steady::obligations>,
        typename Drain::obligations
    >;

    using epilogue_obligations = typename EpilogueBuild::obligations;

    using barrier_obligations = iro::util::concat_t<
        typename ABarrierInitHelper::obligations,
        typename BBarrierInitHelper::obligations
    >;

    using phase_edges = iro::util::concat_t<
        iro::util::concat_t<typename Prologue::edges, typename Steady::edges>,
        typename Drain::edges
    >;

    using epilogue_edges = typename EpilogueBuild::edges;

    using barrier_edges = iro::util::concat_t<
        typename build_barrier_edges_a<0, KTiles, kHasTmaA>::edges,
        typename build_barrier_edges_b<0, KTiles, kHasTmaB>::edges
    >;

    using global_edges = iro::util::concat_t<
        iro::util::concat_t<
            typename build_wait_edges<0, KTiles>::edges,
            typename build_reuse_edges<0, KTiles>::edges
        >,
        typename build_accum_edges<0, KTiles>::edges
    >;

    template<bool Enable, class = void>
    struct boundary_tilein_a {
        using edges = iro::util::type_list<>;
        using obligations = iro::util::type_list<>;
    };

    template<bool Enable>
    struct boundary_tilein_a<Enable, std::enable_if_t<Enable>> {
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<TileInA, 0>, detail::in_port_t<typename Prefetch<0>::IssueA, 0>>
        >;
        using obligations = iro::util::type_list<TileInA>;
    };

    template<bool Enable, class = void>
    struct boundary_tilein_b {
        using edges = iro::util::type_list<>;
        using obligations = iro::util::type_list<>;
    };

    template<bool Enable>
    struct boundary_tilein_b<Enable, std::enable_if_t<Enable>> {
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<TileInB, 0>, detail::in_port_t<typename Prefetch<0>::IssueB, 0>>
        >;
        using obligations = iro::util::type_list<TileInB>;
    };

    using boundary_edges = iro::util::concat_t<
        iro::util::concat_t<
            typename boundary_tilein_a<!kHasTmaA>::edges,
            typename boundary_tilein_b<!kHasTmaB>::edges
        >,
        iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<Fence, 0>, detail::in_port_t<TileOut, 0>>
        >
    >;

    using obligations = iro::util::concat_t<
        iro::util::concat_t<phase_obligations, epilogue_obligations>,
        iro::util::concat_t<barrier_obligations,
                            iro::util::type_list<RegPressure, TileInA, TileInB, Store, Fence, TileOut>>
    >;

    using edges = iro::util::concat_t<
        iro::util::concat_t<iro::util::concat_t<iro::util::concat_t<iro::util::concat_t<phase_edges, epilogue_edges>, global_edges>, barrier_edges>, boundary_edges>,
        iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<EpilogueProducer, 0>, detail::in_port_t<Store, 0>>,
            iro::compose::Edge<detail::out_port_t<Store, 0>, detail::in_port_t<Fence, 0>>
        >
    >;

    using type = axp::level3::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

// ----------------------------------------------------------------------------
// Warpgroup WGMMA tile (explicit prologue/steady/drain pipeline)
// ----------------------------------------------------------------------------

template<
    class Recipe,
    int BlockM, int BlockN, int BlockK,
    int Stages, int KTiles,
    class ASubj, class BSubj, class CSubj,
    class ScaleASubj, class ScaleBSubj,
    class WgmmaSubj,
    class MemoryPatternA, class MemoryPatternB,
    class LoadModeA, class LoadModeB,
    class Schedule,
    class PipeATag, class PipeBTag,
    class ATma = void, class BTma = void,
    class CapT = axp::target_cap,
    class EpiloguePolicy = epilogue::None
>
struct GemmTileWgmmaImpl {
    static_assert(std::is_same_v<typename Recipe::acc, iro::elem::f32>, "GemmTileWgmma: Recipe::acc must be f32");
    static_assert(Stages >= 2 && Stages <= 4, "GemmTileWgmma: Stages must be 2-4");
    static_assert(KTiles > 0, "GemmTileWgmma: KTiles must be > 0");
    static_assert(BlockM == 64, "GemmTileWgmma: WGMMA requires BlockM == 64");

    using ExecGroup = iro::exec::warpgroup_t<CapT::warpgroup_warps>;
    using AccRecipe = iro::recipe::Accumulate<Recipe>;
    using ElemA = iro::verify::recipe_in_a_t<Recipe>;
    using ElemB = iro::verify::recipe_in_b_t<Recipe>;
    using ScaleElemA = iro::verify::recipe_scale_a_t<Recipe>;
    using ScaleElemB = iro::verify::recipe_scale_b_t<Recipe>;
    static constexpr int kScaleVecA = iro::verify::recipe_scale_vec_a_v<Recipe>;
    static constexpr int kScaleVecB = iro::verify::recipe_scale_vec_b_v<Recipe>;
    static constexpr bool kHasScaleA = iro::verify::recipe_has_scale_a_v<Recipe>;
    static constexpr bool kHasScaleB = iro::recipe::is_precision_ab_v<Recipe> &&
                                       iro::verify::recipe_has_scale_b_v<Recipe>;
    using ScaleTileA = typename detail::scale_tile<kHasScaleA, ScaleElemA, kScaleVecA>::type;
    using ScaleTileB = typename detail::scale_tile<kHasScaleB, ScaleElemB, kScaleVecB>::type;
    using ScheduleT = axp::kit::detail::select_schedule_t<Schedule, CapT>;
    static constexpr bool kBulkSchedule = std::is_same_v<ScheduleT, axp::intent::schedule::BulkSynchronous>;
    static constexpr bool kProducerConsumer = std::is_same_v<ScheduleT, axp::intent::schedule::ProducerConsumer>;
    template<class Op>
    using ProducerOp = std::conditional_t<
        kProducerConsumer,
        axp::level0::SpecializedOp<axp::level0::role::producer, Op>,
        Op
    >;
    template<class Op>
    using ConsumerOp = std::conditional_t<
        kProducerConsumer,
        axp::level0::SpecializedOp<axp::level0::role::consumer, Op>,
        Op
    >;
    template<class Op>
    using ConsumerMaybe = std::conditional_t<
        kProducerConsumer && !std::is_void_v<Op>,
        ConsumerOp<Op>,
        Op
    >;
    using ScheduleReq = axp::level0::RequireWarpgroupCount<2, CapT::warpgroup_warps>;
    using ScheduleObligations = std::conditional_t<
        kProducerConsumer,
        iro::util::type_list<ScheduleReq>,
        iro::util::type_list<>
    >;
    using SwizzleAtomA = std::conditional_t<
        std::is_same_v<LoadModeA, axp::intent::load_mode::AsyncPrefetch>,
        axp::kit::detail::select_swizzle_t<MemoryPatternA, ElemA, BlockK, CapT, true>,
        axp::swizzle::None>;
    using SwizzleAtomB = std::conditional_t<
        std::is_same_v<LoadModeB, axp::intent::load_mode::AsyncPrefetch>,
        axp::kit::detail::select_swizzle_t<MemoryPatternB, ElemB, BlockK, CapT, false>,
        axp::swizzle::None>;
    using StageSwizzleA = std::conditional_t<std::is_same_v<SwizzleAtomA, axp::swizzle::None>, void, SwizzleAtomA>;
    using StageSwizzleB = std::conditional_t<std::is_same_v<SwizzleAtomB, axp::swizzle::None>, void, SwizzleAtomB>;
    using WgmmaSwizzleA = SwizzleAtomA;
    using WgmmaSwizzleB = SwizzleAtomB;
    static constexpr bool kSwizzleA = !std::is_same_v<SwizzleAtomA, axp::swizzle::None>;
    static constexpr bool kSwizzleB = !std::is_same_v<SwizzleAtomB, axp::swizzle::None>;
    static constexpr int kASmemAlign = kSwizzleA ? (1 << (SwizzleAtomA::B_bits + SwizzleAtomA::S_bits)) : 16;
    static constexpr int kBSmemAlign = kSwizzleB ? (1 << (SwizzleAtomB::B_bits + SwizzleAtomB::S_bits)) : 16;
    using ATileSLayout = std::conditional_t<
        kSwizzleA,
        iro::contract::layout::Swizzled<BlockK, SwizzleAtomA::B, SwizzleAtomA::S>,
        iro::contract::layout::RowMajor<BlockK>
    >;
    using FenceHandleImpl = axp::level2::wgmma::Fence<
        Recipe, WgmmaSubj, ExecGroup, iro::util::type_list<>, iro::util::type_list<>, CapT
    >;
    using FenceHandle = ConsumerOp<FenceHandleImpl>;

    using ATileG = iro::contract::Tile<
        iro::contract::Shape<BlockM, BlockK>,
        ElemA,
        iro::contract::layout::RowMajor<BlockK>,
        iro::contract::space::global,
        iro::contract::Align<16>
    >;

    using BTileG = iro::contract::Tile<
        iro::contract::Shape<BlockK, BlockN>,
        ElemB,
        iro::contract::layout::ColMajor<BlockK>,
        iro::contract::space::global,
        iro::contract::Align<16>
    >;

    using AutoATma = axp::kit::detail::select_tma_t<LoadModeA, CapT, ATileG, ASubj, PipeATag>;
    using AutoBTma = axp::kit::detail::select_tma_t<LoadModeB, CapT, BTileG, BSubj, PipeBTag>;
    using ATmaT = std::conditional_t<!std::is_void_v<ATma>, ATma, AutoATma>;
    using BTmaT = std::conditional_t<!std::is_void_v<BTma>, BTma, AutoBTma>;
    static constexpr bool kHasTmaAConfig =
        axp::level2::staging::tma_traits<ATmaT>::valid || axp::level2::staging::tma_multicast_traits<ATmaT>::valid;
    static constexpr bool kHasTmaBConfig =
        axp::level2::staging::tma_traits<BTmaT>::valid || axp::level2::staging::tma_multicast_traits<BTmaT>::valid;
    static_assert(!kSwizzleA || kHasTmaAConfig,
                  "GemmTileWgmma: SwizzleAtom requires TMA staging for A operand");
    static_assert(!kSwizzleB || kHasTmaBConfig,
                  "GemmTileWgmma: SwizzleAtom requires TMA staging for B operand");

    using ATileS = iro::contract::Tile<
        iro::contract::Shape<BlockM, BlockK>,
        ElemA,
        ATileSLayout,
        iro::contract::space::shared,
        iro::contract::Align<kASmemAlign>
    >;

    using BTileSLayout = std::conditional_t<
        kSwizzleB,
        iro::contract::layout::SwizzledColMajor<BlockK, SwizzleAtomB::B, SwizzleAtomB::S>,
        iro::contract::layout::ColMajor<BlockK>
    >;
    using BTileS = iro::contract::Tile<
        iro::contract::Shape<BlockK, BlockN>,
        ElemB,
        BTileSLayout,
        iro::contract::space::shared,
        iro::contract::Align<kBSmemAlign>
    >;

    using CTileS = iro::contract::Tile<
        iro::contract::Shape<BlockM, BlockN>,
        typename Recipe::out,
        iro::contract::layout::RowMajor<BlockN>,
        iro::contract::space::shared,
        iro::contract::Align<16>
    >;

    using TileInA = axp::level0::TileBoundaryIn<
        Recipe, ATileG, ASubj, iro::exec::block, iro::token::lifetime::block
    >;
    using TileInB = axp::level0::TileBoundaryIn<
        Recipe, BTileG, BSubj, iro::exec::block, iro::token::lifetime::block
    >;
    using TileOut = axp::level0::TileBoundaryOut<
        Recipe, CTileS, CSubj, iro::exec::block, iro::token::lifetime::block
    >;

    using MmaShape = axp::protocol::compute::MmaShape<BlockM, BlockN, BlockK, ElemA, ElemB, typename Recipe::acc,
                                                     typename ATileS::layout, typename BTileS::layout>;

    using AccFrag = iro::contract::FragmentDesc<
        iro::contract::Shape<BlockM, BlockN>,
        typename Recipe::acc,
        iro::dist::accumulator,
        BlockN / 2
    >;
    static constexpr int kAccRegs = detail::frag_reg_count_v<AccFrag>;
    static constexpr int kBaseRegs = 48 + Stages * 8;
    using RegPressure = detail::reg_pressure_obligation<kBaseRegs, AccFrag>;

    using PipeA = iro::contract::res::smem_pipeline<PipeATag, Stages, ATileS::bytes, ATileS::align::bytes>;
    using PipeB = iro::contract::res::smem_pipeline<PipeBTag, Stages, BTileS::bytes, BTileS::align::bytes>;

    template<class Tma, class Enable = void>
    struct barrier_init {
        using init_type = void;
        using obligations = iro::util::type_list<>;
    };

    template<class Tma>
    struct barrier_init<Tma, std::enable_if_t<axp::level2::staging::tma_traits<Tma>::valid>> {
        using BarrierSubj = typename axp::level2::staging::tma_traits<Tma>::BarrierSubj;
        using init_type = axp::level0::BarrierInit<Recipe, BarrierSubj, iro::exec::block, 1>;
        using obligations = iro::util::type_list<init_type>;
    };

    template<class Tma>
    struct barrier_init<Tma, std::enable_if_t<axp::level2::staging::tma_multicast_traits<Tma>::valid>> {
        using BarrierSubj = typename axp::level2::staging::tma_multicast_traits<Tma>::BarrierSubj;
        using init_type = axp::level0::ClusterBarrierInit<Recipe, BarrierSubj, iro::exec::cluster, 1>;
        using obligations = iro::util::type_list<init_type>;
    };

    using ABarrierInitHelper = barrier_init<ATmaT>;
    using BBarrierInitHelper = barrier_init<BTmaT>;
    using ABarrierInit = typename ABarrierInitHelper::init_type;
    using BBarrierInit = typename BBarrierInitHelper::init_type;
    static constexpr bool kHasTmaA = !std::is_void_v<ABarrierInit>;
    static constexpr bool kHasTmaB = !std::is_void_v<BBarrierInit>;
    static constexpr bool kStreamingA = axp::level2::staging::streaming_traits<ATmaT>::valid;
    static constexpr bool kStreamingB = axp::level2::staging::streaming_traits<BTmaT>::valid;
    using AStageIssueExec = axp::level2::staging::tma_issue_exec_group_t<ATmaT>;
    using BStageIssueExec = axp::level2::staging::tma_issue_exec_group_t<BTmaT>;

    static constexpr int kPrefetchTiles = kBulkSchedule ? 0 : (KTiles < Stages ? KTiles : Stages);
    static constexpr int kDescTiles = kBulkSchedule ? 0 : kPrefetchTiles;
    static constexpr int kSteadyTiles = kBulkSchedule ? KTiles : (KTiles > Stages ? (KTiles - Stages) : 0);
    static constexpr int kDrainStart = kBulkSchedule ? KTiles : kSteadyTiles;
    static constexpr int kWgmmaWaitDepth = (KTiles > 1)
        ? ((Stages - 1) < (KTiles - 1) ? (Stages - 1) : (KTiles - 1))
        : 0;
    static_assert(kWgmmaWaitDepth >= 0 && kWgmmaWaitDepth <= 7, "GemmTileWgmma: wait depth out of range");

    template<int I>
    struct wait_depth {
        static constexpr int remaining = KTiles - 1 - I;
        static constexpr int value = (remaining < kWgmmaWaitDepth) ? remaining : kWgmmaWaitDepth;
    };

    template<int Max, int I = 0, class Enable = void>
    struct wgmma_commit_extra;

    template<int Max, int I>
    struct wgmma_commit_extra<Max, I, std::enable_if_t<(I < Max)>> {
        using type = iro::util::concat_t<
            iro::util::type_list<axp::protocol::compute::wgmma_committed<WgmmaSubj, I>>,
            typename wgmma_commit_extra<Max, I + 1>::type
        >;
    };

    template<int Max, int I>
    struct wgmma_commit_extra<Max, I, std::enable_if_t<(I >= Max)>> {
        using type = iro::util::type_list<>;
    };

    template<int I>
    struct slot_traits {
        static constexpr int slot = I % Stages;
        using SlotA = iro::contract::res::slot_subject<PipeA, slot>;
        using SlotB = iro::contract::res::slot_subject<PipeB, slot>;
    };

    template<int I>
    struct Prefetch {
        using SlotA = typename slot_traits<I>::SlotA;
        using SlotB = typename slot_traits<I>::SlotB;

        using StageA = axp::level2::staging::StageGmemToSmem<
            Recipe, ATileG, ATileS, ASubj, PipeATag, SlotA,
            iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleA, ATmaT,
            AStageIssueExec, CapT
        >;
        using StageB = axp::level2::staging::StageGmemToSmem<
            Recipe, BTileG, BTileS, BSubj, PipeBTag, SlotB,
            iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleB, BTmaT,
            BStageIssueExec, CapT
        >;

        struct IssueA : StageA::Issue {};
        struct IssueB : StageB::Issue {};
        struct WaitA : StageA::Wait {};
        struct WaitB : StageB::Wait {};

        using obligations = iro::util::type_list<IssueA, IssueB, WaitA, WaitB>;
        using edges = iro::util::concat_t<
            detail::stage_issue_wait_edges_t<IssueA, WaitA, kHasTmaA, kStreamingA>,
            detail::stage_issue_wait_edges_t<IssueB, WaitB, kHasTmaB, kStreamingB>
        >;
    };

    template<int I>
    struct DescCache {
        using SlotA = typename slot_traits<I>::SlotA;
        using SlotB = typename slot_traits<I>::SlotB;

        struct MakeDescAImpl : axp::level0::MakeDesc<
            Recipe, ATileS, SlotA, detail::desc_a_subj<I>, ExecGroup, iro::token::lifetime::block, WgmmaSwizzleA
        > {};
        struct MakeDescBImpl : axp::level0::MakeDesc<
            Recipe, BTileS, SlotB, detail::desc_b_subj<I>, ExecGroup, iro::token::lifetime::block, WgmmaSwizzleB
        > {};
        using MakeDescA = ConsumerOp<MakeDescAImpl>;
        using MakeDescB = ConsumerOp<MakeDescBImpl>;

        using obligations = iro::util::type_list<MakeDescA, MakeDescB>;
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<typename Prefetch<I>::WaitA, 0>, detail::in_port_t<MakeDescA, 0>>,
            iro::compose::Edge<detail::out_port_t<typename Prefetch<I>::WaitB, 0>, detail::in_port_t<MakeDescB, 0>>
        >;
    };

    struct EmptyPhase {
        using obligations = iro::util::type_list<>;
        using edges = iro::util::type_list<>;
    };

    template<int I>
    struct Issue {
        using SlotA = typename slot_traits<I>::SlotA;
        using SlotB = typename slot_traits<I>::SlotB;
        static constexpr int slot = slot_traits<I>::slot;
        using DescSubjA = detail::desc_a_subj<slot>;
        using DescSubjB = detail::desc_b_subj<slot>;
        using ADesc = axp::level0::WgmmaSmemDesc<ATileS, SlotA, WgmmaSwizzleA>;
        using BDesc = axp::level0::WgmmaSmemDesc<BTileS, SlotB, WgmmaSwizzleB>;

        using ScaleAHelper = detail::scale_op<kHasScaleA, Recipe, ATileS, ScaleTileA, SlotA, ScaleASubj, CapT>;
        using ScaleBHelper = detail::scale_op<kHasScaleB, Recipe, BTileS, ScaleTileB, SlotB, ScaleBSubj, CapT>;
        using ScaleA = typename ScaleAHelper::type;
        using ScaleB = typename ScaleBHelper::type;

        struct DescAImpl : axp::level0::UseWgmmaSmemDesc<
            Recipe, ATileS, SlotA, DescSubjA, ExecGroup, iro::token::lifetime::block, WgmmaSwizzleA
        > {};
        struct DescBImpl : axp::level0::UseWgmmaSmemDesc<
            Recipe, BTileS, SlotB, DescSubjB, ExecGroup, iro::token::lifetime::block, WgmmaSwizzleB
        > {};
        using DescA = ConsumerOp<DescAImpl>;
        using DescB = ConsumerOp<DescBImpl>;

        struct MmaImpl : axp::level2::Matmul<
            Recipe, MmaShape, ADesc, BDesc, AccFrag,
            DescSubjA, DescSubjB, detail::acc_raw_subj<I>, ExecGroup, WgmmaSubj, CapT
        > {};
        using Mma = ConsumerOp<MmaImpl>;

        using CommitExtra = typename wgmma_commit_extra<kWgmmaWaitDepth>::type;
        struct CommitImpl : axp::level2::wgmma::CommitGroup<
            AccRecipe, WgmmaSubj, ExecGroup, kWgmmaWaitDepth,
            iro::util::type_list<>, CommitExtra, CapT
        > {};
        using Commit = ConsumerOp<CommitImpl>;

        using base_obligations = iro::util::type_list<DescA, DescB, Mma, Commit>;
        using obligations = iro::util::concat_t<
            iro::util::concat_t<typename ScaleAHelper::obligations, typename ScaleBHelper::obligations>,
            base_obligations
        >;
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<DescA, 0>, detail::in_port_t<Mma, 0>>,
            iro::compose::Edge<detail::out_port_t<DescB, 0>, detail::in_port_t<Mma, 1>>,
            iro::compose::Edge<detail::out_port_t<Mma, 1>, detail::in_port_t<Commit, 0>>
        >;
    };

    template<int I, bool HasPrev = (I > 0)>
    struct RetireBase;

    template<int I>
    struct RetireBase<I, false> {
        using SlotA = typename slot_traits<I>::SlotA;
        using SlotB = typename slot_traits<I>::SlotB;
        using StageA = axp::level2::staging::StageGmemToSmem<
            Recipe, ATileG, ATileS, ASubj, PipeATag, SlotA,
            iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleA, ATmaT,
            AStageIssueExec, CapT
        >;
        using StageB = axp::level2::staging::StageGmemToSmem<
            Recipe, BTileG, BTileS, BSubj, PipeBTag, SlotB,
            iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleB, BTmaT,
            BStageIssueExec, CapT
        >;

        struct WaitImpl : axp::level2::wgmma::WaitGroup<
            AccRecipe, WgmmaSubj, ExecGroup, wait_depth<I>::value,
            iro::util::type_list<>, iro::util::type_list<>, CapT
        > {};
        struct WaitAccImpl : axp::level2::wgmma::WaitAcc<
            AccRecipe, AccFrag, detail::acc_raw_subj<I>, detail::acc_wait_subj<I>, WgmmaSubj, ExecGroup,
            wait_depth<I>::value, iro::util::type_list<>, iro::util::type_list<>, CapT
        > {};
        using Wait = ConsumerOp<WaitImpl>;
        using WaitAcc = ConsumerOp<WaitAccImpl>;

        struct MarkAImpl : StageA::Mark {};
        struct MarkBImpl : StageB::Mark {};
        struct ReleaseAImpl : StageA::Release {};
        struct ReleaseBImpl : StageB::Release {};
        using MarkA = ConsumerOp<MarkAImpl>;
        using MarkB = ConsumerOp<MarkBImpl>;
        using ReleaseA = ConsumerOp<ReleaseAImpl>;
        using ReleaseB = ConsumerOp<ReleaseBImpl>;
        struct HoldA : axp::level0::SlotAfter<
            Recipe, SlotA, iro::exec::block, iro::token::lifetime::block, ATileS::bytes,
            AccFrag, detail::acc_wait_subj<I>, ExecGroup, typename AccFrag::dist
        > {};
        struct HoldB : axp::level0::SlotAfter<
            Recipe, SlotB, iro::exec::block, iro::token::lifetime::block, BTileS::bytes,
            AccFrag, detail::acc_wait_subj<I>, ExecGroup, typename AccFrag::dist
        > {};

        using obligations = iro::util::type_list<Wait, WaitAcc, HoldA, HoldB, MarkA, MarkB, ReleaseA, ReleaseB>;
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<typename Issue<I>::Mma, 0>,
                               detail::in_port_t<WaitAcc, 0>>,
            iro::compose::Edge<detail::out_port_t<Wait, 0>, detail::in_port_t<WaitAcc, 1>>,
            iro::compose::Edge<detail::out_port_t<WaitAcc, 0>, detail::in_port_t<HoldA, 1>>,
            iro::compose::Edge<detail::out_port_t<WaitAcc, 0>, detail::in_port_t<HoldB, 1>>,
            iro::compose::Edge<detail::out_port_t<HoldA, 0>, detail::in_port_t<MarkA, 0>>,
            iro::compose::Edge<detail::out_port_t<HoldB, 0>, detail::in_port_t<MarkB, 0>>,
            iro::compose::Edge<detail::out_port_t<MarkA, 0>, detail::in_port_t<ReleaseA, 0>>,
            iro::compose::Edge<detail::out_port_t<MarkB, 0>, detail::in_port_t<ReleaseB, 0>>
        >;
        using accum_subj = detail::acc_wait_subj<I>;
        using accum_obligation = WaitAcc;
    };

    template<int I>
    struct RetireBase<I, true> {
        using SlotA = typename slot_traits<I>::SlotA;
        using SlotB = typename slot_traits<I>::SlotB;
        using StageA = axp::level2::staging::StageGmemToSmem<
            Recipe, ATileG, ATileS, ASubj, PipeATag, SlotA,
            iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleA, ATmaT,
            AStageIssueExec, CapT
        >;
        using StageB = axp::level2::staging::StageGmemToSmem<
            Recipe, BTileG, BTileS, BSubj, PipeBTag, SlotB,
            iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleB, BTmaT,
            BStageIssueExec, CapT
        >;

        struct WaitImpl : axp::level2::wgmma::WaitGroup<
            AccRecipe, WgmmaSubj, ExecGroup, wait_depth<I>::value,
            iro::util::type_list<>, iro::util::type_list<>, CapT
        > {};
        struct WaitAccImpl : axp::level2::wgmma::WaitAcc<
            AccRecipe, AccFrag, detail::acc_raw_subj<I>, detail::acc_wait_subj<I>, WgmmaSubj, ExecGroup,
            wait_depth<I>::value, iro::util::type_list<>, iro::util::type_list<>, CapT
        > {};
        using Wait = ConsumerOp<WaitImpl>;
        using WaitAcc = ConsumerOp<WaitAccImpl>;

        using PrevAccum = std::conditional_t<
            (I == 1),
            detail::acc_wait_subj<0>,
            detail::acc_accum_subj<I - 1>
        >;

        struct AddImpl : axp::level0::Add<
            AccRecipe, AccFrag, PrevAccum, detail::acc_wait_subj<I>, detail::acc_accum_subj<I>, ExecGroup
        > {};
        using Add = ConsumerOp<AddImpl>;

        struct MarkAImpl : StageA::Mark {};
        struct MarkBImpl : StageB::Mark {};
        struct ReleaseAImpl : StageA::Release {};
        struct ReleaseBImpl : StageB::Release {};
        using MarkA = ConsumerOp<MarkAImpl>;
        using MarkB = ConsumerOp<MarkBImpl>;
        using ReleaseA = ConsumerOp<ReleaseAImpl>;
        using ReleaseB = ConsumerOp<ReleaseBImpl>;
        struct HoldA : axp::level0::SlotAfter<
            Recipe, SlotA, iro::exec::block, iro::token::lifetime::block, ATileS::bytes,
            AccFrag, detail::acc_wait_subj<I>, ExecGroup, typename AccFrag::dist
        > {};
        struct HoldB : axp::level0::SlotAfter<
            Recipe, SlotB, iro::exec::block, iro::token::lifetime::block, BTileS::bytes,
            AccFrag, detail::acc_wait_subj<I>, ExecGroup, typename AccFrag::dist
        > {};

        using obligations = iro::util::type_list<Wait, WaitAcc, Add, HoldA, HoldB, MarkA, MarkB, ReleaseA, ReleaseB>;
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<typename Issue<I>::Mma, 0>,
                               detail::in_port_t<WaitAcc, 0>>,
            iro::compose::Edge<detail::out_port_t<Wait, 0>, detail::in_port_t<WaitAcc, 1>>,
            iro::compose::Edge<detail::out_port_t<WaitAcc, 0>, detail::in_port_t<Add, 1>>,
            iro::compose::Edge<detail::out_port_t<WaitAcc, 0>, detail::in_port_t<HoldA, 1>>,
            iro::compose::Edge<detail::out_port_t<WaitAcc, 0>, detail::in_port_t<HoldB, 1>>,
            iro::compose::Edge<detail::out_port_t<HoldA, 0>, detail::in_port_t<MarkA, 0>>,
            iro::compose::Edge<detail::out_port_t<HoldB, 0>, detail::in_port_t<MarkB, 0>>,
            iro::compose::Edge<detail::out_port_t<MarkA, 0>, detail::in_port_t<ReleaseA, 0>>,
            iro::compose::Edge<detail::out_port_t<MarkB, 0>, detail::in_port_t<ReleaseB, 0>>
        >;
        using accum_subj = detail::acc_accum_subj<I>;
        using accum_obligation = Add;
    };

    template<int I>
    using Retire = RetireBase<I, (I > 0)>;

    template<int I, bool Enable>
    struct retire_for {
        using type = EmptyPhase;
    };

    template<int I>
    struct retire_for<I, true> {
        using type = Retire<I - kWgmmaWaitDepth>;
    };

    template<int I>
    struct ComputePrefetch {
        using IssueCurr = Issue<I>;
        using PrefetchNext = Prefetch<I + Stages>;
        using RetirePrev = typename retire_for<I, (I >= kWgmmaWaitDepth)>::type;
        using obligations = iro::util::concat_t<
            iro::util::concat_t<typename IssueCurr::obligations, typename PrefetchNext::obligations>,
            typename RetirePrev::obligations
        >;
        using edges = iro::util::concat_t<
            iro::util::concat_t<typename IssueCurr::edges, typename PrefetchNext::edges>,
            typename RetirePrev::edges
        >;
    };

    template<int I>
    struct ComputeDrain {
        using IssueCurr = Issue<I>;
        using RetirePrev = typename retire_for<I, (I >= kWgmmaWaitDepth)>::type;
        using obligations = iro::util::concat_t<typename IssueCurr::obligations, typename RetirePrev::obligations>;
        using edges = iro::util::concat_t<typename IssueCurr::edges, typename RetirePrev::edges>;
    };

    template<int I>
    struct ComputeBulk {
        using IssueCurr = Issue<I>;
        using PrefetchCurr = Prefetch<I>;
        using DescMaybe = std::conditional_t<(I < Stages), DescCache<I>, EmptyPhase>;
        using RetirePrev = typename retire_for<I, (I >= kWgmmaWaitDepth)>::type;
        using obligations = iro::util::concat_t<
            iro::util::concat_t<
                iro::util::concat_t<typename PrefetchCurr::obligations, typename DescMaybe::obligations>,
                typename IssueCurr::obligations
            >,
            typename RetirePrev::obligations
        >;
        using edges = iro::util::concat_t<
            iro::util::concat_t<
                iro::util::concat_t<typename PrefetchCurr::edges, typename DescMaybe::edges>,
                typename IssueCurr::edges
            >,
            typename RetirePrev::edges
        >;
    };

    template<int I, int End>
    struct build_prefetch {
        using curr = Prefetch<I>;
        using next = build_prefetch<I + 1, End>;
        using obligations = iro::util::concat_t<typename curr::obligations, typename next::obligations>;
        using edges = iro::util::concat_t<typename curr::edges, typename next::edges>;
    };

    template<int End>
    struct build_prefetch<End, End> {
        using obligations = iro::util::type_list<>;
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_desc_cache {
        using curr = DescCache<I>;
        using next = build_desc_cache<I + 1, End>;
        using obligations = iro::util::concat_t<typename curr::obligations, typename next::obligations>;
        using edges = iro::util::concat_t<typename curr::edges, typename next::edges>;
    };

    template<int End>
    struct build_desc_cache<End, End> {
        using obligations = iro::util::type_list<>;
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_desc_use_edges {
        static constexpr int slot = slot_traits<I>::slot;
        using curr = iro::util::type_list<
            iro::compose::Edge<
                detail::out_port_t<typename DescCache<slot>::MakeDescA, 0>,
                detail::in_port_t<typename Issue<I>::DescA, 1>
            >,
            iro::compose::Edge<
                detail::out_port_t<typename DescCache<slot>::MakeDescB, 0>,
                detail::in_port_t<typename Issue<I>::DescB, 1>
            >
        >;
        using next = build_desc_use_edges<I + 1, End>;
        using edges = iro::util::concat_t<curr, typename next::edges>;
    };

    template<int End>
    struct build_desc_use_edges<End, End> {
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_steady {
        using curr = std::conditional_t<kBulkSchedule, ComputeBulk<I>, ComputePrefetch<I>>;
        using next = build_steady<I + 1, End>;
        using obligations = iro::util::concat_t<typename curr::obligations, typename next::obligations>;
        using edges = iro::util::concat_t<typename curr::edges, typename next::edges>;
    };

    template<int End>
    struct build_steady<End, End> {
        using obligations = iro::util::type_list<>;
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_drain {
        using curr = ComputeDrain<I>;
        using next = build_drain<I + 1, End>;
        using obligations = iro::util::concat_t<typename curr::obligations, typename next::obligations>;
        using edges = iro::util::concat_t<typename curr::edges, typename next::edges>;
    };

    template<int End>
    struct build_drain<End, End> {
        using obligations = iro::util::type_list<>;
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_retire_tail {
        using curr = Retire<I>;
        using next = build_retire_tail<I + 1, End>;
        using obligations = iro::util::concat_t<typename curr::obligations, typename next::obligations>;
        using edges = iro::util::concat_t<typename curr::edges, typename next::edges>;
    };

    template<int End>
    struct build_retire_tail<End, End> {
        using obligations = iro::util::type_list<>;
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_wait_edges {
        using wait_a_edges = detail::wait_scale_edges_t<
            typename Prefetch<I>::WaitA,
            typename Issue<I>::ScaleA,
            typename Issue<I>::DescA,
            kHasScaleA,
            0
        >;
        using wait_b_edges = detail::wait_scale_edges_t<
            typename Prefetch<I>::WaitB,
            typename Issue<I>::ScaleB,
            typename Issue<I>::DescB,
            kHasScaleB,
            0
        >;
        using curr = iro::util::concat_t<
            iro::util::concat_t<wait_a_edges, wait_b_edges>,
            iro::util::type_list<
                iro::compose::Edge<detail::out_port_t<typename Prefetch<I>::WaitA, 1>,
                                   detail::in_port_t<typename Retire<I>::HoldA, 0>>,
                iro::compose::Edge<detail::out_port_t<typename Prefetch<I>::WaitB, 1>,
                                   detail::in_port_t<typename Retire<I>::HoldB, 0>>
            >
        >;
        using next = build_wait_edges<I + 1, End>;
        using edges = iro::util::concat_t<curr, typename next::edges>;
    };

    template<int End>
    struct build_wait_edges<End, End> {
        using edges = iro::util::type_list<>;
    };

    template<int I, bool Enable>
    struct reuse_edge_a {
        using edges = iro::util::type_list<>;
    };

    template<int I>
    struct reuse_edge_a<I, true> {
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<typename Retire<I>::ReleaseA, 0>,
                               detail::in_port_t<typename Prefetch<I + Stages>::IssueA, 1>>
        >;
    };

    template<int I, bool Enable>
    struct reuse_edge_b {
        using edges = iro::util::type_list<>;
    };

    template<int I>
    struct reuse_edge_b<I, true> {
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<typename Retire<I>::ReleaseB, 0>,
                               detail::in_port_t<typename Prefetch<I + Stages>::IssueB, 1>>
        >;
    };

    template<int I, int End>
    struct build_reuse_edges {
        static constexpr bool has_next = (I + Stages < End);
        using curr = iro::util::concat_t<
            typename reuse_edge_a<I, has_next && !kHasTmaA>::edges,
            typename reuse_edge_b<I, has_next && !kHasTmaB>::edges
        >;
        using next = build_reuse_edges<I + 1, End>;
        using edges = iro::util::concat_t<curr, typename next::edges>;
    };

    template<int End>
    struct build_reuse_edges<End, End> {
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_wgmma_wait_edges {
        static constexpr int depth = wait_depth<I>::value;
        static constexpr int src = I + depth;
        using curr = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<typename Issue<src>::Commit, 0>,
                               detail::in_port_t<typename Retire<I>::Wait, 0>>
        >;
        using next = build_wgmma_wait_edges<I + 1, End>;
        using edges = iro::util::concat_t<curr, typename next::edges>;
    };

    template<int End>
    struct build_wgmma_wait_edges<End, End> {
        using edges = iro::util::type_list<>;
    };

    template<int I, int End, bool Enable, class = void>
    struct build_barrier_edges_a {
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_barrier_edges_a<I, End, true, std::enable_if_t<(I < End)>> {
        using curr = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<ABarrierInit, 0>,
                               detail::in_port_t<typename Prefetch<I>::IssueA, 1>>
        >;
        using next = build_barrier_edges_a<I + 1, End, true>;
        using edges = iro::util::concat_t<curr, typename next::edges>;
    };

    template<int End, bool Enable>
    struct build_barrier_edges_a<End, End, Enable, void> {
        using edges = iro::util::type_list<>;
    };

    template<int I, int End, bool Enable, class = void>
    struct build_barrier_edges_b {
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_barrier_edges_b<I, End, true, std::enable_if_t<(I < End)>> {
        using curr = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<BBarrierInit, 0>,
                               detail::in_port_t<typename Prefetch<I>::IssueB, 1>>
        >;
        using next = build_barrier_edges_b<I + 1, End, true>;
        using edges = iro::util::concat_t<curr, typename next::edges>;
    };

    template<int End, bool Enable>
    struct build_barrier_edges_b<End, End, Enable, void> {
        using edges = iro::util::type_list<>;
    };

    template<int I, bool HasPrev>
    struct accum_edge {
        using edges = iro::util::type_list<>;
    };

    template<int I>
    struct accum_edge<I, true> {
        using edges = iro::util::type_list<
            iro::compose::Edge<
                detail::out_port_t<typename Retire<I - 1>::accum_obligation, 0>,
                detail::in_port_t<typename Retire<I>::Add, 0>
            >
        >;
    };

    template<int I, int End>
    struct build_accum_edges {
        using curr = typename accum_edge<I, (I > 0)>::edges;
        using next = build_accum_edges<I + 1, End>;
        using edges = iro::util::concat_t<curr, typename next::edges>;
    };

    template<int End>
    struct build_accum_edges<End, End> {
        using edges = iro::util::type_list<>;
    };

    template<int I, int End>
    struct build_fence_edges {
        using curr = iro::util::type_list<
            iro::compose::Edge<
                detail::out_port_t<FenceHandle, 0>,
                detail::in_port_t<typename Issue<I>::Mma, 2>
            >
        >;
        using next = build_fence_edges<I + 1, End>;
        using edges = iro::util::concat_t<curr, typename next::edges>;
    };

    template<int End>
    struct build_fence_edges<End, End> {
        using edges = iro::util::type_list<>;
    };

    using Prologue = axp::level3::detail::make_composition_t<
        iro::util::concat_t<
            typename build_prefetch<0, kPrefetchTiles>::obligations,
            typename build_desc_cache<0, kDescTiles>::obligations
        >,
        iro::util::concat_t<
            typename build_prefetch<0, kPrefetchTiles>::edges,
            typename build_desc_cache<0, kDescTiles>::edges
        >
    >;

    using Steady = axp::level3::detail::make_composition_t<
        typename build_steady<0, kSteadyTiles>::obligations,
        typename build_steady<0, kSteadyTiles>::edges
    >;

    using Drain = axp::level3::detail::make_composition_t<
        typename build_drain<kDrainStart, KTiles>::obligations,
        typename build_drain<kDrainStart, KTiles>::edges
    >;

    static constexpr int kRetireTailStart = (kWgmmaWaitDepth > 0) ? (KTiles - kWgmmaWaitDepth) : KTiles;

    using Tail = build_retire_tail<kRetireTailStart, KTiles>;

    using FinalAccum = typename Retire<KTiles - 1>::accum_subj;
    using FinalAccumObligation = typename Retire<KTiles - 1>::accum_obligation;

    using EpilogueBuild = detail::build_vec_epilogue<
        Recipe, AccFrag, ExecGroup, EpiloguePolicy, FinalAccum, FinalAccumObligation, CapT
    >;
    using EpilogueAccum = typename EpilogueBuild::out_frag_subj;
    using EpilogueProducer = typename EpilogueBuild::out_frag_producer;

    using Store = axp::level0::FragmentToSharedTile<
        Recipe, AccFrag, CTileS, EpilogueAccum, CSubj, ExecGroup, iro::token::lifetime::warpgroup
    >;

    using Fence = axp::level0::TileFence<
        Recipe, CTileS, CSubj, iro::exec::block
    >;

    using phase_obligations = iro::util::concat_t<
        iro::util::concat_t<iro::util::concat_t<typename Prologue::obligations, typename Steady::obligations>,
                            typename Drain::obligations>,
        typename Tail::obligations
    >;

    using epilogue_obligations = typename EpilogueBuild::obligations;

    using barrier_obligations = iro::util::concat_t<
        typename ABarrierInitHelper::obligations,
        typename BBarrierInitHelper::obligations
    >;

    using phase_edges = iro::util::concat_t<
        iro::util::concat_t<iro::util::concat_t<typename Prologue::edges, typename Steady::edges>,
                            typename Drain::edges>,
        typename Tail::edges
    >;

    using epilogue_edges = typename EpilogueBuild::edges;

    using barrier_edges = iro::util::concat_t<
        typename build_barrier_edges_a<0, KTiles, kHasTmaA>::edges,
        typename build_barrier_edges_b<0, KTiles, kHasTmaB>::edges
    >;

    using global_edges = iro::util::concat_t<
        iro::util::concat_t<
            iro::util::concat_t<
                typename build_wait_edges<0, KTiles>::edges,
                typename build_desc_use_edges<0, KTiles>::edges
            >,
            typename build_reuse_edges<0, KTiles>::edges
        >,
        iro::util::concat_t<
            typename build_wgmma_wait_edges<0, KTiles>::edges,
            iro::util::concat_t<
                typename build_accum_edges<0, KTiles>::edges,
                typename build_fence_edges<0, KTiles>::edges
            >
        >
    >;

    template<bool Enable, class = void>
    struct boundary_tilein_a {
        using edges = iro::util::type_list<>;
        using obligations = iro::util::type_list<>;
    };

    template<bool Enable>
    struct boundary_tilein_a<Enable, std::enable_if_t<Enable>> {
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<TileInA, 0>, detail::in_port_t<typename Prefetch<0>::IssueA, 0>>
        >;
        using obligations = iro::util::type_list<TileInA>;
    };

    template<bool Enable, class = void>
    struct boundary_tilein_b {
        using edges = iro::util::type_list<>;
        using obligations = iro::util::type_list<>;
    };

    template<bool Enable>
    struct boundary_tilein_b<Enable, std::enable_if_t<Enable>> {
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<TileInB, 0>, detail::in_port_t<typename Prefetch<0>::IssueB, 0>>
        >;
        using obligations = iro::util::type_list<TileInB>;
    };

    using boundary_edges = iro::util::concat_t<
        iro::util::concat_t<
            typename boundary_tilein_a<!kHasTmaA>::edges,
            typename boundary_tilein_b<!kHasTmaB>::edges
        >,
        iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<Fence, 0>, detail::in_port_t<TileOut, 0>>
        >
    >;

    using boundary_obligations = iro::util::concat_t<
        typename boundary_tilein_a<!kHasTmaA>::obligations,
        typename boundary_tilein_b<!kHasTmaB>::obligations
    >;

    using obligations = iro::util::concat_t<
        iro::util::concat_t<phase_obligations, epilogue_obligations>,
        iro::util::concat_t<barrier_obligations,
                            iro::util::concat_t<
                                ScheduleObligations,
                                iro::util::concat_t<
                                    boundary_obligations,
                                    iro::util::type_list<RegPressure, FenceHandle, Store, Fence, TileOut>
                                >
                            >>
    >;

    using edges = iro::util::concat_t<
        iro::util::concat_t<iro::util::concat_t<iro::util::concat_t<iro::util::concat_t<phase_edges, epilogue_edges>, global_edges>, barrier_edges>, boundary_edges>,
        iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<EpilogueProducer, 0>, detail::in_port_t<Store, 0>>,
            iro::compose::Edge<detail::out_port_t<Store, 0>, detail::in_port_t<Fence, 0>>
        >
    >;

    using type = axp::level3::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

} // namespace axp::level3::gemm

namespace axp::level3 {

template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj,
         class WgmmaSubj,
         class MemoryPatternA = axp::intent::memory_pattern::Optimized,
         class MemoryPatternB = axp::intent::memory_pattern::Optimized,
         class LoadModeA = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeB = axp::intent::load_mode::AsyncPrefetch,
         class Schedule = axp::intent::schedule::Pipelined,
         class ScaleASubj = iro::contract::subject::global,
         class ScaleBSubj = iro::contract::subject::global>
struct GemmTileConfig {
    using recipe = Recipe;
    static constexpr int block_m = BlockM;
    static constexpr int block_n = BlockN;
    static constexpr int block_k = BlockK;
    static constexpr int stages = Stages;
    static constexpr int k_tiles = KTiles;
    using a_subj = ASubj;
    using b_subj = BSubj;
    using c_subj = CSubj;
    using wgmma_subj = WgmmaSubj;
    using memory_pattern_a = MemoryPatternA;
    using memory_pattern_b = MemoryPatternB;
    using load_mode_a = LoadModeA;
    using load_mode_b = LoadModeB;
    using schedule = Schedule;
    using scale_a_subj = ScaleASubj;
    using scale_b_subj = ScaleBSubj;
};

template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj,
         class WgmmaSubj,
         class MemoryPatternA = axp::intent::memory_pattern::Optimized,
         class MemoryPatternB = axp::intent::memory_pattern::Optimized,
         class LoadModeA = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeB = axp::intent::load_mode::AsyncPrefetch,
         class Schedule = axp::intent::schedule::Pipelined,
         class ScaleASubj = iro::contract::subject::global,
         class ScaleBSubj = iro::contract::subject::global,
         class ATma = void, class BTma = void>
struct GemmTileConfigSm90Multicast
    : GemmTileConfig<Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
                     ASubj, BSubj, CSubj, WgmmaSubj,
                     MemoryPatternA, MemoryPatternB,
                     LoadModeA, LoadModeB,
                     Schedule,
                     ScaleASubj, ScaleBSubj> {
    static_assert(std::is_same_v<LoadModeA, axp::intent::load_mode::AsyncPrefetch>,
                  "GemmTileConfigSm90Multicast requires load_mode_a = AsyncPrefetch");
    static_assert(std::is_same_v<LoadModeB, axp::intent::load_mode::AsyncPrefetch>,
                  "GemmTileConfigSm90Multicast requires load_mode_b = AsyncPrefetch");
    static_assert(axp::level2::staging::tma_multicast_traits<ATma>::valid,
                  "GemmTileConfigSm90Multicast requires ATma multicast config");
    static_assert(axp::level2::staging::tma_multicast_traits<BTma>::valid,
                  "GemmTileConfigSm90Multicast requires BTma multicast config");
    using a_tma = ATma;
    using b_tma = BTma;
};

template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj,
         class WgmmaSubj,
         class EpiloguePolicy,
         class MemoryPatternA = axp::intent::memory_pattern::Optimized,
         class MemoryPatternB = axp::intent::memory_pattern::Optimized,
         class LoadModeA = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeB = axp::intent::load_mode::AsyncPrefetch,
         class Schedule = axp::intent::schedule::Pipelined,
         class ScaleASubj = iro::contract::subject::global,
         class ScaleBSubj = iro::contract::subject::global>
struct GemmTileFusedConfig {
    using recipe = Recipe;
    static constexpr int block_m = BlockM;
    static constexpr int block_n = BlockN;
    static constexpr int block_k = BlockK;
    static constexpr int stages = Stages;
    static constexpr int k_tiles = KTiles;
    using a_subj = ASubj;
    using b_subj = BSubj;
    using c_subj = CSubj;
    using wgmma_subj = WgmmaSubj;
    using epilogue_policy = EpiloguePolicy;
    using memory_pattern_a = MemoryPatternA;
    using memory_pattern_b = MemoryPatternB;
    using load_mode_a = LoadModeA;
    using load_mode_b = LoadModeB;
    using schedule = Schedule;
    using scale_a_subj = ScaleASubj;
    using scale_b_subj = ScaleBSubj;
};

namespace gemm::detail {
template<class Config, class CapT, bool IsA, class = void>
struct config_tma_or_void {
    using type = void;
};

template<class Config, class CapT, bool IsA>
struct config_tma_or_void<Config, CapT, IsA, std::void_t<typename Config::a_tma>> {
    using type = std::conditional_t<IsA, typename Config::a_tma, typename Config::b_tma>;
};

template<class Config, class CapT, bool IsA>
using config_tma_or_void_t = typename config_tma_or_void<Config, CapT, IsA>::type;

template<bool UseWarpPath,
         class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj, class ScaleASubj, class ScaleBSubj, class WgmmaSubj,
         class MemoryPatternA, class MemoryPatternB,
         class LoadModeA, class LoadModeB, class Schedule,
         class ATma, class BTma, class Cap, class EpiloguePolicy = axp::level3::gemm::epilogue::None>
struct select_non_wgmma_gemm;

template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj, class ScaleASubj, class ScaleBSubj, class WgmmaSubj,
         class MemoryPatternA, class MemoryPatternB,
         class LoadModeA, class LoadModeB, class Schedule,
         class ATma, class BTma, class Cap, class EpiloguePolicy>
struct select_non_wgmma_gemm<true,
                             Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
                             ASubj, BSubj, CSubj, ScaleASubj, ScaleBSubj, WgmmaSubj,
                             MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB, Schedule,
                             ATma, BTma, Cap, EpiloguePolicy> {
    using type = typename axp::level3::gemm::GemmTileWarpImpl<
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
        ASubj, BSubj, CSubj, ScaleASubj, ScaleBSubj, WgmmaSubj,
        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB, Schedule,
        axp::level3::gemm::detail::pipe_a_tag,
        axp::level3::gemm::detail::pipe_b_tag,
        ATma, BTma, Cap, EpiloguePolicy
    >::type;
};

template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj, class ScaleASubj, class ScaleBSubj, class WgmmaSubj,
         class MemoryPatternA, class MemoryPatternB,
         class LoadModeA, class LoadModeB, class Schedule,
         class ATma, class BTma, class Cap, class EpiloguePolicy>
struct select_non_wgmma_gemm<false,
                             Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
                             ASubj, BSubj, CSubj, ScaleASubj, ScaleBSubj, WgmmaSubj,
                             MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB, Schedule,
                             ATma, BTma, Cap, EpiloguePolicy> {
    static_assert(BlockM == -1,
                  "axp::level3::GemmTile: unsupported non-WGMMA shape. "
                  "Provide a warp-path configuration or a WGMMA-capable target.");
    using type = void;
};

template<bool UseWarpPath,
         class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj, class ScaleASubj, class ScaleBSubj, class WgmmaSubj,
         class MemoryPatternA, class MemoryPatternB,
         class LoadModeA, class LoadModeB, class Schedule,
         class ATma, class BTma, class Cap>
struct select_has_wgmma_gemm;

template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj, class ScaleASubj, class ScaleBSubj, class WgmmaSubj,
         class MemoryPatternA, class MemoryPatternB,
         class LoadModeA, class LoadModeB, class Schedule,
         class ATma, class BTma, class Cap>
struct select_has_wgmma_gemm<true,
                             Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
                             ASubj, BSubj, CSubj, ScaleASubj, ScaleBSubj, WgmmaSubj,
                             MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB, Schedule,
                             ATma, BTma, Cap> {
    using type = typename select_non_wgmma_gemm<
        true,
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
        ASubj, BSubj, CSubj, ScaleASubj, ScaleBSubj, WgmmaSubj,
        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB, Schedule,
        ATma, BTma, Cap
    >::type;
};

template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj, class ScaleASubj, class ScaleBSubj, class WgmmaSubj,
         class MemoryPatternA, class MemoryPatternB,
         class LoadModeA, class LoadModeB, class Schedule,
         class ATma, class BTma, class Cap>
struct select_has_wgmma_gemm<false,
                             Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
                             ASubj, BSubj, CSubj, ScaleASubj, ScaleBSubj, WgmmaSubj,
                             MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB, Schedule,
                             ATma, BTma, Cap> {
    using type = typename axp::level3::gemm::GemmTileWgmmaImpl<
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles, ASubj, BSubj, CSubj, ScaleASubj, ScaleBSubj,
        WgmmaSubj, MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB, Schedule,
        axp::level3::gemm::detail::pipe_a_tag,
        axp::level3::gemm::detail::pipe_b_tag,
        ATma, BTma,
        Cap
    >::type;
};
} // namespace gemm::detail

template<class Config, class CapT = axp::target_cap>
using GemmTile = registry::Select<registry::GemmTilePattern<
    typename Config::recipe,
    Config::block_m, Config::block_n, Config::block_k,
    Config::stages, Config::k_tiles,
    typename Config::a_subj, typename Config::b_subj, typename Config::c_subj,
    typename Config::wgmma_subj,
    typename Config::memory_pattern_a,
    typename Config::memory_pattern_b,
    typename Config::load_mode_a,
    typename Config::load_mode_b,
    typename Config::schedule,
    typename Config::scale_a_subj, typename Config::scale_b_subj,
    gemm::detail::config_tma_or_void_t<Config, CapT, /*is_a=*/true>,
    gemm::detail::config_tma_or_void_t<Config, CapT, /*is_a=*/false>>, CapT>;

template<class Config, class CapT = axp::target_cap>
using GemmTileFused = registry::Select<registry::GemmTileFusedPattern<
    typename Config::recipe,
    Config::block_m, Config::block_n, Config::block_k,
    Config::stages, Config::k_tiles,
    typename Config::a_subj, typename Config::b_subj, typename Config::c_subj,
    typename Config::wgmma_subj,
    typename Config::epilogue_policy,
    typename Config::memory_pattern_a,
    typename Config::memory_pattern_b,
    typename Config::load_mode_a,
    typename Config::load_mode_b,
    typename Config::schedule,
    typename Config::scale_a_subj, typename Config::scale_b_subj,
    gemm::detail::config_tma_or_void_t<Config, CapT, /*is_a=*/true>,
    gemm::detail::config_tma_or_void_t<Config, CapT, /*is_a=*/false>>, CapT>;

} // namespace axp::level3

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level3::registry {

template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj, class WgmmaSubj,
         class MemoryPatternA, class MemoryPatternB,
         class LoadModeA, class LoadModeB,
         class Schedule,
         class ScaleASubj, class ScaleBSubj,
         class ATma, class BTma, class Cap>
struct resolve_impl<GemmTilePattern<Recipe, BlockM, BlockN, BlockK, Stages, KTiles, ASubj, BSubj, CSubj,
                                    WgmmaSubj, MemoryPatternA, MemoryPatternB,
                                    LoadModeA, LoadModeB, Schedule,
                                    ScaleASubj, ScaleBSubj, ATma, BTma>,
                    Cap,
                    std::enable_if_t<!Cap::has_wgmma && std::is_same_v<typename Recipe::acc, iro::elem::f32>>> {
    static constexpr bool supported = true;
    static constexpr bool kWmmaMacroShape =
        axp::protocol::compute::detail::is_wmma_shape_v<
            BlockM, BlockN, BlockK,
            iro::verify::recipe_in_a_t<Recipe>,
            iro::verify::recipe_in_b_t<Recipe>,
            typename Recipe::acc>;
    using type = typename axp::level3::gemm::detail::select_non_wgmma_gemm<
        kWmmaMacroShape,
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
        ASubj, BSubj, CSubj, ScaleASubj, ScaleBSubj, WgmmaSubj,
        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB, Schedule,
        ATma, BTma, Cap
    >::type;
};

template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj, class WgmmaSubj,
         class MemoryPatternA, class MemoryPatternB,
         class LoadModeA, class LoadModeB,
         class Schedule,
         class ScaleASubj, class ScaleBSubj,
         class ATma, class BTma, class Cap>
struct resolve_impl<GemmTilePattern<Recipe, BlockM, BlockN, BlockK, Stages, KTiles, ASubj, BSubj, CSubj,
                                    WgmmaSubj, MemoryPatternA, MemoryPatternB,
                                    LoadModeA, LoadModeB, Schedule,
                                    ScaleASubj, ScaleBSubj, ATma, BTma>,
                    Cap,
                    std::enable_if_t<Cap::has_wgmma && std::is_same_v<typename Recipe::acc, iro::elem::f32>>> {
    static constexpr bool supported = true;
    static constexpr bool kWmmaMacroShape =
        axp::protocol::compute::detail::is_wmma_shape_v<
            BlockM, BlockN, BlockK,
            iro::verify::recipe_in_a_t<Recipe>,
            iro::verify::recipe_in_b_t<Recipe>,
            typename Recipe::acc>;
    using type = typename axp::level3::gemm::detail::select_has_wgmma_gemm<
        kWmmaMacroShape,
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
        ASubj, BSubj, CSubj, ScaleASubj, ScaleBSubj, WgmmaSubj,
        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB, Schedule,
        ATma, BTma, Cap
    >::type;
};

template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj, class WgmmaSubj, class EpiloguePolicy,
         class MemoryPatternA, class MemoryPatternB,
         class LoadModeA, class LoadModeB,
         class Schedule,
         class ScaleASubj, class ScaleBSubj, class ATma, class BTma, class Cap>
struct resolve_impl<GemmTileFusedPattern<
                        Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
                        ASubj, BSubj, CSubj, WgmmaSubj, EpiloguePolicy,
                        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB, Schedule,
                        ScaleASubj, ScaleBSubj, ATma, BTma>,
                    Cap,
                    std::enable_if_t<!Cap::has_wgmma && std::is_same_v<typename Recipe::acc, iro::elem::f32>>> {
    static constexpr bool supported = true;
    using type = typename axp::level3::gemm::GemmTileWarpImpl<
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles, ASubj, BSubj, CSubj, ScaleASubj, ScaleBSubj, WgmmaSubj,
        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB, Schedule,
        axp::level3::gemm::detail::pipe_a_tag,
        axp::level3::gemm::detail::pipe_b_tag,
        ATma, BTma,
        Cap,
        EpiloguePolicy
    >::type;
};

template<class Recipe, int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj, class BSubj, class CSubj, class WgmmaSubj, class EpiloguePolicy,
         class MemoryPatternA, class MemoryPatternB,
         class LoadModeA, class LoadModeB,
         class Schedule,
         class ScaleASubj, class ScaleBSubj, class ATma, class BTma, class Cap>
struct resolve_impl<GemmTileFusedPattern<
                        Recipe, BlockM, BlockN, BlockK, Stages, KTiles,
                        ASubj, BSubj, CSubj, WgmmaSubj, EpiloguePolicy,
                        MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB, Schedule,
                        ScaleASubj, ScaleBSubj, ATma, BTma>,
                    Cap,
                    std::enable_if_t<Cap::has_wgmma && std::is_same_v<typename Recipe::acc, iro::elem::f32>>> {
    static constexpr bool supported = true;
    using type = typename axp::level3::gemm::GemmTileWgmmaImpl<
        Recipe, BlockM, BlockN, BlockK, Stages, KTiles, ASubj, BSubj, CSubj, ScaleASubj, ScaleBSubj,
        WgmmaSubj, MemoryPatternA, MemoryPatternB, LoadModeA, LoadModeB, Schedule,
        axp::level3::gemm::detail::pipe_a_tag,
        axp::level3::gemm::detail::pipe_b_tag,
        ATma, BTma,
        Cap,
        EpiloguePolicy
    >::type;
};

} // namespace axp::level3::registry
#endif // AXP_LIBRARY_BUILD
