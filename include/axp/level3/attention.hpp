#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../level0/compute_alias.hpp"
#include "../level0/compute.hpp"
#include "../level0/ownership.hpp"
#include "../level0/convert.hpp"
#include "../level0/sync.hpp"
#include "../level0/stage.hpp"
#include "../level2/matmul.hpp"
#include "../level2/staging.hpp"
#include "../level2/attention.hpp"
#include "../swizzle.hpp"
#include "../intent.hpp"
#include "../kits/intent.hpp"
#include "detail/compose.hpp"
#include "detail/reg_pressure.hpp"
#include "registry.hpp"

namespace axp::level3::attention {

namespace detail {
struct qk_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.attn.qk_frag"); };
struct weights_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.attn.weights_frag"); };
struct tile_state_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.attn.tile_state"); };
struct combined_state_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.attn.combined_state"); };
struct scale_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.attn.scale"); };
struct scale_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.attn.scale_frag"); };
struct weights_scaled_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.attn.weights_scaled"); };
struct weights_f32_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.attn.weights_f32"); };
struct weights_f16_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.attn.weights_f16"); };
struct pv_frag_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.attn.pv_frag"); };
struct wgmma_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.attn.wgmma"); };
struct q_pipe_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.attn.q_pipe"); };
struct k_pipe_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.attn.k_pipe"); };
struct v_pipe_tag { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.level3.attn.v_pipe"); };

using qk_frag_subj = iro::contract::subject::indexed<qk_frag_tag, 0>;
using weights_frag_subj = iro::contract::subject::indexed<weights_frag_tag, 0>;
using tile_state_subj = iro::contract::subject::indexed<tile_state_tag, 0>;
using combined_state_subj = iro::contract::subject::indexed<combined_state_tag, 0>;
using scale_subj = iro::contract::subject::indexed<scale_tag, 0>;
using scale_frag_subj = iro::contract::subject::indexed<scale_frag_tag, 0>;
using weights_scaled_subj = iro::contract::subject::indexed<weights_scaled_tag, 0>;
using weights_f32_subj = iro::contract::subject::indexed<weights_f32_tag, 0>;
using weights_f16_subj = iro::contract::subject::indexed<weights_f16_tag, 0>;
using pv_frag_subj = iro::contract::subject::indexed<pv_frag_tag, 0>;

template<class Obligation, int I>
using in_port_t = axp::level3::detail::in_port_t<Obligation, I>;

template<class Obligation, int I>
using out_port_t = axp::level3::detail::out_port_t<Obligation, I>;

using axp::level3::detail::reg_pressure_obligation;

template<class Issue, class Wait, bool HasTma>
struct stage_issue_wait_edges;

template<class Issue, class Wait>
struct stage_issue_wait_edges<Issue, Wait, false> {
    using type = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<Issue, 0>, detail::in_port_t<Wait, 0>>
    >;
};

template<class Issue, class Wait>
struct stage_issue_wait_edges<Issue, Wait, true> {
    using type = iro::util::type_list<
        iro::compose::Edge<detail::out_port_t<Issue, 0>, detail::in_port_t<Wait, 0>>,
        iro::compose::Edge<detail::out_port_t<Issue, 1>, detail::in_port_t<Wait, 1>>
    >;
};

template<class Issue, class Wait, bool HasTma>
using stage_issue_wait_edges_t = typename stage_issue_wait_edges<Issue, Wait, HasTma>::type;
} // namespace detail

// AttentionTile: QK -> tile softmax -> online softmax update -> PV (warp-level, WMMA only).
// Inputs: Q/K/V tiles in shared, old softmax state, accumulator fragment.
// Outputs: updated accumulator fragment and new softmax state.
template<
    class Recipe,
    int TileM, int TileN, int HeadDim,
    int Stages, int SlotIdx,
    class QSubj, class KSubj, class VSubj,
    class AccSubj, class OldStateSubj, class OutStateSubj,
    class MemoryPatternQ, class MemoryPatternK, class MemoryPatternV,
    class LoadModeQ, class LoadModeK, class LoadModeV,
    class Schedule,
    class TileSkip,
    class QTma = void, class KTma = void, class VTma = void,
    class CapT = axp::target_cap>
struct AttentionTileImpl {
    static_assert(std::is_same_v<typename Recipe::acc, iro::elem::f32>,
                  "AttentionTile: Recipe::acc must be f32");
    using ElemA = iro::verify::recipe_in_a_t<Recipe>;
    using ElemB = iro::verify::recipe_in_b_t<Recipe>;
    static_assert(std::is_same_v<ElemA, ElemB>, "AttentionTile: Q/K/V elem must match");
    static_assert(std::is_same_v<ElemA, iro::elem::f16> || std::is_same_v<ElemA, iro::elem::bf16>,
                  "AttentionTile: only f16/bf16 inputs supported");
    static_assert(std::is_same_v<TileSkip, axp::intent::tile_skip::None> ||
                  std::is_same_v<TileSkip, axp::intent::tile_skip::Causal>,
                  "AttentionTile: TileSkip must be intent::tile_skip::None or Causal");

    static_assert(TileM == 16 && TileN == 16 && HeadDim == 16,
                  "AttentionTile: WMMA 16x16x16 only (TileM=16, TileN=16, HeadDim=16)");
    static_assert(Stages >= 2 && Stages <= 4, "AttentionTile: Stages must be 2-4");
    static_assert(SlotIdx >= 0 && SlotIdx < Stages, "AttentionTile: SlotIdx must be in [0, Stages)");
    using ScheduleT = axp::kit::detail::select_schedule_t<Schedule, CapT>;
    static constexpr bool kProducerConsumer = std::is_same_v<ScheduleT, axp::intent::schedule::ProducerConsumer>;
    static_assert(!kProducerConsumer,
                  "AttentionTile: ProducerConsumer schedule requires WGMMA-capable path");
    using SwizzleAtomQ = std::conditional_t<
        std::is_same_v<LoadModeQ, axp::intent::load_mode::AsyncPrefetch>,
        axp::kit::detail::select_swizzle_t<MemoryPatternQ, ElemA, HeadDim, CapT, true>,
        axp::swizzle::None>;
    using SwizzleAtomK = std::conditional_t<
        std::is_same_v<LoadModeK, axp::intent::load_mode::AsyncPrefetch>,
        axp::kit::detail::select_swizzle_t<MemoryPatternK, ElemB, HeadDim, CapT, false>,
        axp::swizzle::None>;
    using SwizzleAtomV = std::conditional_t<
        std::is_same_v<LoadModeV, axp::intent::load_mode::AsyncPrefetch>,
        axp::kit::detail::select_swizzle_t<MemoryPatternV, ElemB, TileN, CapT, false>,
        axp::swizzle::None>;
    static_assert(std::is_same_v<SwizzleAtomQ, axp::swizzle::None> &&
                  std::is_same_v<SwizzleAtomK, axp::swizzle::None> &&
                  std::is_same_v<SwizzleAtomV, axp::swizzle::None>,
                  "AttentionTile: Swizzle is only supported on WGMMA path");

    using ExecGroup = iro::exec::warp;
    using AccRecipe = iro::recipe::Accumulate<Recipe>;
    using ScalePayload = iro::contract::ScalarDesc<iro::elem::f32, iro::dist::replicated>;
    static constexpr bool kTileSkip = !std::is_same_v<TileSkip, axp::intent::tile_skip::None>;
    using TileSkipHookOp = axp::level2::attention::TileSkipHook<Recipe, axp::subject::TileSkip, ExecGroup>;
    template<bool Enable, class = void>
    struct tile_skip_hook {
        using obligations = iro::util::type_list<>;
        using edges = iro::util::type_list<>;
    };
    template<bool Enable>
    struct tile_skip_hook<Enable, std::enable_if_t<Enable>> {
        using obligations = iro::util::type_list<TileSkipHookOp>;
        using edges = iro::util::type_list<>;
    };

    using QTileG = iro::contract::Tile<
        iro::contract::Shape<TileM, HeadDim>,
        ElemA,
        iro::contract::layout::RowMajor<HeadDim>,
        iro::contract::space::global,
        iro::contract::Align<16>
    >;

    using KTileG = iro::contract::Tile<
        iro::contract::Shape<HeadDim, TileN>,
        ElemB,
        iro::contract::layout::ColMajor<HeadDim>,
        iro::contract::space::global,
        iro::contract::Align<16>
    >;

    using VTileG = iro::contract::Tile<
        iro::contract::Shape<TileN, HeadDim>,
        ElemB,
        iro::contract::layout::ColMajor<TileN>,
        iro::contract::space::global,
        iro::contract::Align<16>
    >;

    using AutoQTma = axp::kit::detail::select_tma_t<LoadModeQ, CapT, QTileG, QSubj, detail::q_pipe_tag>;
    using AutoKTma = axp::kit::detail::select_tma_t<LoadModeK, CapT, KTileG, KSubj, detail::k_pipe_tag>;
    using AutoVTma = axp::kit::detail::select_tma_t<LoadModeV, CapT, VTileG, VSubj, detail::v_pipe_tag>;
    using QTmaT = std::conditional_t<!std::is_void_v<QTma>, QTma, AutoQTma>;
    using KTmaT = std::conditional_t<!std::is_void_v<KTma>, KTma, AutoKTma>;
    using VTmaT = std::conditional_t<!std::is_void_v<VTma>, VTma, AutoVTma>;

    using StageSwizzleQ = std::conditional_t<std::is_same_v<SwizzleAtomQ, axp::swizzle::None>, void, SwizzleAtomQ>;
    using StageSwizzleK = std::conditional_t<std::is_same_v<SwizzleAtomK, axp::swizzle::None>, void, SwizzleAtomK>;
    using StageSwizzleV = std::conditional_t<std::is_same_v<SwizzleAtomV, axp::swizzle::None>, void, SwizzleAtomV>;

    using QTileS = iro::contract::Tile<
        iro::contract::Shape<TileM, HeadDim>,
        ElemA,
        iro::contract::layout::RowMajor<HeadDim>,
        iro::contract::space::shared,
        iro::contract::Align<16>
    >;

    using KTileS = iro::contract::Tile<
        iro::contract::Shape<HeadDim, TileN>,
        ElemB,
        iro::contract::layout::ColMajor<HeadDim>,
        iro::contract::space::shared,
        iro::contract::Align<16>
    >;

    using VTileS = iro::contract::Tile<
        iro::contract::Shape<TileN, HeadDim>,
        ElemB,
        iro::contract::layout::ColMajor<TileN>,
        iro::contract::space::shared,
        iro::contract::Align<16>
    >;

    using PTileF32 = iro::contract::Tile<
        iro::contract::Shape<TileM, TileN>,
        iro::elem::f32,
        iro::contract::layout::RowMajor<TileN>,
        iro::contract::space::shared,
        iro::contract::Align<16>
    >;

    using PTileF16 = iro::contract::Tile<
        iro::contract::Shape<TileM, TileN>,
        ElemA,
        iro::contract::layout::RowMajor<TileN>,
        iro::contract::space::shared,
        iro::contract::Align<16>
    >;

    using PipeQ = iro::contract::res::smem_pipeline<
        detail::q_pipe_tag, Stages, QTileS::bytes, QTileS::align::bytes
    >;
    using PipeK = iro::contract::res::smem_pipeline<
        detail::k_pipe_tag, Stages, KTileS::bytes, KTileS::align::bytes
    >;
    using PipeV = iro::contract::res::smem_pipeline<
        detail::v_pipe_tag, Stages, VTileS::bytes, VTileS::align::bytes
    >;

    using SlotQ = iro::contract::res::slot_subject<PipeQ, SlotIdx>;
    using SlotK = iro::contract::res::slot_subject<PipeK, SlotIdx>;
    using SlotV = iro::contract::res::slot_subject<PipeV, SlotIdx>;

    using TileInQ = axp::level0::TileBoundaryIn<
        Recipe, QTileG, QSubj, iro::exec::block, iro::token::lifetime::block
    >;
    using TileInK = axp::level0::TileBoundaryIn<
        Recipe, KTileG, KSubj, iro::exec::block, iro::token::lifetime::block
    >;
    using TileInV = axp::level0::TileBoundaryIn<
        Recipe, VTileG, VSubj, iro::exec::block, iro::token::lifetime::block
    >;

    using QStageIssueExec = axp::level2::staging::tma_issue_exec_group_t<QTmaT>;
    using KStageIssueExec = axp::level2::staging::tma_issue_exec_group_t<KTmaT>;
    using VStageIssueExec = axp::level2::staging::tma_issue_exec_group_t<VTmaT>;

    using StageQ = axp::level2::staging::StageGmemToSmem<
        Recipe, QTileG, QTileS, QSubj, detail::q_pipe_tag, SlotQ,
        iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleQ, QTmaT,
        QStageIssueExec, CapT
    >;
    using StageK = axp::level2::staging::StageGmemToSmem<
        Recipe, KTileG, KTileS, KSubj, detail::k_pipe_tag, SlotK,
        iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleK, KTmaT,
        KStageIssueExec, CapT
    >;
    using StageV = axp::level2::staging::StageGmemToSmem<
        Recipe, VTileG, VTileS, VSubj, detail::v_pipe_tag, SlotV,
        iro::exec::block, iro::token::lifetime::block, Stages, StageSwizzleV, VTmaT,
        VStageIssueExec, CapT
    >;

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

    using QBarrierInitHelper = barrier_init<QTmaT>;
    using KBarrierInitHelper = barrier_init<KTmaT>;
    using VBarrierInitHelper = barrier_init<VTmaT>;
    using QBarrierInit = typename QBarrierInitHelper::init_type;
    using KBarrierInit = typename KBarrierInitHelper::init_type;
    using VBarrierInit = typename VBarrierInitHelper::init_type;
    static constexpr bool kHasTmaQ = !std::is_void_v<QBarrierInit>;
    static constexpr bool kHasTmaK = !std::is_void_v<KBarrierInit>;
    static constexpr bool kHasTmaV = !std::is_void_v<VBarrierInit>;

    struct IssueQ : StageQ::Issue {};
    struct IssueK : StageK::Issue {};
    struct IssueV : StageV::Issue {};

    struct WaitQ : StageQ::Wait {};
    struct WaitK : StageK::Wait {};
    struct WaitV : StageV::Wait {};

    struct MarkQ : StageQ::Mark {};
    struct MarkK : StageK::Mark {};
    struct MarkV : StageV::Mark {};

    struct ReleaseQ : StageQ::Release {};
    struct ReleaseK : StageK::Release {};
    struct ReleaseV : StageV::Release {};

    template<bool Enable, class Init, class Issue>
    struct barrier_edge {
        using edges = iro::util::type_list<>;
    };

    template<class Init, class Issue>
    struct barrier_edge<true, Init, Issue> {
        using edges = iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<Init, 0>, detail::in_port_t<Issue, 1>>
        >;
    };


    using QKShape = axp::protocol::compute::MmaShape<
        TileM, TileN, HeadDim,
        ElemA, ElemB, typename Recipe::acc,
        typename QTileS::layout, typename KTileS::layout
    >;

    using PVShape = axp::protocol::compute::MmaShape<
        TileM, HeadDim, TileN,
        ElemA, ElemB, typename Recipe::acc,
        typename PTileF16::layout, typename VTileS::layout
    >;

    using QKFrag = iro::contract::FragmentDesc<
        iro::contract::Shape<TileM, TileN>,
        typename Recipe::acc,
        iro::dist::accumulator,
        TileN / 2
    >;

    using OFrag = iro::contract::FragmentDesc<
        iro::contract::Shape<TileM, HeadDim>,
        typename Recipe::acc,
        iro::dist::accumulator,
        HeadDim / 2
    >;
    static constexpr int kBaseRegs = 48 + Stages * 4;
    using RegPressure = detail::reg_pressure_obligation<kBaseRegs, QKFrag, OFrag>;

    struct HoldQ : axp::level0::SlotAfter<
        Recipe, SlotQ, iro::exec::block, iro::token::lifetime::block, QTileS::bytes,
        QKFrag, detail::qk_frag_subj, ExecGroup, typename QKFrag::dist
    > {};
    struct HoldK : axp::level0::SlotAfter<
        Recipe, SlotK, iro::exec::block, iro::token::lifetime::block, KTileS::bytes,
        QKFrag, detail::qk_frag_subj, ExecGroup, typename QKFrag::dist
    > {};
    struct HoldV : axp::level0::SlotAfter<
        Recipe, SlotV, iro::exec::block, iro::token::lifetime::block, VTileS::bytes,
        OFrag, detail::pv_frag_subj, ExecGroup, typename OFrag::dist
    > {};

    static constexpr int cast_vec_bytes = Recipe::vec_bytes;
    static constexpr int cast_out_vec_bytes = (cast_vec_bytes * PTileF16::elem::bytes) / PTileF32::elem::bytes;
    static_assert(cast_vec_bytes == 8 || cast_vec_bytes == 16, "AttentionTile: Recipe::vec_bytes must be 8 or 16");
    static_assert(cast_out_vec_bytes == 4 || cast_out_vec_bytes == 8 || cast_out_vec_bytes == 16,
                  "AttentionTile: cast output vec bytes must be 4/8/16");

    template<class R, class Enable = void>
    struct weight_recipe_impl;

    template<class R>
    struct weight_recipe_impl<R, std::void_t<typename R::scale>> {
        using type = iro::recipe::Precision<
            ElemA,
            typename R::acc,
            typename R::out,
            cast_out_vec_bytes,
            typename R::math,
            typename R::fp8_policy,
            typename R::scale,
            R::scale_vec
        >;
    };

    template<class R>
    struct weight_recipe_impl<R, std::void_t<typename R::scale_a>> {
        using type = iro::recipe::Precision<
            ElemA,
            typename R::acc,
            typename R::out,
            cast_out_vec_bytes,
            typename R::math,
            typename R::fp8_policy,
            typename R::scale_a,
            R::scale_vec_a
        >;
    };

    using WeightRecipe = typename weight_recipe_impl<Recipe>::type;

    using QK = axp::level2::Matmul<
        Recipe,
        QKShape,
        QTileS,
        KTileS,
        QKFrag,
        SlotQ,
        SlotK,
        detail::qk_frag_subj,
        ExecGroup,
        detail::wgmma_tag,
        CapT
    >;

    using Softmax = axp::level2::attention::WarpSoftmaxState<
        AccRecipe,
        QKFrag,
        detail::qk_frag_subj,
        detail::weights_frag_subj,
        detail::tile_state_subj,
        ExecGroup,
        CapT
    >;

    using Combine = axp::level2::attention::SoftmaxStateCombine<
        AccRecipe,
        ExecGroup,
        OldStateSubj,
        detail::tile_state_subj,
        detail::combined_state_subj
    >;

    using Rescale = axp::level2::attention::RescaleAccumulator<
        AccRecipe,
        OFrag,
        AccSubj,
        OldStateSubj,
        detail::combined_state_subj,
        ExecGroup
    >;

    using StateCopy = axp::level2::attention::SoftmaxStateCopy<
        AccRecipe,
        detail::combined_state_subj,
        OutStateSubj,
        ExecGroup,
        CapT
    >;

    using Scale = axp::level2::attention::SoftmaxStateScale<
        AccRecipe,
        detail::tile_state_subj,
        detail::combined_state_subj,
        detail::scale_subj,
        ExecGroup,
        CapT
    >;

    using ScaleFrag = axp::level0::FragmentBroadcast<
        AccRecipe,
        QKFrag,
        ScalePayload,
        detail::scale_subj,
        detail::scale_frag_subj,
        ExecGroup
    >;

    using ScaleWeights = axp::level0::Mul<
        AccRecipe,
        QKFrag,
        detail::weights_frag_subj,
        detail::scale_frag_subj,
        detail::weights_scaled_subj,
        ExecGroup
    >;

    using StoreWeights = axp::level0::FragmentToSharedTile<
        AccRecipe,
        QKFrag,
        PTileF32,
        detail::weights_scaled_subj,
        detail::weights_f32_subj,
        ExecGroup,
        iro::token::lifetime::warp
    >;

    using CastWeights = axp::level0::CastTile<
        AccRecipe,
        WeightRecipe,
        PTileF32,
        PTileF16,
        detail::weights_f32_subj,
        detail::weights_f16_subj,
        ExecGroup,
        cast_vec_bytes
    >;

    using PV = axp::level0::WarpMmaShared<
        Recipe,
        PVShape,
        PTileF16,
        VTileS,
        OFrag,
        detail::weights_f16_subj,
        SlotV,
        detail::pv_frag_subj
    >;

    using Add = axp::level0::Add<
        AccRecipe,
        OFrag,
        AccSubj,
        detail::pv_frag_subj,
        AccSubj,
        ExecGroup
    >;

    using barrier_obligations = iro::util::concat_t<
        typename QBarrierInitHelper::obligations,
        iro::util::concat_t<typename KBarrierInitHelper::obligations, typename VBarrierInitHelper::obligations>
    >;

    using obligations = iro::util::concat_t<
        barrier_obligations,
        typename tile_skip_hook<kTileSkip>::obligations,
        iro::util::type_list<
        RegPressure,
        TileInQ,
        TileInK,
        TileInV,
        IssueQ,
        IssueK,
        IssueV,
        WaitQ,
        WaitK,
        WaitV,
        QK,
        Softmax,
        Combine,
        Rescale,
        StateCopy,
        Scale,
        ScaleFrag,
        ScaleWeights,
        StoreWeights,
        CastWeights,
        PV,
        Add,
        HoldQ,
        HoldK,
        HoldV,
        MarkQ,
        MarkK,
        MarkV,
        ReleaseQ,
        ReleaseK,
        ReleaseV
    >
    >;

    using stage_q_edges = detail::stage_issue_wait_edges_t<IssueQ, WaitQ, kHasTmaQ>;
    using stage_k_edges = detail::stage_issue_wait_edges_t<IssueK, WaitK, kHasTmaK>;
    using stage_v_edges = detail::stage_issue_wait_edges_t<IssueV, WaitV, kHasTmaV>;

    using barrier_edges = iro::util::concat_t<
        typename barrier_edge<kHasTmaQ, QBarrierInit, IssueQ>::edges,
        iro::util::concat_t<
            typename barrier_edge<kHasTmaK, KBarrierInit, IssueK>::edges,
            typename barrier_edge<kHasTmaV, VBarrierInit, IssueV>::edges
        >
    >;

    using edges = iro::util::concat_t<
        typename tile_skip_hook<kTileSkip>::edges,
        iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<TileInQ, 0>, detail::in_port_t<IssueQ, 0>>,
            iro::compose::Edge<detail::out_port_t<TileInK, 0>, detail::in_port_t<IssueK, 0>>,
            iro::compose::Edge<detail::out_port_t<TileInV, 0>, detail::in_port_t<IssueV, 0>>
        >,
        barrier_edges,
        stage_q_edges,
        stage_k_edges,
        stage_v_edges,
        iro::util::type_list<
            iro::compose::Edge<detail::out_port_t<WaitQ, 0>, detail::in_port_t<QK, 0>>,
            iro::compose::Edge<detail::out_port_t<WaitK, 0>, detail::in_port_t<QK, 1>>,
            iro::compose::Edge<detail::out_port_t<WaitV, 0>, detail::in_port_t<PV, 1>>,
            iro::compose::Edge<detail::out_port_t<WaitQ, 1>, detail::in_port_t<HoldQ, 0>>,
            iro::compose::Edge<detail::out_port_t<WaitK, 1>, detail::in_port_t<HoldK, 0>>,
            iro::compose::Edge<detail::out_port_t<WaitV, 1>, detail::in_port_t<HoldV, 0>>,
            iro::compose::Edge<detail::out_port_t<QK, 0>, detail::in_port_t<HoldQ, 1>>,
            iro::compose::Edge<detail::out_port_t<QK, 0>, detail::in_port_t<HoldK, 1>>,
            iro::compose::Edge<detail::out_port_t<PV, 0>, detail::in_port_t<HoldV, 1>>,
            iro::compose::Edge<detail::out_port_t<HoldQ, 0>, detail::in_port_t<MarkQ, 0>>,
            iro::compose::Edge<detail::out_port_t<HoldK, 0>, detail::in_port_t<MarkK, 0>>,
            iro::compose::Edge<detail::out_port_t<HoldV, 0>, detail::in_port_t<MarkV, 0>>,
            iro::compose::Edge<detail::out_port_t<MarkQ, 0>, detail::in_port_t<ReleaseQ, 0>>,
            iro::compose::Edge<detail::out_port_t<MarkK, 0>, detail::in_port_t<ReleaseK, 0>>,
            iro::compose::Edge<detail::out_port_t<MarkV, 0>, detail::in_port_t<ReleaseV, 0>>,
            iro::compose::Edge<detail::out_port_t<QK, 0>, detail::in_port_t<Softmax, 0>>,
            iro::compose::Edge<detail::out_port_t<Softmax, 1>, detail::in_port_t<Combine, 1>>,
            iro::compose::Edge<detail::out_port_t<Softmax, 1>, detail::in_port_t<Scale, 0>>,
            iro::compose::Edge<detail::out_port_t<Combine, 0>, detail::in_port_t<Rescale, 2>>,
            iro::compose::Edge<detail::out_port_t<Combine, 0>, detail::in_port_t<Scale, 1>>,
            iro::compose::Edge<detail::out_port_t<Combine, 0>, detail::in_port_t<StateCopy, 0>>,
            iro::compose::Edge<detail::out_port_t<Scale, 0>, detail::in_port_t<ScaleFrag, 0>>,
            iro::compose::Edge<detail::out_port_t<ScaleFrag, 0>, detail::in_port_t<ScaleWeights, 1>>,
            iro::compose::Edge<detail::out_port_t<Softmax, 0>, detail::in_port_t<ScaleWeights, 0>>,
            iro::compose::Edge<detail::out_port_t<ScaleWeights, 0>, detail::in_port_t<StoreWeights, 0>>,
            iro::compose::Edge<detail::out_port_t<StoreWeights, 0>, detail::in_port_t<CastWeights, 0>>,
            iro::compose::Edge<detail::out_port_t<CastWeights, 0>, detail::in_port_t<PV, 0>>,
            iro::compose::Edge<detail::out_port_t<Rescale, 0>, detail::in_port_t<Add, 0>>,
            iro::compose::Edge<detail::out_port_t<PV, 0>, detail::in_port_t<Add, 1>>
        >
    >;

    using type = axp::level3::detail::make_composition_t<obligations, edges, iro::profile::BudgetMax, CapT>;
};

// AttentionTile: WGMMA path wrapped at L3 to enforce reg_pressure and preserve boundary obligations.
template<
    class Recipe,
    int TileM, int TileN, int HeadDim,
    int Stages,
    class QSubj, class KSubj, class VSubj,
    class AccSubj, class OldStateSubj, class OutStateSubj,
    class MemoryPatternQ, class MemoryPatternK, class MemoryPatternV,
    class LoadModeQ, class LoadModeK, class LoadModeV,
    class Schedule,
    class TileSkip,
    class QTma = void, class KTma = void, class VTma = void,
    class CapT = axp::target_cap>
struct AttentionTileWgmmaImpl {
    static_assert(std::is_same_v<typename Recipe::acc, iro::elem::f32>,
                  "AttentionTileWgmma: Recipe::acc must be f32");
    static_assert(Stages >= 2 && Stages <= 4, "AttentionTileWgmma: Stages must be 2-4");

    using QKFrag = iro::contract::FragmentDesc<
        iro::contract::Shape<TileM, TileN>,
        typename Recipe::acc,
        iro::dist::accumulator,
        TileN / 2
    >;

    using OFrag = iro::contract::FragmentDesc<
        iro::contract::Shape<TileM, HeadDim>,
        typename Recipe::acc,
        iro::dist::accumulator,
        HeadDim / 2
    >;

    static constexpr int kBaseRegs = 64 + Stages * 8;
    using RegPressure = detail::reg_pressure_obligation<kBaseRegs, QKFrag, OFrag>;

    using Core = axp::level2::attention::AttentionWgmma<
        Recipe, TileM, TileN, HeadDim, Stages,
        QSubj, KSubj, VSubj,
        AccSubj, OldStateSubj, OutStateSubj,
        MemoryPatternQ, MemoryPatternK, MemoryPatternV,
        LoadModeQ, LoadModeK, LoadModeV,
        Schedule,
        TileSkip,
        QTma, KTma, VTma, CapT
    >;

    using RegComp = axp::level3::detail::as_composition_t<RegPressure, CapT>;
    using type = iro::compose::join_t<Core, RegComp>;
};

} // namespace axp::level3::attention

namespace axp::level3 {

template<class Recipe, int TileQ, int TileK, int TileV, int HeadDim, int Stages, int SlotIdx,
         class QSubj, class KSubj, class VSubj,
         class AccSubj, class OldStateSubj, class OutStateSubj,
         class MemoryPatternQ = axp::intent::memory_pattern::Optimized,
         class MemoryPatternK = axp::intent::memory_pattern::Optimized,
         class MemoryPatternV = axp::intent::memory_pattern::Optimized,
         class LoadModeQ = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeK = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeV = axp::intent::load_mode::AsyncPrefetch,
         class Schedule = axp::intent::schedule::Pipelined,
         class TileSkip = axp::intent::tile_skip::None>
struct AttentionTileConfig {
    using recipe = Recipe;
    static constexpr int tile_q = TileQ;
    static constexpr int tile_k = TileK;
    static constexpr int tile_v = TileV;
    static constexpr int head_dim = HeadDim;
    static constexpr int stages = Stages;
    static constexpr int slot_idx = SlotIdx;
    using q_subj = QSubj;
    using k_subj = KSubj;
    using v_subj = VSubj;
    using acc_subj = AccSubj;
    using old_state_subj = OldStateSubj;
    using out_state_subj = OutStateSubj;
    using memory_pattern_q = MemoryPatternQ;
    using memory_pattern_k = MemoryPatternK;
    using memory_pattern_v = MemoryPatternV;
    using load_mode_q = LoadModeQ;
    using load_mode_k = LoadModeK;
    using load_mode_v = LoadModeV;
    using schedule = Schedule;
    using tile_skip = TileSkip;
};

template<class Recipe, int TileQ, int TileK, int TileV, int HeadDim, int Stages, int SlotIdx,
         class QSubj, class KSubj, class VSubj,
         class AccSubj, class OldStateSubj, class OutStateSubj,
         class MemoryPatternQ = axp::intent::memory_pattern::Optimized,
         class MemoryPatternK = axp::intent::memory_pattern::Optimized,
         class MemoryPatternV = axp::intent::memory_pattern::Optimized,
         class LoadModeQ = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeK = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeV = axp::intent::load_mode::AsyncPrefetch,
         class Schedule = axp::intent::schedule::Pipelined,
         class TileSkip = axp::intent::tile_skip::None,
         class QTma = void, class KTma = void, class VTma = void>
struct AttentionTileConfigSm90Multicast
    : AttentionTileConfig<Recipe, TileQ, TileK, TileV, HeadDim, Stages, SlotIdx,
                          QSubj, KSubj, VSubj, AccSubj, OldStateSubj, OutStateSubj,
                          MemoryPatternQ, MemoryPatternK, MemoryPatternV,
                          LoadModeQ, LoadModeK, LoadModeV,
                          Schedule, TileSkip> {
    static_assert(std::is_same_v<LoadModeQ, axp::intent::load_mode::AsyncPrefetch>,
                  "AttentionTileConfigSm90Multicast requires load_mode_q = AsyncPrefetch");
    static_assert(std::is_same_v<LoadModeK, axp::intent::load_mode::AsyncPrefetch>,
                  "AttentionTileConfigSm90Multicast requires load_mode_k = AsyncPrefetch");
    static_assert(std::is_same_v<LoadModeV, axp::intent::load_mode::AsyncPrefetch>,
                  "AttentionTileConfigSm90Multicast requires load_mode_v = AsyncPrefetch");
    static_assert(axp::level2::staging::tma_multicast_traits<QTma>::valid,
                  "AttentionTileConfigSm90Multicast requires QTma multicast config");
    static_assert(axp::level2::staging::tma_multicast_traits<KTma>::valid,
                  "AttentionTileConfigSm90Multicast requires KTma multicast config");
    static_assert(axp::level2::staging::tma_multicast_traits<VTma>::valid,
                  "AttentionTileConfigSm90Multicast requires VTma multicast config");
    using q_tma = QTma;
    using k_tma = KTma;
    using v_tma = VTma;
};

namespace detail {
template<class Config, class CapT, int Index, class = void>
struct config_tma_or_void {
    using type = void;
};

template<class Config, class CapT>
struct config_tma_or_void<Config, CapT, 0, std::void_t<typename Config::q_tma>> {
    using type = typename Config::q_tma;
};

template<class Config, class CapT>
struct config_tma_or_void<Config, CapT, 1, std::void_t<typename Config::k_tma>> {
    using type = typename Config::k_tma;
};

template<class Config, class CapT>
struct config_tma_or_void<Config, CapT, 2, std::void_t<typename Config::v_tma>> {
    using type = typename Config::v_tma;
};

template<class Config, class CapT, int Index>
using config_tma_or_void_t = typename config_tma_or_void<Config, CapT, Index>::type;
} // namespace detail

template<class Config, class CapT = axp::target_cap>
using AttentionTile = registry::Select<registry::AttentionTilePattern<
    typename Config::recipe,
    Config::tile_q, Config::tile_k, Config::tile_v,
    Config::head_dim, Config::stages, Config::slot_idx,
    typename Config::q_subj, typename Config::k_subj, typename Config::v_subj,
    typename Config::acc_subj, typename Config::old_state_subj, typename Config::out_state_subj,
    typename Config::memory_pattern_q,
    typename Config::memory_pattern_k,
    typename Config::memory_pattern_v,
    typename Config::load_mode_q,
    typename Config::load_mode_k,
    typename Config::load_mode_v,
    typename Config::schedule,
    typename Config::tile_skip,
    detail::config_tma_or_void_t<Config, CapT, 0>,
    detail::config_tma_or_void_t<Config, CapT, 1>,
    detail::config_tma_or_void_t<Config, CapT, 2>>, CapT>;

} // namespace axp::level3

#if defined(AXP_LIBRARY_BUILD)
namespace axp::level3::registry {

template<class Recipe, int TileQ, int TileK, int TileV, int HeadDim, int Stages, int SlotIdx,
         class QSubj, class KSubj, class VSubj,
         class AccSubj, class OldStateSubj, class OutStateSubj,
         class MemoryPatternQ, class MemoryPatternK, class MemoryPatternV,
         class LoadModeQ, class LoadModeK, class LoadModeV,
         class Schedule,
         class TileSkip,
         class QTma, class KTma, class VTma, class Cap>
struct resolve_impl<AttentionTilePattern<Recipe, TileQ, TileK, TileV, HeadDim, Stages, SlotIdx,
                                         QSubj, KSubj, VSubj, AccSubj, OldStateSubj, OutStateSubj,
                                         MemoryPatternQ, MemoryPatternK, MemoryPatternV,
                                         LoadModeQ, LoadModeK, LoadModeV, Schedule,
                                         TileSkip,
                                         QTma, KTma, VTma>, Cap,
                   std::enable_if_t<!Cap::has_wgmma>> {
    static constexpr bool supported = true;
    static_assert(TileK == TileV, "AttentionTile: TileK must equal TileV");
    using type = typename axp::level3::attention::AttentionTileImpl<
        Recipe, TileQ, TileK, HeadDim, Stages, SlotIdx, QSubj, KSubj, VSubj,
        AccSubj, OldStateSubj, OutStateSubj,
        MemoryPatternQ, MemoryPatternK, MemoryPatternV,
        LoadModeQ, LoadModeK, LoadModeV,
        Schedule,
        TileSkip,
        QTma, KTma, VTma, Cap
    >::type;
};

template<class Recipe, int TileQ, int TileK, int TileV, int HeadDim, int Stages, int SlotIdx,
         class QSubj, class KSubj, class VSubj,
         class AccSubj, class OldStateSubj, class OutStateSubj,
         class MemoryPatternQ, class MemoryPatternK, class MemoryPatternV,
         class LoadModeQ, class LoadModeK, class LoadModeV,
         class Schedule,
         class TileSkip,
         class QTma, class KTma, class VTma, class Cap>
struct resolve_impl<AttentionTilePattern<Recipe, TileQ, TileK, TileV, HeadDim, Stages, SlotIdx,
                                         QSubj, KSubj, VSubj, AccSubj, OldStateSubj, OutStateSubj,
                                         MemoryPatternQ, MemoryPatternK, MemoryPatternV,
                                         LoadModeQ, LoadModeK, LoadModeV, Schedule,
                                         TileSkip,
                                         QTma, KTma, VTma>, Cap,
                   std::enable_if_t<Cap::has_wgmma>> {
    static constexpr bool supported = true;
    static_assert(TileK == TileV, "AttentionTile: TileK must equal TileV");
    using type = typename axp::level3::attention::AttentionTileWgmmaImpl<
        Recipe, TileQ, TileK, HeadDim, Stages,
        QSubj, KSubj, VSubj, AccSubj, OldStateSubj, OutStateSubj,
        MemoryPatternQ, MemoryPatternK, MemoryPatternV,
        LoadModeQ, LoadModeK, LoadModeV,
        Schedule,
        TileSkip,
        QTma, KTma, VTma, Cap
    >::type;
};

} // namespace axp::level3::registry
#endif // AXP_LIBRARY_BUILD
