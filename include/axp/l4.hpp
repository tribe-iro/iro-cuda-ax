#pragma once

#include "prelude.hpp"
#include "bundles/checklist.hpp"

namespace axp::l4 {

namespace detail {

template<class... Subjects>
consteval bool subjects_follow_policy() {
    return (axp::bundle::check::subject_follows_derivation_policy<Subjects>() && ...);
}

} // namespace detail

// L4 intent patterns (portable descriptions only).
template<class Recipe,
         int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj = axp::subject::MatrixA,
         class BSubj = axp::subject::MatrixB,
         class CSubj = axp::subject::MatrixC,
         class WgmmaSubj = axp::subject::Accumulator,
         class MemoryPatternA = axp::intent::memory_pattern::Optimized,
         class MemoryPatternB = axp::intent::memory_pattern::Optimized,
         class LoadModeA = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeB = axp::intent::load_mode::AsyncPrefetch,
         class Schedule = axp::intent::schedule::Pipelined,
         class ScaleASubj = iro::contract::subject::global,
         class ScaleBSubj = iro::contract::subject::global>
struct GemmPattern {
    static_assert(detail::subjects_follow_policy<ASubj, BSubj, CSubj, WgmmaSubj, ScaleASubj, ScaleBSubj>(),
                  "GemmPattern: subjects must follow canonical derivation policy");
};

template<class Recipe,
         int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class EpiloguePolicy,
         class ASubj = axp::subject::MatrixA,
         class BSubj = axp::subject::MatrixB,
         class CSubj = axp::subject::MatrixC,
         class WgmmaSubj = axp::subject::Accumulator,
         class MemoryPatternA = axp::intent::memory_pattern::Optimized,
         class MemoryPatternB = axp::intent::memory_pattern::Optimized,
         class LoadModeA = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeB = axp::intent::load_mode::AsyncPrefetch,
         class Schedule = axp::intent::schedule::Pipelined,
         class ScaleASubj = iro::contract::subject::global,
         class ScaleBSubj = iro::contract::subject::global>
struct GemmFusedPattern {
    static_assert(detail::subjects_follow_policy<ASubj, BSubj, CSubj, WgmmaSubj, ScaleASubj, ScaleBSubj>(),
                  "GemmFusedPattern: subjects must follow canonical derivation policy");
};

template<class Recipe,
         int TileQ, int TileK, int TileV, int HeadDim, int Stages, int SlotIdx,
         class QSubj = axp::subject::AttentionQ,
         class KSubj = axp::subject::AttentionK,
         class VSubj = axp::subject::AttentionV,
         class AccSubj = axp::subject::Accumulator,
         class OldStateSubj = axp::subject::AttentionS,
         class OutStateSubj = axp::subject::wire<axp::tag::S, 1>,
         class MemoryPatternQ = axp::intent::memory_pattern::Optimized,
         class MemoryPatternK = axp::intent::memory_pattern::Optimized,
         class MemoryPatternV = axp::intent::memory_pattern::Optimized,
         class LoadModeQ = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeK = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeV = axp::intent::load_mode::AsyncPrefetch,
         class Schedule = axp::intent::schedule::Pipelined,
         class TileSkip = axp::intent::tile_skip::None>
struct AttentionPattern {
    static_assert(detail::subjects_follow_policy<QSubj, KSubj, VSubj, AccSubj, OldStateSubj, OutStateSubj>(),
                  "AttentionPattern: subjects must follow canonical derivation policy");
};

template<class Recipe, int ElementsPerThread,
         class InSubj = axp::subject::MatrixA,
         class OutSubj = axp::subject::Output>
struct SoftmaxRowPattern {
    static_assert(detail::subjects_follow_policy<InSubj, OutSubj>(),
                  "SoftmaxRowPattern: subjects must follow canonical derivation policy");
};

template<class Recipe, int TileRows, int TileCols,
         class InSubj = axp::subject::MatrixA,
         class OutSubj = axp::subject::Output>
struct ElementwisePattern {
    static_assert(detail::subjects_follow_policy<InSubj, OutSubj>(),
                  "ElementwisePattern: subjects must follow canonical derivation policy");
};

template<class Recipe, int TileRows, int TileCols,
         class InSubj = axp::subject::MatrixA,
         class OutSubj = axp::subject::Output,
         class GammaSubj = axp::subject::wire<axp::tag::B, 1>,
         class BetaSubj = axp::subject::wire<axp::tag::C, 1>,
         class EpsSubj = axp::subject::wire<axp::tag::S, 2>>
struct LayerNormPattern {
    static_assert(detail::subjects_follow_policy<InSubj, OutSubj, GammaSubj, BetaSubj, EpsSubj>(),
                  "LayerNormPattern: subjects must follow canonical derivation policy");
};

template<class Recipe, int TileRows, int TileCols,
         class InSubj = axp::subject::MatrixA,
         class OutSubj = axp::subject::Output,
         class WeightSubj = axp::subject::wire<axp::tag::B, 2>,
         class EpsSubj = axp::subject::wire<axp::tag::S, 2>>
struct RMSNormPattern {
    static_assert(detail::subjects_follow_policy<InSubj, OutSubj, WeightSubj, EpsSubj>(),
                  "RMSNormPattern: subjects must follow canonical derivation policy");
};

template<class Recipe, class ValuePayload, class IndexPayload, class SharedTile, class OutTile,
         class ValueSubj, class IndexSubj, class SharedSubj, class OutValSubj, class OutSubj,
         class ExecGroup = iro::exec::block>
struct HistogramPattern {
    static_assert(detail::subjects_follow_policy<ValueSubj, IndexSubj, SharedSubj, OutValSubj, OutSubj>(),
                  "HistogramPattern: subjects must follow canonical derivation policy");
};

template<class Recipe, int TileElems, class InSubj, class OutSubj>
struct SortPattern {
    static_assert(detail::subjects_follow_policy<InSubj, OutSubj>(),
                  "SortPattern: subjects must follow canonical derivation policy");
};

template<class Recipe, class ValuePayload, class IndexPayload, class StateTile,
         class ValueSubj, class IndexSubj, class StateSubj, class AtomicOutSubj,
         class DependEventTag, class PhaseTag, class DoneEventTag,
         class ExecGroup = iro::exec::block>
struct StreamingMicrobatchPattern {
    static_assert(detail::subjects_follow_policy<ValueSubj, IndexSubj, StateSubj, AtomicOutSubj>(),
                  "StreamingMicrobatchPattern: subjects must follow canonical derivation policy");
};

template<class Recipe, class InTile, class OutTile, class GatherPayload, class IndexPayload,
         class InSubj, class IndexSubj, class GatherSubj, class OutSubj, class EmitEventTag,
         int SegmentWidth, class ExecGroup = iro::exec::warp>
struct ScientificSparseSegmentedPattern {
    static_assert(detail::subjects_follow_policy<InSubj, IndexSubj, GatherSubj, OutSubj>(),
                  "ScientificSparseSegmentedPattern: subjects must follow canonical derivation policy");
};

template<class Recipe, class InTile, class OutTile, class TileSubj,
         class SwizzleAtom, class EmitEventTag, class ExecGroup = iro::exec::block>
struct ScientificSwizzlePattern {
    static_assert(detail::subjects_follow_policy<TileSubj>(),
                  "ScientificSwizzlePattern: subjects must follow canonical derivation policy");
};

// Ergonomic config wrappers for the highest-arity pattern families.
template<class Recipe,
         int BlockM, int BlockN, int BlockK, int Stages, int KTiles,
         class ASubj = axp::subject::MatrixA,
         class BSubj = axp::subject::MatrixB,
         class CSubj = axp::subject::MatrixC,
         class WgmmaSubj = axp::subject::Accumulator,
         class MemoryPatternA = axp::intent::memory_pattern::Optimized,
         class MemoryPatternB = axp::intent::memory_pattern::Optimized,
         class LoadModeA = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeB = axp::intent::load_mode::AsyncPrefetch,
         class Schedule = axp::intent::schedule::Pipelined,
         class ScaleASubj = iro::contract::subject::global,
         class ScaleBSubj = iro::contract::subject::global>
struct GemmConfig {
    using recipe = Recipe;
    static constexpr int block_m = BlockM;
    static constexpr int block_n = BlockN;
    static constexpr int block_k = BlockK;
    static constexpr int stages = Stages;
    static constexpr int k_tiles = KTiles;
    using a_subject = ASubj;
    using b_subject = BSubj;
    using c_subject = CSubj;
    using wgmma_subject = WgmmaSubj;
    using memory_pattern_a = MemoryPatternA;
    using memory_pattern_b = MemoryPatternB;
    using load_mode_a = LoadModeA;
    using load_mode_b = LoadModeB;
    using schedule = Schedule;
    using scale_a_subject = ScaleASubj;
    using scale_b_subject = ScaleBSubj;
};

template<class Config>
concept GemmConfigLike = requires {
    typename Config::recipe;
    typename Config::a_subject;
    typename Config::b_subject;
    typename Config::c_subject;
    typename Config::wgmma_subject;
    typename Config::memory_pattern_a;
    typename Config::memory_pattern_b;
    typename Config::load_mode_a;
    typename Config::load_mode_b;
    typename Config::schedule;
    typename Config::scale_a_subject;
    typename Config::scale_b_subject;
    { Config::block_m } -> std::convertible_to<int>;
    { Config::block_n } -> std::convertible_to<int>;
    { Config::block_k } -> std::convertible_to<int>;
    { Config::stages } -> std::convertible_to<int>;
    { Config::k_tiles } -> std::convertible_to<int>;
};

template<class Config>
struct GemmFromConfig {
    static_assert(GemmConfigLike<Config>,
                  "axp::l4::GemmFromConfig: Config must define recipe, shape/stage ints, subjects, "
                  "memory/load intent, schedule, and scale subjects.");
    using type = GemmPattern<
        typename Config::recipe,
        Config::block_m, Config::block_n, Config::block_k, Config::stages, Config::k_tiles,
        typename Config::a_subject, typename Config::b_subject, typename Config::c_subject, typename Config::wgmma_subject,
        typename Config::memory_pattern_a, typename Config::memory_pattern_b,
        typename Config::load_mode_a, typename Config::load_mode_b,
        typename Config::schedule,
        typename Config::scale_a_subject, typename Config::scale_b_subject
    >;
};

template<class Config>
using GemmPatternT = typename GemmFromConfig<Config>::type;

template<class Recipe,
         int TileQ, int TileK, int TileV, int HeadDim, int Stages, int SlotIdx,
         class QSubj = axp::subject::AttentionQ,
         class KSubj = axp::subject::AttentionK,
         class VSubj = axp::subject::AttentionV,
         class AccSubj = axp::subject::Accumulator,
         class OldStateSubj = axp::subject::AttentionS,
         class OutStateSubj = axp::subject::wire<axp::tag::S, 1>,
         class MemoryPatternQ = axp::intent::memory_pattern::Optimized,
         class MemoryPatternK = axp::intent::memory_pattern::Optimized,
         class MemoryPatternV = axp::intent::memory_pattern::Optimized,
         class LoadModeQ = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeK = axp::intent::load_mode::AsyncPrefetch,
         class LoadModeV = axp::intent::load_mode::AsyncPrefetch,
         class Schedule = axp::intent::schedule::Pipelined,
         class TileSkip = axp::intent::tile_skip::None>
struct AttentionConfig {
    using recipe = Recipe;
    static constexpr int tile_q = TileQ;
    static constexpr int tile_k = TileK;
    static constexpr int tile_v = TileV;
    static constexpr int head_dim = HeadDim;
    static constexpr int stages = Stages;
    static constexpr int slot_idx = SlotIdx;
    using q_subject = QSubj;
    using k_subject = KSubj;
    using v_subject = VSubj;
    using acc_subject = AccSubj;
    using old_state_subject = OldStateSubj;
    using out_state_subject = OutStateSubj;
    using memory_pattern_q = MemoryPatternQ;
    using memory_pattern_k = MemoryPatternK;
    using memory_pattern_v = MemoryPatternV;
    using load_mode_q = LoadModeQ;
    using load_mode_k = LoadModeK;
    using load_mode_v = LoadModeV;
    using schedule = Schedule;
    using tile_skip = TileSkip;
};

template<class Config>
concept AttentionConfigLike = requires {
    typename Config::recipe;
    typename Config::q_subject;
    typename Config::k_subject;
    typename Config::v_subject;
    typename Config::acc_subject;
    typename Config::old_state_subject;
    typename Config::out_state_subject;
    typename Config::memory_pattern_q;
    typename Config::memory_pattern_k;
    typename Config::memory_pattern_v;
    typename Config::load_mode_q;
    typename Config::load_mode_k;
    typename Config::load_mode_v;
    typename Config::schedule;
    typename Config::tile_skip;
    { Config::tile_q } -> std::convertible_to<int>;
    { Config::tile_k } -> std::convertible_to<int>;
    { Config::tile_v } -> std::convertible_to<int>;
    { Config::head_dim } -> std::convertible_to<int>;
    { Config::stages } -> std::convertible_to<int>;
    { Config::slot_idx } -> std::convertible_to<int>;
};

template<class Config>
struct AttentionFromConfig {
    static_assert(AttentionConfigLike<Config>,
                  "axp::l4::AttentionFromConfig: Config must define recipe, tile/head/stage ints, "
                  "subjects, memory/load intent, schedule, and tile_skip.");
    using type = AttentionPattern<
        typename Config::recipe,
        Config::tile_q, Config::tile_k, Config::tile_v, Config::head_dim, Config::stages, Config::slot_idx,
        typename Config::q_subject, typename Config::k_subject, typename Config::v_subject,
        typename Config::acc_subject, typename Config::old_state_subject, typename Config::out_state_subject,
        typename Config::memory_pattern_q, typename Config::memory_pattern_k, typename Config::memory_pattern_v,
        typename Config::load_mode_q, typename Config::load_mode_k, typename Config::load_mode_v,
        typename Config::schedule, typename Config::tile_skip
    >;
};

template<class Config>
using AttentionPatternT = typename AttentionFromConfig<Config>::type;

template<class Recipe, class ValuePayload, class IndexPayload, class StateTile,
         class ValueSubj, class IndexSubj, class StateSubj, class AtomicOutSubj,
         class DependEventTag, class PhaseTag, class DoneEventTag,
         class ExecGroup = iro::exec::block>
struct StreamingMicrobatchConfig {
    using recipe = Recipe;
    using value_payload = ValuePayload;
    using index_payload = IndexPayload;
    using state_tile = StateTile;
    using value_subj = ValueSubj;
    using index_subj = IndexSubj;
    using state_subj = StateSubj;
    using atomic_out_subj = AtomicOutSubj;
    using depend_event_tag = DependEventTag;
    using phase_tag = PhaseTag;
    using done_event_tag = DoneEventTag;
    using exec_group = ExecGroup;
};

template<class Config>
concept StreamingMicrobatchConfigLike = requires {
    typename Config::recipe;
    typename Config::value_payload;
    typename Config::index_payload;
    typename Config::state_tile;
    typename Config::value_subj;
    typename Config::index_subj;
    typename Config::state_subj;
    typename Config::atomic_out_subj;
    typename Config::depend_event_tag;
    typename Config::phase_tag;
    typename Config::done_event_tag;
    typename Config::exec_group;
};

template<class Config>
struct StreamingMicrobatchFromConfig {
    static_assert(StreamingMicrobatchConfigLike<Config>,
                  "axp::l4::StreamingMicrobatchFromConfig: Config must define recipe, payload/tile,"
                  " subjects, event tags, and exec group.");
    using type = StreamingMicrobatchPattern<
        typename Config::recipe,
        typename Config::value_payload,
        typename Config::index_payload,
        typename Config::state_tile,
        typename Config::value_subj,
        typename Config::index_subj,
        typename Config::state_subj,
        typename Config::atomic_out_subj,
        typename Config::depend_event_tag,
        typename Config::phase_tag,
        typename Config::done_event_tag,
        typename Config::exec_group
    >;
};

template<class Config>
using StreamingMicrobatchPatternT = typename StreamingMicrobatchFromConfig<Config>::type;

template<class Recipe, class InTile, class OutTile, class GatherPayload, class IndexPayload,
         class InSubj, class IndexSubj, class GatherSubj, class OutSubj, class EmitEventTag,
         int SegmentWidth, class ExecGroup = iro::exec::warp>
struct ScientificSparseSegmentedConfig {
    using recipe = Recipe;
    using in_tile = InTile;
    using out_tile = OutTile;
    using gather_payload = GatherPayload;
    using index_payload = IndexPayload;
    using in_subj = InSubj;
    using index_subj = IndexSubj;
    using gather_subj = GatherSubj;
    using out_subj = OutSubj;
    using emit_event_tag = EmitEventTag;
    static constexpr int segment_width = SegmentWidth;
    using exec_group = ExecGroup;
};

template<class Config>
concept ScientificSparseSegmentedConfigLike = requires {
    typename Config::recipe;
    typename Config::in_tile;
    typename Config::out_tile;
    typename Config::gather_payload;
    typename Config::index_payload;
    typename Config::in_subj;
    typename Config::index_subj;
    typename Config::gather_subj;
    typename Config::out_subj;
    typename Config::emit_event_tag;
    typename Config::exec_group;
    { Config::segment_width } -> std::convertible_to<int>;
};

template<class Config>
struct ScientificSparseSegmentedFromConfig {
    static_assert(ScientificSparseSegmentedConfigLike<Config>,
                  "axp::l4::ScientificSparseSegmentedFromConfig: Config must define recipe, tiles/payloads,"
                  " subjects, event tag, segment width, and exec group.");
    using type = ScientificSparseSegmentedPattern<
        typename Config::recipe,
        typename Config::in_tile,
        typename Config::out_tile,
        typename Config::gather_payload,
        typename Config::index_payload,
        typename Config::in_subj,
        typename Config::index_subj,
        typename Config::gather_subj,
        typename Config::out_subj,
        typename Config::emit_event_tag,
        Config::segment_width,
        typename Config::exec_group
    >;
};

template<class Config>
using ScientificSparseSegmentedPatternT = typename ScientificSparseSegmentedFromConfig<Config>::type;

template<class Recipe, class InTile, class OutTile, class TileSubj,
         class SwizzleAtom, class EmitEventTag, class ExecGroup = iro::exec::block>
struct ScientificSwizzleConfig {
    using recipe = Recipe;
    using in_tile = InTile;
    using out_tile = OutTile;
    using tile_subj = TileSubj;
    using swizzle_atom = SwizzleAtom;
    using emit_event_tag = EmitEventTag;
    using exec_group = ExecGroup;
};

template<class Config>
concept ScientificSwizzleConfigLike = requires {
    typename Config::recipe;
    typename Config::in_tile;
    typename Config::out_tile;
    typename Config::tile_subj;
    typename Config::swizzle_atom;
    typename Config::emit_event_tag;
    typename Config::exec_group;
};

template<class Config>
struct ScientificSwizzleFromConfig {
    static_assert(ScientificSwizzleConfigLike<Config>,
                  "axp::l4::ScientificSwizzleFromConfig: Config must define recipe, tiles,"
                  " subject, swizzle atom, event tag, and exec group.");
    using type = ScientificSwizzlePattern<
        typename Config::recipe,
        typename Config::in_tile,
        typename Config::out_tile,
        typename Config::tile_subj,
        typename Config::swizzle_atom,
        typename Config::emit_event_tag,
        typename Config::exec_group
    >;
};

template<class Config>
using ScientificSwizzlePatternT = typename ScientificSwizzleFromConfig<Config>::type;

#include "l4/preset/gemm.hpp"
#include "l4/preset/attention.hpp"
#include "l4/preset/streaming.hpp"
#include "l4/preset/scientific.hpp"
#include "l4/preset/elementwise_norm_sort_hist.hpp"

namespace manifest {

// Explicit deterministic tie-break key per public L4 preset.
template<class Pattern>
struct tie_break_key;

template<class Pattern, class = void>
struct has_tie_break_key : std::false_type {};

template<class Pattern>
struct has_tie_break_key<Pattern, std::void_t<decltype(tie_break_key<Pattern>::value)>> : std::true_type {};

template<class Pattern>
inline constexpr bool has_tie_break_key_v = has_tie_break_key<Pattern>::value;

template<class Pattern, class Cap>
struct enabled : std::false_type {};

template<class Pattern, class Cap>
inline constexpr bool enabled_v = enabled<Pattern, Cap>::value;
// Specializations for tie_break_key and enabled<Pattern, Cap> are generated in:
// axp/l4/graph_registry_index.hpp

} // namespace manifest

} // namespace axp::l4

#include "l4/lowering.hpp"
#include "l4/lowering_presets.hpp"
