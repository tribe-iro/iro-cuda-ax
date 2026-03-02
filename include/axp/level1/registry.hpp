#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../target.hpp"

namespace axp::level1::registry {

template<class>
inline constexpr bool always_false_v = false;

// Reduction patterns
template<class Recipe, class Frag, class Subj, class ExecGroup,
         template<class, class, class, class, class, class, class, class> class Op>
struct WarpReducePattern {};

template<class Recipe, class Frag, class SmemTile, class Subj, class SmemSubj, class ExecGroup,
         template<class, class, class, class, class, class, class, class> class Op>
struct BlockReducePattern {};

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag>
struct WarpAllReducePattern {};

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag, int SegmentWidth>
struct WarpSegmentedReducePattern {};

template<class Recipe, class Frag, class Subj, class ExecGroup, class OpTag,
         int BarrierId = 1, int WarpgroupCount = 1>
struct WarpgroupReducePattern {};

template<class Recipe, class Payload, class MaskPayload, class Subj, class MaskSubj, class ExecGroup, class OpTag>
struct ShuffleReduceTreePattern {};

// Scan patterns
template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct WarpScanPattern {};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode, int SegmentWidth,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct WarpSegmentedScanPattern {};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         int BarrierId = 1, int WarpgroupCount = 1,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct WarpgroupScanPattern {};

template<class Recipe, class Payload, class Subj, class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct BlockScanPattern {};

template<class Recipe, class Payload, class CarryPayload, class Subj, class CarryInSubj, class CarryOutSubj,
         class ExecGroup, class OpTag, class Mode,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct ChainedScanPattern {};

// Order patterns
template<class Recipe, class Subject, class EventTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct PublishEventPattern {};

template<class Recipe, class Payload, class PayloadSubj, class Subject, class EventTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct EmitEventAfterPattern {};

template<class Recipe, class Subject, class EventTag, class PhaseTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct DependOnEventPattern {};

template<class Recipe, class Payload, class PayloadSubj, class Subject, class EventTag, class PhaseTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct DependOnEventGatePattern {};

template<class Recipe, class Subject, class PrevEpochTag, class NextEpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct AdvanceEpochPattern {};

template<class Recipe, class Subject, class ScopeT, class EventTag, class ExecGroup,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block>
struct EventFromAtomicDonePattern {};

// Epoch patterns
template<class Recipe, class Subject, class EpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct InitEpochPattern {};

template<class Recipe, class Subject, class PrevEpochTag, class NextEpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct AdvanceEpochTokenPattern {};

template<class Recipe, class Subject, class EpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct RequireEpochPattern {};

// Atomic patterns
template<class Recipe, class Subject, class ExecGroup, class ScopeT,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block>
struct MarkAtomicDonePattern {};

template<class Recipe, class Subject, class ExecGroup, class ScopeT,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block>
struct RequireAtomicDonePattern {};

template<class Recipe, class TilePayload, class Subject, class ExecGroup, class ScopeT,
         class OrderT = iro::memory_order::seq_cst,
         class Lifetime = iro::token::lifetime::block>
struct MarkAtomicDoneFromTilePattern {};

// Row reduction patterns
template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup>
struct RowSumPattern {};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup>
struct RowSumVecPattern {};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup>
struct RowMaxPattern {};

template<class Recipe, class Frag, class InSubj, class OutSubj, class ExecGroup>
struct RowMaxVecPattern {};

// Gather/scatter patterns
template<class Recipe, class InTile, class IndexPayload, class OutPayload,
         class InSubj, class IndexSubj, class OutSubj, class ExecGroup, class CachePolicy>
struct GatherGlobalPattern {};

template<class Recipe, class InPayload, class IndexPayload, class OutTile,
         class InSubj, class IndexSubj, class OutSubj, class ExecGroup, class CachePolicy>
struct ScatterGlobalPattern {};

// Permute patterns
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup, class Pattern,
         int BlockThreads = 0, class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct PermutePattern {};

// Count bits pattern (warp ballot + popc + exclusive prefix)
template<class Recipe, class InPayload, class InSubj, class PrefixSubj, class CountSubj, class ExecGroup>
struct CountBitsPattern {};

// Sort/merge patterns (bitonic, fragment payload)
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup>
struct BitonicSortPattern {};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup>
struct BitonicMergePattern {};

// Cross-lane merge (warp bitonic, scalar payload)
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup>
struct BitonicMergeCrossPattern {};

// Causal mask + tile-skip predicate pattern
template<class Recipe, class MaskPayload, class PredPayload,
         class QCoordPayload, class KCoordPayload,
         class QCoordSubj, class KCoordSubj,
         class MaskSubj, class PredSubj,
         class ExecGroup, int TileM, int TileN>
struct CausalMaskPattern {};

template<class Pattern, class Cap = axp::target_cap, class = void>
struct resolve_impl {
    static constexpr bool supported = false;
};

template<class Pattern, class Cap = axp::target_cap>
struct resolve {
    static_assert(resolve_impl<Pattern, Cap>::supported,
                  "axp::level1::registry::resolve: unsupported pattern");
    using type = typename resolve_impl<Pattern, Cap>::type;
};

template<class Pattern, class Cap = axp::target_cap>
using Select = typename resolve<Pattern, Cap>::type;

template<class Pattern, class Cap = axp::target_cap>
struct supports : std::bool_constant<resolve_impl<Pattern, Cap>::supported> {};

} // namespace axp::level1::registry
