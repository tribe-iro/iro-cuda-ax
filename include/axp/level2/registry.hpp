#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../target.hpp"
#include "../swizzle.hpp"
#include "../intent.hpp"

namespace axp::level2::registry {

template<class>
inline constexpr bool always_false_v = false;

// Scale shared tile pattern (FP8 scale prelude)
template<class Recipe, class Tile, class ScaleTile, class TileSubj, class ScaleSubj, class ExecGroup>
struct ScaleSharedTilePattern {};

// Matmul patterns (registry chooses warp vs warpgroup)
template<class Recipe, class Shape, class ATile, class BTile, class AccFrag,
         class ASubj, class BSubj, class AccSubj, class ExecGroup, class WgmmaSubj>
struct MatmulPattern {};

template<class Recipe, class Shape, class ATile, class BTile, class AccFrag,
         class ASubj, class BSubj, class AccSubj>
struct MatmulWarpPattern {};

template<class Recipe, class Shape, class ADesc, class BDesc, class AccFrag,
         class ADescSubj, class BDescSubj, class AccSubj, class WgmmaSubj, class ExecGroup>
struct MatmulWarpgroupPattern {};

// Staging patterns (GMEM -> SMEM slot pipelines)
template<class Recipe, class InTile, class OutTile, class InSubj, class PipeTag, class SlotSubj,
         class MarkExecGroup, class Lifetime, int Slots, class SwizzleAtom,
         class Tma = void, class IssueExecGroup = iro::exec::block>
struct StageGmemToSmemPattern {};

template<class Recipe, class InTile, class OutTile, class SlotSubj, class OutSubj,
         class MarkExecGroup, class Lifetime, class Tma = void>
struct StageSmemToGmemPattern {};

// Ordering / atomic protocol patterns
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

template<class Recipe, class Subject, class EpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct InitEpochPattern {};

template<class Recipe, class Subject, class PrevEpochTag, class NextEpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct AdvanceEpochTokenPattern {};

template<class Recipe, class Subject, class EpochTag, class ExecGroup,
         class Lifetime = iro::token::lifetime::block>
struct RequireEpochPattern {};

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

// Registry plumbing
template<class Pattern, class Cap = axp::target_cap, class = void>
struct resolve_impl {
    static constexpr bool supported = false;
};

template<class Pattern, class Cap = axp::target_cap>
struct resolve {
    static_assert(resolve_impl<Pattern, Cap>::supported,
                  "axp::level2::registry::resolve: unsupported pattern");
    using type = typename resolve_impl<Pattern, Cap>::type;
};

template<class Pattern, class Cap = axp::target_cap>
using Select = typename resolve<Pattern, Cap>::type;

template<class Pattern, class Cap = axp::target_cap>
struct supports : std::bool_constant<resolve_impl<Pattern, Cap>::supported> {};

} // namespace axp::level2::registry
