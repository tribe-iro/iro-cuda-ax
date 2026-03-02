#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>

namespace axp::bundle::check {

template<int AlignBytes>
consteval bool is_power_of_two_alignment() {
    static_assert(AlignBytes > 0, "Alignment must be positive");
    return (AlignBytes & (AlignBytes - 1)) == 0;
}

template<int AlignBytes>
consteval bool is_smem_alignment_sota() {
    return AlignBytes >= 16 && is_power_of_two_alignment<AlignBytes>();
}

template<int Slots>
consteval bool is_pipeline_depth_sota() {
    return Slots >= 2 && Slots <= 4;
}

template<long long Bytes, long long MaxBytes>
consteval bool fits_in_smem_budget() {
    static_assert(MaxBytes > 0, "MaxBytes must be positive");
    return Bytes <= MaxBytes;
}

template<class Bundle, class SlotSubj>
consteval bool is_smem_ready_bundle() {
    using list_t = iro::token::bundle_list<Bundle>;
    if constexpr (!iro::verify::has_token_kind_subject<list_t, iro::token::kind_slot_state, SlotSubj>()) {
        return false;
    }
    using slot_tok = iro::verify::get_token_kind_subject<list_t, iro::token::kind_slot_state, SlotSubj>;
    if constexpr (!std::is_same_v<typename slot_tok::state, iro::token::state::ready>) {
        return false;
    }
    return iro::verify::has_token_kind_subject<list_t, iro::token::kind_visible_at, SlotSubj>() &&
           iro::verify::has_token_kind_subject<list_t, iro::token::kind_alive, SlotSubj>() &&
           iro::verify::has_token_kind_subject<list_t, iro::token::kind_sync_at, SlotSubj>();
}

template<class ExecGroup, class Scope>
consteval bool scope_matches_exec_group() {
    return Scope::level >= iro::scope::min_scope_for_t<ExecGroup>::level;
}

template<class SwizzleAtom>
consteval bool is_swizzle_atom_valid() {
    return (SwizzleAtom::B >= 0) && (SwizzleAtom::M > 0) && (SwizzleAtom::S >= 0);
}

template<class Tile>
consteval bool is_wmma_tile_16x16_f16() {
    return std::is_same_v<typename Tile::elem, iro::elem::f16> &&
           (Tile::shape::rank == 2) &&
           (Tile::shape::template dim<0>() == 16) &&
           (Tile::shape::template dim<1>() == 16);
}

template<class Subject>
struct is_slot_subject : std::false_type {};

template<class PipeRes, int SlotIdx>
struct is_slot_subject<iro::contract::res::slot_subject<PipeRes, SlotIdx>> : std::true_type {};

template<class Subject>
inline constexpr bool is_slot_subject_v = is_slot_subject<Subject>::value;

template<class Subject>
struct is_indexed_subject : std::false_type {};

template<class Tag, int I>
struct is_indexed_subject<iro::contract::subject::indexed<Tag, I>> : std::true_type {};

template<class Subject>
inline constexpr bool is_indexed_subject_v = is_indexed_subject<Subject>::value;

template<class Subject>
struct is_pair_subject : std::false_type {};

template<class A, class B>
struct is_pair_subject<iro::contract::subject::pair<A, B>> : std::true_type {};

template<class Subject>
inline constexpr bool is_pair_subject_v = is_pair_subject<Subject>::value;

template<class Subject>
consteval bool subject_follows_derivation_policy() {
    return is_slot_subject_v<Subject> ||
           is_indexed_subject_v<Subject> ||
           is_pair_subject_v<Subject> ||
           std::is_same_v<Subject, iro::contract::subject::global>;
}

template<class Subject>
consteval bool subject_is_pipeline_slot() {
    return is_slot_subject_v<Subject>;
}

template<class Subject>
consteval bool subject_is_intermediate_wire() {
    return is_indexed_subject_v<Subject>;
}

template<class Subject>
consteval bool subject_is_composite_wire() {
    return is_pair_subject_v<Subject>;
}

}
