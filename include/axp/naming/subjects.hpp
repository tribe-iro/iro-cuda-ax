#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../bundles/checklist.hpp"
#include "tags.hpp"

namespace axp::subject {

using global = iro::contract::subject::global;

template<class Tag, int I>
using indexed = iro::contract::subject::indexed<Tag, I>;

template<class A, class B>
using pair = iro::contract::subject::pair<A, B>;

template<class PipeRes, int SlotIdx>
using slot = iro::contract::res::slot_subject<PipeRes, SlotIdx>;

template<class Tag, int I = 0>
using wire = indexed<Tag, I>;

template<class A, class B>
using composite = pair<A, B>;

template<class Subject>
inline constexpr bool follows_derivation_policy_v =
    axp::bundle::check::subject_follows_derivation_policy<Subject>();

template<class Subject>
consteval void enforce_derivation_policy() {
    static_assert(follows_derivation_policy_v<Subject>,
                  "axp::subject: Subject must be slot_subject, indexed, pair, or global");
}

// Common GEMM subjects
using MatrixA = wire<axp::tag::A, 0>;
using MatrixB = wire<axp::tag::B, 0>;
using MatrixC = wire<axp::tag::C, 0>;
using MatrixD = wire<axp::tag::D, 0>;
using Accumulator = wire<axp::tag::Acc, 0>;
using Output = wire<axp::tag::O, 0>;

// Attention subjects
using AttentionQ = wire<axp::tag::Q, 0>;
using AttentionK = wire<axp::tag::K, 0>;
using AttentionV = wire<axp::tag::V, 0>;
using AttentionS = wire<axp::tag::S, 0>;
using AttentionP = wire<axp::tag::P, 0>;
using TileSkip = wire<axp::tag::TileSkip, 0>;

} // namespace axp::subject
