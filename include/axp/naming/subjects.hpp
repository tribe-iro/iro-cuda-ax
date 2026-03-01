#pragma once

#include <iro_cuda_ax_core.hpp>
#include "tags.hpp"

namespace axp::subject {

using global = iro::contract::subject::global;

template<class Tag, int I>
using indexed = iro::contract::subject::indexed<Tag, I>;

template<class A, class B>
using pair = iro::contract::subject::pair<A, B>;

// Common GEMM subjects
using MatrixA = indexed<axp::tag::A, 0>;
using MatrixB = indexed<axp::tag::B, 0>;
using MatrixC = indexed<axp::tag::C, 0>;
using MatrixD = indexed<axp::tag::D, 0>;
using Accumulator = indexed<axp::tag::Acc, 0>;
using Output = indexed<axp::tag::O, 0>;

// Attention subjects
using AttentionQ = indexed<axp::tag::Q, 0>;
using AttentionK = indexed<axp::tag::K, 0>;
using AttentionV = indexed<axp::tag::V, 0>;
using AttentionS = indexed<axp::tag::S, 0>;
using AttentionP = indexed<axp::tag::P, 0>;
using TileSkip = indexed<axp::tag::TileSkip, 0>;

// Pipeline and mask subjects
using PipeA = indexed<axp::tag::PipeA, 0>;
using PipeB = indexed<axp::tag::PipeB, 0>;
using PipeO = indexed<axp::tag::PipeO, 0>;

using SlotA0 = indexed<axp::tag::SlotA0, 0>;
using SlotA1 = indexed<axp::tag::SlotA1, 1>;
using SlotA2 = indexed<axp::tag::SlotA2, 2>;
using SlotA3 = indexed<axp::tag::SlotA3, 3>;
using SlotB0 = indexed<axp::tag::SlotB0, 0>;
using SlotB1 = indexed<axp::tag::SlotB1, 1>;
using SlotB2 = indexed<axp::tag::SlotB2, 2>;
using SlotB3 = indexed<axp::tag::SlotB3, 3>;
using SlotO0 = indexed<axp::tag::SlotO0, 0>;
using SlotO1 = indexed<axp::tag::SlotO1, 1>;

using Mask = indexed<axp::tag::Mask, 0>;

using Coord0 = indexed<axp::tag::Coord0, 0>;
using Coord1 = indexed<axp::tag::Coord1, 1>;

} // namespace axp::subject
