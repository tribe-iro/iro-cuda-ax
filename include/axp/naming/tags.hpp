#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::tag {

struct A { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.A"); };
struct B { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.B"); };
struct C { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.C"); };
struct D { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.D"); };
struct O { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.O"); };
struct Acc { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.Acc"); };

struct Q { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.Q"); };
struct K { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.K"); };
struct V { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.V"); };
struct S { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.S"); };
struct P { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.P"); };
struct TileSkip { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.TileSkip"); };

struct PipeA { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.PipeA"); };
struct PipeB { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.PipeB"); };
struct PipeO { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.PipeO"); };

struct SlotA0 { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.SlotA0"); };
struct SlotA1 { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.SlotA1"); };
struct SlotA2 { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.SlotA2"); };
struct SlotA3 { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.SlotA3"); };
struct SlotB0 { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.SlotB0"); };
struct SlotB1 { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.SlotB1"); };
struct SlotB2 { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.SlotB2"); };
struct SlotB3 { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.SlotB3"); };
struct SlotO0 { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.SlotO0"); };
struct SlotO1 { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.SlotO1"); };

struct Mask { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.Mask"); };

struct Coord0 { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.Coord0"); };
struct Coord1 { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.tag.Coord1"); };

} // namespace axp::tag
