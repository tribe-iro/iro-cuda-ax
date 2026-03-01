#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::cache {

struct ca { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.cache.ca"); };
struct cg { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.cache.cg"); };
struct cs { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.cache.cs"); };
struct lu { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.cache.lu"); };
struct cv { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.cache.cv"); };

struct wb { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.cache.wb"); };
struct wt { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.cache.wt"); };
struct cs_store { static constexpr auto id = iro::util::fnv1a_64_cstr("axp.cache.cs_store"); };

} // namespace axp::cache
