#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>
#include "../ownership/dist_tags.hpp"
#include "../../detail/resources.hpp"

namespace axp::protocol::tmem {

namespace detail {
template<bool Cond, class T>
struct type_list_if { using type = iro::util::type_list<>; };
template<class T>
struct type_list_if<true, T> { using type = iro::util::type_list<T>; };

template<class A, class B>
using type_list_concat_t = typename iro::util::concat<A, B>::type;
} // namespace detail

// Shared/Reg -> TMEM
template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup, class Lifetime,
         class InDist, class OutDist>
struct TileToTmem {
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::tmem>);
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::acc>, "TileToTmem: InTile elem != Recipe::acc");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::acc>, "TileToTmem: OutTile elem != Recipe::acc");
    static_assert(iro::util::HasId<InDist>, "TileToTmem: InDist must have id");
    static_assert(iro::util::HasId<OutDist>, "TileToTmem: OutDist must have id");

    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::shared> ||
                  std::is_same_v<typename InTile::space, iro::contract::space::reg>);

    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;
    using in_base = iro::util::type_list<
        iro::token::visible_at<InSubj, scope_t>,
        iro::token::alive<InSubj, Lifetime>
    >;
    using in_tokens = detail::type_list_concat_t<
        in_base,
        typename detail::type_list_if<
            std::is_same_v<typename InTile::space, iro::contract::space::shared>,
            iro::token::slot_state<InSubj, iro::token::state::ready>
        >::type
    >;

    using out_base = iro::util::type_list<
        iro::token::visible_at<OutSubj, scope_t>,
        iro::token::alive<OutSubj, Lifetime>
    >;
    using out_tokens = std::conditional_t<
        (scope_t::level >= iro::scope::warpgroup::level),
        detail::type_list_concat_t<out_base, iro::util::type_list<iro::token::sync_at<OutSubj, scope_t>>>,
        out_base
    >;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            in_tokens,
            InDist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            out_tokens,
            OutDist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// TMEM -> Shared/Reg
template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup, class Lifetime,
         class InDist, class OutDist>
struct TmemToTile {
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::tmem>);
    static_assert(std::is_same_v<typename InTile::elem, typename Recipe::acc>, "TmemToTile: InTile elem != Recipe::acc");
    static_assert(std::is_same_v<typename OutTile::elem, typename Recipe::acc>, "TmemToTile: OutTile elem != Recipe::acc");
    static_assert(std::is_same_v<typename OutTile::space, iro::contract::space::shared> ||
                  std::is_same_v<typename OutTile::space, iro::contract::space::reg>);
    static_assert(iro::util::HasId<InDist>, "TmemToTile: InDist must have id");
    static_assert(iro::util::HasId<OutDist>, "TmemToTile: OutDist must have id");

    using scope_t = iro::scope::min_scope_for_t<ExecGroup>;
    using in_tokens = iro::util::type_list<
        iro::token::visible_at<InSubj, scope_t>,
        iro::token::alive<InSubj, Lifetime>
    >;

    using out_base = iro::util::type_list<
        iro::token::visible_at<OutSubj, scope_t>,
        iro::token::alive<OutSubj, Lifetime>
    >;
    using out_tokens = std::conditional_t<
        (scope_t::level >= iro::scope::warpgroup::level),
        detail::type_list_concat_t<out_base, iro::util::type_list<iro::token::sync_at<OutSubj, scope_t>>>,
        out_base
    >;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            InTile,
            InSubj,
            ExecGroup,
            in_tokens,
            InDist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            OutTile,
            OutSubj,
            ExecGroup,
            out_tokens,
            OutDist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

} // namespace axp::protocol::tmem
