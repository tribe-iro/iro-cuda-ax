#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::protocol::audit {

namespace detail {

template<class Port, bool IsInput = Port::is_input>
struct token_list;

template<class Port>
struct token_list<Port, true> { using type = typename Port::required; };

template<class Port>
struct token_list<Port, false> { using type = typename Port::provided; };

template<class Tokens, class Subject>
consteval bool has_alive() {
    return iro::verify::has_token_kind_subject<Tokens, iro::token::kind_alive, Subject>();
}

template<class Tokens, class Subject>
consteval bool has_visible() {
    return iro::verify::has_token_kind_subject<Tokens, iro::token::kind_visible_at, Subject>();
}

template<class Tokens, class Subject>
consteval bool has_sync() {
    return iro::verify::has_token_kind_subject<Tokens, iro::token::kind_sync_at, Subject>();
}

} // namespace detail

template<class Port>
consteval bool port_tokens_complete() {
    using payload = typename Port::payload;
    using subject = typename Port::subject;
    using exec = typename Port::exec_group;
    using tokens = typename detail::token_list<Port>::type;

    if constexpr (iro::contract::TilePayload<payload>) {
        constexpr bool has_visible = detail::has_visible<tokens, subject>();
        constexpr bool has_alive = detail::has_alive<tokens, subject>();
        constexpr bool needs_sync = (iro::scope::min_scope_for_t<exec>::level >= iro::scope::warpgroup::level);
        if constexpr (Port::is_input) {
            return has_visible && has_alive;
        } else {
            if constexpr (needs_sync) {
                return has_visible && has_alive && detail::has_sync<tokens, subject>();
            } else {
                return has_visible && has_alive;
            }
        }
    } else if constexpr (iro::contract::HandlePayload<payload> || iro::contract::RefPayload<payload>) {
        return detail::has_alive<tokens, subject>();
    } else {
        return detail::has_alive<tokens, subject>();
    }
}

template<class List>
struct list_tokens_complete;

template<>
struct list_tokens_complete<iro::util::type_list<>> {
    static constexpr bool value = true;
};

template<class P, class... Ps>
struct list_tokens_complete<iro::util::type_list<P, Ps...>> {
    static constexpr bool value =
        port_tokens_complete<P>() && list_tokens_complete<iro::util::type_list<Ps...>>::value;
};

template<class Contract>
consteval bool contract_tokens_complete() {
    return list_tokens_complete<typename Contract::inputs>::value &&
           list_tokens_complete<typename Contract::outputs>::value;
}

} // namespace axp::protocol::audit
