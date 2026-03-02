#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>

namespace axp::protocol::token_policy {

template<class Token, class = void>
struct is_semantic_token : std::false_type {};

template<class Token>
struct is_semantic_token<Token, std::void_t<typename Token::kind, typename Token::subject, decltype(Token::id)>>
    : std::true_type {};

template<class Token>
inline constexpr bool is_semantic_token_v = is_semantic_token<Token>::value;

template<class Subject>
struct canonical_subject {
    using type = Subject;
};

template<class Subject>
using canonical_subject_t = typename canonical_subject<Subject>::type;

template<class Token, class = void>
struct canonical_token {
    using type = Token;
};

template<class Token>
struct canonical_token<Token, std::enable_if_t<is_semantic_token_v<Token>>> {
    using type = Token;
};

template<class Token>
using canonical_token_t = typename canonical_token<Token>::type;

template<class Required, class Provided, class = void>
struct satisfiable : std::false_type {};

template<class Required, class Provided>
struct satisfiable<Required, Provided,
                   std::enable_if_t<is_semantic_token_v<Required> && is_semantic_token_v<Provided>>>
    : std::bool_constant<
          (Required::kind::id == Provided::kind::id) &&
          std::is_same_v<canonical_subject_t<typename Required::subject>,
                         canonical_subject_t<typename Provided::subject>>> {};

template<class Required, class Provided>
inline constexpr bool satisfiable_v = satisfiable<Required, Provided>::value;

template<class Required, class Provided>
consteval void enforce_satisfiable() {
    static_assert(satisfiable_v<Required, Provided>,
                  "axp::protocol::token_policy: token requirement is not satisfiable by provided token");
}

} // namespace axp::protocol::token_policy

