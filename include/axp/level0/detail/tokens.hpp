#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../../detail/participation_tokens.hpp"
#include "../../bundles/token_bundles.hpp"

namespace axp::level0::detail {

template<class Payload>
struct is_value_payload : std::bool_constant<
    iro::contract::FragmentPayload<Payload> ||
    iro::contract::ScalarPayload<Payload> ||
    iro::contract::VectorPayload<Payload>
> {};

template<class Payload>
consteval int payload_count() {
    if constexpr (iro::contract::FragmentPayload<Payload>) {
        return static_cast<int>(Payload::count);
    } else if constexpr (iro::contract::ScalarPayload<Payload>) {
        return 1;
    } else if constexpr (iro::contract::VectorPayload<Payload>) {
        return Payload::lanes;
    } else {
        return 0;
    }
}

template<class ExecGroup>
struct is_supported_exec : std::false_type {};

template<> struct is_supported_exec<iro::exec::lane> : std::true_type {};
template<> struct is_supported_exec<iro::exec::warp> : std::true_type {};
template<> struct is_supported_exec<iro::exec::block> : std::true_type {};
template<> struct is_supported_exec<iro::exec::cluster> : std::true_type {};
template<int Warps> struct is_supported_exec<iro::exec::warpgroup_t<Warps>> : std::true_type {};

template<class ExecGroup>
struct is_supported_exec_memory : std::false_type {};

template<> struct is_supported_exec_memory<iro::exec::lane> : std::true_type {};
template<> struct is_supported_exec_memory<iro::exec::warp> : std::true_type {};
template<> struct is_supported_exec_memory<iro::exec::block> : std::true_type {};
template<> struct is_supported_exec_memory<iro::exec::cluster> : std::true_type {};
template<int Warps> struct is_supported_exec_memory<iro::exec::warpgroup_t<Warps>> : std::true_type {};

template<class ExecGroup>
struct lifetime_for_exec {
    static_assert(!std::is_same_v<ExecGroup, ExecGroup>,
                  "AX: unsupported ExecGroup");
    using type = iro::token::lifetime::block;
};

template<> struct lifetime_for_exec<iro::exec::lane> { using type = iro::token::lifetime::instruction; };
template<> struct lifetime_for_exec<iro::exec::warp> { using type = iro::token::lifetime::warp; };
template<> struct lifetime_for_exec<iro::exec::block> { using type = iro::token::lifetime::block; };
template<> struct lifetime_for_exec<iro::exec::cluster> { using type = iro::token::lifetime::cluster; };
template<int Warps> struct lifetime_for_exec<iro::exec::warpgroup_t<Warps>> {
    using type = iro::token::lifetime::warpgroup;
};

template<class ExecGroup>
using lifetime_for_exec_t = typename lifetime_for_exec<ExecGroup>::type;

template<class Subject, class ExecGroup>
using participation_tokens = axp::detail::participation_tokens<Subject, ExecGroup>;

template<class Payload, class Subject, class ExecGroup>
using in_tokens = std::conditional_t<
    iro::contract::TilePayload<Payload>,
    axp::bundle::TileInTokens<Subject, ExecGroup, lifetime_for_exec_t<ExecGroup>>,
    std::conditional_t<
        iro::contract::MaskPayload<Payload>,
        iro::util::type_list<
            iro::token::alive<Subject, lifetime_for_exec_t<ExecGroup>>,
            iro::token::mask_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>
        >,
        iro::util::concat_t<
            iro::util::type_list<iro::token::alive<Subject, lifetime_for_exec_t<ExecGroup>>>,
            participation_tokens<Subject, ExecGroup>
        >
    >
>;

template<class Payload, class Subject, class ExecGroup>
using out_tokens = std::conditional_t<
    iro::contract::TilePayload<Payload>,
    axp::bundle::TileOutTokens<Subject, ExecGroup, lifetime_for_exec_t<ExecGroup>>,
    std::conditional_t<
        iro::contract::MaskPayload<Payload>,
        iro::util::type_list<
            iro::token::alive<Subject, lifetime_for_exec_t<ExecGroup>>,
            iro::token::mask_at<Subject, iro::scope::min_scope_for_t<ExecGroup>>
        >,
        iro::util::concat_t<
            iro::util::type_list<iro::token::alive<Subject, lifetime_for_exec_t<ExecGroup>>>,
            participation_tokens<Subject, ExecGroup>
        >
    >
>;

} // namespace axp::level0::detail
