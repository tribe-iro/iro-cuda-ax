#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../protocol/compute/bundles.hpp"
#include "../detail/resources.hpp"
#include "detail/tokens.hpp"

namespace axp::level0 {

namespace detail {

template<class ExecGroup>
struct is_supported_exec_fragment : std::false_type {};

template<> struct is_supported_exec_fragment<iro::exec::warp> : std::true_type {};
template<int Warps> struct is_supported_exec_fragment<iro::exec::warpgroup_t<Warps>> : std::true_type {};

template<class ExecGroup>
struct is_supported_exec_value : std::false_type {};

template<> struct is_supported_exec_value<iro::exec::lane> : std::true_type {};
template<> struct is_supported_exec_value<iro::exec::warp> : std::true_type {};
template<> struct is_supported_exec_value<iro::exec::block> : std::true_type {};
template<int Warps> struct is_supported_exec_value<iro::exec::warpgroup_t<Warps>> : std::true_type {};

template<class Payload, class Recipe>
consteval bool payload_recipe_compatible() {
    static_assert(detail::is_value_payload<Payload>::value,
                  "L0 compute requires Fragment/Scalar/Vector payloads");
    static_assert(std::is_same_v<typename Payload::elem, typename Recipe::in>,
                  "L0 compute: payload elem must match Recipe::in");
    static_assert(std::is_same_v<typename Payload::elem, typename Recipe::out>,
                  "L0 compute: payload elem must match Recipe::out");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "L0 compute: Recipe must be explicit");
    return true;
}

template<class Payload, class ExecGroup>
consteval bool exec_supported() {
    if constexpr (iro::contract::FragmentPayload<Payload>) {
        return is_supported_exec_fragment<ExecGroup>::value;
    } else {
        return is_supported_exec_value<ExecGroup>::value;
    }
}

} // namespace detail

// Unary elementwise op (value -> value)
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct UnaryOp {
    static_assert(detail::exec_supported<Payload, ExecGroup>(),
                  "UnaryOp: unsupported ExecGroup for payload");
    static_assert(detail::payload_recipe_compatible<Payload, Recipe>(),
                  "UnaryOp: payload/recipe mismatch");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, InSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, OutSubj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Binary elementwise op (value -> value)
template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct BinaryOp {
    static_assert(detail::exec_supported<Payload, ExecGroup>(),
                  "BinaryOp: unsupported ExecGroup for payload");
    static_assert(detail::payload_recipe_compatible<Payload, Recipe>(),
                  "BinaryOp: payload/recipe mismatch");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            ASubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, ASubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            Payload,
            BSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, BSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, OutSubj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Ternary fused multiply-add (value -> value)
template<class Recipe, class Payload, class ASubj, class BSubj, class CSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Fma {
    static_assert(detail::exec_supported<Payload, ExecGroup>(),
                  "Fma: unsupported ExecGroup for payload");
    static_assert(detail::payload_recipe_compatible<Payload, Recipe>(),
                  "Fma: payload/recipe mismatch");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            ASubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, ASubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            Payload,
            BSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, BSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            Payload,
            CSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, CSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, OutSubj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Select (value) using mask (mask payload)
template<class Recipe, class Payload, class MaskPayload, class ASubj, class BSubj, class MaskSubj, class OutSubj,
         class ExecGroup, class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Select {
    static_assert(detail::exec_supported<Payload, ExecGroup>(),
                  "Select: unsupported ExecGroup for payload");
    static_assert(iro::contract::MaskPayload<MaskPayload>,
                  "Select: MaskPayload required");
    static_assert(detail::payload_recipe_compatible<Payload, Recipe>(),
                  "Select: payload/recipe mismatch");
    static_assert(MaskPayload::width >= detail::payload_count<Payload>(),
                  "Select: mask width must cover payload element count");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            ASubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, ASubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            Payload,
            BSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, BSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            MaskPayload,
            MaskSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<MaskPayload, MaskSubj, ExecGroup>, InExtra>,
            typename MaskPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, OutSubj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Mask logical ops (mask -> mask)
template<class Recipe, class MaskPayload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct MaskUnaryOp {
    static_assert(iro::contract::MaskPayload<MaskPayload>, "MaskUnaryOp: MaskPayload required");
    static_assert(detail::is_supported_exec_value<ExecGroup>::value,
                  "MaskUnaryOp: ExecGroup must be warp/warpgroup/block");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "MaskUnaryOp: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            MaskPayload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<MaskPayload, InSubj, ExecGroup>, InExtra>,
            typename MaskPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            MaskPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<MaskPayload, OutSubj, ExecGroup>, OutExtra>,
            typename MaskPayload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

template<class Recipe, class MaskPayload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct MaskBinaryOp {
    static_assert(iro::contract::MaskPayload<MaskPayload>, "MaskBinaryOp: MaskPayload required");
    static_assert(detail::is_supported_exec_value<ExecGroup>::value,
                  "MaskBinaryOp: ExecGroup must be warp/warpgroup/block");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "MaskBinaryOp: Recipe must be explicit");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            MaskPayload,
            ASubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<MaskPayload, ASubj, ExecGroup>, InExtra>,
            typename MaskPayload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            MaskPayload,
            BSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<MaskPayload, BSubj, ExecGroup>, InExtra>,
            typename MaskPayload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            MaskPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<MaskPayload, OutSubj, ExecGroup>, OutExtra>,
            typename MaskPayload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Unary ops (distinct types)
template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Alias : UnaryOp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Exp : UnaryOp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Log : UnaryOp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Tanh : UnaryOp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Rsqrt : UnaryOp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Abs : UnaryOp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Neg : UnaryOp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Rcp : UnaryOp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Sqrt : UnaryOp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Sigmoid : UnaryOp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct SiLU : UnaryOp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Gelu : UnaryOp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Popc : UnaryOp<Recipe, Payload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra> {
    static_assert(std::is_same_v<typename Payload::elem, iro::elem::u32>,
                  "Popc: Payload elem must be u32");
};

// Binary ops (distinct types)
template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Add : BinaryOp<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Sub : BinaryOp<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Mul : BinaryOp<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Div : BinaryOp<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Max : BinaryOp<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class Payload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Min : BinaryOp<Recipe, Payload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

// Clamp (value -> value) with explicit min/max inputs
template<class Recipe, class Payload, class InSubj, class MinSubj, class MaxSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct Clamp {
    static_assert(detail::exec_supported<Payload, ExecGroup>(),
                  "Clamp: unsupported ExecGroup for payload");
    static_assert(detail::payload_recipe_compatible<Payload, Recipe>(),
                  "Clamp: payload/recipe mismatch");

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            Payload,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, InSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            Payload,
            MinSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, MinSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >,
        iro::contract::InputPort<
            Payload,
            MaxSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<Payload, MaxSubj, ExecGroup>, InExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            Payload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<Payload, OutSubj, ExecGroup>, OutExtra>,
            typename Payload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

namespace detail {
template<class Pattern, class = void>
struct mask_pattern_ok : std::false_type {};

template<class Pattern>
struct mask_pattern_ok<Pattern, std::void_t<decltype(Pattern::width), decltype(Pattern::word(0))>>
    : std::true_type {};

template<class Pattern, class T, class = void>
struct scalar_pattern_ok : std::false_type {};

template<class Pattern, class T>
struct scalar_pattern_ok<Pattern, T, std::void_t<decltype(Pattern::value)>>
    : std::bool_constant<std::is_convertible_v<decltype(Pattern::value), T>> {};
} // namespace detail

// Scalar constant (no inputs, compile-time value)
template<class Recipe, class ScalarPayload, class OutSubj, class ExecGroup, class Pattern,
         class OutExtra = iro::util::type_list<>>
struct ScalarConst {
    static_assert(iro::contract::ScalarPayload<ScalarPayload>, "ScalarConst: ScalarPayload required");
    static_assert(detail::is_supported_exec_value<ExecGroup>::value,
                  "ScalarConst: ExecGroup must be lane/warp/warpgroup/block");
    static_assert(detail::scalar_pattern_ok<Pattern, typename ScalarPayload::elem::storage_t>::value,
                  "ScalarConst: Pattern must provide static value convertible to payload storage_t");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "ScalarConst: Recipe must be explicit");

    using inputs = iro::util::type_list<>;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            ScalarPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<ScalarPayload, OutSubj, ExecGroup>, OutExtra>,
            typename ScalarPayload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Mask constant (no inputs, compile-time pattern)
template<class Recipe, class MaskPayload, class OutSubj, class ExecGroup, class Pattern,
         class OutExtra = iro::util::type_list<>>
struct MaskConst {
    static_assert(iro::contract::MaskPayload<MaskPayload>, "MaskConst: MaskPayload required");
    static_assert(detail::is_supported_exec_value<ExecGroup>::value,
                  "MaskConst: ExecGroup must be warp/warpgroup/block");
    static_assert(detail::mask_pattern_ok<Pattern>::value,
                  "MaskConst: Pattern must provide width and word(int)");
    static_assert(Pattern::width == MaskPayload::width,
                  "MaskConst: Pattern width must match MaskPayload width");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "MaskConst: Recipe must be explicit");

    using inputs = iro::util::type_list<>;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            MaskPayload,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<MaskPayload, OutSubj, ExecGroup>, OutExtra>,
            typename MaskPayload::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Mask ops (distinct types)
template<class Recipe, class MaskPayload, class InSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct MaskNot : MaskUnaryOp<Recipe, MaskPayload, InSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class MaskPayload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct MaskAnd : MaskBinaryOp<Recipe, MaskPayload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class MaskPayload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct MaskOr : MaskBinaryOp<Recipe, MaskPayload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

template<class Recipe, class MaskPayload, class ASubj, class BSubj, class OutSubj, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct MaskXor : MaskBinaryOp<Recipe, MaskPayload, ASubj, BSubj, OutSubj, ExecGroup, InExtra, OutExtra> {};

// ---------------------------------------------------------------------
// SM90 WGMMA control atoms (warpgroup exec)
// ---------------------------------------------------------------------

template<class Recipe, class Subject, class ExecGroup,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct WgmmaFence {
    static_assert(iro::exec::is_warpgroup_v<ExecGroup>,
                  "WgmmaFence requires warpgroup exec");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "WgmmaFence: Recipe must be explicit");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            axp::protocol::compute::WgmmaHandle,
            Subject,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    iro::token::visible_at<Subject, iro::scope::warpgroup>,
                    iro::token::alive<Subject, iro::token::lifetime::warpgroup>
                >,
                InExtra
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            axp::protocol::compute::WgmmaHandle,
            Subject,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    axp::protocol::compute::wgmma_fenced<Subject>,
                    iro::token::alive<Subject, iro::token::lifetime::warpgroup>
                >,
                OutExtra
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

template<class Recipe, class Subject, class ExecGroup, int Group,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct WgmmaCommitGroup {
    static_assert(iro::exec::is_warpgroup_v<ExecGroup>,
                  "WgmmaCommitGroup requires warpgroup exec");
    static_assert(Group >= 0 && Group <= 7, "WgmmaCommitGroup group out of range");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "WgmmaCommitGroup: Recipe must be explicit");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            axp::protocol::compute::WgmmaHandle,
            Subject,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    axp::protocol::compute::wgmma_issued<Subject>,
                    iro::token::alive<Subject, iro::token::lifetime::warpgroup>
                >,
                InExtra
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            axp::protocol::compute::WgmmaHandle,
            Subject,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    axp::protocol::compute::wgmma_committed<Subject, Group>,
                    iro::token::alive<Subject, iro::token::lifetime::warpgroup>
                >,
                OutExtra
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

template<class Recipe, class Subject, class ExecGroup, int Group,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct WgmmaWaitGroup {
    static_assert(iro::exec::is_warpgroup_v<ExecGroup>,
                  "WgmmaWaitGroup requires warpgroup exec");
    static_assert(Group >= 0 && Group <= 7, "WgmmaWaitGroup group out of range");
    static_assert(!std::is_same_v<Recipe, iro::recipe::no_recipe>,
                  "WgmmaWaitGroup: Recipe must be explicit");
    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            axp::protocol::compute::WgmmaHandle,
            Subject,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    axp::protocol::compute::wgmma_committed<Subject, Group>,
                    iro::token::alive<Subject, iro::token::lifetime::warpgroup>
                >,
                InExtra
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            axp::protocol::compute::WgmmaHandle,
            Subject,
            ExecGroup,
            iro::util::concat_t<
                iro::util::type_list<
                    axp::protocol::compute::wgmma_waited<Subject, Group>,
                    iro::token::sync_at<Subject, iro::scope::warpgroup>,
                    iro::token::alive<Subject, iro::token::lifetime::warpgroup>
                >,
                OutExtra
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;
    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

// Enforce WGMMA wait before consuming accumulator (passes AccFrag through).
template<class Recipe, class AccFrag, class InSubj, class OutSubj, class WgmmaSubj, class ExecGroup, int Group,
         class InExtra = iro::util::type_list<>, class OutExtra = iro::util::type_list<>>
struct WgmmaWaitAcc {
    static_assert(iro::exec::is_warpgroup_v<ExecGroup>,
                  "WgmmaWaitAcc requires warpgroup exec");
    static_assert(iro::contract::FragmentPayload<AccFrag>, "WgmmaWaitAcc requires Fragment payload");
    static_assert(std::is_same_v<typename AccFrag::elem, typename Recipe::acc>,
                  "WgmmaWaitAcc: AccFrag elem must match Recipe::acc");
    using lifetime = iro::token::lifetime::warpgroup;

    using inputs = iro::util::type_list<
        iro::contract::InputPort<
            AccFrag,
            InSubj,
            ExecGroup,
            iro::util::concat_t<detail::in_tokens<AccFrag, InSubj, ExecGroup>, InExtra>,
            typename AccFrag::dist,
            Recipe
        >,
        iro::contract::InputPort<
            axp::protocol::compute::WgmmaHandle,
            WgmmaSubj,
            ExecGroup,
            iro::util::type_list<
                axp::protocol::compute::wgmma_waited<WgmmaSubj, Group>,
                iro::token::alive<WgmmaSubj, lifetime>
            >,
            iro::contract::no_dist,
            Recipe
        >
    >;

    using outputs = iro::util::type_list<
        iro::contract::OutputPort<
            AccFrag,
            OutSubj,
            ExecGroup,
            iro::util::concat_t<detail::out_tokens<AccFrag, OutSubj, ExecGroup>, OutExtra>,
            typename AccFrag::dist,
            Recipe
        >
    >;

    using resources = axp::detail::warpgroup_layout_resources_t<ExecGroup>;
};

} // namespace axp::level0
