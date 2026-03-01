#pragma once

#include <iro_cuda_ax_core.hpp>
#include <concepts>

namespace axp::concepts {

template<class R>
concept Recipe = requires {
    typename R::in;
    typename R::acc;
    typename R::out;
    { R::vec_bytes } -> std::convertible_to<int>;
};

template<class R>
concept RecipeAccF32 = Recipe<R> && std::same_as<typename R::acc, iro::elem::f32>;

template<class R>
concept RecipeInF16 = Recipe<R> && std::same_as<typename R::in, iro::elem::f16>;

template<class R>
concept RecipeInBF16 = Recipe<R> && std::same_as<typename R::in, iro::elem::bf16>;

template<class R>
concept RecipeF16AccF32 = RecipeInF16<R> && RecipeAccF32<R>;

template<class R>
concept RecipeF32 = Recipe<R> &&
    std::same_as<typename R::in, iro::elem::f32> &&
    std::same_as<typename R::acc, iro::elem::f32> &&
    std::same_as<typename R::out, iro::elem::f32>;

template<int K>
concept K16 = (K == 16);

template<int M, int N>
concept MultipleOf16MN = (M % 16 == 0) && (N % 16 == 0);

template<int S>
concept StageCount = (S >= 2) && (S <= 4);

template<int V>
concept VecBytes = (V == 4 || V == 8 || V == 16);

template<int B>
concept BlockThreads32 = (B > 0) && (B % 32 == 0) && (B <= 1024);

template<class E>
concept BlockExec = std::same_as<E, iro::exec::block>;

template<class E>
concept WarpExec = std::same_as<E, iro::exec::warp>;

template<class E>
concept WarpgroupExec = iro::exec::is_warpgroup_v<E>;

} // namespace axp::concepts
