#pragma once
/**
 * @file iro_cuda_ax_core.hpp
 * @brief iro-cuda-ax CORE v0.2.4 — Protocol-Token Kernel Composition Contract
 *
 * This header implements the full iro-cuda-ax CORE specification v0.2.4:
 * - Explicit composition of small, ultra-optimized GPU building blocks
 * - FlashInfer/vLLM-class kernels without JIT and without hidden effects
 * - Protocol tokens and resources for inter-block correctness
 * - Consteval verification at compile time
 *
 * Requirements:
 * - C++20 standard (--std=c++20)
 * - CUDA headers optional (storage_t only available under nvcc)
 *
 * Namespaces (normative per spec):
 * - iro::util      - Type lists, stable IDs, checked arithmetic
 * - iro::schema    - Concepts (Tag, ElemTag, Layout, Token, Resource, ...)
 * - iro::contract  - Shape, Tile, TensorRefDesc, FragmentDesc, Align
 * - iro::contract::subject - Subject constructors (global, indexed, pair)
 * - iro::contract::space   - Memory spaces (global, shared, reg, fragment)
 * - iro::contract::res     - Resources (smem_region, smem_pipeline, ...)
 * - iro::exec      - Execution groups (warp, warpgroup, block, cluster)
 * - iro::scope     - Visibility scopes (lane, warp, warpgroup, block, cluster, device)
 * - iro::token     - Protocol tokens (visible_at, lanes_valid, warps_valid, alive, slot_state, lease, sync_at)
 * - iro::cap       - Capabilities
 * - iro::profile   - Profiles
 * - iro::registry  - Registries
 * - iro::verify    - Verification predicates
 * - iro::compose   - Composition
 * - iro::bind      - Binding (explicit realization lookup)
 * - iro::diag      - Diagnostics
 *
 * Core axiom: NO HIDDEN EFFECTS
 * - Inter-block correctness via explicit protocol tokens and resources only
 * - AX never infers/inserts waits, barriers, fences, conversions, adapters,
 *   lifetimes, ownership, aliasing, reuse, or capability fallbacks
 *
 * @see docs/architecture/layer_contract_law.md and docs/architecture/protocol_planes.md
 *      for the current normative architecture/spec references.
 */

#include <cstdint>
#include <type_traits>
#include <concepts>

// Optional CUDA headers for storage types
#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#endif

// Require C++20
#if __cplusplus < 202002L
    #error "iro-cuda-ax requires C++20 (--std=c++20)"
#endif

// Programmatic AX version constants.
inline constexpr int version_major = 0;
inline constexpr int version_minor = 2;
inline constexpr int version_patch = 4;

// Forward declaration for element tags used in early traits
namespace iro::elem {
struct u32;
struct e4m3;
struct e5m2;
struct e4m3fn;
struct e5m2fnuz;
}

// =============================================================================
// §1. Utilities and stable identity (iro::util)
// =============================================================================

namespace iro::util {

// -----------------------------------------------------------------------------
// §1.1 Type lists (normative)
// -----------------------------------------------------------------------------

/// Type list container
template<class... Ts> struct type_list {};

/// Get size of type list
template<class List> struct size;
template<class... Ts>
struct size<type_list<Ts...>> : std::integral_constant<int, sizeof...(Ts)> {};

template<class List>
inline constexpr int size_v = size<List>::value;

/// Get element at index I
template<class List, int I> struct at;
template<class T0, class... Ts>
struct at<type_list<T0, Ts...>, 0> { using type = T0; };
template<class T0, class... Ts, int I>
struct at<type_list<T0, Ts...>, I> {
    static_assert(I > 0);
    using type = typename at<type_list<Ts...>, I - 1>::type;
};

template<class List, int I>
using at_t = typename at<List, I>::type;

/// Concatenate two type lists
template<class A, class B> struct concat;
template<class... As, class... Bs>
struct concat<type_list<As...>, type_list<Bs...>> {
    using type = type_list<As..., Bs...>;
};

template<class A, class B>
using concat2_t = typename concat<A, B>::type;

template<class... Lists>
struct concat_n;

template<>
struct concat_n<> {
    using type = type_list<>;
};

template<class... Ts>
struct concat_n<type_list<Ts...>> {
    using type = type_list<Ts...>;
};

template<class... Ts, class... Us, class... Rest>
struct concat_n<type_list<Ts...>, type_list<Us...>, Rest...> {
    using type = typename concat_n<type_list<Ts..., Us...>, Rest...>::type;
};

template<class... Lists>
using concat_n_t = typename concat_n<Lists...>::type;

template<class... Lists>
using concat_t = concat_n_t<Lists...>;

template<class A, class B, class C>
using concat3_t = concat_t<A, B, C>;

/// Remove element at index I
template<class List, int I> struct remove_at;
template<int I>
struct remove_at<type_list<>, I> { using type = type_list<>; };
template<class T0, class... Ts>
struct remove_at<type_list<T0, Ts...>, 0> { using type = type_list<Ts...>; };
template<class T0, class... Ts, int I>
struct remove_at<type_list<T0, Ts...>, I> {
    static_assert(I > 0);
    using type = typename concat<type_list<T0>, typename remove_at<type_list<Ts...>, I - 1>::type>::type;
};

template<class List, int I>
using remove_at_t = typename remove_at<List, I>::type;

/// Append element to type list
template<class List, class T> struct append;
template<class... Ts, class T>
struct append<type_list<Ts...>, T> { using type = type_list<Ts..., T>; };

template<class List, class T>
using append_t = typename append<List, T>::type;

/// Prepend element to type list
template<class List, class T> struct prepend;
template<class... Ts, class T>
struct prepend<type_list<Ts...>, T> { using type = type_list<T, Ts...>; };

template<class List, class T>
using prepend_t = typename prepend<List, T>::type;

/// Check if type is in list
template<class List, class T> struct contains;
template<class T>
struct contains<type_list<>, T> : std::false_type {};
template<class T, class... Ts>
struct contains<type_list<T, Ts...>, T> : std::true_type {};
template<class T0, class... Ts, class T>
struct contains<type_list<T0, Ts...>, T> : contains<type_list<Ts...>, T> {};

template<class List, class T>
inline constexpr bool contains_v = contains<List, T>::value;

/// Count occurrences of a type in a list
template<class List, class T> struct count;
template<class T>
struct count<type_list<>, T> : std::integral_constant<int, 0> {};
template<class T0, class... Ts, class T>
struct count<type_list<T0, Ts...>, T>
    : std::integral_constant<int, (std::is_same_v<T0, T> ? 1 : 0) + count<type_list<Ts...>, T>::value> {};

template<class List, class T>
inline constexpr int count_v = count<List, T>::value;

/// Remove the first occurrence of T from a list
template<class Acc, class List, class T, bool Removed>
struct remove_one_impl;

template<class... AccTs, class T, bool Removed>
struct remove_one_impl<type_list<AccTs...>, type_list<>, T, Removed> {
    using type = type_list<AccTs...>;
    static constexpr bool removed = Removed;
};

template<class... AccTs, class T0, class... Ts, class T, bool Removed>
struct remove_one_impl<type_list<AccTs...>, type_list<T0, Ts...>, T, Removed> {
    static constexpr bool match = !Removed && std::is_same_v<T0, T>;
    using next_acc = std::conditional_t<match, type_list<AccTs...>, type_list<AccTs..., T0>>;
    using next = remove_one_impl<next_acc, type_list<Ts...>, T, Removed || match>;
    using type = typename next::type;
    static constexpr bool removed = next::removed;
};

template<class List, class T> struct remove_one;
template<class T>
struct remove_one<type_list<>, T> {
    using type = type_list<>;
    static constexpr bool removed = false;
};
template<class T0, class... Ts, class T>
struct remove_one<type_list<T0, Ts...>, T> {
    using impl = remove_one_impl<type_list<>, type_list<T0, Ts...>, T, false>;
    using type = typename impl::type;
    static constexpr bool removed = impl::removed;
};

template<class List, class T>
using remove_one_t = typename remove_one<List, T>::type;

/// Unsigned 64-bit type for stable IDs
using u64 = unsigned long long;

/// Concept: type has a stable id
template<class T>
concept HasId = requires { { T::id } -> std::convertible_to<u64>; };

// -----------------------------------------------------------------------------
// §1.2 Stable IDs (normative)
// -----------------------------------------------------------------------------

/// FNV-1a 64-bit hash of a null-terminated string (consteval)
consteval u64 fnv1a_64_cstr(const char* s) {
    u64 h = 14695981039346656037ULL;
    for (; *s; ++s) {
        h ^= static_cast<unsigned char>(*s);
        h *= 1099511628211ULL;
    }
    return h;
}

/// Mix two u64 values (consteval, high-quality mixing)
consteval u64 mix_u64(u64 h, u64 x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return h ^ x;
}

// Hash a type list using element ids (order-sensitive).
template<u64 Seed, class List>
struct hash_list_acc;

template<u64 Seed>
struct hash_list_acc<Seed, type_list<>> {
    static constexpr u64 value = Seed;
};

template<u64 Seed, class T, class... Ts>
struct hash_list_acc<Seed, type_list<T, Ts...>> {
    static_assert(HasId<T>, "hash_list requires HasId elements");
    static constexpr u64 value =
        hash_list_acc<mix_u64(Seed, static_cast<u64>(T::id)), type_list<Ts...>>::value;
};

template<class List>
struct hash_list {
    static constexpr u64 value =
        hash_list_acc<fnv1a_64_cstr("iro.util.hash_list"), List>::value;
};

template<class List>
inline constexpr u64 hash_list_v = hash_list<List>::value;

// -----------------------------------------------------------------------------
// §1.3 Checked arithmetic helpers (normative; prevents UB)
// -----------------------------------------------------------------------------

/// Checked multiplication with overflow detection
consteval u64 checked_mul_u64(u64 a, u64 b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t p = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    // Note: can't use static_assert here on runtime value, use conditional
    return (p <= 0xFFFFFFFFFFFFFFFFULL) ? static_cast<u64>(p) : throw "overflow in checked_mul_u64";
#else
    if (a != 0ULL && b > 0xFFFFFFFFFFFFFFFFULL / a) {
        throw "overflow in checked_mul_u64";
    }
    return a * b;
#endif
}

/// Checked addition with overflow detection
consteval u64 checked_add_u64(u64 a, u64 b) {
    if (b > 0xFFFFFFFFFFFFFFFFULL - a) {
        throw "overflow in checked_add_u64";
    }
    return a + b;
}

} // namespace iro::util

// =============================================================================
// §2. Schema concepts (iro::schema) - Forward declarations
// =============================================================================

namespace iro::schema {

/// Tag concept: subjects with stable identities
template<class T>
concept Tag = iro::util::HasId<T>;

/// Element tag concept: types with size/alignment metadata
template<class E>
concept ElemTag = requires {
    { E::bytes } -> std::convertible_to<int>;
    { E::align } -> std::convertible_to<int>;
    { E::id }    -> std::convertible_to<iro::util::u64>;
};

/// Check if element has storage type (CUDA-specific)
template<class E>
concept ElemHasStorage = requires { typename E::storage_t; };

/// Layout concept (rank-aware; addressable regions only)
template<class L>
concept Layout =
    requires { { L::rank } -> std::convertible_to<int>; { L::id } -> std::convertible_to<iro::util::u64>; } &&
    (
        (L::rank == 1 && requires(int i) { { L::offset(i) } -> std::convertible_to<long long>; }) ||
        (L::rank == 2 && requires(int r, int c) { { L::offset(r, c) } -> std::convertible_to<long long>; })
    );

// Forward declarations for Token and Resource concepts (defined after token types)
template<class T> concept Token = requires {
    typename T::kind;
    typename T::subject;
    { T::id } -> std::convertible_to<iro::util::u64>;
} && iro::util::HasId<typename T::kind> && iro::util::HasId<typename T::subject>;

template<class R>
concept Resource = iro::util::HasId<R>;

template<class T>
struct is_dist : std::false_type {};

template<class T>
inline constexpr bool is_dist_v = is_dist<T>::value;

template<class T>
concept Dist = is_dist_v<T>;

} // namespace iro::schema

// =============================================================================
// §2. Structural payloads (iro::contract)
// =============================================================================

namespace iro::contract {

// -----------------------------------------------------------------------------
// §2.1 Tags (subjects) - Subject constructors
// -----------------------------------------------------------------------------

namespace subject {

/// Global subject (singleton)
struct global {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.subject.global");
};

/// Indexed subject (parametric by tag and index)
template<class Tag, int I>
struct indexed {
    static_assert(iro::schema::Tag<Tag>);
    static_assert(I >= 0);
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.subject.indexed"),
                           iro::util::mix_u64(Tag::id, static_cast<iro::util::u64>(I)));
};

/// Pair subject (compound of two tags)
template<class A, class B>
struct pair {
    static_assert(iro::schema::Tag<A> && iro::schema::Tag<B>);
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.subject.pair"),
                           iro::util::mix_u64(A::id, B::id));
};

} // namespace subject

// -----------------------------------------------------------------------------
// §2.2 Shapes (overflow-safe size; normative)
// -----------------------------------------------------------------------------

/// Shape descriptor with compile-time size computation
template<int... Dims>
struct Shape {
    static_assert(((Dims > 0) && ...));
    static constexpr int rank = sizeof...(Dims);

    static constexpr long long size = [] {
        iro::util::u64 p = 1ULL;
        ((p = iro::util::checked_mul_u64(p, static_cast<iro::util::u64>(Dims))), ...);
        if (p > static_cast<iro::util::u64>(0x7FFFFFFFFFFFFFFFULL)) {
            throw "Shape::size overflows signed 64-bit";
        }
        return static_cast<long long>(p);
    }();

    static constexpr iro::util::u64 id = [] {
        iro::util::u64 h = iro::util::fnv1a_64_cstr("iro.contract.Shape");
        ((h = iro::util::mix_u64(h, static_cast<iro::util::u64>(Dims))), ...);
        return h;
    }();

    // Accessor for individual dimensions
    template<int I>
    static constexpr int dim() {
        static_assert(I >= 0 && I < rank);
        constexpr int dims[] = {Dims...};
        return dims[I];
    }
};

// Specialization for 0-rank (scalar)
template<>
struct Shape<> {
    static constexpr int rank = 0;
    static constexpr long long size = 1;
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.contract.Shape.scalar");
};

// -----------------------------------------------------------------------------
// §2.4 Memory spaces
// -----------------------------------------------------------------------------

namespace space {

struct global {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.space.global");
};

struct local {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.space.local");
};

struct constant {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.space.constant");
};

struct texture {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.space.texture");
};

struct surface {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.space.surface");
};

struct shared {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.space.shared");
};

struct reg {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.space.reg");
};

struct fragment {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.space.fragment");
};

// Tensor memory (Blackwell)
struct tmem {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.space.tmem");
};

} // namespace space

// -----------------------------------------------------------------------------
// §2.5 Alignment
// -----------------------------------------------------------------------------

template<int Bytes>
struct Align {
    static_assert(Bytes > 0);
    static constexpr int bytes = Bytes;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.contract.Align"), static_cast<iro::util::u64>(Bytes));
};

// Common alignment aliases
using Align1   = Align<1>;
using Align2   = Align<2>;
using Align4   = Align<4>;
using Align8   = Align<8>;
using Align16  = Align<16>;
using Align32  = Align<32>;
using Align64  = Align<64>;
using Align128 = Align<128>;

// -----------------------------------------------------------------------------
// §2.6 Standard layouts
// -----------------------------------------------------------------------------

namespace layout {

/// Row-major layout (C order)
template<int Cols>
struct RowMajor {
    static_assert(Cols > 0);
    static constexpr int rank = 2;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.layout.RowMajor"), static_cast<iro::util::u64>(Cols));

    static constexpr long long offset(int r, int c) {
        return static_cast<long long>(r) * Cols + c;
    }
};

/// Column-major layout (Fortran order)
template<int Rows>
struct ColMajor {
    static_assert(Rows > 0);
    static constexpr int rank = 2;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.layout.ColMajor"), static_cast<iro::util::u64>(Rows));

    static constexpr long long offset(int r, int c) {
        return static_cast<long long>(c) * Rows + r;
    }
};

/// Contiguous 1D layout
struct Contiguous {
    static constexpr int rank = 1;
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.layout.Contiguous");

    static constexpr long long offset(int i) {
        return i;
    }
};

/// Strided 1D layout
template<int Stride>
struct Strided {
    static_assert(Stride != 0);
    static constexpr int rank = 1;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.layout.Strided"), static_cast<iro::util::u64>(Stride));

    static constexpr long long offset(int i) {
        return static_cast<long long>(i) * Stride;
    }
};

/// Swizzled layout for bank conflict avoidance
template<int BaseLayout, int SwizzleBits, int SwizzleShift>
struct Swizzled {
    static constexpr int rank = 2;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.layout.Swizzled"),
            iro::util::mix_u64(static_cast<iro::util::u64>(BaseLayout),
                iro::util::mix_u64(static_cast<iro::util::u64>(SwizzleBits),
                    static_cast<iro::util::u64>(SwizzleShift))));

    static constexpr long long offset(int r, int c) {
        // XOR-based swizzle pattern
        int swizzled_c = c ^ ((r >> SwizzleShift) & ((1 << SwizzleBits) - 1));
        return static_cast<long long>(r) * BaseLayout + swizzled_c;
    }
};

/// Swizzled layout for column-major tiles (bank conflict avoidance)
template<int BaseLayout, int SwizzleBits, int SwizzleShift>
struct SwizzledColMajor {
    static constexpr int rank = 2;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.layout.SwizzledColMajor"),
            iro::util::mix_u64(static_cast<iro::util::u64>(BaseLayout),
                iro::util::mix_u64(static_cast<iro::util::u64>(SwizzleBits),
                    static_cast<iro::util::u64>(SwizzleShift))));

    static constexpr long long offset(int r, int c) {
        // XOR-based swizzle pattern on the row index for column-major storage
        int swizzled_r = r ^ ((c >> SwizzleShift) & ((1 << SwizzleBits) - 1));
        return static_cast<long long>(c) * BaseLayout + swizzled_r;
    }
};

} // namespace layout

// -----------------------------------------------------------------------------
// §2.7 Tile payload (purely structural)
// -----------------------------------------------------------------------------

template<class ShapeT, class ElemT, class LayoutT, class SpaceT, class AlignT>
struct Tile {
    static_assert(iro::schema::ElemTag<ElemT>);
    static_assert(iro::schema::Layout<LayoutT>);
    static_assert(AlignT::bytes >= ElemT::align);

    using shape = ShapeT;
    using elem = ElemT;
    using layout = LayoutT;
    using space = SpaceT;
    using align = AlignT;

    // Use checked arithmetic to prevent overflow (§1.3, normative)
    static constexpr long long bytes = [] {
        iro::util::u64 result = iro::util::checked_mul_u64(
            static_cast<iro::util::u64>(ShapeT::size),
            static_cast<iro::util::u64>(ElemT::bytes)
        );
        if (result > static_cast<iro::util::u64>(0x7FFFFFFFFFFFFFFFULL)) {
            throw "Tile::bytes overflows signed 64-bit";
        }
        return static_cast<long long>(result);
    }();

    static constexpr iro::util::u64 id = [] {
        iro::util::u64 h = iro::util::fnv1a_64_cstr("iro.contract.Tile");
        h = iro::util::mix_u64(h, ShapeT::id);
        h = iro::util::mix_u64(h, ElemT::id);
        h = iro::util::mix_u64(h, LayoutT::id);
        h = iro::util::mix_u64(h, SpaceT::id);
        h = iro::util::mix_u64(h, AlignT::id);
        return h;
    }();
};

// -----------------------------------------------------------------------------
// §2.8 Tensor reference descriptor (strided views; structural)
// -----------------------------------------------------------------------------

template<int Rank, class ElemT, class SpaceT, class AlignT>
struct TensorRefDesc {
    static_assert(Rank > 0);
    static_assert(iro::schema::ElemTag<ElemT>);

    using elem = ElemT;
    using space = SpaceT;
    using align = AlignT;
    static constexpr int rank = Rank;

    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.contract.TensorRefDesc"),
            iro::util::mix_u64(static_cast<iro::util::u64>(Rank),
                iro::util::mix_u64(ElemT::id,
                    iro::util::mix_u64(SpaceT::id, AlignT::id))));
};

// -----------------------------------------------------------------------------
// §6.1 Fragment payload descriptor (opaque fragments; no Layout)
// -----------------------------------------------------------------------------

template<class ShapeT, class ElemT, class DistT, int Count = ShapeT::size>
struct FragmentDesc {
    static_assert(iro::schema::ElemTag<ElemT>);
    static_assert(iro::schema::Dist<DistT>, "FragmentDesc requires a schema::Dist type");
    static_assert(Count > 0, "FragmentDesc: Count must be positive");
    static_assert(Count <= static_cast<int>(ShapeT::size),
                  "FragmentDesc: Count cannot exceed Shape size");

    using shape = ShapeT;
    using elem = ElemT;
    using dist = DistT;
    static constexpr int count = Count;

    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.contract.FragmentDesc"),
            iro::util::mix_u64(ShapeT::id,
                iro::util::mix_u64(ElemT::id,
                    iro::util::mix_u64(DistT::id, static_cast<iro::util::u64>(Count)))));
};

// -----------------------------------------------------------------------------
// §6.1b Scalar payload descriptor (opaque scalar/register value)
// -----------------------------------------------------------------------------

template<class ElemT, class DistT>
struct ScalarDesc {
    static_assert(iro::schema::ElemTag<ElemT>);
    static_assert(iro::schema::Dist<DistT>, "ScalarDesc requires a schema::Dist type");

    using elem = ElemT;
    using dist = DistT;
    static constexpr int bytes = ElemT::bytes;
    static constexpr int align = ElemT::align;

    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.contract.ScalarDesc"),
            iro::util::mix_u64(ElemT::id, DistT::id));
};

// -----------------------------------------------------------------------------
// §6.1c Vector payload descriptor (opaque vector/register value)
// -----------------------------------------------------------------------------

template<class ElemT, int N, class DistT>
struct VectorDesc {
    static_assert(iro::schema::ElemTag<ElemT>);
    static_assert(iro::schema::Dist<DistT>, "VectorDesc requires a schema::Dist type");
    static_assert(N > 0);

    using elem = ElemT;
    using dist = DistT;
    static constexpr int lanes = N;
    static constexpr int bytes = ElemT::bytes * N;
    static constexpr int align = ElemT::align;

    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.contract.VectorDesc"),
            iro::util::mix_u64(ElemT::id,
                iro::util::mix_u64(static_cast<iro::util::u64>(N), DistT::id)));
};

// -----------------------------------------------------------------------------
// §6.1d Mask payload descriptor (explicit execution mask value)
// -----------------------------------------------------------------------------

template<int Width, class DistT>
struct MaskDesc {
    static_assert(iro::schema::Dist<DistT>, "MaskDesc requires a schema::Dist type");
    static_assert(Width > 0);

    using elem = iro::elem::u32;
    using dist = DistT;
    static constexpr int width = Width;
    static constexpr int words = (Width + 31) / 32;
    // Avoid dependency on full elem definition at this point.
    static constexpr int bytes = words * 4;
    static constexpr int align = 4;

    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.contract.MaskDesc"),
            iro::util::mix_u64(static_cast<iro::util::u64>(Width), DistT::id));
};

} // namespace iro::contract

// =============================================================================
// §3. Execution groups vs visibility scopes
// =============================================================================

// -----------------------------------------------------------------------------
// §3.1 Exec groups (who executes; no lanes)
// -----------------------------------------------------------------------------

namespace iro::exec {

struct lane {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.exec.lane");
};

struct warp {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.exec.warp");
};

template<int Warps>
struct warpgroup_t {
    static_assert(Warps >= 4, "warpgroup_t requires at least 4 warps");
    static constexpr int warps = Warps;
    static constexpr auto id = iro::util::mix_u64(
        iro::util::fnv1a_64_cstr("iro.exec.warpgroup"),
        static_cast<iro::util::u64>(Warps));
};

using warpgroup = warpgroup_t<4>;

struct block {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.exec.block");
};

struct cluster {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.exec.cluster");
};

// CTA-group execution (Blackwell tcgen05)
struct cta_group1 {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.exec.cta_group1");
};

struct cta_group2 {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.exec.cta_group2");
};

template<class ExecGroup>
struct is_warpgroup : std::false_type {};

template<int Warps>
struct is_warpgroup<warpgroup_t<Warps>> : std::true_type {};

template<class ExecGroup>
inline constexpr bool is_warpgroup_v = is_warpgroup<ExecGroup>::value;

template<class ExecGroup>
struct warpgroup_warps : std::integral_constant<int, 0> {};

template<int Warps>
struct warpgroup_warps<warpgroup_t<Warps>> : std::integral_constant<int, Warps> {};

} // namespace iro::exec

// -----------------------------------------------------------------------------
// §3.2 Scopes (who can observe)
// -----------------------------------------------------------------------------

namespace iro::scope {

struct lane {
    static constexpr int level = 0;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.scope.lane");
};

struct warp {
    static constexpr int level = 1;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.scope.warp");
};

struct warpgroup {
    static constexpr int level = 2;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.scope.warpgroup");
};

struct block {
    static constexpr int level = 3;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.scope.block");
};

struct cluster {
    static constexpr int level = 4;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.scope.cluster");
};

struct device {
    static constexpr int level = 5;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.scope.device");
};

// -----------------------------------------------------------------------------
// §3.3 Minimum required visibility scope per exec group (normative)
// -----------------------------------------------------------------------------

template<class ExecGroup> struct min_scope_for;

template<> struct min_scope_for<iro::exec::lane> {
    using type = iro::scope::lane;
};

template<> struct min_scope_for<iro::exec::warp> {
    using type = iro::scope::warp;
};

template<int Warps> struct min_scope_for<iro::exec::warpgroup_t<Warps>> {
    using type = iro::scope::warpgroup;
};

template<> struct min_scope_for<iro::exec::block> {
    using type = iro::scope::block;
};

template<> struct min_scope_for<iro::exec::cluster> {
    using type = iro::scope::cluster;
};

template<> struct min_scope_for<iro::exec::cta_group1> {
    using type = iro::scope::cluster;
};

template<> struct min_scope_for<iro::exec::cta_group2> {
    using type = iro::scope::cluster;
};

template<class ExecGroup>
using min_scope_for_t = typename min_scope_for<ExecGroup>::type;

} // namespace iro::scope

// =============================================================================
// §4. Protocol tokens (iro::token)
// =============================================================================

// ---------------------------------------------------------------------------
// Memory ordering and determinism tags (normative)
// ---------------------------------------------------------------------------

namespace iro::memory_order {

struct relaxed {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.memory_order.relaxed");
};

struct acquire {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.memory_order.acquire");
};

struct release {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.memory_order.release");
};

struct acq_rel {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.memory_order.acq_rel");
};

struct seq_cst {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.memory_order.seq_cst");
};

} // namespace iro::memory_order

namespace iro::determinism {

struct fast {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.determinism.fast");
};

struct reproducible {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.determinism.reproducible");
};

} // namespace iro::determinism

namespace iro::token {

// -----------------------------------------------------------------------------
// §4.2 Token bundles (alias-only; LLM-friendly)
// -----------------------------------------------------------------------------

template<class... Ts>
struct bundle {
    using list = iro::util::type_list<Ts...>;
};

template<class B>
using bundle_list = typename B::list;

// -----------------------------------------------------------------------------
// §4.4 Standard token families (normative)
// -----------------------------------------------------------------------------

// §4.4.1 visible_at
struct kind_visible_at {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.token.kind_visible_at");
};

template<class SubjectT, class ScopeT>
struct visible_at {
    using kind = kind_visible_at;
    using subject = SubjectT;
    using scope = ScopeT;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.token.visible_at"),
            iro::util::mix_u64(SubjectT::id, ScopeT::id));
};

// §4.4.2 lanes_valid
struct kind_lanes_valid {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.token.kind_lanes_valid");
};

template<class SubjectT, int N>
struct lanes_valid {
    static_assert(0 <= N && N <= 32);
    using kind = kind_lanes_valid;
    using subject = SubjectT;
    static constexpr int lanes = N;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.token.lanes_valid"),
            iro::util::mix_u64(SubjectT::id, static_cast<iro::util::u64>(N)));
};

// §4.4.2a issued_by_lane0 (single-lane issuance for collective side-effects)
struct kind_issued_by_lane0 {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.token.kind_issued_by_lane0");
};

template<class SubjectT>
struct issued_by_lane0 {
    using kind = kind_issued_by_lane0;
    using subject = SubjectT;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.token.issued_by_lane0"), SubjectT::id);
};

// §4.4.2b warps_valid
struct kind_warps_valid {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.token.kind_warps_valid");
};

template<class SubjectT, int N>
struct warps_valid {
    static_assert(0 <= N && N <= 32);
    using kind = kind_warps_valid;
    using subject = SubjectT;
    static constexpr int warps = N;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.token.warps_valid"),
            iro::util::mix_u64(SubjectT::id, static_cast<iro::util::u64>(N)));
};

// §4.4.2c warpgroup_participates
struct kind_warpgroup_participates {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.token.kind_warpgroup_participates");
};

template<class SubjectT, int N>
struct warpgroup_participates {
    static_assert(0 <= N && N <= 32);
    using kind = kind_warpgroup_participates;
    using subject = SubjectT;
    static constexpr int warps = N;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.token.warpgroup_participates"),
            iro::util::mix_u64(SubjectT::id, static_cast<iro::util::u64>(N)));
};

// §4.4.2d mask_at (explicit execution mask)
struct kind_mask_at {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.token.kind_mask_at");
};

template<class SubjectT, class ScopeT>
struct mask_at {
    using kind = kind_mask_at;
    using subject = SubjectT;
    using scope = ScopeT;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.token.mask_at"),
            iro::util::mix_u64(SubjectT::id, ScopeT::id));
};

// §4.4.3 alive
struct kind_alive {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.token.kind_alive");
};

namespace lifetime {

struct instruction {
    static constexpr int level = 0;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.lifetime.instruction");
};

struct warp {
    static constexpr int level = 1;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.lifetime.warp");
};

struct warpgroup {
    static constexpr int level = 2;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.lifetime.warpgroup");
};

struct block {
    static constexpr int level = 3;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.lifetime.block");
};

struct cluster {
    static constexpr int level = 4;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.lifetime.cluster");
};

template<int Stages>
struct pipeline {
    static_assert(Stages > 0);
    static constexpr int level = 5;
    static constexpr int stages = Stages;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.lifetime.pipeline"),
            static_cast<iro::util::u64>(Stages));
};

} // namespace lifetime

template<class SubjectT, class LifetimeT>
struct alive {
    using kind = kind_alive;
    using subject = SubjectT;
    using lifetime = LifetimeT;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.token.alive"),
            iro::util::mix_u64(SubjectT::id, LifetimeT::id));
};

// §4.4.4 slot_state
struct kind_slot_state {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.token.kind_slot_state");
};

namespace state {

struct free {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.state.free");
};

struct filling {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.state.filling");
};

struct ready {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.state.ready");
};

struct used {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.state.used");
};

} // namespace state

template<class SlotSubjectT, class StateT>
struct slot_state {
    using kind = kind_slot_state;
    using subject = SlotSubjectT;
    using state = StateT;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.token.slot_state"),
            iro::util::mix_u64(SlotSubjectT::id, StateT::id));
};

// §4.4.5 lease (arena exclusivity)
struct kind_lease {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.token.kind_lease");
};

template<class ArenaTag>
struct lease {
    using kind = kind_lease;
    using subject = ArenaTag;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.token.lease"), ArenaTag::id);
};

// §4.4.6 sync_at (explicit synchronization proof)
struct kind_sync_at {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.token.kind_sync_at");
};

template<class SubjectT, class ScopeT>
struct sync_at {
    using kind = kind_sync_at;
    using subject = SubjectT;
    using scope = ScopeT;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.token.sync_at"),
            iro::util::mix_u64(SubjectT::id, ScopeT::id));
};

// §4.4.7 memory_order (explicit ordering/fence semantics)
struct kind_memory_order {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.token.kind_memory_order");
};

template<class SubjectT, class OrderT, class ScopeT>
struct memory_order {
    using kind = kind_memory_order;
    using subject = SubjectT;
    using order = OrderT;
    using scope = ScopeT;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.token.memory_order"),
            iro::util::mix_u64(SubjectT::id,
                iro::util::mix_u64(OrderT::id, ScopeT::id)));
};

// §4.4.7b version (explicit dependency versioning)
struct kind_version {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.token.kind_version");
};

template<class SubjectT, int Version>
struct version {
    using kind = kind_version;
    using subject = SubjectT;
    static constexpr int value = Version;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.token.version"),
            iro::util::mix_u64(SubjectT::id, static_cast<iro::util::u64>(Version)));
};

// §4.4.8 determinism (explicit reproducibility contract)
struct kind_determinism {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.token.kind_determinism");
};

template<class SubjectT, class ModeT>
struct determinism {
    using kind = kind_determinism;
    using subject = SubjectT;
    using mode = ModeT;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.token.determinism"),
            iro::util::mix_u64(SubjectT::id, ModeT::id));
};

} // namespace iro::token

// =============================================================================
// §5. Resources (iro::contract::res)
// =============================================================================

namespace iro::contract::res {

/// Shared memory region resource
template<class Tag, long long Bytes, int AlignBytes>
struct smem_region {
    static_assert(iro::schema::Tag<Tag>);
    static_assert(Bytes >= 0);
    static_assert(AlignBytes > 0);

    static constexpr long long bytes = Bytes;
    static constexpr int align = AlignBytes;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.smem_region"),
            iro::util::mix_u64(Tag::id,
                iro::util::mix_u64(static_cast<iro::util::u64>(Bytes),
                    static_cast<iro::util::u64>(AlignBytes))));
};

/// Tensor memory region resource (SM100+)
template<class Tag, long long Bytes, int AlignBytes>
struct tmem_region {
    static_assert(iro::schema::Tag<Tag>);
    static_assert(Bytes >= 0);
    static_assert(AlignBytes > 0);

    static constexpr long long bytes = Bytes;
    static constexpr int align = AlignBytes;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.tmem_region"),
            iro::util::mix_u64(Tag::id,
                iro::util::mix_u64(static_cast<iro::util::u64>(Bytes),
                    static_cast<iro::util::u64>(AlignBytes))));
};

/// Shared memory pipeline resource (multi-stage)
template<class Tag, int Slots, long long BytesPerSlot, int AlignBytes>
struct smem_pipeline {
    static_assert(iro::schema::Tag<Tag>);
    static_assert(Slots > 0);
    static_assert(BytesPerSlot >= 0);
    static_assert(AlignBytes > 0);

    static constexpr int slots = Slots;
    static constexpr long long bytes_per_slot = BytesPerSlot;
    static constexpr int align = AlignBytes;
    static constexpr long long total_bytes = Slots * BytesPerSlot;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.smem_pipeline"),
            iro::util::mix_u64(Tag::id,
                iro::util::mix_u64(static_cast<iro::util::u64>(Slots),
                    iro::util::mix_u64(static_cast<iro::util::u64>(BytesPerSlot),
                        static_cast<iro::util::u64>(AlignBytes)))));
};

/// Tensor memory arena resource (for arena-style allocation)
template<class Tag, long long Bytes, int AlignBytes>
struct tmem_arena {
    static_assert(iro::schema::Tag<Tag>);
    static_assert(Bytes >= 0);
    static_assert(AlignBytes > 0);

    static constexpr long long bytes = Bytes;
    static constexpr int align = AlignBytes;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.tmem_arena"),
            iro::util::mix_u64(Tag::id,
                iro::util::mix_u64(static_cast<iro::util::u64>(Bytes),
                    static_cast<iro::util::u64>(AlignBytes))));
};

/// Shared memory arena resource (for arena-style allocation)
template<class Tag, long long Bytes, int AlignBytes>
struct smem_arena {
    static_assert(iro::schema::Tag<Tag>);
    static_assert(Bytes >= 0);
    static_assert(AlignBytes > 0);

    static constexpr long long bytes = Bytes;
    static constexpr int align = AlignBytes;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.smem_arena"),
            iro::util::mix_u64(Tag::id,
                iro::util::mix_u64(static_cast<iro::util::u64>(Bytes),
                    static_cast<iro::util::u64>(AlignBytes))));
};

/// Block threads requirement (launch constraint)
template<int Threads>
struct block_threads {
    static_assert(Threads > 0);
    static constexpr int threads = Threads;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.block_threads"),
            static_cast<iro::util::u64>(Threads));
};

/// Block threads multiple-of requirement (launch constraint)
template<int Threads>
struct block_threads_multiple_of {
    static_assert(Threads > 0);
    static constexpr int multiple = Threads;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.block_threads_multiple_of"),
            static_cast<iro::util::u64>(Threads));
};

/// Warpgroup layout requirement (warpgroup shape contract)
template<int WarpsPerGroup, int WarpSize = 32>
struct warpgroup_layout {
    static_assert(WarpsPerGroup > 0);
    static_assert(WarpSize > 0);
    static constexpr int warps = WarpsPerGroup;
    static constexpr int warp_size = WarpSize;
    static constexpr int threads = WarpsPerGroup * WarpSize;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.warpgroup_layout"),
            iro::util::mix_u64(static_cast<iro::util::u64>(WarpsPerGroup),
                static_cast<iro::util::u64>(WarpSize)));
};

/// Warpgroup linear mapping in threadIdx.x (layout mapping contract)
struct warpgroup_linear_x {
    static constexpr iro::util::u64 id =
        iro::util::fnv1a_64_cstr("iro.res.warpgroup_linear_x");
};

/// Warpgroup count requirement (number of warpgroups participating in the block)
template<int Count>
struct warpgroup_count {
    static_assert(Count >= 1, "warpgroup_count requires Count >= 1");
    static constexpr int count = Count;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.warpgroup_count"),
            static_cast<iro::util::u64>(Count));
};

/// Cache policy requirement (load/store policy)
template<class Policy>
struct cache_policy {
    static_assert(iro::util::HasId<Policy>);
    using policy = Policy;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.cache_policy"), Policy::id);
};

/// Register usage requirement (launch constraint / budget accounting)
template<int Regs>
struct reg_usage {
    static_assert(Regs >= 0);
    static constexpr int regs = Regs;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.reg_usage"),
            static_cast<iro::util::u64>(Regs));
};

/// Cluster dimensions requirement (launch constraint)
template<int X, int Y, int Z>
struct cluster_dims {
    static_assert(X > 0 && Y > 0 && Z > 0);
    static constexpr int x = X;
    static constexpr int y = Y;
    static constexpr int z = Z;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.cluster_dims"),
            iro::util::mix_u64(static_cast<iro::util::u64>(X),
                iro::util::mix_u64(static_cast<iro::util::u64>(Y),
                    static_cast<iro::util::u64>(Z))));
};

/// Named barrier resource
template<class Tag>
struct named_barrier {
    static_assert(iro::schema::Tag<Tag>);
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.named_barrier"), Tag::id);
};

/// Warpgroup barrier resource (warpgroup-scoped synchronization)
template<class Tag, int BaseId>
struct warpgroup_barrier {
    static_assert(iro::schema::Tag<Tag>);
    static_assert(BaseId >= 1 && BaseId <= 8, "warpgroup_barrier BaseId must be 1..8");
    using subject = Tag;
    static constexpr int base_id = BaseId;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.warpgroup_barrier"),
            iro::util::mix_u64(Tag::id, static_cast<iro::util::u64>(BaseId)));
};

/// Slot subject (refers to a specific slot in a pipeline)
template<class PipelineRes, int SlotIdx>
struct slot_subject {
    static_assert(iro::schema::Resource<PipelineRes>);
    static_assert(SlotIdx >= 0 && SlotIdx < PipelineRes::slots);
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.slot_subject"),
            iro::util::mix_u64(PipelineRes::id, static_cast<iro::util::u64>(SlotIdx)));
};

/// Register file pressure hint
template<int NumRegs>
struct reg_pressure {
    static_assert(NumRegs >= 0);
    static constexpr int regs = NumRegs;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.res.reg_pressure"),
            static_cast<iro::util::u64>(NumRegs));
};

} // namespace iro::contract::res

// Forward recipe declarations (used by iro::verify specializations below)
namespace iro::recipe {
template<class InElem, class AccElem, class OutElem, int VecBytes, class MathPolicy,
         class Fp8Policy, class ScaleElem, int ScaleVec>
struct Precision;
template<class ElemA, class ElemB, class AccElem, class OutElem, int VecBytes, class MathPolicy,
         class Fp8Policy, class ScaleElemA, int ScaleVecA, class ScaleElemB, int ScaleVecB>
struct PrecisionAB;
template<class BaseRecipe, class Pattern>
struct Sparse;
template<class R>
struct Accumulate;
template<class R>
struct is_accumulate : std::false_type {};
template<class R>
inline constexpr bool is_accumulate_v = is_accumulate<R>::value;
template<class R>
struct unwrap_accumulate { using type = R; };
template<class R>
using unwrap_accumulate_t = typename unwrap_accumulate<R>::type;
} // namespace iro::recipe

// =============================================================================
// §4.5 Token verification (iro::verify)
// =============================================================================

namespace iro::recipe {
struct no_scale;
template<class R> struct is_precision_ab;
} // namespace iro::recipe

namespace iro::verify {

// -----------------------------------------------------------------------------
// Scope subsumption
// -----------------------------------------------------------------------------

consteval bool scope_subsumes(int prov, int req) {
    return prov >= req;
}

template<class ProvScope, class ReqScope>
consteval bool scope_subsumes_t() {
    return scope_subsumes(ProvScope::level, ReqScope::level);
}

// -----------------------------------------------------------------------------
// Lifetime subsumption
// -----------------------------------------------------------------------------

namespace detail {

template<class T>
struct is_pipeline_lifetime : std::false_type {};

template<int Stages>
struct is_pipeline_lifetime<iro::token::lifetime::pipeline<Stages>> : std::true_type {};

template<class T>
inline constexpr bool is_pipeline_lifetime_v = is_pipeline_lifetime<T>::value;

template<class T>
struct pipeline_stages { static constexpr int value = 0; };

template<int Stages>
struct pipeline_stages<iro::token::lifetime::pipeline<Stages>> {
    static constexpr int value = Stages;
};

template<class T>
inline constexpr int pipeline_stages_v = pipeline_stages<T>::value;

} // namespace detail

template<class ProvLifetime, class ReqLifetime>
consteval bool lifetime_subsumes() {
    // Pipeline lifetimes: stages subsume
    if constexpr (detail::is_pipeline_lifetime_v<ProvLifetime> &&
                  detail::is_pipeline_lifetime_v<ReqLifetime>) {
        return detail::pipeline_stages_v<ProvLifetime> >= detail::pipeline_stages_v<ReqLifetime>;
    }
    // Otherwise: level subsumption
    else {
        return ProvLifetime::level >= ReqLifetime::level;
    }
}

// -----------------------------------------------------------------------------
// Memory order subsumption
// -----------------------------------------------------------------------------

namespace detail {

template<class OrderT>
consteval int memory_order_rank() {
    if constexpr (std::is_same_v<OrderT, iro::memory_order::relaxed>) {
        return 0;
    } else if constexpr (std::is_same_v<OrderT, iro::memory_order::acquire>) {
        return 1;
    } else if constexpr (std::is_same_v<OrderT, iro::memory_order::release>) {
        return 1;
    } else if constexpr (std::is_same_v<OrderT, iro::memory_order::acq_rel>) {
        return 2;
    } else if constexpr (std::is_same_v<OrderT, iro::memory_order::seq_cst>) {
        return 3;
    } else {
        return -1;
    }
}

template<class ProvOrder, class ReqOrder>
consteval bool memory_order_subsumes() {
    if constexpr (std::is_same_v<ProvOrder, ReqOrder>) {
        return true;
    } else if constexpr (std::is_same_v<ProvOrder, iro::memory_order::seq_cst>) {
        return true;
    } else if constexpr (std::is_same_v<ProvOrder, iro::memory_order::acq_rel>) {
        return std::is_same_v<ReqOrder, iro::memory_order::acquire> ||
               std::is_same_v<ReqOrder, iro::memory_order::release> ||
               std::is_same_v<ReqOrder, iro::memory_order::acq_rel>;
    } else if constexpr (std::is_same_v<ProvOrder, iro::memory_order::acquire>) {
        return std::is_same_v<ReqOrder, iro::memory_order::acquire>;
    } else if constexpr (std::is_same_v<ProvOrder, iro::memory_order::release>) {
        return std::is_same_v<ReqOrder, iro::memory_order::release>;
    } else if constexpr (std::is_same_v<ProvOrder, iro::memory_order::relaxed>) {
        return std::is_same_v<ReqOrder, iro::memory_order::relaxed>;
    } else {
        return false;
    }
}

// -----------------------------------------------------------------------------
// Determinism subsumption
// -----------------------------------------------------------------------------

template<class ProvMode, class ReqMode>
consteval bool determinism_subsumes() {
    if constexpr (std::is_same_v<ProvMode, ReqMode>) {
        return true;
    } else if constexpr (std::is_same_v<ProvMode, iro::determinism::reproducible> &&
                         std::is_same_v<ReqMode, iro::determinism::fast>) {
        return true;
    } else {
        return false;
    }
}

} // namespace detail

// -----------------------------------------------------------------------------
// Token canonicality checking
// -----------------------------------------------------------------------------

namespace detail {

template<class List>
struct token_ids;

template<>
struct token_ids<iro::util::type_list<>> {
    static constexpr bool has_duplicate = false;
};

template<class T, class... Ts>
struct token_ids<iro::util::type_list<T, Ts...>> {
    static constexpr iro::util::u64 first_id =
        iro::util::mix_u64(T::kind::id, T::subject::id);

    template<class U>
    static consteval bool matches() {
        return iro::util::mix_u64(U::kind::id, U::subject::id) == first_id;
    }

    static constexpr bool has_duplicate =
        (matches<Ts>() || ...) || token_ids<iro::util::type_list<Ts...>>::has_duplicate;
};

} // namespace detail

/// Check if token list is canonical (no duplicate kind+subject pairs)
template<class TokenList>
consteval bool token_list_canonical() {
    return !detail::token_ids<TokenList>::has_duplicate;
}

// -----------------------------------------------------------------------------
// Token list queries
// -----------------------------------------------------------------------------

namespace detail {

template<class List, class Kind, class Subject>
struct find_token_impl;

template<class Kind, class Subject>
struct find_token_impl<iro::util::type_list<>, Kind, Subject> {
    static constexpr bool found = false;
    using type = void;
};

template<class T, class... Ts, class Kind, class Subject>
struct find_token_impl<iro::util::type_list<T, Ts...>, Kind, Subject> {
    static constexpr bool this_matches =
        std::is_same_v<typename T::kind, Kind> && std::is_same_v<typename T::subject, Subject>;
    static constexpr bool found =
        this_matches || find_token_impl<iro::util::type_list<Ts...>, Kind, Subject>::found;
    using type = std::conditional_t<
        this_matches,
        T,
        typename find_token_impl<iro::util::type_list<Ts...>, Kind, Subject>::type
    >;
};

} // namespace detail

/// Check if list has token with given kind and subject
template<class List, class Kind, class Subject>
consteval bool has_token_kind_subject() {
    return detail::find_token_impl<List, Kind, Subject>::found;
}

/// Get token with given kind and subject (undefined if not found)
template<class List, class Kind, class Subject>
using get_token_kind_subject = typename detail::find_token_impl<List, Kind, Subject>::type;

// -----------------------------------------------------------------------------
// Token list canonicalization helpers (explicit; no implicit adapters)
// -----------------------------------------------------------------------------

namespace detail {

template<class Existing, class Incoming, class Enable = void>
struct canonical_pick {
    static_assert(std::is_same_v<typename Existing::kind, typename Incoming::kind>);
    static_assert(std::is_same_v<typename Existing::subject, typename Incoming::subject>);

    using type = Existing; // Default: keep existing
};

// visible_at: keep max scope
template<class E, class I>
struct canonical_pick_visible_at {
    using type = std::conditional_t<(I::scope::level > E::scope::level), I, E>;
};

template<class Existing, class Incoming>
struct canonical_pick<Existing, Incoming,
    std::enable_if_t<std::is_same_v<typename Existing::kind, iro::token::kind_visible_at>>> {
    using type = typename canonical_pick_visible_at<Existing, Incoming>::type;
};

// mask_at: keep max scope
template<class Existing, class Incoming>
struct canonical_pick<Existing, Incoming,
    std::enable_if_t<std::is_same_v<typename Existing::kind, iro::token::kind_mask_at>>> {
    using type = std::conditional_t<(Incoming::scope::level > Existing::scope::level), Incoming, Existing>;
};

// sync_at: keep max scope
template<class Existing, class Incoming>
struct canonical_pick<Existing, Incoming,
    std::enable_if_t<std::is_same_v<typename Existing::kind, iro::token::kind_sync_at>>> {
    using type = std::conditional_t<(Incoming::scope::level > Existing::scope::level), Incoming, Existing>;
};

// lanes_valid: keep max lanes
template<class Existing, class Incoming>
struct canonical_pick<Existing, Incoming,
    std::enable_if_t<std::is_same_v<typename Existing::kind, iro::token::kind_lanes_valid>>> {
    using type = std::conditional_t<(Incoming::lanes > Existing::lanes), Incoming, Existing>;
};

// warps_valid: keep max warps
template<class Existing, class Incoming>
struct canonical_pick<Existing, Incoming,
    std::enable_if_t<std::is_same_v<typename Existing::kind, iro::token::kind_warps_valid>>> {
    using type = std::conditional_t<(Incoming::warps > Existing::warps), Incoming, Existing>;
};

// warpgroup_participates: keep max warps
template<class Existing, class Incoming>
struct canonical_pick<Existing, Incoming,
    std::enable_if_t<std::is_same_v<typename Existing::kind, iro::token::kind_warpgroup_participates>>> {
    using type = std::conditional_t<(Incoming::warps > Existing::warps), Incoming, Existing>;
};

// alive: keep max lifetime; pipeline only comparable to pipeline
template<class Existing, class Incoming>
struct canonical_pick<Existing, Incoming,
    std::enable_if_t<std::is_same_v<typename Existing::kind, iro::token::kind_alive>>> {
    static constexpr bool e_pipeline = detail::is_pipeline_lifetime_v<typename Existing::lifetime>;
    static constexpr bool i_pipeline = detail::is_pipeline_lifetime_v<typename Incoming::lifetime>;

    static_assert(e_pipeline == i_pipeline,
        "token canonicalization: alive pipeline and non-pipeline lifetimes are incomparable");

    using type = std::conditional_t<
        (e_pipeline && (detail::pipeline_stages_v<typename Incoming::lifetime> >
                        detail::pipeline_stages_v<typename Existing::lifetime>)) ||
        (!e_pipeline && (Incoming::lifetime::level > Existing::lifetime::level)),
        Incoming,
        Existing
    >;
};

// slot_state: exact match only
template<class Existing, class Incoming>
struct canonical_pick<Existing, Incoming,
    std::enable_if_t<std::is_same_v<typename Existing::kind, iro::token::kind_slot_state>>> {
    static_assert(std::is_same_v<Existing, Incoming>,
        "token canonicalization: slot_state duplicates must match exactly");
    using type = Existing;
};

// lease: exact match only
template<class Existing, class Incoming>
struct canonical_pick<Existing, Incoming,
    std::enable_if_t<std::is_same_v<typename Existing::kind, iro::token::kind_lease>>> {
    static_assert(std::is_same_v<Existing, Incoming>,
        "token canonicalization: lease duplicates must match exactly");
    using type = Existing;
};

// memory_order: keep stronger order if comparable (order+scope subsumption)
template<class Existing, class Incoming>
struct canonical_pick<Existing, Incoming,
    std::enable_if_t<std::is_same_v<typename Existing::kind, iro::token::kind_memory_order>>> {
    static constexpr bool incoming_subsumes =
        detail::memory_order_subsumes<typename Incoming::order, typename Existing::order>() &&
        iro::verify::scope_subsumes(Incoming::scope::level, Existing::scope::level);
    static constexpr bool existing_subsumes =
        detail::memory_order_subsumes<typename Existing::order, typename Incoming::order>() &&
        iro::verify::scope_subsumes(Existing::scope::level, Incoming::scope::level);
    static_assert(incoming_subsumes || existing_subsumes,
        "token canonicalization: memory_order duplicates are incomparable");
    using type = std::conditional_t<incoming_subsumes, Incoming, Existing>;
};

// version: exact match only
template<class Existing, class Incoming>
struct canonical_pick<Existing, Incoming,
    std::enable_if_t<std::is_same_v<typename Existing::kind, iro::token::kind_version>>> {
    static_assert(std::is_same_v<Existing, Incoming>,
        "token canonicalization: version duplicates must match exactly");
    using type = Existing;
};

// determinism: keep stronger mode (reproducible > fast)
template<class Existing, class Incoming>
struct canonical_pick<Existing, Incoming,
    std::enable_if_t<std::is_same_v<typename Existing::kind, iro::token::kind_determinism>>> {
    static constexpr bool incoming_subsumes =
        detail::determinism_subsumes<typename Incoming::mode, typename Existing::mode>();
    static constexpr bool existing_subsumes =
        detail::determinism_subsumes<typename Existing::mode, typename Incoming::mode>();
    static_assert(incoming_subsumes || existing_subsumes,
        "token canonicalization: determinism duplicates are incomparable");
    using type = std::conditional_t<incoming_subsumes, Incoming, Existing>;
};

template<class List, class Kind, class Subject, class NewToken>
struct replace_token_kind_subject;

template<class Kind, class Subject, class NewToken>
struct replace_token_kind_subject<iro::util::type_list<>, Kind, Subject, NewToken> {
    using type = iro::util::type_list<>;
};

template<class T, class... Ts, class Kind, class Subject, class NewToken>
struct replace_token_kind_subject<iro::util::type_list<T, Ts...>, Kind, Subject, NewToken> {
    static constexpr bool this_matches =
        std::is_same_v<typename T::kind, Kind> && std::is_same_v<typename T::subject, Subject>;
    using type = std::conditional_t<
        this_matches,
        iro::util::type_list<NewToken, Ts...>,
        iro::util::prepend_t<
            typename replace_token_kind_subject<iro::util::type_list<Ts...>, Kind, Subject, NewToken>::type,
            T
        >
    >;
};

template<class List, class Token>
struct canonical_insert;

template<class List, class Token>
struct canonical_insert {
    using kind = typename Token::kind;
    using subject = typename Token::subject;

    static constexpr bool exists = has_token_kind_subject<List, kind, subject>();
    using type = std::conditional_t<
        !exists,
        iro::util::append_t<List, Token>,
        typename replace_token_kind_subject<
            List,
            kind,
            subject,
            typename canonical_pick<
                get_token_kind_subject<List, kind, subject>,
                Token
            >::type
        >::type
    >;
};

template<class Acc, class Remaining>
struct canonicalize_impl;

template<class Acc>
struct canonicalize_impl<Acc, iro::util::type_list<>> {
    using type = Acc;
};

template<class Acc, class T, class... Ts>
struct canonicalize_impl<Acc, iro::util::type_list<T, Ts...>> {
    using next = typename canonical_insert<Acc, T>::type;
    using type = typename canonicalize_impl<next, iro::util::type_list<Ts...>>::type;
};

} // namespace detail

/// Canonicalize a token list by (kind, subject) with family-specific dominance.
template<class TokenList>
using canonicalize_token_list = typename detail::canonicalize_impl<
    iro::util::type_list<>, TokenList>::type;

// -----------------------------------------------------------------------------
// Token satisfaction (§4.5)
// -----------------------------------------------------------------------------

namespace detail {

// visible_at satisfaction: scope subsumption
template<class P, class R>
consteval bool visible_at_satisfies() {
    static_assert(std::is_same_v<typename P::kind, iro::token::kind_visible_at>);
    static_assert(std::is_same_v<typename R::kind, iro::token::kind_visible_at>);
    static_assert(std::is_same_v<typename P::subject, typename R::subject>);
    return scope_subsumes(P::scope::level, R::scope::level);
}

// lanes_valid satisfaction: lane count subsumption
template<class P, class R>
consteval bool lanes_valid_satisfies() {
    static_assert(std::is_same_v<typename P::kind, iro::token::kind_lanes_valid>);
    static_assert(std::is_same_v<typename R::kind, iro::token::kind_lanes_valid>);
    static_assert(std::is_same_v<typename P::subject, typename R::subject>);
    return P::lanes >= R::lanes;
}

// warps_valid satisfaction: warp count subsumption
template<class P, class R>
consteval bool warps_valid_satisfies() {
    static_assert(std::is_same_v<typename P::kind, iro::token::kind_warps_valid>);
    static_assert(std::is_same_v<typename R::kind, iro::token::kind_warps_valid>);
    static_assert(std::is_same_v<typename P::subject, typename R::subject>);
    return P::warps >= R::warps;
}

// warpgroup_participates satisfaction: warp count subsumption
template<class P, class R>
consteval bool warpgroup_participates_satisfies() {
    static_assert(std::is_same_v<typename P::kind, iro::token::kind_warpgroup_participates>);
    static_assert(std::is_same_v<typename R::kind, iro::token::kind_warpgroup_participates>);
    static_assert(std::is_same_v<typename P::subject, typename R::subject>);
    return P::warps >= R::warps;
}

// mask_at satisfaction: scope subsumption
template<class P, class R>
consteval bool mask_at_satisfies() {
    static_assert(std::is_same_v<typename P::kind, iro::token::kind_mask_at>);
    static_assert(std::is_same_v<typename R::kind, iro::token::kind_mask_at>);
    static_assert(std::is_same_v<typename P::subject, typename R::subject>);
    return scope_subsumes(P::scope::level, R::scope::level);
}

// alive satisfaction: lifetime subsumption
template<class P, class R>
consteval bool alive_satisfies() {
    static_assert(std::is_same_v<typename P::kind, iro::token::kind_alive>);
    static_assert(std::is_same_v<typename R::kind, iro::token::kind_alive>);
    static_assert(std::is_same_v<typename P::subject, typename R::subject>);
    return lifetime_subsumes<typename P::lifetime, typename R::lifetime>();
}

// slot_state satisfaction: exact equality
template<class P, class R>
consteval bool slot_state_satisfies() {
    static_assert(std::is_same_v<typename P::kind, iro::token::kind_slot_state>);
    static_assert(std::is_same_v<typename R::kind, iro::token::kind_slot_state>);
    static_assert(std::is_same_v<typename P::subject, typename R::subject>);
    return std::is_same_v<typename P::state, typename R::state>;
}

// lease satisfaction: exact equality
template<class P, class R>
consteval bool lease_satisfies() {
    static_assert(std::is_same_v<typename P::kind, iro::token::kind_lease>);
    static_assert(std::is_same_v<typename R::kind, iro::token::kind_lease>);
    return std::is_same_v<typename P::subject, typename R::subject>;
}

// sync_at satisfaction: scope subsumption
template<class P, class R>
consteval bool sync_at_satisfies() {
    static_assert(std::is_same_v<typename P::kind, iro::token::kind_sync_at>);
    static_assert(std::is_same_v<typename R::kind, iro::token::kind_sync_at>);
    static_assert(std::is_same_v<typename P::subject, typename R::subject>);
    return scope_subsumes(P::scope::level, R::scope::level);
}

// memory_order satisfaction: order + scope subsumption
template<class P, class R>
consteval bool memory_order_satisfies() {
    static_assert(std::is_same_v<typename P::kind, iro::token::kind_memory_order>);
    static_assert(std::is_same_v<typename R::kind, iro::token::kind_memory_order>);
    static_assert(std::is_same_v<typename P::subject, typename R::subject>);
    return detail::memory_order_subsumes<typename P::order, typename R::order>() &&
           scope_subsumes(P::scope::level, R::scope::level);
}

// version satisfaction: exact match
template<class P, class R>
consteval bool version_satisfies() {
    static_assert(std::is_same_v<typename P::kind, iro::token::kind_version>);
    static_assert(std::is_same_v<typename R::kind, iro::token::kind_version>);
    static_assert(std::is_same_v<typename P::subject, typename R::subject>);
    return P::value == R::value;
}

// determinism satisfaction: reproducible subsumes fast
template<class P, class R>
consteval bool determinism_satisfies() {
    static_assert(std::is_same_v<typename P::kind, iro::token::kind_determinism>);
    static_assert(std::is_same_v<typename R::kind, iro::token::kind_determinism>);
    static_assert(std::is_same_v<typename P::subject, typename R::subject>);
    return detail::determinism_subsumes<typename P::mode, typename R::mode>();
}

} // namespace detail

/// Check if provided token satisfies required token
template<class Provided, class Required>
consteval bool token_satisfies() {
    // Must have same kind and subject
    if constexpr (!std::is_same_v<typename Provided::kind, typename Required::kind> ||
                  !std::is_same_v<typename Provided::subject, typename Required::subject>) {
        return false;
    }
    // Dispatch by kind
    else if constexpr (std::is_same_v<typename Provided::kind, iro::token::kind_visible_at>) {
        return detail::visible_at_satisfies<Provided, Required>();
    }
    else if constexpr (std::is_same_v<typename Provided::kind, iro::token::kind_lanes_valid>) {
        return detail::lanes_valid_satisfies<Provided, Required>();
    }
    else if constexpr (std::is_same_v<typename Provided::kind, iro::token::kind_warps_valid>) {
        return detail::warps_valid_satisfies<Provided, Required>();
    }
    else if constexpr (std::is_same_v<typename Provided::kind, iro::token::kind_warpgroup_participates>) {
        return detail::warpgroup_participates_satisfies<Provided, Required>();
    }
    else if constexpr (std::is_same_v<typename Provided::kind, iro::token::kind_mask_at>) {
        return detail::mask_at_satisfies<Provided, Required>();
    }
    else if constexpr (std::is_same_v<typename Provided::kind, iro::token::kind_alive>) {
        return detail::alive_satisfies<Provided, Required>();
    }
    else if constexpr (std::is_same_v<typename Provided::kind, iro::token::kind_slot_state>) {
        return detail::slot_state_satisfies<Provided, Required>();
    }
    else if constexpr (std::is_same_v<typename Provided::kind, iro::token::kind_lease>) {
        return detail::lease_satisfies<Provided, Required>();
    }
    else if constexpr (std::is_same_v<typename Provided::kind, iro::token::kind_sync_at>) {
        return detail::sync_at_satisfies<Provided, Required>();
    }
    else if constexpr (std::is_same_v<typename Provided::kind, iro::token::kind_memory_order>) {
        return detail::memory_order_satisfies<Provided, Required>();
    }
    else if constexpr (std::is_same_v<typename Provided::kind, iro::token::kind_version>) {
        return detail::version_satisfies<Provided, Required>();
    }
    else if constexpr (std::is_same_v<typename Provided::kind, iro::token::kind_determinism>) {
        return detail::determinism_satisfies<Provided, Required>();
    }
    else {
        // Unknown token kind: require exact match
        return Provided::id == Required::id;
    }
}

/// Check if provided token list satisfies all required tokens
template<class ProvidedList, class RequiredList>
consteval bool token_list_satisfies();

namespace detail {

template<class ProvidedList>
struct list_satisfies_impl {
    template<class Req>
    static consteval bool single_satisfied() {
        using Found = get_token_kind_subject<ProvidedList, typename Req::kind, typename Req::subject>;
        if constexpr (std::is_same_v<Found, void>) {
            return false;
        } else {
            return token_satisfies<Found, Req>();
        }
    }

    template<class... Reqs>
    static consteval bool all_satisfied(iro::util::type_list<Reqs...>) {
        return (single_satisfied<Reqs>() && ...);
    }
};

// Specialization for empty provided list - only satisfied if required list is also empty
template<>
struct list_satisfies_impl<iro::util::type_list<>> {
    template<class Req>
    static consteval bool single_satisfied() {
        return false;  // Can't satisfy any requirement with empty provided list
    }

    template<class... Reqs>
    static consteval bool all_satisfied(iro::util::type_list<Reqs...>) {
        return sizeof...(Reqs) == 0;  // Only satisfied if nothing is required
    }
};

} // namespace detail

template<class ProvidedList, class RequiredList>
consteval bool token_list_satisfies() {
    return detail::list_satisfies_impl<ProvidedList>::all_satisfied(RequiredList{});
}

// -----------------------------------------------------------------------------
// Resource verification
// -----------------------------------------------------------------------------

/// Check if two resources are compatible (same id implies type-identical)
template<class R1, class R2>
consteval bool resources_compatible() {
    if constexpr (R1::id == R2::id) {
        return std::is_same_v<R1, R2>;
    } else {
        return true; // Different ids are always compatible
    }
}

/// Check resource list for conflicts
template<class ResourceList>
consteval bool resources_ok_union();

namespace detail {

template<class List>
struct resources_ok_impl;

template<>
struct resources_ok_impl<iro::util::type_list<>> {
    static constexpr bool ok = true;
};

template<class R, class... Rs>
struct resources_ok_impl<iro::util::type_list<R, Rs...>> {
    template<class Other>
    static consteval bool compatible_with_rest() {
        return resources_compatible<R, Other>();
    }

    static constexpr bool ok =
        (compatible_with_rest<Rs>() && ...) &&
        resources_ok_impl<iro::util::type_list<Rs...>>::ok;
};

} // namespace detail

template<class ResourceList>
consteval bool resources_ok_union() {
    return detail::resources_ok_impl<ResourceList>::ok;
}

// -----------------------------------------------------------------------------
// Resource list canonicality (§5.1: resources canonical by id)
// -----------------------------------------------------------------------------

namespace detail {

template<class List>
struct resource_ids;

template<>
struct resource_ids<iro::util::type_list<>> {
    static constexpr bool has_duplicate = false;
};

template<class R, class... Rs>
struct resource_ids<iro::util::type_list<R, Rs...>> {
    static constexpr iro::util::u64 first_id = R::id;

    template<class U>
    static consteval bool matches() {
        return U::id == first_id;
    }

    static constexpr bool has_duplicate =
        (matches<Rs>() || ...) || resource_ids<iro::util::type_list<Rs...>>::has_duplicate;
};

} // namespace detail

/// Check if resource list is canonical (no duplicate ids)
template<class ResourceList>
consteval bool resource_list_canonical() {
    return !detail::resource_ids<ResourceList>::has_duplicate;
}

// -----------------------------------------------------------------------------
// Resource canonicalization + budget aggregation
// -----------------------------------------------------------------------------

namespace detail {

template<class List, iro::util::u64 Id>
struct resource_id_exists;

template<iro::util::u64 Id>
struct resource_id_exists<iro::util::type_list<>, Id> : std::false_type {};

template<class R, class... Rs, iro::util::u64 Id>
struct resource_id_exists<iro::util::type_list<R, Rs...>, Id>
    : std::bool_constant<(R::id == Id) || resource_id_exists<iro::util::type_list<Rs...>, Id>::value> {};

template<class Acc, class R>
struct resource_insert {
    using type = std::conditional_t<
        resource_id_exists<Acc, R::id>::value,
        Acc,
        iro::util::append_t<Acc, R>
    >;
};

template<class Acc, class List>
struct canonicalize_resources_impl;

template<class Acc>
struct canonicalize_resources_impl<Acc, iro::util::type_list<>> {
    using type = Acc;
};

template<class Acc, class R, class... Rs>
struct canonicalize_resources_impl<Acc, iro::util::type_list<R, Rs...>> {
    using next = typename resource_insert<Acc, R>::type;
    using type = typename canonicalize_resources_impl<next, iro::util::type_list<Rs...>>::type;
};

template<class R>
struct smem_bytes : std::integral_constant<long long, 0> {};

template<class Tag, long long Bytes, int AlignBytes>
struct smem_bytes<iro::contract::res::smem_region<Tag, Bytes, AlignBytes>>
    : std::integral_constant<long long, Bytes> {};

template<class Tag, int Slots, long long BytesPerSlot, int AlignBytes>
struct smem_bytes<iro::contract::res::smem_pipeline<Tag, Slots, BytesPerSlot, AlignBytes>>
    : std::integral_constant<long long, Slots * BytesPerSlot> {};

template<class Tag, long long Bytes, int AlignBytes>
struct smem_bytes<iro::contract::res::smem_arena<Tag, Bytes, AlignBytes>>
    : std::integral_constant<long long, Bytes> {};

template<class R>
struct barrier_count : std::integral_constant<int, 0> {};

template<class Tag>
struct barrier_count<iro::contract::res::named_barrier<Tag>>
    : std::integral_constant<int, 1> {};

template<class Tag, int BaseId>
struct barrier_count<iro::contract::res::warpgroup_barrier<Tag, BaseId>>
    : std::integral_constant<int, 1> {};

template<class R>
struct is_named_barrier : std::false_type {};

template<class Tag>
struct is_named_barrier<iro::contract::res::named_barrier<Tag>> : std::true_type {};

template<class R>
struct is_warpgroup_barrier : std::false_type {};

template<class Tag, int BaseId>
struct is_warpgroup_barrier<iro::contract::res::warpgroup_barrier<Tag, BaseId>> : std::true_type {};

template<class List>
struct count_named_barriers;

template<>
struct count_named_barriers<iro::util::type_list<>> : std::integral_constant<int, 0> {};

template<class R, class... Rs>
struct count_named_barriers<iro::util::type_list<R, Rs...>> {
    static constexpr int value =
        (is_named_barrier<R>::value ? 1 : 0) + count_named_barriers<iro::util::type_list<Rs...>>::value;
};

template<class List>
struct count_warpgroup_barriers;

template<>
struct count_warpgroup_barriers<iro::util::type_list<>> : std::integral_constant<int, 0> {};

template<class R, class... Rs>
struct count_warpgroup_barriers<iro::util::type_list<R, Rs...>> {
    static constexpr int value =
        (is_warpgroup_barrier<R>::value ? 1 : 0) + count_warpgroup_barriers<iro::util::type_list<Rs...>>::value;
};

template<class List>
struct warpgroup_count_info {
    static constexpr bool found = false;
    static constexpr bool conflict = false;
    static constexpr int value = 1;
};

template<int Count, class... Rs>
struct warpgroup_count_info<iro::util::type_list<iro::contract::res::warpgroup_count<Count>, Rs...>> {
    using tail = warpgroup_count_info<iro::util::type_list<Rs...>>;
    static constexpr bool found = true;
    static constexpr bool conflict = tail::found && (tail::value != Count);
    static constexpr int value = tail::found ? tail::value : Count;
};

template<class R, class... Rs>
struct warpgroup_count_info<iro::util::type_list<R, Rs...>> {
    using tail = warpgroup_count_info<iro::util::type_list<Rs...>>;
    static constexpr bool found = tail::found;
    static constexpr bool conflict = tail::conflict;
    static constexpr int value = tail::value;
};

template<class List>
struct barrier_count_total {
    using info = warpgroup_count_info<List>;
    static_assert(!info::conflict,
        "barrier_count_total: multiple warpgroup_count resources with different values");
    static constexpr int value =
        count_named_barriers<List>::value +
        count_warpgroup_barriers<List>::value * info::value;
};

template<class R>
struct reg_count : std::integral_constant<int, 0> {};

template<int Regs>
struct reg_count<iro::contract::res::reg_usage<Regs>>
    : std::integral_constant<int, Regs> {};

template<class R>
struct reg_pressure_count : std::integral_constant<int, 0> {};

template<int Regs>
struct reg_pressure_count<iro::contract::res::reg_pressure<Regs>>
    : std::integral_constant<int, Regs> {};

template<class List>
struct sum_smem;

template<>
struct sum_smem<iro::util::type_list<>> : std::integral_constant<long long, 0> {};

template<class R, class... Rs>
struct sum_smem<iro::util::type_list<R, Rs...>> {
    static constexpr long long value =
        static_cast<long long>(iro::util::checked_add_u64(
            static_cast<iro::util::u64>(smem_bytes<R>::value),
            static_cast<iro::util::u64>(sum_smem<iro::util::type_list<Rs...>>::value)
        ));
};

template<class List>
struct sum_barriers;

template<>
struct sum_barriers<iro::util::type_list<>> : std::integral_constant<int, 0> {};

template<class R, class... Rs>
struct sum_barriers<iro::util::type_list<R, Rs...>> {
    static constexpr int value = barrier_count<R>::value + sum_barriers<iro::util::type_list<Rs...>>::value;
};

template<class List>
struct sum_regs;

template<>
struct sum_regs<iro::util::type_list<>> : std::integral_constant<int, 0> {};

template<class R, class... Rs>
struct sum_regs<iro::util::type_list<R, Rs...>> {
    static constexpr int value = reg_count<R>::value + sum_regs<iro::util::type_list<Rs...>>::value;
};

template<class List>
struct max_reg_pressure;

template<>
struct max_reg_pressure<iro::util::type_list<>> : std::integral_constant<int, 0> {};

template<class R, class... Rs>
struct max_reg_pressure<iro::util::type_list<R, Rs...>> {
    static constexpr int tail = max_reg_pressure<iro::util::type_list<Rs...>>::value;
    static constexpr int head = reg_pressure_count<R>::value;
    static constexpr int value = (head > tail) ? head : tail;
};

} // namespace detail

template<class ResourceList>
using canonicalize_resource_list = typename detail::canonicalize_resources_impl<
    iro::util::type_list<>, ResourceList>::type;

template<class ResourceList>
inline constexpr long long smem_bytes_total_v =
    detail::sum_smem<canonicalize_resource_list<ResourceList>>::value;

template<class ResourceList>
inline constexpr int barrier_count_total_v =
    detail::barrier_count_total<canonicalize_resource_list<ResourceList>>::value;

template<class ResourceList>
inline constexpr int reg_count_total_v =
    detail::sum_regs<canonicalize_resource_list<ResourceList>>::value;

template<class ResourceList>
inline constexpr int reg_pressure_max_v =
    detail::max_reg_pressure<canonicalize_resource_list<ResourceList>>::value;

template<class Recipe>
struct recipe_in_a {
    using type = typename Recipe::in;
};

template<class Recipe>
struct recipe_in_b {
    using type = typename Recipe::in;
};

template<class Recipe>
struct recipe_scale_a {
    using type = typename Recipe::scale;
    static constexpr int vec = Recipe::scale_vec;
};

template<class Recipe>
struct recipe_scale_b {
    using type = typename Recipe::scale;
    static constexpr int vec = Recipe::scale_vec;
};

template<class ElemA, class ElemB, class AccElem, class OutElem, int VecBytes, class MathPolicy,
         class Fp8Policy, class ScaleElemA, int ScaleVecA, class ScaleElemB, int ScaleVecB>
struct recipe_in_a<iro::recipe::PrecisionAB<ElemA, ElemB, AccElem, OutElem, VecBytes, MathPolicy,
                                           Fp8Policy, ScaleElemA, ScaleVecA, ScaleElemB, ScaleVecB>> {
    using type = ElemA;
};

template<class ElemA, class ElemB, class AccElem, class OutElem, int VecBytes, class MathPolicy,
         class Fp8Policy, class ScaleElemA, int ScaleVecA, class ScaleElemB, int ScaleVecB>
struct recipe_in_b<iro::recipe::PrecisionAB<ElemA, ElemB, AccElem, OutElem, VecBytes, MathPolicy,
                                           Fp8Policy, ScaleElemA, ScaleVecA, ScaleElemB, ScaleVecB>> {
    using type = ElemB;
};

template<class ElemA, class ElemB, class AccElem, class OutElem, int VecBytes, class MathPolicy,
         class Fp8Policy, class ScaleElemA, int ScaleVecA, class ScaleElemB, int ScaleVecB>
struct recipe_scale_a<iro::recipe::PrecisionAB<ElemA, ElemB, AccElem, OutElem, VecBytes, MathPolicy,
                                              Fp8Policy, ScaleElemA, ScaleVecA, ScaleElemB, ScaleVecB>> {
    using type = ScaleElemA;
    static constexpr int vec = ScaleVecA;
};

template<class ElemA, class ElemB, class AccElem, class OutElem, int VecBytes, class MathPolicy,
         class Fp8Policy, class ScaleElemA, int ScaleVecA, class ScaleElemB, int ScaleVecB>
struct recipe_scale_b<iro::recipe::PrecisionAB<ElemA, ElemB, AccElem, OutElem, VecBytes, MathPolicy,
                                              Fp8Policy, ScaleElemA, ScaleVecA, ScaleElemB, ScaleVecB>> {
    using type = ScaleElemB;
    static constexpr int vec = ScaleVecB;
};

template<class BaseRecipe, class Pattern>
struct recipe_in_a<iro::recipe::Sparse<BaseRecipe, Pattern>> {
    using type = typename recipe_in_a<BaseRecipe>::type;
};

template<class BaseRecipe, class Pattern>
struct recipe_in_b<iro::recipe::Sparse<BaseRecipe, Pattern>> {
    using type = typename recipe_in_b<BaseRecipe>::type;
};

template<class BaseRecipe, class Pattern>
struct recipe_scale_a<iro::recipe::Sparse<BaseRecipe, Pattern>> {
    using type = typename recipe_scale_a<BaseRecipe>::type;
    static constexpr int vec = recipe_scale_a<BaseRecipe>::vec;
};

template<class BaseRecipe, class Pattern>
struct recipe_scale_b<iro::recipe::Sparse<BaseRecipe, Pattern>> {
    using type = typename recipe_scale_b<BaseRecipe>::type;
    static constexpr int vec = recipe_scale_b<BaseRecipe>::vec;
};

template<class Recipe>
using recipe_in_a_t = typename recipe_in_a<Recipe>::type;

template<class Recipe>
using recipe_in_b_t = typename recipe_in_b<Recipe>::type;

template<class Recipe>
using recipe_scale_a_t = typename recipe_scale_a<Recipe>::type;

template<class Recipe>
using recipe_scale_b_t = typename recipe_scale_b<Recipe>::type;

template<class Recipe>
inline constexpr int recipe_scale_vec_a_v = recipe_scale_a<Recipe>::vec;

template<class Recipe>
inline constexpr int recipe_scale_vec_b_v = recipe_scale_b<Recipe>::vec;

template<class Recipe>
inline constexpr bool recipe_has_scale_a_v =
    !std::is_same_v<recipe_scale_a_t<Recipe>, iro::recipe::no_scale>;

template<class Recipe>
inline constexpr bool recipe_has_scale_b_v =
    !std::is_same_v<recipe_scale_b_t<Recipe>, iro::recipe::no_scale>;

template<class Recipe>
struct recipe_traits {
    using in = typename Recipe::in;
    using acc = typename Recipe::acc;
    using out = typename Recipe::out;
    using math = typename Recipe::math;
    static constexpr int vec_bytes = Recipe::vec_bytes;
    using in_a = recipe_in_a_t<Recipe>;
    using in_b = recipe_in_b_t<Recipe>;
    using scale_a = recipe_scale_a_t<Recipe>;
    using scale_b = recipe_scale_b_t<Recipe>;
    static constexpr int scale_vec_a = recipe_scale_vec_a_v<Recipe>;
    static constexpr int scale_vec_b = recipe_scale_vec_b_v<Recipe>;
    static constexpr bool has_scale_a = recipe_has_scale_a_v<Recipe>;
    static constexpr bool has_scale_b = recipe_has_scale_b_v<Recipe>;
};

template<class RecipeA, class RecipeB>
consteval bool recipe_compatible() {
    using A0 = iro::recipe::unwrap_accumulate_t<RecipeA>;
    using B0 = iro::recipe::unwrap_accumulate_t<RecipeB>;
    constexpr bool A_is_ab = requires { typename A0::in_a; typename A0::in_b; };
    constexpr bool B_is_ab = requires { typename B0::in_a; typename B0::in_b; };
    if constexpr (std::is_same_v<A0, B0>) {
        return true;
    } else if constexpr (A_is_ab && B_is_ab) {
        return std::is_same_v<typename A0::in_a, typename B0::in_a> &&
               std::is_same_v<typename A0::in_b, typename B0::in_b> &&
               std::is_same_v<typename A0::out, typename B0::out> &&
               std::is_same_v<typename A0::math, typename B0::math> &&
               std::is_same_v<typename A0::fp8_policy, typename B0::fp8_policy> &&
               std::is_same_v<typename A0::scale_a, typename B0::scale_a> &&
               std::is_same_v<typename A0::scale_b, typename B0::scale_b> &&
               (A0::scale_vec_a == B0::scale_vec_a) &&
               (A0::scale_vec_b == B0::scale_vec_b);
    } else if constexpr (A_is_ab && !B_is_ab) {
        return (std::is_same_v<typename B0::in, recipe_in_a_t<A0>> ||
                std::is_same_v<typename B0::in, recipe_in_b_t<A0>>) &&
               std::is_same_v<typename B0::out, typename A0::out> &&
               std::is_same_v<typename B0::math, typename A0::math> &&
               std::is_same_v<typename B0::fp8_policy, typename A0::fp8_policy>;
    } else if constexpr (!A_is_ab && B_is_ab) {
        return (std::is_same_v<typename A0::in, recipe_in_a_t<B0>> ||
                std::is_same_v<typename A0::in, recipe_in_b_t<B0>>) &&
               std::is_same_v<typename A0::out, typename B0::out> &&
               std::is_same_v<typename A0::math, typename B0::math> &&
               std::is_same_v<typename A0::fp8_policy, typename B0::fp8_policy>;
    } else {
        return std::is_same_v<typename A0::in, typename B0::in> &&
               std::is_same_v<typename A0::out, typename B0::out> &&
               std::is_same_v<typename A0::math, typename B0::math> &&
               std::is_same_v<typename A0::fp8_policy, typename B0::fp8_policy> &&
               std::is_same_v<typename A0::scale, typename B0::scale> &&
               (A0::scale_vec == B0::scale_vec);
    }
}

} // namespace iro::verify

// =============================================================================
// §6. Ports and payload categories
// =============================================================================

namespace iro::contract {

// -----------------------------------------------------------------------------
// §6.2 Payload concepts (normative)
// -----------------------------------------------------------------------------

namespace detail {

template<class T> struct is_tile : std::false_type {};
template<class S, class E, class L, class Sp, class A>
struct is_tile<Tile<S, E, L, Sp, A>> : std::true_type {};

template<class T> struct is_tensor_ref_desc : std::false_type {};
template<int R, class E, class S, class A>
struct is_tensor_ref_desc<TensorRefDesc<R, E, S, A>> : std::true_type {};

template<class T> struct is_fragment_desc : std::false_type {};
template<class S, class E, class D, int C>
struct is_fragment_desc<FragmentDesc<S, E, D, C>> : std::true_type {};

template<class T> struct is_scalar_desc : std::false_type {};
template<class E, class D>
struct is_scalar_desc<ScalarDesc<E, D>> : std::true_type {};

template<class T> struct is_vector_desc : std::false_type {};
template<class E, int N, class D>
struct is_vector_desc<VectorDesc<E, N, D>> : std::true_type {};

template<class T> struct is_mask_desc : std::false_type {};
template<int W, class D>
struct is_mask_desc<MaskDesc<W, D>> : std::true_type {};

} // namespace detail

template<class T>
concept TilePayload = detail::is_tile<T>::value;

template<class T>
concept RefPayload = detail::is_tensor_ref_desc<T>::value;

template<class T>
concept FragmentPayload = detail::is_fragment_desc<T>::value;

template<class T>
concept ScalarPayload = detail::is_scalar_desc<T>::value;

template<class T>
concept VectorPayload = detail::is_vector_desc<T>::value;

template<class T>
concept MaskPayload = detail::is_mask_desc<T>::value;

template<class T>
concept HandlePayload = iro::util::HasId<T> &&
    !TilePayload<T> && !RefPayload<T> && !FragmentPayload<T> &&
    !ScalarPayload<T> && !VectorPayload<T> && !MaskPayload<T>;

// -----------------------------------------------------------------------------
// §6.3 Port type (normative)
// -----------------------------------------------------------------------------

/// No-dist marker for payloads that don't need dist (global/shared space tiles, refs, handles)
struct no_dist {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.contract.no_dist");
};

// -----------------------------------------------------------------------------
// Precision recipes (explicit, no hidden effects)
// -----------------------------------------------------------------------------

} // namespace iro::contract

namespace iro::schema {
template<>
struct is_dist<iro::contract::no_dist> : std::true_type {};
} // namespace iro::schema

namespace iro::recipe {

struct Exact {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.recipe.math.exact");
};

struct ApproxExp {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.recipe.math.approx_exp");
};

struct Fast {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.recipe.math.fast");
};

struct no_recipe {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.recipe.none");
};

struct no_scale {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.recipe.scale.none");
};

struct no_fp8_policy {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.recipe.fp8.none");
};

struct fp8_native {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.recipe.fp8.native");
};

struct fp8_saturate {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.recipe.fp8.saturate");
};

struct fp8_nan_to_zero {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.recipe.fp8.nan_to_zero");
};

template<class Elem>
inline constexpr bool is_fp8_elem_v =
    std::is_same_v<Elem, iro::elem::e4m3> ||
    std::is_same_v<Elem, iro::elem::e4m3fn> ||
    std::is_same_v<Elem, iro::elem::e5m2> ||
    std::is_same_v<Elem, iro::elem::e5m2fnuz>;

template<class Policy>
inline constexpr bool implemented_fp8_policy_v =
    std::is_same_v<Policy, no_fp8_policy> ||
    std::is_same_v<Policy, fp8_native> ||
    std::is_same_v<Policy, fp8_saturate> ||
    std::is_same_v<Policy, fp8_nan_to_zero>;

template<class R, class = void>
struct has_fp8_policy : std::false_type {};

template<class R>
struct has_fp8_policy<R, std::void_t<typename R::fp8_policy>> : std::true_type {};

template<class InElem, class AccElem, class OutElem, int VecBytes, class MathPolicy,
         class Fp8Policy = no_fp8_policy, class ScaleElem = no_scale, int ScaleVec = 0>
struct Precision {
    static_assert(VecBytes == 4 || VecBytes == 8 || VecBytes == 16, "Precision recipe VecBytes must be 4, 8, or 16");
    static_assert(ScaleVec >= 0, "Precision recipe ScaleVec must be >= 0");
    static_assert((std::is_same_v<ScaleElem, no_scale> && ScaleVec == 0) ||
                  (!std::is_same_v<ScaleElem, no_scale> && ScaleVec > 0),
                  "Precision recipe ScaleVec must be >0 when scale elem is set");
    static constexpr bool uses_fp8 =
        is_fp8_elem_v<InElem> || is_fp8_elem_v<AccElem> || is_fp8_elem_v<OutElem>;
    static_assert(!uses_fp8 || !std::is_same_v<Fp8Policy, no_fp8_policy>,
                  "Precision recipe: FP8 types require explicit fp8_policy");
    static_assert(!uses_fp8 || implemented_fp8_policy_v<Fp8Policy>,
                  "Precision recipe: fp8_policy not implemented in this build");
    using in = InElem;
    using acc = AccElem;
    using out = OutElem;
    static constexpr int vec_bytes = VecBytes;
    using math = MathPolicy;
    using fp8_policy = Fp8Policy;
    using scale = ScaleElem;
    static constexpr int scale_vec = ScaleVec;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.recipe.Precision"),
            iro::util::mix_u64(InElem::id,
                iro::util::mix_u64(AccElem::id,
                    iro::util::mix_u64(OutElem::id,
                        iro::util::mix_u64(static_cast<iro::util::u64>(VecBytes),
                            iro::util::mix_u64(MathPolicy::id,
                                iro::util::mix_u64(Fp8Policy::id,
                                    iro::util::mix_u64(ScaleElem::id, static_cast<iro::util::u64>(ScaleVec)))))))));
};

template<class InElem, class AccElem, class OutElem, int VecBytes, class MathPolicy,
         class Fp8Policy = no_fp8_policy, class ScaleElem = no_scale, int ScaleVec = 0>
struct ScaledFp8 : Precision<InElem, AccElem, OutElem, VecBytes, MathPolicy, Fp8Policy, ScaleElem, ScaleVec> {
    static_assert(is_fp8_elem_v<InElem> || is_fp8_elem_v<AccElem> || is_fp8_elem_v<OutElem>,
                  "ScaledFp8 requires FP8 element types");
    static_assert(!std::is_same_v<ScaleElem, no_scale>, "ScaledFp8 requires explicit scale element");
    static_assert(ScaleVec > 0, "ScaledFp8 requires ScaleVec > 0");
    using base = Precision<InElem, AccElem, OutElem, VecBytes, MathPolicy, Fp8Policy, ScaleElem, ScaleVec>;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.recipe.ScaledFp8"), base::id);
};

template<class ElemA, class ElemB, class AccElem, class OutElem, int VecBytes, class MathPolicy,
         class Fp8Policy = no_fp8_policy,
         class ScaleElemA = no_scale, int ScaleVecA = 0,
         class ScaleElemB = no_scale, int ScaleVecB = 0>
struct PrecisionAB {
    static_assert(VecBytes == 4 || VecBytes == 8 || VecBytes == 16, "PrecisionAB VecBytes must be 4, 8, or 16");
    static_assert(ScaleVecA >= 0 && ScaleVecB >= 0, "PrecisionAB ScaleVec must be >= 0");
    static_assert((std::is_same_v<ScaleElemA, no_scale> && ScaleVecA == 0) ||
                  (!std::is_same_v<ScaleElemA, no_scale> && ScaleVecA > 0),
                  "PrecisionAB ScaleVecA must be >0 when ScaleElemA is set");
    static_assert((std::is_same_v<ScaleElemB, no_scale> && ScaleVecB == 0) ||
                  (!std::is_same_v<ScaleElemB, no_scale> && ScaleVecB > 0),
                  "PrecisionAB ScaleVecB must be >0 when ScaleElemB is set");
    static constexpr bool uses_fp8 =
        is_fp8_elem_v<ElemA> || is_fp8_elem_v<ElemB> ||
        is_fp8_elem_v<AccElem> || is_fp8_elem_v<OutElem>;
    static_assert(!uses_fp8 || !std::is_same_v<Fp8Policy, no_fp8_policy>,
                  "PrecisionAB: FP8 types require explicit fp8_policy");
    static_assert(!uses_fp8 || implemented_fp8_policy_v<Fp8Policy>,
                  "PrecisionAB: fp8_policy not implemented in this build");
    using in_a = ElemA;
    using in_b = ElemB;
    using acc = AccElem;
    using out = OutElem;
    static constexpr int vec_bytes = VecBytes;
    using math = MathPolicy;
    using fp8_policy = Fp8Policy;
    using scale_a = ScaleElemA;
    static constexpr int scale_vec_a = ScaleVecA;
    using scale_b = ScaleElemB;
    static constexpr int scale_vec_b = ScaleVecB;
    using in = in_a;
    using scale = scale_a;
    static constexpr int scale_vec = scale_vec_a;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.recipe.PrecisionAB"),
            iro::util::mix_u64(ElemA::id,
                iro::util::mix_u64(ElemB::id,
                    iro::util::mix_u64(AccElem::id,
                        iro::util::mix_u64(OutElem::id,
                            iro::util::mix_u64(static_cast<iro::util::u64>(VecBytes),
                                iro::util::mix_u64(MathPolicy::id,
                                    iro::util::mix_u64(Fp8Policy::id,
                                        iro::util::mix_u64(ScaleElemA::id,
                                            iro::util::mix_u64(static_cast<iro::util::u64>(ScaleVecA),
                                                iro::util::mix_u64(ScaleElemB::id,
                                                    static_cast<iro::util::u64>(ScaleVecB))))))))))));
};

template<class R>
struct Accumulate {
    static_assert(iro::util::HasId<R>, "Accumulate requires Recipe with id");
    static_assert(requires { typename R::acc; }, "Accumulate requires Recipe::acc");
    static_assert(requires { typename R::math; }, "Accumulate requires Recipe::math");
    static_assert(requires { R::vec_bytes; }, "Accumulate requires Recipe::vec_bytes");
    static_assert(requires { typename R::fp8_policy; }, "Accumulate requires Recipe::fp8_policy");
    static_assert(requires { typename R::scale; }, "Accumulate requires Recipe::scale");
    static_assert(requires { R::scale_vec; }, "Accumulate requires Recipe::scale_vec");

    using base = R;
    using in = typename R::acc;
    using acc = typename R::acc;
    using out = typename R::acc;
    static constexpr int vec_bytes = R::vec_bytes;
    using math = typename R::math;
    using fp8_policy = typename R::fp8_policy;
    using scale = typename R::scale;
    static constexpr int scale_vec = R::scale_vec;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.recipe.Accumulate"), R::id);
};

template<class R>
struct is_accumulate<Accumulate<R>> : std::true_type {};

template<class R>
struct unwrap_accumulate<Accumulate<R>> { using type = typename unwrap_accumulate<R>::type; };

struct sparse_2to4 {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.recipe.sparse.2to4");
    static constexpr int k = 2;
    static constexpr int n = 4;
};

struct sparse_4to8 {
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("iro.recipe.sparse.4to8");
    static constexpr int k = 4;
    static constexpr int n = 8;
};

template<class BaseRecipe, class Pattern>
struct Sparse : BaseRecipe {
    static_assert(iro::util::HasId<BaseRecipe>, "Sparse recipe requires BaseRecipe with id");
    static_assert(iro::util::HasId<Pattern>, "Sparse recipe requires Pattern with id");
    using base = BaseRecipe;
    using sparse_pattern = Pattern;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.recipe.Sparse"),
            iro::util::mix_u64(BaseRecipe::id, Pattern::id));
};

template<class R>
struct is_precision_ab : std::false_type {};

template<class ElemA, class ElemB, class AccElem, class OutElem, int VecBytes, class MathPolicy,
         class Fp8Policy, class ScaleElemA, int ScaleVecA, class ScaleElemB, int ScaleVecB>
struct is_precision_ab<PrecisionAB<ElemA, ElemB, AccElem, OutElem, VecBytes, MathPolicy,
                                   Fp8Policy, ScaleElemA, ScaleVecA, ScaleElemB, ScaleVecB>>
    : std::true_type {};

template<class R>
inline constexpr bool is_precision_ab_v = is_precision_ab<R>::value;

template<class R>
struct is_sparse_recipe : std::false_type {};

template<class BaseRecipe, class Pattern>
struct is_sparse_recipe<Sparse<BaseRecipe, Pattern>> : std::true_type {};

template<class R>
inline constexpr bool is_sparse_recipe_v = is_sparse_recipe<R>::value;

template<class R>
inline constexpr bool implemented_recipe_math_v =
    std::is_same_v<typename R::math, Exact> ||
    std::is_same_v<typename R::math, ApproxExp> ||
    std::is_same_v<typename R::math, Fast>;

} // namespace iro::recipe

namespace iro::contract {

namespace detail {

/// Check if payload requires dist (reg/fragment space tiles and fragments)
template<class PayloadT>
consteval bool payload_requires_dist() {
    if constexpr (is_tile<PayloadT>::value) {
        return std::is_same_v<typename PayloadT::space, space::reg> ||
               std::is_same_v<typename PayloadT::space, space::fragment> ||
               std::is_same_v<typename PayloadT::space, space::tmem>;
    } else if constexpr (is_fragment_desc<PayloadT>::value) {
        return true;
    } else if constexpr (is_scalar_desc<PayloadT>::value ||
                         is_vector_desc<PayloadT>::value ||
                         is_mask_desc<PayloadT>::value) {
        return true;
    } else {
        return false;
    }
}

/// Check if scope is block-or-higher
template<class ScopeT>
consteval bool is_block_or_higher_scope() {
    return ScopeT::level >= iro::scope::block::level;
}

template<class ExecGroupT>
consteval bool exec_group_requires_sync() {
    using MinScope = iro::scope::min_scope_for_t<ExecGroupT>;
    return MinScope::level >= iro::scope::block::level;
}

} // namespace detail

// Forward declaration for mandatory token rule checks
namespace verify {

/// §6.4: Check input tile port has required visible_at + alive tokens
template<class PayloadT, class SubjectT, class ExecGroupT, class RequiredTokens>
consteval bool input_tile_port_well_formed() {
    if constexpr (!detail::is_tile<PayloadT>::value) {
        return true; // Not a tile payload, no mandatory rules
    } else {
        // Must have visible_at<subject, S> (scope may be upgraded by the obligation)
        constexpr bool has_visible = iro::verify::has_token_kind_subject<
            RequiredTokens, iro::token::kind_visible_at, SubjectT>();

        // Must have alive<subject, L>
        constexpr bool has_alive = iro::verify::has_token_kind_subject<
            RequiredTokens, iro::token::kind_alive, SubjectT>();

        return has_visible && has_alive;
    }
}

/// §6.4: Check output tile port has required visible_at + alive + sync_at tokens
template<class PayloadT, class SubjectT, class ExecGroupT, class ProvidedTokens>
consteval bool output_tile_port_well_formed() {
    if constexpr (!detail::is_tile<PayloadT>::value) {
        return true; // Not a tile payload, no mandatory rules
    } else {
        using MinScope = iro::scope::min_scope_for_t<ExecGroupT>;

        // Must have visible_at<subject, S> where S subsumes min_scope
        constexpr bool has_visible = iro::verify::has_token_kind_subject<
            ProvidedTokens, iro::token::kind_visible_at, SubjectT>();

        // Must have alive<subject, L>
        constexpr bool has_alive = iro::verify::has_token_kind_subject<
            ProvidedTokens, iro::token::kind_alive, SubjectT>();

        if constexpr (!has_visible || !has_alive) {
            return false;
        } else {
            using VisibleToken = iro::verify::get_token_kind_subject<
                ProvidedTokens, iro::token::kind_visible_at, SubjectT>;

            // Check scope subsumption
            if constexpr (!iro::verify::scope_subsumes(
                    VisibleToken::scope::level, MinScope::level)) {
                return false;
            }
            // If exec group is warpgroup-or-higher, must also have sync_at
            else if constexpr (detail::exec_group_requires_sync<ExecGroupT>()) {
                constexpr bool has_sync = iro::verify::has_token_kind_subject<
                    ProvidedTokens, iro::token::kind_sync_at, SubjectT>();
                if constexpr (!has_sync) {
                    return false;
                } else {
                    using SyncToken = iro::verify::get_token_kind_subject<
                        ProvidedTokens, iro::token::kind_sync_at, SubjectT>;
                    return iro::verify::scope_subsumes(
                        SyncToken::scope::level, MinScope::level);
                }
            } else {
                return true;
            }
        }
    }
}

} // namespace verify

/// Input port with dist support (§6.3, §6.6) and explicit precision recipe
template<class PayloadT, class SubjectT, class ExecGroupT, class RequiredTokens,
         class DistT = no_dist, class RecipeT = iro::recipe::no_recipe>
struct InputPort {
    // §6.6: Dist rule - reg/fragment tiles and fragments MUST have non-no_dist
    static_assert(!detail::payload_requires_dist<PayloadT>() || !std::is_same_v<DistT, no_dist>,
        "InputPort: reg/fragment tiles and FragmentDesc MUST specify dist");

    // §6.3: Recipe math must be implemented (unless no_recipe)
    static_assert(std::is_same_v<RecipeT, iro::recipe::no_recipe> ||
                  iro::recipe::implemented_recipe_math_v<RecipeT>,
        "InputPort: Recipe math policy not implemented (use Exact/Fast)");

    // §6.4: Mandatory token rules for tile payloads
    static_assert(verify::input_tile_port_well_formed<PayloadT, SubjectT, ExecGroupT, RequiredTokens>(),
        "InputPort: TilePayload MUST require visible_at + alive tokens (§6.4)");

    using payload = PayloadT;
    using subject = SubjectT;
    using exec_group = ExecGroupT;
    using required = RequiredTokens;
    using dist = DistT;
    using recipe = RecipeT;

    static constexpr bool is_input = true;
    static constexpr bool is_output = false;

    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.port.Input"),
            iro::util::mix_u64(PayloadT::id,
                iro::util::mix_u64(SubjectT::id,
                    iro::util::mix_u64(ExecGroupT::id,
                        iro::util::mix_u64(DistT::id, RecipeT::id)))));
};

/// Output port with dist support (§6.3, §6.6) and explicit precision recipe
template<class PayloadT, class SubjectT, class ExecGroupT, class ProvidedTokens,
         class DistT = no_dist, class RecipeT = iro::recipe::no_recipe>
struct OutputPort {
    // §6.6: Dist rule - reg/fragment tiles and fragments MUST have non-no_dist
    static_assert(!detail::payload_requires_dist<PayloadT>() || !std::is_same_v<DistT, no_dist>,
        "OutputPort: reg/fragment tiles and FragmentDesc MUST specify dist");

    // §6.3: Recipe math must be implemented (unless no_recipe)
    static_assert(std::is_same_v<RecipeT, iro::recipe::no_recipe> ||
                  iro::recipe::implemented_recipe_math_v<RecipeT>,
        "OutputPort: Recipe math policy not implemented (use Exact/Fast)");

    // §6.4: Mandatory token rules for tile payloads
    static_assert(verify::output_tile_port_well_formed<PayloadT, SubjectT, ExecGroupT, ProvidedTokens>(),
        "OutputPort: TilePayload MUST provide visible_at + alive + sync_at (if block+) tokens (§6.4)");

    using payload = PayloadT;
    using subject = SubjectT;
    using exec_group = ExecGroupT;
    using provided = ProvidedTokens;
    using dist = DistT;
    using recipe = RecipeT;

    static constexpr bool is_input = false;
    static constexpr bool is_output = true;

    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.port.Output"),
            iro::util::mix_u64(PayloadT::id,
                iro::util::mix_u64(SubjectT::id,
                    iro::util::mix_u64(ExecGroupT::id,
                        iro::util::mix_u64(DistT::id, RecipeT::id)))));
};

// -----------------------------------------------------------------------------
// §6.5 Payload compatibility (normative; strict)
// -----------------------------------------------------------------------------

namespace verify {

/// Check payload compatibility for port connection
template<class OutPayload, class InPayload>
consteval bool payload_compatible() {
    // Cross-kind: incompatible
    if constexpr (TilePayload<OutPayload> && TilePayload<InPayload>) {
        // Tile <-> Tile: exact shape/elem/space/layout; alignment subsumption ok
        return std::is_same_v<typename OutPayload::shape, typename InPayload::shape> &&
               std::is_same_v<typename OutPayload::elem, typename InPayload::elem> &&
               std::is_same_v<typename OutPayload::space, typename InPayload::space> &&
               std::is_same_v<typename OutPayload::layout, typename InPayload::layout> &&
               (OutPayload::align::bytes >= InPayload::align::bytes);
    }
    else if constexpr (RefPayload<OutPayload> && RefPayload<InPayload>) {
        // Ref <-> Ref: exact type equality
        return std::is_same_v<OutPayload, InPayload>;
    }
    else if constexpr (FragmentPayload<OutPayload> && FragmentPayload<InPayload>) {
        // Fragment <-> Fragment: exact shape/elem/dist
        return std::is_same_v<typename OutPayload::shape, typename InPayload::shape> &&
               std::is_same_v<typename OutPayload::elem, typename InPayload::elem> &&
               std::is_same_v<typename OutPayload::dist, typename InPayload::dist>;
    }
    else if constexpr (ScalarPayload<OutPayload> && ScalarPayload<InPayload>) {
        // Scalar <-> Scalar: exact elem/dist
        return std::is_same_v<typename OutPayload::elem, typename InPayload::elem> &&
               std::is_same_v<typename OutPayload::dist, typename InPayload::dist>;
    }
    else if constexpr (VectorPayload<OutPayload> && VectorPayload<InPayload>) {
        // Vector <-> Vector: exact elem/lanes/dist
        return std::is_same_v<typename OutPayload::elem, typename InPayload::elem> &&
               (OutPayload::lanes == InPayload::lanes) &&
               std::is_same_v<typename OutPayload::dist, typename InPayload::dist>;
    }
    else if constexpr (MaskPayload<OutPayload> && MaskPayload<InPayload>) {
        // Mask <-> Mask: exact width/dist
        return (OutPayload::width == InPayload::width) &&
               std::is_same_v<typename OutPayload::dist, typename InPayload::dist>;
    }
    else if constexpr (HandlePayload<OutPayload> && HandlePayload<InPayload>) {
        // Handle <-> Handle: exact type equality
        return std::is_same_v<OutPayload, InPayload>;
    }
    else {
        // Cross-kind: incompatible
        return false;
    }
}

/// Check port connection validity (§6.5, §6.6)
template<class OutPort, class InPort>
consteval bool port_satisfies() {
    static_assert(OutPort::is_output && InPort::is_input);

    // Payload must be compatible
    if constexpr (!payload_compatible<typename OutPort::payload, typename InPort::payload>()) {
        return false;
    }

    // Subject must match
    if constexpr (!std::is_same_v<typename OutPort::subject, typename InPort::subject>) {
        return false;
    }

    // §6.6: Dist must match for reg/fragment tiles and fragments
    if constexpr (detail::payload_requires_dist<typename OutPort::payload>()) {
        if constexpr (!std::is_same_v<typename OutPort::dist, typename InPort::dist>) {
            return false;
        }
    }

    // Provided tokens must satisfy required tokens
    return iro::verify::token_list_satisfies<
        typename OutPort::provided,
        typename InPort::required
    >();
}

} // namespace verify

} // namespace iro::contract

// =============================================================================
// §7. Explicit adapters: TileView
// =============================================================================

namespace iro::contract {
template<class InputPortsT, class OutputPortsT, class ResourcesT>
struct Obligation;
}

namespace iro::contract::adapter {

/// Explicit tile view adapter: reinterprets a tile without implicit conversion.
/// This is an obligation helper with no resources and explicit tokens.
template<
    class InTile,
    class OutTile,
    class SubjectT,
    class ExecGroupT,
    class RecipeT,
    class RequiredTokens,
    class ProvidedTokens
>
struct TileView {
    static_assert(TilePayload<InTile> && TilePayload<OutTile>,
        "TileView: InTile and OutTile must be TilePayloads");

    static_assert(std::is_same_v<typename InTile::elem, typename OutTile::elem>,
        "TileView: element types must match");

    static_assert(std::is_same_v<typename InTile::space, typename OutTile::space>,
        "TileView: memory spaces must match");

    static_assert(InTile::align::bytes >= OutTile::align::bytes,
        "TileView: input alignment must subsume output alignment");

    static_assert(InTile::shape::size == OutTile::shape::size,
        "TileView: shape sizes must match");

    using input_port = InputPort<InTile, SubjectT, ExecGroupT, RequiredTokens, no_dist, RecipeT>;
    using output_port = OutputPort<OutTile, SubjectT, ExecGroupT, ProvidedTokens, no_dist, RecipeT>;

    using obligation = iro::contract::Obligation<
        iro::util::type_list<input_port>,
        iro::util::type_list<output_port>,
        iro::util::type_list<>
    >;
};

} // namespace iro::contract::adapter

// =============================================================================
// §8. Obligations and realizations
// =============================================================================

namespace iro::contract {

// Mark obligations that are fused/semantic (not decomposed into lower-level atoms).
// Default: not fused.
template<class T>
struct is_fused_atom : std::false_type {};

template<class T>
inline constexpr bool is_fused_atom_v = is_fused_atom<T>::value;

// -----------------------------------------------------------------------------
// §8 Obligation well-formedness helpers
// -----------------------------------------------------------------------------

namespace detail {

/// Check token canonicality for all input ports
template<class InputPorts>
struct input_ports_tokens_canonical;

template<>
struct input_ports_tokens_canonical<iro::util::type_list<>> {
    static constexpr bool value = true;
};

template<class Port, class... Ports>
struct input_ports_tokens_canonical<iro::util::type_list<Port, Ports...>> {
    static constexpr bool value =
        iro::verify::token_list_canonical<typename Port::required>() &&
        input_ports_tokens_canonical<iro::util::type_list<Ports...>>::value;
};

/// Check token canonicality for all output ports
template<class OutputPorts>
struct output_ports_tokens_canonical;

template<>
struct output_ports_tokens_canonical<iro::util::type_list<>> {
    static constexpr bool value = true;
};

template<class Port, class... Ports>
struct output_ports_tokens_canonical<iro::util::type_list<Port, Ports...>> {
    static constexpr bool value =
        iro::verify::token_list_canonical<typename Port::provided>() &&
        output_ports_tokens_canonical<iro::util::type_list<Ports...>>::value;
};

template<class Port>
struct required_layout_resources {
    using exec_group = typename Port::exec_group;
    template<class G, bool IsWarpgroup = iro::exec::is_warpgroup_v<G>>
    struct impl {
        using type = iro::util::type_list<>;
    };
    template<class G>
    struct impl<G, true> {
        using type = iro::util::type_list<
            iro::contract::res::block_threads_multiple_of<iro::exec::warpgroup_warps<G>::value * 32>,
            iro::contract::res::warpgroup_layout<iro::exec::warpgroup_warps<G>::value, 32>,
            iro::contract::res::warpgroup_linear_x
        >;
    };
    using type = typename impl<exec_group>::type;
};

template<class Ports>
struct required_layout_resources_for_ports;

template<>
struct required_layout_resources_for_ports<iro::util::type_list<>> {
    using type = iro::util::type_list<>;
};

template<class P0, class... Ps>
struct required_layout_resources_for_ports<iro::util::type_list<P0, Ps...>> {
    using type = iro::util::concat_t<
        typename required_layout_resources<P0>::type,
        typename required_layout_resources_for_ports<iro::util::type_list<Ps...>>::type
    >;
};

template<class Resources, class Required>
struct resources_contain_all;

template<class Resources>
struct resources_contain_all<Resources, iro::util::type_list<>> : std::true_type {};

template<class Resources, class R0, class... Rs>
struct resources_contain_all<Resources, iro::util::type_list<R0, Rs...>>
    : std::bool_constant<
        iro::util::contains<Resources, R0>::value &&
        resources_contain_all<Resources, iro::util::type_list<Rs...>>::value> {};

} // namespace detail

/// Obligation descriptor (what a kernel step requires/provides)
/// Enforces well-formedness per §8: token canonicality + mandatory tokens + resource canonicality
template<class InputPortsT, class OutputPortsT, class ResourcesT>
struct Obligation {
    // §8: Token canonicality for all input port required tokens
    static_assert(detail::input_ports_tokens_canonical<InputPortsT>::value,
        "Obligation: all input port required token lists must be canonical (no duplicate kind+subject)");

    // §8: Token canonicality for all output port provided tokens
    static_assert(detail::output_ports_tokens_canonical<OutputPortsT>::value,
        "Obligation: all output port provided token lists must be canonical (no duplicate kind+subject)");

    // §8: Resource canonicality
    static_assert(iro::verify::resource_list_canonical<ResourcesT>(),
        "Obligation: resource list must be canonical (no duplicate ids)");

    using required_layout_resources = iro::verify::canonicalize_resource_list<
        iro::util::concat_t<
            typename detail::required_layout_resources_for_ports<InputPortsT>::type,
            typename detail::required_layout_resources_for_ports<OutputPortsT>::type
        >
    >;

    static_assert(detail::resources_contain_all<ResourcesT, required_layout_resources>::value,
        "Obligation: warpgroup exec_group requires warpgroup_layout + block_threads_multiple_of");

    // Note: Mandatory token rules (§6.4) are enforced by InputPort/OutputPort static_asserts

    using inputs = InputPortsT;
    using outputs = OutputPortsT;
    using resources = ResourcesT;

    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.contract.Obligation"),
            iro::util::mix_u64(iro::util::hash_list_v<InputPortsT>,
                iro::util::mix_u64(iro::util::hash_list_v<OutputPortsT>,
                    iro::util::hash_list_v<ResourcesT>)));
};

/// Realization stub (actual implementation provided externally)
template<class ObligationT, iro::util::u64 ImplId>
struct Realization {
    using obligation = ObligationT;
    static constexpr iro::util::u64 impl_id = ImplId;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.contract.Realization"),
            iro::util::mix_u64(ObligationT::id, ImplId));
};

} // namespace iro::contract

// =============================================================================
// §9. Capabilities (iro::cap)
// =============================================================================

namespace iro::cap {

// SM architecture capabilities
struct sm89 {
    static constexpr int sm_version = 89;
    static constexpr bool has_fp8 = true;
    static constexpr bool has_wgmma = false;
    static constexpr bool has_tma = false;
    static constexpr bool has_f16_atomics = false;
    static constexpr int warpgroup_warps = 4;
    static constexpr int max_smem_per_block = 100 * 1024;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.cap.sm89");
};

struct sm90 {
    static constexpr int sm_version = 90;
    static constexpr bool has_fp8 = true;
    static constexpr bool has_wgmma = true;
    static constexpr bool has_tma = true;
    static constexpr bool has_f16_atomics = true;
    static constexpr int warpgroup_warps = 4;
    static constexpr int max_smem_per_block = 227 * 1024;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.cap.sm90");
};

struct sm100 {
    static constexpr int sm_version = 100;
    static constexpr bool has_fp8 = true;
    static constexpr bool has_wgmma = true;
    static constexpr bool has_tma = true;
    static constexpr bool has_f16_atomics = true;
    static constexpr int warpgroup_warps = 4;
    static constexpr int max_smem_per_block = 227 * 1024;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.cap.sm100");
};

/// Check if capability C supports required capability R
template<class C, class R>
consteval bool cap_supports() {
    return C::sm_version >= R::sm_version;
}

} // namespace iro::cap

// =============================================================================
// §10. Profiles (iro::profile)
// =============================================================================

namespace iro::profile {

/// Budget descriptor
template<long long MaxSmem, int MaxRegs, int MaxBarriers>
struct Budget {
    static constexpr long long max_smem = MaxSmem;
    static constexpr int max_regs = MaxRegs;
    static constexpr int max_barriers = MaxBarriers;

    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.profile.Budget"),
            iro::util::mix_u64(static_cast<iro::util::u64>(MaxSmem),
                iro::util::mix_u64(static_cast<iro::util::u64>(MaxRegs),
                    static_cast<iro::util::u64>(MaxBarriers))));
};

/// Standard budget aliases
using BudgetSmall  = Budget<16 * 1024, 64, 2>;
using BudgetMedium = Budget<48 * 1024, 128, 8>;
using BudgetLarge  = Budget<100 * 1024, 255, 16>;
using BudgetMax    = Budget<228 * 1024, 255, 16>;

} // namespace iro::profile

// =============================================================================
// §11. Registry (iro::registry)
// =============================================================================

namespace iro::registry {

/// Element registry entry
template<class ElemT>
struct ElemEntry {
    using elem = ElemT;
    static constexpr auto id = ElemT::id;
};

/// Realization registry entry
template<class RealizationT, class... Caps>
struct RealizationEntry {
    using realization = RealizationT;
    using caps = iro::util::type_list<Caps...>;
    static constexpr auto id = RealizationT::id;
};

} // namespace iro::registry

// =============================================================================
// §12. Composition (iro::compose)
// =============================================================================

namespace iro::compose {

namespace detail {

template<class T, class = void>
struct has_port : std::false_type {};

template<class T>
struct has_port<T, std::void_t<typename T::port>> : std::true_type {};

template<class T>
inline constexpr bool has_port_v = has_port<T>::value;

template<class T, class = void>
struct port_of {
    using type = T;
};

template<class T>
struct port_of<T, std::void_t<typename T::port>> {
    using type = typename T::port;
};

template<class T>
using port_of_t = typename port_of<T>::type;

} // namespace detail

template<class Obligation, int Index>
struct in_port_ref {
    static_assert(Index >= 0, "in_port_ref: index must be non-negative");
    static_assert(Index < iro::util::size_v<typename Obligation::inputs>,
                  "in_port_ref: index out of range");
    using obligation = Obligation;
    using port = iro::util::at_t<typename Obligation::inputs, Index>;
    using payload = typename port::payload;
    using subject = typename port::subject;
    using exec_group = typename port::exec_group;
    using required = typename port::required;
    using dist = typename port::dist;
    using recipe = typename port::recipe;
    static constexpr int index = Index;
    static constexpr bool is_input = true;
    static constexpr bool is_output = false;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.compose.in_port_ref"),
            iro::util::mix_u64(Obligation::id, port::id));
};

template<class Obligation, int Index>
struct out_port_ref {
    static_assert(Index >= 0, "out_port_ref: index must be non-negative");
    static_assert(Index < iro::util::size_v<typename Obligation::outputs>,
                  "out_port_ref: index out of range");
    using obligation = Obligation;
    using port = iro::util::at_t<typename Obligation::outputs, Index>;
    using payload = typename port::payload;
    using subject = typename port::subject;
    using exec_group = typename port::exec_group;
    using provided = typename port::provided;
    using dist = typename port::dist;
    using recipe = typename port::recipe;
    static constexpr int index = Index;
    static constexpr bool is_input = false;
    static constexpr bool is_output = true;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.compose.out_port_ref"),
            iro::util::mix_u64(Obligation::id, port::id));
};

/// Edge connecting two ports
template<class OutRefT, class InRefT>
struct Edge {
    static_assert(detail::has_port_v<OutRefT> && detail::has_port_v<InRefT>,
        "Edge: ports must be compose::in_port_ref/out_port_ref");
    using out_ref = OutRefT;
    using in_ref = InRefT;
    using out_port = detail::port_of_t<OutRefT>;
    using in_port = detail::port_of_t<InRefT>;

    static_assert(out_port::is_output && in_port::is_input);
    static_assert(iro::contract::verify::port_satisfies<out_port, in_port>(),
        "Edge: output port does not satisfy input port requirements");
    static_assert(
        (std::is_same_v<typename out_port::recipe, iro::recipe::no_recipe> &&
         std::is_same_v<typename in_port::recipe, iro::recipe::no_recipe>) ||
        (!std::is_same_v<typename out_port::recipe, iro::recipe::no_recipe> &&
         !std::is_same_v<typename in_port::recipe, iro::recipe::no_recipe> &&
         iro::verify::recipe_compatible<typename out_port::recipe, typename in_port::recipe>()),
        "Edge: recipes must be both no_recipe or both explicit and compatible");

    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.compose.Edge"),
            iro::util::mix_u64(out_ref::id, in_ref::id));
};

namespace detail {

template<class ObligationList>
struct inputs_of;
template<>
struct inputs_of<iro::util::type_list<>> {
    using type = iro::util::type_list<>;
};

template<class Obligation, int I, int N>
struct input_refs_impl;

template<class Obligation, int N>
struct input_refs_impl<Obligation, N, N> {
    using type = iro::util::type_list<>;
};

template<class Obligation, int I, int N>
struct input_refs_impl {
    using type = iro::util::concat_t<
        iro::util::type_list<iro::compose::in_port_ref<Obligation, I>>,
        typename input_refs_impl<Obligation, I + 1, N>::type
    >;
};

template<class O0, class... Os>
struct inputs_of<iro::util::type_list<O0, Os...>> {
    using self = typename input_refs_impl<O0, 0, iro::util::size_v<typename O0::inputs>>::type;
    using type = iro::util::concat_t<self, typename inputs_of<iro::util::type_list<Os...>>::type>;
};

template<class ObligationList>
struct outputs_of;
template<>
struct outputs_of<iro::util::type_list<>> {
    using type = iro::util::type_list<>;
};

template<class Obligation, int I, int N>
struct output_refs_impl;

template<class Obligation, int N>
struct output_refs_impl<Obligation, N, N> {
    using type = iro::util::type_list<>;
};

template<class Obligation, int I, int N>
struct output_refs_impl {
    using type = iro::util::concat_t<
        iro::util::type_list<iro::compose::out_port_ref<Obligation, I>>,
        typename output_refs_impl<Obligation, I + 1, N>::type
    >;
};

template<class O0, class... Os>
struct outputs_of<iro::util::type_list<O0, Os...>> {
    using self = typename output_refs_impl<O0, 0, iro::util::size_v<typename O0::outputs>>::type;
    using type = iro::util::concat_t<self, typename outputs_of<iro::util::type_list<Os...>>::type>;
};

template<class ObligationList>
struct resources_of;
template<>
struct resources_of<iro::util::type_list<>> {
    using type = iro::util::type_list<>;
};
template<class O0, class... Os>
struct resources_of<iro::util::type_list<O0, Os...>> {
    using type = iro::util::concat_t<typename O0::resources,
                                     typename resources_of<iro::util::type_list<Os...>>::type>;
};

template<class ObligationList>
struct has_fused_obligation;

template<>
struct has_fused_obligation<iro::util::type_list<>> : std::false_type {};

template<class O0, class... Os>
struct has_fused_obligation<iro::util::type_list<O0, Os...>>
    : std::bool_constant<
        iro::contract::is_fused_atom_v<O0> ||
        has_fused_obligation<iro::util::type_list<Os...>>::value> {};

template<class Ports, class EdgeList>
struct remove_edge_inputs;
template<class Ports>
struct remove_edge_inputs<Ports, iro::util::type_list<>> {
    using type = Ports;
    static constexpr bool ok = true;
};
template<class Ports, class E0, class... Es>
struct remove_edge_inputs<Ports, iro::util::type_list<E0, Es...>> {
    using removed = iro::util::remove_one<Ports, typename E0::in_ref>;
    static constexpr bool ok =
        removed::removed && remove_edge_inputs<typename removed::type, iro::util::type_list<Es...>>::ok;
    using type = typename remove_edge_inputs<typename removed::type, iro::util::type_list<Es...>>::type;
};

template<class EdgeList>
struct edge_out_ports;
template<>
struct edge_out_ports<iro::util::type_list<>> {
    using type = iro::util::type_list<>;
};
template<class E0, class... Es>
struct edge_out_ports<iro::util::type_list<E0, Es...>> {
    using type = iro::util::concat_t<iro::util::type_list<typename E0::out_ref>,
                                     typename edge_out_ports<iro::util::type_list<Es...>>::type>;
};

template<class Ports, class RemoveList>
struct remove_ports_by_type;
template<class RemoveList>
struct remove_ports_by_type<iro::util::type_list<>, RemoveList> {
    using type = iro::util::type_list<>;
};
template<class P0, class... Ps, class RemoveList>
struct remove_ports_by_type<iro::util::type_list<P0, Ps...>, RemoveList> {
    using rest = typename remove_ports_by_type<iro::util::type_list<Ps...>, RemoveList>::type;
    using type = std::conditional_t<iro::util::contains<RemoveList, P0>::value,
        rest,
        iro::util::prepend_t<rest, P0>
    >;
};

template<class Ports, class EdgeList>
struct remove_edge_outputs {
    using type = typename remove_ports_by_type<
        Ports,
        typename edge_out_ports<EdgeList>::type
    >::type;
    static constexpr bool ok = true;
};

template<class Ports, class EdgeList>
struct edges_outputs_exist;
template<class Ports>
struct edges_outputs_exist<Ports, iro::util::type_list<>> {
    static constexpr bool ok = true;
};
template<class Ports, class E0, class... Es>
struct edges_outputs_exist<Ports, iro::util::type_list<E0, Es...>> {
    static constexpr bool ok =
        iro::util::contains<Ports, typename E0::out_ref>::value &&
        edges_outputs_exist<Ports, iro::util::type_list<Es...>>::ok;
};

template<class Obligation, class EdgeList>
struct incoming_count;
template<class Obligation>
struct incoming_count<Obligation, iro::util::type_list<>> : std::integral_constant<int, 0> {};
template<class Obligation, class E0, class... Es>
struct incoming_count<Obligation, iro::util::type_list<E0, Es...>>
    : std::integral_constant<int,
        (std::is_same_v<typename E0::in_ref::obligation, Obligation> ? 1 : 0) +
        incoming_count<Obligation, iro::util::type_list<Es...>>::value> {};

template<class Obligation, class EdgeList>
struct outgoing_neighbors;
template<class Obligation>
struct outgoing_neighbors<Obligation, iro::util::type_list<>> {
    using type = iro::util::type_list<>;
};
template<class Obligation, class E0, class... Es>
struct outgoing_neighbors<Obligation, iro::util::type_list<E0, Es...>> {
    using rest = typename outgoing_neighbors<Obligation, iro::util::type_list<Es...>>::type;
    using type = std::conditional_t<
        std::is_same_v<typename E0::out_ref::obligation, Obligation>,
        iro::util::prepend_t<rest, typename E0::in_ref::obligation>,
        rest
    >;
};

template<class ObligationList, class EdgeList>
struct incoming_counts;
template<class EdgeList>
struct incoming_counts<iro::util::type_list<>, EdgeList> {
    using type = iro::util::type_list<>;
};
template<class O0, class... Os, class EdgeList>
struct incoming_counts<iro::util::type_list<O0, Os...>, EdgeList> {
    using head = std::integral_constant<int, incoming_count<O0, EdgeList>::value>;
    using tail = typename incoming_counts<iro::util::type_list<Os...>, EdgeList>::type;
    using type = iro::util::prepend_t<tail, head>;
};

template<class ObligationList, class EdgeList>
struct adjacency_list;
template<class EdgeList>
struct adjacency_list<iro::util::type_list<>, EdgeList> {
    using type = iro::util::type_list<>;
};
template<class O0, class... Os, class EdgeList>
struct adjacency_list<iro::util::type_list<O0, Os...>, EdgeList> {
    using head = typename outgoing_neighbors<O0, EdgeList>::type;
    using tail = typename adjacency_list<iro::util::type_list<Os...>, EdgeList>::type;
    using type = iro::util::prepend_t<tail, head>;
};

template<class NeighborList, class ObligationList, class CountList>
struct decrement_counts;
template<class NeighborList>
struct decrement_counts<NeighborList, iro::util::type_list<>, iro::util::type_list<>> {
    using type = iro::util::type_list<>;
};
template<class NeighborList, class O0, class... Os, class C0, class... Cs>
struct decrement_counts<NeighborList, iro::util::type_list<O0, Os...>, iro::util::type_list<C0, Cs...>> {
    static constexpr int dec = iro::util::count_v<NeighborList, O0>;
    using head = std::conditional_t<
        (dec > 0),
        std::integral_constant<int, C0::value - dec>,
        C0
    >;
    using tail = typename decrement_counts<
        NeighborList,
        iro::util::type_list<Os...>,
        iro::util::type_list<Cs...>>::type;
    using type = iro::util::prepend_t<tail, head>;
};

template<class CountList>
struct find_zero_count;
template<>
struct find_zero_count<iro::util::type_list<>> {
    static constexpr int value = -1;
};
template<class C0, class... Cs>
struct find_zero_count<iro::util::type_list<C0, Cs...>> {
    static constexpr int tail = find_zero_count<iro::util::type_list<Cs...>>::value;
    static constexpr int value = (C0::value == 0) ? 0 : (tail < 0 ? -1 : tail + 1);
};

template<class CountList, class ObligationList, class AdjList>
struct acyclic_impl_counts;
template<int ZeroIndex, class ObligationList, class CountList, class AdjList>
struct acyclic_step;
template<class AdjList>
struct acyclic_impl_counts<iro::util::type_list<>, iro::util::type_list<>, AdjList> : std::true_type {};

template<class CountList, class ObligationList, class AdjList>
struct acyclic_impl_counts {
    static constexpr int zero = find_zero_count<CountList>::value;
    static constexpr bool value =
        (zero < 0) ? false : acyclic_step<zero, ObligationList, CountList, AdjList>::value;
};

template<int ZeroIndex, class ObligationList, class CountList, class AdjList>
struct acyclic_step {
    using neighbors = iro::util::at_t<AdjList, ZeroIndex>;
    using dec_counts = typename decrement_counts<neighbors, ObligationList, CountList>::type;
    using next_obligations = iro::util::remove_at_t<ObligationList, ZeroIndex>;
    using next_counts = iro::util::remove_at_t<dec_counts, ZeroIndex>;
    using next_adj = iro::util::remove_at_t<AdjList, ZeroIndex>;
    static constexpr bool value =
        acyclic_impl_counts<next_counts, next_obligations, next_adj>::value;
};

template<class ObligationList, class EdgeList>
struct acyclic {
    using counts = typename incoming_counts<ObligationList, EdgeList>::type;
    using adjacency = typename adjacency_list<ObligationList, EdgeList>::type;
    static constexpr bool value = acyclic_impl_counts<counts, ObligationList, adjacency>::value;
};

} // namespace detail

template<class ObligationList, class EdgeList, class ResourceList, class ProfileT, class CapT>
struct Composition;

template<class ObligationList, class EdgeList, class ProfileT, class CapT>
using CompositionAutoResources = Composition<
    ObligationList,
    EdgeList,
    iro::verify::canonicalize_resource_list<typename detail::resources_of<ObligationList>::type>,
    ProfileT,
    CapT>;

template<class CompA, class CompB, class ExtraEdges = iro::util::type_list<>>
struct join {
    static_assert(std::is_same_v<typename CompA::profile, typename CompB::profile>,
        "compose::join: profile mismatch");
    static_assert(std::is_same_v<typename CompA::cap, typename CompB::cap>,
        "compose::join: cap mismatch");

    using obligations = iro::util::concat_t<typename CompA::obligations, typename CompB::obligations>;
    using edges = iro::util::concat_t<
        iro::util::concat_t<typename CompA::edges, typename CompB::edges>,
        ExtraEdges
    >;

    using type = CompositionAutoResources<obligations, edges, typename CompA::profile, typename CompA::cap>;
};

template<class CompA, class CompB, class ExtraEdges = iro::util::type_list<>>
using join_t = typename join<CompA, CompB, ExtraEdges>::type;

/// Composition of multiple obligations with edges
template<class ObligationList, class EdgeList, class ResourceList, class ProfileT, class CapT>
struct Composition {
    using obligations = ObligationList;
    using edges = EdgeList;
    using resources = ResourceList;
    using profile = ProfileT;
    using cap = CapT;
    using inputs = typename detail::remove_edge_inputs<
        typename detail::inputs_of<ObligationList>::type,
        EdgeList
    >::type;
    using outputs = typename detail::remove_edge_outputs<
        typename detail::outputs_of<ObligationList>::type,
        EdgeList
    >::type;
    using expected_resources_raw = typename detail::resources_of<ObligationList>::type;
    using expected_resources = iro::verify::canonicalize_resource_list<expected_resources_raw>;
    using canonical_resources = iro::verify::canonicalize_resource_list<ResourceList>;
    using provided_resources = canonical_resources;
    static constexpr bool has_fused_obligation =
        detail::has_fused_obligation<ObligationList>::value;

    static_assert(iro::verify::resources_ok_union<expected_resources_raw>(),
        "Composition: conflicting resources across obligations");
    static_assert(iro::verify::resources_ok_union<ResourceList>(),
        "Composition: resource conflict detected");
    static_assert(std::is_same_v<expected_resources, provided_resources>,
        "Composition: resources must equal canonical union of obligations");
    static_assert(iro::verify::detail::sum_smem<canonical_resources>::value <= ProfileT::max_smem,
        "Composition: shared memory exceeds profile budget");
    static_assert(iro::verify::detail::sum_smem<canonical_resources>::value <= CapT::max_smem_per_block,
        "Composition: shared memory exceeds target cap");
    static_assert(iro::verify::detail::sum_barriers<canonical_resources>::value <= ProfileT::max_barriers,
        "Composition: barrier count exceeds profile budget");
    static_assert(iro::verify::detail::sum_regs<canonical_resources>::value <= ProfileT::max_regs,
        "Composition: register budget exceeds profile");
    static_assert(iro::verify::detail::max_reg_pressure<canonical_resources>::value <= ProfileT::max_regs,
        "Composition: register pressure exceeds profile");
    static_assert(detail::remove_edge_inputs<
                      typename detail::inputs_of<ObligationList>::type, EdgeList>::ok,
        "Composition: edge input port not found or connected more than once");
    static_assert(detail::edges_outputs_exist<
                      typename detail::outputs_of<ObligationList>::type, EdgeList>::ok,
        "Composition: edge output port not found");
    static_assert(detail::acyclic<ObligationList, EdgeList>::value,
        "Composition: edge graph contains a cycle");

    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.compose.Composition"),
            iro::util::mix_u64(iro::util::hash_list_v<ObligationList>,
                iro::util::mix_u64(iro::util::hash_list_v<EdgeList>,
                    iro::util::mix_u64(iro::util::hash_list_v<ResourceList>,
                        iro::util::mix_u64(ProfileT::id, CapT::id)))));
};

} // namespace iro::compose

// =============================================================================
// §13. Binding (iro::bind)
// =============================================================================

namespace iro::bind {

/// Binding result for a single obligation
template<class ObligationT, class RealizationT>
struct Resolved {
    using obligation = ObligationT;
    using realization = RealizationT;

    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.bind.Resolved"),
            iro::util::mix_u64(ObligationT::id, RealizationT::id));
};

/// Full binding of a composition
template<class CompositionT, class ResolvedList>
struct Resolution {
    using composition = CompositionT;
    using resolved = ResolvedList;

    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.bind.Resolution"),
            iro::util::mix_u64(CompositionT::id,
                static_cast<iro::util::u64>(iro::util::size_v<ResolvedList>)));
};

// -----------------------------------------------------------------------------
// Spec §8: Deterministic per-leaf binding selection
// -----------------------------------------------------------------------------

/// Marker for "no realization found"
struct no_realization {
    static constexpr iro::util::u64 id = 0;
};

namespace detail {

template<class T, class = void>
struct entry_traits {
    using realization = T;
    using caps = iro::util::type_list<>;
    static constexpr bool has_caps = false;
};

template<class T>
struct entry_traits<T, std::void_t<typename T::realization, typename T::caps>> {
    using realization = typename T::realization;
    using caps = typename T::caps;
    static constexpr bool has_caps = true;
};

/// Check if registry entry's realization matches obligation by id
template<class Entry, class ObligationT>
consteval bool realization_matches_obligation() {
    using RealizationT = typename entry_traits<Entry>::realization;
    return std::is_same_v<typename RealizationT::obligation, ObligationT>;
}

template<class CapsList, class RequiredCap>
struct caps_supports;

template<class RequiredCap>
struct caps_supports<iro::util::type_list<>, RequiredCap> {
    static constexpr bool value = false;
};

template<class Cap, class... Caps, class RequiredCap>
struct caps_supports<iro::util::type_list<Cap, Caps...>, RequiredCap> {
    static constexpr bool value =
        iro::cap::cap_supports<Cap, RequiredCap>() ||
        caps_supports<iro::util::type_list<Caps...>, RequiredCap>::value;
};

/// Check if registry entry supports required capability
template<class Entry, class RequiredCap>
consteval bool realization_supports_cap() {
    if constexpr (!entry_traits<Entry>::has_caps) {
        return false; // Caps are required for explicit resolution
    } else {
        using CapsList = typename entry_traits<Entry>::caps;
        return caps_supports<CapsList, RequiredCap>::value;
    }
}

/// Find first matching realization in a registry list
template<class ObligationT, class RequiredCap, class RealizationList>
struct find_realization;

template<class ObligationT, class RequiredCap>
struct find_realization<ObligationT, RequiredCap, iro::util::type_list<>> {
    using type = no_realization;
    static constexpr bool found = false;
};

template<class ObligationT, class RequiredCap, class R, class... Rs>
struct find_realization<ObligationT, RequiredCap, iro::util::type_list<R, Rs...>> {
private:
    static constexpr bool this_matches =
        realization_matches_obligation<R, ObligationT>() &&
        realization_supports_cap<R, RequiredCap>();
public:
    using type = std::conditional_t<
        this_matches,
        R,
        typename find_realization<ObligationT, RequiredCap, iro::util::type_list<Rs...>>::type
    >;
    static constexpr bool found = this_matches ||
        find_realization<ObligationT, RequiredCap, iro::util::type_list<Rs...>>::found;
};

/// Count matching realizations (for ambiguity detection)
template<class ObligationT, class RequiredCap, class RealizationList>
struct count_realizations;

template<class ObligationT, class RequiredCap>
struct count_realizations<ObligationT, RequiredCap, iro::util::type_list<>> {
    static constexpr int value = 0;
};

template<class ObligationT, class RequiredCap, class R, class... Rs>
struct count_realizations<ObligationT, RequiredCap, iro::util::type_list<R, Rs...>> {
    static constexpr int value =
        ((realization_matches_obligation<R, ObligationT>() &&
          realization_supports_cap<R, RequiredCap>()) ? 1 : 0) +
        count_realizations<ObligationT, RequiredCap, iro::util::type_list<Rs...>>::value;
};

} // namespace detail

/// Match a realization for an obligation from a registry
/// Returns no_realization if not found
/// Deterministic: always picks first matching (by list order)
template<class ObligationT, class RequiredCap, class RealizationRegistry>
struct match_realization {
    using found_entry = typename detail::find_realization<
        ObligationT, RequiredCap, RealizationRegistry>::type;
    using type = typename detail::entry_traits<found_entry>::realization;

    static constexpr bool found =
        detail::find_realization<ObligationT, RequiredCap, RealizationRegistry>::found;

    static constexpr int match_count =
        detail::count_realizations<ObligationT, RequiredCap, RealizationRegistry>::value;

    // Warn on ambiguity (multiple matches)
    static constexpr bool ambiguous = match_count > 1;
};

/// Convenience alias for matching
template<class ObligationT, class RequiredCap, class RealizationRegistry>
using match_realization_t = typename match_realization<ObligationT, RequiredCap, RealizationRegistry>::type;

/// Lookup a realization for an obligation (deterministic per-leaf)
/// Produces Resolved<Obligation, Realization> if found, or compile error if not
template<class ObligationT, class RequiredCap, class RealizationRegistry>
struct lookup_realization {
    using selected = match_realization<ObligationT, RequiredCap, RealizationRegistry>;

    static_assert(selected::found,
        "lookup_realization: no realization found for obligation (§8: binding failure)");

    static_assert(!selected::ambiguous,
        "lookup_realization: multiple realizations found (ambiguous)");

    using type = Resolved<ObligationT, typename selected::type>;
};

template<class ObligationT, class RequiredCap, class RealizationRegistry>
using lookup_realization_t = typename lookup_realization<ObligationT, RequiredCap, RealizationRegistry>::type;

} // namespace iro::bind

// =============================================================================
// §14. Diagnostics (iro::diag)
// =============================================================================

namespace iro::diag {

/// Diagnostic codes
enum class Code : int {
    OK = 0,

    // Token errors (100-199)
    MISSING_VISIBLE_AT = 100,
    MISSING_ALIVE = 101,
    MISSING_SYNC_AT = 102,
    SCOPE_INSUFFICIENT = 103,
    LIFETIME_INSUFFICIENT = 104,
    LANES_INSUFFICIENT = 105,
    SLOT_STATE_MISMATCH = 106,
    LEASE_MISMATCH = 107,
    TOKEN_LIST_NOT_CANONICAL = 108,

    // Payload errors (200-299)
    PAYLOAD_INCOMPATIBLE = 200,
    SHAPE_MISMATCH = 201,
    ELEM_MISMATCH = 202,
    SPACE_MISMATCH = 203,
    LAYOUT_MISMATCH = 204,
    ALIGNMENT_INSUFFICIENT = 205,
    DIST_MISMATCH = 206,

    // Resource errors (300-399)
    RESOURCE_CONFLICT = 300,
    SMEM_OVERFLOW = 301,
    REG_OVERFLOW = 302,
    BARRIER_OVERFLOW = 303,

    // Port errors (400-499)
    PORT_SUBJECT_MISMATCH = 400,
    PORT_DIRECTION_ERROR = 401,

    // Composition errors (500-599)
    EDGE_INVALID = 500,
    CYCLE_DETECTED = 501,
    UNCONNECTED_INPUT = 502,

    // Capability errors (600-699)
    CAP_UNSUPPORTED = 600,
    SM_VERSION_INSUFFICIENT = 601,

    // Resolution errors (700-799)
    NO_REALIZATION_FOUND = 700,
    AMBIGUOUS_REALIZATION = 701,
};

/// Diagnostic message template
template<Code C, iro::util::u64 Context = 0>
struct Diagnostic {
    static constexpr Code code = C;
    static constexpr iro::util::u64 context = Context;
    static constexpr bool is_error = static_cast<int>(C) != 0;
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.diag.Diagnostic"),
            iro::util::mix_u64(static_cast<iro::util::u64>(static_cast<int>(C)), Context));
};

using OK = Diagnostic<Code::OK>;

} // namespace iro::diag

// =============================================================================
// §15. Standard element types
// =============================================================================

namespace iro::elem {

// Floating point types
struct f16 {
    static constexpr int bytes = 2;
    static constexpr int align = 2;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.f16");
#ifdef __CUDACC__
    using storage_t = __half;
#endif
};

struct bf16 {
    static constexpr int bytes = 2;
    static constexpr int align = 2;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.bf16");
#ifdef __CUDACC__
    using storage_t = __nv_bfloat16;
#endif
};

struct f32 {
    static constexpr int bytes = 4;
    static constexpr int align = 4;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.f32");
#ifdef __CUDACC__
    using storage_t = float;
#endif
};

struct f64 {
    static constexpr int bytes = 8;
    static constexpr int align = 8;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.f64");
#ifdef __CUDACC__
    using storage_t = double;
#endif
};

// TF32 (tensor-float-32)
struct tf32 {
    static constexpr int bytes = 4;
    static constexpr int align = 4;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.tf32");
#ifdef __CUDACC__
    using storage_t = float;
#endif
};

// FP8 types
struct e4m3 {
    static constexpr int bytes = 1;
    static constexpr int align = 1;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.e4m3");
#ifdef __CUDACC__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    using storage_t = __nv_fp8_e4m3;
#endif
#endif
};

struct e5m2 {
    static constexpr int bytes = 1;
    static constexpr int align = 1;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.e5m2");
#ifdef __CUDACC__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    using storage_t = __nv_fp8_e5m2;
#endif
#endif
};

// FP8 variants (finite-only / non-NaN)
struct e4m3fn {
    static constexpr int bytes = 1;
    static constexpr int align = 1;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.e4m3fn");
#ifdef __CUDACC__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    using storage_t = __nv_fp8_e4m3;
#endif
#endif
};

struct e5m2fnuz {
    static constexpr int bytes = 1;
    static constexpr int align = 1;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.e5m2fnuz");
#ifdef __CUDACC__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    using storage_t = __nv_fp8_e5m2;
#endif
#endif
};

// FP4 (Blackwell)
struct e2m1 {
    static constexpr int bytes = 1;
    static constexpr int align = 1;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.e2m1");
#ifdef __CUDACC__
    struct storage_t { uint8_t v; };
#endif
};

// Integer types
struct i8 {
    static constexpr int bytes = 1;
    static constexpr int align = 1;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.i8");
#ifdef __CUDACC__
    using storage_t = int8_t;
#endif
};

struct i16 {
    static constexpr int bytes = 2;
    static constexpr int align = 2;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.i16");
#ifdef __CUDACC__
    using storage_t = int16_t;
#endif
};

struct i32 {
    static constexpr int bytes = 4;
    static constexpr int align = 4;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.i32");
#ifdef __CUDACC__
    using storage_t = int32_t;
#endif
};

struct i64 {
    static constexpr int bytes = 8;
    static constexpr int align = 8;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.i64");
#ifdef __CUDACC__
    using storage_t = int64_t;
#endif
};

struct u8 {
    static constexpr int bytes = 1;
    static constexpr int align = 1;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.u8");
#ifdef __CUDACC__
    using storage_t = uint8_t;
#endif
};

struct u16 {
    static constexpr int bytes = 2;
    static constexpr int align = 2;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.u16");
#ifdef __CUDACC__
    using storage_t = uint16_t;
#endif
};

struct u32 {
    static constexpr int bytes = 4;
    static constexpr int align = 4;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.u32");
#ifdef __CUDACC__
    using storage_t = uint32_t;
#endif
};

struct u64 {
    static constexpr int bytes = 8;
    static constexpr int align = 8;
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.elem.u64");
#ifdef __CUDACC__
    using storage_t = uint64_t;
#endif
};

} // namespace iro::elem

// =============================================================================
// §16. Fragment distribution patterns
// =============================================================================

namespace iro::dist {

/// Warp-level distribution (for WMMA)
struct warp_row_major {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.dist.warp_row_major");
};

struct warp_col_major {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.dist.warp_col_major");
};

/// Warpgroup-level distribution (for WGMMA)
struct warpgroup_row_major {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.dist.warpgroup_row_major");
};

struct warpgroup_col_major {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.dist.warpgroup_col_major");
};

/// Accumulator distribution
struct accumulator {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.dist.accumulator");
};

/// Register tile ownership (per-thread register tile)
struct reg_owned {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.dist.reg_owned");
};

/// TMEM layouts (Blackwell)
struct tmem_row_major {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.dist.tmem_row_major");
};

struct tmem_col_major {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.dist.tmem_col_major");
};

// -------------------------------------------------------------------------
// Scalar/Vector/Mask distributions (general-purpose)
// -------------------------------------------------------------------------

/// Per-lane distribution (one value per lane/thread)
struct lane {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.dist.lane");
};

/// Replicated distribution (identical value stored per lane)
struct replicated {
    static constexpr auto id = iro::util::fnv1a_64_cstr("iro.dist.replicated");
};

/// Uniform distribution across a scope (one value shared)
template<class ScopeT>
struct uniform {
    static constexpr auto id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.dist.uniform"), ScopeT::id);
    using scope = ScopeT;
};

/// Cyclic distribution across a scope (periodic lane mapping)
template<class ScopeT, int Period>
struct cyclic {
    static_assert(Period > 0, "dist::cyclic requires positive period");
    static constexpr auto id =
        iro::util::mix_u64(
            iro::util::fnv1a_64_cstr("iro.dist.cyclic"),
            iro::util::mix_u64(ScopeT::id, static_cast<iro::util::u64>(Period)));
    using scope = ScopeT;
    static constexpr int period = Period;
};

/// Striped distribution across a scope (contiguous stripe per lane group)
template<int Lanes>
struct striped {
    static_assert(Lanes > 0, "dist::striped requires positive lane count");
    static constexpr auto id =
        iro::util::mix_u64(
            iro::util::fnv1a_64_cstr("iro.dist.striped"),
            static_cast<iro::util::u64>(Lanes));
    static constexpr int lanes = Lanes;
};

/// Mask distribution scoped to a scope (bitmask for active lanes)
template<class ScopeT>
struct mask {
    static constexpr auto id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("iro.dist.mask"), ScopeT::id);
    using scope = ScopeT;
};

} // namespace iro::dist

namespace iro::schema {
template<> struct is_dist<iro::dist::warp_row_major> : std::true_type {};
template<> struct is_dist<iro::dist::warp_col_major> : std::true_type {};
template<> struct is_dist<iro::dist::warpgroup_row_major> : std::true_type {};
template<> struct is_dist<iro::dist::warpgroup_col_major> : std::true_type {};
template<> struct is_dist<iro::dist::accumulator> : std::true_type {};
template<> struct is_dist<iro::dist::reg_owned> : std::true_type {};
template<> struct is_dist<iro::dist::tmem_row_major> : std::true_type {};
template<> struct is_dist<iro::dist::tmem_col_major> : std::true_type {};
template<> struct is_dist<iro::dist::lane> : std::true_type {};
template<> struct is_dist<iro::dist::replicated> : std::true_type {};
template<class ScopeT> struct is_dist<iro::dist::uniform<ScopeT>> : std::true_type {};
template<class ScopeT, int Period> struct is_dist<iro::dist::cyclic<ScopeT, Period>> : std::true_type {};
template<int Lanes> struct is_dist<iro::dist::striped<Lanes>> : std::true_type {};
template<class ScopeT> struct is_dist<iro::dist::mask<ScopeT>> : std::true_type {};
} // namespace iro::schema

// =============================================================================
// §17. Convenience type aliases and helpers
// =============================================================================

namespace iro {

// Re-export commonly used types at namespace root
using util::type_list;
using util::u64;

// Tile construction helpers
template<int... Dims>
using Shape = contract::Shape<Dims...>;

template<class Shape, class Elem, class Layout, class Space, class Align>
using Tile = contract::Tile<Shape, Elem, Layout, Space, Align>;

template<int Rank, class Elem, class Space, class Align>
using TensorRef = contract::TensorRefDesc<Rank, Elem, Space, Align>;

template<class Shape, class Elem, class Dist>
using Fragment = contract::FragmentDesc<Shape, Elem, Dist>;

// Space aliases (within iro namespace, refers to iro::contract::space)
namespace space = contract::space;

// Alignment aliases
template<int N>
using Align = contract::Align<N>;

// Layout aliases (within iro namespace, refers to iro::contract::layout)
namespace layout = contract::layout;

// Resource aliases (within iro namespace, refers to iro::contract::res)
namespace res = contract::res;

// NOTE: elem, token, scope, exec, cap, dist are already direct children of iro::
// so no namespace aliases needed - they're accessible as iro::elem, iro::token, etc.

} // namespace iro

// =============================================================================
// §18. Static assertions to verify spec compliance
// =============================================================================

namespace iro::verify::spec {

// Verify type_list operations
static_assert(util::size_v<util::type_list<int, float, double>> == 3);
static_assert(std::is_same_v<util::at_t<util::type_list<int, float, double>, 1>, float>);
static_assert(util::contains_v<util::type_list<int, float>, int>);
static_assert(!util::contains_v<util::type_list<int, float>, double>);

// Verify Shape
static_assert(contract::Shape<4, 8, 16>::rank == 3);
static_assert(contract::Shape<4, 8, 16>::size == 4 * 8 * 16);
static_assert(contract::Shape<>::size == 1);

// Verify scope ordering
static_assert(scope::lane::level < scope::warp::level);
static_assert(scope::warp::level < scope::warpgroup::level);
static_assert(scope::warpgroup::level < scope::block::level);
static_assert(scope::block::level < scope::cluster::level);
static_assert(scope::cluster::level < scope::device::level);

// Verify scope subsumption
static_assert(scope_subsumes(scope::block::level, scope::warp::level));
static_assert(!scope_subsumes(scope::warp::level, scope::block::level));

// Verify element tags satisfy concept
static_assert(schema::ElemTag<elem::f32>);
static_assert(schema::ElemTag<elem::f16>);
static_assert(schema::ElemTag<elem::bf16>);

// Verify layout concept
static_assert(schema::Layout<contract::layout::Contiguous>);
static_assert(schema::Layout<contract::layout::RowMajor<16>>);

// Verify token concept
static_assert(schema::Token<token::visible_at<contract::subject::global, scope::warp>>);
static_assert(schema::Token<token::alive<contract::subject::global, token::lifetime::block>>);
static_assert(schema::Token<token::sync_at<contract::subject::global, scope::block>>);
static_assert(schema::Token<token::mask_at<contract::subject::global, scope::warp>>);
static_assert(schema::Token<token::memory_order<contract::subject::global, memory_order::relaxed, scope::warp>>);
static_assert(schema::Token<token::determinism<contract::subject::global, determinism::fast>>);

// Verify fnv1a produces stable, non-zero IDs
static_assert(util::fnv1a_64_cstr("test") != 0);
static_assert(util::fnv1a_64_cstr("test") == util::fnv1a_64_cstr("test"));
static_assert(util::fnv1a_64_cstr("test") != util::fnv1a_64_cstr("test2"));

// Verify token list canonicality
namespace token_canonical_test {
    struct TestSubject { static constexpr auto id = util::fnv1a_64_cstr("test.subject"); };
    using Tok1 = token::visible_at<TestSubject, scope::warp>;
    using Tok2 = token::alive<TestSubject, token::lifetime::block>;
    using Tok3 = token::visible_at<TestSubject, scope::block>;  // Same kind+subject as Tok1

    // No duplicates
    static_assert(token_list_canonical<util::type_list<Tok1, Tok2>>());
    // Duplicate kind+subject (both visible_at<TestSubject>)
    static_assert(!token_list_canonical<util::type_list<Tok1, Tok3>>());
}

// Verify resource list canonicality
namespace resource_canonical_test {
    struct Tag1 { static constexpr auto id = util::fnv1a_64_cstr("test.tag1"); };
    struct Tag2 { static constexpr auto id = util::fnv1a_64_cstr("test.tag2"); };
    using Res1 = contract::res::smem_region<Tag1, 1024, 16>;
    using Res2 = contract::res::smem_region<Tag2, 1024, 16>;
    using Res1Dup = contract::res::smem_region<Tag1, 1024, 16>;  // Same as Res1

    // No duplicates
    static_assert(resource_list_canonical<util::type_list<Res1, Res2>>());
    // Duplicate id
    static_assert(!resource_list_canonical<util::type_list<Res1, Res1Dup>>());
}

// Verify resolution selection
namespace resolve_test {
    struct TestSubject { static constexpr auto id = util::fnv1a_64_cstr("bind.test.subject"); };

    // Simple obligation
    using TestObligation = contract::Obligation<
        util::type_list<>,  // No inputs
        util::type_list<>,  // No outputs
        util::type_list<>   // No resources
    >;

    // Realization for the obligation
    using TestRealization = contract::Realization<TestObligation, util::fnv1a_64_cstr("test.impl")>;
    using TestEntry = registry::RealizationEntry<TestRealization, cap::sm90>;

    // Registry with one realization entry
    using TestRegistry = util::type_list<TestEntry>;

    // Resolution should find the realization
    static_assert(bind::match_realization<TestObligation, cap::sm90, TestRegistry>::found);
    static_assert(!bind::match_realization<TestObligation, cap::sm90, TestRegistry>::ambiguous);
    static_assert(std::is_same_v<
        bind::match_realization_t<TestObligation, cap::sm90, TestRegistry>,
        TestRealization
    >);
}

} // namespace iro::verify::spec
