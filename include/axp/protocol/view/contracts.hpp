#pragma once

#include <iro_cuda_ax_core.hpp>
#include <type_traits>

namespace axp::protocol::view {

namespace detail {

template<class T>
struct is_row_major : std::false_type {};
template<int Cols>
struct is_row_major<iro::contract::layout::RowMajor<Cols>> : std::true_type {};

template<class T>
struct is_col_major : std::false_type {};
template<int Rows>
struct is_col_major<iro::contract::layout::ColMajor<Rows>> : std::true_type {};

template<class T>
struct row_major_cols;
template<int Cols>
struct row_major_cols<iro::contract::layout::RowMajor<Cols>> {
    static constexpr int value = Cols;
};

template<class T>
struct col_major_rows;
template<int Rows>
struct col_major_rows<iro::contract::layout::ColMajor<Rows>> {
    static constexpr int value = Rows;
};

template<class T>
struct is_contiguous_layout : std::false_type {};
template<int Cols>
struct is_contiguous_layout<iro::contract::layout::RowMajor<Cols>> : std::true_type {};
template<int Rows>
struct is_contiguous_layout<iro::contract::layout::ColMajor<Rows>> : std::true_type {};
template<>
struct is_contiguous_layout<iro::contract::layout::Contiguous> : std::true_type {};

template<class T>
inline constexpr bool is_row_major_v = is_row_major<T>::value;
template<class T>
inline constexpr bool is_col_major_v = is_col_major<T>::value;
template<class T>
inline constexpr bool is_contiguous_layout_v = is_contiguous_layout<T>::value;

template<class InLayout, class OutLayout, class InShape, class OutShape>
consteval bool transpose_layout_ok() {
    if constexpr (detail::is_row_major_v<InLayout> && detail::is_col_major_v<OutLayout>) {
        return detail::row_major_cols<InLayout>::value == InShape::template dim<1>() &&
               detail::col_major_rows<OutLayout>::value == OutShape::template dim<0>();
    } else if constexpr (detail::is_col_major_v<InLayout> && detail::is_row_major_v<OutLayout>) {
        return detail::col_major_rows<InLayout>::value == InShape::template dim<0>() &&
               detail::row_major_cols<OutLayout>::value == OutShape::template dim<1>();
    } else {
        return false;
    }
}

} // namespace detail

// Base TileView: zero-copy view with layout equality only.
template<class Recipe, class InTile, class OutTile, class SubjectT, class ExecGroupT, class RequiredTokens, class ProvidedTokens>
struct TileView {
    static_assert(std::is_same_v<typename InTile::layout, typename OutTile::layout>,
                  "TileView requires identical layouts (use explicit Transpose/Swizzle/Reshape views)");
    using obligation = typename iro::contract::adapter::TileView<
        InTile,
        OutTile,
        SubjectT,
        ExecGroupT,
        Recipe,
        RequiredTokens,
        ProvidedTokens
    >::obligation;
};

// Transpose view (RowMajor <-> ColMajor, rank-2 only).
template<class Recipe, class InTile, class OutTile, class SubjectT, class ExecGroupT, class RequiredTokens, class ProvidedTokens>
struct TransposeView {
    static_assert(InTile::shape::rank == 2 && OutTile::shape::rank == 2, "TransposeView requires rank-2 tiles");
    static_assert(InTile::shape::template dim<0>() == OutTile::shape::template dim<1>() &&
                  InTile::shape::template dim<1>() == OutTile::shape::template dim<0>(),
                  "TransposeView requires swapped dimensions");
    static_assert(detail::transpose_layout_ok<
                      typename InTile::layout,
                      typename OutTile::layout,
                      typename InTile::shape,
                      typename OutTile::shape>(),
                  "TransposeView requires RowMajor<->ColMajor layouts with matching extents");

    using obligation = typename iro::contract::adapter::TileView<
        InTile,
        OutTile,
        SubjectT,
        ExecGroupT,
        Recipe,
        RequiredTokens,
        ProvidedTokens
    >::obligation;
};

// Swizzle view (shared-memory only, RowMajor -> Swizzled).
template<class Recipe, class InTile, class OutTile, class SubjectT, class ExecGroupT, class RequiredTokens, class ProvidedTokens, class SwizzleAtom>
struct SwizzleView {
    static_assert(std::is_same_v<typename InTile::space, iro::contract::space::shared>,
                  "SwizzleView requires shared memory tiles");
    static_assert(detail::is_row_major_v<typename InTile::layout>,
                  "SwizzleView requires RowMajor input layout");
    static_assert(std::is_same_v<typename OutTile::layout,
                  iro::contract::layout::Swizzled<
                      InTile::shape::template dim<1>(),
                      SwizzleAtom::B,
                      SwizzleAtom::S>>,
                  "SwizzleView requires Swizzled<RowMajorCols, B, S> output layout");
    using obligation = typename iro::contract::adapter::TileView<
        InTile,
        OutTile,
        SubjectT,
        ExecGroupT,
        Recipe,
        RequiredTokens,
        ProvidedTokens
    >::obligation;
};

// Reshape view (size-preserving, contiguous layouts only).
template<class Recipe, class InTile, class OutTile, class SubjectT, class ExecGroupT, class RequiredTokens, class ProvidedTokens>
struct ReshapeView {
    static_assert(InTile::shape::size == OutTile::shape::size, "ReshapeView requires equal total size");
    static_assert(detail::is_contiguous_layout_v<typename InTile::layout>,
                  "ReshapeView requires contiguous input layout");
    static_assert(detail::is_contiguous_layout_v<typename OutTile::layout>,
                  "ReshapeView requires contiguous output layout");
    using obligation = typename iro::contract::adapter::TileView<
        InTile,
        OutTile,
        SubjectT,
        ExecGroupT,
        Recipe,
        RequiredTokens,
        ProvidedTokens
    >::obligation;
};

} // namespace axp::protocol::view
