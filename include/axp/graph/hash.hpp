#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/graph/hash.hpp is library-only; use axp/l4.hpp in application code."
#endif

#include <type_traits>

#include "spec.hpp"

namespace axp::graph {

template<class CompositionT>
struct graph_hash_override {
    static constexpr bool enabled = false;
    static constexpr iro::util::u64 value = 0;
};

namespace detail {

template<class T, class = void>
struct has_direct_id : std::false_type {};

template<class T>
struct has_direct_id<T, std::void_t<decltype(T::id)>> : std::true_type {};

template<class T, class = void>
struct has_obligation_id : std::false_type {};

template<class T>
struct has_obligation_id<T, std::void_t<typename T::obligation, decltype(T::obligation::id)>> : std::true_type {};

template<class T, class = void>
struct has_obligation_shape : std::false_type {};

template<class T>
struct has_obligation_shape<T, std::void_t<typename T::inputs, typename T::outputs, typename T::resources>> : std::true_type {};

template<class T, bool HasId = has_direct_id<T>::value, bool HasObligationId = has_obligation_id<T>::value,
         bool HasShape = has_obligation_shape<T>::value>
struct stable_id_impl;

template<class T, bool HasObligationId, bool HasShape>
struct stable_id_impl<T, true, HasObligationId, HasShape> {
    static constexpr iro::util::u64 value = T::id;
};

template<class T, bool HasShape>
struct stable_id_impl<T, false, true, HasShape> {
    static constexpr iro::util::u64 value = T::obligation::id;
};

template<class T>
struct stable_id_impl<T, false, false, true> {
    static constexpr iro::util::u64 value =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.graph.synthetic_obligation_id.v1"),
                           iro::util::mix_u64(iro::util::hash_list_v<typename T::inputs>,
                                              iro::util::mix_u64(iro::util::hash_list_v<typename T::outputs>,
                                                                 iro::util::hash_list_v<typename T::resources>)));
};

template<class T>
struct stable_id_impl<T, false, false, false> {
    static_assert(has_direct_id<T>::value || has_obligation_id<T>::value || has_obligation_shape<T>::value,
                  "axp::graph::hash: type has no stable id and no obligation shape");
    static constexpr iro::util::u64 value = 0;
};

template<class T>
inline constexpr iro::util::u64 stable_id_v = stable_id_impl<T>::value;

template<class T>
struct id_token {
    static constexpr iro::util::u64 id = stable_id_v<T>;
};

template<class A, class B>
struct id_less : std::bool_constant<(A::id < B::id)> {};

template<class A, class B>
struct edge_less : std::bool_constant<
    (A::from_node_id < B::from_node_id) ||
    ((A::from_node_id == B::from_node_id) && (A::from_port < B::from_port)) ||
    ((A::from_node_id == B::from_node_id) && (A::from_port == B::from_port) && (A::to_node_id < B::to_node_id)) ||
    ((A::from_node_id == B::from_node_id) && (A::from_port == B::from_port) &&
     (A::to_node_id == B::to_node_id) && (A::to_port < B::to_port))> {};

template<class Sorted, class T, template<class, class> class Less>
struct insert_sorted;

template<class T, template<class, class> class Less>
struct insert_sorted<iro::util::type_list<>, T, Less> {
    using type = iro::util::type_list<T>;
};

template<class Head, class... Tail, class T, template<class, class> class Less>
struct insert_sorted<iro::util::type_list<Head, Tail...>, T, Less> {
    using type = std::conditional_t<
        Less<T, Head>::value,
        iro::util::type_list<T, Head, Tail...>,
        iro::util::prepend_t<
            typename insert_sorted<iro::util::type_list<Tail...>, T, Less>::type,
            Head
        >
    >;
};

template<class Acc, class Remaining, template<class, class> class Less>
struct sort_impl;

template<class Acc, template<class, class> class Less>
struct sort_impl<Acc, iro::util::type_list<>, Less> {
    using type = Acc;
};

template<class Acc, class T, class... Ts, template<class, class> class Less>
struct sort_impl<Acc, iro::util::type_list<T, Ts...>, Less> {
    using next = typename insert_sorted<Acc, T, Less>::type;
    using type = typename sort_impl<next, iro::util::type_list<Ts...>, Less>::type;
};

template<class List, template<class, class> class Less>
using sort_t = typename sort_impl<iro::util::type_list<>, List, Less>::type;

template<class ObligationList>
struct node_tokens;

template<class... Os>
struct node_tokens<iro::util::type_list<Os...>> {
    using type = iro::util::type_list<id_token<Os>...>;
};

template<class Edge>
struct edge_token {
    static constexpr iro::util::u64 from_node_id = stable_id_v<typename Edge::out_ref::obligation>;
    static constexpr iro::util::u64 to_node_id = stable_id_v<typename Edge::in_ref::obligation>;
    static constexpr iro::util::u64 from_port =
        static_cast<iro::util::u64>(Edge::out_ref::index);
    static constexpr iro::util::u64 to_port =
        static_cast<iro::util::u64>(Edge::in_ref::index);
    static constexpr iro::util::u64 id =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.graph.edge.v1"),
                           iro::util::mix_u64(from_node_id,
                                              iro::util::mix_u64(from_port,
                                                                 iro::util::mix_u64(to_node_id, to_port))));
};

template<class EdgeList>
struct edge_tokens;

template<class... Es>
struct edge_tokens<iro::util::type_list<Es...>> {
    using type = iro::util::type_list<edge_token<Es>...>;
};

template<class CompositionT>
struct computed_graph_hash {
    using node_token_list = typename node_tokens<typename CompositionT::obligations>::type;
    using sorted_nodes = sort_t<node_token_list, id_less>;
    using edge_token_list = typename edge_tokens<typename CompositionT::edges>::type;
    using sorted_edges = sort_t<edge_token_list, edge_less>;
    using canonical_resources = iro::verify::canonicalize_resource_list<typename CompositionT::resources>;
    using sorted_resources = sort_t<canonical_resources, id_less>;

    static constexpr iro::util::u64 value =
        iro::util::mix_u64(iro::util::fnv1a_64_cstr("axp.graph.hash.v1"),
                           iro::util::mix_u64(iro::util::hash_list_v<sorted_nodes>,
                                              iro::util::mix_u64(iro::util::hash_list_v<sorted_edges>,
                                                                 iro::util::hash_list_v<sorted_resources>)));
};

} // namespace detail

template<class G>
struct graph_hash {
    static_assert(CompositionLike<G>,
                  "axp::graph::graph_hash: graph must provide an iro::compose::Composition shape");
    using composition = composition_of_t<G>;

    template<class C, bool HasOverride = graph_hash_override<C>::enabled>
    struct select_impl;

    template<class C>
    struct select_impl<C, true> {
        static constexpr iro::util::u64 value = graph_hash_override<C>::value;
    };

    template<class C>
    struct select_impl<C, false> {
        static constexpr iro::util::u64 value = detail::computed_graph_hash<C>::value;
    };

    static constexpr iro::util::u64 value = select_impl<composition>::value;
};

template<class G>
inline constexpr iro::util::u64 graph_hash_v = graph_hash<G>::value;

} // namespace axp::graph
