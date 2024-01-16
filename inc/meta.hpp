#ifndef META_HPP
#define META_HPP

#include "types.hpp"

namespace pcx {

namespace detail_ {
template<auto... Vs>
struct equal_impl {
    static constexpr bool value = false;
};
template<auto V, auto... Vs>
    requires(std::equality_comparable_with<decltype(V), decltype(Vs)> && ...)
struct equal_impl<V, Vs...> {
    static constexpr bool value = ((V == Vs) && ...) && equal_impl<Vs...>::value;
};
template<auto V>
struct equal_impl<V> {
    static constexpr bool value = true;
};
}    // namespace detail_

/**
 * @brief Checks if all template parameter values are equality comparible and equal.
 * 
 * @tparam Vs... Values to check the equality.
 */
template<auto... Vs>
concept equal_values = detail_::equal_impl<Vs...>::value;

namespace detail_ {
template<auto... Vs>
struct unique_values_impl {
    static constexpr bool value = true;
};
template<auto V, auto... Vs>
struct unique_values_impl<V, Vs...> {
    static constexpr bool value = (!equal_values<V, Vs> && ...) && unique_values_impl<Vs...>::value;
};
}    // namespace detail_

/**
 * @brief Checks if template parameters Vs... do not contain repeating values.
 * 
 * @tparam Vs...
 */
template<auto... Vs>
concept unique_values = detail_::unique_values_impl<Vs...>::value;

/**
 * @brief Checks if a value V matches any of the values in Vs...
 * 
 * @tparam V 
 * @tparam Vs...
 */
template<auto V, auto... Vs>
concept value_matched = (!unique_values<V, Vs...>);

namespace detail_ {
template<auto... Values>
struct value_sequence {};

template<typename Sequence, auto... NewValues>
struct expand_value_sequence_impl;
template<auto... Values, auto... NewValues>
struct expand_value_sequence_impl<value_sequence<Values...>, NewValues...> {
    using type = value_sequence<Values..., NewValues...>;
};

template<typename Sequence, auto... NewValues>
using expand_value_sequence = typename expand_value_sequence_impl<Sequence, NewValues...>::type;

template<typename S, auto Vfilter, auto... Vs>
struct filter_value_sequence_impl {};

template<typename S, auto Vfilter, auto V, auto... Vs>
struct filter_value_sequence_impl<S, Vfilter, V, Vs...> {
    using type = std::conditional_t<
        equal_values<Vfilter, V>,
        expand_value_sequence<S, Vs...>,
        typename filter_value_sequence_impl<expand_value_sequence<S, V>, Vfilter, Vs...>::type>;
};
template<typename S, auto Vfilter>
struct filter_value_sequence_impl<S, Vfilter> {
    using type = S;
};
template<typename S, auto Vfilter>
struct filter_value_sequence_adapter {};
template<auto... Vs, auto Vfilter>
struct filter_value_sequence_adapter<value_sequence<Vs...>, Vfilter> {
    using type = typename filter_value_sequence_impl<value_sequence<>, Vfilter, Vs...>::type;
};

template<typename Sequence, auto FilterValue>
using filter_value_sequence = typename filter_value_sequence_adapter<Sequence, FilterValue>::type;

template<typename S>
struct value_to_index_sequence_impl {};
template<uZ... Is>
struct value_to_index_sequence_impl<detail_::value_sequence<Is...>> {
    using type = std::index_sequence<Is...>;
};
template<typename S>
struct index_to_value_sequence_impl {};
template<uZ... Is>
struct index_to_value_sequence_impl<std::index_sequence<Is...>> {
    using type = detail_::value_sequence<Is...>;
};
template<typename S>
using value_to_index_sequence = typename value_to_index_sequence_impl<S>::type;
template<typename S>
using index_to_value_sequence = typename index_to_value_sequence_impl<S>::type;

// template<uZ I, auto Vmatch, auto V, auto... Vs>
// struct idx_impl {
//     static constexpr uZ value = equal_values<Vmatch, V> ? I : idx_impl<I + 1, Vmatch, Vs...>::value;
// };
// template<uZ I, auto Vmatch, auto V>
// struct idx_impl<I, Vmatch, V> {
//     static constexpr uZ value = I;
// };
// template<typename S, uZ Index>
// get_value_from_sequence =  idx_impl<Index, auto Vmatch, auto V, auto Vs>

template<uZ I, uZ Imatch, auto V, auto... Vs>
struct index_value_sequence_impl {
    static constexpr auto value = index_value_sequence_impl<I + 1, Imatch, Vs...>::value;
};
template<uZ Imatch, auto V, auto... Vs>
struct index_value_sequence_impl<Imatch, Imatch, V, Vs...> {
    static constexpr auto value = V;
};

template<typename Sequence, uZ Index>
struct index_value_sequence;

template<uZ Index, auto... Vs>
struct index_value_sequence<value_sequence<Vs...>, Index> {
    static constexpr auto value = index_value_sequence_impl<0, Index, Vs...>::value;
};

template<typename Sequence, uZ Index>
static constexpr auto index_value_sequence_v = index_value_sequence<Sequence, Index>::value;
}    // namespace detail_

namespace meta {
template<auto... Values>
struct value_sequence {};

namespace detail_ {
template<uZ I, auto Vmatch, auto V, auto... Vs>
struct find_first_impl {
    static constexpr uZ index = equal_values<Vmatch, V> ? I : find_first_impl<I + 1, Vmatch, Vs...>::index;
};
template<uZ I, auto Vmatch, auto V>
struct find_first_impl<I, Vmatch, V> {
    static_assert(equal_values<Vmatch, V>);
    static constexpr uZ index = I;
};

template<uZ I, auto Imatch, auto V, auto... Vs>
struct index_into_impl {
    static constexpr auto value = index_into_impl<I + 1, Imatch, Vs...>::value;
};
template<auto Imatch, auto V, auto... Vs>
struct index_into_impl<Imatch, Imatch, V, Vs...> {
    static constexpr auto value = V;
};
}    // namespace detail_

template<auto V, auto... Vs>
    requires value_matched<V, Vs...>
static constexpr uZ find_first_in_values = detail_::find_first_impl<0, V, Vs...>::index;

template<uZ I, auto... Vs>
    requires /**/ (I <= sizeof...(Vs))
static constexpr auto index_into_values = detail_::index_into_impl<0, I, Vs...>::value;

template<uZ I, typename Sequence>
static constexpr auto index_into_sequence = 0;

}    // namespace meta

}    // namespace pcx


#endif