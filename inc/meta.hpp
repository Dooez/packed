#ifndef META_HPP
#define META_HPP

#include "types.hpp"

#include <type_traits>

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
struct value_sequence {
    static constexpr uZ size = sizeof...(Values);
};

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
/**
    template<auto V>
    struct value_constant{
        static constexpr auto value = V;
    };
  
    template<auto... Vs>
    concept equal_values;

    template<auto... Vs>
    concept unique_values;

    template<auto V, auto... Vs>
    concept value_matched;

    template<auto... Values>
    struct value_sequence{
        static constexpr uZ size;
    };

    template<typename T>
    concept any_value_sequence;

    template<typename Sequence, auto... NewValues>
    using expand_value_sequence;

    template<typename Sequence1, typename Sequence2>
    using concat_value_sequences;

    template<typename Sequence>
    using reverse_value_sequence;

    template<typename Sequence, auto V>
    concept contains_value;

    template<auto V, auto... Vs>
        requires value_matched<V, Vs...>
    static constexpr uZ find_first_in_values;

    template<uZ I, auto... Vs>
        requires (I < sizeof...(Vs))
    static constexpr auto index_into_values;

    template<uZ I, any_value_sequene Sequence>
        requires (I < Sequence::size)
    static constexpr auto index_into_sequence;

    template<auto V, any_value_sequence Sequence>
        requires contains_value<Sequence, V>
    static constexpr uZ find_first_in_sequence;

    template<any_value_sequence Sequence, auto V>
    using filter_value_sequence;
 * 
 * 
 */

template<auto V>
struct value_constant {
    static constexpr auto value = V;
};

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
template<auto... Values>
struct value_sequence {
    static constexpr uZ size = sizeof...(Values);
};

namespace detail_ {
template<typename T>
struct is_value_sequence : std::false_type {};
template<auto... Values>
struct is_value_sequence<value_sequence<Values...>> : std::true_type {};

template<typename T>
struct is_value_sequence_of_unique;
template<auto... Values>
struct is_value_sequence_of_unique<value_sequence<Values...>> {
    static constexpr bool value = unique_values<Values...>;
};

template<typename Sequence, auto... NewValues>
struct expand_value_sequence_impl;
template<auto... Values, auto... NewValues>
struct expand_value_sequence_impl<value_sequence<Values...>, NewValues...> {
    using type = value_sequence<Values..., NewValues...>;
};
template<typename, typename>
struct concat_value_sequences_impl;
template<auto... Values1, auto... Values2>
struct concat_value_sequences_impl<value_sequence<Values1...>, value_sequence<Values2...>> {
    using type = value_sequence<Values1..., Values2...>;
};
template<typename Sequence>
struct reverse_value_sequence_impl {
    using type = Sequence;
};
template<auto V, auto... Vs>
struct reverse_value_sequence_impl<value_sequence<V, Vs...>> {
    using type =
        typename expand_value_sequence_impl<typename reverse_value_sequence_impl<value_sequence<Vs...>>::type,
                                            V>::type;
};

}    // namespace detail_
template<typename T>
concept any_value_sequence = detail_::is_value_sequence<T>::value;

template<typename T>
concept value_sequence_of_unique = detail_::is_value_sequence_of_unique<T>::value;

template<typename Sequence, auto... NewValues>
using expand_value_sequence_t = typename detail_::expand_value_sequence_impl<Sequence, NewValues...>::type;

template<typename Sequence1, typename Sequence2>
using concat_value_sequences_t = typename detail_::concat_value_sequences_impl<Sequence1, Sequence2>::type;

template<typename Sequence>
using reverse_value_sequence_t = typename detail_::reverse_value_sequence_impl<Sequence>::type;

namespace detail_ {
template<uZ I, auto Vmatch, auto V, auto... Vs>
struct find_first_impl {
    static constexpr uZ value = std::
        conditional_t<equal_values<Vmatch, V>, uZ_constant<I>, find_first_impl<I + 1, Vmatch, Vs...>>::value;
};
template<uZ I, auto Vmatch, auto V>
struct find_first_impl<I, Vmatch, V> {
    static_assert(equal_values<Vmatch, V>);
    static constexpr uZ value = I;
};

template<uZ I, auto Imatch, auto V, auto... Vs>
struct index_into_impl {
    static constexpr auto value = index_into_impl<I + 1, Imatch, Vs...>::value;
};
template<auto Imatch, auto V, auto... Vs>
struct index_into_impl<Imatch, Imatch, V, Vs...> {
    static constexpr auto value = V;
};

template<typename S, auto Vfilter, auto... Vs>
struct filter_impl {
    static_assert(sizeof...(Vs) == 0);
    using type = S;
};
template<typename S, auto Vfilter, auto V, auto... Vs>
struct filter_impl<S, Vfilter, V, Vs...> {
    using type =
        std::conditional_t<equal_values<Vfilter, V>,
                           concat_value_sequences_t<S, filter_impl<value_sequence<>, Vfilter, Vs...>>,
                           typename filter_impl<expand_value_sequence_t<S, V>, Vfilter, Vs...>::type>;
};
// template<typename S, auto Vfilter>
// struct filter_impl<S, Vfilter> {
//     using type = S;
// };
template<typename S>
struct sequence_adapter;

template<auto... Vs>
struct sequence_adapter<value_sequence<Vs...>> {
    template<auto V>
    struct matched {
        static constexpr bool value = value_matched<V, Vs...>;
    };
    template<uZ I>
    struct index_into {
        static constexpr auto value = index_into_impl<0, I, Vs...>::value;
    };
    template<auto V>
    struct find_first {
        static constexpr uZ index = find_first_impl<0, V, Vs...>::value;
    };
    template<auto V>
    struct filter {
        using type = filter_impl<value_sequence<>, V, Vs...>;
    };
};
}    // namespace detail_

template<typename Sequence, auto V>
concept contains_value = any_value_sequence<Sequence> &&    //
                         detail_::sequence_adapter<Sequence>::template matched<V>::value;

template<auto V, auto... Vs>
    requires value_matched<V, Vs...>
static constexpr uZ find_first_in_values = detail_::find_first_impl<0, V, Vs...>::value;

template<uZ I, auto... Vs>
    requires /**/ (I < sizeof...(Vs))
static constexpr auto index_into_values = detail_::index_into_impl<0, I, Vs...>::value;

template<uZ I, any_value_sequence Sequence>
    requires /**/ (I < Sequence::size)
static constexpr auto index_into_sequence =
    detail_::sequence_adapter<Sequence>::template index_into<I>::value;

template<auto V, any_value_sequence Sequence>
    requires contains_value<Sequence, V>
static constexpr uZ find_first_in_sequence =
    detail_::sequence_adapter<Sequence>::template find_first<V>::value;

template<any_value_sequence Sequence, auto V>
using filter_value_sequence = typename detail_::sequence_adapter<Sequence>::template filter<V>::type;

namespace detail_ {
template<typename S>
struct value_to_index_sequence_impl;
template<uZ... Is>
struct value_to_index_sequence_impl<value_sequence<Is...>> {
    using type = std::index_sequence<Is...>;
};
template<typename S>
struct index_to_value_sequence_impl;
template<uZ... Is>
struct index_to_value_sequence_impl<std::index_sequence<Is...>> {
    using type = value_sequence<Is...>;
};
}    // namespace detail_
template<typename S>
using value_to_index_sequence = typename detail_::value_to_index_sequence_impl<S>::type;
template<typename S>
using index_to_value_sequence = typename detail_::index_to_value_sequence_impl<S>::type;

template<typename... Ts>
struct any_types {
    static constexpr auto size = sizeof...(Ts);
};
namespace detail_ {
template<typename... Ts>
struct expand_types;
template<typename Tl, typename Tr>
struct expand_types<Tl, Tr> {
    using type = any_types<Tl, Tr>;
};
template<typename... Ts, typename Tr>
struct expand_types<any_types<Ts...>, Tr> {
    using type = any_types<Ts..., Tr>;
};
template<typename Tl, typename... Ts>
struct expand_types<Tl, any_types<Ts...>> {
    using type = any_types<Tl, Ts...>;
};
template<typename Tl, typename... Ts>
struct expand_types<Tl, Ts...> {
    using type = typename expand_types<Tl, typename expand_types<Ts...>::type>::type;
};
template<>
struct expand_types<> {
    using type = any_types<>;
};
}    // namespace detail_
template<typename... Ts>
using expand_types_t = typename detail_::expand_types<Ts...>::type;
}    // namespace meta

}    // namespace pcx


#endif
