#ifndef TUPLE_UTIL_HPP
#define TUPLE_UTIL_HPP

#include "types.hpp"

#include <algorithm>
#include <tuple>

namespace pcx::detail_ {

template<uZ I, typename... Tups>
constexpr auto zip_tuple_element(Tups&&... tuples) {
    return std::tuple<std::tuple_element_t<I, std::remove_reference_t<Tups>>...>{
        std::get<I>(std::forward<Tups>(tuples))...};
}

template<uZ... I, typename... Tups>
constexpr auto zip_tuples_impl(std::index_sequence<I...>, Tups&&... tuples) {
    return std::make_tuple(zip_tuple_element<I>(std::forward<Tups>(tuples)...)...);
}

template<typename... Tups>
    requires /**/ (sizeof...(Tups) > 0)
constexpr auto zip_tuples(Tups&&... tuples) {
    constexpr auto min_size = std::min({std::tuple_size_v<std::remove_reference_t<Tups>>...});
    return zip_tuples_impl(std::make_index_sequence<min_size>{}, std::forward<Tups>(tuples)...);
}

template<typename F, typename Tuple, typename I>
struct apply_result_;

template<typename F, typename Tuple, uZ... I>
struct apply_result_<F, Tuple, std::index_sequence<I...>> {
    using type = typename std::invoke_result_t<std::remove_reference_t<F>,
                                               std::tuple_element_t<I, std::remove_reference_t<Tuple>>...>;
};

template<typename F, typename Tuple>
struct apply_result {
    using type = typename apply_result_<
        F,
        Tuple,
        std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>>::type;
};

template<typename F, typename Tuple>
using apply_result_t = typename apply_result<F, std::remove_reference_t<Tuple>>::type;

template<typename F, typename TupleTuple, typename I>
struct has_result_;

template<typename F, typename TupleTuple, uZ... I>
struct has_result_<F, TupleTuple, std::index_sequence<I...>> {
    static constexpr bool value =
        !(std::same_as<apply_result_t<F, std::tuple_element_t<I, TupleTuple>>, void> || ...);
};

template<typename F, typename... Tups>
concept has_result = has_result_<
    F,
    decltype(zip_tuples(std::declval<Tups>()...)),
    std::make_index_sequence<std::tuple_size_v<decltype(zip_tuples(std::declval<Tups>()...))>>>::value;

template<uZ... I, typename F, typename... Tups>
constexpr auto apply_for_each_impl(std::index_sequence<I...>, F&& f, Tups&&... args) {
    return std::make_tuple(
        std::apply(std::forward<F>(f), zip_tuple_element<I>(std::forward<Tups>(args)...))...);
}

template<uZ... I, typename F, typename... Tups>
constexpr void void_apply_for_each_impl(std::index_sequence<I...>, F&& f, Tups&&... args) {
    (std::apply(std::forward<F>(f), zip_tuple_element<I>(std::forward<Tups>(args)...)), ...);
}

template<typename F, typename... Tups>
    requires has_result<F, Tups...>
constexpr auto apply_for_each(F&& f, Tups&&... args) {
    constexpr auto min_size = std::min({std::tuple_size_v<std::remove_reference_t<Tups>>...});
    return apply_for_each_impl(
        std::make_index_sequence<min_size>{}, std::forward<F>(f), std::forward<Tups>(args)...);
}

template<typename F, typename... Tups>
constexpr void apply_for_each(F&& f, Tups&&... args) {
    constexpr auto min_size = std::min({std::tuple_size_v<std::remove_reference_t<Tups>>...});
    void_apply_for_each_impl(
        std::make_index_sequence<min_size>{}, std::forward<F>(f), std::forward<Tups>(args)...);
}

template<typename T>
struct is_std_tuple : std::false_type {};
template<typename... T>
struct is_std_tuple<std::tuple<T...>> : std::true_type {};
template<typename T>
concept std_tuple = is_std_tuple<T>::value;

template<typename... Ts>
struct any_types {
    static constexpr auto size = sizeof...(Ts);
};
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
template<typename... Ts>
using expand_types_t = typename expand_types<Ts...>::type;


template<typename Tups, typename... Ts>
struct select_std_tuples_impl;

template<typename Tups, typename T, typename... Ts>
struct select_std_tuples_impl<Tups, T, Ts...> {
    using type = std::conditional_t<
        std_tuple<std::remove_cvref_t<T>>,
        typename select_std_tuples_impl<expand_types_t<Tups, std::remove_cvref_t<T>>, Ts...>::type,
        typename select_std_tuples_impl<Tups, Ts...>::type    //
        >;
};
template<typename Tups>
struct select_std_tuples_impl<Tups> {
    using type = Tups;
};

template<typename... Ts>
using select_std_tuples_t = typename select_std_tuples_impl<any_types<>, Ts...>::type;


template<typename T>
struct tuples_of_common_size;
template<typename T, typename... Ts>
struct tuples_of_common_size<any_types<T, Ts...>> {
    static constexpr bool value = ((std::tuple_size_v<T> == std::tuple_size_v<Ts>) && ...);
    static constexpr auto size  = std::tuple_size_v<T>;
};
template<typename T>
struct tuples_of_common_size<any_types<T>> {
    static constexpr bool value = true;
    static constexpr auto size  = std::tuple_size_v<T>;
};
template<>
struct tuples_of_common_size<any_types<>> {
    static constexpr bool value = true;
    static constexpr auto size  = 1;
};

/**
 * @brief Invokes the callable `f` with the provided arguments, potentially
 * multiple times simultaneously iterating over tuple elements 
 * of arguments of `std::tuple<...>` types.
 *
 * 
 * @tparam F 
 * @tparam Tups 
 * @param f     The callable to inoke. 
 * @param args  Arguments passed to the callable. 
 * If `args...` include `std::tuple<...>`, the tuples must be of the same tuple size.
 * The callable is invoked with each set of arguments, iterating over tuple elements.
 *
 * @return A tuple containing results of individual invocations.
 */
template<typename F, typename... Args>
    requires has_result<F, Args...> && (tuples_of_common_size<select_std_tuples_t<Args...>>::value)
constexpr auto mass_invoke(F&& f, Args&&... args) {
    // constexpr auto count = (..., std::tuple_size_v<std::remove_reference_t<Args>>);
    constexpr auto count = tuples_of_common_size<select_std_tuples_t<Args...>>::size;
    static_assert(count != 0);


    constexpr auto iterate = []<uZ... Is>(std::index_sequence<Is...>, auto&& f, Args&&... args) {
        constexpr auto invoke = []<uZ I>(uZ_constant<I>, auto& f, auto&... args) {
            constexpr auto get = []<uZ Iget, typename T>(uZ_constant<Iget>, T&& arg) {
                if constexpr (std_tuple<std::remove_cvref_t<T>>) {
                    return std::get<Iget>(std::forward<T>(arg));
                } else {
                    return arg;
                }
            };
            return f(get(uZ_constant<I>{}, args)...);
        };
        return std::make_tuple(invoke(uZ_constant<Is>{}, f, args...)...);
    };
    return iterate(std::make_index_sequence<count>{}, std::forward<F>(f), std::forward<Args>(args)...);
}    // namespace pcx::detail_

}    // namespace pcx::detail_

#endif
