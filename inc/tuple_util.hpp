#ifndef TUPLE_UTIL_HPP
#define TUPLE_UTIL_HPP

#include "meta.hpp"
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
concept has_result_old = has_result_<
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
    requires has_result_old<F, Tups...>
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

template<typename Tups, typename... Ts>
struct select_std_tuples_impl;

template<typename Tups, typename T, typename... Ts>
struct select_std_tuples_impl<Tups, T, Ts...> {
    using type = std::conditional_t<
        std_tuple<std::remove_cvref_t<T>>,
        typename select_std_tuples_impl<meta::expand_types_t<Tups, std::remove_cvref_t<T>>, Ts...>::type,
        typename select_std_tuples_impl<Tups, Ts...>::type    //
        >;
};
template<typename Tups>
struct select_std_tuples_impl<Tups> {
    using type = Tups;
};

template<typename... Ts>
using select_std_tuples_t = typename select_std_tuples_impl<meta::any_types<>, Ts...>::type;


template<typename T>
struct tuples_of_common_size;
template<typename T, typename... Ts>
struct tuples_of_common_size<meta::any_types<T, Ts...>> {
    static constexpr bool value = ((std::tuple_size_v<T> == std::tuple_size_v<Ts>) && ...);
    static constexpr auto size  = std::tuple_size_v<T>;
};
template<typename T>
struct tuples_of_common_size<meta::any_types<T>> {
    static constexpr bool value = true;
    static constexpr auto size  = std::tuple_size_v<T>;
};
template<>
struct tuples_of_common_size<meta::any_types<>> {
    static constexpr bool value = true;
    static constexpr auto size  = 1;
};
template<typename T>
concept same_size = tuples_of_common_size<T>::value;

template<typename T>
struct nonzero_tuple_size_if_any;
template<std_tuple... Ts>
struct nonzero_tuple_size_if_any<meta::any_types<Ts...>> {
    static constexpr bool value = ((std::tuple_size_v<Ts> > 0) && ...);
};
template<>
struct nonzero_tuple_size_if_any<meta::any_types<>> {
    static constexpr bool value = true;
};
template<typename T>
concept nonzero_size = nonzero_tuple_size_if_any<T>::value;


template<uZ I, typename T>
struct get_type {
    using type = std::conditional_t<std_tuple<T>, std::tuple_element_t<I, T>, T>;
};

template<typename F, typename... Args>
concept mass_invocable =
    same_size<select_std_tuples_t<Args...>>          //
    && nonzero_size<select_std_tuples_t<Args...>>    //
    && ([]<uZ... Is>(std::index_sequence<Is...>) {
           constexpr auto invocable_i = []<uZ I>(uZ_constant<I>) {
               return std::invocable<F, typename get_type<I, Args>::type...>;
           };
           return (invocable_i(uZ_constant<Is>{}) && ...);
       }(std::make_index_sequence<tuples_of_common_size<select_std_tuples_t<Args...>>::size>{}));

template<typename F, typename... Args>
concept has_result =
    mass_invocable<F, Args...>    //
    && ([]<uZ... Is>(std::index_sequence<Is...>) {
           constexpr auto nonvoid_result = []<uZ I>(uZ_constant<I>) {
               return !std::same_as<std::invoke_result_t<F, typename get_type<I, Args>::type...>, void>;
           };
           return (nonvoid_result(uZ_constant<Is>{}) && ...);
       }(std::make_index_sequence<tuples_of_common_size<select_std_tuples_t<Args...>>::size>{}));

/**
 * @brief Invokes the callable `f` with the provided arguments, potentially
 * multiple times simultaneously iterating over tuple elements of arguments 
 * of `std::tuple<...>` types.
 *
 * 
 * @tparam F 
 * @tparam Tups 
 * @param f     The callable to inoke. 
 * @param args  Arguments passed to the callable. 
 * If `args...` include `std::tuple<...>`, all the tuples must be of the same tuple size.
 * The callable is invoked with each set of arguments, iterating over tuple elements.
 *
 * @return A tuple containing results of individual invocations.
 */
template<typename F, typename... Args>
    requires same_size<select_std_tuples_t<Args...>>          //
             && nonzero_size<select_std_tuples_t<Args...>>    //
             && mass_invocable<F, Args...>                    //
             && has_result<F, Args...>
constexpr auto mass_invoke(F&& f, Args&&... args) {
    constexpr auto count   = tuples_of_common_size<select_std_tuples_t<Args...>>::size;
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
}

}    // namespace pcx::detail_

#endif
