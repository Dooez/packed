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
}    // namespace pcx::detail_

#endif