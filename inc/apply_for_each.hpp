#ifndef APPLY_FOR_EACH
#define APPLY_FOR_EACH

#include <tuple>
#include <concepts>

template<std::size_t I>
auto extract(auto arg0)
{
    return std::make_tuple(std::get<I>(arg0));
}
template<std::size_t I>
auto extract(auto arg0, auto... args)
{
    return std::tuple_cat(extract<I>(arg0), extract<I>(args...));
}

template<std::size_t N, typename T>
auto apply_for_each_(auto& callable, T arg0, auto... args)
{
    auto row = extract<N>(arg0, args...);
    auto res = std::make_tuple(std::apply(callable, row));
    if constexpr (N < std::tuple_size_v<T> - 1)
    {
        return std::tuple_cat(res, apply_for_each_<N + 1>(callable, arg0, args...));
    } else
    {
        return res;
    }
}

template<typename C, typename... T>
auto apply_for_each(C& callable, T... args)
{
    return apply_for_each_<0>(callable, args...);
}
#endif