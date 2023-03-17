#ifndef VECTOR_UTIL_HPP
#define VECTOR_UTIL_HPP

#include <array>
#include <complex>
#include <cstring>
#include <immintrin.h>
#include <tuple>

namespace pcx {

template<typename T>
    requires std::same_as<T, float> || std::same_as<T, double>
constexpr const std::size_t default_pack_size = 32 / sizeof(T);

constexpr const std::size_t dynamic_size = -1;

template<std::size_t N>
concept power_of_two = (N & (N - 1)) == 0;

template<typename T, std::size_t PackSize>
concept packed_floating_point = std::floating_point<T> && power_of_two<PackSize> &&
                                (PackSize >= pcx::default_pack_size<T>);

template<typename T, typename Allocator, std::size_t PackSize>
    requires packed_floating_point<T, PackSize>
class vector;

template<typename T, bool Const, std::size_t PackSize>
class cx_ref;

template<typename T, bool Const, std::size_t PackSize>
class iterator;

template<typename T, bool Const, std::size_t Size, std::size_t PackSize>
class subrange;

namespace internal {
    template<std::size_t I, typename... Tups>
    constexpr auto zip_tuple_element(Tups&&... tuples)
    {
        return std::tuple<std::tuple_element_t<I, std::remove_reference_t<Tups>>...>{
            std::get<I>(std::forward<Tups>(tuples))...};
    }

    template<std::size_t... I, typename... Tups>
    constexpr auto zip_tuples_impl(std::index_sequence<I...>, Tups&&... tuples)
    {
        return std::make_tuple(zip_tuple_element<I>(std::forward<Tups>(tuples)...)...);
    }

    template<typename... Tups>
    constexpr auto zip_tuples(Tups&&... tuples)
    {
        static_assert(sizeof...(tuples) > 0);

        constexpr auto min_size =
            std::min({std::tuple_size_v<std::remove_reference_t<Tups>>...});
        return zip_tuples_impl(std::make_index_sequence<min_size>{},
                               std::forward<Tups>(tuples)...);
    }

    template<typename F, typename Tuple>
    concept appliable = requires(F&& f, Tuple&& tuple) {
                            std::apply(std::forward<F>(f), std::forward<Tuple>(tuple));
                        };

    template<typename F, typename TupleTuple, typename I>
    struct appliable_seq;

    template<typename F, typename TupleTuple, std::size_t... I>
    struct appliable_seq<F, TupleTuple, std::index_sequence<I...>>
    {
        static constexpr bool value =
            (appliable<F, std::tuple_element_t<I, TupleTuple>> && ...);
    };

    template<typename F, typename... Tups>
    concept appliable_m =
        appliable_seq<F,
                      decltype(zip_tuples(std::declval<Tups>()...)),
                      std::make_index_sequence<std::tuple_size_v<decltype(zip_tuples(
                          std::declval<Tups>()...))>>>::value;

    template<typename F, typename Tuple, typename I>
    struct apply_result_impl;

    template<typename F, typename Tuple, std::size_t... I>
    struct apply_result_impl<F, Tuple, std::index_sequence<I...>>
    {
        using type = typename std::invoke_result_t<
            std::remove_reference_t<F>,
            std::tuple_element_t<I, std::remove_reference_t<Tuple>>...>;
    };

    template<typename F, typename Tuple>
    struct apply_result
    {
        using type =
            typename apply_result_impl<F,
                                       Tuple,
                                       std::make_index_sequence<std::tuple_size_v<
                                           std::remove_reference_t<Tuple>>>>::type;
    };

    template<typename F, typename Tuple>
    using apply_result_t = typename apply_result<F, std::remove_reference_t<Tuple>>::type;

    template<typename F, typename TupleTuple, typename I>
    struct has_result_seq;

    template<typename F, typename TupleTuple, std::size_t... I>
    struct has_result_seq<F, TupleTuple, std::index_sequence<I...>>
    {
        static constexpr bool value = !(
            std::same_as<apply_result_t<F, std::tuple_element_t<I, TupleTuple>>, void> ||
            ...);
    };

    template<typename F, typename... Tups>
    concept has_result =
        has_result_seq<F,
                       decltype(zip_tuples(std::declval<Tups>()...)),
                       std::make_index_sequence<std::tuple_size_v<decltype(zip_tuples(
                           std::declval<Tups>()...))>>>::value;

    template<std::size_t... I, typename C, typename Tup>
    constexpr auto apply_for_each_impl(std::index_sequence<I...>,
                                       C&&   callable,
                                       Tup&& args)
    {
        return std::make_tuple(std::apply(callable, std::get<I>(args))...);
    }

    template<std::size_t... I, typename C, typename Tup>
    constexpr void void_apply_for_each_impl(std::index_sequence<I...>,
                                            C&&   callable,
                                            Tup&& args)
    {
        (std::apply(callable, std::get<I>(args)), ...);
    }

    template<typename C, typename... Tups>
        requires appliable_m<C, Tups...> && has_result<C, Tups...>
    constexpr auto apply_for_each(C&& callable, Tups&&... args)
    {
        auto args_zip = zip_tuples(args...);

        constexpr auto N = std::tuple_size_v<std::remove_cvref_t<decltype(args_zip)>>;
        return apply_for_each_impl(std::make_index_sequence<N>{},
                                   std::forward<C>(callable),
                                   std::move(args_zip));
    }

    template<typename C, typename... Tups>
    constexpr void apply_for_each(C&& callable, Tups&&... args)
    {
        auto args_zip = zip_tuples(args...);

        constexpr auto N = std::tuple_size_v<std::remove_cvref_t<decltype(args_zip)>>;
        void_apply_for_each_impl(std::make_index_sequence<N>{},
                                 std::forward<C>(callable),
                                 std::move(args_zip));
    }
}    // namespace internal

/**
 * @brief alias for templated avx2 types and functions
 *
 */
namespace avx {
    template<typename T>
    struct reg;
    template<>
    struct reg<float>
    {
        using type = __m256;
    };
    template<>
    struct reg<double>
    {
        using type = __m256d;
    };
    template<typename T>
    struct cx_reg
    {
        typename reg<T>::type real;
        typename reg<T>::type imag;
    };

    inline auto load(const float* source) -> reg<float>::type
    {
        return _mm256_loadu_ps(source);
    }
    inline auto load(const double* source) -> reg<double>::type
    {
        return _mm256_loadu_pd(source);
    }
    template<std::size_t PackSize, typename T>
    inline auto cxload(const T* ptr) -> cx_reg<T>
    {
        return {load(ptr), load(ptr + PackSize)};
    }

    inline auto broadcast(const float* source) -> reg<float>::type
    {
        return _mm256_broadcast_ss(source);
    }
    inline auto broadcast(const double* source) -> reg<double>::type
    {
        return _mm256_broadcast_sd(source);
    }

    template<typename T>
    inline auto broadcast(const std::complex<T>& source) -> cx_reg<T>
    {
        const auto& value = reinterpret_cast<const T(&)[2]>(source);
        return {avx::broadcast(&(value[0])), avx::broadcast(&(value[1]))};
    }
    template<typename T>
    inline auto broadcast(const T& source) -> typename reg<T>::type
    {
        return avx::broadcast(&source);
    }

    inline void store(float* dest, reg<float>::type reg)
    {
        return _mm256_storeu_ps(dest, reg);
    }
    inline void store(double* dest, reg<double>::type reg)
    {
        return _mm256_storeu_pd(dest, reg);
    }
    template<std::size_t PackSize, typename T>
    inline void cxstore(T* ptr, cx_reg<T> reg)
    {
        store(ptr, reg.real);
        store(ptr + PackSize, reg.imag);
    }

    inline auto unpacklo_ps(reg<float>::type a, reg<float>::type b) -> reg<float>::type
    {
        return _mm256_unpacklo_ps(a, b);
    };
    inline auto unpackhi_ps(reg<float>::type a, reg<float>::type b) -> reg<float>::type
    {
        return _mm256_unpackhi_ps(a, b);
    };

    inline auto unpacklo_pd(reg<float>::type a, reg<float>::type b) -> reg<float>::type
    {
        return _mm256_castpd_ps(
            _mm256_unpacklo_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
    };
    inline auto unpackhi_pd(reg<float>::type a, reg<float>::type b) -> reg<float>::type
    {
        return _mm256_castpd_ps(
            _mm256_unpackhi_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
    };
    inline auto unpacklo_pd(reg<double>::type a, reg<double>::type b) -> reg<double>::type
    {
        return _mm256_unpacklo_pd(a, b);
    };
    inline auto unpackhi_pd(reg<double>::type a, reg<double>::type b) -> reg<double>::type
    {
        return _mm256_unpackhi_pd(a, b);
    };

    inline auto unpacklo_128(reg<float>::type a, reg<float>::type b) -> reg<float>::type
    {
        return _mm256_permute2f128_ps(a, b, 0b00100000);
    };
    inline auto unpackhi_128(reg<float>::type a, reg<float>::type b) -> reg<float>::type
    {
        return _mm256_permute2f128_ps(a, b, 0b00110001);
    };
    inline auto unpacklo_128(reg<double>::type a, reg<double>::type b)
        -> reg<double>::type
    {
        return _mm256_permute2f128_pd(a, b, 0b00110001);
    };
    inline auto unpackhi_128(reg<double>::type a, reg<double>::type b)
        -> reg<double>::type
    {
        return _mm256_permute2f128_pd(a, b, 0b00110001);
    };

    inline auto unpack_ps(cx_reg<float> a, cx_reg<float> b)
        -> std::array<cx_reg<float>, 2>
    {
        auto real_lo = unpacklo_ps(a.real, b.real);
        auto real_hi = unpackhi_ps(a.real, b.real);
        auto imag_lo = unpacklo_ps(a.imag, b.imag);
        auto imag_hi = unpackhi_ps(a.imag, b.imag);

        return {cx_reg<float>({real_lo, imag_lo}), cx_reg<float>({real_hi, imag_hi})};
    };

    template<typename T>
    inline auto unpack_pd(cx_reg<T> a, cx_reg<T> b) -> std::array<cx_reg<T>, 2>
    {
        auto real_lo = unpacklo_pd(a.real, b.real);
        auto real_hi = unpackhi_pd(a.real, b.real);
        auto imag_lo = unpacklo_pd(a.imag, b.imag);
        auto imag_hi = unpackhi_pd(a.imag, b.imag);

        return {cx_reg<float>({real_lo, imag_lo}), cx_reg<float>({real_hi, imag_hi})};
    };

    template<typename T>
    inline auto unpack_128(cx_reg<T> a, cx_reg<T> b) -> std::array<cx_reg<T>, 2>
    {
        auto real_lo = unpacklo_128(a.real, b.real);
        auto real_hi = unpackhi_128(a.real, b.real);
        auto imag_lo = unpacklo_128(a.imag, b.imag);
        auto imag_hi = unpackhi_128(a.imag, b.imag);

        return {cx_reg<float>({real_lo, imag_lo}), cx_reg<float>({real_hi, imag_hi})};
    };

    template<typename T>
    struct convert
    {
        static inline auto packed_to_interleaved(auto... args);
        static inline auto interleaved_to_packed(auto... args);
    };

    template<>
    struct convert<float>
    {
        static inline auto packed_to_interleaved(auto... args)
        {
            auto tup = std::make_tuple(args...);

            auto unpack_ps = [](cx_reg<float> reg) {
                auto real = unpacklo_ps(reg.real, reg.imag);
                auto imag = unpackhi_ps(reg.real, reg.imag);
                return cx_reg<float>({real, imag});
            };
            auto unpack_128 = [](cx_reg<float> reg) {
                auto real = unpacklo_128(reg.real, reg.imag);
                auto imag = unpackhi_128(reg.real, reg.imag);
                return cx_reg<float>({real, imag});
            };

            auto tmp = internal::apply_for_each(unpack_ps, tup);
            return internal::apply_for_each(unpack_128, tmp);
        }

        static inline auto interleaved_to_packed(auto... args)
        {
            auto tup = std::make_tuple(args...);

            auto pack_128 = [](cx_reg<float> reg) {
                auto real = unpacklo_128(reg.real, reg.imag);
                auto imag = unpackhi_128(reg.real, reg.imag);
                return cx_reg<float>({real, imag});
            };
            auto pack_ps = [](cx_reg<float> reg) {
                auto real = _mm256_shuffle_ps(reg.real, reg.imag, 0b10001000);
                auto imag = _mm256_shuffle_ps(reg.real, reg.imag, 0b11011101);
                return cx_reg<float>({real, imag});
            };

            auto tmp = internal::apply_for_each(pack_128, tup);
            return internal::apply_for_each(pack_ps, tmp);
        }

        static constexpr auto swap_12 = [](cx_reg<float> reg) {
            auto real = _mm256_shuffle_ps(reg.real, reg.real, 0b11011000);
            auto imag = _mm256_shuffle_ps(reg.imag, reg.imag, 0b11011000);
            return cx_reg<float>({real, imag});
        };

        static constexpr auto swap_24 = [](cx_reg<float> reg) {
            auto real = _mm256_permute4x64_pd(_mm256_castps_pd(reg.real), 0b11011000);
            auto imag = _mm256_permute4x64_pd(_mm256_castps_pd(reg.imag), 0b11011000);
            return cx_reg<float>({_mm256_castpd_ps(real), _mm256_castpd_ps(imag)});
        };

        static constexpr auto swap_48 = [](cx_reg<float> reg) {
            auto real = unpacklo_128(reg.real, reg.imag);
            auto imag = unpackhi_128(reg.real, reg.imag);
            return cx_reg<float>({real, imag});
        };

        template<std::size_t PackFrom, std::size_t PackTo>
        static inline auto repack(auto... args)
        {
            auto tup = std::make_tuple(args...);
            if constexpr (PackFrom == PackTo)
            {
                return tup;
            } else if constexpr (PackFrom == 1)
            {
                if constexpr (PackTo == 8)
                {
                    auto pack_1 = [](cx_reg<float> reg) {
                        auto real = _mm256_shuffle_ps(reg.real, reg.imag, 0b10001000);
                        auto imag = _mm256_shuffle_ps(reg.real, reg.imag, 0b11011101);
                        return cx_reg<float>({real, imag});
                    };

                    auto tmp = internal::apply_for_each(swap_48, tup);
                    return internal::apply_for_each(pack_1, tmp);
                } else if constexpr (PackTo == 4)
                {
                    auto tmp = internal::apply_for_each(swap_12, tup);
                    return internal::apply_for_each(swap_24, tmp);

                } else if constexpr (PackTo == 2)
                {
                    return internal::apply_for_each(swap_12, tup);
                }
            } else if constexpr (PackFrom == 2)
            {
                if constexpr (PackTo == 8)
                {
                    auto pack_1 = [](cx_reg<float> reg) {
                        auto real = unpacklo_pd(reg.real, reg.imag);
                        auto imag = unpackhi_pd(reg.real, reg.imag);
                        return cx_reg<float>({real, imag});
                    };
                    auto tmp = internal::apply_for_each(swap_48, tup);
                    return internal::apply_for_each(pack_1, tmp);
                } else if constexpr (PackTo == 4)
                {
                    return internal::apply_for_each(swap_24, tup);
                } else if constexpr (PackTo == 1)
                {
                    return internal::apply_for_each(swap_12, tup);
                }
            } else if constexpr (PackFrom == 4)
            {
                if constexpr (PackTo == 8)
                {
                    return internal::apply_for_each(swap_48, tup);
                } else if constexpr (PackTo == 2)
                {
                    return internal::apply_for_each(swap_24, tup);
                } else if constexpr (PackTo == 1)
                {
                    auto tmp = internal::apply_for_each(swap_24, tup);
                    return internal::apply_for_each(swap_12, tmp);
                }
            } else if constexpr (PackFrom == 8)
            {
                if constexpr (PackTo == 4)
                {
                    return internal::apply_for_each(swap_48, tup);
                } else if constexpr (PackTo == 2)
                {
                    auto tmp = internal::apply_for_each(swap_48, tup);
                    return internal::apply_for_each(swap_24, tmp);
                } else if constexpr (PackTo == 1)
                {
                    auto pack_0 = [](cx_reg<float> reg) {
                        auto real = unpacklo_ps(reg.real, reg.imag);
                        auto imag = unpackhi_ps(reg.real, reg.imag);
                        return cx_reg<float>({real, imag});
                    };
                    auto tmp = internal::apply_for_each(pack_0, tup);
                    return internal::apply_for_each(swap_48, tmp);
                }
            }
        };
    };
}    // namespace avx

template<typename T, bool Const, std::size_t PackSize>
inline void packed_copy(iterator<T, Const, PackSize> first,
                        iterator<T, Const, PackSize> last,
                        iterator<T, false, PackSize> d_first)
{
    packed_copy(iterator<T, true, PackSize>(first),
                iterator<T, true, PackSize>(last),
                d_first);
}

template<typename T, std::size_t PackSize>
void packed_copy(iterator<T, true, PackSize>  first,
                 iterator<T, true, PackSize>  last,
                 iterator<T, false, PackSize> d_first)
{
    while (first < std::min(first.align_upper(), last))
    {
        *d_first++ = *first;
        ++first;
    }
    if (first.aligned() && d_first.aligned() && first < last)
    {
        auto size         = (last - first);
        auto aligned_size = size / PackSize * PackSize * 2;
        std::memcpy(&(*d_first), &(*first), (aligned_size + size % PackSize) * sizeof(T));
        std::memcpy(&(*d_first) + aligned_size + PackSize,
                    &(*first) + aligned_size + PackSize,
                    size % PackSize * sizeof(T));
        return;
    }
    while (first < last)
    {
        *d_first++ = *first;
        ++first;
    }
};


template<typename T, std::size_t PackSize>
void fill(iterator<T, false, PackSize> first,
          iterator<T, false, PackSize> last,
          std::complex<T>              value)
{
    constexpr const std::size_t reg_size = 32 / sizeof(T);

    while (first < std::min(first.align_upper(), last))
    {
        *first = value;
        ++first;
    }
    const auto scalar = avx::broadcast(value);
    while (first < last.align_lower())
    {
        auto ptr = &(*first);
        for (uint i = 0; i < PackSize / reg_size; ++i)
        {
            avx::store(ptr + reg_size * i, scalar.real);
            avx::store(ptr + reg_size * i + PackSize, scalar.imag);
        }
        first += PackSize;
    }
    while (first < last)
    {
        *first = value;
        ++first;
    }
};

template<typename T, std::size_t PackSize, typename U>
    requires std::convertible_to<U, std::complex<T>>
inline void fill(iterator<T, false, PackSize> first,
                 iterator<T, false, PackSize> last,
                 U                            value)
{
    auto value_cx = std::complex<T>(value);
    fill(first, last, value_cx);
};
}    // namespace pcx
#endif