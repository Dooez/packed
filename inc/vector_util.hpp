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
    inline auto zip_tuple_element(Tups&&... tuples)
    {
        return std::make_tuple(std::get<I>(std::forward<Tups>(tuples))...);
    }

    template<std::size_t... I, typename... Tups>
    inline auto zip_tuples_impl(std::index_sequence<I...>, Tups&&... tuples)
    {
        return std::make_tuple(zip_tuple_element<I>(std::forward<Tups>(tuples)...)...);
    }

    template<typename... Tups>
    inline auto zip_tuples(Tups&&... tuples)
    {
        constexpr auto I = std::min(std::tuple_size_v<std::remove_cvref_t<Tups>>...);
        return zip_tuples_impl(std::make_index_sequence<I>{},
                               std::forward<Tups>(tuples)...);
    }

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