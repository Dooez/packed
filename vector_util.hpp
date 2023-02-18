#ifndef VECTOR_UTIL_HPP
#define VECTOR_UTIL_HPP

#include <complex>
#include <cstring>
#include <immintrin.h>
#include <span>
#include <variant>

namespace pcx {

constexpr const std::size_t dynamic_size = -1;

template<std::size_t N>
concept power_of_two = (N & (N - 1)) == 0;

template<typename T, std::size_t PackSize>
concept packed_floating_point = std::floating_point<T> && power_of_two<PackSize> &&
                                (PackSize >= 32 / sizeof(T));

template<typename T, typename Allocator, std::size_t PackSize>
    requires packed_floating_point<T, PackSize>
class vector;

template<typename T, bool Const, std::size_t PackSize>
class cx_ref;

template<typename T, bool Const, std::size_t PackSize>
class iterator;

template<typename T, bool Const, std::size_t Size, std::size_t PackSize>
class subrange;

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

    inline auto load(const float* source) -> reg<float>::type
    {
        return _mm256_loadu_ps(source);
    }
    inline auto load(const double* source) -> reg<double>::type
    {
        return _mm256_loadu_pd(source);
    }

    inline auto broadcast(const float* source) -> reg<float>::type
    {
        return _mm256_broadcast_ss(source);
    }
    inline auto broadcast(const double* source) -> reg<double>::type
    {
        return _mm256_broadcast_sd(source);
    }

    void store(float* dest, reg<float>::type reg)
    {
        return _mm256_storeu_ps(dest, reg);
    }
    void store(double* dest, reg<double>::type reg)
    {
        return _mm256_storeu_pd(dest, reg);
    }

    void store_s(float* dest, reg<float>::type reg)
    {
        const auto reg128 = _mm256_castps256_ps128(reg);
        return _mm_store_ss(dest, reg128);
    }
    void store_s(double* dest, reg<double>::type reg)
    {
        const auto reg128 = _mm256_castpd256_pd128(reg);
        return _mm_store_sd(dest, reg128);
    }


}    // namespace avx

template<typename T, std::size_t PackSize, bool Const>
inline void packed_copy(iterator<T, Const, PackSize> first,
                        iterator<T, Const, PackSize> last,
                        iterator<T, false, PackSize> d_first)
{
    packed_copy(iterator<T, PackSize, true>(first),
                iterator<T, PackSize, true>(last),
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
    constexpr std::size_t reg_size = 32 / sizeof(T);

    const auto real = avx::broadcast(&reinterpret_cast<T(&)[2]>(value)[0]);
    const auto imag = avx::broadcast(&reinterpret_cast<T(&)[2]>(value)[1]);

    while (first < std::min(first.align_upper(), last))
    {
        auto ptr = &(*first);
        avx::store_s(ptr, real);
        avx::store_s(ptr + PackSize, imag);
        ++first;
    }
    while (first < last.align_lower())
    {
        auto ptr = &(*first);
        for (uint i = 0; i < PackSize / reg_size; ++i)
        {
            avx::store(ptr + reg_size * i, real);
            avx::store(ptr + reg_size * i + PackSize, imag);
        }
        first += PackSize;
    }
    while (first < last)
    {
        auto ptr = &(*first);
        avx::store_s(ptr, real);
        avx::store_s(ptr + PackSize, imag);
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