#ifndef VECTOR_UTIL_HPP
#define VECTOR_UTIL_HPP

#include <complex>
#include <cstring>
#include <immintrin.h>

template<std::size_t N>
concept power_of_two = requires { (N & (N - 1)) == 0; };

template<typename T, std::size_t PackSize>
concept packed_floating_point = std::floating_point<T> && power_of_two<PackSize> &&
                                requires { PackSize >= 64 / sizeof(T); };

template<typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize>
class packed_cx_vector;

template<typename T, std::size_t PackSize, typename Allocator, bool Const = false>
class packed_cx_ref;

template<typename T, std::size_t PackSize, typename Allocator, bool Const = false>
class packed_iterator;

template<typename T, std::size_t PackSize, typename Allocator, bool Const = false>
class packed_subrange;


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
inline auto load(const T* source) -> typename reg<T>::type;
template<>
inline auto load<float>(const float* source) -> reg<float>::type
{
    return _mm256_loadu_ps(source);
}
template<>
inline auto load<double>(const double* source) -> reg<double>::type
{
    return _mm256_loadu_pd(source);
}

template<typename T>
inline auto broadcast(const T* source) -> typename reg<T>::type;
template<>
inline auto broadcast<float>(const float* source) -> reg<float>::type
{
    return _mm256_broadcast_ss(source);
}
template<>
inline auto broadcast<double>(const double* source) -> reg<double>::type
{
    return _mm256_broadcast_sd(source);
}

template<typename T>
inline auto store(T* dest, typename reg<T>::type reg) -> void;
template<>
inline auto store<float>(float* dest, reg<float>::type reg) -> void
{
    return _mm256_storeu_ps(dest, reg);
}
template<>
inline auto store<double>(double* dest, reg<double>::type reg) -> void
{
    return _mm256_storeu_pd(dest, reg);
}

template<typename T>
inline auto store_s(T* dest, typename reg<T>::type reg) -> void;
template<>
inline auto store_s<float>(float* dest, reg<float>::type reg) -> void
{
    const auto reg128 = _mm256_castps256_ps128(reg);
    return _mm_store_ss(dest, reg128);
}
template<>
inline auto store_s<double>(double* dest, reg<double>::type reg) -> void
{
    const auto reg128 = _mm256_castpd256_pd128(reg);
    return _mm_store_sd(dest, reg128);
}


}    // namespace avx

template<typename T, std::size_t PackSize, typename Allocator, bool Const>
inline void packed_copy(packed_iterator<T, PackSize, Allocator, Const> first,
                        packed_iterator<T, PackSize, Allocator, Const> last,
                        packed_iterator<T, PackSize, Allocator>        d_first)
{
    packed_copy(packed_iterator<T, PackSize, Allocator, true>(first),
                packed_iterator<T, PackSize, Allocator, true>(last),
                d_first);
}

template<typename T, std::size_t PackSize, typename Allocator>
void packed_copy(packed_iterator<T, PackSize, Allocator, true> first,
                 packed_iterator<T, PackSize, Allocator, true> last,
                 packed_iterator<T, PackSize, Allocator>       d_first)
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


template<typename T, std::size_t PackSize, typename Allocator>
void set(packed_iterator<T, PackSize, Allocator> first,
         packed_iterator<T, PackSize, Allocator> last,
         std::complex<T>                         value)
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

template<typename T, std::size_t PackSize, typename Allocator, typename U>
    requires std::convertible_to<U, std::complex<T>>
inline void set(packed_iterator<T, PackSize, Allocator> first,
                packed_iterator<T, PackSize, Allocator> last,
                U                                       value)
{
    auto value_cx = std::complex<T>(value);
    set(first, last, value_cx);
};
#endif