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
    constexpr uint reg_size = 32;

    const auto re_256 = _mm256_broadcast_ss(&reinterpret_cast<T(&)[2]>(value)[0]);
    const auto im_256 = _mm256_broadcast_ss(&reinterpret_cast<T(&)[2]>(value)[1]);
    const auto re_128 = _mm256_castps256_ps128(re_256);
    const auto im_128 = _mm256_castps256_ps128(im_256);

    while (first < std::min(first.align_upper(), last))
    {
        auto ptr = &(*first);
        _mm_store_ss(ptr, re_128);
        _mm_store_ss(ptr + PackSize, im_128);
        ++first;
    }
    while (first < last.align_lower())
    {
        auto ptr = &(*first);
        for (uint i = 0; i < PackSize / (reg_size / sizeof(T)); ++i)
        {
            _mm256_storeu_ps(ptr + (reg_size / sizeof(T)) * i, re_256);
            _mm256_storeu_ps(ptr + (reg_size / sizeof(T)) * i + PackSize, im_256);
        }
        first += PackSize;
    }
    while (first < last)
    {
        auto ptr = &(*first);
        _mm_store_ss(ptr, re_128);
        _mm_store_ss(ptr + PackSize, im_128);
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