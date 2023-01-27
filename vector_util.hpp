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

template<typename T, std::size_t PackSize, typename Allocator, bool Const>
class packed_cx_ref;

template<typename T, std::size_t PackSize, typename Allocator, bool Const>
class packed_iterator;

template<typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize>
void aligned_copy(packed_cx_ref<T, PackSize, Allocator, false>         dest,
                  packed_cx_ref<T, PackSize, Allocator, true>          source,
                  typename std::allocator_traits<Allocator>::size_type size)
{
    auto aligned_size = size / PackSize * 2 * PackSize;
    std::memcpy(&dest, &source, (aligned_size + size % PackSize) * sizeof(T));
    std::memcpy(&dest + aligned_size + PackSize,
                &source + aligned_size + PackSize,
                size % PackSize * sizeof(T));
};


template<typename Vec, typename T, std::size_t PackSize, typename Allocator>
    requires std::same_as<Vec, packed_cx_vector<T, PackSize, Allocator>>
void copy(typename Vec::const_iterator first,
          typename Vec::const_iterator last,
          typename Vec::iterator       d_first)
{
    if (first.aligned() && d_first.aligned())
    {
        aligned_copy(*d_first, *first, last - first);
        return;
    }
    for (; first < last; ++first)
    {
        *d_first++ = *first;
    }
};

template<typename T, std::size_t PackSize, typename Allocator>
void set(packed_iterator<T, PackSize, Allocator, false> first,    //
         packed_iterator<T, PackSize, Allocator, false> last,     //
         std::complex<T>                                value)
{
    constexpr uint reg_size = 32;

    const auto re_256 = _mm256_broadcast_ss(&reinterpret_cast<T(&)[2]>(value)[0]);
    const auto im_256 = _mm256_broadcast_ss(&reinterpret_cast<T(&)[2]>(value)[1]);
    const auto re_128 = _mm256_castps256_ps128(re_256);
    const auto im_128 = _mm256_castps256_ps128(im_256);

    while (first < first.align_upper())
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
void set(packed_iterator<T, PackSize, Allocator, false> first,    //
         packed_iterator<T, PackSize, Allocator, false> last,     //
         U                                              value)
{
    auto value_cx = std::complex<T>(value);
    set(first, last, value_cx);
};
#endif