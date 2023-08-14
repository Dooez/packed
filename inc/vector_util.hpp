#ifndef VECTOR_UTIL_HPP
#define VECTOR_UTIL_HPP

#include "simd_impl/avx2_common.hpp"
#include "types.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <new>
#include <tuple>

namespace pcx {
using f32 = float;
using f64 = float;

using uZ  = std::size_t;
using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8  = uint8_t;

using iZ  = std::ptrdiff_t;
using i64 = int64_t;
using i32 = int32_t;
using i16 = int16_t;
using i8  = int8_t;

template<typename T>
    requires std::same_as<T, float> || std::same_as<T, double>
constexpr const std::size_t default_pack_size = 32 / sizeof(T);

constexpr const std::size_t dynamic_size = -1;

template<typename T, uZ PackSize>
concept packed_floating_point = std::floating_point<T> && pack_size<PackSize>;

template<typename T, uZ PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize>
class vector;

template<typename T, bool Const, std::size_t PackSize>
class iterator;

template<typename T, bool Const, std::size_t PackSize>
class subrange;

template<typename T, bool Const, std::size_t PackSize>
class cx_ref;


template<std::size_t PackSize>
constexpr auto pidx(std::size_t idx) -> std::size_t {
    return idx + idx / PackSize * PackSize;
}

template<typename T, std::align_val_t Alignment = std::align_val_t{64}>
class aligned_allocator {
public:
    using value_type      = T;
    using is_always_equal = std::true_type;

    aligned_allocator() = default;

    template<typename U>
    explicit aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {};

    aligned_allocator(const aligned_allocator&)     = default;
    aligned_allocator(aligned_allocator&&) noexcept = default;

    ~aligned_allocator() = default;

    aligned_allocator& operator=(const aligned_allocator&)     = default;
    aligned_allocator& operator=(aligned_allocator&&) noexcept = default;

    [[nodiscard]] auto allocate(std::size_t n) -> value_type* {
        return reinterpret_cast<value_type*>(::operator new[](n * sizeof(value_type), Alignment));
    }

    void deallocate(value_type* p, std::size_t) {
        ::operator delete[](reinterpret_cast<void*>(p), Alignment);
    }

    template<typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

private:
};


template<typename T, std::align_val_t Alignment>
bool operator==(const aligned_allocator<T, Alignment>&, const aligned_allocator<T, Alignment>&) noexcept {
    return true;
}
template<typename T, std::align_val_t Alignment>
bool operator!=(const aligned_allocator<T, Alignment>&, const aligned_allocator<T, Alignment>&) noexcept {
    return false;
}

namespace simd {

template<typename T>
struct convert;

template<>
struct convert<float> {
    static constexpr auto swap_48 = [](cx_reg<float, false> reg) {
        auto real = avx2::unpacklo_128(reg.real, reg.imag);
        auto imag = avx2::unpackhi_128(reg.real, reg.imag);
        return cx_reg<float, false>({real, imag});
    };

    /**
     * @brief Shuffles data to put I/Q into separate simd registers.
     * Resulting data order is dependent on input pack size.
     * if PackFrom < 4 the order of values across 2 registers changes
     * [0 1 2 3][4 5 6 7] -> [0 1 4 5][2 3 6 7]
     * Faster than true repack.
     */
    template<uZ PackFrom>
    static inline auto split(auto... args) {
        auto tup = std::make_tuple(args...);
        if constexpr (PackFrom == 1) {
            auto split = []<bool Conj>(cx_reg<float, Conj> reg) {
                auto real = _mm256_shuffle_ps(reg.real, reg.imag, 0b10001000);
                auto imag = _mm256_shuffle_ps(reg.real, reg.imag, 0b11011101);
                return cx_reg<float, Conj>({real, imag});
            };
            return pcx::detail_::apply_for_each(split, tup);
        } else if constexpr (PackFrom == 2) {
            auto split = []<bool Conj>(cx_reg<float, Conj> reg) {
                auto real = avx2::unpacklo_pd(reg.real, reg.imag);
                auto imag = avx2::unpackhi_pd(reg.real, reg.imag);
                return cx_reg<float, Conj>({real, imag});
            };
            return pcx::detail_::apply_for_each(split, tup);
        } else if constexpr (PackFrom == 4) {
            return pcx::detail_::apply_for_each(swap_48, tup);
        } else {
            return tup;
        }
    }

    template<std::size_t PackTo>
    static inline auto combine(auto... args) {
        auto tup = std::make_tuple(args...);
        if constexpr (PackTo == 1) {
            auto combine = []<bool Conj>(cx_reg<float, Conj> reg) {
                auto real = simd::avx2::unpacklo_ps(reg.real, reg.imag);
                auto imag = simd::avx2::unpackhi_ps(reg.real, reg.imag);
                return cx_reg<float, Conj>({real, imag});
            };
            return pcx::detail_::apply_for_each(combine, tup);
        } else if constexpr (PackTo == 2) {
            auto combine = []<bool Conj>(cx_reg<float, Conj> reg) {
                auto real = avx2::unpacklo_pd(reg.real, reg.imag);
                auto imag = avx2::unpackhi_pd(reg.real, reg.imag);
                return cx_reg<float, Conj>({real, imag});
            };
            return pcx::detail_::apply_for_each(combine, tup);
        } else if constexpr (PackTo == 4) {
            return pcx::detail_::apply_for_each(swap_48, tup);
        } else {
            return tup;
        }
    }
};

}    // namespace simd

template<typename T, bool Const, std::size_t PackSize>
inline void packed_copy(iterator<T, Const, PackSize> first,
                        iterator<T, Const, PackSize> last,
                        iterator<T, false, PackSize> d_first) {
    packed_copy(iterator<T, true, PackSize>(first), iterator<T, true, PackSize>(last), d_first);
}

template<typename T, std::size_t PackSize>
void packed_copy(iterator<T, true, PackSize>  first,
                 iterator<T, true, PackSize>  last,
                 iterator<T, false, PackSize> d_first) {
    while (first < std::min(first.align_upper(), last)) {
        *d_first++ = *first;
        ++first;
    }
    if (first.aligned() && d_first.aligned() && first < last) {
        auto size         = (last - first);
        auto aligned_size = size / PackSize * PackSize * 2;
        std::memcpy(&(*d_first), &(*first), (aligned_size + size % PackSize) * sizeof(T));
        std::memcpy(&(*d_first) + aligned_size + PackSize,
                    &(*first) + aligned_size + PackSize,
                    size % PackSize * sizeof(T));
        return;
    }
    while (first < last) {
        *d_first++ = *first;
        ++first;
    }
};


template<typename T, std::size_t PackSize>
void fill(iterator<T, false, PackSize> first, iterator<T, false, PackSize> last, std::complex<T> value) {
    constexpr const std::size_t reg_size = 32 / sizeof(T);

    while (first < std::min(first.align_upper(), last)) {
        *first = value;
        ++first;
    }
    const auto scalar = simd::broadcast(value);
    while (first < last.align_lower()) {
        auto ptr = &(*first);
        for (uint i = 0; i < PackSize / reg_size; ++i) {
            simd::store(ptr + reg_size * i, scalar.real);
            simd::store(ptr + reg_size * i + PackSize, scalar.imag);
        }
        first += PackSize;
    }
    while (first < last) {
        *first = value;
        ++first;
    }
};

template<typename T, std::size_t PackSize, typename U>
    requires std::convertible_to<U, std::complex<T>>
inline void fill(iterator<T, false, PackSize> first, iterator<T, false, PackSize> last, U value) {
    auto value_cx = std::complex<T>(value);
    fill(first, last, value_cx);
};
}    // namespace pcx
#endif