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
#include <type_traits>

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
constexpr const uZ default_pack_size = 32 / sizeof(T);

constexpr const uZ dynamic_size = -1;

template<typename T, uZ PackSize>
concept packed_floating_point = floating_point<T> && pack_size<PackSize>;

template<typename T, uZ PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize>
class vector;

template<typename T, bool Const, uZ PackSize>
class iterator;
namespace detail_ {
template<typename T, bool Const, uZ PackSize>
struct iterator_maker;
}

template<typename T, bool Const, uZ PackSize>
class subrange;

template<typename T, bool Const, uZ PackSize>
class cx_ref;


template<uZ PackSize>
constexpr auto pidx(uZ idx) -> uZ {
    return idx + idx / PackSize * PackSize;
}

template<typename T, std::align_val_t Alignment = std::align_val_t{64}>
class aligned_allocator {
public:
    using value_type      = T;
    using is_always_equal = std::true_type;

    aligned_allocator() noexcept = default;

    template<typename U>
    explicit aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {};

    aligned_allocator(const aligned_allocator&) noexcept = default;
    aligned_allocator(aligned_allocator&&) noexcept      = default;

    ~aligned_allocator() = default;

    aligned_allocator& operator=(const aligned_allocator&) noexcept = default;
    aligned_allocator& operator=(aligned_allocator&&) noexcept      = default;

    [[nodiscard]] auto allocate(uZ n) -> value_type* {
        return reinterpret_cast<value_type*>(::operator new[](n * sizeof(value_type), Alignment));
    }

    void deallocate(value_type* p, uZ) {
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
// template<typename T, std::align_val_t Alignment>
// bool operator!=(const aligned_allocator<T, Alignment>&, const aligned_allocator<T, Alignment>&) noexcept {
//     return false;
// }

namespace detail_ {

template<typename T>
struct is_std_complex_floating_point {
    static constexpr bool value = false;
};

template<std::floating_point F>
struct is_std_complex_floating_point<std::complex<F>> {
    using real_type             = F;
    static constexpr bool value = true;
};

template<typename T>
struct is_pcx_iterator {
    static constexpr auto value = std::false_type{};
};

template<typename T, bool Const, uZ PackSize>
struct is_pcx_iterator<iterator<T, Const, PackSize>> {
    static constexpr auto value = std::true_type{};
};

template<bool Always>
struct aligned_base {};
template<>
struct aligned_base<true> {
    static constexpr std::true_type always_aligned{};
};
}    // namespace detail_

template<typename T>
concept always_aligned = std::derived_from<T, detail_::aligned_base<true>>;

template<typename V>
struct cx_vector_traits {
    using real_type               = void;
    static constexpr uZ pack_size = 0;
};

template<typename R>
    requires rv::contiguous_range<R> && detail_::is_std_complex_floating_point<rv::range_value_t<R>>::value
struct cx_vector_traits<R> {
    using real_type = typename detail_::is_std_complex_floating_point<rv::range_value_t<R>>::real_type;
    static constexpr uZ pack_size = 1;

    static auto re_data(R& vector) -> real_type* {
        return reinterpret_cast<real_type*>(rv::data(vector));
    }
    static auto re_data(const R& vector) -> const real_type* {
        return reinterpret_cast<const real_type*>(rv::data(vector));
    }
    static auto size(const R& vector) {
        return rv::size(vector);
    }
};

template<typename R>
    requires(detail_::is_pcx_iterator<rv::iterator_t<R>>::value)
struct cx_vector_traits<R> {
    using real_type               = typename rv::iterator_t<R>::real_type;
    static constexpr uZ pack_size = rv::iterator_t<R>::pack_size;

    inline static auto re_data(R& vector) {
        auto it = vector.begin();
        if constexpr (!always_aligned<R>) {
            if (!it.aligned()) {
                throw(
                    std::invalid_argument("Packed complex range is not aligned. Packed complex range must be "
                                          "aligned to be accessed as a vector."));
            }
        }
        return &(*it);
    }

    inline static auto re_data(const R& vector) {
        auto it = vector.begin();
        if constexpr (!always_aligned<R>) {
            if (!it.aligned()) {
                throw(
                    std::invalid_argument("Packed complex range is not aligned. Packed complex range must be "
                                          "aligned to be accessed as a vector."));
            }
        }
        return &(*it);
    }

    inline static auto size(const R& vector) {
        return rv::size(vector);
    }
};

// template<typename T_, uZ PackSize_, typename Alloc_>
// struct cx_vector_traits<pcx::vector<T_, PackSize_, Alloc_>> {
//     using real_type               = T_;
//     static constexpr uZ pack_size = PackSize_;
//
//     static auto re_data(pcx::vector<T_, PackSize_, Alloc_>& vector) -> real_type* {
//         return vector.data();
//     }
//     static auto re_data(const pcx::vector<T_, PackSize_, Alloc_>& vector) -> const real_type* {
//         return vector.data();
//     }
//     static auto size(const pcx::vector<T_, PackSize_, Alloc_>& vector) {
//         return vector.size();
//     }
// };

// template<typename T_, bool Const_, uZ PackSize_>
// struct cx_vector_traits<pcx::subrange<T_, Const_, PackSize_>> {
//     using real_type = T_;
//
//     static auto re_data(pcx::subrange<T_, Const_, PackSize_> subrange) {
//         if (!subrange.aligned()) {
//             throw(std::invalid_argument(
//                 "subrange is not aligned. pcx::subrange must be aligned to be accessed as a vector"));
//         }
//         return &(*subrange.begin());
//     }
//     static auto size(pcx::subrange<T_, Const_, PackSize_> subrange) {
//         return subrange.size();
//     }
// };

template<typename T, typename V>
concept complex_vector_of = std::same_as<T, typename cx_vector_traits<V>::real_type>;

template<typename T, typename R>
concept range_of_complex_vector_of = rv::random_access_range<R> &&    //
                                     complex_vector_of<T, std::remove_pointer_t<rv::range_value_t<R>>>;


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

    template<uZ PackTo>
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

template<typename T, bool Const, uZ PackSize>
inline void packed_copy(iterator<T, Const, PackSize> first,
                        iterator<T, Const, PackSize> last,
                        iterator<T, false, PackSize> d_first) {
    packed_copy(iterator<T, true, PackSize>(first), iterator<T, true, PackSize>(last), d_first);
}

template<typename T, uZ PackSize>
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


template<typename T, uZ PackSize>
void fill(iterator<T, false, PackSize> first, iterator<T, false, PackSize> last, std::complex<T> value) {
    constexpr const uZ reg_size = 32 / sizeof(T);

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

template<typename T, uZ PackSize, typename U>
    requires std::convertible_to<U, std::complex<T>>
inline void fill(iterator<T, false, PackSize> first, iterator<T, false, PackSize> last, U value) {
    auto value_cx = std::complex<T>(value);
    fill(first, last, value_cx);
};
}    // namespace pcx
#endif