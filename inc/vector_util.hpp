#ifndef VECTOR_UTIL_HPP
#define VECTOR_UTIL_HPP

#include "simd_impl/avx2.hpp"
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

template<std::size_t N>
concept power_of_two = N > 0 && (N & (N - 1)) == 0;

template<typename T, std::size_t PackSize>
concept packed_floating_point = std::floating_point<T> && power_of_two<PackSize>;

template<typename T, std::size_t PackSize, typename Allocator>
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

/**
 * @brief forward declarations
 *
 */
namespace simd {

template<typename T, bool Conj>
auto conj(cx_reg<T, Conj> reg) -> cx_reg<T, !Conj> {
    return {reg.real, reg.imag};
}

/**
 * @brief Conditionaly swaps real and imaginary parts of complex simd vectors.
 *
 * @tparam Inverse If true performs swap.
 * @param args Variable number of complex simd vectors.
 * @return Tuple of invrsed simd complex vectors.
 */
// template<bool Inverse>
// inline auto inverse(auto... args) {
//     auto tup = std::make_tuple(args...);
//     if constexpr (Inverse) {
//         auto inverse = [](auto reg) {
//             using reg_t = decltype(reg);
//             return reg_t{reg.imag, reg.real};
//         };
//         return detail_::apply_for_each(inverse, tup);
//     } else {
//         return tup;
//     }
// };

/**
* @brief Register aligned adress
*
* @tparam PackSize
* @tparam T
* @param data Base address. Must be aligned by simd register size.
* @param offset New address offset. Must be a multiple of simd register size.
* If data in-pack index I is non-zero, offset must be less then PackSize - I;
* @return T*
*/
template<uZ PackSize, typename T>
constexpr auto ra_addr(T* data, uZ offset) -> T* {
    return data + offset + (offset / PackSize) * PackSize;
}
template<uZ PackSize, typename T>
constexpr auto ra_addr(const T* data, uZ offset) -> const T* {
    return data + offset + (offset / PackSize) * PackSize;
}
template<uZ PackSize, typename T>
    requires(PackSize <= reg<T>::size)
constexpr auto ra_addr(T* data, uZ offset) -> T* {
    return data + offset * 2;
}
template<uZ PackSize, typename T>
    requires(PackSize <= reg<T>::size)
constexpr auto ra_addr(const T* data, uZ offset) -> const T* {
    return data + offset * 2;
}

}    // namespace simd

/**
 * @brief alias for templated avx2 types and functions
 *
 */
namespace simd {

inline auto unpacklo_ps(reg<float>::type a, reg<float>::type b) -> reg<float>::type {
    return _mm256_unpacklo_ps(a, b);
};
inline auto unpackhi_ps(reg<float>::type a, reg<float>::type b) -> reg<float>::type {
    return _mm256_unpackhi_ps(a, b);
};

inline auto unpacklo_pd(reg<float>::type a, reg<float>::type b) -> reg<float>::type {
    return _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
};
inline auto unpackhi_pd(reg<float>::type a, reg<float>::type b) -> reg<float>::type {
    return _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
};
inline auto unpacklo_pd(reg<double>::type a, reg<double>::type b) -> reg<double>::type {
    return _mm256_unpacklo_pd(a, b);
};
inline auto unpackhi_pd(reg<double>::type a, reg<double>::type b) -> reg<double>::type {
    return _mm256_unpackhi_pd(a, b);
};

inline auto unpacklo_128(reg<float>::type a, reg<float>::type b) -> reg<float>::type {
    return _mm256_insertf128_ps(a, _mm256_extractf128_ps(b, 0), 1);
};
inline auto unpackhi_128(reg<float>::type a, reg<float>::type b) -> reg<float>::type {
    return _mm256_permute2f128_ps(a, b, 0b00110001);
};
inline auto unpacklo_128(reg<double>::type a, reg<double>::type b) -> reg<double>::type {
    return _mm256_insertf128_pd(a, _mm256_extractf128_pd(b, 0), 1);
};
inline auto unpackhi_128(reg<double>::type a, reg<double>::type b) -> reg<double>::type {
    return _mm256_permute2f128_pd(a, b, 0b00110001);
};

template<bool Conj>
inline auto unpack_ps(cx_reg<float, Conj> a, cx_reg<float, Conj> b)
    -> std::tuple<cx_reg<float, Conj>, cx_reg<float, Conj>> {
    auto real_lo = unpacklo_ps(a.real, b.real);
    auto real_hi = unpackhi_ps(a.real, b.real);
    auto imag_lo = unpacklo_ps(a.imag, b.imag);
    auto imag_hi = unpackhi_ps(a.imag, b.imag);

    return {cx_reg<float, Conj>({real_lo, imag_lo}), cx_reg<float, Conj>({real_hi, imag_hi})};
};

template<typename T, bool Conj>
inline auto unpack_pd(cx_reg<T, Conj> a, cx_reg<T, Conj> b) -> std::tuple<cx_reg<T, Conj>, cx_reg<T, Conj>> {
    auto real_lo = unpacklo_pd(a.real, b.real);
    auto real_hi = unpackhi_pd(a.real, b.real);
    auto imag_lo = unpacklo_pd(a.imag, b.imag);
    auto imag_hi = unpackhi_pd(a.imag, b.imag);

    return {cx_reg<T, Conj>({real_lo, imag_lo}), cx_reg<T, Conj>({real_hi, imag_hi})};
};

template<typename T, bool Conj>
inline auto unpack_128(cx_reg<T, Conj> a, cx_reg<T, Conj> b) -> std::tuple<cx_reg<T, Conj>, cx_reg<T, Conj>> {
    auto real_hi = unpackhi_128(a.real, b.real);
    auto real_lo = unpacklo_128(a.real, b.real);
    auto imag_hi = unpackhi_128(a.imag, b.imag);
    auto imag_lo = unpacklo_128(a.imag, b.imag);

    return {cx_reg<T, Conj>({real_lo, imag_lo}), cx_reg<T, Conj>({real_hi, imag_hi})};
};

template<typename T>
struct convert;

template<>
struct convert<float> {
    static constexpr auto swap_12 = [](cx_reg<float, false> reg) {
        auto real = _mm256_shuffle_ps(reg.real, reg.real, 0b11011000);
        auto imag = _mm256_shuffle_ps(reg.imag, reg.imag, 0b11011000);
        return cx_reg<float, false>({real, imag});
    };

    static constexpr auto swap_24 = [](cx_reg<float, false> reg) {
        auto real = _mm256_permute4x64_pd(_mm256_castps_pd(reg.real), 0b11011000);
        auto imag = _mm256_permute4x64_pd(_mm256_castps_pd(reg.imag), 0b11011000);
        return cx_reg<float, false>({_mm256_castpd_ps(real), _mm256_castpd_ps(imag)});
    };

    static constexpr auto swap_48 = [](cx_reg<float, false> reg) {
        auto real = unpacklo_128(reg.real, reg.imag);
        auto imag = unpackhi_128(reg.real, reg.imag);
        return cx_reg<float, false>({real, imag});
    };

    /**
     * @brief Shuffles data to convert from PackFrom to PackTo pack size;
     *
     * @param args arbitrary number of cx_reg<T>
     */
    template<std::size_t PackFrom, std::size_t PackTo>
        requires(PackFrom > 0) && (PackTo > 0)
    static inline auto repack(auto... args) {
        auto tup = std::make_tuple(args...);
        if constexpr (PackFrom == PackTo || (PackFrom >= 8 && PackTo >= 8)) {
            return tup;
        } else if constexpr (PackFrom == 1) {
            if constexpr (PackTo >= 8) {
                auto pack_1 = [](cx_reg<float, false> reg) {
                    auto real = _mm256_shuffle_ps(reg.real, reg.imag, 0b10001000);
                    auto imag = _mm256_shuffle_ps(reg.real, reg.imag, 0b11011101);
                    return cx_reg<float, false>({real, imag});
                };

                auto tmp = pcx::detail_::apply_for_each(swap_48, tup);
                return pcx::detail_::apply_for_each(pack_1, tmp);
            } else if constexpr (PackTo == 4) {
                auto tmp = pcx::detail_::apply_for_each(swap_12, tup);
                return pcx::detail_::apply_for_each(swap_24, tmp);
            } else if constexpr (PackTo == 2) {
                return pcx::detail_::apply_for_each(swap_12, tup);
            }
        } else if constexpr (PackFrom == 2) {
            if constexpr (PackTo >= 8) {
                auto pack_1 = [](cx_reg<float, false> reg) {
                    auto real = unpacklo_pd(reg.real, reg.imag);
                    auto imag = unpackhi_pd(reg.real, reg.imag);
                    return cx_reg<float, false>({real, imag});
                };
                auto tmp = pcx::detail_::apply_for_each(swap_48, tup);
                return pcx::detail_::apply_for_each(pack_1, tmp);
            } else if constexpr (PackTo == 4) {
                return pcx::detail_::apply_for_each(swap_24, tup);
            } else if constexpr (PackTo == 1) {
                return pcx::detail_::apply_for_each(swap_12, tup);
            }
        } else if constexpr (PackFrom == 4) {
            if constexpr (PackTo >= 8) {
                return pcx::detail_::apply_for_each(swap_48, tup);
            } else if constexpr (PackTo == 2) {
                return pcx::detail_::apply_for_each(swap_24, tup);
            } else if constexpr (PackTo == 1) {
                auto tmp = pcx::detail_::apply_for_each(swap_24, tup);
                return pcx::detail_::apply_for_each(swap_12, tmp);
            }
        } else if constexpr (PackFrom >= 8) {
            auto conj = []<bool Conj>(cx_reg<float, Conj> reg) {
                if constexpr (Conj) {
                    auto zero = _mm256_setzero_ps();
                    return cx_reg<float, false>{reg.real, _mm256_sub_ps(zero, reg.imag)};
                } else {
                    return reg;
                }
            };
            auto tup_ = pcx::detail_::apply_for_each(conj, tup);
            if constexpr (PackTo == 4) {
                return pcx::detail_::apply_for_each(swap_48, tup_);
            } else if constexpr (PackTo == 2) {
                auto tmp = pcx::detail_::apply_for_each(swap_48, tup_);
                return pcx::detail_::apply_for_each(swap_24, tmp);
            } else if constexpr (PackTo == 1) {
                auto pack_0 = [](cx_reg<float, false> reg) {
                    auto real = unpacklo_ps(reg.real, reg.imag);
                    auto imag = unpackhi_ps(reg.real, reg.imag);
                    return cx_reg<float, false>({real, imag});
                };
                auto tmp = pcx::detail_::apply_for_each(pack_0, tup_);
                return pcx::detail_::apply_for_each(swap_48, tmp);
            }
        }
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
                auto real = unpacklo_pd(reg.real, reg.imag);
                auto imag = unpackhi_pd(reg.real, reg.imag);
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
                auto real = unpacklo_ps(reg.real, reg.imag);
                auto imag = unpackhi_ps(reg.real, reg.imag);
                return cx_reg<float, Conj>({real, imag});
            };
            return pcx::detail_::apply_for_each(combine, tup);
        } else if constexpr (PackTo == 2) {
            auto combine = []<bool Conj>(cx_reg<float, Conj> reg) {
                auto real = unpacklo_pd(reg.real, reg.imag);
                auto imag = unpackhi_pd(reg.real, reg.imag);
                return cx_reg<float, Conj>({real, imag});
            };
            return pcx::detail_::apply_for_each(combine, tup);
        } else if constexpr (PackTo == 4) {
            return pcx::detail_::apply_for_each(swap_48, tup);
        } else {
            return tup;
        }
    }

    template<bool Inverse = true>
    static inline auto inverse(auto... args) {
        auto tup = std::make_tuple(args...);
        if constexpr (Inverse) {
            auto inverse = [](auto reg) {
                using reg_t = decltype(reg);
                return reg_t{reg.imag, reg.real};
            };
            return pcx::detail_::apply_for_each(inverse, tup);
        } else {
            return tup;
        }
    };
};

template<>
struct convert<double> {
    static constexpr auto swap_12 = []<bool Conj>(cx_reg<double, Conj> reg) {
        auto real = _mm256_permute4x64_pd(reg.real, 0b11011000);
        auto imag = _mm256_permute4x64_pd(reg.imag, 0b11011000);
        return cx_reg<double, Conj>({real, imag});
    };

    static constexpr auto swap_24 = []<bool Conj>(cx_reg<double, Conj> reg) {
        auto real = unpacklo_128(reg.real, reg.imag);
        auto imag = unpackhi_128(reg.real, reg.imag);
        return cx_reg<double, Conj>({real, imag});
    };

    template<std::size_t PackFrom, std::size_t PackTo>
        requires(PackFrom > 0) && (PackTo > 0)
    static inline auto repack(auto... args) {
        auto tup = std::make_tuple(args...);
        if constexpr (PackFrom == PackTo || (PackFrom >= 4 && PackTo >= 4)) {
            return tup;
        } else if constexpr (PackFrom == 1) {
            if constexpr (PackTo >= 4) {
                auto pack_1 = []<bool Conj>(cx_reg<double, Conj> reg) {
                    auto real = unpacklo_pd(reg.real, reg.imag);
                    auto imag = unpackhi_pd(reg.real, reg.imag);
                    return cx_reg<double, Conj>({real, imag});
                };
                auto tmp = pcx::detail_::apply_for_each(pack_1, tup);
                return pcx::detail_::apply_for_each(swap_12, tmp);
            } else if constexpr (PackTo == 2) {
                return pcx::detail_::apply_for_each(swap_12, tup);
            }
        } else if constexpr (PackFrom == 2) {
            if constexpr (PackTo >= 4) {
                return pcx::detail_::apply_for_each(swap_24, tup);
            } else if constexpr (PackTo == 1) {
                return pcx::detail_::apply_for_each(swap_12, tup);
            }
        } else if constexpr (PackFrom >= 4) {
            auto conj = []<bool Conj>(cx_reg<double, Conj> reg) {
                if constexpr (Conj) {
                    auto zero = _mm256_setzero_ps();
                    return cx_reg<double, false>{reg.real, _mm256_sub_ps(zero, reg.imag)};
                } else {
                    return reg;
                }
            };
            auto tup_ = pcx::detail_::apply_for_each(conj, tup);
            if constexpr (PackTo == 2) {
                return pcx::detail_::apply_for_each(swap_24, tup_);
            } else if constexpr (PackTo == 1) {
                auto pack_1 = []<bool Conj>(cx_reg<double, Conj> reg) {
                    auto real = unpacklo_pd(reg.real, reg.imag);
                    auto imag = unpackhi_pd(reg.real, reg.imag);
                    return cx_reg<double, Conj>({real, imag});
                };
                auto tmp = pcx::detail_::apply_for_each(pack_1, tup_);
                return pcx::detail_::apply_for_each(swap_24, tmp);
            }
        }
    };

    template<std::size_t PackFrom>
    static inline auto split(auto... args) {
        auto tup = std::make_tuple(args...);
        { return tup; }
    }

    template<std::size_t PackTo>
    static inline auto combine(auto... args) {
        auto tup = std::make_tuple(args...);
        { return tup; }
    }

    template<bool Inverse = true>
    static inline auto inverse(auto... args) {
        auto tup = std::make_tuple(args...);
        if constexpr (Inverse) {
            auto inverse = [](auto reg) {
                using reg_t = decltype(reg);
                return reg_t{reg.imag, reg.real};
            };
            return pcx::detail_::apply_for_each(inverse, tup);
        } else {
            return tup;
        }
    };
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