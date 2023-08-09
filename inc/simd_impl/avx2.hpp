#ifndef AVX2_HPP
#define AVX2_HPP

#include "simd_common.hpp"

#include <immintrin.h>

namespace pcx::simd {

template<>
struct reg<float> {
    using type = __m256;
    static constexpr uZ size{32 / sizeof(float)};
};
template<>
struct reg<double> {
    using type = __m256d;
    static constexpr uZ size{32 / sizeof(double)};
};

template<typename T>
using reg_t = typename reg<T>::type;

template<>
struct simd_traits<reg_t<float>> {
    static constexpr uZ size = reg<float>::size;
};

template<>
struct simd_traits<reg_t<double>> {
    static constexpr uZ size = reg<double>::size;
};


template<>
inline auto zero<float>() -> reg_t<float> {
    return _mm256_setzero_ps();
};
template<>
inline auto zero<double>() -> reg_t<double> {
    return _mm256_setzero_pd();
};
inline auto broadcast(const float* source) -> reg_t<float> {
    return _mm256_broadcast_ss(source);
}
inline auto broadcast(const double* source) -> reg_t<double> {
    return _mm256_broadcast_sd(source);
}
template<typename T>
inline auto broadcast(T source) -> reg_t<T> {
    return broadcast(&source);
}
inline auto load(const float* source) -> reg_t<float> {
    return _mm256_loadu_ps(source);
}
inline auto load(const double* source) -> reg_t<double> {
    return _mm256_loadu_pd(source);
}
inline void store(float* dest, reg_t<float> reg) {
    return _mm256_storeu_ps(dest, reg);
}
inline void store(double* dest, reg_t<double> reg) {
    return _mm256_storeu_pd(dest, reg);
}

template<typename T>
inline auto broadcast(std::complex<T> source) -> cx_reg<T, false> {
    return {broadcast(source.real()), broadcast(source.imag())};
}
template<uZ PackSize, typename T>
    requires(PackSize >= reg<T>::size)
inline auto cxload(const T* ptr) -> cx_reg<T, false> {
    return {load(ptr), load(ptr + PackSize)};
}
template<uZ PackSize, typename T, bool Conj>
    requires(PackSize >= reg<T>::size)
inline void cxstore(T* ptr, cx_reg<T, Conj> reg) {
    store(ptr, reg.real);
    if constexpr (Conj) {
        store(ptr + PackSize, simd::sub(zero<T>(), reg.imag));
    } else {
        store(ptr + PackSize, reg.imag);
    }
}

template<uZ PackSize, typename T, bool Conj>
    requires(PackSize >= reg<T>::size)
inline auto cxloadstore(T* ptr, cx_reg<T, Conj> reg) -> cx_reg<T, false> {
    auto tmp = cxload<PackSize>(ptr);
    cxstore<PackSize>(ptr, reg);
    return tmp;
}

namespace avx2 {

inline auto unpacklo_ps(reg_t<float> a, reg_t<float> b) -> reg_t<float> {
    return _mm256_unpacklo_ps(a, b);
};
inline auto unpackhi_ps(reg_t<float> a, reg_t<float> b) -> reg_t<float> {
    return _mm256_unpackhi_ps(a, b);
};

inline auto unpacklo_pd(reg_t<float> a, reg_t<float> b) -> reg_t<float> {
    return _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
};
inline auto unpackhi_pd(reg_t<float> a, reg_t<float> b) -> reg_t<float> {
    return _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
};
inline auto unpacklo_pd(reg_t<double> a, reg_t<double> b) -> reg_t<double> {
    return _mm256_unpacklo_pd(a, b);
};
inline auto unpackhi_pd(reg_t<double> a, reg_t<double> b) -> reg_t<double> {
    return _mm256_unpackhi_pd(a, b);
};

inline auto unpacklo_128(reg_t<float> a, reg_t<float> b) -> reg_t<float> {
    return _mm256_insertf128_ps(a, _mm256_extractf128_ps(b, 0), 1);
};
inline auto unpackhi_128(reg_t<float> a, reg_t<float> b) -> reg_t<float> {
    return _mm256_permute2f128_ps(a, b, 0b00110001);
};
inline auto unpacklo_128(reg_t<double> a, reg_t<double> b) -> reg_t<double> {
    return _mm256_insertf128_pd(a, _mm256_extractf128_pd(b, 0), 1);
};
inline auto unpackhi_128(reg_t<double> a, reg_t<double> b) -> reg_t<double> {
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

inline constexpr auto swap_12 = []<bool Conj>(cx_reg<float, Conj> reg) /* static */ {
    auto real = _mm256_shuffle_ps(reg.real, reg.real, 0b11011000);
    auto imag = _mm256_shuffle_ps(reg.imag, reg.imag, 0b11011000);
    return cx_reg<float, Conj>({real, imag});
};

inline constexpr auto swap_24 = []<bool Conj>(cx_reg<float, Conj> reg) /* static */ {
    auto real = _mm256_permute4x64_pd(_mm256_castps_pd(reg.real), 0b11011000);
    auto imag = _mm256_permute4x64_pd(_mm256_castps_pd(reg.imag), 0b11011000);
    return cx_reg<float, Conj>({_mm256_castpd_ps(real), _mm256_castpd_ps(imag)});
};

inline constexpr auto swap_48 = []<bool Conj>(cx_reg<float, Conj> reg) /* static */ {
    auto real = unpacklo_128(reg.real, reg.imag);
    auto imag = unpackhi_128(reg.real, reg.imag);
    return cx_reg<float, Conj>({real, imag});
};

}    // namespace avx2

template<uZ PackFrom, uZ PackTo, bool... Conj>
    requires(!(Conj || ...))
inline auto repack(cx_reg<float, Conj>... args) {
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

            auto tmp = pcx::detail_::apply_for_each(avx2::swap_48, tup);
            return pcx::detail_::apply_for_each(pack_1, tmp);
        } else if constexpr (PackTo == 4) {
            auto tmp = pcx::detail_::apply_for_each(avx2::swap_12, tup);
            return pcx::detail_::apply_for_each(avx2::swap_24, tmp);
        } else if constexpr (PackTo == 2) {
            return pcx::detail_::apply_for_each(avx2::swap_12, tup);
        }
    } else if constexpr (PackFrom == 2) {
        if constexpr (PackTo >= 8) {
            auto pack_1 = [](cx_reg<float, false> reg) {
                auto real = avx2::unpacklo_pd(reg.real, reg.imag);
                auto imag = avx2::unpackhi_pd(reg.real, reg.imag);
                return cx_reg<float, false>({real, imag});
            };
            auto tmp = pcx::detail_::apply_for_each(avx2::swap_48, tup);
            return pcx::detail_::apply_for_each(pack_1, tmp);
        } else if constexpr (PackTo == 4) {
            return pcx::detail_::apply_for_each(avx2::swap_24, tup);
        } else if constexpr (PackTo == 1) {
            return pcx::detail_::apply_for_each(avx2::swap_12, tup);
        }
    } else if constexpr (PackFrom == 4) {
        if constexpr (PackTo >= 8) {
            return pcx::detail_::apply_for_each(avx2::swap_48, tup);
        } else if constexpr (PackTo == 2) {
            return pcx::detail_::apply_for_each(avx2::swap_24, tup);
        } else if constexpr (PackTo == 1) {
            auto tmp = pcx::detail_::apply_for_each(avx2::swap_24, tup);
            return pcx::detail_::apply_for_each(avx2::swap_12, tmp);
        }
    } else if constexpr (PackFrom >= 8) {
        if constexpr (PackTo == 4) {
            return pcx::detail_::apply_for_each(avx2::swap_48, tup);
        } else if constexpr (PackTo == 2) {
            auto tmp = pcx::detail_::apply_for_each(avx2::swap_48, tup);
            return pcx::detail_::apply_for_each(avx2::swap_24, tmp);
        } else if constexpr (PackTo == 1) {
            auto pack_0 = [](cx_reg<float, false> reg) {
                auto real = simd::avx2::unpacklo_ps(reg.real, reg.imag);
                auto imag = simd::avx2::unpackhi_ps(reg.real, reg.imag);
                return cx_reg<float, false>({real, imag});
            };
            auto tmp = pcx::detail_::apply_for_each(pack_0, tup);
            return pcx::detail_::apply_for_each(avx2::swap_48, tmp);
        }
    }
};

template<>
inline auto add(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm256_add_ps(lhs, rhs);
}
template<>
inline auto add(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm256_add_pd(lhs, rhs);
}
template<>
inline auto sub(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm256_sub_ps(lhs, rhs);
}
template<>
inline auto sub(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm256_sub_pd(lhs, rhs);
}
template<>
inline auto mul(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm256_mul_ps(lhs, rhs);
}
template<>
inline auto mul(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm256_mul_pd(lhs, rhs);
}
template<>
inline auto div(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm256_div_ps(lhs, rhs);
}
template<>
inline auto div(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm256_div_pd(lhs, rhs);
}

template<>
inline auto fmadd(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm256_fmadd_ps(a, b, c);
}
template<>
inline auto fmadd(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm256_fmadd_pd(a, b, c);
}
template<>
inline auto fnmadd(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm256_fnmadd_ps(a, b, c);
}
template<>
inline auto fnmadd(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm256_fnmadd_pd(a, b, c);
}
template<>
inline auto fmsub(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm256_fmsub_ps(a, b, c);
}
template<>
inline auto fmsub(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm256_fmsub_pd(a, b, c);
}
template<>
inline auto fnmsub(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm256_fnmsub_ps(a, b, c);
}
template<>
inline auto fnmsub(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm256_fnmsub_pd(a, b, c);
}

}    // namespace pcx::simd

#endif