#ifndef AVX2_HPP
#define AVX2_HPP

#include "simd_base.hpp"

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

template<>
inline auto setzero<float>() -> reg_t<float> {
    return _mm256_setzero_ps();
};
template<>
inline auto setzero<double>() -> reg_t<double> {
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
inline auto cxload(const T* ptr) -> cx_reg<T, false> {
    return {load(ptr), load(ptr + PackSize)};
}
template<uZ PackSize, typename T, bool Conj>
inline void cxstore(T* ptr, cx_reg<T, Conj> reg) {
    store(ptr, reg.real);
    if constexpr (Conj) {
        auto zero = setzero<T>();
        store(ptr + PackSize, sub(zero, reg.imag));
    } else {
        store(ptr + PackSize, reg.imag);
    }
}

template<uZ PackSize, typename T, bool Conj>
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

}    // namespace avx2


inline auto add(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm256_add_ps(lhs, rhs);
}
inline auto add(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm256_add_pd(lhs, rhs);
}
inline auto sub(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm256_sub_ps(lhs, rhs);
}
inline auto sub(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm256_sub_pd(lhs, rhs);
}
inline auto mul(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm256_mul_ps(lhs, rhs);
}
inline auto mul(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm256_mul_pd(lhs, rhs);
}
inline auto div(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm256_div_ps(lhs, rhs);
}
inline auto div(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm256_div_pd(lhs, rhs);
}

inline auto fmadd(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm256_fmadd_ps(a, b, c);
}
inline auto fmadd(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm256_fmadd_pd(a, b, c);
}
inline auto fnmadd(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm256_fnmadd_ps(a, b, c);
}
inline auto fnmadd(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm256_fnmadd_pd(a, b, c);
}
inline auto fmsub(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm256_fmsub_ps(a, b, c);
}
inline auto fmsub(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm256_fmsub_pd(a, b, c);
}
inline auto fnmsub(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm256_fnmsub_ps(a, b, c);
}
inline auto fnmsub(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm256_fnmsub_pd(a, b, c);
}

// Non-specific
template<typename T, bool Conj>
inline auto add(cx_reg<T, Conj> lhs, cx_reg<T, Conj> rhs) -> cx_reg<T, Conj> {
    return {add(lhs.real, rhs.real), add(lhs.imag, rhs.imag)};
}
template<typename T>
inline auto add(cx_reg<T, true> lhs, cx_reg<T, false> rhs) -> cx_reg<T, false> {
    return {add(lhs.real, rhs.real), sub(rhs.imag, lhs.imag)};
}
template<typename T>
inline auto add(cx_reg<T, false> lhs, cx_reg<T, true> rhs) -> cx_reg<T, false> {
    return {add(lhs.real, rhs.real), sub(lhs.imag, rhs.imag)};
}

template<typename T>
inline auto sub(cx_reg<T, false> lhs, cx_reg<T, false> rhs) -> cx_reg<T, false> {
    return {sub(lhs.real, rhs.real), sub(lhs.imag, rhs.imag)};
}
template<typename T>
inline auto sub(cx_reg<T, true> lhs, cx_reg<T, false> rhs) -> cx_reg<T, true> {
    return {sub(lhs.real, rhs.real), add(lhs.imag, rhs.imag)};
}
template<typename T>
inline auto sub(cx_reg<T, false> lhs, cx_reg<T, true> rhs) -> cx_reg<T, false> {
    return {sub(lhs.real, rhs.real), add(lhs.imag, rhs.imag)};
}
template<typename T>
inline auto sub(cx_reg<T, true> lhs, cx_reg<T, true> rhs) -> cx_reg<T, false> {
    return {sub(lhs.real, rhs.real), sub(rhs.imag, lhs.imag)};
}

template<typename T>
inline auto mul(cx_reg<T, false> lhs, cx_reg<T, false> rhs) -> cx_reg<T, false> {
    auto real = mul(lhs.real, rhs.real);
    auto imag = mul(lhs.real, rhs.imag);
    return {fnmadd(lhs.imag, rhs.imag, real), fmadd(lhs.imag, rhs.real, imag)};
}
template<typename T>
inline auto mul(cx_reg<T, true> lhs, cx_reg<T, false> rhs) -> cx_reg<T, false> {
    auto real = mul(lhs.real, rhs.real);
    auto imag = mul(lhs.real, rhs.imag);
    return {fmadd(lhs.imag, rhs.imag, real), fnmadd(lhs.imag, rhs.real, imag)};
}
template<typename T>
inline auto mul(cx_reg<T, false> lhs, cx_reg<T, true> rhs) -> cx_reg<T, false> {
    auto real = mul(lhs.real, rhs.real);
    auto imag = mul(lhs.real, rhs.imag);
    return {fmadd(lhs.imag, rhs.imag, real), fmsub(lhs.imag, rhs.real, imag)};
}
template<typename T>
inline auto mul(cx_reg<T, true> lhs, cx_reg<T, true> rhs) -> cx_reg<T, false> {
    auto real = mul(lhs.real, rhs.real);
    auto imag = mul(lhs.real, rhs.imag);
    return {fnmadd(lhs.imag, rhs.imag, real), fnmsub(lhs.imag, rhs.real, imag)};
}

template<typename T>
inline auto div(cx_reg<T, false> lhs, cx_reg<T, false> rhs) -> cx_reg<T, false> {
    auto rhs_abs = mul(rhs.real, rhs.real);
    auto real_   = mul(lhs.real, rhs.real);
    auto imag_   = mul(lhs.real, rhs.imag);

    rhs_abs = fmadd(rhs.imag, rhs.imag, rhs_abs);
    real_   = fmadd(lhs.imag, rhs.imag, real_);
    imag_   = fmsub(lhs.imag, rhs.real, imag_);

    return {div(real_, rhs_abs), div(imag_, rhs_abs)};
}
template<typename T>
inline auto div(cx_reg<T, true> lhs, cx_reg<T, false> rhs) -> cx_reg<T, false> {
    auto rhs_abs = mul(rhs.real, rhs.real);
    auto real_   = mul(lhs.real, rhs.real);
    auto imag_   = mul(lhs.real, rhs.imag);

    rhs_abs = fmadd(rhs.imag, rhs.imag, rhs_abs);
    real_   = fnmadd(lhs.imag, rhs.imag, real_);
    imag_   = fnmsub(lhs.imag, rhs.real, imag_);

    return {div(real_, rhs_abs), div(imag_, rhs_abs)};
}
template<typename T>
inline auto div(cx_reg<T, false> lhs, cx_reg<T, true> rhs) -> cx_reg<T, false> {
    auto rhs_abs = mul(rhs.real, rhs.real);
    auto real_   = mul(lhs.real, rhs.real);
    auto imag_   = mul(lhs.real, rhs.imag);

    rhs_abs = fmadd(rhs.imag, rhs.imag, rhs_abs);
    real_   = fnmadd(lhs.imag, rhs.imag, real_);
    imag_   = fmadd(lhs.imag, rhs.real, imag_);

    return {div(real_, rhs_abs), div(imag_, rhs_abs)};
}
template<typename T>
inline auto div(cx_reg<T, true> lhs, cx_reg<T, true> rhs) -> cx_reg<T, false> {
    auto rhs_abs = mul(rhs.real, rhs.real);
    auto real_   = mul(lhs.real, rhs.real);
    auto imag_   = mul(lhs.real, rhs.imag);

    rhs_abs = fmadd(rhs.imag, rhs.imag, rhs_abs);
    real_   = fmadd(lhs.imag, rhs.imag, real_);
    imag_   = fnmadd(lhs.imag, rhs.real, imag_);

    return {div(real_, rhs_abs), div(imag_, rhs_abs)};
}

template<typename T, bool Conj>
inline auto add(reg_t<T> lhs, cx_reg<T, Conj> rhs) -> cx_reg<T, Conj> {
    return {add(lhs, rhs.real), rhs.imag};
}
template<typename T, bool Conj>
inline auto sub(reg_t<T> lhs, cx_reg<T, Conj> rhs) -> cx_reg<T, false> {
    if constexpr (Conj) {
        return {sub(lhs, rhs.real), rhs.imag};
    } else {
        auto zero = setzero<T>();
        return {sub(lhs, rhs.real), sub(zero, rhs.imag)};
    }
}
template<typename T, bool Conj>
inline auto mul(reg_t<T> lhs, cx_reg<T, Conj> rhs) -> cx_reg<T, Conj> {
    return {mul(lhs, rhs.real), mul(lhs, rhs.imag)};
}
template<typename T, bool Conj>
inline auto div(reg_t<T> lhs, cx_reg<T, Conj> rhs) -> cx_reg<T, false> {
    auto     rhs_abs = mul(rhs.real, rhs.real);
    auto     real_   = mul(lhs, rhs.real);
    reg_t<T> imag_;
    if constexpr (Conj) {
        imag_ = mul(lhs, rhs.imag);
    } else {
        auto zero = setzero<T>();
        imag_ = mul(lhs, sub(zero, rhs.imag));
    }
    rhs_abs = fmadd(rhs.imag, rhs.imag, rhs_abs);

    return {div(real_, rhs_abs), div(imag_, rhs_abs)};
}

template<typename T, bool Conj>
inline auto add(cx_reg<T, Conj> lhs, reg_t<T> rhs) -> cx_reg<T, Conj> {
    return {add(lhs.real, rhs), lhs.imag};
}
template<typename T, bool Conj>
inline auto sub(cx_reg<T, Conj> lhs, reg_t<T> rhs) -> cx_reg<T, Conj> {
    return {sub(lhs.real, rhs), lhs.imag};
}
template<typename T, bool Conj>
inline auto mul(cx_reg<T, Conj> lhs, reg_t<T> rhs) -> cx_reg<T, Conj> {
    return {mul(lhs.real, rhs), mul(lhs.imag, rhs)};
}
template<typename T, bool Conj>
inline auto div(cx_reg<T, Conj> lhs, reg_t<T> rhs) -> cx_reg<T, Conj> {
    return {div(lhs.real, rhs), div(lhs.imag, rhs)};
}


}    // namespace pcx::simd

#endif