#ifndef SIMD_COMMON_HPP
#define SIMD_COMMON_HPP

#include "tuple_util.hpp"
#include "types.hpp"

#include <complex>

namespace pcx::simd {

/**
 * @brief Overload for architecture specific intrinsic simd type must be provided.
 */
template<typename T>
struct simd_traits {
    static constexpr uZ size = 0;
};

template<typename T>
concept simd_vector = simd_traits<T>::size != 0;

template<typename T>
inline auto zero() -> reg_t<T>;
template<typename T>
inline auto broadcast(const T* source) -> reg_t<T>;
template<typename T>
inline auto broadcast(T source) -> reg_t<T>;
template<typename T>
inline auto load(const T* source) -> reg_t<T>;
template<typename T>
inline void store(T* dest, reg_t<T> reg);

template<typename T>
inline auto broadcast(std::complex<T> source) -> cx_reg<T, false>;
template<uZ PackSize, typename T>
inline auto cxload(const T* ptr) -> cx_reg<T, false>;
template<uZ PackSize, typename T, bool Conj>
inline void cxstore(T* ptr, cx_reg<T, Conj> reg);

/**
 * @param args Variable number of complex simd vectors.
 * @return Tuple of repacked complex simd vectors in the order of passing.
 */
template<uZ PackFrom, uZ PackTo>
inline auto repack(auto... args);

/**
 * @brief Conditionaly swaps real and imaginary parts of complex simd vectors.
 *
 * @tparam Inverse If true performs swap.
 * @param args Variable number of complex simd vectors.
 * @return Tuple of invrsed simd complex vectors.
 */
template<bool Inverse>
inline auto inverse(auto... args) {
    auto tup = std::make_tuple(args...);
    if constexpr (Inverse) {
        auto inverse = [](auto reg) {
            using reg_t = decltype(reg);
            return reg_t{reg.imag, reg.real};
        };
        return detail_::apply_for_each(inverse, tup);
    } else {
        return tup;
    }
};

/*
 * Simd specific arithmetic declarations.
 * Declaring with templates because simd vector register is
 * not defined at this point yet.
 * Architecture specific definitions must be defined.
 */

/**/
template<simd_vector Reg>
inline auto add(Reg lhs, Reg rhs) -> Reg;
template<simd_vector Reg>
inline auto sub(Reg lhs, Reg rhs) -> Reg;
template<simd_vector Reg>
inline auto mul(Reg lhs, Reg rhs) -> Reg;
template<simd_vector Reg>
inline auto div(Reg lhs, Reg rhs) -> Reg;

template<simd_vector Reg>
inline auto fmadd(Reg a, Reg b, Reg c) -> Reg;
template<simd_vector Reg>
inline auto fmsub(Reg a, Reg b, Reg c) -> Reg;
template<simd_vector Reg>
inline auto fnmadd(Reg a, Reg b, Reg c) -> Reg;
template<simd_vector Reg>
inline auto fnmsub(Reg a, Reg b, Reg c) -> Reg;

/*
 * Implementation provided below.
 */
/**/
template<typename T, bool ConjLhs, bool ConjRhs>
inline auto add(cx_reg<T, ConjLhs> lhs, cx_reg<T, ConjRhs> rhs);
template<typename T, bool ConjLhs, bool ConjRhs>
inline auto sub(cx_reg<T, ConjLhs> lhs, cx_reg<T, ConjRhs> rhs);
template<typename T, bool ConjLhs, bool ConjRhs>
inline auto mul(cx_reg<T, ConjLhs> lhs, cx_reg<T, ConjRhs> rhs);
template<typename T, bool ConjLhs, bool ConjRhs>
inline auto div(cx_reg<T, ConjLhs> lhs, cx_reg<T, ConjRhs> rhs);

/**
 * @brief Preforms multiple complex multiplications.
 * Arguments are multiplied in pairs in the order of passing.
 * Complex multiplication can be viewed as two stages:
 * 1. two multiplications;
 * 2. addition;
 * This function explicitly reorders stages of multiple multiplications
 * to reduce possible latency effect on performance.
 */
template<typename T, bool... Conj>
    requires(sizeof...(Conj) % 2 == 0)
inline auto mul_pairs(cx_reg<T, Conj>... args);

template<typename T, bool... Conj>
    requires(sizeof...(Conj) % 2 == 0)
inline auto div_pairs(cx_reg<T, Conj>... args);

// Imlpementations

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

namespace detail_ {

template<typename T, bool ConjLhs, bool ConjRhs>
inline auto mul_real_rhs(cx_reg<T, ConjLhs> lhs, cx_reg<T, ConjRhs> rhs) -> cx_reg<T, false> {
    return {mul(lhs.real, rhs.real), mul(lhs.real, rhs.imag)};
};
template<typename T>
inline auto mul_imag_rhs(cx_reg<T, false> prod_real_rhs, cx_reg<T, false> lhs, cx_reg<T, false> rhs)
    -> cx_reg<T, false> {
    return {fnmadd(lhs.imag, rhs.imag, prod_real_rhs.real), fmadd(lhs.imag, rhs.real, prod_real_rhs.imag)};
}
template<typename T>
inline auto mul_imag_rhs(cx_reg<T, false> prod_real_rhs, cx_reg<T, true> lhs, cx_reg<T, false> rhs)
    -> cx_reg<T, false> {
    return {fmadd(lhs.imag, rhs.imag, prod_real_rhs.real), fnmadd(lhs.imag, rhs.real, prod_real_rhs.imag)};
}
template<typename T>
inline auto mul_imag_rhs(cx_reg<T, false> prod_real_rhs, cx_reg<T, false> lhs, cx_reg<T, true> rhs)
    -> cx_reg<T, false> {
    return {fmadd(lhs.imag, rhs.imag, prod_real_rhs.real), fmsub(lhs.imag, rhs.real, prod_real_rhs.imag)};
}
template<typename T>
inline auto mul_imag_rhs(cx_reg<T, false> prod_real_rhs, cx_reg<T, true> lhs, cx_reg<T, true> rhs)
    -> cx_reg<T, false> {
    return {fnmadd(lhs.imag, rhs.imag, prod_real_rhs.real), fnmsub(lhs.imag, rhs.real, prod_real_rhs.imag)};
}
}    // namespace detail_

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
        return {sub(lhs, rhs.real), sub(zero<T>(), rhs.imag)};
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
        imag_ = mul(lhs, sub(zero<T>(), rhs.imag));
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