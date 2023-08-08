#ifndef SIMD_BASE_HPP
#define SIMD_BASE_HPP

#include "types.hpp"

#include <complex>

namespace pcx::simd {

template<typename T>
inline auto setzero() -> reg_t<T>;
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
inline auto inverse(auto... args);

// Arithmetic

template<typename T>
inline auto add(reg_t<T> lhs, reg_t<T> rhs) -> reg_t<T>;
template<typename T>
inline auto sub(reg_t<T> lhs, reg_t<T> rhs) -> reg_t<T>;
template<typename T>
inline auto mul(reg_t<T> lhs, reg_t<T> rhs) -> reg_t<T>;
template<typename T>
inline auto div(reg_t<T> lhs, reg_t<T> rhs) -> reg_t<T>;

template<typename T>
inline auto fmadd(reg_t<T> lhs, reg_t<T> rhs) -> reg_t<T>;
template<typename T>
inline auto fmsub(reg_t<T> lhs, reg_t<T> rhs) -> reg_t<T>;
template<typename T>
inline auto fnmadd(reg_t<T> lhs, reg_t<T> rhs) -> reg_t<T>;
template<typename T>
inline auto fnmsub(reg_t<T> lhs, reg_t<T> rhs) -> reg_t<T>;

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


}    // namespace pcx::simd
#endif