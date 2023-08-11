#ifndef SIMD_FFT_HPP
#define SIMD_FFT_HPP


#include "avx2_common.hpp"
namespace pcx::detail_ {
template<typename T, typename... U>
concept has_type = (std::same_as<T, U> || ...);

namespace fft {
constexpr auto log2i(u64 num) -> uZ;
constexpr auto reverse_bit_order(u64 num, u64 depth) -> u64;
constexpr auto n_reversals(uZ max) -> uZ;
template<typename T>
inline auto wnk(std::size_t n, std::size_t k) -> std::complex<T>;
}    // namespace fft
}    // namespace pcx::detail_

namespace pcx::simd {

/**
  * @brief Performs butterfly operation, then multiplies diff by imaginary unit RhsRotI times;
  *
  * @tparam RhsRotI number of multiplications by imaginary unity
  */
template<uint RhsRotI = 0, uZ PackSize, typename T>
    requires(RhsRotI < 4)
inline auto ibtfly(cx_reg<T, false, PackSize> lhs, cx_reg<T, false, PackSize> rhs) {
    cx_reg<T, false, PackSize> s;
    cx_reg<T, false, PackSize> d;
    if constexpr (RhsRotI == 0) {
        auto s_re = add(lhs.real, rhs.real);
        auto d_re = sub(lhs.real, rhs.real);
        auto s_im = add(lhs.imag, rhs.imag);
        auto d_im = sub(lhs.imag, rhs.imag);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    } else if constexpr (RhsRotI == 1) {
        auto s_re = add(lhs.real, rhs.real);
        auto d_im = sub(lhs.real, rhs.real);
        auto s_im = add(lhs.imag, rhs.imag);
        auto d_re = sub(rhs.imag, lhs.imag);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    } else if constexpr (RhsRotI == 2) {
        auto s_re = add(lhs.real, rhs.real);
        auto d_re = sub(rhs.real, lhs.real);
        auto s_im = add(lhs.imag, rhs.imag);
        auto d_im = sub(rhs.imag, lhs.imag);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    } else {
        auto s_re = add(lhs.real, rhs.real);
        auto d_im = sub(rhs.real, lhs.real);
        auto s_im = add(lhs.imag, rhs.imag);
        auto d_re = sub(lhs.imag, rhs.imag);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    }

    return std::make_tuple(s, d);
}

/**
  * @brief Multiplies rhs by imaginary unit RhsRotI times, then performs butterfly operation;
  *
  * @tparam RhsRotI number of multiplications by imaginary unity
  */
template<uint RhsRotI = 0, uZ PackSize, typename T>
    requires(RhsRotI < 4)
inline auto btfly(cx_reg<T, false, PackSize> lhs, cx_reg<T, false, PackSize> rhs) {
    cx_reg<T, false, PackSize> s;
    cx_reg<T, false, PackSize> d;
    if constexpr (RhsRotI == 0) {
        auto s_re = add(lhs.real, rhs.real);
        auto d_re = sub(lhs.real, rhs.real);
        auto s_im = add(lhs.imag, rhs.imag);
        auto d_im = sub(lhs.imag, rhs.imag);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    } else if constexpr (RhsRotI == 1) {
        auto s_re = sub(lhs.real, rhs.imag);
        auto d_re = add(lhs.real, rhs.imag);
        auto s_im = add(lhs.imag, rhs.real);
        auto d_im = sub(lhs.imag, rhs.real);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    } else if constexpr (RhsRotI == 2) {
        auto s_re = sub(lhs.real, rhs.real);
        auto d_re = add(lhs.real, rhs.real);
        auto s_im = sub(lhs.imag, rhs.imag);
        auto d_im = add(lhs.imag, rhs.imag);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    } else {
        auto s_re = add(lhs.real, rhs.imag);
        auto d_re = sub(lhs.real, rhs.imag);
        auto s_im = sub(lhs.imag, rhs.real);
        auto d_im = add(lhs.imag, rhs.real);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    }
    return std::make_tuple(s, d);
}

}    // namespace pcx::simd
#endif