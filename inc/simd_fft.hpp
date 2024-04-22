#ifndef SIMD_FFT_HPP
#define SIMD_FFT_HPP

#include "simd_common.hpp"
#include "types.hpp"

#define _AINLINE_ [[gnu::always_inline, clang::always_inline]]

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
  * @tparam RhsRotI Number of multiplications by imaginary unity
  */
template<uZ RhsRotI = 0, uZ PackSize, typename T>
    requires(RhsRotI < 4)
_AINLINE_ auto ibtfly(cx_reg<T, false, PackSize> lhs, cx_reg<T, false, PackSize> rhs) {
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
  * @tparam RhsRotI Number of multiplications by imaginary unity
  */
template<uint RhsRotI = 0, uZ PackSize, typename T>
    requires(RhsRotI < 4)
_AINLINE_ auto btfly(cx_reg<T, false, PackSize> lhs, cx_reg<T, false, PackSize> rhs) {
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

/* Collection of functions that perfrom fixed size FFT.
FFT size is determined by hardware simd size.
The amount of loads is minimized.
*/
namespace size_specific {

template<typename T>
inline constexpr uZ unsorted_size = 0;

/**
 * @brief Performs usorted DIF FFT over a segment of data.
    FFT is performed in blocks of `unsorted_size` length.
    Each block is loaded a single time.
    If not bit-reversed the order is arbitrary, but equal to input order of `unsorted_reverse()`

 * @tparam PDest        Pack size of the data after transform.
 * @tparam PTform       Pack size of the data input.
 * @tparam BitReversed  If true, the data after transform is in bit-reversed order. Otherwise the order is arbitrary.
 * @param dest          Pointer to the data segment.
 * @param twiddle_ptr   Pointer to the first twiddle of this transform.
 * @param size          Total size of the data segment.
 * @return const float* Pointer to the first twiddle after this transform.
 */
template<uZ PDest, uZ PTform, bool BitReversed>
inline auto unsorted(float* dest, const float* twiddle_ptr, uZ size) -> const float*;
template<uZ PDest, uZ PTform, bool BitReversed>
inline auto unsorted(double* dest, const double* twiddle_ptr, uZ size) -> const double*;

/**
 * @brief Performes DIF IFFT over a segment of data.
    IFFT is performed in blocks of `unsorted_size` length.
    Each block is loaded a single time.
    If not bit-reversed the order is arbitrary, but equal to output order of `unsorted()`

 * @tparam PTform       Pack size of the data after transform.
 * @tparam PSrc         Pack size of the data input.
 * @tparam Scale        If true performs normalized IFFT.
 * @tparam BitReversed  If true the input is in bit-reversed order. Otherwise the order is arbitrary.
 * @param dest          Pointer to the output data segment. 
 * @param twiddle_ptr   Pointer to the first twiddle after corresponding forward transform.
 * @param size          Total size of the data segment.
 * @param fft_size      Total size of IFFT.
 * @param optional      Optionally containts `const float* source`, pointing to the data source.
    Otherwise `dest` is the pointer to the input.
 * @return const float* 
 */
template<uZ PTform, uZ PSrc, bool Scale, bool BitReversed>
inline auto unsorted_reverse(float*       dest,    //
                             const float* twiddle_ptr,
                             uZ           size,
                             uZ           fft_size,
                             auto... optional) -> const float*;
template<uZ PTform, uZ PSrc, bool Scale, bool BitReversed>
inline auto unsorted_reverse(double*       dest,    //
                             const double* twiddle_ptr,
                             uZ            size,
                             uZ            fft_size,
                             auto... optional) -> const double*;

/**
 * @brief Insert a group of twiddle blocks into storage range.
    A group consist of `n_blocks` blocks.
    Each twiddle block corresponds to a single `unsorted()` and `unsorted_reverse()` block.
    l_size is equal to total FFT size divided by `unsorted_size`.

 * @tparam T        float.
 * @param twiddles  Twiddle storage range.
 * @param n_blocks  Number of twiddle blocks to insert.     
 * @param l_size    Size of FFT iteration of the first level.
 * @param i_group   Group index from 0 to `fft_size`/(`unsorted_size`*`n_blocks`).
 */
template<typename T>
inline void insert_unsorted(auto& twiddles, uZ n_blocks, uZ l_size, uZ i_group);

template<typename T>
static constexpr uZ sorted_size = 0;

template<uZ PTform, uZ PSrc, bool Inverse, floating_point T>
static inline void tform_sort(T* data, uZ size, const auto& sort);

template<uZ PTform, uZ PSrc, bool Inverse, floating_point T>
static inline void tform_sort(T* dest, const T* source, uZ size, const auto& sort);

}    // namespace size_specific
}    // namespace pcx::simd

#undef _AINLINE_

#endif
