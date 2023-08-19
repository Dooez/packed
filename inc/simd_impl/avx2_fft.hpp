#ifndef AVX2_FFT_HPP
#define AVX2_FFT_HPP

#include "simd_fft.hpp"

namespace pcx::simd {

constexpr auto swap_48 = [](cx_reg<float, false> reg) {
    auto real = avx2::unpacklo_128(reg.real, reg.imag);
    auto imag = avx2::unpackhi_128(reg.real, reg.imag);
    return cx_reg<float, false>({real, imag});
};

namespace detail_ {
template<uZ N, uZ M, uZ... L>
struct equal_ {
    static constexpr bool value = (N == M) && (sizeof...(L) < 2 || equal_<M, L...>::value);
};
template<uZ N, uZ M>
struct equal_<N, M> {
    static constexpr bool value = N == M;
};
template<uZ... I>
concept equal = (sizeof...(I) < 2 || (equal_<I...>::value));

}    // namespace detail_

namespace avx2 {
/**
 * @brief Shuffles data to put I/Q into separate simd registers.
 * Resulting data order is dependent on input pack size.
 * if PackFrom < 4 the order of values across 2 registers changes
 * [0 1 2 3][4 5 6 7] -> [0 1 4 5][2 3 6 7]
 * Faster than true repack.
 */
template<uZ... PackSize>
    requires detail_::equal<PackSize...>
static inline auto split(cx_reg<float, false, PackSize>... args) {
    constexpr auto swap_48 = []<uZ PackSize_>(cx_reg<float, false, PackSize_> reg) {
        auto real = avx2::unpacklo_128(reg.real, reg.imag);
        auto imag = avx2::unpackhi_128(reg.real, reg.imag);
        return cx_reg<float, false, reg_size<float>>({real, imag});
    };
    auto tup = std::make_tuple(args...);
    if constexpr (((PackSize == 1) && ...)) {
        auto split = []<bool Conj, uZ PackSize_>(cx_reg<float, Conj, PackSize_> reg) {
            auto real = _mm256_shuffle_ps(reg.real, reg.imag, 0b10001000);
            auto imag = _mm256_shuffle_ps(reg.real, reg.imag, 0b11011101);
            return cx_reg<float, Conj, reg_size<float>>({real, imag});
        };
        return pcx::detail_::apply_for_each(split, tup);
    } else if constexpr (((PackSize == 2) && ...)) {
        auto split = []<bool Conj, uZ PackSize_>(cx_reg<float, Conj, PackSize_> reg) {
            auto real = avx2::unpacklo_pd(reg.real, reg.imag);
            auto imag = avx2::unpackhi_pd(reg.real, reg.imag);
            return cx_reg<float, Conj, reg_size<float>>({real, imag});
        };
        return pcx::detail_::apply_for_each(split, tup);
    } else if constexpr (((PackSize == 4) && ...)) {
        return pcx::detail_::apply_for_each(swap_48, tup);
    } else {
        return tup;
    }
}
template<uZ PackTo, uZ... PackSize>
    requires detail_::equal<PackSize...>
static inline auto combine(cx_reg<float, false, PackSize>... args) {
    constexpr auto swap_48 = []<uZ PackSize_>(cx_reg<float, false, PackSize_> reg) {
        auto real = avx2::unpacklo_128(reg.real, reg.imag);
        auto imag = avx2::unpackhi_128(reg.real, reg.imag);
        return cx_reg<float, false, PackTo>({real, imag});
    };
    auto tup = std::make_tuple(args...);
    if constexpr (PackTo == 1) {
        auto combine = []<bool Conj, uZ PackSize_>(cx_reg<float, Conj, PackSize_> reg) {
            auto real = simd::avx2::unpacklo_ps(reg.real, reg.imag);
            auto imag = simd::avx2::unpackhi_ps(reg.real, reg.imag);
            return cx_reg<float, Conj, PackTo>({real, imag});
        };
        return pcx::detail_::apply_for_each(combine, tup);
    } else if constexpr (PackTo == 2) {
        auto combine = []<bool Conj, uZ PackSize_>(cx_reg<float, Conj, PackSize_> reg) {
            auto real = avx2::unpacklo_pd(reg.real, reg.imag);
            auto imag = avx2::unpackhi_pd(reg.real, reg.imag);
            return cx_reg<float, Conj, PackTo>({real, imag});
        };
        return pcx::detail_::apply_for_each(combine, tup);
    } else if constexpr (PackTo == 4) {
        return pcx::detail_::apply_for_each(swap_48, tup);
    } else {
        return tup;
    }
}
}    // namespace avx2
// NOLINTBEGIN(*pointer-arithmetic*)

/**
 * @brief Contains functions specific to data type and simd sizes
 *
 * @tparam SimdSize number of type T values in simd registers in, e.g. 4 for SSE (128 bits) and float (32 bits)
 * @tparam SimdCount number of simd registers
 */
struct size_specific {
    template<typename T>
    static constexpr uZ unsorted_size = 0;
    template<>
    static constexpr uZ unsorted_size<float> = 32;
    template<>
    static constexpr uZ unsorted_size<double> = 16;

    template<typename T>
    static constexpr uZ sorted_size = 0;
    template<>
    static constexpr uZ sorted_size<float> = 8;
    template<>
    static constexpr uZ sorted_size<double> = 4;

    template<uZ PDest, uZ PTform, bool BitReversed>
    static inline auto unsorted(float* dest, const float* twiddle_ptr, uZ size) {
        constexpr auto reg_size = reg<float>::size;
        using cx_reg            = cx_reg<float, false, reg_size>;

        for (uZ i_group = 0; i_group < size / unsorted_size<float>; ++i_group) {
            cx_reg tw0 = {broadcast(twiddle_ptr++), broadcast(twiddle_ptr++)};

            cx_reg tw1 = {broadcast(twiddle_ptr++), broadcast(twiddle_ptr++)};
            cx_reg tw2 = {broadcast(twiddle_ptr++), broadcast(twiddle_ptr++)};

            auto* ptr0 = ra_addr<PTform>(dest, reg_size * (i_group * 4));
            auto* ptr1 = ra_addr<PTform>(dest, reg_size * (i_group * 4 + 1));
            auto* ptr2 = ra_addr<PTform>(dest, reg_size * (i_group * 4 + 2));
            auto* ptr3 = ra_addr<PTform>(dest, reg_size * (i_group * 4 + 3));

            auto p2 = cxload<PTform>(ptr2);
            auto p3 = cxload<PTform>(ptr3);
            auto p0 = cxload<PTform>(ptr0);
            auto p1 = cxload<PTform>(ptr1);

            auto [p2tw, p3tw] = mul_pairs(p2, tw0, p3, tw0);

            auto [a1, a3] = btfly(p1, p3tw);
            auto [a0, a2] = btfly(p0, p2tw);

            auto [a1tw, a3tw] = mul_pairs(a1, tw1, a3, tw2);

            auto tw3 = cxload<reg_size>(twiddle_ptr);
            auto tw4 = cxload<reg_size>(twiddle_ptr + reg_size * 2);
            twiddle_ptr += reg_size * 4;

            auto [b0, b1] = btfly(a0, a1tw);
            auto [b2, b3] = btfly(a2, a3tw);

            auto [shb0, shb1] = avx2::unpack_128(b0, b1);
            auto [shb2, shb3] = avx2::unpack_128(b2, b3);

            auto [shb1tw, shb3tw] = mul_pairs(shb1, tw3, shb3, tw4);

            auto tw5 = cxload<reg_size>(twiddle_ptr);
            auto tw6 = cxload<reg_size>(twiddle_ptr + reg_size * 2);
            twiddle_ptr += reg_size * 4;

            auto [c0, c1] = btfly(shb0, shb1tw);
            auto [c2, c3] = btfly(shb2, shb3tw);

            auto [shc0, shc1] = avx2::unpack_pd(c0, c1);
            auto [shc2, shc3] = avx2::unpack_pd(c2, c3);

            auto [shc1tw, shc3tw] = mul_pairs(shc1, tw5, shc3, tw6);

            auto tw7 = cxload<reg_size>(twiddle_ptr);
            auto tw8 = cxload<reg_size>(twiddle_ptr + reg_size * 2);
            twiddle_ptr += reg_size * 4;

            auto [d0, d1] = btfly(shc0, shc1tw);
            auto [d2, d3] = btfly(shc2, shc3tw);

            auto shuf = [](cx_reg lhs, cx_reg rhs) {
                auto lhs_re = _mm256_shuffle_ps(lhs.real, rhs.real, 0b10001000);
                auto rhs_re = _mm256_shuffle_ps(lhs.real, rhs.real, 0b11011101);
                auto lhs_im = _mm256_shuffle_ps(lhs.imag, rhs.imag, 0b10001000);
                auto rhs_im = _mm256_shuffle_ps(lhs.imag, rhs.imag, 0b11011101);
                return std::make_tuple(cx_reg{lhs_re, lhs_im}, cx_reg{rhs_re, rhs_im});
            };
            auto [shd0, shd1] = shuf(d0, d1);
            auto [shd2, shd3] = shuf(d2, d3);

            auto [shd1tw, shd3tw] = mul_pairs(shd1, tw7, shd3, tw8);

            auto [e0, e1] = btfly(shd0, shd1tw);
            auto [e2, e3] = btfly(shd2, shd3tw);

            if constexpr (BitReversed) {
                cx_reg she0_, she1_, she2_, she3_;    //NOLINT (*declaration*)
                if constexpr (PDest < 4) {
                    std::tie(she0_, she1_) = avx2::unpack_ps(e0, e1);
                    std::tie(she2_, she3_) = avx2::unpack_ps(e2, e3);
                    std::tie(she0_, she1_) = avx2::unpack_128(she0_, she1_);
                    std::tie(she2_, she3_) = avx2::unpack_128(she2_, she3_);
                } else {
                    std::tie(she0_, she1_) = avx2::unpack_ps(e0, e1);
                    std::tie(she2_, she3_) = avx2::unpack_ps(e2, e3);
                    std::tie(she0_, she1_) = avx2::unpack_pd(she0_, she1_);
                    std::tie(she2_, she3_) = avx2::unpack_pd(she2_, she3_);
                    std::tie(she0_, she1_) = avx2::unpack_128(she0_, she1_);
                    std::tie(she2_, she3_) = avx2::unpack_128(she2_, she3_);
                }
                auto [she0, she1, she2, she3] = avx2::combine<PDest>(she0_, she1_, she2_, she3_);
                cxstore<PDest>(ptr0, she0);
                cxstore<PDest>(ptr1, she1);
                cxstore<PDest>(ptr2, she2);
                cxstore<PDest>(ptr3, she3);
            } else {
                constexpr auto PStore = std::max(PDest, reg_size);
                cxstore<PStore>(ptr0, e0);
                cxstore<PStore>(ptr1, e1);
                cxstore<PStore>(ptr2, e2);
                cxstore<PStore>(ptr3, e3);
            }
        }
        return twiddle_ptr;
    }

    template<uZ PTform, uZ PSrc, bool Scale, bool BitReversed>
    static inline auto unsorted_reverse(float*       dest,    //
                                        const float* twiddle_ptr,
                                        uZ           size,
                                        uZ           fft_size,
                                        auto... optional) {
        constexpr auto reg_size = reg<float>::size;
        using cx_reg            = cx_reg<float, false, reg_size>;
        constexpr auto PLoad    = BitReversed ? PSrc : std::max(PSrc, reg_size);

        using source_type  = const float*;
        constexpr bool Src = pcx::detail_::has_type<source_type, decltype(optional)...>;

        auto scale = [](uZ size) {
            if constexpr (Scale) {
                return static_cast<float>(1. / static_cast<double>(size));
            } else {
                return [] {};
            }
        }(fft_size);

        for (iZ i_group = static_cast<iZ>(size / unsorted_size<float> - 1); i_group >= 0; --i_group) {
            auto* ptr0 = ra_addr<PTform>(dest, reg_size * (i_group * 4));
            auto* ptr1 = ra_addr<PTform>(dest, reg_size * (i_group * 4 + 1));
            auto* ptr2 = ra_addr<PTform>(dest, reg_size * (i_group * 4 + 2));
            auto* ptr3 = ra_addr<PTform>(dest, reg_size * (i_group * 4 + 3));

            simd::cx_reg<float, false, PLoad> she0_, she1_, she2_, she3_;    // NOLINT (*declaration*)
            if constexpr (Src) {
                auto src = std::get<source_type&>(std::tie(optional...));

                she0_ = cxload<PLoad>(ra_addr<PLoad>(src, reg_size * (i_group * 4)));
                she1_ = cxload<PLoad>(ra_addr<PLoad>(src, reg_size * (i_group * 4 + 1)));
                she2_ = cxload<PLoad>(ra_addr<PLoad>(src, reg_size * (i_group * 4 + 2)));
                she3_ = cxload<PLoad>(ra_addr<PLoad>(src, reg_size * (i_group * 4 + 3)));
            } else {
                she0_ = cxload<PLoad>(ptr0);
                she1_ = cxload<PLoad>(ptr1);
                she2_ = cxload<PLoad>(ptr2);
                she3_ = cxload<PLoad>(ptr3);
            }

            cx_reg e0, e1, e2, e3;    // NOLINT (*declaration*)

            auto [she0, she1, she2, she3]    = repack2<reg_size>(she0_, she1_, she2_, she3_);
            std::tie(she0, she1, she2, she3) = inverse<true>(she0, she1, she2, she3);

            if constexpr (BitReversed) {
                std::tie(e0, e1) = avx2::unpack_128(she0, she1);
                std::tie(e2, e3) = avx2::unpack_128(she2, she3);
            } else {
                std::tie(e0, e1, e2, e3) = std::tie(she0, she1, she2, she3);
            }

            twiddle_ptr -= reg_size * 4;
            auto tw7 = cxload<reg_size>(twiddle_ptr);
            auto tw8 = cxload<reg_size>(twiddle_ptr + reg_size * 2);

            if constexpr (BitReversed) {
                std::tie(e0, e1) = avx2::unpack_ps(e0, e1);
                std::tie(e2, e3) = avx2::unpack_ps(e2, e3);
                std::tie(e0, e1) = avx2::unpack_pd(e0, e1);
                std::tie(e2, e3) = avx2::unpack_pd(e2, e3);
            }
            auto [shd0, shd1tw] = ibtfly(e0, e1);
            auto [shd2, shd3tw] = ibtfly(e2, e3);

            auto [shd1, shd3] = mul_pairs(shd1tw, tw7, shd3tw, tw8);

            twiddle_ptr -= reg_size * 4;
            auto tw5 = cxload<reg_size>(twiddle_ptr);
            auto tw6 = cxload<reg_size>(twiddle_ptr + reg_size * 2);

            auto [d0, d1] = avx2::unpack_ps(shd0, shd1);
            auto [d2, d3] = avx2::unpack_ps(shd2, shd3);

            auto [shc0, shc1tw] = btfly(d0, d1);
            auto [shc2, shc3tw] = btfly(d2, d3);

            auto [shc1, shc3] = mul_pairs(shc1tw, tw5, shc3tw, tw6);

            twiddle_ptr -= reg_size * 4;
            auto tw3 = cxload<reg_size>(twiddle_ptr);
            auto tw4 = cxload<reg_size>(twiddle_ptr + reg_size * 2);

            auto [c0, c1] = avx2::unpack_pd(shc0, shc1);
            auto [c2, c3] = avx2::unpack_pd(shc2, shc3);

            auto [shb0, shb1tw] = btfly(c0, c1);
            auto [shb2, shb3tw] = btfly(c2, c3);

            auto [shb1, shb3] = mul_pairs(shb1tw, tw3, shb3tw, tw4);

            twiddle_ptr -= 6;    //NOLINT (*magic-numbers*)
            cx_reg tw1 = {broadcast(twiddle_ptr + 2), broadcast(twiddle_ptr + 3)};
            cx_reg tw2 = {broadcast(twiddle_ptr + 4),
                          broadcast(twiddle_ptr + 5)};    //NOLINT (*magic-numbers*)
            cx_reg tw0 = {broadcast(twiddle_ptr + 0), broadcast(twiddle_ptr + 1)};

            auto [b0, b1] = avx2::unpack_128(shb0, shb1);
            auto [b2, b3] = avx2::unpack_128(shb2, shb3);

            auto [a0, a1tw] = btfly(b0, b1);
            auto [a2, a3tw] = btfly(b2, b3);

            auto [a1, a3] = mul_pairs(a1tw, tw1, a3tw, tw2);

            auto [p0, p2tw] = btfly(a0, a2);
            auto [p1, p3tw] = btfly(a1, a3);

            auto [p2, p3] = mul_pairs(p2tw, tw0, p3tw, tw0);

            if constexpr (Scale) {
                auto scaling = broadcast(scale);

                p0 = mul(p0, scaling);
                p1 = mul(p1, scaling);
                p2 = mul(p2, scaling);
                p3 = mul(p3, scaling);
            }

            std::tie(p0, p2, p1, p3) = inverse<true>(p0, p2, p1, p3);

            cxstore<PTform>(ptr0, p0);
            cxstore<PTform>(ptr2, p2);
            cxstore<PTform>(ptr1, p1);
            cxstore<PTform>(ptr3, p3);
        }

        return twiddle_ptr;
    }

    template<typename T>
    static inline void insert_unsorted(auto& twiddles, uZ group_size, uZ l_size, uZ i_group) {
        using pcx::detail_::fft::log2i;
        using pcx::detail_::fft::reverse_bit_order;
        using pcx::detail_::fft::wnk;

        for (uint i = 0; i < group_size; ++i) {
            std::size_t start = group_size * i_group + i;

            auto tw0 = wnk<T>(l_size, reverse_bit_order(start, log2i(l_size / 2)));

            twiddles.push_back(tw0.real());
            twiddles.push_back(tw0.imag());

            auto tw1 = wnk<T>(l_size * 2, reverse_bit_order(start * 2, log2i(l_size)));
            auto tw2 = wnk<T>(l_size * 2, reverse_bit_order(start * 2 + 1, log2i(l_size)));

            twiddles.push_back(tw1.real());
            twiddles.push_back(tw1.imag());
            twiddles.push_back(tw2.real());
            twiddles.push_back(tw2.imag());

            auto tw3_1 = wnk<T>(l_size * 4, reverse_bit_order(start * 4, log2i(l_size * 2)));
            auto tw3_2 = wnk<T>(l_size * 4, reverse_bit_order(start * 4 + 1, log2i(l_size * 2)));
            auto tw4_1 = wnk<T>(l_size * 4, reverse_bit_order(start * 4 + 2, log2i(l_size * 2)));
            auto tw4_2 = wnk<T>(l_size * 4, reverse_bit_order(start * 4 + 3, log2i(l_size * 2)));

            auto ins4 = [&twiddles](float tw) {
                for (uint i = 0; i < 4; ++i) {
                    twiddles.push_back(tw);
                }
            };
            ins4(tw3_1.real());
            ins4(tw3_2.real());
            ins4(tw3_1.imag());
            ins4(tw3_2.imag());
            ins4(tw4_1.real());
            ins4(tw4_2.real());
            ins4(tw4_1.imag());
            ins4(tw4_2.imag());

            auto tw7  = wnk<T>(l_size * 8, reverse_bit_order(start * 8, log2i(l_size * 4)));
            auto tw8  = wnk<T>(l_size * 8, reverse_bit_order(start * 8 + 1, log2i(l_size * 4)));
            auto tw9  = wnk<T>(l_size * 8, reverse_bit_order(start * 8 + 2, log2i(l_size * 4)));
            auto tw10 = wnk<T>(l_size * 8, reverse_bit_order(start * 8 + 3, log2i(l_size * 4)));
            auto tw11 = wnk<T>(l_size * 8, reverse_bit_order(start * 8 + 4, log2i(l_size * 4)));
            auto tw12 = wnk<T>(l_size * 8, reverse_bit_order(start * 8 + 5, log2i(l_size * 4)));
            auto tw13 = wnk<T>(l_size * 8, reverse_bit_order(start * 8 + 6, log2i(l_size * 4)));
            auto tw14 = wnk<T>(l_size * 8, reverse_bit_order(start * 8 + 7, log2i(l_size * 4)));

            auto ins2 = [&twiddles](float tw) {
                twiddles.push_back(tw);
                twiddles.push_back(tw);
            };

            ins2(tw7.real());
            ins2(tw8.real());
            ins2(tw9.real());
            ins2(tw10.real());
            ins2(tw7.imag());
            ins2(tw8.imag());
            ins2(tw9.imag());
            ins2(tw10.imag());

            ins2(tw11.real());
            ins2(tw12.real());
            ins2(tw13.real());
            ins2(tw14.real());
            ins2(tw11.imag());
            ins2(tw12.imag());
            ins2(tw13.imag());
            ins2(tw14.imag());

            const std::array<uint, 8> switched_k = {0, 2, 1, 3, 4, 6, 5, 7};
            for (auto k: switched_k) {
                auto tw = wnk<T>(l_size * 16, reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                twiddles.push_back(tw.real());
            }
            for (auto k: switched_k) {
                auto tw = wnk<T>(l_size * 16, reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                twiddles.push_back(tw.imag());
            }

            for (auto k: switched_k) {
                auto tw = wnk<T>(l_size * 16, reverse_bit_order(start * 16 + k + 8, log2i(l_size * 8)));
                twiddles.push_back(tw.real());
            }
            for (auto k: switched_k) {
                auto tw = wnk<T>(l_size * 16, reverse_bit_order(start * 16 + k + 8, log2i(l_size * 8)));
                twiddles.push_back(tw.imag());
            }
        }
    }


    template<uZ PTform, uZ PSrc, bool Inverse>
    static inline void tform_sort(float* data, uZ size, const auto& sort) {
        constexpr auto reg_size = reg<float>::size;
        using cx_reg            = cx_reg<float, false, reg_size>;

        const auto sq2 = pcx::detail_::fft::wnk<float>(8, 1);
        // auto       twsq2 = broadcast(sq2.real());

        auto* src0 = ra_addr<PSrc>(data, 0);
        auto* src1 = ra_addr<PSrc>(data, 1 * size / 8);
        auto* src2 = ra_addr<PSrc>(data, 2 * size / 8);
        auto* src3 = ra_addr<PSrc>(data, 3 * size / 8);
        auto* src4 = ra_addr<PSrc>(data, 4 * size / 8);
        auto* src5 = ra_addr<PSrc>(data, 5 * size / 8);
        auto* src6 = ra_addr<PSrc>(data, 6 * size / 8);
        auto* src7 = ra_addr<PSrc>(data, 7 * size / 8);

        uint i = 0;
        for (; i < pcx::detail_::fft::n_reversals(size / 64); i += 2) {
            auto offset1 = sort[i] * reg_size;
            auto offset2 = sort[i + 1] * reg_size;

            auto p1_ = cxload<PSrc>(ra_addr<PSrc>(src1, offset1));
            auto p5_ = cxload<PSrc>(ra_addr<PSrc>(src5, offset1));
            auto p3_ = cxload<PSrc>(ra_addr<PSrc>(src3, offset1));
            auto p7_ = cxload<PSrc>(ra_addr<PSrc>(src7, offset1));

            // _mm_prefetch(ra_addr<PTform>(src0, offset1), _MM_HINT_T0);
            // _mm_prefetch(ra_addr<PTform>(src4, offset1), _MM_HINT_T0);
            // _mm_prefetch(ra_addr<PTform>(src2, offset1), _MM_HINT_T0);
            // _mm_prefetch(ra_addr<PTform>(src6, offset1), _MM_HINT_T0);

            auto [p1, p5, p3, p7]    = avx2::split(p1_, p5_, p3_, p7_);
            std::tie(p1, p5, p3, p7) = inverse<Inverse>(p1, p5, p3, p7);

            auto [a1, a5] = btfly(p1, p5);
            auto [a3, a7] = btfly(p3, p7);

            auto [b5, b7] = btfly<3>(a5, a7);

            auto twsq2 = broadcast(sq2.real());

            cx_reg b5_tw = {add(b5.real, b5.imag), sub(b5.imag, b5.real)};
            cx_reg b7_tw = {sub(b7.real, b7.imag), add(b7.real, b7.imag)};

            auto [b1, b3] = btfly(a1, a3);

            b5_tw = mul(b5_tw, twsq2);
            b7_tw = mul(b7_tw, twsq2);

            auto p0_ = cxload<PSrc>(ra_addr<PSrc>(src0, offset1));
            auto p4_ = cxload<PSrc>(ra_addr<PSrc>(src4, offset1));
            auto p2_ = cxload<PSrc>(ra_addr<PSrc>(src2, offset1));
            auto p6_ = cxload<PSrc>(ra_addr<PSrc>(src6, offset1));


            // _mm_prefetch(ra_addr<PTform>(src0, offset2), _MM_HINT_T0);
            // _mm_prefetch(ra_addr<PTform>(src4, offset2), _MM_HINT_T0);
            // if constexpr (PSrc < 4) {
            //     _mm_prefetch(ra_addr<PTform>(src2, offset2), _MM_HINT_T0);
            //     _mm_prefetch(ra_addr<PTform>(src6, offset2), _MM_HINT_T0);
            // } else {
            //     _mm_prefetch(ra_addr<PTform>(src1, offset2), _MM_HINT_T0);
            //     _mm_prefetch(ra_addr<PTform>(src5, offset2), _MM_HINT_T0);
            // }

            auto [p0, p4, p2, p6]    = avx2::split(p0_, p4_, p2_, p6_);
            std::tie(p0, p4, p2, p6) = inverse<Inverse>(p0, p4, p2, p6);

            auto [a0, a4] = btfly(p0, p4);
            auto [a2, a6] = btfly(p2, p6);

            auto [b0, b2] = btfly(a0, a2);
            auto [b4, b6] = btfly<3>(a4, a6);

            auto [c0, c1] = btfly(b0, b1);
            auto [c2, c3] = btfly<3>(b2, b3);
            auto [c4, c5] = btfly(b4, b5_tw);
            auto [c6, c7] = btfly<2>(b6, b7_tw);

            auto [sha0, sha4] = avx2::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx2::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx2::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx2::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx2::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx2::unpack_pd(sha1, sha3);

            auto [shc0, shc1] = avx2::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx2::unpack_128(shb2, shb3);

            cx_reg q0, q1, q2, q3, q4, q5, q6, q7;    // NOLINT (*declaration*)

            std::tie(shc0, shc1, shc2, shc3) = inverse<Inverse>(shc0, shc1, shc2, shc3);

            auto q0_ = cxloadstore<PSrc, reg_size>(ra_addr<PSrc>(src0, offset2), shc0);
            auto q4_ = cxloadstore<PSrc, reg_size>(ra_addr<PSrc>(src4, offset2), shc2);

            if constexpr (PSrc < 4) {
                auto q2_ = cxloadstore<PSrc, reg_size>(ra_addr<PSrc>(src2, offset2), shc1);
                auto q6_ = cxloadstore<PSrc, reg_size>(ra_addr<PSrc>(src6, offset2), shc3);

                std::tie(q0, q4, q2, q6) = avx2::split(q0_, q4_, q2_, q6_);
                std::tie(q0, q4, q2, q6) = inverse<Inverse>(q0, q4, q2, q6);
            } else {
                auto q1_ = cxloadstore<PSrc, reg_size>(ra_addr<PSrc>(src1, offset2), shc1);
                auto q5_ = cxloadstore<PSrc, reg_size>(ra_addr<PSrc>(src5, offset2), shc3);

                std::tie(q0, q4, q1, q5) = avx2::split(q0_, q4_, q1_, q5_);
                std::tie(q0, q4, q1, q5) = inverse<Inverse>(q0, q4, q1, q5);
            }

            // if constexpr (PSrc < 4) {
            //     _mm_prefetch(ra_addr<PTform>(src1, offset2), _MM_HINT_T0);
            //     _mm_prefetch(ra_addr<PTform>(src5, offset2), _MM_HINT_T0);
            // } else {
            //     _mm_prefetch(ra_addr<PTform>(src2, offset2), _MM_HINT_T0);
            //     _mm_prefetch(ra_addr<PTform>(src6, offset2), _MM_HINT_T0);
            // }
            // _mm_prefetch(ra_addr<PTform>(src3, offset2), _MM_HINT_T0);
            // _mm_prefetch(ra_addr<PTform>(src7, offset2), _MM_HINT_T0);

            auto [shb4, shb6] = avx2::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx2::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx2::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx2::unpack_128(shb6, shb7);

            std::tie(shc4, shc5, shc6, shc7) = inverse<Inverse>(shc4, shc5, shc6, shc7);
            if constexpr (PSrc < 4) {
                auto q1_ = cxloadstore<PSrc, reg_size>(ra_addr<PSrc>(src1, offset2), shc4);
                auto q5_ = cxloadstore<PSrc, reg_size>(ra_addr<PSrc>(src5, offset2), shc6);
                auto q3_ = cxloadstore<PSrc, reg_size>(ra_addr<PSrc>(src3, offset2), shc5);
                auto q7_ = cxloadstore<PSrc, reg_size>(ra_addr<PSrc>(src7, offset2), shc7);

                std::tie(q1, q5, q3, q7) = avx2::split(q1_, q5_, q3_, q7_);
                std::tie(q1, q5, q3, q7) = inverse<Inverse>(q1, q5, q3, q7);
            } else {
                auto q2_ = cxloadstore<PSrc, reg_size>(ra_addr<PSrc>(src2, offset2), shc4);
                auto q6_ = cxloadstore<PSrc, reg_size>(ra_addr<PSrc>(src6, offset2), shc6);
                auto q3_ = cxloadstore<PSrc, reg_size>(ra_addr<PSrc>(src3, offset2), shc5);
                auto q7_ = cxloadstore<PSrc, reg_size>(ra_addr<PSrc>(src7, offset2), shc7);

                std::tie(q2, q6, q3, q7) = avx2::split(q2_, q6_, q3_, q7_);
                std::tie(q2, q6, q3, q7) = inverse<Inverse>(q2, q6, q3, q7);
            }

            auto [x1, x5] = btfly(q1, q5);
            auto [x3, x7] = btfly(q3, q7);

            auto [y5, y7] = btfly<3>(x5, x7);

            auto [y1, y3] = btfly(x1, x3);

            cx_reg y5_tw = {add(y5.real, y5.imag), sub(y5.imag, y5.real)};
            cx_reg y7_tw = {sub(y7.real, y7.imag), add(y7.real, y7.imag)};

            auto [x0, x4] = btfly(q0, q4);

            y5_tw = mul(y5_tw, twsq2);
            y7_tw = mul(y7_tw, twsq2);

            auto [x2, x6] = btfly(q2, q6);

            auto [y0, y2] = btfly(x0, x2);
            auto [y4, y6] = btfly<3>(x4, x6);

            auto [z0, z1] = btfly(y0, y1);
            auto [z2, z3] = btfly<3>(y2, y3);
            auto [z4, z5] = btfly(y4, y5_tw);
            auto [z6, z7] = btfly<2>(y6, y7_tw);

            auto [shx0, shx4] = avx2::unpack_ps(z0, z4);
            auto [shx1, shx5] = avx2::unpack_ps(z1, z5);
            auto [shx2, shx6] = avx2::unpack_ps(z2, z6);
            auto [shx3, shx7] = avx2::unpack_ps(z3, z7);

            auto [shy0, shy2] = avx2::unpack_pd(shx0, shx2);
            auto [shy1, shy3] = avx2::unpack_pd(shx1, shx3);

            auto [shz0, shz1] = avx2::unpack_128(shy0, shy1);
            auto [shz2, shz3] = avx2::unpack_128(shy2, shy3);

            std::tie(shz0, shz1, shz2, shz3) = inverse<Inverse>(shz0, shz1, shz2, shz3);

            cxstore<PTform>(ra_addr<PTform>(src4, offset1), shz2);
            cxstore<PTform>(ra_addr<PTform>(src0, offset1), shz0);
            if constexpr (PSrc < 4) {
                cxstore<PTform>(ra_addr<PTform>(src2, offset1), shz1);
                cxstore<PTform>(ra_addr<PTform>(src6, offset1), shz3);
            } else {
                cxstore<PTform>(ra_addr<PTform>(src1, offset1), shz1);
                cxstore<PTform>(ra_addr<PTform>(src5, offset1), shz3);
            }

            auto [shy4, shy6] = avx2::unpack_pd(shx4, shx6);
            auto [shy5, shy7] = avx2::unpack_pd(shx5, shx7);

            auto [shz4, shz5] = avx2::unpack_128(shy4, shy5);
            auto [shz6, shz7] = avx2::unpack_128(shy6, shy7);

            std::tie(shz4, shz5, shz6, shz7) = inverse<Inverse>(shz4, shz5, shz6, shz7);

            if constexpr (PSrc < 4) {
                cxstore<PTform>(ra_addr<PTform>(src1, offset1), shz4);
                cxstore<PTform>(ra_addr<PTform>(src5, offset1), shz6);
            } else {
                cxstore<PTform>(ra_addr<PTform>(src2, offset1), shz4);
                cxstore<PTform>(ra_addr<PTform>(src6, offset1), shz6);
            }
            cxstore<PTform>(ra_addr<PTform>(src3, offset1), shz5);
            cxstore<PTform>(ra_addr<PTform>(src7, offset1), shz7);
        };

        for (; i < size / 64; ++i) {
            auto offset = sort[i] * reg_size;

            auto p1_ = cxload<PSrc>(ra_addr<PSrc>(src1, offset));
            auto p5_ = cxload<PSrc>(ra_addr<PSrc>(src5, offset));
            auto p3_ = cxload<PSrc>(ra_addr<PSrc>(src3, offset));
            auto p7_ = cxload<PSrc>(ra_addr<PSrc>(src7, offset));

            auto [p1, p5, p3, p7]    = avx2::split(p1_, p5_, p3_, p7_);
            std::tie(p1, p5, p3, p7) = inverse<Inverse>(p1, p5, p3, p7);

            auto [a1, a5] = btfly(p1, p5);
            auto [a3, a7] = btfly(p3, p7);

            auto [b5, b7] = btfly<3>(a5, a7);
            auto [b1, b3] = btfly(a1, a3);

            auto twsq2 = broadcast(sq2.real());

            cx_reg b5_tw = {add(b5.real, b5.imag), sub(b5.imag, b5.real)};
            cx_reg b7_tw = {sub(b7.real, b7.imag), add(b7.real, b7.imag)};

            b5_tw = mul(b5_tw, twsq2);
            b7_tw = mul(b7_tw, twsq2);

            auto p0_ = cxload<PSrc>(ra_addr<PSrc>(src0, offset));
            auto p4_ = cxload<PSrc>(ra_addr<PSrc>(src4, offset));
            auto p2_ = cxload<PSrc>(ra_addr<PSrc>(src2, offset));
            auto p6_ = cxload<PSrc>(ra_addr<PSrc>(src6, offset));

            auto [p0, p4, p2, p6]    = avx2::split(p0_, p4_, p2_, p6_);
            std::tie(p0, p4, p2, p6) = inverse<Inverse>(p0, p4, p2, p6);

            auto [a0, a4] = btfly(p0, p4);
            auto [a2, a6] = btfly(p2, p6);

            auto [b0, b2] = btfly(a0, a2);
            auto [b4, b6] = btfly<3>(a4, a6);

            auto [c0, c1] = btfly(b0, b1);
            auto [c2, c3] = btfly<3>(b2, b3);
            auto [c4, c5] = btfly(b4, b5_tw);
            auto [c6, c7] = btfly<2>(b6, b7_tw);

            auto [sha0, sha4] = avx2::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx2::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx2::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx2::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx2::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx2::unpack_pd(sha1, sha3);

            auto [shc0, shc1] = avx2::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx2::unpack_128(shb2, shb3);

            std::tie(shc0, shc1, shc2, shc3) = inverse<Inverse>(shc0, shc1, shc2, shc3);

            cxstore<PTform>(ra_addr<PTform>(src0, offset), shc0);
            cxstore<PTform>(ra_addr<PTform>(src4, offset), shc2);
            if constexpr (PSrc < 4) {
                cxstore<PTform>(ra_addr<PTform>(src2, offset), shc1);
                cxstore<PTform>(ra_addr<PTform>(src6, offset), shc3);
            } else {
                cxstore<PTform>(ra_addr<PTform>(src1, offset), shc1);
                cxstore<PTform>(ra_addr<PTform>(src5, offset), shc3);
            }

            auto [shb4, shb6] = avx2::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx2::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx2::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx2::unpack_128(shb6, shb7);

            std::tie(shc4, shc5, shc6, shc7) = inverse<Inverse>(shc4, shc5, shc6, shc7);

            if constexpr (PSrc < 4) {
                cxstore<PTform>(ra_addr<PTform>(src1, offset), shc4);
                cxstore<PTform>(ra_addr<PTform>(src5, offset), shc6);
            } else {
                cxstore<PTform>(ra_addr<PTform>(src2, offset), shc4);
                cxstore<PTform>(ra_addr<PTform>(src6, offset), shc6);
            }
            cxstore<PTform>(ra_addr<PTform>(src3, offset), shc5);
            cxstore<PTform>(ra_addr<PTform>(src7, offset), shc7);
        }
    }

    template<uZ PTform, uZ PSrc, bool Inverse>
    static inline void tform_sort(float* dest, const float* source, uZ size, const auto& sort) {
        constexpr auto reg_size = reg<float>::size;
        using cx_reg            = cx_reg<float, false, reg_size>;

        const auto sq2   = pcx::detail_::fft::wnk<float>(8, 1);
        auto       twsq2 = broadcast(sq2.real());

        const auto* const src0 = source;
        const auto* const src1 = ra_addr<PSrc>(source, 1 * size / 8);
        const auto* const src2 = ra_addr<PSrc>(source, 2 * size / 8);
        const auto* const src3 = ra_addr<PSrc>(source, 3 * size / 8);
        const auto* const src4 = ra_addr<PSrc>(source, 4 * size / 8);
        const auto* const src5 = ra_addr<PSrc>(source, 5 * size / 8);
        const auto* const src6 = ra_addr<PSrc>(source, 6 * size / 8);
        const auto* const src7 = ra_addr<PSrc>(source, 7 * size / 8);

        auto* const dst0 = dest;
        auto* const dst1 = ra_addr<PTform>(dest, 1 * size / 8);
        auto* const dst2 = ra_addr<PTform>(dest, 2 * size / 8);
        auto* const dst3 = ra_addr<PTform>(dest, 3 * size / 8);
        auto* const dst4 = ra_addr<PTform>(dest, 4 * size / 8);
        auto* const dst5 = ra_addr<PTform>(dest, 5 * size / 8);
        auto* const dst6 = ra_addr<PTform>(dest, 6 * size / 8);
        auto* const dst7 = ra_addr<PTform>(dest, 7 * size / 8);

        uint i = 0;
        for (; i < pcx::detail_::fft::n_reversals(size / 64); i += 2) {
            auto offset_src  = sort[i] * reg_size;
            auto offset_dest = sort[i + 1] * reg_size;

            for (uint k = 0; k < 2; ++k) {
                auto p1_ = cxload<PSrc>(ra_addr<PSrc>(src1, offset_src));
                auto p5_ = cxload<PSrc>(ra_addr<PSrc>(src5, offset_src));
                auto p3_ = cxload<PSrc>(ra_addr<PSrc>(src3, offset_src));
                auto p7_ = cxload<PSrc>(ra_addr<PSrc>(src7, offset_src));

                auto [p1, p5, p3, p7] = avx2::split(p1_, p5_, p3_, p7_);

                auto [a1, a5] = btfly(p1, p5);
                auto [a3, a7] = btfly(p3, p7);

                cx_reg b5 = {add(a5.real, a7.imag), sub(a5.imag, a7.real)};
                cx_reg b7 = {sub(a5.real, a7.imag), add(a5.imag, a7.real)};

                cx_reg b5_tw = {add(b5.real, b5.imag), sub(b5.imag, b5.real)};
                cx_reg b7_tw = {sub(b7.real, b7.imag), add(b7.real, b7.imag)};

                auto [b1, b3] = btfly(a1, a3);

                b5_tw = mul(b5_tw, twsq2);
                b7_tw = mul(b7_tw, twsq2);

                auto p0_ = cxload<PSrc>(ra_addr<PSrc>(src0, offset_src));
                auto p4_ = cxload<PSrc>(ra_addr<PSrc>(src4, offset_src));
                auto p2_ = cxload<PSrc>(ra_addr<PSrc>(src2, offset_src));
                auto p6_ = cxload<PSrc>(ra_addr<PSrc>(src6, offset_src));

                auto [p0, p4, p2, p6] = avx2::split(p0_, p4_, p2_, p6_);


                auto [a0, a4] = btfly(p0, p4);
                auto [a2, a6] = btfly(p2, p6);

                auto [b0, b2] = btfly(a0, a2);
                cx_reg b4     = {add(a4.real, a6.imag), sub(a4.imag, a6.real)};
                cx_reg b6     = {sub(a4.real, a6.imag), add(a4.imag, a6.real)};

                auto [c0, c1] = btfly(b0, b1);
                cx_reg c2     = {add(b2.real, b3.imag), sub(b2.imag, b3.real)};
                cx_reg c3     = {sub(b2.real, b3.imag), add(b2.imag, b3.real)};

                auto [c4, c5] = btfly(b4, b5_tw);
                auto [c7, c6] = btfly(b6, b7_tw);

                auto [sha0, sha4] = avx2::unpack_ps(c0, c4);
                auto [sha2, sha6] = avx2::unpack_ps(c2, c6);
                auto [sha1, sha5] = avx2::unpack_ps(c1, c5);
                auto [sha3, sha7] = avx2::unpack_ps(c3, c7);

                auto [shb0, shb2] = avx2::unpack_pd(sha0, sha2);
                auto [shb1, shb3] = avx2::unpack_pd(sha1, sha3);
                auto [shc0, shc1] = avx2::unpack_128(shb0, shb1);
                auto [shc2, shc3] = avx2::unpack_128(shb2, shb3);

                cxstore<PTform>(ra_addr<PTform>(dst0, offset_dest), shc0);
                cxstore<PTform>(ra_addr<PTform>(dst4, offset_dest), shc2);
                if constexpr (PSrc < 4) {
                    cxstore<PTform>(ra_addr<PTform>(dst2, offset_dest), shc1);
                    cxstore<PTform>(ra_addr<PTform>(dst6, offset_dest), shc3);
                } else {
                    cxstore<PTform>(ra_addr<PTform>(dst1, offset_dest), shc1);
                    cxstore<PTform>(ra_addr<PTform>(dst5, offset_dest), shc3);
                }

                auto [shb4, shb6] = avx2::unpack_pd(sha4, sha6);
                auto [shb5, shb7] = avx2::unpack_pd(sha5, sha7);
                auto [shc4, shc5] = avx2::unpack_128(shb4, shb5);
                auto [shc6, shc7] = avx2::unpack_128(shb6, shb7);

                if constexpr (PSrc < 4) {
                    cxstore<PTform>(ra_addr<PTform>(dst1, offset_dest), shc4);
                    cxstore<PTform>(ra_addr<PTform>(dst5, offset_dest), shc6);
                } else {
                    cxstore<PTform>(ra_addr<PTform>(dst2, offset_dest), shc4);
                    cxstore<PTform>(ra_addr<PTform>(dst6, offset_dest), shc6);
                }
                cxstore<PTform>(ra_addr<PTform>(dst3, offset_dest), shc5);
                cxstore<PTform>(ra_addr<PTform>(dst7, offset_dest), shc7);

                offset_src  = sort[i + 1] * reg_size;
                offset_dest = sort[i] * reg_size;
            }
        };
        for (; i < size / 64; ++i) {
            auto offset_src  = sort[i] * reg_size;
            auto offset_dest = sort[i] * reg_size;

            auto p1_ = cxload<PSrc>(ra_addr<PSrc>(src1, offset_src));
            auto p5_ = cxload<PSrc>(ra_addr<PSrc>(src5, offset_src));
            auto p3_ = cxload<PSrc>(ra_addr<PSrc>(src3, offset_src));
            auto p7_ = cxload<PSrc>(ra_addr<PSrc>(src7, offset_src));

            auto [p1, p5, p3, p7] = avx2::split(p1_, p5_, p3_, p7_);

            auto [a1, a5] = btfly(p1, p5);
            auto [a3, a7] = btfly(p3, p7);

            cx_reg b5 = {add(a5.real, a7.imag), sub(a5.imag, a7.real)};
            cx_reg b7 = {sub(a5.real, a7.imag), add(a5.imag, a7.real)};

            cx_reg b5_tw = {add(b5.real, b5.imag), sub(b5.imag, b5.real)};
            cx_reg b7_tw = {sub(b7.real, b7.imag), add(b7.real, b7.imag)};

            auto [b1, b3] = btfly(a1, a3);

            b5_tw = mul(b5_tw, twsq2);
            b7_tw = mul(b7_tw, twsq2);

            auto p0_ = cxload<PSrc>(ra_addr<PSrc>(src0, offset_src));
            auto p4_ = cxload<PSrc>(ra_addr<PSrc>(src4, offset_src));
            auto p2_ = cxload<PSrc>(ra_addr<PSrc>(src2, offset_src));
            auto p6_ = cxload<PSrc>(ra_addr<PSrc>(src6, offset_src));

            auto [p0, p4, p2, p6] = avx2::split(p0_, p4_, p2_, p6_);

            auto [a0, a4] = btfly(p0, p4);
            auto [a2, a6] = btfly(p2, p6);

            auto [b0, b2] = btfly(a0, a2);
            cx_reg b4     = {add(a4.real, a6.imag), sub(a4.imag, a6.real)};
            cx_reg b6     = {sub(a4.real, a6.imag), add(a4.imag, a6.real)};

            auto [c0, c1] = btfly(b0, b1);
            cx_reg c2     = {add(b2.real, b3.imag), sub(b2.imag, b3.real)};
            cx_reg c3     = {sub(b2.real, b3.imag), add(b2.imag, b3.real)};

            auto [c4, c5] = btfly(b4, b5_tw);
            auto [c7, c6] = btfly(b6, b7_tw);

            auto [sha0, sha4] = avx2::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx2::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx2::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx2::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx2::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx2::unpack_pd(sha1, sha3);
            auto [shc0, shc1] = avx2::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx2::unpack_128(shb2, shb3);

            cxstore<PTform>(ra_addr<PTform>(dst0, offset_dest), shc0);
            cxstore<PTform>(ra_addr<PTform>(dst4, offset_dest), shc2);
            if constexpr (PSrc < 4) {
                cxstore<PTform>(ra_addr<PTform>(dst2, offset_dest), shc1);
                cxstore<PTform>(ra_addr<PTform>(dst6, offset_dest), shc3);
            } else {
                cxstore<PTform>(ra_addr<PTform>(dst1, offset_dest), shc1);
                cxstore<PTform>(ra_addr<PTform>(dst5, offset_dest), shc3);
            }

            auto [shb4, shb6] = avx2::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx2::unpack_pd(sha5, sha7);
            auto [shc4, shc5] = avx2::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx2::unpack_128(shb6, shb7);

            if constexpr (PSrc < 4) {
                cxstore<PTform>(ra_addr<PTform>(dst1, offset_dest), shc4);
                cxstore<PTform>(ra_addr<PTform>(dst5, offset_dest), shc6);
            } else {
                cxstore<PTform>(ra_addr<PTform>(dst2, offset_dest), shc4);
                cxstore<PTform>(ra_addr<PTform>(dst6, offset_dest), shc6);
            }
            cxstore<PTform>(ra_addr<PTform>(dst3, offset_dest), shc5);
            cxstore<PTform>(ra_addr<PTform>(dst7, offset_dest), shc7);
        }
    }
};

// NOLINTEND(*pointer-arithmetic*)

}    // namespace pcx::simd
#endif