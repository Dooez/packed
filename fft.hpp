#ifndef FFT_HPP
#define FFT_HPP

#include "avx2_common.hpp"
#include "avx2_fft.hpp"
#include "simd_common.hpp"
#include "types.hpp"
#include "vector.hpp"
#include "vector_arithm.hpp"
#include "vector_util.hpp"

#include <array>
#include <bits/ranges_base.h>
// #include <bits/utility.h>
#include <bits/utility.h>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <iostream>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <xmmintrin.h>

// NOLINTBEGIN (*magic-numbers)
namespace pcx {
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


namespace fft {
//TODO: Replace with tables.
constexpr auto log2i(u64 num) -> uZ {
    u64 order = 0;
    for (u8 shift = 32; shift > 0; shift /= 2) {
        if (num >> shift > 0) {
            order += num >> shift > 0 ? shift : 0;
            num >>= shift;
        }
    }
    return order;
}

constexpr auto powi(uint64_t num, uint64_t pow) -> uint64_t {
    auto res = (pow % 2) == 1 ? num : 1UL;
    if (pow > 1) {
        auto half_pow = powi(num, pow / 2UL);
        res *= half_pow * half_pow;
    }
    return res;
}

constexpr auto reverse_bit_order(u64 num, u64 depth) -> u64 {
    num = num >> 32 | num << 32;
    num = (num & 0xFFFF0000FFFF0000) >> 16 | (num & 0x0000FFFF0000FFFF) << 16;
    num = (num & 0xFF00FF00FF00FF00) >> 8 | (num & 0x00FF00FF00FF00FF) << 8;
    num = (num & 0xF0F0F0F0F0F0F0F0) >> 4 | (num & 0x0F0F0F0F0F0F0F0F) << 4;
    num = (num & 0xCCCCCCCCCCCCCCCC) >> 2 | (num & 0x3333333333333333) << 2;
    num = (num & 0xAAAAAAAAAAAAAAAA) >> 1 | (num & 0x5555555555555555) << 1;
    return num >> (64 - depth);
}

/**
 * @brief Returns number of unique bit-reversed pairs from 0 to max-1
 */
constexpr auto n_reversals(uZ max) -> uZ {
    return max - (1U << ((log2i(max) + 1) / 2));
}

template<typename T>
inline auto wnk(uZ n, uZ k) -> std::complex<T> {
    constexpr double pi = 3.14159265358979323846;
    if (n == k * 4) {
        return {0, -1};
    }
    if (n == k * 2) {
        return {-1, 0};
    }
    return exp(std::complex<T>(0, -2 * pi * static_cast<double>(k) / static_cast<double>(n)));
}

template<uZ Size, bool DecInTime>
struct order {
    static constexpr std::array<uZ, Size> data = [] {
        std::array<uZ, Size> order;
        for (uZ i = 0; i < Size; ++i) {
            if constexpr (DecInTime) {
                order[i] = reverse_bit_order(i, log2i(Size));
            } else {
                order[i] = i;
            }
        }
        return order;
    }();

    static constexpr std::array<uZ, Size - 1> tw = []<uZ... N>(std::index_sequence<N...>) {
        if constexpr (DecInTime) {
            return std::array<uZ, sizeof...(N)>{(
                N > 0
                    ? (1U << log2i(N + 1)) - 1 + reverse_bit_order(1 + N - (1U << log2i(N + 1)), log2i(N + 1))
                    : 0)...};
        } else {
            return std::array<uZ, sizeof...(N)>{N...};
        }
    }(std::make_index_sequence<Size - 1>{});
};

template<uZ NodeSize>
struct node {
    template<uZ... I, uZ... J>
    static inline auto combine(std::index_sequence<I...>, std::index_sequence<J...>) {
        return std::index_sequence<I..., J...>{};
    }
    template<uZ I, uZ... L>
    static inline auto offset_seqence(std::index_sequence<I>, std::index_sequence<L...>) {
        return std::index_sequence<I + L...>{};
    }
    template<uZ I, uZ... K, uZ... L>
    static inline auto offset_sequence(std::index_sequence<I, K...>, std::index_sequence<L...> s) {
        return combine(std::index_sequence<I + L...>{}, foo(std::index_sequence<K...>{}, s));
    }
    template<uZ M, uZ... I>
    static inline auto mul_sequence(std::index_sequence<I...>) {
        return std::index_sequence<M * I...>{};
    }

    //     template<typename T, uZ PDest, uZ PSrc, bool ConjTw, bool Reverse, bool DIT = false, typename... Args>
    //     static inline void perform123(std::array<T*, 4> dest, Args... args) {
    //         constexpr auto reg_size = simd::reg<T>::size;
    //         using cx_reg            = simd::cx_reg<T, false, reg_size>;
    //
    //         using src_type = std::array<const T*, NodeSize>;
    //         using tw_type  = std::array<simd::cx_reg<T>, NodeSize - 1>;
    //
    //         constexpr bool Inverse = ConjTw || Reverse;
    //
    //         constexpr bool Src   = has_type<src_type, Args...>;
    //         constexpr bool Tw    = has_type<tw_type, Args...>;
    //         constexpr bool Scale = has_type<simd::reg_t<T>, Args...>;
    //
    //         constexpr auto& data_idx = order<NodeSize, DIT>::data;
    //         constexpr auto& tw_idx   = order<NodeSize, DIT>::tw;
    //
    //         constexpr auto load = [](auto&& dest, auto&&... args) {
    //             constexpr auto load_impl = []<uZ... I>(auto&& src, std::index_sequence<I...>) {
    //                 constexpr auto& data_idx = order<NodeSize, DIT>::data;
    //                 return std::make_tuple(src[data_idx[I]]...);
    //             };
    //
    //             if constexpr (Src) {
    //                 auto& src = std::get<src_type&>(std::tie(args...));
    //                 return load_impl(std::forward(src), std::make_index_sequence<NodeSize>{});
    //             } else {
    //                 return load_impl(std::forward(dest), std::make_index_sequence<NodeSize>{});
    //             }
    //         };
    //         constexpr auto repack_lmbd  = [](auto&&... regs) { return simd::repack2<reg_size>(regs...); };
    //         constexpr auto inverse_lmbd = [](auto&&... regs) { return simd::inverse<Inverse>(regs...); };
    //
    //         auto dat = load(std::forward(dest, args...));
    //         dat      = std::apply(repack_lmbd, dat);
    //         dat      = std::apply(inverse_lmbd, dat);
    //
    //         constexpr auto btfly = []<uZ Stride>(auto arg_tuple) {
    //             constexpr auto extract = []<uZ... I>(auto tuple) {
    //                 return std::make_tuple(std::get<I>(tuple)...);
    //             };
    //             constexpr auto n_g = NodeSize / Stride / 2;
    //             constexpr auto l   = offset_seqence(mul_sequence<Stride * 2>(std::make_index_sequence<n_g>{}),
    //                                               std::make_index_sequence<Stride>{});
    //         };
    //
    //         cx_reg b1, b3, b5, b7;
    //         if constexpr (Tw) {
    //             auto& tw          = std::get<tw_type&>(std::tie(args...));
    //             auto [p5tw, p7tw] = simd::mul({p5, tw[tw_idx[0]]}, {p7, tw[tw_idx[0]]});
    //             auto [a1, a5]     = simd::btfly(p1, p5tw);
    //             auto [a3, a7]     = simd::btfly(p3, p7tw);
    //
    //             auto [a3tw, a7tw] = simd::mul({a3, tw[tw_idx[1]]}, {a7, tw[tw_idx[2]]});
    //             std::tie(b1, b3)  = simd::btfly(a1, a3tw);
    //             std::tie(b5, b7)  = simd::btfly(a5, a7tw);
    //
    //             std::tie(b1, b3, b5, b7) =
    //                 simd::mul({b1, tw[tw_idx[3]]}, {b3, tw[tw_idx[4]]}, {b5, tw[tw_idx[5]]}, {b7, tw[tw_idx[6]]});
    //         } else {
    //             const T sq2   = std::sqrt(double{2}) / 2;
    //             auto [a1, a5] = simd::btfly(p1, p5);
    //             auto [a3, a7] = simd::btfly(p3, p7);
    //
    //             std::tie(b1, b3) = simd::btfly(a1, a3);
    //             std::tie(b5, b7) = simd::btfly<3>(a5, a7);
    //             auto b5_tw       = simd::cx_reg<T>{simd::add(b5.real, b5.imag), simd::sub(b5.imag, b5.real)};
    //             auto b7_tw       = simd::cx_reg<T>{simd::sub(b7.real, b7.imag), simd::add(b7.real, b7.imag)};
    //             auto twsq2       = simd::broadcast(sq2);
    //             b5               = simd::mul(b5_tw, twsq2);
    //             b7               = simd::mul(b7_tw, twsq2);
    //         }
    //         simd::cx_reg<T, false, PSrc> p0_, p2_, p4_, p6_;
    //         if constexpr (Src) {
    //             auto& src = std::get<src_type&>(std::tie(args...));
    //             p4_       = simd::cxload<PSrc>(src[data_idx[4]]);
    //             p0_       = simd::cxload<PSrc>(src[data_idx[0]]);
    //             p6_       = simd::cxload<PSrc>(src[data_idx[6]]);
    //             p2_       = simd::cxload<PSrc>(src[data_idx[2]]);
    //         } else {
    //             p4_ = simd::cxload<PSrc>(dest[data_idx[4]]);
    //             p0_ = simd::cxload<PSrc>(dest[data_idx[0]]);
    //             p6_ = simd::cxload<PSrc>(dest[data_idx[6]]);
    //             p2_ = simd::cxload<PSrc>(dest[data_idx[2]]);
    //         }
    //         auto [p4, p0, p6, p2]    = simd::repack2<reg_size>(p4_, p0_, p6_, p2_);
    //         std::tie(p4, p0, p6, p2) = simd::inverse<Inverse>(p4, p0, p6, p2);
    //
    //         simd::cx_reg<T> c0, c1, c2, c3, b4, b6;
    //         if constexpr (Tw) {
    //             auto& tw          = std::get<tw_type&>(std::tie(args...));
    //             auto [p4tw, p6tw] = simd::mul({p4, tw[tw_idx[0]]}, {p6, tw[tw_idx[0]]});
    //             auto [a0, a4]     = simd::btfly(p0, p4tw);
    //             auto [a2, a6]     = simd::btfly(p2, p6tw);
    //
    //             auto [a2tw, a6tw] = simd::mul({a2, tw[tw_idx[1]]}, {a6, tw[tw_idx[2]]});
    //             auto [b0, b2]     = simd::btfly(a0, a2tw);
    //             std::tie(b4, b6)  = simd::btfly(a4, a6tw);
    //
    //             std::tie(c0, c1) = simd::btfly(b0, b1);
    //             std::tie(c2, c3) = simd::btfly(b2, b3);
    //         } else {
    //             auto [a0, a4] = simd::btfly(p0, p4);
    //             auto [a2, a6] = simd::btfly(p2, p6);
    //
    //             auto [b0, b2]    = simd::btfly(a0, a2);
    //             std::tie(b4, b6) = simd::btfly<3>(a4, a6);
    //
    //             std::tie(c0, c1) = simd::btfly(b0, b1);
    //             std::tie(c2, c3) = simd::btfly<3>(b2, b3);
    //         }
    //         if constexpr (Scale) {
    //             auto& scaling = std::get<simd::reg_t<T>&>(std::tie(args...));
    //             c0            = simd::mul(c0, scaling);
    //             c1            = simd::mul(c1, scaling);
    //             c2            = simd::mul(c2, scaling);
    //             c3            = simd::mul(c3, scaling);
    //         }
    //         std::tie(c0, c1, c2, c3)  = simd::inverse<Inverse>(c0, c1, c2, c3);
    //         auto [c0_, c1_, c2_, c3_] = simd::repack2<PDest>(c0, c1, c2, c3);
    //
    //         cxstore<PDest>(dest[data_idx[0]], c0_);
    //         cxstore<PDest>(dest[data_idx[1]], c1_);
    //         cxstore<PDest>(dest[data_idx[2]], c2_);
    //         cxstore<PDest>(dest[data_idx[3]], c3_);
    //
    //         simd::cx_reg<T> c4, c5, c6, c7;
    //         if constexpr (Tw) {
    //             std::tie(c4, c5) = simd::btfly(b4, b5);
    //             std::tie(c6, c7) = simd::btfly(b6, b7);
    //         } else {
    //             std::tie(c4, c5) = simd::btfly(b4, b5);
    //             std::tie(c6, c7) = simd::btfly<2>(b6, b7);
    //         }
    //         if constexpr (Scale) {
    //             auto& scaling = std::get<simd::reg_t<T>&>(std::tie(args...));
    //             c4            = simd::mul(c4, scaling);
    //             c5            = simd::mul(c5, scaling);
    //             c6            = simd::mul(c6, scaling);
    //             c7            = simd::mul(c7, scaling);
    //         }
    //         std::tie(c4, c5, c6, c7)  = simd::inverse<Inverse>(c4, c5, c6, c7);
    //         auto [c4_, c5_, c6_, c7_] = simd::repack2<PDest>(c4, c5, c6, c7);
    //
    //         cxstore<PDest>(dest[data_idx[4]], c4_);
    //         cxstore<PDest>(dest[data_idx[5]], c5_);
    //         cxstore<PDest>(dest[data_idx[6]], c6_);
    //         cxstore<PDest>(dest[data_idx[7]], c7_);
    //     }
};
template<>
struct node<2> {
    template<typename T, uZ PDest, uZ PSrc, bool ConjTw, bool Reverse, bool DIT = false, typename... Args>
    static inline void perform(std::array<T*, 2> dest, Args... args) {
        constexpr auto reg_size = simd::reg<T>::size;
        using cx_reg            = simd::cx_reg<T, false, reg_size>;

        constexpr bool Inverse = ConjTw || Reverse;

        using src_type       = std::array<const T*, 2>;
        using tw_type        = std::array<simd::cx_reg<T>, 1>;
        constexpr bool Src   = has_type<src_type, Args...>;
        constexpr bool Tw    = has_type<tw_type, Args...>;
        constexpr bool Scale = has_type<simd::reg_t<T>, Args...>;

        // NOLINTNEXTLINE(*-declaration)
        simd::cx_reg<T, false, PSrc> p0_, p1_;
        if constexpr (Src) {
            auto& src = std::get<src_type&>(std::tie(args...));
            p0_       = simd::cxload<PSrc>(src[0]);
            p1_       = simd::cxload<PSrc>(src[1]);
        } else {
            p0_ = simd::cxload<PSrc>(dest[0]);
            p1_ = simd::cxload<PSrc>(dest[1]);
        }
        auto [p0, p1]    = simd::repack2<reg_size>(p0_, p1_);
        std::tie(p0, p1) = simd::inverse<Inverse>(p0, p1);
        // NOLINTNEXTLINE(*-declaration)
        cx_reg a0, a1;
        if constexpr (Reverse) {
            std::tie(a0, a1) = simd::ibtfly(p0, p1);
            if constexpr (Tw) {
                auto& tw = std::get<tw_type&>(std::tie(args...));
                a1       = simd::mul(a1, tw[0]);
            }
        } else {
            if constexpr (Tw) {
                auto& tw = std::get<tw_type&>(std::tie(args...));
                p1       = simd::mul(p1, tw[0]);
            }
            std::tie(a0, a1) = simd::btfly(p0, p1);
        }
        if constexpr (Scale) {
            auto& scaling = std::get<simd::reg_t<T>&>(std::tie(args...));
            a0            = simd::mul(a0, scaling);
            a1            = simd::mul(a1, scaling);
        }
        std::tie(a0, a1) = simd::inverse<Inverse>(a0, a1);
        auto [a0_, a1_]  = simd::repack2<PDest>(a0, a1);

        cxstore<PDest>(dest[0], a0_);
        cxstore<PDest>(dest[1], a1_);
    };
};
template<>
struct node<4> {
    template<typename T, uZ PDest, uZ PSrc, bool ConjTw, bool Reverse, bool DIT = false, typename... Args>
    static inline void perform(std::array<T*, 4> dest, Args... args) {
        constexpr auto reg_size = simd::reg<T>::size;
        using cx_reg            = simd::cx_reg<T, false, reg_size>;

        constexpr bool Inverse = ConjTw || Reverse;

        using src_type       = std::array<const T*, 4>;
        using tw_type        = std::array<simd::cx_reg<T>, 3>;
        constexpr bool Src   = has_type<src_type, Args...>;
        constexpr bool Tw    = has_type<tw_type, Args...>;
        constexpr bool Scale = has_type<simd::reg_t<T>, Args...>;

        constexpr auto& data_idx = order<4, DIT>::data;
        // NOLINTNEXTLINE(*-declaration)
        simd::cx_reg<T, false, PSrc> p0_, p1_, p2_, p3_;
        if constexpr (Src) {
            auto& src = std::get<src_type&>(std::tie(args...));
            p2_       = simd::cxload<PSrc>(src[data_idx[2]]);
            p3_       = simd::cxload<PSrc>(src[data_idx[3]]);
            p0_       = simd::cxload<PSrc>(src[data_idx[0]]);
            p1_       = simd::cxload<PSrc>(src[data_idx[1]]);
        } else {
            p2_ = simd::cxload<PSrc>(dest[data_idx[2]]);
            p3_ = simd::cxload<PSrc>(dest[data_idx[3]]);
            p0_ = simd::cxload<PSrc>(dest[data_idx[0]]);
            p1_ = simd::cxload<PSrc>(dest[data_idx[1]]);
        }
        // std::tie(p2, p3, p0, p1) = simd::repack<PSrc, PLoad>(p2, p3, p0, p1)
        auto [p2, p3, p0, p1]    = simd::repack2<reg_size>(p2_, p3_, p0_, p1_);
        std::tie(p2, p3, p0, p1) = simd::inverse<Inverse>(p2, p3, p0, p1);
        // NOLINTNEXTLINE(*-declaration)
        cx_reg b0, b1, b2, b3;
        if constexpr (Tw) {
            auto& tw = std::get<tw_type&>(std::tie(args...));
            if constexpr (Reverse) {
                auto [a2, a3tw] = simd::ibtfly(p2, p3);
                auto [a0, a1tw] = simd::ibtfly(p0, p1);
                // auto [a3, a1]    = simd::mul({a3tw, tw[2]}, {a1tw, tw[1]});
                auto [a3, a1]    = simd::mul_pairs(a3tw, tw[2], a1tw, tw[1]);
                std::tie(b0, b2) = simd::ibtfly(a0, a2);
                std::tie(b1, b3) = simd::ibtfly(a1, a3);
                // std::tie(b2, b3) = simd::mul({b2, tw[0]}, {b3, tw[0]});
                std::tie(b2, b3) = simd::mul_pairs(b2, tw[0], b3, tw[0]);
            } else {
                auto [p2tw, p3tw] = simd::mul({p2, tw[0]}, {p3, tw[0]});
                auto [a0, a2]     = simd::btfly(p0, p2tw);
                auto [a1, a3]     = simd::btfly(p1, p3tw);
                auto [a1tw, a3tw] = simd::mul({a1, tw[1]}, {a3, tw[2]});
                std::tie(b0, b1)  = simd::btfly(a0, a1tw);
                std::tie(b2, b3)  = simd::btfly(a2, a3tw);
            }
        } else {
            if constexpr (Reverse) {
                auto [a2, a3]    = simd::ibtfly<3>(p2, p3);
                auto [a0, a1]    = simd::ibtfly(p0, p1);
                std::tie(b0, b2) = simd::ibtfly(a0, a2);
                std::tie(b1, b3) = simd::ibtfly(a1, a3);
            } else {
                auto [a0, a2]    = simd::btfly(p0, p2);
                auto [a1, a3]    = simd::btfly(p1, p3);
                std::tie(b0, b1) = simd::btfly(a0, a1);
                std::tie(b2, b3) = simd::btfly<3>(a2, a3);
            }
        }
        if constexpr (Scale) {
            auto& scaling = std::get<simd::reg_t<T>&>(std::tie(args...));
            b0            = simd::mul(b0, scaling);
            b1            = simd::mul(b1, scaling);
            b2            = simd::mul(b2, scaling);
            b3            = simd::mul(b3, scaling);
        }
        std::tie(b0, b1, b2, b3)  = simd::inverse<Inverse>(b0, b1, b2, b3);
        auto [b0_, b1_, b2_, b3_] = simd::repack2<PDest>(b0, b1, b2, b3);

        cxstore<PDest>(dest[data_idx[0]], b0_);
        cxstore<PDest>(dest[data_idx[1]], b1_);
        cxstore<PDest>(dest[data_idx[2]], b2_);
        cxstore<PDest>(dest[data_idx[3]], b3_);
    };
};
template<>
struct node<8> {
    template<typename T, uZ PDest, uZ PSrc, bool ConjTw, bool Reverse, bool DIT = false, typename... Args>
    static inline void perform(std::array<T*, 8> dest, Args... args) {
        constexpr auto reg_size = simd::reg<T>::size;
        using cx_reg            = simd::cx_reg<T, false, reg_size>;

        constexpr bool Inverse = ConjTw || Reverse;

        using src_type       = std::array<const T*, 8>;
        using tw_type        = std::array<simd::cx_reg<T>, 7>;
        constexpr bool Src   = has_type<src_type, Args...>;
        constexpr bool Tw    = has_type<tw_type, Args...>;
        constexpr bool Scale = has_type<simd::reg_t<T>, Args...>;

        constexpr auto& data_idx = order<8, DIT>::data;
        constexpr auto& tw_idx   = order<8, DIT>::tw;

        if constexpr (Reverse) {
            simd::cx_reg<T, false, PSrc> c4_, c5_, c6_, c7_;
            if constexpr (Src) {
                auto& src = std::get<src_type&>(std::tie(args...));
                c4_       = simd::cxload<PSrc>(src[data_idx[4]]);
                c5_       = simd::cxload<PSrc>(src[data_idx[5]]);
                c6_       = simd::cxload<PSrc>(src[data_idx[6]]);
                c7_       = simd::cxload<PSrc>(src[data_idx[7]]);
            } else {
                c4_ = simd::cxload<PSrc>(dest[data_idx[4]]);
                c5_ = simd::cxload<PSrc>(dest[data_idx[5]]);
                c6_ = simd::cxload<PSrc>(dest[data_idx[6]]);
                c7_ = simd::cxload<PSrc>(dest[data_idx[7]]);
            }
            auto [c4, c5, c6, c7] = simd::repack2<reg_size>(c4_, c5_, c6_, c7_);

            std::tie(c4, c5, c6, c7) = simd::inverse<Inverse>(c4, c5, c6, c7);
            cx_reg a4, a5, a6, a7;
            if constexpr (Tw) {
                auto& tw        = std::get<tw_type&>(std::tie(args...));
                auto [b4, b5tw] = simd::ibtfly(c4, c5);
                auto [b6, b7tw] = simd::ibtfly(c6, c7);
                auto [b5, b7]   = simd::mul({b5tw, tw[tw_idx[5]]}, {b7tw, tw[tw_idx[6]]});

                std::tie(a4, a6) = simd::ibtfly(b4, b6);
                std::tie(a5, a7) = simd::ibtfly(b5, b7);
                std::tie(a6, a7) = simd::mul({a6, tw[tw_idx[2]]}, {a7, tw[tw_idx[2]]});
            } else {
                const T sq2      = std::sqrt(double{2}) / 2;
                auto [b4, b5_tw] = simd::ibtfly(c4, c5);
                auto [b6, b7_tw] = simd::ibtfly<2>(c6, c7);
                auto twsq2       = simd::broadcast(sq2);
                b5_tw            = simd::mul(b5_tw, twsq2);
                b7_tw            = simd::mul(b7_tw, twsq2);

                std::tie(a4, a6) = simd::ibtfly<3>(b4, b6);
                auto b5 = cx_reg{simd::add(b5_tw.real, b5_tw.imag), simd::sub(b5_tw.imag, b5_tw.real)};
                auto b7 = cx_reg{simd::sub(b7_tw.real, b7_tw.imag), simd::add(b7_tw.real, b7_tw.imag)};
                std::tie(a5, a7) = simd::ibtfly<3>(b5, b7);
            }
            simd::cx_reg<T, false, PSrc> c0_, c1_, c2_, c3_;
            if constexpr (Src) {
                auto& src = std::get<src_type&>(std::tie(args...));
                c0_       = simd::cxload<PSrc>(src[data_idx[0]]);
                c1_       = simd::cxload<PSrc>(src[data_idx[1]]);
                c2_       = simd::cxload<PSrc>(src[data_idx[2]]);
                c3_       = simd::cxload<PSrc>(src[data_idx[3]]);
            } else {
                c0_ = simd::cxload<PSrc>(dest[data_idx[0]]);
                c1_ = simd::cxload<PSrc>(dest[data_idx[1]]);
                c2_ = simd::cxload<PSrc>(dest[data_idx[2]]);
                c3_ = simd::cxload<PSrc>(dest[data_idx[3]]);
            }
            auto [c0, c1, c2, c3]    = simd::repack2<reg_size>(c0_, c1_, c2_, c3_);
            std::tie(c0, c1, c2, c3) = simd::inverse<Inverse>(c0, c1, c2, c3);

            cx_reg p0, p2, p4, p6, a1, a3;
            if constexpr (Tw) {
                auto& tw        = std::get<tw_type&>(std::tie(args...));
                auto [b0, b1tw] = simd::ibtfly(c0, c1);
                auto [b2, b3tw] = simd::ibtfly(c2, c3);
                auto [b1, b3]   = simd::mul({b1tw, tw[tw_idx[3]]}, {b3tw, tw[tw_idx[4]]});

                auto [a0, a2]    = simd::ibtfly(b0, b2);
                std::tie(a1, a3) = simd::ibtfly(b1, b3);
                std::tie(a2, a3) = simd::mul({a2, tw[tw_idx[1]]}, {a3, tw[tw_idx[1]]});

                std::tie(p0, p4) = simd::ibtfly(a0, a4);
                std::tie(p2, p6) = simd::ibtfly(a2, a6);
                std::tie(p4, p6) = simd::mul({p4, tw[tw_idx[0]]}, {p6, tw[tw_idx[0]]});
            } else {
                auto [b0, b1] = simd::ibtfly(c0, c1);
                auto [b2, b3] = simd::ibtfly<3>(c2, c3);

                auto [a0, a2]    = simd::ibtfly(b0, b2);
                std::tie(a1, a3) = simd::ibtfly(b1, b3);

                std::tie(p0, p4) = simd::ibtfly(a0, a4);
                std::tie(p2, p6) = simd::ibtfly(a2, a6);
            }
            if constexpr (Scale) {
                auto& scaling = std::get<simd::reg_t<T>&>(std::tie(args...));
                p0            = simd::mul(p0, scaling);
                p4            = simd::mul(p4, scaling);
                p2            = simd::mul(p2, scaling);
                p6            = simd::mul(p6, scaling);
            }
            std::tie(p0, p4, p2, p6) = simd::inverse<Inverse>(p0, p4, p2, p6);
            std::tie(p0, p4, p2, p6) = simd::repack2<PDest>(p0, p4, p2, p6);

            simd::cxstore<PDest>(dest[data_idx[0]], p0);
            simd::cxstore<PDest>(dest[data_idx[4]], p4);
            simd::cxstore<PDest>(dest[data_idx[2]], p2);
            simd::cxstore<PDest>(dest[data_idx[6]], p6);

            auto [p1, p5] = simd::ibtfly(a1, a5);
            auto [p3, p7] = simd::ibtfly(a3, a7);
            if constexpr (Tw) {
                auto& tw         = std::get<tw_type&>(std::tie(args...));
                std::tie(p5, p7) = simd::mul({p5, tw[tw_idx[0]]}, {p7, tw[tw_idx[0]]});
            }
            if constexpr (Scale) {
                auto& scaling = std::get<simd::reg_t<T>&>(std::tie(args...));
                p1            = simd::mul(p1, scaling);
                p5            = simd::mul(p5, scaling);
                p3            = simd::mul(p3, scaling);
                p7            = simd::mul(p7, scaling);
            }
            std::tie(p1, p5, p3, p7) = simd::inverse<Inverse>(p1, p5, p3, p7);
            std::tie(p1, p5, p3, p7) = simd::repack2<PDest>(p1, p5, p3, p7);

            simd::cxstore<PDest>(dest[data_idx[1]], p1);
            simd::cxstore<PDest>(dest[data_idx[5]], p5);
            simd::cxstore<PDest>(dest[data_idx[3]], p3);
            simd::cxstore<PDest>(dest[data_idx[7]], p7);
        } else {
            simd::cx_reg<T, false, PSrc> p1_, p3_, p5_, p7_;
            if constexpr (Src) {
                auto& src = std::get<src_type&>(std::tie(args...));
                p5_       = simd::cxload<PSrc>(src[data_idx[5]]);
                p1_       = simd::cxload<PSrc>(src[data_idx[1]]);
                p7_       = simd::cxload<PSrc>(src[data_idx[7]]);
                p3_       = simd::cxload<PSrc>(src[data_idx[3]]);
            } else {
                p5_ = simd::cxload<PSrc>(dest[data_idx[5]]);
                p1_ = simd::cxload<PSrc>(dest[data_idx[1]]);
                p7_ = simd::cxload<PSrc>(dest[data_idx[7]]);
                p3_ = simd::cxload<PSrc>(dest[data_idx[3]]);
            }
            auto [p5, p1, p7, p3]    = simd::repack2<reg_size>(p5_, p1_, p7_, p3_);
            std::tie(p5, p1, p7, p3) = simd::inverse<Inverse>(p5, p1, p7, p3);

            cx_reg b1, b3, b5, b7;
            if constexpr (Tw) {
                auto& tw          = std::get<tw_type&>(std::tie(args...));
                auto [p5tw, p7tw] = simd::mul({p5, tw[tw_idx[0]]}, {p7, tw[tw_idx[0]]});
                auto [a1, a5]     = simd::btfly(p1, p5tw);
                auto [a3, a7]     = simd::btfly(p3, p7tw);

                auto [a3tw, a7tw] = simd::mul({a3, tw[tw_idx[1]]}, {a7, tw[tw_idx[2]]});
                std::tie(b1, b3)  = simd::btfly(a1, a3tw);
                std::tie(b5, b7)  = simd::btfly(a5, a7tw);

                std::tie(b1, b3, b5, b7) = simd::mul(
                    {b1, tw[tw_idx[3]]}, {b3, tw[tw_idx[4]]}, {b5, tw[tw_idx[5]]}, {b7, tw[tw_idx[6]]});
            } else {
                const T sq2   = std::sqrt(double{2}) / 2;
                auto [a1, a5] = simd::btfly(p1, p5);
                auto [a3, a7] = simd::btfly(p3, p7);

                std::tie(b1, b3) = simd::btfly(a1, a3);
                std::tie(b5, b7) = simd::btfly<3>(a5, a7);
                auto b5_tw       = simd::cx_reg<T>{simd::add(b5.real, b5.imag), simd::sub(b5.imag, b5.real)};
                auto b7_tw       = simd::cx_reg<T>{simd::sub(b7.real, b7.imag), simd::add(b7.real, b7.imag)};
                auto twsq2       = simd::broadcast(sq2);
                b5               = simd::mul(b5_tw, twsq2);
                b7               = simd::mul(b7_tw, twsq2);
            }
            simd::cx_reg<T, false, PSrc> p0_, p2_, p4_, p6_;
            if constexpr (Src) {
                auto& src = std::get<src_type&>(std::tie(args...));
                p4_       = simd::cxload<PSrc>(src[data_idx[4]]);
                p0_       = simd::cxload<PSrc>(src[data_idx[0]]);
                p6_       = simd::cxload<PSrc>(src[data_idx[6]]);
                p2_       = simd::cxload<PSrc>(src[data_idx[2]]);
            } else {
                p4_ = simd::cxload<PSrc>(dest[data_idx[4]]);
                p0_ = simd::cxload<PSrc>(dest[data_idx[0]]);
                p6_ = simd::cxload<PSrc>(dest[data_idx[6]]);
                p2_ = simd::cxload<PSrc>(dest[data_idx[2]]);
            }
            auto [p4, p0, p6, p2]    = simd::repack2<reg_size>(p4_, p0_, p6_, p2_);
            std::tie(p4, p0, p6, p2) = simd::inverse<Inverse>(p4, p0, p6, p2);

            simd::cx_reg<T> c0, c1, c2, c3, b4, b6;
            if constexpr (Tw) {
                auto& tw          = std::get<tw_type&>(std::tie(args...));
                auto [p4tw, p6tw] = simd::mul({p4, tw[tw_idx[0]]}, {p6, tw[tw_idx[0]]});
                auto [a0, a4]     = simd::btfly(p0, p4tw);
                auto [a2, a6]     = simd::btfly(p2, p6tw);

                auto [a2tw, a6tw] = simd::mul({a2, tw[tw_idx[1]]}, {a6, tw[tw_idx[2]]});
                auto [b0, b2]     = simd::btfly(a0, a2tw);
                std::tie(b4, b6)  = simd::btfly(a4, a6tw);

                std::tie(c0, c1) = simd::btfly(b0, b1);
                std::tie(c2, c3) = simd::btfly(b2, b3);
            } else {
                auto [a0, a4] = simd::btfly(p0, p4);
                auto [a2, a6] = simd::btfly(p2, p6);

                auto [b0, b2]    = simd::btfly(a0, a2);
                std::tie(b4, b6) = simd::btfly<3>(a4, a6);

                std::tie(c0, c1) = simd::btfly(b0, b1);
                std::tie(c2, c3) = simd::btfly<3>(b2, b3);
            }
            if constexpr (Scale) {
                auto& scaling = std::get<simd::reg_t<T>&>(std::tie(args...));
                c0            = simd::mul(c0, scaling);
                c1            = simd::mul(c1, scaling);
                c2            = simd::mul(c2, scaling);
                c3            = simd::mul(c3, scaling);
            }
            std::tie(c0, c1, c2, c3)  = simd::inverse<Inverse>(c0, c1, c2, c3);
            auto [c0_, c1_, c2_, c3_] = simd::repack2<PDest>(c0, c1, c2, c3);

            cxstore<PDest>(dest[data_idx[0]], c0_);
            cxstore<PDest>(dest[data_idx[1]], c1_);
            cxstore<PDest>(dest[data_idx[2]], c2_);
            cxstore<PDest>(dest[data_idx[3]], c3_);

            simd::cx_reg<T> c4, c5, c6, c7;
            if constexpr (Tw) {
                std::tie(c4, c5) = simd::btfly(b4, b5);
                std::tie(c6, c7) = simd::btfly(b6, b7);
            } else {
                std::tie(c4, c5) = simd::btfly(b4, b5);
                std::tie(c6, c7) = simd::btfly<2>(b6, b7);
            }
            if constexpr (Scale) {
                auto& scaling = std::get<simd::reg_t<T>&>(std::tie(args...));
                c4            = simd::mul(c4, scaling);
                c5            = simd::mul(c5, scaling);
                c6            = simd::mul(c6, scaling);
                c7            = simd::mul(c7, scaling);
            }
            std::tie(c4, c5, c6, c7)  = simd::inverse<Inverse>(c4, c5, c6, c7);
            auto [c4_, c5_, c6_, c7_] = simd::repack2<PDest>(c4, c5, c6, c7);

            cxstore<PDest>(dest[data_idx[4]], c4_);
            cxstore<PDest>(dest[data_idx[5]], c5_);
            cxstore<PDest>(dest[data_idx[6]], c6_);
            cxstore<PDest>(dest[data_idx[7]], c7_);
        }
    };
};


}    // namespace fft
}    // namespace detail_

template<typename V>
struct cx_vector_traits {
    using real_type               = decltype([] {});
    static constexpr uZ pack_size = 0;
};

template<typename R>
    requires rv::contiguous_range<R> && detail_::is_std_complex_floating_point<rv::range_value_t<R>>::value
struct cx_vector_traits<R> {
    using real_type = typename detail_::is_std_complex_floating_point<rv::range_value_t<R>>::real_type;
    static constexpr uZ pack_size = 1;

    static auto re_data(R& vector) {
        return reinterpret_cast<real_type*>(rv::data(vector));
    }
    static auto re_data(const R& vector) {
        return reinterpret_cast<const real_type*>(rv::data(vector));
    }
    static auto size(const R& vector) {
        return rv::size(vector);
    }
};

template<typename T_, uZ PackSize_, typename Alloc_>
struct cx_vector_traits<pcx::vector<T_, PackSize_, Alloc_>> {
    using real_type               = T_;
    static constexpr uZ pack_size = PackSize_;

    static auto re_data(pcx::vector<T_, PackSize_, Alloc_>& vector) {
        return vector.data();
    }
    static auto re_data(const pcx::vector<T_, PackSize_, Alloc_>& vector) {
        return vector.data();
    }
    static auto size(const pcx::vector<T_, PackSize_, Alloc_>& vector) {
        return vector.size();
    }
};

template<typename T_, bool Const_, uZ PackSize_>
struct cx_vector_traits<pcx::subrange<T_, Const_, PackSize_>> {
    using real_type               = T_;
    static constexpr uZ pack_size = PackSize_;

    static auto re_data(pcx::subrange<T_, false, PackSize_> subrange) {
        if (!subrange.aligned()) {
            throw(std::invalid_argument(std::string(
                "subrange is not aligned. pcx::subrange must be aligned to be accessed as a vector")));
        }
        return &(*subrange.begin());
    }
    static auto size(pcx::subrange<T_, Const_, PackSize_> subrange) {
        return subrange.size();
    }
};

template<typename T, typename V>
concept complex_vector_of = std::same_as<T, typename cx_vector_traits<V>::real_type>;

template<typename T, typename R>
concept range_complex_vector_of = rv::random_access_range<R> &&    //
                                  complex_vector_of<T, std::remove_pointer_t<rv::range_value_t<R>>>;

/**
 * @brief Controls how fft output and ifft input is ordered;
 * This may have an impact on performance of fft/ifft;
 *
 * normal:          default, elements are in ascending frequency order, starting from zero
 * bit_reversed:    element indexes are in a bit-reversed order
 * unordered:
 */
enum class fft_order {
    normal,
    bit_reversed,
    unordered
};

struct strategy42 {
    static constexpr uZ node_size = 4;

    static constexpr std::array<uZ, 2> align_node_size{2, 2};
    static constexpr std::array<uZ, 2> align_node_count{1, 1};
};
struct strategy48 {
    static constexpr uZ node_size = 4;

    static constexpr std::array<uZ, 2> align_node_size{8, 2};
    static constexpr std::array<uZ, 2> align_node_count{1, 1};
};
struct strategy8 {
    static constexpr uZ node_size = 8;

    static constexpr std::array<uZ, 2> align_node_size{4, 2};
    static constexpr std::array<uZ, 3> align_node_count{1, 1, 1};
};

template<typename T,
         fft_order Order      = fft_order::normal,
         typename Allocator   = std::allocator<T>,
         uZ SubSize           = pcx::dynamic_size,
         uZ NodeSizeRec       = 4,
         typename Strategy    = strategy42,
         typename StrategyRec = strategy42>
    requires(std::same_as<T, float> || std::same_as<T, double>) &&
            (pcx::pack_size<SubSize> && SubSize >= pcx::default_pack_size<T> ||
             (SubSize == pcx::dynamic_size))
class fft_unit {
public:
    using real_type      = T;
    using allocator_type = Allocator;

private:
    using size_type              = uZ;
    using subsize_t              = std::conditional_t<SubSize == pcx::dynamic_size, uZ, decltype([]() {})>;
    using sort_allocator_type    = typename std::allocator_traits<allocator_type>::template rebind_alloc<uZ>;
    static constexpr bool sorted = Order == fft_order::normal;

    using sort_t    = std::conditional_t<sorted,    //
                                      std::vector<uZ, sort_allocator_type>,
                                      decltype([]() {})>;
    using twiddle_t = std::conditional_t<sorted,
                                         pcx::vector<real_type, simd::reg<T>::size, allocator_type>,
                                         std::vector<real_type, allocator_type>>;

public:
    explicit fft_unit(uZ fft_size, allocator_type allocator = allocator_type())
        requires(SubSize != pcx::dynamic_size)
    : m_size(check_size(fft_size))
    , m_sort(get_sort(size(), static_cast<sort_allocator_type>(allocator)))
    , m_twiddles(get_twiddles(size(), sub_size(), allocator)){};

    explicit fft_unit(uZ fft_size, uZ sub_size = 2048, allocator_type allocator = allocator_type())
        requires(SubSize == pcx::dynamic_size)
    : m_size(check_size(fft_size))
    , m_sub_size(check_sub_size(sub_size))
    , m_sort(get_sort(size(), static_cast<sort_allocator_type>(allocator)))
    , m_twiddles(get_twiddles(size(), sub_size, allocator)){};

    fft_unit(const fft_unit& other)     = default;
    fft_unit(fft_unit&& other) noexcept = default;

    ~fft_unit() = default;

    fft_unit& operator=(const fft_unit& other)     = delete;
    fft_unit& operator=(fft_unit&& other) noexcept = delete;

    [[nodiscard]] constexpr auto size() const -> size_type {
        return m_size;
    }
    /**
     * @brief Performs in place FFT.
     *
     * @tparam PackSize
     * @param[in, out] data
     */
    template<uZ PackSize = 1>
        requires pack_size<PackSize>
    void fft_raw(T* data) {
        if constexpr (!sorted) {
            apply_unsorted<PackSize, PackSize, false, false>(data);
        } else {
            constexpr auto PTform = std::max(PackSize, simd::reg<T>::size);
            // depth3_and_sort<PTform, PackSize, false>(data);
            // size_specific::template tform_sort<PTform, PackSize, false>(data, size(), m_sort);
            simd::size_specific::tform_sort<PTform, PackSize, false>(data, size(), m_sort);
            apply_subtform<PackSize, PTform, false, false>(data, size());
        }
    }
    /**
     * @brief Perfomrs out of place FFT.
     *
     * @tparam PackSizeDest
     * @tparam PackSizeSrc
     * @param[out]  dest
     * @param[in]   source
     */
    template<uZ PackSizeDest = 1, uZ PackSizeSrc = 1>
        requires pack_size<PackSizeDest> && pack_size<PackSizeSrc>
    void fft_raw(T* dest, const T* source) {
        if constexpr (!sorted) {
            apply_unsorted<PackSizeDest, PackSizeSrc, false, false>(dest, source, size());
        } else {
            constexpr auto PTform = std::max(PackSizeDest, simd::reg<T>::size);
            if (dest == source) {
                simd::size_specific::tform_sort<PTform, PackSizeDest, false>(dest, size(), m_sort);
                apply_subtform<PackSizeDest, PTform, false, false>(dest, size());
            } else {
                simd::size_specific::tform_sort<PTform, PackSizeSrc, false>(dest, source, size(), m_sort);
                apply_subtform<PackSizeDest, PTform, false, false>(dest, size());
            }
        }
    }
    /**
     * @brief Performs zero-extended out of place FFT.
     *
     * @tparam PackSizeDest
     * @tparam PackSizeSrc
     * @param[out]  dest
     * @param[in]   source
     * @param[in]   source_size Source size, must be lower or equal to fft size.
       Values after source_size are zero-extended.
     */
    template<uZ PackSizeDest = 1, uZ PackSizeSrc = 1>
        requires pack_size<PackSizeDest> && pack_size<PackSizeSrc>
    void fft_raw(T* dest, const T* source, uZ source_size) {
        if constexpr (!sorted) {
            apply_unsorted<PackSizeDest, PackSizeSrc, false, false>(dest, source, source_size);
        } else {
            constexpr auto PTform = std::max(PackSizeDest, simd::reg<T>::size);
            simd::size_specific::tform_sort<PTform, PackSizeSrc, false>(dest, source, source_size, m_sort);
            apply_subtform<PackSizeDest, PTform, false, false>(dest, size());
        }
    }
    /**
     * @brief Performs in place IFFT.
     *
     * @tparam Normalized If true output is normalized by 1/<fft size>.
     * @tparam PackSize Data pack size.
     * @param[in, out] data
     */
    template<bool Normalized = true, uZ PackSize = 1>
        requires pack_size<PackSize>
    void ifft_raw(T* data) {
        if constexpr (!sorted) {
            apply_unsorted<PackSize, PackSize, true, Normalized>(data);
        } else {
            constexpr auto PTform = std::max(PackSize, simd::reg<T>::size);
            // depth3_and_sort<PTform, PackSize, false>(data);
            // size_specific::template tform_sort<PTform, PackSize, false>(data, size(), m_sort);
            simd::size_specific::tform_sort<PTform, PackSize, false>(data, size(), m_sort);
            apply_subtform<PackSize, PTform, true, Normalized>(data, size());
        }
    }
    /**
     * @brief Performs out of place IFFT.
     *
     * @tparam Normalized If true output is normilized by 1/<fft size>.
     * @tparam PackSizeDest Destination pack size.
     * @tparam PackSizeSrc Source paxk size.
     * @param[out] dest
     * @param[in] source
     */
    template<bool Normalized = true, uZ PackSizeDest = 1, uZ PackSizeSrc = 1>
        requires pack_size<PackSizeDest> && pack_size<PackSizeSrc>
    void ifft_raw(T* dest, const T* source) {
        if constexpr (!sorted) {
            // TODO:Add source size
            apply_unsorted<PackSizeDest, PackSizeSrc, true, Normalized>(dest, source, size());
        } else {
            simd::size_specific::tform_sort<PackSizeDest, PackSizeSrc, true>(dest, source, size(), m_sort);
            apply_subtform<PackSizeDest, PackSizeSrc, true, Normalized>(dest, size());
        }
    }
    /**
     * @brief Performs in place FFT.
     *
     * @tparam Vect_ Must satisfy complex_vector_of<T>.
     * @param[in, out] vector
     */
    template<typename Vect_>
        requires complex_vector_of<T, Vect_>
    void operator()(Vect_& vector) {
        using v_traits = cx_vector_traits<Vect_>;
        if (v_traits::size(vector) != m_size) {
            throw(std::invalid_argument(std::string("input size (which is ")
                                            .append(std::to_string(v_traits::size(vector)))
                                            .append(") is not equal to fft size (which is ")
                                            .append(std::to_string(m_size))
                                            .append(")")));
        }
        constexpr auto PData = v_traits::pack_size;
        if constexpr (!sorted) {
            // fftu_internal<PData>(v_traits::re_data(vector));
            apply_unsorted<PData, PData, false, false>(v_traits::re_data(vector));
        } else {
            constexpr auto PTform = std::max(PData, simd::reg<T>::size);
            // depth3_and_sort<PTform, PData, false>(v_traits::re_data(vector));
            // size_specific::template tform_sort<PTform, PData, false>(v_traits::re_data(vector), size(), m_sort);
            simd::size_specific::tform_sort<PTform, PData, false>(v_traits::re_data(vector), size(), m_sort);
            apply_subtform<PData, PTform, false, false>(v_traits::re_data(vector), size());
        }
    }
    /**
     * @brief Performs out of place FFT.
     Source values are potentially zero-extended.
     * @tparam DestVect_
     * @tparam SrcVect_
     * @param[out]  dest
     * @param[in]   source
     */
    template<typename DestVect_, typename SrcVect_>
        requires complex_vector_of<T, DestVect_> && complex_vector_of<T, SrcVect_>
    void operator()(DestVect_& dest, const SrcVect_& source) {
        using src_traits = cx_vector_traits<SrcVect_>;
        using dst_traits = cx_vector_traits<DestVect_>;
        if (dst_traits::size(dest) != m_size) {
            throw(std::invalid_argument(std::string("destination size (which is ")
                                            .append(std::to_string(dst_traits::size(dest)))
                                            .append(") is not equal to fft size (which is ")
                                            .append(std::to_string(size()))
                                            .append(")")));
        }
        if (src_traits::size(source) > m_size) {
            throw(std::invalid_argument(std::string("source size (which is ")
                                            .append(std::to_string(src_traits::size(source)))
                                            .append(") is bigger than fft size (which is ")
                                            .append(std::to_string(size()))
                                            .append(")")));
        }
        constexpr auto PDest = dst_traits::pack_size;
        constexpr auto PSrc  = src_traits::pack_size;

        auto dst_ptr = dst_traits::re_data(dest);
        auto src_ptr = src_traits::re_data(source);
        if constexpr (PSrc != PDest && std::max(PSrc, PDest) > simd::reg<T>::size) {
            if (dst_ptr == src_ptr) {
                throw(std::invalid_argument(
                    std::string("cannot perform in-place repack from source pack size (which is ")
                        .append(std::to_string(PSrc))
                        .append(") to destination pack size (which is ")
                        .append(std::to_string(PDest))
                        .append(")")));
            }
        }
        if constexpr (!sorted) {
            apply_unsorted<PDest, PSrc, false, false>(dst_ptr, src_ptr, src_traits::size(source));
            // if (src_traits::size(source) < size()) {
            //     fftu_internal<PDest, PSrc>(dst_ptr, src_ptr, src_traits::size(source));
            // } else {
            //     fftu_internal<PDest, PSrc>(dst_ptr, src_ptr);
            // }
        } else {
            // TODO:  source upsize
            constexpr auto PTform = std::max(PDest, simd::reg<T>::size);
            if (dst_ptr == src_ptr) {
                // size_specific::template tform_sort<PTform, PDest, false>(
                //     dst_traits::re_data(dest), size(), m_sort);

                simd::size_specific::tform_sort<PTform, PDest, false>(
                    dst_traits::re_data(dest), size(), m_sort);
                // depth3_and_sort<PTform, PDest, false>(dst_traits::re_data(dest));
                apply_subtform<PDest, PTform, false, false>(dst_traits::re_data(dest), size());
            } else {
                // size_specific::template tform_sort<PTform, PDest, false>(
                //     dst_traits::re_data(dest), src_traits::re_data(source), size(), m_sort);
                simd::size_specific::tform_sort<PTform, PSrc, false>(
                    dst_traits::re_data(dest), src_traits::re_data(source), size(), m_sort);
                // depth3_and_sort<PTform, PSrc, false>(dst_traits::re_data(dest));
                apply_subtform<PDest, PTform, false, false>(dst_traits::re_data(dest), size());
            }
        }
    }
    /**
     * @brief Performs in place IFFT.
     *
     * @tparam Normalized If true output is normalized by 1/<fft size>.
     * @tparam Vect_
     * @param[in, out] vector
     */
    template<bool Normalized = true, typename Vect_>
        requires complex_vector_of<T, Vect_>
    void ifft(Vect_& vector) {
        using v_traits = cx_vector_traits<Vect_>;
        if (v_traits::size(vector) != m_size) {
            throw(std::invalid_argument(std::string("input size (which is ")
                                            .append(std::to_string(v_traits::size(vector)))
                                            .append(" is not equal to fft size (which is ")
                                            .append(std::to_string(m_size))
                                            .append(")")));
        }
        constexpr auto PData = v_traits::pack_size;
        if constexpr (!sorted) {
            // ifftu_internal<PData>(v_traits::re_data(vector));

            apply_unsorted<PData, PData, true, Normalized>(v_traits::re_data(vector));
        } else {
            constexpr auto PTform = std::max(PData, simd::reg<T>::size);

            // size_specific::template tform_sort<PTform, PData, true>(v_traits::re_data(vector), size(), m_sort);
            simd::size_specific::tform_sort<PTform, PData, true>(v_traits::re_data(vector), size(), m_sort);

            // depth3_and_sort<PTform, PData, true>(v_traits::re_data(vector));
            apply_subtform<PData, PTform, true, true>(v_traits::re_data(vector), size());
        }
    }
    /**
     * @brief Performs out of place IFFT.
     *
     * @tparam Normalized If true output is normalized by 1/<fft size>.
     * @tparam DestVect_
     * @tparam SrcVect_
     * @param[out]  dest
     * @param[in]   source
     */
    template<bool Normalized = true, typename DestVect_, typename SrcVect_>
        requires complex_vector_of<T, DestVect_> && complex_vector_of<T, SrcVect_>
    void ifft(DestVect_& dest, const SrcVect_& source) {
        using src_traits = cx_vector_traits<SrcVect_>;
        using dst_traits = cx_vector_traits<DestVect_>;
        if (dst_traits::size(dest) != m_size) {
            throw(std::invalid_argument(std::string("destination size (which is ")
                                            .append(std::to_string(dst_traits::size(dest)))
                                            .append(") is not equal to fft size (which is ")
                                            .append(std::to_string(size()))
                                            .append(")")));
        }
        if (src_traits::size(source) != m_size) {
            throw(std::invalid_argument(std::string("source size (which is ")
                                            .append(std::to_string(src_traits::size(source)))
                                            .append(") is not equal to fft size (which is ")
                                            .append(std::to_string(size()))
                                            .append(")")));
        }

        constexpr auto PDest = dst_traits::pack_size;
        constexpr auto PSrc  = src_traits::pack_size;

        auto dst_ptr = dst_traits::re_data(dest);
        auto src_ptr = src_traits::re_data(source);

        if constexpr (PSrc != PDest && std::max(PSrc, PDest) > simd::reg<T>::size) {
            if (dst_ptr == src_ptr) {
                throw(std::invalid_argument(
                    std::string("cannot perform in-place repack from source pack size (which is ")
                        .append(std::to_string(PSrc))
                        .append(") to destination pack size (which is ")
                        .append(std::to_string(PDest))
                        .append(")")));
            }
        }

        if constexpr (!sorted) {
            apply_unsorted<PDest, PSrc, true, Normalized>(dst_ptr, src_ptr, src_traits::size(source));
        } else {
            // TODO:  source upsize
            constexpr auto PTform = std::max(PDest, simd::reg<T>::size);
            if (dst_ptr == src_ptr) {
                simd::size_specific::tform_sort<PTform, PDest, true>(
                    dst_traits::re_data(dest), size(), m_sort);
                apply_subtform<PDest, PTform, true, Normalized>(dst_traits::re_data(dest), size());
            } else {
                simd::size_specific::tform_sort<PTform, PSrc, true>(
                    dst_traits::re_data(dest), src_traits::re_data(source), size(), m_sort);
                apply_subtform<PDest, PTform, true, Normalized>(dst_traits::re_data(dest), size());
            }
        }
    }

private:
    const size_type                 m_size;
    [[no_unique_address]] subsize_t m_sub_size;
    [[no_unique_address]] sort_t    m_sort;
    const twiddle_t                 m_twiddles;

    size_type m_align_count     = 0;
    size_type m_align_count_rec = 0;
    size_type m_subtform_idx    = get_subtform_idx_strategic(m_size,
                                                          sub_size(),
                                                          sorted ? simd::size_specific::sorted_size<T>
                                                                    : simd::size_specific::unsorted_size<T>);

    [[nodiscard]] constexpr auto sub_size() const -> size_type {
        if constexpr (SubSize == pcx::dynamic_size) {
            return m_sub_size;
        } else {
            return SubSize;
        }
    }

public:
    template<uZ PDest, uZ PTform, bool Inverse, bool Scale, uZ AlignSize>
    inline auto subtform(T* data, uZ max_size, uZ align_count) -> const T* {
        namespace d_             = detail_;
        constexpr auto node_size = Strategy::node_size;

        const auto* twiddle_ptr = m_twiddles.data();

        uZ l_size     = simd::size_specific::sorted_size<T> * 2;
        uZ group_size = max_size / simd::size_specific::sorted_size<T>;
        uZ n_groups   = 1;

        if constexpr (AlignSize > 1) {
            constexpr auto align_ce = uZ_constant<AlignSize>{};
            if constexpr (Scale) {
                group_size /= AlignSize;
                auto scaling = simd::broadcast(static_cast<T>(1 / static_cast<double>(size())));

                for (uZ i_group = 0; i_group < n_groups; ++i_group) {
                    uZ offset = i_group * simd::reg<T>::size;

                    auto tw = []<uZ... I>(auto tw_ptr, std::index_sequence<I...>) {
                        return std::array<simd::cx_reg<T>, sizeof...(I)>{
                            simd::cxload<simd::reg<T>::size>(tw_ptr + simd::reg<T>::size * 2 * I)...};
                    }(twiddle_ptr, std::make_index_sequence<AlignSize - 1>{});
                    twiddle_ptr += simd::reg<T>::size * 2 * (AlignSize - 1);
                    for (uZ i = 0; i < group_size; ++i) {
                        node_along<AlignSize, PTform, PTform, true, Inverse>(
                            data, l_size * (AlignSize / 2), offset, tw, scaling);
                        offset += l_size * (AlignSize / 2);
                    }
                }

                l_size *= AlignSize;
                n_groups *= AlignSize;
            }
            uZ i = Scale ? 1 : 0;
            for (; i < align_count; ++i) {
                group_size /= AlignSize;

                for (uZ i_group = 0; i_group < n_groups; ++i_group) {
                    uZ offset = i_group * simd::reg<T>::size;

                    auto tw = []<uZ... I>(auto tw_ptr, std::index_sequence<I...>) {
                        return std::array<simd::cx_reg<T>, sizeof...(I)>{
                            simd::cxload<simd::reg<T>::size>(tw_ptr + simd::reg<T>::size * 2 * I)...};
                    }(twiddle_ptr, std::make_index_sequence<AlignSize - 1>{});
                    twiddle_ptr += simd::reg<T>::size * 2 * (AlignSize - 1);
                    for (uZ i = 0; i < group_size; ++i) {
                        node_along<AlignSize, PTform, PTform, true, Inverse>(
                            data, l_size * (AlignSize / 2), offset, tw);
                        offset += l_size * (AlignSize / 2);
                    }
                }

                l_size *= AlignSize;
                n_groups *= AlignSize;
            }
        } else if constexpr (Scale) {
            group_size /= node_size;
            auto scaling = simd::broadcast(static_cast<T>(1 / static_cast<double>(size())));

            for (uZ i_group = 0; i_group < n_groups; ++i_group) {
                uZ offset = i_group * simd::reg<T>::size;

                auto tw = []<uZ... I>(auto tw_ptr, std::index_sequence<I...>) {
                    return std::array<simd::cx_reg<T>, sizeof...(I)>{
                        simd::cxload<simd::reg<T>::size>(tw_ptr + simd::reg<T>::size * 2 * I)...};
                }(twiddle_ptr, std::make_index_sequence<node_size - 1>{});
                twiddle_ptr += simd::reg<T>::size * 2 * (node_size - 1);
                for (uZ i = 0; i < group_size; ++i) {
                    node_along<node_size, PTform, PTform, true, Inverse>(
                        data, l_size * (node_size / 2), offset, tw, scaling);
                    offset += l_size * (node_size / 2);
                }
            }

            l_size *= node_size;
            n_groups *= node_size;
        }

        if constexpr (PDest != PTform) {
            max_size /= node_size;
        }
        while (l_size <= max_size) {
            group_size /= node_size;
            for (uZ i_group = 0; i_group < n_groups; ++i_group) {
                uZ offset = i_group * simd::reg<T>::size;

                auto tw = []<uZ... I>(auto tw_ptr, std::index_sequence<I...>) {
                    return std::array<simd::cx_reg<T>, sizeof...(I)>{
                        simd::cxload<simd::reg<T>::size>(tw_ptr + simd::reg<T>::size * 2 * I)...};
                }(twiddle_ptr, std::make_index_sequence<node_size - 1>{});
                twiddle_ptr += simd::reg<T>::size * 2 * (node_size - 1);
                for (uZ i = 0; i < group_size; ++i) {
                    node_along<node_size, PTform, PTform, true, Inverse>(
                        data, l_size * (node_size / 2), offset, tw);
                    offset += l_size * (node_size / 2);
                }
            }
            l_size *= node_size;
            n_groups *= node_size;
        }
        if constexpr (PDest != PTform) {
            group_size /= node_size;

            for (uZ i_group = 0; i_group < n_groups; ++i_group) {
                uZ offset = i_group * simd::reg<T>::size;

                auto tw = []<uZ... I>(auto tw_ptr, std::index_sequence<I...>) {
                    return std::array<simd::cx_reg<T>, sizeof...(I)>{
                        simd::cxload<simd::reg<T>::size>(tw_ptr + simd::reg<T>::size * 2 * I)...};
                }(twiddle_ptr, std::make_index_sequence<node_size - 1>{});
                twiddle_ptr += simd::reg<T>::size * 2 * (node_size - 1);
                for (uZ i = 0; i < group_size; ++i) {
                    node_along<node_size, PDest, PTform, true, Inverse>(
                        data, l_size * (node_size / 2), offset, tw);
                    offset += l_size * (node_size / 2);
                }
            }
        }
        return twiddle_ptr;
    };

    template<uZ PDest, uZ PTform, bool Inverse, bool Scale, uZ AlignSize, uZ AlignSizeR>
    inline auto subtform_recursive(T* data, uZ size, uZ align_count, uZ align_rec_count) -> const T* {
        if (size <= sub_size()) {
            return subtform<PDest, PTform, Inverse, Scale, AlignSize>(data, size, align_count);
        }
        if constexpr (AlignSizeR > 1) {
            if (align_rec_count > 0) {
                const T* twiddle_ptr;
                for (uZ i = 0; i < AlignSizeR; ++i) {
                    twiddle_ptr = subtform_recursive<PTform, PTform, Inverse, false, AlignSize, AlignSizeR>(
                        simd::ra_addr<PTform>(data, size * i / AlignSizeR),
                        size / AlignSizeR,
                        align_count,
                        align_rec_count - 1);
                }
                uZ   n_groups = size / simd::reg<T>::size / AlignSizeR;
                auto scaling  = [](uZ size) {
                    if constexpr (Scale) {
                        return simd::broadcast(static_cast<T>(1. / static_cast<double>(size)));
                    } else {
                        return [] {};
                    }
                }(size);
                for (uZ i_group = 0; i_group < n_groups; ++i_group) {
                    auto tw = []<uZ... Itw>(auto* tw_ptr, std::index_sequence<Itw...>) {
                        return std::array<simd::cx_reg<T>, sizeof...(Itw)>{
                            simd::cxload<simd::reg<T>::size>(tw_ptr + simd::reg<T>::size * 2 * Itw)...};
                    }(twiddle_ptr, std::make_index_sequence<AlignSizeR - 1>{});
                    twiddle_ptr += simd::reg<T>::size * 2 * (AlignSizeR - 1);
                    node_along<AlignSizeR, PDest, PTform, true, Inverse>(
                        data, size, i_group * simd::reg<T>::size, tw, scaling);
                }
                return twiddle_ptr;
            }
        }
        constexpr auto node_size = StrategyRec::node_size;
        const T*       twiddle_ptr;
        for (uZ i = 0; i < node_size; ++i) {
            twiddle_ptr = subtform_recursive<PTform, PTform, Inverse, false, AlignSize, 0>(
                simd::ra_addr<PTform>(data, size * i / node_size), size / node_size, align_count, 0);
        }

        uZ   n_groups = size / simd::reg<T>::size / node_size;
        auto scaling  = [](uZ size) {
            if constexpr (Scale) {
                return simd::broadcast(static_cast<T>(1. / static_cast<double>(size)));
            } else {
                return [] {};
            }
        }(size);
        for (uZ i_group = 0; i_group < n_groups; ++i_group) {
            auto tw = []<uZ... Itw>(auto* tw_ptr, std::index_sequence<Itw...>) {
                return std::array<simd::cx_reg<T>, sizeof...(Itw)>{
                    simd::cxload<simd::reg<T>::size>(tw_ptr + simd::reg<T>::size * 2 * Itw)...};
            }(twiddle_ptr, std::make_index_sequence<node_size - 1>{});
            twiddle_ptr += simd::reg<T>::size * 2 * (node_size - 1);
            node_along<node_size, PDest, PTform, true, Inverse>(
                data, size, i_group * simd::reg<T>::size, tw, scaling);
        }
        return twiddle_ptr;
    };

    template<uZ PDest, uZ PTform, bool Inverse = false, bool Scale = false>
    inline auto apply_subtform(T* data, uZ size) {
        static constexpr auto subtform_array = []() {
            auto add_first_zero = []<uZ... I>(std::array<uZ, sizeof...(I)> sizes, std::index_sequence<I...>) {
                return std::array<uZ, sizeof...(I) + 1>{0, sizes[I]...};
            };
            using subtform_t     = const T* (fft_unit::*)(T*, uZ, uZ, uZ);
            constexpr auto align = add_first_zero(
                Strategy::align_node_size, std::make_index_sequence<Strategy::align_node_size.size()>{});
            constexpr auto align_rec =
                add_first_zero(StrategyRec::align_node_size,
                               std::make_index_sequence<StrategyRec::align_node_size.size()>{});
            constexpr auto n_combine_rem = align.size() * align_rec.size();

            std::array<subtform_t, n_combine_rem> subtform_table{};

            constexpr auto fill_rec = [=]<uZ AlignSize, uZ... I>(    //
                                          subtform_t* begin,
                                          uZ_constant<AlignSize>,
                                          std::index_sequence<I...>) {
                ((*(begin + I) =
                      &fft_unit::subtform_recursive<PDest, PTform, Inverse, Scale, AlignSize, align_rec[I]>),
                 ...);
            };
            [=]<uZ... I>(subtform_t* begin, std::index_sequence<I...>) {
                fill_rec(begin, uZ_constant<0>{}, std::make_index_sequence<align_rec.size()>{});
                (fill_rec(begin + (I + 1) * align_rec.size(),
                          uZ_constant<Strategy::align_node_size[I]>{},
                          std::make_index_sequence<align_rec.size()>{}),
                 ...);
            }(subtform_table.data(), std::make_index_sequence<Strategy::align_node_size.size()>{});
            return subtform_table;
        }();
        (this->*subtform_array[m_subtform_idx])(data, size, m_align_count, m_align_count_rec);
    };

    template<uZ PDest, uZ PSrc, bool First, uZ AlignSize>
    inline auto usubtform(T*       dest,    //
                          uZ       size,
                          const T* twiddle_ptr,
                          uZ       align_count = 0,
                          auto... optional) -> const T* {
        constexpr uZ reg_size = simd::reg<T>::size;
        using cx_reg          = simd::cx_reg<T, false, reg_size>;

        constexpr auto PTform    = std::max(PDest, reg_size);
        constexpr auto node_size = Strategy::node_size;

        using source_type  = const T*;
        constexpr bool Src = detail_::has_type<source_type, decltype(optional)...>;

        uZ l_size   = size;
        uZ n_groups = 1;

        if constexpr (AlignSize > 1) {
            uZ i_align = 0;
            if constexpr (PSrc < PTform || Src) {
                auto tw = []<uZ... I>(const T* tw_ptr, std::index_sequence<I...>) {
                    if constexpr (First) {
                        return [] {};
                    } else {
                        return std::array<cx_reg, sizeof...(I)>{
                            cx_reg{simd::broadcast(tw_ptr + I * 2), simd::broadcast(tw_ptr + I * 2 + 1)}
                            ...
                        };
                    }
                }(twiddle_ptr, std::make_index_sequence<AlignSize - 1>{});
                twiddle_ptr += 2 * (AlignSize - 1);

                for (uZ i = 0; i < l_size / (reg_size * AlignSize); ++i) {
                    node_along<AlignSize, PTform, PSrc, false>(dest, l_size, i * reg_size, tw, optional...);
                }
                l_size /= AlignSize;
                n_groups *= AlignSize;
                ++i_align;
            }

            constexpr bool countable_misalign = Strategy::align_node_size.size() > 1;
            if constexpr (countable_misalign || !(PSrc < PTform || Src)) {
                for (; i_align < align_count; ++i_align) {
                    uZ i_group = 0;
                    if constexpr (First) {
                        twiddle_ptr += 2 * (AlignSize - 1);
                        for (uZ i = 0; i < l_size / (reg_size * AlignSize); ++i) {
                            node_along<AlignSize, PTform, PTform, false>(dest, l_size, i * reg_size);
                        }
                        ++i_group;
                    }
                    for (; i_group < n_groups; ++i_group) {
                        auto tw = []<uZ... I>(const T* tw_ptr, std::index_sequence<I...>) {
                            return std::array<cx_reg, sizeof...(I)>{
                                cx_reg{simd::broadcast(tw_ptr + I * 2),
                                       simd::broadcast(tw_ptr + I * 2 + 1)}
                                ...
                            };
                        }(twiddle_ptr, std::make_index_sequence<AlignSize - 1>{});
                        twiddle_ptr += 2 * (AlignSize - 1);
                        auto* group_ptr = simd::ra_addr<PTform>(dest, i_group * l_size);
                        for (uZ i = 0; i < l_size / (reg_size * AlignSize); ++i) {
                            node_along<AlignSize, PTform, PTform, false>(group_ptr, l_size, i * reg_size, tw);
                        }
                    }
                    l_size /= AlignSize;
                    n_groups *= AlignSize;
                }
            }
        } else if constexpr (PSrc < PTform || Src) {
            auto tw = []<uZ... I>(const T* tw_ptr, std::index_sequence<I...>) {
                if constexpr (First) {
                    return [] {};
                } else {
                    return std::array<cx_reg, sizeof...(I)>{
                        cx_reg{simd::broadcast(tw_ptr + I * 2), simd::broadcast(tw_ptr + I * 2 + 1)}
                        ...
                    };
                }
            }(twiddle_ptr, std::make_index_sequence<node_size - 1>{});
            twiddle_ptr += 2 * (node_size - 1);

            for (uZ i = 0; i < l_size / (reg_size * node_size); ++i) {
                node_along<node_size, PTform, PSrc, false>(dest, l_size, i * reg_size, tw, optional...);
            }
            l_size /= node_size;
            n_groups *= node_size;
        }

        while (l_size > simd::size_specific::unsorted_size<T>) {
            uZ i_group = 0;
            if constexpr (First) {
                twiddle_ptr += 2 * (node_size - 1);
                for (uZ i = 0; i < l_size / (reg_size * node_size); ++i) {
                    node_along<node_size, PTform, PTform, false>(dest, l_size, i * reg_size);
                }
                ++i_group;
            }
            for (; i_group < n_groups; ++i_group) {
                auto tw = []<uZ... I>(const T* tw_ptr, std::index_sequence<I...>) {
                    return std::array<cx_reg, sizeof...(I)>{
                        cx_reg{simd::broadcast(tw_ptr + I * 2), simd::broadcast(tw_ptr + I * 2 + 1)}
                        ...
                    };
                }(twiddle_ptr, std::make_index_sequence<node_size - 1>{});

                twiddle_ptr += 2 * (node_size - 1);
                auto* group_ptr = simd::ra_addr<PTform>(dest, i_group * l_size);
                for (uZ i = 0; i < l_size / (reg_size * node_size); ++i) {
                    node_along<node_size, PTform, PTform, false>(group_ptr, l_size, i * reg_size, tw);
                }
            }
            l_size /= node_size;
            n_groups *= node_size;
        }

        return simd::size_specific::unsorted<PDest, PTform, Order == fft_order::bit_reversed>(
            dest, twiddle_ptr, size);
    };

    template<uZ PDest, uZ PSrc, bool First, bool Scale, uZ AlignSize>
    inline auto usubtform_reverse(T*       dest,    //
                                  uZ       size,
                                  const T* twiddle_ptr,
                                  uZ       align_count,
                                  auto... optional) -> const T* {
        constexpr auto reg_size = simd::reg<T>::size;
        using cx_reg            = simd::cx_reg<T, false, reg_size>;

        constexpr auto PTform    = std::max(PDest, reg_size);
        constexpr auto node_size = Strategy::node_size;

        twiddle_ptr =
            simd::size_specific::unsorted_reverse<PTform, PSrc, Scale, Order == fft_order::bit_reversed>(
                dest, twiddle_ptr, size, this->size(), optional...);

        uZ l_size   = simd::size_specific::unsorted_size<T>;
        uZ n_groups = size / l_size;

        if constexpr (AlignSize > 1) {
            constexpr bool countable_misalign = Strategy::align_node_size.size() > 1;
            if constexpr (countable_misalign) {
                size /= detail_::fft::powi(AlignSize, align_count);
            } else {
                size /= AlignSize;
            }
        } else if constexpr (PDest < PTform) {
            size /= node_size;
        }

        while (l_size < size) {
            l_size *= node_size;
            n_groups /= node_size;
            constexpr auto min_g = First ? 1 : 0;
            for (iZ i_group = n_groups - 1; i_group >= min_g; --i_group) {
                twiddle_ptr -= 2 * (node_size - 1);
                auto tw = []<uZ... I>(const T* tw_ptr, std::index_sequence<I...>) {
                    return std::array<cx_reg, sizeof...(I)>{
                        cx_reg{simd::broadcast(tw_ptr + I * 2), simd::broadcast(tw_ptr + I * 2 + 1)}
                        ...
                    };
                }(twiddle_ptr, std::make_index_sequence<node_size - 1>{});

                auto* group_ptr = simd::ra_addr<PTform>(dest, i_group * l_size);
                for (uZ i = 0; i < l_size / (reg_size * node_size); ++i) {
                    node_along<node_size, PTform, PTform, false, false, true>(
                        group_ptr, l_size, i * reg_size, tw);
                }
            }
            if constexpr (First) {
                twiddle_ptr -= 2 * (node_size - 1);
                for (uZ i = 0; i < l_size / (reg_size * node_size); ++i) {
                    node_along<node_size, PTform, PTform, false, false, true>(dest, l_size, i * reg_size);
                }
            }
        }

        if constexpr (AlignSize > 1) {
            constexpr bool countable_misalign = Strategy::align_node_size.size() > 1;
            if constexpr (countable_misalign) {
                constexpr auto min_align = PDest < PTform ? 1 : 0;
                for (iZ i_align = align_count - 1; i_align >= min_align; --i_align) {
                    l_size *= AlignSize;
                    n_groups /= AlignSize;
                    constexpr auto min_g = First ? 1 : 0;
                    for (iZ i_group = n_groups - 1; i_group >= min_g; --i_group) {
                        twiddle_ptr -= 2 * (AlignSize - 1);
                        auto tw = []<uZ... I>(const T* tw_ptr, std::index_sequence<I...>) {
                            return std::array<cx_reg, sizeof...(I)>{
                                cx_reg{simd::broadcast(tw_ptr + I * 2),
                                       simd::broadcast(tw_ptr + I * 2 + 1)}
                                ...
                            };
                        }(twiddle_ptr, std::make_index_sequence<AlignSize - 1>{});

                        auto* group_ptr = simd::ra_addr<PTform>(dest, i_group * l_size);
                        for (uZ i = 0; i < l_size / (reg_size * AlignSize); ++i) {
                            node_along<AlignSize, PTform, PTform, false, false, true>(
                                group_ptr, l_size, i * reg_size, tw);
                        }
                    }
                    if constexpr (First) {
                        twiddle_ptr -= 2 * (AlignSize - 1);
                        for (uZ i = 0; i < l_size / (reg_size * AlignSize); ++i) {
                            node_along<AlignSize, PTform, PTform, false, false, true>(
                                dest, l_size, i * reg_size);
                        }
                    }
                }
            }
            if constexpr (PDest < PTform || !countable_misalign) {
                l_size *= AlignSize;
                n_groups /= AlignSize;
                constexpr auto min_g = First ? 1 : 0;
                for (iZ i_group = n_groups - 1; i_group >= min_g; --i_group) {
                    twiddle_ptr -= 2 * (AlignSize - 1);
                    auto tw = []<uZ... I>(const T* tw_ptr, std::index_sequence<I...>) {
                        return std::array<cx_reg, sizeof...(I)>{
                            cx_reg{simd::broadcast(tw_ptr + I * 2), simd::broadcast(tw_ptr + I * 2 + 1)}
                            ...
                        };
                    }(twiddle_ptr, std::make_index_sequence<AlignSize - 1>{});

                    auto* group_ptr = simd::ra_addr<PTform>(dest, i_group * l_size);
                    for (uZ i = 0; i < l_size / (reg_size * AlignSize); ++i) {
                        node_along<AlignSize, PDest, PTform, false, false, true>(
                            group_ptr, l_size, i * reg_size, tw);
                    }
                }
                if constexpr (First) {
                    twiddle_ptr -= 2 * (AlignSize - 1);
                    for (uZ i = 0; i < l_size / (reg_size * AlignSize); ++i) {
                        node_along<AlignSize, PDest, PTform, false, false, true>(dest, l_size, i * reg_size);
                    }
                }
            }
        } else if constexpr (PDest < PTform) {
            l_size *= node_size;
            n_groups /= node_size;
            constexpr auto min_g = First ? 1 : 0;
            for (iZ i_group = n_groups - 1; i_group >= min_g; --i_group) {
                twiddle_ptr -= 2 * (node_size - 1);
                auto tw = []<uZ... I>(const T* tw_ptr, std::index_sequence<I...>) {
                    return std::array<cx_reg, sizeof...(I)>{
                        cx_reg{simd::broadcast(tw_ptr + I * 2), simd::broadcast(tw_ptr + I * 2 + 1)}
                        ...
                    };
                }(twiddle_ptr, std::make_index_sequence<node_size - 1>{});

                auto* group_ptr = simd::ra_addr<PTform>(dest, i_group * l_size);
                for (uZ i = 0; i < l_size / (reg_size * node_size); ++i) {
                    node_along<node_size, PDest, PTform, false, false, true>(
                        group_ptr, l_size, i * reg_size, tw);
                }
            }
            if constexpr (First) {
                twiddle_ptr -= 2 * (node_size - 1);
                for (uZ i = 0; i < l_size / (reg_size * node_size); ++i) {
                    node_along<node_size, PDest, PTform, false, false, true>(dest, l_size, i * reg_size);
                }
            }
        }

        return twiddle_ptr;
    };

    template<uZ   PDest,
             uZ   PSrc,
             bool First,
             uZ   AlignSize,
             uZ   AlignSizeRec>
    inline auto usubtform_recursive(T*       dest,    //
                                    uZ       size,
                                    const T* twiddle_ptr,
                                    uZ       align_count,
                                    uZ       align_count_rec,
                                    auto... optional) -> const T* {
        if (size <= sub_size()) {
            return usubtform<PDest, PSrc, First, AlignSize>(
                dest, size, twiddle_ptr, align_count, optional...);
        }
        constexpr auto PTform    = std::max(PDest, simd::reg<T>::size);
        constexpr auto node_size = StrategyRec::node_size;

        if constexpr (AlignSizeRec > 1) {
            constexpr bool countable_misalign = StrategyRec::align_node_size.size() > 1;
            constexpr auto next_align         = countable_misalign ? AlignSizeRec : 0;
            if (!countable_misalign || align_count_rec > 0) {
                auto tw = []<uZ... I>(const T* tw_ptr, std::index_sequence<I...>) {
                    if constexpr (First) {
                        return [] {};
                    } else {
                        return std::array<simd::cx_reg<T>, sizeof...(I)>{
                            simd::cx_reg<T>{simd::broadcast(tw_ptr + I * 2),
                                            simd::broadcast(tw_ptr + I * 2 + 1)}
                            ...
                        };
                    }
                }(twiddle_ptr, std::make_index_sequence<AlignSizeRec - 1>{});
                twiddle_ptr += 2 * (AlignSizeRec - 1);
                for (uZ i_group = 0; i_group < size / AlignSizeRec / simd::reg<T>::size; ++i_group) {
                    node_along<AlignSizeRec, PTform, PSrc, false, false, false>(
                        dest, size, i_group * simd::reg<T>::size, tw, optional...);
                }

                twiddle_ptr = usubtform_recursive<PDest, PTform, First, AlignSize, next_align>(
                    dest, size / AlignSizeRec, twiddle_ptr, align_count, align_count_rec - 1);
                for (uZ i = 1; i < AlignSizeRec; ++i) {
                    twiddle_ptr = usubtform_recursive<PDest, PTform, false, AlignSize, next_align>(
                        simd::ra_addr<PTform>(dest, i * size / AlignSizeRec),
                        size / AlignSizeRec,
                        twiddle_ptr,
                        align_count,
                        align_count_rec - 1);
                };
                return twiddle_ptr;
            }
        }
        auto tw = []<uZ... I>(const T* tw_ptr, std::index_sequence<I...>) {
            if constexpr (First) {
                return [] {};
            } else {
                return std::array<simd::cx_reg<T>, sizeof...(I)>{
                    simd::cx_reg<T>{simd::broadcast(tw_ptr + I * 2), simd::broadcast(tw_ptr + I * 2 + 1)}
                    ...
                };
            }
        }(twiddle_ptr, std::make_index_sequence<node_size - 1>{});
        twiddle_ptr += 2 * (node_size - 1);
        for (uZ i_group = 0; i_group < size / node_size / simd::reg<T>::size; ++i_group) {
            node_along<node_size, PTform, PSrc, false, false, false>(
                dest, size, i_group * simd::reg<T>::size, tw, optional...);
        }

        twiddle_ptr = usubtform_recursive<PDest, PTform, First, AlignSize, 0>(
            dest, size / node_size, twiddle_ptr, align_count, 0);
        for (uZ i = 1; i < node_size; ++i) {
            twiddle_ptr = usubtform_recursive<PDest, PTform, false, AlignSize, 0>(
                simd::ra_addr<PTform>(dest, i * size / node_size),
                size / node_size,
                twiddle_ptr,
                align_count,
                0);
        };
        return twiddle_ptr;
    };
    template<uZ PDest, uZ PSrc, bool First, uZ AlignSize, uZ AlignSizeRec>
    inline auto usubtform_recursive_src(T*       dest,
                                        uZ       size,
                                        const T* twiddle_ptr,
                                        uZ       align_count,
                                        const uZ align_count_rec,
                                        const T* src,
                                        uZ       src_size) {
        if (src_size < size) {
            return usubtform_recursive<PDest, PSrc, First, AlignSize, AlignSizeRec>(
                dest, size, twiddle_ptr, align_count, align_count_rec, src, src_size);
        } else {
            return usubtform_recursive<PDest, PSrc, First, AlignSize, AlignSizeRec>(
                dest, size, twiddle_ptr, align_count, align_count_rec, src);
        }
    }

    template<uZ   PDest,
             uZ   PSrc,
             bool First,
             bool Scale,
             uZ   AlignSize,
             uZ   AlignSizeRec>
    inline auto usubtform_recursive_reverse(T*       dest,    //
                                            uZ       size,
                                            const T* twiddle_ptr,
                                            uZ       align_count,
                                            uZ       align_count_rec,
                                            auto... optional) -> const T* {
        if (size <= sub_size()) {
            return usubtform_reverse<PDest, PSrc, First, Scale, AlignSize>(
                dest, size, twiddle_ptr, align_count, optional...);
        }
        constexpr auto PTform    = std::max(PDest, simd::reg<T>::size);
        constexpr auto node_size = StrategyRec::node_size;

        if constexpr (AlignSizeRec > 1) {
            constexpr bool countable_misalign = StrategyRec::align_node_size.size() > 1;
            constexpr auto next_align         = countable_misalign ? AlignSizeRec : 0;
            if (!countable_misalign || align_count_rec > 0) {
                for (uZ i = AlignSizeRec - 1; i > 0; --i) {
                    twiddle_ptr =
                        usubtform_recursive_reverse<PTform, PSrc, false, Scale, AlignSize, next_align>(
                            simd::ra_addr<PSrc>(dest, i * size / AlignSizeRec),
                            size / AlignSizeRec,
                            twiddle_ptr,
                            align_count,
                            align_count_rec - 1);
                };
                twiddle_ptr = usubtform_recursive_reverse<PTform, PSrc, First, Scale, AlignSize, next_align>(
                    dest, size / AlignSizeRec, twiddle_ptr, align_count, align_count_rec - 1);

                twiddle_ptr -= 2 * (AlignSizeRec - 1);
                auto tw = []<uZ... I>(const T* tw_ptr, std::index_sequence<I...>) {
                    if constexpr (First) {
                        return [] {};
                    } else {
                        return std::array<simd::cx_reg<T>, sizeof...(I)>{
                            simd::cx_reg<T>{simd::broadcast(tw_ptr + I * 2),
                                            simd::broadcast(tw_ptr + I * 2 + 1)}
                            ...
                        };
                    }
                }(twiddle_ptr, std::make_index_sequence<AlignSizeRec - 1>{});
                for (uZ i_group = 0; i_group < size / AlignSizeRec / simd::reg<T>::size; ++i_group) {
                    node_along<AlignSizeRec, PDest, PTform, false, false, true>(
                        dest, size, i_group * simd::reg<T>::size, tw, optional...);
                }
                return twiddle_ptr;
            }
        }
        for (uZ i = node_size - 1; i > 0; --i) {
            twiddle_ptr = usubtform_recursive_reverse<PTform, PSrc, false, Scale, AlignSize, 0>(
                simd::ra_addr<PSrc>(dest, i * size / node_size),
                size / node_size,
                twiddle_ptr,
                align_count,
                0);
        };
        twiddle_ptr = usubtform_recursive_reverse<PTform, PSrc, First, Scale, AlignSize, 0>(
            dest, size / node_size, twiddle_ptr, align_count, 0);

        twiddle_ptr -= 2 * (node_size - 1);
        auto tw = []<uZ... I>(const T* tw_ptr, std::index_sequence<I...>) {
            if constexpr (First) {
                return [] {};
            } else {
                return std::array<simd::cx_reg<T>, sizeof...(I)>{
                    simd::cx_reg<T>{simd::broadcast(tw_ptr + I * 2), simd::broadcast(tw_ptr + I * 2 + 1)}
                    ...
                };
            }
        }(twiddle_ptr, std::make_index_sequence<node_size - 1>{});
        for (uZ i_group = 0; i_group < size / node_size / simd::reg<T>::size; ++i_group) {
            node_along<node_size, PDest, PTform, false, false, true>(
                dest, size, i_group * simd::reg<T>::size, tw, optional...);
        }

        return twiddle_ptr;
    };

    template<uZ   PDest,
             uZ   PSrc,
             bool First,
             bool Scale,
             uZ   AlignSize,
             uZ   AlignSizeRec>
    inline auto usubtform_recursive_reverse_src(T*       dest,    //
                                                uZ       size,
                                                const T* twiddle_ptr,
                                                uZ       align_count,
                                                uZ       align_count_rec,
                                                const T* src,
                                                uZ = {}) -> const T* {
        return usubtform_recursive_reverse<PDest, PSrc, First, Scale, AlignSize, AlignSizeRec>(
            dest, size, twiddle_ptr, align_count, align_count_rec, src);
    };

    template<uZ PDest, uZ PSrc, bool Reverse, bool Normalized>
    inline auto apply_unsorted(T* data) {
        static constexpr auto subtform_array = []() {
            auto add_first_zero = []<uZ... I>(std::array<uZ, sizeof...(I)> sizes, std::index_sequence<I...>) {
                return std::array<uZ, sizeof...(I) + 1>{0, sizes[I]...};
            };
            using subtform_t     = const T* (fft_unit::*)(T*, uZ, const T*, uZ, uZ);
            constexpr auto align = add_first_zero(
                Strategy::align_node_size, std::make_index_sequence<Strategy::align_node_size.size()>{});
            constexpr auto align_rec =
                add_first_zero(StrategyRec::align_node_size,
                               std::make_index_sequence<StrategyRec::align_node_size.size()>{});
            constexpr auto n_combine_rem = align.size() * align_rec.size();

            std::array<subtform_t, n_combine_rem> subtform_table{};

            constexpr auto fill_rec = [=]<uZ AlignSize, uZ... I>(    //
                                          subtform_t* begin,
                                          uZ_constant<AlignSize>,
                                          std::index_sequence<I...>) {
                if (Reverse)
                    ((*(begin + I) = &fft_unit::usubtform_recursive_reverse<PDest,
                                                                            PSrc,
                                                                            true,
                                                                            Normalized,
                                                                            AlignSize,
                                                                            align_rec[I]>),
                     ...);
                else
                    ((*(begin + I) =
                          &fft_unit::usubtform_recursive<PDest, PSrc, true, AlignSize, align_rec[I]>),
                     ...);
            };
            [=]<uZ... I>(subtform_t* begin, std::index_sequence<I...>) {
                fill_rec(begin, uZ_constant<0>{}, std::make_index_sequence<align_rec.size()>{});
                (fill_rec(begin + (I + 1) * align_rec.size(),
                          uZ_constant<Strategy::align_node_size[I]>{},
                          std::make_index_sequence<align_rec.size()>{}),
                 ...);
            }(subtform_table.data(), std::make_index_sequence<Strategy::align_node_size.size()>{});
            return subtform_table;
        }();

        if constexpr (Reverse) {
            (this->*subtform_array[m_subtform_idx])(
                data, size(), &*m_twiddles.end(), m_align_count, m_align_count_rec);
        } else {
            (this->*subtform_array[m_subtform_idx])(
                data, size(), m_twiddles.data(), m_align_count, m_align_count_rec);
        }
    };

    template<uZ PDest, uZ PSrc, bool Reverse, bool Normalized>
    inline auto apply_unsorted(T* data, const T* src, uZ src_size) {
        static constexpr auto subtform_array = []() {
            auto add_first_zero = []<uZ... I>(std::array<uZ, sizeof...(I)> sizes, std::index_sequence<I...>) {
                return std::array<uZ, sizeof...(I) + 1>{0, sizes[I]...};
            };
            using subtform_t     = const T* (fft_unit::*)(T*, uZ, const T*, uZ, uZ, const T*, uZ);
            constexpr auto align = add_first_zero(
                Strategy::align_node_size, std::make_index_sequence<Strategy::align_node_size.size()>{});
            constexpr auto align_rec =
                add_first_zero(StrategyRec::align_node_size,
                               std::make_index_sequence<StrategyRec::align_node_size.size()>{});
            constexpr auto n_combine_rem = align.size() * align_rec.size();

            std::array<subtform_t, n_combine_rem> subtform_table{};

            constexpr auto fill_rec = [=]<uZ AlignSize, uZ... I>(    //
                                          subtform_t* begin,
                                          uZ_constant<AlignSize>,
                                          std::index_sequence<I...>) {
                if (Reverse)
                    ((*(begin + I) = &fft_unit::usubtform_recursive_reverse_src<PDest,
                                                                                PSrc,
                                                                                true,
                                                                                Normalized,
                                                                                AlignSize,
                                                                                align_rec[I]>),
                     ...);
                else
                    ((*(begin + I) =
                          &fft_unit::usubtform_recursive_src<PDest, PSrc, true, AlignSize, align_rec[I]>),
                     ...);
            };
            [=]<uZ... I>(subtform_t* begin, std::index_sequence<I...>) {
                fill_rec(begin, uZ_constant<0>{}, std::make_index_sequence<align_rec.size()>{});
                (fill_rec(begin + (I + 1) * align_rec.size(),
                          uZ_constant<Strategy::align_node_size[I]>{},
                          std::make_index_sequence<align_rec.size()>{}),
                 ...);
            }(subtform_table.data(), std::make_index_sequence<Strategy::align_node_size.size()>{});
            return subtform_table;
        }();

        if constexpr (Reverse) {
            (this->*subtform_array[m_subtform_idx])(
                data, size(), &*m_twiddles.end(), m_align_count, m_align_count_rec, src, 0);
        } else {
            (this->*subtform_array[m_subtform_idx])(
                data, size(), m_twiddles.data(), m_align_count, m_align_count_rec, src, src_size);
        }
    };

    template<uZ   NodeSize,
             uZ   PDest,
             uZ   PSrc,
             bool DecInTime,
             bool ConjTw  = false,
             bool Reverse = false,
             typename... Optional>
    inline static void node_along(T* dest, uZ l_size, uZ offset, Optional... optional) {
        constexpr auto PLoad  = std::max(PSrc, simd::reg<T>::size);
        constexpr auto PStore = std::max(PDest, simd::reg<T>::size);

        using source_type     = const T*;
        constexpr bool Src    = detail_::has_type<source_type, Optional...>;
        constexpr bool Upsize = Src && detail_::has_type<uZ, Optional...>;

        constexpr auto perform = [](auto&&... args) {
            detail_::fft::node<NodeSize>::template perform<T, PDest, PSrc, ConjTw, Reverse, DecInTime>(
                std::forward<decltype(args)>(args)...);
        };

        constexpr auto Idxs = std::make_index_sequence<NodeSize>{};
        constexpr auto get_data_array =
            []<uZ PackSize, uZ... I>(
                auto data, auto offset, auto l_size, uZ_constant<PackSize>, std::index_sequence<I...>) {
                constexpr auto Size = sizeof...(I);
                return std::array<decltype(data), Size>{
                    simd::ra_addr<PackSize>(data, offset + l_size / Size * I)...};
            };

        auto dst = get_data_array(dest, offset, l_size, uZ_constant<PDest>{}, Idxs);

        if constexpr (Src) {
            auto source = std::get<source_type&>(std::tie(optional...));
            if constexpr (Upsize) {
                auto data_size = std::get<uZ&>(std::tie(optional...));

                if (offset + l_size / NodeSize * (NodeSize - 1) + simd::reg<T>::size <= data_size) {
                    auto src = get_data_array(source, offset, l_size, uZ_constant<PSrc>{}, Idxs);
                    perform(dst, src, optional...);
                } else {
                    std::array<T, simd::reg<T>::size * 2> zeros{};

                    auto src = []<uZ... I>(const T* ptr, std::index_sequence<I...>) {
                        constexpr auto Size = sizeof...(I);
                        return std::array<const T*, Size>{ptr + 0 * I...};
                    }(zeros.data(), Idxs);

                    for (uZ i = 0; i < NodeSize; ++i) {
                        auto           l_offset = offset + l_size / NodeSize * i;
                        std::ptrdiff_t diff     = data_size - l_offset - 1;
                        if (diff >= static_cast<int>(simd::reg<T>::size)) {
                            src[i] = simd::ra_addr<PLoad>(source, l_offset);
                        } else if (diff >= 0) {
                            std::array<T, simd::reg<T>::size * 2> align{};
                            for (; diff >= 0; --diff) {
                                align[diff] = *(source + pidx<PSrc>(l_offset + diff));
                                align[diff + simd::reg<T>::size] =
                                    *(source + pidx<PSrc>(l_offset + diff) + PSrc);
                            }
                            src[i] = align.data();
                            perform(dst, src, optional...);
                            return;
                        } else {
                            break;
                        }
                    }
                    perform(dst, src, optional...);
                };
            } else {
                auto src = get_data_array(source, offset, l_size, uZ_constant<PSrc>{}, Idxs);
                perform(dst, src, optional...);
            }
        } else {
            perform(dst, optional...);
        }
    };

private:
    static constexpr auto check_size(uZ size) -> uZ {
        if (size > 1 && (size & (size - 1)) == 0) {
            return size;
        }
        throw(std::invalid_argument("fft_size (which is  " + std::to_string(size) +
                                    ") is not an integer power of two"));
    }

    static constexpr auto check_sub_size(uZ sub_size) -> uZ {
        if (sub_size >= pcx::default_pack_size<T> && (sub_size & (sub_size - 1)) == 0) {
            return sub_size;
        } else if (sub_size < pcx::default_pack_size<T>) {
            throw(std::invalid_argument("fft_sub_size (which is  " + std::to_string(sub_size) +
                                        ") is smaller than minimum (which is " +
                                        std::to_string(pcx::default_pack_size<T>) + ")"));
        } else if ((sub_size & (sub_size - 1)) != 0) {
            throw(std::invalid_argument("fft_sub_size (which is  " + std::to_string(sub_size) +
                                        ") is not an integer power of two"));
        }
        return 0;
    }

    static auto get_sort(uZ fft_size, sort_allocator_type allocator) -> std::vector<uZ, sort_allocator_type>
        requires(sorted)
    {
        using detail_::fft::reverse_bit_order;

        const auto packed_sort_size =
            fft_size / simd::size_specific::sorted_size<T> / simd::size_specific::sorted_size<T>;
        const auto order = detail_::fft::log2i(packed_sort_size);
        auto       sort  = std::vector<uZ, sort_allocator_type>(allocator);
        sort.reserve(packed_sort_size);

        for (uZ i = 0; i < packed_sort_size; ++i) {
            if (i >= reverse_bit_order(i, order)) {
                continue;
            }
            sort.push_back(i);
            sort.push_back(reverse_bit_order(i, order));
        }
        for (uZ i = 0; i < packed_sort_size; ++i) {
            if (i == reverse_bit_order(i, order)) {
                sort.push_back(i);
            }
        }
        return sort;
    }
    static auto get_sort(uZ fft_size, sort_allocator_type allocator)
        requires(!sorted)
    {
        return sort_t{};
    };

    static auto get_subtform_idx_strategic_(uZ fft_size, uZ sub_size, uZ specific_size) {
        using detail_::fft::log2i;
        constexpr auto n_rem     = log2i(Strategy::node_size);
        constexpr auto n_rem_rec = log2i(StrategyRec::node_size);

        auto sub_size_ = std::min(fft_size, sub_size);

        auto get_cost = []<typename Strategy_>(auto size, Strategy_) {
            uZ node_count = 0;
            uZ weight     = 1;
            using namespace detail_::fft;


            auto misalign = log2i(size) % log2i(Strategy_::node_size);
            if (misalign == 0) {
                auto target_node_count = log2i(size) / log2i(Strategy_::node_size);
                auto target_weight     = powi(log2i(Strategy_::node_size), target_node_count);
                return std::pair(target_node_count, target_weight);
            }

            uZ align_idx   = 0;
            uZ align_count = 0;

            using namespace detail_::fft;
            for (; align_idx < Strategy_::align_node_size.size(); ++align_idx) {
                auto i_align_size = Strategy_::align_node_size[align_idx];
                auto l_size       = size;
                align_count       = 0;
                while (l_size > 1 && log2i(l_size) % log2i(Strategy::node_size) != 0) {
                    l_size /= i_align_size;
                    ++align_count;
                }
                if (l_size > 0) {
                    auto target_node_count = log2i(l_size) / log2i(Strategy_::node_size);
                    auto target_weight     = powi(log2i(Strategy_::node_size), target_node_count);
                    node_count += target_node_count;
                    weight *= target_weight;
                    break;
                }
            }
            weight *= powi(log2i(Strategy_::align_node_size[align_idx]), align_count);
            node_count += align_count;

            return std::pair(node_count, weight);
        };

        auto rec_size = fft_size / sub_size_;

        auto rec_cost  = get_cost(rec_size, StrategyRec{});
        auto targ_cost = get_cost(sub_size_ / specific_size, Strategy{});

        auto count1  = rec_cost.first + targ_cost.first;
        auto weight1 = rec_cost.second * targ_cost.second;

        rec_cost  = get_cost(rec_size * 2, StrategyRec{});
        targ_cost = get_cost(sub_size_ / 2 / specific_size, Strategy{});

        auto count2  = rec_cost.first + targ_cost.first;
        auto weight2 = rec_cost.second * targ_cost.second;

        if ((count2 < count1) || (count2 == count1) && (weight2 > weight1)) {
            sub_size_ /= 2;
        }
        auto misalign    = log2i(sub_size_ / specific_size) % log2i(Strategy::node_size);
        uZ   align_idx   = 0;
        uZ   align_count = 0;
        if (misalign > 0) {
            using namespace detail_::fft;
            for (; align_idx < Strategy::align_node_size.size(); ++align_idx) {
                auto i_align_size = Strategy::align_node_size[align_idx];
                auto size         = sub_size_ / 8;
                align_count       = 0;
                while (size > 1 && log2i(size) % log2i(Strategy::node_size) != 0) {
                    size /= i_align_size;
                    ++align_count;
                }
                if (size > 0)
                    break;
            }
        }
        auto misalign_rec    = log2i(fft_size / sub_size_) % log2i(StrategyRec::node_size);
        uZ   align_idx_rec   = 0;
        uZ   align_count_rec = 0;
        if (misalign_rec > 0) {
            using namespace detail_::fft;
            for (; align_idx_rec < StrategyRec::align_node_size.size(); ++align_idx_rec) {
                auto i_align_size = StrategyRec::align_node_size[align_idx_rec];
                auto size         = fft_size / sub_size_;
                align_count_rec   = 0;
                while (size > 1 && log2i(size) % log2i(StrategyRec::node_size) != 0) {
                    size /= i_align_size;
                    ++align_count_rec;
                }
                if (size > 0)
                    break;
            }
        }
        uZ idx = 0;
        idx += misalign ? (StrategyRec::align_node_size.size() + 1) * (align_idx + 1) : 0;
        idx += misalign_rec ? align_idx_rec + 1 : 0;

        return std::make_tuple(idx, align_count, align_idx, align_count_rec, align_idx_rec, sub_size_);
    }

    auto get_subtform_idx_strategic(uZ fft_size, uZ sub_size, uZ specific_size) {
        auto [idx, align_c, align_i, align_c_rec, align_i_rec, sub_size_] =
            get_subtform_idx_strategic_(fft_size, sub_size, specific_size);

        m_align_count     = align_c;
        m_align_count_rec = align_c_rec;
        return idx;
    }

    static auto get_twiddles(uZ fft_size, uZ sub_size, allocator_type allocator) -> twiddle_t
        requires(sorted)
    {
        using detail_::fft::log2i;
        using detail_::fft::wnk;
        auto [idx, align_c, align_i, align_c_rec, align_i_rec, sub_size_] =
            get_subtform_idx_strategic_(fft_size, sub_size, simd::size_specific::sorted_size<T>);

        const auto     depth            = log2i(fft_size);
        constexpr auto tform_sort_depth = log2i(simd::size_specific::sorted_size<T>);

        const uZ n_twiddles = 8 * ((1U << (depth - tform_sort_depth)) - 1U);

        auto twiddles = pcx::vector<real_type, simd::reg<T>::size, allocator_type>(n_twiddles, allocator);

        auto tw_it = twiddles.begin();

        uZ l_size   = simd::size_specific::sorted_size<T> * 2;
        uZ n_groups = 1;

        auto insert_tw = [](auto tw_it, auto size, auto n_groups, auto max_btfly_size) {
            using namespace detail_::fft;
            for (uZ i_group = 0; i_group < n_groups; ++i_group) {
                for (uZ i_btfly = 0; i_btfly < log2i(max_btfly_size); ++i_btfly) {
                    for (uZ i_subgroup = 0; i_subgroup < (1U << i_btfly); ++i_subgroup) {
                        for (uZ k = 0; k < simd::reg<T>::size; ++k) {
                            *(tw_it++) = wnk<T>(size * (1U << i_btfly),
                                                k + i_group * simd::reg<T>::size + i_subgroup * size / 2);
                        }
                    }
                }
            }
            return tw_it;
        };

        auto align_node_size = Strategy::align_node_size.at(align_i);
        for (uZ i = 0; i < align_c; ++i) {
            tw_it = insert_tw(tw_it, l_size, n_groups, align_node_size);
            l_size *= align_node_size;
            n_groups *= align_node_size;
        }
        while (l_size <= sub_size_) {
            tw_it = insert_tw(tw_it, l_size, n_groups, Strategy::node_size);
            l_size *= Strategy::node_size;
            n_groups *= Strategy::node_size;
        }
        align_node_size = StrategyRec::align_node_size.at(align_i_rec);
        fft_size /= detail_::fft::powi(align_node_size, align_c_rec);
        while (l_size <= fft_size) {
            tw_it = insert_tw(tw_it, l_size, n_groups, StrategyRec::node_size);
            l_size *= StrategyRec::node_size;
            n_groups *= StrategyRec::node_size;
        };

        for (uZ i = 0; i < align_c_rec; ++i) {
            tw_it = insert_tw(tw_it, l_size, n_groups, align_node_size);
            l_size *= align_node_size;
            n_groups *= align_node_size;
        }
        return twiddles;
    }

    static auto get_twiddles(uZ fft_size, uZ sub_size, allocator_type allocator) -> twiddle_t
        requires(!sorted)
    {
        auto [idx, align_c, align_i, align_c_rec, align_i_rec, sub_size_] =
            get_subtform_idx_strategic_(fft_size, sub_size, simd::size_specific::unsorted_size<T>);

        auto twiddles = std::vector<T, allocator_type>(allocator);

        insert_tw_unsorted(fft_size,
                           2,
                           sub_size_,
                           0,
                           twiddles,
                           Strategy::align_node_size[align_i],
                           align_c,
                           StrategyRec::align_node_size[align_i_rec],
                           align_c_rec);

        return twiddles;
    }

    static void insert_tw_unsorted(uZ         fft_size,
                                   uZ         l_size,
                                   uZ         sub_size,
                                   uZ         i_group,
                                   twiddle_t& twiddles,
                                   uZ         align_size,
                                   uZ         align_count,
                                   uZ         align_size_rec,
                                   uZ         align_count_rec) {
        using detail_::fft::log2i;
        using detail_::fft::reverse_bit_order;
        using detail_::fft::wnk;

        if ((fft_size / l_size) < sub_size) {
            uZ start_size = twiddles.size();
            uZ single_load_size =
                fft_size / (simd::size_specific::unsorted_size<T> / 2);    // TODO: single load strategic
            uZ group_size = 1;

            for (uZ i_align = 0; i_align < align_count; ++i_align) {
                uZ start = group_size * i_group;
                for (uZ i = 0; i < group_size; ++i) {
                    for (uZ i_btfly = 0; i_btfly < log2i(align_size); ++i_btfly) {
                        auto btfly_size = (1U << i_btfly);
                        for (uZ i_tw = 0; i_tw < btfly_size; ++i_tw) {
                            auto k0 = (start + i) * btfly_size + i_tw;
                            auto k  = reverse_bit_order(k0, log2i(l_size / 2 * btfly_size));
                            auto tw = wnk<T>(l_size * btfly_size, k);
                            twiddles.push_back(tw.real());
                            twiddles.push_back(tw.imag());
                        }
                    }
                }
                l_size *= align_size;
                group_size *= align_size;
            }

            constexpr auto node_size = Strategy::node_size;
            while (l_size < single_load_size) {
                uZ start = group_size * i_group;
                for (uZ i = 0; i < group_size; ++i) {
                    for (uZ i_btfly = 0; i_btfly < log2i(node_size); ++i_btfly) {
                        auto btfly_size = (1U << i_btfly);
                        for (uZ i_tw = 0; i_tw < btfly_size; ++i_tw) {
                            auto k0 = (start + i) * btfly_size + i_tw;
                            auto k  = reverse_bit_order(k0, log2i(l_size / 2 * btfly_size));
                            auto tw = wnk<T>(l_size * btfly_size, k);
                            twiddles.push_back(tw.real());
                            twiddles.push_back(tw.imag());
                        }
                    }
                }
                l_size *= node_size;
                group_size *= node_size;
            }

            if (l_size == single_load_size) {
                simd::size_specific::insert_unsorted<T>(twiddles, group_size, l_size, i_group);
            }
        } else {
            if (align_count_rec > 0) {
                for (uZ i_btfly = 0; i_btfly < log2i(align_size_rec); ++i_btfly) {
                    auto btfly_size = (1U << i_btfly);
                    for (uZ i_tw = 0; i_tw < btfly_size; ++i_tw) {
                        auto k0 = i_group * btfly_size + i_tw;
                        auto k  = reverse_bit_order(k0, log2i(l_size / 2 * btfly_size));
                        auto tw = wnk<T>(l_size * btfly_size, k);
                        twiddles.push_back(tw.real());
                        twiddles.push_back(tw.imag());
                    }
                }
                l_size *= align_size_rec;
                i_group *= align_size_rec;

                for (uZ i = 0; i < align_size_rec; ++i) {
                    insert_tw_unsorted(fft_size,
                                       l_size,
                                       sub_size,
                                       i_group + i,
                                       twiddles,
                                       align_size,
                                       align_count,
                                       align_size_rec,
                                       align_count_rec - 1);
                }
                return;
            }
            constexpr auto node_size = StrategyRec::node_size;
            for (uZ i_btfly = 0; i_btfly < log2i(node_size); ++i_btfly) {
                auto btfly_size = (1U << i_btfly);
                for (uZ i_tw = 0; i_tw < btfly_size; ++i_tw) {
                    auto k0 = i_group * btfly_size + i_tw;
                    auto k  = reverse_bit_order(k0, log2i(l_size / 2 * btfly_size));
                    auto tw = wnk<T>(l_size * btfly_size, k);
                    twiddles.push_back(tw.real());
                    twiddles.push_back(tw.imag());
                }
            }
            l_size *= node_size;
            i_group *= node_size;

            for (uZ i = 0; i < node_size; ++i) {
                insert_tw_unsorted(fft_size,
                                   l_size,
                                   sub_size,
                                   i_group + i,
                                   twiddles,
                                   align_size,
                                   align_count,
                                   align_size_rec,
                                   align_count_rec);
            }
        }
    }
};

template<typename T,
         fft_order Order     = fft_order::normal,
         bool      BigG      = false,
         bool      BiggerG   = false,
         typename Allocator  = pcx::aligned_allocator<T, std::align_val_t(64)>,
         uZ NodeSizeStrategy = 2>
    requires(std::same_as<T, float> || std::same_as<T, double>)
class fft_unit_par {
public:
    using real_type      = T;
    using allocator_type = Allocator;
    using size_type      = uZ;

private:
    using sort_allocator_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<uZ>;
    using tw_allocator_type =
        typename std::allocator_traits<allocator_type>::template rebind_alloc<std::complex<real_type>>;
    static constexpr bool sorted = Order == fft_order::normal;

    using sort_t    = std::conditional_t<sorted,    //
                                      std::vector<uZ, sort_allocator_type>,
                                      decltype([]() {})>;
    using twiddle_t = std::vector<std::complex<real_type>, tw_allocator_type>;

public:
    fft_unit_par(uZ size, allocator_type allocator = allocator_type{})
    : m_size(size)
    , m_sort(get_sort(size, allocator))
    , m_twiddles(get_twiddles(size, allocator))
    , m_twiddles_new(get_twiddles_new(size, allocator)){};

    template<typename DestR_, typename SrcR_>
        requires range_complex_vector_of<T, DestR_> && range_complex_vector_of<T, SrcR_>
    void operator()(DestR_& dest, const SrcR_& source) {
        if (dest.size() != m_size) {
            throw(std::invalid_argument(std::string("destination size (which is ")
                                            .append(std::to_string(dest.size()))
                                            .append(" is not equal to fft size (which is ")
                                            .append(std::to_string(m_size))
                                            .append(")")));
        }
        if (source.size() != m_size) {
            throw(std::invalid_argument(std::string("source size (which is ")
                                            .append(std::to_string(source.size()))
                                            .append(" is not equal to fft size (which is ")
                                            .append(std::to_string(m_size))
                                            .append(")")));
        }

        using dest_vector_t = std::remove_pointer_t<rv::range_value_t<DestR_>>;
        using src_vector_t  = std::remove_pointer_t<rv::range_value_t<SrcR_>>;

        constexpr auto PDest = cx_vector_traits<dest_vector_t>::pack_size;
        constexpr auto PSrc  = cx_vector_traits<src_vector_t>::pack_size;

        auto get_vector = [](auto&& R, uZ i) -> auto& {
            using vector_t = rv::range_value_t<std::remove_cvref_t<decltype(R)>>;
            if constexpr (std::is_pointer_v<vector_t>) {
                return *R[i];
            } else {
                return R[i];
            }
        };

        auto tw_it     = m_twiddles.begin();
        auto data_size = get_vector(dest, 0).size();

        uZ l_size = 1;
        if constexpr (BigG && !BiggerG) {
            if (log2i(m_size) % 2 != 0) {
                const auto grp_size = m_size / l_size / 8;
                for (uZ grp = 0; grp < grp_size; ++grp) {
                    std::array<T*, 8> dst{
                        get_vector(dest, grp).data(),
                        get_vector(dest, grp + grp_size * 1).data(),
                        get_vector(dest, grp + grp_size * 2).data(),
                        get_vector(dest, grp + grp_size * 3).data(),
                        get_vector(dest, grp + grp_size * 4).data(),
                        get_vector(dest, grp + grp_size * 5).data(),
                        get_vector(dest, grp + grp_size * 6).data(),
                        get_vector(dest, grp + grp_size * 7).data(),
                    };
                    std::array<T*, 8> src{
                        get_vector(source, grp).data(),
                        get_vector(source, grp + grp_size * 1).data(),
                        get_vector(source, grp + grp_size * 2).data(),
                        get_vector(source, grp + grp_size * 3).data(),
                        get_vector(source, grp + grp_size * 4).data(),
                        get_vector(source, grp + grp_size * 5).data(),
                        get_vector(source, grp + grp_size * 6).data(),
                        get_vector(source, grp + grp_size * 7).data(),
                    };
                    long_btfly8<PDest, PSrc>(dst, data_size, src);
                }
                l_size *= 8;
            }
        }
        if constexpr (BiggerG) {
            auto max_size = m_size;
            if (log2i(m_size) % 3 == 1) {
                max_size /= 4;
            };
            const auto grp_size = m_size / l_size / 8;
            for (uZ grp = 0; grp < grp_size; ++grp) {
                std::array<T*, 8> dst{
                    get_vector(dest, grp).data(),
                    get_vector(dest, grp + grp_size * 1).data(),
                    get_vector(dest, grp + grp_size * 2).data(),
                    get_vector(dest, grp + grp_size * 3).data(),
                    get_vector(dest, grp + grp_size * 4).data(),
                    get_vector(dest, grp + grp_size * 5).data(),
                    get_vector(dest, grp + grp_size * 6).data(),
                    get_vector(dest, grp + grp_size * 7).data(),
                };
                std::array<T*, 8> src{
                    get_vector(source, grp).data(),
                    get_vector(source, grp + grp_size * 1).data(),
                    get_vector(source, grp + grp_size * 2).data(),
                    get_vector(source, grp + grp_size * 3).data(),
                    get_vector(source, grp + grp_size * 4).data(),
                    get_vector(source, grp + grp_size * 5).data(),
                    get_vector(source, grp + grp_size * 6).data(),
                    get_vector(source, grp + grp_size * 7).data(),
                };
                long_btfly8<PDest, PSrc>(dst, data_size, src);
            }
            l_size *= 8;

            for (; l_size < max_size / 4; l_size *= 8) {
                const auto grp_size = m_size / l_size / 8;
                for (uZ grp = 0; grp < grp_size; ++grp) {
                    std::array<T*, 8> dst{
                        get_vector(dest, grp).data(),
                        get_vector(dest, grp + grp_size * 1).data(),
                        get_vector(dest, grp + grp_size * 2).data(),
                        get_vector(dest, grp + grp_size * 3).data(),
                        get_vector(dest, grp + grp_size * 4).data(),
                        get_vector(dest, grp + grp_size * 5).data(),
                        get_vector(dest, grp + grp_size * 6).data(),
                        get_vector(dest, grp + grp_size * 7).data(),
                    };
                    long_btfly8<PDest, PSrc>(dst, data_size);
                }
                for (uZ idx = 1; idx < l_size; ++idx) {
                    std::array<std::complex<T>, 7> tw = {
                        *(tw_it++),
                        *(tw_it++),
                        *(tw_it++),
                        *(tw_it++),
                        *(tw_it++),
                        *(tw_it++),
                        *(tw_it++),
                    };
                    for (uZ grp = 0; grp < grp_size; ++grp) {
                        std::array<T*, 8> dst{
                            get_vector(dest, grp + idx * grp_size * 8).data(),
                            get_vector(dest, grp + idx * grp_size * 8 + grp_size * 1).data(),
                            get_vector(dest, grp + idx * grp_size * 8 + grp_size * 2).data(),
                            get_vector(dest, grp + idx * grp_size * 8 + grp_size * 3).data(),
                            get_vector(dest, grp + idx * grp_size * 8 + grp_size * 4).data(),
                            get_vector(dest, grp + idx * grp_size * 8 + grp_size * 5).data(),
                            get_vector(dest, grp + idx * grp_size * 8 + grp_size * 6).data(),
                            get_vector(dest, grp + idx * grp_size * 8 + grp_size * 7).data(),
                        };
                        long_btfly8<PDest, PSrc>(dst, data_size, tw);
                    }
                }
            }
        } else if constexpr (!BigG) {
            const auto grp_size = m_size / l_size / 4;
            for (uZ grp = 0; grp < grp_size; ++grp) {
                std::array<T*, 4> dst{
                    get_vector(dest, grp).data(),
                    get_vector(dest, grp + grp_size * 1).data(),
                    get_vector(dest, grp + grp_size * 2).data(),
                    get_vector(dest, grp + grp_size * 3).data(),
                };
                std::array<T*, 4> src{
                    get_vector(source, grp).data(),
                    get_vector(source, grp + grp_size * 1).data(),
                    get_vector(source, grp + grp_size * 2).data(),
                    get_vector(source, grp + grp_size * 3).data(),
                };
                long_btfly4<PDest, PSrc>(dst, data_size, src);
            }
            l_size *= 4;
        }

        for (; l_size < m_size / 2; l_size *= 4) {
            const auto grp_size = m_size / l_size / 4;
            for (uZ grp = 0; grp < grp_size; ++grp) {
                std::array<T*, 4> dst{
                    get_vector(dest, grp).data(),
                    get_vector(dest, grp + grp_size * 1).data(),
                    get_vector(dest, grp + grp_size * 2).data(),
                    get_vector(dest, grp + grp_size * 3).data(),
                };
                long_btfly4<PDest, PSrc>(dst, data_size);
            }
            for (uZ idx = 1; idx < l_size; ++idx) {
                std::array<std::complex<T>, 3> tw = {
                    *(tw_it++),
                    *(tw_it++),
                    *(tw_it++),
                };
                for (uZ grp = 0; grp < grp_size; ++grp) {
                    std::array<T*, 4> dst{
                        get_vector(dest, grp + idx * grp_size * 4).data(),
                        get_vector(dest, grp + idx * grp_size * 4 + grp_size * 1).data(),
                        get_vector(dest, grp + idx * grp_size * 4 + grp_size * 2).data(),
                        get_vector(dest, grp + idx * grp_size * 4 + grp_size * 3).data(),
                    };
                    long_btfly4<PDest, PSrc>(dst, data_size, tw);
                }
            }
        }
        if constexpr (!BigG && !BiggerG) {
            if (l_size == m_size / 2) {
                std::array<T*, 2> dst{
                    get_vector(dest, 0).data(),
                    get_vector(dest, 1).data(),
                };
                long_btfly2<PDest, PSrc>(dst, data_size);

                for (uZ idx = 1; idx < l_size; ++idx) {
                    auto tw = *(tw_it++);

                    std::array<T*, 2> dst{
                        get_vector(dest, idx * 2).data(),
                        get_vector(dest, idx * 2 + 1).data(),
                    };
                    long_btfly2<PDest, PSrc>(dst, data_size, tw);
                }
            }
        }
        if constexpr (sorted) {
            for (uZ i = 0; i < m_sort.size(); i += 2) {
                using std::swap;
                swap(get_vector(dest, m_sort[i]), get_vector(dest, m_sort[i + 1]));
            }
        }
    };

    template<typename DestR_, typename SrcR_>
        requires range_complex_vector_of<T, DestR_> && range_complex_vector_of<T, SrcR_>
    void new_tform(DestR_& dest, const SrcR_& source) {
        if (dest.size() != m_size) {
            throw(std::invalid_argument(std::string("destination size (which is ")
                                            .append(std::to_string(dest.size()))
                                            .append(" is not equal to fft size (which is ")
                                            .append(std::to_string(m_size))
                                            .append(")")));
        }
        if (source.size() != m_size) {
            throw(std::invalid_argument(std::string("source size (which is ")
                                            .append(std::to_string(source.size()))
                                            .append(" is not equal to fft size (which is ")
                                            .append(std::to_string(m_size))
                                            .append(")")));
        }

        using dest_vector_t = std::remove_pointer_t<rv::range_value_t<DestR_>>;
        using src_vector_t  = std::remove_pointer_t<rv::range_value_t<SrcR_>>;

        constexpr auto PDest  = cx_vector_traits<dest_vector_t>::pack_size;
        constexpr auto PSrc   = cx_vector_traits<src_vector_t>::pack_size;
        constexpr auto PTform = std::max(PDest, simd::reg<T>::size);

        constexpr auto get_vector = [](auto& R, uZ i) -> auto& {
            using vector_t = rv::range_value_t<std::remove_cvref_t<decltype(R)>>;
            if constexpr (std::is_pointer_v<vector_t>) {
                return *R[i];
            } else {
                return R[i];
            }
        };

        auto tw_it     = m_twiddles_new.begin();
        auto data_size = get_vector(dest, 0).size();

        uZ l_size = 1;

        constexpr auto get_data_ptr =
            [get_vector]<uZ... I>(auto& data, uZ i, uZ grp_size, std::index_sequence<I...>) {
                using vector_t = rv::range_value_t<std::remove_cvref_t<decltype(data)>>;
                using traits   = cx_vector_traits<vector_t>;
                // using data_ptr_t      = typename traits::real_type*;
                constexpr uZ NodeSize = sizeof...(I);
                return std::array{traits::re_data(get_vector(data, i + grp_size * I))...};
            };
        auto misalign = log2i(m_size) % log2i(NodeSizeStrategy);
        using detail_::fft::powi;

        constexpr auto realign = [get_data_ptr]<uZ... Pow>(auto& dest,
                                                           auto& source,
                                                           auto& tw_it,
                                                           uZ    size,
                                                           uZ    data_size,
                                                           uZ    align_size,
                                                           uZ    align_count,
                                                           std::index_sequence<Pow...>) {
            constexpr auto multi_step = [get_data_ptr]<uZ NodeSize>(uZ_constant<NodeSize>,    //
                                                                    uZ    align_count,
                                                                    uZ    size,
                                                                    uZ    data_size,
                                                                    auto& dest,
                                                                    auto& source,
                                                                    auto  tw_it) {
                const auto     grp_size = size / NodeSize;
                constexpr auto idxs     = std::make_index_sequence<NodeSize>{};
                for (uZ grp = 0; grp < grp_size; ++grp) {
                    auto dst = get_data_ptr(dest, grp, grp_size, idxs);
                    auto src = get_data_ptr(source, grp, grp_size, idxs);
                    long_node<NodeSize, PTform, PSrc, false>(dst, data_size, src);
                }
                if (PTform != PDest && powi(NodeSize, align_count) == size) {
                    --align_count;
                }
                uZ l_size = NodeSize;
                for (uZ i = 1; i < align_count; ++i) {
                    const auto grp_size = size / l_size / NodeSize;
                    for (uZ grp = 0; grp < grp_size; ++grp) {
                        auto dst = get_data_ptr(dest, grp, grp_size, idxs);
                        long_node<NodeSize, PTform, PTform, false>(dst, data_size);
                    }
                    for (uZ idx = 1; idx < l_size; ++idx) {
                        auto tw = []<uZ... I>(auto& tw_it, std::index_sequence<I...>) {
                            return std::array{simd::broadcast(*(tw_it + I))...};
                        }(tw_it, std::make_index_sequence<NodeSize - 1>{});
                        tw_it += NodeSize - 1;
                        for (uZ grp = 0; grp < grp_size; ++grp) {
                            auto dst = get_data_ptr(dest, grp, grp_size, idxs);
                            long_node<NodeSize, PTform, PTform, false>(dst, data_size, tw);
                        }
                    }
                    l_size *= NodeSize;
                }
                if (PTform != PDest && powi(NodeSize, align_count) == size) {
                    const auto grp_size = size / l_size / NodeSize;
                    for (uZ grp = 0; grp < grp_size; ++grp) {
                        auto dst = get_data_ptr(dest, grp, grp_size, idxs);
                        long_node<NodeSize, PDest, PTform, false>(dst, data_size);
                    }
                    for (uZ idx = 1; idx < l_size; ++idx) {
                        auto tw = []<uZ... I>(auto& tw_it, std::index_sequence<I...>) {
                            return std::array{simd::broadcast(*(tw_it + I))...};
                        }(tw_it, std::make_index_sequence<NodeSize - 1>{});
                        tw_it += NodeSize - 1;
                        for (uZ grp = 0; grp < grp_size; ++grp) {
                            auto dst = get_data_ptr(dest, grp, grp_size, idxs);
                            long_node<NodeSize, PDest, PTform, false>(dst, data_size, tw);
                        }
                    }
                    l_size *= NodeSize;
                }
                return std::make_tuple(l_size, tw_it);
            };
            uZ l_size;
            (void)((align_size == powi(2, log2i(NodeSizeStrategy) - Pow - 1) &&
                    (std::tie(l_size, tw_it) =
                         multi_step(uZ_constant<powi(2, log2i(NodeSizeStrategy) - Pow - 1)>{},
                                    align_count,
                                    size,
                                    data_size,
                                    dest,
                                    source,
                                    tw_it),
                     true)) ||
                   ...);
            return std::make_tuple(l_size, tw_it);
        };

        auto idxs = std::make_index_sequence<NodeSizeStrategy>{};
        if (m_align_count != 0) {
            std::tie(l_size, tw_it) = realign(dest,
                                              source,
                                              tw_it,
                                              m_size,
                                              data_size,
                                              m_align_size,
                                              m_align_count,
                                              std::make_index_sequence<log2i(NodeSizeStrategy) - 1>{});
        } else {
            if (PTform != PDest && m_size == NodeSizeStrategy) {
                const auto grp_size = m_size / l_size / NodeSizeStrategy;
                for (uZ grp = 0; grp < grp_size; ++grp) {
                    auto dst = get_data_ptr(dest, grp, grp_size, idxs);
                    auto src = get_data_ptr(source, grp, grp_size, idxs);
                    long_node<NodeSizeStrategy, PDest, PSrc>(dst, data_size, src);
                }
                l_size *= NodeSizeStrategy;
            } else {
                const auto grp_size = m_size / l_size / NodeSizeStrategy;
                for (uZ grp = 0; grp < grp_size; ++grp) {
                    auto dst = get_data_ptr(dest, grp, grp_size, idxs);
                    auto src = get_data_ptr(source, grp, grp_size, idxs);
                    long_node<NodeSizeStrategy, PTform, PSrc>(dst, data_size, src);
                }
                l_size *= NodeSizeStrategy;
            }
        }

        uZ max_size = m_size;
        if constexpr (PTform != PDest) {
            max_size /= NodeSizeStrategy;
        }
        while (l_size < max_size) {
            const auto grp_size = m_size / l_size / NodeSizeStrategy;
            for (uZ i = 0; i < grp_size; ++i) {
                auto dst = get_data_ptr(dest, i, grp_size, idxs);
                long_node<NodeSizeStrategy, PTform, PTform>(dst, data_size);
            }
            for (uZ i_grp = 1; i_grp < l_size; ++i_grp) {
                auto tw = []<uZ... I>(auto& tw_it, std::index_sequence<I...>) {
                    return std::array{simd::broadcast(*(tw_it + I))...};
                }(tw_it, std::make_index_sequence<NodeSizeStrategy - 1>{});
                tw_it += NodeSizeStrategy - 1;
                for (uZ i = 0; i < grp_size; ++i) {
                    uZ   start = grp_size * i_grp * NodeSizeStrategy + i;
                    auto dst   = get_data_ptr(dest, start, grp_size, idxs);
                    long_node<NodeSizeStrategy, PTform, PTform>(dst, data_size, tw);
                }
            }
            l_size *= NodeSizeStrategy;
        }

        if (PTform != PDest && l_size <= m_size) {
            const auto grp_size = m_size / l_size / NodeSizeStrategy;
            for (uZ grp = 0; grp < grp_size; ++grp) {
                auto dst = get_data_ptr(dest, grp, grp_size, idxs);
                long_node<NodeSizeStrategy, PDest, PTform>(dst, data_size);
            }
            for (uZ idx = 1; idx < l_size; ++idx) {
                auto tw = []<uZ... I>(auto& tw_it, std::index_sequence<I...>) {
                    return std::array{simd::broadcast(*(tw_it + I))...};
                }(tw_it, std::make_index_sequence<NodeSizeStrategy - 1>{});
                tw_it += NodeSizeStrategy - 1;
                for (uZ grp = 0; grp < grp_size; ++grp) {
                    auto dst = get_data_ptr(dest, grp, grp_size, idxs);
                    long_node<NodeSizeStrategy, PDest, PTform>(dst, data_size, tw);
                }
            }
            l_size *= NodeSizeStrategy;
        }
        return;

        for (uZ i = 0; i < m_sort.size(); i += 2) {
            using std::swap;
            swap(get_vector(dest, m_sort[i]), get_vector(dest, m_sort[i + 1]));
        }
    }

private:
    template<uZ PDest, uZ PSrc, typename... Optional>
    inline void long_btfly8(std::array<T*, 8> dest, uZ size, Optional... optional) {
        using src_type       = std::array<const T*, 8>;
        using tw_type        = std::array<std::complex<T>, 7>;
        constexpr bool Src   = detail_::has_type<src_type, Optional...>;
        constexpr bool Tw    = detail_::has_type<tw_type, Optional...>;
        constexpr bool Scale = detail_::has_type<T, Optional...>;

        const auto& source = [](auto... optional) {
            if constexpr (Src) {
                return std::get<src_type&>(std::tie(optional...));
            } else {
                return [] {};
            }
        }(optional...);

        auto tw = [](auto... optional) {
            if constexpr (Tw) {
                auto& tw = std::get<tw_type&>(std::tie(optional...));
                return std::array<simd::cx_reg<T>, 7>{
                    simd::broadcast(tw[0]),
                    simd::broadcast(tw[1]),
                    simd::broadcast(tw[2]),
                    simd::broadcast(tw[3]),
                    simd::broadcast(tw[4]),
                    simd::broadcast(tw[5]),
                    simd::broadcast(tw[6]),
                };
            } else {
                return [] {};
            }
        }(optional...);

        auto scaling = [](auto... optional) {
            if constexpr (Scale) {
                auto scale = std::get<T>(std::tie(optional...));
                return simd::broadcast(scale);
            } else {
                return [] {};
            }
        }(optional...);

        for (uZ i = 0; i < size; i += simd::reg<T>::size) {
            std::array<T*, 8> dst{
                simd::ra_addr<PDest>(dest[0], i),
                simd::ra_addr<PDest>(dest[1], i),
                simd::ra_addr<PDest>(dest[2], i),
                simd::ra_addr<PDest>(dest[3], i),
                simd::ra_addr<PDest>(dest[4], i),
                simd::ra_addr<PDest>(dest[5], i),
                simd::ra_addr<PDest>(dest[6], i),
                simd::ra_addr<PDest>(dest[7], i),
            };
            auto src = [](auto source, auto i) {
                if constexpr (Src) {
                    return std::array<const T*, 8>{
                        simd::ra_addr<PSrc>(source[0], i),
                        simd::ra_addr<PSrc>(source[1], i),
                        simd::ra_addr<PSrc>(source[2], i),
                        simd::ra_addr<PSrc>(source[3], i),
                        simd::ra_addr<PSrc>(source[4], i),
                        simd::ra_addr<PSrc>(source[5], i),
                        simd::ra_addr<PSrc>(source[6], i),
                        simd::ra_addr<PSrc>(source[7], i),
                    };
                } else {
                    return [] {};
                }
            }(source, i);

            detail_::fft::node<8>::template perform<T, PDest, PSrc, false, false>(dst, src, tw, scaling);
        }
    }

    template<uZ PDest, uZ PSrc, typename... Optional>
    inline void long_btfly4(std::array<T*, 4> dest, uZ size, Optional... optional) {
        using src_type       = std::array<const T*, 4>;
        using tw_type        = std::array<std::complex<T>, 3>;
        constexpr bool Src   = detail_::has_type<src_type, Optional...>;
        constexpr bool Tw    = detail_::has_type<tw_type, Optional...>;
        constexpr bool Scale = detail_::has_type<T, Optional...>;

        const auto& source = [](auto... optional) {
            if constexpr (Src) {
                return std::get<src_type&>(std::tie(optional...));
            } else {
                return [] {};
            }
        }(optional...);

        auto tw = [](auto... optional) {
            if constexpr (Tw) {
                auto& tw = std::get<tw_type&>(std::tie(optional...));
                return std::array<simd::cx_reg<T>, 3>{
                    simd::broadcast(tw[0]),
                    simd::broadcast(tw[1]),
                    simd::broadcast(tw[2]),
                };
            } else {
                return [] {};
            }
        }(optional...);

        auto scaling = [](auto... optional) {
            if constexpr (Scale) {
                auto scale = std::get<T>(std::tie(optional...));
                return simd::broadcast(scale);
            } else {
                return [] {};
            }
        }(optional...);

        for (uZ i = 0; i < size; i += simd::reg<T>::size) {
            std::array<T*, 4> dst{
                simd::ra_addr<PDest>(dest[0], i),
                simd::ra_addr<PDest>(dest[1], i),
                simd::ra_addr<PDest>(dest[2], i),
                simd::ra_addr<PDest>(dest[3], i),
            };
            auto src = [](auto source, auto i) {
                if constexpr (Src) {
                    return std::array<const T*, 4>{
                        simd::ra_addr<PSrc>(source[0], i),
                        simd::ra_addr<PSrc>(source[1], i),
                        simd::ra_addr<PSrc>(source[2], i),
                        simd::ra_addr<PSrc>(source[3], i),
                    };
                } else {
                    return [] {};
                }
            }(source, i);

            detail_::fft::node<4>::template perform<T, PDest, PSrc, false, false>(dst, src, tw, scaling);
        }
    }

    template<uZ PDest, uZ PSrc, typename... Optional>
    inline void long_btfly2(std::array<T*, 2> dest, uZ size, Optional... optional) {
        using src_type       = std::array<const T*, 2>;
        using tw_type        = std::complex<T>;
        constexpr bool Src   = detail_::has_type<src_type, Optional...>;
        constexpr bool Tw    = detail_::has_type<tw_type, Optional...>;
        constexpr bool Scale = detail_::has_type<T, Optional...>;

        const auto& source = [](auto... optional) {
            if constexpr (Src) {
                return std::get<src_type&>(std::tie(optional...));
            } else {
                return [] {};
            }
        }(optional...);

        auto tw = [](auto... optional) {
            if constexpr (Tw) {
                auto& tw = std::get<tw_type&>(std::tie(optional...));
                return simd::broadcast(tw);
            } else {
                return [] {};
            }
        }(optional...);

        auto scaling = [](auto... optional) {
            if constexpr (Scale) {
                auto scale = std::get<T>(std::tie(optional...));
                return simd::broadcast(scale);
            } else {
                return [] {};
            }
        }(optional...);

        for (uZ i = 0; i < size; i += simd::reg<T>::size) {
            std::array<T*, 2> dst{
                simd::ra_addr<PDest>(dest[0], i),
                simd::ra_addr<PDest>(dest[1], i),
            };
            auto src = [](auto source, auto i) {
                if constexpr (Src) {
                    return std::array<const T*, 2>{
                        simd::ra_addr<PSrc>(source[0], i),
                        simd::ra_addr<PSrc>(source[1], i),
                    };
                } else {
                    return [] {};
                }
            }(source, i);

            detail_::fft::node<2>::template perform<T, PDest, PSrc, false, false>(dst, src, tw, scaling);
        }
    }

    template<uZ   NodeSize,
             uZ   PDest,
             uZ   PSrc,
             bool DecInTime = false,
             bool ConjTw    = false,
             bool Reverse   = false,
             typename... Optional>
    inline static void long_node(const std::array<T*, NodeSize>& dest, uZ size, const Optional&... optional) {
        using src_type     = std::array<const T*, NodeSize>;
        constexpr bool Src = detail_::has_type<src_type, Optional...>;

        const auto& source = [](const auto&... optional) {
            if constexpr (Src) {
                return std::get<src_type&>(std::tie(optional...));
            } else {
                return [] {};
            }
        }(optional...);

        constexpr auto get_data_array = []<uZ PackSize, uZ... I>(auto& data,
                                                                 uZ    offset,
                                                                 uZ_constant<PackSize>,
                                                                 std::index_sequence<I...>) {
            constexpr auto Size = sizeof...(I);
            static_assert(std::same_as<T*, std::remove_cvref_t<decltype(data[0])>>);
            return std::array<std::remove_cvref_t<decltype(data[0])>, Size>{
                simd::ra_addr<PackSize>(data[I], offset)...};
        };
        constexpr auto Idxs = std::make_index_sequence<NodeSize>{};

        for (uZ i = 0; i < size; i += simd::reg<T>::size) {
            auto dst = get_data_array(dest, i, uZ_constant<PDest>{}, Idxs);
            auto src = [](auto& source, auto i) {
                if constexpr (Src) {
                    return get_data_array(source, i, uZ_constant<PSrc>{}, Idxs);
                } else {
                    return [] {};
                }
            }(source, i);

            detail_::fft::node<NodeSize>::template perform<T, PDest, PSrc, false, false>(
                dst, src, optional...);
        }
    }

private:
    const uZ                           m_size;
    [[no_unique_address]] const sort_t m_sort;
    uZ                                 m_align_size  = 0;
    uZ                                 m_align_count = 0;
    const twiddle_t                    m_twiddles;
    const twiddle_t                    m_twiddles_new;

    auto get_twiddles(size_type size, allocator_type allocator) -> twiddle_t {
        auto twiddles = twiddle_t(static_cast<tw_allocator_type>(allocator));
        //twiddles.reserve(?);
        uZ l_size = 1;

        using namespace detail_::fft;
        if constexpr (BigG) {
            if (log2i(size) % 2 != 0) {
                l_size *= 8;
            }
        }

        if constexpr (BiggerG) {
            auto max_size = m_size;
            if (log2i(m_size) % 3 == 1) {
                max_size /= 4;
            };
            for (; l_size < max_size / 4; l_size *= 8) {
                const auto grp_size = std::max(m_size / l_size / 4, 0UL);
                for (uZ idx = 1; idx < l_size; ++idx) {
                    twiddles.push_back(wnk<T>(l_size * 2, reverse_bit_order(idx, log2i(l_size))));
                    twiddles.push_back(wnk<T>(l_size * 4, reverse_bit_order(idx * 2 + 0, log2i(l_size * 2))));
                    twiddles.push_back(wnk<T>(l_size * 4, reverse_bit_order(idx * 2 + 1, log2i(l_size * 2))));
                    twiddles.push_back(wnk<T>(l_size * 8, reverse_bit_order(idx * 4 + 0, log2i(l_size * 4))));
                    twiddles.push_back(wnk<T>(l_size * 8, reverse_bit_order(idx * 4 + 1, log2i(l_size * 4))));
                    twiddles.push_back(wnk<T>(l_size * 8, reverse_bit_order(idx * 4 + 2, log2i(l_size * 4))));
                    twiddles.push_back(wnk<T>(l_size * 8, reverse_bit_order(idx * 4 + 3, log2i(l_size * 4))));
                }
            }
        }
        for (; l_size < m_size / 2; l_size *= 4) {
            const auto grp_size = std::max(m_size / l_size / 4, 0UL);
            for (uZ idx = 1; idx < l_size; ++idx) {
                twiddles.push_back(wnk<T>(l_size * 2, reverse_bit_order(idx, log2i(l_size))));
                twiddles.push_back(wnk<T>(l_size * 4, reverse_bit_order(idx * 2, log2i(l_size * 2))));
                twiddles.push_back(wnk<T>(l_size * 4, reverse_bit_order(idx * 2 + 1, log2i(l_size * 2))));
            }
        }

        if constexpr (!BiggerG) {
            if (l_size == m_size / 2) {
                for (uZ idx = 1; idx < l_size; ++idx) {
                    twiddles.push_back(wnk<T>(l_size * 2, reverse_bit_order(idx, log2i(l_size))));
                }
            }
        }
        std::cout << "<get twiddles> real: " << twiddles.size() << "\n";
        return twiddles;
    };

    auto get_twiddles_new(uZ size, allocator_type allocator) -> twiddle_t {
        auto twiddles = twiddle_t(static_cast<tw_allocator_type>(allocator));
        //twiddles.reserve(?);

        using namespace detail_::fft;

        auto misalign = log2i(size) % log2i(NodeSizeStrategy);
        if (misalign != 0) {
            for (uZ pow = 0; pow < log2i(NodeSizeStrategy); ++pow) {
                uZ align_size  = powi(2, log2i(NodeSizeStrategy) - pow - 1);
                uZ align_count = 1;
                while ((log2i(align_size) * align_count) % log2i(NodeSizeStrategy) != misalign)
                    ++align_count;
                if (powi(align_size, align_count) <= size) {
                    m_align_count = align_count;
                    m_align_size  = align_size;
                    break;
                }
            }
        }

        // uZ n_twiddles = (m_align_size - 1) * m_align_count +    //
        //                 (NodeSizeStrategy - 1) * (log2i(size) - log2i(m_align_size) * m_align_count);

        /**
         * @brief calculates number of twiddles

         * @param btfly_size    butterfly size
         * @param count         number of butterflies to perform
         * @param start_size    starting trandform size
         */
        constexpr auto get_num_twiddles = [](uZ btfly_size, uZ count, uZ start_size) -> uZ {
            return (btfly_size - 1)                                            // twiddles per butterfly
                   * (start_size                                               // geometric progression sum
                          * ((1 - static_cast<iZ>(powi(btfly_size, count)))    //
                             / (1 - static_cast<iZ>(btfly_size)))              //
                      - count);    // first butterfly on each level is performed with fixed twiddles
        };
        uZ aligned_size = powi(m_align_size, m_align_count);
        uZ n_twiddles   = get_num_twiddles(m_align_size, m_align_count, 1) +
                        get_num_twiddles(NodeSizeStrategy,
                                         log2i(size / aligned_size) / log2i(NodeSizeStrategy),
                                         aligned_size);
        // twiddles.reserve(n_twiddles);

        constexpr auto insert = [](uZ size, auto& twiddles, uZ node_size) {
            for (uZ idx = 1; idx < size; ++idx) {
                for (uZ i_node = 2; i_node <= node_size; i_node *= 2) {
                    for (uZ i = 0; i < i_node / 2; ++i) {
                        twiddles.push_back(wnk<T>(size * i_node,                             //
                                                  reverse_bit_order(idx * i_node / 2 + i,    //
                                                                    log2i(size * i_node / 2))));
                    }
                }
            }
        };
        // for (uZ idx = 1; idx < l_size; ++idx) {
        //     twiddles.push_back(wnk<T>(l_size * 2, reverse_bit_order(idx, log2i(l_size))));
        //     twiddles.push_back(wnk<T>(l_size * 4, reverse_bit_order(idx * 2 + 0, log2i(l_size * 2))));
        //     twiddles.push_back(wnk<T>(l_size * 4, reverse_bit_order(idx * 2 + 1, log2i(l_size * 2))));
        //     twiddles.push_back(wnk<T>(l_size * 8, reverse_bit_order(idx * 4 + 0, log2i(l_size * 4))));
        //     twiddles.push_back(wnk<T>(l_size * 8, reverse_bit_order(idx * 4 + 1, log2i(l_size * 4))));
        //     twiddles.push_back(wnk<T>(l_size * 8, reverse_bit_order(idx * 4 + 2, log2i(l_size * 4))));
        //     twiddles.push_back(wnk<T>(l_size * 8, reverse_bit_order(idx * 4 + 3, log2i(l_size * 4))));
        // }

        uZ l_size = 1;
        for (uZ i = 0; i < m_align_count; ++i) {
            insert(l_size, twiddles, m_align_size);
            l_size *= m_align_size;
        }
        while (l_size < size) {
            insert(l_size, twiddles, NodeSizeStrategy);
            l_size *= NodeSizeStrategy;
        }
        std::cout << "<get twiddles new> equation: " << n_twiddles << " real: " << twiddles.size() << "\n";
        return twiddles;
    };

    static auto get_sort(size_type size, allocator_type allocator) -> sort_t {
        if constexpr (!sorted) {
            return {};
        } else {
            auto sort = sort_t(static_cast<sort_allocator_type>(allocator));
            //sort.reserve(?);
            for (uZ i = 0; i < size; ++i) {
                auto rev = detail_::fft::reverse_bit_order(i, log2i(size));
                if (rev > i) {
                    sort.push_back(i);
                    sort.push_back(rev);
                }
            }
            return sort;
        }
    };
    static constexpr auto log2i(uZ num) -> uZ {
        uZ order = 0;
        while ((num >>= 1U) != 0) {
            order++;
        }
        return order;
    }
};

// NOLINTEND (*magic-numbers)
}    // namespace pcx
#endif