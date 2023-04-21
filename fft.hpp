#ifndef FFT_HPP
#define FFT_HPP

#include "vector.hpp"
#include "vector_arithm.hpp"
#include "vector_util.hpp"

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <xmmintrin.h>

// NOLINTBEGIN (*magic-numbers)
namespace pcx {
namespace internal {
template<typename T, typename... U>
concept has_type = (std::same_as<T, U> || ...);
namespace fft {

enum class decimation : bool {
    in_time      = true,
    in_frequency = false
};

template<std::size_t Size, bool DecInTime>
struct order {
    static constexpr auto reverse_bit_order(uint64_t num, uint64_t depth) -> uint64_t {
        num = num >> 32 | num << 32;
        num = (num & 0xFFFF0000FFFF0000) >> 16 | (num & 0x0000FFFF0000FFFF) << 16;
        num = (num & 0xFF00FF00FF00FF00) >> 8 | (num & 0x00FF00FF00FF00FF) << 8;
        num = (num & 0xF0F0F0F0F0F0F0F0) >> 4 | (num & 0x0F0F0F0F0F0F0F0F) << 4;
        num = (num & 0xCCCCCCCCCCCCCCCC) >> 2 | (num & 0x3333333333333333) << 2;
        num = (num & 0xAAAAAAAAAAAAAAAA) >> 1 | (num & 0x5555555555555555) << 1;
        return num >> (64 - depth);
    }
    static constexpr auto log2i(std::size_t num) -> std::size_t {
        std::size_t order = 0;
        while ((num >>= 1U) != 0) {
            order++;
        }
        return order;
    }
    static constexpr std::array<std::size_t, Size> get = [] {
        std::array<std::size_t, Size> order;
        for (uint i = 0; i < Size; ++i) {
            if constexpr (DecInTime) {
                order[i] = reverse_bit_order(i, log2i(Size));
            } else {
                order[i] = i;
            }
        }
        return order;
    }();
};

template<typename T, std::size_t PDest, std::size_t PSrc, bool ConjTw, bool Reverse, typename... Args>
inline void node8(std::array<T*, 8> dest, Args... args) {
    constexpr auto PLoad   = std::max(PSrc, avx::reg<T>::size);
    constexpr auto PStore  = std::max(PDest, avx::reg<T>::size);
    constexpr bool Inverse = ConjTw || Reverse;

    using src_type       = std::array<const T*, 8>;
    using tw_type        = std::array<avx::cx_reg<T>, 7>;
    constexpr bool Src   = has_type<src_type, Args...>;
    constexpr bool Tw    = has_type<tw_type, Args...>;
    constexpr bool Scale = has_type<avx::reg_t<T>, Args...>;

    if constexpr (Reverse) {
        avx::cx_reg<T> c4, c5, c6, c7;
        if constexpr (Src) {
            auto& src = std::get<src_type&>(std::tie(args...));
            c4        = avx::cxload<PLoad>(src[4]);
            c5        = avx::cxload<PLoad>(src[5]);
            c6        = avx::cxload<PLoad>(src[6]);
            c7        = avx::cxload<PLoad>(src[7]);
        } else {
            c4 = avx::cxload<PLoad>(dest[4]);
            c5 = avx::cxload<PLoad>(dest[5]);
            c6 = avx::cxload<PLoad>(dest[6]);
            c7 = avx::cxload<PLoad>(dest[7]);
        }
        std::tie(c4, c5, c6, c7) = avx::convert<T>::template repack<PSrc, PLoad>(c4, c5, c6, c7);
        std::tie(c4, c5, c6, c7) = avx::convert<T>::template inverse<Inverse>(c4, c5, c6, c7);
        avx::cx_reg<T> a4, a5, a6, a7;
        if constexpr (Tw) {
            auto& tw        = std::get<tw_type&>(std::tie(args...));
            auto [b4, b5tw] = avx::ibtfly(c4, c5);
            auto [b6, b7tw] = avx::ibtfly(c6, c7);
            auto [b5, b7]   = avx::mul({b5tw, tw[5]}, {b7tw, tw[6]});

            std::tie(a4, a6) = avx::ibtfly(b4, b6);
            std::tie(a5, a7) = avx::ibtfly(b5, b7);
            std::tie(a6, a7) = avx::mul({a6, tw[2]}, {a7, tw[2]});
        } else {
            static const T sq2 = std::sqrt(double{2}) / 2;
            auto [b4, b5_tw]   = avx::ibtfly(c4, c5);
            auto [b6, b7_tw]   = avx::ibtfly<2>(c6, c7);
            auto twsq2         = avx::broadcast(sq2);
            b5_tw              = avx::mul(b5_tw, twsq2);
            b7_tw              = avx::mul(b7_tw, twsq2);

            std::tie(a4, a6) = avx::ibtfly<3>(b4, b6);
            auto b5 = avx::cx_reg<T>{avx::add(b5_tw.real, b5_tw.imag), avx::sub(b5_tw.imag, b5_tw.real)};
            auto b7 = avx::cx_reg<T>{avx::sub(b7_tw.real, b7_tw.imag), avx::add(b7_tw.real, b7_tw.imag)};
            std::tie(a5, a7) = avx::ibtfly<3>(b5, b7);
        }
        avx::cx_reg<T> c0, c1, c2, c3;
        if constexpr (Src) {
            auto& src = std::get<src_type&>(std::tie(args...));
            c0        = avx::cxload<PLoad>(src[0]);
            c1        = avx::cxload<PLoad>(src[1]);
            c2        = avx::cxload<PLoad>(src[2]);
            c3        = avx::cxload<PLoad>(src[3]);
        } else {
            c0 = avx::cxload<PLoad>(dest[0]);
            c1 = avx::cxload<PLoad>(dest[1]);
            c2 = avx::cxload<PLoad>(dest[2]);
            c3 = avx::cxload<PLoad>(dest[3]);
        }
        std::tie(c0, c1, c2, c3) = avx::convert<T>::template repack<PSrc, PLoad>(c0, c1, c2, c3);
        std::tie(c0, c1, c2, c3) = avx::convert<T>::template inverse<Inverse>(c0, c1, c2, c3);

        avx::cx_reg<T> p0, p2, p4, p6, a1, a3;
        if constexpr (Tw) {
            auto& tw        = std::get<tw_type&>(std::tie(args...));
            auto [b0, b1tw] = avx::ibtfly(c0, c1);
            auto [b2, b3tw] = avx::ibtfly(c2, c3);
            auto [b1, b3]   = avx::mul({b1tw, tw[3]}, {b3tw, tw[4]});

            auto [a0, a2]    = avx::ibtfly(b0, b2);
            std::tie(a1, a3) = avx::ibtfly(b1, b3);
            std::tie(a2, a3) = avx::mul({a2, tw[1]}, {a3, tw[1]});

            std::tie(p0, p4) = avx::ibtfly(a0, a4);
            std::tie(p2, p6) = avx::ibtfly(a2, a6);
            std::tie(p4, p6) = avx::mul({p4, tw[0]}, {p6, tw[0]});
        } else {
            auto [b0, b1] = avx::ibtfly(c0, c1);
            auto [b2, b3] = avx::ibtfly<3>(c2, c3);

            auto [a0, a2]    = avx::ibtfly(b0, b2);
            std::tie(a1, a3) = avx::ibtfly(b1, b3);

            std::tie(p0, p4) = avx::ibtfly(a0, a4);
            std::tie(p2, p6) = avx::ibtfly(a2, a6);
        }
        if constexpr (Scale) {
            auto& scaling = std::get<avx::reg_t<T>&>(std::tie(args...));
            p0            = avx::mul(p0, scaling);
            p4            = avx::mul(p4, scaling);
            p2            = avx::mul(p2, scaling);
            p6            = avx::mul(p6, scaling);
        }
        std::tie(p0, p4, p2, p6) = avx::convert<T>::template inverse<Inverse>(p0, p4, p2, p6);
        std::tie(p0, p4, p2, p6) = avx::convert<T>::template repack<PStore, PDest>(p0, p4, p2, p6);

        avx::cxstore<PStore>(dest[0], p0);
        avx::cxstore<PStore>(dest[4], p4);
        avx::cxstore<PStore>(dest[2], p2);
        avx::cxstore<PStore>(dest[6], p6);

        auto [p1, p5] = avx::ibtfly(a1, a5);
        auto [p3, p7] = avx::ibtfly(a3, a7);
        if constexpr (Tw) {
            auto& tw         = std::get<tw_type&>(std::tie(args...));
            std::tie(p5, p7) = avx::mul({p5, tw[0]}, {p7, tw[0]});
        }
        if constexpr (Scale) {
            auto& scaling = std::get<avx::reg_t<T>&>(std::tie(args...));
            p1            = avx::mul(p1, scaling);
            p5            = avx::mul(p5, scaling);
            p3            = avx::mul(p3, scaling);
            p7            = avx::mul(p7, scaling);
        }
        std::tie(p1, p5, p3, p7) = avx::convert<T>::template inverse<Inverse>(p1, p5, p3, p7);
        std::tie(p1, p5, p3, p7) = avx::convert<T>::template repack<PStore, PDest>(p1, p5, p3, p7);

        avx::cxstore<PStore>(dest[1], p1);
        avx::cxstore<PStore>(dest[5], p5);
        avx::cxstore<PStore>(dest[3], p3);
        avx::cxstore<PStore>(dest[7], p7);
    } else {
        avx::cx_reg<T> p1, p3, p5, p7;
        if constexpr (Src) {
            auto& src = std::get<src_type&>(std::tie(args...));
            p5        = avx::cxload<PLoad>(src[5]);
            p1        = avx::cxload<PLoad>(src[1]);
            p7        = avx::cxload<PLoad>(src[7]);
            p3        = avx::cxload<PLoad>(src[3]);
        } else {
            p5 = avx::cxload<PLoad>(dest[5]);
            p1 = avx::cxload<PLoad>(dest[1]);
            p7 = avx::cxload<PLoad>(dest[7]);
            p3 = avx::cxload<PLoad>(dest[3]);
        }
        std::tie(p5, p1, p7, p3) = avx::convert<T>::template repack<PSrc, PLoad>(p5, p1, p7, p3);
        std::tie(p5, p1, p7, p3) = avx::convert<T>::template inverse<Inverse>(p5, p1, p7, p3);

        avx::cx_reg<T> b1, b3, b5, b7;
        if constexpr (Tw) {
            auto& tw          = std::get<tw_type&>(std::tie(args...));
            auto [p5tw, p7tw] = avx::mul({p5, tw[0]}, {p7, tw[0]});
            auto [a1, a5]     = avx::btfly(p1, p5tw);
            auto [a3, a7]     = avx::btfly(p3, p7tw);

            auto [a3tw, a7tw] = avx::mul({a3, tw[1]}, {a7, tw[2]});
            std::tie(b1, b3)  = avx::btfly(a1, a3tw);
            std::tie(b5, b7)  = avx::btfly(a5, a7tw);

            std::tie(b1, b3, b5, b7) = avx::mul({b1, tw[3]}, {b3, tw[4]}, {b5, tw[5]}, {b7, tw[6]});
        } else {
            static const T sq2 = std::sqrt(double{2}) / 2;
            auto [a1, a5]      = avx::btfly(p1, p5);
            auto [a3, a7]      = avx::btfly(p3, p7);

            std::tie(b1, b3) = avx::btfly(a1, a3);
            std::tie(b5, b7) = avx::btfly<3>(a5, a7);
            auto b5_tw       = avx::cx_reg<T>{avx::add(b5.real, b5.imag), avx::sub(b5.imag, b5.real)};
            auto b7_tw       = avx::cx_reg<T>{avx::sub(b7.real, b7.imag), avx::add(b7.real, b7.imag)};
            auto twsq2       = avx::broadcast(sq2);
            b5               = avx::mul(b5_tw, twsq2);
            b7               = avx::mul(b7_tw, twsq2);
        }
        avx::cx_reg<T> p0, p2, p4, p6;
        if constexpr (Src) {
            auto& src = std::get<src_type&>(std::tie(args...));
            p4        = avx::cxload<PLoad>(src[4]);
            p0        = avx::cxload<PLoad>(src[0]);
            p6        = avx::cxload<PLoad>(src[6]);
            p2        = avx::cxload<PLoad>(src[2]);
        } else {
            p4 = avx::cxload<PLoad>(dest[4]);
            p0 = avx::cxload<PLoad>(dest[0]);
            p6 = avx::cxload<PLoad>(dest[6]);
            p2 = avx::cxload<PLoad>(dest[2]);
        }
        std::tie(p4, p0, p6, p2) = avx::convert<T>::template repack<PSrc, PLoad>(p4, p0, p6, p2);
        std::tie(p4, p0, p6, p2) = avx::convert<T>::template inverse<Inverse>(p4, p0, p6, p2);

        avx::cx_reg<T> c0, c1, c2, c3, b4, b6;
        if constexpr (Tw) {
            auto& tw          = std::get<tw_type&>(std::tie(args...));
            auto [p4tw, p6tw] = avx::mul({p4, tw[0]}, {p6, tw[0]});
            auto [a0, a4]     = avx::btfly(p0, p4tw);
            auto [a2, a6]     = avx::btfly(p2, p6tw);

            auto [a2tw, a6tw] = avx::mul({a2, tw[1]}, {a6, tw[2]});
            auto [b0, b2]     = avx::btfly(a0, a2tw);
            std::tie(b4, b6)  = avx::btfly(a4, a6tw);

            std::tie(c0, c1) = avx::btfly(b0, b1);
            std::tie(c2, c3) = avx::btfly(b2, b3);
        } else {
            auto [a0, a4] = avx::btfly(p0, p4);
            auto [a2, a6] = avx::btfly(p2, p6);

            auto [b0, b2]    = avx::btfly(a0, a2);
            std::tie(b4, b6) = avx::btfly<3>(a4, a6);

            std::tie(c0, c1) = avx::btfly(b0, b1);
            std::tie(c2, c3) = avx::btfly<3>(b2, b3);
        }
        if constexpr (Scale) {
            auto& scaling = std::get<avx::reg_t<T>&>(std::tie(args...));
            c0            = avx::mul(c0, scaling);
            c1            = avx::mul(c1, scaling);
            c2            = avx::mul(c2, scaling);
            c3            = avx::mul(c3, scaling);
        }
        std::tie(c0, c1, c2, c3) = avx::convert<T>::template inverse<Inverse>(c0, c1, c2, c3);
        std::tie(c0, c1, c2, c3) = avx::convert<T>::template repack<PStore, PDest>(c0, c1, c2, c3);

        cxstore<PStore>(dest[0], c0);
        cxstore<PStore>(dest[1], c1);
        cxstore<PStore>(dest[2], c2);
        cxstore<PStore>(dest[3], c3);

        avx::cx_reg<T> c4, c5, c6, c7;
        if constexpr (Tw) {
            std::tie(c4, c5) = avx::btfly(b4, b5);
            std::tie(c6, c7) = avx::btfly(b6, b7);
        } else {
            std::tie(c4, c5) = avx::btfly(b4, b5);
            std::tie(c6, c7) = avx::btfly<2>(b6, b7);
        }
        if constexpr (Scale) {
            auto& scaling = std::get<avx::reg_t<T>&>(std::tie(args...));
            c4            = avx::mul(c4, scaling);
            c5            = avx::mul(c5, scaling);
            c6            = avx::mul(c6, scaling);
            c7            = avx::mul(c7, scaling);
        }
        std::tie(c4, c5, c6, c7) = avx::convert<T>::template inverse<Inverse>(c4, c5, c6, c7);
        std::tie(c4, c5, c6, c7) = avx::convert<T>::template repack<PStore, PDest>(c4, c5, c6, c7);

        cxstore<PStore>(dest[4], c4);
        cxstore<PStore>(dest[5], c5);
        cxstore<PStore>(dest[6], c6);
        cxstore<PStore>(dest[7], c7);
    }
};

/**
  * @brief Performes two levels of FFT butterflies.
  * If data is in ascending order, equivalent to decimation in frequency.
  * To perform decimation in time switch data[1] and data[2].
  *
  * First step performs data[0,1]±tw[0]×data[2,3].
  * Second step performs data[0,2]±tw[1,2]×data[1,3].
  *
  * If an array of pointers 'src' is provided, uses 'src' as source and dest as destination.
  * If an array of complex SIMD vectors 'tw' is provided, uses 'tw' as twiddles for butterflies,
  * otherwise uses fixed twiddles for FFT sizes 2, 4 (tw = {1, 1, -j}).
  * If a single SIMD vector 'scaling' is provided, multiplies the result by 'scaling'.
  *
  * @tparam T
  * @tparam PDest pack size of destination
  * @tparam PSrc pack size of source
  * @tparam ConjTw use complex conjugate of twiddles (by switching data's real and imaginary parts)
  * @tparam Reverse perform operations in reverse order with complex conjugate of twiddles (unoffected by ConjTw)
  * @tparam Args
  * @param dest destination pointers (also source pointers if no explicit source pointers provided)
  * @param args any combination of std::array<const T*, 4> src, std::array<avx::cx_reg<T>, 3> tw, avx::reg_t<T> scaling
  */
template<typename T,
         std::size_t PDest,
         std::size_t PSrc,
         bool        ConjTw,
         bool        Reverse,
         bool        DIT = false,
         typename... Args>
inline void node4(std::array<T*, 4> dest, Args... args) {
    constexpr auto PLoad   = std::max(PSrc, avx::reg<T>::size);
    constexpr auto PStore  = std::max(PDest, avx::reg<T>::size);
    constexpr bool Inverse = ConjTw || Reverse;

    using src_type       = std::array<const T*, 4>;
    using tw_type        = std::array<avx::cx_reg<T>, 3>;
    constexpr bool Src   = has_type<src_type, Args...>;
    constexpr bool Tw    = has_type<tw_type, Args...>;
    constexpr bool Scale = has_type<avx::reg_t<T>, Args...>;

    // NOLINTNEXTLINE(*-declaration)
    avx::cx_reg<T> p0, p1, p2, p3;
    if constexpr (Src) {
        auto& src = std::get<src_type&>(std::tie(args...));
        p2        = avx::cxload<PLoad>(src[order<4, DIT>::get[2]]);
        p3        = avx::cxload<PLoad>(src[order<4, DIT>::get[3]]);
        p0        = avx::cxload<PLoad>(src[order<4, DIT>::get[0]]);
        p1        = avx::cxload<PLoad>(src[order<4, DIT>::get[1]]);
    } else {
        p2 = avx::cxload<PLoad>(dest[order<4, DIT>::get[2]]);
        p3 = avx::cxload<PLoad>(dest[order<4, DIT>::get[3]]);
        p0 = avx::cxload<PLoad>(dest[order<4, DIT>::get[0]]);
        p1 = avx::cxload<PLoad>(dest[order<4, DIT>::get[1]]);
    }
    std::tie(p2, p3, p0, p1) = avx::convert<T>::template repack<PSrc, PLoad>(p2, p3, p0, p1);
    std::tie(p2, p3, p0, p1) = avx::convert<T>::template inverse<Inverse>(p2, p3, p0, p1);
    // NOLINTNEXTLINE(*-declaration)
    avx::cx_reg<T> b0, b1, b2, b3;
    if constexpr (Tw) {
        auto& tw = std::get<tw_type&>(std::tie(args...));
        if constexpr (Reverse) {
            auto [a2, a3tw]  = avx::ibtfly(p2, p3);
            auto [a0, a1tw]  = avx::ibtfly(p0, p1);
            auto [a3, a1]    = avx::mul({a3tw, tw[2]}, {a1tw, tw[1]});
            std::tie(b0, b2) = avx::ibtfly(a0, a2);
            std::tie(b1, b3) = avx::ibtfly(a1, a3);
            std::tie(b2, b3) = avx::mul({b2, tw[0]}, {b3, tw[0]});
        } else {
            auto [p2tw, p3tw] = avx::mul({p2, tw[0]}, {p3, tw[0]});
            auto [a0, a2]     = avx::btfly(p0, p2tw);
            auto [a1, a3]     = avx::btfly(p1, p3tw);
            auto [a1tw, a3tw] = avx::mul({a1, tw[1]}, {a3, tw[2]});
            std::tie(b0, b1)  = avx::btfly(a0, a1tw);
            std::tie(b2, b3)  = avx::btfly(a2, a3tw);
        }
    } else {
        if constexpr (Reverse) {
            auto [a2, a3]    = avx::ibtfly<3>(p2, p3);
            auto [a0, a1]    = avx::ibtfly(p0, p1);
            std::tie(b0, b2) = avx::ibtfly(a0, a2);
            std::tie(b1, b3) = avx::ibtfly(a1, a3);
        } else {
            auto [a0, a2]    = avx::btfly(p0, p2);
            auto [a1, a3]    = avx::btfly(p1, p3);
            std::tie(b0, b1) = avx::btfly(a0, a1);
            std::tie(b2, b3) = avx::btfly<3>(a2, a3);
        }
    }
    if constexpr (Scale) {
        auto& scaling = std::get<avx::reg_t<T>&>(std::tie(args...));
        b0            = avx::mul(b0, scaling);
        b1            = avx::mul(b1, scaling);
        b2            = avx::mul(b2, scaling);
        b3            = avx::mul(b3, scaling);
    }
    std::tie(b0, b1, b2, b3) = avx::convert<T>::template inverse<Inverse>(b0, b1, b2, b3);
    std::tie(b0, b1, b2, b3) = avx::convert<T>::template repack<PStore, PDest>(b0, b1, b2, b3);

    cxstore<PStore>(dest[order<4, DIT>::get[0]], b0);
    cxstore<PStore>(dest[order<4, DIT>::get[1]], b1);
    cxstore<PStore>(dest[order<4, DIT>::get[2]], b2);
    cxstore<PStore>(dest[order<4, DIT>::get[3]], b3);
};

template<typename T, std::size_t PDest, std::size_t PSrc, bool ConjTw, bool Reverse, typename... Args>
inline void node2(std::array<T*, 2> dest, Args... args) {
    constexpr auto PLoad   = std::max(PSrc, avx::reg<T>::size);
    constexpr auto PStore  = std::max(PDest, avx::reg<T>::size);
    constexpr bool Inverse = ConjTw || Reverse;

    using src_type       = std::array<const T*, 2>;
    using tw_type        = avx::cx_reg<T>;
    constexpr bool Src   = has_type<src_type, Args...>;
    constexpr bool Tw    = has_type<tw_type, Args...>;
    constexpr bool Scale = has_type<avx::reg_t<T>, Args...>;

    // NOLINTNEXTLINE(*-declaration)
    avx::cx_reg<T> p0, p1;
    if constexpr (Src) {
        auto& src = std::get<src_type&>(std::tie(args...));
        p0        = avx::cxload<PLoad>(src[0]);
        p1        = avx::cxload<PLoad>(src[1]);
    } else {
        p0 = avx::cxload<PLoad>(dest[0]);
        p1 = avx::cxload<PLoad>(dest[1]);
    }
    std::tie(p0, p1) = avx::convert<T>::template repack<PSrc, PLoad>(p0, p1);
    std::tie(p0, p1) = avx::convert<T>::template inverse<Inverse>(p0, p1);
    // NOLINTNEXTLINE(*-declaration)
    avx::cx_reg<T> a0, a1;
    if constexpr (Reverse) {
        std::tie(a0, a1) = avx::ibtfly(p0, p1);
        if constexpr (Tw) {
            auto& tw = std::get<tw_type&>(std::tie(args...));
            a1       = avx::mul(a1, tw);
        }
    } else {
        if constexpr (Tw) {
            auto& tw = std::get<tw_type&>(std::tie(args...));
            p1       = avx::mul(p1, tw);
        }
        std::tie(a0, a1) = avx::btfly(p0, p1);
    }
    if constexpr (Scale) {
        auto& scaling = std::get<avx::reg_t<T>&>(std::tie(args...));
        a0            = avx::mul(a0, scaling);
        a1            = avx::mul(a1, scaling);
    }
    std::tie(a0, a1) = avx::convert<T>::template inverse<Inverse>(a0, a1);
    std::tie(a0, a1) = avx::convert<T>::template repack<PStore, PDest>(a0, a1);

    cxstore<PStore>(dest[0], a0);
    cxstore<PStore>(dest[1], a1);
};

template<typename T>
inline auto wnk(std::size_t n, std::size_t k) -> std::complex<T> {
    constexpr double pi = 3.14159265358979323846;
    if (n == k * 4) {
        return {0, -1};
    }
    if (n == k * 2) {
        return {-1, 0};
    }
    return exp(std::complex<T>(0, -2 * pi * static_cast<double>(k) / static_cast<double>(n)));
}
}    // namespace fft
}    // namespace internal


template<typename T,
         std::size_t Size    = pcx::dynamic_size,
         std::size_t SubSize = pcx::default_pack_size<T>,
         typename Allocator  = pcx::aligned_allocator<T, std::align_val_t(64)>>
    requires(std::same_as<T, float> || std::same_as<T, double>) &&
            (pcx::power_of_two<Size> || (Size == pcx::dynamic_size)) &&
            (pcx::power_of_two<SubSize> && SubSize >= pcx::default_pack_size<T> ||
             (SubSize == pcx::dynamic_size))
class fft_unit {
public:
    using real_type      = T;
    using allocator_type = Allocator;

    // static constexpr auto pack_size = default_pack_size<T>;
    static constexpr std::size_t reg_size = 32 / sizeof(real_type);

private:
    using size_t    = std::conditional_t<Size == pcx::dynamic_size, std::size_t, decltype([]() {})>;
    using subsize_t = std::conditional_t<SubSize == pcx::dynamic_size, std::size_t, decltype([]() {})>;
    using sort_allocator_type =
        typename std::allocator_traits<allocator_type>::template rebind_alloc<std::size_t>;

public:
    explicit fft_unit(allocator_type allocator = allocator_type())
        requires(Size != pcx::dynamic_size) && (SubSize != pcx::dynamic_size)
    : m_sort(get_sort(size(), static_cast<sort_allocator_type>(allocator)))
    , m_twiddles(get_twiddles(size(), sub_size(), allocator))
    , m_twiddles_unsorted(get_twiddles_unsorted(size(), sub_size(), allocator)){};

    explicit fft_unit(std::size_t sub_size = 1, allocator_type allocator = allocator_type())
        requires(Size != pcx::dynamic_size) && (SubSize == pcx::dynamic_size)
    : m_sub_size(check_sub_size(sub_size))
    , m_sort(get_sort(size(), static_cast<sort_allocator_type>(allocator)))
    , m_twiddles(get_twiddles(size(), sub_size, allocator))
    , m_twiddles_unsorted(get_twiddles_unsorted(size(), sub_size, allocator)){};

    explicit fft_unit(std::size_t fft_size, allocator_type allocator = allocator_type())
        requires(Size == pcx::dynamic_size) && (SubSize != pcx::dynamic_size)
    : m_size(check_size(fft_size))
    , m_sort(get_sort(size(), static_cast<sort_allocator_type>(allocator)))
    , m_twiddles(get_twiddles(size(), sub_size(), allocator))
    , m_twiddles_unsorted(get_twiddles_unsorted(size(), sub_size(), allocator)){};

    explicit fft_unit(std::size_t    fft_size,
                      std::size_t    sub_size  = 1,
                      allocator_type allocator = allocator_type())
        requires(Size == pcx::dynamic_size) && (SubSize == pcx::dynamic_size)
    : m_size(check_size(fft_size))
    , m_sub_size(check_sub_size(sub_size))
    , m_sort(get_sort(size(), static_cast<sort_allocator_type>(allocator)))
    , m_twiddles(get_twiddles(size(), sub_size, allocator))
    , m_twiddles_unsorted(get_twiddles_unsorted(size(), sub_size, allocator)){};

    fft_unit(const fft_unit& other)     = default;
    fft_unit(fft_unit&& other) noexcept = default;

    ~fft_unit() = default;

    fft_unit& operator=(const fft_unit& other)     = default;
    fft_unit& operator=(fft_unit&& other) noexcept = default;

    [[nodiscard]] constexpr auto size() const -> std::size_t {
        if constexpr (Size == pcx::dynamic_size) {
            return m_size;
        } else {
            return Size;
        }
    }

    template<typename VAllocator, std::size_t VPackSize>
    void operator()(pcx::vector<T, VAllocator, VPackSize>& vector) {
        assert(size() == vector.size());
        fft_internal<VPackSize>(vector.data());
    };

    template<typename VAllocator>
    void operator()(std::vector<std::complex<T>, VAllocator>& vector) {
        assert(size() == vector.size());
        fft_internal<1>(reinterpret_cast<T*>(vector.data()));
    };

    template<typename VAllocator, std::size_t VPackSize>
    void unsorted(pcx::vector<T, VAllocator, VPackSize>& vector) {
        if (size() != vector.size()) {
            throw(std::string("Size, which is ") + std::to_string(size()) + "not equal vector.size() " +
                  std::to_string(vector.size()));
        }
        fftu_internal<VPackSize>(vector.data());
    };
    template<typename VAllocator>
    void unsorted(std::vector<std::complex<T>, VAllocator>& vector) {
        assert(size() == vector.size());
        fftu_internal<1>(reinterpret_cast<T*>(vector.data()));
    };

    template<typename VAllocator1, std::size_t VPackSize1, typename VAllocator2, std::size_t VPackSize2>
    void unsorted(pcx::vector<T, VAllocator1, VPackSize1>&       dest,
                  const pcx::vector<T, VAllocator2, VPackSize2>& source) {
        if (size() != dest.size()) {
            throw(std::string("Size, which is ") + std::to_string(size()) + "not equal dest.size() " +
                  std::to_string(dest.size()));
        }
        fftu_internal<VPackSize1, VPackSize2>(
            dest.data(), source.data(), source.size() < size() ? source.size() : 0);
    };

    template<typename VAllocator, std::size_t VPackSize>
    void ifft(pcx::vector<T, VAllocator, VPackSize>& vector) {
        assert(size() == vector.size());
        ifft_internal<VPackSize>(vector.data());
    };
    template<typename VAllocator>
    void ifft(std::vector<std::complex<T>, VAllocator>& vector) {
        assert(size() == vector.size());
        ifft_internal<1>(reinterpret_cast<T*>(vector.data()));
    };

    template<typename VAllocator>
    void operator()(std::vector<std::complex<T>, VAllocator>&       dest,
                    const std::vector<std::complex<T>, VAllocator>& source) {
        assert(size() == dest.size() && size() == source.size());
        if (&dest == &source) {
            fft_internal<1>(reinterpret_cast<T*>(dest.data()));
        } else {
            fft_internal<1, 1>(reinterpret_cast<T*>(dest.data()), reinterpret_cast<const T*>(source.data()));
        }
    };

    template<typename VAllocator, std::size_t VPackSize>
    void operator()(pcx::vector<T, VAllocator, VPackSize>&       dest,
                    const pcx::vector<T, VAllocator, VPackSize>& source) {
        assert(size() == dest.size() && size() == source.size());
        if (&dest == &source) {
            fft_internal<VPackSize>(dest.data());
        } else {
            fft_internal<VPackSize, VPackSize>(dest.data(), source.data());
        }
    };

private:
    [[no_unique_address]] size_t                           m_size;
    [[no_unique_address]] subsize_t                        m_sub_size;
    const std::vector<std::size_t, sort_allocator_type>    m_sort;
    const pcx::vector<real_type, allocator_type, reg_size> m_twiddles;
    const std::vector<real_type, allocator_type>           m_twiddles_unsorted;

    [[nodiscard]] constexpr auto sub_size() const -> std::size_t {
        if constexpr (SubSize == pcx::dynamic_size) {
            return m_sub_size;
        } else {
            return SubSize;
        }
    }

public:
    template<std::size_t PData>
    inline void fft_internal(float* data) {
        constexpr auto PTform = std::max(PData, reg_size);
        depth3_and_sort<PTform, PData>(data);
        subtransform_recursive<PData, PTform>(data, size());
    }

    template<std::size_t PData, std::size_t Scale = true>
    inline void ifft_internal(float* data) {
        constexpr auto PTform = std::max(PData, reg_size);
        depth3_and_sort<PTform, PData, true>(data);
        subtransform_recursive<PData, PTform, true, Scale>(data, size());
    }

    template<std::size_t PDest, std::size_t PSrc>
    inline void fft_internal(float* dest, const float* source) {
        constexpr auto PTform = std::max(PDest, reg_size);
        depth3_and_sort<PTform, PSrc>(dest, source);
        subtransform_recursive<PDest, PTform>(dest, size());
    };

    template<std::size_t PData, bool BitReversed = true>
    void fftu_internal(float* data) {
        auto* twiddle_ptr = m_twiddles_unsorted.data();
        if (log2i(size() / (reg_size * 4)) % 2 == 0) {
            unsorted_subtransform_recursive<PData, PData, true, BitReversed>(data, size(), twiddle_ptr);
        } else if (size() / (reg_size * 4) > 8) {
            constexpr auto PTform = std::max(PData, reg_size);
            for (std::size_t i_group = 0; i_group < size() / 8 / reg_size; ++i_group) {
                node8_dif_along<PTform, PData>(data, size(), i_group * reg_size);
            }
            twiddle_ptr = unsorted_subtransform_recursive<PData, PTform, true, BitReversed>(
                data, size() / 8, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive<PData, PTform, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 8 * 1), size() / 8, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive<PData, PTform, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 8 * 2), size() / 8, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive<PData, PTform, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 8 * 3), size() / 8, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive<PData, PTform, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 8 * 4), size() / 8, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive<PData, PTform, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 8 * 5), size() / 8, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive<PData, PTform, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 8 * 6), size() / 8, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive<PData, PTform, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 8 * 7), size() / 8, twiddle_ptr);
        } else {
            constexpr auto PTform = std::max(PData, reg_size);

            using reg_t = avx::cx_reg<float>;
            reg_t tw0   = {
                avx::broadcast(twiddle_ptr++),
                avx::broadcast(twiddle_ptr++),
            };

            for (std::size_t i_group = 0; i_group < size() / 2 / reg_size; ++i_group) {
                node2_along<PTform, PData>(data, size(), i_group * reg_size);
            }
            twiddle_ptr = unsorted_subtransform_recursive<PData, PTform, true, BitReversed>(
                data, size() / 2, twiddle_ptr);
            unsorted_subtransform_recursive<PData, PTform, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 2), size() / 2, twiddle_ptr);
        }
    }

    template<std::size_t PDest, std::size_t PSrc, bool BitReversed = true>
    void fftu_internal(float* dest, const float* source, std::size_t source_size = 0) {
        auto* twiddle_ptr = m_twiddles_unsorted.data();
        if (source_size == 0) {
            if (log2i(size() / (reg_size * 4)) % 2 == 0) {
                unsorted_subtransform_recursive<PDest, PDest, true, BitReversed>(dest, size(), twiddle_ptr);
            } else if (size() / (reg_size * 4) > 8) {
                constexpr auto PTform = std::max(PDest, reg_size);
                for (std::size_t i_group = 0; i_group < size() / 8 / reg_size; ++i_group) {
                    node8_dif_along<PTform, PDest>(dest, size(), i_group * reg_size);
                }
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, true, BitReversed>(
                    dest, size() / 8, twiddle_ptr);
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 8 * 1), size() / 8, twiddle_ptr);
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 8 * 2), size() / 8, twiddle_ptr);
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 8 * 3), size() / 8, twiddle_ptr);
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 8 * 4), size() / 8, twiddle_ptr);
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 8 * 5), size() / 8, twiddle_ptr);
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 8 * 6), size() / 8, twiddle_ptr);
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 8 * 7), size() / 8, twiddle_ptr);
            } else {
                constexpr auto PTform = std::max(PDest, reg_size);

                using reg_t = avx::cx_reg<float>;
                reg_t tw0   = {
                    avx::broadcast(twiddle_ptr++),
                    avx::broadcast(twiddle_ptr++),
                };

                for (std::size_t i_group = 0; i_group < size() / 2 / reg_size; ++i_group) {
                    auto* ptr0 = avx::ra_addr<PTform>(dest, i_group * reg_size);
                    auto* ptr1 = avx::ra_addr<PTform>(dest, i_group * reg_size + size() / 2);

                    auto p1 = avx::cxload<PTform>(ptr1);
                    auto p0 = avx::cxload<PTform>(ptr0);

                    if constexpr (PDest < PTform) {
                        std::tie(p1, p0) = avx::convert<float>::repack<PDest, PTform>(p1, p0);
                    }

                    auto p1tw = avx::mul(p1, tw0);

                    auto [a0, a1] = avx::btfly(p0, p1tw);

                    cxstore<PTform>(ptr0, a0);
                    cxstore<PTform>(ptr1, a1);
                }

                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, true, BitReversed>(
                    dest, size() / 2, twiddle_ptr);
                unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 2), size() / 2, twiddle_ptr);
            }
        } else {
            if (log2i(size() / (reg_size * 4)) % 2 == 0) {
                unsorted_subtransform_recursive_fill0<PDest, PDest, true, BitReversed>(
                    dest, source, size(), source_size, twiddle_ptr);
            } else if (size() / (reg_size * 4) > 8) {
                constexpr auto PTform = std::max(PDest, reg_size);
                for (std::size_t i_group = 0; i_group < size() / 8 / reg_size; ++i_group) {
                    node8_dif_along_fill0<PTform, PSrc>(
                        dest, source, size(), source_size, i_group * reg_size);
                }
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, true, BitReversed>(
                    dest, size() / 8, twiddle_ptr);
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 8 * 1), size() / 8, twiddle_ptr);
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 8 * 2), size() / 8, twiddle_ptr);
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 8 * 3), size() / 8, twiddle_ptr);
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 8 * 4), size() / 8, twiddle_ptr);
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 8 * 5), size() / 8, twiddle_ptr);
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 8 * 6), size() / 8, twiddle_ptr);
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 8 * 7), size() / 8, twiddle_ptr);
            } else {
                constexpr auto PTform = std::max(PDest, reg_size);

                using reg_t = avx::cx_reg<float>;
                reg_t tw0   = {
                    avx::broadcast(twiddle_ptr++),
                    avx::broadcast(twiddle_ptr++),
                };
                for (std::size_t i_group = 0; i_group < size() / 2 / reg_size; ++i_group) {
                    node2_along<PTform, PSrc>(dest, size(), i_group * reg_size, tw0, source, source_size);
                }
                twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, true, BitReversed>(
                    dest, size() / 2, twiddle_ptr);
                unsorted_subtransform_recursive<PDest, PTform, false, BitReversed>(
                    avx::ra_addr<PTform>(dest, size() / 2), size() / 2, twiddle_ptr);
            }
        }
    }

    template<std::size_t PData, bool BitReversed = true>
    void fftu_internal_inverse(float* data) {
        auto* twiddle_ptr = &(*m_twiddles_unsorted.end());
        if (log2i(size() / (reg_size * 4)) % 2 == 0) {
            unsorted_subtransform_recursive_inverse<PData, PData, true, BitReversed>(
                data, size(), twiddle_ptr);
        } else if (size() / (reg_size * 4) > 8) {
            constexpr auto PTform = std::max(PData, reg_size);

            twiddle_ptr = unsorted_subtransform_recursive_inverse<PTform, PData, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 8 * 7), size() / 8, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive_inverse<PTform, PData, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 8 * 6), size() / 8, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive_inverse<PTform, PData, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 8 * 5), size() / 8, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive_inverse<PTform, PData, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 8 * 4), size() / 8, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive_inverse<PTform, PData, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 8 * 3), size() / 8, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive_inverse<PTform, PData, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 8 * 2), size() / 8, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive_inverse<PTform, PData, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 8 * 1), size() / 8, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive_inverse<PTform, PData, true, BitReversed>(
                data, size() / 8, twiddle_ptr);
            for (std::size_t i_group = 0; i_group < size() / 8 / reg_size; ++i_group) {
                node8_dif_along<PData, PTform, true>(data, size(), i_group * reg_size);
            }
        } else {
            constexpr auto PTform = std::max(PData, reg_size);

            twiddle_ptr = unsorted_subtransform_recursive_inverse<PData, PTform, false, BitReversed>(
                avx::ra_addr<PTform>(data, size() / 2), size() / 2, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive_inverse<PData, PTform, true, BitReversed>(
                data, size() / 2, twiddle_ptr);

            twiddle_ptr -= 2;

            using reg_t = avx::cx_reg<float>;
            reg_t tw0   = {
                avx::broadcast(twiddle_ptr),
                avx::broadcast(twiddle_ptr + 1),
            };
            for (std::size_t i_group = 0; i_group < size() / 2 / reg_size; ++i_group) {
                auto* ptr0 = avx::ra_addr<PTform>(data, i_group * reg_size);
                auto* ptr1 = avx::ra_addr<PTform>(data, i_group * reg_size + size() / 2);

                auto a1 = avx::cxload<PTform>(ptr1);
                auto a0 = avx::cxload<PTform>(ptr0);

                std::tie(a1, a0) = avx::convert<float>::repack<PData, PTform>(a1, a0);
                std::tie(a1, a0) = avx::convert<float>::inverse(a1, a0);

                auto [p0, p1] = avx::ibtfly(a0, a1);

                std::tie(p1, p0) = avx::convert<float>::inverse(p1, p0);

                cxstore<PTform>(ptr0, p0);
                cxstore<PTform>(ptr1, p1);
            }
        }
    }

    template<std::size_t PTform, std::size_t PSrc, bool Inverse = false>
    inline void depth3_and_sort(float* data) {
        const auto sq2 = internal::fft::wnk<T>(8, 1);
        // auto       twsq2 = avx::broadcast(sq2.real());

        auto* src0 = avx::ra_addr<PTform>(data, 0);
        auto* src1 = avx::ra_addr<PTform>(data, 1 * size() / 8);
        auto* src2 = avx::ra_addr<PTform>(data, 2 * size() / 8);
        auto* src3 = avx::ra_addr<PTform>(data, 3 * size() / 8);
        auto* src4 = avx::ra_addr<PTform>(data, 4 * size() / 8);
        auto* src5 = avx::ra_addr<PTform>(data, 5 * size() / 8);
        auto* src6 = avx::ra_addr<PTform>(data, 6 * size() / 8);
        auto* src7 = avx::ra_addr<PTform>(data, 7 * size() / 8);

        uint i = 0;
        for (; i < n_reversals(size() / 64); i += 2) {
            using reg_t = avx::cx_reg<float>;

            auto offset1 = m_sort[i] * reg_size;
            auto offset2 = m_sort[i + 1] * reg_size;

            auto p1 = avx::cxload<PTform>(avx::ra_addr<PTform>(src1, offset1));
            auto p5 = avx::cxload<PTform>(avx::ra_addr<PTform>(src5, offset1));
            auto p3 = avx::cxload<PTform>(avx::ra_addr<PTform>(src3, offset1));
            auto p7 = avx::cxload<PTform>(avx::ra_addr<PTform>(src7, offset1));

            _mm_prefetch(avx::ra_addr<PTform>(src0, offset1), _MM_HINT_T0);
            _mm_prefetch(avx::ra_addr<PTform>(src4, offset1), _MM_HINT_T0);
            _mm_prefetch(avx::ra_addr<PTform>(src2, offset1), _MM_HINT_T0);
            _mm_prefetch(avx::ra_addr<PTform>(src6, offset1), _MM_HINT_T0);


            std::tie(p1, p5, p3, p7) = avx::convert<float>::split<PSrc>(p1, p5, p3, p7);
            std::tie(p1, p5, p3, p7) = avx::convert<float>::inverse<Inverse>(p1, p5, p3, p7);

            auto [a1, a5] = avx::btfly(p1, p5);
            auto [a3, a7] = avx::btfly(p3, p7);

            auto [b5, b7] = avx::btfly<3>(a5, a7);

            auto twsq2 = avx::broadcast(sq2.real());

            reg_t b5_tw = {avx::add(b5.real, b5.imag), avx::sub(b5.imag, b5.real)};
            reg_t b7_tw = {avx::sub(b7.real, b7.imag), avx::add(b7.real, b7.imag)};

            auto [b1, b3] = avx::btfly(a1, a3);

            b5_tw = avx::mul(b5_tw, twsq2);
            b7_tw = avx::mul(b7_tw, twsq2);

            auto p0 = avx::cxload<PTform>(avx::ra_addr<PTform>(src0, offset1));
            auto p4 = avx::cxload<PTform>(avx::ra_addr<PTform>(src4, offset1));
            auto p2 = avx::cxload<PTform>(avx::ra_addr<PTform>(src2, offset1));
            auto p6 = avx::cxload<PTform>(avx::ra_addr<PTform>(src6, offset1));


            _mm_prefetch(avx::ra_addr<PTform>(src0, offset2), _MM_HINT_T0);
            _mm_prefetch(avx::ra_addr<PTform>(src4, offset2), _MM_HINT_T0);
            if constexpr (PSrc < 4) {
                _mm_prefetch(avx::ra_addr<PTform>(src2, offset2), _MM_HINT_T0);
                _mm_prefetch(avx::ra_addr<PTform>(src6, offset2), _MM_HINT_T0);
            } else {
                _mm_prefetch(avx::ra_addr<PTform>(src1, offset2), _MM_HINT_T0);
                _mm_prefetch(avx::ra_addr<PTform>(src5, offset2), _MM_HINT_T0);
            }

            std::tie(p0, p4, p2, p6) = avx::convert<float>::split<PSrc>(p0, p4, p2, p6);
            std::tie(p0, p4, p2, p6) = avx::convert<float>::inverse<Inverse>(p0, p4, p2, p6);

            auto [a0, a4] = avx::btfly(p0, p4);
            auto [a2, a6] = avx::btfly(p2, p6);

            auto [b0, b2] = avx::btfly(a0, a2);
            auto [b4, b6] = avx::btfly<3>(a4, a6);

            auto [c0, c1] = avx::btfly(b0, b1);
            auto [c2, c3] = avx::btfly<3>(b2, b3);
            auto [c4, c5] = avx::btfly(b4, b5_tw);
            auto [c6, c7] = avx::btfly<2>(b6, b7_tw);

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);

            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            reg_t q0, q1, q2, q3, q4, q5, q6, q7;

            std::tie(shc0, shc1, shc2, shc3) = avx::convert<float>::inverse<Inverse>(shc0, shc1, shc2, shc3);

            q0 = cxloadstore<PTform>(avx::ra_addr<PTform>(src0, offset2), shc0);
            q4 = cxloadstore<PTform>(avx::ra_addr<PTform>(src4, offset2), shc2);

            if constexpr (PSrc < 4) {
                q2 = cxloadstore<PTform>(avx::ra_addr<PTform>(src2, offset2), shc1);
                q6 = cxloadstore<PTform>(avx::ra_addr<PTform>(src6, offset2), shc3);

                std::tie(q0, q4, q2, q6) = avx::convert<float>::split<PSrc>(q0, q4, q2, q6);
                std::tie(q0, q4, q2, q6) = avx::convert<float>::inverse<Inverse>(q0, q4, q2, q6);
            } else {
                q1 = cxloadstore<PTform>(avx::ra_addr<PTform>(src1, offset2), shc1);
                q5 = cxloadstore<PTform>(avx::ra_addr<PTform>(src5, offset2), shc3);

                std::tie(q0, q4, q1, q5) = avx::convert<float>::split<PSrc>(q0, q4, q1, q5);
                std::tie(q0, q4, q1, q5) = avx::convert<float>::inverse<Inverse>(q0, q4, q1, q5);
            }

            if constexpr (PSrc < 4) {
                _mm_prefetch(avx::ra_addr<PTform>(src1, offset2), _MM_HINT_T0);
                _mm_prefetch(avx::ra_addr<PTform>(src5, offset2), _MM_HINT_T0);
            } else {
                _mm_prefetch(avx::ra_addr<PTform>(src2, offset2), _MM_HINT_T0);
                _mm_prefetch(avx::ra_addr<PTform>(src6, offset2), _MM_HINT_T0);
            }
            _mm_prefetch(avx::ra_addr<PTform>(src3, offset2), _MM_HINT_T0);
            _mm_prefetch(avx::ra_addr<PTform>(src7, offset2), _MM_HINT_T0);

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            std::tie(shc4, shc5, shc6, shc7) = avx::convert<float>::inverse<Inverse>(shc4, shc5, shc6, shc7);

            if constexpr (PSrc < 4) {
                q1 = cxloadstore<PTform>(avx::ra_addr<PTform>(src1, offset2), shc4);
                q5 = cxloadstore<PTform>(avx::ra_addr<PTform>(src5, offset2), shc6);
            } else {
                q2 = cxloadstore<PTform>(avx::ra_addr<PTform>(src2, offset2), shc4);
                q6 = cxloadstore<PTform>(avx::ra_addr<PTform>(src6, offset2), shc6);
            }
            q3 = cxloadstore<PTform>(avx::ra_addr<PTform>(src3, offset2), shc5);
            q7 = cxloadstore<PTform>(avx::ra_addr<PTform>(src7, offset2), shc7);

            if constexpr (PSrc < 4) {
                std::tie(q1, q5, q3, q7) = avx::convert<float>::split<PSrc>(q1, q5, q3, q7);
                std::tie(q1, q5, q3, q7) = avx::convert<float>::inverse<Inverse>(q1, q5, q3, q7);
            } else {
                std::tie(q2, q6, q3, q7) = avx::convert<float>::split<PSrc>(q2, q6, q3, q7);
                std::tie(q2, q6, q3, q7) = avx::convert<float>::inverse<Inverse>(q2, q6, q3, q7);
            }

            auto [x1, x5] = avx::btfly(q1, q5);
            auto [x3, x7] = avx::btfly(q3, q7);

            auto [y5, y7] = avx::btfly<3>(x5, x7);

            auto [y1, y3] = avx::btfly(x1, x3);

            reg_t y5_tw = {avx::add(y5.real, y5.imag), avx::sub(y5.imag, y5.real)};
            reg_t y7_tw = {avx::sub(y7.real, y7.imag), avx::add(y7.real, y7.imag)};

            auto [x0, x4] = avx::btfly(q0, q4);

            y5_tw = avx::mul(y5_tw, twsq2);
            y7_tw = avx::mul(y7_tw, twsq2);

            auto [x2, x6] = avx::btfly(q2, q6);

            auto [y0, y2] = avx::btfly(x0, x2);
            auto [y4, y6] = avx::btfly<3>(x4, x6);

            auto [z0, z1] = avx::btfly(y0, y1);
            auto [z2, z3] = avx::btfly<3>(y2, y3);
            auto [z4, z5] = avx::btfly(y4, y5_tw);
            auto [z6, z7] = avx::btfly<2>(y6, y7_tw);

            auto [shx0, shx4] = avx::unpack_ps(z0, z4);
            auto [shx1, shx5] = avx::unpack_ps(z1, z5);
            auto [shx2, shx6] = avx::unpack_ps(z2, z6);
            auto [shx3, shx7] = avx::unpack_ps(z3, z7);

            auto [shy0, shy2] = avx::unpack_pd(shx0, shx2);
            auto [shy1, shy3] = avx::unpack_pd(shx1, shx3);

            auto [shz0, shz1] = avx::unpack_128(shy0, shy1);
            auto [shz2, shz3] = avx::unpack_128(shy2, shy3);

            std::tie(shz0, shz1, shz2, shz3) = avx::convert<float>::inverse<Inverse>(shz0, shz1, shz2, shz3);

            avx::cxstore<PTform>(avx::ra_addr<PTform>(src4, offset1), shz2);
            avx::cxstore<PTform>(avx::ra_addr<PTform>(src0, offset1), shz0);
            if constexpr (PSrc < 4) {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src2, offset1), shz1);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src6, offset1), shz3);
            } else {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src1, offset1), shz1);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src5, offset1), shz3);
            }

            auto [shy4, shy6] = avx::unpack_pd(shx4, shx6);
            auto [shy5, shy7] = avx::unpack_pd(shx5, shx7);

            auto [shz4, shz5] = avx::unpack_128(shy4, shy5);
            auto [shz6, shz7] = avx::unpack_128(shy6, shy7);

            std::tie(shz4, shz5, shz6, shz7) = avx::convert<float>::inverse<Inverse>(shz4, shz5, shz6, shz7);

            if constexpr (PSrc < 4) {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src1, offset1), shz4);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src5, offset1), shz6);
            } else {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src2, offset1), shz4);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src6, offset1), shz6);
            }
            avx::cxstore<PTform>(avx::ra_addr<PTform>(src3, offset1), shz5);
            avx::cxstore<PTform>(avx::ra_addr<PTform>(src7, offset1), shz7);
        };
        for (; i < size() / 64; ++i) {
            using reg_t = avx::cx_reg<float>;
            auto offset = m_sort[i] * reg_size;

            auto p1 = avx::cxload<PTform>(avx::ra_addr<PTform>(src1, offset));
            auto p5 = avx::cxload<PTform>(avx::ra_addr<PTform>(src5, offset));
            auto p3 = avx::cxload<PTform>(avx::ra_addr<PTform>(src3, offset));
            auto p7 = avx::cxload<PTform>(avx::ra_addr<PTform>(src7, offset));

            std::tie(p1, p5, p3, p7) = avx::convert<float>::split<PSrc>(p1, p5, p3, p7);
            std::tie(p1, p5, p3, p7) = avx::convert<float>::inverse<Inverse>(p1, p5, p3, p7);

            auto [a1, a5] = avx::btfly(p1, p5);
            auto [a3, a7] = avx::btfly(p3, p7);

            auto [b5, b7] = avx::btfly<3>(a5, a7);
            auto [b1, b3] = avx::btfly(a1, a3);

            auto twsq2 = avx::broadcast(sq2.real());

            reg_t b5_tw = {avx::add(b5.real, b5.imag), avx::sub(b5.imag, b5.real)};
            reg_t b7_tw = {avx::sub(b7.real, b7.imag), avx::add(b7.real, b7.imag)};

            b5_tw = avx::mul(b5_tw, twsq2);
            b7_tw = avx::mul(b7_tw, twsq2);

            auto p0 = avx::cxload<PTform>(avx::ra_addr<PTform>(src0, offset));
            auto p4 = avx::cxload<PTform>(avx::ra_addr<PTform>(src4, offset));
            auto p2 = avx::cxload<PTform>(avx::ra_addr<PTform>(src2, offset));
            auto p6 = avx::cxload<PTform>(avx::ra_addr<PTform>(src6, offset));

            std::tie(p0, p4, p2, p6) = avx::convert<float>::split<PSrc>(p0, p4, p2, p6);
            std::tie(p0, p4, p2, p6) = avx::convert<float>::inverse<Inverse>(p0, p4, p2, p6);

            auto [a0, a4] = avx::btfly(p0, p4);
            auto [a2, a6] = avx::btfly(p2, p6);

            auto [b0, b2] = avx::btfly(a0, a2);
            auto [b4, b6] = avx::btfly<3>(a4, a6);

            auto [c0, c1] = avx::btfly(b0, b1);
            auto [c2, c3] = avx::btfly<3>(b2, b3);
            auto [c4, c5] = avx::btfly(b4, b5_tw);
            auto [c6, c7] = avx::btfly<2>(b6, b7_tw);

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);

            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            std::tie(shc0, shc1, shc2, shc3) = avx::convert<float>::inverse<Inverse>(shc0, shc1, shc2, shc3);

            avx::cxstore<PTform>(avx::ra_addr<PTform>(src0, offset), shc0);
            avx::cxstore<PTform>(avx::ra_addr<PTform>(src4, offset), shc2);
            if constexpr (PSrc < 4) {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src2, offset), shc1);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src6, offset), shc3);
            } else {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src1, offset), shc1);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src5, offset), shc3);
            }

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            std::tie(shc4, shc5, shc6, shc7) = avx::convert<float>::inverse<Inverse>(shc4, shc5, shc6, shc7);

            if constexpr (PSrc < 4) {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src1, offset), shc4);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src5, offset), shc6);
            } else {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src2, offset), shc4);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src6, offset), shc6);
            }
            avx::cxstore<PTform>(avx::ra_addr<PTform>(src3, offset), shc5);
            avx::cxstore<PTform>(avx::ra_addr<PTform>(src7, offset), shc7);
        }
    }

    template<std::size_t PTform, std::size_t PSrc>
    inline void depth3_and_sort(float* dest, const float* source) {
        constexpr auto PLoad = std::max(PSrc, reg_size);

        const auto sq2   = internal::fft::wnk<T>(8, 1);
        auto       twsq2 = avx::broadcast(sq2.real());

        const auto* const src0 = source;
        const auto* const src1 = avx::ra_addr<PTform>(source, 1 * size() / 8);
        const auto* const src2 = avx::ra_addr<PTform>(source, 2 * size() / 8);
        const auto* const src3 = avx::ra_addr<PTform>(source, 3 * size() / 8);
        const auto* const src4 = avx::ra_addr<PTform>(source, 4 * size() / 8);
        const auto* const src5 = avx::ra_addr<PTform>(source, 5 * size() / 8);
        const auto* const src6 = avx::ra_addr<PTform>(source, 6 * size() / 8);
        const auto* const src7 = avx::ra_addr<PTform>(source, 7 * size() / 8);

        auto* const dst0 = dest;
        auto* const dst1 = avx::ra_addr<PTform>(dest, 1 * size() / 8);
        auto* const dst2 = avx::ra_addr<PTform>(dest, 2 * size() / 8);
        auto* const dst3 = avx::ra_addr<PTform>(dest, 3 * size() / 8);
        auto* const dst4 = avx::ra_addr<PTform>(dest, 4 * size() / 8);
        auto* const dst5 = avx::ra_addr<PTform>(dest, 5 * size() / 8);
        auto* const dst6 = avx::ra_addr<PTform>(dest, 6 * size() / 8);
        auto* const dst7 = avx::ra_addr<PTform>(dest, 7 * size() / 8);

        uint i = 0;
        for (; i < n_reversals(size() / 64); i += 2) {
            using reg_t = avx::cx_reg<float>;

            auto offset_src  = m_sort[i] * reg_size;
            auto offset_dest = m_sort[i + 1] * reg_size;

            for (uint k = 0; k < 2; ++k) {
                auto p1 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src1, offset_src));
                auto p5 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src5, offset_src));
                auto p3 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src3, offset_src));
                auto p7 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src7, offset_src));

                if constexpr (PSrc < PLoad) {
                    std::tie(p1, p5, p3, p7) = avx::convert<float>::split<PSrc>(p1, p5, p3, p7);
                }

                auto [a1, a5] = avx::btfly(p1, p5);
                auto [a3, a7] = avx::btfly(p3, p7);

                reg_t b5 = {avx::add(a5.real, a7.imag), avx::sub(a5.imag, a7.real)};
                reg_t b7 = {avx::sub(a5.real, a7.imag), avx::add(a5.imag, a7.real)};

                reg_t b5_tw = {avx::add(b5.real, b5.imag), avx::sub(b5.imag, b5.real)};
                reg_t b7_tw = {avx::sub(b7.real, b7.imag), avx::add(b7.real, b7.imag)};

                auto [b1, b3] = avx::btfly(a1, a3);

                b5_tw = avx::mul(b5_tw, twsq2);
                b7_tw = avx::mul(b7_tw, twsq2);

                auto p0 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src0, offset_src));
                auto p4 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src4, offset_src));
                auto p2 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src2, offset_src));
                auto p6 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src6, offset_src));

                if constexpr (PSrc < PLoad) {
                    std::tie(p0, p4, p2, p6) = avx::convert<float>::split<PSrc>(p0, p4, p2, p6);
                }

                auto [a0, a4] = avx::btfly(p0, p4);
                auto [a2, a6] = avx::btfly(p2, p6);

                auto [b0, b2] = avx::btfly(a0, a2);
                reg_t b4      = {avx::add(a4.real, a6.imag), avx::sub(a4.imag, a6.real)};
                reg_t b6      = {avx::sub(a4.real, a6.imag), avx::add(a4.imag, a6.real)};

                auto [c0, c1] = avx::btfly(b0, b1);
                reg_t c2      = {avx::add(b2.real, b3.imag), avx::sub(b2.imag, b3.real)};
                reg_t c3      = {avx::sub(b2.real, b3.imag), avx::add(b2.imag, b3.real)};

                auto [c4, c5] = avx::btfly(b4, b5_tw);
                auto [c7, c6] = avx::btfly(b6, b7_tw);

                auto [sha0, sha4] = avx::unpack_ps(c0, c4);
                auto [sha2, sha6] = avx::unpack_ps(c2, c6);
                auto [sha1, sha5] = avx::unpack_ps(c1, c5);
                auto [sha3, sha7] = avx::unpack_ps(c3, c7);

                auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
                auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);
                auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
                auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst0, offset_dest), shc0);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst4, offset_dest), shc2);
                if constexpr (PSrc < 4) {
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst2, offset_dest), shc1);
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst6, offset_dest), shc3);
                } else {
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst1, offset_dest), shc1);
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst5, offset_dest), shc3);
                }

                auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
                auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);
                auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
                auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

                if constexpr (PSrc < 4) {
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst1, offset_dest), shc4);
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst5, offset_dest), shc6);
                } else {
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst2, offset_dest), shc4);
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst6, offset_dest), shc6);
                }
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst3, offset_dest), shc5);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst7, offset_dest), shc7);

                offset_src  = m_sort[i + 1] * reg_size;
                offset_dest = m_sort[i] * reg_size;
            }
        };
        for (; i < size() / 64; ++i) {
            using reg_t = avx::cx_reg<float>;

            auto offset_src  = m_sort[i] * reg_size;
            auto offset_dest = m_sort[i] * reg_size;

            auto p1 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src1, offset_src));
            auto p5 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src5, offset_src));
            auto p3 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src3, offset_src));
            auto p7 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src7, offset_src));

            if constexpr (PSrc < PLoad) {
                std::tie(p1, p5, p3, p7) = avx::convert<float>::split<PSrc>(p1, p5, p3, p7);
            }

            auto [a1, a5] = avx::btfly(p1, p5);
            auto [a3, a7] = avx::btfly(p3, p7);

            reg_t b5 = {avx::add(a5.real, a7.imag), avx::sub(a5.imag, a7.real)};
            reg_t b7 = {avx::sub(a5.real, a7.imag), avx::add(a5.imag, a7.real)};

            reg_t b5_tw = {avx::add(b5.real, b5.imag), avx::sub(b5.imag, b5.real)};
            reg_t b7_tw = {avx::sub(b7.real, b7.imag), avx::add(b7.real, b7.imag)};

            auto [b1, b3] = avx::btfly(a1, a3);

            b5_tw = avx::mul(b5_tw, twsq2);
            b7_tw = avx::mul(b7_tw, twsq2);

            auto p0 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src0, offset_src));
            auto p4 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src4, offset_src));
            auto p2 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src2, offset_src));
            auto p6 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src6, offset_src));

            if constexpr (PSrc < PLoad) {
                std::tie(p0, p4, p2, p6) = avx::convert<float>::split<PSrc>(p0, p4, p2, p6);
            }

            auto [a0, a4] = avx::btfly(p0, p4);
            auto [a2, a6] = avx::btfly(p2, p6);

            auto [b0, b2] = avx::btfly(a0, a2);
            reg_t b4      = {avx::add(a4.real, a6.imag), avx::sub(a4.imag, a6.real)};
            reg_t b6      = {avx::sub(a4.real, a6.imag), avx::add(a4.imag, a6.real)};

            auto [c0, c1] = avx::btfly(b0, b1);
            reg_t c2      = {avx::add(b2.real, b3.imag), avx::sub(b2.imag, b3.real)};
            reg_t c3      = {avx::sub(b2.real, b3.imag), avx::add(b2.imag, b3.real)};

            auto [c4, c5] = avx::btfly(b4, b5_tw);
            auto [c7, c6] = avx::btfly(b6, b7_tw);

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);
            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            avx::cxstore<PTform>(avx::ra_addr<PTform>(dst0, offset_dest), shc0);
            avx::cxstore<PTform>(avx::ra_addr<PTform>(dst4, offset_dest), shc2);
            if constexpr (PSrc < 4) {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst2, offset_dest), shc1);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst6, offset_dest), shc3);
            } else {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst1, offset_dest), shc1);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst5, offset_dest), shc3);
            }

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);
            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            if constexpr (PSrc < 4) {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst1, offset_dest), shc4);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst5, offset_dest), shc6);
            } else {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst2, offset_dest), shc4);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst6, offset_dest), shc6);
            }
            avx::cxstore<PTform>(avx::ra_addr<PTform>(dst3, offset_dest), shc5);
            avx::cxstore<PTform>(avx::ra_addr<PTform>(dst7, offset_dest), shc7);
        }
    }

    template<std::size_t PDest, std::size_t PTform, bool Inverse = false, bool Scale = false>
    inline auto subtransform(float* data, std::size_t max_size) -> const float* {
        const auto* twiddle_ptr = m_twiddles.data();

        std::size_t l_size     = reg_size * 2;
        std::size_t group_size = max_size / reg_size / 4;
        std::size_t n_groups   = 1;
        std::size_t tw_offset  = 0;

        std::size_t max_size_ = max_size;
        if constexpr ((PDest < reg_size) || Scale) {
            max_size_ = max_size_ / 2;
        }
        if (log2i(max_size / (reg_size * 2)) % 2 == 0) {
            std::size_t offset = 0;

            auto scaling = std::conditional_t<Scale, typename avx::reg<float>::type, decltype([] {})>{};
            if constexpr (Scale) {
                scaling = avx::broadcast(static_cast<float>(1 / static_cast<double>(max_size)));
            }

            const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
            const auto tw1 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 2);
            const auto tw2 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 4);
            const auto tw3 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 6);
            const auto tw4 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 8);
            const auto tw5 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 10);
            const auto tw6 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 12);
            twiddle_ptr += reg_size * 14;

            group_size /= 2;
            if (max_size / (reg_size * 2) == 4) {
                for (std::size_t i = 0; i < group_size; ++i) {
                    node8_dit_along<PDest, PTform, Inverse, Scale>(
                        data, l_size, offset, tw0, tw1, tw2, tw3, tw4, tw5, tw6, scaling);
                    offset += l_size * 4;
                }
                return twiddle_ptr;
            }

            for (std::size_t i = 0; i < group_size; ++i) {
                node8_dit_along<PTform, PTform, Inverse>(
                    data, l_size, offset, tw0, tw1, tw2, tw3, tw4, tw5, tw6);
                offset += l_size * 4;
            }

            l_size *= 8;
            n_groups *= 8;
            group_size /= 4;
        }
        while (l_size < max_size_) {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                std::size_t offset = i_group * reg_size;

                std::array<avx::cx_reg<T>, 3> tw{
                    avx::cxload<reg_size>(twiddle_ptr),
                    avx::cxload<reg_size>(twiddle_ptr + reg_size * 2),
                    avx::cxload<reg_size>(twiddle_ptr + reg_size * 4),
                };
                twiddle_ptr += reg_size * 6;
                for (std::size_t i = 0; i < group_size; ++i) {
                    node4_along<PTform, PTform, true, Inverse>(data, l_size * 2, offset, tw);
                    offset += l_size * 2;
                }
            }
            l_size *= 4;
            n_groups *= 4;
            group_size /= 4;
        }
        if constexpr ((PDest < reg_size) || Scale) {
            auto scaling = std::conditional_t<Scale, typename avx::reg<float>::type, decltype([] {})>{};
            if constexpr (Scale) {
                scaling = avx::broadcast(static_cast<float>(1 / static_cast<double>(max_size)));
            }
            if (l_size == max_size / 2) {
                for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                    std::size_t offset = i_group * reg_size;

                    std::array<avx::cx_reg<T>, 3> tw{
                        avx::cxload<reg_size>(twiddle_ptr),
                        avx::cxload<reg_size>(twiddle_ptr + reg_size * 2),
                        avx::cxload<reg_size>(twiddle_ptr + reg_size * 4),
                    };
                    twiddle_ptr += reg_size * 6;
                    for (std::size_t i = 0; i < group_size; ++i) {
                        node4_along<PDest, PTform, true, Inverse>(data, l_size * 2, offset, tw, scaling);
                        offset += l_size * 2;
                    }
                }
                return twiddle_ptr;
            }
        }
        return twiddle_ptr;
    };

    template<std::size_t PDest, std::size_t PTform, bool Inverse = false, bool Scale = false>
    inline auto subtransform_recursive(T* data, std::size_t size) -> const T* {
        if (size <= sub_size()) {
            return subtransform<PDest, PTform, Inverse, Scale>(data, size);
        } else {
            subtransform_recursive<PTform, PTform, Inverse>(data, size / 4);
            subtransform_recursive<PTform, PTform, Inverse>(avx::ra_addr<PTform>(data, size / 4), size / 4);
            subtransform_recursive<PTform, PTform, Inverse>(avx::ra_addr<PTform>(data, size / 2), size / 4);
            auto twiddle_ptr = subtransform_recursive<PTform, PTform, Inverse>(
                avx::ra_addr<PTform>(data, size * 3 / 4), size / 4);

            std::size_t n_groups = size / reg_size / 4;
            auto scaling = std::conditional_t<Scale, typename avx::reg<float>::type, decltype([] {})>{};
            if constexpr (Scale) {
                scaling = avx::broadcast(static_cast<T>(1. / static_cast<double>(size)));
            }
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                std::array<avx::cx_reg<T>, 3> tw{
                    avx::cxload<reg_size>(twiddle_ptr),
                    avx::cxload<reg_size>(twiddle_ptr + reg_size * 2),
                    avx::cxload<reg_size>(twiddle_ptr + reg_size * 4),
                };
                twiddle_ptr += reg_size * 6;
                node4_along<PDest, PTform, true, Inverse>(data, size, i_group * reg_size, tw, scaling);
            }
            return twiddle_ptr;
        }
    };

    template<std::size_t PDest, std::size_t PSrc, bool First = false, bool BitReverse = true>
    inline auto unsorted_subtransform_recursive(T* data, std::size_t size, const T* twiddle_ptr) -> const T* {
        if (size <= sub_size()) {
            return unsorted_subtransform<PDest, PSrc, First, BitReverse>(data, size, twiddle_ptr);
        }
        constexpr auto PTform = std::max(PSrc, reg_size);
        if constexpr (First) {
            twiddle_ptr += 6;
            for (std::size_t i_group = 0; i_group < size / 4 / reg_size; ++i_group) {
                node4_along<PTform, PSrc, false>(data, size, i_group * reg_size);
            }
        } else {
            std::array<avx::cx_reg<T>, 3> tw{
                {{avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)},
                 {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)},
                 {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)}}
            };
            for (std::size_t i_group = 0; i_group < size / 4 / reg_size; ++i_group) {
                node4_along<PTform, PSrc, false>(data, size, i_group * reg_size, tw);
            }
        }
        twiddle_ptr =
            unsorted_subtransform_recursive<PDest, PTform, First, BitReverse>(data, size / 4, twiddle_ptr);
        twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReverse>(
            avx::ra_addr<PTform>(data, size / 4), size / 4, twiddle_ptr);
        twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReverse>(
            avx::ra_addr<PTform>(data, size / 2), size / 4, twiddle_ptr);
        twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReverse>(
            avx::ra_addr<PTform>(data, size / 4 * 3), size / 4, twiddle_ptr);
        return twiddle_ptr;
    };

    template<std::size_t PDest,
             std::size_t PSrc,
             bool        First      = false,
             bool        BitReverse = true,
             typename... Optional>
    inline auto unsorted_subtransform(float*       dest,
                                      std::size_t  size,
                                      const float* twiddle_ptr,
                                      Optional... optional) -> const float* {
        constexpr auto PTform = std::max(PSrc, reg_size);

        using source_type  = const T*;
        constexpr bool Src = internal::has_type<source_type, Optional...>;

        using reg_t = avx::cx_reg<float>;

        std::size_t l_size   = size;
        std::size_t n_groups = 1;

        if constexpr (PSrc < PTform || Src) {
            if (l_size > reg_size * 8) {
                if constexpr (First) {
                    twiddle_ptr += 6;
                    for (std::size_t i = 0; i < l_size / reg_size / 4; ++i) {
                        node4_along<PTform, PSrc, false>(dest, l_size, i * reg_size, optional...);
                    }
                } else {
                    std::array<avx::reg_t<T>, 3> tw{
                        {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)},
                        {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)},
                        {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)},
                    };
                    for (std::size_t i = 0; i < l_size / reg_size / 4; ++i) {
                        node4_along<PTform, PSrc, false>(dest, l_size, i * reg_size, tw, optional...);
                    }
                }
                l_size /= 4;
                n_groups *= 4;
            } else if (l_size == reg_size * 8) {
                reg_t tw0 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};
                for (std::size_t i = 0; i < l_size / reg_size / 2; ++i) {
                    node2_along<PTform, PSrc>(dest, l_size, i * reg_size, tw0, optional...);
                }
                l_size /= 2;
                n_groups *= 2;
            }
        }

        while (l_size > reg_size * 8) {
            uint i_group = 0;
            if constexpr (First) {
                twiddle_ptr += 6;
                for (std::size_t i = 0; i < l_size / reg_size / 4; ++i) {
                    node4_along<PTform, PTform, false>(dest, l_size, i * reg_size);
                }
                ++i_group;
            }
            for (; i_group < n_groups; ++i_group) {
                std::array<avx::cx_reg<T>, 3> tw{
                    {{avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)},
                     {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)},
                     {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)}}
                };
                auto* group_ptr = avx::ra_addr<PTform>(dest, i_group * l_size);
                for (std::size_t i = 0; i < l_size / reg_size / 4; ++i) {
                    node4_along<PTform, PTform, false>(group_ptr, l_size, i * reg_size, tw);
                }
            }
            l_size /= 4;
            n_groups *= 4;
        }


        for (std::size_t i_group = 0; i_group < size / reg_size / 4; ++i_group) {
            reg_t tw0 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};

            reg_t tw1 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};
            reg_t tw2 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};

            auto* ptr0 = avx::ra_addr<PTform>(dest, reg_size * (i_group * 4));
            auto* ptr1 = avx::ra_addr<PTform>(dest, reg_size * (i_group * 4 + 1));
            auto* ptr2 = avx::ra_addr<PTform>(dest, reg_size * (i_group * 4 + 2));
            auto* ptr3 = avx::ra_addr<PTform>(dest, reg_size * (i_group * 4 + 3));

            auto p2 = avx::cxload<PTform>(ptr2);
            auto p3 = avx::cxload<PTform>(ptr3);
            auto p0 = avx::cxload<PTform>(ptr0);
            auto p1 = avx::cxload<PTform>(ptr1);

            auto [p2tw, p3tw] = avx::mul({p2, tw0}, {p3, tw0});

            auto [a1, a3] = avx::btfly(p1, p3tw);
            auto [a0, a2] = avx::btfly(p0, p2tw);

            auto [a1tw, a3tw] = avx::mul({a1, tw1}, {a3, tw2});

            auto tw3 = avx::cxload<reg_size>(twiddle_ptr);
            auto tw4 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 2);
            twiddle_ptr += reg_size * 4;

            auto [b0, b1] = avx::btfly(a0, a1tw);
            auto [b2, b3] = avx::btfly(a2, a3tw);

            auto [shb0, shb1] = avx::unpack_128(b0, b1);
            auto [shb2, shb3] = avx::unpack_128(b2, b3);

            auto [shb1tw, shb3tw] = avx::mul({shb1, tw3}, {shb3, tw4});

            auto tw5 = avx::cxload<reg_size>(twiddle_ptr);
            auto tw6 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 2);
            twiddle_ptr += reg_size * 4;

            auto [c0, c1] = avx::btfly(shb0, shb1tw);
            auto [c2, c3] = avx::btfly(shb2, shb3tw);

            auto [shc0, shc1] = avx::unpack_pd(c0, c1);
            auto [shc2, shc3] = avx::unpack_pd(c2, c3);

            auto [shc1tw, shc3tw] = avx::mul({shc1, tw5}, {shc3, tw6});

            auto tw7 = avx::cxload<reg_size>(twiddle_ptr);
            auto tw8 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 2);
            twiddle_ptr += reg_size * 4;

            auto [d0, d1] = avx::btfly(shc0, shc1tw);
            auto [d2, d3] = avx::btfly(shc2, shc3tw);

            auto shuf = [](reg_t lhs, reg_t rhs) {
                auto lhs_re = _mm256_shuffle_ps(lhs.real, rhs.real, 0b10001000);
                auto rhs_re = _mm256_shuffle_ps(lhs.real, rhs.real, 0b11011101);
                auto lhs_im = _mm256_shuffle_ps(lhs.imag, rhs.imag, 0b10001000);
                auto rhs_im = _mm256_shuffle_ps(lhs.imag, rhs.imag, 0b11011101);
                return std::make_tuple(reg_t{lhs_re, lhs_im}, reg_t{rhs_re, rhs_im});
            };
            auto [shd0, shd1] = shuf(d0, d1);
            auto [shd2, shd3] = shuf(d2, d3);

            auto [shd1tw, shd3tw] = avx::mul({shd1, tw7}, {shd3, tw8});

            auto [e0, e1] = avx::btfly(shd0, shd1tw);
            auto [e2, e3] = avx::btfly(shd2, shd3tw);

            reg_t she0, she1, she2, she3;
            if constexpr (BitReverse) {
                if constexpr (PDest < 4) {
                    std::tie(she0, she1) = avx::unpack_ps(e0, e1);
                    std::tie(she2, she3) = avx::unpack_ps(e2, e3);
                    std::tie(she0, she1) = avx::unpack_128(she0, she1);
                    std::tie(she2, she3) = avx::unpack_128(she2, she3);
                } else {
                    std::tie(she0, she1) = avx::unpack_ps(e0, e1);
                    std::tie(she2, she3) = avx::unpack_ps(e2, e3);
                    std::tie(she0, she1) = avx::unpack_pd(she0, she1);
                    std::tie(she2, she3) = avx::unpack_pd(she2, she3);
                    std::tie(she0, she1) = avx::unpack_128(she0, she1);
                    std::tie(she2, she3) = avx::unpack_128(she2, she3);
                }
                std::tie(she0, she1, she2, she3) =
                    avx::convert<float>::combine<PDest>(she0, she1, she2, she3);
            } else {
                std::tie(she0, she1, she2, she3) = std::tie(e0, e1, e2, e3);
            }
            cxstore<PTform>(ptr0, she0);
            cxstore<PTform>(ptr1, she1);
            cxstore<PTform>(ptr2, she2);
            cxstore<PTform>(ptr3, she3);
        }
        return twiddle_ptr;
    }

    template<std::size_t PDest, std::size_t PSrc, bool First = false, bool BitReverse = true>
    inline auto unsorted_subtransform_recursive_fill0(T*          dest,    //
                                                      const T*    src,
                                                      std::size_t size,
                                                      std::size_t data_size,
                                                      const T*    twiddle_ptr) -> const T* {
        if (size <= sub_size()) {
            return unsorted_subtransform<PDest, PSrc, First, BitReverse>(
                dest, size, twiddle_ptr, src, data_size);
        }
        constexpr auto PTform = std::max(PSrc, reg_size);
        if constexpr (First) {
            twiddle_ptr += 6;
            for (std::size_t i_group = 0; i_group < size / 4 / reg_size; ++i_group) {
                node4_along<PTform, PSrc, false>(dest, size, i_group * reg_size, src, data_size);
            }
        } else {
            std::array<avx::reg_t<T>, 3> tw{
                {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)},
                {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)},
                {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)},
            };

            for (std::size_t i_group = 0; i_group < size / 4 / reg_size; ++i_group) {
                node4_along<PTform, PSrc, false>(dest, size, i_group * reg_size, src, data_size, tw);
            }
        }
        twiddle_ptr =
            unsorted_subtransform_recursive<PDest, PTform, First, BitReverse>(dest, size / 4, twiddle_ptr);
        twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReverse>(
            avx::ra_addr<PTform>(dest, size / 4), size / 4, twiddle_ptr);
        twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReverse>(
            avx::ra_addr<PTform>(dest, size / 2), size / 4, twiddle_ptr);
        twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform, false, BitReverse>(
            avx::ra_addr<PTform>(dest, size / 4 * 3), size / 4, twiddle_ptr);
        return twiddle_ptr;
    };

    template<std::size_t PDest,
             std::size_t PSrc,
             bool        First      = false,
             bool        Scale      = true,
             bool        BitReverse = true>
    inline auto unsorted_subtransform_inverse(float* data, std::size_t size, const float* twiddle_ptr)
        -> const float* {
        constexpr auto PTform = std::max(PSrc, reg_size);
        using reg_t           = avx::cx_reg<float>;

        const auto scale = static_cast<float>(1. / static_cast<double>(this->size()));

        for (int i_group = size / reg_size / 4 - 1; i_group >= 0; --i_group) {
            auto* ptr0 = avx::ra_addr<PTform>(data, reg_size * (i_group * 4));
            auto* ptr1 = avx::ra_addr<PTform>(data, reg_size * (i_group * 4 + 1));
            auto* ptr2 = avx::ra_addr<PTform>(data, reg_size * (i_group * 4 + 2));
            auto* ptr3 = avx::ra_addr<PTform>(data, reg_size * (i_group * 4 + 3));

            auto she0 = avx::cxload<PTform>(ptr0);
            auto she1 = avx::cxload<PTform>(ptr1);
            auto she2 = avx::cxload<PTform>(ptr2);
            auto she3 = avx::cxload<PTform>(ptr3);

            reg_t e0, e1, e2, e3;
            if constexpr (BitReverse) {
                std::tie(she0, she1, she2, she3) =
                    avx::convert<float>::repack<PSrc, PTform>(she0, she1, she2, she3);
            }
            std::tie(she0, she1, she2, she3) = avx::convert<float>::inverse<true>(she0, she1, she2, she3);
            if constexpr (BitReverse) {
                std::tie(e0, e1) = avx::unpack_128(she0, she1);
                std::tie(e2, e3) = avx::unpack_128(she2, she3);
            } else {
                std::tie(e0, e1, e2, e3) = std::tie(she0, she1, she2, she3);
            }

            twiddle_ptr -= reg_size * 4;
            auto tw7 = avx::cxload<reg_size>(twiddle_ptr);
            auto tw8 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 2);

            if constexpr (BitReverse) {
                std::tie(e0, e1) = avx::unpack_ps(e0, e1);
                std::tie(e2, e3) = avx::unpack_ps(e2, e3);
                std::tie(e0, e1) = avx::unpack_pd(e0, e1);
                std::tie(e2, e3) = avx::unpack_pd(e2, e3);
            }
            auto [shd0, shd1tw] = avx::ibtfly(e0, e1);
            auto [shd2, shd3tw] = avx::ibtfly(e2, e3);

            auto [shd1, shd3] = avx::mul({shd1tw, tw7}, {shd3tw, tw8});

            twiddle_ptr -= reg_size * 4;
            auto tw5 = avx::cxload<reg_size>(twiddle_ptr);
            auto tw6 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 2);

            auto [d0, d1] = avx::unpack_ps(shd0, shd1);
            auto [d2, d3] = avx::unpack_ps(shd2, shd3);

            auto [shc0, shc1tw] = avx::btfly(d0, d1);
            auto [shc2, shc3tw] = avx::btfly(d2, d3);

            auto [shc1, shc3] = avx::mul({shc1tw, tw5}, {shc3tw, tw6});

            twiddle_ptr -= reg_size * 4;
            auto tw3 = avx::cxload<reg_size>(twiddle_ptr);
            auto tw4 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 2);

            auto [c0, c1] = avx::unpack_pd(shc0, shc1);
            auto [c2, c3] = avx::unpack_pd(shc2, shc3);

            auto [shb0, shb1tw] = avx::btfly(c0, c1);
            auto [shb2, shb3tw] = avx::btfly(c2, c3);

            auto [shb1, shb3] = avx::mul({shb1tw, tw3}, {shb3tw, tw4});

            twiddle_ptr -= 6;
            reg_t tw1 = {avx::broadcast(twiddle_ptr + 2), avx::broadcast(twiddle_ptr + 3)};
            reg_t tw2 = {avx::broadcast(twiddle_ptr + 4), avx::broadcast(twiddle_ptr + 5)};
            reg_t tw0 = {avx::broadcast(twiddle_ptr + 0), avx::broadcast(twiddle_ptr + 1)};

            auto [b0, b1] = avx::unpack_128(shb0, shb1);
            auto [b2, b3] = avx::unpack_128(shb2, shb3);

            auto [a0, a1tw] = avx::btfly(b0, b1);
            auto [a2, a3tw] = avx::btfly(b2, b3);

            auto [a1, a3] = avx::mul({a1tw, tw1}, {a3tw, tw2});

            auto [p0, p2tw] = avx::btfly(a0, a2);
            auto [p1, p3tw] = avx::btfly(a1, a3);

            auto [p2, p3] = avx::mul({p2tw, tw0}, {p3tw, tw0});

            if constexpr (Scale) {
                auto scaling = avx::broadcast(scale);

                p0 = avx::mul(p0, scaling);
                p1 = avx::mul(p1, scaling);
                p2 = avx::mul(p2, scaling);
                p3 = avx::mul(p3, scaling);
            }

            std::tie(p0, p2, p1, p3) = avx::convert<float>::inverse<true>(p0, p2, p1, p3);

            cxstore<PTform>(ptr0, p0);
            cxstore<PTform>(ptr2, p2);
            cxstore<PTform>(ptr1, p1);
            cxstore<PTform>(ptr3, p3);
        }

        std::size_t size_ = size;
        if constexpr (PDest < PTform) {
            size_ /= 2;
        }
        std::size_t l_size   = reg_size * 8;
        std::size_t n_groups = size / l_size / 2;

        while (l_size < size_) {
            constexpr auto max_g = First ? 1 : 0;
            for (int i_group = n_groups - 1; i_group >= max_g; --i_group) {
                twiddle_ptr -= 6;
                std::array<avx::cx_reg<T>, 3> tw{
                    {{avx::broadcast(twiddle_ptr + 0), avx::broadcast(twiddle_ptr + 1)},
                     {avx::broadcast(twiddle_ptr + 2), avx::broadcast(twiddle_ptr + 3)},
                     {avx::broadcast(twiddle_ptr + 4), avx::broadcast(twiddle_ptr + 5)}}
                };
                auto* group_ptr = avx::ra_addr<PTform>(data, i_group * l_size * 2);

                for (std::size_t i = 0; i < l_size / reg_size / 2; ++i) {
                    node4_along<PTform, PTform, false, false, true>(group_ptr, l_size * 2, i * reg_size, tw);
                }
            }
            if constexpr (First) {
                twiddle_ptr -= 6;
                for (std::size_t i = 0; i < l_size / reg_size / 2; ++i) {
                    node4_along<PTform, PTform, false, false, true>(data, l_size * 2, i * reg_size);
                }
            }
            l_size *= 4;
            n_groups /= 4;
        }

        if constexpr (PDest < PTform) {
            if (l_size < size) {
                if constexpr (First) {
                    twiddle_ptr -= 6;
                    for (std::size_t i = 0; i < l_size / reg_size / 2; ++i) {
                        node4_along<PDest, PTform, false, false, true>(data, l_size * 2, i * reg_size);
                    }
                } else {
                    twiddle_ptr -= 6;
                    std::array<avx::cx_reg<T>, 3> tw{
                        {{avx::broadcast(twiddle_ptr + 0), avx::broadcast(twiddle_ptr + 1)},
                         {avx::broadcast(twiddle_ptr + 2), avx::broadcast(twiddle_ptr + 3)},
                         {avx::broadcast(twiddle_ptr + 4), avx::broadcast(twiddle_ptr + 5)}}
                    };
                    for (std::size_t i = 0; i < l_size / reg_size / 2; ++i) {
                        node4_along<PDest, PTform, false, false, true>(data, l_size * 2, i * reg_size, tw);
                    }
                }
                l_size *= 4;
                n_groups /= 4;
            } else if (l_size == size) {
                twiddle_ptr -= 2;
                reg_t tw0 = {avx::broadcast(twiddle_ptr + 0), avx::broadcast(twiddle_ptr + 1)};
                for (std::size_t i = 0; i < l_size / reg_size / 2; ++i) {
                    node2_along<PDest, PTform, true>(data, l_size, i * reg_size, tw0);
                }

                l_size *= 2;
                n_groups /= 2;
            }
        }

        return twiddle_ptr;
    }

    template<std::size_t PDest, std::size_t PSrc, bool First = false, bool BitReverse = true>
    inline auto unsorted_subtransform_recursive_inverse(T* data, std::size_t size, const T* twiddle_ptr)
        -> const T* {
        if (size <= sub_size()) {
            return unsorted_subtransform_inverse<PDest, PSrc, First, true, BitReverse>(
                data, size, twiddle_ptr);
        }
        constexpr auto PTform = std::max(PSrc, reg_size);

        twiddle_ptr = unsorted_subtransform_recursive_inverse<PDest, PTform, false, BitReverse>(
            avx::ra_addr<PTform>(data, size / 4 * 3), size / 4, twiddle_ptr);
        twiddle_ptr = unsorted_subtransform_recursive_inverse<PDest, PTform, false, BitReverse>(
            avx::ra_addr<PTform>(data, size / 2), size / 4, twiddle_ptr);
        twiddle_ptr = unsorted_subtransform_recursive_inverse<PDest, PTform, false, BitReverse>(
            avx::ra_addr<PTform>(data, size / 4), size / 4, twiddle_ptr);
        twiddle_ptr = unsorted_subtransform_recursive_inverse<PDest, PTform, First, BitReverse>(
            data, size / 4, twiddle_ptr);
        if constexpr (First) {
            twiddle_ptr -= 6;
            for (std::size_t i_group = 0; i_group < size / 4 / reg_size; ++i_group) {
                node4_along<PTform, PSrc, false, false, true>(data, size, i_group * reg_size);
            }
        } else {
            twiddle_ptr -= 6;
            std::array<avx::cx_reg<T>, 3> tw{
                {{avx::broadcast(twiddle_ptr + 0), avx::broadcast(twiddle_ptr + 1)},
                 {avx::broadcast(twiddle_ptr + 2), avx::broadcast(twiddle_ptr + 3)},
                 {avx::broadcast(twiddle_ptr + 4), avx::broadcast(twiddle_ptr + 5)}}
            };
            for (std::size_t i_group = 0; i_group < size / 4 / reg_size; ++i_group) {
                node4_along<PTform, PSrc, false, false, true>(data, size, i_group * reg_size, tw);
            }
        }
        return twiddle_ptr;
    };

    template<std::size_t PDest,
             std::size_t PSrc    = PDest,
             bool        Inverse = false,
             bool        IScale  = false,
             typename ScaleT     = uint>
    inline void node8_dit_along(T*             data,
                                std::size_t    l_size,
                                std::size_t    offset,
                                avx::cx_reg<T> tw0,
                                avx::cx_reg<T> tw1,
                                avx::cx_reg<T> tw2,
                                avx::cx_reg<T> tw3,
                                avx::cx_reg<T> tw4,
                                avx::cx_reg<T> tw5,
                                avx::cx_reg<T> tw6,
                                ScaleT         scaling = ScaleT{}) {
        constexpr auto    PLoad = std::max(PSrc, avx::reg<T>::size);
        std::array<T*, 8> ptr04261537{
            avx::ra_addr<PLoad>(data, offset),
            avx::ra_addr<PLoad>(data, offset + l_size * 4 / 2),
            avx::ra_addr<PLoad>(data, offset + l_size * 2 / 2),
            avx::ra_addr<PLoad>(data, offset + l_size * 6 / 2),
            avx::ra_addr<PLoad>(data, offset + l_size * 1 / 2),
            avx::ra_addr<PLoad>(data, offset + l_size * 5 / 2),
            avx::ra_addr<PLoad>(data, offset + l_size * 3 / 2),
            avx::ra_addr<PLoad>(data, offset + l_size * 7 / 2),
        };
        auto tw = std::array<avx::cx_reg<T>, 7>{tw0, tw1, tw2, tw3, tw5, tw4, tw6};
        if constexpr (IScale) {
            internal::fft::node8<T, PDest, PSrc, Inverse, false>(ptr04261537, tw, scaling);
        } else {
            internal::fft::node8<T, PDest, PSrc, Inverse, false>(ptr04261537, tw);
        }
    }

    template<std::size_t PDest, std::size_t PSrc = PDest, bool Reverse = false>
    inline void node8_dif_along(T* data, std::size_t l_size, std::size_t offset) {
        constexpr auto    PLoad = std::max(PSrc, reg_size);
        std::array<T*, 8> ptr{
            avx::ra_addr<PLoad>(data, offset),
            avx::ra_addr<PLoad>(data, offset + l_size / 8 * 1),
            avx::ra_addr<PLoad>(data, offset + l_size / 8 * 2),
            avx::ra_addr<PLoad>(data, offset + l_size / 8 * 3),
            avx::ra_addr<PLoad>(data, offset + l_size / 8 * 4),
            avx::ra_addr<PLoad>(data, offset + l_size / 8 * 5),
            avx::ra_addr<PLoad>(data, offset + l_size / 8 * 6),
            avx::ra_addr<PLoad>(data, offset + l_size / 8 * 7),
        };
        internal::fft::node8<T, PDest, PSrc, false, Reverse>(ptr);
    };

    template<std::size_t PDest,
             std::size_t PSrc,
             bool        ConjTw  = false,
             bool        Reverse = false,
             typename... Optional>
    inline void node2_along(T* dest, std::size_t l_size, std::size_t offset, Optional... optional) {
        constexpr auto PLoad  = std::max(PSrc, avx::reg<T>::size);
        constexpr auto PStore = std::max(PDest, avx::reg<T>::size);
        using source_type     = const T*;
        constexpr bool Src    = internal::has_type<source_type, Optional...>;
        constexpr bool Upsize = Src && internal::has_type<std::size_t, Optional...>;

        std::array<T*, 2> dst{
            avx::ra_addr<PStore>(dest, offset),
            avx::ra_addr<PStore>(dest, offset + l_size / 2),
        };
        if constexpr (Src) {
            auto source = std::get<source_type&>(std::tie(optional...));
            if constexpr (Upsize) {
                auto data_size = std::get<std::size_t&>(std::tie(optional...));

                if (offset + l_size / 2 + avx::reg<T>::size <= data_size) {
                    std::array<const T*, 2> src{
                        avx::ra_addr<PLoad>(source, offset),
                        avx::ra_addr<PLoad>(source, offset + l_size / 2),
                    };
                    internal::fft::node2<T, PDest, PSrc, ConjTw, Reverse>(dst, src, optional...);
                } else {
                    std::array<T, avx::reg<T>::size * 2> zeros{};

                    std::array<const T*, 2> src{
                        zeros.data(),
                        zeros.data(),
                    };
                    for (uint i = 0; i < 2; ++i) {
                        auto           l_offset = offset + l_size / 2 * i;
                        std::ptrdiff_t diff     = data_size - l_offset - 1;
                        if (diff >= static_cast<int>(avx::reg<T>::size)) {
                            src[i] = avx::ra_addr<PLoad>(source, l_offset);
                        } else if (diff >= 0) {
                            std::array<T, avx::reg<T>::size * 2> align{};
                            for (; diff >= 0; --diff) {
                                align[diff] = *(source + pidx<PSrc>(l_offset + diff));
                                align[diff + avx::reg<T>::size] =
                                    *(source + pidx<PSrc>(l_offset + diff) + PSrc);
                            }
                            src[i] = align.data();
                            internal::fft::node2<T, PDest, PSrc, ConjTw, Reverse>(dst, src, optional...);
                            return;
                        } else {
                            break;
                        }
                    }
                    internal::fft::node2<T, PDest, PSrc, ConjTw, Reverse>(dst, src, optional...);
                };
            } else {
                std::array<const T*, 2> src{
                    avx::ra_addr<PLoad>(source, offset),
                    avx::ra_addr<PLoad>(source, offset + l_size / 2),
                };
                internal::fft::node2<T, PDest, PSrc, ConjTw, Reverse>(dst, src, optional...);
            }
        } else {
            internal::fft::node2<T, PDest, PSrc, ConjTw, Reverse>(dst, optional...);
        }
    };

    /**
      * @brief
      *
      * @tparam PDest
      * @tparam PSrc
      * @tparam DecInTime
      * @tparam ConjTw
      * @tparam Reverse
      * @tparam Optional
      * @param dest
      * @param l_size
      * @param offset
      * @param optional may be any combination of const T* source, std::array<avx::cx_reg<T>,3> twiddles, avx::reg_t<T> scaling
      */
    template<std::size_t PDest,
             std::size_t PSrc,
             bool        DecInTime,
             bool        ConjTw  = false,
             bool        Reverse = false,
             typename... Optional>
    inline void node4_along(T* dest, std::size_t l_size, std::size_t offset, Optional... optional) {
        constexpr auto PLoad  = std::max(PSrc, avx::reg<T>::size);
        constexpr auto PStore = std::max(PDest, avx::reg<T>::size);

        using source_type     = const T*;
        constexpr bool Src    = internal::has_type<source_type, Optional...>;
        constexpr bool Upsize = Src && internal::has_type<std::size_t, Optional...>;

        std::array<T*, 4> dst{
            avx::ra_addr<PStore>(dest, offset),
            avx::ra_addr<PStore>(dest, offset + l_size / 4),
            avx::ra_addr<PStore>(dest, offset + l_size / 2),
            avx::ra_addr<PStore>(dest, offset + l_size / 4 * 3),
        };
        if constexpr (Src) {
            auto source = std::get<source_type&>(std::tie(optional...));
            if constexpr (Upsize) {
                auto data_size = std::get<std::size_t&>(std::tie(optional...));

                if (offset + l_size / 4 * 3 + avx::reg<T>::size <= data_size) {
                    std::array<const T*, 4> src{
                        avx::ra_addr<PLoad>(source, offset),
                        avx::ra_addr<PLoad>(source, offset + l_size / 4),
                        avx::ra_addr<PLoad>(source, offset + l_size / 2),
                        avx::ra_addr<PLoad>(source, offset + l_size / 4 * 3),
                    };
                    internal::fft::node4<T, PDest, PSrc, ConjTw, Reverse, DecInTime>(dst, src, optional...);
                } else {
                    std::array<T, avx::reg<T>::size * 2> zeros{};

                    std::array<const T*, 4> src{
                        zeros.data(),
                        zeros.data(),
                        zeros.data(),
                        zeros.data(),
                    };
                    for (uint i = 0; i < 4; ++i) {
                        auto           l_offset = offset + l_size / 4 * i;
                        std::ptrdiff_t diff     = data_size - l_offset - 1;
                        if (diff >= static_cast<int>(avx::reg<T>::size)) {
                            src[i] = avx::ra_addr<PLoad>(source, l_offset);
                        } else if (diff >= 0) {
                            std::array<T, avx::reg<T>::size * 2> align{};
                            for (; diff >= 0; --diff) {
                                align[diff] = *(source + pidx<PSrc>(l_offset + diff));
                                align[diff + avx::reg<T>::size] =
                                    *(source + pidx<PSrc>(l_offset + diff) + PSrc);
                            }
                            src[i] = align.data();
                            internal::fft::node4<T, PDest, PSrc, ConjTw, Reverse, DecInTime>(
                                dst, src, optional...);
                            return;
                        } else {
                            break;
                        }
                    }
                    internal::fft::node4<T, PDest, PSrc, ConjTw, Reverse, DecInTime>(dst, src, optional...);
                };
            } else {
                std::array<const T*, 4> src{
                    avx::ra_addr<PLoad>(source, offset),
                    avx::ra_addr<PLoad>(source, offset + l_size / 4),
                    avx::ra_addr<PLoad>(source, offset + l_size / 2),
                    avx::ra_addr<PLoad>(source, offset + l_size / 4 * 3),
                };
                internal::fft::node4<T, PDest, PSrc, ConjTw, Reverse, DecInTime>(dst, src, optional...);
            }
        } else {
            internal::fft::node4<T, PDest, PSrc, ConjTw, Reverse, DecInTime>(dst, optional...);
        }
    }

    template<std::size_t PDest, std::size_t PSrc = PDest, bool Reverse = false>
    inline void node8_dif_along_fill0(    //
        T*          dest,
        const T*    source,
        std::size_t l_size,
        std::size_t data_size,
        std::size_t offset) {
        constexpr auto    PLoad  = std::max(PSrc, reg_size);
        constexpr auto    PStore = std::max(PDest, reg_size);
        std::array<T*, 8> dst{
            avx::ra_addr<PStore>(dest, offset),
            avx::ra_addr<PStore>(dest, offset + l_size / 8 * 1),
            avx::ra_addr<PStore>(dest, offset + l_size / 8 * 2),
            avx::ra_addr<PStore>(dest, offset + l_size / 8 * 3),
            avx::ra_addr<PStore>(dest, offset + l_size / 8 * 4),
            avx::ra_addr<PStore>(dest, offset + l_size / 8 * 5),
            avx::ra_addr<PStore>(dest, offset + l_size / 8 * 6),
            avx::ra_addr<PStore>(dest, offset + l_size / 8 * 7),
        };
        if (offset + l_size / 8 * 7 + avx::reg<T>::size <= data_size) {
            std::array<const T*, 8> src{
                avx::ra_addr<PLoad>(source, offset),
                avx::ra_addr<PLoad>(source, offset + l_size / 8 * 1),
                avx::ra_addr<PLoad>(source, offset + l_size / 8 * 2),
                avx::ra_addr<PLoad>(source, offset + l_size / 8 * 3),
                avx::ra_addr<PLoad>(source, offset + l_size / 8 * 4),
                avx::ra_addr<PLoad>(source, offset + l_size / 8 * 5),
                avx::ra_addr<PLoad>(source, offset + l_size / 8 * 6),
                avx::ra_addr<PLoad>(source, offset + l_size / 8 * 7),
            };
            internal::fft::node8<T, PDest, PSrc, false, Reverse>(dst, src);
        } else {
            std::array<T, avx::reg<T>::size * 2> zeros{};

            std::array<const T*, 8> src{
                zeros.data(),
                zeros.data(),
                zeros.data(),
                zeros.data(),
                zeros.data(),
                zeros.data(),
                zeros.data(),
                zeros.data(),
            };
            for (uint i = 0; i < 8; ++i) {
                auto l_offset = offset + l_size / 8 * i;

                std::ptrdiff_t diff = data_size - l_offset - 1;
                if (diff >= static_cast<int>(avx::reg<T>::size)) {
                    src[i] = avx::ra_addr<PLoad>(source, l_offset);
                } else if (diff >= 0) {
                    std::array<T, avx::reg<T>::size * 2> align{};
                    for (; diff >= 0; --diff) {
                        align[diff]                     = *(source + pidx<PSrc>(l_offset + diff));
                        align[diff + avx::reg<T>::size] = *(source + pidx<PSrc>(l_offset + diff) + PSrc);
                    }
                    src[i] = align.data();
                    internal::fft::node8<T, PDest, PSrc, false, Reverse>(dst, src);
                    return;
                } else {
                    break;
                }
            }
            internal::fft::node8<T, PDest, PSrc, false, Reverse>(dst, src);
        }
    };

private:
    static constexpr auto check_size(std::size_t size) -> std::size_t {
        if (size > 1 && (size & (size - 1)) == 0) {
            return size;
        }
        throw(std::invalid_argument("fft_size (which is  " + std::to_string(size) +
                                    ") is not an integer power of two"));
    }

    static constexpr auto check_sub_size(std::size_t sub_size) -> std::size_t {
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

    static constexpr auto log2i(std::size_t num) -> std::size_t {
        std::size_t order = 0;
        while ((num >>= 1U) != 0) {
            order++;
        }
        return order;
    }

    static constexpr auto reverse_bit_order(uint64_t num, uint64_t depth) -> uint64_t {
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
     *
     * @param max
     * @return std::size_t
     */
    static constexpr auto n_reversals(std::size_t max) -> std::size_t {
        return max - (1U << ((log2i(max) + 1) / 2));
    }

    static auto get_sort(std::size_t fft_size, sort_allocator_type allocator)
        -> std::vector<std::size_t, sort_allocator_type> {
        const auto packed_sort_size = fft_size / reg_size / reg_size;
        const auto order            = log2i(packed_sort_size);
        auto       sort             = std::vector<std::size_t, sort_allocator_type>(allocator);
        sort.reserve(packed_sort_size);

        for (uint i = 0; i < packed_sort_size; ++i) {
            if (i >= reverse_bit_order(i, order)) {
                continue;
            }
            sort.push_back(i);
            sort.push_back(reverse_bit_order(i, order));
        }
        for (uint i = 0; i < packed_sort_size; ++i) {
            if (i == reverse_bit_order(i, order)) {
                sort.push_back(i);
            }
        }
        return sort;
    }

    static auto get_twiddles(std::size_t fft_size, std::size_t sub_size, allocator_type allocator)
        -> pcx::vector<real_type, allocator_type, reg_size> {
        auto       wnk   = internal::fft::wnk<T>;
        const auto depth = log2i(fft_size);

        const std::size_t n_twiddles = 8 * ((1U << (depth - 3)) - 1U);

        auto twiddles = pcx::vector<real_type, allocator_type, reg_size>(n_twiddles, allocator);

        auto tw_it = twiddles.begin();

        std::size_t l_size   = reg_size * 2;
        std::size_t n_groups = 1;

        std::size_t sub_size_ = std::min(fft_size, sub_size);
        // if (fft_size > sub_size_ ) {
        //     sub_size_ = sub_size_ / 2;
        // }

        if (log2i(sub_size_ / (reg_size * 2)) % 2 == 0) {
            for (uint k = 0; k < reg_size; ++k) {
                *(tw_it++) = wnk(l_size, k);
            }
            for (uint k = 0; k < reg_size * 2; ++k) {
                *(tw_it++) = wnk(l_size * 2UL, k);
            }

            for (uint k = 0; k < reg_size * 4; ++k) {
                *(tw_it++) = wnk(l_size * 4UL, k);
            }

            l_size *= 8;
            n_groups *= 8;
        }
        while (l_size < sub_size_) {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                for (uint k = 0; k < reg_size; ++k) {
                    *(tw_it++) = wnk(l_size, k + i_group * reg_size);
                }
                for (uint k = 0; k < reg_size; ++k) {
                    *(tw_it++) = wnk(l_size * 2UL, k + i_group * reg_size);
                }
                for (uint k = 0; k < reg_size; ++k) {
                    *(tw_it++) = wnk(l_size * 2UL, k + i_group * reg_size + l_size / 2);
                }
            }
            l_size *= 4;
            n_groups *= 4;
        }

        if (l_size == sub_size_) {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                for (uint k = 0; k < reg_size; ++k) {
                    *(tw_it++) = wnk(l_size, k + i_group * reg_size);
                }
            }
            l_size *= 2;
            n_groups *= 2;
        };

        while (l_size < fft_size) {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                for (uint k = 0; k < reg_size; ++k) {
                    *(tw_it++) = wnk(l_size, k + i_group * reg_size);
                }
                for (uint k = 0; k < reg_size; ++k) {
                    auto a     = wnk(l_size * 2UL, k + i_group * reg_size);
                    *(tw_it++) = wnk(l_size * 2UL, k + i_group * reg_size);
                }
                for (uint k = 0; k < reg_size; ++k) {
                    *(tw_it++) = wnk(l_size * 2UL, k + i_group * reg_size + l_size / 2);
                }
            }
            l_size *= 4;
            n_groups *= 4;
        };
        if (l_size == fft_size) {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                for (uint k = 0; k < reg_size; ++k) {
                    *(tw_it++) = wnk(l_size, k + i_group * reg_size);
                }
            }
            l_size *= 2;
            n_groups *= 2;
        };
        return twiddles;
    }

    static auto get_twiddles_unsorted(std::size_t fft_size, std::size_t sub_size, allocator_type allocator)
        -> std::vector<T, allocator_type> {
        constexpr auto wnk = internal::fft::wnk<T>;

        auto twiddles = std::vector<T, allocator_type>(allocator);

        std::size_t l_size = 2;
        if (log2i(fft_size / (reg_size * 4)) % 2 == 0) {
            insert_tw_unsorted(fft_size, l_size, sub_size, 0, twiddles);
        } else if (fft_size / (reg_size * 4) > 8) {
            insert_tw_unsorted(fft_size, l_size * 8, sub_size, 0, twiddles);
            insert_tw_unsorted(fft_size, l_size * 8, sub_size, 1, twiddles);
            insert_tw_unsorted(fft_size, l_size * 8, sub_size, 2, twiddles);
            insert_tw_unsorted(fft_size, l_size * 8, sub_size, 3, twiddles);
            insert_tw_unsorted(fft_size, l_size * 8, sub_size, 4, twiddles);
            insert_tw_unsorted(fft_size, l_size * 8, sub_size, 5, twiddles);
            insert_tw_unsorted(fft_size, l_size * 8, sub_size, 6, twiddles);
            insert_tw_unsorted(fft_size, l_size * 8, sub_size, 7, twiddles);
        } else {
            auto tw0 = wnk(l_size, reverse_bit_order(0, log2i(l_size / 2)));

            twiddles.push_back(tw0.real());
            twiddles.push_back(tw0.imag());

            insert_tw_unsorted(fft_size, l_size * 2, sub_size, 0, twiddles);
            insert_tw_unsorted(fft_size, l_size * 2, sub_size, 1, twiddles);
        }
        return twiddles;
    }

    static void insert_tw_unsorted(std::size_t                             fft_size,
                                   std::size_t                             l_size,
                                   std::size_t                             sub_size,
                                   std::size_t                             i_group,
                                   std::vector<real_type, allocator_type>& twiddles) {
        constexpr auto wnk = internal::fft::wnk<T>;
        if ((fft_size / l_size) < sub_size) {
            std::size_t start_size       = twiddles.size();
            std::size_t single_load_size = fft_size / (reg_size * 2);
            std::size_t group_size       = 1;

            while (l_size < single_load_size / 2) {
                std::size_t start = group_size * i_group;
                for (uint i = 0; i < group_size; ++i) {
                    auto tw0 = wnk(l_size,    //
                                   reverse_bit_order((start + i), log2i(l_size / 2)));
                    auto tw1 = wnk(l_size * 2,    //
                                   reverse_bit_order((start + i) * 2, log2i(l_size)));
                    auto tw2 = wnk(l_size * 2,    //
                                   reverse_bit_order((start + i) * 2 + 1, log2i(l_size)));

                    twiddles.push_back(tw0.real());
                    twiddles.push_back(tw0.imag());
                    twiddles.push_back(tw1.real());
                    twiddles.push_back(tw1.imag());
                    twiddles.push_back(tw2.real());
                    twiddles.push_back(tw2.imag());
                }
                l_size *= 4;
                group_size *= 4;
            }

            if (l_size == single_load_size / 2) {
                std::size_t start = group_size * i_group;

                for (uint i = 0; i < group_size; ++i) {
                    auto tw0 = wnk(l_size,    //
                                   reverse_bit_order(start + i, log2i(l_size / 2)));

                    twiddles.push_back(tw0.real());
                    twiddles.push_back(tw0.imag());
                }
                l_size *= 2;
                group_size *= 2;
            }

            if (l_size == single_load_size) {
                for (uint i = 0; i < group_size; ++i) {
                    std::size_t start = group_size * i_group + i;

                    auto tw0 = wnk(l_size,    //
                                   reverse_bit_order(start, log2i(l_size / 2)));

                    twiddles.push_back(tw0.real());
                    twiddles.push_back(tw0.imag());

                    auto tw1 = wnk(l_size * 2,    //
                                   reverse_bit_order(start * 2, log2i(l_size)));
                    auto tw2 = wnk(l_size * 2,    //
                                   reverse_bit_order(start * 2 + 1, log2i(l_size)));

                    twiddles.push_back(tw1.real());
                    twiddles.push_back(tw1.imag());
                    twiddles.push_back(tw2.real());
                    twiddles.push_back(tw2.imag());

                    auto tw3_1 = wnk(l_size * 4,    //
                                     reverse_bit_order(start * 4, log2i(l_size * 2)));
                    auto tw3_2 = wnk(l_size * 4,    //
                                     reverse_bit_order(start * 4 + 1, log2i(l_size * 2)));
                    auto tw4_1 = wnk(l_size * 4,    //
                                     reverse_bit_order(start * 4 + 2, log2i(l_size * 2)));
                    auto tw4_2 = wnk(l_size * 4,    //
                                     reverse_bit_order(start * 4 + 3, log2i(l_size * 2)));

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


                    auto tw7  = wnk(l_size * 8,    //
                                   reverse_bit_order(start * 8, log2i(l_size * 4)));
                    auto tw8  = wnk(l_size * 8,    //
                                   reverse_bit_order(start * 8 + 1, log2i(l_size * 4)));
                    auto tw9  = wnk(l_size * 8,    //
                                   reverse_bit_order(start * 8 + 2, log2i(l_size * 4)));
                    auto tw10 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 3, log2i(l_size * 4)));
                    auto tw11 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 4, log2i(l_size * 4)));
                    auto tw12 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 5, log2i(l_size * 4)));
                    auto tw13 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 6, log2i(l_size * 4)));
                    auto tw14 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 7, log2i(l_size * 4)));

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

                    std::array<uint, 8> switched_k = {0, 2, 1, 3, 4, 6, 5, 7};
                    for (auto k: switched_k) {
                        auto tw = wnk(l_size * 16,    //
                                      reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                        twiddles.push_back(tw.real());
                    }
                    for (auto k: switched_k) {
                        auto tw = wnk(l_size * 16,    //
                                      reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                        twiddles.push_back(tw.imag());
                    }

                    for (auto k: switched_k) {
                        auto tw = wnk(l_size * 16,    //
                                      reverse_bit_order(start * 16 + k + 8, log2i(l_size * 8)));
                        twiddles.push_back(tw.real());
                    }
                    for (auto k: switched_k) {
                        auto tw = wnk(l_size * 16,    //
                                      reverse_bit_order(start * 16 + k + 8, log2i(l_size * 8)));
                        twiddles.push_back(tw.imag());
                    }


                    //                     for (uint k = 0; k < 8; ++k) {
                    //                         auto tw = wnk(l_size * 16,    //
                    //                                       reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                    //                         twiddles.push_back(tw.real());
                    //                     }
                    //                     for (uint k = 0; k < 8; ++k) {
                    //                         auto tw = wnk(l_size * 16,    //
                    //                                       reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                    //                         twiddles.push_back(tw.imag());
                    //                     }
                    //
                    //                     for (uint k = 8; k < 16; ++k) {
                    //                         auto tw = wnk(l_size * 16,    //
                    //                                       reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                    //                         twiddles.push_back(tw.real());
                    //                     }
                    //                     for (uint k = 8; k < 16; ++k) {
                    //                         auto tw = wnk(l_size * 16,    //
                    //                                       reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                    //                         twiddles.push_back(tw.imag());
                    //                     }
                }
            }
        } else {
            auto tw0 = wnk(l_size, reverse_bit_order(i_group, log2i(l_size / 2)));
            auto tw1 = wnk(l_size * 2, reverse_bit_order(i_group * 2, log2i(l_size)));
            auto tw2 = wnk(l_size * 2, reverse_bit_order(i_group * 2 + 1, log2i(l_size)));

            twiddles.push_back(tw0.real());
            twiddles.push_back(tw0.imag());
            twiddles.push_back(tw1.real());
            twiddles.push_back(tw1.imag());
            twiddles.push_back(tw2.real());
            twiddles.push_back(tw2.imag());

            l_size *= 4;
            i_group *= 4;

            insert_tw_unsorted(fft_size, l_size, sub_size, i_group + 0, twiddles);
            insert_tw_unsorted(fft_size, l_size, sub_size, i_group + 1, twiddles);
            insert_tw_unsorted(fft_size, l_size, sub_size, i_group + 2, twiddles);
            insert_tw_unsorted(fft_size, l_size, sub_size, i_group + 3, twiddles);
        }
    }
};

// NOLINTEND (*magic-numbers)
}    // namespace pcx
#endif