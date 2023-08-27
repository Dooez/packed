#ifndef AVX512_COMMON_HPP
#define AVX512_COMMON_HPP

#include "simd_common.hpp"
#include "types.hpp"

#include <immintrin.h>

namespace pcx::simd {

template<>
struct reg<float> {
    using type = __m512;
    static constexpr uZ size{64 / sizeof(float)};
};
template<>
struct reg<double> {
    using type = __m512d;
    static constexpr uZ size{64 / sizeof(double)};
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
    return _mm512_setzero_ps();
};
template<>
inline auto zero<double>() -> reg_t<double> {
    return _mm512_setzero_pd();
};
inline auto broadcast(const float* source) -> reg_t<float> {
    return _mm512_set1_ps(*source);
}
inline auto broadcast(const double* source) -> reg_t<double> {
    return _mm512_set1_pd(*source);
}
template<typename T>
inline auto broadcast(T source) -> reg_t<T> {
    return broadcast(&source);
}
inline auto load(const float* source) -> reg_t<float> {
    return _mm512_loadu_ps(source);
}
inline auto load(const double* source) -> reg_t<double> {
    return _mm512_loadu_pd(source);
}
inline void store(float* dest, reg_t<float> reg) {
    return _mm512_storeu_ps(dest, reg);
}
inline void store(double* dest, reg_t<double> reg) {
    return _mm512_storeu_pd(dest, reg);
}

template<uZ PackSize, typename T>
inline auto broadcast(std::complex<T> source) -> cx_reg<T, false, PackSize> {
    return {broadcast(source.real()), broadcast(source.imag())};
}
template<typename T>
inline auto broadcast(std::complex<T> source) -> cx_reg<T, false, reg<T>::size> {
    return {broadcast(source.real()), broadcast(source.imag())};
}
namespace avx512 {

inline auto unpacklo_32(reg_t<float> a, reg_t<float> b) -> reg_t<float> {
    return _mm512_unpacklo_ps(a, b);
};
inline auto unpackhi_32(reg_t<float> a, reg_t<float> b) -> reg_t<float> {
    return _mm512_unpackhi_ps(a, b);
};

inline auto unpacklo_64(reg_t<float> a, reg_t<float> b) -> reg_t<float> {
    return _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(a), _mm512_castps_pd(b)));
};
inline auto unpackhi_64(reg_t<float> a, reg_t<float> b) -> reg_t<float> {
    return _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(a), _mm512_castps_pd(b)));
};
inline auto unpacklo_64(reg_t<double> a, reg_t<double> b) -> reg_t<double> {
    return _mm512_unpacklo_pd(a, b);
};
inline auto unpackhi_64(reg_t<double> a, reg_t<double> b) -> reg_t<double> {
    return _mm512_unpackhi_pd(a, b);
};

inline auto unpacklo_128(reg_t<float> a, reg_t<float> b) -> reg_t<float> {
    return _mm512_shuffle_f32x4(a, b, 0b10001000);
};
inline auto unpackhi_128(reg_t<float> a, reg_t<float> b) -> reg_t<float> {
    return _mm512_shuffle_f32x4(a, b, 0b11011101);
};
inline auto unpacklo_128(reg_t<double> a, reg_t<double> b) -> reg_t<double> {
    return _mm512_shuffle_f64x2(a, b, 0b10001000);
};
inline auto unpackhi_128(reg_t<double> a, reg_t<double> b) -> reg_t<double> {
    return _mm512_shuffle_f64x2(a, b, 0b11011101);
};

inline auto unpacklo_256(reg_t<float> a, reg_t<float> b) -> reg_t<float> {
    return _mm512_shuffle_f32x4(a, b, 0b01000100);
};
inline auto unpackhi_256(reg_t<float> a, reg_t<float> b) -> reg_t<float> {
    return _mm512_shuffle_f32x4(a, b, 0b11101110);
};
inline auto unpacklo_256(reg_t<double> a, reg_t<double> b) -> reg_t<double> {
    return _mm512_shuffle_f64x2(a, b, 0b01000100);
};
inline auto unpackhi_256(reg_t<double> a, reg_t<double> b) -> reg_t<double> {
    return _mm512_shuffle_f64x2(a, b, 0b11101110);
};

}    // namespace avx512

template<uZ PackTo, uZ PackFrom, bool... Conj>
    requires pack_size<PackTo> && (PackTo <= reg<float>::size)
inline auto repack2(cx_reg<float, Conj, PackFrom>... args) {
    auto tup = std::make_tuple(args...);

    constexpr auto swap_12 =
        []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) /* C++23 static */ {
            auto real = _mm512_shuffle_ps(reg.real, reg.real, 0b11011000);
            auto imag = _mm512_shuffle_ps(reg.imag, reg.imag, 0b11011000);
            return cx_reg<float, Conj_, PackTo>({real, imag});
        };
    constexpr auto swap_24 =
        []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) /* C++23 static */ {
            auto real = _mm512_permutex_pd(_mm512_castps_pd(reg.real), 0b11011000);
            auto imag = _mm512_permutex_pd(_mm512_castps_pd(reg.imag), 0b11011000);
            return cx_reg<float, Conj_, PackTo>({_mm512_castpd_ps(real), _mm512_castpd_ps(imag)});
        };
    constexpr auto swap_48 =
        []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) /* C++23 static */ {
            auto real = _mm512_shuffle_f32x4(reg.real, reg.real, 0b11011000);
            auto imag = _mm512_shuffle_f32x4(reg.imag, reg.imag, 0b11011000);
            return cx_reg<float, Conj_, PackTo>({real, imag});
        };
    constexpr auto swap_816 =
        []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) /* C++23 static */ {
            auto real = avx512::unpacklo_256(reg.real, reg.imag);
            auto imag = avx512::unpackhi_256(reg.real, reg.imag);
            return cx_reg<float, Conj_, PackTo>({real, imag});
        };
    if constexpr (PackFrom == PackTo) {
        return tup;
    } else if constexpr (PackFrom == 1) {
        if constexpr (PackTo == 16) {
            auto pack_1 = []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) /* C++23 static */ {
                auto real = _mm512_shuffle_ps(reg.real, reg.imag, 0b10001000);
                auto imag = _mm512_shuffle_ps(reg.real, reg.imag, 0b11011101);
                return cx_reg<float, Conj_, PackTo>({real, imag});
            };
            auto tmp1 = pcx::detail_::apply_for_each(pack_1, tup);
            auto tmp2 = pcx::detail_::apply_for_each(swap_24, tmp1);
            return pcx::detail_::apply_for_each(swap_48, tmp2);
        } else if constexpr (PackTo == 8) {
            auto tmp1 = pcx::detail_::apply_for_each(swap_12, tup);
            auto tmp2 = pcx::detail_::apply_for_each(swap_24, tmp1);
            return pcx::detail_::apply_for_each(swap_48, tmp2);
        } else if constexpr (PackTo == 4) {
            auto tmp = pcx::detail_::apply_for_each(swap_12, tup);
            return pcx::detail_::apply_for_each(swap_24, tmp);
        } else if constexpr (PackTo == 2) {
            return pcx::detail_::apply_for_each(swap_12, tup);
        }
    } else if constexpr (PackFrom == 2) {
        if constexpr (PackTo == 16) {
            auto pack_1 = []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) {
                auto real = avx512::unpacklo_64(reg.real, reg.imag);
                auto imag = avx512::unpackhi_64(reg.real, reg.imag);
                return cx_reg<float, Conj_, PackTo>{real, imag};
            };
            auto tmp1 = pcx::detail_::apply_for_each(pack_1, tup);
            auto tmp2 = pcx::detail_::apply_for_each(swap_24, tmp1);
            return pcx::detail_::apply_for_each(swap_48, tmp2);
        } else if constexpr (PackTo == 8) {
            auto pack_1 = []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) {
                auto real = avx512::unpacklo_64(reg.real, reg.imag);
                auto imag = avx512::unpackhi_64(reg.real, reg.imag);
                return cx_reg<float, Conj_, PackTo>{real, imag};
            };
            auto tmp1 = pcx::detail_::apply_for_each(pack_1, tup);
            auto tmp2 = pcx::detail_::apply_for_each(swap_24, tmp1);
            auto tmp3 = pcx::detail_::apply_for_each(swap_48, tmp2);
            return pcx::detail_::apply_for_each(swap_816, tmp3);
        } else if constexpr (PackTo == 4) {
            return pcx::detail_::apply_for_each(swap_24, tup);
        } else if constexpr (PackTo == 1) {
            return pcx::detail_::apply_for_each(swap_12, tup);
        }
    } else if constexpr (PackFrom == 4) {
        if constexpr (PackTo == 16) {
            auto pack_1 = []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) {
                auto real = avx512::unpacklo_128(reg.real, reg.imag);
                auto imag = avx512::unpackhi_128(reg.real, reg.imag);
                return cx_reg<float, Conj_, PackTo>{real, imag};
            };
            return pcx::detail_::apply_for_each(pack_1, tup);
        } else if constexpr (PackTo == 8) {
            return pcx::detail_::apply_for_each(swap_48, tup);
        } else if constexpr (PackTo == 2) {
            return pcx::detail_::apply_for_each(swap_24, tup);
        } else if constexpr (PackTo == 1) {
            auto tmp = pcx::detail_::apply_for_each(swap_24, tup);
            return pcx::detail_::apply_for_each(swap_12, tmp);
        }
    } else if constexpr (PackFrom == 8) {
        if constexpr (PackTo == 16) {
            return pcx::detail_::apply_for_each(swap_816, tup);
        } else if constexpr (PackTo == 4) {
            auto pack_1 = []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) {
                auto real = avx512::unpacklo_128(reg.real, reg.imag);
                auto imag = avx512::unpackhi_128(reg.real, reg.imag);
                return cx_reg<float, Conj_, PackTo>{real, imag};
            };
            auto tmp = pcx::detail_::apply_for_each(pack_1, tup);
            return pcx::detail_::apply_for_each(swap_816, tmp);
        } else if constexpr (PackTo == 2) {
            auto tmp = pcx::detail_::apply_for_each(swap_48, tup);
            return pcx::detail_::apply_for_each(swap_24, tmp);
        } else if constexpr (PackTo == 1) {
            auto pack_0 = []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) {
                auto real = simd::avx512::unpacklo_32(reg.real, reg.imag);
                auto imag = simd::avx512::unpackhi_32(reg.real, reg.imag);
                return cx_reg<float, Conj_, PackTo>{real, imag};
            };
            auto tmp1 = pcx::detail_::apply_for_each(swap_816, tup);
            auto tmp2 = pcx::detail_::apply_for_each(swap_48, tmp1);
            auto tmp3 = pcx::detail_::apply_for_each(swap_24, tmp2);
            return pcx::detail_::apply_for_each(pack_0, tmp3);
        }
    } else if constexpr (PackFrom == 16) {
        if constexpr (PackTo == 8) {
            return pcx::detail_::apply_for_each(swap_816, tup);
        } else if constexpr (PackTo == 4) {
            auto tmp = pcx::detail_::apply_for_each(swap_816, tup);
            return pcx::detail_::apply_for_each(swap_48, tmp);
        } else if constexpr (PackTo == 2) {
            auto tmp1 = pcx::detail_::apply_for_each(swap_816, tup);
            auto tmp2 = pcx::detail_::apply_for_each(swap_48, tmp1);
            return pcx::detail_::apply_for_each(swap_24, tmp2);
        } else if constexpr (PackTo == 1) {
            auto pack_0 = []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) {
                auto real = simd::avx512::unpacklo_32(reg.real, reg.imag);
                auto imag = simd::avx512::unpackhi_32(reg.real, reg.imag);
                return cx_reg<float, Conj_, PackTo>{real, imag};
            };
            auto tmp1 = pcx::detail_::apply_for_each(swap_48, tup);
            auto tmp2 = pcx::detail_::apply_for_each(swap_24, tmp1);
            return pcx::detail_::apply_for_each(pack_0, tmp2);
        }
    }
};

// TODO: remove nested if since they are constexpr anyway
template<uZ PackTo, uZ PackFrom, bool... Conj>
    requires pack_size<PackTo> && (PackTo <= reg<double>::size)
static inline auto repack2(cx_reg<double, Conj, PackFrom>... args) {
    constexpr auto swap_12 = []<bool Conj_, uZ PackSize>(cx_reg<double, Conj_, PackSize> reg) {
        auto real = _mm512_permute4x64_pd(reg.real, 0b11011000);
        auto imag = _mm512_permute4x64_pd(reg.imag, 0b11011000);
        return cx_reg<double, Conj_, PackTo>({real, imag});
    };
    constexpr auto swap_24 = []<bool Conj_, uZ PackSize>(cx_reg<double, Conj_, PackSize> reg) {
        auto real = avx512::unpacklo_128(reg.real, reg.imag);
        auto imag = avx512::unpackhi_128(reg.real, reg.imag);
        return cx_reg<double, Conj_, PackTo>({real, imag});
    };

    auto tup = std::make_tuple(args...);
    if constexpr (PackFrom == PackTo || (PackFrom >= 4 && PackTo >= 4)) {
        auto pack = []<bool Conj_>(cx_reg<double, Conj_, PackFrom> reg) {
            return cx_reg<double, Conj_, PackTo>({reg.real, reg.imag});
        };
        return pcx::detail_::apply_for_each(pack, tup);
    } else if constexpr (PackFrom == 1) {
        if constexpr (PackTo >= 4) {
            constexpr auto pack_1 = []<bool Conj_, uZ PackSize>(cx_reg<double, Conj_, PackSize> reg) {
                auto real = avx512::unpacklo_64(reg.real, reg.imag);
                auto imag = avx512::unpackhi_64(reg.real, reg.imag);
                return cx_reg<double, Conj_, PackTo>({real, imag});
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
        if constexpr (PackTo == 2) {
            return pcx::detail_::apply_for_each(swap_24, tup);
        } else if constexpr (PackTo == 1) {
            constexpr auto pack_1 = []<bool Conj_, uZ PackSize>(cx_reg<double, Conj_, PackSize> reg) {
                auto real = avx512::unpacklo_64(reg.real, reg.imag);
                auto imag = avx512::unpackhi_64(reg.real, reg.imag);
                return cx_reg<double, Conj_, PackTo>({real, imag});
            };
            auto tmp = pcx::detail_::apply_for_each(pack_1, tup);
            return pcx::detail_::apply_for_each(swap_24, tmp);
        }
    }
};

template<uZ SrcSize, uZ PackSize, typename T>
inline auto cxload(const T* ptr) {
    constexpr auto LoadSize  = std::max(SrcSize, reg<T>::size);
    constexpr auto PackSize_ = std::min(PackSize, reg<T>::size);

    auto data_  = cx_reg<T, false, std::min(SrcSize, reg<T>::size)>{load(ptr), load(ptr + LoadSize)};
    auto [data] = repack2<PackSize_>(data_);
    return data;
}
template<uZ DestSize, uZ PackSize_, typename T>
inline void cxstore(T* ptr, cx_reg<T, false, PackSize_> data) {
    constexpr auto StoreSize = std::max(DestSize, reg<T>::size);

    auto [data_] = repack2<std::min(DestSize, reg<T>::size)>(data);
    store(ptr, data_.real);
    store(ptr + StoreSize, data_.imag);
}

template<uZ PackSize, typename T, bool Conj>
inline auto cxloadstore(T* ptr, cx_reg<T, Conj, PackSize> reg) -> cx_reg<T, false, PackSize> {
    auto tmp = cxload<PackSize>(ptr);
    cxstore<PackSize>(ptr, reg);
    return tmp;
}
template<uZ LoadSize, uZ StoreSize, typename T, bool Conj, uZ PackSize>
inline auto cxloadstore(T* ptr, cx_reg<T, Conj, PackSize> reg) -> cx_reg<T, false, LoadSize> {
    auto tmp = cxload<LoadSize>(ptr);
    cxstore<StoreSize>(ptr, reg);
    return tmp;
}

template<>
inline auto add(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm512_add_ps(lhs, rhs);
}
template<>
inline auto add(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm512_add_pd(lhs, rhs);
}
template<>
inline auto sub(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm512_sub_ps(lhs, rhs);
}
template<>
inline auto sub(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm512_sub_pd(lhs, rhs);
}
template<>
inline auto mul(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm512_mul_ps(lhs, rhs);
}
template<>
inline auto mul(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm512_mul_pd(lhs, rhs);
}
template<>
inline auto div(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm512_div_ps(lhs, rhs);
}
template<>
inline auto div(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm512_div_pd(lhs, rhs);
}

template<>
inline auto fmadd(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm512_fmadd_ps(a, b, c);
}
template<>
inline auto fmadd(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm512_fmadd_pd(a, b, c);
}
template<>
inline auto fnmadd(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm512_fnmadd_ps(a, b, c);
}
template<>
inline auto fnmadd(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm512_fnmadd_pd(a, b, c);
}
template<>
inline auto fmsub(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm512_fmsub_ps(a, b, c);
}
template<>
inline auto fmsub(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm512_fmsub_pd(a, b, c);
}
template<>
inline auto fnmsub(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm512_fnmsub_ps(a, b, c);
}
template<>
inline auto fnmsub(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm512_fnmsub_pd(a, b, c);
}

template<uZ PackSize, bool Conj_>
auto apply_conj(cx_reg<float, Conj_, PackSize> reg) -> cx_reg<float, false, PackSize> {
    if constexpr (!Conj_) {
        return reg;
    } else if constexpr (PackSize == 16) {
        return {reg.real, sub(zero<float>(), reg.imag)};
    } else if constexpr (PackSize == 8) {
        auto pos  = zero<float>();
        auto neg  = broadcast(-0.F);
        auto mask = avx512::unpacklo_256(pos, neg);
        return {_mm512_xor_ps(reg.real, mask), _mm512_xor_ps(reg.imag, mask)};
    } else if constexpr (PackSize == 4) {
        auto pos  = zero<float>();
        auto neg  = broadcast(-0.F);
        auto mask = avx512::unpacklo_256(pos, neg);
        mask      = avx512::unpacklo_128(mask, mask);
        return {_mm512_xor_ps(reg.real, mask), _mm512_xor_ps(reg.imag, mask)};
    } else if constexpr (PackSize == 2) {
        auto pos  = zero<float>();
        auto neg  = broadcast(-0.F);
        auto mask = avx512::unpacklo_64(pos, neg);
        return {_mm512_xor_ps(reg.real, mask), _mm512_xor_ps(reg.imag, mask)};
    } else {
        auto pos  = zero<float>();
        auto neg  = broadcast(-0.F);
        auto mask = avx512::unpacklo_32(pos, neg);
        return {_mm512_xor_ps(reg.real, mask), _mm512_xor_ps(reg.imag, mask)};
    }
}
template<uZ PackSize, bool Conj_>
auto apply_conj(cx_reg<double, Conj_, PackSize> reg) -> cx_reg<double, false, PackSize> {
    if constexpr (!Conj_) {
        return reg;
    } else if constexpr (PackSize >= 4) {
        return {reg.real, sub(zero<double>(), reg.imag)};
    } else if constexpr (PackSize == 2) {
        auto pos  = zero<double>();
        auto neg  = broadcast(-0.);
        auto mask = avx512::unpacklo_128(pos, neg);
        return {_mm512_xor_pd(reg.real, mask), _mm512_xor_pd(reg.imag, mask)};
    } else {
        auto pos  = zero<double>();
        auto neg  = broadcast(-0.);
        auto mask = avx512::unpacklo_64(pos, neg);
        return {_mm512_xor_pd(reg.real, mask), _mm512_xor_pd(reg.imag, mask)};
    };
}
}    // namespace pcx::simd

#endif