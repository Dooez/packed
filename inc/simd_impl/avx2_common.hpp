#ifndef AVX2_COMMON_HPP
#define AVX2_COMMON_HPP

#include "simd_common.hpp"
#include "types.hpp"

#include <immintrin.h>

namespace pcx::simd {

template<>
struct reg<float> {
    using type               = __m256;
    static constexpr uZ size = 32 / sizeof(float);
};
template<>
struct reg<double> {
    using type               = __m256d;
    static constexpr uZ size = 32 / sizeof(double);
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

template<uZ PackSize, typename T>
inline auto broadcast(std::complex<T> source) -> cx_reg<T, false, PackSize> {
    return {broadcast(source.real()), broadcast(source.imag())};
}
template<typename T>
inline auto broadcast(std::complex<T> source) -> cx_reg<T, false, reg<T>::size> {
    return {broadcast(source.real()), broadcast(source.imag())};
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

template<bool Conj, uZ PackSize>
inline auto unpack_ps(cx_reg<float, Conj, PackSize> a, cx_reg<float, Conj, PackSize> b) {
    auto real_lo = unpacklo_ps(a.real, b.real);
    auto real_hi = unpackhi_ps(a.real, b.real);
    auto imag_lo = unpacklo_ps(a.imag, b.imag);
    auto imag_hi = unpackhi_ps(a.imag, b.imag);

    return std::make_tuple(cx_reg<float, Conj, PackSize>({real_lo, imag_lo}),
                           cx_reg<float, Conj, PackSize>({real_hi, imag_hi}));
};

template<typename T, bool Conj, uZ PackSize>
inline auto unpack_pd(cx_reg<T, Conj, PackSize> a, cx_reg<T, Conj, PackSize> b) {
    auto real_lo = unpacklo_pd(a.real, b.real);
    auto real_hi = unpackhi_pd(a.real, b.real);
    auto imag_lo = unpacklo_pd(a.imag, b.imag);
    auto imag_hi = unpackhi_pd(a.imag, b.imag);

    return std::make_tuple(cx_reg<T, Conj, PackSize>({real_lo, imag_lo}),
                           cx_reg<T, Conj, PackSize>({real_hi, imag_hi}));
};

template<typename T, bool Conj, uZ PackSize>
inline auto unpack_128(cx_reg<T, Conj, PackSize> a, cx_reg<T, Conj, PackSize> b) {
    auto real_hi = unpackhi_128(a.real, b.real);
    auto real_lo = unpacklo_128(a.real, b.real);
    auto imag_hi = unpackhi_128(a.imag, b.imag);
    auto imag_lo = unpacklo_128(a.imag, b.imag);

    return std::make_tuple(cx_reg<T, Conj, PackSize>({real_lo, imag_lo}),
                           cx_reg<T, Conj, PackSize>({real_hi, imag_hi}));
};

inline constexpr auto float_swap_12 =
    []<bool Conj, uZ PackSize>(cx_reg<float, Conj, PackSize> reg) /* static */ {
        auto real = _mm256_shuffle_ps(reg.real, reg.real, 0b11011000);
        auto imag = _mm256_shuffle_ps(reg.imag, reg.imag, 0b11011000);
        return cx_reg<float, Conj, PackSize>({real, imag});
    };

inline constexpr auto float_swap_24 =
    []<bool Conj, uZ PackSize>(cx_reg<float, Conj, PackSize> reg) /* static */ {
        auto real = _mm256_permute4x64_pd(_mm256_castps_pd(reg.real), 0b11011000);
        auto imag = _mm256_permute4x64_pd(_mm256_castps_pd(reg.imag), 0b11011000);
        return cx_reg<float, Conj, PackSize>({_mm256_castpd_ps(real), _mm256_castpd_ps(imag)});
    };

inline constexpr auto float_swap_48 =
    []<bool Conj, uZ PackSize>(cx_reg<float, Conj, PackSize> reg) /* static */ {
        auto real = unpacklo_128(reg.real, reg.imag);
        auto imag = unpackhi_128(reg.real, reg.imag);
        return cx_reg<float, Conj, PackSize>({real, imag});
    };

inline constexpr auto double_swap_12 = []<bool Conj, uZ PackSize>(cx_reg<double, Conj, PackSize> reg) {
    auto real = _mm256_permute4x64_pd(reg.real, 0b11011000);
    auto imag = _mm256_permute4x64_pd(reg.imag, 0b11011000);
    return cx_reg<double, Conj, PackSize>({real, imag});
};

inline constexpr auto double_swap_24 = []<bool Conj, uZ PackSize>(cx_reg<double, Conj, PackSize> reg) {
    auto real = avx2::unpacklo_128(reg.real, reg.imag);
    auto imag = avx2::unpackhi_128(reg.real, reg.imag);
    return cx_reg<double, Conj, PackSize>({real, imag});
};

}    // namespace avx2


template<uZ PackFrom, uZ PackTo, bool... Conj>
    requires((PackFrom > 0) && (PackTo > 0))
inline auto repack(cx_reg<float, Conj>... args) {
    auto tup = std::make_tuple(args...);
    if constexpr (PackFrom == PackTo || (PackFrom >= 8 && PackTo >= 8)) {
        return tup;
    } else if constexpr (PackFrom == 1) {
        if constexpr (PackTo >= 8) {
            auto pack_1 = []<bool Conj_>(cx_reg<float, Conj_> reg) {
                auto real = _mm256_shuffle_ps(reg.real, reg.imag, 0b10001000);
                auto imag = _mm256_shuffle_ps(reg.real, reg.imag, 0b11011101);
                return cx_reg<float, Conj_>({real, imag});
            };
            auto tmp = pcx::detail_::apply_for_each(avx2::float_swap_48, tup);
            return pcx::detail_::apply_for_each(pack_1, tmp);
        } else if constexpr (PackTo == 4) {
            auto tmp = pcx::detail_::apply_for_each(avx2::float_swap_12, tup);
            return pcx::detail_::apply_for_each(avx2::float_swap_24, tmp);
        } else if constexpr (PackTo == 2) {
            return pcx::detail_::apply_for_each(avx2::float_swap_12, tup);
        }
    } else if constexpr (PackFrom == 2) {
        if constexpr (PackTo >= 8) {
            auto pack_1 = []<bool Conj_>(cx_reg<float, Conj_> reg) {
                auto real = avx2::unpacklo_pd(reg.real, reg.imag);
                auto imag = avx2::unpackhi_pd(reg.real, reg.imag);
                return cx_reg<float, Conj_>({real, imag});
            };
            auto tmp = pcx::detail_::apply_for_each(avx2::float_swap_48, tup);
            return pcx::detail_::apply_for_each(pack_1, tmp);
        } else if constexpr (PackTo == 4) {
            return pcx::detail_::apply_for_each(avx2::float_swap_24, tup);
        } else if constexpr (PackTo == 1) {
            return pcx::detail_::apply_for_each(avx2::float_swap_12, tup);
        }
    } else if constexpr (PackFrom == 4) {
        if constexpr (PackTo >= 8) {
            return pcx::detail_::apply_for_each(avx2::float_swap_48, tup);
        } else if constexpr (PackTo == 2) {
            return pcx::detail_::apply_for_each(avx2::float_swap_24, tup);
        } else if constexpr (PackTo == 1) {
            auto tmp = pcx::detail_::apply_for_each(avx2::float_swap_24, tup);
            return pcx::detail_::apply_for_each(avx2::float_swap_12, tmp);
        }
    } else if constexpr (PackFrom >= 8) {
        if constexpr (PackTo == 4) {
            return pcx::detail_::apply_for_each(avx2::float_swap_48, tup);
        } else if constexpr (PackTo == 2) {
            auto tmp = pcx::detail_::apply_for_each(avx2::float_swap_48, tup);
            return pcx::detail_::apply_for_each(avx2::float_swap_24, tmp);
        } else if constexpr (PackTo == 1) {
            auto pack_0 = []<bool Conj_>(cx_reg<float, Conj_> reg) {
                auto real = simd::avx2::unpacklo_ps(reg.real, reg.imag);
                auto imag = simd::avx2::unpackhi_ps(reg.real, reg.imag);
                return cx_reg<float, Conj_>({real, imag});
            };
            auto tmp = pcx::detail_::apply_for_each(pack_0, tup);
            return pcx::detail_::apply_for_each(avx2::float_swap_48, tmp);
        }
    }
};

template<uZ PackTo, uZ PackFrom, bool... Conj>
    requires pack_size<PackTo> && (PackTo <= reg<float>::size)
inline auto repack2(cx_reg<float, Conj, PackFrom>... args) {
    auto tup = std::make_tuple(args...);

    constexpr auto swap_12 = []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) /* static */ {
        auto real = _mm256_shuffle_ps(reg.real, reg.real, 0b11011000);
        auto imag = _mm256_shuffle_ps(reg.imag, reg.imag, 0b11011000);
        return cx_reg<float, Conj_, PackTo>({real, imag});
    };
    constexpr auto swap_24 = []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) /* static */ {
        auto real = _mm256_permute4x64_pd(_mm256_castps_pd(reg.real), 0b11011000);
        auto imag = _mm256_permute4x64_pd(_mm256_castps_pd(reg.imag), 0b11011000);
        return cx_reg<float, Conj_, PackTo>({_mm256_castpd_ps(real), _mm256_castpd_ps(imag)});
    };
    constexpr auto swap_48 = []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) /* static */ {
        auto real = avx2::unpacklo_128(reg.real, reg.imag);
        auto imag = avx2::unpackhi_128(reg.real, reg.imag);
        return cx_reg<float, Conj_, PackTo>({real, imag});
    };

    if constexpr (PackFrom == PackTo || (PackFrom >= 8 && PackTo >= 8)) {
        auto pack = []<bool Conj_>(cx_reg<float, Conj_, PackFrom> reg) {
            return cx_reg<float, Conj_, PackTo>{reg.real, reg.imag};
        };
        return pcx::detail_::apply_for_each(pack, tup);
    } else if constexpr (PackFrom == 1) {
        if constexpr (PackTo >= 8) {
            auto pack_1 = []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) {
                auto real = _mm256_shuffle_ps(reg.real, reg.imag, 0b10001000);
                auto imag = _mm256_shuffle_ps(reg.real, reg.imag, 0b11011101);
                return cx_reg<float, Conj_, PackTo>({real, imag});
            };
            auto tmp = pcx::detail_::apply_for_each(swap_48, tup);
            return pcx::detail_::apply_for_each(pack_1, tmp);
        } else if constexpr (PackTo == 4) {
            auto tmp = pcx::detail_::apply_for_each(swap_12, tup);

            return pcx::detail_::apply_for_each(swap_24, tmp);
        } else if constexpr (PackTo == 2) {
            return pcx::detail_::apply_for_each(swap_12, tup);
        }
    } else if constexpr (PackFrom == 2) {
        if constexpr (PackTo >= 8) {
            auto pack_1 = []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) {
                auto real = avx2::unpacklo_pd(reg.real, reg.imag);
                auto imag = avx2::unpackhi_pd(reg.real, reg.imag);
                return cx_reg<float, Conj_, PackTo>({real, imag});
            };
            auto tmp = pcx::detail_::apply_for_each(swap_48, tup);
            return pcx::detail_::apply_for_each(pack_1, tmp);
        } else if constexpr (PackTo == 4) {
            return pcx::detail_::apply_for_each(swap_24, tup);
        } else if constexpr (PackTo == 1) {
            return pcx::detail_::apply_for_each(swap_12, tup);
        }
    } else if constexpr (PackFrom == 4) {
        if constexpr (PackTo >= 8) {
            return pcx::detail_::apply_for_each(swap_48, tup);
        } else if constexpr (PackTo == 2) {
            return pcx::detail_::apply_for_each(swap_24, tup);
        } else if constexpr (PackTo == 1) {
            auto tmp = pcx::detail_::apply_for_each(swap_24, tup);
            return pcx::detail_::apply_for_each(swap_12, tmp);
        }
    } else if constexpr (PackFrom >= 8) {
        if constexpr (PackTo == 4) {
            return pcx::detail_::apply_for_each(swap_48, tup);
        } else if constexpr (PackTo == 2) {
            auto tmp = pcx::detail_::apply_for_each(swap_48, tup);
            return pcx::detail_::apply_for_each(swap_24, tmp);
        } else if constexpr (PackTo == 1) {
            auto pack_0 = []<bool Conj_, uZ PackSize>(cx_reg<float, Conj_, PackSize> reg) {
                auto real = simd::avx2::unpacklo_ps(reg.real, reg.imag);
                auto imag = simd::avx2::unpackhi_ps(reg.real, reg.imag);
                return cx_reg<float, Conj_, PackTo>({real, imag});
            };
            auto tmp = pcx::detail_::apply_for_each(pack_0, tup);
            return pcx::detail_::apply_for_each(swap_48, tmp);
        }
    }
};

template<uZ PackFrom, uZ PackTo, bool... Conj>
    requires((PackFrom > 0) && (PackTo > 0))
static inline auto repack(cx_reg<double, Conj>... args) {
    auto tup = std::make_tuple(args...);
    if constexpr (PackFrom == PackTo || (PackFrom >= 4 && PackTo >= 4)) {
        return tup;
    } else if constexpr (PackFrom == 1) {
        if constexpr (PackTo >= 4) {
            auto pack_1 = []<bool Conj_>(cx_reg<double, Conj_> reg) {
                auto real = avx2::unpacklo_pd(reg.real, reg.imag);
                auto imag = avx2::unpackhi_pd(reg.real, reg.imag);
                return cx_reg<double, Conj_>({real, imag});
            };
            auto tmp = pcx::detail_::apply_for_each(pack_1, tup);
            return pcx::detail_::apply_for_each(avx2::double_swap_12, tmp);
        } else if constexpr (PackTo == 2) {
            return pcx::detail_::apply_for_each(avx2::double_swap_12, tup);
        }
    } else if constexpr (PackFrom == 2) {
        if constexpr (PackTo >= 4) {
            return pcx::detail_::apply_for_each(avx2::double_swap_24, tup);
        } else if constexpr (PackTo == 1) {
            return pcx::detail_::apply_for_each(avx2::double_swap_12, tup);
        }
    } else if constexpr (PackFrom >= 4) {
        if constexpr (PackTo == 2) {
            return pcx::detail_::apply_for_each(avx2::double_swap_24, tup);
        } else if constexpr (PackTo == 1) {
            auto pack_1 = []<bool Conj_>(cx_reg<double, Conj_> reg) {
                auto real = avx2::unpacklo_pd(reg.real, reg.imag);
                auto imag = avx2::unpackhi_pd(reg.real, reg.imag);
                return cx_reg<double, Conj_>({real, imag});
            };
            auto tmp = pcx::detail_::apply_for_each(pack_1, tup);
            return pcx::detail_::apply_for_each(avx2::double_swap_24, tmp);
        }
    }
};

// TODO: remove nested if since they are constexpr anyway
template<uZ PackTo, uZ PackFrom, bool... Conj>
    requires pack_size<PackTo> && (PackTo <= reg<double>::size)
static inline auto repack2(cx_reg<double, Conj, PackFrom>... args) {
    constexpr auto swap_12 = []<bool Conj_, uZ PackSize>(cx_reg<double, Conj_, PackSize> reg) {
        auto real = _mm256_permute4x64_pd(reg.real, 0b11011000);
        auto imag = _mm256_permute4x64_pd(reg.imag, 0b11011000);
        return cx_reg<double, Conj_, PackTo>({real, imag});
    };
    constexpr auto swap_24 = []<bool Conj_, uZ PackSize>(cx_reg<double, Conj_, PackSize> reg) {
        auto real = avx2::unpacklo_128(reg.real, reg.imag);
        auto imag = avx2::unpackhi_128(reg.real, reg.imag);
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
                auto real = avx2::unpacklo_pd(reg.real, reg.imag);
                auto imag = avx2::unpackhi_pd(reg.real, reg.imag);
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
                auto real = avx2::unpacklo_pd(reg.real, reg.imag);
                auto imag = avx2::unpackhi_pd(reg.real, reg.imag);
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

template<uZ PackSize, bool Conj_>
auto apply_conj(cx_reg<float, Conj_, PackSize> reg) -> cx_reg<float, false, PackSize> {
    if constexpr (!Conj_) {
        return reg;
    } else if constexpr (PackSize >= 8) {
        return {reg.real, sub(zero<float>(), reg.imag)};
    } else if constexpr (PackSize == 4) {
        auto pos  = zero<float>();
        auto neg  = broadcast(-0.F);
        auto mask = avx2::unpacklo_128(pos, neg);
        return {_mm256_xor_ps(reg.real, mask), _mm256_xor_ps(reg.imag, mask)};
    } else if constexpr (PackSize == 2) {
        auto pos  = zero<float>();
        auto neg  = broadcast(-0.F);
        auto mask = avx2::unpacklo_pd(pos, neg);
        return {_mm256_xor_ps(reg.real, mask), _mm256_xor_ps(reg.imag, mask)};
    } else {
        auto pos  = zero<float>();
        auto neg  = broadcast(-0.F);
        auto mask = avx2::unpacklo_ps(pos, neg);
        return {_mm256_xor_ps(reg.real, mask), _mm256_xor_ps(reg.imag, mask)};
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
        auto mask = avx2::unpacklo_128(pos, neg);
        return {_mm256_xor_pd(reg.real, mask), _mm256_xor_pd(reg.imag, mask)};
    } else {
        auto pos  = zero<double>();
        auto neg  = broadcast(-0.);
        auto mask = avx2::unpacklo_pd(pos, neg);
        return {_mm256_xor_pd(reg.real, mask), _mm256_xor_pd(reg.imag, mask)};
    };
}
}    // namespace pcx::simd

#endif
