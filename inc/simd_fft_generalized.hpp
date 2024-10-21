#pragma once

#include "avx2_common.hpp"
#include "fft.hpp"
#include "simd_common.hpp"
#include "simd_fft.hpp"
#include "tuple_util.hpp"
#include "types.hpp"


#define PCX_LAINLINE [[gnu::always_inline, clang::always_inline]]
namespace pcx::detail_::fft {

// constexpr auto log2i(u64 num) -> uZ {
//     u64 order = 0;
//     for (u8 shift = 32; shift > 0; shift /= 2) {
//         if (num >> shift > 0) {
//             order += num >> shift > 0 ? shift : 0;
//             num >>= shift;
//         }
//     }
//     return order;
// }
//
// constexpr auto powi(uint64_t num, uint64_t pow) -> uint64_t {
//     auto res = (pow % 2) == 1 ? num : 1UL;
//     if (pow > 1) {
//         auto half_pow = powi(num, pow / 2UL);
//         res *= half_pow * half_pow;
//     }
//     return res;
// }
//
// constexpr auto reverse_bit_order(u64 num, u64 depth) -> u64 {
//     //TODO(Timofey): possibly find better solution to prevent UB
//     if (depth == 0)
//         return 0;
//     num = num >> 32U | num << 32U;
//     // NOLINTBEGIN (*magic-numbers*)
//     num = (num & 0xFFFF0000FFFF0000U) >> 16U | (num & 0x0000FFFF0000FFFFU) << 16U;
//     num = (num & 0xFF00FF00FF00FF00U) >> 8U | (num & 0x00FF00FF00FF00FFU) << 8U;
//     num = (num & 0xF0F0F0F0F0F0F0F0U) >> 4U | (num & 0x0F0F0F0F0F0F0F0FU) << 4U;
//     num = (num & 0xCCCCCCCCCCCCCCCCU) >> 2U | (num & 0x3333333333333333U) << 2U;
//     num = (num & 0xAAAAAAAAAAAAAAAAU) >> 1U | (num & 0x5555555555555555U) << 1U;
//     // NOLINTEND (*magic-numbers*)
//     return num >> (64 - depth);
// }
//
// /**
//  * @brief Returns number of unique bit-reversed pairs from 0 to max-1
//  */
// constexpr auto n_reversals(uZ max) -> uZ {
//     return max - (1U << ((log2i(max) + 1) / 2));
// }

template<typename T, typename... U>
concept has_type = (std::same_as<T, U> || ...);

template<uZ Size, bool DecInTime>
struct order2 {
    static constexpr auto data = [] {
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

    static constexpr auto tw = []<uZ... N>(std::index_sequence<N...>) -> std::array<uZ, sizeof...(N)> {
        if constexpr (DecInTime) {
            return {(N > 0 ? (1U << log2i(N + 1)) - 1 +
                                 reverse_bit_order(1 + N - (1U << log2i(N + 1)), log2i(N + 1))
                           : 0)...};
        } else {
            return {N...};
        }
    }
    (std::make_index_sequence<Size - 1>{});
};
static constexpr auto next_pow_2(u64 v) {
    v--;
    v |= v >> 1U;
    v |= v >> 2U;
    v |= v >> 4U;
    v |= v >> 8U;
    v |= v >> 16U;
    v |= v >> 32U;
    v++;
    return v;
}


/**
 * @brief Twiddles of constant bit-reversed indexes.
 * Indexes are bit-reversed, and thus independent on the actual transform size e.g.
 * if twiddle is defined as tw = exp(-2 * pi * i * k / N)
 * `ITw == 0` => k == 0
 * `ITw == 1` => k == N/2
 * `ITw == 2` => k == N/4
 * `ITw == 3` => k == 3N/4
 */
template<typename T, uZ BitRevI>
struct const_tw {
    static constexpr uZ min_fft_size = next_pow_2(BitRevI);
    static constexpr uZ real_index   = reverse_bit_order(BitRevI, min_fft_size);

    static auto value = static_cast<std::complex<T>>(    //
        std::exp(                                        //
            std::complex<double>{0,                      //
                                 2. * std::numbers::pi* static_cast<double>(real_index) /
                                     static_cast<double>(min_fft_size)}));
};

template<uZ NodeSize, typename T>
struct newnode {
    struct settings {
        uZ   pack_dest;
        uZ   pack_src;
        bool conj_tw;
        bool dit;
    };

    using full_tw_t   = std::array<simd::cx_reg<T>, NodeSize - 1>;
    using pruned_tw_t = std::conditional_t<(NodeSize > 4),    //
                                           std::array<simd::cx_reg<T>, NodeSize / 4 - 1>,
                                           decltype([] {})>;
    using dest_t      = std::array<T*, NodeSize>;


    template<settings Settings>
    static inline void perform(const dest_t& dest, const pruned_tw_t& tw) {
        auto data_tup = load<Settings>(dest);
    };

    template<settings Settings>
    static inline void perform(const dest_t&                         dest,
                               const std::array<const T*, NodeSize>& src,
                               const full_tw_t&                      tw) {
        auto data_tup = load<Settings>(src);
        /*auto p1 = std::apply(&simd::inverse<Settings.conj_tw>, p0);*/

        auto p1 = data_tup;

        auto res = []<uZ L = 0>(this auto&& f, const auto& data, const auto& tw, uZ_constant<L> = {}) {
            if constexpr (powi(2, L + 1) == NodeSize) {
                return btfly<L>(data, tw);
            } else {
                auto tmp = btfly<L>(data, tw);
                return f(tmp, tw, uZ_constant<L + 1>{});
            }
        }(p1, tw);

        /*auto res1 = std::apply(&simd::inverse<Settings.conj_tw>, res0);*/
        store<Settings>(dest, res);
    }

    template<settings Settings>
    static inline void perform(std::array<T*, NodeSize>                         dest,
                               const std::array<simd::cx_reg<T>, NodeSize - 1>& tw) {
        auto data_tup = load<Settings>(dest);
        /*auto p1 = std::apply(&simd::inverse<Settings.conj_tw>, p0);*/

        auto p1 = data_tup;

        auto res = []<uZ I>(this auto&& f, uZ_constant<I>, const auto& data, const auto& tw) {
            if constexpr (powi(2, I + 1) == NodeSize) {
                return btfly<I>(data, tw);
            } else {
                auto tmp = btfly<I>(data, tw);
                return f(uZ_constant<I + 1>{}, tmp, tw);
            }
        }(uZ_constant<0>{}, p1, tw);

        /*auto res1 = std::apply(&simd::inverse<Settings.conj_tw>, res0);*/
        store<Settings>(dest, res);
    }

private:
    template<settings Settings, typename U>
    static auto load(const std::array<U*, NodeSize>& data) {
        constexpr auto reg_size = simd::reg<T>::size;
        return []<uZ... Is>(std::index_sequence<Is...>, const auto& data) {
            constexpr auto& data_idx = order2<NodeSize, Settings.dit>::data;
            return std::make_tuple(simd::cxload<Settings.pack_src, reg_size>(data[data_idx[Is]])...);
        }(std::make_index_sequence<NodeSize>{}, data);
    }

    template<settings Settings>
    static void store(const std::array<T*, NodeSize>& dest, const auto& data) {
        []<uZ... Is>(std::index_sequence<Is...>, const auto& dest, const auto& data) {
            constexpr auto& data_idx = order2<NodeSize, Settings.dit>::data;
            (simd::cxstore<Settings.pack_dest>(dest[data_idx[Is]], std::get<Is>(data)), ...);
        }(std::make_index_sequence<NodeSize>{}, dest, data);
    }

    template<uZ Level, bool Top, typename... Ts>
    static auto get_half(const std::tuple<Ts...>& data) {
        constexpr uZ size   = sizeof...(Ts);
        constexpr uZ stride = size / powi(2, Level);
        constexpr uZ start  = Top ? 0 : stride / 2;

        return []<uZ... Grp>(std::index_sequence<Grp...>, const auto& data) {
            constexpr auto iterate = []<uZ... Iters, uZ Offset>(std::index_sequence<Iters...>,
                                                                uZ_constant<Offset>,
                                                                const auto& data) {
                return std::make_tuple(std::get<start + Offset + Iters>(data)...);
            };
            return std::tuple_cat(iterate(std::make_index_sequence<stride / 2>{},    //
                                          uZ_constant<Grp * stride>{},
                                          data)...);
        }(std::make_index_sequence<size / stride>{}, data);
    }
    template<uZ Level>
    static auto get_top_half(const auto& data) {
        return get_half<Level, true>(data);
    }
    template<uZ Level>
    static auto get_bot_half(const auto& data) {
        return get_half<Level, false>(data);
    }

    template<uZ Level>
    static auto btfly(const auto& data, const auto& tw) {
        auto bottom = get_bot_half<Level>(data);

        auto tws = []<uZ... Itw>(std::index_sequence<Itw...>, const auto& tw) {
            constexpr auto make_rep =
                []<uZ... Reps, uZ I>(std::index_sequence<Reps...>, uZ_constant<I>, auto tw) {
                    return std::make_tuple(((void)Reps, tw[powi(2UL, Level) - 1 + I])...);
                };
            return std::tuple_cat(make_rep(std::make_index_sequence<NodeSize / 2 / sizeof...(Itw)>{},    //
                                           uZ_constant<Itw>{},
                                           tw)...);
        }(std::make_index_sequence<powi(2UL, Level)>{}, tw);

        auto bottom_tw = simd::mul_tuples(bottom, tws);
        auto top       = get_top_half<Level>(data);

        return detail_::make_flat_tuple(mass_invoke(simd::btfly, top, bottom_tw));
    };


    static constexpr auto next_pow_2(u64 v) {
        v--;
        v |= v >> 1U;
        v |= v >> 2U;
        v |= v >> 4U;
        v |= v >> 8U;
        v |= v >> 16U;
        v |= v >> 32U;
        v++;
        return v;
    }

    /**
     * @brief Provides butterfly operations for a compile time constant indexes.
     * Indexes are bit-reversed, and thus independent on the actual transform size e.g.
     * if twiddle is defined as tw = exp(-2 * pi * i * k / N)
     * `ITw == 0` => k == 0
     * `ITw == 1` => k == N/2
     * `ITw == 2` => k == N/4
     * `ITw == 3` => k == 3N/4
     * ...
     * Low index butterflies are optimized. 
     */
    template<uZ ITw>
    struct const_btfly_impl {
        template<uZ Offset, uZ... Is>
        static auto step0(const auto& /*top*/,
                          const auto& bottom,    //
                          uZ_constant<Offset>,
                          std::index_sequence<Is...>) {
            auto tw = simd::broadcast(&const_tw<T, ITw>::value);
            return std::make_tuple(simd::detail_::mul_real_rhs(std::get<Offset + Is>(bottom), tw)...);
        }
        template<uZ Offset, uZ... Is>
        static auto step1(const auto& /*top*/,    //
                          const auto& bottom,
                          const auto& res0,
                          uZ_constant<Offset>,
                          std::index_sequence<Is...>) {
            auto tw = simd::broadcast(&const_tw<T, ITw>::value);
            return std::make_tuple(simd::detail_::mul_imag_rhs(std::get<Offset + Is>(res0),    //
                                                               std::get<Offset + Is>(bottom),
                                                               tw)...);
        }
        template<uZ Offset, uZ... Is>
        static auto step2(const auto& top,    //
                          const auto& /*bottom*/,
                          const auto& res1,
                          uZ_constant<Offset>,
                          std::index_sequence<Is...>) {
            std::make_tuple(simd::btfly(std::get<Offset + Is>(top),    //
                                        std::get<Offset + Is>(res1))...);
        }
    };

    template<>
    struct const_btfly_impl<0> {
        template<uZ Offset, uZ... Is>
        static auto step0(const auto& /*top*/,    //
                          const auto& /*bottom*/,
                          uZ_constant<Offset>,
                          std::index_sequence<Is...>) {
            return [] {};
        }
        template<uZ Offset, uZ... Is>
        static auto step1(const auto& /*top*/,    //
                          const auto& /*bottom*/,
                          const auto& /*res0*/,
                          uZ_constant<Offset>,
                          std::index_sequence<Is...>) {
            return [] {};
        }
        template<uZ Offset, uZ... Is>
        static auto step2(const auto& top,    //
                          const auto& bottom,
                          const auto& /*res1*/,
                          uZ_constant<Offset>,
                          std::index_sequence<Is...>) {
            std::make_tuple(simd::btfly(std::get<Offset + Is>(top),    //
                                        std::get<Offset + Is>(bottom))...);
        }
    };
    template<>
    struct const_btfly_impl<1> {
        template<uZ Offset, uZ... Is>
        static auto step0(const auto& /*top*/,    //
                          const auto& /*bottom*/,
                          uZ_constant<Offset>,
                          std::index_sequence<Is...>) {
            return [] {};
        }
        template<uZ Offset, uZ... Is>
        static auto step1(const auto& /*top*/,    //
                          const auto& /*bottom*/,
                          const auto& /*res0*/,
                          uZ_constant<Offset>,
                          std::index_sequence<Is...>) {
            return [] {};
        }
        template<uZ Offset, uZ... Is>
        static auto step2(const auto& top,    //
                          const auto& bottom,
                          const auto& /*res1*/,
                          uZ_constant<Offset>,
                          std::index_sequence<Is...>) {
            std::make_tuple(simd::btfly_t<3>{}(std::get<Offset + Is>(top),    //
                                               std::get<Offset + Is>(bottom))...);
        }
    };
    static inline auto sq2 = static_cast<T>(std::sqrt(2.));

    template<uZ Level>
    static auto const_btfly(const auto& data) {
        constexpr uZ stride = NodeSize / powi(2, Level);

        constexpr auto idxs = std::make_index_sequence<stride / 2>{};

        constexpr auto step0 = []<uZ... Is>(const auto& top, const auto& bottom, std::index_sequence<Is...>) {
            return detail_::make_flat_tuple(const_btfly_impl<Is>::step0(top,    //
                                                                        bottom,
                                                                        idxs,
                                                                        Is * stride)...);
        };
        constexpr auto step1 =
            []<uZ... Is>(const auto& top, const auto& bottom, const auto& res0, std::index_sequence<Is...>) {
                return detail_::make_flat_tuple(const_btfly_impl<Is>::step1(top,    //
                                                                            bottom,
                                                                            res0,
                                                                            idxs,
                                                                            Is * stride)...);
            };
        constexpr auto step2 =
            []<uZ... Is>(const auto& top, const auto& bottom, const auto& res1, std::index_sequence<Is...>) {
                return detail_::make_flat_tuple(const_btfly_impl<Is>::step2(top,    //
                                                                            bottom,
                                                                            res1,
                                                                            idxs,
                                                                            Is * stride)...);
            };
        constexpr auto twidxs = std::make_index_sequence<powi(2, Level)>{};

        auto top    = get_top_half<Level>(data);
        auto bottom = get_bot_half<Level>(data);
        auto res0   = step0(top, bottom, twidxs);
        auto res1   = step1(top, bottom, res0, twidxs);
        return step2(top, bottom, res1, twidxs);
    };
};

}    // namespace pcx::detail_::fft

namespace pcx::simd {

inline void tform() {
    constexpr uZ btfly_size = 4;
    using pcx::detail_::mass_invoke;
    constexpr uZ src_size = 4;

    constexpr auto load = [](const auto* ptr, uZ offset) { return cxload<src_size>(ptr + offset); };

    auto* data    = (f32*){nullptr};
    auto  offsets = []<uZ... Is>(std::index_sequence<Is...>, uZ step) {
        return std::make_tuple((step * Is)...);
    }(std::make_index_sequence<btfly_size>{}, 512);

    auto rdata = mass_invoke(load, data, offsets);
}
}    // namespace pcx::simd

#undef PCX_LAINLINE
