#pragma once

#include "avx2_common.hpp"
#include "fft.hpp"
#include "simd_common.hpp"
#include "simd_fft.hpp"
#include "tuple_util.hpp"
#include "types.hpp"

namespace pcxo::detail_::fft {

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

template<uZ NodeSize>
struct newnode {
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

    struct settings {
        uZ   pack_dest;
        uZ   pack_src;
        bool conj_tw;
        bool dit;
    };

    template<typename T, settings Settings, typename... Args>
    static inline void perform(std::array<T*, NodeSize> dest, Args&&... args) {
        constexpr auto reg_size = simd::reg<T>::size;
        // using cx_reg            = simd::cx_reg<T, false, reg_size>;

        using src_type     = std::array<const T*, NodeSize>;
        using tw_type      = std::array<simd::cx_reg<T>, NodeSize - 1>;
        constexpr bool Src = has_type<src_type, Args...>;
        // constexpr bool Tw    = has_type<tw_type, Args...>;
        // constexpr bool Scale = has_type<simd::reg_t<T>, Args...>;

        constexpr auto load =
            []<uZ... Is>(std::index_sequence<Is...>, const auto& dest, const auto&... args) {
                constexpr auto& data_idx = order2<NodeSize, Settings.dit>::data;
                if constexpr (Src) {
                    auto& src = std::get<src_type&>(std::tie(args...));
                    return std::make_tuple(simd::cxload<Settings.pack_src, reg_size>(src[data_idx[Is]])...);
                } else {
                    return std::make_tuple(simd::cxload<Settings.pack_src, reg_size>(dest[data_idx[Is]])...);
                }
            };
        constexpr auto store =
            []<uZ... Is>(std::index_sequence<Is...>, const auto& dest, const auto&... data) {
                constexpr auto& data_idx = order2<NodeSize, Settings.dit>::data;
                (simd::cxstore<Settings.pack_dest>(dest[data_idx[Is]], std::get<Is>(data)), ...);
            };


        constexpr auto btfly = []<uZ Level>(uZ_constant<Level>, const auto& data, const auto& tw) {
            constexpr uZ stride = NodeSize / powi(2, Level);

            constexpr auto get_half = []<uZ Start>(uZ_constant<Start>, const auto& data) {
                return []<uZ... Grp>(std::index_sequence<Grp...>, const auto& data) {
                    constexpr auto iterate = []<uZ... Iters, uZ Offset>(std::index_sequence<Iters...>,
                                                                        uZ_constant<Offset>,
                                                                        const auto& data) {
                        return std::make_tuple(std::get<Start + Offset + Iters>(data)...);
                    };
                    return std::tuple_cat(iterate(std::make_index_sequence<stride / 2>{},    //
                                                  uZ_constant<Grp * stride>{},
                                                  data)...);
                }(std::make_index_sequence<NodeSize / stride>{}, data);
            };

            auto bottom = get_half(uZ_constant<stride / 2>{}, data);

            auto tws = []<uZ... Itw>(std::index_sequence<Itw...>, const auto& tw) {
                constexpr auto make_rep =
                    []<uZ... Reps, uZ I>(std::index_sequence<Reps...>, uZ_constant<I>, auto tw) {
                        return std::make_tuple(((void)Reps, tw[powi(2UL, Level) - 1 + I])...);
                    };
                return std::tuple_cat(make_rep(std::make_index_sequence<NodeSize / sizeof...(Itw)>{},    //
                                               uZ_constant<Itw>{},
                                               tw)...);
            }(std::make_index_sequence<powi(2UL, Level)>{}, tw);
            /*auto bottom_tw = std::apply(simd::mul_pairs, zip_tuples(bottom, tws));*/
            auto bottom_tw = simd::mul_tuples(bottom, tws);
            auto top       = get_half(uZ_constant<0>{}, data);

            return mass_invoke(simd::btfly, top, bottom_tw);
        };

        auto p0 = load(std::make_index_sequence<NodeSize>{}, dest, args...);
        /*auto p1 = std::apply(&simd::inverse<Settings.conj_tw>, p0);*/

        auto p1   = p0;
        auto res0 = []<uZ I>(uZ_constant<I>, const auto& data, const auto& tw) {
            if constexpr (powi(2, I + 1) == NodeSize) {
                return btfly(uZ_constant<I>{}, data, tw);
            } else {
                auto tmp = btfly(uZ_constant<I>{}, data, tw);
                return btfly(uZ_constant<I + 1>{}, tmp, tw);
            }
        }(uZ_constant<0>{}, p1, std::get<tw_type&>(std::tie(args...)));
        /*auto res1 = std::apply(&simd::inverse<Settings.conj_tw>, res0);*/
        auto res1 = res0;
        store(dest, res1);
    }
};

}    // namespace pcxo::detail_::fft

namespace pcxo::simd {

inline void tform() {
    // constexpr uZ btfly_size = 4;
    // using pcxo::detail_::mass_invoke;
    // constexpr uZ src_size = 4;
    //
    // constexpr auto load = [](const auto* ptr, uZ offset) { return cxload<src_size>(ptr + offset); };
    //
    // auto* data    = (f32*){nullptr};
    // auto  offsets = []<uZ... Is>(std::index_sequence<Is...>, uZ step) {
    //     return std::make_tuple((step * Is)...);
    // }(std::make_index_sequence<btfly_size>{}, 512);
    //
    // auto rdata = mass_invoke(load, data, offsets);
}
}    // namespace pcxo::simd
