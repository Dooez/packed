#include "fft.hpp"
#include "test_pcx.hpp"
#include "vector.hpp"

#include <iostream>
#include <utility>

// NOLINTBEGIN

void test_que(pcx::fft_unit<float, pcx::fft_ordering::unordered>& unit,
              std::vector<std::complex<float>>&                   v1) {
    unit(v1);
}

// void test_que(pcx::fft_unit<float, 8192, 512>& unit, pcx::vector<float>& v1) {
//     unit.subtransform<1, 8>(v1.data(), v1.size());
// }

// void test_que(pcx::fft_unit<float, 8192, 512>&       unit,
//               float*                                 data,
//               std::size_t                            l_size,
//               std::size_t                            offset,
//               std::array<pcx::avx::cx_reg<float>, 7> tw,
//               pcx::avx::reg_t<float>                 scaling) {
//     unit.node8_along<8, 8, false, true, false>(data, l_size, offset, tw, scaling);
// }


// void test_que(pcx::fft_unit<float, 8192, 512>& unit,
//               float*                           data,
//               std::size_t                      l_size,
//               std::size_t                      offset,
//               pcx::avx::cx_reg<float>          tw0,
//               pcx::avx::cx_reg<float>          tw1,
//               pcx::avx::cx_reg<float>          tw2,
//               pcx::avx::cx_reg<float>          tw3,
//               pcx::avx::cx_reg<float>          tw4,
//               pcx::avx::cx_reg<float>          tw5,
//               pcx::avx::cx_reg<float>          tw6) {
//     unit.node8_dit<8,8,true>(data, l_size, offset, tw0, tw1, tw2, tw3, tw4, tw5, tw6);
// }
template<typename T>
auto fmul(std::complex<T> lhs, std::complex<T> rhs) -> std::complex<T> {
    auto lhsv = pcx::avx::broadcast(lhs);
    auto rhsv = pcx::avx::broadcast(rhs);

    auto resv = pcx::avx::mul(lhsv, rhsv);

    T re;
    T im;

    _mm_store_ss(&re, _mm256_castps256_ps128(resv.real));
    _mm_store_ss(&im, _mm256_castps256_ps128(resv.imag));

    return {re, im};
}

constexpr auto log2i(std::size_t num) -> std::size_t {
    std::size_t order = 0;
    while ((num >>= 1U) != 0) {
        order++;
    }
    return order;
}

constexpr auto reverse_bit_order(uint64_t num, uint64_t depth) -> uint64_t {
    num = num >> 32 | num << 32;
    num = (num & 0xFFFF0000FFFF0000) >> 16 | (num & 0x0000FFFF0000FFFF) << 16;
    num = (num & 0xFF00FF00FF00FF00) >> 8 | (num & 0x00FF00FF00FF00FF) << 8;
    num = (num & 0xF0F0F0F0F0F0F0F0) >> 4 | (num & 0x0F0F0F0F0F0F0F0F) << 4;
    num = (num & 0xCCCCCCCCCCCCCCCC) >> 2 | (num & 0x3333333333333333) << 2;
    num = (num & 0xAAAAAAAAAAAAAAAA) >> 1 | (num & 0x5555555555555555) << 1;
    return num >> (64 - depth);
}

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
template<typename T, typename Allocator, std::size_t PackSize>
auto fft(const pcx::vector<T, PackSize, Allocator>& vector) {
    using vector_t = pcx::vector<T, PackSize, Allocator>;
    auto fft_size  = vector.size();
    auto res       = vector_t(fft_size);
    if (fft_size == 1) {
        res = vector;
        return res;
    }

    auto even = vector_t(fft_size / 2);
    auto odd  = vector_t(fft_size / 2);

    for (uint i = 0; i < fft_size / 2; ++i) {
        even[i] = vector[i * 2];
        odd[i]  = vector[i * 2 + 1];
    }

    even = fft(even);
    odd  = fft(odd);

    for (uint i = 0; i < fft_size / 2; ++i) {
        auto even_v = even[i].value();
        auto odd_v  = fmul(odd[i].value(), wnk<T>(fft_size, i));

        res[i]                = even_v + odd_v;
        res[i + fft_size / 2] = even_v - odd_v;

        // res[i]                = wnk<T>(fft_size, i);
        // res[i + fft_size / 2] = wnk<T>(fft_size, i);
    }

    return res;
}


template<typename T, typename Allocator, std::size_t PackSize>
auto fftu(const pcx::vector<T, PackSize, Allocator>& vector) {
    auto        size       = vector.size();
    auto        u          = vector;
    std::size_t n_groups   = 1;
    std::size_t group_size = size / 2;
    std::size_t l_size     = 2;

    while (l_size <= size) {
        for (uint i_group = 0; i_group < n_groups; ++i_group) {
            auto group_offset = i_group * group_size * 2;
            auto w            = wnk<T>(l_size, reverse_bit_order(i_group, log2i(l_size / 2)));
            // std::cout << "tw: " << w << "\n";

            for (uint i = 0; i < group_size; ++i) {
                auto p0 = u[i + group_offset].value();
                auto p1 = u[i + group_offset + group_size].value();

                auto p1tw = fmul(p1, w);

                u[i + group_offset]              = p0 + p1tw;
                u[i + group_offset + group_size] = p0 - p1tw;
                // u[i + group_offset]              = w;
                // u[i + group_offset + group_size] = w;
            }
        }
        l_size *= 2;
        n_groups *= 2;
        group_size /= 2;
    }
    return u;
}

template<typename T, typename Allocator, std::size_t PackSize>
auto ifftu(const pcx::vector<T, PackSize, Allocator>& vector) {
    auto        size       = vector.size();
    auto        u          = vector;
    std::size_t n_groups   = size / 2;
    std::size_t group_size = 1;
    std::size_t l_size     = size;

    while (l_size >= 2) {
        for (uint i_group = 0; i_group < n_groups; ++i_group) {
            auto group_offset = i_group * group_size * 2;
            auto w            = std::conj(wnk<T>(l_size, reverse_bit_order(i_group, log2i(l_size / 2))));
            // w                 = {1, 0};
            for (uint i = 0; i < group_size; ++i) {
                auto p0 = u[i + group_offset].value();
                auto p1 = u[i + group_offset + group_size].value();

                auto a1tw = p0 - p1;
                auto a0   = p0 + p1;

                auto a1 = fmul(a1tw, w);

                u[i + group_offset]              = a0;
                u[i + group_offset + group_size] = a1;
            }
        }
        l_size /= 2;
        n_groups /= 2;
        group_size *= 2;
    }
    for (auto a: u) {
        a = a.value() / static_cast<float>(size);
    }
    return u;
}

template<std::size_t PackSize = 8>
int test_fft_float(std::size_t size) {
    constexpr float  pi  = 3.14159265358979323846;
    constexpr double dpi = 3.14159265358979323846;

    auto depth = log2i(size);

    auto vec      = pcx::vector<float, PackSize, std::allocator<float>>(size);
    auto vec_out  = pcx::vector<float, PackSize, std::allocator<float>>(size);
    auto svec_out = std::vector<std::complex<float>>(size);
    for (uint i = 0; i < size; ++i) {
        vec[i]      = std::exp(std::complex(0.F, 2 * pi * i / size * 13.37F));
        svec_out[i] = vec[i];
    }

    auto ff = fft(vec);

    for (std::size_t sub_size = 64; sub_size <= size * 4; sub_size *= 2) {
        auto unit = pcx::fft_unit<float, pcx::fft_ordering::normal>(size, sub_size);

        vec_out = vec;

        unit(vec_out);
        int ret = 0;
        for (uint i = 0; i < size; ++i) {
            auto val = std::complex<float>(ff[i].value());
            if (!equal_eps(val, vec_out[i].value(), 1U << (depth))) {
                std::cout << "vec  " << size << ":" << sub_size << " #" << i << ": "
                          << abs(val - vec_out[i].value()) << "  " << val << vec_out[i].value() << "\n";
                ++ret;
            }
            if (ret > 31) {
                return ret;
            }
        }


        unit.ifft(vec_out);
        for (uint i = 0; i < size; ++i) {
            auto val = std::complex<float>(vec[i].value());
            if (!equal_eps(val, vec_out[i].value(), 1U << (depth))) {
                std::cout << "ifftvec  " << size << ":" << sub_size << " #" << i << ": "
                          << abs(val - vec_out[i].value()) << "  " << val << vec_out[i].value() << "\n";
                return 1;
            }
        }

        vec_out = vec;
        unit(vec_out, vec);
        for (uint i = 0; i < size; ++i) {
            auto val = std::complex<float>(ff[i].value());
            if (!equal_eps(val, vec_out[i].value(), 1U << (depth))) {
                std::cout << "veco " << size << ":" << sub_size << " #" << i << ": "
                          << abs(val - vec_out[i].value()) << "  " << val << vec_out[i].value() << "\n";
                return 1;
            }
        }

        for (uint i = 0; i < size; ++i) {
            svec_out[i] = vec[i];
        }

        unit(svec_out);
        for (uint i = 0; i < size; ++i) {
            auto val = std::complex<float>(ff[i].value());
            if (!equal_eps(val, svec_out[i], 1U << (depth))) {
                std::cout << "svec " << size << ":" << sub_size << " #" << i << ": " << abs(val - svec_out[i])
                          << "  " << val << svec_out[i] << "\n";
                return 1;
            }
        }

        unit.ifft(svec_out);
        for (uint i = 0; i < size; ++i) {
            auto val = std::complex<float>(vec[i].value());
            if (!equal_eps(val, svec_out[i], 1U << (depth))) {
                std::cout << "ifft svec " << size << ":" << sub_size << " #" << i << ": "
                          << abs(val - svec_out[i]) << "  " << val << svec_out[i] << "\n";
                return 1;
            }
        }

        for (uint i = 0; i < size; ++i) {
            svec_out[i] = vec[i];
        }
        auto svec2 = svec_out;
        unit(svec_out, svec2);
        for (uint i = 0; i < size; ++i) {
            auto val = std::complex<float>(ff[i].value());
            if (!equal_eps(val, svec_out[i], 1U << (depth))) {
                std::cout << "svec " << size << ":" << sub_size << " #" << i << ": " << abs(val - svec_out[i])
                          << "  " << val << svec_out[i] << "\n";
                return 1;
            }
        }
    }

    return 0;
}

template<std::size_t PackSize = 8>
int test_fftu_float(std::size_t size) {
    constexpr float pi = 3.14159265358979323846;

    auto depth = log2i(size);

    auto vec      = pcx::vector<float, PackSize, std::allocator<float>>(size);
    auto vec_out  = pcx::vector<float, PackSize, std::allocator<float>>(size);
    auto svec_out = std::vector<std::complex<float>>(size);
    for (uint i = 0; i < size; ++i) {
        vec[i]      = std::exp(std::complex(0.F, 2 * pi * i / size * 13.37F));
        svec_out[i] = vec[i];
    }

    for (std::size_t sub_size = 64; sub_size <= size; sub_size *= 2) {
        vec_out = vec;

        auto unit   = pcx::fft_unit<float, pcx::fft_ordering::bit_reversed>(size, sub_size);
        auto unit_u = pcx::fft_unit<float, pcx::fft_ordering::unordered>(size, sub_size);

        auto ffu   = fftu(vec);
        auto eps_u = 1U << (depth - 1);
        vec_out    = vec;
        unit(vec_out);
        int ret = 0;
        for (uint i = 0; i < size; ++i) {
            auto val = std::complex<float>(ffu[i].value());
            if (!equal_eps(val, vec_out[i].value(), eps_u)) {
                std::cout << PackSize << " fftu " << size << ":" << sub_size << " #" << i << ": "
                          << abs(val - vec_out[i].value()) << "  " << val << vec_out[i].value() << "\n";
                ++ret;
            }
            if (ret > 16) {
                return ret;
            }
        }
        if (ret != 0) {
            return ret;
        }

        ret = 0;
        unit.ifftu_internal<PackSize>(vec_out.data());
        for (uint i = 0; i < size; ++i) {
            auto val = std::complex<float>(vec[i].value());
            if (!equal_eps(val, vec_out[i].value(), 1U << (depth))) {
                std::cout << "ifftvec  " << size << ":" << sub_size << " #" << i << ": "
                          << abs(val - vec_out[i].value()) << "  " << val << vec_out[i].value() << "\n";
                ret++;
            }
            if (ret > 16) {
                return ret;
            }
        }
        if (ret != 0) {
            return ret;
        }

        for (uint i = 0; i < size; ++i) {
            svec_out[i] = vec[i];
        }
        unit(svec_out);
        for (uint i = 0; i < size; ++i) {
            auto val = std::complex<float>(ffu[i].value());
            if (!equal_eps(val, svec_out[i], eps_u)) {
                std::cout << PackSize << " fftu svec " << size << ":" << sub_size << " #" << i << ": "
                          << abs(val - svec_out[i]) << "  " << val << svec_out[i] << "\n";
                return 1;
            }
        }


        vec_out = vec;
        unit_u.fftu_internal<PackSize>(vec_out.data());
        unit_u.ifftu_internal<PackSize>(vec_out.data());
        for (uint i = 0; i < size; ++i) {
            auto val = std::complex<float>(vec[i].value());
            if (!equal_eps(val, vec_out[i].value(), 1U << (depth))) {
                std::cout << "ifft vec true  " << size << ":" << sub_size << " #" << i << ": "
                          << abs(val - vec_out[i].value()) << "  " << val << vec_out[i].value() << "\n";
                ret++;
            }
            if (ret > 16) {
                return ret;
            }
        }
        if (ret != 0) {
            return ret;
        }
    }
    return 0;
}
template<std::size_t PackSize = 8>
int test_fftu_float_0(std::size_t size) {
    constexpr float pi = 3.14159265358979323846;

    auto depth = log2i(size);

    auto vec      = pcx::vector<float, PackSize, std::allocator<float>>(size);
    auto vec_out  = pcx::vector<float, PackSize, std::allocator<float>>(size);
    auto svec_out = std::vector<std::complex<float>>(size);
    for (uint i = 0; i < size; ++i) {
        vec[i]      = std::exp(std::complex(0.F, 2 * pi * i / size * 13.37F));
        svec_out[i] = vec[i];
    }

    for (std::size_t sub_size = 64; sub_size <= size; sub_size *= 2) {
        vec_out = vec;

        auto unit = pcx::fft_unit<float, pcx::fft_ordering::bit_reversed>(size, sub_size);

        auto ffu   = fftu(vec);
        auto eps_u = 1U << (depth - 1);

        for (auto n_empty = size / 2; n_empty < std::min(size, size / 2 + pcx::avx::reg<float>::size);
             ++n_empty) {
            auto vec_short = pcx::vector<float, PackSize, std::allocator<float>>(size - n_empty);
            auto vec_zero  = pcx::vector<float, PackSize, std::allocator<float>>(size);

            vec_short = pcx::subrange(vec.begin(), vec_short.size());
            pcx::subrange(vec_zero.begin(), vec_short.size())
                .assign(pcx::subrange(vec.begin(), vec_short.size()));

            for (uint i = 0; i < size; ++i) {
                svec_out[i] = vec_zero[i];
            }

            auto vec_out_zero = vec_zero;
            vec_out           = vec_out_zero;
            unit(vec_out_zero);
            unit(vec_out, vec_short);
            int ret = 0;
            for (uint i = 0; i < size; ++i) {
                auto val = std::complex<float>(vec_out_zero[i].value());
                if (!equal_eps(val, vec_out[i].value(), eps_u)) {
                    std::cout << PackSize << " fftu " << size << ":" << sub_size << " #" << i << ": "
                              << abs(val - vec_out[i].value()) << "  " << val << vec_out[i].value() << "\n";
                    ++ret;
                }
                if (ret > 16) {
                    return ret;
                }
            }
            if (ret != 0) {
                return ret;
            }
        }
    }
    return 0;
}
constexpr float pi = 3.14159265358979323846;

int main() {
    int ret = 0;

    // for (uint i = 6; i < 16; ++i) {
    //     std::cout << (1U << i) << "\n";
    //     // ret += test_ifftu_float(1U << i);
    //     // ret += test_fft_float4(1U << i);
    //     // ret += test_fft_float<1024>(1U << i);
    //     ret += test_fft_float(1U << i);
    //     ret += test_fftu_float(1U << i);
    //     // ret += test_fftu_float_0(1U << i);
    //     // ret += test_fftu_float<1024>(1U << i);
    //     if (ret > 0) {
    //         return ret;
    //     }
    // }

    //     constexpr std::size_t size = 64;
    //
    //     auto vec  = pcx::vector<float>(size);
    //     auto vec2 = pcx::vector<float>(size);
    //
    //     std::cout << pcx::fft_unit_par<float>::test(vec) << "\n";
    //     std::cout << pcx::fft_unit_par<float>::test(std::vector<std::complex<float>>{}) << "\n";
    //     std::cout << pcx::fft_unit_par<float>::test(std::vector<float>{}) << "\n";
    //
    //     static_assert(pcx::complex_vector_of<float, pcx::vector<float>>);
    //     static_assert(pcx::complex_vector_of<float, std::vector<std::complex<float>>>);
    //     // static_assert(pcx::complex_vector_of<float, std::vector<float>>);
    //

    std::size_t par_size  = 4096 ;
    auto        st_par    = std::vector<pcx::vector<float>>(par_size);
    auto        vec_check = pcx::vector<float>(par_size);
    for (uint i = 0; auto& vec: st_par) {
        vec.resize(128);
        auto val = std::exp(std::complex(0.F, 2 * pi * i / par_size * 13.37F));
        pcx::subrange(vec).fill(val);
        vec_check[i] = val;
        ++i;
    }

    pcx::fft_unit_par<float> par_unit(par_size);
    pcx::fft_unit<float>     check_unit(par_size);

    par_unit(st_par, st_par);
    check_unit(vec_check);

    // for (uint i = 0; auto& vec: st_par) {
    //     std::cout << std::to_string(abs(vec[0].value())) << "  " << std::to_string(abs(vec_check[i].value()))
    //               << "  " << std::to_string(abs(vec_check[i].value() - vec[0].value())) << "\n";
    //     if (++i > 32)
    //         break;
    // }
    uint q = 0;
    for (uint i = 0; i < par_size; ++i) {
        auto val = std::complex<float>(st_par[i][0].value());
        if (!equal_eps(val, vec_check[i].value(), 1U << 10)) {
            std::cout << "svec " << par_size << " #" << i << ": " << abs(val - vec_check[i].value())
                      << "  " << val << vec_check[i].value() << "\n";
            ++q;
            if (q >32U){
                return 1;
            }
        }
    }
    return 0;
}

//NOLINTEND