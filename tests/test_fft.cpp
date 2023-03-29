#include "fft.hpp"
#include "test_pcx.hpp"

#include <iostream>

void test_que(pcx::fft_unit<float>& unit, std::vector<std::complex<float>>& v1) {
    unit.unsorted(v1);
}

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
    return exp(std::complex<T>(0, -2 * pi * static_cast<double>(k) / static_cast<double>(n)));
}
template<typename T, typename Allocator, std::size_t PackSize>
auto fft(const pcx::vector<T, Allocator, PackSize>& vector) {
    using vector_t = pcx::vector<T, Allocator, PackSize>;
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
auto fftu(const pcx::vector<T, Allocator, PackSize>& vector) {
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
auto ifftu(const pcx::vector<T, Allocator, PackSize>& vector) {
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
        a = a.value() / float(size);
    }
    return u;
}

template<std::size_t PackSize = 8>
int test_fft_float(std::size_t size) {
    constexpr float  pi  = 3.14159265358979323846;
    constexpr double dpi = 3.14159265358979323846;

    auto depth = log2i(size);

    auto vec      = pcx::vector<float, std::allocator<float>, PackSize>(size);
    auto vec_out  = pcx::vector<float, std::allocator<float>, PackSize>(size);
    auto svec_out = std::vector<std::complex<float>>(size);
    for (uint i = 0; i < size; ++i) {
        vec[i]      = std::exp(std::complex(0.F, 2 * pi * i / size * 13.37F));
        svec_out[i] = vec[i];
    }

    auto ff = fft(vec);

    for (std::size_t sub_size = 64; sub_size <= size * 2; sub_size *= 2) {
        auto unit = pcx::fft_unit<float, pcx::dynamic_size, pcx::dynamic_size>(size, sub_size);

        vec_out = vec;

        unit(vec_out);
        for (uint i = 0; i < size; ++i) {
            auto val = std::complex<float>(ff[i].value());
            if (!equal_eps(val, vec_out[i].value(), 1U << (depth))) {
                std::cout << "vec  " << size << ":" << sub_size << " #" << i << ": "
                          << abs(val - vec_out[i].value()) << "  " << val << vec_out[i].value() << "\n";
                return 1;
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

    auto vec      = pcx::vector<float, std::allocator<float>, PackSize>(size);
    auto vec_out  = pcx::vector<float, std::allocator<float>, PackSize>(size);
    auto svec_out = std::vector<std::complex<float>>(size);
    for (uint i = 0; i < size; ++i) {
        vec[i]      = std::exp(std::complex(0.F, 2 * pi * i / size * 13.37F));
        svec_out[i] = vec[i];
    }

    for (std::size_t sub_size = 64; sub_size <= size * 2; sub_size *= 2) {
        vec_out = vec;

        auto unit = pcx::fft_unit<float, pcx::dynamic_size, pcx::dynamic_size>(size, sub_size);

        auto ffu = fftu(vec);

        vec_out = vec;
        unit.unsorted(vec_out);
        for (uint i = 0; i < size; ++i) {
            auto val = std::complex<float>(ffu[i].value());
            if (!equal_eps(val, vec_out[i].value(), 1U)) {
                std::cout << PackSize << " fftu " << size << ":" << sub_size << " #" << i << ": "
                          << abs(val - vec_out[i].value()) << "  " << val << vec_out[i].value() << "\n";
                return 1;
            }
        }
        for (uint i = 0; i < size; ++i) {
            svec_out[i] = vec[i];
        }
        unit.unsorted(svec_out);
        for (uint i = 0; i < size; ++i) {
            auto val = std::complex<float>(ffu[i].value());
            if (!equal_eps(val, svec_out[i], 4U)) {
                std::cout << PackSize << " fftu svec " << size << ":" << sub_size << " #" << i << ": "
                          << abs(val - svec_out[i]) << "  " << val << svec_out[i] << "\n";
                return 1;
            }
        }
    }
    return 0;
}

int main() {
    int ret = 0;

    for (uint i = 6; i < 16; ++i) {
        std::cout << (1U << i) << "\n";
        // ret += test_fft_float4(1U << i);
        ret += test_fft_float(1U << i);
        // ret += test_fft_float<1024>(1U << i);
        ret += test_fftu_float(1U << i);
        // ret += test_fftu_float<1024>(1U << i);
        if (ret > 0) {
            return ret;
        }
    }

    // ret += test_fft_float4(64*2);

    //
    constexpr std::size_t size = 64;
    constexpr float       pi   = 3.14159265358979323846;

    auto vec  = pcx::vector<float>(size);
    auto vec2 = pcx::vector<float>(size);
    for (uint i = 0; i < size; ++i) {
        vec[i] = std::exp(std::complex(0.F, 2 * pi * i / size * 1.F));
    }
    vec2 = vec;

    auto v3 = fftu(vec);
    auto v4 = ifftu(v3);
    // for (uint i = 0; i < size; ++i) {
    //     std::cout << abs(v4[i].value() - vec[i].value()) << " " << v4[i].value() << "  " << vec[i].value()
    //               << "\n";
    // }


    //     //
    //     //     auto vec3 = fft(vec);
    //     //     unit.binary4(vec);
    //     //
    //     //     for (uint i = 0; i < size; ++i)
    //     //     {
    //     //         std::cout << "#" << i << " " << (vec3[i].value()) << "  " << (vec[i].value())
    //     //                   << "  " << abs(vec[i].value() - vec3[i].value()) << "\n";
    //     //     }

    //     auto til = std::vector<std::complex<float>>(8 * 4);
    //
    //     for (uint i = 0; i < 32; ++i)
    //     {
    //         til.at(i) = std::complex<float>(i, 100 + i);
    //     }
    //
    //     unit.test_itrlv(reinterpret_cast<float*>(til.data()));
    //
    //     std::cout << til[0] << " " << til[1];

    return 0;
}