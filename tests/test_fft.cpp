#include "fft.hpp"
#include "test_pcx.hpp"

#include <iostream>

void test(pcx::fft_unit<float>& unit, float* v1, float* v2)
{
    unit.fft_internal(v1, v2);
}

constexpr auto log2i(std::size_t num) -> std::size_t
{
    std::size_t order = 0;
    while ((num >>= 1U) != 0)
    {
        order++;
    }
    return order;
}

constexpr auto reverse_bit_order(uint64_t num, uint64_t depth) -> uint64_t
{
    num = num >> 32 | num << 32;
    num = (num & 0xFFFF0000FFFF0000) >> 16 | (num & 0x0000FFFF0000FFFF) << 16;
    num = (num & 0xFF00FF00FF00FF00) >> 8 | (num & 0x00FF00FF00FF00FF) << 8;
    num = (num & 0xF0F0F0F0F0F0F0F0) >> 4 | (num & 0x0F0F0F0F0F0F0F0F) << 4;
    num = (num & 0xCCCCCCCCCCCCCCCC) >> 2 | (num & 0x3333333333333333) << 2;
    num = (num & 0xAAAAAAAAAAAAAAAA) >> 1 | (num & 0x5555555555555555) << 1;
    return num >> (64 - depth);
}

template<typename T>
inline auto wnk(std::size_t n, std::size_t k) -> std::complex<T>
{
    constexpr double pi = 3.14159265358979323846;
    return exp(
        std::complex<T>(0, -2 * pi * static_cast<double>(k) / static_cast<double>(n)));
}
template<typename T>
auto fft(const pcx::vector<T>& vector)
{
    auto fft_size = vector.size();
    auto res      = pcx::vector<T>(fft_size);
    if (fft_size == 1)
    {
        res = vector;
        return res;
    }

    auto even = pcx::vector<T>(fft_size / 2);
    auto odd  = pcx::vector<T>(fft_size / 2);

    for (uint i = 0; i < fft_size / 2; ++i)
    {
        even[i] = vector[i * 2];
        odd[i]  = vector[i * 2 + 1];
    }

    even = fft(even);
    odd  = fft(odd);

    for (uint i = 0; i < fft_size / 2; ++i)
    {
        auto even_v = even[i].value();
        auto odd_v  = odd[i].value() * wnk<T>(fft_size, i);

        res[i]                = even_v + odd_v;
        res[i + fft_size / 2] = even_v - odd_v;
    }

    return res;
}


template<typename T>
auto fftu(const pcx::vector<T>& vector)
{
    auto        size       = vector.size();
    auto        u          = vector;
    std::size_t n_groups   = 1;
    std::size_t group_size = size / 2;
    std::size_t l_size     = 2;

    // while (l_size <= size)
    while (l_size <= size)
    {
        for (uint i_group = 0; i_group < n_groups; ++i_group)
        {
            auto group_offset = i_group * group_size * 2;
            auto w = wnk<T>(l_size, reverse_bit_order(i_group, log2i(l_size / 2)));

            for (uint i = 0; i < group_size; ++i)
            {
                auto p0 = u[i + group_offset].value();
                auto p1 = u[i + group_offset + group_size].value();

                auto p1tw = p1 * w;

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

int test_fft_float(std::size_t size)
{
    constexpr float  pi  = 3.14159265358979323846;
    constexpr double dpi = 3.14159265358979323846;

    auto depth = log2i(size);

    auto vec     = pcx::vector<float>(size);
    auto vec_out = pcx::vector<float>(size);
    for (uint i = 0; i < size; ++i)
    {
        vec[i] = std::exp(std::complex(0.F, 2 * pi * i / size * 13.37F));
        // vec[i] = 0;
    }

    auto ff   = fft(vec);
    auto unit = pcx::fft_unit<float, pcx::dynamic_size, 2048>(size);
    unit(vec_out, vec);
    for (uint i = 0; i < size; ++i)
    {
        auto val = std::complex<float>(ff[i].value());
        if (!equal_eps(val, vec_out[i].value(), 1U << (depth)))
        {
            std::cout << size << " #" << i << ": " << abs(val - vec_out[i].value())
                      << "  " << val << vec_out[i].value() << "\n";
            return 1;
        }
    }
    vec_out = vec;
    unit(vec_out);
    for (uint i = 0; i < size; ++i)
    {
        auto val = std::complex<float>(ff[i].value());
        if (!equal_eps(val, vec_out[i].value(), 1U << (depth)))
        {
            std::cout << size << " #" << i << ": " << abs(val - vec_out[i].value())
                      << "  " << val << vec_out[i].value() << "\n";
            return 1;
        }
    }
    vec_out = vec;

    unit.binary(vec_out);
    for (uint i = 0; i < size; ++i)
    {
        auto val = std::complex<float>(ff[i].value());
        if (!equal_eps(val, vec_out[i].value(), 1U << (depth)))
        {
            std::cout << size << " #" << i << ": " << abs(val - vec_out[i].value())
                      << "  " << val << vec_out[i].value() << "\n";
            return 1;
        }
    }
    return 0;
}


int main()
{
    //     int ret = 0;
    //     for (uint i = 20; i > 5; --i)
    //     {
    //         std::cout << (1U << i) << "\n";
    //
    //         ret += test_fft_float(1U << i);
    //         if (ret > 0)
    //         {
    //             return ret;
    //         }
    //     }

    constexpr std::size_t size = 128;
    constexpr float       pi   = 3.14159265358979323846;

    auto vec  = pcx::vector<float>(size);
    auto vec2 = pcx::vector<float>(size);
    for (uint i = 0; i < size; ++i)
    {
        vec[i] = std::exp(std::complex(0.F, 2 * pi * i / size * 1.F));
    }

    auto unit = pcx::fft_unit<float, pcx::dynamic_size, size>(size);

    auto vec3 = fftu(vec);
    unit.unsorted(vec);

    for (uint i = 0; i < size; ++i)
    {
        std::cout << "#" << i << " " << (vec3[i].value()) << "  " << (vec[i].value())
                  << "  " << abs(vec[i].value() - vec3[i].value()) << "\n";
    }

    return 0;
}