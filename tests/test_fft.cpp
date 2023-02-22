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

int test_fft_float(std::size_t size)
{
    constexpr float  pi  = 3.14159265358979323846;
    constexpr double dpi = 3.14159265358979323846;

    auto depth = log2i(size);

    auto vec = pcx::vector<float>(size);
    auto vec_out     = pcx::vector<float>(size);
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
    int ret = 0;
    for (uint i = 6; i < 21; ++i)
    {
        std::cout << (1U << i) << "\n";

        ret += test_fft_float(1U << i);
        if (ret > 0)
        {
            return ret;
        }
    }

    //     constexpr std::size_t size = 64;
    //     constexpr float       pi   = 3.14159265358979323846;
    //
    //     auto vec  = pcx::vector<float>(size);
    //     auto vec2 = pcx::vector<float>(size);
    //     for (uint i = 0; i < size; ++i)
    //     {
    //         vec[i] = std::exp(std::complex(0.F, 2 * pi * i / size * 13.37F));
    //     }
    //
    //     auto unit = pcx::fft_unit<float, pcx::dynamic_size, 16>(size);
    //
    //     unit(vec2, vec);
    //     auto ff = fft(vec);
    //
    //     for (uint i = 0; i < size; ++i)
    //     {
    //         std::cout << "#" << i << " " << vec2[i].value() << "  " << ff[i].value() << "\n";
    //     }

    return 0;
}