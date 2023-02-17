#include "fft.hpp"

#include <iostream>

template void pcx::fft_unit<float, 32>::fft_internal<32, std::allocator<float>>(
    pcx::vector<float, 32, std::allocator<float>>& vector);

inline auto wnk(std::size_t n, std::size_t k) -> std::complex<float>
{
    constexpr double pi = 3.14159265358979323846;
    return exp(
        std::complex<float>(0,
                            -2 * pi * static_cast<double>(k) / static_cast<double>(n)));
}

auto fft(const pcx::vector<float>& vector)
{
    auto fft_size = vector.size();
    auto res      = pcx::vector<float>(fft_size);
    if (fft_size == 1)
    {
        res = vector;
        return res;
    }

    auto even = pcx::vector<float>(fft_size / 2);
    auto odd  = pcx::vector<float>(fft_size / 2);

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
        auto odd_v  = odd[i].value() * wnk(fft_size, i);

        res[i]                = even_v + odd_v;
        res[i + fft_size / 2] = even_v - odd_v;
    }

    return res;
}


int main()
{
    const uint fsize = 128;
    auto       unit  = pcx::fft_unit<float, 32>(fsize);
    auto       vec   = pcx::vector<float>(fsize);
    // pcx::fft_internal(vec);
    vec[0] = 1;

    constexpr float pi = 3.14159265358979323846;

    for (uint i = 0; i < fsize; ++i)
    {
        vec[i] = std::exp(std::complex(0.F, 2 * pi * i / 2));
        // vec[i] = 1;
    }

    auto ff = fft(vec);
    unit(vec);
    for (uint i = 0; i < fsize; ++i)
    {
        // std::cout << abs(ff[i].value()) << "\n";
        // std::cout << abs(vec[i].value()) << "\n";
        // std::cout << ff[i].value() << " " << vec[i].value() << "\n";
        std::cout << abs(ff[i].value() - vec[i].value()) << "\n";
    }

    std::cout << wnk(8, 1);
    std::cout << sizeof(double) << "  " << sizeof(float) << "\n";
    return 0;
}