#include "fft.hpp"

#include <iostream>

template void pcx::fft_unit<float, 32>::fft_internal<32, std::allocator<float>>(
    pcx::vector<float, 32, std::allocator<float>>& vector);

int main()
{
    auto vec = pcx::vector<float>(1024);
    // pcx::fft_internal(vec);

    pcx::fft_unit<float, 8>(1024);
    pcx::fft_unit<float, 32>(1024);
    std::cout << sizeof(double) << "  " << sizeof(float) << "\n";
    return 0;
}