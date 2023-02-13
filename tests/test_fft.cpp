#include "fft.hpp"

template void fft_internal<float, 32, std::allocator<float>>(
    pcx::vector<float, 32, std::allocator<float>>& vector);

int main()
{
    auto vec = pcx::vector<float>(1024);
    fft_internal(vec);

    fft_unit<float, 32>(1024);
    return 0;
}