#ifndef FFT_HPP
#define FFT_HPP

#include "vector.hpp"

template<typename T, std::size_t PackSize, typename Allocator = std::allocator<T>>
    requires packed_floating_point<T, PackSize>
class fft_unit
{
public:
    template<typename VT, std::size_t VPackSize, typename VAllocator>
        requires std::same_as<VT, T> && requires { PackSize == VPackSize; }
    fft_unit(const packed_cx_vector<VT, VPackSize, VAllocator>& test_vecor){};

    fft_unit(const fft_unit& other)     = default;
    fft_unit(fft_unit&& other) noexcept = default;

    ~fft_unit() = default;

    fft_unit& operator=(const fft_unit& other)     = delete;
    fft_unit& operator=(fft_unit&& other) noexcept = delete;

private:
};


#endif