#ifndef FFT_HPP
#define FFT_HPP

#include "vector.hpp"

template<typename T, std::size_t PackSize, typename Allocator = std::allocator<T>>
    requires packed_floating_point<T, PackSize>
class fft_unit
{
public:
    fft_unit(std::size_t fft_size){};

    fft_unit(const fft_unit& other)     = delete;
    fft_unit(fft_unit&& other) noexcept = delete;

    ~fft_unit() = default;

    fft_unit& operator=(const fft_unit& other)     = delete;
    fft_unit& operator=(fft_unit&& other) noexcept = delete;


    template<typename VT, std::size_t VPackSize, typename VAllocator>
        requires std::same_as<VT, T> && requires { PackSize == VPackSize; }
    void fft_unit_test(const packed_cx_vector<VT, VPackSize, VAllocator>& test_vecor){};

private:
};

template<typename T, std::size_t PackSize, typename Allocator>
    requires std::same_as<T, float>
void fft_internal(packed_cx_vector<T, PackSize, Allocator>& vector)
{
    auto size = vector.size();

    constexpr double pi  = 3.14159265358979323846;
    const auto       sq2 = std::exp(std::complex<float>(0, pi / 4));

    auto twsq2 = _mm256_broadcast_ss(&reinterpret_cast<T(&)[2]>(sq2)[0]);

    auto p0re = _mm256_loadu_ps(&vector[0 * size / 8]);
    auto p1re = _mm256_loadu_ps(&vector[1 * size / 8]);
    auto p2re = _mm256_loadu_ps(&vector[2 * size / 8]);
    auto p3re = _mm256_loadu_ps(&vector[3 * size / 8]);
    auto p4re = _mm256_loadu_ps(&vector[4 * size / 8]);
    auto p5re = _mm256_loadu_ps(&vector[5 * size / 8]);
    auto p6re = _mm256_loadu_ps(&vector[6 * size / 8]);
    auto p7re = _mm256_loadu_ps(&vector[7 * size / 8]);

    auto p0im = _mm256_loadu_ps(&vector[0 * size / 8] + PackSize);
    auto p1im = _mm256_loadu_ps(&vector[1 * size / 8] + PackSize);
    auto p2im = _mm256_loadu_ps(&vector[2 * size / 8] + PackSize);
    auto p3im = _mm256_loadu_ps(&vector[3 * size / 8] + PackSize);
    auto p4im = _mm256_loadu_ps(&vector[4 * size / 8] + PackSize);
    auto p5im = _mm256_loadu_ps(&vector[5 * size / 8] + PackSize);
    auto p6im = _mm256_loadu_ps(&vector[6 * size / 8] + PackSize);
    auto p7im = _mm256_loadu_ps(&vector[7 * size / 8] + PackSize);

    auto a0re = _mm256_add_ps(p0re, p4re);
    auto a4re = _mm256_sub_ps(p0re, p4re);
    auto a1re = _mm256_add_ps(p1re, p5re);
    auto a5re = _mm256_sub_ps(p1re, p5re);
    auto a2re = _mm256_add_ps(p2re, p6re);
    auto a6re = _mm256_sub_ps(p2re, p6re);
    auto a3re = _mm256_add_ps(p3re, p7re);
    auto a7re = _mm256_sub_ps(p3re, p7re);

    auto a0im = _mm256_add_ps(p0im, p4im);
    auto a4im = _mm256_sub_ps(p0im, p4im);
    auto a1im = _mm256_add_ps(p1im, p5im);
    auto a5im = _mm256_sub_ps(p1im, p5im);
    auto a2im = _mm256_add_ps(p2im, p6im);
    auto a6im = _mm256_sub_ps(p2im, p6im);
    auto a3im = _mm256_add_ps(p3im, p7im);
    auto a7im = _mm256_sub_ps(p3im, p7im);

    auto b0re = _mm256_add_ps(a0re, a2re);
    auto b2re = _mm256_sub_ps(a0re, a2re);
    auto b4re = _mm256_add_ps(a4re, a6im);
    auto b6re = _mm256_sub_ps(a4re, a6im);
    auto b0im = _mm256_add_ps(a0im, a2im);
    auto b2im = _mm256_sub_ps(a0im, a2im);
    auto b4im = _mm256_add_ps(a4im, a6re);
    auto b6im = _mm256_sub_ps(a4im, a6re);

    auto b1re = _mm256_add_ps(a1re, a3re);
    auto b3re = _mm256_sub_ps(a1re, a3re);
    auto b5re = _mm256_add_ps(a5re, a7im);
    auto b7re = _mm256_sub_ps(a5re, a7im);
    auto b1im = _mm256_add_ps(a1im, a3im);
    auto b3im = _mm256_sub_ps(a1im, a3im);
    auto b5im = _mm256_add_ps(a5im, a7re);
    auto b7im = _mm256_sub_ps(a5im, a7re);

    auto b5re_tw = _mm256_add_ps(b5re, b5im);
    auto b5im_tw = _mm256_sub_ps(b5im, b5re);
    auto b7re_tw = _mm256_add_ps(b7re, b7im);
    auto b7im_tw = _mm256_sub_ps(b7im, b7re);

    b5re_tw = _mm256_mul_ps(b5re_tw, twsq2);
    b5im_tw = _mm256_mul_ps(b5im_tw, twsq2);
    b7re_tw = _mm256_mul_ps(b7re_tw, twsq2);
    b7im_tw = _mm256_mul_ps(b7im_tw, twsq2);

    auto c0re = _mm256_add_ps(b0re, b1re);
    auto c1re = _mm256_sub_ps(b0re, b1re);
    auto c0im = _mm256_add_ps(b0im, b1im);
    auto c1im = _mm256_sub_ps(b0im, b1im);
    auto c2re = _mm256_add_ps(b2re, b3im);
    auto c3re = _mm256_sub_ps(b2re, b3im);
    auto c2im = _mm256_add_ps(b2im, b3re);
    auto c3im = _mm256_sub_ps(b2im, b3re);
    auto c4re = _mm256_add_ps(b4re, b5re_tw);
    auto c5re = _mm256_sub_ps(b4re, b5re_tw);
    auto c4im = _mm256_add_ps(b4im, b5im_tw);
    auto c5im = _mm256_sub_ps(b4im, b5im_tw);
    auto c6re = _mm256_sub_ps(b6re, b7re_tw);
    auto c7re = _mm256_add_ps(b6re, b7re_tw);
    auto c6im = _mm256_sub_ps(b6im, b7im_tw);
    auto c7im = _mm256_add_ps(b6im, b7im_tw);

    auto sha0re = _mm256_unpacklo_ps(c0re, c4re);
    auto sha4re = _mm256_unpackhi_ps(c0re, c4re);
    auto sha1re = _mm256_unpacklo_ps(c1re, c5re);
    auto sha5re = _mm256_unpackhi_ps(c1re, c5re);
    auto sha2re = _mm256_unpacklo_ps(c4re, c6re);
    auto sha6re = _mm256_unpackhi_ps(c4re, c6re);
    auto sha3re = _mm256_unpacklo_ps(c3re, c7re);
    auto sha7re = _mm256_unpackhi_ps(c3re, c7re);

    auto sha0im = _mm256_unpacklo_ps(c0im, c4im);
    auto sha4im = _mm256_unpackhi_ps(c0im, c4im);
    auto sha1im = _mm256_unpacklo_ps(c1im, c5im);
    auto sha5im = _mm256_unpackhi_ps(c1im, c5im);
    auto sha2im = _mm256_unpacklo_ps(c4im, c6im);
    auto sha6im = _mm256_unpackhi_ps(c4im, c6im);
    auto sha3im = _mm256_unpacklo_ps(c3im, c7im);
    auto sha7im = _mm256_unpackhi_ps(c3im, c7im);

    auto shb0re = _mm256_unpacklo_pd(sha0re, sha2re);
    auto shb2re = _mm256_unpackhi_pd(sha0re, sha2re);
    auto shb1re = _mm256_unpacklo_pd(sha1re, sha3re);
    auto shb3re = _mm256_unpackhi_pd(sha1re, sha3re);
    auto shb0re = _mm256_unpacklo_pd(sha4re, sha6re);
    auto shb2re = _mm256_unpackhi_pd(sha4re, sha6re);
    auto shb1re = _mm256_unpacklo_pd(sha5re, sha7re);
    auto shb3re = _mm256_unpackhi_pd(sha5re, sha7re);

    auto shb0im = _mm256_unpacklo_pd(sha0im, sha2im);
    auto shb2im = _mm256_unpackhi_pd(sha0im, sha2im);
    auto shb1im = _mm256_unpacklo_pd(sha1im, sha3im);
    auto shb3im = _mm256_unpackhi_pd(sha1im, sha3im);
    auto shb0im = _mm256_unpacklo_pd(sha4im, sha6im);
    auto shb2im = _mm256_unpackhi_pd(sha4im, sha6im);
    auto shb1im = _mm256_unpacklo_pd(sha5im, sha7im);
    auto shb3im = _mm256_unpackhi_pd(sha5im, sha7im);

    auto shc0re = _mm256_permute2f128_ps(shb0re, shb1re, 0b00100000);
    auto shc1re = _mm256_permute2f128_ps(shb0re, shb1re, 0b00110001);

    shc0_re = reg.vec(); cod.e('vperm2f128', shc0_re, shb0_re, shb1_re, '00100000b');
    shc1_re = shb1_re;   cod.e('vperm2f128', shc1_re, shb0_re, shb1_re, '00110001b');
}


#endif