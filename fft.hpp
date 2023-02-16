#ifndef FFT_HPP
#define FFT_HPP

#include "vector.hpp"

#include <memory>

namespace pcx {

namespace avx {
    template<std::size_t PackSize, typename T>
        requires packed_floating_point<T, PackSize>
    auto cxload(const T* ptr) -> cx_reg<T>
    {
        return {load(ptr), load(ptr + PackSize)};
    }

    template<std::size_t PackSize, typename T>
        requires packed_floating_point<T, PackSize>
    void cxstore(T* ptr, cx_reg<T> reg)
    {
        store(ptr, reg.real);
        store(ptr + PackSize, reg.imag);
    }

}    // namespace avx

constexpr const std::size_t dynamic_size = -1;

template<typename T,
         std::size_t PackSize,
         typename Allocator = std::allocator<T>,
         std::size_t Size   = pcx::dynamic_size>
    requires pcx::packed_floating_point<T, PackSize>
class fft_unit
{
public:
    using real_type      = T;
    using allocator_type = Allocator;

    static constexpr auto pack_size = PackSize;

private:
    using size_t =
        std::conditional_t<Size == pcx::dynamic_size, std::size_t, decltype([] {})>;

    using sort_allocator_type = typename std::allocator_traits<
        allocator_type>::template rebind_alloc<std::size_t>;

public:
    fft_unit(std::size_t fft_size, allocator_type allocator = allocator_type())
        requires(Size == pcx::dynamic_size)
    : m_size(fft_size)
    , m_sort(get_sort(fft_size, static_cast<sort_allocator_type>(allocator)))
    , m_twiddles(get_twiddles(fft_size, allocator)){};

    fft_unit(const fft_unit& other)     = default;
    fft_unit(fft_unit&& other) noexcept = default;

    ~fft_unit() = default;

    fft_unit& operator=(const fft_unit& other)     = default;
    fft_unit& operator=(fft_unit&& other) noexcept = default;

    [[nodiscard]] constexpr auto size() const -> std::size_t
    {
        if constexpr (Size == pcx::dynamic_size)
        {
            return m_size;
        } else
        {
            return Size;
        }
    }


    template<typename VT, std::size_t VPackSize, typename VAllocator>
        requires std::same_as<VT, real_type> && requires { VPackSize == pack_size; }
    void operator()(pcx::vector<VT, VPackSize, VAllocator>& test_vecor)
    {
        fft_internal(test_vecor);
    };

private:
    [[no_unique_address]] size_t m_size;
    // std::shared_ptr<packed_fft_core>                  m_core{};
    const std::vector<std::size_t, sort_allocator_type>     m_sort;
    const pcx::vector<real_type, pack_size, allocator_type> m_twiddles;

public:
    template<std::size_t VPackSize, typename VAllocator>
    void fft_internal(pcx::vector<T, VPackSize, VAllocator>& vector)
    {
        constexpr const double pi  = 3.14159265358979323846;
        const auto             sq2 = std::exp(std::complex<float>(0, pi / 4));

        auto  twsq2       = avx::broadcast(sq2.real());
        auto* data_ptr    = &vector[0];
        auto* twiddle_ptr = &m_twiddles[0];

        auto sh0 = 0;
        auto sh1 = pidx(1 * size() / 8);
        auto sh2 = pidx(2 * size() / 8);
        auto sh3 = pidx(3 * size() / 8);
        auto sh4 = pidx(4 * size() / 8);
        auto sh5 = pidx(5 * size() / 8);
        auto sh6 = pidx(6 * size() / 8);
        auto sh7 = pidx(7 * size() / 8);

        for (uint i = 0; i < n_reversals(size() / 64); i += 2)
        {
            auto offset_first  = m_sort[i];
            auto offset_second = m_sort[i + 1];

            auto p1re = _mm256_loadu_ps(data_ptr + sh1 + offset_first);
            auto p1im = _mm256_loadu_ps(data_ptr + sh1 + offset_first + PackSize);
            auto p5re = _mm256_loadu_ps(data_ptr + sh5 + offset_first);
            auto p5im = _mm256_loadu_ps(data_ptr + sh5 + offset_first + PackSize);
            auto p3re = _mm256_loadu_ps(data_ptr + sh3 + offset_first);
            auto p3im = _mm256_loadu_ps(data_ptr + sh3 + offset_first + PackSize);
            auto p7re = _mm256_loadu_ps(data_ptr + sh7 + offset_first);
            auto p7im = _mm256_loadu_ps(data_ptr + sh7 + offset_first + PackSize);

            auto a5re = _mm256_sub_ps(p1re, p5re);
            auto a7im = _mm256_sub_ps(p3im, p7im);
            auto a5im = _mm256_sub_ps(p1im, p5im);
            auto a7re = _mm256_sub_ps(p3re, p7re);
            auto a1re = _mm256_add_ps(p1re, p5re);
            auto a3im = _mm256_add_ps(p3im, p7im);

            auto b5re = _mm256_add_ps(a5re, a7im);
            auto b7re = _mm256_sub_ps(a5re, a7im);
            auto b5im = _mm256_add_ps(a5im, a7re);
            auto b7im = _mm256_sub_ps(a5im, a7re);
            auto a1im = _mm256_add_ps(p1im, p5im);
            auto a3re = _mm256_add_ps(p3re, p7re);

            auto b5re_tw = _mm256_add_ps(b5re, b5im);
            auto b5im_tw = _mm256_sub_ps(b5im, b5re);
            auto b7re_tw = _mm256_add_ps(b7re, b7im);
            auto b7im_tw = _mm256_sub_ps(b7im, b7re);

            auto b1re = _mm256_add_ps(a1re, a3re);
            auto b3re = _mm256_sub_ps(a1re, a3re);
            auto b1im = _mm256_add_ps(a1im, a3im);
            auto b3im = _mm256_sub_ps(a1im, a3im);

            b5re_tw = _mm256_mul_ps(b5re_tw, twsq2);
            b5im_tw = _mm256_mul_ps(b5im_tw, twsq2);
            b7re_tw = _mm256_mul_ps(b7re_tw, twsq2);
            b7im_tw = _mm256_mul_ps(b7im_tw, twsq2);

            auto p0re = _mm256_loadu_ps(data_ptr + sh0 + offset_first);
            auto p0im = _mm256_loadu_ps(data_ptr + sh0 + offset_first + PackSize);
            auto p4re = _mm256_loadu_ps(data_ptr + sh4 + offset_first);
            auto p4im = _mm256_loadu_ps(data_ptr + sh4 + offset_first + PackSize);
            auto p2re = _mm256_loadu_ps(data_ptr + sh2 + offset_first);
            auto p2im = _mm256_loadu_ps(data_ptr + sh2 + offset_first + PackSize);
            auto p6re = _mm256_loadu_ps(data_ptr + sh6 + offset_first);
            auto p6im = _mm256_loadu_ps(data_ptr + sh6 + offset_first + PackSize);

            auto a0re = _mm256_add_ps(p0re, p4re);
            auto a0im = _mm256_add_ps(p0im, p4im);
            auto a2re = _mm256_add_ps(p2re, p6re);
            auto a2im = _mm256_add_ps(p2im, p6im);
            auto a4re = _mm256_sub_ps(p0re, p4re);
            auto a4im = _mm256_sub_ps(p0im, p4im);
            auto a6re = _mm256_sub_ps(p2re, p6re);
            auto a6im = _mm256_sub_ps(p2im, p6im);

            auto b0re = _mm256_add_ps(a0re, a2re);
            auto b2re = _mm256_sub_ps(a0re, a2re);
            auto b0im = _mm256_add_ps(a0im, a2im);
            auto b2im = _mm256_sub_ps(a0im, a2im);
            auto b4re = _mm256_add_ps(a4re, a6im);
            auto b6re = _mm256_sub_ps(a4re, a6im);
            auto b4im = _mm256_add_ps(a4im, a6re);
            auto b6im = _mm256_sub_ps(a4im, a6re);

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

            auto sha1re = _mm256_castps_pd(_mm256_unpacklo_ps(c1re, c5re));
            auto sha5re = _mm256_castps_pd(_mm256_unpackhi_ps(c1re, c5re));
            auto sha1im = _mm256_castps_pd(_mm256_unpacklo_ps(c1im, c5im));
            auto sha5im = _mm256_castps_pd(_mm256_unpackhi_ps(c1im, c5im));
            auto sha3re = _mm256_castps_pd(_mm256_unpacklo_ps(c3re, c7re));
            auto sha7re = _mm256_castps_pd(_mm256_unpackhi_ps(c3re, c7re));
            auto sha3im = _mm256_castps_pd(_mm256_unpacklo_ps(c3im, c7im));
            auto sha7im = _mm256_castps_pd(_mm256_unpackhi_ps(c3im, c7im));

            auto sha0re = _mm256_castps_pd(_mm256_unpacklo_ps(c0re, c4re));
            auto sha4re = _mm256_castps_pd(_mm256_unpackhi_ps(c0re, c4re));
            auto sha0im = _mm256_castps_pd(_mm256_unpacklo_ps(c0im, c4im));
            auto sha4im = _mm256_castps_pd(_mm256_unpackhi_ps(c0im, c4im));
            auto sha2re = _mm256_castps_pd(_mm256_unpacklo_ps(c2re, c6re));
            auto sha6re = _mm256_castps_pd(_mm256_unpackhi_ps(c2re, c6re));
            auto sha2im = _mm256_castps_pd(_mm256_unpacklo_ps(c2im, c6im));
            auto sha6im = _mm256_castps_pd(_mm256_unpackhi_ps(c2im, c6im));

            auto shb0re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha0re, sha2re));
            auto shb2re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha0re, sha2re));
            auto shb0im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha0im, sha2im));
            auto shb2im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha0im, sha2im));
            auto shb1re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha1re, sha3re));
            auto shb3re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha1re, sha3re));
            auto shb1im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha1im, sha3im));
            auto shb3im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha1im, sha3im));

            auto shc0re = _mm256_permute2f128_ps(shb0re, shb1re, 0b00100000);
            auto shc1re = _mm256_permute2f128_ps(shb0re, shb1re, 0b00110001);
            auto shc0im = _mm256_permute2f128_ps(shb0im, shb1im, 0b00100000);
            auto shc1im = _mm256_permute2f128_ps(shb0im, shb1im, 0b00110001);

            auto q0re = _mm256_loadu_ps(data_ptr + sh0 + offset_second);
            _mm256_storeu_ps(data_ptr + sh0 + offset_second, shc0re);
            auto q0im = _mm256_loadu_ps(data_ptr + sh0 + offset_second + PackSize);
            _mm256_storeu_ps(data_ptr + sh0 + offset_second + PackSize, shc0im);
            auto q1re = _mm256_loadu_ps(data_ptr + sh1 + offset_second);
            _mm256_storeu_ps(data_ptr + sh1 + offset_second, shc1re);
            auto q1im = _mm256_loadu_ps(data_ptr + sh1 + offset_second + PackSize);
            _mm256_storeu_ps(data_ptr + sh1 + offset_second + PackSize, shc1im);

            auto shc2re = _mm256_permute2f128_ps(shb2re, shb3re, 0b00100000);
            auto shc3re = _mm256_permute2f128_ps(shb2re, shb3re, 0b00110001);
            auto shc2im = _mm256_permute2f128_ps(shb2im, shb3im, 0b00100000);
            auto shc3im = _mm256_permute2f128_ps(shb2im, shb3im, 0b00110001);

            auto q2re = _mm256_loadu_ps(data_ptr + sh2 + offset_second);
            _mm256_storeu_ps(data_ptr + sh2 + offset_second, shc2re);
            auto q2im = _mm256_loadu_ps(data_ptr + sh2 + offset_second + PackSize);
            _mm256_storeu_ps(data_ptr + sh2 + offset_second + PackSize, shc2im);
            auto q3re = _mm256_loadu_ps(data_ptr + sh3 + offset_second);
            _mm256_storeu_ps(data_ptr + sh3 + offset_second, shc3re);
            auto q3im = _mm256_loadu_ps(data_ptr + sh3 + offset_second + PackSize);
            _mm256_storeu_ps(data_ptr + sh3 + offset_second + PackSize, shc3im);

            auto shb4re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha4re, sha6re));
            auto shb6re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha4re, sha6re));
            auto shb5re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha5re, sha7re));
            auto shb7re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha5re, sha7re));

            auto shc4re = _mm256_permute2f128_ps(shb4re, shb5re, 0b00100000);
            auto shc5re = _mm256_permute2f128_ps(shb4re, shb5re, 0b00110001);
            auto shc6re = _mm256_permute2f128_ps(shb6re, shb7re, 0b00100000);
            auto shc7re = _mm256_permute2f128_ps(shb6re, shb7re, 0b00110001);

            auto shb4im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha4im, sha6im));
            auto shb6im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha4im, sha6im));
            auto shb5im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha5im, sha7im));
            auto shb7im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha5im, sha7im));

            auto shc4im = _mm256_permute2f128_ps(shb4im, shb5im, 0b00100000);
            auto shc5im = _mm256_permute2f128_ps(shb4im, shb5im, 0b00110001);
            auto shc6im = _mm256_permute2f128_ps(shb6im, shb7im, 0b00100000);
            auto shc7im = _mm256_permute2f128_ps(shb6im, shb7im, 0b00110001);


            auto q4re = _mm256_loadu_ps(data_ptr + sh4 + offset_second);
            _mm256_storeu_ps(data_ptr + sh4 + offset_second, shc4re);
            auto q4im = _mm256_loadu_ps(data_ptr + sh4 + offset_second + PackSize);
            _mm256_storeu_ps(data_ptr + sh4 + offset_second + PackSize, shc4im);
            auto q6re = _mm256_loadu_ps(data_ptr + sh6 + offset_second);
            _mm256_storeu_ps(data_ptr + sh6 + offset_second, shc6re);
            auto q6im = _mm256_loadu_ps(data_ptr + sh6 + offset_second + PackSize);
            _mm256_storeu_ps(data_ptr + sh6 + offset_second + PackSize, shc6im);

            auto q5re = _mm256_loadu_ps(data_ptr + sh5 + offset_second);
            _mm256_storeu_ps(data_ptr + sh5 + offset_second, shc5re);
            auto q5im = _mm256_loadu_ps(data_ptr + sh5 + offset_second + PackSize);
            _mm256_storeu_ps(data_ptr + sh5 + offset_second + PackSize, shc5im);
            auto q7re = _mm256_loadu_ps(data_ptr + sh7 + offset_second);
            _mm256_storeu_ps(data_ptr + sh7 + offset_second, shc7re);
            auto q7im = _mm256_loadu_ps(data_ptr + sh7 + offset_second + PackSize);
            _mm256_storeu_ps(data_ptr + sh7 + offset_second + PackSize, shc7im);

            auto x1re = _mm256_add_ps(q1re, q5re);
            auto x5re = _mm256_sub_ps(q1re, q5re);
            auto x3im = _mm256_add_ps(q3im, q7im);
            auto x7im = _mm256_sub_ps(q3im, q7im);
            auto x1im = _mm256_add_ps(q1im, q5im);
            auto x5im = _mm256_sub_ps(q1im, q5im);
            auto x3re = _mm256_add_ps(q3re, q7re);
            auto x7re = _mm256_sub_ps(q3re, q7re);

            auto y5re = _mm256_add_ps(x5re, x7im);
            auto y7re = _mm256_sub_ps(x5re, x7im);
            auto y5im = _mm256_add_ps(x5im, x7re);
            auto y7im = _mm256_sub_ps(x5im, x7re);

            auto y5re_tw = _mm256_add_ps(y5re, y5im);
            auto y5im_tw = _mm256_sub_ps(y5im, y5re);
            auto y7re_tw = _mm256_add_ps(y7re, y7im);
            auto y7im_tw = _mm256_sub_ps(y7im, y7re);

            auto y1re = _mm256_add_ps(x1re, x3re);
            auto y3re = _mm256_sub_ps(x1re, x3re);
            auto y1im = _mm256_add_ps(x1im, x3im);
            auto y3im = _mm256_sub_ps(x1im, x3im);
            y5re_tw   = _mm256_mul_ps(y5re_tw, twsq2);
            y5im_tw   = _mm256_mul_ps(y5im_tw, twsq2);
            y7re_tw   = _mm256_mul_ps(y7re_tw, twsq2);
            y7im_tw   = _mm256_mul_ps(y7im_tw, twsq2);

            auto x0re = _mm256_add_ps(q0re, q4re);
            auto x4re = _mm256_sub_ps(q0re, q4re);
            auto x0im = _mm256_add_ps(q0im, q4im);
            auto x4im = _mm256_sub_ps(q0im, q4im);
            auto x2re = _mm256_add_ps(q2re, q6re);
            auto x6re = _mm256_sub_ps(q2re, q6re);
            auto x2im = _mm256_add_ps(q2im, q6im);
            auto x6im = _mm256_sub_ps(q2im, q6im);

            auto y0re = _mm256_add_ps(x0re, x2re);
            auto y2re = _mm256_sub_ps(x0re, x2re);
            auto y0im = _mm256_add_ps(x0im, x2im);
            auto y2im = _mm256_sub_ps(x0im, x2im);
            auto y4re = _mm256_add_ps(x4re, x6im);
            auto y6re = _mm256_sub_ps(x4re, x6im);
            auto y4im = _mm256_add_ps(x4im, x6re);
            auto y6im = _mm256_sub_ps(x4im, x6re);

            auto z0re = _mm256_add_ps(y0re, y1re);
            auto z1re = _mm256_sub_ps(y0re, y1re);
            auto z0im = _mm256_add_ps(y0im, y1im);
            auto z1im = _mm256_sub_ps(y0im, y1im);
            auto z2re = _mm256_add_ps(y2re, y3im);
            auto z3re = _mm256_sub_ps(y2re, y3im);
            auto z2im = _mm256_add_ps(y2im, y3re);
            auto z3im = _mm256_sub_ps(y2im, y3re);

            auto z4re = _mm256_add_ps(y4re, y5re_tw);
            auto z5re = _mm256_sub_ps(y4re, y5re_tw);
            auto z4im = _mm256_add_ps(y4im, y5im_tw);
            auto z5im = _mm256_sub_ps(y4im, y5im_tw);
            auto z6re = _mm256_sub_ps(y6re, y7re_tw);
            auto z7re = _mm256_add_ps(y6re, y7re_tw);
            auto z6im = _mm256_sub_ps(y6im, y7im_tw);
            auto z7im = _mm256_add_ps(y6im, y7im_tw);

            auto shx1re = _mm256_castps_pd(_mm256_unpacklo_ps(z1re, z5re));
            auto shx5re = _mm256_castps_pd(_mm256_unpackhi_ps(z1re, z5re));
            auto shx3re = _mm256_castps_pd(_mm256_unpacklo_ps(z3re, z7re));
            auto shx7re = _mm256_castps_pd(_mm256_unpackhi_ps(z3re, z7re));
            auto shx0re = _mm256_castps_pd(_mm256_unpacklo_ps(z0re, z4re));
            auto shx4re = _mm256_castps_pd(_mm256_unpackhi_ps(z0re, z4re));
            auto shx2re = _mm256_castps_pd(_mm256_unpacklo_ps(z2re, z6re));
            auto shx6re = _mm256_castps_pd(_mm256_unpackhi_ps(z2re, z6re));

            auto shx0im = _mm256_castps_pd(_mm256_unpacklo_ps(z0im, z4im));
            auto shx4im = _mm256_castps_pd(_mm256_unpackhi_ps(z0im, z4im));
            auto shx2im = _mm256_castps_pd(_mm256_unpacklo_ps(z2im, z6im));
            auto shx6im = _mm256_castps_pd(_mm256_unpackhi_ps(z2im, z6im));
            auto shx1im = _mm256_castps_pd(_mm256_unpacklo_ps(z1im, z5im));
            auto shx5im = _mm256_castps_pd(_mm256_unpackhi_ps(z1im, z5im));
            auto shx3im = _mm256_castps_pd(_mm256_unpacklo_ps(z3im, z7im));
            auto shx7im = _mm256_castps_pd(_mm256_unpackhi_ps(z3im, z7im));

            auto shy0re = _mm256_castpd_ps(_mm256_unpacklo_pd(shx0re, shx2re));
            auto shy2re = _mm256_castpd_ps(_mm256_unpackhi_pd(shx0re, shx2re));
            auto shy1re = _mm256_castpd_ps(_mm256_unpacklo_pd(shx1re, shx3re));
            auto shy3re = _mm256_castpd_ps(_mm256_unpackhi_pd(shx1re, shx3re));

            auto shy0im = _mm256_castpd_ps(_mm256_unpacklo_pd(shx0im, shx2im));
            auto shy2im = _mm256_castpd_ps(_mm256_unpackhi_pd(shx0im, shx2im));
            auto shy1im = _mm256_castpd_ps(_mm256_unpacklo_pd(shx1im, shx3im));
            auto shy3im = _mm256_castpd_ps(_mm256_unpackhi_pd(shx1im, shx3im));

            auto shz0re = _mm256_permute2f128_ps(shy0re, shy1re, 0b00100000);
            auto shz1re = _mm256_permute2f128_ps(shy0re, shy1re, 0b00110001);
            auto shz2re = _mm256_permute2f128_ps(shy2re, shy3re, 0b00100000);
            auto shz3re = _mm256_permute2f128_ps(shy2re, shy3re, 0b00110001);
            auto shz0im = _mm256_permute2f128_ps(shy0im, shy1im, 0b00100000);
            auto shz1im = _mm256_permute2f128_ps(shy0im, shy1im, 0b00110001);
            auto shz2im = _mm256_permute2f128_ps(shy2im, shy3im, 0b00100000);
            auto shz3im = _mm256_permute2f128_ps(shy2im, shy3im, 0b00110001);

            _mm256_storeu_ps(data_ptr + sh0 + offset_first, shz0re);
            _mm256_storeu_ps(data_ptr + sh0 + offset_first + PackSize, shz0im);
            _mm256_storeu_ps(data_ptr + sh1 + offset_first, shz1re);
            _mm256_storeu_ps(data_ptr + sh1 + offset_first + PackSize, shz1im);
            _mm256_storeu_ps(data_ptr + sh2 + offset_first, shz2re);
            _mm256_storeu_ps(data_ptr + sh2 + offset_first + PackSize, shz2im);
            _mm256_storeu_ps(data_ptr + sh3 + offset_first, shz3re);
            _mm256_storeu_ps(data_ptr + sh3 + offset_first + PackSize, shz3im);

            auto shy4re = _mm256_castpd_ps(_mm256_unpacklo_pd(shx4re, shx6re));
            auto shy6re = _mm256_castpd_ps(_mm256_unpackhi_pd(shx4re, shx6re));
            auto shy5re = _mm256_castpd_ps(_mm256_unpacklo_pd(shx5re, shx7re));
            auto shy7re = _mm256_castpd_ps(_mm256_unpackhi_pd(shx5re, shx7re));

            auto shz4re = _mm256_permute2f128_ps(shy4re, shy5re, 0b00100000);
            auto shz5re = _mm256_permute2f128_ps(shy4re, shy5re, 0b00110001);
            auto shz6re = _mm256_permute2f128_ps(shy6re, shy7re, 0b00100000);
            auto shz7re = _mm256_permute2f128_ps(shy6re, shy7re, 0b00110001);

            auto shy4im = _mm256_castpd_ps(_mm256_unpacklo_pd(shx4im, shx6im));
            auto shy6im = _mm256_castpd_ps(_mm256_unpackhi_pd(shx4im, shx6im));
            auto shy5im = _mm256_castpd_ps(_mm256_unpacklo_pd(shx5im, shx7im));
            auto shy7im = _mm256_castpd_ps(_mm256_unpackhi_pd(shx5im, shx7im));

            auto shz4im = _mm256_permute2f128_ps(shy4im, shy5im, 0b00100000);
            auto shz5im = _mm256_permute2f128_ps(shy4im, shy5im, 0b00110001);
            auto shz6im = _mm256_permute2f128_ps(shy6im, shy7im, 0b00100000);
            auto shz7im = _mm256_permute2f128_ps(shy6im, shy7im, 0b00110001);

            _mm256_storeu_ps(data_ptr + sh4 + offset_first, shz4re);
            _mm256_storeu_ps(data_ptr + sh4 + offset_first + PackSize, shz4im);
            _mm256_storeu_ps(data_ptr + sh5 + offset_first, shz5re);
            _mm256_storeu_ps(data_ptr + sh5 + offset_first + PackSize, shz5im);
            _mm256_storeu_ps(data_ptr + sh6 + offset_first, shz6re);
            _mm256_storeu_ps(data_ptr + sh6 + offset_first + PackSize, shz6im);
            _mm256_storeu_ps(data_ptr + sh7 + offset_first, shz7re);
            _mm256_storeu_ps(data_ptr + sh7 + offset_first + PackSize, shz7im);
        };

        for (uint i = n_reversals(size() / 64); i < size() / 64; ++i)
        {
            auto offset = m_sort[i];

            auto p1re = _mm256_loadu_ps(data_ptr + sh1 + offset);
            auto p1im = _mm256_loadu_ps(data_ptr + sh1 + offset + PackSize);
            auto p5re = _mm256_loadu_ps(data_ptr + sh5 + offset);
            auto p5im = _mm256_loadu_ps(data_ptr + sh5 + offset + PackSize);
            auto p3re = _mm256_loadu_ps(data_ptr + sh3 + offset);
            auto p3im = _mm256_loadu_ps(data_ptr + sh3 + offset + PackSize);
            auto p7re = _mm256_loadu_ps(data_ptr + sh7 + offset);
            auto p7im = _mm256_loadu_ps(data_ptr + sh7 + offset + PackSize);

            auto a5re = _mm256_sub_ps(p1re, p5re);
            auto a7im = _mm256_sub_ps(p3im, p7im);
            auto a5im = _mm256_sub_ps(p1im, p5im);
            auto a7re = _mm256_sub_ps(p3re, p7re);
            auto a1re = _mm256_add_ps(p1re, p5re);
            auto a3im = _mm256_add_ps(p3im, p7im);

            auto b5re = _mm256_add_ps(a5re, a7im);
            auto b7re = _mm256_sub_ps(a5re, a7im);
            auto b5im = _mm256_add_ps(a5im, a7re);
            auto b7im = _mm256_sub_ps(a5im, a7re);
            auto a1im = _mm256_add_ps(p1im, p5im);
            auto a3re = _mm256_add_ps(p3re, p7re);

            auto b5re_tw = _mm256_add_ps(b5re, b5im);
            auto b5im_tw = _mm256_sub_ps(b5im, b5re);
            auto b7re_tw = _mm256_add_ps(b7re, b7im);
            auto b7im_tw = _mm256_sub_ps(b7im, b7re);

            auto b1re = _mm256_add_ps(a1re, a3re);
            auto b3re = _mm256_sub_ps(a1re, a3re);
            auto b1im = _mm256_add_ps(a1im, a3im);
            auto b3im = _mm256_sub_ps(a1im, a3im);

            b5re_tw = _mm256_mul_ps(b5re_tw, twsq2);
            b5im_tw = _mm256_mul_ps(b5im_tw, twsq2);
            b7re_tw = _mm256_mul_ps(b7re_tw, twsq2);
            b7im_tw = _mm256_mul_ps(b7im_tw, twsq2);

            auto p0re = _mm256_loadu_ps(data_ptr + sh0 + offset);
            auto p0im = _mm256_loadu_ps(data_ptr + sh0 + offset + PackSize);
            auto p4re = _mm256_loadu_ps(data_ptr + sh4 + offset);
            auto p4im = _mm256_loadu_ps(data_ptr + sh4 + offset + PackSize);
            auto p2re = _mm256_loadu_ps(data_ptr + sh2 + offset);
            auto p2im = _mm256_loadu_ps(data_ptr + sh2 + offset + PackSize);
            auto p6re = _mm256_loadu_ps(data_ptr + sh6 + offset);
            auto p6im = _mm256_loadu_ps(data_ptr + sh6 + offset + PackSize);

            auto a0re = _mm256_add_ps(p0re, p4re);
            auto a0im = _mm256_add_ps(p0im, p4im);
            auto a2re = _mm256_add_ps(p2re, p6re);
            auto a2im = _mm256_add_ps(p2im, p6im);
            auto a4re = _mm256_sub_ps(p0re, p4re);
            auto a4im = _mm256_sub_ps(p0im, p4im);
            auto a6re = _mm256_sub_ps(p2re, p6re);
            auto a6im = _mm256_sub_ps(p2im, p6im);

            auto b0re = _mm256_add_ps(a0re, a2re);
            auto b2re = _mm256_sub_ps(a0re, a2re);
            auto b0im = _mm256_add_ps(a0im, a2im);
            auto b2im = _mm256_sub_ps(a0im, a2im);
            auto b4re = _mm256_add_ps(a4re, a6im);
            auto b6re = _mm256_sub_ps(a4re, a6im);
            auto b4im = _mm256_add_ps(a4im, a6re);
            auto b6im = _mm256_sub_ps(a4im, a6re);

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

            auto sha1re = _mm256_castps_pd(_mm256_unpacklo_ps(c1re, c5re));
            auto sha5re = _mm256_castps_pd(_mm256_unpackhi_ps(c1re, c5re));
            auto sha1im = _mm256_castps_pd(_mm256_unpacklo_ps(c1im, c5im));
            auto sha5im = _mm256_castps_pd(_mm256_unpackhi_ps(c1im, c5im));
            auto sha3re = _mm256_castps_pd(_mm256_unpacklo_ps(c3re, c7re));
            auto sha7re = _mm256_castps_pd(_mm256_unpackhi_ps(c3re, c7re));
            auto sha3im = _mm256_castps_pd(_mm256_unpacklo_ps(c3im, c7im));
            auto sha7im = _mm256_castps_pd(_mm256_unpackhi_ps(c3im, c7im));

            auto sha0re = _mm256_castps_pd(_mm256_unpacklo_ps(c0re, c4re));
            auto sha4re = _mm256_castps_pd(_mm256_unpackhi_ps(c0re, c4re));
            auto sha0im = _mm256_castps_pd(_mm256_unpacklo_ps(c0im, c4im));
            auto sha4im = _mm256_castps_pd(_mm256_unpackhi_ps(c0im, c4im));
            auto sha2re = _mm256_castps_pd(_mm256_unpacklo_ps(c2re, c6re));
            auto sha6re = _mm256_castps_pd(_mm256_unpackhi_ps(c2re, c6re));
            auto sha2im = _mm256_castps_pd(_mm256_unpacklo_ps(c2im, c6im));
            auto sha6im = _mm256_castps_pd(_mm256_unpackhi_ps(c2im, c6im));

            auto shb0re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha0re, sha2re));
            auto shb2re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha0re, sha2re));
            auto shb0im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha0im, sha2im));
            auto shb2im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha0im, sha2im));
            auto shb1re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha1re, sha3re));
            auto shb3re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha1re, sha3re));
            auto shb1im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha1im, sha3im));
            auto shb3im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha1im, sha3im));

            auto shc0re = _mm256_permute2f128_ps(shb0re, shb1re, 0b00100000);
            auto shc1re = _mm256_permute2f128_ps(shb0re, shb1re, 0b00110001);
            auto shc0im = _mm256_permute2f128_ps(shb0im, shb1im, 0b00100000);
            auto shc1im = _mm256_permute2f128_ps(shb0im, shb1im, 0b00110001);

            _mm256_storeu_ps(data_ptr + sh0 + offset, shc0re);
            _mm256_storeu_ps(data_ptr + sh0 + offset + PackSize, shc0im);
            _mm256_storeu_ps(data_ptr + sh1 + offset, shc1re);
            _mm256_storeu_ps(data_ptr + sh1 + offset + PackSize, shc1im);

            auto shc2re = _mm256_permute2f128_ps(shb2re, shb3re, 0b00100000);
            auto shc3re = _mm256_permute2f128_ps(shb2re, shb3re, 0b00110001);
            auto shc2im = _mm256_permute2f128_ps(shb2im, shb3im, 0b00100000);
            auto shc3im = _mm256_permute2f128_ps(shb2im, shb3im, 0b00110001);

            _mm256_storeu_ps(data_ptr + sh2 + offset, shc2re);
            _mm256_storeu_ps(data_ptr + sh2 + offset + PackSize, shc2im);
            _mm256_storeu_ps(data_ptr + sh3 + offset, shc3re);
            _mm256_storeu_ps(data_ptr + sh3 + offset + PackSize, shc3im);

            auto shb4re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha4re, sha6re));
            auto shb6re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha4re, sha6re));
            auto shb5re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha5re, sha7re));
            auto shb7re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha5re, sha7re));

            auto shc4re = _mm256_permute2f128_ps(shb4re, shb5re, 0b00100000);
            auto shc5re = _mm256_permute2f128_ps(shb4re, shb5re, 0b00110001);
            auto shc6re = _mm256_permute2f128_ps(shb6re, shb7re, 0b00100000);
            auto shc7re = _mm256_permute2f128_ps(shb6re, shb7re, 0b00110001);

            auto shb4im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha4im, sha6im));
            auto shb6im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha4im, sha6im));
            auto shb5im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha5im, sha7im));
            auto shb7im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha5im, sha7im));

            auto shc4im = _mm256_permute2f128_ps(shb4im, shb5im, 0b00100000);
            auto shc5im = _mm256_permute2f128_ps(shb4im, shb5im, 0b00110001);
            auto shc6im = _mm256_permute2f128_ps(shb6im, shb7im, 0b00100000);
            auto shc7im = _mm256_permute2f128_ps(shb6im, shb7im, 0b00110001);

            _mm256_storeu_ps(data_ptr + sh4 + offset, shc4re);
            _mm256_storeu_ps(data_ptr + sh4 + offset + PackSize, shc4im);
            _mm256_storeu_ps(data_ptr + sh6 + offset, shc6re);
            _mm256_storeu_ps(data_ptr + sh6 + offset + PackSize, shc6im);

            _mm256_storeu_ps(data_ptr + sh5 + offset, shc5re);
            _mm256_storeu_ps(data_ptr + sh5 + offset + PackSize, shc5im);
            _mm256_storeu_ps(data_ptr + sh7 + offset, shc7re);
            _mm256_storeu_ps(data_ptr + sh7 + offset + PackSize, shc7im);
        }

        std::size_t l_size     = 16;
        std::size_t group_size = size() / reg_size / 4;
        std::size_t n_groups   = 1;
        std::size_t tw_offset  = 0;

        while (l_size < size() / 2)
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                auto*       grp_ptr = data_ptr + pidx(i_group * reg_size);
                std::size_t offset  = 0;

                const auto tw0 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset));
                const auto tw1 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset + 8));
                const auto tw2 =
                    avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset + 16));

                tw_offset += 24;

                for (std::size_t i = 0; i < group_size; ++i)
                {
                    auto* ptr0 = grp_ptr + pidx(offset);
                    auto* ptr1 = grp_ptr + pidx(offset + l_size / 2);
                    auto* ptr2 = grp_ptr + pidx(offset + l_size);
                    auto* ptr3 = grp_ptr + pidx(offset + l_size / 2 * 3);

                    auto p1 = avx::cxload<PackSize>(ptr1);
                    auto p3 = avx::cxload<PackSize>(ptr3);
                    auto p0 = avx::cxload<PackSize>(ptr0);
                    auto p2 = avx::cxload<PackSize>(ptr2);

                    auto p1tw_re = avx::mul(p1.real, tw0.real);
                    auto p3tw_re = avx::mul(p3.real, tw0.real);
                    auto p1tw_im = avx::mul(p1.real, tw0.imag);
                    auto p3tw_im = avx::mul(p3.real, tw0.imag);

                    p1tw_re = avx::fnmadd(p1.imag, tw0.imag, p1tw_re);
                    p3tw_re = avx::fnmadd(p3.imag, tw0.imag, p3tw_re);
                    p1tw_im = avx::fmadd(p1.imag, tw0.real, p1tw_im);
                    p3tw_im = avx::fmadd(p3.imag, tw0.real, p3tw_im);

                    avx::cx_reg<float> p1tw = {p1tw_re, p1tw_im};
                    avx::cx_reg<float> p3tw = {p3tw_re, p3tw_im};

                    auto a2 = avx::add(p2, p3tw);
                    auto a3 = avx::sub(p2, p3tw);
                    auto a0 = avx::add(p0, p1tw);
                    auto a1 = avx::sub(p0, p1tw);

                    auto a2tw_re = avx::mul(a1.real, tw1.real);
                    auto a3tw_re = avx::mul(a3.real, tw2.real);
                    auto a2tw_im = avx::mul(a1.real, tw1.imag);
                    auto a3tw_im = avx::mul(a3.real, tw2.imag);

                    a2tw_re = avx::fnmadd(a2.imag, tw1.imag, a2tw_re);
                    a3tw_re = avx::fnmadd(a3.imag, tw2.imag, a3tw_re);
                    a2tw_im = avx::fmadd(a2.imag, tw1.real, a2tw_im);
                    a3tw_im = avx::fmadd(a3.imag, tw2.real, a3tw_im);

                    avx::cx_reg<float> a2tw = {a2tw_re, a2tw_im};
                    avx::cx_reg<float> a3tw = {a3tw_re, a3tw_im};

                    auto b0 = avx::add(a0, a2tw);
                    auto b1 = avx::sub(a0, a2tw);
                    auto b2 = avx::add(a1, a3tw);
                    auto b3 = avx::sub(a1, a3tw);

                    cxstore<PackSize>(ptr0, b0);
                    cxstore<PackSize>(ptr1, b1);
                    cxstore<PackSize>(ptr2, b2);
                    cxstore<PackSize>(ptr3, b3);

                    offset += l_size * 2;
                }
            }

            l_size *= 4;
            group_size /= 4;
            n_groups *= 4;
        }

        if (l_size < size())
        {
        }
    };

private:
    static constexpr auto pidx(std::size_t idx) -> std::size_t
    {
        return idx + idx / PackSize * PackSize;
    }

    static constexpr std::size_t reg_size = 32 / sizeof(real_type);

    static constexpr auto log2i(std::size_t num) -> std::size_t
    {
        std::size_t order = 0;
        while ((num >>= 1U) != 0)
        {
            order++;
        }
        return order;
    }

    static constexpr auto reverse_bit_order(uint64_t num, uint64_t depth) -> uint64_t
    {
        num = num >> 32 | num << 32;
        num = (num & 0xFFFF0000FFFF0000) >> 16 | (num & 0x0000FFFF0000FFFF) << 16;
        num = (num & 0xFF00FF00FF00FF00) >> 8 | (num & 0x00FF00FF00FF00FF) << 8;
        num = (num & 0xF0F0F0F0F0F0F0F0) >> 4 | (num & 0x0F0F0F0F0F0F0F0F) << 4;
        num = (num & 0xCCCCCCCCCCCCCCCC) >> 2 | (num & 0x3333333333333333) << 2;
        num = (num & 0xAAAAAAAAAAAAAAAA) >> 1 | (num & 0x5555555555555555) << 1;
        return num >> (64 - depth);
    }

    /**
     * @brief Returns number of unique bit-reversed pairs from 0 to max-1
     *
     * @param max
     * @return std::size_t
     */
    static constexpr auto n_reversals(std::size_t max) -> std::size_t
    {
        return max - (1U << (log2i(max + 1) / 2));
    }

    static inline auto wnk(std::size_t n, std::size_t k) -> std::complex<real_type>
    {
        constexpr double pi = 3.14159265358979323846;
        return exp(std::complex<real_type>(0,
                                           -2 * pi * static_cast<double>(k) /
                                               static_cast<double>(n)));
    }

    static auto get_sort(std::size_t fft_size, sort_allocator_type allocator)
        -> std::vector<std::size_t, sort_allocator_type>
    {
        const auto packed_sort_size = fft_size / reg_size / reg_size;
        const auto order            = log2i(packed_sort_size);
        auto       sort             = std::vector<std::size_t, sort_allocator_type>();
        sort.reserve(packed_sort_size);

        for (uint i = 0; i < packed_sort_size / 2; ++i)
        {
            if (i == reverse_bit_order(i, order))
            {
                continue;
            }
            sort.push_back(pidx(i * reg_size));
            sort.push_back(pidx(reverse_bit_order(i, order) * reg_size));
        }
        for (uint i = 0; i < packed_sort_size; ++i)
        {
            if (i == reverse_bit_order(i, order))
            {
                sort.push_back(pidx(i * reg_size));
            }
        }
        return sort;
    }

    static auto get_twiddles(std::size_t fft_size, allocator_type allocator)
        -> pcx::vector<real_type, pack_size, allocator_type>
    {
        const auto depth = log2i(fft_size);

        const std::size_t n_twiddles = 8 * ((1U << (depth - 3)) - 1U);

        auto twiddles =
            pcx::vector<real_type, pack_size, allocator_type>(n_twiddles, allocator);

        auto tw_it       = twiddles.begin();
        uint l_size      = 16;
        uint small_i_max = 1;
        //
        //         if (depth % 2 == 0)
        //         {
        //             for (uint k = 0; k < 8; ++k)
        //             {
        //                 *(tw_it++) = wnk(l_fft_size, k);
        //             }
        //             l_fft_size *= 2;
        //             small_i_max *= 2;
        //         }

        while (l_size < fft_size)
        {
            for (uint small_i = 0; small_i < small_i_max; ++small_i)
            {
                for (uint k = 0; k < reg_size; ++k)
                {
                    *(tw_it++) = wnk(l_size, k + small_i * reg_size);
                }
                for (uint k = 0; k < reg_size; ++k)
                {
                    *(tw_it++) = wnk(l_size * 2UL, k + small_i * reg_size);
                }
                for (uint k = 0; k < reg_size; ++k)
                {
                    *(tw_it++) = wnk(l_size * 2UL, k + small_i * reg_size + l_size / 2);
                }
            }
            small_i_max *= 4;
            l_size *= 4;
        }
        return twiddles;
    }
};


}    // namespace pcx
#endif