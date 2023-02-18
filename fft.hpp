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

template<typename T,
         typename Allocator   = std::allocator<T>,
         std::size_t PackSize = 32 / sizeof(T),
         std::size_t Size     = pcx::dynamic_size>
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


    template<typename VAllocator>
    void operator()(pcx::vector<T, VAllocator, PackSize>& test_vecor)
    {
        fft_internal(test_vecor);
    };

private:
    [[no_unique_address]] size_t                            m_size;
    const std::vector<std::size_t, sort_allocator_type>     m_sort;
    const pcx::vector<real_type, allocator_type, pack_size> m_twiddles;

public:
    template<typename VAllocator>
    void fft_internal(pcx::vector<T, VAllocator, PackSize>& vector)
        requires std::same_as<T, float>
    {
        const auto sq2 = wnk(8, 1);

        auto twsq2 = avx::broadcast(sq2.real());

        auto* data_ptr    = &vector[0];
        auto* twiddle_ptr = &m_twiddles[0];

        const auto sh0 = 0;
        const auto sh1 = pidx(1 * size() / 8);
        const auto sh2 = pidx(2 * size() / 8);
        const auto sh3 = pidx(3 * size() / 8);
        const auto sh4 = pidx(4 * size() / 8);
        const auto sh5 = pidx(5 * size() / 8);
        const auto sh6 = pidx(6 * size() / 8);
        const auto sh7 = pidx(7 * size() / 8);

        for (uint i = 0; i < n_reversals(size() / 64); i += 2)
        {
            using reg_t = avx::cx_reg<float>;

            auto offset_first  = m_sort[i];
            auto offset_second = m_sort[i + 1];

            auto p1 = avx::cxload<PackSize>(data_ptr + sh1 + offset_first);
            auto p5 = avx::cxload<PackSize>(data_ptr + sh5 + offset_first);
            auto p3 = avx::cxload<PackSize>(data_ptr + sh3 + offset_first);
            auto p7 = avx::cxload<PackSize>(data_ptr + sh7 + offset_first);

            auto a5 = avx::sub(p1, p5);
            auto a1 = avx::add(p1, p5);
            auto a7 = avx::sub(p3, p7);
            auto a3 = avx::add(p3, p7);

            reg_t b5 = {avx::add(a5.real, a7.imag), avx::sub(a5.imag, a7.real)};
            reg_t b7 = {avx::sub(a5.real, a7.imag), avx::add(a5.imag, a7.real)};

            reg_t b5_tw = {avx::add(b5.real, b5.imag), avx::sub(b5.imag, b5.real)};
            reg_t b7_tw = {avx::sub(b7.real, b7.imag), avx::add(b7.real, b7.imag)};

            auto b1 = avx::add(a1, a3);
            auto b3 = avx::sub(a1, a3);

            b5_tw = avx::mul(b5_tw, twsq2);
            b7_tw = avx::mul(b7_tw, twsq2);

            auto p0 = avx::cxload<PackSize>(data_ptr + sh0 + offset_first);
            auto p4 = avx::cxload<PackSize>(data_ptr + sh4 + offset_first);
            auto p2 = avx::cxload<PackSize>(data_ptr + sh2 + offset_first);
            auto p6 = avx::cxload<PackSize>(data_ptr + sh6 + offset_first);

            auto a0 = avx::add(p0, p4);
            auto a4 = avx::sub(p0, p4);
            auto a2 = avx::add(p2, p6);
            auto a6 = avx::sub(p2, p6);

            auto  b0 = avx::add(a0, a2);
            auto  b2 = avx::sub(a0, a2);
            reg_t b4 = {avx::add(a4.real, a6.imag), avx::sub(a4.imag, a6.real)};
            reg_t b6 = {avx::sub(a4.real, a6.imag), avx::add(a4.imag, a6.real)};

            auto  c0 = avx::add(b0, b1);
            auto  c1 = avx::sub(b0, b1);
            reg_t c2 = {avx::add(b2.real, b3.imag), avx::sub(b2.imag, b3.real)};
            reg_t c3 = {avx::sub(b2.real, b3.imag), avx::add(b2.imag, b3.real)};

            reg_t c4 = avx::add(b4, b5_tw);
            reg_t c5 = avx::sub(b4, b5_tw);
            reg_t c6 = avx::sub(b6, b7_tw);
            reg_t c7 = avx::add(b6, b7_tw);

            auto sha0re = _mm256_castps_pd(_mm256_unpacklo_ps(c0.real, c4.real));
            auto sha4re = _mm256_castps_pd(_mm256_unpackhi_ps(c0.real, c4.real));
            auto sha0im = _mm256_castps_pd(_mm256_unpacklo_ps(c0.imag, c4.imag));
            auto sha4im = _mm256_castps_pd(_mm256_unpackhi_ps(c0.imag, c4.imag));
            auto sha2re = _mm256_castps_pd(_mm256_unpacklo_ps(c2.real, c6.real));
            auto sha6re = _mm256_castps_pd(_mm256_unpackhi_ps(c2.real, c6.real));
            auto sha2im = _mm256_castps_pd(_mm256_unpacklo_ps(c2.imag, c6.imag));
            auto sha6im = _mm256_castps_pd(_mm256_unpackhi_ps(c2.imag, c6.imag));

            auto sha1re = _mm256_castps_pd(_mm256_unpacklo_ps(c1.real, c5.real));
            auto sha5re = _mm256_castps_pd(_mm256_unpackhi_ps(c1.real, c5.real));
            auto sha1im = _mm256_castps_pd(_mm256_unpacklo_ps(c1.imag, c5.imag));
            auto sha5im = _mm256_castps_pd(_mm256_unpackhi_ps(c1.imag, c5.imag));
            auto sha3re = _mm256_castps_pd(_mm256_unpacklo_ps(c3.real, c7.real));
            auto sha7re = _mm256_castps_pd(_mm256_unpackhi_ps(c3.real, c7.real));
            auto sha3im = _mm256_castps_pd(_mm256_unpacklo_ps(c3.imag, c7.imag));
            auto sha7im = _mm256_castps_pd(_mm256_unpackhi_ps(c3.imag, c7.imag));

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
            auto shc2re = _mm256_permute2f128_ps(shb2re, shb3re, 0b00100000);
            auto shc3re = _mm256_permute2f128_ps(shb2re, shb3re, 0b00110001);
            auto shc2im = _mm256_permute2f128_ps(shb2im, shb3im, 0b00100000);
            auto shc3im = _mm256_permute2f128_ps(shb2im, shb3im, 0b00110001);

            auto q0 = avx::cxload<PackSize>(data_ptr + sh0 + offset_second);
            _mm256_storeu_ps(data_ptr + sh0 + offset_second, shc0re);
            _mm256_storeu_ps(data_ptr + sh0 + offset_second + PackSize, shc0im);
            auto q1 = avx::cxload<PackSize>(data_ptr + sh1 + offset_second);
            _mm256_storeu_ps(data_ptr + sh1 + offset_second, shc1re);
            _mm256_storeu_ps(data_ptr + sh1 + offset_second + PackSize, shc1im);
            auto q4 = avx::cxload<PackSize>(data_ptr + sh4 + offset_second);
            _mm256_storeu_ps(data_ptr + sh4 + offset_second, shc2re);
            _mm256_storeu_ps(data_ptr + sh4 + offset_second + PackSize, shc2im);
            auto q5 = avx::cxload<PackSize>(data_ptr + sh5 + offset_second);
            _mm256_storeu_ps(data_ptr + sh5 + offset_second, shc3re);
            _mm256_storeu_ps(data_ptr + sh5 + offset_second + PackSize, shc3im);

            auto shb4re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha4re, sha6re));
            auto shb6re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha4re, sha6re));
            auto shb4im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha4im, sha6im));
            auto shb6im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha4im, sha6im));

            auto shb5re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha5re, sha7re));
            auto shb7re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha5re, sha7re));
            auto shb5im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha5im, sha7im));
            auto shb7im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha5im, sha7im));

            auto shc4re = _mm256_permute2f128_ps(shb4re, shb5re, 0b00100000);
            auto shc5re = _mm256_permute2f128_ps(shb4re, shb5re, 0b00110001);
            auto shc4im = _mm256_permute2f128_ps(shb4im, shb5im, 0b00100000);
            auto shc5im = _mm256_permute2f128_ps(shb4im, shb5im, 0b00110001);

            auto q2 = avx::cxload<PackSize>(data_ptr + sh2 + offset_second);
            _mm256_storeu_ps(data_ptr + sh2 + offset_second, shc4re);
            _mm256_storeu_ps(data_ptr + sh2 + offset_second + PackSize, shc4im);
            auto q3 = avx::cxload<PackSize>(data_ptr + sh3 + offset_second);
            _mm256_storeu_ps(data_ptr + sh3 + offset_second, shc5re);
            _mm256_storeu_ps(data_ptr + sh3 + offset_second + PackSize, shc5im);

            auto shc6re = _mm256_permute2f128_ps(shb6re, shb7re, 0b00100000);
            auto shc7re = _mm256_permute2f128_ps(shb6re, shb7re, 0b00110001);
            auto shc6im = _mm256_permute2f128_ps(shb6im, shb7im, 0b00100000);
            auto shc7im = _mm256_permute2f128_ps(shb6im, shb7im, 0b00110001);

            auto q6 = avx::cxload<PackSize>(data_ptr + sh6 + offset_second);
            _mm256_storeu_ps(data_ptr + sh6 + offset_second, shc6re);
            _mm256_storeu_ps(data_ptr + sh6 + offset_second + PackSize, shc6im);
            auto q7 = avx::cxload<PackSize>(data_ptr + sh7 + offset_second);
            _mm256_storeu_ps(data_ptr + sh7 + offset_second, shc7re);
            _mm256_storeu_ps(data_ptr + sh7 + offset_second + PackSize, shc7im);

            auto x5 = avx::sub(q1, q5);
            auto x1 = avx::add(q1, q5);
            auto x7 = avx::sub(q3, q7);
            auto x3 = avx::add(q3, q7);

            reg_t y5 = {avx::add(x5.real, x7.imag), avx::sub(x5.imag, x7.real)};
            reg_t y7 = {avx::sub(x5.real, x7.imag), avx::add(x5.imag, x7.real)};

            auto y1 = avx::add(x1, x3);

            reg_t y5_tw = {avx::add(y5.real, y5.imag), avx::sub(y5.imag, y5.real)};
            reg_t y7_tw = {avx::sub(y7.real, y7.imag), avx::add(y7.real, y7.imag)};

            auto y3 = avx::sub(x1, x3);

            y5_tw = avx::mul(y5_tw, twsq2);
            y7_tw = avx::mul(y7_tw, twsq2);

            auto x0 = avx::add(q0, q4);
            auto x4 = avx::sub(q0, q4);
            auto x2 = avx::add(q2, q6);
            auto x6 = avx::sub(q2, q6);

            auto  y0 = avx::add(x0, x2);
            auto  y2 = avx::sub(x0, x2);
            reg_t y4 = {avx::add(x4.real, x6.imag), avx::sub(x4.imag, x6.real)};
            reg_t y6 = {avx::sub(x4.real, x6.imag), avx::add(x4.imag, x6.real)};

            auto z0 = avx::add(y0, y1);
            auto z1 = avx::sub(y0, y1);

            reg_t z2 = {avx::add(y2.real, y3.imag), avx::sub(y2.imag, y3.real)};
            reg_t z3 = {avx::sub(y2.real, y3.imag), avx::add(y2.imag, y3.real)};

            reg_t z4 = avx::add(y4, y5_tw);
            reg_t z5 = avx::sub(y4, y5_tw);
            reg_t z6 = avx::sub(y6, y7_tw);
            reg_t z7 = avx::add(y6, y7_tw);

            auto shx0re = _mm256_castps_pd(_mm256_unpacklo_ps(z0.real, z4.real));
            auto shx4re = _mm256_castps_pd(_mm256_unpackhi_ps(z0.real, z4.real));
            auto shx0im = _mm256_castps_pd(_mm256_unpacklo_ps(z0.imag, z4.imag));
            auto shx4im = _mm256_castps_pd(_mm256_unpackhi_ps(z0.imag, z4.imag));
            auto shx1re = _mm256_castps_pd(_mm256_unpacklo_ps(z1.real, z5.real));
            auto shx5re = _mm256_castps_pd(_mm256_unpackhi_ps(z1.real, z5.real));
            auto shx1im = _mm256_castps_pd(_mm256_unpacklo_ps(z1.imag, z5.imag));
            auto shx5im = _mm256_castps_pd(_mm256_unpackhi_ps(z1.imag, z5.imag));

            auto shx2re = _mm256_castps_pd(_mm256_unpacklo_ps(z2.real, z6.real));
            auto shx6re = _mm256_castps_pd(_mm256_unpackhi_ps(z2.real, z6.real));
            auto shx2im = _mm256_castps_pd(_mm256_unpacklo_ps(z2.imag, z6.imag));
            auto shx6im = _mm256_castps_pd(_mm256_unpackhi_ps(z2.imag, z6.imag));
            auto shx3re = _mm256_castps_pd(_mm256_unpacklo_ps(z3.real, z7.real));
            auto shx7re = _mm256_castps_pd(_mm256_unpackhi_ps(z3.real, z7.real));
            auto shx3im = _mm256_castps_pd(_mm256_unpacklo_ps(z3.imag, z7.imag));
            auto shx7im = _mm256_castps_pd(_mm256_unpackhi_ps(z3.imag, z7.imag));

            auto shy0re = _mm256_castpd_ps(_mm256_unpacklo_pd(shx0re, shx2re));
            auto shy2re = _mm256_castpd_ps(_mm256_unpackhi_pd(shx0re, shx2re));
            auto shy0im = _mm256_castpd_ps(_mm256_unpacklo_pd(shx0im, shx2im));
            auto shy2im = _mm256_castpd_ps(_mm256_unpackhi_pd(shx0im, shx2im));
            auto shy1re = _mm256_castpd_ps(_mm256_unpacklo_pd(shx1re, shx3re));
            auto shy3re = _mm256_castpd_ps(_mm256_unpackhi_pd(shx1re, shx3re));
            auto shy1im = _mm256_castpd_ps(_mm256_unpacklo_pd(shx1im, shx3im));
            auto shy3im = _mm256_castpd_ps(_mm256_unpackhi_pd(shx1im, shx3im));

            auto shz0re = _mm256_permute2f128_ps(shy0re, shy1re, 0b00100000);
            auto shz1re = _mm256_permute2f128_ps(shy0re, shy1re, 0b00110001);
            auto shz0im = _mm256_permute2f128_ps(shy0im, shy1im, 0b00100000);
            auto shz1im = _mm256_permute2f128_ps(shy0im, shy1im, 0b00110001);
            auto shz2re = _mm256_permute2f128_ps(shy2re, shy3re, 0b00100000);
            auto shz3re = _mm256_permute2f128_ps(shy2re, shy3re, 0b00110001);
            auto shz2im = _mm256_permute2f128_ps(shy2im, shy3im, 0b00100000);
            auto shz3im = _mm256_permute2f128_ps(shy2im, shy3im, 0b00110001);

            _mm256_storeu_ps(data_ptr + sh0 + offset_first, shz0re);
            _mm256_storeu_ps(data_ptr + sh0 + offset_first + PackSize, shz0im);
            _mm256_storeu_ps(data_ptr + sh1 + offset_first, shz1re);
            _mm256_storeu_ps(data_ptr + sh1 + offset_first + PackSize, shz1im);
            _mm256_storeu_ps(data_ptr + sh4 + offset_first, shz2re);
            _mm256_storeu_ps(data_ptr + sh4 + offset_first + PackSize, shz2im);
            _mm256_storeu_ps(data_ptr + sh5 + offset_first, shz3re);
            _mm256_storeu_ps(data_ptr + sh5 + offset_first + PackSize, shz3im);

            auto shy4re = _mm256_castpd_ps(_mm256_unpacklo_pd(shx4re, shx6re));
            auto shy6re = _mm256_castpd_ps(_mm256_unpackhi_pd(shx4re, shx6re));
            auto shy4im = _mm256_castpd_ps(_mm256_unpacklo_pd(shx4im, shx6im));
            auto shy6im = _mm256_castpd_ps(_mm256_unpackhi_pd(shx4im, shx6im));
            auto shy5re = _mm256_castpd_ps(_mm256_unpacklo_pd(shx5re, shx7re));
            auto shy7re = _mm256_castpd_ps(_mm256_unpackhi_pd(shx5re, shx7re));
            auto shy5im = _mm256_castpd_ps(_mm256_unpacklo_pd(shx5im, shx7im));
            auto shy7im = _mm256_castpd_ps(_mm256_unpackhi_pd(shx5im, shx7im));

            auto shz4re = _mm256_permute2f128_ps(shy4re, shy5re, 0b00100000);
            auto shz5re = _mm256_permute2f128_ps(shy4re, shy5re, 0b00110001);
            auto shz4im = _mm256_permute2f128_ps(shy4im, shy5im, 0b00100000);
            auto shz5im = _mm256_permute2f128_ps(shy4im, shy5im, 0b00110001);
            auto shz6re = _mm256_permute2f128_ps(shy6re, shy7re, 0b00100000);
            auto shz7re = _mm256_permute2f128_ps(shy6re, shy7re, 0b00110001);
            auto shz6im = _mm256_permute2f128_ps(shy6im, shy7im, 0b00100000);
            auto shz7im = _mm256_permute2f128_ps(shy6im, shy7im, 0b00110001);

            _mm256_storeu_ps(data_ptr + sh2 + offset_first, shz4re);
            _mm256_storeu_ps(data_ptr + sh2 + offset_first + PackSize, shz4im);
            _mm256_storeu_ps(data_ptr + sh3 + offset_first, shz5re);
            _mm256_storeu_ps(data_ptr + sh3 + offset_first + PackSize, shz5im);
            _mm256_storeu_ps(data_ptr + sh6 + offset_first, shz6re);
            _mm256_storeu_ps(data_ptr + sh6 + offset_first + PackSize, shz6im);
            _mm256_storeu_ps(data_ptr + sh7 + offset_first, shz7re);
            _mm256_storeu_ps(data_ptr + sh7 + offset_first + PackSize, shz7im);
        };

        for (uint i = n_reversals(size() / 64); i < size() / 64; ++i)
        {
            using reg_t = avx::cx_reg<float>;
            auto offset = m_sort[i];

            auto p1 = avx::cxload<PackSize>(data_ptr + sh1 + offset);
            auto p3 = avx::cxload<PackSize>(data_ptr + sh3 + offset);
            auto p5 = avx::cxload<PackSize>(data_ptr + sh5 + offset);
            auto p7 = avx::cxload<PackSize>(data_ptr + sh7 + offset);

            auto a5 = avx::sub(p1, p5);
            auto a1 = avx::add(p1, p5);
            auto a7 = avx::sub(p3, p7);
            auto a3 = avx::add(p3, p7);

            reg_t b5 = {avx::add(a5.real, a7.imag), avx::sub(a5.imag, a7.real)};
            reg_t b7 = {avx::sub(a5.real, a7.imag), avx::add(a5.imag, a7.real)};

            reg_t b5_tw = {avx::add(b5.real, b5.imag), avx::sub(b5.imag, b5.real)};
            reg_t b7_tw = {avx::sub(b7.real, b7.imag), avx::add(b7.real, b7.imag)};

            auto b1 = avx::add(a1, a3);
            auto b3 = avx::sub(a1, a3);

            b5_tw = avx::mul(b5_tw, twsq2);
            b7_tw = avx::mul(b7_tw, twsq2);

            auto p0 = avx::cxload<PackSize>(data_ptr + sh0 + offset);
            auto p2 = avx::cxload<PackSize>(data_ptr + sh2 + offset);
            auto p4 = avx::cxload<PackSize>(data_ptr + sh4 + offset);
            auto p6 = avx::cxload<PackSize>(data_ptr + sh6 + offset);

            auto a0 = avx::add(p0, p4);
            auto a4 = avx::sub(p0, p4);
            auto a2 = avx::add(p2, p6);
            auto a6 = avx::sub(p2, p6);

            auto  b0 = avx::add(a0, a2);
            auto  b2 = avx::sub(a0, a2);
            reg_t b4 = {avx::add(a4.real, a6.imag), avx::sub(a4.imag, a6.real)};
            reg_t b6 = {avx::sub(a4.real, a6.imag), avx::add(a4.imag, a6.real)};

            auto  c0 = avx::add(b0, b1);
            auto  c1 = avx::sub(b0, b1);
            reg_t c2 = {avx::add(b2.real, b3.imag), avx::sub(b2.imag, b3.real)};
            reg_t c3 = {avx::sub(b2.real, b3.imag), avx::add(b2.imag, b3.real)};

            reg_t c4 = avx::add(b4, b5_tw);
            reg_t c5 = avx::sub(b4, b5_tw);
            reg_t c6 = avx::sub(b6, b7_tw);
            reg_t c7 = avx::add(b6, b7_tw);

            auto sha0re = _mm256_castps_pd(_mm256_unpacklo_ps(c0.real, c4.real));
            auto sha4re = _mm256_castps_pd(_mm256_unpackhi_ps(c0.real, c4.real));
            auto sha0im = _mm256_castps_pd(_mm256_unpacklo_ps(c0.imag, c4.imag));
            auto sha4im = _mm256_castps_pd(_mm256_unpackhi_ps(c0.imag, c4.imag));
            auto sha2re = _mm256_castps_pd(_mm256_unpacklo_ps(c2.real, c6.real));
            auto sha6re = _mm256_castps_pd(_mm256_unpackhi_ps(c2.real, c6.real));
            auto sha2im = _mm256_castps_pd(_mm256_unpacklo_ps(c2.imag, c6.imag));
            auto sha6im = _mm256_castps_pd(_mm256_unpackhi_ps(c2.imag, c6.imag));

            auto sha1re = _mm256_castps_pd(_mm256_unpacklo_ps(c1.real, c5.real));
            auto sha5re = _mm256_castps_pd(_mm256_unpackhi_ps(c1.real, c5.real));
            auto sha1im = _mm256_castps_pd(_mm256_unpacklo_ps(c1.imag, c5.imag));
            auto sha5im = _mm256_castps_pd(_mm256_unpackhi_ps(c1.imag, c5.imag));
            auto sha3re = _mm256_castps_pd(_mm256_unpacklo_ps(c3.real, c7.real));
            auto sha7re = _mm256_castps_pd(_mm256_unpackhi_ps(c3.real, c7.real));
            auto sha3im = _mm256_castps_pd(_mm256_unpacklo_ps(c3.imag, c7.imag));
            auto sha7im = _mm256_castps_pd(_mm256_unpackhi_ps(c3.imag, c7.imag));

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
            auto shc2re = _mm256_permute2f128_ps(shb2re, shb3re, 0b00100000);
            auto shc3re = _mm256_permute2f128_ps(shb2re, shb3re, 0b00110001);
            auto shc2im = _mm256_permute2f128_ps(shb2im, shb3im, 0b00100000);
            auto shc3im = _mm256_permute2f128_ps(shb2im, shb3im, 0b00110001);

            _mm256_storeu_ps(data_ptr + sh0 + offset, shc0re);
            _mm256_storeu_ps(data_ptr + sh0 + offset + PackSize, shc0im);
            _mm256_storeu_ps(data_ptr + sh1 + offset, shc1re);
            _mm256_storeu_ps(data_ptr + sh1 + offset + PackSize, shc1im);
            _mm256_storeu_ps(data_ptr + sh4 + offset, shc2re);
            _mm256_storeu_ps(data_ptr + sh4 + offset + PackSize, shc2im);
            _mm256_storeu_ps(data_ptr + sh5 + offset, shc3re);
            _mm256_storeu_ps(data_ptr + sh5 + offset + PackSize, shc3im);

            auto shb4re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha4re, sha6re));
            auto shb6re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha4re, sha6re));
            auto shb4im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha4im, sha6im));
            auto shb6im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha4im, sha6im));
            auto shb5re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha5re, sha7re));
            auto shb7re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha5re, sha7re));
            auto shb5im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha5im, sha7im));
            auto shb7im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha5im, sha7im));

            auto shc4re = _mm256_permute2f128_ps(shb4re, shb5re, 0b00100000);
            auto shc5re = _mm256_permute2f128_ps(shb4re, shb5re, 0b00110001);
            auto shc4im = _mm256_permute2f128_ps(shb4im, shb5im, 0b00100000);
            auto shc5im = _mm256_permute2f128_ps(shb4im, shb5im, 0b00110001);
            auto shc6re = _mm256_permute2f128_ps(shb6re, shb7re, 0b00100000);
            auto shc7re = _mm256_permute2f128_ps(shb6re, shb7re, 0b00110001);
            auto shc6im = _mm256_permute2f128_ps(shb6im, shb7im, 0b00100000);
            auto shc7im = _mm256_permute2f128_ps(shb6im, shb7im, 0b00110001);

            _mm256_storeu_ps(data_ptr + sh2 + offset, shc4re);
            _mm256_storeu_ps(data_ptr + sh2 + offset + PackSize, shc4im);
            _mm256_storeu_ps(data_ptr + sh3 + offset, shc5re);
            _mm256_storeu_ps(data_ptr + sh3 + offset + PackSize, shc5im);
            _mm256_storeu_ps(data_ptr + sh6 + offset, shc6re);
            _mm256_storeu_ps(data_ptr + sh6 + offset + PackSize, shc6im);
            _mm256_storeu_ps(data_ptr + sh7 + offset, shc7re);
            _mm256_storeu_ps(data_ptr + sh7 + offset + PackSize, shc7im);
        }

        std::size_t l_size     = reg_size * 2;
        std::size_t group_size = size() / reg_size / 4;
        std::size_t n_groups   = 1;
        std::size_t tw_offset  = 0;

        while (l_size < size())
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset));
                const auto tw1 =
                    avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset + reg_size));
                const auto tw2 =
                    avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset + reg_size * 2));

                tw_offset += reg_size * 3;

                for (std::size_t i = 0; i < group_size; ++i)
                {
                    auto* ptr0 = data_ptr + pidx(offset);
                    auto* ptr1 = data_ptr + pidx(offset + l_size / 2);
                    auto* ptr2 = data_ptr + pidx(offset + l_size);
                    auto* ptr3 = data_ptr + pidx(offset + l_size / 2 * 3);

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

                    auto a2tw_re = avx::mul(a2.real, tw1.real);
                    auto a2tw_im = avx::mul(a2.real, tw1.imag);
                    auto a3tw_re = avx::mul(a3.real, tw2.real);
                    auto a3tw_im = avx::mul(a3.real, tw2.imag);

                    a2tw_re = avx::fnmadd(a2.imag, tw1.imag, a2tw_re);
                    a2tw_im = avx::fmadd(a2.imag, tw1.real, a2tw_im);
                    a3tw_re = avx::fnmadd(a3.imag, tw2.imag, a3tw_re);
                    a3tw_im = avx::fmadd(a3.imag, tw2.real, a3tw_im);

                    avx::cx_reg<float> a2tw = {a2tw_re, a2tw_im};
                    avx::cx_reg<float> a3tw = {a3tw_re, a3tw_im};

                    auto b0 = avx::add(a0, a2tw);
                    auto b2 = avx::sub(a0, a2tw);
                    auto b1 = avx::add(a1, a3tw);
                    auto b3 = avx::sub(a1, a3tw);

                    cxstore<PackSize>(ptr0, b0);
                    cxstore<PackSize>(ptr1, b1);
                    cxstore<PackSize>(ptr2, b2);
                    cxstore<PackSize>(ptr3, b3);

                    offset += l_size * 2;
                }
            }

            l_size *= 4;
            n_groups *= 4;
            group_size /= 4;
        }

        if (l_size == size())
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset));

                tw_offset += reg_size;

                auto* ptr0 = data_ptr + pidx(offset);
                auto* ptr1 = data_ptr + pidx(offset + l_size / 2);

                auto p1 = avx::cxload<PackSize>(ptr1);
                auto p0 = avx::cxload<PackSize>(ptr0);

                auto p1tw = avx::mul(p1, tw0);

                auto a0 = avx::add(p0, p1tw);
                auto a1 = avx::sub(p0, p1tw);

                cxstore<PackSize>(ptr0, a0);
                cxstore<PackSize>(ptr1, a1);
            }
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
        return max - (1U << ((log2i(max) + 1) / 2));
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
        auto       sort = std::vector<std::size_t, sort_allocator_type>(allocator);
        sort.reserve(packed_sort_size);

        for (uint i = 0; i < packed_sort_size; ++i)
        {
            if (i >= reverse_bit_order(i, order))
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
        -> pcx::vector<real_type, allocator_type, pack_size>
    {
        const auto depth = log2i(fft_size);

        const std::size_t n_twiddles = 8 * ((1U << (depth - 3)) - 1U);

        auto twiddles =
            pcx::vector<real_type, allocator_type, pack_size>(n_twiddles, allocator);

        auto tw_it = twiddles.begin();

        std::size_t l_size   = reg_size * 2;
        std::size_t n_groups = 1;

        while (l_size < fft_size)
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                for (uint k = 0; k < reg_size; ++k)
                {
                    *(tw_it++) = wnk(l_size, k + i_group * reg_size);
                }
                for (uint k = 0; k < reg_size; ++k)
                {
                    *(tw_it++) = wnk(l_size * 2UL, k + i_group * reg_size);
                }
                for (uint k = 0; k < reg_size; ++k)
                {
                    *(tw_it++) = wnk(l_size * 2UL, k + i_group * reg_size + l_size / 2);
                }
            }
            l_size *= 4;
            n_groups *= 4;
        }

        if (l_size == fft_size)
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                for (uint k = 0; k < reg_size; ++k)
                {
                    *(tw_it++) = wnk(l_size, k + i_group * reg_size);
                }
            }
        }
        return twiddles;
    }
};


}    // namespace pcx
#endif