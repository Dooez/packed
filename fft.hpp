#ifndef FFT_HPP
#define FFT_HPP

#include "vector.hpp"

#include <memory>

namespace pcx {
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

    struct packed_fft_core
    {
        packed_fft_core(std::size_t fft_size, allocator_type allocator)
            requires(Size == pcx::dynamic_size)
        : sort(get_sort(fft_size, static_cast<sort_allocator_type>(allocator)))
        , twiddles(get_twiddles(fft_size, allocator)){};

        std::pair<std::vector<std::size_t, sort_allocator_type>, std::size_t> sort;
        pcx::vector<real_type, pack_size, allocator_type>                     twiddles;
    };

    using core_allocator_type = typename std::allocator_traits<
        allocator_type>::template rebind_alloc<packed_fft_core>;

public:
    fft_unit(std::size_t fft_size, allocator_type allocator = allocator_type())
    {
        auto core_alloc = core_allocator_type(allocator);
        m_core = std::allocate_shared<packed_fft_core>(core_alloc, fft_size, allocator);
    };

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
    void fft_unit_test(const pcx::vector<VT, VPackSize, VAllocator>& test_vecor){};

private:
    [[no_unique_address]] size_t                      m_size;
    std::shared_ptr<packed_fft_core>                  m_core{};
    std::vector<std::size_t, sort_allocator_type>     m_sort;
    pcx::vector<real_type, pack_size, allocator_type> m_twiddles;

public:
    template<std::size_t VPackSize, typename VAllocator>
    void fft_internal(pcx::vector<T, VPackSize, VAllocator>& vector)
    {
        auto size = vector.size();

        constexpr const double pi  = 3.14159265358979323846;
        const auto             sq2 = std::exp(std::complex<float>(0, pi / 4));

        auto  twsq2       = avx::broadcast(sq2.real());
        auto* data_ptr    = &vector[0];
        auto* twiddle_ptr = &m_twiddles[0];

        for (uint i = 0; i < n_reversals(size / 64); i += 2)
        {
            auto idx_first  = m_sort[i];
            auto idx_second = m_sort[i + 1];

            auto p1re = _mm256_loadu_ps(data_ptr + 1 * size / 8 + idx_first);
            auto p1im = _mm256_loadu_ps(data_ptr + 1 * size / 8 + idx_first + PackSize);
            auto p5re = _mm256_loadu_ps(data_ptr + 5 * size / 8 + idx_first);
            auto p5im = _mm256_loadu_ps(data_ptr + 5 * size / 8 + idx_first + PackSize);
            auto p3re = _mm256_loadu_ps(data_ptr + 3 * size / 8 + idx_first);
            auto p3im = _mm256_loadu_ps(data_ptr + 3 * size / 8 + idx_first + PackSize);
            auto p7re = _mm256_loadu_ps(data_ptr + 7 * size / 8 + idx_first);
            auto p7im = _mm256_loadu_ps(data_ptr + 7 * size / 8 + idx_first + PackSize);

            auto a1re = _mm256_add_ps(p1re, p5re);
            auto a5re = _mm256_sub_ps(p1re, p5re);
            auto a1im = _mm256_add_ps(p1im, p5im);
            auto a5im = _mm256_sub_ps(p1im, p5im);
            auto a3re = _mm256_add_ps(p3re, p7re);
            auto a7re = _mm256_sub_ps(p3re, p7re);
            auto a3im = _mm256_add_ps(p3im, p7im);
            auto a7im = _mm256_sub_ps(p3im, p7im);

            auto b5re    = _mm256_add_ps(a5re, a7im);
            auto b7re    = _mm256_sub_ps(a5re, a7im);
            auto b5im    = _mm256_add_ps(a5im, a7re);
            auto b7im    = _mm256_sub_ps(a5im, a7re);
            auto b5re_tw = _mm256_add_ps(b5re, b5im);
            auto b5im_tw = _mm256_sub_ps(b5im, b5re);
            auto b7re_tw = _mm256_add_ps(b7re, b7im);
            auto b7im_tw = _mm256_sub_ps(b7im, b7re);

            auto b1re = _mm256_add_ps(a1re, a3re);
            auto b3re = _mm256_sub_ps(a1re, a3re);
            auto b1im = _mm256_add_ps(a1im, a3im);
            auto b3im = _mm256_sub_ps(a1im, a3im);
            b5re_tw   = _mm256_mul_ps(b5re_tw, twsq2);
            b5im_tw   = _mm256_mul_ps(b5im_tw, twsq2);
            b7re_tw   = _mm256_mul_ps(b7re_tw, twsq2);
            b7im_tw   = _mm256_mul_ps(b7im_tw, twsq2);

            auto p0re = _mm256_loadu_ps(data_ptr + 0 * size / 8 + idx_first);
            auto p0im = _mm256_loadu_ps(data_ptr + 0 * size / 8 + idx_first + PackSize);
            auto p4re = _mm256_loadu_ps(data_ptr + 4 * size / 8 + idx_first);
            auto p4im = _mm256_loadu_ps(data_ptr + 4 * size / 8 + idx_first + PackSize);
            auto p2re = _mm256_loadu_ps(data_ptr + 2 * size / 8 + idx_first);
            auto p2im = _mm256_loadu_ps(data_ptr + 2 * size / 8 + idx_first + PackSize);
            auto p6re = _mm256_loadu_ps(data_ptr + 6 * size / 8 + idx_first);
            auto p6im = _mm256_loadu_ps(data_ptr + 6 * size / 8 + idx_first + PackSize);

            auto a0re = _mm256_add_ps(p0re, p4re);
            auto a4re = _mm256_sub_ps(p0re, p4re);
            auto a0im = _mm256_add_ps(p0im, p4im);
            auto a4im = _mm256_sub_ps(p0im, p4im);
            auto a2re = _mm256_add_ps(p2re, p6re);
            auto a6re = _mm256_sub_ps(p2re, p6re);
            auto a2im = _mm256_add_ps(p2im, p6im);
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
            auto sha3re = _mm256_castps_pd(_mm256_unpacklo_ps(c3re, c7re));
            auto sha7re = _mm256_castps_pd(_mm256_unpackhi_ps(c3re, c7re));
            auto sha0re = _mm256_castps_pd(_mm256_unpacklo_ps(c0re, c4re));
            auto sha4re = _mm256_castps_pd(_mm256_unpackhi_ps(c0re, c4re));
            auto sha2re = _mm256_castps_pd(_mm256_unpacklo_ps(c2re, c6re));
            auto sha6re = _mm256_castps_pd(_mm256_unpackhi_ps(c2re, c6re));

            auto shb0re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha0re, sha2re));
            auto shb2re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha0re, sha2re));
            auto shb4re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha4re, sha6re));
            auto shb6re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha4re, sha6re));
            auto shb5re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha5re, sha7re));
            auto shb7re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha5re, sha7re));
            auto shb1re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha1re, sha3re));
            auto shb3re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha1re, sha3re));

            auto shc0re = _mm256_permute2f128_ps(shb0re, shb1re, 0b00100000);
            auto shc1re = _mm256_permute2f128_ps(shb0re, shb1re, 0b00110001);
            auto shc2re = _mm256_permute2f128_ps(shb2re, shb3re, 0b00100000);
            auto shc3re = _mm256_permute2f128_ps(shb2re, shb3re, 0b00110001);
            auto shc4re = _mm256_permute2f128_ps(shb4re, shb5re, 0b00100000);
            auto shc5re = _mm256_permute2f128_ps(shb4re, shb5re, 0b00110001);
            auto shc6re = _mm256_permute2f128_ps(shb6re, shb7re, 0b00100000);
            auto shc7re = _mm256_permute2f128_ps(shb6re, shb7re, 0b00110001);

            auto sha0im = _mm256_castps_pd(_mm256_unpacklo_ps(c0im, c4im));
            auto sha4im = _mm256_castps_pd(_mm256_unpackhi_ps(c0im, c4im));
            auto sha2im = _mm256_castps_pd(_mm256_unpacklo_ps(c2im, c6im));
            auto sha6im = _mm256_castps_pd(_mm256_unpackhi_ps(c2im, c6im));
            auto sha1im = _mm256_castps_pd(_mm256_unpacklo_ps(c1im, c5im));
            auto sha5im = _mm256_castps_pd(_mm256_unpackhi_ps(c1im, c5im));
            auto sha3im = _mm256_castps_pd(_mm256_unpacklo_ps(c3im, c7im));
            auto sha7im = _mm256_castps_pd(_mm256_unpackhi_ps(c3im, c7im));

            auto shb0im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha0im, sha2im));
            auto shb2im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha0im, sha2im));
            auto shb4im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha4im, sha6im));
            auto shb6im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha4im, sha6im));
            auto shb1im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha1im, sha3im));
            auto shb3im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha1im, sha3im));
            auto shb5im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha5im, sha7im));
            auto shb7im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha5im, sha7im));

            auto shc0im = _mm256_permute2f128_ps(shb0im, shb1im, 0b00100000);
            auto shc1im = _mm256_permute2f128_ps(shb0im, shb1im, 0b00110001);
            auto shc2im = _mm256_permute2f128_ps(shb2im, shb3im, 0b00100000);
            auto shc3im = _mm256_permute2f128_ps(shb2im, shb3im, 0b00110001);
            auto shc4im = _mm256_permute2f128_ps(shb4im, shb5im, 0b00100000);
            auto shc5im = _mm256_permute2f128_ps(shb4im, shb5im, 0b00110001);
            auto shc6im = _mm256_permute2f128_ps(shb6im, shb7im, 0b00100000);
            auto shc7im = _mm256_permute2f128_ps(shb6im, shb7im, 0b00110001);

            auto q0re = _mm256_loadu_ps(data_ptr + 0 * size / 8 + idx_second);
            _mm256_storeu_ps(data_ptr + 0 * size / 8 + idx_second, shc0re);
            auto q0im = _mm256_loadu_ps(data_ptr + 0 * size / 8 + idx_second + PackSize);
            _mm256_storeu_ps(data_ptr + 0 * size / 8 + idx_second + PackSize, shc0im);
            auto q2re = _mm256_loadu_ps(data_ptr + 2 * size / 8 + idx_second);
            _mm256_storeu_ps(data_ptr + 2 * size / 8 + idx_second, shc2re);
            auto q2im = _mm256_loadu_ps(data_ptr + 2 * size / 8 + idx_second + PackSize);
            _mm256_storeu_ps(data_ptr + 2 * size / 8 + idx_second + PackSize, shc2im);
            auto q4re = _mm256_loadu_ps(data_ptr + 4 * size / 8 + idx_second);
            _mm256_storeu_ps(data_ptr + 4 * size / 8 + idx_second, shc4re);
            auto q4im = _mm256_loadu_ps(data_ptr + 4 * size / 8 + idx_second + PackSize);
            _mm256_storeu_ps(data_ptr + 4 * size / 8 + idx_second + PackSize, shc4im);
            auto q6re = _mm256_loadu_ps(data_ptr + 6 * size / 8 + idx_second);
            _mm256_storeu_ps(data_ptr + 6 * size / 8 + idx_second, shc6re);
            auto q6im = _mm256_loadu_ps(data_ptr + 6 * size / 8 + idx_second + PackSize);
            _mm256_storeu_ps(data_ptr + 6 * size / 8 + idx_second + PackSize, shc6im);

            auto q1re = _mm256_loadu_ps(data_ptr + 1 * size / 8 + idx_second);
            _mm256_storeu_ps(data_ptr + 1 * size / 8 + idx_second, shc1re);
            auto q1im = _mm256_loadu_ps(data_ptr + 1 * size / 8 + idx_second + PackSize);
            _mm256_storeu_ps(data_ptr + 1 * size / 8 + idx_second + PackSize, shc1im);
            auto q5re = _mm256_loadu_ps(data_ptr + 5 * size / 8 + idx_second);
            _mm256_storeu_ps(data_ptr + 5 * size / 8 + idx_second, shc5re);
            auto q5im = _mm256_loadu_ps(data_ptr + 5 * size / 8 + idx_second + PackSize);
            _mm256_storeu_ps(data_ptr + 5 * size / 8 + idx_second + PackSize, shc5im);
            auto q3re = _mm256_loadu_ps(data_ptr + 3 * size / 8 + idx_second);
            _mm256_storeu_ps(data_ptr + 3 * size / 8 + idx_second, shc3re);
            auto q3im = _mm256_loadu_ps(data_ptr + 3 * size / 8 + idx_second + PackSize);
            _mm256_storeu_ps(data_ptr + 3 * size / 8 + idx_second + PackSize, shc3im);
            auto q7re = _mm256_loadu_ps(data_ptr + 7 * size / 8 + idx_second);
            _mm256_storeu_ps(data_ptr + 7 * size / 8 + idx_second, shc7re);
            auto q7im = _mm256_loadu_ps(data_ptr + 7 * size / 8 + idx_second + PackSize);
            _mm256_storeu_ps(data_ptr + 7 * size / 8 + idx_second + PackSize, shc7im);

            auto x1re = _mm256_add_ps(q1re, q5re);
            auto x5re = _mm256_sub_ps(q1re, q5re);
            auto x1im = _mm256_add_ps(q1im, q5im);
            auto x5im = _mm256_sub_ps(q1im, q5im);
            auto x3re = _mm256_add_ps(q3re, q7re);
            auto x7re = _mm256_sub_ps(q3re, q7re);
            auto x3im = _mm256_add_ps(q3im, q7im);
            auto x7im = _mm256_sub_ps(q3im, q7im);

            auto y5re    = _mm256_add_ps(x5re, x7im);
            auto y7re    = _mm256_sub_ps(x5re, x7im);
            auto y5im    = _mm256_add_ps(x5im, x7re);
            auto y7im    = _mm256_sub_ps(x5im, x7re);
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

            auto shy0re = _mm256_castpd_ps(_mm256_unpacklo_pd(shx0re, shx2re));
            auto shy2re = _mm256_castpd_ps(_mm256_unpackhi_pd(shx0re, shx2re));
            auto shy4re = _mm256_castpd_ps(_mm256_unpacklo_pd(shx4re, shx6re));
            auto shy6re = _mm256_castpd_ps(_mm256_unpackhi_pd(shx4re, shx6re));
            auto shy5re = _mm256_castpd_ps(_mm256_unpacklo_pd(shx5re, shx7re));
            auto shy7re = _mm256_castpd_ps(_mm256_unpackhi_pd(shx5re, shx7re));
            auto shy1re = _mm256_castpd_ps(_mm256_unpacklo_pd(shx1re, shx3re));
            auto shy3re = _mm256_castpd_ps(_mm256_unpackhi_pd(shx1re, shx3re));

            auto shz0re = _mm256_permute2f128_ps(shy0re, shy1re, 0b00100000);
            auto shz1re = _mm256_permute2f128_ps(shy0re, shy1re, 0b00110001);
            auto shz2re = _mm256_permute2f128_ps(shy2re, shy3re, 0b00100000);
            auto shz3re = _mm256_permute2f128_ps(shy2re, shy3re, 0b00110001);
            auto shz4re = _mm256_permute2f128_ps(shy4re, shy5re, 0b00100000);
            auto shz5re = _mm256_permute2f128_ps(shy4re, shy5re, 0b00110001);
            auto shz6re = _mm256_permute2f128_ps(shy6re, shy7re, 0b00100000);
            auto shz7re = _mm256_permute2f128_ps(shy6re, shy7re, 0b00110001);

            auto shx0im = _mm256_castps_pd(_mm256_unpacklo_ps(z0im, z4im));
            auto shx4im = _mm256_castps_pd(_mm256_unpackhi_ps(z0im, z4im));
            auto shx2im = _mm256_castps_pd(_mm256_unpacklo_ps(z2im, z6im));
            auto shx6im = _mm256_castps_pd(_mm256_unpackhi_ps(z2im, z6im));
            auto shx1im = _mm256_castps_pd(_mm256_unpacklo_ps(z1im, z5im));
            auto shx5im = _mm256_castps_pd(_mm256_unpackhi_ps(z1im, z5im));
            auto shx3im = _mm256_castps_pd(_mm256_unpacklo_ps(z3im, z7im));
            auto shx7im = _mm256_castps_pd(_mm256_unpackhi_ps(z3im, z7im));

            auto shy0im = _mm256_castpd_ps(_mm256_unpacklo_pd(shx0im, shx2im));
            auto shy2im = _mm256_castpd_ps(_mm256_unpackhi_pd(shx0im, shx2im));
            auto shy4im = _mm256_castpd_ps(_mm256_unpacklo_pd(shx4im, shx6im));
            auto shy6im = _mm256_castpd_ps(_mm256_unpackhi_pd(shx4im, shx6im));
            auto shy1im = _mm256_castpd_ps(_mm256_unpacklo_pd(shx1im, shx3im));
            auto shy3im = _mm256_castpd_ps(_mm256_unpackhi_pd(shx1im, shx3im));
            auto shy5im = _mm256_castpd_ps(_mm256_unpacklo_pd(shx5im, shx7im));
            auto shy7im = _mm256_castpd_ps(_mm256_unpackhi_pd(shx5im, shx7im));

            auto shz0im = _mm256_permute2f128_ps(shy0im, shy1im, 0b00100000);
            auto shz1im = _mm256_permute2f128_ps(shy0im, shy1im, 0b00110001);
            auto shz2im = _mm256_permute2f128_ps(shy2im, shy3im, 0b00100000);
            auto shz3im = _mm256_permute2f128_ps(shy2im, shy3im, 0b00110001);
            auto shz4im = _mm256_permute2f128_ps(shy4im, shy5im, 0b00100000);
            auto shz5im = _mm256_permute2f128_ps(shy4im, shy5im, 0b00110001);
            auto shz6im = _mm256_permute2f128_ps(shy6im, shy7im, 0b00100000);
            auto shz7im = _mm256_permute2f128_ps(shy6im, shy7im, 0b00110001);

            _mm256_storeu_ps(data_ptr + 0 * size / 8 + idx_first, shz0re);
            _mm256_storeu_ps(data_ptr + 0 * size / 8 + idx_first + PackSize, shz0im);
            _mm256_storeu_ps(data_ptr + 1 * size / 8 + idx_first, shz1re);
            _mm256_storeu_ps(data_ptr + 1 * size / 8 + idx_first + PackSize, shz1im);
            _mm256_storeu_ps(data_ptr + 2 * size / 8 + idx_first, shz2re);
            _mm256_storeu_ps(data_ptr + 2 * size / 8 + idx_first + PackSize, shz2im);
            _mm256_storeu_ps(data_ptr + 3 * size / 8 + idx_first, shz3re);
            _mm256_storeu_ps(data_ptr + 3 * size / 8 + idx_first + PackSize, shz3im);

            _mm256_storeu_ps(data_ptr + 4 * size / 8 + idx_first, shz4re);
            _mm256_storeu_ps(data_ptr + 4 * size / 8 + idx_first + PackSize, shz4im);
            _mm256_storeu_ps(data_ptr + 5 * size / 8 + idx_first, shz5re);
            _mm256_storeu_ps(data_ptr + 5 * size / 8 + idx_first + PackSize, shz5im);
            _mm256_storeu_ps(data_ptr + 6 * size / 8 + idx_first, shz6re);
            _mm256_storeu_ps(data_ptr + 6 * size / 8 + idx_first + PackSize, shz6im);
            _mm256_storeu_ps(data_ptr + 7 * size / 8 + idx_first, shz7re);
            _mm256_storeu_ps(data_ptr + 7 * size / 8 + idx_first + PackSize, shz7im);
        };

        for (uint i = n_reversals(size / 64); i < size / 64; ++i)
        {
            auto idx = m_sort[i];

            auto p0re = _mm256_loadu_ps(data_ptr + 0 * size / 8 + idx);
            auto p0im = _mm256_loadu_ps(data_ptr + 0 * size / 8 + idx + PackSize);
            auto p4re = _mm256_loadu_ps(data_ptr + 4 * size / 8 + idx);
            auto p4im = _mm256_loadu_ps(data_ptr + 4 * size / 8 + idx + PackSize);
            auto p2re = _mm256_loadu_ps(data_ptr + 2 * size / 8 + idx);
            auto p2im = _mm256_loadu_ps(data_ptr + 2 * size / 8 + idx + PackSize);
            auto p6re = _mm256_loadu_ps(data_ptr + 6 * size / 8 + idx);
            auto p6im = _mm256_loadu_ps(data_ptr + 6 * size / 8 + idx + PackSize);

            auto a0re = _mm256_add_ps(p0re, p4re);
            auto a4re = _mm256_sub_ps(p0re, p4re);
            auto a0im = _mm256_add_ps(p0im, p4im);
            auto a4im = _mm256_sub_ps(p0im, p4im);
            auto a2re = _mm256_add_ps(p2re, p6re);
            auto a6re = _mm256_sub_ps(p2re, p6re);
            auto a2im = _mm256_add_ps(p2im, p6im);
            auto a6im = _mm256_sub_ps(p2im, p6im);

            auto b0re = _mm256_add_ps(a0re, a2re);
            auto b2re = _mm256_sub_ps(a0re, a2re);
            auto b0im = _mm256_add_ps(a0im, a2im);
            auto b2im = _mm256_sub_ps(a0im, a2im);
            auto b4re = _mm256_add_ps(a4re, a6im);
            auto b6re = _mm256_sub_ps(a4re, a6im);
            auto b4im = _mm256_add_ps(a4im, a6re);
            auto b6im = _mm256_sub_ps(a4im, a6re);

            auto p1re = _mm256_loadu_ps(data_ptr + 1 * size / 8 + idx);
            auto p1im = _mm256_loadu_ps(data_ptr + 1 * size / 8 + idx + PackSize);
            auto p5re = _mm256_loadu_ps(data_ptr + 5 * size / 8 + idx);
            auto p5im = _mm256_loadu_ps(data_ptr + 5 * size / 8 + idx + PackSize);
            auto p3re = _mm256_loadu_ps(data_ptr + 3 * size / 8 + idx);
            auto p3im = _mm256_loadu_ps(data_ptr + 3 * size / 8 + idx + PackSize);
            auto p7re = _mm256_loadu_ps(data_ptr + 7 * size / 8 + idx);
            auto p7im = _mm256_loadu_ps(data_ptr + 7 * size / 8 + idx + PackSize);

            auto a1re = _mm256_add_ps(p1re, p5re);
            auto a5re = _mm256_sub_ps(p1re, p5re);
            auto a1im = _mm256_add_ps(p1im, p5im);
            auto a5im = _mm256_sub_ps(p1im, p5im);
            auto a3re = _mm256_add_ps(p3re, p7re);
            auto a7re = _mm256_sub_ps(p3re, p7re);
            auto a3im = _mm256_add_ps(p3im, p7im);
            auto a7im = _mm256_sub_ps(p3im, p7im);

            auto b5re    = _mm256_add_ps(a5re, a7im);
            auto b7re    = _mm256_sub_ps(a5re, a7im);
            auto b5im    = _mm256_add_ps(a5im, a7re);
            auto b7im    = _mm256_sub_ps(a5im, a7re);
            auto b5re_tw = _mm256_add_ps(b5re, b5im);
            auto b5im_tw = _mm256_sub_ps(b5im, b5re);
            auto b7re_tw = _mm256_add_ps(b7re, b7im);
            auto b7im_tw = _mm256_sub_ps(b7im, b7re);

            auto b1re = _mm256_add_ps(a1re, a3re);
            auto b3re = _mm256_sub_ps(a1re, a3re);
            auto b1im = _mm256_add_ps(a1im, a3im);
            auto b3im = _mm256_sub_ps(a1im, a3im);
            b5re_tw   = _mm256_mul_ps(b5re_tw, twsq2);
            b5im_tw   = _mm256_mul_ps(b5im_tw, twsq2);
            b7re_tw   = _mm256_mul_ps(b7re_tw, twsq2);
            b7im_tw   = _mm256_mul_ps(b7im_tw, twsq2);

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
            auto sha3re = _mm256_castps_pd(_mm256_unpacklo_ps(c3re, c7re));
            auto sha7re = _mm256_castps_pd(_mm256_unpackhi_ps(c3re, c7re));
            auto sha0re = _mm256_castps_pd(_mm256_unpacklo_ps(c0re, c4re));
            auto sha4re = _mm256_castps_pd(_mm256_unpackhi_ps(c0re, c4re));
            auto sha2re = _mm256_castps_pd(_mm256_unpacklo_ps(c2re, c6re));
            auto sha6re = _mm256_castps_pd(_mm256_unpackhi_ps(c2re, c6re));

            auto shb0re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha0re, sha2re));
            auto shb2re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha0re, sha2re));
            auto shb4re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha4re, sha6re));
            auto shb6re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha4re, sha6re));
            auto shb5re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha5re, sha7re));
            auto shb7re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha5re, sha7re));
            auto shb1re = _mm256_castpd_ps(_mm256_unpacklo_pd(sha1re, sha3re));
            auto shb3re = _mm256_castpd_ps(_mm256_unpackhi_pd(sha1re, sha3re));

            auto shc0re = _mm256_permute2f128_ps(shb0re, shb1re, 0b00100000);
            auto shc1re = _mm256_permute2f128_ps(shb0re, shb1re, 0b00110001);
            auto shc2re = _mm256_permute2f128_ps(shb2re, shb3re, 0b00100000);
            auto shc3re = _mm256_permute2f128_ps(shb2re, shb3re, 0b00110001);
            auto shc4re = _mm256_permute2f128_ps(shb4re, shb5re, 0b00100000);
            auto shc5re = _mm256_permute2f128_ps(shb4re, shb5re, 0b00110001);
            auto shc6re = _mm256_permute2f128_ps(shb6re, shb7re, 0b00100000);
            auto shc7re = _mm256_permute2f128_ps(shb6re, shb7re, 0b00110001);

            auto sha0im = _mm256_castps_pd(_mm256_unpacklo_ps(c0im, c4im));
            auto sha4im = _mm256_castps_pd(_mm256_unpackhi_ps(c0im, c4im));
            auto sha2im = _mm256_castps_pd(_mm256_unpacklo_ps(c2im, c6im));
            auto sha6im = _mm256_castps_pd(_mm256_unpackhi_ps(c2im, c6im));
            auto sha1im = _mm256_castps_pd(_mm256_unpacklo_ps(c1im, c5im));
            auto sha5im = _mm256_castps_pd(_mm256_unpackhi_ps(c1im, c5im));
            auto sha3im = _mm256_castps_pd(_mm256_unpacklo_ps(c3im, c7im));
            auto sha7im = _mm256_castps_pd(_mm256_unpackhi_ps(c3im, c7im));

            auto shb0im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha0im, sha2im));
            auto shb2im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha0im, sha2im));
            auto shb4im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha4im, sha6im));
            auto shb6im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha4im, sha6im));
            auto shb1im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha1im, sha3im));
            auto shb3im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha1im, sha3im));
            auto shb5im = _mm256_castpd_ps(_mm256_unpacklo_pd(sha5im, sha7im));
            auto shb7im = _mm256_castpd_ps(_mm256_unpackhi_pd(sha5im, sha7im));

            auto shc0im = _mm256_permute2f128_ps(shb0im, shb1im, 0b00100000);
            auto shc1im = _mm256_permute2f128_ps(shb0im, shb1im, 0b00110001);
            auto shc2im = _mm256_permute2f128_ps(shb2im, shb3im, 0b00100000);
            auto shc3im = _mm256_permute2f128_ps(shb2im, shb3im, 0b00110001);
            auto shc4im = _mm256_permute2f128_ps(shb4im, shb5im, 0b00100000);
            auto shc5im = _mm256_permute2f128_ps(shb4im, shb5im, 0b00110001);
            auto shc6im = _mm256_permute2f128_ps(shb6im, shb7im, 0b00100000);
            auto shc7im = _mm256_permute2f128_ps(shb6im, shb7im, 0b00110001);

            _mm256_storeu_ps(data_ptr + 0 * size / 8 + idx, shc0re);
            _mm256_storeu_ps(data_ptr + 0 * size / 8 + idx + PackSize, shc0im);
            _mm256_storeu_ps(data_ptr + 1 * size / 8 + idx, shc1re);
            _mm256_storeu_ps(data_ptr + 1 * size / 8 + idx + PackSize, shc1im);
            _mm256_storeu_ps(data_ptr + 2 * size / 8 + idx, shc2re);
            _mm256_storeu_ps(data_ptr + 2 * size / 8 + idx + PackSize, shc2im);
            _mm256_storeu_ps(data_ptr + 3 * size / 8 + idx, shc3re);
            _mm256_storeu_ps(data_ptr + 3 * size / 8 + idx + PackSize, shc3im);

            _mm256_storeu_ps(data_ptr + 4 * size / 8 + idx, shc4re);
            _mm256_storeu_ps(data_ptr + 4 * size / 8 + idx + PackSize, shc4im);
            _mm256_storeu_ps(data_ptr + 5 * size / 8 + idx, shc5re);
            _mm256_storeu_ps(data_ptr + 5 * size / 8 + idx + PackSize, shc5im);
            _mm256_storeu_ps(data_ptr + 6 * size / 8 + idx, shc6re);
            _mm256_storeu_ps(data_ptr + 6 * size / 8 + idx + PackSize, shc6im);
            _mm256_storeu_ps(data_ptr + 7 * size / 8 + idx, shc7re);
            _mm256_storeu_ps(data_ptr + 7 * size / 8 + idx + PackSize, shc7im);
        }

        if ((size & 0x5555555555555555) != 0)
        {
            auto tw0re = _mm256_loadu_ps(twiddle_ptr);
            auto tw0im = _mm256_loadu_ps(twiddle_ptr + PackSize);

            avx::cx_reg<float> tw0 = {tw0re, tw0im};

            for (uint i = 0; i < size / 16; ++i)
            {
                auto p0re = _mm256_loadu_ps(data_ptr + 1 * size / 8 + i * 8);
                auto p0im = _mm256_loadu_ps(data_ptr + 1 * size / 8 + i * 8 + PackSize);
                auto p1re = _mm256_loadu_ps(data_ptr + 1 * size / 8 + i * 8);
                auto p1im = _mm256_loadu_ps(data_ptr + 1 * size / 8 + i * 8 + PackSize);
                auto p2re = _mm256_loadu_ps(data_ptr + 1 * size / 8 + i * 8);
                auto p2im = _mm256_loadu_ps(data_ptr + 1 * size / 8 + i * 8 + PackSize);
                auto p3re = _mm256_loadu_ps(data_ptr + 1 * size / 8 + i * 8);
                auto p3im = _mm256_loadu_ps(data_ptr + 1 * size / 8 + i * 8 + PackSize);

                avx::cx_reg<float> p1{p1re, p1im};

                auto p1tw = avx::mul(p1, tw0);




            }
        }
    };

private:
    static constexpr std::size_t register_size = 32 / sizeof(real_type);

    static constexpr auto log2i(std::size_t number) -> std::size_t
    {
        std::size_t order = 0;
        while ((number >>= 1U) != 0)
        {
            order++;
        }
        return order;
    }

    static constexpr auto reverse_bit_order(uint64_t number) -> uint64_t
    {
        number = number >> 32 | number << 32;
        number = (number & 0xFFFF0000FFFF0000) >> 16 | (number & 0x0000FFFF0000FFFF)
                                                           << 16;
        number = (number & 0xFF00FF00FF00FF00) >> 8 | (number & 0x00FF00FF00FF00FF) << 8;
        number = (number & 0xF0F0F0F0F0F0F0F0) >> 4 | (number & 0x0F0F0F0F0F0F0F0F) << 4;
        number = (number & 0xCCCCCCCCCCCCCCCC) >> 2 | (number & 0x3333333333333333) << 2;
        number = (number & 0xAAAAAAAAAAAAAAAA) >> 1 | (number & 0x5555555555555555) << 1;
        return number;
    }

    /**
     * @brief Returns number of unique bit-reversed pairs from 0 to max-1
     * 
     * @param max 
     * @return std::size_t 
     */
    static constexpr auto n_reversals(std::size_t max) -> std::size_t
    {
        return max - 1U << (log2i(max + 1) / 2);
    }

    static inline auto wnk(std::size_t n, std::size_t k) -> std::complex<real_type>
    {
        constexpr double pi = 3.14159265358979323846;
        return exp(std::complex<real_type>(0,
                                           -2 * pi * static_cast<double>(k) /
                                               static_cast<double>(n)));
    }

    static auto get_sort(std::size_t fft_size, sort_allocator_type allocator)
        -> std::pair<std::vector<std::size_t, sort_allocator_type>, std::size_t>
    {
        const auto packed_sort_size = fft_size / register_size / register_size;

        auto sort = std::pair<std::vector<std::size_t, sort_allocator_type>, std::size_t>{
            allocator,
            0};
        auto& indexes = sort.first;
        indexes.reserve(packed_sort_size);

        for (uint i = 0; i < packed_sort_size / 2; ++i)
        {
            if (i == reverse_bit_order(i))
            {
                continue;
            }
            indexes.push_back(i);
            indexes.push_back(reverse_bit_order(i));
        }
        sort.second = indexes.size();
        for (uint i = 0; i < packed_sort_size; ++i)
        {
            if (i == reverse_bit_order(i))
            {
                indexes.push_back(i);
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
        uint l_fft_size  = 16;
        uint small_i_max = 1;

        if (depth % 2 == 0)
        {
            for (uint k = 0; k < 8; ++k)
            {
                *(tw_it++) = wnk(l_fft_size, k);
            }
            l_fft_size *= 2;
            small_i_max *= 2;
        }

        while (l_fft_size < fft_size)
        {
            for (uint small_i = 0; small_i < small_i_max; ++small_i)
            {
                for (uint k = 0; k < 8; ++k)
                {
                    *(tw_it++) = wnk(l_fft_size, k + small_i * 8);
                }

                for (uint k = 0; k < 8; ++k)
                {
                    *(tw_it++) = wnk(l_fft_size * 2UL, k + small_i * 8);
                }
                for (uint k = 0; k < 8; ++k)
                {
                    *(tw_it++) = wnk(l_fft_size * 2UL, k + small_i * 8 + l_fft_size / 2);
                }
            }
            small_i_max *= 4;
            l_fft_size *= 4;
        }
        return twiddles;
    }
};


}    // namespace pcx
#endif