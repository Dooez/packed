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
    [[no_unique_address]] size_t     m_size;
    std::shared_ptr<packed_fft_core> m_core{};

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

    static constexpr auto reverse_bit_order(uint64_t num) -> uint64_t
    {
        num = num >> 32 | num << 32;
        num = (num & 0xFFFF0000FFFF0000) >> 16 | (num & 0x0000FFFF0000FFFF) << 16;
        num = (num & 0xFF00FF00FF00FF00) >> 8 | (num & 0x00FF00FF00FF00FF) << 8;
        num = (num & 0xF0F0F0F0F0F0F0F0) >> 4 | (num & 0x0F0F0F0F0F0F0F0F) << 4;
        num = (num & 0xCCCCCCCCCCCCCCCC) >> 2 | (num & 0x3333333333333333) << 2;
        num = (num & 0xAAAAAAAAAAAAAAAA) >> 1 | (num & 0x5555555555555555) << 1;
        return num;
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

template<typename T, std::size_t PackSize, typename Allocator>
    requires std::same_as<T, float>
void fft_internal(pcx::vector<T, PackSize, Allocator>& vector)
{
    auto size = vector.size();

    constexpr const double pi  = 3.14159265358979323846;
    const auto             sq2 = std::exp(std::complex<float>(0, pi / 4));

    for (uint i = 0; i < size / 64; ++i)
    {
    };

    auto twsq2 = _mm256_broadcast_ss(&reinterpret_cast<const T(&)[2]>(sq2)[0]);

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

    auto shb0re = _mm256_unpacklo_pd(_mm256_castps_pd(sha0re), _mm256_castps_pd(sha2re));
    auto shb2re = _mm256_unpackhi_pd(_mm256_castps_pd(sha0re), _mm256_castps_pd(sha2re));
    auto shb1re = _mm256_unpacklo_pd(_mm256_castps_pd(sha1re), _mm256_castps_pd(sha3re));
    auto shb3re = _mm256_unpackhi_pd(_mm256_castps_pd(sha1re), _mm256_castps_pd(sha3re));
    auto shb4re = _mm256_unpacklo_pd(_mm256_castps_pd(sha4re), _mm256_castps_pd(sha6re));
    auto shb6re = _mm256_unpackhi_pd(_mm256_castps_pd(sha4re), _mm256_castps_pd(sha6re));
    auto shb5re = _mm256_unpacklo_pd(_mm256_castps_pd(sha5re), _mm256_castps_pd(sha7re));
    auto shb7re = _mm256_unpackhi_pd(_mm256_castps_pd(sha5re), _mm256_castps_pd(sha7re));

    auto shb0im = _mm256_unpacklo_pd(_mm256_castps_pd(sha0im), _mm256_castps_pd(sha2im));
    auto shb2im = _mm256_unpackhi_pd(_mm256_castps_pd(sha0im), _mm256_castps_pd(sha2im));
    auto shb1im = _mm256_unpacklo_pd(_mm256_castps_pd(sha1im), _mm256_castps_pd(sha3im));
    auto shb3im = _mm256_unpackhi_pd(_mm256_castps_pd(sha1im), _mm256_castps_pd(sha3im));
    auto shb4im = _mm256_unpacklo_pd(_mm256_castps_pd(sha4im), _mm256_castps_pd(sha6im));
    auto shb5im = _mm256_unpackhi_pd(_mm256_castps_pd(sha4im), _mm256_castps_pd(sha6im));
    auto shb6im = _mm256_unpacklo_pd(_mm256_castps_pd(sha5im), _mm256_castps_pd(sha7im));
    auto shb7im = _mm256_unpackhi_pd(_mm256_castps_pd(sha5im), _mm256_castps_pd(sha7im));

    auto shc0re = _mm256_permute2f128_ps(_mm256_castpd_ps(shb0re),
                                         _mm256_castpd_ps(shb1re),
                                         0b00100000);
    auto shc1re = _mm256_permute2f128_ps(_mm256_castpd_ps(shb0re),
                                         _mm256_castpd_ps(shb1re),
                                         0b00110001);
    auto shc2re = _mm256_permute2f128_ps(_mm256_castpd_ps(shb2re),
                                         _mm256_castpd_ps(shb3re),
                                         0b00100000);
    auto shc3re = _mm256_permute2f128_ps(_mm256_castpd_ps(shb2re),
                                         _mm256_castpd_ps(shb3re),
                                         0b00110001);
    auto shc4re = _mm256_permute2f128_ps(_mm256_castpd_ps(shb4re),
                                         _mm256_castpd_ps(shb5re),
                                         0b00100000);
    auto shc5re = _mm256_permute2f128_ps(_mm256_castpd_ps(shb4re),
                                         _mm256_castpd_ps(shb5re),
                                         0b00110001);
    auto shc6re = _mm256_permute2f128_ps(_mm256_castpd_ps(shb6re),
                                         _mm256_castpd_ps(shb7re),
                                         0b00100000);
    auto shc7re = _mm256_permute2f128_ps(_mm256_castpd_ps(shb6re),
                                         _mm256_castpd_ps(shb7re),
                                         0b00110001);

    auto shc0im = _mm256_permute2f128_ps(_mm256_castpd_ps(shb0im),
                                         _mm256_castpd_ps(shb1im),
                                         0b00100000);
    auto shc1im = _mm256_permute2f128_ps(_mm256_castpd_ps(shb0im),
                                         _mm256_castpd_ps(shb1im),
                                         0b00110001);
    auto shc2im = _mm256_permute2f128_ps(_mm256_castpd_ps(shb2im),
                                         _mm256_castpd_ps(shb3im),
                                         0b00100000);
    auto shc3im = _mm256_permute2f128_ps(_mm256_castpd_ps(shb2im),
                                         _mm256_castpd_ps(shb3im),
                                         0b00110001);
    auto shc4im = _mm256_permute2f128_ps(_mm256_castpd_ps(shb4im),
                                         _mm256_castpd_ps(shb5im),
                                         0b00100000);
    auto shc5im = _mm256_permute2f128_ps(_mm256_castpd_ps(shb4im),
                                         _mm256_castpd_ps(shb5im),
                                         0b00110001);
    auto shc6im = _mm256_permute2f128_ps(_mm256_castpd_ps(shb6im),
                                         _mm256_castpd_ps(shb7im),
                                         0b00100000);
    auto shc7im = _mm256_permute2f128_ps(_mm256_castpd_ps(shb6im),
                                         _mm256_castpd_ps(shb7im),
                                         0b00110001);

    _mm256_storeu_ps(&vector[0 * size / 8], shc0re);
    _mm256_storeu_ps(&vector[1 * size / 8], shc1re);
    _mm256_storeu_ps(&vector[2 * size / 8], shc2re);
    _mm256_storeu_ps(&vector[3 * size / 8], shc3re);
    _mm256_storeu_ps(&vector[4 * size / 8], shc4re);
    _mm256_storeu_ps(&vector[5 * size / 8], shc5re);
    _mm256_storeu_ps(&vector[6 * size / 8], shc6re);
    _mm256_storeu_ps(&vector[7 * size / 8], shc7re);

    _mm256_storeu_ps(&vector[0 * size / 8] + PackSize, shc0im);
    _mm256_storeu_ps(&vector[1 * size / 8] + PackSize, shc1im);
    _mm256_storeu_ps(&vector[2 * size / 8] + PackSize, shc2im);
    _mm256_storeu_ps(&vector[3 * size / 8] + PackSize, shc3im);
    _mm256_storeu_ps(&vector[4 * size / 8] + PackSize, shc4im);
    _mm256_storeu_ps(&vector[5 * size / 8] + PackSize, shc5im);
    _mm256_storeu_ps(&vector[6 * size / 8] + PackSize, shc6im);
    _mm256_storeu_ps(&vector[7 * size / 8] + PackSize, shc7im);
}

}    // namespace pcx
#endif