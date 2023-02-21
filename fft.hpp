#ifndef FFT_HPP
#define FFT_HPP

#include "vector.hpp"

#include <memory>

namespace pcx {

template<typename T,
         std::size_t Size     = pcx::dynamic_size,
         std::size_t SubSize  = pcx::dynamic_size,
         typename Allocator   = std::allocator<T>,
         std::size_t PackSize = pcx::default_pack_size<T>>
    requires(std::same_as<T, float> || std::same_as<T, double>) &&
            (pcx::power_of_two<Size> || (Size == pcx::dynamic_size)) &&
            (pcx::power_of_two<SubSize> || (SubSize == pcx::dynamic_size)) &&
            pcx::power_of_two<PackSize>
class fft_unit
{
public:
    using real_type      = T;
    using allocator_type = Allocator;

    static constexpr auto pack_size = PackSize;

private:
    using size_t =
        std::conditional_t<Size == pcx::dynamic_size, std::size_t, decltype([]() {})>;

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
    void operator()(pcx::vector<T, VAllocator, PackSize>& vector)
    {
        assert(size() == vector.size());
        fft_internal(vector.data());
    };
    template<typename VAllocator>
    void separated(pcx::vector<T, VAllocator, PackSize>& vector)
    {
        assert(size() == vector.size());
        fft_internal_separated(vector.data());
    };
    template<typename VAllocator>
    void binary(pcx::vector<T, VAllocator, PackSize>& vector)
    {
        assert(size() == vector.size());
        fft_internal_binary(vector.data());
    };


    template<typename VAllocator>
    void operator()(pcx::vector<T, VAllocator, PackSize>&       dest,
                    const pcx::vector<T, VAllocator, PackSize>& source)
    {
        assert(size() == dest.size() && size() == source.size());
        if (&dest == &source)
        {
            fft_internal(dest.data());
        } else
        {
            fft_internal(dest.data(), source.data());
        }
    };


private:
    [[no_unique_address]] size_t                            m_size;
    const std::vector<std::size_t, sort_allocator_type>     m_sort;
    const pcx::vector<real_type, allocator_type, pack_size> m_twiddles;

public:
    void fft_internal_binary(float* source)
        requires(SubSize != pcx::dynamic_size)
    {
        dept3_and_sort(source);

        std::size_t sub_size_ = std::min(size(), SubSize);

        recursive_sub_transform(source, size(), sub_size_);
    }


    void fft_internal(float* source)
        requires(SubSize == pcx::dynamic_size)
    {
        auto* twiddle_ptr = m_twiddles.data();

        const auto sq2   = wnk(8, 1);
        auto       twsq2 = avx::broadcast(sq2.real());

        const auto sh0 = 0;
        const auto sh1 = pidx(1 * size() / 8);
        const auto sh2 = pidx(2 * size() / 8);
        const auto sh3 = pidx(3 * size() / 8);
        const auto sh4 = pidx(4 * size() / 8);
        const auto sh5 = pidx(5 * size() / 8);
        const auto sh6 = pidx(6 * size() / 8);
        const auto sh7 = pidx(7 * size() / 8);

        uint i = 0;

        for (; i < n_reversals(size() / 64); i += 2)
        {
            using reg_t = avx::cx_reg<float>;

            auto offset_first  = m_sort[i];
            auto offset_second = m_sort[i + 1];

            auto p1 = avx::cxload<PackSize>(source + sh1 + offset_first);
            auto p5 = avx::cxload<PackSize>(source + sh5 + offset_first);
            auto p3 = avx::cxload<PackSize>(source + sh3 + offset_first);
            auto p7 = avx::cxload<PackSize>(source + sh7 + offset_first);

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

            auto p0 = avx::cxload<PackSize>(source + sh0 + offset_first);
            auto p4 = avx::cxload<PackSize>(source + sh4 + offset_first);
            auto p2 = avx::cxload<PackSize>(source + sh2 + offset_first);
            auto p6 = avx::cxload<PackSize>(source + sh6 + offset_first);

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

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);

            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            auto q0 = avx::cxload<PackSize>(source + sh0 + offset_second);
            avx::cxstore<PackSize>(source + sh0 + offset_second, shc0);
            auto q1 = avx::cxload<PackSize>(source + sh1 + offset_second);
            avx::cxstore<PackSize>(source + sh1 + offset_second, shc1);
            auto q4 = avx::cxload<PackSize>(source + sh4 + offset_second);
            avx::cxstore<PackSize>(source + sh4 + offset_second, shc2);
            auto q5 = avx::cxload<PackSize>(source + sh5 + offset_second);
            avx::cxstore<PackSize>(source + sh5 + offset_second, shc3);

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);

            auto q2 = avx::cxload<PackSize>(source + sh2 + offset_second);
            avx::cxstore<PackSize>(source + sh2 + offset_second, shc4);
            auto q3 = avx::cxload<PackSize>(source + sh3 + offset_second);
            avx::cxstore<PackSize>(source + sh3 + offset_second, shc5);

            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            auto q6 = avx::cxload<PackSize>(source + sh6 + offset_second);
            avx::cxstore<PackSize>(source + sh6 + offset_second, shc6);
            auto q7 = avx::cxload<PackSize>(source + sh7 + offset_second);
            avx::cxstore<PackSize>(source + sh7 + offset_second, shc7);

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

            auto [shx0, shx4] = avx::unpack_ps(z0, z4);
            auto [shx1, shx5] = avx::unpack_ps(z1, z5);
            auto [shx2, shx6] = avx::unpack_ps(z2, z6);
            auto [shx3, shx7] = avx::unpack_ps(z3, z7);

            auto [shy0, shy2] = avx::unpack_pd(shx0, shx2);
            auto [shy1, shy3] = avx::unpack_pd(shx1, shx3);

            auto [shz0, shz1] = avx::unpack_128(shy0, shy1);
            auto [shz2, shz3] = avx::unpack_128(shy2, shy3);

            avx::cxstore<PackSize>(source + sh0 + offset_first, shz0);
            avx::cxstore<PackSize>(source + sh1 + offset_first, shz1);
            avx::cxstore<PackSize>(source + sh4 + offset_first, shz2);
            avx::cxstore<PackSize>(source + sh5 + offset_first, shz3);

            auto [shy4, shy6] = avx::unpack_pd(shx4, shx6);
            auto [shy5, shy7] = avx::unpack_pd(shx5, shx7);

            auto [shz4, shz5] = avx::unpack_128(shy4, shy5);
            auto [shz6, shz7] = avx::unpack_128(shy6, shy7);

            avx::cxstore<PackSize>(source + sh2 + offset_first, shz4);
            avx::cxstore<PackSize>(source + sh3 + offset_first, shz5);
            avx::cxstore<PackSize>(source + sh6 + offset_first, shz6);
            avx::cxstore<PackSize>(source + sh7 + offset_first, shz7);
        };

        for (; i < size() / 64; ++i)
        {
            using reg_t = avx::cx_reg<float>;
            auto offset = m_sort[i];

            auto p1 = avx::cxload<PackSize>(source + sh1 + offset);
            auto p3 = avx::cxload<PackSize>(source + sh3 + offset);
            auto p5 = avx::cxload<PackSize>(source + sh5 + offset);
            auto p7 = avx::cxload<PackSize>(source + sh7 + offset);

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

            auto p0 = avx::cxload<PackSize>(source + sh0 + offset);
            auto p2 = avx::cxload<PackSize>(source + sh2 + offset);
            auto p4 = avx::cxload<PackSize>(source + sh4 + offset);
            auto p6 = avx::cxload<PackSize>(source + sh6 + offset);

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

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);

            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            avx::cxstore<PackSize>(source + sh0 + offset, shc0);
            avx::cxstore<PackSize>(source + sh1 + offset, shc1);
            avx::cxstore<PackSize>(source + sh4 + offset, shc2);
            avx::cxstore<PackSize>(source + sh5 + offset, shc3);

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            avx::cxstore<PackSize>(source + sh2 + offset, shc4);
            avx::cxstore<PackSize>(source + sh3 + offset, shc5);
            avx::cxstore<PackSize>(source + sh6 + offset, shc6);
            avx::cxstore<PackSize>(source + sh7 + offset, shc7);
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
                    auto* ptr0 = source + pidx(offset);
                    auto* ptr1 = source + pidx(offset + l_size / 2);
                    auto* ptr2 = source + pidx(offset + l_size);
                    auto* ptr3 = source + pidx(offset + l_size / 2 * 3);

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

                auto* ptr0 = source + pidx(offset);
                auto* ptr1 = source + pidx(offset + l_size / 2);

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
    void fft_internal(float* dest, const float* source)
        requires(SubSize == pcx::dynamic_size)
    {
        const auto sq2 = wnk(8, 1);

        auto twsq2 = avx::broadcast(sq2.real());

        auto* twiddle_ptr = m_twiddles.data();

        const auto sh0 = 0;
        const auto sh1 = pidx(1 * size() / 8);
        const auto sh2 = pidx(2 * size() / 8);
        const auto sh3 = pidx(3 * size() / 8);
        const auto sh4 = pidx(4 * size() / 8);
        const auto sh5 = pidx(5 * size() / 8);
        const auto sh6 = pidx(6 * size() / 8);
        const auto sh7 = pidx(7 * size() / 8);

        uint i = 0;

        for (; i < n_reversals(size() / 64); i += 2)
        {
            using reg_t = avx::cx_reg<float>;

            auto offset_first  = m_sort[i];
            auto offset_second = m_sort[i + 1];

            auto p1 = avx::cxload<PackSize>(source + sh1 + offset_first);
            auto p5 = avx::cxload<PackSize>(source + sh5 + offset_first);
            auto p3 = avx::cxload<PackSize>(source + sh3 + offset_first);
            auto p7 = avx::cxload<PackSize>(source + sh7 + offset_first);

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

            auto p0 = avx::cxload<PackSize>(source + sh0 + offset_first);
            auto p4 = avx::cxload<PackSize>(source + sh4 + offset_first);
            auto p2 = avx::cxload<PackSize>(source + sh2 + offset_first);
            auto p6 = avx::cxload<PackSize>(source + sh6 + offset_first);

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

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);
            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            avx::cxstore<PackSize>(dest + sh0 + offset_second, shc0);
            avx::cxstore<PackSize>(dest + sh4 + offset_second, shc2);
            avx::cxstore<PackSize>(dest + sh5 + offset_second, shc3);
            avx::cxstore<PackSize>(dest + sh1 + offset_second, shc1);

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);
            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            avx::cxstore<PackSize>(dest + sh2 + offset_second, shc4);
            avx::cxstore<PackSize>(dest + sh3 + offset_second, shc5);
            avx::cxstore<PackSize>(dest + sh6 + offset_second, shc6);
            avx::cxstore<PackSize>(dest + sh7 + offset_second, shc7);

            auto q1 = avx::cxload<PackSize>(source + sh1 + offset_second);
            auto q5 = avx::cxload<PackSize>(source + sh5 + offset_second);
            auto q3 = avx::cxload<PackSize>(source + sh3 + offset_second);
            auto q7 = avx::cxload<PackSize>(source + sh7 + offset_second);

            auto x5 = avx::sub(q1, q5);
            auto x1 = avx::add(q1, q5);
            auto x7 = avx::sub(q3, q7);
            auto x3 = avx::add(q3, q7);

            reg_t y5 = {avx::add(x5.real, x7.imag), avx::sub(x5.imag, x7.real)};
            reg_t y7 = {avx::sub(x5.real, x7.imag), avx::add(x5.imag, x7.real)};


            reg_t y5_tw = {avx::add(y5.real, y5.imag), avx::sub(y5.imag, y5.real)};
            reg_t y7_tw = {avx::sub(y7.real, y7.imag), avx::add(y7.real, y7.imag)};

            auto y1 = avx::add(x1, x3);
            auto y3 = avx::sub(x1, x3);

            y5_tw = avx::mul(y5_tw, twsq2);
            y7_tw = avx::mul(y7_tw, twsq2);

            auto q0 = avx::cxload<PackSize>(source + sh0 + offset_second);
            auto q4 = avx::cxload<PackSize>(source + sh4 + offset_second);
            auto q2 = avx::cxload<PackSize>(source + sh2 + offset_second);
            auto q6 = avx::cxload<PackSize>(source + sh6 + offset_second);

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

            auto [shx0, shx4] = avx::unpack_ps(z0, z4);
            auto [shx1, shx5] = avx::unpack_ps(z1, z5);
            auto [shx2, shx6] = avx::unpack_ps(z2, z6);
            auto [shx3, shx7] = avx::unpack_ps(z3, z7);

            auto [shy0, shy2] = avx::unpack_pd(shx0, shx2);
            auto [shy1, shy3] = avx::unpack_pd(shx1, shx3);

            auto [shz0, shz1] = avx::unpack_128(shy0, shy1);
            auto [shz2, shz3] = avx::unpack_128(shy2, shy3);

            avx::cxstore<PackSize>(dest + sh0 + offset_first, shz0);
            avx::cxstore<PackSize>(dest + sh1 + offset_first, shz1);
            avx::cxstore<PackSize>(dest + sh4 + offset_first, shz2);
            avx::cxstore<PackSize>(dest + sh5 + offset_first, shz3);

            auto [shy4, shy6] = avx::unpack_pd(shx4, shx6);
            auto [shy5, shy7] = avx::unpack_pd(shx5, shx7);

            auto [shz4, shz5] = avx::unpack_128(shy4, shy5);
            auto [shz6, shz7] = avx::unpack_128(shy6, shy7);

            avx::cxstore<PackSize>(dest + sh2 + offset_first, shz4);
            avx::cxstore<PackSize>(dest + sh3 + offset_first, shz5);
            avx::cxstore<PackSize>(dest + sh6 + offset_first, shz6);
            avx::cxstore<PackSize>(dest + sh7 + offset_first, shz7);
        };

        for (; i < size() / 64; ++i)
        {
            using reg_t = avx::cx_reg<float>;
            auto offset = m_sort[i];

            auto p1 = avx::cxload<PackSize>(source + sh1 + offset);
            auto p5 = avx::cxload<PackSize>(source + sh5 + offset);
            auto p3 = avx::cxload<PackSize>(source + sh3 + offset);
            auto p7 = avx::cxload<PackSize>(source + sh7 + offset);

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

            auto p0 = avx::cxload<PackSize>(source + sh0 + offset);
            auto p2 = avx::cxload<PackSize>(source + sh2 + offset);
            auto p4 = avx::cxload<PackSize>(source + sh4 + offset);
            auto p6 = avx::cxload<PackSize>(source + sh6 + offset);

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

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);

            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            avx::cxstore<PackSize>(dest + sh0 + offset, shc0);
            avx::cxstore<PackSize>(dest + sh1 + offset, shc1);
            avx::cxstore<PackSize>(dest + sh4 + offset, shc2);
            avx::cxstore<PackSize>(dest + sh5 + offset, shc3);

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            avx::cxstore<PackSize>(dest + sh2 + offset, shc4);
            avx::cxstore<PackSize>(dest + sh3 + offset, shc5);
            avx::cxstore<PackSize>(dest + sh6 + offset, shc6);
            avx::cxstore<PackSize>(dest + sh7 + offset, shc7);
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
                    auto* ptr0 = dest + pidx(offset);
                    auto* ptr1 = dest + pidx(offset + l_size / 2);
                    auto* ptr2 = dest + pidx(offset + l_size);
                    auto* ptr3 = dest + pidx(offset + l_size / 2 * 3);

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

                auto* ptr0 = dest + pidx(offset);
                auto* ptr1 = dest + pidx(offset + l_size / 2);

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

    void fft_unsorted(float* source)
        requires(SubSize != pcx::dynamic_size)
    {};

    void fft_internal(float* source)
        requires(SubSize != pcx::dynamic_size)
    {
        const auto sq2 = wnk(8, 1);

        auto twsq2 = avx::broadcast(sq2.real());

        auto* twiddle_ptr = m_twiddles.data();

        const auto sh0 = 0;
        const auto sh1 = pidx(1 * size() / 8);
        const auto sh2 = pidx(2 * size() / 8);
        const auto sh3 = pidx(3 * size() / 8);
        const auto sh4 = pidx(4 * size() / 8);
        const auto sh5 = pidx(5 * size() / 8);
        const auto sh6 = pidx(6 * size() / 8);
        const auto sh7 = pidx(7 * size() / 8);

        uint i = 0;

        for (; i < n_reversals(size() / 64); i += 2)
        {
            using reg_t = avx::cx_reg<float>;

            auto offset_first  = m_sort[i];
            auto offset_second = m_sort[i + 1];

            auto p1 = avx::cxload<PackSize>(source + sh1 + offset_first);
            auto p5 = avx::cxload<PackSize>(source + sh5 + offset_first);
            auto p3 = avx::cxload<PackSize>(source + sh3 + offset_first);
            auto p7 = avx::cxload<PackSize>(source + sh7 + offset_first);

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

            auto p0 = avx::cxload<PackSize>(source + sh0 + offset_first);
            auto p4 = avx::cxload<PackSize>(source + sh4 + offset_first);
            auto p2 = avx::cxload<PackSize>(source + sh2 + offset_first);
            auto p6 = avx::cxload<PackSize>(source + sh6 + offset_first);

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

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);

            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            auto q0 = avx::cxload<PackSize>(source + sh0 + offset_second);
            avx::cxstore<PackSize>(source + sh0 + offset_second, shc0);
            auto q1 = avx::cxload<PackSize>(source + sh1 + offset_second);
            avx::cxstore<PackSize>(source + sh1 + offset_second, shc1);
            auto q4 = avx::cxload<PackSize>(source + sh4 + offset_second);
            avx::cxstore<PackSize>(source + sh4 + offset_second, shc2);
            auto q5 = avx::cxload<PackSize>(source + sh5 + offset_second);
            avx::cxstore<PackSize>(source + sh5 + offset_second, shc3);

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);

            auto q2 = avx::cxload<PackSize>(source + sh2 + offset_second);
            avx::cxstore<PackSize>(source + sh2 + offset_second, shc4);
            auto q3 = avx::cxload<PackSize>(source + sh3 + offset_second);
            avx::cxstore<PackSize>(source + sh3 + offset_second, shc5);

            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            auto q6 = avx::cxload<PackSize>(source + sh6 + offset_second);
            avx::cxstore<PackSize>(source + sh6 + offset_second, shc6);
            auto q7 = avx::cxload<PackSize>(source + sh7 + offset_second);
            avx::cxstore<PackSize>(source + sh7 + offset_second, shc7);

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

            auto [shx0, shx4] = avx::unpack_ps(z0, z4);
            auto [shx1, shx5] = avx::unpack_ps(z1, z5);
            auto [shx2, shx6] = avx::unpack_ps(z2, z6);
            auto [shx3, shx7] = avx::unpack_ps(z3, z7);

            auto [shy0, shy2] = avx::unpack_pd(shx0, shx2);
            auto [shy1, shy3] = avx::unpack_pd(shx1, shx3);

            auto [shz0, shz1] = avx::unpack_128(shy0, shy1);
            auto [shz2, shz3] = avx::unpack_128(shy2, shy3);

            avx::cxstore<PackSize>(source + sh0 + offset_first, shz0);
            avx::cxstore<PackSize>(source + sh1 + offset_first, shz1);
            avx::cxstore<PackSize>(source + sh4 + offset_first, shz2);
            avx::cxstore<PackSize>(source + sh5 + offset_first, shz3);

            auto [shy4, shy6] = avx::unpack_pd(shx4, shx6);
            auto [shy5, shy7] = avx::unpack_pd(shx5, shx7);

            auto [shz4, shz5] = avx::unpack_128(shy4, shy5);
            auto [shz6, shz7] = avx::unpack_128(shy6, shy7);

            avx::cxstore<PackSize>(source + sh2 + offset_first, shz4);
            avx::cxstore<PackSize>(source + sh3 + offset_first, shz5);
            avx::cxstore<PackSize>(source + sh6 + offset_first, shz6);
            avx::cxstore<PackSize>(source + sh7 + offset_first, shz7);
        };

        for (; i < size() / 64; ++i)
        {
            using reg_t = avx::cx_reg<float>;
            auto offset = m_sort[i];

            auto p1 = avx::cxload<PackSize>(source + sh1 + offset);
            auto p3 = avx::cxload<PackSize>(source + sh3 + offset);
            auto p5 = avx::cxload<PackSize>(source + sh5 + offset);
            auto p7 = avx::cxload<PackSize>(source + sh7 + offset);

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

            auto p0 = avx::cxload<PackSize>(source + sh0 + offset);
            auto p2 = avx::cxload<PackSize>(source + sh2 + offset);
            auto p4 = avx::cxload<PackSize>(source + sh4 + offset);
            auto p6 = avx::cxload<PackSize>(source + sh6 + offset);

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

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);

            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            avx::cxstore<PackSize>(source + sh0 + offset, shc0);
            avx::cxstore<PackSize>(source + sh1 + offset, shc1);
            avx::cxstore<PackSize>(source + sh4 + offset, shc2);
            avx::cxstore<PackSize>(source + sh5 + offset, shc3);

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            avx::cxstore<PackSize>(source + sh2 + offset, shc4);
            avx::cxstore<PackSize>(source + sh3 + offset, shc5);
            avx::cxstore<PackSize>(source + sh6 + offset, shc6);
            avx::cxstore<PackSize>(source + sh7 + offset, shc7);
        }

        std::size_t l_size     = 0;
        std::size_t group_size = 0;
        std::size_t n_groups   = 0;
        std::size_t tw_offset  = 0;

        std::size_t sub_offset = 0;
        std::size_t sub_size_  = std::min(size(), SubSize);

        for (uint i = 0; i < size() / sub_size_; ++i)
        {
            auto* sub_source = source + pidx(sub_offset);
            l_size           = reg_size * 2;
            group_size       = sub_size_ / reg_size / 4;
            n_groups         = 1;
            tw_offset        = 0;

            while (l_size < sub_size_)
            {
                for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
                {
                    std::size_t offset = i_group * reg_size;

                    const auto tw0 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset));
                    const auto tw1 =
                        avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset + reg_size));
                    const auto tw2 = avx::cxload<PackSize>(
                        twiddle_ptr + pidx(tw_offset + reg_size * 2));

                    tw_offset += reg_size * 3;

                    for (std::size_t i = 0; i < group_size; ++i)
                    {
                        auto* ptr0 = sub_source + pidx(offset);
                        auto* ptr1 = sub_source + pidx(offset + l_size / 2);
                        auto* ptr2 = sub_source + pidx(offset + l_size);
                        auto* ptr3 = sub_source + pidx(offset + l_size / 2 * 3);

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

            if (l_size == sub_size_)
            {
                for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
                {
                    std::size_t offset = i_group * reg_size;

                    const auto tw0 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset));

                    tw_offset += reg_size;

                    auto* ptr0 = sub_source + pidx(offset);
                    auto* ptr1 = sub_source + pidx(offset + l_size / 2);

                    auto p1 = avx::cxload<PackSize>(ptr1);
                    auto p0 = avx::cxload<PackSize>(ptr0);

                    auto p1tw = avx::mul(p1, tw0);

                    auto a0 = avx::add(p0, p1tw);
                    auto a1 = avx::sub(p0, p1tw);

                    cxstore<PackSize>(ptr0, a0);
                    cxstore<PackSize>(ptr1, a1);
                }
                l_size *= 2;
                n_groups *= 2;
                group_size /= 2;
            }

            sub_offset += SubSize;
        }

        group_size = size() / SubSize / 2;
        while (l_size <= size())
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset));

                tw_offset += reg_size;

                for (std::size_t i = 0; i < group_size; ++i)
                {
                    auto* ptr0 = source + pidx(offset);
                    auto* ptr1 = source + pidx(offset + l_size / 2);

                    auto p1 = avx::cxload<PackSize>(ptr1);
                    auto p0 = avx::cxload<PackSize>(ptr0);

                    auto p1tw = avx::mul(p1, tw0);

                    auto a0 = avx::add(p0, p1tw);
                    auto a1 = avx::sub(p0, p1tw);

                    cxstore<PackSize>(ptr0, a0);
                    cxstore<PackSize>(ptr1, a1);

                    offset += l_size;
                }
            }

            l_size *= 2;
            n_groups *= 2;
            group_size /= 2;
        }
    };
    void fft_internal(float* dest, const float* source)
        requires(SubSize != pcx::dynamic_size)
    {
        const auto sq2 = wnk(8, 1);

        auto twsq2 = avx::broadcast(sq2.real());

        auto* twiddle_ptr = m_twiddles.data();

        const auto sh0 = 0;
        const auto sh1 = pidx(1 * size() / 8);
        const auto sh2 = pidx(2 * size() / 8);
        const auto sh3 = pidx(3 * size() / 8);
        const auto sh4 = pidx(4 * size() / 8);
        const auto sh5 = pidx(5 * size() / 8);
        const auto sh6 = pidx(6 * size() / 8);
        const auto sh7 = pidx(7 * size() / 8);

        uint i = 0;

        for (; i < n_reversals(size() / 64); i += 2)
        {
            using reg_t = avx::cx_reg<float>;

            auto offset_first  = m_sort[i];
            auto offset_second = m_sort[i + 1];

            auto p1 = avx::cxload<PackSize>(source + sh1 + offset_first);
            auto p5 = avx::cxload<PackSize>(source + sh5 + offset_first);
            auto p3 = avx::cxload<PackSize>(source + sh3 + offset_first);
            auto p7 = avx::cxload<PackSize>(source + sh7 + offset_first);

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

            auto p0 = avx::cxload<PackSize>(source + sh0 + offset_first);
            auto p4 = avx::cxload<PackSize>(source + sh4 + offset_first);
            auto p2 = avx::cxload<PackSize>(source + sh2 + offset_first);
            auto p6 = avx::cxload<PackSize>(source + sh6 + offset_first);

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

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);
            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            avx::cxstore<PackSize>(dest + sh0 + offset_second, shc0);
            avx::cxstore<PackSize>(dest + sh4 + offset_second, shc2);
            avx::cxstore<PackSize>(dest + sh5 + offset_second, shc3);
            avx::cxstore<PackSize>(dest + sh1 + offset_second, shc1);

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);
            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            avx::cxstore<PackSize>(dest + sh2 + offset_second, shc4);
            avx::cxstore<PackSize>(dest + sh3 + offset_second, shc5);
            avx::cxstore<PackSize>(dest + sh6 + offset_second, shc6);
            avx::cxstore<PackSize>(dest + sh7 + offset_second, shc7);

            auto q1 = avx::cxload<PackSize>(source + sh1 + offset_second);
            auto q5 = avx::cxload<PackSize>(source + sh5 + offset_second);
            auto q3 = avx::cxload<PackSize>(source + sh3 + offset_second);
            auto q7 = avx::cxload<PackSize>(source + sh7 + offset_second);

            auto x5 = avx::sub(q1, q5);
            auto x1 = avx::add(q1, q5);
            auto x7 = avx::sub(q3, q7);
            auto x3 = avx::add(q3, q7);

            reg_t y5 = {avx::add(x5.real, x7.imag), avx::sub(x5.imag, x7.real)};
            reg_t y7 = {avx::sub(x5.real, x7.imag), avx::add(x5.imag, x7.real)};


            reg_t y5_tw = {avx::add(y5.real, y5.imag), avx::sub(y5.imag, y5.real)};
            reg_t y7_tw = {avx::sub(y7.real, y7.imag), avx::add(y7.real, y7.imag)};

            auto y1 = avx::add(x1, x3);
            auto y3 = avx::sub(x1, x3);

            y5_tw = avx::mul(y5_tw, twsq2);
            y7_tw = avx::mul(y7_tw, twsq2);

            auto q0 = avx::cxload<PackSize>(source + sh0 + offset_second);
            auto q4 = avx::cxload<PackSize>(source + sh4 + offset_second);
            auto q2 = avx::cxload<PackSize>(source + sh2 + offset_second);
            auto q6 = avx::cxload<PackSize>(source + sh6 + offset_second);

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

            auto [shx0, shx4] = avx::unpack_ps(z0, z4);
            auto [shx1, shx5] = avx::unpack_ps(z1, z5);
            auto [shx2, shx6] = avx::unpack_ps(z2, z6);
            auto [shx3, shx7] = avx::unpack_ps(z3, z7);

            auto [shy0, shy2] = avx::unpack_pd(shx0, shx2);
            auto [shy1, shy3] = avx::unpack_pd(shx1, shx3);

            auto [shz0, shz1] = avx::unpack_128(shy0, shy1);
            auto [shz2, shz3] = avx::unpack_128(shy2, shy3);

            avx::cxstore<PackSize>(dest + sh0 + offset_first, shz0);
            avx::cxstore<PackSize>(dest + sh1 + offset_first, shz1);
            avx::cxstore<PackSize>(dest + sh4 + offset_first, shz2);
            avx::cxstore<PackSize>(dest + sh5 + offset_first, shz3);

            auto [shy4, shy6] = avx::unpack_pd(shx4, shx6);
            auto [shy5, shy7] = avx::unpack_pd(shx5, shx7);

            auto [shz4, shz5] = avx::unpack_128(shy4, shy5);
            auto [shz6, shz7] = avx::unpack_128(shy6, shy7);

            avx::cxstore<PackSize>(dest + sh2 + offset_first, shz4);
            avx::cxstore<PackSize>(dest + sh3 + offset_first, shz5);
            avx::cxstore<PackSize>(dest + sh6 + offset_first, shz6);
            avx::cxstore<PackSize>(dest + sh7 + offset_first, shz7);
        };

        for (; i < size() / 64; ++i)
        {
            using reg_t = avx::cx_reg<float>;
            auto offset = m_sort[i];

            auto p1 = avx::cxload<PackSize>(source + sh1 + offset);
            auto p5 = avx::cxload<PackSize>(source + sh5 + offset);
            auto p3 = avx::cxload<PackSize>(source + sh3 + offset);
            auto p7 = avx::cxload<PackSize>(source + sh7 + offset);

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

            auto p0 = avx::cxload<PackSize>(source + sh0 + offset);
            auto p2 = avx::cxload<PackSize>(source + sh2 + offset);
            auto p4 = avx::cxload<PackSize>(source + sh4 + offset);
            auto p6 = avx::cxload<PackSize>(source + sh6 + offset);

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

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);

            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            avx::cxstore<PackSize>(dest + sh0 + offset, shc0);
            avx::cxstore<PackSize>(dest + sh1 + offset, shc1);
            avx::cxstore<PackSize>(dest + sh4 + offset, shc2);
            avx::cxstore<PackSize>(dest + sh5 + offset, shc3);

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            avx::cxstore<PackSize>(dest + sh2 + offset, shc4);
            avx::cxstore<PackSize>(dest + sh3 + offset, shc5);
            avx::cxstore<PackSize>(dest + sh6 + offset, shc6);
            avx::cxstore<PackSize>(dest + sh7 + offset, shc7);
        }

        std::size_t l_size     = 0;
        std::size_t group_size = 0;
        std::size_t n_groups   = 0;
        std::size_t tw_offset  = 0;

        std::size_t sub_offset = 0;
        std::size_t sub_size_  = std::min(size(), SubSize);

        for (uint i = 0; i < size() / sub_size_; ++i)
        {
            auto* sub_source = dest + pidx(sub_offset);
            l_size           = reg_size * 2;
            group_size       = sub_size_ / reg_size / 4;
            n_groups         = 1;
            tw_offset        = 0;

            while (l_size < sub_size_)
            {
                for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
                {
                    std::size_t offset = i_group * reg_size;

                    const auto tw0 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset));
                    const auto tw1 =
                        avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset + reg_size));
                    const auto tw2 = avx::cxload<PackSize>(
                        twiddle_ptr + pidx(tw_offset + reg_size * 2));

                    tw_offset += reg_size * 3;

                    for (std::size_t i = 0; i < group_size; ++i)
                    {
                        auto* ptr0 = sub_source + pidx(offset);
                        auto* ptr1 = sub_source + pidx(offset + l_size / 2);
                        auto* ptr2 = sub_source + pidx(offset + l_size);
                        auto* ptr3 = sub_source + pidx(offset + l_size / 2 * 3);

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

            if (l_size == sub_size_)
            {
                for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
                {
                    std::size_t offset = i_group * reg_size;

                    const auto tw0 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset));

                    tw_offset += reg_size;

                    auto* ptr0 = sub_source + pidx(offset);
                    auto* ptr1 = sub_source + pidx(offset + l_size / 2);

                    auto p1 = avx::cxload<PackSize>(ptr1);
                    auto p0 = avx::cxload<PackSize>(ptr0);

                    auto p1tw = avx::mul(p1, tw0);

                    auto a0 = avx::add(p0, p1tw);
                    auto a1 = avx::sub(p0, p1tw);

                    cxstore<PackSize>(ptr0, a0);
                    cxstore<PackSize>(ptr1, a1);
                }
                l_size *= 2;
                n_groups *= 2;
                group_size /= 2;
            }

            sub_offset += SubSize;
        }

        group_size = size() / SubSize / 2;
        while (l_size <= size())
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset));

                tw_offset += reg_size;

                for (std::size_t i = 0; i < group_size; ++i)
                {
                    auto* ptr0 = dest + pidx(offset);
                    auto* ptr1 = dest + pidx(offset + l_size / 2);

                    auto p1 = avx::cxload<PackSize>(ptr1);
                    auto p0 = avx::cxload<PackSize>(ptr0);

                    auto p1tw = avx::mul(p1, tw0);

                    auto a0 = avx::add(p0, p1tw);
                    auto a1 = avx::sub(p0, p1tw);

                    cxstore<PackSize>(ptr0, a0);
                    cxstore<PackSize>(ptr1, a1);

                    offset += l_size;
                }
            }

            l_size *= 2;
            n_groups *= 2;
            group_size /= 2;
        }
    };

    void fft_internal_separated(float* source)
        requires(SubSize != pcx::dynamic_size)
    {
        const auto sq2 = wnk(8, 1);

        auto twsq2 = avx::broadcast(sq2.real());

        auto* twiddle_ptr = m_twiddles.data();

        dept3_sort_separate(source, source);

        std::size_t l_size     = 0;
        std::size_t group_size = 0;
        std::size_t n_groups   = 0;
        std::size_t tw_offset  = 0;

        std::size_t sub_offset = 0;
        std::size_t sub_size_  = std::min(size(), SubSize);

        for (uint i = 0; i < size() / sub_size_; ++i)
        {
            auto* sub_source = source + pidx(sub_offset);
            l_size           = reg_size * 2;
            group_size       = sub_size_ / reg_size / 4;
            n_groups         = 1;
            tw_offset        = 0;

            while (l_size < sub_size_)
            {
                for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
                {
                    std::size_t offset = i_group * reg_size;

                    const auto tw0 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset));
                    const auto tw1 =
                        avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset + reg_size));
                    const auto tw2 = avx::cxload<PackSize>(
                        twiddle_ptr + pidx(tw_offset + reg_size * 2));

                    tw_offset += reg_size * 3;

                    for (std::size_t i = 0; i < group_size; ++i)
                    {
                        auto* ptr0 = sub_source + pidx(offset);
                        auto* ptr1 = sub_source + pidx(offset + l_size / 2);
                        auto* ptr2 = sub_source + pidx(offset + l_size);
                        auto* ptr3 = sub_source + pidx(offset + l_size / 2 * 3);

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

            if (l_size == sub_size_)
            {
                for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
                {
                    std::size_t offset = i_group * reg_size;

                    const auto tw0 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset));

                    tw_offset += reg_size;

                    auto* ptr0 = sub_source + pidx(offset);
                    auto* ptr1 = sub_source + pidx(offset + l_size / 2);

                    auto p1 = avx::cxload<PackSize>(ptr1);
                    auto p0 = avx::cxload<PackSize>(ptr0);

                    auto p1tw = avx::mul(p1, tw0);

                    auto a0 = avx::add(p0, p1tw);
                    auto a1 = avx::sub(p0, p1tw);

                    cxstore<PackSize>(ptr0, a0);
                    cxstore<PackSize>(ptr1, a1);
                }
                l_size *= 2;
                n_groups *= 2;
                group_size /= 2;
            }

            sub_offset += SubSize;
        }

        group_size = size() / SubSize / 2;
        while (l_size <= size())
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset));

                tw_offset += reg_size;

                for (std::size_t i = 0; i < group_size; ++i)
                {
                    auto* ptr0 = source + pidx(offset);
                    auto* ptr1 = source + pidx(offset + l_size / 2);

                    auto p1 = avx::cxload<PackSize>(ptr1);
                    auto p0 = avx::cxload<PackSize>(ptr0);

                    auto p1tw = avx::mul(p1, tw0);

                    auto a0 = avx::add(p0, p1tw);
                    auto a1 = avx::sub(p0, p1tw);

                    cxstore<PackSize>(ptr0, a0);
                    cxstore<PackSize>(ptr1, a1);

                    offset += l_size;
                }
            }

            l_size *= 2;
            n_groups *= 2;
            group_size /= 2;
        }
    };

    inline void dept3_sort_separate(float* dest, const float* source)
    {
        using reg_t = avx::cx_reg<float>;

        const auto sq2   = wnk(8, 1);
        auto       twsq2 = avx::broadcast(sq2.real());

        auto* twiddle_ptr = m_twiddles.data();

        const auto sh0 = 0;
        const auto sh1 = pidx(1 * size() / 8);
        const auto sh2 = pidx(2 * size() / 8);
        const auto sh3 = pidx(3 * size() / 8);
        const auto sh4 = pidx(4 * size() / 8);
        const auto sh5 = pidx(5 * size() / 8);
        const auto sh6 = pidx(6 * size() / 8);
        const auto sh7 = pidx(7 * size() / 8);

        for (uint i = 0; i < size() / reg_size / 2; ++i)
        {
            auto offset = pidx(reg_size * i);

            auto p0 = avx::cxload<PackSize>(source + sh0 + offset);
            auto p4 = avx::cxload<PackSize>(source + sh4 + offset);

            auto a0 = avx::add(p0, p4);
            auto a4 = avx::sub(p0, p4);

            auto [sha0, sha4] = avx::unpack_ps(a0, a4);

            avx::cxstore<PackSize>(dest + sh0 + offset, a0);
            avx::cxstore<PackSize>(dest + sh4 + offset, a4);
        }

        for (uint i = 0; i < size() / reg_size / 4; ++i)
        {
            auto offset = pidx(reg_size * i);

            auto p0 = avx::cxload<PackSize>(dest + sh0 + offset);
            auto p2 = avx::cxload<PackSize>(dest + sh2 + offset);

            auto a0 = avx::add(p0, p2);
            auto a2 = avx::sub(p0, p2);

            auto [sha0, sha2] = avx::unpack_pd(a0, a2);

            avx::cxstore<PackSize>(dest + sh0 + offset, a0);
            avx::cxstore<PackSize>(dest + sh2 + offset, a2);
        }
        for (uint i = 0; i < size() / reg_size / 4; ++i)
        {
            auto offset = pidx(reg_size * i);

            auto p4 = avx::cxload<PackSize>(dest + sh4 + offset);
            auto p6 = avx::cxload<PackSize>(dest + sh6 + offset);

            reg_t a4 = {avx::add(p4.real, p6.imag), avx::sub(p4.imag, p6.real)};
            reg_t a6 = {avx::sub(p4.real, p6.imag), avx::add(p4.imag, p6.real)};

            auto [sha4, sha6] = avx::unpack_pd(a4, a6);

            avx::cxstore<PackSize>(dest + sh4 + offset, a4);
            avx::cxstore<PackSize>(dest + sh6 + offset, a6);
        }

        for (uint i = 0; i < size() / reg_size / 8; ++i)
        {
            auto offset = pidx(reg_size * i);

            auto p0 = avx::cxload<PackSize>(dest + sh0 + offset);
            auto p1 = avx::cxload<PackSize>(dest + sh1 + offset);

            auto a0 = avx::add(p0, p1);
            auto a1 = avx::sub(p0, p1);

            auto [sha0, sha1] = avx::unpack_128(a0, a1);

            avx::cxstore<PackSize>(dest + sh0 + offset, sha0);
            avx::cxstore<PackSize>(dest + sh1 + offset, sha1);
        }
        for (uint i = 0; i < size() / reg_size / 8; ++i)
        {
            auto offset = pidx(reg_size * i);

            auto p2 = avx::cxload<PackSize>(dest + sh2 + offset);
            auto p3 = avx::cxload<PackSize>(dest + sh3 + offset);

            reg_t a2 = {avx::add(p2.real, p3.imag), avx::sub(p2.imag, p3.real)};
            reg_t a3 = {avx::sub(p2.real, p3.imag), avx::add(p2.imag, p3.real)};

            auto [sha2, sha3] = avx::unpack_128(a2, a3);
            avx::cxstore<PackSize>(dest + sh2 + offset, sha2);
            avx::cxstore<PackSize>(dest + sh3 + offset, sha3);
        }
        for (uint i = 0; i < size() / reg_size / 8; ++i)
        {
            auto offset = pidx(reg_size * i);

            auto p4 = avx::cxload<PackSize>(dest + sh4 + offset);
            auto p5 = avx::cxload<PackSize>(dest + sh5 + offset);

            reg_t p5_tw = {avx::add(p5.real, p5.imag), avx::sub(p5.imag, p5.real)};
            p5_tw       = avx::mul(p5_tw, twsq2);

            reg_t a4 = avx::add(p4, p5_tw);
            reg_t a5 = avx::sub(p4, p5_tw);

            auto [sha4, sha5] = avx::unpack_128(a4, a5);

            avx::cxstore<PackSize>(dest + sh4 + offset, sha4);
            avx::cxstore<PackSize>(dest + sh5 + offset, sha5);
        }
        for (uint i = 0; i < size() / reg_size / 8; ++i)
        {
            auto offset = pidx(reg_size * i);

            auto p6 = avx::cxload<PackSize>(dest + sh6 + offset);
            auto p7 = avx::cxload<PackSize>(dest + sh7 + offset);

            reg_t p7_tw = {avx::sub(p7.real, p7.imag), avx::add(p7.real, p7.imag)};
            p7_tw       = avx::mul(p7_tw, twsq2);

            reg_t a6 = avx::sub(p6, p7_tw);
            reg_t a7 = avx::add(p6, p7_tw);

            auto [sha6, sha7] = avx::unpack_128(a6, a7);

            avx::cxstore<PackSize>(dest + sh6 + offset, sha6);
            avx::cxstore<PackSize>(dest + sh7 + offset, sha7);
        }

        uint i = 0;
        for (; i < n_reversals(size() / 64); i += 2)
        {
            auto offset_first  = m_sort[i];
            auto offset_second = m_sort[i + 1];

            auto p0 = avx::cxload<PackSize>(dest + sh0 + offset_first);
            auto p4 = avx::cxload<PackSize>(dest + sh4 + offset_first);
            auto p2 = avx::cxload<PackSize>(dest + sh2 + offset_first);
            auto p6 = avx::cxload<PackSize>(dest + sh6 + offset_first);

            auto [sha0, sha4] = avx::unpack_ps(p0, p4);
            auto [sha2, sha6] = avx::unpack_ps(p2, p6);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);

            auto q0 = avx::cxload<PackSize>(dest + sh0 + offset_second);
            avx::cxstore<PackSize>(dest + sh0 + offset_second, shb0);
            auto q4 = avx::cxload<PackSize>(dest + sh4 + offset_second);
            avx::cxstore<PackSize>(dest + sh4 + offset_second, shb2);
            auto q2 = avx::cxload<PackSize>(dest + sh2 + offset_second);
            avx::cxstore<PackSize>(dest + sh2 + offset_second, shb4);
            auto q6 = avx::cxload<PackSize>(dest + sh6 + offset_second);
            avx::cxstore<PackSize>(dest + sh6 + offset_second, shb6);

            auto [shx0, shx4] = avx::unpack_ps(q0, q4);
            auto [shx2, shx6] = avx::unpack_ps(q2, q6);

            auto [shy0, shy2] = avx::unpack_pd(shx0, shx2);
            auto [shy4, shy6] = avx::unpack_pd(shx4, shx6);

            avx::cxstore<PackSize>(dest + sh0 + offset_first, shy0);
            avx::cxstore<PackSize>(dest + sh4 + offset_first, shy2);
            avx::cxstore<PackSize>(dest + sh2 + offset_first, shy4);
            avx::cxstore<PackSize>(dest + sh6 + offset_first, shy6);
        };
        for (; i < size() / 64; ++i)
        {
            auto offset = m_sort[i];

            auto p0 = avx::cxload<PackSize>(dest + sh0 + offset);
            auto p4 = avx::cxload<PackSize>(dest + sh4 + offset);
            auto p2 = avx::cxload<PackSize>(dest + sh2 + offset);
            auto p6 = avx::cxload<PackSize>(dest + sh6 + offset);

            auto [sha0, sha4] = avx::unpack_ps(p0, p4);
            auto [sha2, sha6] = avx::unpack_ps(p2, p6);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);

            avx::cxstore<PackSize>(dest + sh0 + offset, shb0);
            avx::cxstore<PackSize>(dest + sh4 + offset, shb2);
            avx::cxstore<PackSize>(dest + sh6 + offset, shb6);
            avx::cxstore<PackSize>(dest + sh2 + offset, shb4);
        }
        i = 0;
        for (; i < n_reversals(size() / 64); i += 2)
        {
            auto offset_first  = m_sort[i];
            auto offset_second = m_sort[i + 1];
            auto p1            = avx::cxload<PackSize>(dest + sh1 + offset_first);
            auto p3            = avx::cxload<PackSize>(dest + sh3 + offset_first);
            auto p5            = avx::cxload<PackSize>(dest + sh5 + offset_first);
            auto p7            = avx::cxload<PackSize>(dest + sh7 + offset_first);

            auto [sha1, sha5] = avx::unpack_ps(p1, p5);
            auto [sha3, sha7] = avx::unpack_ps(p3, p7);

            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);


            auto q1 = avx::cxload<PackSize>(dest + sh1 + offset_second);
            avx::cxstore<PackSize>(dest + sh1 + offset_second, shb1);
            auto q5 = avx::cxload<PackSize>(dest + sh5 + offset_second);
            avx::cxstore<PackSize>(dest + sh5 + offset_second, shb3);

            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);
            auto q3           = avx::cxload<PackSize>(dest + sh3 + offset_second);
            avx::cxstore<PackSize>(dest + sh3 + offset_second, shb5);
            auto q7 = avx::cxload<PackSize>(dest + sh7 + offset_second);
            avx::cxstore<PackSize>(dest + sh7 + offset_second, shb7);

            auto [shx1, shx5] = avx::unpack_ps(q1, q5);
            auto [shx3, shx7] = avx::unpack_ps(q3, q7);

            auto [shy1, shy3] = avx::unpack_pd(shx1, shx3);
            auto [shy5, shy7] = avx::unpack_pd(shx5, shx7);

            avx::cxstore<PackSize>(dest + sh1 + offset_first, shy1);
            avx::cxstore<PackSize>(dest + sh5 + offset_first, shy3);
            avx::cxstore<PackSize>(dest + sh3 + offset_first, shy5);
            avx::cxstore<PackSize>(dest + sh7 + offset_first, shy7);
        };
        for (; i < size() / 64; ++i)
        {
            auto offset = m_sort[i];

            auto p1 = avx::cxload<PackSize>(dest + sh1 + offset);
            auto p3 = avx::cxload<PackSize>(dest + sh3 + offset);
            auto p5 = avx::cxload<PackSize>(dest + sh5 + offset);
            auto p7 = avx::cxload<PackSize>(dest + sh7 + offset);

            auto [sha1, sha5] = avx::unpack_ps(p1, p5);
            auto [sha3, sha7] = avx::unpack_ps(p3, p7);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);


            avx::cxstore<PackSize>(dest + sh1 + offset, shb1);
            avx::cxstore<PackSize>(dest + sh5 + offset, shb3);
            avx::cxstore<PackSize>(dest + sh3 + offset, shb5);
            avx::cxstore<PackSize>(dest + sh7 + offset, shb7);
        }
    };

    inline void dept3_and_sort(float* source)
    {
        const auto sq2   = wnk(8, 1);
        auto       twsq2 = avx::broadcast(sq2.real());

        const auto sh0 = 0;
        const auto sh1 = pidx(1 * size() / 8);
        const auto sh2 = pidx(2 * size() / 8);
        const auto sh3 = pidx(3 * size() / 8);
        const auto sh4 = pidx(4 * size() / 8);
        const auto sh5 = pidx(5 * size() / 8);
        const auto sh6 = pidx(6 * size() / 8);
        const auto sh7 = pidx(7 * size() / 8);

        uint i = 0;

        for (; i < n_reversals(size() / 64); i += 2)
        {
            using reg_t = avx::cx_reg<float>;

            auto offset_first  = m_sort[i];
            auto offset_second = m_sort[i + 1];

            auto p1 = avx::cxload<PackSize>(source + sh1 + offset_first);
            auto p5 = avx::cxload<PackSize>(source + sh5 + offset_first);
            auto p3 = avx::cxload<PackSize>(source + sh3 + offset_first);
            auto p7 = avx::cxload<PackSize>(source + sh7 + offset_first);

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

            auto p0 = avx::cxload<PackSize>(source + sh0 + offset_first);
            auto p4 = avx::cxload<PackSize>(source + sh4 + offset_first);
            auto p2 = avx::cxload<PackSize>(source + sh2 + offset_first);
            auto p6 = avx::cxload<PackSize>(source + sh6 + offset_first);

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

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);

            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            auto q0 = avx::cxload<PackSize>(source + sh0 + offset_second);
            avx::cxstore<PackSize>(source + sh0 + offset_second, shc0);
            auto q1 = avx::cxload<PackSize>(source + sh1 + offset_second);
            avx::cxstore<PackSize>(source + sh1 + offset_second, shc1);
            auto q4 = avx::cxload<PackSize>(source + sh4 + offset_second);
            avx::cxstore<PackSize>(source + sh4 + offset_second, shc2);
            auto q5 = avx::cxload<PackSize>(source + sh5 + offset_second);
            avx::cxstore<PackSize>(source + sh5 + offset_second, shc3);

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);

            auto q2 = avx::cxload<PackSize>(source + sh2 + offset_second);
            avx::cxstore<PackSize>(source + sh2 + offset_second, shc4);
            auto q3 = avx::cxload<PackSize>(source + sh3 + offset_second);
            avx::cxstore<PackSize>(source + sh3 + offset_second, shc5);

            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            auto q6 = avx::cxload<PackSize>(source + sh6 + offset_second);
            avx::cxstore<PackSize>(source + sh6 + offset_second, shc6);
            auto q7 = avx::cxload<PackSize>(source + sh7 + offset_second);
            avx::cxstore<PackSize>(source + sh7 + offset_second, shc7);

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

            auto [shx0, shx4] = avx::unpack_ps(z0, z4);
            auto [shx1, shx5] = avx::unpack_ps(z1, z5);
            auto [shx2, shx6] = avx::unpack_ps(z2, z6);
            auto [shx3, shx7] = avx::unpack_ps(z3, z7);

            auto [shy0, shy2] = avx::unpack_pd(shx0, shx2);
            auto [shy1, shy3] = avx::unpack_pd(shx1, shx3);

            auto [shz0, shz1] = avx::unpack_128(shy0, shy1);
            auto [shz2, shz3] = avx::unpack_128(shy2, shy3);

            avx::cxstore<PackSize>(source + sh0 + offset_first, shz0);
            avx::cxstore<PackSize>(source + sh1 + offset_first, shz1);
            avx::cxstore<PackSize>(source + sh4 + offset_first, shz2);
            avx::cxstore<PackSize>(source + sh5 + offset_first, shz3);

            auto [shy4, shy6] = avx::unpack_pd(shx4, shx6);
            auto [shy5, shy7] = avx::unpack_pd(shx5, shx7);

            auto [shz4, shz5] = avx::unpack_128(shy4, shy5);
            auto [shz6, shz7] = avx::unpack_128(shy6, shy7);

            avx::cxstore<PackSize>(source + sh2 + offset_first, shz4);
            avx::cxstore<PackSize>(source + sh3 + offset_first, shz5);
            avx::cxstore<PackSize>(source + sh6 + offset_first, shz6);
            avx::cxstore<PackSize>(source + sh7 + offset_first, shz7);
        };

        for (; i < size() / 64; ++i)
        {
            using reg_t = avx::cx_reg<float>;
            auto offset = m_sort[i];

            auto p1 = avx::cxload<PackSize>(source + sh1 + offset);
            auto p3 = avx::cxload<PackSize>(source + sh3 + offset);
            auto p5 = avx::cxload<PackSize>(source + sh5 + offset);
            auto p7 = avx::cxload<PackSize>(source + sh7 + offset);

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

            auto p0 = avx::cxload<PackSize>(source + sh0 + offset);
            auto p2 = avx::cxload<PackSize>(source + sh2 + offset);
            auto p4 = avx::cxload<PackSize>(source + sh4 + offset);
            auto p6 = avx::cxload<PackSize>(source + sh6 + offset);

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

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);

            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            avx::cxstore<PackSize>(source + sh0 + offset, shc0);
            avx::cxstore<PackSize>(source + sh1 + offset, shc1);
            avx::cxstore<PackSize>(source + sh4 + offset, shc2);
            avx::cxstore<PackSize>(source + sh5 + offset, shc3);

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            avx::cxstore<PackSize>(source + sh2 + offset, shc4);
            avx::cxstore<PackSize>(source + sh3 + offset, shc5);
            avx::cxstore<PackSize>(source + sh6 + offset, shc6);
            avx::cxstore<PackSize>(source + sh7 + offset, shc7);
        }
    }

    inline auto sized_sub_transorm(float* source, std::size_t size)
        -> std::array<std::size_t, 2>
    {
        auto* twiddle_ptr = m_twiddles.data();

        std::size_t l_size     = reg_size * 2;
        std::size_t group_size = size / reg_size / 4;
        std::size_t n_groups   = 1;
        std::size_t tw_offset  = 0;

        while (l_size < size)
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
                    auto* ptr0 = source + pidx(offset);
                    auto* ptr1 = source + pidx(offset + l_size / 2);
                    auto* ptr2 = source + pidx(offset + l_size);
                    auto* ptr3 = source + pidx(offset + l_size / 2 * 3);

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

        if (l_size == size)
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset));

                tw_offset += reg_size;

                auto* ptr0 = source + pidx(offset);
                auto* ptr1 = source + pidx(offset + l_size / 2);

                auto p1 = avx::cxload<PackSize>(ptr1);
                auto p0 = avx::cxload<PackSize>(ptr0);

                auto p1tw = avx::mul(p1, tw0);

                auto a0 = avx::add(p0, p1tw);
                auto a1 = avx::sub(p0, p1tw);

                cxstore<PackSize>(ptr0, a0);
                cxstore<PackSize>(ptr1, a1);
            }
            l_size *= 2;
            n_groups *= 2;
            group_size /= 2;
        }

        return {tw_offset, n_groups};
    }

    inline auto recursive_sub_transform(float*      source,
                                        std::size_t size,
                                        std::size_t sub_size)
        -> std::array<std::size_t, 2>
    {
        if (size == sub_size)
        {
            return sized_sub_transorm(source, sub_size);
        } else
        {
            auto* twiddle_ptr = m_twiddles.data();
            recursive_sub_transform(source, size / 2, sub_size);
            auto [tw_offset, n_groups] =
                recursive_sub_transform(source + pidx(size / 2), size / 2, sub_size);

            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<PackSize>(twiddle_ptr + pidx(tw_offset));

                tw_offset += reg_size;

                auto* ptr0 = source + pidx(offset);
                auto* ptr1 = source + pidx(offset + size / 2);

                auto p1 = avx::cxload<PackSize>(ptr1);
                auto p0 = avx::cxload<PackSize>(ptr0);

                auto p1tw = avx::mul(p1, tw0);

                auto a0 = avx::add(p0, p1tw);
                auto a1 = avx::sub(p0, p1tw);

                cxstore<PackSize>(ptr0, a0);
                cxstore<PackSize>(ptr1, a1);
            }

            return {tw_offset, n_groups * 2};
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

        std::size_t sub_size_ = std::min(fft_size, SubSize);

        while (l_size < sub_size_)
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

        while (l_size <= fft_size)
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                for (uint k = 0; k < reg_size; ++k)
                {
                    *(tw_it++) = wnk(l_size, k + i_group * reg_size);
                }
            }
            l_size *= 2;
            n_groups *= 2;
        };
        return twiddles;
    }
};


}    // namespace pcx
#endif