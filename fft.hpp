#ifndef FFT_HPP
#define FFT_HPP

#include "vector.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>

namespace pcx {

template<typename T,
         std::size_t Size     = pcx::dynamic_size,
         std::size_t SubSize  = pcx::default_pack_size<T>,
         typename Allocator   = std::allocator<T>,
         std::size_t PackSize = pcx::default_pack_size<T>>
    requires(std::same_as<T, float> || std::same_as<T, double>) &&
            (pcx::power_of_two<Size> || (Size == pcx::dynamic_size)) &&
            (pcx::power_of_two<SubSize> && SubSize >= pcx::default_pack_size<T> ||
             (SubSize == pcx::dynamic_size)) &&
            pcx::power_of_two<PackSize> && (PackSize >= pcx::default_pack_size<T>)
class fft_unit
{
public:
    using real_type      = T;
    using allocator_type = Allocator;

    // static constexpr auto pack_size = default_pack_size<T>;
    static constexpr std::size_t reg_size = 32 / sizeof(real_type);

private:
    using size_t =
        std::conditional_t<Size == pcx::dynamic_size, std::size_t, decltype([]() {})>;
    using subsize_t =
        std::conditional_t<SubSize == pcx::dynamic_size, std::size_t, decltype([]() {})>;
    using sort_allocator_type = typename std::allocator_traits<
        allocator_type>::template rebind_alloc<std::size_t>;

public:
    fft_unit(allocator_type allocator = allocator_type())
        requires(Size != pcx::dynamic_size) && (SubSize != pcx::dynamic_size)
    : m_sort(get_sort(size(), static_cast<sort_allocator_type>(allocator)))
    , m_twiddles(get_twiddles(size(), sub_size(), allocator))
    , m_twiddles4(get_twiddles4(size(), sub_size(), allocator))
    , m_twiddles_unsorted(get_twiddles_unsorted(size(), sub_size(), allocator))
    , m_twiddles_unsorted_linear(
          get_twiddles_unsorted_linear(size(), sub_size(), allocator)){};

    fft_unit(std::size_t sub_size = 1, allocator_type allocator = allocator_type())
        requires(Size != pcx::dynamic_size) && (SubSize == pcx::dynamic_size)
    : m_sub_size(check_sub_size(sub_size))
    , m_sort(get_sort(size(), static_cast<sort_allocator_type>(allocator)))
    , m_twiddles(get_twiddles(size(), sub_size, allocator))
    , m_twiddles4(get_twiddles4(size(), sub_size, allocator))
    , m_twiddles_unsorted(get_twiddles_unsorted(size(), sub_size, allocator))
    , m_twiddles_unsorted_linear(
          get_twiddles_unsorted_linear(size(), sub_size, allocator)){};

    fft_unit(std::size_t fft_size, allocator_type allocator = allocator_type())
        requires(Size == pcx::dynamic_size) && (SubSize != pcx::dynamic_size)
    : m_size(check_size(fft_size))
    , m_sort(get_sort(size(), static_cast<sort_allocator_type>(allocator)))
    , m_twiddles(get_twiddles(size(), sub_size(), allocator))
    , m_twiddles4(get_twiddles4(size(), sub_size(), allocator))
    , m_twiddles_unsorted(get_twiddles_unsorted(size(), sub_size(), allocator))
    , m_twiddles_unsorted_linear(
          get_twiddles_unsorted_linear(size(), sub_size(), allocator)){};

    fft_unit(std::size_t    fft_size,
             std::size_t    sub_size  = 1,
             allocator_type allocator = allocator_type())
        requires(Size == pcx::dynamic_size) && (SubSize == pcx::dynamic_size)
    : m_size(check_size(fft_size))
    , m_sub_size(check_sub_size(sub_size))
    , m_sort(get_sort(size(), static_cast<sort_allocator_type>(allocator)))
    , m_twiddles(get_twiddles(size(), sub_size, allocator))
    , m_twiddles4(get_twiddles4(size(), sub_size, allocator))
    , m_twiddles_unsorted(get_twiddles_unsorted(size(), sub_size, allocator))
    , m_twiddles_unsorted_linear(
          get_twiddles_unsorted_linear(size(), sub_size, allocator)){};

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
    void binary4(pcx::vector<T, VAllocator, PackSize>& vector)
    {
        assert(size() == vector.size());
        fft_internal_binary4<true, PackSize>(vector.data());
    };
    template<typename VAllocator>
    void unsorted(pcx::vector<T, VAllocator, PackSize>& vector)
    {
        assert(size() == vector.size());
        recursive_unsorted(vector.data(), size(), m_twiddles_unsorted.data());
        // fixed_size_unsorted(vector.data(), size(), m_twiddles_unsorted.data());
    };


    template<typename VAllocator>
    void unsorted_linear(pcx::vector<T, VAllocator, PackSize>& vector)
    {
        assert(size() == vector.size());
        linear_unsorted(vector.data(), size());
        // fixed_size_unsorted(vector.data(), size(), m_twiddles_unsorted.data());
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
    [[no_unique_address]] size_t                           m_size;
    [[no_unique_address]] subsize_t                        m_sub_size;
    const std::vector<std::size_t, sort_allocator_type>    m_sort;
    const pcx::vector<real_type, allocator_type, reg_size> m_twiddles;
    const pcx::vector<real_type, allocator_type, reg_size> m_twiddles4;
    const std::vector<real_type, allocator_type>           m_twiddles_unsorted;
    const std::vector<real_type, allocator_type>           m_twiddles_unsorted_linear;

    [[nodiscard]] constexpr auto sub_size() const -> std::size_t
    {
        if constexpr (SubSize == pcx::dynamic_size)
        {
            return m_sub_size;
        } else
        {
            return SubSize;
        }
    }

public:
    void fft_internal_binary(float* source)
    {
        dept3_and_sort(source);
        recursive_subtransform(source, size());
    }

    template<bool PackedSrc, std::size_t TMPPackSize>
    void fft_internal_binary4(float* source)
    {
        dept3_and_sort<PackedSrc, TMPPackSize>(source);
        if (size() <= sub_size() || log2i(size() / sub_size()) % 2 == 0)
        {
            recursive_subtransform4(source, size());
        } else
        {
            recursive_subtransform4(source, size() / 2);
            auto twiddle_ptr =
                recursive_subtransform4(source + pidx(size() / 2), size() / 2);
            std::size_t n_groups = size() / reg_size / 2;

            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
                twiddle_ptr += reg_size * 2;

                auto* ptr0 = source + pidx(offset);
                auto* ptr1 = source + pidx(offset + size() / 2);

                auto p1 = avx::cxload<PackSize>(ptr1);
                auto p0 = avx::cxload<PackSize>(ptr0);

                auto p1tw = avx::mul(p1, tw0);

                auto a0 = avx::add(p0, p1tw);
                auto a1 = avx::sub(p0, p1tw);

                cxstore<PackSize>(ptr0, a0);
                cxstore<PackSize>(ptr1, a1);
            }
        }
    }

    void fft_internal(float* source)
    {
        dept3_and_sort(source);
        linear_subtransform(source);
    };

    void fft_internal(float* dest, const float* source)
    {
        dept3_and_sort(dest, source);
        linear_subtransform(dest);
    };

    template<bool PackedSrc, std::size_t TMPPackSize>
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

            auto p1 = reg_t{};
            auto p5 = reg_t{};
            auto p3 = reg_t{};
            auto p7 = reg_t{};

            if constexpr (PackedSrc)
            {
                p1 = avx::cxload<TMPPackSize>(source + sh1 + offset_first);
                p5 = avx::cxload<TMPPackSize>(source + sh5 + offset_first);
                p3 = avx::cxload<TMPPackSize>(source + sh3 + offset_first);
                p7 = avx::cxload<TMPPackSize>(source + sh7 + offset_first);
            } else
            {
                auto p1r = avx::load(source + sh1 + offset_first);
                auto p1i = avx::load(source + sh1 + offset_first + TMPPackSize);
                auto p5r = avx::load(source + sh5 + offset_first);
                auto p5i = avx::load(source + sh5 + offset_first + TMPPackSize);
                auto p3r = avx::load(source + sh3 + offset_first);
                auto p3i = avx::load(source + sh3 + offset_first + TMPPackSize);
                auto p7r = avx::load(source + sh7 + offset_first);
                auto p7i = avx::load(source + sh7 + offset_first + TMPPackSize);

                auto [p1r_d, p1i_d] = avx::unpack_pd(p1r, p1i);
                auto [p5r_d, p5i_d] = avx::unpack_pd(p5r, p5i);
                auto [p3r_d, p3i_d] = avx::unpack_pd(p3r, p3i);
                auto [p7r_d, p7i_d] = avx::unpack_pd(p7r, p7i);

                auto [p1r_s, p1i_s] = avx::unpack_ps(p1r_d, p1i_d);
                auto [p5r_s, p5i_s] = avx::unpack_ps(p5r_d, p5i_d);
                auto [p3r_s, p3i_s] = avx::unpack_ps(p3r_d, p3i_d);
                auto [p7r_s, p7i_s] = avx::unpack_ps(p7r_d, p7i_d);

                auto [p1r_128, p1i_128] = avx::unpack_ps(p1r_s, p1i_s);
                auto [p5r_128, p5i_128] = avx::unpack_ps(p5r_s, p5i_s);
                auto [p3r_128, p3i_128] = avx::unpack_ps(p3r_s, p3i_s);
                auto [p7r_128, p7i_128] = avx::unpack_ps(p7r_s, p7i_s);

                auto [p1_re, p1_im] = avx::unpack_pd(p1r_128, p1i_128);
                auto [p5_re, p5_im] = avx::unpack_pd(p5r_128, p5i_128);
                auto [p3_re, p3_im] = avx::unpack_pd(p3r_128, p3i_128);
                auto [p7_re, p7_im] = avx::unpack_pd(p7r_128, p7i_128);

                p1 = {p1_re, p1_im};
                p5 = {p5_re, p5_im};
                p3 = {p3_re, p3_im};
                p7 = {p7_re, p7_im};
            }

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

    inline void dept3_and_sort(float* dest, const float* source)
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
    }

    inline auto fixed_size_unsorted(float*       data,
                                    std::size_t  size,
                                    const float* twiddle_ptr) -> const float*
    {
        using reg_t = avx::cx_reg<float>;

        std::size_t l_size     = size;
        std::size_t group_size = size / reg_size / 2;
        std::size_t n_groups   = 1;

        while (l_size > reg_size * 8)
        {
            for (uint i_group = 0; i_group < n_groups; ++i_group)
            {
                reg_t tw0 = {avx::broadcast(twiddle_ptr++),
                             avx::broadcast(twiddle_ptr++)};

                reg_t tw1 = {avx::broadcast(twiddle_ptr++),
                             avx::broadcast(twiddle_ptr++)};
                reg_t tw2 = {avx::broadcast(twiddle_ptr++),
                             avx::broadcast(twiddle_ptr++)};

                auto* group_ptr = data + pidx(i_group * l_size);

                for (std::size_t i = 0; i < group_size / 2; ++i)
                {
                    std::size_t offset = i * reg_size;

                    auto* ptr0 = group_ptr + pidx(offset);
                    auto* ptr1 = group_ptr + pidx(offset + l_size / 2);
                    auto* ptr2 = group_ptr + pidx(offset + l_size / 4);
                    auto* ptr3 = group_ptr + pidx(offset + l_size / 4 * 3);

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
                }
            }
            l_size /= 4;
            n_groups *= 4;
            group_size /= 4;
        }

        if (l_size == reg_size * 8)
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                reg_t tw0 = {avx::broadcast(twiddle_ptr++),
                             avx::broadcast(twiddle_ptr++)};

                auto* group_ptr = data + pidx(i_group * l_size);

                for (std::size_t i = 0; i < group_size; ++i)
                {
                    std::size_t offset = i * reg_size;
                    auto*       ptr0   = group_ptr + pidx(offset);
                    auto*       ptr1   = group_ptr + pidx(offset + l_size / 2);

                    auto p1 = avx::cxload<PackSize>(ptr1);
                    auto p0 = avx::cxload<PackSize>(ptr0);

                    auto p1tw = avx::mul(p1, tw0);

                    auto a0 = avx::add(p0, p1tw);
                    auto a1 = avx::sub(p0, p1tw);

                    cxstore<PackSize>(ptr0, a0);
                    cxstore<PackSize>(ptr1, a1);
                }
            }
            l_size /= 2;
            n_groups *= 2;
            group_size /= 2;
        }

        if (l_size == reg_size * 4)
        {
            for (std::size_t i_group = 0; i_group < size / reg_size / 4; ++i_group)
            {
                reg_t tw0 = {avx::broadcast(twiddle_ptr++),
                             avx::broadcast(twiddle_ptr++)};

                reg_t tw1 = {avx::broadcast(twiddle_ptr++),
                             avx::broadcast(twiddle_ptr++)};
                reg_t tw2 = {avx::broadcast(twiddle_ptr++),
                             avx::broadcast(twiddle_ptr++)};

                reg_t tw3_1 = {avx::broadcast(twiddle_ptr++),
                               avx::broadcast(twiddle_ptr++)};
                reg_t tw3_2 = {avx::broadcast(twiddle_ptr++),
                               avx::broadcast(twiddle_ptr++)};
                reg_t tw3   = {avx::unpacklo_128(tw3_1.real, tw3_2.real),
                               avx::unpacklo_128(tw3_1.imag, tw3_2.imag)};
                reg_t tw4_1 = {avx::broadcast(twiddle_ptr++),
                               avx::broadcast(twiddle_ptr++)};
                reg_t tw4_2 = {avx::broadcast(twiddle_ptr++),
                               avx::broadcast(twiddle_ptr++)};
                reg_t tw4   = {avx::unpacklo_128(tw4_1.real, tw4_2.real),
                               avx::unpacklo_128(tw4_1.imag, tw4_2.imag)};

                auto tw56 = avx::cxload<PackSize>(twiddle_ptr);
                twiddle_ptr += reg_size * 2;
                auto [tw5, tw6] = avx::unpack_ps(tw56, tw56);

                auto tw7 = avx::cxload<PackSize>(twiddle_ptr);
                twiddle_ptr += reg_size * 2;
                auto tw8 = avx::cxload<PackSize>(twiddle_ptr);
                twiddle_ptr += reg_size * 2;

                std::size_t offset = i_group * reg_size * 4;

                auto* ptr0 = data + pidx(offset);
                auto* ptr1 = data + pidx(offset + reg_size);
                auto* ptr2 = data + pidx(offset + reg_size * 2);
                auto* ptr3 = data + pidx(offset + reg_size * 3);

                auto p2 = avx::cxload<PackSize>(ptr2);
                auto p3 = avx::cxload<PackSize>(ptr3);
                auto p0 = avx::cxload<PackSize>(ptr0);
                auto p1 = avx::cxload<PackSize>(ptr1);

                auto p2tw_re = avx::mul(p2.real, tw0.real);
                auto p3tw_re = avx::mul(p3.real, tw0.real);
                auto p2tw_im = avx::mul(p2.real, tw0.imag);
                auto p3tw_im = avx::mul(p3.real, tw0.imag);

                p2tw_re = avx::fnmadd(p2.imag, tw0.imag, p2tw_re);
                p3tw_re = avx::fnmadd(p3.imag, tw0.imag, p3tw_re);
                p2tw_im = avx::fmadd(p2.imag, tw0.real, p2tw_im);
                p3tw_im = avx::fmadd(p3.imag, tw0.real, p3tw_im);

                avx::cx_reg<float> p2tw = {p2tw_re, p2tw_im};
                avx::cx_reg<float> p3tw = {p3tw_re, p3tw_im};

                auto a1 = avx::add(p1, p3tw);
                auto a3 = avx::sub(p1, p3tw);
                auto a0 = avx::add(p0, p2tw);
                auto a2 = avx::sub(p0, p2tw);

                // cxstore<PackSize>(ptr0, a0);
                // cxstore<PackSize>(ptr1, a1);
                // cxstore<PackSize>(ptr2, a2);
                // cxstore<PackSize>(ptr3, a3);
                // continue;

                auto a1tw_re = avx::mul(a1.real, tw1.real);
                auto a1tw_im = avx::mul(a1.real, tw1.imag);
                auto a3tw_re = avx::mul(a3.real, tw2.real);
                auto a3tw_im = avx::mul(a3.real, tw2.imag);

                a1tw_re = avx::fnmadd(a1.imag, tw1.imag, a1tw_re);
                a1tw_im = avx::fmadd(a1.imag, tw1.real, a1tw_im);
                a3tw_re = avx::fnmadd(a3.imag, tw2.imag, a3tw_re);
                a3tw_im = avx::fmadd(a3.imag, tw2.real, a3tw_im);

                avx::cx_reg<float> a1tw = {a1tw_re, a1tw_im};
                avx::cx_reg<float> a3tw = {a3tw_re, a3tw_im};

                auto b0 = avx::add(a0, a1tw);
                auto b1 = avx::sub(a0, a1tw);
                auto b2 = avx::add(a2, a3tw);
                auto b3 = avx::sub(a2, a3tw);

                // cxstore<PackSize>(ptr0, b0);
                // cxstore<PackSize>(ptr1, b1);
                // cxstore<PackSize>(ptr2, b2);
                // cxstore<PackSize>(ptr3, b3);
                // continue;

                auto [shb0, shb1] = avx::unpack_128(b0, b1);
                auto [shb2, shb3] = avx::unpack_128(b2, b3);

                auto shb1tw = avx::mul(shb1, tw3);
                auto shb3tw = avx::mul(shb3, tw4);

                auto c0 = avx::add(shb0, shb1tw);
                auto c1 = avx::sub(shb0, shb1tw);
                auto c2 = avx::add(shb2, shb3tw);
                auto c3 = avx::sub(shb2, shb3tw);

                // auto [shcc0, shcc1] = avx::unpack_128(c0, c1);
                // auto [shcc2, shcc3] = avx::unpack_128(c2, c3);
                // cxstore<PackSize>(ptr0, shcc0);
                // cxstore<PackSize>(ptr1, shcc1);
                // cxstore<PackSize>(ptr2, shcc2);
                // cxstore<PackSize>(ptr3, shcc3);
                // continue;

                auto [shc0, shc1] = avx::unpack_pd(c0, c1);
                auto [shc2, shc3] = avx::unpack_pd(c2, c3);

                auto shc1tw = avx::mul(shc1, tw5);
                auto shc3tw = avx::mul(shc3, tw6);

                auto d0 = avx::add(shc0, shc1tw);
                auto d1 = avx::sub(shc0, shc1tw);
                auto d2 = avx::add(shc2, shc3tw);
                auto d3 = avx::sub(shc2, shc3tw);

                // auto [shdd0, shdd1] = avx::unpack_pd(d0, d1);
                // auto [shdd2, shdd3] = avx::unpack_pd(d2, d3);
                // auto [shddd0, shddd1] = avx::unpack_128(shdd0, shdd1);
                // auto [shddd2, shddd3] = avx::unpack_128(shdd2, shdd3);
                // cxstore<PackSize>(ptr0, shddd0);
                // cxstore<PackSize>(ptr1, shddd1);
                // cxstore<PackSize>(ptr2, shddd2);
                // cxstore<PackSize>(ptr3, shddd3);
                // continue;

                auto [shd0s, shd1s] = avx::unpack_ps(d0, d1);
                auto [shd2s, shd3s] = avx::unpack_ps(d2, d3);
                auto [shd0, shd1]   = avx::unpack_pd(shd0s, shd1s);
                auto [shd2, shd3]   = avx::unpack_pd(shd2s, shd3s);

                auto shd1tw = avx::mul(shd1, tw7);
                auto shd3tw = avx::mul(shd3, tw8);

                auto e0 = avx::add(shd0, shd1tw);
                auto e1 = avx::sub(shd0, shd1tw);
                auto e2 = avx::add(shd2, shd3tw);
                auto e3 = avx::sub(shd2, shd3tw);

                auto [she0s, she1s] = avx::unpack_ps(e0, e1);
                auto [she2s, she3s] = avx::unpack_ps(e2, e3);
                auto [she0, she1]   = avx::unpack_128(she0s, she1s);
                auto [she2, she3]   = avx::unpack_128(she2s, she3s);

                //                 auto [sht0, sht1] = avx::unpack_ps(tw0, tw0);
                //                 auto [t0, t1]     = avx::unpack_128(sht0, sht1);
                //
                //                 auto [sht2, sht3] = avx::unpack_ps(tw0, tw0);
                //                 auto [t2, t3]     = avx::unpack_128(sht2, sht3);
                cxstore<PackSize>(ptr0, she0);
                cxstore<PackSize>(ptr1, she1);
                cxstore<PackSize>(ptr2, she2);
                cxstore<PackSize>(ptr3, she3);
            }
        }

        return twiddle_ptr;
    }

    inline auto recursive_unsorted(float*       data,
                                   std::size_t  size,
                                   const float* twiddle_ptr) -> const float*
    {
        if (size <= sub_size())
        {
            return fixed_size_unsorted(data, size, twiddle_ptr);
        } else
        {
            using reg_t = avx::cx_reg<float>;
            reg_t tw0   = {
                avx::broadcast(twiddle_ptr++),
                avx::broadcast(twiddle_ptr++),
            };

            for (std::size_t offset = 0; offset < size / 2; offset += reg_size)
            {
                auto* ptr0 = data + pidx(offset);
                auto* ptr1 = data + pidx(offset + size / 2);

                auto p1 = avx::cxload<PackSize>(ptr1);
                auto p0 = avx::cxload<PackSize>(ptr0);

                auto p1tw = avx::mul(p1, tw0);

                auto a0 = avx::add(p0, p1tw);
                auto a1 = avx::sub(p0, p1tw);

                // {
                cxstore<PackSize>(ptr0, a0);
                cxstore<PackSize>(ptr1, a1);
            }

            twiddle_ptr = recursive_unsorted(data, size / 2, twiddle_ptr);
            return recursive_unsorted(data + (size / 2) * 2, size / 2, twiddle_ptr);
        }
    };

    inline auto linear_unsorted(float* data, std::size_t size)
    {
        auto        twiddle_ptr = m_twiddles_unsorted_linear.data();
        std::size_t n_groups    = 1;

        while (size > sub_size())
        {
            for (uint i_group = 0; i_group < n_groups; ++i_group)
            {
                auto* group_ptr = data + pidx(i_group * size);

                using reg_t = avx::cx_reg<float>;
                reg_t tw0   = {
                    avx::broadcast(twiddle_ptr++),
                    avx::broadcast(twiddle_ptr++),
                };
                for (uint offset = 0; offset < size / 2; offset += reg_size)
                {
                    auto* ptr0 = group_ptr + pidx(offset);
                    auto* ptr1 = group_ptr + pidx(offset + size / 2);

                    auto p1 = avx::cxload<PackSize>(ptr1);
                    auto p0 = avx::cxload<PackSize>(ptr0);

                    auto p1tw = avx::mul(p1, tw0);

                    auto a0 = avx::add(p0, p1tw);
                    auto a1 = avx::sub(p0, p1tw);

                    cxstore<PackSize>(ptr0, a0);
                    cxstore<PackSize>(ptr1, a1);
                }
            }

            n_groups *= 2;
            size /= 2;
        }

        for (uint i = 0; i < n_groups; ++i)
        {
            twiddle_ptr = fixed_size_unsorted(data + pidx(i * size), size, twiddle_ptr);
        }
    };

    inline auto fixed_size_subtransform(float* data, std::size_t max_size) -> const float*
    {
        const auto* twiddle_ptr = m_twiddles.data();

        std::size_t l_size     = reg_size * 2;
        std::size_t group_size = max_size / reg_size / 4;
        std::size_t n_groups   = 1;
        std::size_t tw_offset  = 0;

        while (l_size < max_size)
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
                const auto tw1 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 2);
                const auto tw2 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 4);
                twiddle_ptr += reg_size * 6;

                for (std::size_t i = 0; i < group_size; ++i)
                {
                    auto* ptr0 = data + pidx(offset);
                    auto* ptr1 = data + pidx(offset + l_size / 2);
                    auto* ptr2 = data + pidx(offset + l_size);
                    auto* ptr3 = data + pidx(offset + l_size / 2 * 3);

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

        if (l_size == max_size)
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
                twiddle_ptr += reg_size * 2;

                auto* ptr0 = data + pidx(offset);
                auto* ptr1 = data + pidx(offset + l_size / 2);

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

        return twiddle_ptr;
    };

    inline auto recursive_subtransform(float* data, std::size_t size) -> const float*
    {
        if (size <= sub_size())
        {
            return fixed_size_subtransform(data, size);
        } else
        {
            recursive_subtransform(data, size / 2);
            auto twiddle_ptr = recursive_subtransform(data + pidx(size / 2), size / 2);
            std::size_t n_groups = size / reg_size / 2;

            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
                twiddle_ptr += reg_size * 2;

                auto* ptr0 = data + pidx(offset);
                auto* ptr1 = data + pidx(offset + size / 2);

                auto p1 = avx::cxload<PackSize>(ptr1);
                auto p0 = avx::cxload<PackSize>(ptr0);

                auto p1tw = avx::mul(p1, tw0);

                auto a0 = avx::add(p0, p1tw);
                auto a1 = avx::sub(p0, p1tw);

                cxstore<PackSize>(ptr0, a0);
                cxstore<PackSize>(ptr1, a1);
            }

            return twiddle_ptr;
        }
    };

    inline auto fixed_size_subtransform4(float* data, std::size_t max_size) -> const
        float*
    {
        const auto* twiddle_ptr = m_twiddles4.data();

        std::size_t l_size     = reg_size * 2;
        std::size_t group_size = max_size / reg_size / 4;
        std::size_t n_groups   = 1;
        std::size_t tw_offset  = 0;

        while (l_size < max_size)
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
                const auto tw1 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 2);
                const auto tw2 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 4);
                twiddle_ptr += reg_size * 6;

                for (std::size_t i = 0; i < group_size; ++i)
                {
                    auto* ptr0 = data + pidx(offset);
                    auto* ptr1 = data + pidx(offset + l_size / 2);
                    auto* ptr2 = data + pidx(offset + l_size);
                    auto* ptr3 = data + pidx(offset + l_size / 2 * 3);

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

        if (l_size == max_size)
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
                twiddle_ptr += reg_size * 2;

                auto* ptr0 = data + pidx(offset);
                auto* ptr1 = data + pidx(offset + l_size / 2);

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

        return twiddle_ptr;
    };

    inline auto recursive_subtransform4(float* data, std::size_t size) -> const float*
    {
        if (size <= sub_size())
        {
            return fixed_size_subtransform4(data, size);
        } else
        {
            recursive_subtransform4(data, size / 4);
            recursive_subtransform4(data + pidx(size / 4), size / 4);
            recursive_subtransform4(data + pidx(size / 2), size / 4);
            auto twiddle_ptr =
                recursive_subtransform4(data + pidx(3 * size / 4), size / 4);


            std::size_t n_groups = size / reg_size / 4;


            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
                const auto tw1 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 2);
                const auto tw2 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 4);
                twiddle_ptr += reg_size * 6;

                auto* ptr0 = data + pidx(offset);
                auto* ptr1 = data + pidx(offset + size / 4);
                auto* ptr2 = data + pidx(offset + size / 2);
                auto* ptr3 = data + pidx(offset + size * 3 / 4);

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
            }

            return twiddle_ptr;
        }
    };

    inline void linear_subtransform(float* data)
    {
        std::size_t sub_size_ = std::min(size(), sub_size());

        auto        twiddle_ptr = fixed_size_subtransform(data, sub_size_);
        std::size_t sub_offset  = sub_size_;
        std::size_t n_groups    = sub_size_ / reg_size;

        for (uint i = 1; i < size() / sub_size_; ++i)
        {
            auto* sub_data = data + pidx(sub_offset);

            fixed_size_subtransform(sub_data, sub_size_);
            sub_offset += sub_size_;
        }
        std::size_t l_size     = sub_size_ * 2;
        std::size_t group_size = size() / l_size;
        while (l_size <= size())
        {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
                twiddle_ptr += reg_size * 2;

                for (std::size_t i = 0; i < group_size; ++i)
                {
                    auto* ptr0 = data + pidx(offset);
                    auto* ptr1 = data + pidx(offset + l_size / 2);

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

private:
    static constexpr auto check_size(std::size_t size) -> std::size_t
    {
        if (size > 1 && (size & (size - 1)) == 0)
        {
            return size;
        }
        throw(std::invalid_argument("fft_size (which is  " + std::to_string(size) +
                                    ") is not an integer power of two"));
    }

    static constexpr auto check_sub_size(std::size_t sub_size) -> std::size_t
    {
        if (sub_size >= pcx::default_pack_size<T> && (sub_size & (sub_size - 1)) == 0)
        {
            return sub_size;
        }
        if (sub_size < pcx::default_pack_size<T>)
        {
            throw(std::invalid_argument("fft_sub_size (which is  " +
                                        std::to_string(sub_size) +
                                        ") is smaller than minimum (which is " +
                                        std::to_string(pcx::default_pack_size<T>) + ")"));
        }
        if ((sub_size & (sub_size - 1)) != 0)
        {
            throw(std::invalid_argument("fft_sub_size (which is  " +
                                        std::to_string(sub_size) +
                                        ") is not an integer power of two"));
        }
    }

    static constexpr auto pidx(std::size_t idx) -> std::size_t
    {
        return idx + idx / PackSize * PackSize;
    }

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

    static auto get_twiddles(std::size_t    fft_size,
                             std::size_t    sub_size,
                             allocator_type allocator)
        -> pcx::vector<real_type, allocator_type, reg_size>
    {
        const auto depth = log2i(fft_size);

        const std::size_t n_twiddles = 8 * ((1U << (depth - 3)) - 1U);

        auto twiddles =
            pcx::vector<real_type, allocator_type, reg_size>(n_twiddles, allocator);

        auto tw_it = twiddles.begin();

        std::size_t l_size   = reg_size * 2;
        std::size_t n_groups = 1;

        std::size_t sub_size_ = std::min(fft_size, sub_size);

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

    static auto get_twiddles4(std::size_t    fft_size,
                              std::size_t    sub_size,
                              allocator_type allocator)
        -> pcx::vector<real_type, allocator_type, reg_size>
    {
        const auto depth = log2i(fft_size);

        const std::size_t n_twiddles = 8 * ((1U << (depth - 3)) - 1U);

        auto twiddles =
            pcx::vector<real_type, allocator_type, reg_size>(n_twiddles, allocator);

        auto tw_it = twiddles.begin();

        std::size_t l_size   = reg_size * 2;
        std::size_t n_groups = 1;

        std::size_t sub_size_ = std::min(fft_size, sub_size);

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

        if (l_size == sub_size_)
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
                    auto a     = wnk(l_size * 2UL, k + i_group * reg_size);
                    *(tw_it++) = wnk(l_size * 2UL, k + i_group * reg_size);
                }
                for (uint k = 0; k < reg_size; ++k)
                {
                    *(tw_it++) = wnk(l_size * 2UL, k + i_group * reg_size + l_size / 2);
                }
            }
            l_size *= 4;
            n_groups *= 4;
        };
        if (l_size == fft_size)
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

    static auto get_twiddles_unsorted(std::size_t    fft_size,
                                      std::size_t    sub_size,
                                      allocator_type allocator)
        -> std::vector<T, allocator_type>
    {
        auto twiddles = std::vector<T, allocator_type>(allocator);


        std::size_t l_size = 2;
        insert_twiddles_unsorted(fft_size, l_size, sub_size, 0, twiddles);

        return twiddles;
        //

        auto sub_size_ = fft_size / sub_size;
        //                 while (l_size < sub_size_)
        //                 {
        //                     for (uint i = 0; i < l_size / 2; ++i)
        //                     {
        //                         auto tw0 = wnk(l_size, reverse_bit_order(i, log2i(l_size / 2)));
        //                         twiddles.push_back(tw0.real());
        //                         twiddles.push_back(tw0.imag());
        //                     }
        //                     l_size *= 2;
        //                 }
        //
        //                 std::size_t single_load_size = fft_size / (reg_size * 2);
        //
        //                 for (uint i_group = 0; i_group < sub_size_; ++i_group)
        //                 {
        //                     std::size_t group_size = 1;
        //
        //                     while (l_size < single_load_size / 2)
        //                     {
        //                         std::size_t start = group_size * i_group;
        //
        //                         for (uint i = 0; i < group_size; ++i)
        //                         {
        //                             auto tw0 = wnk(l_size,    //
        //                                            reverse_bit_order(start + i, log2i(l_size / 2)));
        //                             auto tw1 = wnk(l_size * 2,    //
        //                                            reverse_bit_order((start + i) * 2, log2i(l_size)));
        //                             auto tw2 = wnk(l_size * 2,    //
        //                                            reverse_bit_order((start + i) * 2 + 1, log2i(l_size)));
        //
        //                             twiddles.push_back(tw0.real());
        //                             twiddles.push_back(tw0.imag());
        //                             twiddles.push_back(tw1.real());
        //                             twiddles.push_back(tw1.imag());
        //                             twiddles.push_back(tw2.real());
        //                             twiddles.push_back(tw2.imag());
        //                         }
        //                         l_size *= 4;
        //                         group_size *= 4;
        //                     }
        //
        //                     if (l_size == single_load_size / 2)
        //                     {
        //                         std::size_t start = group_size * i_group;
        //
        //                         for (uint i = 0; i < group_size; ++i)
        //                         {
        //                             auto tw0 = wnk(l_size,    //
        //                                            reverse_bit_order(start + i, log2i(l_size / 2)));
        //
        //                             twiddles.push_back(tw0.real());
        //                             twiddles.push_back(tw0.imag());
        //                         }
        //                         l_size *= 2;
        //                         group_size *= 2;
        //                     }
        //
        //                     if (l_size == single_load_size)
        //                     {
        //                         for (uint i = 0; i < group_size; ++i)
        //                         {
        //                             std::size_t start = group_size * i_group + i;
        //
        //                             auto tw0 = wnk(l_size,    //
        //                                            reverse_bit_order(start, log2i(l_size / 2)));
        //
        //                             twiddles.push_back(tw0.real());
        //                             twiddles.push_back(tw0.imag());
        //
        //                             auto tw1 = wnk(l_size * 2,    //
        //                                            reverse_bit_order(start * 2, log2i(l_size)));
        //                             auto tw2 = wnk(l_size * 2,    //
        //                                            reverse_bit_order(start * 2 + 1, log2i(l_size)));
        //
        //                             twiddles.push_back(tw1.real());
        //                             twiddles.push_back(tw1.imag());
        //                             twiddles.push_back(tw2.real());
        //                             twiddles.push_back(tw2.imag());
        //
        //                             auto tw3_1 = wnk(l_size * 4,    //
        //                                              reverse_bit_order(start * 4, log2i(l_size * 2)));
        //                             auto tw3_2 = wnk(l_size * 4,    //
        //                                              reverse_bit_order(start * 4 + 1, log2i(l_size * 2)));
        //                             auto tw4_1 = wnk(l_size * 4,    //
        //                                              reverse_bit_order(start * 4 + 2, log2i(l_size * 2)));
        //                             auto tw4_2 = wnk(l_size * 4,    //
        //                                              reverse_bit_order(start * 4 + 3, log2i(l_size * 2)));
        //
        //                             twiddles.push_back(tw3_1.real());
        //                             twiddles.push_back(tw3_1.imag());
        //                             twiddles.push_back(tw3_2.real());
        //                             twiddles.push_back(tw3_2.imag());
        //                             twiddles.push_back(tw4_1.real());
        //                             twiddles.push_back(tw4_1.imag());
        //                             twiddles.push_back(tw4_2.real());
        //                             twiddles.push_back(tw4_2.imag());
        //
        //
        //                             auto tw7  = wnk(l_size * 8,    //
        //                                            reverse_bit_order(start * 8, log2i(l_size * 4)));
        //                             auto tw8  = wnk(l_size * 8,    //
        //                                            reverse_bit_order(start * 8 + 1, log2i(l_size * 4)));
        //                             auto tw9  = wnk(l_size * 8,    //
        //                                            reverse_bit_order(start * 8 + 2, log2i(l_size * 4)));
        //                             auto tw10 = wnk(l_size * 8,    //
        //                                             reverse_bit_order(start * 8 + 3, log2i(l_size * 4)));
        //                             auto tw11 = wnk(l_size * 8,    //
        //                                             reverse_bit_order(start * 8 + 4, log2i(l_size * 4)));
        //                             auto tw12 = wnk(l_size * 8,    //
        //                                             reverse_bit_order(start * 8 + 5, log2i(l_size * 4)));
        //                             auto tw13 = wnk(l_size * 8,    //
        //                                             reverse_bit_order(start * 8 + 6, log2i(l_size * 4)));
        //                             auto tw14 = wnk(l_size * 8,    //
        //                                             reverse_bit_order(start * 8 + 7, log2i(l_size * 4)));
        //
        //                             twiddles.push_back(tw7.real());
        //                             twiddles.push_back(tw8.real());
        //                             twiddles.push_back(tw11.real());
        //                             twiddles.push_back(tw12.real());
        //                             twiddles.push_back(tw9.real());
        //                             twiddles.push_back(tw10.real());
        //                             twiddles.push_back(tw13.real());
        //                             twiddles.push_back(tw14.real());
        //
        //                             twiddles.push_back(tw7.imag());
        //                             twiddles.push_back(tw8.imag());
        //                             twiddles.push_back(tw11.imag());
        //                             twiddles.push_back(tw12.imag());
        //                             twiddles.push_back(tw9.imag());
        //                             twiddles.push_back(tw10.imag());
        //                             twiddles.push_back(tw13.imag());
        //                             twiddles.push_back(tw14.imag());
        //
        //                             for (uint k = 0; k < 8; ++k)
        //                             {
        //                                 auto tw =
        //                                     wnk(l_size * 16,    //
        //                                         reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
        //                                 twiddles.push_back(tw.real());
        //                             }
        //                             for (uint k = 0; k < 8; ++k)
        //                             {
        //                                 auto tw =
        //                                     wnk(l_size * 16,    //
        //                                         reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
        //                                 twiddles.push_back(tw.imag());
        //                             }
        //
        //                             for (uint k = 8; k < 16; ++k)
        //                             {
        //                                 auto tw =
        //                                     wnk(l_size * 16,    //
        //                                         reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
        //                                 twiddles.push_back(tw.real());
        //                             }
        //                             for (uint k = 8; k < 16; ++k)
        //                             {
        //                                 auto tw =
        //                                     wnk(l_size * 16,    //
        //                                         reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
        //                                 twiddles.push_back(tw.imag());
        //                             }
        //                         }
        //                     }
        //                 }
        //         return twiddles;
    }

    static void insert_twiddles_unsorted(std::size_t             fft_size,
                                         std::size_t             l_size,
                                         std::size_t             sub_size,
                                         std::size_t             i_group,
                                         std::vector<real_type>& twiddles)
    {
        if ((fft_size / l_size) < sub_size)
        {
            std::size_t start_size       = twiddles.size();
            std::size_t single_load_size = fft_size / (reg_size * 2);
            std::size_t group_size       = 1;

            while (l_size < single_load_size / 2)
            {
                std::size_t start = group_size * i_group;
                for (uint i = 0; i < group_size; ++i)
                {
                    auto tw0 = wnk(l_size,    //
                                   reverse_bit_order(start + i, log2i(l_size / 2)));
                    auto tw1 = wnk(l_size * 2,    //
                                   reverse_bit_order((start + i) * 2, log2i(l_size)));
                    auto tw2 = wnk(l_size * 2,    //
                                   reverse_bit_order((start + i) * 2 + 1, log2i(l_size)));

                    twiddles.push_back(tw0.real());
                    twiddles.push_back(tw0.imag());
                    twiddles.push_back(tw1.real());
                    twiddles.push_back(tw1.imag());
                    twiddles.push_back(tw2.real());
                    twiddles.push_back(tw2.imag());
                }
                l_size *= 4;
                group_size *= 4;
            }

            if (l_size == single_load_size / 2)
            {
                std::size_t start = group_size * i_group;

                for (uint i = 0; i < group_size; ++i)
                {
                    auto tw0 = wnk(l_size,    //
                                   reverse_bit_order(start + i, log2i(l_size / 2)));

                    twiddles.push_back(tw0.real());
                    twiddles.push_back(tw0.imag());
                }
                l_size *= 2;
                group_size *= 2;
            }

            if (l_size == single_load_size)
            {
                for (uint i = 0; i < group_size; ++i)
                {
                    std::size_t start = group_size * i_group + i;

                    auto tw0 = wnk(l_size,    //
                                   reverse_bit_order(start, log2i(l_size / 2)));

                    twiddles.push_back(tw0.real());
                    twiddles.push_back(tw0.imag());

                    auto tw1 = wnk(l_size * 2,    //
                                   reverse_bit_order(start * 2, log2i(l_size)));
                    auto tw2 = wnk(l_size * 2,    //
                                   reverse_bit_order(start * 2 + 1, log2i(l_size)));

                    twiddles.push_back(tw1.real());
                    twiddles.push_back(tw1.imag());
                    twiddles.push_back(tw2.real());
                    twiddles.push_back(tw2.imag());

                    auto tw3_1 = wnk(l_size * 4,    //
                                     reverse_bit_order(start * 4, log2i(l_size * 2)));
                    auto tw3_2 = wnk(l_size * 4,    //
                                     reverse_bit_order(start * 4 + 1, log2i(l_size * 2)));
                    auto tw4_1 = wnk(l_size * 4,    //
                                     reverse_bit_order(start * 4 + 2, log2i(l_size * 2)));
                    auto tw4_2 = wnk(l_size * 4,    //
                                     reverse_bit_order(start * 4 + 3, log2i(l_size * 2)));

                    twiddles.push_back(tw3_1.real());
                    twiddles.push_back(tw3_1.imag());
                    twiddles.push_back(tw3_2.real());
                    twiddles.push_back(tw3_2.imag());
                    twiddles.push_back(tw4_1.real());
                    twiddles.push_back(tw4_1.imag());
                    twiddles.push_back(tw4_2.real());
                    twiddles.push_back(tw4_2.imag());


                    auto tw7  = wnk(l_size * 8,    //
                                   reverse_bit_order(start * 8, log2i(l_size * 4)));
                    auto tw8  = wnk(l_size * 8,    //
                                   reverse_bit_order(start * 8 + 1, log2i(l_size * 4)));
                    auto tw9  = wnk(l_size * 8,    //
                                   reverse_bit_order(start * 8 + 2, log2i(l_size * 4)));
                    auto tw10 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 3, log2i(l_size * 4)));
                    auto tw11 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 4, log2i(l_size * 4)));
                    auto tw12 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 5, log2i(l_size * 4)));
                    auto tw13 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 6, log2i(l_size * 4)));
                    auto tw14 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 7, log2i(l_size * 4)));

                    twiddles.push_back(tw7.real());
                    twiddles.push_back(tw8.real());
                    twiddles.push_back(tw11.real());
                    twiddles.push_back(tw12.real());
                    twiddles.push_back(tw9.real());
                    twiddles.push_back(tw10.real());
                    twiddles.push_back(tw13.real());
                    twiddles.push_back(tw14.real());

                    twiddles.push_back(tw7.imag());
                    twiddles.push_back(tw8.imag());
                    twiddles.push_back(tw11.imag());
                    twiddles.push_back(tw12.imag());
                    twiddles.push_back(tw9.imag());
                    twiddles.push_back(tw10.imag());
                    twiddles.push_back(tw13.imag());
                    twiddles.push_back(tw14.imag());

                    for (uint k = 0; k < 8; ++k)
                    {
                        auto tw =
                            wnk(l_size * 16,    //
                                reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                        twiddles.push_back(tw.real());
                    }
                    for (uint k = 0; k < 8; ++k)
                    {
                        auto tw =
                            wnk(l_size * 16,    //
                                reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                        twiddles.push_back(tw.imag());
                    }

                    for (uint k = 8; k < 16; ++k)
                    {
                        auto tw =
                            wnk(l_size * 16,    //
                                reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                        twiddles.push_back(tw.real());
                    }
                    for (uint k = 8; k < 16; ++k)
                    {
                        auto tw =
                            wnk(l_size * 16,    //
                                reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                        twiddles.push_back(tw.imag());
                    }
                }
            }


            // std::cout << "inserted twiddles: " << twiddles.size() - start_size << "\n";

        } else
        {
            auto tw0 = wnk(l_size, reverse_bit_order(i_group, log2i(l_size / 2)));
            // std::cout << "inserted: " << tw0 << "\n";
            twiddles.push_back(tw0.real());
            twiddles.push_back(tw0.imag());

            insert_twiddles_unsorted(fft_size,
                                     l_size * 2,
                                     sub_size,
                                     i_group * 2,
                                     twiddles);

            insert_twiddles_unsorted(fft_size,
                                     l_size * 2,
                                     sub_size,
                                     i_group * 2 + 1,
                                     twiddles);
        }
    }

    static auto get_twiddles_unsorted_linear(std::size_t    fft_size,
                                             std::size_t    sub_size,
                                             allocator_type allocator)
        -> std::vector<T, allocator_type>
    {
        auto twiddles = std::vector<T, allocator_type>(allocator);


        std::size_t l_size = 2;
        //

        auto sub_size_ = fft_size / std::min(fft_size, sub_size);
        while (l_size <= sub_size_)
        {
            for (uint i = 0; i < l_size / 2; ++i)
            {
                auto tw0 = wnk(l_size, reverse_bit_order(i, log2i(l_size / 2)));
                twiddles.push_back(tw0.real());
                twiddles.push_back(tw0.imag());
            }
            l_size *= 2;
        }

        std::size_t single_load_size = fft_size / (reg_size * 2);

        auto l_size_keep = l_size;
        for (uint i_group = 0; i_group < sub_size_; ++i_group)
        {
            std::size_t group_size = 1;
            l_size                 = l_size_keep;

            while (l_size < single_load_size / 2)
            {
                std::size_t start = group_size * i_group;

                for (uint i = 0; i < group_size; ++i)
                {
                    auto tw0 = wnk(l_size,    //
                                   reverse_bit_order(start + i, log2i(l_size / 2)));
                    auto tw1 = wnk(l_size * 2,    //
                                   reverse_bit_order((start + i) * 2, log2i(l_size)));
                    auto tw2 = wnk(l_size * 2,    //
                                   reverse_bit_order((start + i) * 2 + 1, log2i(l_size)));

                    twiddles.push_back(tw0.real());
                    twiddles.push_back(tw0.imag());
                    twiddles.push_back(tw1.real());
                    twiddles.push_back(tw1.imag());
                    twiddles.push_back(tw2.real());
                    twiddles.push_back(tw2.imag());
                }
                l_size *= 4;
                group_size *= 4;
            }

            if (l_size == single_load_size / 2)
            {
                std::size_t start = group_size * i_group;

                for (uint i = 0; i < group_size; ++i)
                {
                    auto tw0 = wnk(l_size,    //
                                   reverse_bit_order(start + i, log2i(l_size / 2)));

                    twiddles.push_back(tw0.real());
                    twiddles.push_back(tw0.imag());
                }
                l_size *= 2;
                group_size *= 2;
            }

            if (l_size == single_load_size)
            {
                for (uint i = 0; i < group_size; ++i)
                {
                    std::size_t start = group_size * i_group + i;

                    auto tw0 = wnk(l_size,    //
                                   reverse_bit_order(start, log2i(l_size / 2)));

                    twiddles.push_back(tw0.real());
                    twiddles.push_back(tw0.imag());

                    auto tw1 = wnk(l_size * 2,    //
                                   reverse_bit_order(start * 2, log2i(l_size)));
                    auto tw2 = wnk(l_size * 2,    //
                                   reverse_bit_order(start * 2 + 1, log2i(l_size)));

                    twiddles.push_back(tw1.real());
                    twiddles.push_back(tw1.imag());
                    twiddles.push_back(tw2.real());
                    twiddles.push_back(tw2.imag());

                    auto tw3_1 = wnk(l_size * 4,    //
                                     reverse_bit_order(start * 4, log2i(l_size * 2)));
                    auto tw3_2 = wnk(l_size * 4,    //
                                     reverse_bit_order(start * 4 + 1, log2i(l_size * 2)));
                    auto tw4_1 = wnk(l_size * 4,    //
                                     reverse_bit_order(start * 4 + 2, log2i(l_size * 2)));
                    auto tw4_2 = wnk(l_size * 4,    //
                                     reverse_bit_order(start * 4 + 3, log2i(l_size * 2)));

                    twiddles.push_back(tw3_1.real());
                    twiddles.push_back(tw3_1.imag());
                    twiddles.push_back(tw3_2.real());
                    twiddles.push_back(tw3_2.imag());
                    twiddles.push_back(tw4_1.real());
                    twiddles.push_back(tw4_1.imag());
                    twiddles.push_back(tw4_2.real());
                    twiddles.push_back(tw4_2.imag());


                    auto tw7  = wnk(l_size * 8,    //
                                   reverse_bit_order(start * 8, log2i(l_size * 4)));
                    auto tw8  = wnk(l_size * 8,    //
                                   reverse_bit_order(start * 8 + 1, log2i(l_size * 4)));
                    auto tw9  = wnk(l_size * 8,    //
                                   reverse_bit_order(start * 8 + 2, log2i(l_size * 4)));
                    auto tw10 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 3, log2i(l_size * 4)));
                    auto tw11 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 4, log2i(l_size * 4)));
                    auto tw12 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 5, log2i(l_size * 4)));
                    auto tw13 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 6, log2i(l_size * 4)));
                    auto tw14 = wnk(l_size * 8,    //
                                    reverse_bit_order(start * 8 + 7, log2i(l_size * 4)));

                    twiddles.push_back(tw7.real());
                    twiddles.push_back(tw8.real());
                    twiddles.push_back(tw11.real());
                    twiddles.push_back(tw12.real());
                    twiddles.push_back(tw9.real());
                    twiddles.push_back(tw10.real());
                    twiddles.push_back(tw13.real());
                    twiddles.push_back(tw14.real());

                    twiddles.push_back(tw7.imag());
                    twiddles.push_back(tw8.imag());
                    twiddles.push_back(tw11.imag());
                    twiddles.push_back(tw12.imag());
                    twiddles.push_back(tw9.imag());
                    twiddles.push_back(tw10.imag());
                    twiddles.push_back(tw13.imag());
                    twiddles.push_back(tw14.imag());

                    for (uint k = 0; k < 8; ++k)
                    {
                        auto tw =
                            wnk(l_size * 16,    //
                                reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                        twiddles.push_back(tw.real());
                    }
                    for (uint k = 0; k < 8; ++k)
                    {
                        auto tw =
                            wnk(l_size * 16,    //
                                reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                        twiddles.push_back(tw.imag());
                    }

                    for (uint k = 8; k < 16; ++k)
                    {
                        auto tw =
                            wnk(l_size * 16,    //
                                reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                        twiddles.push_back(tw.real());
                    }
                    for (uint k = 8; k < 16; ++k)
                    {
                        auto tw =
                            wnk(l_size * 16,    //
                                reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                        twiddles.push_back(tw.imag());
                    }
                }
            }
        }

        return twiddles;
    }
};


}    // namespace pcx
#endif