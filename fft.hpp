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
    , m_twiddles_unsorted(get_twiddles_unsorted(size(), sub_size(), allocator)){};

    fft_unit(std::size_t sub_size = 1, allocator_type allocator = allocator_type())
        requires(Size != pcx::dynamic_size) && (SubSize == pcx::dynamic_size)
    : m_sub_size(check_sub_size(sub_size))
    , m_sort(get_sort(size(), static_cast<sort_allocator_type>(allocator)))
    , m_twiddles(get_twiddles(size(), sub_size, allocator))
    , m_twiddles_unsorted(get_twiddles_unsorted(size(), sub_size, allocator)){};

    fft_unit(std::size_t fft_size, allocator_type allocator = allocator_type())
        requires(Size == pcx::dynamic_size) && (SubSize != pcx::dynamic_size)
    : m_size(check_size(fft_size))
    , m_sort(get_sort(size(), static_cast<sort_allocator_type>(allocator)))
    , m_twiddles(get_twiddles(size(), sub_size(), allocator))
    , m_twiddles_unsorted(get_twiddles_unsorted(size(), sub_size(), allocator)){};

    fft_unit(std::size_t    fft_size,
             std::size_t    sub_size  = 1,
             allocator_type allocator = allocator_type())
        requires(Size == pcx::dynamic_size) && (SubSize == pcx::dynamic_size)
    : m_size(check_size(fft_size))
    , m_sub_size(check_sub_size(sub_size))
    , m_sort(get_sort(size(), static_cast<sort_allocator_type>(allocator)))
    , m_twiddles(get_twiddles(size(), sub_size, allocator))
    , m_twiddles_unsorted(get_twiddles_unsorted(size(), sub_size, allocator)){};

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
        fft_internal<true, PackSize>(vector.data());
    };
    template<typename VAllocator>
    void operator()(std::vector<std::complex<T>, VAllocator>& vector)
    {
        assert(size() == vector.size());
        fft_internal<false, reg_size>(reinterpret_cast<T*>(vector.data()));
    };

    template<typename VAllocator>
    void unsorted(pcx::vector<T, VAllocator, PackSize>& vector)
    {
        assert(size() == vector.size());
        subtransform_unsorted(vector.data(), size(), m_twiddles_unsorted.data());
        // subtransform_cached_unsorted(vector.data(), size(), m_twiddles_unsorted.data());
    };

    template<typename VAllocator>
    void unsorted(std::vector<std::complex<T>, VAllocator>& vector)
    {
        assert(size() == vector.size());
        subtransform_unsorted<false, false, reg_size>(reinterpret_cast<T*>(vector.data()),
                                                      size(),
                                                      m_twiddles_unsorted.data());
    };


    template<typename VAllocator>
    void operator()(pcx::vector<T, VAllocator, PackSize>&       dest,
                    const pcx::vector<T, VAllocator, PackSize>& source)
    {
        assert(size() == dest.size() && size() == source.size());
        if (&dest == &source)
        {
            fft_internal<true, PackSize>(dest.data());
        } else
        {
            fft_internal<true, PackSize>(dest.data(), source.data());
        }
    };

private:
    [[no_unique_address]] size_t                           m_size;
    [[no_unique_address]] subsize_t                        m_sub_size;
    const std::vector<std::size_t, sort_allocator_type>    m_sort;
    const pcx::vector<real_type, allocator_type, reg_size> m_twiddles;
    const std::vector<real_type, allocator_type>           m_twiddles_unsorted;

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
    template<bool PackedSrc, std::size_t TMPPackSize>
    void fft_internal(float* source)
    {
        dept3_and_sort<PackedSrc, TMPPackSize>(source);
        if (size() <= sub_size() || log2i(size() / sub_size()) % 2 == 0)
        {
            subtransform<PackedSrc>(source, size());
        } else
        {
            subtransform<true>(source, size() / 2);
            auto twiddle_ptr = subtransform<true>(
                source + reg_offset<TMPPackSize>(size() / 2 / reg_size),
                size() / 2);
            std::size_t n_groups = size() / reg_size / 2;

            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
                twiddle_ptr += reg_size * 2;

                auto* ptr0 = source + reg_offset<TMPPackSize>(i_group);
                auto* ptr1 =
                    source + reg_offset<TMPPackSize>(i_group + size() / 2 / reg_size);

                auto p1 = avx::cxload<PackSize>(ptr1);
                auto p0 = avx::cxload<PackSize>(ptr0);

                auto p1tw = avx::mul(p1, tw0);

                auto a0 = avx::add(p0, p1tw);
                auto a1 = avx::sub(p0, p1tw);

                if constexpr (!PackedSrc)
                {
                    std::tie(a0, a1) = avx::convert<T>::packed_to_interleaved(a0, a1);
                }

                cxstore<PackSize>(ptr0, a0);
                cxstore<PackSize>(ptr1, a1);
            }
        }
    }

    template<bool PackedSrc, std::size_t TMPPackSize>
    void fft_internal(float* dest, const float* source)
    {
        dept3_and_sort<PackedSrc, TMPPackSize>(dest, source);
        if (size() <= sub_size() || log2i(size() / sub_size()) % 2 == 0)
        {
            subtransform<PackedSrc>(dest, size());
        } else
        {
            subtransform<true>(dest, size() / 2);
            auto twiddle_ptr =
                subtransform<true>(dest + reg_offset<TMPPackSize>(size() / 2 / reg_size),
                                   size() / 2);
            std::size_t n_groups = size() / reg_size / 2;

            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
                twiddle_ptr += reg_size * 2;

                auto* ptr0 = dest + reg_offset<TMPPackSize>(i_group);
                auto* ptr1 =
                    dest + reg_offset<TMPPackSize>(i_group + size() / 2 / reg_size);

                auto p1 = avx::cxload<PackSize>(ptr1);
                auto p0 = avx::cxload<PackSize>(ptr0);

                auto p1tw = avx::mul(p1, tw0);

                auto a0 = avx::add(p0, p1tw);
                auto a1 = avx::sub(p0, p1tw);

                if constexpr (!PackedSrc)
                {
                    std::tie(a0, a1) = avx::convert<T>::packed_to_interleaved(a0, a1);
                }

                cxstore<PackSize>(ptr0, a0);
                cxstore<PackSize>(ptr1, a1);
            }
        }
    };

    template<bool PackedSrc, std::size_t TMPPackSize = PackSize>
    inline void dept3_and_sort(float* source)
    {
        const auto sq2   = wnk(8, 1);
        auto       twsq2 = avx::broadcast(sq2.real());

        const auto sh0 = 0;
        const auto sh1 = reg_offset<TMPPackSize>(1 * size() / 64);
        const auto sh2 = reg_offset<TMPPackSize>(2 * size() / 64);
        const auto sh3 = reg_offset<TMPPackSize>(3 * size() / 64);
        const auto sh4 = reg_offset<TMPPackSize>(4 * size() / 64);
        const auto sh5 = reg_offset<TMPPackSize>(5 * size() / 64);
        const auto sh6 = reg_offset<TMPPackSize>(6 * size() / 64);
        const auto sh7 = reg_offset<TMPPackSize>(7 * size() / 64);

        uint i = 0;

        for (; i < n_reversals(size() / 64); i += 2)
        {
            using reg_t = avx::cx_reg<float>;

            auto offset_first  = m_sort[i];
            auto offset_second = m_sort[i + 1];

            auto p1 = avx::cxload<TMPPackSize>(source + sh1 + offset_first);
            auto p5 = avx::cxload<TMPPackSize>(source + sh5 + offset_first);
            auto p3 = avx::cxload<TMPPackSize>(source + sh3 + offset_first);
            auto p7 = avx::cxload<TMPPackSize>(source + sh7 + offset_first);

            if constexpr (!PackedSrc)
            {
                std::tie(p1, p5, p3, p7) =
                    avx::convert<T>::interleaved_to_packed(p1, p5, p3, p7);
            }

            auto [a1, a5] = avx::btfly(p1, p5);
            auto [a3, a7] = avx::btfly(p3, p7);

            reg_t b5 = {avx::add(a5.real, a7.imag), avx::sub(a5.imag, a7.real)};
            reg_t b7 = {avx::sub(a5.real, a7.imag), avx::add(a5.imag, a7.real)};

            reg_t b5_tw = {avx::add(b5.real, b5.imag), avx::sub(b5.imag, b5.real)};
            reg_t b7_tw = {avx::sub(b7.real, b7.imag), avx::add(b7.real, b7.imag)};

            auto [b1, b3] = avx::btfly(a1, a3);

            b5_tw = avx::mul(b5_tw, twsq2);
            b7_tw = avx::mul(b7_tw, twsq2);

            auto p0 = avx::cxload<TMPPackSize>(source + sh0 + offset_first);
            auto p4 = avx::cxload<TMPPackSize>(source + sh4 + offset_first);
            auto p2 = avx::cxload<TMPPackSize>(source + sh2 + offset_first);
            auto p6 = avx::cxload<TMPPackSize>(source + sh6 + offset_first);

            if constexpr (!PackedSrc)
            {
                std::tie(p0, p4, p2, p6) =
                    avx::convert<T>::interleaved_to_packed(p0, p4, p2, p6);
            }

            auto [a0, a4] = avx::btfly(p0, p4);
            auto [a2, a6] = avx::btfly(p2, p6);

            auto [b0, b2] = avx::btfly(a0, a2);
            reg_t b4      = {avx::add(a4.real, a6.imag), avx::sub(a4.imag, a6.real)};
            reg_t b6      = {avx::sub(a4.real, a6.imag), avx::add(a4.imag, a6.real)};

            auto [c0, c1] = avx::btfly(b0, b1);
            reg_t c2      = {avx::add(b2.real, b3.imag), avx::sub(b2.imag, b3.real)};
            reg_t c3      = {avx::sub(b2.real, b3.imag), avx::add(b2.imag, b3.real)};

            auto [c4, c5] = avx::btfly(b4, b5_tw);
            auto [c7, c6] = avx::btfly(b6, b7_tw);

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);

            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            auto q0 = avx::cxload<TMPPackSize>(source + sh0 + offset_second);
            avx::cxstore<TMPPackSize>(source + sh0 + offset_second, shc0);
            auto q1 = avx::cxload<TMPPackSize>(source + sh1 + offset_second);
            avx::cxstore<TMPPackSize>(source + sh1 + offset_second, shc1);
            auto q4 = avx::cxload<TMPPackSize>(source + sh4 + offset_second);
            avx::cxstore<TMPPackSize>(source + sh4 + offset_second, shc2);
            auto q5 = avx::cxload<TMPPackSize>(source + sh5 + offset_second);
            avx::cxstore<TMPPackSize>(source + sh5 + offset_second, shc3);

            if constexpr (!PackedSrc)
            {
                std::tie(q0, q1, q4, q5) =
                    avx::convert<T>::interleaved_to_packed(q0, q1, q4, q5);
            }

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);

            auto q2 = avx::cxload<TMPPackSize>(source + sh2 + offset_second);
            avx::cxstore<TMPPackSize>(source + sh2 + offset_second, shc4);
            auto q3 = avx::cxload<TMPPackSize>(source + sh3 + offset_second);
            avx::cxstore<TMPPackSize>(source + sh3 + offset_second, shc5);

            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            auto q6 = avx::cxload<TMPPackSize>(source + sh6 + offset_second);
            avx::cxstore<TMPPackSize>(source + sh6 + offset_second, shc6);
            auto q7 = avx::cxload<TMPPackSize>(source + sh7 + offset_second);
            avx::cxstore<TMPPackSize>(source + sh7 + offset_second, shc7);

            if constexpr (!PackedSrc)
            {
                std::tie(q2, q3, q6, q7) =
                    avx::convert<T>::interleaved_to_packed(q2, q3, q6, q7);
            }

            auto [x1, x5] = avx::btfly(q1, q5);
            auto [x3, x7] = avx::btfly(q3, q7);

            reg_t y5 = {avx::add(x5.real, x7.imag), avx::sub(x5.imag, x7.real)};
            reg_t y7 = {avx::sub(x5.real, x7.imag), avx::add(x5.imag, x7.real)};

            auto y1 = avx::add(x1, x3);

            reg_t y5_tw = {avx::add(y5.real, y5.imag), avx::sub(y5.imag, y5.real)};
            reg_t y7_tw = {avx::sub(y7.real, y7.imag), avx::add(y7.real, y7.imag)};

            auto y3 = avx::sub(x1, x3);

            y5_tw = avx::mul(y5_tw, twsq2);
            y7_tw = avx::mul(y7_tw, twsq2);

            auto [x0, x4] = avx::btfly(q0, q4);
            auto [x2, x6] = avx::btfly(q2, q6);

            auto [y0, y2] = avx::btfly(x0, x2);
            reg_t y4      = {avx::add(x4.real, x6.imag), avx::sub(x4.imag, x6.real)};
            reg_t y6      = {avx::sub(x4.real, x6.imag), avx::add(x4.imag, x6.real)};

            auto [z0, z1] = avx::btfly(y0, y1);

            reg_t z2 = {avx::add(y2.real, y3.imag), avx::sub(y2.imag, y3.real)};
            reg_t z3 = {avx::sub(y2.real, y3.imag), avx::add(y2.imag, y3.real)};

            auto [z4, z5] = avx::btfly(y4, y5_tw);
            auto [z7, z6] = avx::btfly(y6, y7_tw);

            auto [shx0, shx4] = avx::unpack_ps(z0, z4);
            auto [shx1, shx5] = avx::unpack_ps(z1, z5);
            auto [shx2, shx6] = avx::unpack_ps(z2, z6);
            auto [shx3, shx7] = avx::unpack_ps(z3, z7);

            auto [shy0, shy2] = avx::unpack_pd(shx0, shx2);
            auto [shy1, shy3] = avx::unpack_pd(shx1, shx3);

            auto [shz0, shz1] = avx::unpack_128(shy0, shy1);
            auto [shz2, shz3] = avx::unpack_128(shy2, shy3);

            avx::cxstore<TMPPackSize>(source + sh0 + offset_first, shz0);
            avx::cxstore<TMPPackSize>(source + sh1 + offset_first, shz1);
            avx::cxstore<TMPPackSize>(source + sh4 + offset_first, shz2);
            avx::cxstore<TMPPackSize>(source + sh5 + offset_first, shz3);

            auto [shy4, shy6] = avx::unpack_pd(shx4, shx6);
            auto [shy5, shy7] = avx::unpack_pd(shx5, shx7);

            auto [shz4, shz5] = avx::unpack_128(shy4, shy5);
            auto [shz6, shz7] = avx::unpack_128(shy6, shy7);

            avx::cxstore<TMPPackSize>(source + sh2 + offset_first, shz4);
            avx::cxstore<TMPPackSize>(source + sh3 + offset_first, shz5);
            avx::cxstore<TMPPackSize>(source + sh6 + offset_first, shz6);
            avx::cxstore<TMPPackSize>(source + sh7 + offset_first, shz7);
        };

        for (; i < size() / 64; ++i)
        {
            using reg_t = avx::cx_reg<float>;
            auto offset = m_sort[i];

            auto p1 = avx::cxload<TMPPackSize>(source + sh1 + offset);
            auto p3 = avx::cxload<TMPPackSize>(source + sh3 + offset);
            auto p5 = avx::cxload<TMPPackSize>(source + sh5 + offset);
            auto p7 = avx::cxload<TMPPackSize>(source + sh7 + offset);

            if constexpr (!PackedSrc)
            {
                std::tie(p1, p3, p5, p7) =
                    avx::convert<T>::interleaved_to_packed(p1, p3, p5, p7);
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

            auto p0 = avx::cxload<TMPPackSize>(source + sh0 + offset);
            auto p2 = avx::cxload<TMPPackSize>(source + sh2 + offset);
            auto p4 = avx::cxload<TMPPackSize>(source + sh4 + offset);
            auto p6 = avx::cxload<TMPPackSize>(source + sh6 + offset);

            if constexpr (!PackedSrc)
            {
                std::tie(p0, p4, p2, p6) =
                    avx::convert<T>::interleaved_to_packed(p0, p4, p2, p6);
            }

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

            avx::cxstore<TMPPackSize>(source + sh0 + offset, shc0);
            avx::cxstore<TMPPackSize>(source + sh1 + offset, shc1);
            avx::cxstore<TMPPackSize>(source + sh4 + offset, shc2);
            avx::cxstore<TMPPackSize>(source + sh5 + offset, shc3);

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            avx::cxstore<TMPPackSize>(source + sh2 + offset, shc4);
            avx::cxstore<TMPPackSize>(source + sh3 + offset, shc5);
            avx::cxstore<TMPPackSize>(source + sh6 + offset, shc6);
            avx::cxstore<TMPPackSize>(source + sh7 + offset, shc7);
        }
    }

    template<bool PackedSrc, std::size_t TMPPackSize = PackSize>
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

            auto offset_src  = m_sort[i];
            auto offset_dest = m_sort[i + 1];

            for (uint k = 0; k < 2; ++k)
            {
                auto p1 = avx::cxload<TMPPackSize>(source + sh1 + offset_src);
                auto p5 = avx::cxload<TMPPackSize>(source + sh5 + offset_src);
                auto p3 = avx::cxload<TMPPackSize>(source + sh3 + offset_src);
                auto p7 = avx::cxload<TMPPackSize>(source + sh7 + offset_src);

                if constexpr (!PackedSrc)
                {
                    std::tie(p1, p5, p3, p7) =
                        avx::convert<T>::interleaved_to_packed(p1, p5, p3, p7);
                }

                auto [a1, a5] = avx::btfly(p1, p5);
                auto [a3, a7] = avx::btfly(p3, p7);

                reg_t b5 = {avx::add(a5.real, a7.imag), avx::sub(a5.imag, a7.real)};
                reg_t b7 = {avx::sub(a5.real, a7.imag), avx::add(a5.imag, a7.real)};

                reg_t b5_tw = {avx::add(b5.real, b5.imag), avx::sub(b5.imag, b5.real)};
                reg_t b7_tw = {avx::sub(b7.real, b7.imag), avx::add(b7.real, b7.imag)};

                auto [b1, b3] = avx::btfly(a1, a3);

                b5_tw = avx::mul(b5_tw, twsq2);
                b7_tw = avx::mul(b7_tw, twsq2);

                auto p0 = avx::cxload<TMPPackSize>(source + sh0 + offset_src);
                auto p4 = avx::cxload<TMPPackSize>(source + sh4 + offset_src);
                auto p2 = avx::cxload<TMPPackSize>(source + sh2 + offset_src);
                auto p6 = avx::cxload<TMPPackSize>(source + sh6 + offset_src);

                if constexpr (!PackedSrc)
                {
                    std::tie(p0, p4, p2, p6) =
                        avx::convert<T>::interleaved_to_packed(p0, p4, p2, p6);
                }

                auto [a0, a4] = avx::btfly(p0, p4);
                auto [a2, a6] = avx::btfly(p2, p6);

                auto [b0, b2] = avx::btfly(a0, a2);
                reg_t b4      = {avx::add(a4.real, a6.imag), avx::sub(a4.imag, a6.real)};
                reg_t b6      = {avx::sub(a4.real, a6.imag), avx::add(a4.imag, a6.real)};

                auto [c0, c1] = avx::btfly(b0, b1);
                reg_t c2      = {avx::add(b2.real, b3.imag), avx::sub(b2.imag, b3.real)};
                reg_t c3      = {avx::sub(b2.real, b3.imag), avx::add(b2.imag, b3.real)};

                auto [c4, c5] = avx::btfly(b4, b5_tw);
                auto [c7, c6] = avx::btfly(b6, b7_tw);

                auto [sha0, sha4] = avx::unpack_ps(c0, c4);
                auto [sha2, sha6] = avx::unpack_ps(c2, c6);
                auto [sha1, sha5] = avx::unpack_ps(c1, c5);
                auto [sha3, sha7] = avx::unpack_ps(c3, c7);

                auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
                auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);
                auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
                auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

                avx::cxstore<TMPPackSize>(dest + sh0 + offset_dest, shc0);
                avx::cxstore<TMPPackSize>(dest + sh4 + offset_dest, shc2);
                avx::cxstore<TMPPackSize>(dest + sh5 + offset_dest, shc3);
                avx::cxstore<TMPPackSize>(dest + sh1 + offset_dest, shc1);

                auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
                auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);
                auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
                auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

                avx::cxstore<TMPPackSize>(dest + sh2 + offset_dest, shc4);
                avx::cxstore<TMPPackSize>(dest + sh3 + offset_dest, shc5);
                avx::cxstore<TMPPackSize>(dest + sh6 + offset_dest, shc6);
                avx::cxstore<TMPPackSize>(dest + sh7 + offset_dest, shc7);

                std::swap(offset_src, offset_dest);
            }
        };

        for (; i < size() / 64; ++i)
        {
            using reg_t = avx::cx_reg<float>;
            auto offset = m_sort[i];

            auto p1 = avx::cxload<TMPPackSize>(source + sh1 + offset);
            auto p5 = avx::cxload<TMPPackSize>(source + sh5 + offset);
            auto p3 = avx::cxload<TMPPackSize>(source + sh3 + offset);
            auto p7 = avx::cxload<TMPPackSize>(source + sh7 + offset);

            if constexpr (!PackedSrc)
            {
                std::tie(p1, p5, p3, p7) =
                    avx::convert<T>::interleaved_to_packed(p1, p5, p3, p7);
            }

            auto [a1, a5] = avx::btfly(p1, p5);
            auto [a3, a7] = avx::btfly(p3, p7);

            reg_t b5 = {avx::add(a5.real, a7.imag), avx::sub(a5.imag, a7.real)};
            reg_t b7 = {avx::sub(a5.real, a7.imag), avx::add(a5.imag, a7.real)};

            reg_t b5_tw = {avx::add(b5.real, b5.imag), avx::sub(b5.imag, b5.real)};
            reg_t b7_tw = {avx::sub(b7.real, b7.imag), avx::add(b7.real, b7.imag)};

            auto [b1, b3] = avx::btfly(a1, a3);

            b5_tw = avx::mul(b5_tw, twsq2);
            b7_tw = avx::mul(b7_tw, twsq2);

            auto p0 = avx::cxload<TMPPackSize>(source + sh0 + offset);
            auto p2 = avx::cxload<TMPPackSize>(source + sh2 + offset);
            auto p4 = avx::cxload<TMPPackSize>(source + sh4 + offset);
            auto p6 = avx::cxload<TMPPackSize>(source + sh6 + offset);

            if constexpr (!PackedSrc)
            {
                std::tie(p0, p4, p2, p6) =
                    avx::convert<T>::interleaved_to_packed(p0, p4, p2, p6);
            }

            auto [a0, a4] = avx::btfly(p0, p4);
            auto [a2, a6] = avx::btfly(p2, p6);

            auto [b0, b2] = avx::btfly(a0, a2);
            reg_t b4      = {avx::add(a4.real, a6.imag), avx::sub(a4.imag, a6.real)};
            reg_t b6      = {avx::sub(a4.real, a6.imag), avx::add(a4.imag, a6.real)};

            auto [c0, c1] = avx::btfly(b0, b1);
            reg_t c2      = {avx::add(b2.real, b3.imag), avx::sub(b2.imag, b3.real)};
            reg_t c3      = {avx::sub(b2.real, b3.imag), avx::add(b2.imag, b3.real)};

            auto [c4, c5] = avx::btfly(b4, b5_tw);
            auto [c7, c6] = avx::btfly(b6, b7_tw);

            auto [sha0, sha4] = avx::unpack_ps(c0, c4);
            auto [sha2, sha6] = avx::unpack_ps(c2, c6);
            auto [sha1, sha5] = avx::unpack_ps(c1, c5);
            auto [sha3, sha7] = avx::unpack_ps(c3, c7);

            auto [shb0, shb2] = avx::unpack_pd(sha0, sha2);
            auto [shb1, shb3] = avx::unpack_pd(sha1, sha3);

            auto [shc0, shc1] = avx::unpack_128(shb0, shb1);
            auto [shc2, shc3] = avx::unpack_128(shb2, shb3);

            avx::cxstore<TMPPackSize>(dest + sh0 + offset, shc0);
            avx::cxstore<TMPPackSize>(dest + sh1 + offset, shc1);
            avx::cxstore<TMPPackSize>(dest + sh4 + offset, shc2);
            avx::cxstore<TMPPackSize>(dest + sh5 + offset, shc3);

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            avx::cxstore<TMPPackSize>(dest + sh2 + offset, shc4);
            avx::cxstore<TMPPackSize>(dest + sh3 + offset, shc5);
            avx::cxstore<TMPPackSize>(dest + sh6 + offset, shc6);
            avx::cxstore<TMPPackSize>(dest + sh7 + offset, shc7);
        }
    }

    template<bool PackedDest, std::size_t TMPPackSize = PackSize>
    inline auto subtransform_cached(float* data, std::size_t max_size) -> const float*
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

                    auto p1 = avx::cxload<TMPPackSize>(ptr1);
                    auto p3 = avx::cxload<TMPPackSize>(ptr3);
                    auto p0 = avx::cxload<TMPPackSize>(ptr0);
                    auto p2 = avx::cxload<TMPPackSize>(ptr2);

                    auto [p1tw, p3tw] = avx::mul({p1, tw0}, {p3, tw0});

                    auto [a2, a3] = avx::btfly(p2, p3tw);
                    auto [a0, a1] = avx::btfly(p0, p1tw);

                    auto [a2tw, a3tw] = avx::mul({a2, tw1}, {a3, tw2});

                    auto [b0, b2] = avx::btfly(a0, a2tw);
                    auto [b1, b3] = avx::btfly(a1, a3tw);

                    if constexpr (!PackedDest)
                    {
                        if (l_size * 2 == max_size)
                        {
                            std::tie(b0, b1, b2, b3) =
                                avx::convert<T>::packed_to_interleaved(b0, b1, b2, b3);
                        }
                    }

                    cxstore<TMPPackSize>(ptr0, b0);
                    cxstore<TMPPackSize>(ptr1, b1);
                    cxstore<TMPPackSize>(ptr2, b2);
                    cxstore<TMPPackSize>(ptr3, b3);

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

                auto p1 = avx::cxload<TMPPackSize>(ptr1);
                auto p0 = avx::cxload<TMPPackSize>(ptr0);

                auto p1tw = avx::mul(p1, tw0);

                auto [a0, a1] = avx::btfly(p0, p1tw);

                if constexpr (!PackedDest)
                {
                    std::tie(a0, a1) = avx::convert<T>::packed_to_interleaved(a0, a1);
                }

                cxstore<TMPPackSize>(ptr0, a0);
                cxstore<TMPPackSize>(ptr1, a1);
            }
            l_size *= 2;
            n_groups *= 2;
            group_size /= 2;
        }

        return twiddle_ptr;
    };

    template<bool PackedDest, std::size_t TMPPackSize = PackSize>
    inline auto subtransform(float* data, std::size_t size) -> const float*
    {
        if (size <= sub_size())
        {
            return subtransform_cached<PackedDest, TMPPackSize>(data, size);
        } else
        {
            subtransform<true>(data, size / 4);
            subtransform<true>(data + reg_offset<TMPPackSize>(size / 4 / reg_size),
                               size / 4);
            subtransform<true>(data + reg_offset<TMPPackSize>(size / 2 / reg_size),
                               size / 4);
            auto twiddle_ptr = subtransform<true>(
                data + reg_offset<TMPPackSize>(size * 3 / 4 / reg_size),
                size / 4);

            std::size_t n_groups = size / reg_size / 4;

            for (std::size_t i_group = 0; i_group < n_groups; ++i_group)
            {
                const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
                const auto tw1 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 2);
                const auto tw2 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 4);
                twiddle_ptr += reg_size * 6;

                auto* ptr0 = data + reg_offset<TMPPackSize>(i_group);
                auto* ptr1 =
                    data + reg_offset<TMPPackSize>(i_group + size / 4 / reg_size);
                auto* ptr2 =
                    data + reg_offset<TMPPackSize>(i_group + size / 2 / reg_size);
                auto* ptr3 =
                    data + reg_offset<TMPPackSize>(i_group + size * 3 / 4 / reg_size);

                auto p1 = avx::cxload<TMPPackSize>(ptr1);
                auto p3 = avx::cxload<TMPPackSize>(ptr3);
                auto p0 = avx::cxload<TMPPackSize>(ptr0);
                auto p2 = avx::cxload<TMPPackSize>(ptr2);

                auto [p1tw, p3tw] = avx::mul({p1, tw0}, {p3, tw0});

                auto [a2, a3] = avx::btfly(p2, p3tw);
                auto [a0, a1] = avx::btfly(p0, p1tw);

                auto [a2tw, a3tw] = avx::mul({a2, tw1}, {a3, tw2});

                auto [b0, b2] = avx::btfly(a0, a2tw);
                auto [b1, b3] = avx::btfly(a1, a3tw);

                if constexpr (!PackedDest)
                {
                    std::tie(b0, b1, b2, b3) =
                        avx::convert<T>::packed_to_interleaved(b0, b1, b2, b3);
                }

                cxstore<TMPPackSize>(ptr0, b0);
                cxstore<TMPPackSize>(ptr1, b1);
                cxstore<TMPPackSize>(ptr2, b2);
                cxstore<TMPPackSize>(ptr3, b3);
            }

            return twiddle_ptr;
        }
    };

    template<bool        PackedSrc   = true,
             bool        PackedDest  = true,
             std::size_t TMPPackSize = PackSize>
    inline auto subtransform_cached_unsorted(float*       data,
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

                    auto p1 = avx::cxload<TMPPackSize>(ptr1);
                    auto p3 = avx::cxload<TMPPackSize>(ptr3);
                    auto p0 = avx::cxload<TMPPackSize>(ptr0);
                    auto p2 = avx::cxload<TMPPackSize>(ptr2);

                    if constexpr (!PackedSrc)
                    {
                        std::tie(p1, p3, p0, p2) =
                            avx::convert<T>::interleaved_to_packed(p1, p3, p0, p2);
                    }

                    auto [p1tw, p3tw] = avx::mul({p1, tw0}, {p3, tw0});

                    auto [a2, a3] = avx::btfly(p2, p3tw);
                    auto [a0, a1] = avx::btfly(p0, p1tw);

                    auto [a2tw, a3tw] = avx::mul({a2, tw1}, {a3, tw2});

                    auto [b0, b2] = avx::btfly(a0, a2tw);
                    auto [b1, b3] = avx::btfly(a1, a3tw);

                    cxstore<TMPPackSize>(ptr0, b0);
                    cxstore<TMPPackSize>(ptr1, b1);
                    cxstore<TMPPackSize>(ptr2, b2);
                    cxstore<TMPPackSize>(ptr3, b3);
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

                    auto p1 = avx::cxload<TMPPackSize>(ptr1);
                    auto p0 = avx::cxload<TMPPackSize>(ptr0);

                    if constexpr (!PackedSrc)
                    {
                        if (size == reg_size * 8)
                        {
                            std::tie(p1, p0) =
                                avx::convert<T>::interleaved_to_packed(p1, p0);
                        }
                    }

                    auto p1tw = avx::mul(p1, tw0);

                    auto [a0, a1] = avx::btfly(p0, p1tw);

                    cxstore<TMPPackSize>(ptr0, a0);
                    cxstore<TMPPackSize>(ptr1, a1);
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

                std::size_t offset = i_group * reg_size * 4;

                auto* ptr0 = data + pidx(offset);
                auto* ptr1 = data + pidx(offset + reg_size);
                auto* ptr2 = data + pidx(offset + reg_size * 2);
                auto* ptr3 = data + pidx(offset + reg_size * 3);

                auto p2 = avx::cxload<TMPPackSize>(ptr2);
                auto p3 = avx::cxload<TMPPackSize>(ptr3);
                auto p0 = avx::cxload<TMPPackSize>(ptr0);
                auto p1 = avx::cxload<TMPPackSize>(ptr1);

                auto [p2tw, p3tw] = avx::mul({p2, tw0}, {p3, tw0});

                auto [a1, a3] = avx::btfly(p1, p3tw);
                auto [a0, a2] = avx::btfly(p0, p2tw);

                auto [a1tw, a3tw] = avx::mul({a1, tw1}, {a3, tw2});

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
                auto  tw56  = avx::cxload<reg_size>(twiddle_ptr);
                twiddle_ptr += reg_size * 2;
                auto [tw5, tw6] = avx::unpack_ps(tw56, tw56);

                auto [b0, b1] = avx::btfly(a0, a1tw);
                auto [b2, b3] = avx::btfly(a2, a3tw);

                auto [shb0, shb1] = avx::unpack_128(b0, b1);
                auto [shb2, shb3] = avx::unpack_128(b2, b3);

                auto [shb1tw, shb3tw] = avx::mul({shb1, tw3}, {shb3, tw4});

                auto [c0, c1] = avx::btfly(shb0, shb1tw);
                auto [c2, c3] = avx::btfly(shb2, shb3tw);

                auto [shc0, shc1] = avx::unpack_pd(c0, c1);
                auto [shc2, shc3] = avx::unpack_pd(c2, c3);

                auto [shc1tw, shc3tw] = avx::mul({shc1, tw5}, {shc3, tw6});

                auto tw7 = avx::cxload<reg_size>(twiddle_ptr);
                twiddle_ptr += reg_size * 2;
                auto tw8 = avx::cxload<reg_size>(twiddle_ptr);
                twiddle_ptr += reg_size * 2;

                auto [d0, d1] = avx::btfly(shc0, shc1tw);
                auto [d2, d3] = avx::btfly(shc2, shc3tw);

                auto [shd0s, shd1s] = avx::unpack_ps(d0, d1);
                auto [shd2s, shd3s] = avx::unpack_ps(d2, d3);
                auto [shd0, shd1]   = avx::unpack_pd(shd0s, shd1s);
                auto [shd2, shd3]   = avx::unpack_pd(shd2s, shd3s);

                auto [shd1tw, shd3tw] = avx::mul({shd1, tw7}, {shd3, tw8});

                auto [e0, e1] = avx::btfly(shd0, shd1tw);
                auto [e2, e3] = avx::btfly(shd2, shd3tw);

                auto [she0s, she1s] = avx::unpack_ps(e0, e1);
                auto [she2s, she3s] = avx::unpack_ps(e2, e3);
                auto [she0, she1]   = avx::unpack_128(she0s, she1s);
                auto [she2, she3]   = avx::unpack_128(she2s, she3s);

                if constexpr (!PackedDest)
                {
                    std::tie(she0, she1, she2, she3) =
                        avx::convert<T>::packed_to_interleaved(she0, she1, she2, she3);
                }

                cxstore<TMPPackSize>(ptr0, she0);
                cxstore<TMPPackSize>(ptr1, she1);
                cxstore<TMPPackSize>(ptr2, she2);
                cxstore<TMPPackSize>(ptr3, she3);
            }
        }

        return twiddle_ptr;
    }

    template<bool        PackedSrc   = true,
             bool        PackedDest  = true,
             std::size_t TMPPackSize = PackSize>
    inline auto subtransform_unsorted(float*       data,
                                      std::size_t  size,
                                      const float* twiddle_ptr) -> const float*
    {
        if (size <= sub_size())
        {
            return subtransform_cached_unsorted<PackedSrc, PackedDest, TMPPackSize>(
                data,
                size,
                twiddle_ptr);
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

                auto p1 = avx::cxload<TMPPackSize>(ptr1);
                auto p0 = avx::cxload<TMPPackSize>(ptr0);

                if constexpr (!PackedSrc)
                {
                    std::tie(p1, p0) = avx::convert<T>::interleaved_to_packed(p1, p0);
                }

                auto p1tw = avx::mul(p1, tw0);

                auto [a0, a1] = avx::btfly(p0, p1tw);

                cxstore<TMPPackSize>(ptr0, a0);
                cxstore<TMPPackSize>(ptr1, a1);
            }

            twiddle_ptr =
                subtransform_unsorted<true, PackedDest, TMPPackSize>(data,
                                                                     size / 2,
                                                                     twiddle_ptr);
            return subtransform_unsorted<true, PackedDest, TMPPackSize>(
                data + (size / 2) * 2,
                size / 2,
                twiddle_ptr);
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

    template<std::size_t TMPPackSize>
    static constexpr auto pidx(std::size_t idx) -> std::size_t
    {
        return idx + idx / TMPPackSize * TMPPackSize;
    }

    static constexpr auto pidx(std::size_t idx) -> std::size_t
    {
        return idx + idx / PackSize * PackSize;
    }

    template<std::size_t TMPPackSize>
    static constexpr auto reg_offset(std::size_t reg_idx) -> std::size_t
    {
        return reg_idx * reg_size + (reg_idx * reg_size / PackSize) * PackSize;
    }
    template<>
    static constexpr auto reg_offset<reg_size>(std::size_t reg_idx) -> std::size_t
    {
        return reg_idx * reg_size * 2;
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
};


}    // namespace pcx
#endif