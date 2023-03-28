#ifndef FFT_HPP
#define FFT_HPP

#include "vector.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>

namespace pcx {

template<typename T,
         std::size_t Size    = pcx::dynamic_size,
         std::size_t SubSize = pcx::default_pack_size<T>,
         typename Allocator  = pcx::aligned_allocator<T, std::align_val_t(32)>>
    requires(std::same_as<T, float> || std::same_as<T, double>) &&
            (pcx::power_of_two<Size> || (Size == pcx::dynamic_size)) &&
            (pcx::power_of_two<SubSize> && SubSize >= pcx::default_pack_size<T> ||
             (SubSize == pcx::dynamic_size))
class fft_unit {
public:
    using real_type      = T;
    using allocator_type = Allocator;

    // static constexpr auto pack_size = default_pack_size<T>;
    static constexpr std::size_t reg_size = 32 / sizeof(real_type);

private:
    using size_t    = std::conditional_t<Size == pcx::dynamic_size, std::size_t, decltype([]() {})>;
    using subsize_t = std::conditional_t<SubSize == pcx::dynamic_size, std::size_t, decltype([]() {})>;
    using sort_allocator_type =
        typename std::allocator_traits<allocator_type>::template rebind_alloc<std::size_t>;

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

    fft_unit(std::size_t fft_size, std::size_t sub_size = 1, allocator_type allocator = allocator_type())
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

    [[nodiscard]] constexpr auto size() const -> std::size_t {
        if constexpr (Size == pcx::dynamic_size) {
            return m_size;
        } else {
            return Size;
        }
    }

    template<typename VAllocator, std::size_t VPackSize>
    void operator()(pcx::vector<T, VAllocator, VPackSize>& vector) {
        assert(size() == vector.size());
        fft_internal<VPackSize>(vector.data());
    };

    template<typename VAllocator>
    void operator()(std::vector<std::complex<T>, VAllocator>& vector) {
        assert(size() == vector.size());
        fft_internal<1>(reinterpret_cast<T*>(vector.data()));
    };

    template<typename VAllocator, std::size_t VPackSize>
    void unsorted(pcx::vector<T, VAllocator, VPackSize>& vector) {
        assert(size() == vector.size());
        fftu_internal<VPackSize>(vector.data());
    };

    template<typename VAllocator>
    void unsorted(std::vector<std::complex<T>, VAllocator>& vector) {
        assert(size() == vector.size());
        fftu_internal<1>(reinterpret_cast<T*>(vector.data()));
    };

    template<typename VAllocator, std::size_t VPackSize>
    void ifft(pcx::vector<T, VAllocator, VPackSize>& vector) {
        assert(size() == vector.size());
        ifft_internal<VPackSize>(vector.data());
    };
    template<typename VAllocator>
    void ifft(std::vector<std::complex<T>, VAllocator>& vector) {
        assert(size() == vector.size());
        ifft_internal<1>(reinterpret_cast<T*>(vector.data()));
    };

    template<typename VAllocator>
    void operator()(std::vector<std::complex<T>, VAllocator>&       dest,
                    const std::vector<std::complex<T>, VAllocator>& source) {
        assert(size() == dest.size() && size() == source.size());
        if (&dest == &source) {
            fft_internal<1>(reinterpret_cast<T*>(dest.data()));
        } else {
            fft_internal<1, 1>(reinterpret_cast<T*>(dest.data()), reinterpret_cast<const T*>(source.data()));
        }
    };

    template<typename VAllocator, std::size_t VPackSize>
    void operator()(pcx::vector<T, VAllocator, VPackSize>&       dest,
                    const pcx::vector<T, VAllocator, VPackSize>& source) {
        assert(size() == dest.size() && size() == source.size());
        if (&dest == &source) {
            fft_internal<VPackSize>(dest.data());
        } else {
            fft_internal<VPackSize, VPackSize>(dest.data(), source.data());
        }
    };

private:
    [[no_unique_address]] size_t                           m_size;
    [[no_unique_address]] subsize_t                        m_sub_size;
    const std::vector<std::size_t, sort_allocator_type>    m_sort;
    const pcx::vector<real_type, allocator_type, reg_size> m_twiddles;
    const std::vector<real_type, allocator_type>           m_twiddles_unsorted;

    [[nodiscard]] constexpr auto sub_size() const -> std::size_t {
        if constexpr (SubSize == pcx::dynamic_size) {
            return m_sub_size;
        } else {
            return SubSize;
        }
    }

public:
    template<std::size_t PData>
    inline void fft_internal(float* data) {
        constexpr auto PTform = std::max(PData, reg_size);
        depth3_and_sort<PTform, PData>(data);
        subtransform_recursive<PData, PTform>(data, size());
    }

    template<std::size_t PData, std::size_t Scale = true>
    inline void ifft_internal(float* data) {
        constexpr auto PTform = std::max(PData, reg_size);
        depth3_and_sort<PTform, PData, true>(data);
        subtransform_recursive<PData, PTform, true, Scale>(data, size());
    }

    template<std::size_t PDest, std::size_t PSrc>
    inline void fft_internal(float* dest, const float* source) {
        constexpr auto PTform = std::max(PDest, reg_size);
        depth3_and_sort<PTform, PSrc>(dest, source);
        subtransform_recursive<PDest, PTform>(dest, size());
    };

    template<std::size_t PData>
    void fftu_internal(float* data) {
        auto* twiddle_ptr = m_twiddles_unsorted.data();
        if (log2i(size() / sub_size()) % 2 == 0) {
            unsorted_subtransform_recursive<PData, PData>(data, size(), twiddle_ptr);
        } else {
            constexpr auto PTform = std::max(PData, reg_size);

            using reg_t = avx::cx_reg<float>;
            reg_t tw0   = {
                avx::broadcast(twiddle_ptr++),
                avx::broadcast(twiddle_ptr++),
            };

            for (std::size_t i_group = 0; i_group < size() / 2 / reg_size; ++i_group) {
                auto* ptr0 = avx::ra_addr<PTform>(data, i_group * reg_size);
                auto* ptr1 = avx::ra_addr<PTform>(data, i_group * reg_size + size() / 2);

                auto p1 = avx::cxload<PTform>(ptr1);
                auto p0 = avx::cxload<PTform>(ptr0);

                if constexpr (PData < PTform) {
                    std::tie(p1, p0) = avx::convert<float>::repack<PData, PTform>(p1, p0);
                }

                auto p1tw = avx::mul(p1, tw0);

                auto [a0, a1] = avx::btfly(p0, p1tw);

                cxstore<PTform>(ptr0, a0);
                cxstore<PTform>(ptr1, a1);
            }
            twiddle_ptr = unsorted_subtransform_recursive<PData, PTform>(data, size() / 2, twiddle_ptr);
            unsorted_subtransform_recursive<PData, PTform>(
                avx::ra_addr<PTform>(data, size() / 2), size() / 2, twiddle_ptr);
        }
    }

    template<std::size_t PTform, std::size_t PSrc, bool Inverse = false>
    inline void depth3_and_sort(float* data) {
        const auto sq2   = wnk(8, 1);
        auto       twsq2 = avx::broadcast(sq2.real());

        auto* src0 = avx::ra_addr<PTform>(data, 0);
        auto* src1 = avx::ra_addr<PTform>(data, 1 * size() / 8);
        auto* src2 = avx::ra_addr<PTform>(data, 2 * size() / 8);
        auto* src3 = avx::ra_addr<PTform>(data, 3 * size() / 8);
        auto* src4 = avx::ra_addr<PTform>(data, 4 * size() / 8);
        auto* src5 = avx::ra_addr<PTform>(data, 5 * size() / 8);
        auto* src6 = avx::ra_addr<PTform>(data, 6 * size() / 8);
        auto* src7 = avx::ra_addr<PTform>(data, 7 * size() / 8);

        uint i = 0;
        for (; i < n_reversals(size() / 64); i += 2) {
            using reg_t = avx::cx_reg<float>;

            auto offset1 = m_sort[i] * reg_size;
            auto offset2 = m_sort[i + 1] * reg_size;

            auto p1 = avx::cxload<PTform>(avx::ra_addr<PTform>(src1, offset1));
            auto p5 = avx::cxload<PTform>(avx::ra_addr<PTform>(src5, offset1));
            auto p3 = avx::cxload<PTform>(avx::ra_addr<PTform>(src3, offset1));
            auto p7 = avx::cxload<PTform>(avx::ra_addr<PTform>(src7, offset1));

            std::tie(p1, p5, p3, p7) = avx::convert<float>::split<PSrc>(p1, p5, p3, p7);
            std::tie(p1, p5, p3, p7) = avx::convert<float>::inverse<Inverse>(p1, p5, p3, p7);

            auto [a1, a5] = avx::btfly(p1, p5);
            auto [a3, a7] = avx::btfly(p3, p7);

            reg_t b5 = {avx::add(a5.real, a7.imag), avx::sub(a5.imag, a7.real)};
            reg_t b7 = {avx::sub(a5.real, a7.imag), avx::add(a5.imag, a7.real)};

            reg_t b5_tw = {avx::add(b5.real, b5.imag), avx::sub(b5.imag, b5.real)};
            reg_t b7_tw = {avx::sub(b7.real, b7.imag), avx::add(b7.real, b7.imag)};

            auto [b1, b3] = avx::btfly(a1, a3);

            b5_tw = avx::mul(b5_tw, twsq2);
            b7_tw = avx::mul(b7_tw, twsq2);

            auto p0 = avx::cxload<PTform>(avx::ra_addr<PTform>(src0, offset1));
            auto p4 = avx::cxload<PTform>(avx::ra_addr<PTform>(src4, offset1));
            auto p2 = avx::cxload<PTform>(avx::ra_addr<PTform>(src2, offset1));
            auto p6 = avx::cxload<PTform>(avx::ra_addr<PTform>(src6, offset1));

            std::tie(p0, p4, p2, p6) = avx::convert<float>::split<PSrc>(p0, p4, p2, p6);
            std::tie(p0, p4, p2, p6) = avx::convert<float>::inverse<Inverse>(p0, p4, p2, p6);

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

            reg_t q0, q1, q2, q3, q4, q5, q6, q7;

            std::tie(shc0, shc1, shc2, shc3) = avx::convert<float>::inverse<Inverse>(shc0, shc1, shc2, shc3);

            q0 = cxloadstore<PTform>(avx::ra_addr<PTform>(src0, offset2), shc0);
            q4 = cxloadstore<PTform>(avx::ra_addr<PTform>(src4, offset2), shc2);

            if constexpr (PSrc < 4) {
                q2 = cxloadstore<PTform>(avx::ra_addr<PTform>(src2, offset2), shc1);
                q6 = cxloadstore<PTform>(avx::ra_addr<PTform>(src6, offset2), shc3);

                std::tie(q0, q4, q2, q6) = avx::convert<float>::split<PSrc>(q0, q4, q2, q6);
                std::tie(q0, q4, q2, q6) = avx::convert<float>::inverse<Inverse>(q0, q4, q2, q6);
            } else {
                q1 = cxloadstore<PTform>(avx::ra_addr<PTform>(src1, offset2), shc1);
                q5 = cxloadstore<PTform>(avx::ra_addr<PTform>(src5, offset2), shc3);

                std::tie(q0, q4, q1, q5) = avx::convert<float>::split<PSrc>(q0, q4, q1, q5);
                std::tie(q0, q4, q1, q5) = avx::convert<float>::inverse<Inverse>(q0, q4, q1, q5);
            }

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            std::tie(shc4, shc5, shc6, shc7) = avx::convert<float>::inverse<Inverse>(shc4, shc5, shc6, shc7);

            if constexpr (PSrc < 4) {
                q1 = cxloadstore<PTform>(avx::ra_addr<PTform>(src1, offset2), shc4);
                q5 = cxloadstore<PTform>(avx::ra_addr<PTform>(src5, offset2), shc6);
            } else {
                q2 = cxloadstore<PTform>(avx::ra_addr<PTform>(src2, offset2), shc4);
                q6 = cxloadstore<PTform>(avx::ra_addr<PTform>(src6, offset2), shc6);
            }
            q3 = cxloadstore<PTform>(avx::ra_addr<PTform>(src3, offset2), shc5);
            q7 = cxloadstore<PTform>(avx::ra_addr<PTform>(src7, offset2), shc7);

            if constexpr (PSrc < 4) {
                std::tie(q1, q5, q3, q7) = avx::convert<float>::split<PSrc>(q1, q5, q3, q7);
                std::tie(q1, q5, q3, q7) = avx::convert<float>::inverse<Inverse>(q1, q5, q3, q7);
            } else {
                std::tie(q2, q6, q3, q7) = avx::convert<float>::split<PSrc>(q2, q6, q3, q7);
                std::tie(q2, q6, q3, q7) = avx::convert<float>::inverse<Inverse>(q2, q6, q3, q7);
            }

            auto [x1, x5] = avx::btfly(q1, q5);
            auto [x3, x7] = avx::btfly(q3, q7);

            reg_t y5 = {avx::add(x5.real, x7.imag), avx::sub(x5.imag, x7.real)};
            reg_t y7 = {avx::sub(x5.real, x7.imag), avx::add(x5.imag, x7.real)};

            auto [y1, y3] = avx::btfly(x1, x3);

            reg_t y5_tw = {avx::add(y5.real, y5.imag), avx::sub(y5.imag, y5.real)};
            reg_t y7_tw = {avx::sub(y7.real, y7.imag), avx::add(y7.real, y7.imag)};

            auto [x0, x4] = avx::btfly(q0, q4);

            y5_tw = avx::mul(y5_tw, twsq2);
            y7_tw = avx::mul(y7_tw, twsq2);

            auto [x2, x6] = avx::btfly(q2, q6);

            auto [y0, y2] = avx::btfly(x0, x2);

            reg_t y4 = {avx::add(x4.real, x6.imag), avx::sub(x4.imag, x6.real)};
            reg_t y6 = {avx::sub(x4.real, x6.imag), avx::add(x4.imag, x6.real)};

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

            std::tie(shz0, shz1, shz2, shz3) = avx::convert<float>::inverse<Inverse>(shz0, shz1, shz2, shz3);

            avx::cxstore<PTform>(avx::ra_addr<PTform>(src4, offset1), shz2);
            avx::cxstore<PTform>(avx::ra_addr<PTform>(src0, offset1), shz0);
            if constexpr (PSrc < 4) {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src2, offset1), shz1);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src6, offset1), shz3);
            } else {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src1, offset1), shz1);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src5, offset1), shz3);
            }

            auto [shy4, shy6] = avx::unpack_pd(shx4, shx6);
            auto [shy5, shy7] = avx::unpack_pd(shx5, shx7);

            auto [shz4, shz5] = avx::unpack_128(shy4, shy5);
            auto [shz6, shz7] = avx::unpack_128(shy6, shy7);

            std::tie(shz4, shz5, shz6, shz7) = avx::convert<float>::inverse<Inverse>(shz4, shz5, shz6, shz7);

            if constexpr (PSrc < 4) {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src1, offset1), shz4);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src5, offset1), shz6);
            } else {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src2, offset1), shz4);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src6, offset1), shz6);
            }
            avx::cxstore<PTform>(avx::ra_addr<PTform>(src3, offset1), shz5);
            avx::cxstore<PTform>(avx::ra_addr<PTform>(src7, offset1), shz7);
        };

        for (; i < size() / 64; ++i) {
            using reg_t = avx::cx_reg<float>;
            auto offset = m_sort[i] * reg_size;

            auto p1 = avx::cxload<PTform>(avx::ra_addr<PTform>(src1, offset));
            auto p3 = avx::cxload<PTform>(avx::ra_addr<PTform>(src3, offset));
            auto p5 = avx::cxload<PTform>(avx::ra_addr<PTform>(src5, offset));
            auto p7 = avx::cxload<PTform>(avx::ra_addr<PTform>(src7, offset));

            std::tie(p1, p5, p3, p7) = avx::convert<float>::split<PSrc>(p1, p5, p3, p7);
            std::tie(p1, p5, p3, p7) = avx::convert<float>::inverse<Inverse>(p1, p5, p3, p7);

            auto [a1, a5] = avx::btfly(p1, p5);
            auto [a3, a7] = avx::btfly(p3, p7);

            reg_t b5 = {avx::add(a5.real, a7.imag), avx::sub(a5.imag, a7.real)};
            reg_t b7 = {avx::sub(a5.real, a7.imag), avx::add(a5.imag, a7.real)};

            auto [b1, b3] = avx::btfly(a1, a3);

            reg_t b5_tw = {avx::add(b5.real, b5.imag), avx::sub(b5.imag, b5.real)};
            reg_t b7_tw = {avx::sub(b7.real, b7.imag), avx::add(b7.real, b7.imag)};

            b5_tw = avx::mul(b5_tw, twsq2);
            b7_tw = avx::mul(b7_tw, twsq2);

            auto p0 = avx::cxload<PTform>(avx::ra_addr<PTform>(src0, offset));
            auto p2 = avx::cxload<PTform>(avx::ra_addr<PTform>(src2, offset));
            auto p4 = avx::cxload<PTform>(avx::ra_addr<PTform>(src4, offset));
            auto p6 = avx::cxload<PTform>(avx::ra_addr<PTform>(src6, offset));

            std::tie(p0, p4, p2, p6) = avx::convert<float>::split<PSrc>(p0, p4, p2, p6);
            std::tie(p0, p4, p2, p6) = avx::convert<float>::inverse<Inverse>(p0, p4, p2, p6);

            auto [a0, a4] = avx::btfly(p0, p4);
            auto [a2, a6] = avx::btfly(p2, p6);

            auto [b0, b2] = avx::btfly(a0, a2);

            reg_t b4 = {avx::add(a4.real, a6.imag), avx::sub(a4.imag, a6.real)};
            reg_t b6 = {avx::sub(a4.real, a6.imag), avx::add(a4.imag, a6.real)};

            auto [c0, c1] = avx::btfly(b0, b1);

            reg_t c2 = {avx::add(b2.real, b3.imag), avx::sub(b2.imag, b3.real)};
            reg_t c3 = {avx::sub(b2.real, b3.imag), avx::add(b2.imag, b3.real)};

            auto [c4, c5] = avx::btfly(b4, b5_tw);

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

            std::tie(shc0, shc1, shc2, shc3) = avx::convert<float>::inverse<Inverse>(shc0, shc1, shc2, shc3);

            avx::cxstore<PTform>(avx::ra_addr<PTform>(src0, offset), shc0);
            avx::cxstore<PTform>(avx::ra_addr<PTform>(src4, offset), shc2);
            if constexpr (PSrc < 4) {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src2, offset), shc1);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src6, offset), shc3);
            } else {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src1, offset), shc1);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src5, offset), shc3);
            }

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);

            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            std::tie(shc4, shc5, shc6, shc7) = avx::convert<float>::inverse<Inverse>(shc4, shc5, shc6, shc7);

            if constexpr (PSrc < 4) {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src1, offset), shc4);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src5, offset), shc6);
            } else {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src2, offset), shc4);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(src6, offset), shc6);
            }
            avx::cxstore<PTform>(avx::ra_addr<PTform>(src3, offset), shc5);
            avx::cxstore<PTform>(avx::ra_addr<PTform>(src7, offset), shc7);
        }
    }

    template<std::size_t PTform, std::size_t PSrc>
    inline void depth3_and_sort(float* dest, const float* source) {
        constexpr auto PLoad = std::max(PSrc, reg_size);

        const auto sq2   = wnk(8, 1);
        auto       twsq2 = avx::broadcast(sq2.real());

        auto* src0 = source;
        auto* src1 = avx::ra_addr<PTform>(source, 1 * size() / 8);
        auto* src2 = avx::ra_addr<PTform>(source, 2 * size() / 8);
        auto* src3 = avx::ra_addr<PTform>(source, 3 * size() / 8);
        auto* src4 = avx::ra_addr<PTform>(source, 4 * size() / 8);
        auto* src5 = avx::ra_addr<PTform>(source, 5 * size() / 8);
        auto* src6 = avx::ra_addr<PTform>(source, 6 * size() / 8);
        auto* src7 = avx::ra_addr<PTform>(source, 7 * size() / 8);

        auto* dst0 = dest;
        auto* dst1 = avx::ra_addr<PTform>(dest, 1 * size() / 8);
        auto* dst2 = avx::ra_addr<PTform>(dest, 2 * size() / 8);
        auto* dst3 = avx::ra_addr<PTform>(dest, 3 * size() / 8);
        auto* dst4 = avx::ra_addr<PTform>(dest, 4 * size() / 8);
        auto* dst5 = avx::ra_addr<PTform>(dest, 5 * size() / 8);
        auto* dst6 = avx::ra_addr<PTform>(dest, 6 * size() / 8);
        auto* dst7 = avx::ra_addr<PTform>(dest, 7 * size() / 8);

        uint i = 0;
        for (; i < n_reversals(size() / 64); i += 2) {
            using reg_t = avx::cx_reg<float>;

            auto offset_src  = m_sort[i] * reg_size;
            auto offset_dest = m_sort[i + 1] * reg_size;

            for (uint k = 0; k < 2; ++k) {
                auto p1 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src1, offset_src));
                auto p5 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src5, offset_src));
                auto p3 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src3, offset_src));
                auto p7 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src7, offset_src));

                if constexpr (PSrc < PLoad) {
                    std::tie(p1, p5, p3, p7) = avx::convert<float>::split<PSrc>(p1, p5, p3, p7);
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

                auto p0 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src0, offset_src));
                auto p4 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src4, offset_src));
                auto p2 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src2, offset_src));
                auto p6 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src6, offset_src));

                if constexpr (PSrc < PLoad) {
                    std::tie(p0, p4, p2, p6) = avx::convert<float>::split<PSrc>(p0, p4, p2, p6);
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

                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst0, offset_dest), shc0);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst4, offset_dest), shc2);
                if constexpr (PSrc < 4) {
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst2, offset_dest), shc1);
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst6, offset_dest), shc3);
                } else {
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst1, offset_dest), shc1);
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst5, offset_dest), shc3);
                }

                auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
                auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);
                auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
                auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

                if constexpr (PSrc < 4) {
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst1, offset_dest), shc4);
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst5, offset_dest), shc6);
                } else {
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst2, offset_dest), shc4);
                    avx::cxstore<PTform>(avx::ra_addr<PTform>(dst6, offset_dest), shc6);
                }
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst3, offset_dest), shc5);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst7, offset_dest), shc7);

                offset_src  = m_sort[i + 1] * reg_size;
                offset_dest = m_sort[i] * reg_size;
            }
        };

        for (; i < size() / 64; ++i) {
            using reg_t = avx::cx_reg<float>;

            auto offset_src  = m_sort[i] * reg_size;
            auto offset_dest = m_sort[i] * reg_size;

            auto p1 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src1, offset_src));
            auto p5 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src5, offset_src));
            auto p3 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src3, offset_src));
            auto p7 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src7, offset_src));

            if constexpr (PSrc < PLoad) {
                std::tie(p1, p5, p3, p7) = avx::convert<float>::split<PSrc>(p1, p5, p3, p7);
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

            auto p0 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src0, offset_src));
            auto p4 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src4, offset_src));
            auto p2 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src2, offset_src));
            auto p6 = avx::cxload<PLoad>(avx::ra_addr<PLoad>(src6, offset_src));

            if constexpr (PSrc < PLoad) {
                std::tie(p0, p4, p2, p6) = avx::convert<float>::split<PSrc>(p0, p4, p2, p6);
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

            avx::cxstore<PTform>(avx::ra_addr<PTform>(dst0, offset_dest), shc0);
            avx::cxstore<PTform>(avx::ra_addr<PTform>(dst4, offset_dest), shc2);
            if constexpr (PSrc < 4) {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst2, offset_dest), shc1);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst6, offset_dest), shc3);
            } else {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst1, offset_dest), shc1);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst5, offset_dest), shc3);
            }

            auto [shb4, shb6] = avx::unpack_pd(sha4, sha6);
            auto [shb5, shb7] = avx::unpack_pd(sha5, sha7);
            auto [shc4, shc5] = avx::unpack_128(shb4, shb5);
            auto [shc6, shc7] = avx::unpack_128(shb6, shb7);

            if constexpr (PSrc < 4) {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst1, offset_dest), shc4);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst5, offset_dest), shc6);
            } else {
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst2, offset_dest), shc4);
                avx::cxstore<PTform>(avx::ra_addr<PTform>(dst6, offset_dest), shc6);
            }
            avx::cxstore<PTform>(avx::ra_addr<PTform>(dst3, offset_dest), shc5);
            avx::cxstore<PTform>(avx::ra_addr<PTform>(dst7, offset_dest), shc7);
        }
    }

    /**
     * @brief
     *
     * @tparam PDest Destination pack size.
     * @tparam PTform Transform pack size. Must be equal to the maximum of PDest and reg_size.
     * @param data
     * @param max_size
     * @return const float* twiddle pointer.
     */
    template<std::size_t PDest, std::size_t PTform, bool Inverse = false, bool Scale = false>
    inline auto subtransform(float* data, std::size_t max_size) -> const float* {
        const auto* twiddle_ptr = m_twiddles.data();

        std::size_t l_size     = reg_size * 2;
        std::size_t group_size = max_size / reg_size / 4;
        std::size_t n_groups   = 1;
        std::size_t tw_offset  = 0;

        std::size_t max_size_ = max_size;
        if constexpr ((PDest < reg_size) || Scale) {
            max_size_ = max_size_ / 2;
        }

        while (l_size < max_size_) {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                std::size_t offset = i_group * reg_size;

                const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
                const auto tw1 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 2);
                const auto tw2 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 4);
                twiddle_ptr += reg_size * 6;

                for (std::size_t i = 0; i < group_size; ++i) {
                    node4_dit<PTform, PTform, Inverse>(data, l_size, offset, tw0, tw1, tw2);
                    offset += l_size * 2;
                }
            }

            l_size *= 4;
            n_groups *= 4;
            group_size /= 4;
        }

        auto scaling = std::conditional_t<Scale, typename avx::reg<float>::type, decltype([] {})>{};
        if constexpr (Scale) {
            scaling = avx::broadcast(static_cast<float>(1 / static_cast<double>(max_size)));
        }
        if constexpr ((PDest < reg_size) || Scale) {
            if (l_size == max_size / 2) {
                for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                    std::size_t offset = i_group * reg_size;

                    const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
                    const auto tw1 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 2);
                    const auto tw2 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 4);
                    twiddle_ptr += reg_size * 6;

                    for (std::size_t i = 0; i < group_size; ++i) {
                        node4_dit<PDest, PTform, Inverse, Scale>(
                            data, l_size, offset, tw0, tw1, tw2, scaling);
                        offset += l_size * 2;
                    }
                }

                return twiddle_ptr;
            }
        }

        if (l_size == max_size) {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                std::size_t offset = i_group * reg_size;
                const auto  tw0    = avx::cxload<reg_size>(twiddle_ptr);
                twiddle_ptr += reg_size * 2;

                node2<PDest, PTform, Inverse, Scale>(data, l_size, offset, tw0, scaling);
            }
        }

        return twiddle_ptr;
    };

    template<std::size_t PDest, std::size_t PTform, bool Inverse = false, bool Scale = false>
    inline auto subtransform_recursive(float* data, std::size_t size) -> const float* {
        if (size <= sub_size()) {
            return subtransform<PDest, PTform, Inverse, Scale>(data, size);
        } else {
            subtransform_recursive<PTform, PTform, Inverse>(data, size / 4);
            subtransform_recursive<PTform, PTform, Inverse>(avx::ra_addr<PTform>(data, size / 4), size / 4);
            subtransform_recursive<PTform, PTform, Inverse>(avx::ra_addr<PTform>(data, size / 2), size / 4);
            auto twiddle_ptr = subtransform_recursive<PTform, PTform, Inverse>(
                avx::ra_addr<PTform>(data, size * 3 / 4), size / 4);

            std::size_t n_groups = size / reg_size / 4;

            auto scaling = std::conditional_t<Scale, typename avx::reg<float>::type, decltype([] {})>{};
            if constexpr (Scale) {
                scaling = avx::broadcast(static_cast<float>(1 / static_cast<double>(size)));
            }

            for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                auto offset = i_group * reg_size;

                const auto tw0 = avx::cxload<reg_size>(twiddle_ptr);
                const auto tw1 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 2);
                const auto tw2 = avx::cxload<reg_size>(twiddle_ptr + reg_size * 4);
                twiddle_ptr += reg_size * 6;

                node4_dit<PDest, PTform, Inverse, Scale>(data, size / 2, offset, tw0, tw1, tw2, scaling);
            }

            return twiddle_ptr;
        }
    };

    template<std::size_t PDest, std::size_t PSrc>
    inline auto unsorted_subtransform(float* data, std::size_t size, const float* twiddle_ptr) -> const
        float* {
        constexpr auto PTform = std::max(PSrc, reg_size);

        using reg_t = avx::cx_reg<float>;

        std::size_t l_size   = size;
        std::size_t n_groups = 1;

        if constexpr (PSrc < PTform) {
            if (l_size > reg_size * 8) {
                reg_t tw0 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};

                reg_t tw1 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};
                reg_t tw2 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};

                auto* group_ptr = data;

                for (std::size_t i = 0; i < l_size / reg_size / 4; ++i) {
                    auto offset = i * reg_size;
                    node4_dif<PTform, PSrc>(data, l_size, offset, tw0, tw1, tw2);
                }

                l_size /= 4;
                n_groups *= 4;
            } else if (l_size == reg_size * 8) {
                reg_t tw0 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};
                for (std::size_t i = 0; i < l_size / reg_size / 2; ++i) {
                    std::size_t offset = i * reg_size;
                    node2<PTform, PSrc>(data, l_size, offset, tw0);
                }

                l_size /= 2;
                n_groups *= 2;
            }
        }

        while (l_size > reg_size * 8) {
            for (uint i_group = 0; i_group < n_groups; ++i_group) {
                reg_t tw0 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};

                reg_t tw1 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};
                reg_t tw2 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};

                auto* group_ptr = avx::ra_addr<PTform>(data, i_group * l_size);

                for (std::size_t i = 0; i < l_size / reg_size / 4; ++i) {
                    auto offset = i * reg_size;
                    node4_dif<PTform, PTform>(group_ptr, l_size, offset, tw0, tw1, tw2);
                }
            }
            l_size /= 4;
            n_groups *= 4;
        }

        if (l_size == reg_size * 8) {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                reg_t tw0 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};

                auto* group_ptr = avx::ra_addr<PTform>(data, i_group * l_size);

                for (std::size_t i = 0; i < l_size / reg_size / 2; ++i) {
                    std::size_t offset = i * reg_size;
                    node2<PTform>(group_ptr, l_size, offset, tw0);
                }
            }
            l_size /= 2;
            n_groups *= 2;
        }

        if (l_size == reg_size * 4) {
            for (std::size_t i_group = 0; i_group < size / reg_size / 4; ++i_group) {
                reg_t tw0 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};

                reg_t tw1 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};
                reg_t tw2 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};

                auto* ptr0 = avx::ra_addr<PTform>(data, reg_size * (i_group * 4));
                auto* ptr1 = avx::ra_addr<PTform>(data, reg_size * (i_group * 4 + 1));
                auto* ptr2 = avx::ra_addr<PTform>(data, reg_size * (i_group * 4 + 2));
                auto* ptr3 = avx::ra_addr<PTform>(data, reg_size * (i_group * 4 + 3));

                auto p2 = avx::cxload<PTform>(ptr2);
                auto p3 = avx::cxload<PTform>(ptr3);
                auto p0 = avx::cxload<PTform>(ptr0);
                auto p1 = avx::cxload<PTform>(ptr1);

                auto [p2tw, p3tw] = avx::mul({p2, tw0}, {p3, tw0});

                auto [a1, a3] = avx::btfly(p1, p3tw);
                auto [a0, a2] = avx::btfly(p0, p2tw);

                auto [a1tw, a3tw] = avx::mul({a1, tw1}, {a3, tw2});

                reg_t tw3_1 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};
                reg_t tw3_2 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};
                reg_t tw3   = {avx::unpacklo_128(tw3_1.real, tw3_2.real),
                               avx::unpacklo_128(tw3_1.imag, tw3_2.imag)};
                reg_t tw4_1 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};
                reg_t tw4_2 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};
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

                if constexpr (PDest < PTform) {
                    std::tie(she0, she1, she2, she3) =
                        avx::convert<float>::repack<PTform, PDest>(she0, she1, she2, she3);
                }

                cxstore<PTform>(ptr0, she0);
                cxstore<PTform>(ptr1, she1);
                cxstore<PTform>(ptr2, she2);
                cxstore<PTform>(ptr3, she3);
            }
        }

        return twiddle_ptr;
    }

    template<std::size_t PDest, std::size_t PSrc>
    inline auto unsorted_subtransform_recursive(float* data, std::size_t size, const float* twiddle_ptr)
        -> const float* {
        if (size <= sub_size()) {
            return unsorted_subtransform<PDest, PSrc>(data, size, twiddle_ptr);
        } else {
            constexpr auto PTform = std::max(PSrc, reg_size);

            using reg_t = avx::cx_reg<float>;

            reg_t tw0 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};
            reg_t tw1 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};
            reg_t tw2 = {avx::broadcast(twiddle_ptr++), avx::broadcast(twiddle_ptr++)};

            for (std::size_t i_group = 0; i_group < size / 4 / reg_size; ++i_group) {
                auto offset = i_group * reg_size;
                node4_dif<PTform, PSrc>(data, size, offset, tw0, tw1, tw2);
            }

            twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform>(    //
                data,
                size / 4,
                twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform>(
                avx::ra_addr<PTform>(data, size / 4), size / 4, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform>(
                avx::ra_addr<PTform>(data, size / 2), size / 4, twiddle_ptr);
            twiddle_ptr = unsorted_subtransform_recursive<PDest, PTform>(
                avx::ra_addr<PTform>(data, size / 4 * 3), size / 4, twiddle_ptr);
            return twiddle_ptr;
        }
    };

    template<std::size_t PDest,
             std::size_t PSrc    = PDest,
             bool        Inverse = false,
             bool        IScale  = false,
             typename ScaleT     = uint>
    inline void
    node2(float* data, std::size_t l_size, std::size_t offset, avx::cx_reg<float> tw0, ScaleT scaling = 0) {
        constexpr auto PLoad  = std::max(PSrc, reg_size);
        constexpr auto PStore = std::max(PDest, reg_size);

        auto* ptr0 = avx::ra_addr<PLoad>(data, offset);
        auto* ptr1 = avx::ra_addr<PLoad>(data, offset + l_size / 2);

        auto p1 = avx::cxload<PLoad>(ptr1);
        auto p0 = avx::cxload<PLoad>(ptr0);

        std::tie(p1, p0) = avx::convert<float>::repack<PSrc, PLoad>(p1, p0);
        std::tie(p1, p0) = avx::convert<float>::inverse<Inverse>(p1, p0);

        auto p1tw = avx::mul(p1, tw0);

        auto [a0, a1] = avx::btfly(p0, p1tw);

        if constexpr (IScale) {
            a0 = avx::mul(a0, scaling);
            a1 = avx::mul(a1, scaling);
        }

        std::tie(a0, a1) = avx::convert<float>::inverse<Inverse>(a0, a1);
        std::tie(a0, a1) = avx::convert<float>::repack<PStore, PDest>(a0, a1);

        cxstore<PStore>(ptr0, a0);
        cxstore<PStore>(ptr1, a1);
    }

    template<std::size_t PDest,
             std::size_t PSrc    = PDest,
             bool        Inverse = false,
             bool        IScale  = false,
             typename ScaleT     = uint>
    inline void node4_dit(float*             data,
                          std::size_t        l_size,
                          std::size_t        offset,
                          avx::cx_reg<float> tw0,
                          avx::cx_reg<float> tw1,
                          avx::cx_reg<float> tw2,
                          ScaleT             scaling = 0) {
        constexpr auto PLoad  = std::max(PSrc, reg_size);
        constexpr auto PStore = std::max(PDest, reg_size);

        auto* ptr0 = avx::ra_addr<PLoad>(data, offset);
        auto* ptr1 = avx::ra_addr<PLoad>(data, offset + l_size / 2);
        auto* ptr2 = avx::ra_addr<PLoad>(data, offset + l_size);
        auto* ptr3 = avx::ra_addr<PLoad>(data, offset + l_size / 2 * 3);

        auto p1 = avx::cxload<PLoad>(ptr1);
        auto p3 = avx::cxload<PLoad>(ptr3);
        auto p0 = avx::cxload<PLoad>(ptr0);
        auto p2 = avx::cxload<PLoad>(ptr2);

        std::tie(p1, p3, p0, p2) = avx::convert<float>::repack<PSrc, PLoad>(p1, p3, p0, p2);
        std::tie(p1, p3, p0, p2) = avx::convert<float>::inverse<Inverse>(p1, p3, p0, p2);

        auto [p1tw, p3tw] = avx::mul({p1, tw0}, {p3, tw0});

        auto [a2, a3] = avx::btfly(p2, p3tw);
        auto [a0, a1] = avx::btfly(p0, p1tw);

        auto [a2tw, a3tw] = avx::mul({a2, tw1}, {a3, tw2});

        auto [b0, b2] = avx::btfly(a0, a2tw);
        auto [b1, b3] = avx::btfly(a1, a3tw);

        if constexpr (IScale) {
            b0 = avx::mul(b0, scaling);
            b1 = avx::mul(b1, scaling);
            b2 = avx::mul(b2, scaling);
            b3 = avx::mul(b3, scaling);
        }

        std::tie(b0, b1, b2, b3) = avx::convert<float>::inverse<Inverse>(b0, b1, b2, b3);
        std::tie(b0, b1, b2, b3) = avx::convert<float>::repack<PStore, PDest>(b0, b1, b2, b3);

        cxstore<PStore>(ptr0, b0);
        cxstore<PStore>(ptr1, b1);
        cxstore<PStore>(ptr2, b2);
        cxstore<PStore>(ptr3, b3);
    }

    template<std::size_t PDest, std::size_t PSrc = PDest>
    inline void node4_dif(float*             data,
                          std::size_t        l_size,
                          std::size_t        offset,
                          avx::cx_reg<float> tw0,
                          avx::cx_reg<float> tw1,
                          avx::cx_reg<float> tw2) {
        constexpr auto PLoad  = std::max(PSrc, reg_size);
        constexpr auto PStore = std::max(PDest, reg_size);

        auto* ptr0 = avx::ra_addr<PLoad>(data, offset);
        auto* ptr1 = avx::ra_addr<PLoad>(data, offset + l_size / 4);
        auto* ptr2 = avx::ra_addr<PLoad>(data, offset + l_size / 2);
        auto* ptr3 = avx::ra_addr<PLoad>(data, offset + l_size / 4 * 3);

        auto p2 = avx::cxload<PLoad>(ptr2);
        auto p3 = avx::cxload<PLoad>(ptr3);
        auto p0 = avx::cxload<PLoad>(ptr0);
        auto p1 = avx::cxload<PLoad>(ptr1);

        std::tie(p2, p3, p0, p1) = avx::convert<float>::repack<PSrc, PLoad>(p2, p3, p0, p1);

        auto [p2tw, p3tw] = avx::mul({p2, tw0}, {p3, tw0});

        auto [a0, a2] = avx::btfly(p0, p2tw);
        auto [a1, a3] = avx::btfly(p1, p3tw);

        auto [a1tw, a3tw] = avx::mul({a1, tw1}, {a3, tw2});

        auto [b0, b1] = avx::btfly(a0, a1tw);
        auto [b2, b3] = avx::btfly(a2, a3tw);

        std::tie(b0, b1, b2, b3) = avx::convert<float>::repack<PStore, PDest>(b0, b1, b2, b3);

        cxstore<PStore>(ptr0, b0);
        cxstore<PStore>(ptr1, b1);
        cxstore<PStore>(ptr2, b2);
        cxstore<PStore>(ptr3, b3);
    };

private:
    static constexpr auto check_size(std::size_t size) -> std::size_t {
        if (size > 1 && (size & (size - 1)) == 0) {
            return size;
        }
        throw(std::invalid_argument("fft_size (which is  " + std::to_string(size) +
                                    ") is not an integer power of two"));
    }

    static constexpr auto check_sub_size(std::size_t sub_size) -> std::size_t {
        if (sub_size >= pcx::default_pack_size<T> && (sub_size & (sub_size - 1)) == 0) {
            return sub_size;
        }
        if (sub_size < pcx::default_pack_size<T>) {
            throw(std::invalid_argument("fft_sub_size (which is  " + std::to_string(sub_size) +
                                        ") is smaller than minimum (which is " +
                                        std::to_string(pcx::default_pack_size<T>) + ")"));
        }
        if ((sub_size & (sub_size - 1)) != 0) {
            throw(std::invalid_argument("fft_sub_size (which is  " + std::to_string(sub_size) +
                                        ") is not an integer power of two"));
        }
    }

    static constexpr auto log2i(std::size_t num) -> std::size_t {
        std::size_t order = 0;
        while ((num >>= 1U) != 0) {
            order++;
        }
        return order;
    }

    static constexpr auto reverse_bit_order(uint64_t num, uint64_t depth) -> uint64_t {
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
    static constexpr auto n_reversals(std::size_t max) -> std::size_t {
        return max - (1U << ((log2i(max) + 1) / 2));
    }

    static inline auto wnk(std::size_t n, std::size_t k) -> std::complex<real_type> {
        constexpr double pi = 3.14159265358979323846;
        return exp(std::complex<real_type>(0, -2 * pi * static_cast<double>(k) / static_cast<double>(n)));
    }

    static auto get_sort(std::size_t fft_size, sort_allocator_type allocator)
        -> std::vector<std::size_t, sort_allocator_type> {
        const auto packed_sort_size = fft_size / reg_size / reg_size;
        const auto order            = log2i(packed_sort_size);
        auto       sort             = std::vector<std::size_t, sort_allocator_type>(allocator);
        sort.reserve(packed_sort_size);

        for (uint i = 0; i < packed_sort_size; ++i) {
            if (i >= reverse_bit_order(i, order)) {
                continue;
            }
            sort.push_back(i);
            sort.push_back(reverse_bit_order(i, order));
        }
        for (uint i = 0; i < packed_sort_size; ++i) {
            if (i == reverse_bit_order(i, order)) {
                sort.push_back(i);
            }
        }
        return sort;
    }

    static auto get_twiddles(std::size_t fft_size, std::size_t sub_size, allocator_type allocator)
        -> pcx::vector<real_type, allocator_type, reg_size> {
        const auto depth = log2i(fft_size);

        const std::size_t n_twiddles = 8 * ((1U << (depth - 3)) - 1U);

        auto twiddles = pcx::vector<real_type, allocator_type, reg_size>(n_twiddles, allocator);

        auto tw_it = twiddles.begin();

        std::size_t l_size   = reg_size * 2;
        std::size_t n_groups = 1;

        std::size_t sub_size_ = std::min(fft_size, sub_size);
        if (fft_size > sub_size_ && log2i(fft_size / sub_size_) % 2 != 0) {
            sub_size_ = sub_size_ / 2;
        }

        while (l_size < sub_size_) {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                for (uint k = 0; k < reg_size; ++k) {
                    *(tw_it++) = wnk(l_size, k + i_group * reg_size);
                }
                for (uint k = 0; k < reg_size; ++k) {
                    *(tw_it++) = wnk(l_size * 2UL, k + i_group * reg_size);
                }
                for (uint k = 0; k < reg_size; ++k) {
                    *(tw_it++) = wnk(l_size * 2UL, k + i_group * reg_size + l_size / 2);
                }
            }
            l_size *= 4;
            n_groups *= 4;
        }

        if (l_size == sub_size_) {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                for (uint k = 0; k < reg_size; ++k) {
                    *(tw_it++) = wnk(l_size, k + i_group * reg_size);
                }
            }
            l_size *= 2;
            n_groups *= 2;
        };

        while (l_size < fft_size) {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                for (uint k = 0; k < reg_size; ++k) {
                    *(tw_it++) = wnk(l_size, k + i_group * reg_size);
                }
                for (uint k = 0; k < reg_size; ++k) {
                    auto a     = wnk(l_size * 2UL, k + i_group * reg_size);
                    *(tw_it++) = wnk(l_size * 2UL, k + i_group * reg_size);
                }
                for (uint k = 0; k < reg_size; ++k) {
                    *(tw_it++) = wnk(l_size * 2UL, k + i_group * reg_size + l_size / 2);
                }
            }
            l_size *= 4;
            n_groups *= 4;
        };
        if (l_size == fft_size) {
            for (std::size_t i_group = 0; i_group < n_groups; ++i_group) {
                for (uint k = 0; k < reg_size; ++k) {
                    *(tw_it++) = wnk(l_size, k + i_group * reg_size);
                }
            }
            l_size *= 2;
            n_groups *= 2;
        };
        return twiddles;
    }


    static auto get_twiddles_unsorted(std::size_t fft_size, std::size_t sub_size, allocator_type allocator)
        -> std::vector<T, allocator_type> {
        auto twiddles = std::vector<T, allocator_type>(allocator);

        std::size_t l_size = 2;
        if (log2i(fft_size / sub_size) % 2 == 0) {
            insert_tw_unsorted(fft_size, l_size, sub_size, 0, twiddles);
        } else {
            auto tw0 = wnk(l_size, reverse_bit_order(0, log2i(l_size / 2)));

            twiddles.push_back(tw0.real());
            twiddles.push_back(tw0.imag());

            insert_tw_unsorted(fft_size, l_size * 2, sub_size, 0, twiddles);
            insert_tw_unsorted(fft_size, l_size * 2, sub_size, 1, twiddles);
        }
        return twiddles;
    }

    static void insert_tw_unsorted(std::size_t                             fft_size,
                                   std::size_t                             l_size,
                                   std::size_t                             sub_size,
                                   std::size_t                             i_group,
                                   std::vector<real_type, allocator_type>& twiddles) {
        if ((fft_size / l_size) < sub_size) {
            std::size_t start_size       = twiddles.size();
            std::size_t single_load_size = fft_size / (reg_size * 2);
            std::size_t group_size       = 1;

            while (l_size < single_load_size / 2) {
                std::size_t start = group_size * i_group;
                for (uint i = 0; i < group_size; ++i) {
                    auto tw0 = wnk(l_size,    //
                                   reverse_bit_order((start + i), log2i(l_size / 2)));
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

            if (l_size == single_load_size / 2) {
                std::size_t start = group_size * i_group;

                for (uint i = 0; i < group_size; ++i) {
                    auto tw0 = wnk(l_size,    //
                                   reverse_bit_order(start + i, log2i(l_size / 2)));

                    twiddles.push_back(tw0.real());
                    twiddles.push_back(tw0.imag());
                }
                l_size *= 2;
                group_size *= 2;
            }

            if (l_size == single_load_size) {
                for (uint i = 0; i < group_size; ++i) {
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

                    for (uint k = 0; k < 8; ++k) {
                        auto tw = wnk(l_size * 16,    //
                                      reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                        twiddles.push_back(tw.real());
                    }
                    for (uint k = 0; k < 8; ++k) {
                        auto tw = wnk(l_size * 16,    //
                                      reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                        twiddles.push_back(tw.imag());
                    }

                    for (uint k = 8; k < 16; ++k) {
                        auto tw = wnk(l_size * 16,    //
                                      reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                        twiddles.push_back(tw.real());
                    }
                    for (uint k = 8; k < 16; ++k) {
                        auto tw = wnk(l_size * 16,    //
                                      reverse_bit_order(start * 16 + k, log2i(l_size * 8)));
                        twiddles.push_back(tw.imag());
                    }
                }
            }
        } else {
            auto tw0 = wnk(l_size, reverse_bit_order(i_group, log2i(l_size / 2)));
            auto tw1 = wnk(l_size * 2, reverse_bit_order(i_group * 2, log2i(l_size)));
            auto tw2 = wnk(l_size * 2, reverse_bit_order(i_group * 2 + 1, log2i(l_size)));

            twiddles.push_back(tw0.real());
            twiddles.push_back(tw0.imag());
            twiddles.push_back(tw1.real());
            twiddles.push_back(tw1.imag());
            twiddles.push_back(tw2.real());
            twiddles.push_back(tw2.imag());

            l_size *= 4;
            i_group *= 4;

            insert_tw_unsorted(fft_size, l_size, sub_size, i_group + 0, twiddles);
            insert_tw_unsorted(fft_size, l_size, sub_size, i_group + 1, twiddles);
            insert_tw_unsorted(fft_size, l_size, sub_size, i_group + 2, twiddles);
            insert_tw_unsorted(fft_size, l_size, sub_size, i_group + 3, twiddles);
        }
    }
};


}    // namespace pcx
#endif