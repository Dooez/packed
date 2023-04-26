#include "fft.hpp"
#include "vector_util.hpp"

#include <cstddef>

namespace pcx {

template<typename T,
         std::size_t PackSize = pcx::default_pack_size<T>,
         typename Allocator   = pcx::aligned_allocator<T>>
class fxcorr_unit {
    using real_type      = T;
    using allocator_type = Allocator;

public:
    fxcorr_unit() = delete;

    fxcorr_unit(pcx::vector<T> g, std::size_t fft_size, allocator_type allocator = allocator_type{})
    : m_fft(fft_size, 2048, allocator)
    , m_kernel(fft_size, allocator)
    , m_tmp(fft_size, allocator)
    , m_overlap(g.size() - 1) {
        m_fft(m_kernel, g);
        m_kernel = m_kernel / static_cast<T>(fft_size);
    };

    fxcorr_unit(const fxcorr_unit& other)     = default;
    fxcorr_unit(fxcorr_unit&& other) noexcept = default;

    ~fxcorr_unit() = default;

    fxcorr_unit& operator=(const fxcorr_unit& other)     = delete;
    fxcorr_unit& operator=(fxcorr_unit&& other) noexcept = delete;

    template<typename VAllocator, std::size_t VPackSize>
    void operator()(pcx::vector<T, VAllocator, VPackSize>& vec) {
        uint offset = 0;
        auto step   = (m_fft.size() - m_overlap) / VPackSize * VPackSize;
        for (; offset + m_fft.size() < vec.size(); offset += step) {
            m_fft(m_tmp, subrange(vec.begin() + offset, m_fft.size()));
            m_tmp = m_tmp * conj(m_kernel);
            m_fft.ifft(m_tmp);
            subrange(vec.begin() + offset, step).assign(m_tmp);
        }
        while (offset < vec.size()) {
            m_fft(m_tmp, subrange(vec.begin() + offset, vec.end()));
            m_tmp = m_tmp * conj(m_kernel);
            m_fft.ifft(m_tmp);
            auto l_step = std::min(step, vec.size() - offset);
            subrange(vec.begin() + offset, l_step).assign(subrange(m_tmp.begin(), l_step));
            offset += step;
        }
    }

private:
    fft_unit<T, pcx::fft_output::unsorted, pcx::dynamic_size, pcx::dynamic_size, allocator_type> m_fft;
    pcx::vector<T, allocator_type>                                                               m_kernel;
    pcx::vector<T, allocator_type>                                                               m_tmp;
    std::size_t                                                                                  m_overlap;
};
}    // namespace pcx