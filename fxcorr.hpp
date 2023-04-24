#include "fft.hpp"
#include "vector_util.hpp"

namespace pcx {

template<typename T,
         std::size_t PackSize = pcx::default_pack_size<T>,
         typename Allocator   = pcx::aligned_allocator<T, std::align_val_t{64}>>
class fxcorr_unit {
    using real_type      = T;
    using allocator_type = Allocator;

public:
    fxcorr_unit() = delete;

    fxcorr_unit(pcx::vector<T> g, std::size_t fft_size, allocator_type allocator = allocator_type{})
    : m_fft(fft_size, 2048, allocator)
    , m_kernel(fft_size, allocator)
    , m_tmp(fft_size, allocator) {
        m_fft.unsorted(m_kernel, g);
        m_kernel    = conj(m_kernel / static_cast<T>(fft_size));
        auto l_size = (g.size() + PackSize - 1) / PackSize * PackSize + 1;
        m_overlap   = l_size - 1;
        auto m_save = fft_size - m_overlap;
    };

    fxcorr_unit(const fxcorr_unit& other)     = default;
    fxcorr_unit(fxcorr_unit&& other) noexcept = default;

    ~fxcorr_unit() = default;

    fxcorr_unit& operator=(const fxcorr_unit& other)     = delete;
    fxcorr_unit& operator=(fxcorr_unit&& other) noexcept = delete;

    void operator()(pcx::vector<T> f) {
        m_fft.unsorted(m_tmp, f);
        m_tmp = m_tmp * m_kernel;
    }

private:
    fft_unit<T, pcx::dynamic_size, pcx::dynamic_size, allocator_type> m_fft;
    pcx::vector<T, allocator_type>                                    m_kernel;
    pcx::vector<T, allocator_type>                                    m_tmp;
    std::size_t                                                       m_overlap;
};
}    // namespace pcx