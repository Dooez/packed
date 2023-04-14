#include "fft.hpp"

namespace pcx {

template<typename T, typename Allocator = pcx::aligned_allocator<T, std::align_val_t(64)>>
class fconv_unit {
    using real_type      = T;
    using allocator_type = Allocator;

public:
    fconv_unit() = delete;

    fconv_unit(pcx::vector<T> g,
               std::size_t    f_size,
               std::size_t    fft_size,
               allocator_type allocator = allocator_type{})
    : m_fft(fft_size, 2048, allocator)
    , m_kernel(fft_size) {
        pcx::subrange(m_kernel.begin(), g.size()).assign(g);
        m_fft(m_kernel, g);
    };

    fconv_unit(const fconv_unit& other)     = default;
    fconv_unit(fconv_unit&& other) noexcept = default;

    ~fconv_unit() = default;

    fconv_unit& operator=(const fconv_unit& other)     = delete;
    fconv_unit& operator=(fconv_unit&& other) noexcept = delete;

private:
    fft_unit<T, pcx::dynamic_size, pcx::dynamic_size, allocator_type> m_fft;
    pcx::vector<T, allocator_type>                                    m_kernel;
};
}    // namespace pcx