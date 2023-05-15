#include "fft.hpp"
#include "vector_util.hpp"

#include <concepts>
#include <cstddef>
#include <memory>

namespace pcx {

namespace internal {
template<typename T, typename Allocator = pcx::aligned_allocator<T>>
class pseudo_vector_factory {
public:
    explicit pseudo_vector_factory(std::size_t size, const Allocator& allocator = Allocator{})
    : m_vector(size, allocator){};

    pseudo_vector_factory(const pseudo_vector_factory& other)     = delete;
    pseudo_vector_factory(pseudo_vector_factory&& other) noexcept = delete;

    ~pseudo_vector_factory() = default;

    pseudo_vector_factory& operator=(const pseudo_vector_factory& other)     = delete;
    pseudo_vector_factory& operator=(pseudo_vector_factory&& other) noexcept = delete;

    auto* operator()() {
        return &m_vector;
    }

private:
    pcx::vector<T, Allocator> m_vector;
};
}    // namespace internal


/**
 * @brief
 *
 * @tparam T
 * @tparam Allocator
 * @tparam FFTUnit
 * @tparam TMPFactory A type with operator() returning a pointer_t. Dereferencing pointer_t must return pcx::vector<T, ...>&.
   Size of the returned vector must be equal to fft size;
 */
template<typename T,
         typename Allocator   = pcx::aligned_allocator<T>,
         typename FFT_        = pcx::fft_unit<T, pcx::fft_ordering::unordered, Allocator>,
         typename TmpFactory_ = internal::pseudo_vector_factory<T, Allocator>>
    requires std::floating_point<T> && std::same_as<typename Allocator::value_type, T>
class fxcorr_unit {
public:
    using real_type      = T;
    using allocator_type = Allocator;

    fxcorr_unit() = delete;
    fxcorr_unit(pcx::vector<T> g, std::size_t fft_size, allocator_type allocator = allocator_type{})
    : m_fft(std::make_shared<FFT_>(fft_size, 2048, allocator))
    , m_kernel(m_fft->size(), allocator)
    , m_tmp_factory(m_fft->size(), allocator)
    , m_overlap(g.size() - 1) {
        (*m_fft)(m_kernel, g);
        m_kernel = m_kernel / static_cast<T>(m_fft->size());
    };

    fxcorr_unit(pcx::vector<T> g, std::shared_ptr<FFT_> fft_unit, allocator_type allocator = allocator_type{})
    : m_fft(std::move(fft_unit))
    , m_kernel(m_fft->size(), allocator)
    , m_tmp_factory(m_fft->size(), allocator)
    , m_overlap(g.size() - 1) {
        (*m_fft)(m_kernel, g);
        m_kernel = m_kernel / static_cast<T>(m_fft->size());
    };

    fxcorr_unit(pcx::vector<T>        g,
                std::shared_ptr<FFT_> fft_unit,
                TmpFactory_           tmp_factory,
                allocator_type        allocator = allocator_type{})
    : m_fft(fft_unit)
    , m_kernel(m_fft->size(), allocator)
    , m_tmp_factory(tmp_factory)
    , m_overlap(g.size() - 1) {
        (*m_fft)(m_kernel, g);
        m_kernel = m_kernel / static_cast<T>(m_fft->size());
    };

    fxcorr_unit(const fxcorr_unit& other)     = default;
    fxcorr_unit(fxcorr_unit&& other) noexcept = default;

    ~fxcorr_unit() = default;

    fxcorr_unit& operator=(const fxcorr_unit& other)     = delete;
    fxcorr_unit& operator=(fxcorr_unit&& other) noexcept = delete;

    template<typename Alloc_, std::size_t PackSize_>
    void operator()(pcx::vector<T, Alloc_, PackSize_>& vec) {
        uint offset = 0;
        auto step   = (m_fft->size() - m_overlap) / PackSize_ * PackSize_;
        for (; offset + m_fft->size() < vec.size(); offset += step) {
            auto tmp = *m_tmp_factory();
            (*m_fft)(tmp, subrange(vec.begin() + offset, m_fft->size()));
            tmp = tmp * conj(m_kernel);
            (*m_fft).template ifft<false>(tmp);
            subrange(vec.begin() + offset, step).assign(tmp);
        }
        while (offset < vec.size()) {
            auto tmp = *m_tmp_factory();
            (*m_fft)(tmp, subrange(vec.begin() + offset, vec.end()));
            tmp = tmp * conj(m_kernel);
            (*m_fft).template ifft<false>(tmp);
            auto l_step = std::min(step, vec.size() - offset);
            subrange(vec.begin() + offset, l_step).assign(subrange(tmp.begin(), l_step));
            offset += step;
        }
    }

private:
    std::shared_ptr<FFT_>          m_fft;
    pcx::vector<T, allocator_type> m_kernel;
    TmpFactory_                    m_tmp_factory;
    std::size_t                    m_overlap;
};
}    // namespace pcx