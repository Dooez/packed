#include "fft.hpp"
#include "vector_util.hpp"

#include <concepts>
#include <cstddef>
#include <memory>

namespace pcx {

namespace detail_ {
template<typename T, typename Allocator = pcx::aligned_allocator<T>>
class pseudo_vector_factory {
public:
    explicit pseudo_vector_factory(uZ size, const Allocator& allocator = Allocator{})
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
    pcx::vector<T, default_pack_size<T>, Allocator> m_vector;
};
}    // namespace detail_


/**
 * @brief
 *
 * @tparam T
 * @tparam Allocator
 * @tparam FFT_
 * @tparam TMPFactory_ A type with operator() returning a pointer_t. Dereferencing pointer_t must return pcx::vector<T, ...>&.
   Size of the returned vector must be equal to fft size;
 */
template<typename T,
         typename Allocator   = pcx::aligned_allocator<T>,
         typename FFT_        = pcx::fft_unit<T, pcx::fft_order::unordered, Allocator>,
         typename TmpFactory_ = detail_::pseudo_vector_factory<T, Allocator>>
    requires (std::floating_point<T> && std::same_as<typename Allocator::value_type, T>)
class fxcorr_unit {
public:
    using real_type      = T;
    using allocator_type = Allocator;

    fxcorr_unit() = delete;
    /**
     * @brief Construct a new fxcorr unit object that performs cross-correlation with g function.
     *
     * @param[in] g Base function to perform correlation with.
     * @param[in] fft_size
     * @param[in] allocator
     */
    fxcorr_unit(pcx::vector<T> g, uZ fft_size, allocator_type allocator = allocator_type{})
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

    template<typename Vect_>
        requires complex_vector_of<T, Vect_>
    void operator()(Vect_& vector) {
        constexpr auto src_pack_size = cx_vector_traits<Vect_>::pack_size;
        auto           tmp           = m_tmp_factory();
        constexpr auto tmp_pack_size =
            cx_vector_traits<std::remove_reference_t<decltype(*tmp)>>::pack_size;

        uZ   offset = 0;
        auto step   = (m_fft->size() - m_overlap) / src_pack_size * src_pack_size;
        for (; offset + m_fft->size() < vector.size(); offset += step) {
            m_fft->template fft_raw<tmp_pack_size, src_pack_size>(tmp->data(), vector.data() + offset);
            *tmp = *tmp * conj(m_kernel);
            if constexpr (tmp_pack_size >= src_pack_size) {
                m_fft->template ifft_raw<false, src_pack_size, tmp_pack_size>(tmp->data(), tmp->data());
                std::memcpy(vector.data() + offset, tmp->data(), sizeof(T) * 2 * step);
            } else {
                m_fft->template ifft_raw<false, tmp_pack_size>(tmp->data());
                rv::copy(pcx::subrange(tmp->begin(), step), vector.begin() + offset);
            }
        }
        while (offset < vector.size()) {
            m_fft->template fft_raw<tmp_pack_size, src_pack_size>(
                tmp->data(), vector.data() + offset, vector.size() - offset);
            *tmp        = *tmp * conj(m_kernel);
            auto l_step = std::min(step, vector.size() - offset);
            if constexpr (tmp_pack_size >= src_pack_size) {
                m_fft->template ifft_raw<false, src_pack_size, tmp_pack_size>(tmp->data(), tmp->data());
                std::memcpy(vector.data() + offset, tmp->data(), sizeof(T) * 2 * l_step);
            } else {
                m_fft->template ifft_raw<false, tmp_pack_size>(tmp->data());
                rv::copy(pcx::subrange(tmp->begin(), l_step), vector.begin() + offset);
            }
            offset += l_step;
        }
    };

    template<typename DestVect_, typename SrcVect_>
        requires complex_vector_of<T, DestVect_> && complex_vector_of<T, SrcVect_>
    void operator()(DestVect_& dest, const SrcVect_& source) {}

private:
    std::shared_ptr<FFT_>                                m_fft;
    pcx::vector<T, default_pack_size<T>, allocator_type> m_kernel;
    TmpFactory_                                          m_tmp_factory;
    uZ                                                   m_overlap;
};
}    // namespace pcx