#ifndef ELEMENT_ACCESS_HPP
#define ELEMENT_ACCESS_HPP

#include "types.hpp"

namespace pcx {

template<uZ PackSize>
constexpr auto pidx(uZ idx) -> uZ {
    return idx + idx / PackSize * PackSize;
}

namespace detail_ {
template<typename T, bool Const, uZ PackSize>
auto make_iterator(auto* data_ptr, iZ index) noexcept {
    return iterator<T, Const, PackSize>(data_ptr, index);
};
}    // namespace detail_

template<typename T, bool Const = false, uZ PackSize = pcx::default_pack_size<T>>
class iterator {
    template<typename VT, uZ VPackSize, typename>
        requires packed_floating_point<VT, VPackSize>
    friend class vector;

    friend class iterator<T, true, PackSize>;

    friend auto detail_::make_iterator<T, Const, PackSize>(T* ptr, iZ index) noexcept;
    friend auto detail_::make_iterator<T, Const, PackSize>(const T* ptr, iZ index) noexcept;

public:
    using real_type       = T;
    using real_pointer    = std::conditional_t<Const, const T*, T*>;
    using difference_type = ptrdiff_t;
    using reference =
        typename std::conditional_t<Const, const cx_ref<T, true, PackSize>, cx_ref<T, false, PackSize>>;
    using value_type       = cx_ref<T, Const, PackSize>;
    using iterator_concept = std::random_access_iterator_tag;

    static constexpr auto pack_size = PackSize;

private:
    iterator(real_pointer data_ptr, difference_type index) noexcept
    : m_ptr(data_ptr)
    , m_idx(index){};

public:
    iterator() noexcept = default;

    // NOLINTNEXTLINE(*explicit*)
    iterator(const iterator<T, false, PackSize>& other) noexcept
        requires Const
    : m_ptr(other.m_ptr)
    , m_idx(other.m_idx){};

    iterator(const iterator& other) noexcept = default;
    iterator(iterator&& other) noexcept      = default;

    iterator& operator=(const iterator& other) noexcept = default;
    iterator& operator=(iterator&& other) noexcept      = default;

    ~iterator() noexcept = default;

    [[nodiscard]] reference operator*() const {
        return reference(m_ptr);
    }
    [[nodiscard]] reference operator[](difference_type idx) const noexcept {
        return *(*this + idx);
    }

    [[nodiscard]] bool operator==(const iterator& other) const noexcept {
        return (m_ptr == other.m_ptr) && (m_idx == other.m_idx);
    }
    [[nodiscard]] auto operator<=>(const iterator& other) const noexcept {
        return m_ptr <=> other.m_ptr;
    }

    iterator& operator++() noexcept {
        if (++m_idx % PackSize == 0) {
            m_ptr += PackSize;
        }
        ++m_ptr;
        return *this;
    }
    iterator operator++(int) noexcept {
        auto copy = *this;
        ++*this;
        return copy;
    }
    iterator& operator--() noexcept {
        if (m_idx % PackSize == 0) {
            m_ptr -= PackSize;
        };
        --m_idx;
        --m_ptr;
        return *this;
    }
    iterator operator--(int) noexcept {
        auto copy = *this;
        --*this;
        return copy;
    }

    iterator& operator+=(difference_type n) noexcept {
        m_ptr = m_ptr + n + (m_idx % PackSize + n) / PackSize * PackSize;
        m_idx += n;
        return *this;
    }
    iterator& operator-=(difference_type n) noexcept {
        return (*this) += -n;
    }

    [[nodiscard]] friend iterator operator+(iterator it, difference_type n) noexcept {
        it += n;
        return it;
    }
    [[nodiscard]] friend iterator operator+(difference_type n, iterator it) noexcept {
        it += n;
        return it;
    }
    [[nodiscard]] friend iterator operator-(iterator it, difference_type n) noexcept {
        it -= n;
        return it;
    }

    template<bool OConst>
    [[nodiscard]] friend auto operator-(const iterator&                      lhs,
                                        const iterator<T, OConst, PackSize>& rhs) noexcept
        -> difference_type {
        return lhs.m_idx - rhs.m_idx;
    }

    [[nodiscard]] bool aligned(difference_type idx = 0) const noexcept {
        return (m_idx + idx) % PackSize == 0;
    }
    /**
     * @brief Returns aligned iterator not bigger then this itertor;
     *
     * @return iterator
     */
    [[nodiscard]] auto align_lower() const noexcept -> iterator {
        return *this - m_idx % PackSize;
    }
    /**
     * @brief Return aligned iterator not smaller then this itertator;
     *
     * @return iterator
     */
    [[nodiscard]] auto align_upper() const noexcept -> iterator {
        return m_idx % PackSize == 0 ? *this : *this + PackSize - m_idx % PackSize;
    }

private:
    real_pointer    m_ptr{};
    difference_type m_idx = 0;
};

template<typename T, bool Const>
class iterator<T, Const, 1> {
    template<typename VT, uZ VPackSize, typename>
        requires packed_floating_point<VT, VPackSize>
    friend class vector;
    friend class iterator<T, true, 1>;

    friend auto detail_::make_iterator<T, Const, 1>(T* ptr, iZ index) noexcept;
    friend auto detail_::make_iterator<T, Const, 1>(const T* ptr, iZ index) noexcept;

public:
    using real_type       = T;
    using real_pointer    = std::conditional_t<Const, const T*, T*>;
    using difference_type = ptrdiff_t;
    using reference       = typename std::conditional_t<Const, const cx_ref<T, true, 1>, cx_ref<T, false, 1>>;
    using value_type      = cx_ref<T, Const, 1>;
    using iterator_concept = std::random_access_iterator_tag;

    static constexpr auto pack_size = 1;

private:
    explicit iterator(real_pointer data_ptr, difference_type) noexcept
    : m_ptr(data_ptr){};

public:
    iterator() noexcept = default;

    // NOLINTNEXTLINE(*explicit*)
    iterator(const iterator<T, false, 1>& other) noexcept
        requires Const
    : m_ptr(other.m_ptr){};

    iterator(const iterator& other) noexcept = default;
    iterator(iterator&& other) noexcept      = default;

    iterator& operator=(const iterator& other) noexcept = default;
    iterator& operator=(iterator&& other) noexcept      = default;

    ~iterator() noexcept = default;

    [[nodiscard]] reference operator*() const {
        return reference(m_ptr);
    }
    [[nodiscard]] reference operator[](difference_type idx) const noexcept {
        return *(*this + idx);
    }

    [[nodiscard]] bool operator==(const iterator& other) const noexcept {
        return (m_ptr == other.m_ptr);
    }
    [[nodiscard]] auto operator<=>(const iterator& other) const noexcept {
        return m_ptr <=> other.m_ptr;
    }

    iterator& operator++() noexcept {
        m_ptr += 2;
        return *this;
    }
    iterator operator++(int) noexcept {
        auto copy = *this;
        ++*this;
        return copy;
    }
    iterator& operator--() noexcept {
        m_ptr -= 2;
        return *this;
    }
    iterator operator--(int) noexcept {
        auto copy = *this;
        --*this;
        return copy;
    }

    iterator& operator+=(difference_type n) noexcept {
        m_ptr = m_ptr + n * 2;
        return *this;
    }
    iterator& operator-=(difference_type n) noexcept {
        return (*this) += -n;
    }

    [[nodiscard]] friend iterator operator+(iterator it, difference_type n) noexcept {
        it += n;
        return it;
    }
    [[nodiscard]] friend iterator operator+(difference_type n, iterator it) noexcept {
        it += n;
        return it;
    }
    [[nodiscard]] friend iterator operator-(iterator it, difference_type n) noexcept {
        it -= n;
        return it;
    }

    template<bool OConst>
    [[nodiscard]] friend auto operator-(const iterator& lhs, const iterator<T, OConst, 1>& rhs) noexcept
        -> difference_type {
        return (lhs.m_ptr - rhs.m_ptr) >> 1U;
    }

    [[nodiscard]] bool aligned(difference_type = 0) const noexcept {
        return true;
    }
    /**
     * @brief Returns aligned iterator not bigger then this itertor;
     *
     * @return iterator
     */
    [[nodiscard]] auto align_lower() const noexcept -> iterator {
        return *this;
    }
    /**
     * @brief Return aligned iterator not smaller then this itertator;
     *
     * @return iterator
     */
    [[nodiscard]] auto align_upper() const noexcept -> iterator {
        return *this;
    }

private:
    real_pointer m_ptr{};
};

namespace detail_ {
template<typename T>
struct is_pcx_iterator {
    static constexpr bool value = false;
};

template<typename T, bool Const, uZ PackSize>
struct is_pcx_iterator<iterator<T, Const, PackSize>> {
    static constexpr bool value = true;
};
}    // namespace detail_


namespace detail_ {
template<typename T, bool Const, uZ PackSize>
[[nodiscard]] auto make_cx_ref(auto* ptr) {
    return cx_ref<T, Const, PackSize>(ptr);
}
}    // namespace detail_

template<typename T, bool Const = false, uZ PackSize = pcx::default_pack_size<T>>
class cx_ref {
    template<typename VT, uZ VPackSize, typename>
        requires packed_floating_point<VT, VPackSize>
    friend class vector;

    friend class iterator<T, Const, PackSize>;
    friend class iterator<T, false, PackSize>;

    template<typename ST, bool SConst, uZ SPackSize>
    friend class subrange;

    friend class cx_ref<T, true, PackSize>;

    friend auto detail_::make_cx_ref<T, Const, PackSize>(T* ptr);
    friend auto detail_::make_cx_ref<T, Const, PackSize>(const T* ptr);

public:
    using real_type    = T;
    using real_pointer = std::conditional_t<Const, const T*, T*>;
    using value_type   = std::complex<real_type>;

private:
    explicit cx_ref(real_pointer ptr) noexcept
    : m_ptr(ptr){};

public:
    cx_ref() = delete;

    // NOLINTNEXTLINE (*explicit*)
    cx_ref(const cx_ref<T, false, PackSize>& other) noexcept
        requires Const
    : m_ptr(other.m_ptr){};

    cx_ref(const cx_ref&) noexcept = default;
    cx_ref(cx_ref&&) noexcept      = default;

    ~cx_ref() = default;

    cx_ref& operator=(const cx_ref& other)
        requires(!Const)
    {
        *m_ptr              = *other.m_ptr;
        *(m_ptr + PackSize) = *(other.m_ptr + PackSize);
        return *this;
    }
    cx_ref& operator=(cx_ref&& other) noexcept
        requires(!Const)
    {
        *m_ptr              = *other.m_ptr;
        *(m_ptr + PackSize) = *(other.m_ptr + PackSize);
        return *this;
    }
    // NOLINTNEXTLINE (*assign*) Proxy reference, see std::indirectly_writable
    const cx_ref& operator=(const cx_ref& other) const
        requires(!Const)
    {
        *m_ptr              = *other.m_ptr;
        *(m_ptr + PackSize) = *(other.m_ptr + PackSize);
        return *this;
    }
    // NOLINTNEXTLINE (*assign*) Proxy reference, see std::indirectly_writable
    const cx_ref& operator=(cx_ref&& other) const
        requires(!Const)
    {
        *m_ptr              = *other.m_ptr;
        *(m_ptr + PackSize) = *(other.m_ptr + PackSize);
        return *this;
    }

    template<typename U>
        requires std::is_convertible_v<U, value_type>
    auto& operator=(const U& other) const
        requires(!Const)
    {
        auto tmp            = static_cast<value_type>(other);
        *m_ptr              = tmp.real();
        *(m_ptr + PackSize) = tmp.imag();
        return *this;
    }
    // NOLINTNEXTLINE(*explicit*)
    [[nodiscard]] operator value_type() const {
        return value_type(*m_ptr, *(m_ptr + PackSize));
    }
    [[nodiscard]] value_type value() const {
        return *this;
    }
    // NOLINTNEXTLINE(google-runtime-operator)
    [[nodiscard]] auto operator&() const noexcept -> real_pointer {
        return m_ptr;
    }

private:
    real_pointer m_ptr{};
};

namespace detail_ {
template<bool Always>
struct pack_aligned_base {
    using always_aligned = std::false_type;
};
template<>
struct pack_aligned_base<true> {
    using always_aligned = std::true_type;
};
}    // namespace detail_

template<typename R>
    requires /**/ (rv::range<R> && (detail_::is_pcx_iterator<rv::iterator_t<R>>::value))
struct cx_vector_traits<R> {
    using real_type                                 = typename rv::iterator_t<R>::real_type;
    static constexpr uZ   pack_size                 = rv::iterator_t<R>::pack_size;
    static constexpr bool enable_vector_expressions = true;

    static constexpr bool always_aligned = pack_size == 1    //
                                           || std::derived_from<R, detail_::pack_aligned_base<true>>;

    using iterator_t       = rv::iterator_t<R>;
    using const_iterator_t = decltype(rv::cbegin(std::declval<R&>()));

    inline static auto re_data(R& vector) {
        auto it = vector.begin();
        if constexpr (!always_aligned) {
            // TODO: no exception verison
            if (!it.aligned()) {
                throw(
                    std::invalid_argument("Packed complex range is not aligned. Packed complex range must be "
                                          "aligned to be accessed as a vector."));
            }
        }
        return &(*it);
    }

    inline static auto re_data(const R& vector) {
        auto it = vector.begin();
        if constexpr (!always_aligned) {
            // TODO: no exception verison
            if (!it.aligned()) {
                throw(
                    std::invalid_argument("Packed complex range is not aligned. Packed complex range must be "
                                          "aligned to be accessed as a vector."));
            }
        }
        return &(*it);
    }

    inline static auto size(const R& vector) {
        return rv::size(vector);
    }

    inline static constexpr auto aligned(const R& /*vector*/) -> bool
        requires(always_aligned)
    {
        return true;
    }

    inline static auto aligned(const R& vector) -> bool
        requires(!always_aligned)
    {
        if constexpr (requires(const R& v) {
                          { v.aligned() } -> std::same_as<bool>;
                      }) {
            return vector.aligned();
        } else {
            return vector.begin().aligned();
        }
    }

    static constexpr auto aligned(const iterator_t& iterator) {
        return iterator.aligned();
    }

    static constexpr auto aligned(const const_iterator_t& iterator)
        requires(!std::same_as<iterator_t, const_iterator_t>)
    {
        return iterator.aligned();
    }
};
}    // namespace pcx
#endif
