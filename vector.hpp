#ifndef PCX_VECTOR_HPP
#define PCX_VECTOR_HPP
#include "vector_arithm.hpp"
#include "vector_util.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

namespace pcx {
/**
 * @brief Vector of complex floating point values.
 * Values are stored in packs. Inside pack continious real values
 * are followed by continious imaginary values.
 *
 * @tparam T real type
 * @tparam PackSize number of complex values in pack default is 32/16 for float/double
 * @tparam Allocator
 */
template<typename T,
         std::size_t PackSize = pcx::default_pack_size<T>,
         typename Allocator   = pcx::aligned_allocator<T>>
    requires packed_floating_point<T, PackSize>
class vector {
    friend class internal::expression_traits;
    template<typename OT, std::size_t OPackSize, typename OAllocator>
        requires packed_floating_point<OT, OPackSize>
    friend class pcx::vector;

private:
    using alloc_traits = std::allocator_traits<Allocator>;

public:
    using real_type    = T;
    using real_pointer = T*;

    using allocator_type  = Allocator;
    using size_type       = std::size_t;
    using difference_type = ptrdiff_t;
    using reference       = cx_ref<T, false, PackSize>;
    using const_reference = const cx_ref<T, true, PackSize>;

    using iterator       = pcx::iterator<T, false, PackSize>;
    using const_iterator = pcx::iterator<T, true, PackSize>;

    static constexpr size_type pack_size = PackSize;

    vector() noexcept(noexcept(allocator_type())) = default;

    explicit vector(const allocator_type& allocator) noexcept(
        std::is_nothrow_copy_constructible_v<allocator_type>)
    : m_allocator(allocator)
    , m_ptr(nullptr){};

    explicit vector(size_type size, const allocator_type& allocator = allocator_type())
    : m_allocator(allocator)
    , m_size(size)
    , m_ptr(alloc_traits::allocate(m_allocator, real_size(m_size))) {
        fill(begin(), end(), 0);
    };

    template<typename U>
        requires std::is_convertible_v<U, std::complex<real_type>>
    vector(size_type size, U value, const allocator_type& allocator = allocator_type())
    : m_allocator(allocator)
    , m_size(size)
    , m_ptr(alloc_traits::allocate(m_allocator, num_packs(size) * PackSize * 2)) {
        fill(begin(), end(), value);
    };

    vector(const vector& other)
    : m_allocator(alloc_traits::select_on_container_copy_construction(other.m_allocator))
    , m_size(other.m_size) {
        if (m_size == 0) {
            return;
        }
        m_ptr = alloc_traits::allocate(m_allocator, real_size(m_size));

        packed_copy(other.begin(), other.end(), begin());
    }

    vector(const vector& other, const allocator_type& allocator)
    : m_allocator(allocator)
    , m_size(other.m_size) {
        if (m_size == 0) {
            return;
        }
        m_ptr = alloc_traits::allocate(m_allocator, real_size(m_size));

        packed_copy(other.begin(), other.end(), begin());
    }

    vector(vector&& other) noexcept(std::is_nothrow_move_constructible_v<allocator_type>)
    : m_allocator(std::move(other.m_allocator)) {
        using std::swap;
        swap(m_size, other.m_size);
        swap(m_ptr, other.m_ptr);
    };

    vector(vector&& other, const allocator_type& allocator) noexcept(alloc_traits::is_always_equal::value)
    : m_allocator(allocator) {
        if constexpr (alloc_traits::is_always_equal::value) {
            using std::swap;
            swap(m_size, other.m_size);
            swap(m_ptr, other.m_ptr);
        } else {
            if (m_allocator == other.m_allocator) {
                using std::swap;
                swap(m_size, other.m_size);
                swap(m_ptr, other.m_ptr);
                return;
            }
            m_ptr = alloc_traits::allocate(m_allocator, real_size(m_size));
            packed_copy(other.begin(), other.end(), begin());
        }
    };

    vector& operator=(const vector& other) {
        if (this == &other) {
            return *this;
        }

        if constexpr (alloc_traits::propagate_on_container_copy_assignment::value &&
                      !alloc_traits::is_always_equal::value) {
            if (m_allocator != other.m_allocator || num_packs(m_size) != num_packs(other.m_size)) {
                deallocate();
                m_allocator = other.m_allocator;
                m_size      = other.m_size;
                m_ptr       = alloc_traits::allocate(m_allocator, real_size(m_size));
            }
        } else {
            if constexpr (alloc_traits::propagate_on_container_copy_assignment::value) {
                m_allocator = other.m_allocator;
            }
            if (num_packs(m_size) != num_packs(other.m_size)) {
                deallocate();
                m_size = other.m_size;
                m_ptr  = alloc_traits::allocate(m_allocator, real_size(m_size));
            }
        }

        packed_copy(other.begin(), other.end(), begin());
        return *this;
    }

    vector& operator=(vector&& other) noexcept(alloc_traits::propagate_on_container_move_assignment::value&&
                                                   std::is_nothrow_swappable_v<allocator_type> ||
                                               alloc_traits::is_always_equal::value) {
        if constexpr (alloc_traits::propagate_on_container_move_assignment::value) {
            deallocate();
            using std::swap;
            swap(m_allocator, other.m_allocator);
            swap(m_size, other.m_size);
            swap(m_ptr, other.m_ptr);
            return *this;
        } else {
            if constexpr (alloc_traits::is_always_equal::value) {
                deallocate();
                using std::swap;
                swap(m_size, other.m_size);
                swap(m_ptr, other.m_ptr);
                return *this;
            } else {
                if (m_allocator == other.m_allocator) {
                    deallocate();
                    using std::swap;
                    swap(m_size, other.m_size);
                    swap(m_ptr, other.m_ptr);
                    return *this;
                }
                if (num_packs(m_size) != num_packs(other.m_size)) {
                    deallocate();
                    m_size = other.m_size;
                    m_ptr  = alloc_traits::allocate(m_allocator, real_size(m_size));
                }
                packed_copy(other.begin(), other.end(), begin());
                return *this;
            }
        }
    }

    template<typename E>
        requires(!std::same_as<E, vector>) && internal::vector_expression<E>
    vector& operator=(const E& other) {
        assert(size() == other.size());

        auto it_this = begin();
        auto it_expr = other.begin();

        if (it_expr.aligned()) {
            constexpr auto reg_size   = 32 / sizeof(T);
            constexpr auto store_size = std::max(reg_size, pack_size);

            auto aligned_size = (end().align_lower() - it_this) / store_size;
            auto ptr          = &(*it_this);
            for (uint i = 0; i < aligned_size; ++i) {
                for (uint i_reg = 0; i_reg < store_size; i_reg += reg_size) {
                    auto offset = i * store_size + i_reg;
                    auto data   = internal::expression_traits::cx_reg<pack_size>(it_expr, offset);
                    avx::cxstore<store_size>(avx::ra_addr<store_size>(ptr, offset), data);
                }
            }
            it_this += aligned_size * store_size;
            it_expr += aligned_size * store_size;
        }
        while (it_this < end()) {
            *it_this = *it_expr;
            ++it_this;
            ++it_expr;
        }
        return *this;
    };

    ~vector() {
        deallocate();
    };

    void swap(vector& other) noexcept(
        alloc_traits::propagate_on_container_swap::value&& std::is_nothrow_swappable_v<allocator_type> ||
        alloc_traits::is_always_equal::value) {
        if constexpr (alloc_traits::propagate_on_container_swap::value) {
            using std::swap;
            swap(m_allocator, other.m_allocator);
            swap(m_size, other.m_size);
            swap(m_ptr, other.m_ptr);
        } else {
            using std::swap;
            swap(m_size, other.m_size);
            swap(m_ptr, other.m_ptr);
        }
    }

    friend void swap(vector& first, vector& second) noexcept(noexcept(first.swap(second))) {
        first.swap(second);
    }

    [[nodiscard]] auto get_allocator() const noexcept(std::is_nothrow_copy_constructible_v<allocator_type>)
        -> allocator_type {
        return m_allocator;
    }

    [[nodiscard]] auto operator[](size_type idx) -> reference {
        auto p_idx = packed_idx(idx);
        return reference(m_ptr + p_idx);
    }
    // NOLINTNEXLINE (*const*) proxy reference
    [[nodiscard]] auto operator[](size_type idx) const -> const_reference {
        auto p_idx = packed_idx(idx);
        return reference(m_ptr + p_idx);
    }

    [[nodiscard]] auto at(size_type idx) -> reference {
        if (idx >= size()) {
            throw std::out_of_range(std::string("idx (which is") + std::to_string(idx) +
                                    std::string(") >= vector size (which is ") + std::to_string(size()) +
                                    std::string(")"));
        }
        auto p_idx = packed_idx(idx);
        return {m_ptr + p_idx};
    }
    [[nodiscard]] auto at(size_type idx) const -> const_reference {
        if (idx >= size()) {
            throw std::out_of_range(std::string("idx (which is") + std::to_string(idx) +
                                    std::string(") >= vector size (which is ") + std::to_string(size()) +
                                    std::string(")"));
        }
        auto p_idx = packed_idx(idx);
        return {m_ptr + p_idx};
    }

    [[nodiscard]] auto data() -> real_pointer {
        return m_ptr;
    }
    [[nodiscard]] auto data() const -> const real_pointer {
        return m_ptr;
    }

    [[nodiscard]] constexpr auto size() const noexcept -> size_type {
        return m_size;
    };
    [[nodiscard]] constexpr auto length() const noexcept -> size_type {
        return size();
    };
    void resize(size_type new_size) {
        if (num_packs(new_size) != num_packs(m_size)) {
            real_type* new_ptr = alloc_traits::allocate(m_allocator, real_size(new_size));

            packed_copy(begin(), end(), iterator(new_ptr, 0));
            deallocate();

            m_size = new_size;
            m_ptr  = new_ptr;
        }
    }

    [[nodiscard]] auto begin() noexcept -> iterator {
        return iterator(m_ptr, 0);
    }
    [[nodiscard]] auto begin() const noexcept -> const_iterator {
        return const_iterator(m_ptr, 0);
    }
    [[nodiscard]] auto cbegin() const noexcept -> const_iterator {
        return const_iterator(m_ptr, 0);
    }

    [[nodiscard]] auto end() noexcept -> iterator {
        return iterator(m_ptr + packed_idx(size()), size());
    }
    [[nodiscard]] auto end() const noexcept -> const_iterator {
        return const_iterator(m_ptr + packed_idx(size()), size());
    }
    [[nodiscard]] auto cend() const noexcept -> const_iterator {
        return const_iterator(m_ptr + packed_idx(size()), size());
    }

    template<typename SAllocator>
    explicit operator std::vector<std::complex<T>, SAllocator>() const {
        auto it_this      = begin();
        auto aligned_size = end().align_lower() - it_this;

        auto svec = std::vector<std::complex<T>, SAllocator>();
        svec.reserve(m_size);
        svec.resize(aligned_size);
        auto* svec_ptr = reinterpret_cast<T*>(svec.data());

        auto ptr = &(*it_this);
        for (uint i = 0; i < aligned_size; i += pack_size) {
            for (uint i_reg = 0; i_reg < pack_size; i_reg += avx::reg<real_type>::size) {
                auto data = avx::cxload<pack_size>(m_ptr, i + i_reg);
                avx::cxstore<avx::reg<real_type>::size>(avx::ra_addr<1>(svec_ptr, i + i_reg), data);
            }
        }
        it_this += aligned_size;

        while (it_this < end()) {
            svec.push_back(it_this.vale());
            ++it_this;
        }
    }

    template<std::size_t OPackSize>
    explicit vector(const vector<real_type, OPackSize, allocator_type>& other)
    : m_allocator(alloc_traits::select_on_container_copy_construction(other.m_allocator))
    , m_size(other.m_size) {
        if (m_size == 0) {
            return;
        }
        m_ptr = alloc_traits::allocate(m_allocator, real_size(m_size));
        *this = pcx::subrange(other.begin(), other.end());
    }

    template<typename OAllocator, std::size_t OPackSize>
    explicit vector(const vector<T, OPackSize, OAllocator>& other, const OAllocator& allocator = OAllocator{})
    : m_allocator(allocator)
    , m_size(other.m_size) {
        if (m_size == 0) {
            return;
        }
        m_ptr = alloc_traits::allocate(m_allocator, real_size(m_size));
        *this = pcx::subrange(other.begin(), other.end());
    }

private:
    [[no_unique_address]] allocator_type m_allocator{};

    size_type    m_size = 0;
    real_pointer m_ptr{};

    void deallocate() {
        if (m_ptr != real_pointer() && size() != 0) {
            alloc_traits::deallocate(m_allocator, m_ptr, num_packs(size()) * PackSize * 2);
            m_ptr  = real_pointer();
            m_size = 0;
        }
    }

    static constexpr auto num_packs(size_type vector_size) -> size_type {
        return (vector_size % PackSize > 0) ? vector_size / PackSize + 1 : vector_size / PackSize;
    }
    static constexpr auto real_size(size_type vector_size) -> size_type {
        return num_packs(vector_size) * PackSize * 2;
    }
    static constexpr auto packed_idx(size_type idx) -> size_type {
        return idx + idx / PackSize * PackSize;
    }
};

template<typename T, bool Const = false, std::size_t PackSize = pcx::default_pack_size<T>>
class iterator {
    template<typename VT, std::size_t VPackSize, typename>
        requires packed_floating_point<VT, VPackSize>
    friend class vector;
    friend class iterator<T, true, PackSize>;

public:
    using real_type       = T;
    using real_pointer    = T*;
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
    template<typename VT, std::size_t VPackSize, typename>
        requires packed_floating_point<VT, VPackSize>
    friend class vector;
    friend class iterator<T, true, 1>;

public:
    using real_type       = T;
    using real_pointer    = T*;
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

template<typename T, bool Const = false, std::size_t PackSize = pcx::default_pack_size<T>>
class subrange : public std::ranges::view_base {
    template<typename VT, std::size_t VPackSize, typename>
        requires packed_floating_point<VT, VPackSize>
    friend class vector;

    friend class internal::expression_traits;

public:
    using real_type       = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    using iterator       = pcx::iterator<T, Const, PackSize>;
    using const_iterator = pcx::iterator<T, true, PackSize>;

    using reference = typename iterator::reference;

    static constexpr size_type pack_size = PackSize;

    subrange() noexcept = default;

    subrange(const iterator& begin, size_type size) noexcept
    : m_begin(begin)
    , m_size(size){};

    subrange(const iterator& begin, const iterator& end) noexcept
    : m_begin(begin)
    , m_size(end - begin){};

    template<typename VAllocator>
    explicit subrange(vector<real_type, pack_size, VAllocator>& vector) noexcept
    : m_begin(vector.begin())
    , m_size(vector.size()){};

    template<typename VAllocator>
    explicit subrange(const vector<real_type, pack_size, VAllocator>& vector) noexcept
        requires Const
    : m_begin(vector.begin())
    , m_size(vector.size()){};

    subrange(const subrange&) noexcept = default;
    subrange(subrange&&) noexcept      = default;

    ~subrange() noexcept = default;

    subrange& operator=(subrange&&) noexcept      = default;
    subrange& operator=(const subrange&) noexcept = default;

    void swap(subrange& other) noexcept {
        using std::swap;
        swap(m_begin, other.m_begin);
        swap(m_size, other.m_size);
    }
    friend void swap(subrange& first, subrange& second) noexcept {
        first.swap(second);
    }

    [[nodiscard]] auto begin() const -> iterator {
        return m_begin;
    }
    [[nodiscard]] auto cbegin() const -> const_iterator {
        return m_begin;
    }
    [[nodiscard]] auto end() const -> iterator {
        return m_begin + size();
    }
    [[nodiscard]] auto cend() const -> const_iterator {
        return m_begin + size();
    }

    [[nodiscard]] auto operator[](difference_type idx) const -> reference {
        return *(m_begin + idx);
    }

    [[nodiscard]] constexpr auto size() const -> size_type {
        return m_size;
    };
    [[nodiscard]] bool aligned() const {
        return m_begin.aligned();
    }
    [[nodiscard]] constexpr bool empty() const {
        return size() == 0;
    }
    // NOLINTNEXTLINE (*explicit*)
    [[nodiscard]] operator bool() const {
        return size() != 0;
    }

    template<typename U>
        requires std::convertible_to<U, std::complex<T>>
    void fill(U value) {
        pcx::fill(begin(), end(), value);
    };

    template<typename R>
        requires(!Const) && (!internal::vector_expression<R>) && std::ranges::input_range<R> &&
                std::indirectly_copyable<std::ranges::iterator_t<R>, iterator>
    void assign(const R& range) {
        std::ranges::copy(range, begin());
    };

    template<typename E>
        requires(!Const) && internal::vector_expression<E>
    void assign(const E& expression) {
        assert(size() == expression.size());

        auto it_this       = begin();
        auto it_expr       = expression.begin();
        auto aligned_begin = std::min(it_this.align_upper(), end());
        while (it_this < aligned_begin) {
            *it_this = *it_expr;
            ++it_this;
            ++it_expr;
        }

        if (it_this.aligned() && it_expr.aligned()) {
            constexpr auto reg_size   = 32 / sizeof(T);
            constexpr auto store_size = std::max(reg_size, pack_size);

            auto aligned_size = (end().align_lower() - it_this) / store_size;
            auto ptr          = &(*it_this);
            for (uint i = 0; i < aligned_size; ++i) {
                for (uint i_reg = 0; i_reg < store_size; i_reg += reg_size) {
                    auto offset = i * store_size + i_reg;
                    auto data   = internal::expression_traits::cx_reg<pack_size>(it_expr, offset);
                    avx::cxstore<store_size>(avx::ra_addr<store_size>(ptr, offset), data);
                }
            }
            it_this += aligned_size * store_size;
            it_expr += aligned_size * store_size;
        }
        while (it_this < end()) {
            *it_this = *it_expr;
            ++it_this;
            ++it_expr;
        }
    };

private:
    size_type m_size{};
    iterator  m_begin{};
};

template<typename T, bool Const = false, std::size_t PackSize = pcx::default_pack_size<T>>
class cx_ref {
    template<typename VT, std::size_t VPackSize, typename>
        requires packed_floating_point<VT, VPackSize>
    friend class vector;

    friend class iterator<T, Const, PackSize>;
    friend class iterator<T, false, PackSize>;

    template<typename ST, bool SConst, std::size_t SPackSize>
    friend class subrange;

    friend class cx_ref<T, true, PackSize>;

public:
    using real_type  = T;
    using pointer    = typename std::conditional<Const, const T*, T*>::type;
    using value_type = std::complex<real_type>;

private:
    explicit cx_ref(pointer ptr) noexcept
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
    [[nodiscard]] auto operator&() const noexcept -> pointer {
        return m_ptr;
    }

private:
    pointer m_ptr{};
};
}    // namespace pcx
#endif