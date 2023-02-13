#include "vector_arithm.hpp"

#include <complex>
// #include <iostream>
#include <memory>
#include <vector>
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
         std::size_t PackSize = 128 / sizeof(T),
         typename Allocator   = std::allocator<T>>
    requires packed_floating_point<T, PackSize>
class packed_cx_vector
{
    friend class internal::expression_traits;

private:
    using alloc_traits = std::allocator_traits<Allocator>;

public:
    using real_type    = T;
    using real_pointer = T*;

    using allocator_type  = Allocator;
    using size_type       = std::size_t;
    using difference_type = ptrdiff_t;
    using reference       = packed_cx_ref<T, PackSize, false>;
    using const_reference = const packed_cx_ref<T, PackSize, true>;

    using iterator       = packed_iterator<T, PackSize, false>;
    using const_iterator = packed_iterator<T, PackSize, true>;

    static constexpr size_type pack_size = PackSize;

    packed_cx_vector() noexcept(noexcept(allocator_type())) = default;

    explicit packed_cx_vector(const allocator_type& allocator) noexcept(
        std::is_nothrow_copy_constructible_v<allocator_type>)
    : m_allocator(allocator)
    , m_size(0)
    , m_ptr(nullptr){};

    explicit packed_cx_vector(size_type             size,
                              const allocator_type& allocator = allocator_type())
    : m_allocator(allocator)
    , m_size(size)
    , m_ptr(alloc_traits::allocate(m_allocator, real_size(m_size)))
    {
        fill(begin(), end(), 0);
    };

    template<typename U>
        requires std::is_convertible_v<U, std::complex<real_type>>
    packed_cx_vector(size_type             length,
                     U                     value,
                     const allocator_type& allocator = allocator_type())
    : m_allocator(allocator)
    , m_size(length)
    , m_ptr(alloc_traits::allocate(m_allocator, num_packs(length) * PackSize * 2))
    {
        fill(begin(), end(), value);
    };

    packed_cx_vector(const packed_cx_vector& other)
    : m_allocator(alloc_traits::select_on_container_copy_construction(other.m_allocator))
    , m_size(other.m_size)
    {
        if (m_size == 0)
        {
            return;
        }
        m_ptr = alloc_traits::allocate(m_allocator, real_size(m_size));

        packed_copy(other.begin(), other.end(), begin());
    }

    packed_cx_vector(const packed_cx_vector& other, const allocator_type& allocator)
    : m_allocator(allocator)
    , m_size(other.m_size)
    {
        if (m_size == 0)
        {
            return;
        }
        m_ptr = alloc_traits::allocate(m_allocator, real_size(m_size));

        packed_copy(other.begin(), other.end(), begin());
    }

    packed_cx_vector(packed_cx_vector&& other) noexcept(
        std::is_nothrow_move_constructible_v<allocator_type>)
    : m_allocator(std::move(other.m_allocator))
    {
        using std::swap;
        swap(m_size, other.m_size);
        swap(m_ptr, other.m_ptr);
    };

    packed_cx_vector(packed_cx_vector&& other, const allocator_type& allocator) noexcept(
        alloc_traits::is_always_equal::value)
    : m_allocator(allocator)
    {
        if constexpr (alloc_traits::is_always_equal::value)
        {
            using std::swap;
            swap(m_size, other.m_size);
            swap(m_ptr, other.m_ptr);
        } else
        {
            if (m_allocator == other.m_allocator)
            {
                using std::swap;
                swap(m_size, other.m_size);
                swap(m_ptr, other.m_ptr);
                return;
            }
            m_ptr = alloc_traits::allocate(m_allocator, real_size(m_size));
            packed_copy(other.begin(), other.end(), begin());
        }
    };

    packed_cx_vector& operator=(const packed_cx_vector& other)
    {
        if (this == &other)
        {
            return *this;
        }

        if constexpr (alloc_traits::propagate_on_container_copy_assignment::value &&
                      !alloc_traits::is_always_equal::value)
        {
            if (m_allocator != other.m_allocator ||
                num_packs(m_size) != num_packs(other.m_size))
            {
                deallocate();
                m_allocator = other.m_allocator;
                m_size      = other.m_size;
                m_ptr       = alloc_traits::allocate(m_allocator, real_size(m_size));
            }
        } else
        {
            if constexpr (alloc_traits::propagate_on_container_copy_assignment::value)
            {
                m_allocator = other.m_allocator;
            }
            if (num_packs(m_size) != num_packs(other.m_size))
            {
                deallocate();
                m_size = other.m_size;
                m_ptr  = alloc_traits::allocate(m_allocator, real_size(m_size));
            }
        }

        packed_copy(other.begin(), other.end(), begin());
        return *this;
    }

    packed_cx_vector& operator=(packed_cx_vector&& other) noexcept(
        alloc_traits::propagate_on_container_move_assignment::value&&
            std::is_nothrow_swappable_v<allocator_type> ||
        alloc_traits::is_always_equal::value)
    {
        if constexpr (alloc_traits::propagate_on_container_move_assignment::value)
        {
            deallocate();
            using std::swap;
            swap(m_allocator, other.m_allocator);
            swap(m_size, other.m_size);
            swap(m_ptr, other.m_ptr);
            return *this;
        } else
        {
            if constexpr (alloc_traits::is_always_equal::value)
            {
                deallocate();
                using std::swap;
                swap(m_size, other.m_size);
                swap(m_ptr, other.m_ptr);
                return *this;
            } else
            {
                if (m_allocator == other.m_allocator)
                {
                    deallocate();
                    using std::swap;
                    swap(m_size, other.m_size);
                    swap(m_ptr, other.m_ptr);
                    return *this;
                }
                if (num_packs(m_size) != num_packs(other.m_size))
                {
                    deallocate();
                    m_size = other.m_size;
                    m_ptr  = alloc_traits::allocate(m_allocator, real_size(m_size));
                }
                packed_copy(other.begin(), other.end(), begin());
                return *this;
            }
        }
    }

    template<typename E>    // clang preferes this overload to normal copy assignment
        requires(!std::same_as<E, packed_cx_vector>) && internal::vector_expression<E>
    packed_cx_vector& operator=(const E& other)
    {
        assert(size() == other.size());

        auto it_this  = begin();
        auto it_other = other.begin();

        if (it_other.aligned())
        {
            constexpr auto reg_size = 32 / sizeof(T);

            auto aligned_size = end().align_lower() - it_this;

            auto ptr = &(*it_this);
            for (long i = 0; i < aligned_size; i += pack_size)
            {
                auto offset = i * 2;
                for (uint i_reg = 0; i_reg < pack_size; i_reg += reg_size)
                {
                    auto data =
                        internal::expression_traits::cx_reg(it_other, offset + i_reg);

                    avx::store(ptr + offset + i_reg, data.real);
                    avx::store(ptr + offset + i_reg + PackSize, data.imag);
                }
            }
            it_this += aligned_size;
            it_other += aligned_size;
        }
        while (it_this < end())
        {
            *it_this = *it_other;
            ++it_this;
            ++it_other;
        }
        return *this;
    };

    ~packed_cx_vector()
    {
        deallocate();
    };

    void swap(packed_cx_vector& other) noexcept(
        alloc_traits::propagate_on_container_swap::value&&
            std::is_nothrow_swappable_v<allocator_type> ||
        alloc_traits::is_always_equal::value)
    {
        if constexpr (alloc_traits::propagate_on_container_swap::value)
        {
            using std::swap;
            swap(m_allocator, other.m_allocator);
            swap(m_size, other.m_size);
            swap(m_ptr, other.m_ptr);
        } else
        {
            using std::swap;
            swap(m_size, other.m_size);
            swap(m_ptr, other.m_ptr);
        }
    }

    friend void swap(packed_cx_vector& first,
                     packed_cx_vector& second) noexcept(noexcept(first.swap(second)))
    {
        first.swap(second);
    }

    [[nodiscard]] auto get_allocator() const
        noexcept(std::is_nothrow_copy_constructible_v<allocator_type>) -> allocator_type
    {
        return m_allocator;
    }

    [[nodiscard]] auto operator[](size_type idx) -> reference
    {
        auto p_idx = packed_idx(idx);
        return reference(m_ptr + p_idx);
    }
    // NOLINTNEXLINE (*const*) proxy reference
    [[nodiscard]] auto operator[](size_type idx) const -> const_reference
    {
        auto p_idx = packed_idx(idx);
        return reference(m_ptr + p_idx);
    }

    [[nodiscard]] auto at(size_type idx) -> reference
    {
        if (idx >= size())
        {
            throw std::out_of_range(std::string("idx (which is") + std::to_string(idx) +
                                    std::string(") >= vector size (which is ") +
                                    std::to_string(size()) + std::string(")"));
        }
        auto p_idx = packed_idx(idx);
        return {m_ptr + p_idx};
    }
    [[nodiscard]] auto at(size_type idx) const -> const_reference
    {
        if (idx >= size())
        {
            throw std::out_of_range(std::string("idx (which is") + std::to_string(idx) +
                                    std::string(") >= vector size (which is ") +
                                    std::to_string(size()) + std::string(")"));
        }
        auto p_idx = packed_idx(idx);
        return {m_ptr + p_idx};
    }

    [[nodiscard]] constexpr auto size() const noexcept -> size_type
    {
        return m_size;
    };
    [[nodiscard]] constexpr auto length() const noexcept -> size_type
    {
        return size();
    };
    void resize(size_type new_size)
    {
        if (num_packs(new_size) != num_packs(m_size))
        {
            real_type* new_ptr = alloc_traits::allocate(m_allocator, real_size(new_size));

            packed_copy(begin(), end(), iterator(new_ptr, 0));
            deallocate();

            m_size = new_size;
            m_ptr  = new_ptr;
        }
    }

    [[nodiscard]] auto begin() noexcept -> iterator
    {
        return iterator(m_ptr, 0);
    }
    [[nodiscard]] auto begin() const noexcept -> const_iterator
    {
        return const_iterator(m_ptr, 0);
    }
    [[nodiscard]] auto cbegin() const noexcept -> const_iterator
    {
        return const_iterator(m_ptr, 0);
    }

    [[nodiscard]] auto end() noexcept -> iterator
    {
        return iterator(m_ptr + packed_idx(size()), size());
    }
    [[nodiscard]] auto end() const noexcept -> const_iterator
    {
        return const_iterator(m_ptr + packed_idx(size()), size());
    }
    [[nodiscard]] auto cend() const noexcept -> const_iterator
    {
        return const_iterator(m_ptr + packed_idx(size()), size());
    }

private:
    [[no_unique_address]] allocator_type m_allocator{};

    size_type    m_size = 0;
    real_pointer m_ptr{};

    void deallocate()
    {
        if (m_ptr != real_pointer() && size() != 0)
        {
            alloc_traits::deallocate(m_allocator,
                                     m_ptr,
                                     num_packs(size()) * PackSize * 2);
            m_ptr  = real_pointer();
            m_size = 0;
        }
    }

    static constexpr auto num_packs(size_type vector_size) -> size_type
    {
        return (vector_size % PackSize > 0) ? vector_size / PackSize + 1
                                            : vector_size / PackSize;
    }
    static constexpr auto real_size(size_type vector_size) -> size_type
    {
        return num_packs(vector_size) * PackSize * 2;
    }
    static constexpr auto packed_idx(size_type idx) -> size_type
    {
        return idx + idx / PackSize * PackSize;
    }
};

template<typename T, std::size_t PackSize, bool Const>
class packed_iterator
{
    template<typename TVec, std::size_t PackSizeVec, typename>
        requires packed_floating_point<TVec, PackSizeVec>
    friend class packed_cx_vector;
    friend class packed_iterator<T, PackSize, true>;

public:
    using real_type        = T;
    using real_pointer     = T*;
    using difference_type  = ptrdiff_t;
    using reference        = typename std::conditional_t<Const,
                                                  const packed_cx_ref<T, PackSize, true>,
                                                  packed_cx_ref<T, PackSize>>;
    using value_type       = packed_cx_ref<T, PackSize, Const>;
    using iterator_concept = std::random_access_iterator_tag;

    static constexpr auto pack_size = PackSize;

private:
    packed_iterator(real_pointer data_ptr, difference_type index) noexcept
    : m_ptr(data_ptr)
    , m_idx(index){};

public:
    packed_iterator() noexcept = default;

    // NOLINTNEXTLINE(*explicit*)
    packed_iterator(const packed_iterator<T, PackSize>& other) noexcept
        requires Const
    : m_ptr(other.m_ptr)
    , m_idx(other.m_idx){};

    packed_iterator(const packed_iterator& other) noexcept = default;
    packed_iterator(packed_iterator&& other) noexcept      = default;

    packed_iterator& operator=(const packed_iterator& other) noexcept = default;
    packed_iterator& operator=(packed_iterator&& other) noexcept      = default;

    ~packed_iterator() noexcept = default;

    [[nodiscard]] reference operator*() const
    {
        return reference(m_ptr);
    }
    [[nodiscard]] reference operator[](difference_type idx) const noexcept
    {
        return *(*this + idx);
    }

    [[nodiscard]] bool operator==(const packed_iterator& other) const noexcept
    {
        return (m_ptr == other.m_ptr) && (m_idx == other.m_idx);
    }
    [[nodiscard]] auto operator<=>(const packed_iterator& other) const noexcept
    {
        return m_ptr <=> other.m_ptr;
    }

    packed_iterator& operator++() noexcept
    {
        if (++m_idx % PackSize == 0)
        {
            m_ptr += PackSize;
        }
        ++m_ptr;
        return *this;
    }
    packed_iterator operator++(int) noexcept
    {
        auto copy = *this;
        ++*this;
        return copy;
    }
    packed_iterator& operator--() noexcept
    {
        if (m_idx % PackSize == 0)
        {
            m_ptr -= PackSize;
        };
        --m_idx;
        --m_ptr;
        return *this;
    }
    packed_iterator operator--(int) noexcept
    {
        auto copy = *this;
        --*this;
        return copy;
    }

    packed_iterator& operator+=(difference_type n) noexcept
    {
        m_ptr = m_ptr + n + (m_idx % PackSize + n) / PackSize * PackSize;
        m_idx += n;
        return *this;
    }
    packed_iterator& operator-=(difference_type n) noexcept
    {
        return (*this) += -n;
    }

    [[nodiscard]] friend packed_iterator operator+(packed_iterator it,
                                                   difference_type n) noexcept
    {
        it += n;
        return it;
    }
    [[nodiscard]] friend packed_iterator operator+(difference_type n,
                                                   packed_iterator it) noexcept
    {
        it += n;
        return it;
    }
    [[nodiscard]] friend packed_iterator operator-(packed_iterator it,
                                                   difference_type n) noexcept
    {
        it -= n;
        return it;
    }

    template<bool OConst>
    [[nodiscard]] friend auto
    operator-(const packed_iterator&                      lhs,
              const packed_iterator<T, PackSize, OConst>& rhs) noexcept -> difference_type
    {
        return lhs.m_idx - rhs.m_idx;
    }

    [[nodiscard]] bool aligned(difference_type idx = 0) const noexcept
    {
        return (m_idx + idx) % PackSize == 0;
    }
    /**
     * @brief Returns aligned iterator not bigger then this itertor;
     *
     * @return packed_iterator
     */
    [[nodiscard]] auto align_lower() const noexcept -> packed_iterator
    {
        return *this - m_idx % PackSize;
    }
    /**
     * @brief Return aligned iterator not smaller then this itertator;
     *
     * @return packed_iterator
     */
    [[nodiscard]] auto align_upper() const noexcept -> packed_iterator
    {
        return m_idx % PackSize == 0 ? *this : *this + PackSize - m_idx % PackSize;
    }

private:
    real_pointer    m_ptr{};
    difference_type m_idx = 0;
};

template<typename T, std::size_t PackSize, bool Const, std::size_t Extent>
class packed_subrange : public std::ranges::view_base
{
    template<typename TVec, std::size_t PackSizeVec, typename>
        requires packed_floating_point<TVec, PackSizeVec>
    friend class packed_cx_vector;

    friend class internal::expression_traits;

public:
    using real_type       = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    using iterator       = packed_iterator<T, PackSize, Const>;
    using const_iterator = packed_iterator<T, PackSize, true>;

    using reference = typename iterator::reference;

    static constexpr size_type pack_size = PackSize;

private:
    using size_t =
        std::conditional_t<Extent == std::dynamic_extent, size_type, std::monostate>;

public:
    packed_subrange() noexcept = default;

    packed_subrange(const iterator& begin, size_type size) noexcept
        requires(Extent == std::dynamic_extent)
    : m_begin(begin)
    , m_size(size){};

    explicit packed_subrange(const iterator& begin) noexcept
        requires(Extent != std::dynamic_extent)
    : m_begin(begin){};

    packed_subrange(const packed_subrange&) noexcept = default;
    packed_subrange(packed_subrange&&) noexcept      = default;

    ~packed_subrange() noexcept = default;

    packed_subrange& operator=(packed_subrange&&) noexcept      = default;
    packed_subrange& operator=(const packed_subrange&) noexcept = default;

    void swap(packed_subrange& other) noexcept
    {
        using std::swap;
        swap(m_begin, other.m_begin);
        swap(m_size, other.m_size);
    }
    friend void swap(packed_subrange& first, packed_subrange& second) noexcept
    {
        first.swap(second);
    }

    [[nodiscard]] auto begin() const -> iterator
    {
        return m_begin;
    }
    [[nodiscard]] auto cbegin() const -> const_iterator
    {
        return m_begin;
    }
    [[nodiscard]] auto end() const -> iterator
    {
        return m_begin + size();
    }
    [[nodiscard]] auto cend() const -> const_iterator
    {
        return m_begin + size();
    }

    [[nodiscard]] auto operator[](difference_type idx) const -> reference
    {
        return *(m_begin + idx);
    }

    [[nodiscard]] constexpr auto size() const -> size_type
    {
        if constexpr (Extent == std::dynamic_extent)
        {
            return m_size;
        } else
        {
            return Extent;
        }
    };
    [[nodiscard]] constexpr bool empty() const
    {
        return size() == 0;
    }
    // NOLINTNEXTLINE (*explicit*)
    [[nodiscard]] operator bool() const
    {
        return size() != 0;
    }

    template<typename U>
        requires std::convertible_to<U, std::complex<T>>
    void fill(U value)
    {
        ::fill(begin(), end(), value);
    };

    template<typename R>
        requires(!Const) &&
                (!internal::vector_expression<R>) && std::ranges::input_range<R> &&
                std::indirectly_copyable<std::ranges::iterator_t<R>, iterator>
    void assign(const R& range)
    {
        std::ranges::copy(range, begin());
    };

    template<typename E>
        requires(!Const) && internal::vector_expression<E>
    void assign(const E& expression)
    {
        assert(size() == expression.size());

        auto it_this = begin();
        auto it_expr = expression.begin();
        auto aligned_begin = std::min(it_this.align_upper(), end());
        while (it_this < aligned_begin)
        {
            *it_this = *it_expr;
            ++it_this;
            ++it_expr;
        }
        if (it_this.aligned() && it_expr.aligned())
        {
            constexpr auto reg_size = 32 / sizeof(T);

            auto aligned_size = end().align_lower() - it_this;

            auto ptr = &(*it_this);
            for (long i = 0; i < aligned_size; i += pack_size)
            {
                auto offset = i * 2;
                for (uint i_reg = 0; i_reg < pack_size; i_reg += reg_size)
                {
                    auto data =
                        internal::expression_traits::cx_reg(it_expr, offset + i_reg);

                    avx::store(ptr + offset + i_reg, data.real);
                    avx::store(ptr + offset + i_reg + PackSize, data.imag);
                }
            }
            it_this += aligned_size;
            it_expr += aligned_size;
        }
        while (it_this < end())
        {
            *it_this = *it_expr;
            ++it_this;
            ++it_expr;
        }
    };

private:
    [[no_unique_address]] size_t m_size{};
    iterator                     m_begin{};
};

template<typename T, std::size_t PackSize, bool Const>
class packed_cx_ref
{
    template<typename TVec, std::size_t PackSizeVec, typename>
        requires packed_floating_point<TVec, PackSizeVec>
    friend class packed_cx_vector;

    friend class packed_iterator<T, PackSize, Const>;
    friend class packed_iterator<T, PackSize, false>;

    friend class packed_subrange<T, PackSize, Const>;
    friend class packed_subrange<T, PackSize, false>;

    friend class packed_cx_ref<T, PackSize, true>;

public:
    using real_type  = T;
    using pointer    = typename std::conditional<Const, const T*, T*>::type;
    using value_type = std::complex<real_type>;

private:
    explicit packed_cx_ref(pointer ptr) noexcept
    : m_ptr(ptr){};

public:
    packed_cx_ref() = delete;

    // NOLINTNEXTLINE (*explicit*)
    packed_cx_ref(const packed_cx_ref<T, PackSize, false>& other) noexcept
        requires Const
    : m_ptr(other.m_ptr){};

    packed_cx_ref(const packed_cx_ref&) noexcept = default;
    packed_cx_ref(packed_cx_ref&&) noexcept      = default;

    ~packed_cx_ref() = default;

    packed_cx_ref& operator=(const packed_cx_ref& other)
        requires(!Const)
    {
        *m_ptr              = *other.m_ptr;
        *(m_ptr + PackSize) = *(other.m_ptr + PackSize);
        return *this;
    }
    packed_cx_ref& operator=(packed_cx_ref&& other)
        requires(!Const)
    {
        *m_ptr              = *other.m_ptr;
        *(m_ptr + PackSize) = *(other.m_ptr + PackSize);
        return *this;
    }
    // NOLINTNEXTLINE (*assign*) Proxy reference, see std::indirectly_writable
    const packed_cx_ref& operator=(const packed_cx_ref& other) const
        requires(!Const)
    {
        *m_ptr              = *other.m_ptr;
        *(m_ptr + PackSize) = *(other.m_ptr + PackSize);
        return *this;
    }
    // NOLINTNEXTLINE (*assign*) Proxy reference, see std::indirectly_writable
    const packed_cx_ref& operator=(packed_cx_ref&& other) const
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

    [[nodiscard]] operator value_type() const
    {
        return value_type(*m_ptr, *(m_ptr + PackSize));
    }
    [[nodiscard]] value_type value() const
    {
        return *this;
    }

    [[nodiscard]] auto operator&() const noexcept -> pointer
    {
        return m_ptr;
    }

private:
    pointer m_ptr{};
};