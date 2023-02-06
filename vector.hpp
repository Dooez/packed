#include "vector_util.hpp"

#include <complex>
#include <memory>
#include <ranges>
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
private:
    using alloc_traits = std::allocator_traits<Allocator>;

public:
    using real_type    = T;
    using real_pointer = typename alloc_traits::pointer;

    using allocator_type  = Allocator;
    using size_type       = typename alloc_traits::size_type;
    using difference_type = typename alloc_traits::difference_type;
    using reference       = packed_cx_ref<T, PackSize, Allocator, false>;
    using const_reference = packed_cx_ref<T, PackSize, Allocator, true>;

    using iterator       = packed_iterator<T, PackSize, Allocator, false>;
    using const_iterator = packed_iterator<T, PackSize, Allocator, true>;

public:
    packed_cx_vector() noexcept(noexcept(allocator_type())) = default;

    explicit packed_cx_vector(const allocator_type& allocator) noexcept(
        std::is_nothrow_copy_constructible_v<allocator_type>)
    : m_allocator(allocator)
    , m_length(0)
    , m_ptr(nullptr){};

    explicit packed_cx_vector(size_type             length,
                              const allocator_type& allocator = allocator_type())
    : m_allocator(allocator)
    , m_length(length)
    , m_ptr(alloc_traits::allocate(m_allocator, num_packs(length) * PackSize * 2))
    {
        set(begin(), end(), 0);
    };

    template<typename U>
        requires std::is_convertible_v<U, std::complex<real_type>>
    packed_cx_vector(size_type             length,
                     U                     value,
                     const allocator_type& allocator = allocator_type())
    : m_allocator(allocator)
    , m_length(length)
    , m_ptr(alloc_traits::allocate(m_allocator, num_packs(length) * PackSize * 2))
    {
        set(begin(), end(), value);
    };

    packed_cx_vector(const packed_cx_vector& other)
    : m_allocator(alloc_traits::select_on_container_copy_construction(other.m_allocator))
    , m_length(other.m_length)
    {
        if (m_length > 0)
        {
            m_ptr =
                alloc_traits::allocate(m_allocator, num_packs(m_length) * PackSize * 2);
        }
        packed_copy(other.begin(), other.end(), begin());
    }

    packed_cx_vector(const packed_cx_vector& other, const allocator_type& allocator)
    : m_allocator(allocator)
    , m_length(other.m_length)
    {
        if (m_length == 0)
        {
            return;
        }
        m_ptr = alloc_traits::allocate(m_allocator, num_packs(m_length) * PackSize * 2);

        packed_copy(other.begin(), other.end(), begin());
    }

    packed_cx_vector(packed_cx_vector&& other) noexcept(
        std::is_nothrow_move_constructible_v<allocator_type>)
    : m_allocator(std::move(other.m_allocator))
    {
        using std::swap;
        swap(m_length, other.m_length);
        swap(m_ptr, other.m_ptr);
    };

    packed_cx_vector(packed_cx_vector&& other, const allocator_type& allocator) noexcept(
        alloc_traits::is_always_equal::value)
    : m_allocator(allocator)
    {
        if constexpr (alloc_traits::is_always_equal::value)
        {
            using std::swap;
            swap(m_length, other.m_length);
            swap(m_ptr, other.m_ptr);
        } else
        {
            if (m_allocator == other.m_allocator)
            {
                using std::swap;
                swap(m_length, other.m_length);
                swap(m_ptr, other.m_ptr);
                return;
            }
            m_ptr =
                alloc_traits::allocate(m_allocator, num_packs(m_length) * PackSize * 2);
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
                num_packs(m_length) != num_packs(other.m_length))
            {
                deallocate();
                m_allocator = other.m_allocator;
                m_length    = other.m_length;
                m_ptr       = alloc_traits::allocate(m_allocator,
                                               num_packs(m_length) * PackSize * 2);
            }
        } else
        {
            if constexpr (alloc_traits::propagate_on_container_copy_assignment::value)
            {
                m_allocator = other.m_allocator;
            }
            if (num_packs(m_length) != num_packs(other.m_length))
            {
                deallocate();
                m_length = other.m_length;
                m_ptr    = alloc_traits::allocate(m_allocator,
                                               num_packs(m_length) * PackSize * 2);
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
            swap(m_length, other.m_length);
            swap(m_ptr, other.m_ptr);
            return *this;
        } else
        {
            if constexpr (alloc_traits::is_always_equal::value)
            {
                deallocate();
                using std::swap;
                swap(m_length, other.m_length);
                swap(m_ptr, other.m_ptr);
                return *this;
            } else
            {
                if (m_allocator == other.m_allocator)
                {
                    deallocate();
                    using std::swap;
                    swap(m_length, other.m_length);
                    swap(m_ptr, other.m_ptr);
                    return *this;
                }
                if (num_packs(m_length) != num_packs(other.m_length))
                {
                    deallocate();
                    m_length = other.m_length;
                    m_ptr    = alloc_traits::allocate(m_allocator,
                                                   num_packs(m_length) * PackSize * 2);
                }
                packed_copy(other.begin(), other.end(), begin());
                return *this;
            }
        }
    }

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
            swap(m_length, other.m_length);
            swap(m_ptr, other.m_ptr);
        } else
        {
            using std::swap;
            swap(m_length, other.m_length);
            swap(m_ptr, other.m_ptr);
        }
    }

    friend void swap(packed_cx_vector& first,
                     packed_cx_vector& second) noexcept(noexcept(first.swap(second)))
    {
        first.swap(second);
    }

    [[nodiscard]] allocator_type get_allocator() const
        noexcept(std::is_nothrow_copy_constructible_v<allocator_type>)
    {
        return m_allocator;
    }

    [[nodiscard]] reference operator[](size_type idx)
    {
        auto p_idx = packed_idx(idx);
        return {m_ptr + p_idx};
    }
    [[nodiscard]] const_reference operator[](size_type idx) const
    {
        auto p_idx = packed_idx(idx);
        return {m_ptr + p_idx};
    }

    [[nodiscard]] reference at(size_type idx)
    {
        if (idx >= m_length)
        {
            throw std::out_of_range(std::string("idx (which is") + std::to_string(idx) +
                                    std::string(") >= vector size (which is ") +
                                    std::to_string(m_length) + std::string(")"));
        }
        auto p_idx = packed_idx(idx);
        return {m_ptr + p_idx};
    }
    [[nodiscard]] const_reference at(size_type idx) const
    {
        if (idx >= m_length)
        {
            throw std::out_of_range(std::string("idx (which is") + std::to_string(idx) +
                                    std::string(") >= vector size (which is ") +
                                    std::to_string(m_length) + std::string(")"));
        }
        auto p_idx = packed_idx(idx);
        return {m_ptr + p_idx};
    }

    [[nodiscard]] size_type size() const noexcept
    {
        return m_length;
    };
    [[nodiscard]] size_type length() const noexcept
    {
        return m_length;
    };
    void resize(size_type new_length)
    {
        if (num_packs(new_length) != num_packs(m_length))
        {
            real_type* new_ptr =
                alloc_traits::allocate(m_allocator, num_packs(new_length) * PackSize * 2);

            packed_copy(begin(), end(), iterator(new_ptr, 0));
            deallocate();

            m_length = new_length;
            m_ptr    = new_ptr;
        }
    }

    [[nodiscard]] iterator begin() noexcept
    {
        return iterator(m_ptr, 0);
    }
    [[nodiscard]] const_iterator begin() const noexcept
    {
        return const_iterator(m_ptr, 0);
    }
    [[nodiscard]] const_iterator cbegin() const noexcept
    {
        return const_iterator(m_ptr, 0);
    }

    [[nodiscard]] iterator end() noexcept
    {
        return iterator(m_ptr + packed_idx(m_length), m_length);
    }
    [[nodiscard]] const_iterator end() const noexcept
    {
        return const_iterator(m_ptr + packed_idx(m_length), m_length);
    }
    [[nodiscard]] const_iterator cend() const noexcept
    {
        return const_iterator(m_ptr + packed_idx(m_length), m_length);
    }

private:
    [[no_unique_address]] allocator_type m_allocator{};

    size_type    m_length = 0;
    real_pointer m_ptr{};

    void deallocate()
    {
        if (m_ptr != real_pointer() && m_length != 0)
        {
            alloc_traits::deallocate(m_allocator,
                                     m_ptr,
                                     num_packs(m_length) * PackSize * 2);
            m_ptr    = real_pointer();
            m_length = 0;
        }
    }

    static constexpr auto num_packs(size_type vector_size) -> size_type
    {
        return (vector_size % PackSize > 0) ? vector_size / PackSize + 1
                                            : vector_size / PackSize;
    }
    static constexpr auto packed_idx(size_type idx) -> size_type
    {
        return idx + idx / PackSize * PackSize;
    }
};

template<typename T, std::size_t PackSize, typename Allocator, bool Const>
class packed_iterator
{
    friend class packed_cx_vector<T, PackSize, Allocator>;
    friend class packed_iterator<T, PackSize, Allocator, true>;

public:
    using real_type    = T;
    using real_pointer = typename packed_cx_vector<T, PackSize, Allocator>::real_pointer;
    using difference_type =
        typename packed_cx_vector<T, PackSize, Allocator>::difference_type;
    using reference        = packed_cx_ref<T, PackSize, Allocator, Const>;
    using value_type       = std::complex<T>;
    using iterator_concept = std::random_access_iterator_tag;

    static constexpr auto pack_size = PackSize;

private:
    using alloc_traits = std::allocator_traits<Allocator>;

    packed_iterator(real_pointer data_ptr, difference_type index) noexcept
    : m_ptr(data_ptr)
    , m_idx(index){};

public:
    packed_iterator() noexcept = default;

    packed_iterator(const packed_iterator<T, PackSize, Allocator, false>& other) noexcept
        requires requires { Const; }
    : m_ptr(other.m_ptr)
    , m_idx(other.m_idx){};

    packed_iterator(const packed_iterator& other) noexcept = default;
    packed_iterator(packed_iterator&& other) noexcept      = default;

    packed_iterator& operator=(const packed_iterator& other) noexcept = default;
    packed_iterator& operator=(packed_iterator&& other) noexcept      = default;

    ~packed_iterator() = default;

    [[nodiscard]] value_type value() const
    {
        return {*m_ptr, *(m_ptr + PackSize)};
    }
    [[nodiscard]] reference operator*() const noexcept
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
    [[nodiscard]] difference_type
    operator-(const packed_iterator<T, PackSize, Allocator, OConst>& other) const noexcept
    {
        return m_idx - other.m_idx;
    }

    [[nodiscard]] bool aligned() const noexcept
    {
        return m_idx % PackSize == 0;
    }
    /**
     * @brief Returns aligned iterator not bigger then this itertor;
     *
     * @return packed_iterator
     */
    [[nodiscard]] packed_iterator align_lower() const noexcept
    {
        return *this - m_idx % PackSize;
    }
    /**
     * @brief Return aligned iterator not smaller then this itertator;
     *
     * @return packed_iterator
     */
    [[nodiscard]] packed_iterator align_upper() const noexcept
    {
        return m_idx % PackSize == 0 ? *this : *this + PackSize - m_idx % PackSize;
    }

private:
    real_pointer    m_ptr{};
    difference_type m_idx = 0;
};

template<typename T, std::size_t PackSize, typename Allocator, bool Const>
class packed_subrange : std::ranges::view_base
{
    using vector_t = packed_cx_vector<T, PackSize, Allocator>;

    friend vector_t;

public:
    using real_type       = T;
    using size_type       = typename vector_t::size_type;
    using difference_type = typename vector_t::difference_type;

    using iterator       = packed_iterator<T, PackSize, Allocator, Const>;
    using reference_type = typename iterator::reference_type;

    packed_subrange() = default;

    packed_subrange(const packed_subrange&) noexcept = default;
    packed_subrange(packed_subrange&&) noexcept      = default;

    ~packed_subrange() = default;

    packed_subrange& operator=(packed_subrange&&) noexcept = default;

    void swap(packed_subrange& other) noexcept(std::is_nothrow_swappable_v<iterator>)
    {
        using std::swap;
        swap(m_begin, other.m_begin);
        swap(m_size, other.m_size);
    }

    friend void swap(packed_subrange& first,
                     packed_subrange& second) noexcept(first.swap(second))
    {
        first.swap(second);
    }

    auto begin() const -> iterator
    {
        return m_begin;
    }
    auto end() const -> iterator
    {
        return m_begin + m_size;
    }
    auto operator[](difference_type idx) -> reference_type
    {
        return *(m_begin + idx);
    }

    auto size() const -> size_type
    {
        return m_size;
    };
    bool empty() const
    {
        return m_size == 0;
    }
    operator bool() const
    {
        return m_size != 0;
    }

    packed_subrange& operator=(const packed_subrange&) = delete;

private:
    iterator  m_begin{};
    size_type m_size{};
};

template<typename T, std::size_t PackSize, typename Allocator, bool Const>
class packed_cx_ref
{
    friend class packed_cx_vector<T, PackSize, Allocator>;
    friend class packed_iterator<T, PackSize, Allocator, Const>;
    friend class packed_cx_ref<T, PackSize, Allocator, true>;

public:
    using real_type = T;
    using pointer   = typename std::conditional<
        Const,
        typename std::allocator_traits<Allocator>::const_pointer,
        typename std::allocator_traits<Allocator>::pointer>::type;
    using value_type = std::complex<real_type>;

private:
    packed_cx_ref(pointer ptr) noexcept
    : m_ptr(ptr){};

public:
    packed_cx_ref() = delete;

    packed_cx_ref(const packed_cx_ref<T, PackSize, Allocator, false>& other) noexcept
        requires requires { Const; }
    : m_ptr(other.m_ptr){};

    packed_cx_ref(const packed_cx_ref&) noexcept = default;
    packed_cx_ref(packed_cx_ref&&) noexcept      = default;

    ~packed_cx_ref() = default;

    packed_cx_ref& operator=(const packed_cx_ref& other)
        requires requires { !Const; }
    {
        *m_ptr              = *other.m_ptr;
        *(m_ptr + PackSize) = *(other.m_ptr + PackSize);
        return *this;
    }
    template<typename U>
        requires std::is_convertible_v<U, value_type>
    packed_cx_ref& operator=(const U& other)
        requires requires { !Const; }
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

    [[nodiscard]] pointer operator&() const noexcept
    {
        return m_ptr;
    }

private:
    pointer m_ptr{};
};