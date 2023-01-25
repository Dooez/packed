#include <complex>
#include <memory>
#include <vector>

template<std::size_t N>
concept power_of_two = requires { (N & (N - 1)) == 0; };

template<typename T, std::size_t PackSize>
concept packed_floating_point = std::floating_point<T> && power_of_two<PackSize> &&
                                requires { PackSize >= 64 / sizeof(T); };

template<typename T, std::size_t PackSize, typename Allocator, bool Const>
class packed_cx_ref;

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
    template<bool>
    class iterator_base;

    using alloc_traits = std::allocator_traits<Allocator>;

public:
    using real_type    = T;
    using real_pointer = typename alloc_traits::pointer;

    using allocator_type  = Allocator;
    using size_type       = typename alloc_traits::size_type;
    using difference_type = typename alloc_traits::difference_type;
    using reference       = packed_cx_ref<T, PackSize, Allocator, false>;
    using const_reference = packed_cx_ref<T, PackSize, Allocator, true>;

    using iterator       = iterator_base<false>;
    using const_iterator = iterator_base<true>;

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
        auto a = 0;
        // TODO: set to 0
    };

    packed_cx_vector(size_type               length,
                     std::complex<real_type> value,
                     const allocator_type&   allocator = allocator_type())
    : m_allocator(allocator)
    , m_length(length)
    , m_ptr(alloc_traits::allocate(m_allocator, num_packs(length) * PackSize * 2))
    {
        auto a = 0;
        // TODO: set to value
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
        // TODO: Copy data
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

        // TODO: Copy data
    }

    packed_cx_vector(packed_cx_vector&& other) noexcept(
        std::is_nothrow_move_constructible_v<allocator_type>) = default;

    packed_cx_vector(packed_cx_vector&& other, const allocator_type& allocator) noexcept
        requires(alloc_traits::is_always_equal::value)
    : m_allocator(allocator)
    , m_length(std::move(other.m_length))
    , m_ptr(std::move(other.m_ptr)){};

    packed_cx_vector(packed_cx_vector&& other, const allocator_type& allocator)
    : m_allocator(allocator)
    , m_length(std::move(other.m_length))
    {
        if (m_allocator == other.m_allocator)
        {
            m_ptr = std::move(other.m_ptr);
            return;
        }
        m_ptr = alloc_traits::allocate(m_allocator, num_packs(m_length) * PackSize * 2);
        // TODO: Copy data
    }

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

        //TODO: Copy contents
        return *this;
    }

    packed_cx_vector& operator=(packed_cx_vector&& other) noexcept(
        alloc_traits::propagate_on_container_move_assignment::value&&
            std::is_nothrow_move_constructible_v<allocator_type> ||
        alloc_traits::is_always_equal::value)
    {
        if constexpr (alloc_traits::propagate_on_container_move_assignment::value)
        {
            deallocate();
            m_allocator = std::move(other.m_allocator);
            m_length    = std::move(other.m_length);
            m_ptr       = std::move(other.m_ptr);
            return *this;
        } else
        {
            if constexpr (alloc_traits::is_always_equal::value)
            {
                deallocate();
                m_length = std::move(other.m_length);
                m_ptr    = std::move(other.m_ptr);
                return *this;
            } else
            {
                if (m_allocator == other.m_allocator)
                {
                    deallocate();
                    m_length = std::move(other.m_length);
                    m_ptr    = std::move(other.m_ptr);
                    return *this;
                }
                if (num_packs(m_length) != num_packs(other.m_length))
                {
                    deallocate();
                    m_length = other.m_length;
                    m_ptr    = alloc_traits::allocate(m_allocator,
                                                   num_packs(m_length) * PackSize * 2);
                }
                //TODO: Copy contents
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
            std::is_nothrow_move_constructible_v<allocator_type> ||
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

    allocator_type get_allocator() const
    {
        return m_allocator;
    }

    // void set(std::complex<T> z);

    [[nodiscard]] reference operator[](size_type idx)
    {
        auto p_idx = packed_idx(idx);
        return {m_ptr[p_idx]};
    }
    [[nodiscard]] const_reference operator[](size_type idx) const
    {
        auto p_idx = packed_idx(idx);
        return {m_ptr[p_idx]};
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
        return {m_ptr[p_idx]};
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
        return {m_ptr[p_idx]};
    }

    [[nodiscard]] size_type size() const
    {
        return m_length;
    };
    [[nodiscard]] size_type length() const
    {
        return m_length;
    };
    void resize(size_type new_length)
    {
        if (num_packs(new_length) != num_packs(m_length))
        {
            real_type* new_ptr =
                alloc_traits::allocate(m_allocator, num_packs(new_length) * PackSize * 2);
            // TODO: copy data
            deallocate();

            m_length = new_length;
            m_ptr    = new_ptr;
        }
    }

    iterator begin()
    {
        return iterator(m_ptr, 0);
    }

    const_iterator begin() const
    {
        return const_iterator(m_ptr, 0);
    }

    iterator end()
    {
        return iterator(m_ptr + m_length, m_length);
    }

    const_iterator end() const
    {
        return const_iterator(m_ptr + m_length, m_length);
    }

private:
    [[no_unique_address]] allocator_type m_allocator{};

    size_type    m_length = 0;
    real_pointer m_ptr{};

    void deallocate()
    {
        if (m_ptr != nullptr && m_length != 0)
        {
            alloc_traits::deallocate(m_allocator,
                                     m_ptr,
                                     num_packs(m_length) * PackSize * 2);
        }
    }

    static constexpr auto num_packs(size_type vector_size) -> size_type
    {
        return (vector_size % PackSize > 0) ? vector_size / PackSize + 1
                                            : vector_size / PackSize;
    }
    static constexpr auto packed_idx(size_type idx) -> size_type
    {
        return (idx / PackSize) * PackSize * 2 + idx % PackSize;
    }
};

template<typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize>
template<bool Const>
class packed_cx_vector<T, PackSize, Allocator>::iterator_base
{
    friend class packed_cx_vector<T, PackSize, Allocator>;

public:
    using real_type       = T;
    using real_pointer    = packed_cx_vector::real_pointer;
    using difference_type = packed_cx_vector::difference_type;
    using reference       = packed_cx_ref<T, PackSize, Allocator, Const>;
    using value_type      = std::complex<T>;
    // using pointer          = packed_cx_ref<T, PackSize, Allocator, Const>;
    using iterator_concept = std::random_access_iterator_tag;

    static constexpr auto pack_size = PackSize;

private:
    using alloc_traits = std::allocator_traits<Allocator>;

    iterator_base(real_pointer data_ptr, difference_type index) noexcept
    : m_ptr(data_ptr)
    , m_idx(index){};

public:
    iterator_base() = default;

    iterator_base(const iterator_base<false>& other) noexcept
        requires requires { Const; }
    : m_ptr(other.m_ptr)
    , m_idx(other.m_idx){};

    iterator_base(const iterator_base& other) noexcept = default;
    iterator_base(iterator_base&& other) noexcept      = default;

    iterator_base& operator=(const iterator_base& other) noexcept = default;
    iterator_base& operator=(iterator_base&& other) noexcept      = default;

    ~iterator_base() = default;

    value_type value() const
    {
        return {*m_ptr, *(m_ptr + PackSize)};
    }
    reference operator*() const
    {
        return reference(m_ptr);
    }
    reference operator[](difference_type idx) const
    {
        return *(*this + idx);
    }

    bool operator==(const iterator_base& other) const
    {
        return (m_ptr == other.m_ptr) && (m_idx == other.m_idx);
    }
    auto operator<=>(const iterator_base& other) const
    {
        return m_ptr <=> other.m_ptr;
    }

    iterator_base& operator++() noexcept
    {
        if (++m_idx % PackSize == 0)
        {
            m_ptr += PackSize;
        }
        ++m_ptr;
        return *this;
    }
    iterator_base operator++(int) noexcept
    {
        auto copy = *this;
        ++*this;
        return copy;
    }
    iterator_base& operator--() noexcept
    {
        if (m_idx % PackSize == 0)
        {
            m_ptr -= PackSize;
        };
        --m_idx;
        --m_ptr;
        return *this;
    }
    iterator_base operator--(int) noexcept
    {
        auto copy = *this;
        --*this;
        return copy;
    }

    iterator_base& operator+=(difference_type n)
    {
        m_ptr = m_ptr + n + (m_idx % PackSize + n) / PackSize * PackSize;
        m_idx += n;
        return *this;
    }
    iterator_base& operator-=(difference_type n)
    {
        return (*this) += -n;
    }

    friend iterator_base operator+(iterator_base it, difference_type n)
    {
        it += n;
        return it;
    }
    friend iterator_base operator+(difference_type n, iterator_base it)
    {
        it += n;
        return it;
    }
    friend iterator_base operator-(iterator_base it, difference_type n)
    {
        it -= n;
        return it;
    }

    template<bool OConst>
    difference_type operator-(const iterator_base<OConst>& other) const
    {
        return m_idx - other.m_idx;
    }

private:
    real_pointer    m_ptr{};
    difference_type m_idx = 0;
};

template<typename T, std::size_t PackSize, typename Allocator, bool Const>
class packed_cx_ref
{
    friend class packed_cx_vector<T, PackSize, Allocator>;


public:
    using real_type    = T;
    using real_pointer = typename std::allocator_traits<Allocator>::pointer;
    using value_type   = std::complex<real_type>;

private:
    packed_cx_ref(real_pointer ptr)
    : m_ptr(ptr){};

public:
    packed_cx_ref() = delete;

    packed_cx_ref(const packed_cx_ref&)     = delete;
    packed_cx_ref(packed_cx_ref&&) noexcept = default;

    ~packed_cx_ref() = default;

    template<typename U>
        requires std::is_convertible_v<U, value_type>
    packed_cx_ref operator=(const U& other)
        requires requires { !Const; }
    {
        auto tmp            = static_cast<value_type>(other);
        *m_ptr              = tmp.real();
        *(m_ptr + PackSize) = tmp.imag();
        return *this;
    }

    operator value_type() const
    {
        return value_type(*m_ptr, *(m_ptr + PackSize));
    }
    value_type value() const
    {
        return *this;
    }


private:
    real_pointer m_ptr{};
};