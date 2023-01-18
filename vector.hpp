#include <memory>

template<std::size_t N>
concept power_of_two = requires { (N & (N - 1)) == 0; };

template<typename T, std::size_t PackSize>
class packed_cx_ref
{
};

template<typename T, std::size_t PackSize>
class const_packed_cx_ref
{
};

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
    requires std::floating_point<T> && power_of_two<PackSize> &&
             requires { PackSize >= 64 / sizeof(T); }
class packed_cx_vector
{
public:
    using real_type       = T;
    using allocator_type  = Allocator;
    using size_type       = std::size_t;
    using reference       = packed_cx_ref<T, PackSize>;
    using const_reference = const_packed_cx_ref<T, PackSize>;
    class iterator;
    class const_iterator;

private:
    using alloc_traits = std::allocator_traits<allocator_type>;

public:
    packed_cx_vector() noexcept(noexcept(allocator_type())) = default;

    // explicit packed_cx_vector(size_type length)
    //     : m_length(length), m_ptr(
    //                             std::allocator_traits<Allocator>::allocate(num_packs(length) * PackSize * 2)){
    //                             // TODO: set to 0
    //                         };

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


private:
    [[no_unique_address]] allocator_type m_allocator{};
    size_type                            m_length = 0;
    real_type*                           m_ptr    = nullptr;

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