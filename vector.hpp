#include <memory>

template <std::size_t N>
concept power_of_two = requires { (N & (N - 1)) == 0; };

template <typename T, std::size_t PackSize>
class packed_cx_ref
{
};

template <typename T, std::size_t PackSize>
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
template <typename T,
          std::size_t PackSize = 128 / sizeof(T),
          typename Allocator = std::allocator<T>>
    requires std::floating_point<T> && power_of_two<PackSize> &&
             requires { PackSize >= 64 / sizeof(T); }
class packed_cx_vector
{
public:
    using real_type = T;
    using allocator_type = Allocator;
    using size_type = std::size_t;
    using reference = packed_cx_ref<T, PackSize>;
    using const_reference = const_packed_cx_ref<T, PackSize>;
    class iterator;
    class const_iterator;

public:
    packed_cx_vector() = default;

    explicit packed_cx_vector(size_type length)
        : m_length(length), m_ptr(
                                std::allocator_traits<Allocator>::allocate(num_packs(length) * PackSize * 2)){
                                // TODO: set to 0
                            };

    packed_cx_vector(const packed_cx_vector &other);
    packed_cx_vector(packed_cx_vector &&other) noexcept
        : m_length(other.m_length), m_ptr(other.m_ptr)
    {
        other.m_length = 0;
        other.m_ptr = nullptr;
    };

    ~packed_cx_vector()
    {
        if (m_ptr != nullptr)
        {
            std::allocator_traits<Allocator>::deallocate(m_ptr,
                                                         num_packs(m_length) * PackSize * 2);
        }
    };

    packed_cx_vector &operator=(packed_cx_vector &&other) noexcept
    {
        swap(*this, other);
        return *this;
    };
    packed_cx_vector &operator=(const packed_cx_vector &other);

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
            real_type *new_ptr = std::allocator_traits<Allocator>::allocate(num_packs(new_length) * PackSize * 2);
            // TODO: copy data
            std::allocator_traits<Allocator>::deallocate(m_ptr,
                                                         num_packs(m_length) * PackSize *
                                                             2);

            m_length = new_length;
            m_ptr = new_ptr;
        }
    }

    friend void swap(packed_cx_vector &first, packed_cx_vector &second)
    {
        using std::swap;
        swap(first.m_ptr, second.m_ptr);
        swap(first.m_length, second.m_length);
    };

private:
    size_type m_length = 0;
    real_type *m_ptr = nullptr;

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