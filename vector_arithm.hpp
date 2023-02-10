#ifndef VECTOR_ARITHM_HPP
#define VECTOR_ARITHM_HPP
#include "vector_util.hpp"

#include <assert.h>
#include <complex>
#include <concepts>
#include <immintrin.h>
#include <ranges>
#include <type_traits>

namespace avx {

inline auto add(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
{
    return _mm256_add_ps(lhs, rhs);
}
inline auto add(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
{
    return _mm256_add_pd(lhs, rhs);
}

inline auto sub(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
{
    return _mm256_sub_ps(lhs, rhs);
}
inline auto sub(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
{
    return _mm256_sub_pd(lhs, rhs);
}

inline auto mul(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
{
    return _mm256_mul_ps(lhs, rhs);
}
inline auto mul(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
{
    return _mm256_mul_pd(lhs, rhs);
}

inline auto div(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
{
    return _mm256_div_ps(lhs, rhs);
}
inline auto div(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
{
    return _mm256_div_pd(lhs, rhs);
}

template<typename T>
struct cx_reg
{
    typename reg<T>::type real;
    typename reg<T>::type imag;
};

template<typename T>
inline auto broadcast(const std::complex<T>& source) -> cx_reg<T>
{
    const auto& value = reinterpret_cast<const T(&)[2]>(source);
    return {avx::broadcast(&(value[0])), avx::broadcast(&(value[1]))};
}

template<typename T>
inline auto broadcast(const T& source) -> typename reg<T>::type
{
    return avx::broadcast(&source);
}

template<typename T>
inline auto add(cx_reg<T> lhs, cx_reg<T> rhs) -> cx_reg<T>
{
    return {add(lhs.real, rhs.real), add(lhs.imag, rhs.imag)};
}
template<typename T>
inline auto sub(cx_reg<T> lhs, cx_reg<T> rhs) -> cx_reg<T>
{
    return {sub(lhs.real, rhs.real), sub(lhs.imag, rhs.imag)};
}
template<typename T>
inline auto mul(cx_reg<T> lhs, cx_reg<T> rhs) -> cx_reg<T>
{
    const auto real =
        avx::sub(avx::mul(lhs.real, rhs.real), avx::mul(lhs.imag, rhs.imag));
    const auto imag =
        avx::add(avx::mul(lhs.real, rhs.imag), avx::mul(lhs.imag, rhs.real));

    return {real, imag};
}
template<typename T>
inline auto div(cx_reg<T> lhs, cx_reg<T> rhs) -> cx_reg<T>
{
    const auto rhs_abs =
        avx::add(avx::mul(rhs.real, rhs.real), avx::mul(rhs.imag, rhs.imag));

    const auto real_ =
        avx::add(avx::mul(lhs.real, rhs.real), avx::mul(lhs.imag, rhs.imag));

    const auto imag_ =
        avx::sub(avx::mul(lhs.imag, rhs.real), avx::mul(lhs.real, rhs.imag));

    return {avx::div(real_, rhs_abs), avx::div(imag_, rhs_abs)};
}

template<typename T>
inline auto add(typename reg<T>::type lhs, cx_reg<T> rhs) -> cx_reg<T>
{
    return {add(lhs, rhs.real), rhs.imag};
}
template<typename T>
inline auto sub(typename reg<T>::type lhs, cx_reg<T> rhs) -> cx_reg<T>
{
    return {sub(lhs, rhs.real), rhs.imag};
}
template<typename T>
inline auto mul(typename reg<T>::type lhs, cx_reg<T> rhs) -> cx_reg<T>
{
    return {avx::mul(lhs, rhs.real), avx::mul(lhs, rhs.imag)};
}
template<typename T>
inline auto div(typename reg<T>::type lhs, cx_reg<T> rhs) -> cx_reg<T>
{
    return {avx::div(lhs, rhs.real), avx::div(lhs, rhs.imag)};
}

template<typename T>
inline auto add(cx_reg<T> lhs, typename reg<T>::type rhs) -> cx_reg<T>
{
    return {add(lhs.real, rhs), lhs.imag};
}
template<typename T>
inline auto sub(cx_reg<T> lhs, typename reg<T>::type rhs) -> cx_reg<T>
{
    return {sub(lhs.real, rhs), lhs.imag};
}
template<typename T>
inline auto mul(cx_reg<T> lhs, typename reg<T>::type rhs) -> cx_reg<T>
{
    return {avx::mul(lhs.real, rhs), avx::mul(lhs.imag, rhs)};
}
template<typename T>
inline auto div(cx_reg<T> lhs, typename reg<T>::type rhs) -> cx_reg<T>
{
    return {avx::div(lhs.real, rhs), avx::div(lhs.imag, rhs)};
}

}    // namespace avx

namespace internal {

template<typename E>
struct is_scalar
{
    static constexpr bool value = false;
};

class expression_base
{
public:
    template<typename E>
    struct is_expression
    {
        static constexpr bool value =
            requires(E expression, std::size_t idx) {
                requires std::ranges::view<E>;

                requires std::ranges::random_access_range<E>;

                typename E::real_type;

                typename E::iterator;

                {
                    expression[idx]
                    } -> std::convertible_to<std::complex<typename E::real_type>>;

                {
                    expression.size()
                    } -> std::same_as<std::size_t>;


                requires requires(typename E::iterator iter) {
                             {
                                 iter.aligned(idx)
                                 } -> std::same_as<bool>;

                             {
                                 iter.cx_reg(idx)
                                 } -> std::same_as<avx::cx_reg<typename E::real_type>>;

                             requires std::convertible_to<
                                 std::iter_value_t<typename E::iterator>,
                                 std::complex<typename E::real_type>>;
                         };
            };
    };

    template<typename T, std::size_t PackSize, bool Const, std::size_t Extent>
    struct is_expression<packed_subrange<T, PackSize, Const, Extent>>
    {
        static constexpr bool value = true;
    };

protected:
    /**
     * @brief Evaluates slice of the expression with offset;
     * Slice size is determined by avx register size; No checks are performed.
     *
     * @param expression
     * @param idx offset
     * @return auto evaluated complex register
     */
    template<typename I>
    static constexpr auto _cx_reg(const I& iterator, std::size_t idx)
    {
        return iterator.cx_reg(idx);
    }
    /**
     * @brief Loads data from iterator with offset.
     * No checks are performed.
     *
     * @param it iterator
     * @param idx data offset
     * @return avx::cx_reg<T> loaded complex register
     */
    template<typename T, std::size_t PackSize, bool Const>
    static constexpr auto _cx_reg(const packed_iterator<T, PackSize, Const>& iterator,
                                  std::size_t idx) -> avx::cx_reg<T>
    {
        auto real = avx::load(&(*iterator) + idx);
        auto imag = avx::load(&(*iterator) + idx + PackSize);
        return {real, imag};
    }
};

template<typename E>
concept vector_expression = expression_base::is_expression<E>::value;

template<typename E1, typename E2>
concept compatible_expressions =
    vector_expression<E1> && vector_expression<E2> &&
    std::same_as<typename E1::real_type, typename E2::real_type>;

template<typename Expression, typename Scalar>
concept compatible_scalar =
    vector_expression<Expression> &&
    (std::same_as<typename Expression::real_type, Scalar> ||
     std::same_as<std::complex<typename Expression::real_type>, Scalar>);
}    // namespace internal


// #region expression

template<typename E1, typename E2>
    requires internal::compatible_expressions<E1, E2>
auto operator+(const E1& lhs, const E2& rhs);

template<typename E1, typename E2>
    requires internal::compatible_expressions<E1, E2>
auto operator-(const E1& lhs, const E2& rhs);

template<typename E1, typename E2>
    requires internal::compatible_expressions<E1, E2>
auto operator*(const E1& lhs, const E2& rhs);

template<typename E1, typename E2>
    requires internal::compatible_expressions<E1, E2>
auto operator/(const E1& lhs, const E2& rhs);

namespace internal {

template<typename E1, typename E2>
    requires compatible_expressions<E1, E2>
class add
: public std::ranges::view_base
, private expression_base
{
    template<typename T, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<T, PackSize>
    friend class ::packed_cx_vector;
    friend class expression_base;
    friend auto operator+<E1, E2>(const E1& lhs, const E2& rhs);

public:
    using real_type = typename E1::real_type;

    class iterator : private expression_base
    {
        template<typename T, std::size_t PackSize, typename Allocator>
            requires packed_floating_point<T, PackSize>
        friend class ::packed_cx_vector;
        friend class expression_base;
        friend class add;

        using lhs_iterator = typename E1::iterator;
        using rhs_iterator = typename E2::iterator;

        iterator(lhs_iterator lhs, rhs_iterator rhs)
        : m_lhs(std::move(lhs))
        , m_rhs(std::move(rhs)){};

    public:
        using real_type       = add::real_type;
        using value_type      = const std::complex<real_type>;
        using difference_type = std::ptrdiff_t;

        using iterator_concept = std::random_access_iterator_tag;


        iterator() = default;

        iterator(const iterator& other) noexcept = default;
        iterator(iterator&& other) noexcept      = default;

        ~iterator() = default;

        iterator& operator=(const iterator& other) noexcept = default;
        iterator& operator=(iterator&& other) noexcept      = default;

        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type
        {
            return value_type(*m_lhs) + value_type(*m_rhs);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type
        {
            return value_type(*(m_lhs + idx)) + value_type(*(m_rhs + idx));
        }

        [[nodiscard]] bool operator==(const iterator& other) const noexcept
        {
            return (m_lhs == other.m_lhs);
        }
        [[nodiscard]] auto operator<=>(const iterator& other) const noexcept
        {
            return (m_lhs <=> other.m_lhs);
        }

        auto operator++() noexcept -> iterator&
        {
            ++m_lhs;
            ++m_rhs;
            return *this;
        }
        auto operator++(int) noexcept -> iterator
        {
            auto copy = *this;
            ++(*this);
            return copy;
        }
        auto operator--() noexcept -> iterator&
        {
            --m_lhs;
            --m_rhs;
        }
        auto operator--(int) noexcept -> iterator
        {
            auto copy = *this;
            --(*this);
            return copy;
        }

        auto operator+=(difference_type n) noexcept -> iterator&
        {
            m_lhs += n;
            m_rhs += n;
            return *this;
        }
        auto operator-=(difference_type n) noexcept -> iterator&
        {
            return (*this) += -n;
        }

        [[nodiscard]] friend auto operator+(iterator it, difference_type n) noexcept
            -> iterator
        {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator+(difference_type n, iterator it) noexcept
            -> iterator
        {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator it, difference_type n) noexcept
            -> iterator
        {
            it -= n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator lhs, iterator rhs) noexcept
        {
            return lhs.m_lhs - rhs.m_lhs;
        }

        [[nodiscard]] bool aligned(std::size_t offset = 0) const noexcept
        {
            return m_lhs.aligned(offset) && m_rhs.aligned(offset);
        }

    private:
        auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
        {
            const auto lhs = _cx_reg(m_lhs, idx);
            const auto rhs = _cx_reg(m_rhs, idx);

            return {avx::add(lhs.real, rhs.real), avx::add(lhs.imag, rhs.imag)};
        }

        lhs_iterator m_lhs;
        rhs_iterator m_rhs;
    };

    add() noexcept = default;

    add(add&& other) noexcept
    : m_lhs(std::move(other.m_lhs))
    , m_rhs(std::move(other.m_rhs)){};

    add(const add&) noexcept = delete;

    add& operator=(add&& other) noexcept
    {
        m_lhs = other.m_lhs;
        m_rhs = other.m_rhs;
    };

    add& operator=(const add&) noexcept = delete;

    ~add() noexcept = default;

    constexpr auto size() const noexcept -> std::size_t
    {
        return m_lhs.size();
    }

    auto operator[](std::size_t idx) const
    {
        return std::complex<real_type>(m_lhs[idx]) + std::complex<real_type>(m_rhs[idx]);
    };

    [[nodiscard]] auto begin() const noexcept -> iterator
    {
        return iterator(m_lhs.begin(), m_rhs.begin());
    }
    [[nodiscard]] auto end() const noexcept -> iterator
    {
        return iterator(m_lhs.end(), m_rhs.end());
    }

    constexpr bool aligned(std::size_t offset = 0) const noexcept
    {
        return _aligned(m_lhs, offset) && _aligned(m_rhs, offset);
    }

private:
    add(const E1& lhs, const E2& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs)
    {
        if constexpr (!is_scalar<E1>::value && !is_scalar<E2>::value)
        {
            assert(lhs.size() == rhs.size());
        };
    };

    const E1 m_lhs;
    const E2 m_rhs;
};

template<typename E1, typename E2>
    requires compatible_expressions<E1, E2>
class sub
: public std::ranges::view_base
, private expression_base
{
    template<typename T, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<T, PackSize>
    friend class ::packed_cx_vector;
    friend class expression_base;
    friend auto operator-<E1, E2>(const E1& lhs, const E2& rhs);

public:
    using real_type = typename E1::real_type;

    class iterator : private expression_base
    {
        template<typename T, std::size_t PackSize, typename Allocator>
            requires packed_floating_point<T, PackSize>
        friend class ::packed_cx_vector;
        friend class expression_base;
        friend class sub;

        using lhs_iterator = typename E1::iterator;
        using rhs_iterator = typename E2::iterator;

        iterator(lhs_iterator lhs, rhs_iterator rhs)
        : m_lhs(std::move(lhs))
        , m_rhs(std::move(rhs)){};

    public:
        using real_type       = sub::real_type;
        using value_type      = const std::complex<real_type>;
        using difference_type = std::ptrdiff_t;

        using iterator_concept = std::random_access_iterator_tag;


        iterator() = default;

        iterator(const iterator& other) noexcept = default;
        iterator(iterator&& other) noexcept      = default;

        ~iterator() = default;

        iterator& operator=(const iterator& other) noexcept = default;
        iterator& operator=(iterator&& other) noexcept      = default;

        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type
        {
            return value_type(*m_lhs) - value_type(*m_rhs);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type
        {
            return value_type(*(m_lhs + idx)) - value_type(*(m_rhs + idx));
        }

        [[nodiscard]] bool operator==(const iterator& other) const noexcept
        {
            return (m_lhs == other.m_lhs);
        }
        [[nodiscard]] auto operator<=>(const iterator& other) const noexcept
        {
            return (m_lhs <=> other.m_lhs);
        }

        auto operator++() noexcept -> iterator&
        {
            ++m_lhs;
            ++m_rhs;
            return *this;
        }
        auto operator++(int) noexcept -> iterator
        {
            auto copy = *this;
            ++(*this);
            return copy;
        }
        auto operator--() noexcept -> iterator&
        {
            --m_lhs;
            --m_rhs;
        }
        auto operator--(int) noexcept -> iterator
        {
            auto copy = *this;
            --(*this);
            return copy;
        }

        auto operator+=(difference_type n) noexcept -> iterator&
        {
            m_lhs += n;
            m_rhs += n;
            return *this;
        }
        auto operator-=(difference_type n) noexcept -> iterator&
        {
            return (*this) += -n;
        }

        [[nodiscard]] friend auto operator+(iterator it, difference_type n) noexcept
            -> iterator
        {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator+(difference_type n, iterator it) noexcept
            -> iterator
        {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator it, difference_type n) noexcept
            -> iterator
        {
            it -= n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator lhs, iterator rhs) noexcept
        {
            return lhs.m_lhs - rhs.m_lhs;
        }

        [[nodiscard]] bool aligned(std::size_t offset = 0) const noexcept
        {
            return m_lhs.aligned(offset) && m_rhs.aligned(offset);
        }

    private:
        auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
        {
            const auto lhs = _cx_reg(m_lhs, idx);
            const auto rhs = _cx_reg(m_rhs, idx);

            return {avx::sub(lhs.real, rhs.real), avx::sub(lhs.imag, rhs.imag)};
        }

        lhs_iterator m_lhs;
        rhs_iterator m_rhs;
    };

    sub() noexcept = default;

    sub(sub&& other) noexcept
    : m_lhs(std::move(other.m_lhs))
    , m_rhs(std::move(other.m_rhs)){};

    sub(const sub&) noexcept = delete;

    sub& operator=(sub&& other) noexcept
    {
        m_lhs = other.m_lhs;
        m_rhs = other.m_rhs;
    };

    sub& operator=(const sub&) noexcept = delete;

    ~sub() noexcept = default;

    constexpr auto size() const noexcept -> std::size_t
    {
        return m_lhs.size();
    }

    auto operator[](std::size_t idx) const
    {
        return std::complex<real_type>(m_lhs[idx]) - std::complex<real_type>(m_rhs[idx]);
    };

    [[nodiscard]] auto begin() const noexcept -> iterator
    {
        return iterator(m_lhs.begin(), m_rhs.begin());
    }
    [[nodiscard]] auto end() const noexcept -> iterator
    {
        return iterator(m_lhs.end(), m_rhs.end());
    }

    constexpr bool aligned(std::size_t offset = 0) const noexcept
    {
        return _aligned(m_lhs, offset) && _aligned(m_rhs, offset);
    }

private:
    sub(const E1& lhs, const E2& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs)
    {
        if constexpr (!is_scalar<E1>::value && !is_scalar<E2>::value)
        {
            assert(lhs.size() == rhs.size());
        };
    };

    const E1 m_lhs;
    const E2 m_rhs;
};

template<typename E1, typename E2>
    requires compatible_expressions<E1, E2>
class mul
: public std::ranges::view_base
, private expression_base
{
    template<typename T, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<T, PackSize>
    friend class ::packed_cx_vector;
    friend class expression_base;
    friend auto operator*<E1, E2>(const E1& lhs, const E2& rhs);

public:
    using real_type = typename E1::real_type;

    class iterator : private expression_base
    {
        template<typename T, std::size_t PackSize, typename Allocator>
            requires packed_floating_point<T, PackSize>
        friend class ::packed_cx_vector;
        friend class expression_base;
        friend class mul;

        using lhs_iterator = typename E1::iterator;
        using rhs_iterator = typename E2::iterator;

        iterator(lhs_iterator lhs, rhs_iterator rhs)
        : m_lhs(std::move(lhs))
        , m_rhs(std::move(rhs)){};

    public:
        using real_type       = mul::real_type;
        using value_type      = const std::complex<real_type>;
        using difference_type = std::ptrdiff_t;

        using iterator_concept = std::random_access_iterator_tag;


        iterator() = default;

        iterator(const iterator& other) noexcept = default;
        iterator(iterator&& other) noexcept      = default;

        ~iterator() = default;

        iterator& operator=(const iterator& other) noexcept = default;
        iterator& operator=(iterator&& other) noexcept      = default;

        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type
        {
            return value_type(*m_lhs) * value_type(*m_rhs);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type
        {
            return value_type(*(m_lhs + idx)) * value_type(*(m_rhs + idx));
        }

        [[nodiscard]] bool operator==(const iterator& other) const noexcept
        {
            return (m_lhs == other.m_lhs);
        }
        [[nodiscard]] auto operator<=>(const iterator& other) const noexcept
        {
            return (m_lhs <=> other.m_lhs);
        }

        auto operator++() noexcept -> iterator&
        {
            ++m_lhs;
            ++m_rhs;
            return *this;
        }
        auto operator++(int) noexcept -> iterator
        {
            auto copy = *this;
            ++(*this);
            return copy;
        }
        auto operator--() noexcept -> iterator&
        {
            --m_lhs;
            --m_rhs;
        }
        auto operator--(int) noexcept -> iterator
        {
            auto copy = *this;
            --(*this);
            return copy;
        }

        auto operator+=(difference_type n) noexcept -> iterator&
        {
            m_lhs += n;
            m_rhs += n;
            return *this;
        }
        auto operator-=(difference_type n) noexcept -> iterator&
        {
            return (*this) += -n;
        }

        [[nodiscard]] friend auto operator+(iterator it, difference_type n) noexcept
            -> iterator
        {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator+(difference_type n, iterator it) noexcept
            -> iterator
        {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator it, difference_type n) noexcept
            -> iterator
        {
            it -= n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator lhs, iterator rhs) noexcept
        {
            return lhs.m_lhs - rhs.m_lhs;
        }

        [[nodiscard]] bool aligned(std::size_t offset = 0) const noexcept
        {
            return m_lhs.aligned(offset) && m_rhs.aligned(offset);
        }

    private:
        auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
        {
            const auto lhs = _cx_reg(m_lhs, idx);
            const auto rhs = _cx_reg(m_rhs, idx);
            return avx::mul(lhs, rhs);
        }

        lhs_iterator m_lhs;
        rhs_iterator m_rhs;
    };

    mul() noexcept = default;

    mul(mul&& other) noexcept
    : m_lhs(std::move(other.m_lhs))
    , m_rhs(std::move(other.m_rhs)){};

    mul(const mul&) noexcept = delete;

    mul& operator=(mul&& other) noexcept
    {
        m_lhs = other.m_lhs;
        m_rhs = other.m_rhs;
    };

    mul& operator=(const mul&) noexcept = delete;

    ~mul() noexcept = default;

    constexpr auto size() const noexcept -> std::size_t
    {
        return m_lhs.size();
    }

    auto operator[](std::size_t idx) const
    {
        return std::complex<real_type>(m_lhs[idx]) * std::complex<real_type>(m_rhs[idx]);
    };

    [[nodiscard]] auto begin() const noexcept -> iterator
    {
        return iterator(m_lhs.begin(), m_rhs.begin());
    }
    [[nodiscard]] auto end() const noexcept -> iterator
    {
        return iterator(m_lhs.end(), m_rhs.end());
    }

    constexpr bool aligned(std::size_t offset = 0) const noexcept
    {
        return _aligned(m_lhs, offset) && _aligned(m_rhs, offset);
    }

private:
    mul(const E1& lhs, const E2& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs)
    {
        if constexpr (!is_scalar<E1>::value && !is_scalar<E2>::value)
        {
            assert(lhs.size() == rhs.size());
        };
    };

    const E1 m_lhs;
    const E2 m_rhs;
};

template<typename E1, typename E2>
    requires compatible_expressions<E1, E2>
class div
: public std::ranges::view_base
, private expression_base
{
    template<typename T, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<T, PackSize>
    friend class ::packed_cx_vector;
    friend class expression_base;
    friend auto operator/<E1, E2>(const E1& lhs, const E2& rhs);

public:
    using real_type = typename E1::real_type;

    class iterator : private expression_base
    {
        template<typename T, std::size_t PackSize, typename Allocator>
            requires packed_floating_point<T, PackSize>
        friend class ::packed_cx_vector;
        friend class expression_base;
        friend class div;

        using lhs_iterator = typename E1::iterator;
        using rhs_iterator = typename E2::iterator;

        iterator(lhs_iterator lhs, rhs_iterator rhs)
        : m_lhs(std::move(lhs))
        , m_rhs(std::move(rhs)){};

    public:
        using real_type       = div::real_type;
        using value_type      = const std::complex<real_type>;
        using difference_type = std::ptrdiff_t;

        using iterator_concept = std::random_access_iterator_tag;


        iterator() = default;

        iterator(const iterator& other) noexcept = default;
        iterator(iterator&& other) noexcept      = default;

        ~iterator() = default;

        iterator& operator=(const iterator& other) noexcept = default;
        iterator& operator=(iterator&& other) noexcept      = default;

        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type
        {
            return value_type(*m_lhs) / value_type(*m_rhs);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type
        {
            return value_type(*(m_lhs + idx)) / value_type(*(m_rhs + idx));
        }

        [[nodiscard]] bool operator==(const iterator& other) const noexcept
        {
            return (m_lhs == other.m_lhs);
        }
        [[nodiscard]] auto operator<=>(const iterator& other) const noexcept
        {
            return (m_lhs <=> other.m_lhs);
        }

        auto operator++() noexcept -> iterator&
        {
            ++m_lhs;
            ++m_rhs;
            return *this;
        }
        auto operator++(int) noexcept -> iterator
        {
            auto copy = *this;
            ++(*this);
            return copy;
        }
        auto operator--() noexcept -> iterator&
        {
            --m_lhs;
            --m_rhs;
        }
        auto operator--(int) noexcept -> iterator
        {
            auto copy = *this;
            --(*this);
            return copy;
        }

        auto operator+=(difference_type n) noexcept -> iterator&
        {
            m_lhs += n;
            m_rhs += n;
            return *this;
        }
        auto operator-=(difference_type n) noexcept -> iterator&
        {
            return (*this) += -n;
        }

        [[nodiscard]] friend auto operator+(iterator it, difference_type n) noexcept
            -> iterator
        {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator+(difference_type n, iterator it) noexcept
            -> iterator
        {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator it, difference_type n) noexcept
            -> iterator
        {
            it -= n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator lhs, iterator rhs) noexcept
        {
            return lhs.m_lhs - rhs.m_lhs;
        }

        [[nodiscard]] bool aligned(std::size_t offset = 0) const noexcept
        {
            return m_lhs.aligned(offset) && m_rhs.aligned(offset);
        }

    private:
        auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
        {
            const auto lhs = _cx_reg(m_lhs, idx);
            const auto rhs = _cx_reg(m_rhs, idx);

            const auto rhs_abs =
                avx::add(avx::mul(rhs.real, rhs.real), avx::mul(rhs.imag, rhs.imag));

            const auto real_ =
                avx::add(avx::mul(lhs.real, rhs.real), avx::mul(lhs.imag, rhs.imag));

            const auto imag_ =
                avx::sub(avx::mul(lhs.imag, rhs.real), avx::mul(lhs.real, rhs.imag));

            return {avx::div(real_, rhs_abs), avx::div(imag_, rhs_abs)};
        }

        lhs_iterator m_lhs;
        rhs_iterator m_rhs;
    };

    div() noexcept = default;

    div(div&& other) noexcept
    : m_lhs(std::move(other.m_lhs))
    , m_rhs(std::move(other.m_rhs)){};

    div(const div&) noexcept = delete;

    div& operator=(div&& other) noexcept
    {
        m_lhs = other.m_lhs;
        m_rhs = other.m_rhs;
    };

    div& operator=(const div&) noexcept = delete;

    ~div() noexcept = default;

    constexpr auto size() const noexcept -> std::size_t
    {
        return m_lhs.size();
    }

    auto operator[](std::size_t idx) const
    {
        return std::complex<real_type>(m_lhs[idx]) / std::complex<real_type>(m_rhs[idx]);
    };

    [[nodiscard]] auto begin() const noexcept -> iterator
    {
        return iterator(m_lhs.begin(), m_rhs.begin());
    }
    [[nodiscard]] auto end() const noexcept -> iterator
    {
        return iterator(m_lhs.end(), m_rhs.end());
    }

    constexpr bool aligned(std::size_t offset = 0) const noexcept
    {
        return _aligned(m_lhs, offset) && _aligned(m_rhs, offset);
    }

private:
    div(const E1& lhs, const E2& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs)
    {
        if constexpr (!is_scalar<E1>::value && !is_scalar<E2>::value)
        {
            assert(lhs.size() == rhs.size());
        };
    };

    const E1 m_lhs;
    const E2 m_rhs;
};

}    // namespace internal

template<typename E1, typename E2>
    requires internal::compatible_expressions<E1, E2>
auto operator+(const E1& lhs, const E2& rhs)
{
    return internal::add(lhs, rhs);
};

template<typename E1, typename E2>
    requires internal::compatible_expressions<E1, E2>
auto operator-(const E1& lhs, const E2& rhs)
{
    return internal::sub(lhs, rhs);
};

template<typename E1, typename E2>
    requires internal::compatible_expressions<E1, E2>
auto operator*(const E1& lhs, const E2& rhs)
{
    return internal::mul(lhs, rhs);
};

template<typename E1, typename E2>
    requires internal::compatible_expressions<E1, E2>
auto operator/(const E1& lhs, const E2& rhs)
{
    return internal::div(lhs, rhs);
};

// #endregion expression

// #region scalar

template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
auto operator+(const E& vector, S scalar);
template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
auto operator+(S scalar, const E& vector);

template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
auto operator-(const E& vector, S scalar);
template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
auto operator-(S scalar, const E& vector);

template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
auto operator*(const E& vector, S scalar);
template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
auto operator*(S scalar, const E& vector);

template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
auto operator/(const E& vector, S scalar);
template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
auto operator/(S scalar, const E& vector);

namespace internal {

template<typename E, typename S>
    requires compatible_scalar<E, S>
class scalar_add
: public std::ranges::view_base
, private expression_base
{
    template<typename T, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<T, PackSize>
    friend class ::packed_cx_vector;
    friend class expression_base;

    friend auto ::operator+(const E& vector, S scalar);
    friend auto ::operator+(S scalar, const E& vector);

public:
    using real_type = typename E::real_type;

    class iterator : private expression_base
    {
        template<typename T, std::size_t PackSize, typename Allocator>
            requires packed_floating_point<T, PackSize>
        friend class ::packed_cx_vector;
        friend class expression_base;
        friend class scalar_add;


        using vector_iterator = typename E::iterator;

        iterator(S scalar, vector_iterator vector)
        : m_vector(std::move(vector))
        , m_scalar(std::move(scalar)){};

    public:
        using real_type       = scalar_add::real_type;
        using value_type      = const std::complex<real_type>;
        using difference_type = std::ptrdiff_t;

        using iterator_concept = std::random_access_iterator_tag;

        iterator() = default;

        iterator(const iterator& other) noexcept = default;
        iterator(iterator&& other) noexcept      = default;

        ~iterator() = default;

        iterator& operator=(const iterator& other) noexcept = default;
        iterator& operator=(iterator&& other) noexcept      = default;

        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type
        {
            return m_scalar + value_type(*m_vector);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type
        {
            return m_scalar + value_type(*(m_vector + idx));
        }

        [[nodiscard]] bool operator==(const iterator& other) const noexcept
        {
            return (m_vector == other.m_vector);
        }
        [[nodiscard]] auto operator<=>(const iterator& other) const noexcept
        {
            return (m_vector <=> other.m_vector);
        }

        auto operator++() noexcept -> iterator&
        {
            ++m_vector;
            return *this;
        }
        auto operator++(int) noexcept -> iterator
        {
            auto copy = *this;
            ++(*this);
            return copy;
        }
        auto operator--() noexcept -> iterator&
        {
            --m_vector;
        }
        auto operator--(int) noexcept -> iterator
        {
            auto copy = *this;
            --(*this);
            return copy;
        }

        auto operator+=(difference_type n) noexcept -> iterator&
        {
            m_vector += n;
            return *this;
        }
        auto operator-=(difference_type n) noexcept -> iterator&
        {
            return (*this) += -n;
        }

        [[nodiscard]] friend auto operator+(iterator it, difference_type n) noexcept
            -> iterator
        {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator+(difference_type n, iterator it) noexcept
            -> iterator
        {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator it, difference_type n) noexcept
            -> iterator
        {
            it -= n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator lhs, iterator rhs) noexcept
        {
            return lhs.m_vector - rhs.m_vector;
        }

        [[nodiscard]] bool aligned(std::size_t offset = 0) const noexcept
        {
            return m_vector.aligned(offset);
        }

    private:
        auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
        {
            const auto scalar = avx::broadcast(m_scalar);
            const auto vector = _cx_reg(m_vector, idx);

            return avx::add(scalar, vector);
        }

        vector_iterator m_vector;
        S               m_scalar;
    };

    scalar_add() noexcept = default;

    scalar_add(scalar_add&& other) noexcept
    : m_scalar(std::move(other.m_scalar))
    , m_vector(std::move(other.m_vector)){};

    scalar_add(const scalar_add&) noexcept = delete;

    scalar_add& operator=(scalar_add&& other) noexcept
    {
        m_scalar = other.m_scalar;
        m_vector = other.m_vector;
    };

    scalar_add& operator=(const scalar_add&) noexcept = delete;

    ~scalar_add() noexcept = default;

    constexpr auto size() const noexcept -> std::size_t
    {
        return m_vector.size();
    }

    auto operator[](std::size_t idx) const
    {
        return m_scalar + std::complex<real_type>(m_vector[idx]);
    };

    [[nodiscard]] auto begin() const noexcept -> iterator
    {
        return iterator(m_scalar, m_vector.begin());
    }
    [[nodiscard]] auto end() const noexcept -> iterator
    {
        return iterator(m_scalar, m_vector.end());
    }

    constexpr bool aligned(std::size_t offset = 0) const noexcept
    {
        return _aligned(m_vector, offset);
    }

private:
    scalar_add(S scalar, const E& vector)
    : m_vector(vector)
    , m_scalar(scalar){};

    const E m_vector;
    S       m_scalar;
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class scalar_sub
: public std::ranges::view_base
, private expression_base
{
    template<typename T, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<T, PackSize>
    friend class ::packed_cx_vector;
    friend class expression_base;

    friend auto ::operator+(const E& vector, S scalar);
    friend auto ::operator+(S scalar, const E& vector);

public:
    using real_type = typename E::real_type;

    class iterator : private expression_base
    {
        template<typename T, std::size_t PackSize, typename Allocator>
            requires packed_floating_point<T, PackSize>
        friend class ::packed_cx_vector;
        friend class expression_base;
        friend class scalar_sub;


        using vector_iterator = typename E::iterator;

        iterator(S scalar, vector_iterator vector)
        : m_vector(std::move(vector))
        , m_scalar(std::move(scalar)){};

    public:
        using real_type       = scalar_sub::real_type;
        using value_type      = const std::complex<real_type>;
        using difference_type = std::ptrdiff_t;

        using iterator_concept = std::random_access_iterator_tag;

        iterator() = default;

        iterator(const iterator& other) noexcept = default;
        iterator(iterator&& other) noexcept      = default;

        ~iterator() = default;

        iterator& operator=(const iterator& other) noexcept = default;
        iterator& operator=(iterator&& other) noexcept      = default;

        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type
        {
            return m_scalar - value_type(*m_vector);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type
        {
            return m_scalar - value_type(*(m_vector + idx));
        }

        [[nodiscard]] bool operator==(const iterator& other) const noexcept
        {
            return (m_vector == other.m_vector);
        }
        [[nodiscard]] auto operator<=>(const iterator& other) const noexcept
        {
            return (m_vector <=> other.m_vector);
        }

        auto operator++() noexcept -> iterator&
        {
            ++m_vector;
            return *this;
        }
        auto operator++(int) noexcept -> iterator
        {
            auto copy = *this;
            ++(*this);
            return copy;
        }
        auto operator--() noexcept -> iterator&
        {
            --m_vector;
        }
        auto operator--(int) noexcept -> iterator
        {
            auto copy = *this;
            --(*this);
            return copy;
        }

        auto operator+=(difference_type n) noexcept -> iterator&
        {
            m_vector += n;
            return *this;
        }
        auto operator-=(difference_type n) noexcept -> iterator&
        {
            return (*this) += -n;
        }

        [[nodiscard]] friend auto operator+(iterator it, difference_type n) noexcept
            -> iterator
        {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator+(difference_type n, iterator it) noexcept
            -> iterator
        {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator it, difference_type n) noexcept
            -> iterator
        {
            it -= n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator lhs, iterator rhs) noexcept
        {
            return lhs.m_vector - rhs.m_vector;
        }

        [[nodiscard]] bool aligned(std::size_t offset = 0) const noexcept
        {
            return m_vector.aligned(offset);
        }

    private:
        auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
        {
            const auto scalar = avx::broadcast(m_scalar);
            const auto vector = _cx_reg(m_vector, idx);

            return avx::sub(scalar, vector);
        }

        vector_iterator m_vector;
        S               m_scalar;
    };

    scalar_sub() noexcept = default;

    scalar_sub(scalar_sub&& other) noexcept
    : m_scalar(std::move(other.m_scalar))
    , m_vector(std::move(other.m_vector)){};

    scalar_sub(const scalar_sub&) noexcept = delete;

    scalar_sub& operator=(scalar_sub&& other) noexcept
    {
        m_scalar = other.m_scalar;
        m_vector = other.m_vector;
    };

    scalar_sub& operator=(const scalar_sub&) noexcept = delete;

    ~scalar_sub() noexcept = default;

    constexpr auto size() const noexcept -> std::size_t
    {
        return m_vector.size();
    }

    auto operator[](std::size_t idx) const
    {
        return m_scalar - std::complex<real_type>(m_vector[idx]);
    };

    [[nodiscard]] auto begin() const noexcept -> iterator
    {
        return iterator(m_scalar, m_vector.begin());
    }
    [[nodiscard]] auto end() const noexcept -> iterator
    {
        return iterator(m_scalar, m_vector.end());
    }

    constexpr bool aligned(std::size_t offset = 0) const noexcept
    {
        return _aligned(m_vector, offset);
    }

private:
    scalar_sub(S scalar, const E& vector)
    : m_vector(vector)
    , m_scalar(scalar){};

    const E m_vector;
    S       m_scalar;
};


}    // namespace internal


template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
auto operator+(const E& vector, S scalar)
{
    return internal::scalar_add(scalar, vector);
}
template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
auto operator+(S scalar, const E& vector)
{
    return internal::scalar_add(scalar, vector);
}

// template<typename E>
//     requires internal::vector_expression<E>
// auto operator+(typename E::real_type lhs, const E& rhs)
// {
//     return internal::scalar_add(lhs, rhs);
// }
//
// template<typename E>
//     requires internal::vector_expression<E>
// auto operator-(const E& lhs, typename E::real_type rhs)
// {
//     return internal::scalar_add(-rhs, lhs);
// }
// template<typename E>
//     requires internal::vector_expression<E>
// auto operator-(typename E::real_type lhs, const E& rhs)
// {
//     return internal::rscalar_sub(lhs, rhs);
// }
//
// template<typename E>
//     requires internal::vector_expression<E>
// auto operator*(const E& lhs, typename E::real_type rhs)
// {
//     return internal::rscalar_mul(rhs, lhs);
// }
// template<typename E>
//     requires internal::vector_expression<E>
// auto operator*(typename E::real_type lhs, const E& rhs)
// {
//     return internal::rscalar_mul(lhs, rhs);
// }
//
// template<typename E>
//     requires internal::vector_expression<E>
// auto operator/(const E& lhs, typename E::real_type rhs)
// {
//     return internal::rscalar_mul(1 / rhs, lhs);
// }
// template<typename E>
//     requires internal::vector_expression<E>
// auto operator/(typename E::real_type lhs, const E& rhs)
// {
//     return internal::rscalar_div(lhs, rhs);
// }

// #endregion scalar

// #region vector

template<typename E, typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize> &&
             (!std::same_as<E, packed_cx_vector<T, PackSize, Allocator>>) &&
             requires(packed_subrange<T, PackSize> subrange, E e) { e + subrange; }
auto operator+(const E&                                        expression,
               const packed_cx_vector<T, PackSize, Allocator>& vector)
{
    return expression + packed_subrange(vector.begin(), vector.size());
};
template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(packed_subrange<T, PackSize> subrange, E e) { subrange + e; }
auto operator+(const packed_cx_vector<T, PackSize, Allocator>& vector,
               const E&                                        expression)
{
    return packed_subrange(vector.begin(), vector.size()) + expression;
};

template<typename E, typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize> &&
             (!std::same_as<E, packed_cx_vector<T, PackSize, Allocator>>) &&
             requires(packed_subrange<T, PackSize> subrange, E e) { e - subrange; }
auto operator-(const E&                                        expression,
               const packed_cx_vector<T, PackSize, Allocator>& vector)
{
    return expression - packed_subrange(vector.begin(), vector.size());
};
template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(packed_subrange<T, PackSize> subrange, E e) { subrange - e; }
auto operator-(const packed_cx_vector<T, PackSize, Allocator>& vector,
               const E&                                        expression)
{
    return packed_subrange(vector.begin(), vector.size()) - expression;
};

template<typename E, typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize> &&
             (!std::same_as<E, packed_cx_vector<T, PackSize, Allocator>>) &&
             requires(packed_subrange<T, PackSize> subrange, E e) { e* subrange; }
auto operator*(const E&                                        expression,
               const packed_cx_vector<T, PackSize, Allocator>& vector)
{
    return expression * packed_subrange(vector.begin(), vector.size());
};
template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(packed_subrange<T, PackSize> subrange, E e) { subrange* e; }
auto operator*(const packed_cx_vector<T, PackSize, Allocator>& vector,
               const E&                                        expression)
{
    return packed_subrange(vector.begin(), vector.size()) * expression;
};

template<typename E, typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize> &&
             (!std::same_as<E, packed_cx_vector<T, PackSize, Allocator>>) &&
             requires(packed_subrange<T, PackSize> subrange, E e) { e / subrange; }
auto operator/(const E&                                        expression,
               const packed_cx_vector<T, PackSize, Allocator>& vector)
{
    return expression / packed_subrange(vector.begin(), vector.size());
};
template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(packed_subrange<T, PackSize> subrange, E e) { subrange / e; }
auto operator/(const packed_cx_vector<T, PackSize, Allocator>& vector,
               const E&                                        expression)
{
    return packed_subrange(vector.begin(), vector.size()) / expression;
};

    // #endregion vector

#endif