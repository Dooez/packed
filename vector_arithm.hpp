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
template<typename T>
inline auto add(typename reg<T>::type lhs, typename reg<T>::type rhs) ->
    typename reg<T>::type;
template<>
inline auto add<float>(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
{
    return _mm256_add_ps(lhs, rhs);
}
template<>
inline auto add<double>(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
{
    return _mm256_add_pd(lhs, rhs);
}

template<typename T>
inline auto sub(typename reg<T>::type lhs, typename reg<T>::type rhs) ->
    typename reg<T>::type;
template<>
inline auto sub<float>(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
{
    return _mm256_sub_ps(lhs, rhs);
}
template<>
inline auto sub<double>(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
{
    return _mm256_sub_pd(lhs, rhs);
}

template<typename T>
inline auto mul(typename reg<T>::type lhs, typename reg<T>::type rhs) ->
    typename reg<T>::type;
template<>
inline auto mul<float>(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
{
    return _mm256_mul_ps(lhs, rhs);
}
template<>
inline auto mul<double>(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
{
    return _mm256_mul_pd(lhs, rhs);
}

template<typename T>
inline auto div(typename reg<T>::type lhs, typename reg<T>::type rhs) ->
    typename reg<T>::type;
template<>
inline auto div<float>(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
{
    return _mm256_div_ps(lhs, rhs);
}
template<>
inline auto div<double>(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
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
        avx::sub<T>(avx::mul<T>(lhs.real, rhs.real), avx::mul<T>(lhs.imag, rhs.imag));
    const auto imag =
        avx::add<T>(avx::mul<T>(lhs.real, rhs.imag), avx::mul<T>(lhs.imag, rhs.real));

    return {real, imag};
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
                typename E::real_type;

                typename E::iterator;

                std::ranges::view<E>;

                requires requires(typename E::iterator iter) {
                             {
                                 iter.aligned(idx)
                                 } -> std::same_as<bool>;

                             requires std::convertible_to<
                                 std::iter_value_t<typename E::iterator>,
                                 std::complex<typename E::real_type>>;
                         };

                {
                    expression[idx]
                    } -> std::convertible_to<std::complex<typename E::real_type>>;

                {
                    expression.size()
                    } -> std::same_as<std::size_t>;
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

            return {avx::add<real_type>(lhs.real, rhs.real),
                    avx::add<real_type>(lhs.imag, rhs.imag)};
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

            return {avx::sub<real_type>(lhs.real, rhs.real),
                    avx::sub<real_type>(lhs.imag, rhs.imag)};
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
                avx::add<real_type>(avx::mul<real_type>(rhs.real, rhs.real),
                                    avx::mul<real_type>(rhs.imag, rhs.imag));

            const auto real_ =
                avx::add<real_type>(avx::mul<real_type>(lhs.real, rhs.real),
                                    avx::mul<real_type>(lhs.imag, rhs.imag));

            const auto imag_ =
                avx::sub<real_type>(avx::mul<real_type>(lhs.imag, rhs.real),
                                    avx::mul<real_type>(lhs.real, rhs.imag));

            return {avx::div<real_type>(real_, rhs_abs),
                    avx::div<real_type>(imag_, rhs_abs)};
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

// #region complex scalar

template<typename E>
    requires internal::vector_expression<E>
auto operator+(const E& lhs, std::complex<typename E::real_type> rhs);
template<typename E>
    requires internal::vector_expression<E>
auto operator+(std::complex<typename E::real_type> lhs, const E& rhs);

template<typename E>
    requires internal::vector_expression<E>
auto operator-(const E& lhs, std::complex<typename E::real_type> rhs);
template<typename E>
    requires internal::vector_expression<E>
auto operator-(std::complex<typename E::real_type> lhs, const E& rhs);
template<typename E>
    requires internal::vector_expression<E>

auto operator*(const E& lhs, std::complex<typename E::real_type> rhs);
template<typename E>
    requires internal::vector_expression<E>
auto operator*(std::complex<typename E::real_type> lhs, const E& rhs);

template<typename E>
    requires internal::vector_expression<E>
auto operator/(const E& lhs, std::complex<typename E::real_type> rhs);
template<typename E>
    requires internal::vector_expression<E>
auto operator/(std::complex<typename E::real_type> lhs, const E& rhs);

namespace internal {

template<typename E>
    requires vector_expression<E>
class packed_scalar : private expression_base
{
public:
    using real_type = typename E::real_type;

private:
    template<typename Tvec, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<Tvec, PackSize>
    friend class packed_cx_vector;
    friend class expression_base;

    friend auto operator+<E>(const E& lhs, std::complex<real_type> rhs);
    friend auto operator+<E>(std::complex<real_type> lhs, const E& rhs);
    friend auto operator-<E>(const E& lhs, std::complex<real_type> rhs);
    friend auto operator-<E>(std::complex<real_type> lhs, const E& rhs);
    friend auto operator*<E>(const E& lhs, std::complex<real_type> rhs);
    friend auto operator*<E>(std::complex<real_type> lhs, const E& rhs);
    friend auto operator/<E>(const E& lhs, std::complex<real_type> rhs);
    friend auto operator/<E>(std::complex<real_type> lhs, const E& rhs);

    packed_scalar(std::complex<real_type> value)
    : m_value(value){};

    constexpr bool aligned(long idx = 0) const
    {
        return true;
    }

    auto operator[](std::size_t idx) const -> std::complex<real_type>
    {
        return m_value;
    };

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
    {
        const auto& value = reinterpret_cast<const real_type(&)[2]>(m_value);
        return {avx::broadcast(&(value[0])), avx::broadcast(&(value[1]))};
    };

    const std::complex<real_type> m_value;
};

template<typename E>
struct is_scalar<packed_scalar<E>>
{
    static constexpr bool value = true;
};

}    // namespace internal

template<typename E>
    requires internal::vector_expression<E>
auto operator+(const E& vector, std::complex<typename E::real_type> scalar)
{
    return vector + internal::packed_scalar<E>(scalar);
}
template<typename E>
    requires internal::vector_expression<E>
auto operator+(std::complex<typename E::real_type> scalar, const E& vector)
{
    return internal::packed_scalar<E>(scalar) + vector;
}

template<typename E>
    requires internal::vector_expression<E>
auto operator-(const E& vector, std::complex<typename E::real_type> scalar)
{
    return vector - internal::packed_scalar<E>(scalar);
}
template<typename E>
    requires internal::vector_expression<E>
auto operator-(std::complex<typename E::real_type> scalar, const E& vector)
{
    return internal::packed_scalar<E>(scalar) - vector;
}

template<typename E>
    requires internal::vector_expression<E>
auto operator*(const E& vector, std::complex<typename E::real_type> scalar)
{
    return vector * internal::packed_scalar<E>(scalar);
}
template<typename E>
    requires internal::vector_expression<E>
auto operator*(std::complex<typename E::real_type> scalar, const E& vector)
{
    return internal::packed_scalar<E>(scalar) * vector;
}

template<typename E>
    requires internal::vector_expression<E>
auto operator/(const E& vector, std::complex<typename E::real_type> scalar)
{
    return vector / internal::packed_scalar<E>(scalar);
}
template<typename E>
    requires internal::vector_expression<E>
auto operator/(std::complex<typename E::real_type> scalar, const E& vector)
{
    return internal::packed_scalar<E>(scalar) / vector;
}

// #endregion complex scalar

// #region real scalar

template<typename E>
    requires internal::vector_expression<E>
auto operator+(const E& lhs, typename E::real_type rhs);
template<typename E>
    requires internal::vector_expression<E>
auto operator+(typename E::real_type lhs, const E& rhs);

template<typename E>
    requires internal::vector_expression<E>
auto operator-(const E& lhs, typename E::real_type rhs);
template<typename E>
    requires internal::vector_expression<E>
auto operator-(typename E::real_type lhs, const E& rhs);

template<typename E>
    requires internal::vector_expression<E>
auto operator*(const E& lhs, typename E::real_type rhs);
template<typename E>
    requires internal::vector_expression<E>
auto operator*(typename E::real_type lhs, const E& rhs);

template<typename E>
    requires internal::vector_expression<E>
auto operator/(const E& lhs, typename E::real_type rhs);
template<typename E>
    requires internal::vector_expression<E>
auto operator/(typename E::real_type lhs, const E& rhs);

namespace internal {

template<typename E>
    requires vector_expression<E>
class rscalar_add : private expression_base
{
public:
    using real_type = typename E::real_type;

    constexpr auto size() const -> std::size_t
    {
        return _size(m_vector);
    }

private:
    template<typename T, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<T, PackSize>
    friend class ::packed_cx_vector;
    friend class expression_base;

    friend auto operator+<E>(const E& lhs, typename E::real_type rhs);
    friend auto operator+<E>(typename E::real_type lhs, const E& rhs);
    friend auto operator-<E>(const E& lhs, typename E::real_type rhs);

    rscalar_add(real_type scalar, const E& vector)
    : m_scalar(scalar)
    , m_vector(vector){};

    constexpr bool aligned(std::size_t offset = 0) const
    {
        return _aligned(m_vector, offset);
    }

    auto operator[](std::size_t idx) const
    {
        return m_scalar + std::complex<real_type>(_element(m_vector, idx));
    };

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
    {
        const auto scalar = avx::broadcast(&m_scalar);
        const auto vector = _cx_reg(m_vector, idx);

        return {avx::add<real_type>(scalar, vector.real), vector.imag};
    };

    const E   m_vector;
    real_type m_scalar;
};

template<typename E>
    requires vector_expression<E>
class rscalar_sub : private expression_base
{
public:
    using real_type = typename E::real_type;

    constexpr auto size() const -> std::size_t
    {
        return _size(m_vector);
    }

private:
    template<typename T, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<T, PackSize>
    friend class ::packed_cx_vector;
    friend class expression_base;

    friend auto operator-<E>(typename E::real_type lhs, const E& rhs);

    rscalar_sub(real_type scalar, const E& vector)
    : m_scalar(scalar)
    , m_vector(vector){};

    constexpr bool aligned(std::size_t offset = 0) const
    {
        return _aligned(m_vector, offset);
    }

    auto operator[](std::size_t idx) const
    {
        return m_scalar - std::complex<real_type>(_element(m_vector, idx));
    };

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
    {
        const auto scalar = avx::broadcast(&m_scalar);
        const auto vector = _cx_reg(m_vector, idx);

        return {avx::sub<real_type>(scalar, vector.real), vector.imag};
    };

    const E   m_vector;
    real_type m_scalar;
};

template<typename E>
    requires vector_expression<E>
class rscalar_mul : private expression_base
{
public:
    using real_type = typename E::real_type;

    constexpr auto size() const -> std::size_t
    {
        return _size(m_vector);
    }

private:
    template<typename T, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<T, PackSize>
    friend class ::packed_cx_vector;
    friend class expression_base;

    friend auto operator*<E>(const E& lhs, typename E::real_type rhs);
    friend auto operator*<E>(typename E::real_type lhs, const E& rhs);
    friend auto operator/<E>(const E& lhs, typename E::real_type rhs);

    rscalar_mul(real_type scalar, const E& vector)
    : m_vector(vector)
    , m_scalar(scalar){};

    constexpr bool aligned(std::size_t offset = 0) const
    {
        return _aligned(m_vector, offset);
    }

    auto operator[](std::size_t idx) const
    {
        return m_scalar * std::complex<real_type>(_element(m_vector, idx));
    };

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
    {
        const auto scalar = avx::broadcast(&m_scalar);
        const auto vector = _cx_reg(m_vector, idx);

        return {avx::mul<real_type>(vector.real, scalar),
                avx::mul<real_type>(vector.imag, scalar)};
    };

    const E   m_vector;
    real_type m_scalar;
};

template<typename E>
    requires vector_expression<E>
class rscalar_div : private expression_base
{
public:
    using real_type = typename E::real_type;

    constexpr auto size() const -> std::size_t
    {
        return _size(m_vector);
    }

private:
    template<typename T, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<T, PackSize>
    friend class ::packed_cx_vector;
    friend class expression_base;
    friend auto operator/<E>(typename E::real_type lhs, const E& rhs);

    rscalar_div(real_type scalar, const E& vector)
    : m_scalar(scalar)
    , m_vector(vector){};

    constexpr bool aligned(std::size_t offset = 0) const
    {
        return _aligned(m_vector, offset);
    }

    auto operator[](std::size_t idx) const
    {
        return m_scalar / std::complex<real_type>(_element(m_vector, idx));
    };

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
    {
        const auto scalar = avx::broadcast<real_type>(&m_scalar);
        const auto vector = _cx_reg(m_vector, idx);

        const auto vector_abs =
            avx::add<real_type>(avx::mul<real_type>(vector.real, vector.real),
                                avx::mul<real_type>(vector.imag, vector.imag));

        const auto real =
            avx::div<real_type>(avx::mul<real_type>(scalar, vector.real), vector_abs);

        const auto imag =
            avx::div<real_type>(avx::mul<real_type>(scalar, vector.imag), vector_abs);

        return {real, imag};
    };

    const E   m_vector;
    real_type m_scalar;
};

}    // namespace internal

template<typename E>
    requires internal::vector_expression<E>
auto operator+(const E& lhs, typename E::real_type rhs)
{
    return internal::rscalar_add(rhs, lhs);
}
template<typename E>
    requires internal::vector_expression<E>
auto operator+(typename E::real_type lhs, const E& rhs)
{
    return internal::rscalar_add(lhs, rhs);
}

template<typename E>
    requires internal::vector_expression<E>
auto operator-(const E& lhs, typename E::real_type rhs)
{
    return internal::rscalar_add(-rhs, lhs);
}
template<typename E>
    requires internal::vector_expression<E>
auto operator-(typename E::real_type lhs, const E& rhs)
{
    return internal::rscalar_sub(lhs, rhs);
}

template<typename E>
    requires internal::vector_expression<E>
auto operator*(const E& lhs, typename E::real_type rhs)
{
    return internal::rscalar_mul(rhs, lhs);
}
template<typename E>
    requires internal::vector_expression<E>
auto operator*(typename E::real_type lhs, const E& rhs)
{
    return internal::rscalar_mul(lhs, rhs);
}

template<typename E>
    requires internal::vector_expression<E>
auto operator/(const E& lhs, typename E::real_type rhs)
{
    return internal::rscalar_mul(1 / rhs, lhs);
}
template<typename E>
    requires internal::vector_expression<E>
auto operator/(typename E::real_type lhs, const E& rhs)
{
    return internal::rscalar_div(lhs, rhs);
}

// #endregion real scalar

// #region vector

template<typename E, typename T, std::size_t PackSize, typename Allocator>
    requires internal::vector_expression<E> && packed_floating_point<T, PackSize>
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
    requires internal::vector_expression<E> && packed_floating_point<T, PackSize>
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
    requires internal::vector_expression<E> && packed_floating_point<T, PackSize>
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
    requires internal::vector_expression<E> && packed_floating_point<T, PackSize>
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