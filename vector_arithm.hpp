#ifndef VECTOR_ARITHM_HPP
#define VECTOR_ARITHM_HPP
#include "vector_util.hpp"

#include <assert.h>
#include <complex>
#include <concepts>
#include <immintrin.h>
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

}    // namespace avx

template<typename T>
class packed_expression_base
{
public:
    packed_expression_base() = default;

    packed_expression_base(const packed_expression_base& other)     = default;
    packed_expression_base(packed_expression_base&& other) noexcept = default;

    ~packed_expression_base() = default;

    packed_expression_base& operator=(const packed_expression_base& other)     = default;
    packed_expression_base& operator=(packed_expression_base&& other) noexcept = default;

private:
};

template<typename T, typename E>
class packed_expression : packed_expression_base<T>
{
    friend class packed_expression_base<T>;

public:
    packed_expression() noexcept  = default;
    ~packed_expression() noexcept = default;

    packed_expression(const packed_expression& other) noexcept = default;
    packed_expression(packed_expression&& other) noexcept      = default;

    packed_expression& operator=(const packed_expression& other) noexcept = default;
    packed_expression& operator=(packed_expression&& other) noexcept      = default;

    auto size()
    {
        return static_cast<E>(*this).size();
    };

    auto operator[](std::size_t idx)
    {
        return static_cast<E>(*this)[idx];
    }

private:
    auto cx_reg(std::size_t idx) const -> avx::cx_reg<T>
    {
        return static_cast<E>(*this).cx_reg(idx);
    };
};

template<typename E>
constexpr bool is_aligned(const E& expression, long offset = 0)
{
    return expression.aligned(offset);
}

template<typename E>
constexpr bool is_scalar()
{
    return false;
};

template<typename T, typename E>
concept concept_packed_expression =
    std::derived_from<E, packed_expression<T, E>> &&
    requires(E expression, std::size_t idx) {
        {
            is_aligned(expression)
            } -> std::same_as<bool>;

        {
            expression.cx_reg(idx)
            } -> std::same_as<avx::cx_reg<T>>;

        {
            expression[idx]
            } -> std::convertible_to<std::complex<T>>;

        is_scalar<E>() || requires(E expression) {
                              {
                                  expression.size()
                                  } -> std::same_as<std::size_t>;
                          };
    };


template<typename T>
class packed_scalar : packed_expression<T, packed_scalar<T>>
{
    friend class packed_expression_base<T>;

public:
    using real_type = T;

    packed_scalar(std::complex<T> value)
    : m_value(value){};

private:
    constexpr bool aligned(long idx = 0)
    {
        return true;
    }

    auto operator[](std::size_t idx) const -> std::complex<T>
    {
        return m_value;
    };

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<T>
    {
        return {avx::broadcast(&m_value.real()), avx::broadcast(&m_value.imag())};
    };

    std::complex<T> m_value;
};

template<>
constexpr bool is_scalar<packed_scalar<float>>()
{
    return true;
};
template<>
constexpr bool is_scalar<packed_scalar<double>>()
{
    return true;
};


template<typename T, typename E1, typename E2>
    requires concept_packed_expression<T, E1> && concept_packed_expression<T, E2>
class packed_sum : packed_expression<T, packed_sum<T, E1, E2>>
{
    friend class packed_expression_base<T>;

public:
    using real_type = typename E1::real_type;

    packed_sum(const E1& lhs, const E2& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs)
    {
        assert(E1::real_type == E2::real_type);
        assert(is_scalar<E2>() || lhs.size() = rhs.size());
    };

    auto size() const -> std::size_t
    {
        return m_lhs.size();
    }

private:
    constexpr bool aligned(std::size_t offset) const
    {
        return is_aligned(m_lhs, offset) && is_aligned(m_rhs, offset);
    }

    auto operator[](std::size_t idx) const
    {
        return m_lhs[idx] + m_rhs[idx];
    };

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<T>
    {
        const auto lhs = m_lhs.cx_reg(idx);
        const auto rhs = m_rhs.cx_reg(idx);

        return {avx::add(lhs.real, rhs.real), avx::add(lhs.imag, rhs.imag)};
    };

    const E1& m_lhs;
    const E2& m_rhs;
};

template<typename T, typename E1, typename E2>
    requires concept_packed_expression<T, E1> && concept_packed_expression<T, E2>
class packed_diff : packed_expression<T, packed_diff<T, E1, E2>>
{
    friend class packed_expression_base<T>;

public:
    using real_type = typename E1::real_type;

    packed_diff(const E1& lhs, const E2& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs)
    {
        assert(E1::real_type == E2::real_type);
        assert(is_scalar<E1>() || is_scalar<E2>() || lhs.size() = rhs.size());
    };

    auto size() const -> std::size_t
    {
        if constexpr (!is_scalar<E1>())
        {
            return m_lhs.size();
        } else
        {
            return m_rhs.size();
        }
    }

private:
    constexpr bool aligned(std::size_t offset) const
    {
        return is_aligned(m_lhs, offset) && is_aligned(m_rhs, offset);
    }

    auto operator[](std::size_t idx) const
    {
        return m_lhs[idx] + m_rhs[idx];
    };

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<T>
    {
        const auto lhs = m_lhs.cx_reg(idx);
        const auto rhs = m_rhs.cx_reg(idx);

        return {avx::sub(lhs.real, rhs.real), avx::sub(lhs.imag, rhs.imag)};
    };

    const E1& m_lhs;
    const E2& m_rhs;
};

template<typename T, typename E1, typename E2>
    requires concept_packed_expression<T, E1> && concept_packed_expression<T, E2>
auto operator+(const E1& lhs, const E2& rhs)
{
    return packed_diff(lhs, rhs);
};

#endif