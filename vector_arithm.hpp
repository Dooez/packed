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
class packed_expression : public packed_expression_base<T>
{
    friend class packed_expression_base<T>;

public:
    using real_type = T;

    packed_expression() noexcept  = default;
    ~packed_expression() noexcept = default;

    packed_expression(const packed_expression& other) noexcept = default;
    packed_expression(packed_expression&& other) noexcept      = default;

    packed_expression& operator=(const packed_expression& other) noexcept = default;
    packed_expression& operator=(packed_expression&& other) noexcept      = default;

    auto size() const
    {
        return static_cast<const E*>(this)->size();
    };


private:
public:
    constexpr bool aligned(long idx = 0) const
    {
        return static_cast<const E*>(this)->aligned(idx);
    }

    auto operator[](std::size_t idx) const
    {
        return (*static_cast<const E*>(this))[idx];
    }

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<T>
    {
        return static_cast<const E*>(this)->cx_reg(idx);
    };
};

template<typename E>
constexpr bool is_scalar()
{
    return false;
};

template<typename T, typename E>
concept concept_packed_expression =
    std::derived_from<E, packed_expression<T, E>> &&
    requires(const E& expression, std::size_t idx) {
        {
            expression.aligned(idx)
            } -> std::same_as<bool>;

        {
            expression[idx]
            } -> std::convertible_to<std::complex<T>>;

        {
            expression.cx_reg(idx)
            } -> std::same_as<avx::cx_reg<T>>;

        is_scalar<E>() || requires(E expression) {
                              {
                                  expression.size()
                                  } -> std::same_as<std::size_t>;
                          };
    };


template<typename T>
class packed_scalar : public packed_expression<T, packed_scalar<T>>
{
    friend class packed_expression_base<T>;

public:
    using real_type = T;

    packed_scalar(std::complex<T> value)
    : m_value(value){};

private:
public:
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
// requires concept_packed_expression<T, E1> && concept_packed_expression<T, E2>
class packed_sum : public packed_expression<T, packed_sum<T, E1, E2>>
{
    friend class packed_expression_base<T>;

public:
    using real_type = typename E1::real_type;

    packed_sum(const E1& lhs, const E2& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs)
    {
        // assert(E1::real_type == E2::real_type);
        assert(is_scalar<E2>() || lhs.size() == rhs.size());
    };

    auto size() const -> std::size_t
    {
        return m_lhs.size();
    }

private:
public:
    constexpr bool aligned(std::size_t offset = 0) const
    {
        return m_lhs.aligned(offset) && m_rhs.aligned(offset);
    }

    auto operator[](std::size_t idx) const
    {
        return std::complex<T>(m_lhs[idx]) + std::complex<T>(m_rhs[idx]);
    };

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<T>
    {
        const auto lhs = m_lhs.cx_reg(idx);
        const auto rhs = m_rhs.cx_reg(idx);

        return {avx::add<T>(lhs.real, rhs.real), avx::add<T>(lhs.imag, rhs.imag)};
    };

    const E1& m_lhs;
    const E2& m_rhs;
};

template<typename T, typename E1, typename E2>
// requires concept_packed_expression<T, E1> && concept_packed_expression<T, E2>
class packed_diff : public packed_expression<T, packed_diff<T, E1, E2>>
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
public:
    constexpr bool aligned(std::size_t offset) const
    {
        return m_lhs.aligned(offset) && m_rhs.aligned(offset);
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

template<typename T,
         template<typename...>
         typename E1,
         template<typename...>
         typename E2,
         typename... Tother>
// requires concept_packed_expression<T, E1<T, Tother...>> &&
//          concept_packed_expression<T, E2<T, Tother...>>
auto operator+(const E1<T, Tother...>& lhs, const E2<T, Tother...>& rhs)
{
    return packed_sum<T, E1<T, Tother...>, E2<T, Tother...>>(lhs, rhs);
};

#endif