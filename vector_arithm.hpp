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
            requires(const E& expression, std::size_t idx) {
                typename E::real_type;

                {
                    expression.aligned(idx)
                    } -> std::same_as<bool>;

                {
                    expression[idx]
                    } -> std::convertible_to<std::complex<typename E::real_type>>;

                {
                    expression.cx_reg(idx)
                    } -> std::same_as<avx::cx_reg<typename E::real_type>>;

                is_scalar<E>::value || requires {
                                           {
                                               expression.size()
                                               } -> std::same_as<std::size_t>;
                                       };
            };
        ;
    };

protected:
    template<typename E>
    static constexpr auto _size(const E& expression)
    {
        return expression.size();
    }

    template<typename E>
    static constexpr bool _aligned(const E& expression, std::size_t offset = 0)
    {
        return expression.aligned(offset);
    }

    template<typename E>
    static constexpr auto _element(const E& expression, std::size_t idx)
    {
        return expression[idx];
    }

    template<typename E>
    static constexpr auto _cx_reg(const E& expression, std::size_t idx)
    {
        return expression.cx_reg(idx);
    }
};

template<typename E>
concept concept_packed_expression = expression_base::is_expression<E>::value;

template<typename E1, typename E2>
concept compatible = concept_packed_expression<E1> && concept_packed_expression<E2> &&
                     std::same_as<typename E1::real_type, typename E2::real_type>;

template<typename E1, typename E2>
    requires compatible<E1, E2>
auto operator+(const E1& lhs, const E2& rhs);

template<typename E1, typename E2>
    requires compatible<E1, E2>
auto operator*(const E1& lhs, const E2& rhs);

template<typename E1, typename E2>
    requires compatible<E1, E2>
class packed_sum : private expression_base
{
public:
    using real_type = typename E1::real_type;

    auto size() const -> std::size_t
    {
        if constexpr (is_scalar<E1>::value)
        {
            return _size(m_rhs);
        } else
        {
            return _size(m_lhs);
        }
    }

private:
    template<typename T, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<T, PackSize>
    friend class packed_cx_vector;
    friend class expression_base;
    friend auto operator+<E1, E2>(const E1& lhs, const E2& rhs);

    packed_sum(const E1& lhs, const E2& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs)
    {
        if constexpr (!is_scalar<E1>::value && !is_scalar<E2>::value)
        {
            assert(lhs.size() == rhs.size());
        }
    };

    constexpr bool aligned(std::size_t offset = 0) const
    {
        return _aligned(m_lhs, offset) && _aligned(m_rhs, offset);
    }

    auto operator[](std::size_t idx) const
    {
        return std::complex<real_type>(_element(m_lhs, idx)) +
               std::complex<real_type>(_element(m_rhs, idx));
    };

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
    {
        const auto lhs = _cx_reg(m_lhs, idx);
        const auto rhs = _cx_reg(m_rhs, idx);

        return {avx::add<real_type>(lhs.real, rhs.real),
                avx::add<real_type>(lhs.imag, rhs.imag)};
    };

    const E1& m_lhs;
    const E2& m_rhs;
};

template<typename E1, typename E2>
    requires compatible<E1, E2>
class mul : private expression_base
{
public:
    using real_type = typename E1::real_type;

    auto size() const -> std::size_t
    {
        if constexpr (is_scalar<E1>::value)
        {
            return _size(m_rhs);
        } else
        {
            return _size(m_lhs);
        }
    }

private:
    template<typename T, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<T, PackSize>
    friend class packed_cx_vector;
    friend class expression_base;
    friend auto operator*<E1, E2>(const E1& lhs, const E2& rhs);

    mul(const E1& lhs, const E2& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs)
    {
        assert(is_scalar<E1>::value || is_scalar<E2>::value || lhs.size() == rhs.size());
    };

    constexpr bool aligned(std::size_t offset = 0) const
    {
        return _aligned(m_lhs, offset) && _aligned(m_rhs, offset);
    }

    auto operator[](std::size_t idx) const
    {
        return std::complex<real_type>(_element(m_lhs, idx)) *
               std::complex<real_type>(_element(m_rhs, idx));
    };

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
    {
        const auto lhs = _cx_reg(m_lhs, idx);
        const auto rhs = _cx_reg(m_rhs, idx);

        auto real = avx::sub<real_type>(avx::mul<real_type>(lhs.real, rhs.real),
                                        avx::mul<real_type>(lhs.imag, rhs.imag));
        auto imag = avx::add<real_type>(avx::mul<real_type>(lhs.real, rhs.imag),
                                        avx::mul<real_type>(lhs.imag, rhs.real));

        return {real, imag};
    };

    const E1& m_lhs;
    const E2& m_rhs;
};

template<typename E1, typename E2>
    requires compatible<E1, E2>
auto operator+(const E1& lhs, const E2& rhs)
{
    return packed_sum<E1, E2>(lhs, rhs);
};

template<typename E1, typename E2>
    requires compatible<E1, E2>
auto operator*(const E1& lhs, const E2& rhs)
{
    return mul(lhs, rhs);
};

// #region complex scalar

template<typename E>
    requires concept_packed_expression<E>
auto operator+(const E& lhs, std::complex<typename E::real_type> rhs);
template<typename E>
    requires concept_packed_expression<E>
auto operator+(std::complex<typename E::real_type> lhs, const E& rhs);

template<typename E>
    requires concept_packed_expression<E>
class packed_scalar : expression_base
{
public:
    using real_type = typename E::real_type;

private:
    template<typename T, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<T, PackSize>
    friend class packed_cx_vector;
    friend class expression_base;

    friend auto operator+<E>(const E& lhs, std::complex<real_type> rhs);
    friend auto operator+<E>(std::complex<real_type> lhs, const E& rhs);


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
        return {avx::broadcast(&m_value.real()), avx::broadcast(&m_value.imag())};
    };

    std::complex<real_type> m_value;
};

template<typename E>
struct is_scalar<packed_scalar<E>>
{
    static constexpr bool value = true;
};

template<typename E>
    requires concept_packed_expression<E>
auto operator+(const E& lhs, std::complex<typename E::real_type> rhs)
{
    return lhs + packed_scalar<E>(rhs);
}

template<typename E>
    requires concept_packed_expression<E>
auto operator+(std::complex<typename E::real_type> lhs, const E& rhs)
{
    return rhs + packed_scalar<E>(lhs);
}

// #endregion complex scalar

// #region real scalar

template<typename E>
    requires concept_packed_expression<E>
auto operator*(const E& lhs, typename E::real_type rhs);

template<typename E>
    requires concept_packed_expression<E>
auto operator*(typename E::real_type lhs, const E& rhs);

template<typename E>
    requires concept_packed_expression<E>
class mulr : private expression_base
{
public:
    using real_type = typename E::real_type;

    auto size() const -> std::size_t
    {
        return _size(m_vector);
    }

private:
    template<typename T, std::size_t PackSize, typename Allocator>
        requires packed_floating_point<T, PackSize>
    friend class packed_cx_vector;
    friend class expression_base;
    friend auto operator*<E>(const E& lhs, typename E::real_type rhs);
    friend auto operator*<E>(typename E::real_type lhs, const E& rhs);

    mulr(const E& vector, real_type scalar)
    : m_vector(vector)
    , m_scalar(scalar){};

    constexpr bool aligned(std::size_t offset = 0) const
    {
        return _aligned(m_vector, offset);
    }

    auto operator[](std::size_t idx) const
    {
        return std::complex<real_type>(_element(m_vector, idx)) * m_scalar;
    };

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
    {
        const auto vector = _cx_reg(m_vector, idx);
        const auto scalar = avx::broadcast(&m_scalar);

        return {avx::mul<real_type>(vector.real, scalar),
                avx::mul<real_type>(vector.imag, scalar)};
    };

    const E&  m_vector;
    real_type m_scalar;
};

template<typename E>
    requires concept_packed_expression<E>
auto operator*(const E& lhs, typename E::real_type rhs)
{
    return mulr(lhs, rhs);
}

template<typename E>
    requires concept_packed_expression<E>
auto operator*(typename E::real_type lhs, const E& rhs)
{
    return mulr(rhs, lhs);
}

// #endregion real scalar

#endif