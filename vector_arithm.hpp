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
auto add(typename reg<T>::type lhs, typename reg<T>::type rhs) -> typename reg<T>::type;
template<>
auto add<float>(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
{
    return _mm256_add_ps(lhs, rhs);
}
template<>
auto add<double>(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
{
    return _mm256_add_pd(lhs, rhs);
}

template<typename T>
auto sub(typename reg<T>::type lhs, typename reg<T>::type rhs) -> typename reg<T>::type;
template<>
auto sub<float>(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
{
    return _mm256_sub_ps(lhs, rhs);
}
template<>
auto sub<double>(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
{
    return _mm256_sub_pd(lhs, rhs);
}

template<typename T>
auto mul(typename reg<T>::type lhs, typename reg<T>::type rhs) -> typename reg<T>::type;
template<>
auto mul<float>(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
{
    return _mm256_mul_ps(lhs, rhs);
}
template<>
auto mul<double>(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
{
    return _mm256_mul_pd(lhs, rhs);
}

template<typename T>
auto div(typename reg<T>::type lhs, typename reg<T>::type rhs) -> typename reg<T>::type;
template<>
auto div<float>(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
{
    return _mm256_div_ps(lhs, rhs);
}
template<>
auto div<double>(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
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

    template<typename T, std::size_t PackSize, bool Const, std::size_t Extent>
    struct is_expression<packed_subrange<T, PackSize, Const, Extent>>
    {
        static constexpr bool value = true;
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

    template<typename T, std::size_t PackSize, bool Const, std::size_t Extent>
    static constexpr bool
    _aligned(const packed_subrange<T, PackSize, Const, Extent>& expression,
             std::size_t                                        offset = 0)
    {
        return expression.begin().aligned(offset);
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

    template<typename T, std::size_t PackSize, bool Const, std::size_t Extent>
    static constexpr auto
    _cx_reg(const packed_subrange<T, PackSize, Const, Extent>& expression,
            std::size_t                                        idx) -> avx::cx_reg<T>
    {
        auto real = avx::load(&(*(expression.begin() + idx)));
        auto imag = avx::load(&(*(expression.begin() + idx)) + PackSize);
        return {real, imag};
    }
};

template<typename E>
concept vector_expression = expression_base::is_expression<E>::value;

template<typename E1, typename E2>
concept compatible_expressions =
    vector_expression<E1> && vector_expression<E2> &&
    std::same_as<typename E1::real_type, typename E2::real_type>;

template<typename T>
concept always_true = true;
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
class add : private expression_base
{
public:
    using real_type = typename E1::real_type;

    constexpr auto size() const -> std::size_t
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
    friend class ::packed_cx_vector;
    friend class expression_base;
    friend auto operator+<E1, E2>(const E1& lhs, const E2& rhs);

    add(const E1& lhs, const E2& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs)
    {
        if constexpr (!is_scalar<E1>::value && !is_scalar<E2>::value)
        {
            assert(lhs.size() == rhs.size());
        };
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

    const E1 m_lhs;
    const E2 m_rhs;
};

template<typename E1, typename E2>
    requires compatible_expressions<E1, E2>
class sub : private expression_base
{
public:
    using real_type = typename E1::real_type;

    constexpr auto size() const -> std::size_t
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
    friend class ::packed_cx_vector;
    friend class expression_base;
    friend auto operator-<E1, E2>(const E1& lhs, const E2& rhs);

    sub(const E1& lhs, const E2& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs)
    {
        if constexpr (!is_scalar<E1>::value && !is_scalar<E2>::value)
        {
            assert(lhs.size() == rhs.size());
        };
    };

    constexpr bool aligned(std::size_t offset = 0) const
    {
        return _aligned(m_lhs, offset) && _aligned(m_rhs, offset);
    }

    auto operator[](std::size_t idx) const
    {
        return std::complex<real_type>(_element(m_lhs, idx)) -
               std::complex<real_type>(_element(m_rhs, idx));
    };

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
    {
        const auto lhs = _cx_reg(m_lhs, idx);
        const auto rhs = _cx_reg(m_rhs, idx);

        return {avx::sub<real_type>(lhs.real, rhs.real),
                avx::sub<real_type>(lhs.imag, rhs.imag)};
    };

    const E1 m_lhs;
    const E2 m_rhs;
};

template<typename E1, typename E2>
    requires compatible_expressions<E1, E2>
class mul : private expression_base
{
public:
    using real_type = typename E1::real_type;

    constexpr auto size() const -> std::size_t
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
    friend class ::packed_cx_vector;
    friend class expression_base;
    friend auto operator*<E1, E2>(const E1& lhs, const E2& rhs);

    mul(const E1& lhs, const E2& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs)
    {
        if constexpr (!is_scalar<E1>::value && !is_scalar<E2>::value)
        {
            assert(lhs.size() == rhs.size());
        };
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

    const E1 m_lhs;
    const E2 m_rhs;
};

template<typename E1, typename E2>
    requires compatible_expressions<E1, E2>
class div : private expression_base
{
public:
    using real_type = typename E1::real_type;

    constexpr auto size() const -> std::size_t
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
    friend class ::packed_cx_vector;
    friend class expression_base;
    friend auto operator/<E1, E2>(const E1& lhs, const E2& rhs);

    div(const E1& lhs, const E2& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs)
    {
        if constexpr (!is_scalar<E1>::value && !is_scalar<E2>::value)
        {
            assert(lhs.size() == rhs.size());
        };
    };

    constexpr bool aligned(std::size_t offset = 0) const
    {
        return _aligned(m_lhs, offset) && _aligned(m_rhs, offset);
    }

    auto operator[](std::size_t idx) const
    {
        return std::complex<real_type>(_element(m_lhs, idx)) /
               std::complex<real_type>(_element(m_rhs, idx));
    };

    auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
    {
        const auto lhs = _cx_reg(m_lhs, idx);
        const auto rhs = _cx_reg(m_rhs, idx);

        const auto rhs_abs = avx::add<real_type>(avx::mul<real_type>(rhs.real, rhs.real),
                                                 avx::mul<real_type>(rhs.imag, rhs.imag));

        const auto real_ = avx::add<real_type>(avx::mul<real_type>(lhs.real, rhs.real),
                                               avx::mul<real_type>(lhs.imag, rhs.imag));

        const auto imag_ = avx::sub<real_type>(avx::mul<real_type>(lhs.imag, rhs.real),
                                               avx::mul<real_type>(lhs.real, rhs.imag));

        return {avx::div<real_type>(real_, rhs_abs), avx::div<real_type>(imag_, rhs_abs)};
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

    const E&  m_vector;
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

    const E&  m_vector;
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

    const E&  m_vector;
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

    const E&  m_vector;
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

template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             (!std::same_as<packed_cx_vector<T, PackSize, Allocator>, E>) &&
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

template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             (!std::same_as<packed_cx_vector<T, PackSize, Allocator>, E>) &&
             requires(packed_subrange<T, PackSize> subrange, E e) { e - subrange; } &&
             true
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

template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             (!std::same_as<packed_cx_vector<T, PackSize, Allocator>, E>) &&
             requires(packed_subrange<T, PackSize> subrange, E e) { e* subrange; } && true
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

template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             (!std::same_as<packed_cx_vector<T, PackSize, Allocator>, E>) &&
             requires(packed_subrange<T, PackSize> subrange, E e) { e / subrange; } &&
             true
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