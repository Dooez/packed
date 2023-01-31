#include <assert.h>
#include <complex>
#include <concepts>
#include <immintrin.h>


template<typename E>
class packed_expression
{
public:
    packed_expression() = default;

    packed_expression(const packed_expression& other)     = delete;
    packed_expression(packed_expression&& other) noexcept = delete;

    ~packed_expression() = default;

    packed_expression& operator=(const packed_expression& other)     = delete;
    packed_expression& operator=(packed_expression&& other) noexcept = delete;

    __mm256 ymmreg(std::size_t id) const
    {
        return static_cast<E>(*this).ymmreg(id);
    };

private:
};

class vector : packed_expression<vector>
{
    template<typename E>
    friend packed_expression<E>;

public:
private:
    __mm256 ymmreg(std::size_t id){};
};

template<typename T>
class packed_scalar : packed_expression<packed_scalar<T>>
{
    template<typename E>
    friend packed_expression<E>;

public:
    using real_type = T;

    constexpr bool is_scalar = true;

    packed_scalar(std::complex<T> value)
    : m_value(value){};

private:
    __mm256 ymmreg(std::size_t id) const
    {
        return _mm256_broadcast_ps(&m_value);
    };

    std::complex<T> m_value;
};


template<typename E1, typename E2>
    requires std::derived_from<E1, packed_expression<E1>> &&
             std::derived_from<E2, packed_expression<E2>>
class packed_sum : packed_expression<packed_sum<E1, E2>>
{
    template<typename E>
    friend packed_expression<E>;

public:
    using real_type = typename E1::real_type;

    constexpr bool is_scalar = false;

    packed_sum(const E1& lhs, const E2& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs)
    {
        assert(E1::real_type == E2::real_type);
        assert(rhs.is_scalar || lhs.size() = rhs.size());
    };

    std::size_t size() const
    {
        return lhs.size();
    }

private:
    __mm256 ymmreg(std::size_t id) const
    {
        return _mm256_add_ps(m_lhs.ymmreg(id), m_rhs.ymmreg(id));
    };

    const E1& m_lhs;
    const E2& m_rhs;
};

template<typename E1, typename E2>
    requires std::derived_from<E1, packed_expression<E1>> &&
             std::derived_from<E2, packed_expression<E2>>
auto operator+(const E1& lhs, const E2& rhs)
{
    return packed_sum(lhs, rhs);
};

template<typename E1, typename Scalar>
    requires std::derived_from<E1, packed_expression<E1>> &&
             std::convertible_to<Scalar, std::complex<typename E1::real_type>>
auto operator+(const E1& lhs, Scalar rhs)
{
    using real_type = typename E1::real_type;
    return packed_sum(lhs, packed_scalar<real_type>(std::complex<real_type>(rhs)));
};