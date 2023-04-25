#ifndef VECTOR_ARITHM_HPP
#define VECTOR_ARITHM_HPP
#include "vector_util.hpp"

#include <cassert>
#include <complex>
#include <concepts>
#include <cstddef>
#include <immintrin.h>
#include <ranges>

namespace pcx {

namespace avx {

inline auto add(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm256_add_ps(lhs, rhs);
}
inline auto add(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm256_add_pd(lhs, rhs);
}

inline auto sub(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm256_sub_ps(lhs, rhs);
}
inline auto sub(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm256_sub_pd(lhs, rhs);
}

inline auto mul(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm256_mul_ps(lhs, rhs);
}
inline auto mul(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm256_mul_pd(lhs, rhs);
}

inline auto div(reg_t<float> lhs, reg_t<float> rhs) -> reg_t<float> {
    return _mm256_div_ps(lhs, rhs);
}
inline auto div(reg_t<double> lhs, reg_t<double> rhs) -> reg_t<double> {
    return _mm256_div_pd(lhs, rhs);
}

inline auto fmadd(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm256_fmadd_ps(a, b, c);
}
inline auto fmadd(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm256_fmadd_pd(a, b, c);
}

inline auto fnmadd(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm256_fnmadd_ps(a, b, c);
}
inline auto fnmadd(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm256_fnmadd_pd(a, b, c);
}

inline auto fmsub(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm256_fmsub_ps(a, b, c);
}
inline auto fmsub(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm256_fmsub_pd(a, b, c);
}

inline auto fnmsub(reg_t<float> a, reg_t<float> b, reg_t<float> c) -> reg_t<float> {
    return _mm256_fnmsub_ps(a, b, c);
}
inline auto fnmsub(reg_t<double> a, reg_t<double> b, reg_t<double> c) -> reg_t<double> {
    return _mm256_fnmsub_pd(a, b, c);
}

template<typename T, bool Conj>
inline auto add(cx_reg<T, Conj> lhs, cx_reg<T, Conj> rhs) -> cx_reg<T, Conj> {
    return {add(lhs.real, rhs.real), add(lhs.imag, rhs.imag)};
}
template<typename T>
inline auto add(cx_reg<T, true> lhs, cx_reg<T, false> rhs) -> cx_reg<T, false> {
    return {add(lhs.real, rhs.real), sub(rhs.imag, lhs.imag)};
}
template<typename T>
inline auto add(cx_reg<T, false> lhs, cx_reg<T, true> rhs) -> cx_reg<T, false> {
    return {add(lhs.real, rhs.real), sub(lhs.imag, rhs.imag)};
}

template<typename T>
inline auto sub(cx_reg<T, false> lhs, cx_reg<T, false> rhs) -> cx_reg<T, false> {
    return {sub(lhs.real, rhs.real), sub(lhs.imag, rhs.imag)};
}
template<typename T>
inline auto sub(cx_reg<T, true> lhs, cx_reg<T, false> rhs) -> cx_reg<T, true> {
    return {sub(lhs.real, rhs.real), add(lhs.imag, rhs.imag)};
}
template<typename T>
inline auto sub(cx_reg<T, false> lhs, cx_reg<T, true> rhs) -> cx_reg<T, false> {
    return {sub(lhs.real, rhs.real), add(lhs.imag, rhs.imag)};
}
template<typename T>
inline auto sub(cx_reg<T, true> lhs, cx_reg<T, true> rhs) -> cx_reg<T, false> {
    return {sub(lhs.real, rhs.real), sub(rhs.imag, lhs.imag)};
}

template<typename T>
inline auto mul(cx_reg<T, false> lhs, cx_reg<T, false> rhs) -> cx_reg<T, false> {
    auto real = mul(lhs.real, rhs.real);
    auto imag = mul(lhs.real, rhs.imag);
    return {fnmadd(lhs.imag, rhs.imag, real), fmadd(lhs.imag, rhs.real, imag)};
}
template<typename T>
inline auto mul(cx_reg<T, true> lhs, cx_reg<T, false> rhs) -> cx_reg<T, false> {
    auto real = mul(lhs.real, rhs.real);
    auto imag = mul(lhs.real, rhs.imag);
    return {fmadd(lhs.imag, rhs.imag, real), fnmadd(lhs.imag, rhs.real, imag)};
}
template<typename T>
inline auto mul(cx_reg<T, false> lhs, cx_reg<T, true> rhs) -> cx_reg<T, false> {
    auto real = mul(lhs.real, rhs.real);
    auto imag = mul(lhs.real, rhs.imag);
    return {fmadd(lhs.imag, rhs.imag, real), fmsub(lhs.imag, rhs.real, imag)};
}
template<typename T>
inline auto mul(cx_reg<T, true> lhs, cx_reg<T, true> rhs) -> cx_reg<T, false> {
    auto real = mul(lhs.real, rhs.real);
    auto imag = mul(lhs.real, rhs.imag);
    return {fnmadd(lhs.imag, rhs.imag, real), fnmsub(lhs.imag, rhs.real, imag)};
}

template<typename T>
inline auto div(cx_reg<T, false> lhs, cx_reg<T, false> rhs) -> cx_reg<T, false> {
    auto rhs_abs = avx::mul(rhs.real, rhs.real);
    auto real_   = avx::mul(lhs.real, rhs.real);
    auto imag_   = avx::mul(lhs.real, rhs.imag);

    rhs_abs = avx::fmadd(rhs.imag, rhs.imag, rhs_abs);
    real_   = avx::fmadd(lhs.imag, rhs.imag, real_);
    imag_   = avx::fmsub(lhs.imag, rhs.real, imag_);

    return {avx::div(real_, rhs_abs), avx::div(imag_, rhs_abs)};
}
template<typename T>
inline auto div(cx_reg<T, true> lhs, cx_reg<T, false> rhs) -> cx_reg<T, false> {
    auto rhs_abs = avx::mul(rhs.real, rhs.real);
    auto real_   = avx::mul(lhs.real, rhs.real);
    auto imag_   = avx::mul(lhs.real, rhs.imag);

    rhs_abs = avx::fmadd(rhs.imag, rhs.imag, rhs_abs);
    real_   = avx::fnmadd(lhs.imag, rhs.imag, real_);
    imag_   = avx::fnmsub(lhs.imag, rhs.real, imag_);

    return {avx::div(real_, rhs_abs), avx::div(imag_, rhs_abs)};
}
template<typename T>
inline auto div(cx_reg<T, false> lhs, cx_reg<T, true> rhs) -> cx_reg<T, false> {
    auto rhs_abs = avx::mul(rhs.real, rhs.real);
    auto real_   = avx::mul(lhs.real, rhs.real);
    auto imag_   = avx::mul(lhs.real, rhs.imag);

    rhs_abs = avx::fmadd(rhs.imag, rhs.imag, rhs_abs);
    real_   = avx::fnmadd(lhs.imag, rhs.imag, real_);
    imag_   = avx::fmadd(lhs.imag, rhs.real, imag_);

    return {avx::div(real_, rhs_abs), avx::div(imag_, rhs_abs)};
}
template<typename T>
inline auto div(cx_reg<T, true> lhs, cx_reg<T, true> rhs) -> cx_reg<T, false> {
    auto rhs_abs = avx::mul(rhs.real, rhs.real);
    auto real_   = avx::mul(lhs.real, rhs.real);
    auto imag_   = avx::mul(lhs.real, rhs.imag);

    rhs_abs = avx::fmadd(rhs.imag, rhs.imag, rhs_abs);
    real_   = avx::fmadd(lhs.imag, rhs.imag, real_);
    imag_   = avx::fnmadd(lhs.imag, rhs.real, imag_);

    return {avx::div(real_, rhs_abs), avx::div(imag_, rhs_abs)};
}

template<typename T, bool Conj>
inline auto add(reg_t<T> lhs, cx_reg<T, Conj> rhs) -> cx_reg<T, Conj> {
    return {add(lhs, rhs.real), rhs.imag};
}
template<typename T, bool Conj>
inline auto sub(reg_t<T> lhs, cx_reg<T, Conj> rhs) -> cx_reg<T, false> {
    if constexpr (Conj) {
        return {sub(lhs, rhs.real), rhs.imag};
    } else {
        reg_t<T> zero;
        if constexpr (std::same_as<T, float>) {
            zero = _mm256_setzero_ps();
        } else {
            zero = _mm256_setzero_pd();
        }
        return {sub(lhs, rhs.real), sub(zero, rhs.imag)};
    }
}
template<typename T, bool Conj>
inline auto mul(reg_t<T> lhs, cx_reg<T, Conj> rhs) -> cx_reg<T, Conj> {
    return {avx::mul(lhs, rhs.real), avx::mul(lhs, rhs.imag)};
}
template<typename T, bool Conj>
inline auto div(reg_t<T> lhs, cx_reg<T, Conj> rhs) -> cx_reg<T, false> {
    auto     rhs_abs = avx::mul(rhs.real, rhs.real);
    auto     real_   = avx::mul(lhs, rhs.real);
    reg_t<T> imag_;
    if constexpr (Conj) {
        imag_ = avx::mul(lhs, rhs.imag);
    } else {
        reg_t<T> zero;
        if constexpr (std::same_as<T, float>) {
            zero = _mm256_setzero_ps();
        } else {
            zero = _mm256_setzero_pd();
        }
        imag_ = avx::mul(lhs, avx::sub(zero, rhs.imag));
    }
    rhs_abs = avx::fmadd(rhs.imag, rhs.imag, rhs_abs);

    return {avx::div(real_, rhs_abs), avx::div(imag_, rhs_abs)};
}

template<typename T, bool Conj>
inline auto add(cx_reg<T, Conj> lhs, reg_t<T> rhs) -> cx_reg<T, Conj> {
    return {add(lhs.real, rhs), lhs.imag};
}
template<typename T, bool Conj>
inline auto sub(cx_reg<T, Conj> lhs, reg_t<T> rhs) -> cx_reg<T, Conj> {
    return {sub(lhs.real, rhs), lhs.imag};
}
template<typename T, bool Conj>
inline auto mul(cx_reg<T, Conj> lhs, reg_t<T> rhs) -> cx_reg<T, Conj> {
    return {avx::mul(lhs.real, rhs), avx::mul(lhs.imag, rhs)};
}
template<typename T, bool Conj>
inline auto div(cx_reg<T, Conj> lhs, reg_t<T> rhs) -> cx_reg<T, Conj> {
    return {avx::div(lhs.real, rhs), avx::div(lhs.imag, rhs)};
}

/**
  * @brief Performs butterfly operation, then multiplies diff by imaginary unit RhsRotI times;
  *
  * @tparam RhsRotI number of multiplications by imaginary unity
  */
template<uint RhsRotI = 0, typename T = double>
    requires(RhsRotI < 4)
inline auto ibtfly(cx_reg<T> lhs, cx_reg<T> rhs) {
    cx_reg<T> s;
    cx_reg<T> d;
    if constexpr (RhsRotI == 0) {
        auto s_re = add(lhs.real, rhs.real);
        auto d_re = sub(lhs.real, rhs.real);
        auto s_im = add(lhs.imag, rhs.imag);
        auto d_im = sub(lhs.imag, rhs.imag);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    } else if constexpr (RhsRotI == 1) {
        auto s_re = add(lhs.real, rhs.real);
        auto d_im = sub(lhs.real, rhs.real);
        auto s_im = add(lhs.imag, rhs.imag);
        auto d_re = sub(rhs.imag, lhs.imag);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    } else if constexpr (RhsRotI == 2) {
        auto s_re = add(lhs.real, rhs.real);
        auto d_re = sub(rhs.real, lhs.real);
        auto s_im = add(lhs.imag, rhs.imag);
        auto d_im = sub(rhs.imag, lhs.imag);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    } else {
        auto s_re = add(lhs.real, rhs.real);
        auto d_im = sub(rhs.real, lhs.real);
        auto s_im = add(lhs.imag, rhs.imag);
        auto d_re = sub(lhs.imag, rhs.imag);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    }

    return std::make_tuple(s, d);
}

/**
  * @brief Multiplies rhs by imaginary unit RhsRotI times, then performs butterfly operation;
  *
  * @tparam RhsRotI number of multiplications by imaginary unity
  */
template<uint RhsRotI = 0, typename T = double>
    requires(RhsRotI < 4)
inline auto btfly(cx_reg<T> lhs, cx_reg<T> rhs) {
    cx_reg<T> s;
    cx_reg<T> d;
    if constexpr (RhsRotI == 0) {
        auto s_re = add(lhs.real, rhs.real);
        auto d_re = sub(lhs.real, rhs.real);
        auto s_im = add(lhs.imag, rhs.imag);
        auto d_im = sub(lhs.imag, rhs.imag);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    } else if constexpr (RhsRotI == 1) {
        auto s_re = sub(lhs.real, rhs.imag);
        auto d_re = add(lhs.real, rhs.imag);
        auto s_im = add(lhs.imag, rhs.real);
        auto d_im = sub(lhs.imag, rhs.real);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    } else if constexpr (RhsRotI == 2) {
        auto s_re = sub(lhs.real, rhs.real);
        auto d_re = add(lhs.real, rhs.real);
        auto s_im = sub(lhs.imag, rhs.imag);
        auto d_im = add(lhs.imag, rhs.imag);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    } else {
        auto s_re = add(lhs.real, rhs.imag);
        auto d_re = sub(lhs.real, rhs.imag);
        auto s_im = sub(lhs.imag, rhs.real);
        auto d_im = add(lhs.imag, rhs.real);
        s         = {s_re, s_im};
        d         = {d_re, d_im};
    }
    return std::make_tuple(s, d);
}

template<typename... Args>
inline auto mul(const cx_reg<Args> (&... args)[2]) {
    auto tup      = std::make_tuple(args...);
    auto real_mul = [](auto opearands) {
        auto lhs  = opearands[0];
        auto rhs  = opearands[1];
        auto real = avx::mul(lhs.real, rhs.real);
        auto imag = avx::mul(lhs.real, rhs.imag);
        return std::make_tuple(lhs, rhs, real, imag);
    };
    auto imag_mul = [](auto opearands) {
        auto lhs   = std::get<0>(opearands);
        auto rhs   = std::get<1>(opearands);
        auto real_ = std::get<2>(opearands);
        auto imag_ = std::get<3>(opearands);

        using reg_t = decltype(lhs);

        auto real = avx::fnmadd(lhs.imag, rhs.imag, real_);
        auto imag = avx::fmadd(lhs.imag, rhs.real, imag_);
        return reg_t{real, imag};
    };

    auto tmp = internal::apply_for_each(real_mul, tup);
    return internal::apply_for_each(imag_mul, tmp);
}
}    // namespace avx

namespace internal {

struct expression_traits {
    /**
     * @brief Extracts simd vector from iterator.
     *
     * @tparam PackSize    required pack size
     * @param iterator     must be aligned
     * @param offset       must be a multiple of SIMD vector size
     * @return constexpr auto
     */
    template<std::size_t PackSize, typename I>
    [[nodiscard]] static constexpr auto cx_reg(const I& iterator, std::size_t offset) {
        auto data      = iterator.cx_reg(offset);
        std::tie(data) = avx::convert<typename I::real_type>::template repack<I::pack_size, PackSize>(data);
        return data;
    }
    /**
     * @brief Extracts simd vector from iterator.
     *
     * @tparam PackSize required pack size
     * @param iterator  must be aligned
     * @param offset    must be a multiple of SIMD vector size
     */
    template<std::size_t PackSize, typename T, bool Const, std::size_t IPackSize>
    [[nodiscard]] static constexpr auto cx_reg(const iterator<T, Const, IPackSize>& iterator,
                                               std::size_t                          offset) {
        constexpr auto PLoad = std::max(avx::reg<T>::size, IPackSize);

        auto addr      = avx::ra_addr<IPackSize>(&(*iterator), offset);
        auto data      = avx::cxload<PLoad>(addr);
        std::tie(data) = avx::convert<T>::template repack<IPackSize, PackSize>(data);
        return data;
    }

    template<typename I>
    [[nodiscard]] static constexpr auto aligned(const I& iterator, std::size_t idx) {
        return iterator.aligned(idx);
    }
};

template<typename E>
concept vector_expression =    //
    requires(E expression, std::size_t idx) {
        requires std::ranges::view<E>;

        requires std::ranges::random_access_range<E>;

        requires std::ranges::sized_range<E>;

        typename E::real_type;

        requires std::convertible_to<std::iter_value_t<decltype(expression.begin())>,
                                     std::complex<typename E::real_type>>;

        { expression_traits::aligned(expression.begin(), idx) } -> std::same_as<bool>;
    } &&
    (
        requires(E expression, std::size_t idx) {
            {
                expression_traits::cx_reg<avx::reg<typename E::real_type>::size>(expression.begin(), idx)
                } -> std::same_as<avx::cx_reg<typename E::real_type, false>>;
        } ||
        requires(E expression, std::size_t idx) {
            {
                expression_traits::cx_reg<avx::reg<typename E::real_type>::size>(expression.begin(), idx)
                } -> std::same_as<avx::cx_reg<typename E::real_type, true>>;
        });

template<typename E1, typename E2>
concept compatible_expression = vector_expression<E1> && vector_expression<E2> &&
                                std::same_as<typename E1::real_type, typename E2::real_type>;

template<typename Expression, typename Scalar>
concept compatible_scalar = vector_expression<Expression> &&
                            (std::same_as<typename Expression::real_type, Scalar> ||
                             std::same_as<std::complex<typename Expression::real_type>, Scalar>);

}    // namespace internal

// #region operator forward declarations

template<typename E1, typename E2>
    requires internal::compatible_expression<E1, E2>
auto operator+(const E1& lhs, const E2& rhs);
template<typename E1, typename E2>
    requires internal::compatible_expression<E1, E2>
auto operator-(const E1& lhs, const E2& rhs);
template<typename E1, typename E2>
    requires internal::compatible_expression<E1, E2>
auto operator*(const E1& lhs, const E2& rhs);
template<typename E1, typename E2>
    requires internal::compatible_expression<E1, E2>
auto operator/(const E1& lhs, const E2& rhs);

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

// #endregion operator forward declarations

namespace internal {

template<typename E1, typename E2>
    requires compatible_expression<E1, E2>
class add : public std::ranges::view_base {
    friend auto operator+<E1, E2>(const E1& lhs, const E2& rhs);
    using lhs_iterator = decltype(std::declval<const E1>().begin());
    using rhs_iterator = decltype(std::declval<const E2>().begin());

public:
    using real_type = typename E1::real_type;
    static constexpr auto pack_size =
        std::min(std::max(E1::pack_size, E2::pack_size), avx::reg<real_type>::size);
    class iterator {
        friend class add;

    private:
        iterator(lhs_iterator lhs, rhs_iterator rhs)
        : m_lhs(std::move(lhs))
        , m_rhs(std::move(rhs)){};

    public:
        using real_type        = add::real_type;
        using value_type       = const std::complex<real_type>;
        using difference_type  = std::ptrdiff_t;
        using iterator_concept = std::random_access_iterator_tag;

        static constexpr auto pack_size = add::pack_size;

        iterator() = default;

        iterator(const iterator& other) noexcept = default;
        iterator(iterator&& other) noexcept      = default;

        ~iterator() = default;

        iterator& operator=(const iterator& other) noexcept = default;
        iterator& operator=(iterator&& other) noexcept      = default;

        [[nodiscard]] bool operator==(const iterator& other) const noexcept {
            return (m_lhs == other.m_lhs);
        }
        [[nodiscard]] auto operator<=>(const iterator& other) const noexcept {
            return (m_lhs <=> other.m_lhs);
        }

        auto operator++() noexcept -> iterator& {
            ++m_lhs;
            ++m_rhs;
            return *this;
        }
        auto operator++(int) noexcept -> iterator {
            auto copy = *this;
            ++(*this);
            return copy;
        }
        auto operator--() noexcept -> iterator& {
            --m_lhs;
            --m_rhs;
        }
        auto operator--(int) noexcept -> iterator {
            auto copy = *this;
            --(*this);
            return copy;
        }

        auto operator+=(difference_type n) noexcept -> iterator& {
            m_lhs += n;
            m_rhs += n;
            return *this;
        }
        auto operator-=(difference_type n) noexcept -> iterator& {
            return (*this) += -n;
        }

        [[nodiscard]] friend auto operator+(iterator it, difference_type n) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator+(difference_type n, iterator it) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator it, difference_type n) noexcept -> iterator {
            it -= n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator lhs, iterator rhs) noexcept {
            return lhs.m_lhs - rhs.m_lhs;
        }

        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return value_type(*m_lhs) + value_type(*m_rhs);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return value_type(*(m_lhs + idx)) + value_type(*(m_rhs + idx));
        }
        [[nodiscard]] auto cx_reg(std::size_t idx) const {
            const auto lhs = expression_traits::cx_reg<pack_size>(m_lhs, idx);
            const auto rhs = expression_traits::cx_reg<pack_size>(m_rhs, idx);

            return avx::add(lhs, rhs);
        }

        [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept {
            return expression_traits::aligned(m_lhs, offset) && expression_traits::aligned(m_rhs, offset);
        }

    private:
        lhs_iterator m_lhs;
        rhs_iterator m_rhs;
    };

private:
    add(const E1& lhs, const E2& rhs)
    : m_lhs(lhs.begin())
    , m_rhs(rhs.begin())
    , m_size(lhs.size()) {
        assert(lhs.size() == rhs.size());
    };

public:
    add() noexcept = delete;

    add(add&& other) noexcept = default;
    add(const add&) noexcept  = default;

    ~add() noexcept = default;

    add& operator=(add&& other) noexcept {
        m_lhs = std::move(other.m_lhs);
        m_rhs = std::move(other.m_rhs);
        return *this;
    };
    add& operator=(const add&) noexcept = delete;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_lhs, m_rhs);
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_lhs + m_size, m_rhs + m_size);
    }

    [[nodiscard]] auto operator[](std::size_t idx) const {
        return std::complex<real_type>(m_lhs[idx]) + std::complex<real_type>(m_rhs[idx]);
    };

    [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
        return m_size;
    }

private:
    lhs_iterator m_lhs;
    rhs_iterator m_rhs;
    std::size_t  m_size;
};

template<typename E1, typename E2>
    requires compatible_expression<E1, E2>
class sub : public std::ranges::view_base {
    friend auto operator-<E1, E2>(const E1& lhs, const E2& rhs);
    using lhs_iterator = decltype(std::declval<const E1>().begin());
    using rhs_iterator = decltype(std::declval<const E2>().begin());

public:
    using real_type = typename E1::real_type;
    static constexpr auto pack_size =
        std::min(std::max(E1::pack_size, E2::pack_size), avx::reg<real_type>::size);
    class iterator {
        friend class sub;

    private:
        iterator(lhs_iterator lhs, rhs_iterator rhs)
        : m_lhs(std::move(lhs))
        , m_rhs(std::move(rhs)){};

    public:
        using real_type        = sub::real_type;
        using value_type       = const std::complex<real_type>;
        using difference_type  = std::ptrdiff_t;
        using iterator_concept = std::random_access_iterator_tag;

        static constexpr auto pack_size = sub::pack_size;

        iterator() = default;

        iterator(const iterator& other) noexcept = default;
        iterator(iterator&& other) noexcept      = default;

        ~iterator() = default;

        iterator& operator=(const iterator& other) noexcept = default;
        iterator& operator=(iterator&& other) noexcept      = default;

        [[nodiscard]] bool operator==(const iterator& other) const noexcept {
            return (m_lhs == other.m_lhs);
        }
        [[nodiscard]] auto operator<=>(const iterator& other) const noexcept {
            return (m_lhs <=> other.m_lhs);
        }

        auto operator++() noexcept -> iterator& {
            ++m_lhs;
            ++m_rhs;
            return *this;
        }
        auto operator++(int) noexcept -> iterator {
            auto copy = *this;
            ++(*this);
            return copy;
        }
        auto operator--() noexcept -> iterator& {
            --m_lhs;
            --m_rhs;
        }
        auto operator--(int) noexcept -> iterator {
            auto copy = *this;
            --(*this);
            return copy;
        }

        auto operator+=(difference_type n) noexcept -> iterator& {
            m_lhs += n;
            m_rhs += n;
            return *this;
        }
        auto operator-=(difference_type n) noexcept -> iterator& {
            return (*this) += -n;
        }

        [[nodiscard]] friend auto operator+(iterator it, difference_type n) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator+(difference_type n, iterator it) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator it, difference_type n) noexcept -> iterator {
            it -= n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator lhs, iterator rhs) noexcept {
            return lhs.m_lhs - rhs.m_lhs;
        }

        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return value_type(*m_lhs) - value_type(*m_rhs);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return value_type(*(m_lhs + idx)) - value_type(*(m_rhs + idx));
        }
        [[nodiscard]] auto cx_reg(std::size_t idx) const {
            const auto lhs = expression_traits::cx_reg<pack_size>(m_lhs, idx);
            const auto rhs = expression_traits::cx_reg<pack_size>(m_rhs, idx);

            return avx::sub(lhs, rhs);
        }

        [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept {
            return expression_traits::aligned(m_lhs, offset) && expression_traits::aligned(m_rhs, offset);
        }

    private:
        lhs_iterator m_lhs;
        rhs_iterator m_rhs;
    };

private:
    sub(const E1& lhs, const E2& rhs)
    : m_lhs(lhs.begin())
    , m_rhs(rhs.begin())
    , m_size(lhs.size()) {
        assert(lhs.size() == rhs.size());
    };

public:
    sub() noexcept = delete;

    sub(sub&& other) noexcept = default;
    sub(const sub&) noexcept  = default;

    ~sub() noexcept = default;

    sub& operator=(sub&& other) noexcept {
        m_lhs = std::move(other.m_lhs);
        m_rhs = std::move(other.m_rhs);
        return *this;
    };
    sub& operator=(const sub&) noexcept = delete;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_lhs, m_rhs);
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_lhs + m_size, m_rhs + m_size);
    }

    [[nodiscard]] auto operator[](std::size_t idx) const {
        return std::complex<real_type>(m_lhs[idx]) - std::complex<real_type>(m_rhs[idx]);
    };

    [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
        return m_size;
    }

private:
    lhs_iterator m_lhs;
    rhs_iterator m_rhs;
    std::size_t  m_size;
};

template<typename E1, typename E2>
    requires compatible_expression<E1, E2>
class mul : public std::ranges::view_base {
    friend auto operator*<E1, E2>(const E1& lhs, const E2& rhs);
    using lhs_iterator = decltype(std::declval<const E1>().begin());
    using rhs_iterator = decltype(std::declval<const E2>().begin());

public:
    using real_type = typename E1::real_type;

    static constexpr auto pack_size = avx::reg<real_type>::size;
    class iterator {
        friend class mul;

    private:
        iterator(lhs_iterator lhs, rhs_iterator rhs)
        : m_lhs(std::move(lhs))
        , m_rhs(std::move(rhs)){};

    public:
        using real_type        = mul::real_type;
        using value_type       = const std::complex<real_type>;
        using difference_type  = std::ptrdiff_t;
        using iterator_concept = std::random_access_iterator_tag;

        static constexpr auto pack_size = mul::pack_size;

        iterator() = default;

        iterator(const iterator& other) noexcept = default;
        iterator(iterator&& other) noexcept      = default;

        ~iterator() = default;

        iterator& operator=(const iterator& other) noexcept = default;
        iterator& operator=(iterator&& other) noexcept      = default;

        [[nodiscard]] bool operator==(const iterator& other) const noexcept {
            return (m_lhs == other.m_lhs);
        }
        [[nodiscard]] auto operator<=>(const iterator& other) const noexcept {
            return (m_lhs <=> other.m_lhs);
        }

        auto operator++() noexcept -> iterator& {
            ++m_lhs;
            ++m_rhs;
            return *this;
        }
        auto operator++(int) noexcept -> iterator {
            auto copy = *this;
            ++(*this);
            return copy;
        }
        auto operator--() noexcept -> iterator& {
            --m_lhs;
            --m_rhs;
        }
        auto operator--(int) noexcept -> iterator {
            auto copy = *this;
            --(*this);
            return copy;
        }

        auto operator+=(difference_type n) noexcept -> iterator& {
            m_lhs += n;
            m_rhs += n;
            return *this;
        }
        auto operator-=(difference_type n) noexcept -> iterator& {
            return (*this) += -n;
        }

        [[nodiscard]] friend auto operator+(iterator it, difference_type n) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator+(difference_type n, iterator it) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator it, difference_type n) noexcept -> iterator {
            it -= n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator lhs, iterator rhs) noexcept {
            return lhs.m_lhs - rhs.m_lhs;
        }

        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return value_type(*m_lhs) * value_type(*m_rhs);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return value_type(*(m_lhs + idx)) * value_type(*(m_rhs + idx));
        }
        [[nodiscard]] auto cx_reg(std::size_t idx) const {
            const auto lhs = expression_traits::cx_reg<pack_size>(m_lhs, idx);
            const auto rhs = expression_traits::cx_reg<pack_size>(m_rhs, idx);

            return avx::mul(lhs, rhs);
        }

        [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept {
            return expression_traits::aligned(m_lhs, offset) && expression_traits::aligned(m_rhs, offset);
        }

    private:
        lhs_iterator m_lhs;
        rhs_iterator m_rhs;
    };

private:
    mul(const E1& lhs, const E2& rhs)
    : m_lhs(lhs.begin())
    , m_rhs(rhs.begin())
    , m_size(lhs.size()) {
        assert(lhs.size() == rhs.size());
    };

public:
    mul() noexcept = default;

    mul(mul&& other) noexcept = default;
    mul(const mul&) noexcept  = default;

    ~mul() noexcept = default;

    mul& operator=(mul&& other) noexcept {
        m_lhs = std::move(other.m_lhs);
        m_rhs = std::move(other.m_rhs);
        return *this;
    };
    mul& operator=(const mul&) noexcept = delete;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_lhs, m_rhs);
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_lhs + m_size, m_rhs + m_size);
    }

    [[nodiscard]] auto operator[](std::size_t idx) const {
        return std::complex<real_type>(m_lhs[idx]) * std::complex<real_type>(m_rhs[idx]);
    };

    [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
        return m_size;
    }

private:
    lhs_iterator m_lhs;
    rhs_iterator m_rhs;
    std::size_t  m_size;
};

template<typename E1, typename E2>
    requires compatible_expression<E1, E2>
class div : public std::ranges::view_base {
    friend auto operator/<E1, E2>(const E1& lhs, const E2& rhs);
    using lhs_iterator = decltype(std::declval<const E1>().begin());
    using rhs_iterator = decltype(std::declval<const E2>().begin());

public:
    using real_type = typename E1::real_type;

    static constexpr auto pack_size = avx::reg<real_type>::size;
    class iterator {
        friend class div;

    private:
        iterator(lhs_iterator lhs, rhs_iterator rhs)
        : m_lhs(std::move(lhs))
        , m_rhs(std::move(rhs)){};

    public:
        using real_type        = div::real_type;
        using value_type       = const std::complex<real_type>;
        using difference_type  = std::ptrdiff_t;
        using iterator_concept = std::random_access_iterator_tag;

        static constexpr auto pack_size = div::pack_size;

        iterator() = default;

        iterator(const iterator& other) noexcept = default;
        iterator(iterator&& other) noexcept      = default;

        ~iterator() = default;

        iterator& operator=(const iterator& other) noexcept = default;
        iterator& operator=(iterator&& other) noexcept      = default;

        [[nodiscard]] bool operator==(const iterator& other) const noexcept {
            return (m_lhs == other.m_lhs);
        }
        [[nodiscard]] auto operator<=>(const iterator& other) const noexcept {
            return (m_lhs <=> other.m_lhs);
        }

        auto operator++() noexcept -> iterator& {
            ++m_lhs;
            ++m_rhs;
            return *this;
        }
        auto operator++(int) noexcept -> iterator {
            auto copy = *this;
            ++(*this);
            return copy;
        }
        auto operator--() noexcept -> iterator& {
            --m_lhs;
            --m_rhs;
        }
        auto operator--(int) noexcept -> iterator {
            auto copy = *this;
            --(*this);
            return copy;
        }

        auto operator+=(difference_type n) noexcept -> iterator& {
            m_lhs += n;
            m_rhs += n;
            return *this;
        }
        auto operator-=(difference_type n) noexcept -> iterator& {
            return (*this) += -n;
        }

        [[nodiscard]] friend auto operator+(iterator it, difference_type n) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator+(difference_type n, iterator it) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator it, difference_type n) noexcept -> iterator {
            it -= n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator lhs, iterator rhs) noexcept {
            return lhs.m_lhs - rhs.m_lhs;
        }

        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return value_type(*m_lhs) / value_type(*m_rhs);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return value_type(*(m_lhs + idx)) / value_type(*(m_rhs + idx));
        }
        [[nodiscard]] auto cx_reg(std::size_t idx) const {
            const auto lhs = expression_traits::cx_reg<pack_size>(m_lhs, idx);
            const auto rhs = expression_traits::cx_reg<pack_size>(m_rhs, idx);

            return avx::div(lhs, rhs);
        }

        [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept {
            return expression_traits::aligned(m_lhs, offset) && expression_traits::aligned(m_rhs, offset);
        }

    private:
        lhs_iterator m_lhs;
        rhs_iterator m_rhs;
    };

private:
    div(const E1& lhs, const E2& rhs)
    : m_lhs(lhs.begin())
    , m_rhs(rhs.begin())
    , m_size(lhs.size()) {
        assert(lhs.size() == rhs.size());
    };

public:
    div() noexcept = delete;

    div(div&& other) noexcept = default;
    div(const div&) noexcept  = default;

    ~div() noexcept = default;

    div& operator=(div&& other) noexcept {
        m_lhs = std::move(other.m_lhs);
        m_rhs = std::move(other.m_rhs);
        return *this;
    };
    div& operator=(const div&) noexcept = delete;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_lhs, m_rhs);
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_lhs + m_size, m_rhs + m_size);
    }

    [[nodiscard]] auto operator[](std::size_t idx) const {
        return std::complex<real_type>(m_lhs[idx]) / std::complex<real_type>(m_rhs[idx]);
    };

    [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
        return m_size;
    }

private:
    lhs_iterator m_lhs;
    rhs_iterator m_rhs;
    std::size_t  m_size;
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class scalar_add : public std::ranges::view_base {
    friend auto operator+<E, S>(const E& vector, S scalar);
    friend auto operator+<E, S>(S scalar, const E& vector);
    friend auto operator-<E, S>(const E& vector, S scalar);

public:
    using real_type = typename E::real_type;

    static constexpr auto pack_size = std::min(E::pack_size, avx::reg<real_type>::size);
    class iterator {
        friend class scalar_add;

    private:
        using vector_iterator = decltype(std::declval<const E>().begin());

        iterator(S scalar, vector_iterator vector)
        : m_vector(std::move(vector))
        , m_scalar(std::move(scalar)){};

    public:
        using real_type        = scalar_add::real_type;
        using value_type       = const std::complex<real_type>;
        using difference_type  = std::ptrdiff_t;
        using iterator_concept = std::random_access_iterator_tag;

        static constexpr auto pack_size = scalar_add::pack_size;

        iterator() = default;

        iterator(const iterator& other) noexcept = default;
        iterator(iterator&& other) noexcept      = default;

        ~iterator() = default;

        iterator& operator=(const iterator& other) noexcept = default;
        iterator& operator=(iterator&& other) noexcept      = default;


        [[nodiscard]] bool operator==(const iterator& other) const noexcept {
            return (m_vector == other.m_vector);
        }
        [[nodiscard]] auto operator<=>(const iterator& other) const noexcept {
            return (m_vector <=> other.m_vector);
        }

        auto operator++() noexcept -> iterator& {
            ++m_vector;
            return *this;
        }
        auto operator++(int) noexcept -> iterator {
            auto copy = *this;
            ++(*this);
            return copy;
        }
        auto operator--() noexcept -> iterator& {
            --m_vector;
        }
        auto operator--(int) noexcept -> iterator {
            auto copy = *this;
            --(*this);
            return copy;
        }

        auto operator+=(difference_type n) noexcept -> iterator& {
            m_vector += n;
            return *this;
        }
        auto operator-=(difference_type n) noexcept -> iterator& {
            return (*this) += -n;
        }

        [[nodiscard]] friend auto operator+(iterator it, difference_type n) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator+(difference_type n, iterator it) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator it, difference_type n) noexcept -> iterator {
            it -= n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator lhs, iterator rhs) noexcept {
            return lhs.m_vector - rhs.m_vector;
        }

        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return m_scalar + value_type(*m_vector);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return m_scalar + value_type(*(m_vector + idx));
        }
        [[nodiscard]] auto cx_reg(std::size_t idx) const {
            const auto scalar = avx::broadcast(m_scalar);
            const auto vector = expression_traits::cx_reg<pack_size>(m_vector, idx);

            return avx::add(scalar, vector);
        }

        [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept {
            return expression_traits::aligned(m_vector, offset);
        }

    private:
        vector_iterator m_vector;
        S               m_scalar;
    };

private:
    scalar_add(S scalar, const E& vector)
    : m_vector(vector)
    , m_scalar(scalar){};

public:
    scalar_add() noexcept = default;

    scalar_add(scalar_add&& other) noexcept = default;
    scalar_add(const scalar_add&) noexcept  = default;

    ~scalar_add() noexcept = default;

    scalar_add& operator=(scalar_add&& other) noexcept {
        m_scalar = std::move(other.m_scalar);
        m_vector = std::move(other.m_vector);
        return *this;
    };
    scalar_add& operator=(const scalar_add&) noexcept = delete;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_scalar, m_vector.begin());
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_scalar, m_vector.end());
    }

    [[nodiscard]] auto operator[](std::size_t idx) const {
        return m_scalar + std::complex<real_type>(m_vector[idx]);
    }
    [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
        return m_vector.size();
    }

private:
    const E m_vector;
    S       m_scalar;
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class scalar_sub : public std::ranges::view_base {
    friend auto operator-<E, S>(S scalar, const E& vector);

public:
    using real_type = typename E::real_type;

    static constexpr auto pack_size = std::min(E::pack_size, avx::reg<real_type>::size);
    class iterator {
        friend class scalar_sub;

    private:
        using vector_iterator = decltype(std::declval<const E>().begin());

        iterator(S scalar, vector_iterator vector)
        : m_vector(std::move(vector))
        , m_scalar(std::move(scalar)){};

    public:
        using real_type        = scalar_sub::real_type;
        using value_type       = const std::complex<real_type>;
        using difference_type  = std::ptrdiff_t;
        using iterator_concept = std::random_access_iterator_tag;

        static constexpr auto pack_size = scalar_sub::pack_size;

        iterator() = default;

        iterator(const iterator& other) noexcept = default;
        iterator(iterator&& other) noexcept      = default;

        ~iterator() = default;

        iterator& operator=(const iterator& other) noexcept = default;
        iterator& operator=(iterator&& other) noexcept      = default;


        [[nodiscard]] bool operator==(const iterator& other) const noexcept {
            return (m_vector == other.m_vector);
        }
        [[nodiscard]] auto operator<=>(const iterator& other) const noexcept {
            return (m_vector <=> other.m_vector);
        }

        auto operator++() noexcept -> iterator& {
            ++m_vector;
            return *this;
        }
        auto operator++(int) noexcept -> iterator {
            auto copy = *this;
            ++(*this);
            return copy;
        }
        auto operator--() noexcept -> iterator& {
            --m_vector;
        }
        auto operator--(int) noexcept -> iterator {
            auto copy = *this;
            --(*this);
            return copy;
        }

        auto operator+=(difference_type n) noexcept -> iterator& {
            m_vector += n;
            return *this;
        }
        auto operator-=(difference_type n) noexcept -> iterator& {
            return (*this) += -n;
        }

        [[nodiscard]] friend auto operator+(iterator it, difference_type n) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator+(difference_type n, iterator it) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator it, difference_type n) noexcept -> iterator {
            it -= n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator lhs, iterator rhs) noexcept {
            return lhs.m_vector - rhs.m_vector;
        }

        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return m_scalar - value_type(*m_vector);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return m_scalar - value_type(*(m_vector + idx));
        }
        [[nodiscard]] auto cx_reg(std::size_t idx) const {
            const auto scalar = avx::broadcast(m_scalar);
            const auto vector = expression_traits::cx_reg<pack_size>(m_vector, idx);

            return avx::sub(scalar, vector);
        }

        [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept {
            return expression_traits::aligned(m_vector, offset);
        }

    private:
        vector_iterator m_vector;
        S               m_scalar;
    };

private:
    scalar_sub(S scalar, const E& vector)
    : m_vector(vector)
    , m_scalar(scalar){};

public:
    scalar_sub() noexcept = default;

    scalar_sub(scalar_sub&& other) noexcept = default;
    scalar_sub(const scalar_sub&) noexcept  = default;

    ~scalar_sub() noexcept = default;

    scalar_sub& operator=(scalar_sub&& other) noexcept {
        m_scalar = std::move(other.m_scalar);
        m_vector = std::move(other.m_vector);
        return *this;
    };
    scalar_sub& operator=(const scalar_sub&) noexcept = delete;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_scalar, m_vector.begin());
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_scalar, m_vector.end());
    }

    [[nodiscard]] auto operator[](std::size_t idx) const {
        return m_scalar - std::complex<real_type>(m_vector[idx]);
    }
    [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
        return m_vector.size();
    }

private:
    const E m_vector;
    S       m_scalar;
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class scalar_mul : public std::ranges::view_base {
    friend auto operator*<E, S>(const E& vector, S scalar);
    friend auto operator*<E, S>(S scalar, const E& vector);
    friend auto operator/<E, S>(const E& vector, S scalar);

public:
    using real_type = typename E::real_type;

    static constexpr auto pack_size = std::min(E::pack_size, avx::reg<real_type>::size);
    class iterator {
        friend class scalar_mul;

    private:
        using vector_iterator = decltype(std::declval<const E>().begin());

        iterator(S scalar, vector_iterator vector)
        : m_vector(std::move(vector))
        , m_scalar(std::move(scalar)){};

    public:
        using real_type        = scalar_mul::real_type;
        using value_type       = const std::complex<real_type>;
        using difference_type  = std::ptrdiff_t;
        using iterator_concept = std::random_access_iterator_tag;

        static constexpr auto pack_size = scalar_mul::pack_size;

        iterator() = default;

        iterator(const iterator& other) noexcept = default;
        iterator(iterator&& other) noexcept      = default;

        ~iterator() = default;

        iterator& operator=(const iterator& other) noexcept = default;
        iterator& operator=(iterator&& other) noexcept      = default;


        [[nodiscard]] bool operator==(const iterator& other) const noexcept {
            return (m_vector == other.m_vector);
        }
        [[nodiscard]] auto operator<=>(const iterator& other) const noexcept {
            return (m_vector <=> other.m_vector);
        }

        auto operator++() noexcept -> iterator& {
            ++m_vector;
            return *this;
        }
        auto operator++(int) noexcept -> iterator {
            auto copy = *this;
            ++(*this);
            return copy;
        }
        auto operator--() noexcept -> iterator& {
            --m_vector;
        }
        auto operator--(int) noexcept -> iterator {
            auto copy = *this;
            --(*this);
            return copy;
        }

        auto operator+=(difference_type n) noexcept -> iterator& {
            m_vector += n;
            return *this;
        }
        auto operator-=(difference_type n) noexcept -> iterator& {
            return (*this) += -n;
        }

        [[nodiscard]] friend auto operator+(iterator it, difference_type n) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator+(difference_type n, iterator it) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator it, difference_type n) noexcept -> iterator {
            it -= n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator lhs, iterator rhs) noexcept {
            return lhs.m_vector - rhs.m_vector;
        }

        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return m_scalar * value_type(*m_vector);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return m_scalar * value_type(*(m_vector + idx));
        }
        [[nodiscard]] auto cx_reg(std::size_t idx) const {
            const auto scalar = avx::broadcast(m_scalar);
            const auto vector = expression_traits::cx_reg<pack_size>(m_vector, idx);

            return avx::mul(scalar, vector);
        }

        [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept {
            return expression_traits::aligned(m_vector, offset);
        }

    private:
        vector_iterator m_vector;
        S               m_scalar;
    };

private:
    scalar_mul(S scalar, const E& vector)
    : m_vector(vector)
    , m_scalar(scalar){};

public:
    scalar_mul() noexcept = default;

    scalar_mul(scalar_mul&& other) noexcept = default;
    scalar_mul(const scalar_mul&) noexcept  = default;

    ~scalar_mul() noexcept = default;

    scalar_mul& operator=(scalar_mul&& other) noexcept {
        m_scalar = std::move(other.m_scalar);
        m_vector = std::move(other.m_vector);
        return *this;
    };
    scalar_mul& operator=(const scalar_mul&) noexcept = delete;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_scalar, m_vector.begin());
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_scalar, m_vector.end());
    }

    [[nodiscard]] auto operator[](std::size_t idx) const {
        return m_scalar * std::complex<real_type>(m_vector[idx]);
    }
    [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
        return m_vector.size();
    }

private:
    const E m_vector;
    S       m_scalar;
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class scalar_div : public std::ranges::view_base {
    friend auto operator/<E, S>(S scalar, const E& vector);

public:
    using real_type = typename E::real_type;

    static constexpr auto pack_size = std::min(E::pack_size, avx::reg<real_type>::size);
    class iterator {
        friend class scalar_div;

    private:
        using vector_iterator = decltype(std::declval<const E>().begin());

        iterator(S scalar, vector_iterator vector)
        : m_vector(std::move(vector))
        , m_scalar(std::move(scalar)){};

    public:
        using real_type        = scalar_div::real_type;
        using value_type       = const std::complex<real_type>;
        using difference_type  = std::ptrdiff_t;
        using iterator_concept = std::random_access_iterator_tag;

        static constexpr auto pack_size = scalar_div::pack_size;

        iterator() = default;

        iterator(const iterator& other) noexcept = default;
        iterator(iterator&& other) noexcept      = default;

        ~iterator() = default;

        iterator& operator=(const iterator& other) noexcept = default;
        iterator& operator=(iterator&& other) noexcept      = default;


        [[nodiscard]] bool operator==(const iterator& other) const noexcept {
            return (m_vector == other.m_vector);
        }
        [[nodiscard]] auto operator<=>(const iterator& other) const noexcept {
            return (m_vector <=> other.m_vector);
        }

        auto operator++() noexcept -> iterator& {
            ++m_vector;
            return *this;
        }
        auto operator++(int) noexcept -> iterator {
            auto copy = *this;
            ++(*this);
            return copy;
        }
        auto operator--() noexcept -> iterator& {
            --m_vector;
        }
        auto operator--(int) noexcept -> iterator {
            auto copy = *this;
            --(*this);
            return copy;
        }

        auto operator+=(difference_type n) noexcept -> iterator& {
            m_vector += n;
            return *this;
        }
        auto operator-=(difference_type n) noexcept -> iterator& {
            return (*this) += -n;
        }

        [[nodiscard]] friend auto operator+(iterator it, difference_type n) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator+(difference_type n, iterator it) noexcept -> iterator {
            it += n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator it, difference_type n) noexcept -> iterator {
            it -= n;
            return it;
        }
        [[nodiscard]] friend auto operator-(iterator lhs, iterator rhs) noexcept {
            return lhs.m_vector - rhs.m_vector;
        }

        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return m_scalar / value_type(*m_vector);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return m_scalar / value_type(*(m_vector + idx));
        }
        [[nodiscard]] auto cx_reg(std::size_t idx) const {
            const auto scalar = avx::broadcast(m_scalar);
            const auto vector = expression_traits::cx_reg<pack_size>(m_vector, idx);

            return avx::div(scalar, vector);
        }

        [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept {
            return expression_traits::aligned(m_vector, offset);
        }

    private:
        vector_iterator m_vector;
        S               m_scalar;
    };

private:
    scalar_div(S scalar, const E& vector)
    : m_vector(vector)
    , m_scalar(scalar){};

public:
    scalar_div() noexcept = default;

    scalar_div(scalar_div&& other) noexcept = default;
    scalar_div(const scalar_div&) noexcept  = default;

    ~scalar_div() noexcept = default;

    scalar_div& operator=(scalar_div&& other) noexcept {
        m_scalar = std::move(other.m_scalar);
        m_vector = std::move(other.m_vector);
        return *this;
    };
    scalar_div& operator=(const scalar_div&) noexcept = delete;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_scalar, m_vector.begin());
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_scalar, m_vector.end());
    }

    [[nodiscard]] auto operator[](std::size_t idx) const {
        return m_scalar / std::complex<real_type>(m_vector[idx]);
    }
    [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
        return m_vector.size();
    }

private:
    const E m_vector;
    S       m_scalar;
};

}    // namespace internal

// #region operator definitions

template<typename E1, typename E2>
    requires internal::compatible_expression<E1, E2>
inline auto operator+(const E1& lhs, const E2& rhs) {
    return internal::add(lhs, rhs);
};
template<typename E1, typename E2>
    requires internal::compatible_expression<E1, E2>
inline auto operator-(const E1& lhs, const E2& rhs) {
    return internal::sub(lhs, rhs);
};
template<typename E1, typename E2>
    requires internal::compatible_expression<E1, E2>
inline auto operator*(const E1& lhs, const E2& rhs) {
    return internal::mul(lhs, rhs);
};
template<typename E1, typename E2>
    requires internal::compatible_expression<E1, E2>
inline auto operator/(const E1& lhs, const E2& rhs) {
    return internal::div(lhs, rhs);
};

template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator+(const E& vector, S scalar) {
    return internal::scalar_add(scalar, vector);
}
template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator+(S scalar, const E& vector) {
    return internal::scalar_add(scalar, vector);
}

template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator-(const E& vector, S scalar) {
    return internal::scalar_add(-scalar, vector);
}
template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator-(S scalar, const E& vector) {
    return internal::scalar_sub(scalar, vector);
}

template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator*(const E& vector, S scalar) {
    return internal::scalar_mul(scalar, vector);
}
template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator*(S scalar, const E& vector) {
    return internal::scalar_mul(scalar, vector);
}

template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator/(const E& vector, S scalar) {
    return internal::scalar_mul(S(1) / scalar, vector);
}
template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator/(S scalar, const E& vector) {
    return internal::scalar_div(scalar, vector);
}

template<typename E, typename T, typename Allocator, std::size_t PackSize>
    requires packed_floating_point<T, PackSize> && (!std::same_as<E, vector<T, Allocator, PackSize>>) &&
             requires(subrange<T, false, pcx::dynamic_size, PackSize> subrange, E e) { e + subrange; }
inline auto operator+(const E& expression, const vector<T, Allocator, PackSize>& vector) {
    return expression + subrange(vector.begin(), vector.size());
};
template<typename T, typename Allocator, std::size_t PackSize, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(subrange<T, false, pcx::dynamic_size, PackSize> subrange, E e) { subrange + e; }
inline auto operator+(const vector<T, Allocator, PackSize>& vector, const E& expression) {
    return subrange(vector.begin(), vector.size()) + expression;
};

template<typename E, typename T, typename Allocator, std::size_t PackSize>
    requires packed_floating_point<T, PackSize> && (!std::same_as<E, vector<T, Allocator, PackSize>>) &&
             requires(subrange<T, false, pcx::dynamic_size, PackSize> subrange, E e) { e - subrange; }
inline auto operator-(const E& expression, const vector<T, Allocator, PackSize>& vector) {
    return expression - subrange(vector.begin(), vector.size());
};
template<typename T, typename Allocator, std::size_t PackSize, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(subrange<T, false, pcx::dynamic_size, PackSize> subrange, E e) { subrange - e; }
inline auto operator-(const vector<T, Allocator, PackSize>& vector, const E& expression) {
    return subrange(vector.begin(), vector.size()) - expression;
};

template<typename E, typename T, typename Allocator, std::size_t PackSize>
    requires packed_floating_point<T, PackSize> && (!std::same_as<E, vector<T, Allocator, PackSize>>) &&
             requires(subrange<T, false, pcx::dynamic_size, PackSize> subrange, E e) { e* subrange; }
inline auto operator*(const E& expression, const vector<T, Allocator, PackSize>& vector) {
    return expression * subrange(vector.begin(), vector.size());
};
template<typename T, typename Allocator, std::size_t PackSize, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(subrange<T, false, pcx::dynamic_size, PackSize> subrange, E e) { subrange* e; }
inline auto operator*(const vector<T, Allocator, PackSize>& vector, const E& expression) {
    return subrange(vector.begin(), vector.size()) * expression;
};

template<typename E, typename T, typename Allocator, std::size_t PackSize>
    requires packed_floating_point<T, PackSize> && (!std::same_as<E, vector<T, Allocator, PackSize>>) &&
             requires(subrange<T, false, pcx::dynamic_size, PackSize> subrange, E e) { e / subrange; }
inline auto operator/(const E& expression, const vector<T, Allocator, PackSize>& vector) {
    return expression / subrange(vector.begin(), vector.size());
};
template<typename T, typename Allocator, std::size_t PackSize, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(subrange<T, false, pcx::dynamic_size, PackSize> subrange, E e) { subrange / e; }
inline auto operator/(const vector<T, Allocator, PackSize>& vector, const E& expression) {
    return subrange(vector.begin(), vector.size()) / expression;
};

// #endregion operator definitions
}    // namespace pcx
#endif