#ifndef VECTOR_ARITHM_HPP
#define VECTOR_ARITHM_HPP
#include "simd_common.hpp"
#include "vector_util.hpp"

#include <cassert>
#include <complex>
#include <concepts>
#include <cstddef>
#include <immintrin.h>
#include <ranges>

namespace pcx {


namespace simd {

// /**
//   * @brief Performs butterfly operation, then multiplies diff by imaginary unit RhsRotI times;
//   *
//   * @tparam RhsRotI number of multiplications by imaginary unity
//   */
// template<uint RhsRotI = 0, uZ PackSize, typename T>
//     requires(RhsRotI < 4)
// inline auto ibtfly(cx_reg<T, false, PackSize> lhs, cx_reg<T, false, PackSize> rhs) {
//     cx_reg<T, false, PackSize> s;
//     cx_reg<T, false, PackSize> d;
//     if constexpr (RhsRotI == 0) {
//         auto s_re = add(lhs.real, rhs.real);
//         auto d_re = sub(lhs.real, rhs.real);
//         auto s_im = add(lhs.imag, rhs.imag);
//         auto d_im = sub(lhs.imag, rhs.imag);
//         s         = {s_re, s_im};
//         d         = {d_re, d_im};
//     } else if constexpr (RhsRotI == 1) {
//         auto s_re = add(lhs.real, rhs.real);
//         auto d_im = sub(lhs.real, rhs.real);
//         auto s_im = add(lhs.imag, rhs.imag);
//         auto d_re = sub(rhs.imag, lhs.imag);
//         s         = {s_re, s_im};
//         d         = {d_re, d_im};
//     } else if constexpr (RhsRotI == 2) {
//         auto s_re = add(lhs.real, rhs.real);
//         auto d_re = sub(rhs.real, lhs.real);
//         auto s_im = add(lhs.imag, rhs.imag);
//         auto d_im = sub(rhs.imag, lhs.imag);
//         s         = {s_re, s_im};
//         d         = {d_re, d_im};
//     } else {
//         auto s_re = add(lhs.real, rhs.real);
//         auto d_im = sub(rhs.real, lhs.real);
//         auto s_im = add(lhs.imag, rhs.imag);
//         auto d_re = sub(lhs.imag, rhs.imag);
//         s         = {s_re, s_im};
//         d         = {d_re, d_im};
//     }
//
//     return std::make_tuple(s, d);
// }
//
// /**
//   * @brief Multiplies rhs by imaginary unit RhsRotI times, then performs butterfly operation;
//   *
//   * @tparam RhsRotI number of multiplications by imaginary unity
//   */
// template<uint RhsRotI = 0, uZ PackSize, typename T>
//     requires(RhsRotI < 4)
// inline auto btfly(cx_reg<T, false, PackSize> lhs, cx_reg<T, false, PackSize> rhs) {
//     cx_reg<T, false, PackSize> s;
//     cx_reg<T, false, PackSize> d;
//     if constexpr (RhsRotI == 0) {
//         auto s_re = add(lhs.real, rhs.real);
//         auto d_re = sub(lhs.real, rhs.real);
//         auto s_im = add(lhs.imag, rhs.imag);
//         auto d_im = sub(lhs.imag, rhs.imag);
//         s         = {s_re, s_im};
//         d         = {d_re, d_im};
//     } else if constexpr (RhsRotI == 1) {
//         auto s_re = sub(lhs.real, rhs.imag);
//         auto d_re = add(lhs.real, rhs.imag);
//         auto s_im = add(lhs.imag, rhs.real);
//         auto d_im = sub(lhs.imag, rhs.real);
//         s         = {s_re, s_im};
//         d         = {d_re, d_im};
//     } else if constexpr (RhsRotI == 2) {
//         auto s_re = sub(lhs.real, rhs.real);
//         auto d_re = add(lhs.real, rhs.real);
//         auto s_im = sub(lhs.imag, rhs.imag);
//         auto d_im = add(lhs.imag, rhs.imag);
//         s         = {s_re, s_im};
//         d         = {d_re, d_im};
//     } else {
//         auto s_re = add(lhs.real, rhs.imag);
//         auto d_re = sub(lhs.real, rhs.imag);
//         auto s_im = sub(lhs.imag, rhs.real);
//         auto d_im = add(lhs.imag, rhs.real);
//         s         = {s_re, s_im};
//         d         = {d_re, d_im};
//     }
//     return std::make_tuple(s, d);
// }

template<typename... Args>
// NOLINTNEXTLINE(*-c-arrays)
inline auto mul(const cx_reg<Args> (&... args)[2]) {
    auto tup      = std::make_tuple(args...);
    auto real_mul = [](auto opearands) {
        auto lhs  = opearands[0];
        auto rhs  = opearands[1];
        auto real = simd::mul(lhs.real, rhs.real);
        auto imag = simd::mul(lhs.real, rhs.imag);
        return std::make_tuple(lhs, rhs, real, imag);
    };
    auto imag_mul = [](auto opearands) {
        auto lhs   = std::get<0>(opearands);
        auto rhs   = std::get<1>(opearands);
        auto real_ = std::get<2>(opearands);
        auto imag_ = std::get<3>(opearands);

        using reg_t = decltype(lhs);

        auto real = simd::fnmadd(lhs.imag, rhs.imag, real_);
        auto imag = simd::fmadd(lhs.imag, rhs.real, imag_);
        return reg_t{real, imag};
    };

    auto tmp = pcx::detail_::apply_for_each(real_mul, tup);
    return pcx::detail_::apply_for_each(imag_mul, tmp);
}
}    // namespace simd

namespace detail_ {

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
        auto data_  = iterator.cx_reg(offset);
        auto [data] = simd::repack2<PackSize>(data_);
        return data;
    }
    /**
     * @brief Extracts simd vector from iterator.
     *
     * @tparam PackSize required pack size
     * @param iterator  must be aligned
     * @param offset    must be a multiple of SIMD vector size
     */
    template<std::size_t PackSize, typename T, bool Const, uZ IPackSize>
    [[nodiscard]] static constexpr auto cx_reg(const iterator<T, Const, IPackSize>& iterator,
                                               std::size_t                          offset) {
        auto addr = simd::ra_addr<IPackSize>(&(*iterator), offset);
        auto data = simd::cxload<IPackSize, PackSize>(addr);
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
                expression_traits::cx_reg<simd::reg<typename E::real_type>::size>(expression.begin(), idx)
            } -> std::same_as<simd::cx_reg<typename E::real_type, false>>;
        } ||
        requires(E expression, std::size_t idx) {
            {
                expression_traits::cx_reg<simd::reg<typename E::real_type>::size>(expression.begin(), idx)
            } -> std::same_as<simd::cx_reg<typename E::real_type, true>>;
        });

template<typename E1, typename E2>
concept compatible_expression = vector_expression<E1> && vector_expression<E2> &&
                                std::same_as<typename E1::real_type, typename E2::real_type>;

template<typename Expression, typename Scalar>
concept compatible_scalar =
    vector_expression<Expression> && (std::same_as<typename Expression::real_type, Scalar> ||
                                      std::same_as<std::complex<typename Expression::real_type>, Scalar>);

}    // namespace detail_

// #region operator forward declarations

template<typename E1, typename E2>
    requires detail_::compatible_expression<E1, E2>
auto operator+(const E1& lhs, const E2& rhs);
template<typename E1, typename E2>
    requires detail_::compatible_expression<E1, E2>
auto operator-(const E1& lhs, const E2& rhs);
template<typename E1, typename E2>
    requires detail_::compatible_expression<E1, E2>
auto operator*(const E1& lhs, const E2& rhs);
template<typename E1, typename E2>
    requires detail_::compatible_expression<E1, E2>
auto operator/(const E1& lhs, const E2& rhs);

template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
auto operator+(const E& vector, S scalar);
template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
auto operator+(S scalar, const E& vector);

template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
auto operator-(const E& vector, S scalar);
template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
auto operator-(S scalar, const E& vector);

template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
auto operator*(const E& vector, S scalar);
template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
auto operator*(S scalar, const E& vector);

template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
auto operator/(const E& vector, S scalar);
template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
auto operator/(S scalar, const E& vector);

template<typename E>
    requires detail_::vector_expression<E>
auto conj(const E& vector);

// #endregion operator forward declarations

namespace detail_ {

template<typename E1, typename E2>
    requires compatible_expression<E1, E2>
class add : public std::ranges::view_base {
    friend auto operator+<E1, E2>(const E1& lhs, const E2& rhs);
    using lhs_iterator = decltype(std::declval<const E1>().begin());
    using rhs_iterator = decltype(std::declval<const E2>().begin());

public:
    using real_type = typename E1::real_type;
    static constexpr auto pack_size =
        std::min(std::max(E1::pack_size, E2::pack_size), simd::reg<real_type>::size);
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
            if constexpr (pack_size < simd::reg<real_type>::size) {
                auto c_lhs = simd::apply_conj(lhs);
                auto c_rhs = simd::apply_conj(rhs);
                return simd::add(c_lhs, c_rhs);
            } else {
                return simd::add(lhs, rhs);
            }
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
        m_lhs  = std::move(other.m_lhs);
        m_rhs  = std::move(other.m_rhs);
        m_size = other.m_size;
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
        std::min(std::max(E1::pack_size, E2::pack_size), simd::reg<real_type>::size);
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
            if constexpr (pack_size < simd::reg<real_type>::size) {
                auto c_lhs = simd::apply_conj(lhs);
                auto c_rhs = simd::apply_conj(rhs);
                return simd::sub(c_lhs, c_rhs);
            } else {
                return simd::sub(lhs, rhs);
            }
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
        m_lhs  = std::move(other.m_lhs);
        m_rhs  = std::move(other.m_rhs);
        m_size = other.m_size;
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

    static constexpr auto pack_size = simd::reg<real_type>::size;
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

            return simd::mul(lhs, rhs);
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
    mul() noexcept = delete;

    mul(mul&& other) noexcept = default;
    mul(const mul&) noexcept  = default;

    ~mul() noexcept = default;

    mul& operator=(mul&& other) noexcept {
        m_lhs  = std::move(other.m_lhs);
        m_rhs  = std::move(other.m_rhs);
        m_size = other.m_size;
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

    static constexpr auto pack_size = simd::reg<real_type>::size;
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

            return simd::div(lhs, rhs);
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
        m_lhs  = std::move(other.m_lhs);
        m_rhs  = std::move(other.m_rhs);
        m_size = other.m_size;
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

    static constexpr auto pack_size = simd::reg<real_type>::size;
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
            const auto scalar = simd::broadcast(m_scalar);
            const auto vector = expression_traits::cx_reg<pack_size>(m_vector, idx);

            return simd::add(scalar, vector);
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
    E m_vector;
    S m_scalar;
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class scalar_sub : public std::ranges::view_base {
    friend auto operator-<E, S>(S scalar, const E& vector);

public:
    using real_type = typename E::real_type;

    static constexpr auto pack_size = simd::reg<real_type>::size;
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
            const auto scalar = simd::broadcast(m_scalar);
            const auto vector = expression_traits::cx_reg<pack_size>(m_vector, idx);

            return simd::sub(scalar, vector);
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
    E m_vector;
    S m_scalar;
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class scalar_mul : public std::ranges::view_base {
    friend auto operator*<E, S>(const E& vector, S scalar);
    friend auto operator*<E, S>(S scalar, const E& vector);
    friend auto operator/<E, S>(const E& vector, S scalar);

public:
    using real_type = typename E::real_type;

    static constexpr auto pack_size = std::same_as<S, std::complex<real_type>>
                                          ? simd::reg<real_type>::size
                                          : std::min(E::pack_size, simd::reg<real_type>::size);
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
            const auto scalar = simd::broadcast(m_scalar);
            const auto vector = expression_traits::cx_reg<pack_size>(m_vector, idx);

            return simd::mul(scalar, vector);
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
    E m_vector;
    S m_scalar;
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class scalar_div : public std::ranges::view_base {
    friend auto operator/<E, S>(S scalar, const E& vector);

public:
    using real_type = typename E::real_type;

    static constexpr auto pack_size = simd::reg<real_type>::size;
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
            const auto scalar = simd::broadcast(m_scalar);
            const auto vector = expression_traits::cx_reg<pack_size>(m_vector, idx);

            return simd::div(scalar, vector);
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
    E m_vector;
    S m_scalar;
};


template<typename E>
    requires vector_expression<E>
class conjugate : public std::ranges::view_base {
    friend auto pcx::conj<>(const E& vector);

public:
    using real_type = typename E::real_type;

    static constexpr auto pack_size = E::pack_size;
    class iterator {
        friend class conjugate;

    private:
        using vector_iterator = decltype(std::declval<const E>().begin());

        explicit iterator(vector_iterator vector)
        : m_vector(std::move(vector)){};

    public:
        using real_type        = conjugate::real_type;
        using value_type       = const std::complex<real_type>;
        using difference_type  = std::ptrdiff_t;
        using iterator_concept = std::random_access_iterator_tag;

        static constexpr auto pack_size = conjugate::pack_size;

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
            return std::conj(value_type(*m_vector));
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return std::conj(value_type(*(m_vector + idx)));
        }
        [[nodiscard]] auto cx_reg(std::size_t idx) const {
            const auto vector = expression_traits::cx_reg<pack_size>(m_vector, idx);
            return simd::conj(vector);
        }

        [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept {
            return expression_traits::aligned(m_vector, offset);
        }

    private:
        vector_iterator m_vector;
    };

private:
    explicit conjugate(const E& vector)
    : m_vector(vector){};

public:
    conjugate() noexcept = default;

    conjugate(conjugate&& other) noexcept = default;
    conjugate(const conjugate&) noexcept  = default;

    ~conjugate() noexcept = default;

    conjugate& operator=(conjugate&& other) noexcept {
        m_vector = std::move(other.m_vector);
        return *this;
    };
    conjugate& operator=(const conjugate&) noexcept = delete;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_vector.begin());
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_vector.end());
    }

    [[nodiscard]] auto operator[](std::size_t idx) const {
        return std::conj(std::complex<real_type>(m_vector[idx]));
    }
    [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
        return m_vector.size();
    }

private:
    E m_vector;
};

}    // namespace detail_

// #region operator definitions

template<typename E1, typename E2>
    requires detail_::compatible_expression<E1, E2>
inline auto operator+(const E1& lhs, const E2& rhs) {
    return detail_::add(lhs, rhs);
};
template<typename E1, typename E2>
    requires detail_::compatible_expression<E1, E2>
inline auto operator-(const E1& lhs, const E2& rhs) {
    return detail_::sub(lhs, rhs);
};
template<typename E1, typename E2>
    requires detail_::compatible_expression<E1, E2>
inline auto operator*(const E1& lhs, const E2& rhs) {
    return detail_::mul(lhs, rhs);
};
template<typename E1, typename E2>
    requires detail_::compatible_expression<E1, E2>
inline auto operator/(const E1& lhs, const E2& rhs) {
    return detail_::div(lhs, rhs);
};

template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator+(const E& vector, S scalar) {
    return detail_::scalar_add(scalar, vector);
}
template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator+(S scalar, const E& vector) {
    return detail_::scalar_add(scalar, vector);
}

template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator-(const E& vector, S scalar) {
    return detail_::scalar_add(-scalar, vector);
}
template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator-(S scalar, const E& vector) {
    return detail_::scalar_sub(scalar, vector);
}

template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator*(const E& vector, S scalar) {
    return detail_::scalar_mul(scalar, vector);
}
template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator*(S scalar, const E& vector) {
    return detail_::scalar_mul(scalar, vector);
}

template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator/(const E& vector, S scalar) {
    return detail_::scalar_mul(S(1) / scalar, vector);
}
template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator/(S scalar, const E& vector) {
    return detail_::scalar_div(scalar, vector);
}

template<typename E>
    requires detail_::vector_expression<E>
auto conj(const E& vector) {
    return detail_::conjugate<E>(vector);
}

template<typename E, typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize> && (!std::same_as<E, vector<T, PackSize, Allocator>>) &&
             requires(subrange<T, false, PackSize> subrange, E e) { e + subrange; }
inline auto operator+(const E& expression, const vector<T, PackSize, Allocator>& vector) {
    return expression + subrange(vector.begin(), vector.size());
};
template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(subrange<T, false, PackSize> subrange, E e) { subrange + e; }
inline auto operator+(const vector<T, PackSize, Allocator>& vector, const E& expression) {
    return subrange(vector.begin(), vector.size()) + expression;
};

template<typename E, typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize> && (!std::same_as<E, vector<T, PackSize, Allocator>>) &&
             requires(subrange<T, false, PackSize> subrange, E e) { e - subrange; }
inline auto operator-(const E& expression, const vector<T, PackSize, Allocator>& vector) {
    return expression - subrange(vector.begin(), vector.size());
};
template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(subrange<T, false, PackSize> subrange, E e) { subrange - e; }
inline auto operator-(const vector<T, PackSize, Allocator>& vector, const E& expression) {
    return subrange(vector.begin(), vector.size()) - expression;
};

template<typename E, typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize> && (!std::same_as<E, vector<T, PackSize, Allocator>>) &&
             requires(subrange<T, false, PackSize> subrange, E e) { e* subrange; }
inline auto operator*(const E& expression, const vector<T, PackSize, Allocator>& vector) {
    return expression * subrange(vector.begin(), vector.size());
};
template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(subrange<T, false, PackSize> subrange, E e) { subrange* e; }
inline auto operator*(const vector<T, PackSize, Allocator>& vector, const E& expression) {
    return subrange(vector.begin(), vector.size()) * expression;
};

template<typename E, typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize> && (!std::same_as<E, vector<T, PackSize, Allocator>>) &&
             requires(subrange<T, false, PackSize> subrange, E e) { e / subrange; }
inline auto operator/(const E& expression, const vector<T, PackSize, Allocator>& vector) {
    return expression / subrange(vector.begin(), vector.size());
};
template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(subrange<T, false, PackSize> subrange, E e) { subrange / e; }
inline auto operator/(const vector<T, PackSize, Allocator>& vector, const E& expression) {
    return subrange(vector.begin(), vector.size()) / expression;
};

template<typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize>    // &&
//  requires(subrange<T, false, PackSize> subrange) { conj(subrange); }
inline auto conj(const vector<T, PackSize, Allocator>& vector) {
    return conj(subrange(vector.begin(), vector.size()));
};

// #endregion operator definitions
}    // namespace pcx
#endif