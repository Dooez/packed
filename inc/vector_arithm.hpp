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
// TODO(Dooez): switch to rv::const_iterator_t
template<typename R>
using const_iterator_t = decltype(rv::cbegin(std::declval<R&>()));

template<typename Iter, typename Range>
concept iterator_of = std::same_as<rv::iterator_t<Range>, Iter>    //
                      || std::same_as<const_iterator_t<Range>, Iter>;

template<typename T>
struct expr_traaits2 {
    using real_type                                 = decltype([] {});
    static constexpr uZ   pack_size                 = 0;
    static constexpr bool enable_vector_expressions = false;
};

template<typename R>
    requires complex_vector<R>
struct expr_traaits2<R> : cx_vector_traits<R> {
    template<uZ PackSize, typename Iter>
        requires iterator_of<Iter, R>
    static auto cx_reg(const Iter& iterator, uZ offset) {
        constexpr uZ pack_size = cx_vector_traits<R>::pack_size;
        auto         addr      = simd::ra_addr<pack_size>(&(*iterator), offset);
        auto         data      = simd::cxload<pack_size, PackSize>(addr);
        return data;
    }
};

class vecexpr_base {};

template<typename R>
    requires std::derived_from<R, vecexpr_base>
struct expr_traaits2<R> {
    static constexpr uZ   pack_size                 = R::pack_size;
    static constexpr bool always_aligned            = false;    // R::always_aligned; TODO:(REMOVE FALSE)
    static constexpr bool enable_vector_expressions = true;

    using real_type        = R::real_type;
    using iterator_t       = rv::iterator_t<R>;
    using const_iterator_t = decltype(rv::cbegin(std::declval<R&>()));

    template<uZ PackSize, typename Iter>
        requires iterator_of<Iter, R> &&
                 requires(const Iter& iterator, uZ offset) { iterator.cx_reg(offset); }
    static auto cx_reg(const Iter& iterator, uZ offset) {
        auto data_  = iterator.cx_reg(offset);
        auto [data] = simd::repack2<PackSize>(data_);
        return data;
    }

    template<typename Iter>
        requires iterator_of<Iter, R>
    static auto aligned(const Iter& iterator) -> bool {
        return iterator.aligned();
    };

    template<typename Iter>
        requires(iterator_of<Iter, R> && always_aligned)
    constexpr static auto aligned(const Iter& /*iterator*/) -> bool {
        return true;
    };
};


template<typename T>
concept vecexpr = expr_traaits2<T>::enable_vector_expressions;
template<typename T, typename U>
concept compatible_vecexpr = vecexpr<T>                                               //
                             && vecexpr<U>                                            //
                             && std::same_as<typename expr_traaits2<T>::real_type,    //
                                             typename expr_traaits2<U>::real_type>;

template<typename Expression, typename Scalar>
concept compatible_scalar =
    vecexpr<Expression> &&
    (std::same_as<typename expr_traaits2<Expression>::real_type, Scalar> ||
     std::same_as<std::complex<typename expr_traaits2<Expression>::real_type>, Scalar>);

template<typename Impl, typename LhsIter, typename RhsIter>
class ee_iter_base {
protected:
    LhsIter m_lhs;    // NOLINT(*non-private*)
    RhsIter m_rhs;    // NOLINT(*non-private*)

    ee_iter_base(const LhsIter& lhs, const RhsIter& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs){};

public:
    using difference_type  = iZ;
    using iterator_concept = std::random_access_iterator_tag;

    ee_iter_base()                               = default;
    ee_iter_base(ee_iter_base&&)                 = default;
    ee_iter_base(const ee_iter_base&)            = default;
    ee_iter_base& operator=(ee_iter_base&&)      = default;
    ee_iter_base& operator=(const ee_iter_base&) = default;
    ~ee_iter_base()                              = default;

    [[nodiscard]] friend bool operator==(const Impl& lhs, const Impl& rhs) noexcept {
        return (lhs.m_lhs == rhs.m_lhs);
    }
    [[nodiscard]] friend auto operator<=>(const Impl& lhs, const Impl& rhs) noexcept {
        return (lhs.m_lhs <=> rhs.m_lhs);
    }

    auto operator++() noexcept -> Impl& {
        ++m_lhs;
        ++m_rhs;
        return *static_cast<Impl*>(this);
    }
    auto operator++(int) noexcept -> Impl {
        auto ithis = static_cast<Impl*>(this);
        auto copy  = *ithis;
        ++(*ithis);
        return copy;
    }
    auto operator--() noexcept -> Impl& {
        --m_lhs;
        --m_rhs;
        return *static_cast<Impl*>(this);
    }
    auto operator--(int) noexcept -> Impl {
        auto ithis = static_cast<Impl*>(this);
        auto copy  = *ithis;
        --(ithis);
        return copy;
    }

    auto operator+=(difference_type n) noexcept -> Impl& {
        m_lhs += n;
        m_rhs += n;
        return *static_cast<Impl*>(this);
    }
    auto operator-=(difference_type n) noexcept -> Impl& {
        return *static_cast<Impl*>(this) += -n;
    }

    [[nodiscard]] friend auto operator+(Impl it, difference_type n) noexcept -> Impl {
        it += n;
        return it;
    }
    [[nodiscard]] friend auto operator+(difference_type n, Impl it) noexcept -> Impl {
        it += n;
        return it;
    }
    [[nodiscard]] friend auto operator-(Impl it, difference_type n) noexcept -> Impl {
        it -= n;
        return it;
    }
    [[nodiscard]] friend auto operator-(Impl lhs, Impl rhs) noexcept {
        return lhs.m_lhs - rhs.m_lhs;
    }
};

template<typename Impl, typename Iter>
class e_iter_base {
protected:
    Iter m_iter;    // NOLINT(*non-private*)

    explicit e_iter_base(const Iter& rhs)
    : m_iter(rhs){};

public:
    using difference_type  = iZ;
    using iterator_concept = std::random_access_iterator_tag;

    e_iter_base()                              = default;
    e_iter_base(e_iter_base&&)                 = default;
    e_iter_base(const e_iter_base&)            = default;
    e_iter_base& operator=(e_iter_base&&)      = default;
    e_iter_base& operator=(const e_iter_base&) = default;
    ~e_iter_base()                             = default;

    [[nodiscard]] friend bool operator==(const Impl& lhs, const Impl& rhs) noexcept {
        return (lhs.m_iter == rhs.m_iter);
    }
    [[nodiscard]] friend auto operator<=>(const Impl& lhs, const Impl& rhs) noexcept {
        return (lhs.m_iter <=> rhs.m_iter);
    }

    auto operator++() noexcept -> Impl& {
        ++m_iter;
        return *static_cast<Impl*>(this);
    }
    auto operator++(int) noexcept -> Impl {
        auto ithis = static_cast<Impl*>(this);
        auto copy  = *ithis;
        ++(*ithis);
        return copy;
    }
    auto operator--() noexcept -> Impl& {
        --m_iter;
        return *static_cast<Impl*>(this);
    }
    auto operator--(int) noexcept -> Impl {
        auto ithis = static_cast<Impl*>(this);
        auto copy  = *ithis;
        --(ithis);
        return copy;
    }

    auto operator+=(difference_type n) noexcept -> Impl& {
        m_iter += n;
        return *static_cast<Impl*>(this);
    }
    auto operator-=(difference_type n) noexcept -> Impl& {
        return *static_cast<Impl*>(this) += -n;
    }

    [[nodiscard]] friend auto operator+(Impl it, difference_type n) noexcept -> Impl {
        it += n;
        return it;
    }
    [[nodiscard]] friend auto operator+(difference_type n, Impl it) noexcept -> Impl {
        it += n;
        return it;
    }
    [[nodiscard]] friend auto operator-(Impl it, difference_type n) noexcept -> Impl {
        it -= n;
        return it;
    }
    [[nodiscard]] friend auto operator-(Impl lhs, Impl rhs) noexcept {
        return lhs.m_iter - rhs.m_iter;
    }
};
}    // namespace detail_


// #region operator forward declarations

template<typename E1, typename E2>
    requires detail_::compatible_vecexpr<E1, E2>
auto operator+(const E1& lhs, const E2& rhs);
template<typename E1, typename E2>
    requires detail_::compatible_vecexpr<E1, E2>
auto operator-(const E1& lhs, const E2& rhs);
template<typename E1, typename E2>
    requires detail_::compatible_vecexpr<E1, E2>
auto operator*(const E1& lhs, const E2& rhs);
template<typename E1, typename E2>
    requires detail_::compatible_vecexpr<E1, E2>
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
    requires detail_::vecexpr<E>
auto conj(const E& vector);

// #endregion operator forward declarations

namespace detail_ {

template<typename ELhs, typename ERhs>
// requires compatible_vecexpr<ELhs, ERhs>
class add2 final
: public rv::view_base
, public vecexpr_base {
    friend auto operator+ <ELhs, ERhs>(const ELhs& lhs, const ERhs& rhs);
    using lhs_traits   = expr_traaits2<ELhs>;
    using rhs_traits   = expr_traaits2<ERhs>;
    using lhs_iterator = typename lhs_traits::const_iterator_t;
    using rhs_iterator = typename rhs_traits::const_iterator_t;

public:
    using real_type = lhs_traits::real_type;

    static constexpr auto pack_size = std::min(std::max(lhs_traits::pack_size,    //
                                                        rhs_traits::pack_size),
                                               simd::reg<real_type>::size);

    static constexpr bool always_aligned = lhs_traits::always_aligned    //
                                           && rhs_traits::always_aligned;

    using value_type      = std::complex<real_type>;
    using difference_type = iZ;

private:
    class iterator : public ee_iter_base<iterator, lhs_iterator, rhs_iterator> {
        friend class add2;

        iterator(const lhs_iterator& lhs, const rhs_iterator& rhs)
        : ee_iter_base<iterator, lhs_iterator, rhs_iterator>(lhs, rhs){};

    public:
        iterator() = default;
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return value_type(*this->m_lhs) + value_type(*this->m_rhs);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return value_type(*(this->m_lhs + idx)) + value_type(*(this->m_rhs + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            const auto lhs = lhs_traits::template cx_reg<pack_size>(this->m_lhs, idx);
            const auto rhs = rhs_traits::template cx_reg<pack_size>(this->m_rhs, idx);
            if constexpr (pack_size < simd::reg<real_type>::size) {
                auto c_lhs = simd::apply_conj(lhs);
                auto c_rhs = simd::apply_conj(rhs);
                return simd::add(c_lhs, c_rhs);
            } else {
                return simd::add(lhs, rhs);
            }
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return lhs_traits::aligned(this->m_lhs) && rhs_traits::aligned(this->m_rhs);
        }
    };

    add2(const ELhs& lhs, const ERhs& rhs)
    : m_lhs(rv::cbegin(lhs))
    , m_rhs(rv::cbegin(rhs))
    , m_size(rv::size(lhs)) {
        assert(rv::size(lhs) == rv::size(rhs));
    };

public:
    add2() noexcept                       = delete;
    add2(add2&&) noexcept                 = default;
    add2(const add2&) noexcept            = default;
    add2& operator=(add2&&) noexcept      = delete;
    add2& operator=(const add2&) noexcept = delete;
    ~add2() noexcept                      = default;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_lhs, m_rhs);
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_lhs + m_size, m_rhs + m_size);
    }

    [[nodiscard]] auto operator[](uZ idx) const {
        return std::complex<real_type>(m_lhs[idx]) + std::complex<real_type>(m_rhs[idx]);
    };

    [[nodiscard]] constexpr auto size() const noexcept -> uZ {
        return m_size;
    }

private:
    lhs_iterator m_lhs{};
    rhs_iterator m_rhs{};
    uZ           m_size{};
};

template<typename ELhs, typename ERhs>
// requires compatible_vecexpr<ELhs, ERhs>
class sub2 final
: public rv::view_base
, public vecexpr_base {
    friend auto operator- <ELhs, ERhs>(const ELhs& lhs, const ERhs& rhs);
    using lhs_traits   = expr_traaits2<ELhs>;
    using rhs_traits   = expr_traaits2<ERhs>;
    using lhs_iterator = typename lhs_traits::const_iterator_t;
    using rhs_iterator = typename rhs_traits::const_iterator_t;

public:
    using real_type = lhs_traits::real_type;

    static constexpr auto pack_size = std::min(std::max(lhs_traits::pack_size,    //
                                                        rhs_traits::pack_size),
                                               simd::reg<real_type>::size);

    static constexpr bool always_aligned = lhs_traits::always_aligned    //
                                           && rhs_traits::always_aligned;

    using value_type      = std::complex<real_type>;
    using difference_type = iZ;

private:
    class iterator : public ee_iter_base<iterator, lhs_iterator, rhs_iterator> {
        friend class sub2;

        iterator(const lhs_iterator& lhs, const rhs_iterator& rhs)
        : ee_iter_base<iterator, lhs_iterator, rhs_iterator>(lhs, rhs){};

    public:
        iterator() = default;
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return value_type(*this->m_lhs) - value_type(*this->m_rhs);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return value_type(*(this->m_lhs + idx)) - value_type(*(this->m_rhs + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            const auto lhs = lhs_traits::template cx_reg<pack_size>(this->m_lhs, idx);
            const auto rhs = rhs_traits::template cx_reg<pack_size>(this->m_rhs, idx);
            if constexpr (pack_size < simd::reg<real_type>::size) {
                auto c_lhs = simd::apply_conj(lhs);
                auto c_rhs = simd::apply_conj(rhs);
                return simd::sub(c_lhs, c_rhs);
            } else {
                return simd::sub(lhs, rhs);
            }
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return lhs_traits::aligned(this->m_lhs) && rhs_traits::aligned(this->m_rhs);
        }
    };

    sub2(const ELhs& lhs, const ERhs& rhs)
    : m_lhs(rv::cbegin(lhs))
    , m_rhs(rv::cbegin(rhs))
    , m_size(rv::size(lhs)) {
        assert(rv::size(lhs) == rv::size(rhs));
    };

public:
    sub2() noexcept                       = delete;
    sub2(sub2&&) noexcept                 = default;
    sub2(const sub2&) noexcept            = default;
    sub2& operator=(sub2&&) noexcept      = delete;
    sub2& operator=(const sub2&) noexcept = delete;
    ~sub2() noexcept                      = default;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_lhs, m_rhs);
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_lhs + m_size, m_rhs + m_size);
    }

    [[nodiscard]] auto operator[](uZ idx) const {
        return std::complex<real_type>(m_lhs[idx]) - std::complex<real_type>(m_rhs[idx]);
    };

    [[nodiscard]] constexpr auto size() const noexcept -> uZ {
        return m_size;
    }

private:
    lhs_iterator m_lhs{};
    rhs_iterator m_rhs{};
    uZ           m_size{};
};

template<typename ELhs, typename ERhs>
// requires compatible_vecexpr<ELhs, ERhs>
class mul2 final
: public rv::view_base
, public vecexpr_base {
    friend auto operator* <ELhs, ERhs>(const ELhs& lhs, const ERhs& rhs);
    using lhs_traits   = expr_traaits2<ELhs>;
    using rhs_traits   = expr_traaits2<ERhs>;
    using lhs_iterator = typename lhs_traits::const_iterator_t;
    using rhs_iterator = typename rhs_traits::const_iterator_t;

public:
    using real_type = lhs_traits::real_type;

    static constexpr auto pack_size = simd::reg<real_type>::size;

    static constexpr bool always_aligned = lhs_traits::always_aligned    //
                                           && rhs_traits::always_aligned;

    using value_type      = std::complex<real_type>;
    using difference_type = iZ;

private:
    class iterator : public ee_iter_base<iterator, lhs_iterator, rhs_iterator> {
        friend class mul2;

        iterator(const lhs_iterator& lhs, const rhs_iterator& rhs)
        : ee_iter_base<iterator, lhs_iterator, rhs_iterator>(lhs, rhs){};

    public:
        iterator() = default;
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return value_type(*this->m_lhs) * value_type(*this->m_rhs);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return value_type(*(this->m_lhs + idx)) * value_type(*(this->m_rhs + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            const auto lhs = lhs_traits::template cx_reg<pack_size>(this->m_lhs, idx);
            const auto rhs = rhs_traits::template cx_reg<pack_size>(this->m_rhs, idx);
            return simd::mul(lhs, rhs);
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return lhs_traits::aligned(this->m_lhs) && rhs_traits::aligned(this->m_rhs);
        }
    };

    mul2(const ELhs& lhs, const ERhs& rhs)
    : m_lhs(rv::cbegin(lhs))
    , m_rhs(rv::cbegin(rhs))
    , m_size(rv::size(lhs)) {
        assert(rv::size(lhs) == rv::size(rhs));
    };

public:
    mul2() noexcept                       = default;
    mul2(mul2&&) noexcept                 = default;
    mul2(const mul2&) noexcept            = default;
    mul2& operator=(mul2&&) noexcept      = delete;
    mul2& operator=(const mul2&) noexcept = delete;
    ~mul2() noexcept                      = default;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_lhs, m_rhs);
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_lhs + m_size, m_rhs + m_size);
    }

    [[nodiscard]] auto operator[](uZ idx) const {
        return std::complex<real_type>(m_lhs[idx]) * std::complex<real_type>(m_rhs[idx]);
    };

    [[nodiscard]] constexpr auto size() const noexcept -> uZ {
        return m_size;
    }

private:
    lhs_iterator m_lhs{};
    rhs_iterator m_rhs{};
    uZ           m_size{};
};

template<typename ELhs, typename ERhs>
// requires compatible_vecexpr<ELhs, ERhs>
class div2 final
: public rv::view_base
, public vecexpr_base {
    friend auto operator/ <ELhs, ERhs>(const ELhs& lhs, const ERhs& rhs);
    using lhs_traits   = expr_traaits2<ELhs>;
    using rhs_traits   = expr_traaits2<ERhs>;
    using lhs_iterator = typename lhs_traits::const_iterator_t;
    using rhs_iterator = typename rhs_traits::const_iterator_t;

public:
    using real_type = lhs_traits::real_type;

    static constexpr auto pack_size = simd::reg<real_type>::size;

    static constexpr bool always_aligned = lhs_traits::always_aligned    //
                                           && rhs_traits::always_aligned;

    using value_type      = std::complex<real_type>;
    using difference_type = iZ;

private:
    class iterator : public ee_iter_base<iterator, lhs_iterator, rhs_iterator> {
        friend class div2;

        iterator(const lhs_iterator& lhs, const rhs_iterator& rhs)
        : ee_iter_base<iterator, lhs_iterator, rhs_iterator>(lhs, rhs){};

    public:
        iterator() = default;
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return value_type(*this->m_lhs) / value_type(*this->m_rhs);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return value_type(*(this->m_lhs + idx)) / value_type(*(this->m_rhs + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            const auto lhs = lhs_traits::template cx_reg<pack_size>(this->m_lhs, idx);
            const auto rhs = rhs_traits::template cx_reg<pack_size>(this->m_rhs, idx);
            return simd::div(lhs, rhs);
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return lhs_traits::aligned(this->m_lhs) && rhs_traits::aligned(this->m_rhs);
        }
    };

    div2(const ELhs& lhs, const ERhs& rhs)
    : m_lhs(rv::cbegin(lhs))
    , m_rhs(rv::cbegin(rhs))
    , m_size(rv::size(lhs)) {
        assert(rv::size(lhs) == rv::size(rhs));
    };

public:
    div2() noexcept                       = delete;
    div2(div2&&) noexcept                 = default;
    div2(const div2&) noexcept            = default;
    div2& operator=(div2&&) noexcept      = delete;
    div2& operator=(const div2&) noexcept = delete;
    ~div2() noexcept                      = default;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_lhs, m_rhs);
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_lhs + m_size, m_rhs + m_size);
    }

    [[nodiscard]] auto operator[](uZ idx) const {
        return std::complex<real_type>(m_lhs[idx]) / std::complex<real_type>(m_rhs[idx]);
    };

    [[nodiscard]] constexpr auto size() const noexcept -> uZ {
        return m_size;
    }

private:
    lhs_iterator m_lhs;
    rhs_iterator m_rhs;
    uZ           m_size;
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class add_scalar2 final
: public vecexpr_base
, rv::view_base {
    friend auto operator+ <E, S>(const E&, S);
    friend auto operator+ <E, S>(S, const E&);
    friend auto operator- <E, S>(const E&, S);

    using expr_traits   = expr_traaits2<E>;
    using expr_iterator = typename expr_traits::const_iterator_t;

public:
    using real_type = expr_traits::real_type;

    static constexpr auto pack_size = simd::reg<real_type>::size;

    static constexpr bool always_aligned = expr_traits::always_aligned;

    using value_type      = std::complex<real_type>;
    using difference_type = iZ;

private:
    class iterator : public e_iter_base<iterator, expr_iterator> {
        friend class add_scalar2;

        using base = e_iter_base<iterator, expr_iterator>;
        S m_scalar;

        iterator(S scalar, expr_iterator iter)
        : base(iter)
        , m_scalar(scalar){};

    public:
        iterator() = default;
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return m_scalar + value_type(*base::m_iter);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return m_scalar + value_type(*(base::m_iter + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            auto scalar = simd::broadcast(m_scalar);
            auto vector = expr_traits::template cx_reg<pack_size>(base::m_iter, idx);
            if constexpr (pack_size < simd::reg<real_type>::size) {
                auto c_vector = simd::apply_conj(vector);
                return simd::add(scalar, c_vector);
            } else {
                return simd::add(scalar, vector);
            }
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return expr_traits::aligned(base::m_iter);
        }
    };

    add_scalar2(S scalar, const E& expr)
    : m_scalar(scalar)
    , m_iter(rv::cbegin(expr))
    , m_size(rv::size(expr)){};

public:
    add_scalar2() noexcept                              = delete;
    add_scalar2(add_scalar2&&) noexcept                 = default;
    add_scalar2(const add_scalar2&) noexcept            = default;
    add_scalar2& operator=(add_scalar2&&) noexcept      = delete;
    add_scalar2& operator=(const add_scalar2&) noexcept = delete;
    ~add_scalar2() noexcept                             = default;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_scalar, m_iter);
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_scalar, m_iter + m_size);
    }

    [[nodiscard]] auto operator[](uZ idx) const {
        return m_scalar + std::complex<real_type>(m_iter[idx]);
    }
    [[nodiscard]] constexpr auto size() const noexcept -> uZ {
        return m_size;
    }

private:
    S             m_scalar;
    expr_iterator m_iter;
    uZ            m_size;
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class sub_scalar2 final
: public vecexpr_base
, rv::view_base {
    friend auto operator- <E, S>(S, const E&);

    using expr_traits   = expr_traaits2<E>;
    using expr_iterator = typename expr_traits::const_iterator_t;

public:
    using real_type = expr_traits::real_type;

    static constexpr auto pack_size = simd::reg<real_type>::size;

    static constexpr bool always_aligned = expr_traits::always_aligned;

    using value_type      = std::complex<real_type>;
    using difference_type = iZ;

private:
    class iterator : public e_iter_base<iterator, expr_iterator> {
        friend class sub_scalar2;

        using base = e_iter_base<iterator, expr_iterator>;
        S m_scalar;

        iterator(S scalar, expr_iterator iter)
        : base(iter)
        , m_scalar(scalar){};

    public:
        iterator() = default;
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return m_scalar - value_type(*base::m_iter);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return m_scalar - value_type(*(base::m_iter + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            auto scalar = simd::broadcast(m_scalar);
            auto vector = expr_traits::template cx_reg<pack_size>(base::m_iter, idx);
            if constexpr (pack_size < simd::reg<real_type>::size) {
                auto c_vector = simd::apply_conj(vector);
                return simd::sub(scalar, c_vector);
            } else {
                return simd::sub(scalar, vector);
            }
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return expr_traits::aligned(base::m_iter);
        }
    };

    sub_scalar2(S scalar, const E& expr)
    : m_scalar(scalar)
    , m_iter(rv::cbegin(expr))
    , m_size(rv::size(expr)){};

public:
    sub_scalar2() noexcept                              = delete;
    sub_scalar2(sub_scalar2&&) noexcept                 = default;
    sub_scalar2(const sub_scalar2&) noexcept            = default;
    sub_scalar2& operator=(sub_scalar2&&) noexcept      = delete;
    sub_scalar2& operator=(const sub_scalar2&) noexcept = delete;
    ~sub_scalar2() noexcept                             = default;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_scalar, m_iter);
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_scalar, m_iter + m_size);
    }

    [[nodiscard]] auto operator[](uZ idx) const {
        return m_scalar - std::complex<real_type>(m_iter[idx]);
    }
    [[nodiscard]] constexpr auto size() const noexcept -> uZ {
        return m_size;
    }

private:
    S             m_scalar;
    expr_iterator m_iter;
    uZ            m_size;
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class mul_scalar2 final
: public vecexpr_base
, rv::view_base {
    friend auto operator* <E, S>(const E&, S);
    friend auto operator* <E, S>(S, const E&);
    friend auto operator/ <E, S>(const E&, S);

    using expr_traits   = expr_traaits2<E>;
    using expr_iterator = typename expr_traits::const_iterator_t;

public:
    using real_type = expr_traits::real_type;

    static constexpr auto pack_size = std::same_as<S, std::complex<real_type>>
                                          ? simd::reg<real_type>::size
                                          : std::min(E::pack_size, simd::reg<real_type>::size);

    static constexpr bool always_aligned = expr_traits::always_aligned;

    using value_type      = std::complex<real_type>;
    using difference_type = iZ;

private:
    class iterator : public e_iter_base<iterator, expr_iterator> {
        friend class mul_scalar2;

        using base = e_iter_base<iterator, expr_iterator>;
        S m_scalar;

        iterator(S scalar, expr_iterator iter)
        : base(iter)
        , m_scalar(scalar){};

    public:
        iterator() = default;
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return m_scalar * value_type(*base::m_iter);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return m_scalar * value_type(*(m_iter + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            auto scalar = simd::broadcast(m_scalar);
            auto vector = expr_traits::template cx_reg<pack_size>(base::m_iter, idx);

            return simd::mul(scalar, vector);
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return expr_traits::aligned(base::m_iter);
        }
    };

    mul_scalar2(S scalar, const E& expr)
    : m_scalar(scalar)
    , m_iter(rv::cbegin(expr))
    , m_size(rv::size(expr)){};

public:
    mul_scalar2() noexcept                              = delete;
    mul_scalar2(mul_scalar2&&) noexcept                 = default;
    mul_scalar2(const mul_scalar2&) noexcept            = default;
    mul_scalar2& operator=(mul_scalar2&&) noexcept      = delete;
    mul_scalar2& operator=(const mul_scalar2&) noexcept = delete;
    ~mul_scalar2() noexcept                             = default;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_scalar, m_iter);
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_scalar, m_iter + m_size);
    }

    [[nodiscard]] auto operator[](uZ idx) const {
        return m_scalar * value_type(m_iter[idx]);
    }
    [[nodiscard]] constexpr auto size() const noexcept -> uZ {
        return m_size;
    }

private:
    S             m_scalar;
    expr_iterator m_iter;
    uZ            m_size;
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class div_scalar2 final
: public vecexpr_base
, rv::view_base {
    friend auto operator/ <E, S>(S, const E&);

    using expr_traits   = expr_traaits2<E>;
    using expr_iterator = typename expr_traits::const_iterator_t;

public:
    using real_type = expr_traits::real_type;

    static constexpr auto pack_size =  simd::reg<real_type>::size;

    static constexpr bool always_aligned = expr_traits::always_aligned;

    using value_type      = std::complex<real_type>;
    using difference_type = iZ;

private:
    class iterator : public e_iter_base<iterator, expr_iterator> {
        friend class div_scalar2;

        using base = e_iter_base<iterator, expr_iterator>;
        S m_scalar;

        iterator(S scalar, expr_iterator iter)
        : base(iter)
        , m_scalar(scalar){};

    public:
        iterator() = default;
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return m_scalar / value_type(*base::m_iter);
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return m_scalar / value_type(*(base::m_iter + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            auto scalar = simd::broadcast(m_scalar);
            auto vector = expr_traits::template cx_reg<pack_size>(base::m_iter, idx);

            return simd::div(scalar, vector);
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return expr_traits::aligned(base::m_iter);
        }
    };

    div_scalar2(S scalar, const E& expr)
    : m_scalar(scalar)
    , m_iter(rv::cbegin(expr))
    , m_size(rv::size(expr)){};

public:
    div_scalar2() noexcept                              = delete;
    div_scalar2(div_scalar2&&) noexcept                 = default;
    div_scalar2(const div_scalar2&) noexcept            = default;
    div_scalar2& operator=(div_scalar2&&) noexcept      = delete;
    div_scalar2& operator=(const div_scalar2&) noexcept = delete;
    ~div_scalar2() noexcept                             = default;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_scalar, m_iter);
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_scalar, m_iter + m_size);
    }

    [[nodiscard]] auto operator[](uZ idx) const {
        return m_scalar / std::complex<real_type>(m_iter[idx]);
    }
    [[nodiscard]] constexpr auto size() const noexcept -> uZ {
        return m_size;
    }

private:
    S             m_scalar;
    expr_iterator m_iter;
    uZ            m_size;
};

template<typename E>
    requires vecexpr<E>
class conjugate final
: public vecexpr_base
, rv::view_base {
    friend auto pcx::conj(const E&);

    using expr_traits   = expr_traaits2<E>;
    using expr_iterator = typename expr_traits::const_iterator_t;

public:
    using real_type = expr_traits::real_type;

    static constexpr auto pack_size = expr_traits::pack_size;

    static constexpr bool always_aligned = expr_traits::always_aligned;

    using value_type      = std::complex<real_type>;
    using difference_type = iZ;

private:
    class iterator : public e_iter_base<iterator, expr_iterator> {
        friend class conjugate;

        using base = e_iter_base<iterator, expr_iterator>;

        explicit iterator(expr_iterator iter)
        : base(iter){};

    public:
        iterator() = default;
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator*() const -> value_type {
            return std::conj(value_type(*base::m_iter));
        }
        // NOLINTNEXTLINE (*const*)
        [[nodiscard]] auto operator[](difference_type idx) const -> value_type {
            return std::conj(value_type(*(base::m_iter + idx)));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            auto vector = expr_traits::template cx_reg<pack_size>(base::m_iter, idx);
            return simd::conj(vector);
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return expr_traits::aligned(base::m_iter);
        }
    };

    explicit conjugate(const E& expr)
    : m_iter(rv::cbegin(expr))
    , m_size(rv::size(expr)){};

public:
    conjugate() noexcept                            = delete;
    conjugate(conjugate&&) noexcept                 = default;
    conjugate(const conjugate&) noexcept            = default;
    conjugate& operator=(conjugate&&) noexcept      = delete;
    conjugate& operator=(const conjugate&) noexcept = delete;
    ~conjugate() noexcept                           = default;

    [[nodiscard]] auto begin() const noexcept -> iterator {
        return iterator(m_iter);
    }
    [[nodiscard]] auto end() const noexcept -> iterator {
        return iterator(m_iter + m_size);
    }

    [[nodiscard]] auto operator[](uZ idx) const {
        return std::conj(std::complex<real_type>(m_iter[idx]));
    }

    [[nodiscard]] constexpr auto size() const noexcept -> uZ {
        return m_size;
    }

private:
    expr_iterator m_iter;
    uZ            m_size;
};

}    // namespace detail_

// #region operator definitions

template<typename E1, typename E2>
    requires detail_::compatible_vecexpr<E1, E2>
inline auto operator+(const E1& lhs, const E2& rhs) {
    return detail_::add2(lhs, rhs);
};
template<typename E1, typename E2>
    requires detail_::compatible_vecexpr<E1, E2>
inline auto operator-(const E1& lhs, const E2& rhs) {
    return detail_::sub2(lhs, rhs);
};
template<typename E1, typename E2>
    requires detail_::compatible_vecexpr<E1, E2>
inline auto operator*(const E1& lhs, const E2& rhs) {
    return detail_::mul2(lhs, rhs);
};
template<typename E1, typename E2>
    requires detail_::compatible_vecexpr<E1, E2>
inline auto operator/(const E1& lhs, const E2& rhs) {
    return detail_::div2(lhs, rhs);
};

template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator+(const E& vector, S scalar) {
    return detail_::add_scalar2(scalar, vector);
}
template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator+(S scalar, const E& vector) {
    return detail_::add_scalar2(scalar, vector);
}

template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator-(const E& vector, S scalar) {
    return detail_::add_scalar2(-scalar, vector);
}
template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator-(S scalar, const E& vector) {
    return detail_::sub_scalar2(scalar, vector);
}

template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator*(const E& vector, S scalar) {
    return detail_::mul_scalar2(scalar, vector);
}
template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator*(S scalar, const E& vector) {
    return detail_::mul_scalar2(scalar, vector);
}

template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator/(const E& vector, S scalar) {
    return detail_::mul_scalar2(S(1) / scalar, vector);
}
template<typename E, typename S>
    requires detail_::compatible_scalar<E, S>
inline auto operator/(S scalar, const E& vector) {
    return detail_::div_scalar2(scalar, vector);
}

template<typename E>
    requires detail_::vecexpr<E>
auto conj(const E& vector) {
    return detail_::conjugate<E>(vector);
}

// #endregion operator definitions
}    // namespace pcx
#endif
