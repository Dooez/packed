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
struct expr_traits {
    using real_type                                 = decltype([] {});
    static constexpr uZ   pack_size                 = 0;
    static constexpr bool enable_vector_expressions = false;
};

template<typename R>
    requires complex_vector<R>
struct expr_traits<R> : cx_vector_traits<R> {
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
struct expr_traits<R> {
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
concept vecexpr = expr_traits<T>::enable_vector_expressions;
template<typename T, typename U>
concept compatible_vecexpr = vecexpr<T>                                             //
                             && vecexpr<U>                                          //
                             && std::same_as<typename expr_traits<T>::real_type,    //
                                             typename expr_traits<U>::real_type>;

template<typename Expression, typename Scalar>
concept compatible_scalar =
    vecexpr<Expression> && (std::same_as<typename expr_traits<Expression>::real_type, Scalar> ||
                            std::same_as<std::complex<typename expr_traits<Expression>::real_type>, Scalar>);

/**
 *  @brief Base class for binary vector expression CRTP injection.
 */
template<typename Impl, typename LhsExpr, typename RhsExpr>
class bi_expr_base
: public rv::view_base
, public vecexpr_base {
    uZ m_size;

protected:
    explicit bi_expr_base(uZ size)
    : m_size(size){};

    using lhs_traits = expr_traits<LhsExpr>;
    using rhs_traits = expr_traits<RhsExpr>;

public:
    using real_type = lhs_traits::real_type;

    static constexpr bool always_aligned = lhs_traits::always_aligned    //
                                           && rhs_traits::always_aligned;

    using difference_type = iZ;

    bi_expr_base() noexcept                               = delete;
    bi_expr_base(bi_expr_base&&) noexcept                 = default;
    bi_expr_base(const bi_expr_base&) noexcept            = default;
    bi_expr_base& operator=(bi_expr_base&&) noexcept      = delete;
    bi_expr_base& operator=(const bi_expr_base&) noexcept = delete;
    ~bi_expr_base() noexcept                              = default;

    [[nodiscard]] auto begin() const noexcept {
        auto* impl_    = static_cast<const Impl*>(this);
        using iterator = Impl::iterator;
        return iterator(impl_->m_lhs, impl_->m_rhs);
    }
    [[nodiscard]] auto end() const noexcept {
        auto* impl_    = static_cast<const Impl*>(this);
        using iterator = Impl::iterator;
        return iterator(impl_->m_lhs + m_size, impl_->m_rhs + m_size);
    }

    [[nodiscard]] constexpr auto size() const noexcept -> uZ {
        return m_size;
    }
};
/**
 *  @brief Base class for vector expression with scalar CRTP injection.
 */
template<typename Impl, typename Expr>
class sca_expr_base
: public rv::view_base
, public vecexpr_base {
    uZ m_size;

protected:
    using expr_traits   = expr_traits<Expr>;
    using expr_iterator = typename expr_traits::const_iterator_t;
    explicit sca_expr_base(uZ size)
    : m_size(size){};

public:
    using real_type = expr_traits::real_type;

    static constexpr bool always_aligned = expr_traits::always_aligned;

    using difference_type = iZ;

    sca_expr_base() noexcept                                = delete;
    sca_expr_base(sca_expr_base&&) noexcept                 = default;
    sca_expr_base(const sca_expr_base&) noexcept            = default;
    sca_expr_base& operator=(sca_expr_base&&) noexcept      = delete;
    sca_expr_base& operator=(const sca_expr_base&) noexcept = delete;
    ~sca_expr_base() noexcept                               = default;

    [[nodiscard]] auto begin() const noexcept {
        auto* impl_    = static_cast<const Impl*>(this);
        using iterator = Impl::iterator;
        return iterator(impl_->m_scalar, impl_->m_iter);
    }
    [[nodiscard]] auto end() const noexcept {
        auto* impl_    = static_cast<const Impl*>(this);
        using iterator = Impl::iterator;
        return iterator(impl_->m_scalar, impl_->m_iter);
    }

    [[nodiscard]] constexpr auto size() const noexcept -> uZ {
        return m_size;
    }
};
/**
 *  @brief Base class for unary vector expression CRTP injection.
 */
template<typename Impl, typename Expr>
class un_expr_base
: public rv::view_base
, public vecexpr_base {
    uZ m_size;

protected:
    using expr_traits   = expr_traits<Expr>;
    using expr_iterator = typename expr_traits::const_iterator_t;
    explicit un_expr_base(uZ size)
    : m_size(size){};

public:
    using real_type = expr_traits::real_type;

    static constexpr bool always_aligned = expr_traits::always_aligned;

    using difference_type = iZ;

    un_expr_base() noexcept                               = delete;
    un_expr_base(un_expr_base&&) noexcept                 = default;
    un_expr_base(const un_expr_base&) noexcept            = default;
    un_expr_base& operator=(un_expr_base&&) noexcept      = delete;
    un_expr_base& operator=(const un_expr_base&) noexcept = delete;
    ~un_expr_base() noexcept                              = default;

    [[nodiscard]] auto begin() const noexcept {
        auto* impl_    = static_cast<const Impl*>(this);
        using iterator = Impl::iterator;
        return iterator(impl_->m_iter);
    }
    [[nodiscard]] auto end() const noexcept {
        auto* impl_    = static_cast<const Impl*>(this);
        using iterator = Impl::iterator;
        return iterator(impl_->m_iter + m_size);
    }

    [[nodiscard]] constexpr auto size() const noexcept -> uZ {
        return m_size;
    }
};
/**
 * @brief Base class for binary vector expression iterator CRTP injection.
 */
template<typename Impl, typename LhsIter, typename RhsIter>
class bi_iter_base {
protected:
    LhsIter m_lhs;    // NOLINT(*non-private*)
    RhsIter m_rhs;    // NOLINT(*non-private*)

    bi_iter_base(const LhsIter& lhs, const RhsIter& rhs)
    : m_lhs(lhs)
    , m_rhs(rhs){};

public:
    using difference_type  = iZ;
    using iterator_concept = std::random_access_iterator_tag;

    bi_iter_base()                               = default;
    bi_iter_base(bi_iter_base&&)                 = default;
    bi_iter_base(const bi_iter_base&)            = default;
    bi_iter_base& operator=(bi_iter_base&&)      = default;
    bi_iter_base& operator=(const bi_iter_base&) = default;
    ~bi_iter_base()                              = default;

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
/**
 * @brief Base class for unary vector expression iterator CRTP injection.
 */
template<typename Impl, typename Iter>
class un_iter_base {
protected:
    Iter m_iter;    // NOLINT(*non-private*)

    explicit un_iter_base(const Iter& iter)
    : m_iter(iter){};

public:
    using difference_type  = iZ;
    using iterator_concept = std::random_access_iterator_tag;

    un_iter_base()                               = default;
    un_iter_base(un_iter_base&&)                 = default;
    un_iter_base(const un_iter_base&)            = default;
    un_iter_base& operator=(un_iter_base&&)      = default;
    un_iter_base& operator=(const un_iter_base&) = default;
    ~un_iter_base()                              = default;

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
}    // namespace pcx

// #region operator forward declarations

template<typename E1, typename E2>
    requires pcx::detail_::compatible_vecexpr<E1, E2>
auto operator+(const E1& lhs, const E2& rhs);
template<typename E1, typename E2>
    requires pcx::detail_::compatible_vecexpr<E1, E2>
auto operator-(const E1& lhs, const E2& rhs);
template<typename E1, typename E2>
    requires pcx::detail_::compatible_vecexpr<E1, E2>
auto operator*(const E1& lhs, const E2& rhs);
template<typename E1, typename E2>
    requires pcx::detail_::compatible_vecexpr<E1, E2>
auto operator/(const E1& lhs, const E2& rhs);

template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
auto operator+(const E& vector, S scalar);
template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
auto operator+(S scalar, const E& vector);

template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
auto operator-(const E& vector, S scalar);
template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
auto operator-(S scalar, const E& vector);

template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
auto operator*(const E& vector, S scalar);
template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
auto operator*(S scalar, const E& vector);

template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
auto operator/(const E& vector, S scalar);
template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
auto operator/(S scalar, const E& vector);

template<typename E>
    requires pcx::detail_::vecexpr<E>
auto conj(const E& vector);

// #endregion operator forward declarations
namespace pcx {
namespace detail_ {

template<typename ELhs, typename ERhs>
    requires compatible_vecexpr<ELhs, ERhs>
class add final : public bi_expr_base<add<ELhs, ERhs>, ELhs, ERhs> {
    friend class bi_expr_base<add, ELhs, ERhs>;
    friend auto operator+ <ELhs, ERhs>(const ELhs& lhs, const ERhs& rhs);

    using lhs_traits   = expr_traits<ELhs>;
    using rhs_traits   = expr_traits<ERhs>;
    using lhs_iterator = typename lhs_traits::const_iterator_t;
    using rhs_iterator = typename rhs_traits::const_iterator_t;

public:
    static constexpr auto pack_size = std::min(std::max(lhs_traits::pack_size,    //
                                                        rhs_traits::pack_size),
                                               simd::reg<typename lhs_traits::real_type>::size);

    using value_type = std::complex<typename lhs_traits::real_type>;

private:
    class iterator : public bi_iter_base<iterator, lhs_iterator, rhs_iterator> {
        friend class bi_expr_base<add, ELhs, ERhs>;

        iterator(const lhs_iterator& lhs, const rhs_iterator& rhs)
        : bi_iter_base<iterator, lhs_iterator, rhs_iterator>(lhs, rhs){};

    public:
        iterator() = default;
        [[nodiscard]] auto operator*() const -> value_type {
            return value_type(*this->m_lhs) + value_type(*this->m_rhs);
        }
        [[nodiscard]] auto operator[](iZ idx) const -> value_type {
            return value_type(*(this->m_lhs + idx)) + value_type(*(this->m_rhs + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            const auto lhs = lhs_traits::template cx_reg<pack_size>(this->m_lhs, idx);
            const auto rhs = rhs_traits::template cx_reg<pack_size>(this->m_rhs, idx);
            if constexpr (pack_size < simd::reg<typename lhs_traits::real_type>::size) {
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

    add(const ELhs& lhs, const ERhs& rhs)
    : bi_expr_base<add, ELhs, ERhs>(rv::size(lhs))
    , m_lhs(rv::cbegin(lhs))
    , m_rhs(rv::cbegin(rhs)) {
        assert(rv::size(lhs) == rv::size(rhs));
    };

public:
    [[nodiscard]] auto operator[](uZ idx) const {
        return value_type(m_lhs[idx]) + value_type(m_rhs[idx]);
    };

private:
    lhs_iterator m_lhs{};
    rhs_iterator m_rhs{};
};

template<typename ELhs, typename ERhs>
    requires compatible_vecexpr<ELhs, ERhs>
class sub final : public bi_expr_base<sub<ELhs, ERhs>, ELhs, ERhs> {
    friend class bi_expr_base<sub, ELhs, ERhs>;
    friend auto operator- <ELhs, ERhs>(const ELhs& lhs, const ERhs& rhs);

    using lhs_traits   = expr_traits<ELhs>;
    using rhs_traits   = expr_traits<ERhs>;
    using lhs_iterator = typename lhs_traits::const_iterator_t;
    using rhs_iterator = typename rhs_traits::const_iterator_t;

public:
    static constexpr auto pack_size = std::min(std::max(lhs_traits::pack_size,    //
                                                        rhs_traits::pack_size),
                                               simd::reg<typename lhs_traits::real_type>::size);

    using value_type = std::complex<typename lhs_traits::real_type>;

private:
    class iterator : public bi_iter_base<iterator, lhs_iterator, rhs_iterator> {
        friend class bi_expr_base<sub, ELhs, ERhs>;

        iterator(const lhs_iterator& lhs, const rhs_iterator& rhs)
        : bi_iter_base<iterator, lhs_iterator, rhs_iterator>(lhs, rhs){};

    public:
        iterator() = default;
        [[nodiscard]] auto operator*() const -> value_type {
            return value_type(*this->m_lhs) - value_type(*this->m_rhs);
        }
        [[nodiscard]] auto operator[](iZ idx) const -> value_type {
            return value_type(*(this->m_lhs + idx)) - value_type(*(this->m_rhs + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            const auto lhs = lhs_traits::template cx_reg<pack_size>(this->m_lhs, idx);
            const auto rhs = rhs_traits::template cx_reg<pack_size>(this->m_rhs, idx);
            if constexpr (pack_size < simd::reg<typename lhs_traits::real_type>::size) {
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

    sub(const ELhs& lhs, const ERhs& rhs)
    : bi_expr_base<sub, ELhs, ERhs>(rv::size(lhs))
    , m_lhs(rv::cbegin(lhs))
    , m_rhs(rv::cbegin(rhs)) {
        assert(rv::size(lhs) == rv::size(rhs));
    };

public:
    [[nodiscard]] auto operator[](uZ idx) const {
        return value_type(m_lhs[idx]) - value_type(m_rhs[idx]);
    };

private:
    lhs_iterator m_lhs{};
    rhs_iterator m_rhs{};
};

template<typename ELhs, typename ERhs>
    requires compatible_vecexpr<ELhs, ERhs>
class mul final : public bi_expr_base<mul<ELhs, ERhs>, ELhs, ERhs> {
    friend class bi_expr_base<mul, ELhs, ERhs>;
    friend auto operator* <ELhs, ERhs>(const ELhs& lhs, const ERhs& rhs);

    using lhs_traits   = expr_traits<ELhs>;
    using rhs_traits   = expr_traits<ERhs>;
    using lhs_iterator = typename lhs_traits::const_iterator_t;
    using rhs_iterator = typename rhs_traits::const_iterator_t;

public:
    static constexpr auto pack_size = simd::reg<typename lhs_traits::real_type>::size;

    using value_type = std::complex<typename lhs_traits::real_type>;

private:
    class iterator : public bi_iter_base<iterator, lhs_iterator, rhs_iterator> {
        friend class bi_expr_base<mul, ELhs, ERhs>;

        iterator(const lhs_iterator& lhs, const rhs_iterator& rhs)
        : bi_iter_base<iterator, lhs_iterator, rhs_iterator>(lhs, rhs){};

    public:
        iterator() = default;
        [[nodiscard]] auto operator*() const -> value_type {
            return value_type(*this->m_lhs) * value_type(*this->m_rhs);
        }
        [[nodiscard]] auto operator[](iZ idx) const -> value_type {
            return value_type(*(this->m_lhs + idx)) * value_type(*(this->m_rhs + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            const auto lhs = lhs_traits::template cx_reg<pack_size>(this->m_lhs, idx);
            const auto rhs = rhs_traits::template cx_reg<pack_size>(this->m_rhs, idx);
            if constexpr (pack_size < simd::reg<typename lhs_traits::real_type>::size) {
                auto c_lhs = simd::apply_conj(lhs);
                auto c_rhs = simd::apply_conj(rhs);
                return simd::mul(c_lhs, c_rhs);
            } else {
                return simd::mul(lhs, rhs);
            }
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return lhs_traits::aligned(this->m_lhs) && rhs_traits::aligned(this->m_rhs);
        }
    };

    mul(const ELhs& lhs, const ERhs& rhs)
    : bi_expr_base<mul, ELhs, ERhs>(rv::size(lhs))
    , m_lhs(rv::cbegin(lhs))
    , m_rhs(rv::cbegin(rhs)) {
        assert(rv::size(lhs) == rv::size(rhs));
    };

public:
    [[nodiscard]] auto operator[](uZ idx) const {
        return value_type(m_lhs[idx]) * value_type(m_rhs[idx]);
    };

private:
    lhs_iterator m_lhs{};
    rhs_iterator m_rhs{};
};

template<typename ELhs, typename ERhs>
    requires compatible_vecexpr<ELhs, ERhs>
class div final : public bi_expr_base<div<ELhs, ERhs>, ELhs, ERhs> {
    friend class bi_expr_base<div, ELhs, ERhs>;
    friend auto operator/ <ELhs, ERhs>(const ELhs& lhs, const ERhs& rhs);

    using lhs_traits   = expr_traits<ELhs>;
    using rhs_traits   = expr_traits<ERhs>;
    using lhs_iterator = typename lhs_traits::const_iterator_t;
    using rhs_iterator = typename rhs_traits::const_iterator_t;

public:
    static constexpr auto pack_size = simd::reg<typename lhs_traits::real_type>::size;

    using value_type = std::complex<typename lhs_traits::real_type>;

private:
    class iterator : public bi_iter_base<iterator, lhs_iterator, rhs_iterator> {
        friend class bi_expr_base<div, ELhs, ERhs>;

        iterator(const lhs_iterator& lhs, const rhs_iterator& rhs)
        : bi_iter_base<iterator, lhs_iterator, rhs_iterator>(lhs, rhs){};

    public:
        iterator() = default;
        [[nodiscard]] auto operator*() const -> value_type {
            return value_type(*this->m_lhs) / value_type(*this->m_rhs);
        }
        [[nodiscard]] auto operator[](iZ idx) const -> value_type {
            return value_type(*(this->m_lhs + idx)) / value_type(*(this->m_rhs + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            const auto lhs = lhs_traits::template cx_reg<pack_size>(this->m_lhs, idx);
            const auto rhs = rhs_traits::template cx_reg<pack_size>(this->m_rhs, idx);
            if constexpr (pack_size < simd::reg<typename lhs_traits::real_type>::size) {
                auto c_lhs = simd::apply_conj(lhs);
                auto c_rhs = simd::apply_conj(rhs);
                return simd::div(c_lhs, c_rhs);
            } else {
                return simd::div(lhs, rhs);
            }
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return lhs_traits::aligned(this->m_lhs) && rhs_traits::aligned(this->m_rhs);
        }
    };

    div(const ELhs& lhs, const ERhs& rhs)
    : bi_expr_base<div, ELhs, ERhs>(rv::size(lhs))
    , m_lhs(rv::cbegin(lhs))
    , m_rhs(rv::cbegin(rhs)) {
        assert(rv::size(lhs) == rv::size(rhs));
    };

public:
    [[nodiscard]] auto operator[](uZ idx) const {
        return value_type(m_lhs[idx]) / value_type(m_rhs[idx]);
    };

private:
    lhs_iterator m_lhs{};
    rhs_iterator m_rhs{};
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class scalar_add final : public sca_expr_base<scalar_add<E, S>, E> {
    friend class sca_expr_base<scalar_add, E>;

    friend auto operator+ <E, S>(const E&, S);
    friend auto operator+ <E, S>(S, const E&);
    friend auto operator- <E, S>(const E&, S);

    using expr_traits   = expr_traits<E>;
    using expr_iterator = typename expr_traits::const_iterator_t;

public:
    static constexpr auto pack_size = simd::reg<typename expr_traits::real_type>::size;
    using value_type                = std::complex<typename expr_traits::real_type>;

private:
    class iterator : public un_iter_base<iterator, expr_iterator> {
        friend class sca_expr_base<scalar_add, E>;
        S m_scalar;

        iterator(S scalar, expr_iterator iter)
        : un_iter_base<iterator, expr_iterator>(iter)
        , m_scalar(scalar){};

    public:
        iterator() = default;
        [[nodiscard]] auto operator*() const -> value_type {
            return m_scalar + value_type(*this->m_iter);
        }
        [[nodiscard]] auto operator[](iZ idx) const -> value_type {
            return m_scalar + value_type(*(this->m_iter + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            auto scalar = simd::broadcast(m_scalar);
            auto vector = expr_traits::template cx_reg<pack_size>(this->m_iter, idx);

            return simd::add(scalar, vector);
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return expr_traits::aligned(this->m_iter);
        }
    };

    explicit scalar_add(S scalar, const E& expr)
    : sca_expr_base<scalar_add, E>(rv::size(expr))
    , m_scalar(scalar)
    , m_iter(rv::cbegin(expr)){};

public:
    [[nodiscard]] auto operator[](uZ idx) const {
        return m_scalar + value_type(m_iter[idx]);
    }

private:
    S             m_scalar;
    expr_iterator m_iter;
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class scalar_sub final : public sca_expr_base<scalar_sub<E, S>, E> {
    friend class sca_expr_base<scalar_sub, E>;

    friend auto operator- <E, S>(S, const E&);

    using expr_traits   = expr_traits<E>;
    using expr_iterator = typename expr_traits::const_iterator_t;

public:
    static constexpr auto pack_size = simd::reg<typename expr_traits::real_type>::size;
    using value_type                = std::complex<typename expr_traits::real_type>;

private:
    class iterator : public un_iter_base<iterator, expr_iterator> {
        friend class sca_expr_base<scalar_sub, E>;
        S m_scalar;

        iterator(S scalar, expr_iterator iter)
        : un_iter_base<iterator, expr_iterator>(iter)
        , m_scalar(scalar){};

    public:
        iterator() = default;
        [[nodiscard]] auto operator*() const -> value_type {
            return m_scalar - value_type(*this->m_iter);
        }
        [[nodiscard]] auto operator[](iZ idx) const -> value_type {
            return m_scalar - value_type(*(this->m_iter + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            auto scalar = simd::broadcast(m_scalar);
            auto vector = expr_traits::template cx_reg<pack_size>(this->m_iter, idx);

            return simd::sub(scalar, vector);
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return expr_traits::aligned(this->m_iter);
        }
    };

    explicit scalar_sub(S scalar, const E& expr)
    : sca_expr_base<scalar_sub, E>(rv::size(expr))
    , m_scalar(scalar)
    , m_iter(rv::cbegin(expr)){};

public:
    [[nodiscard]] auto operator[](uZ idx) const {
        return m_scalar - value_type(m_iter[idx]);
    }

private:
    S             m_scalar;
    expr_iterator m_iter;
};
template<typename E, typename S>
    requires compatible_scalar<E, S>
class scalar_mul final : public sca_expr_base<scalar_mul<E, S>, E> {
    friend class sca_expr_base<scalar_mul, E>;

    friend auto operator* <E, S>(const E&, S);
    friend auto operator* <E, S>(S, const E&);
    friend auto operator/ <E, S>(const E&, S);

    using expr_traits   = expr_traits<E>;
    using expr_iterator = typename expr_traits::const_iterator_t;

public:
    using real_type                 = expr_traits::real_type;
    static constexpr auto pack_size = std::same_as<S, std::complex<real_type>>
                                          ? simd::reg<real_type>::size
                                          : std::min(E::pack_size, simd::reg<real_type>::size);
    using value_type                = std::complex<typename expr_traits::real_type>;

private:
    class iterator : public un_iter_base<iterator, expr_iterator> {
        friend class sca_expr_base<scalar_mul, E>;
        S m_scalar;

        iterator(S scalar, expr_iterator iter)
        : un_iter_base<iterator, expr_iterator>(iter)
        , m_scalar(scalar){};

    public:
        iterator() = default;
        [[nodiscard]] auto operator*() const -> value_type {
            return m_scalar * value_type(*this->m_iter);
        }
        [[nodiscard]] auto operator[](iZ idx) const -> value_type {
            return m_scalar * value_type(*(this->m_iter + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            auto scalar = simd::broadcast(m_scalar);
            auto vector = expr_traits::template cx_reg<pack_size>(this->m_iter, idx);

            return simd::mul(scalar, vector);
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return expr_traits::aligned(this->m_iter);
        }
    };

    explicit scalar_mul(S scalar, const E& expr)
    : sca_expr_base<scalar_mul, E>(rv::size(expr))
    , m_scalar(scalar)
    , m_iter(rv::cbegin(expr)){};

public:
    [[nodiscard]] auto operator[](uZ idx) const {
        return m_scalar * value_type(m_iter[idx]);
    }

private:
    S             m_scalar;
    expr_iterator m_iter;
};

template<typename E, typename S>
    requires compatible_scalar<E, S>
class scalar_div final : public sca_expr_base<scalar_div<E, S>, E> {
    friend class sca_expr_base<scalar_div, E>;

    friend auto operator/ <E, S>(S, const E&);

    using expr_traits   = expr_traits<E>;
    using expr_iterator = typename expr_traits::const_iterator_t;

public:
    static constexpr auto pack_size = simd::reg<typename expr_traits::real_type>::size;
    using value_type                = std::complex<typename expr_traits::real_type>;

private:
    class iterator : public un_iter_base<iterator, expr_iterator> {
        friend class sca_expr_base<scalar_div, E>;
        S m_scalar;

        iterator(S scalar, expr_iterator iter)
        : un_iter_base<iterator, expr_iterator>(iter)
        , m_scalar(scalar){};

    public:
        iterator() = default;
        [[nodiscard]] auto operator*() const -> value_type {
            return m_scalar / value_type(*this->m_iter);
        }
        [[nodiscard]] auto operator[](iZ idx) const -> value_type {
            return m_scalar / value_type(*(this->m_iter + idx));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            auto scalar = simd::broadcast(m_scalar);
            auto vector = expr_traits::template cx_reg<pack_size>(this->m_iter, idx);

            return simd::div(scalar, vector);
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return expr_traits::aligned(this->m_iter);
        }
    };

    explicit scalar_div(S scalar, const E& expr)
    : sca_expr_base<scalar_div, E>(rv::size(expr))
    , m_scalar(scalar)
    , m_iter(rv::cbegin(expr)){};

public:
    [[nodiscard]] auto operator[](uZ idx) const {
        return m_scalar / value_type(m_iter[idx]);
    }

private:
    S             m_scalar;
    expr_iterator m_iter;
};

template<typename E>
    requires vecexpr<E>
class conjugate final : public un_expr_base<conjugate<E>, E> {
    friend class un_expr_base<conjugate, E>;

    friend auto ::conj(const E&);

    using expr_traits   = expr_traits<E>;
    using expr_iterator = typename expr_traits::const_iterator_t;

public:
    static constexpr auto pack_size = expr_traits::pack_size;
    using value_type                = std::complex<typename expr_traits::real_type>;

private:
    class iterator : public un_iter_base<iterator, expr_iterator> {
        friend class un_expr_base<conjugate, E>;

        explicit iterator(expr_iterator iter)
        : un_iter_base<iterator, expr_iterator>(iter){};

    public:
        iterator() = default;
        [[nodiscard]] auto operator*() const -> value_type {
            return std::conj(value_type(*(this->m_iter)));
        }
        [[nodiscard]] auto operator[](uZ idx) const -> value_type {
            return std::conj(value_type(*(this->m_iter + idx)));
        }
        [[nodiscard]] auto cx_reg(uZ idx) const {
            auto vector = expr_traits::template cx_reg<pack_size>(this->m_iter, idx);
            return simd::conj(vector);
        }
        [[nodiscard]] constexpr bool aligned() const noexcept {
            return expr_traits::aligned(this->m_iter);
        }
    };

    explicit conjugate(const E& expr)
    : un_expr_base<conjugate, E>(rv::size(expr))
    , m_iter(rv::cbegin(expr)){};

public:
    [[nodiscard]] auto operator[](uZ idx) const {
        return std::conj(value_type(m_iter[idx]));
    }

private:
    expr_iterator m_iter;
};

}    // namespace detail_
}    // namespace pcx

template<typename E1, typename E2>
    requires pcx::detail_::compatible_vecexpr<E1, E2>
inline auto operator+(const E1& lhs, const E2& rhs) {
    return pcx::detail_::add(lhs, rhs);
};
inline auto add_wrap(const auto& lhs, const auto& rhs) {
    return lhs + rhs;
}

template<typename E1, typename E2>
    requires pcx::detail_::compatible_vecexpr<E1, E2>
inline auto operator-(const E1& lhs, const E2& rhs) {
    return pcx::detail_::sub(lhs, rhs);
};
template<typename E1, typename E2>
    requires pcx::detail_::compatible_vecexpr<E1, E2>
inline auto operator*(const E1& lhs, const E2& rhs) {
    return pcx::detail_::mul(lhs, rhs);
};
template<typename E1, typename E2>
    requires pcx::detail_::compatible_vecexpr<E1, E2>
inline auto operator/(const E1& lhs, const E2& rhs) {
    return pcx::detail_::div(lhs, rhs);
};

template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
inline auto operator+(const E& vector, S scalar) {
    return pcx::detail_::scalar_add(scalar, vector);
}
template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
inline auto operator+(S scalar, const E& vector) {
    return pcx::detail_::scalar_add(scalar, vector);
}

template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
inline auto operator-(const E& vector, S scalar) {
    return pcx::detail_::scalar_add(-scalar, vector);
}
template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
inline auto operator-(S scalar, const E& vector) {
    return pcx::detail_::scalar_sub(scalar, vector);
}

template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
inline auto operator*(const E& vector, S scalar) {
    return pcx::detail_::scalar_mul(scalar, vector);
}
template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
inline auto operator*(S scalar, const E& vector) {
    return pcx::detail_::scalar_mul(scalar, vector);
}

template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
inline auto operator/(const E& vector, S scalar) {
    return pcx::detail_::scalar_mul(S(1) / scalar, vector);
}
template<typename E, typename S>
    requires pcx::detail_::compatible_scalar<E, S>
inline auto operator/(S scalar, const E& vector) {
    return pcx::detail_::scalar_div(scalar, vector);
}

template<typename E>
    requires pcx::detail_::vecexpr<E>
auto conj(const E& vector) {
    return pcx::detail_::conjugate<E>(vector);
}

#endif
