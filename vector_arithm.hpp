#ifndef VECTOR_ARITHM_HPP
#define VECTOR_ARITHM_HPP
#include "vector_util.hpp"

#include <assert.h>
#include <complex>
#include <concepts>
#include <immintrin.h>
#include <ranges>
#include <type_traits>

namespace pcx {

namespace avx {

    inline auto add(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
    {
        return _mm256_add_ps(lhs, rhs);
    }
    inline auto add(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
    {
        return _mm256_add_pd(lhs, rhs);
    }

    inline auto sub(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
    {
        return _mm256_sub_ps(lhs, rhs);
    }
    inline auto sub(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
    {
        return _mm256_sub_pd(lhs, rhs);
    }

    inline auto mul(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
    {
        return _mm256_mul_ps(lhs, rhs);
    }
    inline auto mul(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
    {
        return _mm256_mul_pd(lhs, rhs);
    }

    inline auto div(reg<float>::type lhs, reg<float>::type rhs) -> reg<float>::type
    {
        return _mm256_div_ps(lhs, rhs);
    }
    inline auto div(reg<double>::type lhs, reg<double>::type rhs) -> reg<double>::type
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
    inline auto broadcast(const std::complex<T>& source) -> cx_reg<T>
    {
        const auto& value = reinterpret_cast<const T(&)[2]>(source);
        return {avx::broadcast(&(value[0])), avx::broadcast(&(value[1]))};
    }
    template<typename T>
    inline auto broadcast(const T& source) -> typename reg<T>::type
    {
        return avx::broadcast(&source);
    }

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
            avx::sub(avx::mul(lhs.real, rhs.real), avx::mul(lhs.imag, rhs.imag));
        const auto imag =
            avx::add(avx::mul(lhs.real, rhs.imag), avx::mul(lhs.imag, rhs.real));

        return {real, imag};
    }
    template<typename T>
    inline auto div(cx_reg<T> lhs, cx_reg<T> rhs) -> cx_reg<T>
    {
        const auto rhs_abs =
            avx::add(avx::mul(rhs.real, rhs.real), avx::mul(rhs.imag, rhs.imag));

        const auto real_ =
            avx::add(avx::mul(lhs.real, rhs.real), avx::mul(lhs.imag, rhs.imag));

        const auto imag_ =
            avx::sub(avx::mul(lhs.imag, rhs.real), avx::mul(lhs.real, rhs.imag));

        return {avx::div(real_, rhs_abs), avx::div(imag_, rhs_abs)};
    }

    template<typename T>
    inline auto add(typename reg<T>::type lhs, cx_reg<T> rhs) -> cx_reg<T>
    {
        return {add(lhs, rhs.real), rhs.imag};
    }
    template<typename T>
    inline auto sub(typename reg<T>::type lhs, cx_reg<T> rhs) -> cx_reg<T>
    {
        return {sub(lhs, rhs.real), rhs.imag};
    }
    template<typename T>
    inline auto mul(typename reg<T>::type lhs, cx_reg<T> rhs) -> cx_reg<T>
    {
        return {avx::mul(lhs, rhs.real), avx::mul(lhs, rhs.imag)};
    }
    template<typename T>
    inline auto div(typename reg<T>::type lhs, cx_reg<T> rhs) -> cx_reg<T>
    {
        const auto rhs_abs =
            avx::add(avx::mul(rhs.real, rhs.real), avx::mul(rhs.imag, rhs.imag));

        const auto real_ = avx::mul(lhs, rhs.real);
        const auto imag_ = avx::mul(lhs, rhs.imag);

        return {avx::div(real_, rhs_abs), avx::div(imag_, rhs_abs)};
    }

    template<typename T>
    inline auto add(cx_reg<T> lhs, typename reg<T>::type rhs) -> cx_reg<T>
    {
        return {add(lhs.real, rhs), lhs.imag};
    }
    template<typename T>
    inline auto sub(cx_reg<T> lhs, typename reg<T>::type rhs) -> cx_reg<T>
    {
        return {sub(lhs.real, rhs), lhs.imag};
    }
    template<typename T>
    inline auto mul(cx_reg<T> lhs, typename reg<T>::type rhs) -> cx_reg<T>
    {
        return {avx::mul(lhs.real, rhs), avx::mul(lhs.imag, rhs)};
    }
    template<typename T>
    inline auto div(cx_reg<T> lhs, typename reg<T>::type rhs) -> cx_reg<T>
    {
        return {avx::div(lhs.real, rhs), avx::div(lhs.imag, rhs)};
    }

}    // namespace avx

namespace internal {

    struct expression_traits
    {
        /**
     * @brief Evaluates slice of the expression with offset;
     * Slice size is determined by avx register size; No checks are performed.
     *
     * @param expression
     * @param idx offset
     * @return auto evaluated complex register
     */
        template<typename I>
        [[nodiscard]] static constexpr auto cx_reg(const I& iterator, std::size_t idx)
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
        [[nodiscard]] static constexpr auto
        cx_reg(const iterator<T, PackSize, Const>& iterator, std::size_t idx)
            -> avx::cx_reg<T>
        {
            auto real = avx::load(&(*iterator) + idx);
            auto imag = avx::load(&(*iterator) + idx + PackSize);
            return {real, imag};
        }

        template<typename I>
        [[nodiscard]] static constexpr auto aligned(const I& iterator, std::size_t idx)
        {
            return iterator.aligned(idx);
        }
    };

    template<typename E>
    concept vector_expression =
        requires(E expression, std::size_t idx) {
            requires std::ranges::view<E>;

            requires std::ranges::random_access_range<E>;

            requires std::ranges::sized_range<E>;

            typename E::real_type;

            requires std::convertible_to<std::iter_value_t<decltype(expression.begin())>,
                                         std::complex<typename E::real_type>>;

            {
                expression_traits::aligned(expression.begin(), idx)
                } -> std::same_as<bool>;

            {
                expression_traits::cx_reg(expression.begin(), idx)
                } -> std::same_as<avx::cx_reg<typename E::real_type>>;
        };


    template<typename E1, typename E2>
    concept compatible_expression =
        vector_expression<E1> && vector_expression<E2> &&
        std::same_as<typename E1::real_type, typename E2::real_type>;

    template<typename Expression, typename Scalar>
    concept compatible_scalar =
        vector_expression<Expression> &&
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
    class add : public std::ranges::view_base
    {
        friend auto operator+<E1, E2>(const E1& lhs, const E2& rhs);

    public:
        using real_type = typename E1::real_type;

        class iterator
        {
            friend class add;

        private:
            using lhs_iterator = decltype(std::declval<const E1>().begin());
            using rhs_iterator = decltype(std::declval<const E2>().begin());

            iterator(lhs_iterator lhs, rhs_iterator rhs)
            : m_lhs(std::move(lhs))
            , m_rhs(std::move(rhs)){};

        public:
            using real_type        = add::real_type;
            using value_type       = const std::complex<real_type>;
            using difference_type  = std::ptrdiff_t;
            using iterator_concept = std::random_access_iterator_tag;

            iterator() = default;

            iterator(const iterator& other) noexcept = default;
            iterator(iterator&& other) noexcept      = default;

            ~iterator() = default;

            iterator& operator=(const iterator& other) noexcept = default;
            iterator& operator=(iterator&& other) noexcept      = default;

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
            [[nodiscard]] auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
            {
                const auto lhs = expression_traits::cx_reg(m_lhs, idx);
                const auto rhs = expression_traits::cx_reg(m_rhs, idx);

                return avx::add(lhs, rhs);
            }

            [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept
            {
                return expression_traits::aligned(m_lhs, offset) &&
                       expression_traits::aligned(m_rhs, offset);
            }

        private:
            lhs_iterator m_lhs;
            rhs_iterator m_rhs;
        };

    private:
        add(const E1& lhs, const E2& rhs)
        : m_lhs(lhs)
        , m_rhs(rhs)
        {
            assert(lhs.size() == rhs.size());
        };

    public:
        add() noexcept = default;

        add(add&& other) noexcept = default;
        add(const add&) noexcept  = default;

        ~add() noexcept = default;

        add& operator=(add&& other) noexcept
        {
            m_lhs = std::move(other.m_lhs);
            m_rhs = std::move(other.m_rhs);
            return *this;
        };
        add& operator=(const add&) noexcept = delete;

        [[nodiscard]] auto begin() const noexcept -> iterator
        {
            return iterator(m_lhs.begin(), m_rhs.begin());
        }
        [[nodiscard]] auto end() const noexcept -> iterator
        {
            return iterator(m_lhs.end(), m_rhs.end());
        }

        [[nodiscard]] auto operator[](std::size_t idx) const
        {
            return std::complex<real_type>(m_lhs[idx]) +
                   std::complex<real_type>(m_rhs[idx]);
        };

        [[nodiscard]] constexpr auto size() const noexcept -> std::size_t
        {
            return m_lhs.size();
        }

    private:
        const E1 m_lhs;
        const E2 m_rhs;
    };

    template<typename E1, typename E2>
        requires compatible_expression<E1, E2>
    class sub : public std::ranges::view_base
    {
        friend auto operator-<E1, E2>(const E1& lhs, const E2& rhs);

    public:
        using real_type = typename E1::real_type;

        class iterator
        {
            friend class sub;

        private:
            using lhs_iterator = decltype(std::declval<const E1>().begin());
            using rhs_iterator = decltype(std::declval<const E2>().begin());

            iterator(lhs_iterator lhs, rhs_iterator rhs)
            : m_lhs(std::move(lhs))
            , m_rhs(std::move(rhs)){};

        public:
            using real_type        = sub::real_type;
            using value_type       = const std::complex<real_type>;
            using difference_type  = std::ptrdiff_t;
            using iterator_concept = std::random_access_iterator_tag;

            iterator() = default;

            iterator(const iterator& other) noexcept = default;
            iterator(iterator&& other) noexcept      = default;

            ~iterator() = default;

            iterator& operator=(const iterator& other) noexcept = default;
            iterator& operator=(iterator&& other) noexcept      = default;

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
            [[nodiscard]] auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
            {
                const auto lhs = expression_traits::cx_reg(m_lhs, idx);
                const auto rhs = expression_traits::cx_reg(m_rhs, idx);

                return avx::sub(lhs, rhs);
            }

            [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept
            {
                return expression_traits::aligned(m_lhs, offset) &&
                       expression_traits::aligned(m_rhs, offset);
            }

        private:
            lhs_iterator m_lhs;
            rhs_iterator m_rhs;
        };

    private:
        sub(const E1& lhs, const E2& rhs)
        : m_lhs(lhs)
        , m_rhs(rhs)
        {
            assert(lhs.size() == rhs.size());
        };

    public:
        sub() noexcept = default;

        sub(sub&& other) noexcept = default;
        sub(const sub&) noexcept  = default;

        ~sub() noexcept = default;

        sub& operator=(sub&& other) noexcept
        {
            m_lhs = std::move(other.m_lhs);
            m_rhs = std::move(other.m_rhs);
            return *this;
        };
        sub& operator=(const sub&) noexcept = delete;

        [[nodiscard]] auto begin() const noexcept -> iterator
        {
            return iterator(m_lhs.begin(), m_rhs.begin());
        }
        [[nodiscard]] auto end() const noexcept -> iterator
        {
            return iterator(m_lhs.end(), m_rhs.end());
        }

        [[nodiscard]] auto operator[](std::size_t idx) const
        {
            return std::complex<real_type>(m_lhs[idx]) -
                   std::complex<real_type>(m_rhs[idx]);
        };

        [[nodiscard]] constexpr auto size() const noexcept -> std::size_t
        {
            return m_lhs.size();
        }

    private:
        const E1 m_lhs;
        const E2 m_rhs;
    };

    template<typename E1, typename E2>
        requires compatible_expression<E1, E2>
    class mul : public std::ranges::view_base
    {
        friend auto operator*<E1, E2>(const E1& lhs, const E2& rhs);

    public:
        using real_type = typename E1::real_type;

        class iterator
        {
            friend class mul;

        private:
            using lhs_iterator = decltype(std::declval<const E1>().begin());
            using rhs_iterator = decltype(std::declval<const E2>().begin());

            iterator(lhs_iterator lhs, rhs_iterator rhs)
            : m_lhs(std::move(lhs))
            , m_rhs(std::move(rhs)){};

        public:
            using real_type        = mul::real_type;
            using value_type       = const std::complex<real_type>;
            using difference_type  = std::ptrdiff_t;
            using iterator_concept = std::random_access_iterator_tag;

            iterator() = default;

            iterator(const iterator& other) noexcept = default;
            iterator(iterator&& other) noexcept      = default;

            ~iterator() = default;

            iterator& operator=(const iterator& other) noexcept = default;
            iterator& operator=(iterator&& other) noexcept      = default;

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
            [[nodiscard]] auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
            {
                const auto lhs = expression_traits::cx_reg(m_lhs, idx);
                const auto rhs = expression_traits::cx_reg(m_rhs, idx);

                return avx::mul(lhs, rhs);
            }

            [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept
            {
                return expression_traits::aligned(m_lhs, offset) &&
                       expression_traits::aligned(m_rhs, offset);
            }

        private:
            lhs_iterator m_lhs;
            rhs_iterator m_rhs;
        };

    private:
        mul(const E1& lhs, const E2& rhs)
        : m_lhs(lhs)
        , m_rhs(rhs)
        {
            assert(lhs.size() == rhs.size());
        };

    public:
        mul() noexcept = default;

        mul(mul&& other) noexcept = default;
        mul(const mul&) noexcept  = default;

        ~mul() noexcept = default;

        mul& operator=(mul&& other) noexcept
        {
            m_lhs = std::move(other.m_lhs);
            m_rhs = std::move(other.m_rhs);
            return *this;
        };
        mul& operator=(const mul&) noexcept = delete;

        [[nodiscard]] auto begin() const noexcept -> iterator
        {
            return iterator(m_lhs.begin(), m_rhs.begin());
        }
        [[nodiscard]] auto end() const noexcept -> iterator
        {
            return iterator(m_lhs.end(), m_rhs.end());
        }

        [[nodiscard]] auto operator[](std::size_t idx) const
        {
            return std::complex<real_type>(m_lhs[idx]) *
                   std::complex<real_type>(m_rhs[idx]);
        };

        [[nodiscard]] constexpr auto size() const noexcept -> std::size_t
        {
            return m_lhs.size();
        }

    private:
        const E1 m_lhs;
        const E2 m_rhs;
    };

    template<typename E1, typename E2>
        requires compatible_expression<E1, E2>
    class div : public std::ranges::view_base
    {
        friend auto operator/<E1, E2>(const E1& lhs, const E2& rhs);

    public:
        using real_type = typename E1::real_type;

        class iterator
        {
            friend class div;

        private:
            using lhs_iterator = decltype(std::declval<const E1>().begin());
            using rhs_iterator = decltype(std::declval<const E2>().begin());

            iterator(lhs_iterator lhs, rhs_iterator rhs)
            : m_lhs(std::move(lhs))
            , m_rhs(std::move(rhs)){};

        public:
            using real_type        = div::real_type;
            using value_type       = const std::complex<real_type>;
            using difference_type  = std::ptrdiff_t;
            using iterator_concept = std::random_access_iterator_tag;

            iterator() = default;

            iterator(const iterator& other) noexcept = default;
            iterator(iterator&& other) noexcept      = default;

            ~iterator() = default;

            iterator& operator=(const iterator& other) noexcept = default;
            iterator& operator=(iterator&& other) noexcept      = default;

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
            [[nodiscard]] auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
            {
                const auto lhs = expression_traits::cx_reg(m_lhs, idx);
                const auto rhs = expression_traits::cx_reg(m_rhs, idx);

                return avx::div(lhs, rhs);
            }

            [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept
            {
                return expression_traits::aligned(m_lhs, offset) &&
                       expression_traits::aligned(m_rhs, offset);
            }

        private:
            lhs_iterator m_lhs;
            rhs_iterator m_rhs;
        };

    private:
        div(const E1& lhs, const E2& rhs)
        : m_lhs(lhs)
        , m_rhs(rhs)
        {
            assert(lhs.size() == rhs.size());
        };

    public:
        div() noexcept = default;

        div(div&& other) noexcept = default;
        div(const div&) noexcept  = default;

        ~div() noexcept = default;

        div& operator=(div&& other) noexcept
        {
            m_lhs = std::move(other.m_lhs);
            m_rhs = std::move(other.m_rhs);
            return *this;
        };
        div& operator=(const div&) noexcept = delete;

        [[nodiscard]] auto begin() const noexcept -> iterator
        {
            return iterator(m_lhs.begin(), m_rhs.begin());
        }
        [[nodiscard]] auto end() const noexcept -> iterator
        {
            return iterator(m_lhs.end(), m_rhs.end());
        }

        [[nodiscard]] auto operator[](std::size_t idx) const
        {
            return std::complex<real_type>(m_lhs[idx]) /
                   std::complex<real_type>(m_rhs[idx]);
        };

        [[nodiscard]] constexpr auto size() const noexcept -> std::size_t
        {
            return m_lhs.size();
        }

    private:
        const E1 m_lhs;
        const E2 m_rhs;
    };

    template<typename E, typename S>
        requires compatible_scalar<E, S>
    class scalar_add : public std::ranges::view_base
    {
        friend auto operator+<E, S>(const E& vector, S scalar);
        friend auto operator+<E, S>(S scalar, const E& vector);
        friend auto operator-<E, S>(const E& vector, S scalar);

    public:
        using real_type = typename E::real_type;

        class iterator
        {
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

            iterator() = default;

            iterator(const iterator& other) noexcept = default;
            iterator(iterator&& other) noexcept      = default;

            ~iterator() = default;

            iterator& operator=(const iterator& other) noexcept = default;
            iterator& operator=(iterator&& other) noexcept      = default;


            [[nodiscard]] bool operator==(const iterator& other) const noexcept
            {
                return (m_vector == other.m_vector);
            }
            [[nodiscard]] auto operator<=>(const iterator& other) const noexcept
            {
                return (m_vector <=> other.m_vector);
            }

            auto operator++() noexcept -> iterator&
            {
                ++m_vector;
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
                --m_vector;
            }
            auto operator--(int) noexcept -> iterator
            {
                auto copy = *this;
                --(*this);
                return copy;
            }

            auto operator+=(difference_type n) noexcept -> iterator&
            {
                m_vector += n;
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
                return lhs.m_vector - rhs.m_vector;
            }

            // NOLINTNEXTLINE (*const*)
            [[nodiscard]] auto operator*() const -> value_type
            {
                return m_scalar + value_type(*m_vector);
            }
            // NOLINTNEXTLINE (*const*)
            [[nodiscard]] auto operator[](difference_type idx) const -> value_type
            {
                return m_scalar + value_type(*(m_vector + idx));
            }
            [[nodiscard]] auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
            {
                const auto scalar = avx::broadcast(m_scalar);
                const auto vector = expression_traits::cx_reg(m_vector, idx);

                return avx::add(scalar, vector);
            }

            [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept
            {
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

        scalar_add& operator=(scalar_add&& other) noexcept
        {
            m_scalar = std::move(other.m_scalar);
            m_vector = std::move(other.m_vector);
            return *this;
        };
        scalar_add& operator=(const scalar_add&) noexcept = delete;

        [[nodiscard]] auto begin() const noexcept -> iterator
        {
            return iterator(m_scalar, m_vector.begin());
        }
        [[nodiscard]] auto end() const noexcept -> iterator
        {
            return iterator(m_scalar, m_vector.end());
        }

        [[nodiscard]] auto operator[](std::size_t idx) const
        {
            return m_scalar + std::complex<real_type>(m_vector[idx]);
        }
        [[nodiscard]] constexpr auto size() const noexcept -> std::size_t
        {
            return m_vector.size();
        }

    private:
        const E m_vector;
        S       m_scalar;
    };

    template<typename E, typename S>
        requires compatible_scalar<E, S>
    class scalar_sub : public std::ranges::view_base
    {
        friend auto operator-<E, S>(S scalar, const E& vector);

    public:
        using real_type = typename E::real_type;

        class iterator
        {
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

            iterator() = default;

            iterator(const iterator& other) noexcept = default;
            iterator(iterator&& other) noexcept      = default;

            ~iterator() = default;

            iterator& operator=(const iterator& other) noexcept = default;
            iterator& operator=(iterator&& other) noexcept      = default;


            [[nodiscard]] bool operator==(const iterator& other) const noexcept
            {
                return (m_vector == other.m_vector);
            }
            [[nodiscard]] auto operator<=>(const iterator& other) const noexcept
            {
                return (m_vector <=> other.m_vector);
            }

            auto operator++() noexcept -> iterator&
            {
                ++m_vector;
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
                --m_vector;
            }
            auto operator--(int) noexcept -> iterator
            {
                auto copy = *this;
                --(*this);
                return copy;
            }

            auto operator+=(difference_type n) noexcept -> iterator&
            {
                m_vector += n;
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
                return lhs.m_vector - rhs.m_vector;
            }

            // NOLINTNEXTLINE (*const*)
            [[nodiscard]] auto operator*() const -> value_type
            {
                return m_scalar - value_type(*m_vector);
            }
            // NOLINTNEXTLINE (*const*)
            [[nodiscard]] auto operator[](difference_type idx) const -> value_type
            {
                return m_scalar - value_type(*(m_vector + idx));
            }
            [[nodiscard]] auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
            {
                const auto scalar = avx::broadcast(m_scalar);
                const auto vector = expression_traits::cx_reg(m_vector, idx);

                return avx::sub(scalar, vector);
            }

            [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept
            {
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

        scalar_sub& operator=(scalar_sub&& other) noexcept
        {
            m_scalar = std::move(other.m_scalar);
            m_vector = std::move(other.m_vector);
            return *this;
        };
        scalar_sub& operator=(const scalar_sub&) noexcept = delete;

        [[nodiscard]] auto begin() const noexcept -> iterator
        {
            return iterator(m_scalar, m_vector.begin());
        }
        [[nodiscard]] auto end() const noexcept -> iterator
        {
            return iterator(m_scalar, m_vector.end());
        }

        [[nodiscard]] auto operator[](std::size_t idx) const
        {
            return m_scalar - std::complex<real_type>(m_vector[idx]);
        }
        [[nodiscard]] constexpr auto size() const noexcept -> std::size_t
        {
            return m_vector.size();
        }

    private:
        const E m_vector;
        S       m_scalar;
    };

    template<typename E, typename S>
        requires compatible_scalar<E, S>
    class scalar_mul : public std::ranges::view_base
    {
        friend auto operator*<E, S>(const E& vector, S scalar);
        friend auto operator*<E, S>(S scalar, const E& vector);
        friend auto operator/<E, S>(const E& vector, S scalar);

    public:
        using real_type = typename E::real_type;

        class iterator
        {
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

            iterator() = default;

            iterator(const iterator& other) noexcept = default;
            iterator(iterator&& other) noexcept      = default;

            ~iterator() = default;

            iterator& operator=(const iterator& other) noexcept = default;
            iterator& operator=(iterator&& other) noexcept      = default;


            [[nodiscard]] bool operator==(const iterator& other) const noexcept
            {
                return (m_vector == other.m_vector);
            }
            [[nodiscard]] auto operator<=>(const iterator& other) const noexcept
            {
                return (m_vector <=> other.m_vector);
            }

            auto operator++() noexcept -> iterator&
            {
                ++m_vector;
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
                --m_vector;
            }
            auto operator--(int) noexcept -> iterator
            {
                auto copy = *this;
                --(*this);
                return copy;
            }

            auto operator+=(difference_type n) noexcept -> iterator&
            {
                m_vector += n;
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
                return lhs.m_vector - rhs.m_vector;
            }

            // NOLINTNEXTLINE (*const*)
            [[nodiscard]] auto operator*() const -> value_type
            {
                return m_scalar * value_type(*m_vector);
            }
            // NOLINTNEXTLINE (*const*)
            [[nodiscard]] auto operator[](difference_type idx) const -> value_type
            {
                return m_scalar * value_type(*(m_vector + idx));
            }
            [[nodiscard]] auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
            {
                const auto scalar = avx::broadcast(m_scalar);
                const auto vector = expression_traits::cx_reg(m_vector, idx);

                return avx::mul(scalar, vector);
            }

            [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept
            {
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

        scalar_mul& operator=(scalar_mul&& other) noexcept
        {
            m_scalar = std::move(other.m_scalar);
            m_vector = std::move(other.m_vector);
            return *this;
        };
        scalar_mul& operator=(const scalar_mul&) noexcept = delete;

        [[nodiscard]] auto begin() const noexcept -> iterator
        {
            return iterator(m_scalar, m_vector.begin());
        }
        [[nodiscard]] auto end() const noexcept -> iterator
        {
            return iterator(m_scalar, m_vector.end());
        }

        [[nodiscard]] auto operator[](std::size_t idx) const
        {
            return m_scalar * std::complex<real_type>(m_vector[idx]);
        }
        [[nodiscard]] constexpr auto size() const noexcept -> std::size_t
        {
            return m_vector.size();
        }

    private:
        const E m_vector;
        S       m_scalar;
    };

    template<typename E, typename S>
        requires compatible_scalar<E, S>
    class scalar_div : public std::ranges::view_base
    {
        friend auto operator/<E, S>(S scalar, const E& vector);

    public:
        using real_type = typename E::real_type;

        class iterator
        {
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

            iterator() = default;

            iterator(const iterator& other) noexcept = default;
            iterator(iterator&& other) noexcept      = default;

            ~iterator() = default;

            iterator& operator=(const iterator& other) noexcept = default;
            iterator& operator=(iterator&& other) noexcept      = default;


            [[nodiscard]] bool operator==(const iterator& other) const noexcept
            {
                return (m_vector == other.m_vector);
            }
            [[nodiscard]] auto operator<=>(const iterator& other) const noexcept
            {
                return (m_vector <=> other.m_vector);
            }

            auto operator++() noexcept -> iterator&
            {
                ++m_vector;
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
                --m_vector;
            }
            auto operator--(int) noexcept -> iterator
            {
                auto copy = *this;
                --(*this);
                return copy;
            }

            auto operator+=(difference_type n) noexcept -> iterator&
            {
                m_vector += n;
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
                return lhs.m_vector - rhs.m_vector;
            }

            // NOLINTNEXTLINE (*const*)
            [[nodiscard]] auto operator*() const -> value_type
            {
                return m_scalar / value_type(*m_vector);
            }
            // NOLINTNEXTLINE (*const*)
            [[nodiscard]] auto operator[](difference_type idx) const -> value_type
            {
                return m_scalar / value_type(*(m_vector + idx));
            }
            [[nodiscard]] auto cx_reg(std::size_t idx) const -> avx::cx_reg<real_type>
            {
                const auto scalar = avx::broadcast(m_scalar);
                const auto vector = expression_traits::cx_reg(m_vector, idx);

                return avx::div(scalar, vector);
            }

            [[nodiscard]] constexpr bool aligned(std::size_t offset = 0) const noexcept
            {
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

        scalar_div& operator=(scalar_div&& other) noexcept
        {
            m_scalar = std::move(other.m_scalar);
            m_vector = std::move(other.m_vector);
            return *this;
        };
        scalar_div& operator=(const scalar_div&) noexcept = delete;

        [[nodiscard]] auto begin() const noexcept -> iterator
        {
            return iterator(m_scalar, m_vector.begin());
        }
        [[nodiscard]] auto end() const noexcept -> iterator
        {
            return iterator(m_scalar, m_vector.end());
        }

        [[nodiscard]] auto operator[](std::size_t idx) const
        {
            return m_scalar / std::complex<real_type>(m_vector[idx]);
        }
        [[nodiscard]] constexpr auto size() const noexcept -> std::size_t
        {
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
inline auto operator+(const E1& lhs, const E2& rhs)
{
    return internal::add(lhs, rhs);
};
template<typename E1, typename E2>
    requires internal::compatible_expression<E1, E2>
inline auto operator-(const E1& lhs, const E2& rhs)
{
    return internal::sub(lhs, rhs);
};
template<typename E1, typename E2>
    requires internal::compatible_expression<E1, E2>
inline auto operator*(const E1& lhs, const E2& rhs)
{
    return internal::mul(lhs, rhs);
};
template<typename E1, typename E2>
    requires internal::compatible_expression<E1, E2>
inline auto operator/(const E1& lhs, const E2& rhs)
{
    return internal::div(lhs, rhs);
};

template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator+(const E& vector, S scalar)
{
    return internal::scalar_add(scalar, vector);
}
template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator+(S scalar, const E& vector)
{
    return internal::scalar_add(scalar, vector);
}

template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator-(const E& vector, S scalar)
{
    return internal::scalar_add(-scalar, vector);
}
template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator-(S scalar, const E& vector)
{
    return internal::scalar_sub(scalar, vector);
}

template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator*(const E& vector, S scalar)
{
    return internal::scalar_mul(scalar, vector);
}
template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator*(S scalar, const E& vector)
{
    return internal::scalar_mul(scalar, vector);
}

template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator/(const E& vector, S scalar)
{
    return internal::scalar_mul(S(1) / scalar, vector);
}
template<typename E, typename S>
    requires internal::compatible_scalar<E, S>
inline auto operator/(S scalar, const E& vector)
{
    return internal::scalar_div(scalar, vector);
}

template<typename E, typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize> &&
             (!std::same_as<E, vector<T, PackSize, Allocator>>) &&
             requires(subrange<T, PackSize> subrange, E e) { e + subrange; }
inline auto operator+(const E& expression, const vector<T, PackSize, Allocator>& vector)
{
    return expression + subrange(vector.begin(), vector.size());
};
template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(subrange<T, PackSize> subrange, E e) { subrange + e; }
inline auto operator+(const vector<T, PackSize, Allocator>& vector, const E& expression)
{
    return subrange(vector.begin(), vector.size()) + expression;
};

template<typename E, typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize> &&
             (!std::same_as<E, vector<T, PackSize, Allocator>>) &&
             requires(subrange<T, PackSize> subrange, E e) { e - subrange; }
inline auto operator-(const E& expression, const vector<T, PackSize, Allocator>& vector)
{
    return expression - subrange(vector.begin(), vector.size());
};
template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(subrange<T, PackSize> subrange, E e) { subrange - e; }
inline auto operator-(const vector<T, PackSize, Allocator>& vector, const E& expression)
{
    return subrange(vector.begin(), vector.size()) - expression;
};

template<typename E, typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize> &&
             (!std::same_as<E, vector<T, PackSize, Allocator>>) &&
             requires(subrange<T, PackSize> subrange, E e) { e* subrange; }
inline auto operator*(const E& expression, const vector<T, PackSize, Allocator>& vector)
{
    return expression * subrange(vector.begin(), vector.size());
};
template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(subrange<T, PackSize> subrange, E e) { subrange* e; }
inline auto operator*(const vector<T, PackSize, Allocator>& vector, const E& expression)
{
    return subrange(vector.begin(), vector.size()) * expression;
};

template<typename E, typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize> &&
             (!std::same_as<E, vector<T, PackSize, Allocator>>) &&
             requires(subrange<T, PackSize> subrange, E e) { e / subrange; }
inline auto operator/(const E& expression, const vector<T, PackSize, Allocator>& vector)
{
    return expression / subrange(vector.begin(), vector.size());
};
template<typename T, std::size_t PackSize, typename Allocator, typename E>
    requires packed_floating_point<T, PackSize> &&
             requires(subrange<T, PackSize> subrange, E e) { subrange / e; }
inline auto operator/(const vector<T, PackSize, Allocator>& vector, const E& expression)
{
    return subrange(vector.begin(), vector.size()) / expression;
};

// #endregion operator definitions
}    // namespace pcx
#endif