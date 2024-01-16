#ifndef STATIC_MDSTORAGE_HPP
#define STATIC_MDSTORAGE_HPP

#include "allocators.hpp"
#include "element_access.hpp"
#include "mdstorage.hpp"
#include "meta.hpp"
#include "types.hpp"

#include <array>
#include <concepts>
#include <numeric>

namespace pcx {
//asd

namespace md {

namespace detail_ {
struct basis_base {};
}    // namespace detail_

template<auto... Axes>
class static_basis : detail_::basis_base {
public:
    static constexpr uZ size = sizeof...(Axes);

    template<std::unsigned_integral... Us>
        requires /**/ (sizeof...(Us) == size)
    constexpr explicit static_basis(Us... extents) noexcept
    : m_extents{extents...} {}

    template<auto Axis>
    [[nodiscard]] static constexpr auto index() noexcept {
        return 0;
    }

private:
    std::array<uZ, size> m_extents;
};

/**
 * @brief Basis with left axis being most contigious in memory. Axis index increases left to right.
 * 
 * @tparam Axes 
 */
template<auto... Axes>
    requires(sizeof...(Axes) > 0, unique_values<Axes...>)
struct left_first_basis : detail_::basis_base {
private:
    template<uZ I, auto Vmatch, auto V, auto... Vs>
    struct idx_impl {
        static constexpr uZ value = equal_values<Vmatch, V> ? I : idx_impl<I + 1, Vmatch, Vs...>::value;
    };
    template<uZ I, auto Vmatch, auto V>
    struct idx_impl<I, Vmatch, V> {
        static constexpr uZ value = I;
    };

    template<uZ I, uZ Imatch, auto V, auto... Vs>
    struct value_impl {
        static constexpr auto value = value_impl<I + 1, Imatch, Vs...>::value;
    };
    template<uZ Imatch, auto V, auto... Vs>
    struct value_impl<Imatch, Imatch, V, Vs...> {
        static constexpr auto value = V;
    };

    template<typename S>
    struct basis_from_seq {};
    template<auto... Vs>
    struct basis_from_seq<detail_::value_sequence<Vs...>> {
        using type = left_first_basis<Vs...>;
    };
    template<typename S>
    using basis_from_seq_t = typename basis_from_seq<S>::type;

public:
    static constexpr uZ size = sizeof...(Axes);

    static constexpr auto inner_axis = value_impl<0, 0, Axes...>::value;

    static constexpr auto outer_axis = value_impl<0, size - 1, Axes...>::value;

    /**
     * @brief Index of the axis in basis.
     * 
     * @tparam Axis value representing the axis.
     */
    template<auto Axis>
        requires value_matched<Axis, Axes...>
    static constexpr uZ index = idx_impl<0, Axis, Axes...>::value;

    template<uZ I>
        requires /**/ (I < size)
    static constexpr auto axis = value_impl<0, I, Axes...>::value;

    template<auto Axis>
        requires value_matched<Axis, Axes...>
    using exclude = basis_from_seq_t<detail_::filter_value_sequence<detail_::value_sequence<Axes...>, Axis>>;

    template<auto Axis>
    static constexpr bool contains = value_matched<Axis, Axes...>;
};

/**
 * @brief Basis with right axis being most contigious in memory.
 * 
 * @tparam Axes 
 */
template<auto... Axes>
    requires(sizeof...(Axes) > 0, unique_values<Axes...>)
struct right_first_basis : detail_::basis_base {
private:
    template<uZ I, auto Vmatch, auto V, auto... Vs>
    struct idx_impl {
        static constexpr uZ value = equal_values<Vmatch, V> ? I : idx_impl<I - 1, Vmatch, Vs...>::value;
    };
    template<uZ I, auto Vmatch, auto V>
    struct idx_impl<I, Vmatch, V> {
        static constexpr uZ value = I;
    };

    template<uZ I, uZ Imatch, auto V, auto... Vs>
    struct value_impl {
        static constexpr auto value = value_impl<I - 1, Imatch, Vs...>::value;
    };
    template<uZ Imatch, auto V, auto... Vs>
    struct value_impl<Imatch, Imatch, V, Vs...> {
        static constexpr auto value = V;
    };

    template<typename S>
    struct basis_from_seq {};
    template<auto... Vs>
    struct basis_from_seq<detail_::value_sequence<Vs...>> {
        using type = right_first_basis<Vs...>;
    };

    template<typename S>
    using basis_from_seq_t = typename basis_from_seq<S>::type;

public:
    static constexpr uZ size = sizeof...(Axes);

    static constexpr auto inner_axis = value_impl<size - 1, 0, Axes...>::value;

    static constexpr auto outer_axis = value_impl<size - 1, size - 1, Axes...>::value;

    template<auto Axis>
    static constexpr bool contains = value_matched<Axis, Axes...>;

    /**
     * @brief Index of the axis in basis.
     * 
     * @tparam Axis value representing the axis.
     */
    template<auto Axis>
        requires value_matched<Axis, Axes...>
    static constexpr uZ index = idx_impl<size - 1, Axis, Axes...>::value;

    template<uZ I>
        requires /**/ (I < size)
    static constexpr auto axis = value_impl<size - 1, I, Axes...>::value;

    template<auto Axis>
        requires value_matched<Axis, Axes...>
    using exclude = basis_from_seq_t<detail_::filter_value_sequence<detail_::value_sequence<Axes...>, Axis>>;
};

template<typename T>
concept md_basis = std::derived_from<T, detail_::basis_base>;

class iter_base {
public:
    [[nodiscard]] auto stride() const noexcept -> uZ;
    [[nodiscard]] auto extents() const noexcept;

protected:
    [[nodiscard]] inline auto get(auto* ptr, uZ idx) const noexcept {
        return 0;
    }
};

template<uZ Alignment, uZ... Ns>
struct extents {
public:
    static constexpr uZ alignment = Alignment;
};

using dynamic_extents = extents<1>;

template<uZ Alignment, uZ... Ns>
    requires(sizeof...(Ns) > 0)
struct extents<Alignment, Ns...> {
    static constexpr uZ alignment = Alignment;
    static constexpr uZ count     = sizeof...(Ns);

    template<uZ Index>
        requires /**/ (Index < count)
    static constexpr uZ extent =
        pcx::detail_::index_value_sequence_v<pcx::detail_::value_sequence<Ns...>, Index>;

    static constexpr uZ outer_extent = extent<count - 1>;
    // static constexpr uZ inner_extent = detail_::index_value_sequence_v<detail_::value_sequence<Ns...>, 0>;
    static constexpr uZ inner_extent = extent<count - count>;
    // static constexpr uZ inner_extent = extent<0UL>;

    template<uZ PackSize>
    static constexpr uZ storage_size = [] {
        constexpr uZ align    = std::lcm(Alignment, PackSize * 2);
        constexpr uZ misalign = (extent<0> * 2) % align;
        uZ           size     = extent<0> * 2 + (misalign > 0 ? align - misalign : 0);
        size *= []<uZ... Is>(std::index_sequence<Is...>) {
            return (extent<Is + 1> * ...);
        }(std::make_index_sequence<sizeof...(Ns) - 1>{});
        return size;
    }();

    template<uZ PackSize, md_basis Basis, auto... Excluded>
    static consteval auto stride(pcx::detail_::value_sequence<Excluded...>) -> uZ {
        auto stride = storage_size<PackSize>;
        for (uZ i = 0; i < Basis::size; ++i) {}
        return 0;
    };
};
template<typename T>
struct is_extents : std::false_type {};
template<uZ... Ns>
struct is_extents<extents<Ns...>> : std::true_type {};

template<md_basis Basis, typename Extents, typename ExcludeSeq, uZ PackSize, bool Const, bool Contigious>
class iter_static_base {
public:
    [[nodiscard]] auto stride() const noexcept -> uZ {
        return 0;
    };
    [[nodiscard]] auto extents() const noexcept;

protected:
    [[nodiscard]] inline auto get(auto* ptr, uZ idx) const noexcept {
        return ptr + stride() * idx;
    }

private:
};

template<typename T, md_basis Basis, uZ PackSize, bool Const, bool Contigious>
class iterator : iter_base {
    using pointer = T*;

    explicit iterator(pointer ptr, auto&&... other) noexcept
    : iter_base(std::forward(other...))
    , m_ptr(ptr){};

public:
    iterator()                                    = default;
    iterator(const iterator&) noexcept            = default;
    iterator(iterator&&) noexcept                 = default;
    iterator& operator=(const iterator&) noexcept = default;
    iterator& operator=(iterator&&) noexcept      = default;
    ~iterator()                                   = default;

    using value_type       = decltype(std::declval<iter_base>().get(pointer{}, 0));
    using iterator_concept = std::random_access_iterator_tag;
    using difference_type  = iZ;

    inline auto operator+=(difference_type n) noexcept -> iterator& {
        m_ptr += stride() * n;
        return *this;
    };
    inline auto operator-=(difference_type n) noexcept -> iterator& {
        m_ptr -= stride() * n;
        return *this;
    };

    inline auto operator++() noexcept -> iterator& {
        m_ptr += stride();
        return *this;
    };
    inline auto operator++(int) noexcept -> iterator {
        auto copy = *this;
        m_ptr += stride();
        return copy;
    };
    inline auto operator--() noexcept -> iterator& {
        m_ptr -= stride();
        return *this;
    };
    inline auto operator--(int) noexcept -> iterator {
        auto copy = *this;
        m_ptr -= stride();
        return copy;
    };

    [[nodiscard]] inline friend auto operator+(const iterator& lhs, difference_type rhs) noexcept
        -> iterator {
        return {lhs.m_ptr + rhs * lhs.stride(), lhs.stride(), lhs.m_extents};
    };
    [[nodiscard]] inline friend auto operator+(difference_type lhs, const iterator& rhs) noexcept
        -> iterator {
        return rhs + lhs;
    };

    [[nodiscard]] inline friend auto operator-(const iterator& lhs, difference_type rhs) noexcept
        -> iterator {
        return lhs + (-rhs);
    };

    [[nodiscard]] inline friend auto operator-(const iterator& lhs, const iterator& rhs) noexcept
        -> difference_type {
        return (lhs.m_ptr - rhs.m_ptr) / lhs.stride();
    };

    [[nodiscard]] inline auto operator<=>(const iterator& other) const noexcept {
        return m_ptr <=> other.m_ptr;
    };

    [[nodiscard]] inline auto operator==(const iterator& other) const noexcept {
        return m_ptr == other.m_ptr;
    };

private:
    pointer m_ptr{};
};

class sslice_base {
    [[nodiscard]] auto stride() const noexcept -> uZ;
    [[nodiscard]] auto extents() const noexcept;

protected:
    inline auto get_slice(auto* ptr, uZ idx) const noexcept;

private:
};

template<typename T, uZ PackSize, md_basis Basis, typename Extents, bool Contigious, bool Const>
class sslice
: public std::ranges::view_base
, pcx::detail_::pack_aligned_base<Basis::size == 1 && Contigious>
, {
    static constexpr bool vector_like = Basis::size == 1 && Contigious;

    sslice(T* start, auto&&... args)
    :


        public : sslice() = default;

    sslice(const sslice&) noexcept = default;
    sslice(sslice&&) noexcept      = default;

    sslice& operator=(const sslice&) noexcept = default;
    sslice& operator=(sslice&&) noexcept      = default;

    ~sslice() = default;

    template<auto Axis>
        requires /**/ (Basis::template contains<Axis>)
    [[nodiscard]] auto slice(uZ index) const noexcept {
        return detail_::md_make_slice<T, PackSize, Basis, Axis, Const, Contigious>(
            m_start, m_stride, index, m_extents);
    }

    [[nodiscard]] auto operator[](uZ index) const noexcept {
        return slice<Basis::outer_axis>(index);
    }

    // TODO: `at()`
    // TODO: multidimensional slice and operator[]

    [[nodiscard]] auto begin() const noexcept {
        return detail_::md_make_iterator<T, PackSize, Basis, Const, Contigious>(
            m_start, stride(), 0, m_extents);
    }

    [[nodiscard]] auto end() const noexcept {
        return detail_::md_make_iterator<T, PackSize, Basis, Const, Contigious>(
            m_start, stride(), size(), m_extents);
    }

    [[nodiscard]] auto size() const noexcept -> uZ {
        return m_extents.back();
    }

    template<auto Axis>
        requires /**/ (Basis::template contains<Axis>)
    [[nodiscard]] auto extent() const noexcept -> uZ {
        return m_extents[Basis::template index<Axis>];
    }
    [[nodiscard]] auto extents() const noexcept -> uZ {
        return m_extents;
    }

private:
    [[nodiscard]] auto stride() const noexcept -> uZ {
        if constexpr (vector_like) {
            return 1;
        } else {
            return m_stride;
        }
    }

    using stride_type = std::conditional_t<vector_like, decltype([] {}), uZ>;
    T*           m_start{};
    stride_type  m_stride{};
    extents_type m_extents{};
};
}    // namespace md

}    // namespace pcx


#endif