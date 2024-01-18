#ifndef STATIC_MDSTORAGE_HPP
#define STATIC_MDSTORAGE_HPP

#include "allocators.hpp"
#include "element_access.hpp"
#include "mdstorage.hpp"
#include "meta.hpp"
#include "types.hpp"

#include <algorithm>
#include <array>
#include <bits/utility.h>
#include <concepts>
#include <numeric>

namespace pcx {
//asd

namespace md {

namespace detail_ {
struct basis_base {};
}    // namespace detail_

template<uZ Alignment, auto... Axes>
class static_basis : detail_::basis_base {
    template<typename>
    struct basis_from_value_sequence;
    template<auto... Vs>
    struct basis_from_value_sequence<meta::value_sequence<Vs...>> {
        using type = static_basis<Vs...>;
    };

    template<uZ I, typename Excluded>
    struct outer_remaining_impl {
        static constexpr auto value =
            std::conditional_t<meta::contains_value<Excluded, meta::index_into_values<I, Axes...>>,
                               outer_remaining_impl<I - 1, Excluded>,
                               meta::value_constant<meta::index_into_values<I, Axes...>>>::value;
    };
    template<uZ I, typename Excluded>
    struct inner_remaining_impl {
        static constexpr auto value =
            std::conditional_t<meta::contains_value<Excluded, meta::index_into_values<I, Axes...>>,
                               inner_remaining_impl<I + 1, Excluded>,
                               meta::value_constant<meta::index_into_values<I, Axes...>>>::value;
    };

    template<uZ... Is>
    constexpr static_basis(auto&& extents, std::index_sequence<Is...>)
    : m_extents{extents[Is]...} {};

public:
    static constexpr uZ size = sizeof...(Axes);

    template<std::unsigned_integral... Us>
        requires /**/ (sizeof...(Us) == size)
    constexpr explicit static_basis(Us... extents) noexcept
    : m_extents{extents...} {};

    template<auto Axis>
        requires meta::value_matched<Axis, Axes...>
    [[nodiscard]] static constexpr auto index() noexcept {
        return meta::find_first_in_values<Axis, Axes...>;
    }

    template<uZ Index>
    [[nodiscard]] static constexpr auto axis() noexcept {
        return meta::index_into_values<Index, Axes...>;
    };

    template<auto Axis>
    [[nodiscard]] static constexpr bool contains() noexcept {
        return meta::value_matched<Axis, Axes...>;
    }

    static constexpr auto inner_axis = axis<0>();
    static constexpr auto outer_axis = axis<size - 1>();

    template<auto Axis>
        requires meta::value_matched<Axis, Axes...>
    [[nodiscard]] consteval auto exclude() noexcept {
        using new_basis_t =
            basis_from_value_sequence<meta::filter_value_sequence<meta::value_sequence<Axes...>, Axis>>;

        constexpr uZ index = meta::find_first_in_values<Axis, Axes...>;
        using index_seq    = meta::index_to_value_sequence<std::make_index_sequence<size>>;
        using filtered     = meta::value_to_index_sequence<meta::filter_value_sequence<index_seq, index>>;

        return new_basis_t(m_extents, filtered{});
    };

    template<auto Axis>
        requires meta::value_matched<Axis, Axes...>
    [[nodiscard]] constexpr auto extent() noexcept {
        constexpr uZ index = meta::find_first_in_values<Axis, Axes...>;
        return m_extents[index];
    }

    template<meta::any_value_sequence Excluded>
    static constexpr auto outer_axis_remaining = outer_remaining_impl<size - 1, Excluded>::value;

    template<meta::any_value_sequence Excluded>
    static constexpr auto inner_axis_remaining = outer_remaining_impl<0, Excluded>::value;

private:
    std::array<uZ, size> m_extents;
};

template<typename T>
concept md_basis = std::derived_from<T, detail_::basis_base>;

namespace detail_ {
template<auto Basis, uZ Alignment>
constexpr auto storage_size() -> uZ {
    constexpr uZ inner_extent = Basis.template extent<Basis.inner_axis>();

    uZ size = (inner_extent * 2UL + Alignment - 1UL) / Alignment * Alignment;
    for (uZ i = 1; i < Basis.size; ++i) {
        size *= Basis.template extent<Basis.template axis<i>()>();
    }
    return size;
}
template<auto Basis, meta::any_value_sequence ExcludedAxes, auto Axis, uZ Alignment, uZ PackSize>
    requires /**/ (Basis.template contans<Axis>())
[[nodiscard]] inline auto get_static_slice_offset(auto* ptr, uZ index) noexcept {
    if constexpr (equal_values<Axis, Basis.inner_axis>) {
        return ptr + pidx<PackSize>(index);
    } else {
        constexpr auto div = []<uZ I>(auto&& f, uZ_constant<I>) {
            constexpr auto axis = Basis.template axis<I>();
            if constexpr (equal_values<Axis, axis>) {
                return Basis.template extent<axis>();
            } else {
                return Basis.template extent<axis>() * f(f, uZ_constant<I - 1>{});
            }
        };
        constexpr uZ storage_size = detail_::storage_size<Basis, Alignment>();
        constexpr uZ stride       = storage_size / div(div, uZ_constant<Basis.size - 1>{});

        return ptr + stride * index;
    }
};
};    // namespace detail_

template<auto Basis, meta::any_value_sequence Excluded, uZ Alignment>
class static_iter_base {
public:
    [[nodiscard]] static constexpr auto stride() noexcept -> uZ {
        constexpr auto denom = []<uZ I>(auto&& f, uZ_constant<I>) {
            constexpr auto axis = Basis.template axis<I>();
            if constexpr (meta::contains_value<Excluded, axis>) {
                constexpr auto next_axis = Basis.template axis<I - 1>();
                return Basis.template extent<next_axis>() * f(f, uZ_constant<I - 1>{});
            } else {
                return 1;
            }
        };
        constexpr uZ d = denom(denom, uZ_constant<Basis.size - 1>{});

        constexpr uZ inner_extent = Basis.template extent<Basis.inner_axis>();

        uZ stride = (inner_extent * 2UL + Alignment - 1UL) / Alignment * Alignment;
        for (uZ i = 1; i < Basis.size - 1; ++i) {
            stride *= Basis.template extent<Basis.template axis<i>()>();
        }
        stride /= d;
        return stride;
    };

protected:
private:
};

template<auto Basis>
class dynamic_iter_base {
    using extents_t = std::array<uZ, Basis.size>;

protected:
    [[nodiscard]] auto stride() const noexcept -> uZ {
        return m_stride;
    };

    template<auto Axis>
    [[nodiscard]] auto extent() const noexcept -> uZ {
        return (*m_extents_ptr)[Basis.template extent<Axis>()];
    };

    [[nodiscard]] inline auto get(auto* ptr, uZ idx) const noexcept {
        return 0;
    }

private:
    uZ         m_stride;
    extents_t* m_extents_ptr;
};

template<bool Const, bool Contigious, typename T, uZ PackSize, typename Base>
class iterator : Base {
    using pointer = T*;

    explicit iterator(pointer ptr, auto&&... other) noexcept
    : Base(std::forward(other...))
    , m_ptr(ptr){};

public:
    iterator()                                    = default;
    iterator(const iterator&) noexcept            = default;
    iterator(iterator&&) noexcept                 = default;
    iterator& operator=(const iterator&) noexcept = default;
    iterator& operator=(iterator&&) noexcept      = default;
    ~iterator()                                   = default;

    using value_type       = decltype(std::declval<dynamic_iter_base>().get(pointer{}, 0));
    using iterator_concept = std::random_access_iterator_tag;
    using difference_type  = iZ;

    inline auto operator+=(difference_type n) noexcept -> iterator& {
        m_ptr += Base::stride() * n;
        return *this;
    };
    inline auto operator-=(difference_type n) noexcept -> iterator& {
        m_ptr -= Base::stride() * n;
        return *this;
    };

    inline auto operator++() noexcept -> iterator& {
        m_ptr += Base::stride();
        return *this;
    };
    inline auto operator++(int) noexcept -> iterator {
        auto copy = *this;
        m_ptr += Base::stride();
        return copy;
    };
    inline auto operator--() noexcept -> iterator& {
        m_ptr -= Base::stride();
        return *this;
    };
    inline auto operator--(int) noexcept -> iterator {
        auto copy = *this;
        m_ptr -= Base::stride();
        return copy;
    };

    [[nodiscard]] inline friend auto operator+(const iterator& lhs, difference_type rhs) noexcept
        -> iterator {
        return {lhs.m_ptr + rhs * lhs.stride(), static_cast<const Base&>(lhs)};
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

template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ Alignment, uZ PackSize>
class static_slice_base {
    static constexpr auto outer_axis = Basis.template outer_axis_remaining<ExcludedAxes>;

protected:
    static_slice_base() = default;

    using iterator_base =
        static_iter_base<Basis, meta::expand_value_sequence<ExcludedAxes, outer_axis>, Alignment>;

    static constexpr void iterator_args() noexcept {};

    template<auto Axis>
    using subslice = static_slice_base<Basis,    //
                                       meta::expand_value_sequence<ExcludedAxes, Axis>,
                                       Alignment,
                                       PackSize>;

    template<auto Axis>
    static constexpr void subslice_args(){};

    template<auto Axis>
    static constexpr auto get_slice_offset(auto* ptr, uZ index) noexcept {
        return detail_::get_static_slice_offset<Basis, ExcludedAxes, Axis, Alignment, PackSize>(ptr, index);
    }

    template<auto Axis>
    static constexpr auto get_extent() noexcept -> uZ {
        return Basis.template extent<Axis>;
    }

private:
};

template<auto Basis>
class dynamic_slice_base {
    using extent_type = std::array<uZ, 0>;

protected:
    dynamic_slice_base() = default;

    explicit dynamic_slice_base(extent_type* extents_ptr) noexcept
    : m_extents_ptr(extents_ptr){};

    using iterator_base = dynamic_iter_base;

    void iterator_args() const noexcept {};


    template<auto Axis>
    using subslice = dynamic_slice_base;
    template<auto Axis>
    auto subslice_args() noexcept {
        return m_extents_ptr;
    };
    template<auto Axis>
    [[nodiscard]] auto get_subslice_offset(auto* ptr, uZ index) noexcept {
        return ptr;
    }
    template<auto Axis>
    [[nodiscard]] auto get_extent() const noexcept -> uZ {
        return (*m_extents_ptr)[Basis.template index<Axis>()];
    }

private:
    extent_type* m_extents_ptr;
};

template<bool Const, bool Contigious, auto Basis, typename T, uZ PackSize, typename Base>
class sslice
: public std::ranges::view_base
, pcx::detail_::pack_aligned_base<Basis.size == 1 && Contigious>
, Base {
    using iterator = iterator<Const, Contigious, T, PackSize, typename Base::iterator_base>;

    static constexpr bool vector_like = Basis.size == 1 && Contigious;

    explicit constexpr sslice(T* start, auto&&... args)
    : Base(std::forward(args...))
    , m_start(start){};


public:
    sslice()
    : Base(){};

    sslice(const sslice&) noexcept = default;
    sslice(sslice&&) noexcept      = default;

    sslice& operator=(const sslice&) noexcept = default;
    sslice& operator=(sslice&&) noexcept      = default;

    ~sslice() = default;

    template<auto Axis>
        requires /**/ (Basis.template contains<Axis>)
    [[nodiscard]] auto slice(uZ index) const noexcept {
        using new_base  = Base::template subslice<Axis>;
        using new_slice = sslice<Const, Contigious, Basis, T, PackSize, new_base>;
        auto* new_start = get_subslice_offset<Axis>(m_start, index);
        return new_slice(new_start, subslice_args<Axis>());
    }

    [[nodiscard]] auto operator[](uZ index) const noexcept {
        return slice<Basis.outer_axis>(index);
    }

    // TODO: `at()`
    // TODO: multidimensional slice and operator[]

    [[nodiscard]] auto begin() const noexcept {
        return iterator(m_start, Base::iterator_args());
    }

    [[nodiscard]] auto end() const noexcept {
        return begin() += 0;
    }

    [[nodiscard]] auto size() const noexcept -> uZ {
        return extent<Base::outer_axis>();
    }

    template<auto Axis>
    // requires /**/ (Basis.template contains<Axis>)
    [[nodiscard]] auto extent() const noexcept -> uZ {
        return get_extent<Axis>();
    }

private:
    T* m_start{};
};

template<typename T, auto Basis, uZ PackSize, uZ DataAlignment, typename Allocator>
class static_storage_base {
    static constexpr uZ alignment    = std::lcm(PackSize, DataAlignment);
    static constexpr uZ storage_size = detail_::storage_size<Basis, alignment>();

public:
    template<auto Axis>
    auto slice(uZ index) {
        if constexpr (equal_values<Axis, Basis.inner_axis>) {
            auto* ptr = data() + pidx<PackSize>(index);
        } else {
            constexpr auto div = []<uZ I>(auto&& f, uZ_constant<I>) {
                constexpr auto axis = Basis.template axis<I>();
                if constexpr (equal_values<Axis, axis>) {
                    return Basis.template extent<axis>();
                } else {
                    return Basis.template extent<axis>() * f(f, uZ_constant<I - 1>{});
                }
            };
            constexpr uZ stride = storage_size / div(div, uZ_constant<Basis.size - 1>{});

            auto* ptr = data() + stride * index;
        }
    }

    auto begin(){};

    [[nodiscard]] constexpr auto data() noexcept -> T* {
        return m_data.data();
    }

private:
    std::array<T, storage_size> m_data{};
};

class sslice_base {
    [[nodiscard]] auto stride() const noexcept -> uZ;
    [[nodiscard]] auto extents() const noexcept;

protected:
    inline auto get_slice(auto* ptr, uZ idx) const noexcept;

private:
};

// template<typename T, uZ PackSize, md_basis Basis, typename Extents, bool Contigious, bool Const>
}    // namespace md

}    // namespace pcx


#endif