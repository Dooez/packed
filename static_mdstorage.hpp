#ifndef STATIC_dynamic_storage_base_HPP
#define STATIC_dynamic_storage_base_HPP

#include "allocators.hpp"
#include "dynamic_storage_base.hpp"
#include "element_access.hpp"
#include "meta.hpp"
#include "types.hpp"

#include <algorithm>
#include <array>
#include <bits/utility.h>
#include <concepts>
#include <numeric>
#include <sstream>

namespace pcx {

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
template<uZ Size>
struct dynamic_extents_info {
    uZ                   stride;
    std::array<uZ, Size> extents;
};

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
[[nodiscard]] inline auto get_static_slice_offset(uZ index) noexcept -> uZ {
    if constexpr (equal_values<Axis, Basis.inner_axis>) {
        return pidx<PackSize>(index);
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

        return stride * index;
    }
};
template<auto Basis, meta::any_value_sequence ExcludedAxes, auto Axis, uZ PackSize>
    requires /**/ (Basis.template contans<Axis>())
[[nodiscard]] inline constexpr auto get_dynamic_slice_offset(uZ                                start_stride,
                                                             const std::array<uZ, Basis.size>& extents,
                                                             uZ index) noexcept -> uZ {
    if constexpr (equal_values<Axis, Basis.inner_axis>) {
        return pidx<PackSize>(index);
    } else {
        constexpr auto div = []<uZ I>(auto&& f, auto& extents, uZ_constant<I>) {
            constexpr auto axis = Basis.template axis<I>();
            if constexpr (equal_values<Axis, axis>) {
                return extents[I];
            } else {
                return extents[I] * f(f, extents, uZ_constant<I - 1>{});
            }
        };
        constexpr uZ stride = start_stride / div(div, extents, uZ_constant<Basis.size - 2>{});

        return stride * index;
    }
};

template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ PackSize, uZ Alignment>
class static_slice_base;
template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ PackSize>
class dynamic_slice_base;
};    // namespace detail_

template<bool Const, bool Contigious, auto Basis, typename T, uZ PackSize, typename Base>
class sslice;

template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ PackSize, uZ Alignment>
class static_iter_base {
    static constexpr auto outer_axis = Basis.template outer_axis_remaining<ExcludedAxes>;

protected:
    using slice_base = detail_::
        static_slice_base<Basis, meta::expand_value_sequence<ExcludedAxes, outer_axis>, PackSize, Alignment>;

    static constexpr void slice_base_args() noexcept {};

    static_iter_base() = default;

    [[nodiscard]] static constexpr auto stride() noexcept -> uZ {
        return s_stride;
    };

private:
    static constexpr uZ s_stride = [] {
        constexpr auto denom = []<uZ I>(auto&& f, uZ_constant<I>) {
            constexpr auto axis = Basis.template axis<I>();
            if constexpr (meta::contains_value<ExcludedAxes, axis>) {
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
    }();
};

template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ PackSize>
class dynamic_iter_base {
    static constexpr auto outer_axis = Basis.template outer_axis_remaining<ExcludedAxes>;

    using extents_type = detail_::dynamic_extents_info<Basis.size>;

protected:
    using slice_base =
        detail_::dynamic_slice_base<Basis, meta::expand_value_sequence<ExcludedAxes, outer_axis>, PackSize>;

    [[nodiscard]] auto slice_base_args() const noexcept {
        return m_extents_ptr;
    };

    explicit dynamic_iter_base(extents_type* extents_ptr) noexcept
    : m_stride(calc_stride(extents_ptr))
    , m_extents_ptr(extents_ptr){};

    [[nodiscard]] auto stride() const noexcept -> uZ {
        return m_stride;
    };

private:
    auto calc_stride(extents_type* extents_ptr) {
        auto stride = extents_ptr->stride;

        auto f = []<uZ I>(auto&& f, auto& extents, uZ_constant<I>) {
            constexpr auto axis = Basis.template axis<I>();
            if constexpr (equal_values<outer_axis, axis>) {
                return extents[I];
            } else {
                return extents[I] * f(f, extents, uZ_constant<I - 1>{});
            }
        };
        auto div = f(f, extents_ptr->extents, uZ_constant<Basis.size() - 2>{});
        return stride / div;
    }

    uZ            m_stride{};
    extents_type* m_extents_ptr{};
};

template<bool Const, bool Contigious, auto Basis, typename T, uZ PackSize, typename Base>
class iterator : Base {
    using pointer = T*;

    template<bool, bool, auto, typename, uZ, typename>
    friend class sslice;

    explicit iterator(pointer ptr, auto&&... other) noexcept
    : Base(std::forward(other...))
    , m_ptr(ptr){};

    using sslice = sslice<Const, Contigious, Basis, T, PackSize, typename Base::slice_base>;

public:
    iterator()                                    = default;
    iterator(const iterator&) noexcept            = default;
    iterator(iterator&&) noexcept                 = default;
    iterator& operator=(const iterator&) noexcept = default;
    iterator& operator=(iterator&&) noexcept      = default;
    ~iterator()                                   = default;

    using value_type       = sslice;
    using iterator_concept = std::random_access_iterator_tag;
    using difference_type  = iZ;

    [[nodiscard]] inline auto operator*() const noexcept {
        return sslice(m_ptr, Base::slice_base_args());
    }

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

namespace detail_ {
template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ PackSize, uZ Alignment>
class static_slice_base {
    static constexpr auto outer_axis = Basis.template outer_axis_remaining<ExcludedAxes>;

protected:
    static_slice_base() = default;

    using iterator_base =
        static_iter_base<Basis, meta::expand_value_sequence<ExcludedAxes, outer_axis>, PackSize, Alignment>;

    static constexpr void iterator_base_args() noexcept {};

    template<auto Axis>
    using new_slice_base = static_slice_base<Basis,    //
                                             meta::expand_value_sequence<ExcludedAxes, Axis>,
                                             PackSize,
                                             Alignment>;

    template<auto Axis>
    static constexpr void new_slice_base_args(){};

    template<auto Axis>
    static constexpr auto new_slice_offset(uZ index) noexcept {
        return detail_::get_static_slice_offset<Basis, ExcludedAxes, Axis, PackSize, Alignment>(index);
    }

    template<auto Axis>
    static constexpr auto get_extent() noexcept -> uZ {
        return Basis.template extent<Axis>;
    }

private:
};

template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ PackSize>
class dynamic_slice_base {
    static constexpr auto outer_axis = Basis.template outer_axis_remaining<ExcludedAxes>;

    using extents_type = detail_::dynamic_extents_info<Basis.size>;

protected:
    dynamic_slice_base() = default;

    explicit dynamic_slice_base(extents_type* extents_ptr) noexcept
    : m_extents_ptr(extents_ptr){};

    using iterator_base = dynamic_iter_base<Basis, ExcludedAxes, PackSize>;

    void iterator_base_args() const noexcept {}

    template<auto Axis>
    using new_slice_base =
        dynamic_slice_base<Basis, meta::expand_value_sequence<ExcludedAxes, Axis>, PackSize>;

    template<auto Axis>
    auto new_slice_base_args() noexcept {
        return m_extents_ptr;
    }
    template<auto Axis>
    [[nodiscard]] auto new_slice_offset(uZ index) noexcept -> uZ {
        const auto& stride  = m_extents_ptr->stride;
        const auto& extents = m_extents_ptr->extents;
        return detail_::get_dynamic_slice_offset<Basis, ExcludedAxes, Axis, PackSize>(stride, extents, index);
    }
    template<auto Axis>
    [[nodiscard]] auto get_extent() const noexcept -> uZ {
        return m_extents_ptr->extents[Basis.template index<Axis>()];
    }

private:
    extents_type* m_extents_ptr;
};
}    // namespace detail_

template<bool Const, bool Contigious, auto Basis, typename T, uZ PackSize, typename Base>
class sslice
: public std::ranges::view_base
, pcx::detail_::pack_aligned_base<Basis.size == 1 && Contigious>
, Base {
    template<bool, bool, typename, uZ, typename>
    friend class iterator;

    using iterator = iterator<Const, Contigious, Basis, T, PackSize, typename Base::iterator_base>;

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
        using new_base  = Base::template new_slice_base<Axis>;
        using new_slice = sslice<Const, Contigious, Basis, T, PackSize, new_base>;
        auto* new_start = m_start + Base::template new_slice_offset<Axis>(index);
        return new_slice(new_start, Base::template new_slice_base_args<Axis>());
    }

    [[nodiscard]] auto operator[](uZ index) const noexcept {
        return slice<Basis.outer_axis>(index);
    }

    // TODO: `at()`
    // TODO: multidimensional slice and operator[]

    [[nodiscard]] auto begin() const noexcept {
        return iterator(m_start, Base::iterator_base_args());
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

template<typename T, auto Basis, uZ PackSize, uZ Alignment>
class static_storage_base {
    static constexpr uZ alignment    = std::lcm(PackSize, Alignment);
    static constexpr uZ storage_size = detail_::storage_size<Basis, alignment>();

protected:
    using iterator_base = static_iter_base<Basis, meta::value_sequence<>, PackSize, Alignment>;

    template<auto Axis>
    using slice_base = detail_::static_slice_base<Basis, meta::value_sequence<Axis>, PackSize, Alignment>;

    static constexpr void slice_base_args() noexcept {};

    [[nodiscard]] constexpr auto get_data() noexcept -> T* {
        return m_data.data();
    }

private:
    std::array<T, storage_size> m_data{};
};

template<typename T, auto Basis, uZ PackSize, uZ Alignment, typename Allocator>
class dynamic_storage_base {
    using extents_type    = detail_::dynamic_extents_info<Basis::size>;
    using iterator_traits = std::iterator_traits<Allocator>;

protected:
    using iterator_base = dynamic_iter_base<Basis, meta::value_sequence<>, PackSize>;

    template<auto Axis>
    using slice_base = detail_::dynamic_slice_base<Basis, meta::value_sequence<>, PackSize>;

    /**
     * @brief Construct a new mdarray object.
     * 
     * @param extents   Extents of individual axes. `extents[i]` corresponds to `Basis::axis<i>`.
     * @param alignment Contigious axis storage is padded to a multiple of the least common multiple of `alignment` and `PackSize`.
     * @param allocator 
     */
    explicit dynamic_storage_base(const extents_type& extents, Allocator allocator = {})
    : m_extents{.extents = extents, .stride = 0}
    , m_allocator(allocator) {
        uZ align    = std::lcm(Alignment, PackSize * 2);
        uZ misalign = (extents[0] * 2) % align;
        uZ size     = extents[0] * 2 + (misalign > 0 ? align - misalign : 0);
        if constexpr (!vector_like) {
            uZ stride = size;
            for (uZ i = 1; i < Basis::size - 1; ++i) {
                stride *= m_extents[i];
            }
            size     = stride * m_extents.back();
            m_stride = stride;
        }
        m_ptr = allocator_traits::allocate(m_allocator, size);
        std::memset(m_ptr, 0, size * sizeof(T));
    };

    explicit dynamic_storage_base() = default;

    dynamic_storage_base(const dynamic_storage_base& other)            = delete;
    dynamic_storage_base& operator=(const dynamic_storage_base& other) = delete;

    dynamic_storage_base(dynamic_storage_base&& other) noexcept
    : m_allocator(std::move(other.m_allocator))
    , m_ptr(other.m_ptr)
    , m_extents(other.m_extents) {
        other.m_ptr     = nullptr;
        other.m_extents = {};
    };

    dynamic_storage_base& operator=(dynamic_storage_base&& other) noexcept(
        allocator_traits::propagate_on_container_move_assignment::value ||
        allocator_traits::is_always_equal::value) {
        using std::swap;
        if constexpr (allocator_traits::propagate_on_container_move_assignment::value) {
            if (m_allocator == other.m_allocator) {
                m_allocator = std::move(other.m_allocator);
                swap(m_ptr, other.m_ptr);
                swap(m_extents, other.m_extents);
            } else {
                deallocate();
                m_allocator     = std::move(other.m_allocator);
                m_ptr           = other.m_ptr;
                m_extents       = other.m_extents;
                other.m_ptr     = nullptr;
                other.m_extents = {};
            }
        } else if constexpr (allocator_traits::is_always_equal::value) {
            swap(m_ptr, other.m_ptr);
            swap(m_stride, other.n_stride);
            swap(m_extents, other.m_extents);
        } else {
            if (m_allocator == other.m_allocator) {
                swap(m_ptr, other.m_ptr);
                swap(m_extents, other.m_extents);
            } else {
                auto size = other.m_stride * other.m_extents.back();
                if (size != m_stride * m_extents.back()) {
                    deallocate();
                    m_ptr = allocator_traits::allocate(m_allocator, size);
                }
                m_extents = other.m_extents;
                std::memcpy(m_ptr, other.m_ptr, size * sizeof(T));
            }
        }
    }

    ~dynamic_storage_base() noexcept {
        deallocate();
    }

private:
    void deallocate() noexcept {
        iter_traits::deallocate(m_allocator, m_ptr, m_stride * m_extents.extents.back());
    }

    Allocator    m_allocator;
    extents_type m_extents;
};

}    // namespace md

}    // namespace pcx


#endif