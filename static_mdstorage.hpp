#ifndef STATIC_dynamic_storage_base_HPP
#define STATIC_dynamic_storage_base_HPP

#include "allocators.hpp"
#include "element_access.hpp"
#include "meta.hpp"
#include "types.hpp"

#include <algorithm>
#include <array>
#include <bits/utility.h>
#include <concepts>
#include <cstring>
#include <memory>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

#define _INLINE_   inline __attribute__((always_inline))
#define _NDINLINE_ [[nodiscard]] inline __attribute__((always_inline))
namespace pcx::md {

namespace detail_ {
struct basis_base {};
}    // namespace detail_

template<auto... Axes>
class static_basis : public detail_::basis_base {
    template<uZ I, typename Excluded>
    struct outer_remaining_impl {
        static constexpr auto value =
            std::conditional_t<meta::contains_value<Excluded, meta::index_into_values<I, Axes...>>,
                               outer_remaining_impl<I - 1, Excluded>,
                               meta::value_constant<meta::index_into_values<I, Axes...>>>::value;
    };
    template<typename Excluded>
    struct outer_remaining_impl<0, Excluded> {
        static constexpr auto value = meta::index_into_values<0, Axes...>;
    };

    template<uZ I, typename Excluded>
    struct inner_remaining_impl {
        static constexpr auto value =
            std::conditional_t<meta::contains_value<Excluded, meta::index_into_values<I, Axes...>>,
                               inner_remaining_impl<I + 1, Excluded>,
                               meta::value_constant<meta::index_into_values<I, Axes...>>>::value;
    };

public:
    static constexpr uZ size = sizeof...(Axes);

    using axis_order = std::make_index_sequence<size>;

private:
    using axis_order_as_vseq = meta::index_to_value_sequence<axis_order>;

    template<uZ... Is>
    constexpr explicit static_basis(std::index_sequence<Is...>, auto... extents)
    : extents{std::get<Is>(std::make_tuple(extents...))...} {};

public:
    template<std::unsigned_integral... Us>
        requires /**/ (sizeof...(Us) == size)
    constexpr explicit static_basis(Us... extents) noexcept
    : static_basis(axis_order{}, extents...){};

    /**
     * @brief Returns the axis identifier 0-indexed from inner to outer axis.
     * 
     * @tparam Index 0 corresponds to inner axis.c
     */
    template<uZ Index>
        requires /**/ (Index < size)
    _NDINLINE_ static constexpr auto axis() noexcept {
        constexpr uZ real_index = meta::index_into_sequence<Index, axis_order_as_vseq>;
        return meta::index_into_values<real_index, Axes...>;
    };

    static consteval auto contains(auto axis) -> bool {
        constexpr auto cmp_equal = [](auto a, auto b) {
            if constexpr (std::equality_comparable_with<decltype(a), decltype(b)>) {
                return a == b;
            } else {
                return false;
            }
        };
        return (cmp_equal(axis, Axes) || ...);
    }

    static consteval auto index_of(auto axis) -> uZ {
        assert(contains(axis));
        auto f = [axis]<uZ I>(auto&& f, uZ_constant<I>) {
            if constexpr (I == size - 1) {
                constexpr uZ Index = meta::index_into_sequence<I, axis_order_as_vseq>;
                return Index;
            } else {
                constexpr auto ith_axis = meta::index_into_values<I, Axes...>;
                if constexpr (std::equality_comparable_with<decltype(axis), decltype(ith_axis)>) {
                    if (axis == ith_axis) {
                        constexpr uZ Index = meta::index_into_sequence<I, axis_order_as_vseq>;
                        return Index;
                    }
                }
                return f(f, uZ_constant<I + 1>{});
            }
        };
        return f(f, uZ_constant<0>{});
    }

    consteval auto extent(auto axis) const -> uZ {
        assert(contains(axis));
        return extents[index_of(axis)];
    }

    static constexpr auto inner_axis = axis<meta::index_into_sequence<0, axis_order_as_vseq>>();
    static constexpr auto outer_axis = axis<meta::index_into_sequence<size - 1, axis_order_as_vseq>>();

    //TODO(Timofey): include axis_order
    template<meta::any_value_sequence Excluded>
    static constexpr auto outer_axis_remaining = outer_remaining_impl<size - 1, Excluded>::value;

    template<meta::any_value_sequence Excluded>
    static constexpr auto inner_axis_remaining = outer_remaining_impl<0, Excluded>::value;

    std::array<uZ, size> extents;
};

template<typename T>
concept md_basis = std::derived_from<T, detail_::basis_base>;

namespace detail_ {
template<uZ Size>
struct dynamic_extents_info {
private:
public:
    std::array<uZ, Size> extents;
    uZ                   storage_size;
};

template<auto Basis, uZ Alignment>
constexpr auto storage_size() -> uZ {
    constexpr uZ inner_extent = Basis.extent(Basis.inner_axis);

    uZ size = (inner_extent * 2UL + Alignment - 1UL) / Alignment * Alignment;

    if constexpr (Basis.size > 1) {
        constexpr auto f = []<uZ I>(auto&& f, uZ size, uZ_constant<I>) {
            constexpr auto axis = Basis.template axis<I>();
            if constexpr (Basis.size == 1) {
                return size;
            } else if constexpr (I == Basis.size - 1) {
                return size *= Basis.extent(axis);
            } else {
                return f(f, size *= Basis.extent(axis), uZ_constant<I + 1>{});
            }
        };
        size = f(f, size, uZ_constant<1>{});
    }
    return size;
}
template<auto Basis, meta::any_value_sequence ExcludedAxes, auto Axis, uZ Alignment, uZ PackSize>
    requires /**/ (Basis.contains(Axis))
_NDINLINE_ constexpr auto get_static_slice_offset(uZ index) noexcept -> uZ {
    if constexpr (equal_values<Axis, Basis.inner_axis>) {
        return pidx<PackSize>(index);
    } else {
        constexpr auto div = []<uZ I>(auto&& f, uZ_constant<I>) constexpr {
            constexpr auto axis = Basis.template axis<I>();
            if constexpr (equal_values<Axis, axis>) {
                return Basis.extent(axis);
            } else {
                return Basis.extent(axis) * f(f, uZ_constant<I - 1>{});
            }
        };
        constexpr uZ storage_size = detail_::storage_size<Basis, Alignment>();
        constexpr uZ stride       = storage_size / div(div, uZ_constant<Basis.size - 1>{});

        return stride * index;
    }
};
template<auto Basis, meta::any_value_sequence ExcludedAxes, auto Axis, uZ PackSize>
    requires /**/ (Basis.contains(Axis))
_NDINLINE_ constexpr auto get_dynamic_slice_offset(uZ                                storage_size,
                                                   const std::array<uZ, Basis.size>& extents,
                                                   uZ                                index) noexcept -> uZ {
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
        constexpr uZ stride = storage_size / div(div, extents, uZ_constant<Basis.size - 1>{});

        return stride * index;
    }
};

namespace static_ {
template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ PackSize, uZ Alignment>
class slice_base;
template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ PackSize, uZ Alignment>
class iter_base;
template<typename T, auto Basis, uZ PackSize, uZ Alignment>
class storage_base;
};    // namespace static_
namespace dynamic {
template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ PackSize>
class slice_base;
template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ PackSize>
class iter_base;
template<typename T, auto Basis, uZ PackSize, uZ Alignment, typename Allocator>
class storage_base;
};    // namespace dynamic
};    // namespace detail_

template<bool Const, auto Basis, typename T, uZ PackSize, typename Base>
class sslice;
template<typename T, auto Basis, uZ PackSize, uZ Agignment, typename Base>
class storage;

namespace detail_::static_ {
template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ PackSize, uZ Alignment>
class iter_base {
    static constexpr auto outer_axis = Basis.template outer_axis_remaining<ExcludedAxes>;

protected:
    explicit iter_base(auto) noexcept {};

    iter_base() noexcept                            = default;
    iter_base(const iter_base&) noexcept            = default;
    iter_base(iter_base&&) noexcept                 = default;
    iter_base& operator=(const iter_base&) noexcept = default;
    iter_base& operator=(iter_base&&) noexcept      = default;
    ~iter_base() noexcept                           = default;

    using excluded_axes = ExcludedAxes;

    using slice_base = static_::slice_base<Basis,    //
                                           meta::expand_value_sequence<ExcludedAxes, outer_axis>,
                                           PackSize,
                                           Alignment>;

    _NDINLINE_ static constexpr auto slice_base_args() noexcept {
        return 0;
    };

    _NDINLINE_ static constexpr auto stride() noexcept -> uZ {
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

        constexpr uZ inner_extent = Basis.extent(Basis.inner_axis);

        auto f = []<uZ I>(auto&& f, uZ_constant<I>) {
            constexpr auto axis = Basis.template axis<I>();
            if constexpr (I < Basis.size - 2) {
                return Basis.extent(axis) * f(f, uZ_constant<I + 1>{});
            } else {
                return Basis.extent(axis);
            }
        };

        uZ stride = (inner_extent * 2UL + Alignment - 1UL) / Alignment * Alignment;
        stride *= f(f, uZ_constant<0>{});
        stride /= d;
        return stride;
    }();
};
template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ PackSize, uZ Alignment>
class slice_base {
public:
    static constexpr bool vector_like = (ExcludedAxes::size == Basis.size - 1)    //
                                        && !meta::contains_value<ExcludedAxes, Basis.inner_axis>;

protected:
    explicit slice_base(auto) noexcept {};

    slice_base() noexcept                             = default;
    slice_base(const slice_base&) noexcept            = default;
    slice_base(slice_base&&) noexcept                 = default;
    slice_base& operator=(const slice_base&) noexcept = default;
    slice_base& operator=(slice_base&&) noexcept      = default;
    ~slice_base() noexcept                            = default;

    using excluded_axes = ExcludedAxes;

    static constexpr auto outer_axis = Basis.template outer_axis_remaining<ExcludedAxes>;

    using iterator_base = iter_base<Basis, ExcludedAxes, PackSize, Alignment>;

    _NDINLINE_ static constexpr auto iterator_base_args() noexcept {
        return 0;
    };

    template<auto Axis>
    using new_slice_base = slice_base<Basis,    //
                                      meta::expand_value_sequence<ExcludedAxes, Axis>,
                                      PackSize,
                                      Alignment>;

    _NDINLINE_ static constexpr auto new_slice_base_args() {
        return 0;
    };

    template<auto Axis>
    _NDINLINE_ static constexpr auto new_slice_offset(uZ index) noexcept {
        return get_static_slice_offset<Basis, ExcludedAxes, Axis, PackSize, Alignment>(index);
    }

    template<auto Axis>
    _NDINLINE_ static constexpr auto get_extent() noexcept -> uZ {
        return Basis.extent(Axis);
    }

private:
};
template<typename T, auto Basis, uZ PackSize, uZ Alignment>
class storage_base {
    static constexpr uZ alignment    = std::lcm(PackSize, Alignment);
    static constexpr uZ storage_size = detail_::storage_size<Basis, alignment>();

protected:
    storage_base() noexcept                               = default;
    storage_base(const storage_base&) noexcept            = default;
    storage_base(storage_base&&) noexcept                 = default;
    storage_base& operator=(const storage_base&) noexcept = default;
    storage_base& operator=(storage_base&&) noexcept      = default;
    ~storage_base() noexcept                              = default;

    static constexpr bool dynamic = false;

    using allocator = pcx::null_allocator<T>;

    using iterator_base = iter_base<Basis, meta::value_sequence<>, PackSize, Alignment>;

    _NDINLINE_ static constexpr auto iterator_base_args() noexcept {
        return 0;
    };

    template<auto Axis>
    using slice_base = slice_base<Basis, meta::value_sequence<Axis>, PackSize, Alignment>;

    _NDINLINE_ static constexpr auto slice_base_args() noexcept {
        return 0;
    };

    template<auto Axis>
    _NDINLINE_ static constexpr auto slice_offset(uZ index) noexcept -> uZ {
        return detail_::get_static_slice_offset<Basis, meta::value_sequence<>, Axis, Alignment, PackSize>(
            index);
    }

    _NDINLINE_ constexpr auto get_data() noexcept -> T* {
        return m_data.data();
    }

    _NDINLINE_ static constexpr auto get_size() noexcept -> uZ {
        return Basis.extent(Basis.outer_axis);
    }

private:
    std::array<T, storage_size> m_data{};
};
}    // namespace detail_::static_

namespace detail_::dynamic {
template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ PackSize>
class iter_base {
    static constexpr auto outer_axis = Basis.template outer_axis_remaining<ExcludedAxes>;

    using extents_type = detail_::dynamic_extents_info<Basis.size>;

protected:
    using excluded_axes = ExcludedAxes;

    using slice_base = dynamic::slice_base<Basis,    //
                                           meta::expand_value_sequence<ExcludedAxes, outer_axis>,
                                           PackSize>;

    _NDINLINE_ auto slice_base_args() const noexcept {
        return m_extents_ptr;
    };

    explicit iter_base(extents_type* extents_ptr) noexcept
    : m_stride(calc_stride(extents_ptr))
    , m_extents_ptr(extents_ptr){};

    iter_base() noexcept                            = default;
    iter_base(const iter_base&) noexcept            = default;
    iter_base(iter_base&&) noexcept                 = default;
    iter_base& operator=(const iter_base&) noexcept = default;
    iter_base& operator=(iter_base&&) noexcept      = default;
    ~iter_base() noexcept                           = default;

    _NDINLINE_ auto stride() const noexcept -> uZ {
        return m_stride;
    };

private:
    _NDINLINE_ static auto calc_stride(extents_type* extents_ptr) noexcept {
        auto stride = extents_ptr->storage_size;

        auto f = []<typename F, uZ I>(F&& f, auto& extents, uZ_constant<I>) {
            constexpr auto axis = Basis.template axis<I>();
            if constexpr (equal_values<outer_axis, axis>) {
                return extents[I];
            } else {
                return extents[I] * f(std::forward<F>(f), extents, uZ_constant<I - 1>{});
            }
        };
        auto div = f(f, extents_ptr->extents, uZ_constant<Basis.size() - 1>{});
        return stride / div;
    }

    uZ            m_stride{};
    extents_type* m_extents_ptr{};
};
template<auto Basis, meta::any_value_sequence ExcludedAxes, uZ PackSize>
class slice_base {
    using extents_type = detail_::dynamic_extents_info<Basis.size>;

public:
    static constexpr bool vector_like = (ExcludedAxes::size == Basis.size - 1)    //
                                        && !meta::contains_value<ExcludedAxes, Basis.inner_axis>;

protected:
    explicit slice_base(extents_type* extents_ptr) noexcept
    : m_extents_ptr(extents_ptr){};

    slice_base() noexcept                             = default;
    slice_base(const slice_base&) noexcept            = default;
    slice_base(slice_base&&) noexcept                 = default;
    slice_base& operator=(const slice_base&) noexcept = default;
    slice_base& operator=(slice_base&&) noexcept      = default;
    ~slice_base() noexcept                            = default;

    using excluded_axes = ExcludedAxes;

    static constexpr auto outer_axis = Basis.template outer_axis_remaining<ExcludedAxes>;

    using iterator_base = iter_base<Basis, ExcludedAxes, PackSize>;

    _NDINLINE_ auto iterator_base_args() const noexcept {
        return m_extents_ptr;
    }

    template<auto Axis>
    using new_slice_base = slice_base<Basis, meta::expand_value_sequence<ExcludedAxes, Axis>, PackSize>;

    _NDINLINE_ auto new_slice_base_args() noexcept {
        return m_extents_ptr;
    }

    template<auto Axis>
    _NDINLINE_ auto new_slice_offset(uZ index) noexcept -> uZ {
        const auto& storage_size = m_extents_ptr->storage_size;
        const auto& extents      = m_extents_ptr->extents;
        return detail_::get_dynamic_slice_offset<Basis, ExcludedAxes, Axis, PackSize>(
            storage_size, extents, index);
    }

    template<auto Axis>
    _NDINLINE_ auto get_extent() const noexcept -> uZ {
        return m_extents_ptr->extents[Basis.index_of(Axis)];
    }

private:
    extents_type* m_extents_ptr;
};
template<typename T, auto Basis, uZ PackSize, uZ Alignment, typename Allocator>
class storage_base {
    static constexpr uZ alignment = std::lcm(PackSize * 2, Alignment);

    using extents_type     = detail_::dynamic_extents_info<Basis.size>;
    using allocator_traits = std::allocator_traits<Allocator>;

    template<uZ... Is>
    explicit storage_base(std::index_sequence<Is...>, auto... extents)
    : m_extents{.extents{std::get<Is>(std::make_tuple(extents...))...}} {
        if constexpr (Basis.size > 1) {
            auto stride = calc_stride(m_extents.extents);
            auto size   = stride * m_extents.extents.back();
            m_ptr       = allocator_traits::allocate(m_allocator, size);
            std::memset(m_ptr, 0, size * sizeof(T));
        } else {
            auto size = (m_extents.extents.front() + alignment - 1) / alignment * alignment;
            m_ptr     = allocator_traits::allocate(m_allocator, size);
            std::memset(m_ptr, 0, size * sizeof(T));
        }
    }

    using axis_order    = std::index_sequence<Basis.size>;
    using reverse_order =                                   //
        meta::value_to_index_sequence<                      //
            meta::reverse_value_sequence<                   //
                meta::index_to_value_sequence<              //
                    std::make_index_sequence<Basis.size>    //
                    >>>;

protected:
    storage_base() noexcept
    : m_ptr(nullptr){};

    template<std::unsigned_integral... U>
    explicit storage_base(U... extents)
    : storage_base(axis_order{}, extents...){};

public:
    storage_base(const storage_base& other)            = delete;
    storage_base& operator=(const storage_base& other) = delete;

protected:
    storage_base(storage_base&& other) noexcept
    : m_allocator(std::move(other.m_allocator))
    , m_ptr(other.m_ptr)
    , m_extents(other.m_extents) {
        other.m_ptr     = nullptr;
        other.m_extents = {};
    };

    storage_base& operator=(storage_base&& other) noexcept(
        allocator_traits::propagate_on_container_move_assignment::value ||
        allocator_traits::is_always_equal::value) {
        using std::swap;
        if constexpr (allocator_traits::propagate_on_container_move_assignment::value) {
            if (allocator_traits::is_always_equal::value || m_allocator == other.m_allocator) {
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
            swap(m_extents, other.m_extents);
        } else {
            if (m_allocator == other.m_allocator) {
                swap(m_ptr, other.m_ptr);
                swap(m_extents, other.m_extents);
            } else {
                auto size = other.storage_size();
                if (storage_size() != size) {
                    deallocate();
                    m_ptr = allocator_traits::allocate(m_allocator, size);
                }
                m_extents = other.m_extents;
                std::memcpy(m_ptr, other.m_ptr, size * sizeof(T));
            }
        }
    }

    ~storage_base() noexcept {
        deallocate();
    }

    static constexpr bool dynamic = true;

    using allocator = Allocator;

    using iterator_base = iter_base<Basis, meta::value_sequence<>, PackSize>;

    _NDINLINE_ auto iterator_base_args() noexcept {
        return &m_extents;
    }

    template<auto Axis>
    using slice_base = slice_base<Basis, meta::value_sequence<>, PackSize>;

    _NDINLINE_ auto slice_base_args() noexcept {
        return &m_extents;
    }

    template<auto Axis>
    _NDINLINE_ auto slice_offset(uZ offset) const noexcept -> uZ {
        return get_dynamic_slice_offset<Basis, meta::value_sequence<>, Axis, PackSize>(
            m_extents.storage_size, m_extents.extents, offset);
    }

    auto get_data() const noexcept -> T* {
        return m_ptr;
    }

    _NDINLINE_ auto get_size() const noexcept -> uZ {
        return m_extents.extents.back();
    }

private:
    _NDINLINE_ static auto calc_storage_size(const std::array<uZ, Basis.size>& extents) noexcept -> uZ {
        auto size = (extents.front() + alignment - 1) / alignment * alignment;
        for (uZ i = 1; i < Basis.size; ++i) {
            size *= extents[i];
        }
    }
    _NDINLINE_ auto calc_stride(const std::array<uZ, Basis.size>& extents) const noexcept -> uZ {
        auto inner_storage_size = (extents.front() + alignment - 1) / alignment * alignment;

        auto stride = inner_storage_size;
        for (uZ i = 1; i < Basis.size - 1; ++i) {
            stride *= extents[i];
        }
        return stride;
    }

    _NDINLINE_ auto storage_size() const noexcept -> uZ {
        return m_extents.storage_size;
    }

    _INLINE_ void deallocate() noexcept {
        allocator_traits::deallocate(m_allocator, m_ptr, storage_size());
    }

    [[no_unique_address]] Allocator m_allocator{};
    T*                              m_ptr;
    extents_type                    m_extents{};
};
}    // namespace detail_::dynamic


template<bool Const, auto Basis, typename T, uZ PackSize, typename Base>
class iterator : Base {
    using pointer = T*;

    template<bool, auto, typename, uZ, typename>
    friend class sslice;

    template<typename, auto, uZ, uZ, typename>
    friend class storage;

    template<typename... U>
    explicit iterator(pointer ptr, U&&... other) noexcept
    : Base(std::forward<U>(other)...)
    , m_ptr(ptr){};

    using sslice = md::sslice<Const, Basis, T, PackSize, typename Base::slice_base>;

public:
    iterator() noexcept                           = default;
    iterator(const iterator&) noexcept            = default;
    iterator(iterator&&) noexcept                 = default;
    iterator& operator=(const iterator&) noexcept = default;
    iterator& operator=(iterator&&) noexcept      = default;
    ~iterator() noexcept                          = default;

    using value_type       = sslice;
    using iterator_concept = std::random_access_iterator_tag;
    using difference_type  = iZ;

    [[nodiscard]] auto operator*() const noexcept {
        return sslice(m_ptr, Base::slice_base_args());
    }
    [[nodiscard]] auto operator[](uZ index) const noexcept {
        return sslice(m_ptr + Base::stride() * index, Base::slice_base_args());
    }

    [[nodiscard]] auto operator+=(difference_type n) noexcept -> iterator& {
        m_ptr += Base::stride() * n;
        return *this;
    };
    [[nodiscard]] auto operator-=(difference_type n) noexcept -> iterator& {
        m_ptr -= Base::stride() * n;
        return *this;
    };

    [[nodiscard]] auto operator++() noexcept -> iterator& {
        m_ptr += Base::stride();
        return *this;
    };
    [[nodiscard]] auto operator++(int) noexcept -> iterator {
        auto copy = *this;
        m_ptr += Base::stride();
        return copy;
    };
    [[nodiscard]] auto operator--() noexcept -> iterator& {
        m_ptr -= Base::stride();
        return *this;
    };
    [[nodiscard]] auto operator--(int) noexcept -> iterator {
        auto copy = *this;
        m_ptr -= Base::stride();
        return copy;
    };

    [[nodiscard]] friend auto operator+(const iterator& lhs, difference_type rhs) noexcept -> iterator {
        return iterator{lhs.m_ptr + rhs * lhs.stride(), static_cast<const Base&>(lhs)};
    };
    [[nodiscard]] friend auto operator+(difference_type lhs, const iterator& rhs) noexcept -> iterator {
        return rhs + lhs;
    };

    [[nodiscard]] friend auto operator-(const iterator& lhs, difference_type rhs) noexcept -> iterator {
        return lhs + (-rhs);
    };

    [[nodiscard]] friend auto operator-(const iterator& lhs, const iterator& rhs) noexcept
        -> difference_type {
        return (lhs.m_ptr - rhs.m_ptr) / lhs.stride();
    };

    [[nodiscard]] auto operator<=>(const iterator& other) const noexcept {
        return m_ptr <=> other.m_ptr;
    };

    [[nodiscard]] auto operator==(const iterator& other) const noexcept {
        return m_ptr == other.m_ptr;
    };

private:
    pointer m_ptr{};
};

template<bool Const, auto Basis, typename T, uZ PackSize, typename Base>
class sslice
: public std::ranges::view_base
, pcx::detail_::pack_aligned_base<Base::vector_like>
, Base {
    static constexpr bool contigious = !meta::contains_value<typename Base::excluded_axes, Basis.inner_axis>;

private:
    template<bool, auto, typename, uZ, typename>
    friend class md::iterator;

    template<typename, auto, uZ, uZ, typename>
    friend class storage;

    using iterator =
        std::conditional_t<Base::vector_like,
                           pcx::iterator<T, Const, PackSize>,
                           md::iterator<Const, Basis, T, PackSize, typename Base::iterator_base>>;

    template<bool, auto, typename, uZ, typename>
    friend class sslice;

    template<typename... U>
    explicit constexpr sslice(T* start, U&&... args)
    : Base(std::forward<U>(args)...)
    , m_start(start){};

public:
    sslice()
    : Base(){};

    sslice(const sslice&) noexcept            = default;
    sslice(sslice&&) noexcept                 = default;
    sslice& operator=(const sslice&) noexcept = default;
    sslice& operator=(sslice&&) noexcept      = default;
    ~sslice()                                 = default;

    template<auto Axis>
        requires /**/ (Basis.contains(Axis))
    [[nodiscard]] auto slice(uZ index) const noexcept {
        auto* new_start = m_start + Base::template new_slice_offset<Axis>(index);
        if constexpr (Basis.size - Base::excluded_axes::size == 1) {
            return pcx::detail_::make_cx_ref<T, Const, PackSize>(new_start);
        } else {
            using new_base  = typename Base::template new_slice_base<Axis>;
            using new_slice = sslice<Const, Basis, T, PackSize, new_base>;
            return new_slice(new_start, Base::new_slice_base_args());
        }
    }

    [[nodiscard]] auto operator[](uZ index) const noexcept {
        return slice<Basis.outer_axis>(index);
    }

    // TODO: `at()`
    // TODO: multidimensional slice and operator[]

    [[nodiscard]] auto begin() const noexcept -> iterator {
        if constexpr (Base::vector_like) {
            return pcx::detail_::make_iterator<T, Const, PackSize>(m_start, Base::iterator_base_args());
        } else {
            return iterator(m_start, Base::iterator_base_args());
        }
    }

    [[nodiscard]] auto end() const noexcept {
        return begin() += size();
    }

    [[nodiscard]] auto size() const noexcept -> uZ {
        return extent<Base::outer_axis>();
    }

    template<auto Axis>
    [[nodiscard]] auto extent() const noexcept -> uZ {
        return Base::template get_extent<Axis>();
    }

private:
    T* m_start{};
};

template<typename T, auto Basis, uZ PackSize, uZ Alignment, typename Base>
class storage
: Base
, pcx::detail_::pack_aligned_base<Basis.size == 1> {
    static constexpr bool vector_like = Basis.size == 1;
    static constexpr bool dynamic     = Base::dynamic;

    using iterator =
        std::conditional_t<vector_like,
                           pcx::iterator<T, false, PackSize>,
                           md::iterator<false, Basis, T, PackSize, typename Base::iterator_base>>;
    using const_iterator =
        std::conditional_t<vector_like,
                           pcx::iterator<T, true, PackSize>,
                           md::iterator<true, Basis, T, PackSize, typename Base::iterator_base>>;

public:
    template<std::unsigned_integral... U>
    explicit storage(typename Base::allocator allocator, U... extents)
        requires(dynamic && sizeof...(U) == Basis.size)
    {}
    template<std::unsigned_integral... U>
    explicit storage(U... extents)
        requires(dynamic && sizeof...(U) == Basis.size)
    {}

    storage() = default;

    [[nodiscard]] auto begin() noexcept -> iterator {
        if constexpr (vector_like) {
            return pcx::detail_::make_iterator<T, false, PackSize>(data(), 0);
        } else {
            return iterator(data(), Base::iterator_base_args());
        }
    }
    [[nodiscard]] auto begin() const noexcept -> const_iterator {
        return cbegin();
    }
    [[nodiscard]] auto cbegin() const noexcept -> const_iterator {
        if constexpr (vector_like) {
            return pcx::detail_::make_iterator<T, true, PackSize>(data(), 0);
        } else {
            return const_iterator(data(), Base::iterator_base_args());
        }
    }
    [[nodiscard]] auto end() noexcept -> iterator {
        return begin() + size();
    }
    [[nodiscard]] auto end() const noexcept -> const_iterator {
        return cend();
    }
    [[nodiscard]] auto cend() const noexcept -> const_iterator {
        return cbegin() + size();
    }

    template<auto Axis>
    [[nodiscard]] auto slice(uZ index) noexcept {
        if constexpr (vector_like) {
            return pcx::detail_::make_cx_ref<T, false, PackSize>(data() + pcx::pidx<PackSize>(index));
        } else {
            using slice_base = typename Base::template slice_base<Axis>;
            using slice_type = sslice<false, Basis, T, PackSize, slice_base>;

            auto* slice_start = data() + Base::template slice_offset<Axis>(index);
            return slice_type(slice_start, Base::slice_base_args());
        }
    }
    template<auto Axis>
    [[nodiscard]] auto slice(uZ index) const noexcept {
        return cslice<Axis>(index);
    }
    template<auto Axis>
    [[nodiscard]] auto cslice(uZ index) const noexcept {
        if constexpr (vector_like) {
            return pcx::detail_::make_cx_ref<T, true, PackSize>(data() + pcx::pidx<PackSize>(index));
        } else {
            using slice_base = typename Base::template slice_base<Axis>;
            using slice_type = sslice<true, Basis, T, PackSize, slice_base>;

            auto* slice_start = data() + Base::template slice_offset<Axis>(index);
            return slice_type(slice_start, Base::slice_base_args());
        }
    }

    [[nodiscard]] auto operator[](uZ index) noexcept {
        return slice<Basis.outer_axis>(index);
    }
    [[nodiscard]] auto operator[](uZ index) const noexcept {
        return slice<Basis.outer_axis>(index);
    }

    [[nodiscard]] auto data() noexcept -> T* {
        return Base::get_data();
    }
    [[nodiscard]] auto data() const noexcept -> const T* {
        return Base::get_data();
    }
    [[nodiscard]] auto size() const noexcept -> uZ {
        return Base::get_size();
    }
};

template<typename T, auto Basis, uZ PackSize, uZ Alignment>
using static_stoarge = storage<T,    //
                               Basis,
                               PackSize,
                               Alignment,
                               detail_::static_::storage_base<T, Basis, PackSize, Alignment>>;

template<typename T, auto Basis, uZ PackSize, uZ Alignment, typename Allocator = pcx::aligned_allocator<T>>
using d_stoarge = storage<T,    //
                          Basis,
                          PackSize,
                          Alignment,
                          detail_::dynamic::storage_base<T, Basis, PackSize, Alignment, Allocator>>;
}    // namespace pcx::md
#undef _INLINE_
#undef _NDINLINE_
#endif