#ifndef STATIC_dynamic_storage_base_HPP
#define STATIC_dynamic_storage_base_HPP

#include "allocators.hpp"
#include "element_access.hpp"
#include "meta.hpp"
#include "types.hpp"

#include <algorithm>
#include <array>
#include <bits/utility.h>
#include <cstring>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>

namespace pcx::md {

namespace detail_ {
struct basis_base {};
}    // namespace detail_

template<auto... Axes>
class static_basis : public detail_::basis_base {
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

    template<uZ... Is>
    constexpr static_basis(auto&& extents, std::index_sequence<Is...>)
    : extents{extents[Is]...} {};

public:
    static constexpr uZ size = sizeof...(Axes);

    template<std::unsigned_integral... Us>
        requires /**/ (sizeof...(Us) == size)
    constexpr explicit static_basis(Us... extents) noexcept
    : extents{extents...} {};

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

        return new_basis_t(extents, filtered{});
    };

    template<auto Axis>
        requires meta::value_matched<Axis, Axes...>
    [[nodiscard]] constexpr auto extent() const noexcept -> uZ {
        constexpr uZ index = meta::find_first_in_values<Axis, Axes...>;
        return extents[index];
    }

    template<meta::any_value_sequence Excluded>
    static constexpr auto outer_axis_remaining = outer_remaining_impl<size - 1, Excluded>::value;

    template<meta::any_value_sequence Excluded>
    static constexpr auto inner_axis_remaining = outer_remaining_impl<0, Excluded>::value;

    std::array<uZ, size> extents;

private:
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

    constexpr auto f = []<uZ I>(auto&& f, uZ size, uZ_constant<I>) {
        constexpr auto axis = Basis.template axis<I>();
        if constexpr (Basis.size == 1) {
            return size;
        } else if constexpr (I == Basis.size - 1) {
            return size *= Basis.template extent<axis>();
        } else {
            return f(f, size *= Basis.template extent<axis>(), uZ_constant<I + 1>{});
        }
    };
    size = f(f, size, uZ_constant<1>{});
    return size;
}
template<auto Basis, meta::any_value_sequence ExcludedAxes, auto Axis, uZ Alignment, uZ PackSize>
    requires /**/ (Basis.template contains<Axis>())
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

    using slice_base = slice_base<Basis,    //
                                  meta::expand_value_sequence<ExcludedAxes, outer_axis>,
                                  PackSize,
                                  Alignment>;

    static constexpr auto slice_base_args() noexcept {
        return 0;
    };

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

        auto f = []<uZ I>(auto&& f, uZ_constant<I>) {
            constexpr auto axis = Basis.template axis<I>();
            if constexpr (I < Basis.size - 2) {
                return Basis.template extent<axis>() * f(f, uZ_constant<I + 1>{});
            } else {
                return Basis.template extent<axis>();
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

    static constexpr auto iterator_base_args() noexcept {
        return 0;
    };

    template<auto Axis>
    using new_slice_base = slice_base<Basis,    //
                                      meta::expand_value_sequence<ExcludedAxes, Axis>,
                                      PackSize,
                                      Alignment>;

    static constexpr auto new_slice_base_args() {
        return 0;
    };

    template<auto Axis>
    static constexpr auto new_slice_offset(uZ index) noexcept {
        return get_static_slice_offset<Basis, ExcludedAxes, Axis, PackSize, Alignment>(index);
    }

    template<auto Axis>
    static constexpr auto get_extent() noexcept -> uZ {
        return Basis.template extent<Axis>();
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

    using iterator_base = iter_base<Basis, meta::value_sequence<>, PackSize, Alignment>;

    static constexpr auto iterator_base_args() noexcept {
        return 0;
    };

    template<auto Axis>
    using slice_base = slice_base<Basis, meta::value_sequence<Axis>, PackSize, Alignment>;

    static constexpr auto slice_base_args() noexcept {
        return 0;
    };

    template<auto Axis>
    static constexpr auto slice_offset(uZ index) noexcept -> uZ {
        return detail_::get_static_slice_offset<Basis, meta::value_sequence<>, Axis, Alignment, PackSize>(
            index);
    }

    [[nodiscard]] constexpr auto get_data() noexcept -> T* {
        return m_data.data();
    }

    [[nodiscard]] static constexpr auto get_size() noexcept -> uZ {
        return Basis.template extent<Basis.outer_axis>();
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

    using slice_base = slice_base<Basis, meta::expand_value_sequence<ExcludedAxes, outer_axis>, PackSize>;

    [[nodiscard]] auto slice_base_args() const noexcept {
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

    [[nodiscard]] auto stride() const noexcept -> uZ {
        return m_stride;
    };

private:
    [[nodiscard]] static auto calc_stride(extents_type* extents_ptr) noexcept {
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

    void iterator_base_args() const noexcept {}

    template<auto Axis>
    using new_slice_base = slice_base<Basis, meta::expand_value_sequence<ExcludedAxes, Axis>, PackSize>;

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
template<typename T, auto Basis, uZ PackSize, uZ Alignment, typename Allocator>
class storage_base {
    using extents_type     = detail_::dynamic_extents_info<Basis.size>;
    using allocator_traits = std::allocator_traits<Allocator>;

protected:
    storage_base() noexcept
    : m_ptr(nullptr){};

    explicit storage_base(const std::array<uZ, Basis.size>& extents, Allocator allocator = {})
    : m_extents{.stride = 0, .extents = extents}
    , m_allocator(allocator) {
        uZ align    = std::lcm(Alignment, PackSize * 2);
        uZ misalign = (extents[0] * 2) % align;
        uZ size     = extents[0] * 2 + (misalign > 0 ? align - misalign : 0);
        if constexpr (Basis.size > 1) {
            uZ stride = size;
            for (uZ i = 1; i < Basis.size - 1; ++i) {
                stride *= m_extents.extents[i];
            }
            size             = stride * m_extents.extents.back();
            m_extents.stride = stride;
        }
        m_ptr = allocator_traits::allocate(m_allocator, size);
        std::memset(m_ptr, 0, size * sizeof(T));
    };

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
                auto size = other.stride() * other.get_size();
                if (size != stride() * m_extents.get_size()) {
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

    using iterator_base = iter_base<Basis, meta::value_sequence<>, PackSize>;

    auto iterator_base_args() noexcept {
        return &m_extents;
    }

    template<auto Axis>
    using slice_base = slice_base<Basis, meta::value_sequence<>, PackSize>;

    auto slice_base_args() noexcept {
        return &m_extents;
    }

    template<auto Axis>
    [[nodiscard]] auto slice_offset(uZ offset) const noexcept -> uZ {
        return get_dynamic_slice_offset<Basis, meta::value_sequence<>, Axis, PackSize>(
            m_extents.stride, m_extents.extents, offset);
    }
    auto get_data() const noexcept -> T* {
        return m_ptr;
    }

    [[nodiscard]] auto get_size() const noexcept -> uZ {
        return m_extents.extents.back();
    }

private:
    [[nodiscard]] auto stride() const noexcept -> uZ {
        return m_extents.stride;
    }

    void deallocate() noexcept {
        allocator_traits::deallocate(m_allocator, m_ptr, stride() * get_size());
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

    using sslice = sslice<Const, Basis, T, PackSize, typename Base::slice_base>;

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

    [[nodiscard]] inline auto operator*() const noexcept {
        return sslice(m_ptr, Base::slice_base_args());
    }
    [[nodiscard]] auto operator[](uZ index) const noexcept {
        return sslice(m_ptr + Base::stride() * index, Base::slice_base_args());
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
        return iterator{lhs.m_ptr + rhs * lhs.stride(), static_cast<const Base&>(lhs)};
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

template<bool Const, auto Basis, typename T, uZ PackSize, typename Base>
class sslice
: public std::ranges::view_base
, pcx::detail_::pack_aligned_base<Base::vector_like>
, Base {
    static constexpr bool contigious = !meta::contains_value<typename Base::excluded_axes, Basis.inner_axis>;

private:
    template<bool, auto, typename, uZ, typename>
    friend class iterator;

    template<typename, auto, uZ, uZ, typename>
    friend class storage;

    using iterator = std::conditional_t<Base::vector_like,
                                        pcx::iterator<T, Const, PackSize>,
                                        iterator<Const, Basis, T, PackSize, typename Base::iterator_base>>;

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
        requires /**/ (Basis.template contains<Axis>())
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

template<typename T, auto Basis, uZ PackSize, uZ Agignment, typename Base>
class storage
: Base
, pcx::detail_::pack_aligned_base<Basis.size == 1> {
    static constexpr bool vector_like = Basis.size == 1;
    using iterator =
        std::conditional_t<vector_like,
                           pcx::iterator<T, false, PackSize>,
                           md::iterator<false, Basis, T, PackSize, typename Base::iterator_base>>;
    using const_iterator =
        std::conditional_t<vector_like,
                           pcx::iterator<T, true, PackSize>,
                           md::iterator<true, Basis, T, PackSize, typename Base::iterator_base>>;
    ;

public:
    template<typename... U>
    explicit storage(U&&... args)
    : Base(std::forward<U>(args)...){};

    [[nodiscard]] auto begin() noexcept -> iterator {
        if constexpr (vector_like) {
            return iterator(data(), 0);
        } else {
            return iterator(data(), Base::iterator_base_args());
        }
    }
    [[nodiscard]] auto begin() const noexcept -> const_iterator {
        return cbegin();
    }
    [[nodiscard]] auto cbegin() const noexcept -> const_iterator {
        if constexpr (vector_like) {
            return const_iterator(data(), 0);
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

}    // namespace pcx::md
#endif