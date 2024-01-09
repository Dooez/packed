#ifndef MDVECTOR_HPP
#define MDVECTOR_HPP

#include "types.hpp"
#include "vector_util.hpp"

#include <concepts>
#include <cstddef>
#include <cstring>
#include <memory>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <utility>

namespace pcx {

namespace detail_ {
template<auto... Vs>
struct equal_impl {
    static constexpr bool value = false;
};
template<auto V, auto... Vs>
    requires(std::equality_comparable_with<decltype(V), decltype(Vs)> && ...)
struct equal_impl<V, Vs...> {
    static constexpr bool value = ((V == Vs) && ...) && equal_impl<Vs...>::value;
};
template<auto V>
struct equal_impl<V> {
    static constexpr bool value = true;
};
}    // namespace detail_

/**
 * @brief Checks if all template parameter values are equality comparible and equal.
 * 
 * @tparam Vs... Values to check the equality.
 */
template<auto... Vs>
concept equal_values = detail_::equal_impl<Vs...>::value;

namespace detail_ {
template<auto... Vs>
struct unique_values_impl {
    static constexpr bool value = true;
};
template<auto V, auto... Vs>
struct unique_values_impl<V, Vs...> {
    static constexpr bool value = (!equal_values<V, Vs> && ...) && unique_values_impl<Vs...>::value;
};
}    // namespace detail_

/**
 * @brief Checks if template parameters Vs... do not contain repeating values.
 * 
 * @tparam Vs...
 */
template<auto... Vs>
concept unique_values = detail_::unique_values_impl<Vs...>::value;

/**
 * @brief Checks if a value V matches any of the values in Vs...
 * 
 * @tparam V 
 * @tparam Vs...
 */
template<auto V, auto... Vs>
concept value_matched = (!unique_values<V, Vs...>);

namespace detail_ {
struct basis_base {};

template<auto... Values>
struct value_sequence {};

template<typename Sequence, auto... NewValues>
struct expand_value_sequence_impl;
template<auto... Values, auto... NewValues>
struct expand_value_sequence_impl<value_sequence<Values...>, NewValues...> {
    using type = value_sequence<Values..., NewValues...>;
};

template<typename Sequence, auto... NewValues>
using expand_value_sequence = typename expand_value_sequence_impl<Sequence, NewValues...>::type;

template<typename S, auto Vfilter, auto... Vs>
struct filter_value_sequence_impl {};

template<typename S, auto Vfilter, auto V, auto... Vs>
struct filter_value_sequence_impl<S, Vfilter, V, Vs...> {
    using type = std::conditional_t<
        equal_values<Vfilter, V>,
        expand_value_sequence<S, Vs...>,
        typename filter_value_sequence_impl<expand_value_sequence<S, V>, Vfilter, Vs...>::type>;
};
template<typename S, auto Vfilter>
struct filter_value_sequence_impl<S, Vfilter> {
    using type = S;
};
template<typename S, auto Vfilter>
struct filter_value_sequence_adapter {};
template<auto... Vs, auto Vfilter>
struct filter_value_sequence_adapter<value_sequence<Vs...>, Vfilter> {
    using type = typename filter_value_sequence_impl<value_sequence<>, Vfilter, Vs...>::type;
};

template<typename Sequence, auto FilterValue>
using filter_value_sequence = typename filter_value_sequence_adapter<Sequence, FilterValue>::type;

}    // namespace detail_

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

template<typename T,
         md_basis Basis,
         uZ       PackSize  = default_pack_size<T>,
         typename Allocator = aligned_allocator<T>>
    requires pack_size<PackSize>
class mdarray {
    using allocator_traits = std::allocator_traits<Allocator>;

public:
    /**
     * @brief Construct a new mdarray object.
     * 
     * @param extents   Extents of individual axes. `extents[i]` corresponds to `Basis::axis<i>`.
     * @param alignment Contigious axis storage is padded to multiple of least common multiple `alignment` or PackSize.
     * @param allocator 
     */
    explicit mdarray(std::array<uZ, Basis::size> extents, uZ alignment = 1, Allocator allocator = {})
    : m_extents(extents)
    , m_allocator(allocator) {
        uZ align    = std::lcm(alignment, PackSize);
        uZ misalign = extents[0] % align;
        uZ size     = extents[0] + misalign > 0 ? align - misalign : 0;
        uZ stride   = size;
        for (uZ i = 1; i < Basis::size - 1; ++i) {
            stride *= m_extents[i];
        }
        size     = stride * m_extents.back();
        m_stride = stride;
        m_ptr    = allocator_traits::allocate(allocator, size);
        std::memset(m_ptr, 0, size * sizeof(T));
    };
    explicit mdarray() = default;

    mdarray(const mdarray& other)            = delete;
    mdarray& operator=(const mdarray& other) = delete;

    mdarray(mdarray&& other) noexcept
    : m_allocator(std::move(other.m_allocator))
    , m_ptr(other.m_ptr)
    , m_stride(other.m_stride)
    , m_extents(other.m_extents) {
        other.m_ptr     = nullptr;
        other.m_stride  = 0;
        other.m_extents = {};
    };

    mdarray& operator=(mdarray&& other) noexcept(allocator_traits::propagate_on_move_assignment::value ||
                                                 allocator_traits::is_always_equal::value) {
        using std::swap;
        if constexpr (allocator_traits::propagate_on_move_assignment::value) {
            deallocate();
            m_allocator     = std::move(other.m_allocator);
            m_ptr           = other.m_ptr;
            m_stride        = other.m_stride;
            m_extents       = other.m_extents;
            other.m_ptr     = nullptr;
            other.m_stride  = 0;
            other.m_extents = {};
        } else if constexpr (allocator_traits::is_always_equal::value) {
            swap(m_ptr, other.m_ptr);
            swap(m_stride, other.n_stride);
            swap(m_extents, other.m_extents);
        } else {
            if (m_allocator == other.m_allocator) {
                swap(m_ptr, other.m_ptr);
                swap(m_stride, other.n_stride);
                swap(m_extents, other.m_extents);
            } else {
                auto size = other.m_stride * other.m_extents.back();
                if (size != m_stride * m_extents.back()) {
                    deallocate();
                    m_ptr = allocator_traits::allocate(m_allocator, size);
                }
                m_stride  = other.m_stride;
                m_extents = other.m_extents;
                std::memcpy(m_ptr, other.m_ptr, size * sizeof(T));
            }
        }
    }

    ~mdarray() noexcept {
        deallocate();
    }

    template<auto... Axis, typename... Tidx>
        requires(sizeof...(Axis) == sizeof...(Tidx) && (std::same_as<Tidx, uZ> && ...))
    auto slice(Tidx... indexes){

    };

private:
    [[no_unique_address]] Allocator m_allocator{};
    T*                              m_ptr{};
    uZ                              m_stride{};
    std::array<uZ, Basis::size>     m_extents{};

    inline void deallocate() noexcept {
        if (m_ptr != nullptr) {
            allocator_traits::deallocate(m_ptr, m_stride * m_extents.back());
        }
    };
};

template<typename T, md_basis Basis, bool Contigious, bool Const>
class mdslice : public std::ranges::view_base {
    using extent_t = std::array<uZ, Basis::size>;

    template<typename S>
    struct value_to_index_sequence_impl {};
    template<uZ... Is>
    struct value_to_index_sequence_impl<detail_::value_sequence<Is...>> {
        using type = std::index_sequence<Is...>;
    };
    template<typename S>
    struct index_to_value_sequence_impl {};
    template<uZ... Is>
    struct index_to_value_sequence_impl<std::index_sequence<Is...>> {
        using type = detail_::value_sequence<Is...>;
    };
    template<typename S>
    using value_to_index_sequence = typename value_to_index_sequence_impl<S>::type;
    template<typename S>
    using index_to_value_sequence = typename index_to_value_sequence_impl<S>::type;

    using contigious_size_t = std::conditional_t<Contigious, uZ, decltype([] {})>;

public:
    /**
     * @brief Construct a new mdslice object
     * 
     * @param start First
     * @param stride 
     * @param extents Sizes of axis. 
     */
    mdslice(T* start, uZ stride, extent_t extents)
    : m_start(start)
    , m_stride(stride)
    , m_extents(extents){};

    auto operator[](uZ index) {
        if constexpr (Basis::size == 1) {
            return *(m_start + m_stride * index);
        } else {
            constexpr auto axis = Basis::template axis<Basis::size - 1>;
            return slice_impl<axis>(index);
        }
    };

    auto operator++() {
        m_start += m_stride;
    }
    auto operator*() {
        constexpr auto axis = Basis::template axis<Basis::size - 1>;
        return slice_impl<axis>(0);
    }


    template<auto Axis>
    inline auto slice(uZ index) {
        return slice_impl<Axis>(index);
    }

    template<auto Axis>
    inline auto slice_impl(uZ index) {
        using new_basis               = typename Basis::template exclude<Axis>;
        constexpr auto axis_index     = uZ_constant<Basis::template index<Axis>>{};
        constexpr bool new_contigious = Contigious && Basis::template index<Axis> != 0;

        bool valid_index = index < m_extents[axis_index];

        constexpr auto get_offset = [axis_index](auto&& extents, uZ index, uZ stride) {
            if constexpr (Contigious && axis_index == 0) {
                return packed_offset(index);
            } else {
                for (iZ i = Basis::size - 1; i > axis_index; --i) {
                    stride /= extents[i];
                }
                return stride * index;
            }
        };

        auto new_start   = m_start + get_offset(m_extents, index, m_stride);
        auto new_stride  = (axis_index == (Basis::size - 1)) ? m_stride / m_extents.back() : m_stride;
        auto new_extents = []<uZ I>(auto&& extents, uZ_constant<I>) {
            using indexes  = index_to_value_sequence<decltype(std::make_index_sequence<Basis::size>{})>;
            using filtered = value_to_index_sequence<detail_::filter_value_sequence<indexes, I>>;
            return []<uZ... Is>(auto&& extents, std::index_sequence<Is...>) {
                return std::array{extents[Is]...};
            }(extents, filtered{});
        }(m_extents, axis_index);

        return mdslice<T, new_basis, new_contigious, Const>(new_start, new_stride, new_extents);
    };


private:
    T*                          m_start;
    uZ                          m_stride;
    std::array<uZ, Basis::size> m_extents;

    static auto packed_offset(uZ offset) {
        //TODO: actually calculate index
        return offset * 2;
    };
};

template<typename S>
struct value_to_index_sequence_impl {};
template<uZ... Is>
struct value_to_index_sequence_impl<detail_::value_sequence<Is...>> {
    using type = std::index_sequence<Is...>;
};
template<typename S>
struct index_to_value_sequence_impl {};
template<uZ... Is>
struct index_to_value_sequence_impl<std::index_sequence<Is...>> {
    using type = detail_::value_sequence<Is...>;
};
template<typename S>
using value_to_index_sequence = typename value_to_index_sequence_impl<S>::type;
template<typename S>
using index_to_value_sequence = typename index_to_value_sequence_impl<S>::type;

template<typename T, md_basis Basis, auto Axis, bool Contigious, bool Const>
inline auto generic_slice_impl(T* start, uZ stride, uZ index, const std::array<uZ, Basis::size>& extents) {
    using new_basis = typename Basis::template exclude<Axis>;
    if constexpr (new_basis::size == 1) {}

    constexpr auto axis_index     = uZ_constant<Basis::template index<Axis>>{};
    constexpr bool new_contigious = Contigious && Basis::template index<Axis> != 0;
    constexpr auto packed_offset  = [](uZ offset) { return offset * 2; };

    constexpr auto get_offset = [axis_index](auto&& extents, uZ index, uZ stride) {
        if constexpr (Contigious && axis_index == 0) {
            return packed_offset(index);
        } else {
            for (iZ i = Basis::size - 1; i > axis_index; --i) {
                stride /= extents[i];
            }
            return stride * index;
        }
    };

    auto new_start   = start + get_offset(extents, index, stride);
    auto new_stride  = (axis_index == (Basis::size - 1)) ? stride / extents.back() : stride;
    auto new_extents = []<uZ I>(auto& extents, uZ_constant<I>) {
        using indexes  = index_to_value_sequence<decltype(std::make_index_sequence<Basis::size>{})>;
        using filtered = value_to_index_sequence<detail_::filter_value_sequence<indexes, I>>;
        return []<uZ... Is>(auto& extents, std::index_sequence<Is...>) {
            return std::array{extents[Is]...};
        }(extents, filtered{});
    }(extents, axis_index);

    return mdslice<T, new_basis, new_contigious, Const>(new_start, new_stride, new_extents);
};

template<typename T, md_basis Basis, uZ PackSize, bool Const, bool Contigious>
class md_iterator {
public:
    auto operator*() {
        if constexpr (Basis::size == 1) {
            return cx_ref<T, Const, PackSize>(m_ptr);
        } else {
            constexpr auto axis = Basis::template axis<Basis::size - 1>;
            return generic_slice_impl<T, Basis, axis, Contigious, Const>(m_ptr, m_stride, 0, m_extents);
        }
    }

private:
    T*                          m_ptr{};
    uZ                          m_stride{};
    std::array<uZ, Basis::size> m_extents{};
};

}    // namespace pcx
#endif