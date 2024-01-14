#ifndef MDVECTOR_HPP
#define MDVECTOR_HPP

#include "meta.hpp"
#include "types.hpp"
#include "vector.hpp"
#include "vector_util.hpp"

#include <cstring>
#include <iterator>
#include <memory>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace pcx {

namespace detail_ {
struct basis_base {};
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

namespace detail_ {

template<typename T, uZ PackSize, md_basis Basis, bool Const, bool Contigious>
auto md_get_iterator(T* start, uZ stride, uZ index, const std::array<uZ, Basis::size>& extents) noexcept;

template<typename T,
         uZ       PackSize,
         md_basis Basis,
         auto     ExcludeAxis,
         bool     Const,
         bool     Contigious,
         uZ       ExtentsSize>
    requires(ExtentsSize == Basis::size ||
             equal_values<ExcludeAxis, Basis::outer_axis> && ExtentsSize == Basis::size - 1)
auto md_get_slice(T* start, uZ stride, uZ index, const std::array<uZ, ExtentsSize>& extents) noexcept;

}    // namespace detail_

template<typename T, md_basis Basis, uZ PackSize, bool Const, bool Contigious>
class mditerator;

template<typename T,
         md_basis Basis,
         uZ       PackSize  = default_pack_size<T>,
         typename Allocator = aligned_allocator<T>>
    requires pack_size<PackSize>
class mdstorage : public detail_::aligned_base<Basis::size == 1> {
    using allocator_traits = std::allocator_traits<Allocator>;

    static constexpr bool vector_like = Basis::size == 1;

    using stride_type  = std::conditional_t<vector_like, decltype([] {}), uZ>;
    using extents_type = std::array<uZ, Basis::size>;

public:
    /**
     * @brief Construct a new mdarray object.
     * 
     * @param extents   Extents of individual axes. `extents[i]` corresponds to `Basis::axis<i>`.
     * @param alignment Contigious axis storage is padded to a multiple of the least common multiple of `alignment` and `PackSize`.
     * @param allocator 
     */
    explicit mdstorage(const extents_type& extents, uZ alignment = 1, Allocator allocator = {})
    : m_extents(extents)
    , m_allocator(allocator) {
        uZ align    = std::lcm(alignment, PackSize * 2);
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
        m_ptr = allocator_traits::allocate(allocator, size);
        std::memset(m_ptr, 0, size * sizeof(T));
    };

    explicit mdstorage() = default;

    mdstorage(const mdstorage& other)            = delete;
    mdstorage& operator=(const mdstorage& other) = delete;

    mdstorage(mdstorage&& other) noexcept
    : m_allocator(std::move(other.m_allocator))
    , m_ptr(other.m_ptr)
    , m_stride(other.m_stride)
    , m_extents(other.m_extents) {
        other.m_ptr     = nullptr;
        other.m_stride  = 0;
        other.m_extents = {};
    };

    mdstorage&
    operator=(mdstorage&& other) noexcept(allocator_traits::propagate_on_container_move_assignment::value ||
                                          allocator_traits::is_always_equal::value) {
        using std::swap;
        if constexpr (allocator_traits::propagate_on_container_move_assignment::value) {
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

    ~mdstorage() noexcept {
        deallocate();
    }

    template<auto Axis>
        requires /**/ (Basis::template contains<Axis>)
    [[nodiscard]] auto slice(uZ index) noexcept {
        return detail_::md_get_slice<T, PackSize, Basis, Axis, false, true>(
            m_ptr, m_stride, index, m_extents);
    };

    [[nodiscard]] auto operator[](uZ index) noexcept {
        return slice<Basis::outer_axis>(index);
    }

    [[nodiscard]] auto at(uZ index) {
        //TODO: move to `bounds_check()`
        if (index >= size()) {
            throw(std::out_of_range("index (which is ) >= size (which is)"));
        }
        return slice<Basis::outer_axis>(index);
    }

    // TODO: multidimentional slice(), operator[](), at()
    // TODO: runtime slice() ?

    [[nodiscard]] auto begin() noexcept {
        return detail_::md_get_iterator<T, PackSize, Basis, false, true>(m_ptr, m_stride, 0, m_extents);
    };
    [[nodiscard]] auto begin() const noexcept {
        return cbegin();
    };
    [[nodiscard]] auto cbegin() const noexcept {
        return detail_::md_get_iterator<T, PackSize, Basis, true, true>(m_ptr, m_stride, 0, m_extents);
    }

    [[nodiscard]] auto end() noexcept {
        return detail_::md_get_iterator<T, PackSize, Basis, false, true>(m_ptr, m_stride, size(), m_extents);
    };
    [[nodiscard]] auto end() const noexcept {
        return cend();
    }
    [[nodiscard]] auto cend() const noexcept {
        return detail_::md_get_iterator<T, PackSize, Basis, true, true>(m_ptr, m_stride, size(), m_extents);
    };

    [[nodiscard]] auto size() const noexcept -> uZ {
        return m_extents.back();
    }

    template<auto Axis>
        requires /**/ (Basis::template contains<Axis>)
    [[nodiscard]] auto extent() const noexcept -> uZ {
        return m_extents[Basis::template index<Axis>];
    }
    [[nodiscard]] auto extents() const noexcept -> const extents_type& {
        return m_extents;
    }


private:
    [[no_unique_address]] Allocator m_allocator{};

    T*           m_ptr{};
    stride_type  m_stride{};
    extents_type m_extents{};

    inline void deallocate() noexcept {
        if (m_ptr != nullptr) {
            allocator_traits::deallocate(m_allocator, m_ptr, m_stride * m_extents.back());
        }
    };
};

template<typename T, uZ PackSize, md_basis Basis, bool Contigious, bool Const>
class mdslice
: public std::ranges::view_base
, detail_::aligned_base<Basis::size == 1 && Contigious> {
    using extents_type = std::array<uZ, Basis::size>;

    template<typename T_, uZ, md_basis Basis_, auto ExcludeAxis, bool, bool, uZ ExtentsSize>
        requires(ExtentsSize == Basis_::size ||
                 equal_values<ExcludeAxis, Basis_::outer_axis> && ExtentsSize == Basis_::size - 1)
    friend auto detail_::md_get_slice(T_*                                start,
                                      uZ                                 stride,
                                      uZ                                 index,
                                      const std::array<uZ, ExtentsSize>& extents) noexcept;

    static constexpr bool vector_like = Basis::size == 1 && Contigious;

    template<uZ ExtentsSize, uZ... Is>
    static constexpr bool valid_extents =
        unique_values<Is...> && sizeof...(Is) == Basis::size && ((Is < ExtentsSize) && ...);

    /**
     * @brief Construct a new mdslice object.
     * 
     * @tparam ExtentsSize  
     * @tparam Is...            Indexes of `parent_extents` to initialize slice extents.
                                `Is` order corresponds to `Basis` axis order.
     * @param start             Pointer to the first value.
     * @param stride            Stride of the outermost axis. 
     * @param parent_extents    Array of parent storage/slice extents.
     * @param [unused]      
     */
    template<uZ ExtentsSize, uZ... Is>
    mdslice(T*                                 start,
            uZ                                 stride,
            const std::array<uZ, ExtentsSize>& parent_extents,
            std::index_sequence<Is...>) noexcept
        requires /**/ (!vector_like && valid_extents<ExtentsSize, Is...>)
    : m_start(start)
    , m_stride(stride)
    , m_extents{parent_extents[Is]...} {};

    /**
     * @brief Construct a new mdslice object.
     * 
     * @tparam ExtentsSize  
     * @tparam Is...            Indexes of `parent_extents` to initialize slice extents.
                                `Is` order corresponds to `Basis` axis order.
     * @param start             Pointer to the first value.
     * @param stride            Stride is not used for contigious one-dimentional slice.
     * @param parent_extents    Array of parent storage/slice extents.
     * @param [unused]      
     */
    template<uZ ExtentsSize, uZ... Is>
    mdslice(T* start,
            uZ /*stride*/,
            const std::array<uZ, ExtentsSize>& extents,
            std::index_sequence<Is...>) noexcept
        requires /**/ (vector_like && valid_extents<ExtentsSize, Is...>)
    : m_start(start)
    , m_extents{extents[Is]...} {};

public:
    mdslice() = default;

    mdslice(const mdslice&) noexcept = default;
    mdslice(mdslice&&) noexcept      = default;

    mdslice& operator=(const mdslice&) noexcept = default;
    mdslice& operator=(mdslice&&) noexcept      = default;

    ~mdslice() = default;

    template<auto Axis>
        requires /**/ (Basis::template contains<Axis>)
    [[nodiscard]] auto slice(uZ index) const noexcept {
        return detail_::md_get_slice<T, PackSize, Basis, Axis, Const, Contigious>(
            m_start, m_stride, index, m_extents);
    }

    [[nodiscard]] auto operator[](uZ index) const noexcept {
        return slice<Basis::outer_axis>(index);
    }

    // TODO: `at()`
    // TODO: multidimensional slice and operator[]

    [[nodiscard]] auto begin() const noexcept {
        return detail_::md_get_iterator<T, PackSize, Basis, Const, Contigious>(
            m_start, stride(), 0, m_extents);
    }

    [[nodiscard]] auto end() const noexcept {
        return detail_::md_get_iterator<T, PackSize, Basis, Const, Contigious>(
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

template<typename T, md_basis Basis, uZ PackSize, bool Const, bool Contigious>
class mditerator {
    static constexpr bool one_dimensional = Basis::size == 1;
    using extents_type =
        std::conditional_t<one_dimensional, decltype([] {}), std::array<uZ, Basis::size - 1>>;

    template<typename T_, uZ, md_basis Basis_, bool, bool>
    friend auto detail_::md_get_iterator(T_*                                 start,
                                         uZ                                  stride,
                                         uZ                                  index,
                                         const std::array<uZ, Basis_::size>& extents) noexcept;

    template<uZ... Is>
    mditerator(T*                                 ptr,
               uZ                                 stride,
               const std::array<uZ, Basis::size>& extents,
               std::index_sequence<Is...>) noexcept
    : m_ptr(ptr)
    , m_stride(stride)
    , m_extents{extents[Is]...} {};

    // template<uZ... Is>
    mditerator(T* ptr,
               uZ stride,
               const std::array<uZ, Basis::size>& /*extents*/,
               std::index_sequence<>) noexcept
        requires(one_dimensional)
    : m_ptr(ptr)
    , m_stride(stride){};

    /**
     * @brief Construct a new mditerator object.
     * 
     * @param ptr               Pointer to the data.
     * @param stride            Outer axis stride.  
     * @param parent_extents    Array of parent storage/slice extents.
     */
    mditerator(T* ptr, uZ stride, const std::array<uZ, Basis::size>& parent_extents) noexcept
    : mditerator(ptr, stride, parent_extents, std::make_index_sequence<Basis::size - 1>{}){};

public:
    mditerator()                                      = default;
    mditerator(const mditerator&) noexcept            = default;
    mditerator(mditerator&&) noexcept                 = default;
    mditerator& operator=(const mditerator&) noexcept = default;
    mditerator& operator=(mditerator&&) noexcept      = default;
    ~mditerator()                                     = default;

    [[nodiscard]] inline auto operator*() const noexcept {
        return detail_::md_get_slice<T, PackSize, Basis, Basis::outer_axis, Const, Contigious>(
            m_ptr, m_stride, 0, extents());
    }

    using value_type =
        decltype(detail_::md_get_slice<T, PackSize, Basis, Basis::outer_axis, Const, Contigious>(
            nullptr, 1, 0, std::array<uZ, Basis::size - 1>{}));
    using iterator_concept = std::random_access_iterator_tag;
    using difference_type  = iZ;

    [[nodiscard]] inline auto operator[](uZ index) const noexcept {
        return detail_::md_get_slice<T, PackSize, Basis, Basis::outer_axis, Const, Contigious>(
            m_ptr, m_stride, index, extents());
    };

    inline auto operator+=(difference_type n) noexcept -> mditerator& {
        m_ptr += m_stride * n;
        return *this;
    };
    inline auto operator-=(difference_type n) noexcept -> mditerator& {
        m_ptr -= m_stride * n;
        return *this;
    };

    inline auto operator++() noexcept -> mditerator& {
        m_ptr += m_stride;
        return *this;
    };
    inline auto operator++(int) noexcept -> mditerator {
        auto copy = *this;
        m_ptr += m_stride;
        return copy;
    };
    inline auto operator--() noexcept -> mditerator& {
        m_ptr -= m_stride;
        return *this;
    };
    inline auto operator--(int) noexcept -> mditerator {
        auto copy = *this;
        m_ptr -= m_stride;
        return copy;
    };

    [[nodiscard]] inline friend auto operator+(const mditerator& lhs, difference_type rhs) noexcept
        -> mditerator {
        return {lhs.m_ptr + rhs * lhs.m_stride, lhs.m_stride, lhs.m_extents};
    };
    [[nodiscard]] inline friend auto operator+(difference_type lhs, const mditerator& rhs) noexcept
        -> mditerator {
        return rhs + lhs;
    };

    [[nodiscard]] inline friend auto operator-(const mditerator& lhs, difference_type rhs) noexcept
        -> mditerator {
        return lhs + (-rhs);
    };

    [[nodiscard]] inline friend auto operator-(const mditerator& lhs, const mditerator& rhs) noexcept
        -> difference_type {
        return (lhs.m_ptr - rhs.m_ptr) / lhs.m_stride;
    };

    [[nodiscard]] inline auto operator<=>(const mditerator& other) const noexcept {
        return m_ptr <=> other.m_ptr;
    };

    [[nodiscard]] inline auto operator==(const mditerator& other) const noexcept {
        return m_ptr == other.m_ptr;
    };

private:
    [[nodiscard]] auto extents() const noexcept
        -> std::conditional_t<one_dimensional, std::array<uZ, 0>, const extents_type&> {
        if constexpr (one_dimensional) {
            return {};
        } else {
            return m_extents;
        }
    }

    T*           m_ptr{};
    uZ           m_stride{};
    extents_type m_extents{};
};

namespace detail_ {

template<typename T, uZ PackSize, md_basis Basis, bool Const, bool Contigious>
auto md_get_iterator(T* start, uZ stride, uZ index, const std::array<uZ, Basis::size>& extents) noexcept {
    if constexpr (Basis::size == 1 && Contigious) {
        return iterator_maker<T, Const, PackSize>::make(start + pidx<PackSize>(index), index);
    } else {
        return mditerator<T, Basis, PackSize, Const, Contigious>(start + stride * index, stride, extents);
    }
};

template<typename T,
         uZ       PackSize,
         md_basis Basis,
         auto     ExcludeAxis,
         bool     Const,
         bool     Contigious,
         uZ       ExtentsSize>
    requires(ExtentsSize == Basis::size ||
             equal_values<ExcludeAxis, Basis::outer_axis> && ExtentsSize == Basis::size - 1)
auto md_get_slice(T* start, uZ stride, uZ index, const std::array<uZ, ExtentsSize>& extents) noexcept {
    constexpr uZ   axis_index     = Basis::template index<ExcludeAxis>;
    constexpr bool new_contigious = Contigious && Basis::template index<ExcludeAxis> != 0;
    using new_basis               = typename Basis::template exclude<ExcludeAxis>;

    constexpr auto get_offset = [](auto& extents, uZ index, uZ stride) {
        constexpr uZ axis_index = Basis::template index<ExcludeAxis>;
        if constexpr (Contigious && axis_index == 0) {
            return pidx<PackSize>(index);
        } else {
            for (iZ i = Basis::size - 1; i > axis_index; --i) {
                stride /= extents[i];
            }
            return stride * index;
        }
    };

    start += get_offset(extents, index, stride);
    if constexpr (Basis::size == 1) {
        return detail_::make_cx_ref<T, Const, PackSize>(start);
    } else {
        if constexpr (axis_index == Basis::size - 1 && Basis::size > 2) {
            stride /= extents.back();
        }
        if constexpr (ExtentsSize == Basis::size) {
            using indexes  = index_to_value_sequence<std::make_index_sequence<Basis::size>>;
            using filtered = value_to_index_sequence<filter_value_sequence<indexes, axis_index>>;

            return mdslice<T, PackSize, new_basis, new_contigious, Const>(start, stride, extents, filtered{});
        } else {
            // `mditerator` has outer axis extent removed
            return mdslice<T, PackSize, new_basis, new_contigious, Const>(
                start, stride, extents, std::make_index_sequence<ExtentsSize>{});
        }
    }
};
}    // namespace detail_

}    // namespace pcx
namespace std::ranges {
template<typename T, pcx::uZ PackSize, pcx::md_basis Basis, bool Contigious, bool Const>
inline constexpr bool enable_borrowed_range<pcx::mdslice<T, PackSize, Basis, Contigious, Const>> = true;
}
#endif