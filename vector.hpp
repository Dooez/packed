#ifndef PCX_VECTOR_HPP
#define PCX_VECTOR_HPP

#include "allocators.hpp"
#include "element_access.hpp"
#include "types.hpp"
#include "vector_arithm.hpp"
#include "vector_util.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

namespace pcx {
/**
 * @brief Vector of complex floating point values.
 * Values are stored in packs. Inside pack continious real values
 * are followed by continious imaginary values.
 *
 * @tparam T real type
 * @tparam PackSize number of complex values in pack default is 32/16 for float/double
 * @tparam Allocator
 */
template<typename T, uZ PackSize = pcx::default_pack_size<T>, typename Allocator = pcx::aligned_allocator<T>>
    requires packed_floating_point<T, PackSize>
class vector : public detail_::pack_aligned_base<true> {
    template<typename OT, uZ OPackSize, typename OAllocator>
        requires packed_floating_point<OT, OPackSize>
    friend class pcx::vector;

private:
    using alloc_traits = std::allocator_traits<Allocator>;

public:
    using real_type          = T;
    using real_pointer       = T*;
    using const_real_pointer = const T*;

    using allocator_type  = Allocator;
    using size_type       = uZ;
    using difference_type = ptrdiff_t;
    using reference       = cx_ref<T, false, PackSize>;
    using const_reference = const cx_ref<T, true, PackSize>;

    using iterator       = pcx::iterator<T, false, PackSize>;
    using const_iterator = pcx::iterator<T, true, PackSize>;

    static constexpr size_type pack_size = PackSize;

    vector() noexcept(noexcept(allocator_type())) = default;

    explicit vector(const allocator_type& allocator) noexcept
    : m_allocator(allocator)
    , m_ptr(nullptr){};

    explicit vector(size_type size, const allocator_type& allocator = allocator_type())
    : m_allocator(allocator)
    , m_size(size)
    , m_ptr(alloc_traits::allocate(m_allocator, real_size(m_size))) {
        fill(begin(), end(), 0);
    };

    template<typename U>
        requires std::is_convertible_v<U, std::complex<real_type>>
    vector(size_type size, U value, const allocator_type& allocator = allocator_type())
    : m_allocator(allocator)
    , m_size(size)
    , m_ptr(alloc_traits::allocate(m_allocator, num_packs(size) * PackSize * 2)) {
        fill(begin(), end(), value);
    };

    vector(const vector& other)
    : m_allocator(alloc_traits::select_on_container_copy_construction(other.m_allocator))
    , m_size(other.m_size) {
        if (m_size == 0) {
            return;
        }
        m_ptr = alloc_traits::allocate(m_allocator, real_size(m_size));
        packed_copy(other.begin(), other.end(), begin());
    }

    vector(const vector& other, const allocator_type& allocator)
    : m_allocator(allocator)
    , m_size(other.m_size) {
        if (m_size == 0) {
            return;
        }
        m_ptr = alloc_traits::allocate(m_allocator, real_size(m_size));
        packed_copy(other.begin(), other.end(), begin());
    }

    vector(vector&& other) noexcept
    : m_allocator(std::move(other.m_allocator))
    , m_size(other.m_size)
    , m_ptr(other.m_ptr) {
        other.m_ptr  = {};
        other.m_size = 0;
    };

    vector(vector&& other, const allocator_type& allocator) noexcept(alloc_traits::is_always_equal::value)
    : m_allocator(allocator) {
        if constexpr (alloc_traits::is_always_equal::value) {
            using std::swap;
            swap(m_size, other.m_size);
            swap(m_ptr, other.m_ptr);
        } else {
            if (m_allocator == other.m_allocator) {
                using std::swap;
                swap(m_size, other.m_size);
                swap(m_ptr, other.m_ptr);
                return;
            }
            m_ptr = alloc_traits::allocate(m_allocator, real_size(m_size));
            packed_copy(other.begin(), other.end(), begin());
        }
    };

    vector& operator=(const vector& other) {
        if (this == &other) {
            return *this;
        }
        if constexpr (alloc_traits::propagate_on_container_copy_assignment::value) {
            if (!alloc_traits::is_always_equal::value && (m_allocator != other.m_allocator) ||
                num_packs(m_size) != num_packs(other.m_size)) {
                deallocate();
                m_allocator = other.m_allocator;
                m_size      = other.m_size;
                m_ptr       = alloc_traits::allocate(m_allocator, real_size(m_size));
            } else {
                m_allocator = other.m_allocator;
                m_size      = other.m_size;
            }
        } else {
            if (num_packs(m_size) != num_packs(other.m_size)) {
                deallocate();
                m_size = other.m_size;
                m_ptr  = alloc_traits::allocate(m_allocator, real_size(m_size));
            }
        }
        packed_copy(other.begin(), other.end(), begin());
        return *this;
    }

    vector& operator=(vector&& other) noexcept(alloc_traits::propagate_on_container_move_assignment::value ||
                                               alloc_traits::is_always_equal::value) {
        using std::swap;
        if constexpr (alloc_traits::propagate_on_container_move_assignment::value) {
            deallocate();
            m_allocator  = std::move(other.m_allocator);
            m_size       = other.m_size;
            m_ptr        = other.m_ptr;
            other.m_size = 0;
            other.m_ptr  = real_pointer{};
            return *this;
        } else if constexpr (alloc_traits::is_always_equal::value) {
            swap(m_size, other.m_size);
            swap(m_ptr, other.m_ptr);
            return *this;
        } else {
            if (m_allocator == other.m_allocator) {
                swap(m_size, other.m_size);
                swap(m_ptr, other.m_ptr);
                return *this;
            }
            if (num_packs(m_size) != num_packs(other.m_size)) {
                deallocate();
                m_ptr = alloc_traits::allocate(m_allocator, real_size(other.m_size));
            }
            m_size = other.m_size;
            packed_copy(other.begin(), other.end(), begin());
            return *this;
        }
    }

    template<typename E>
        requires /**/ (!std::same_as<E, vector>) && detail_::vecexpr<E>
    vector& operator=(const E& other) {
        assert(size() == other.size());

        auto it_this = begin();
        auto it_expr = other.begin();

        using expr_traits             = detail_::expr_traits<E>;
        constexpr auto pack_size_expr = expr_traits::pack_size;
        // constexpr auto pack_size_expr = decltype(it_expr)::pack_size;

        if (it_expr.aligned()) {
            constexpr auto reg_size   = 32 / sizeof(T);
            constexpr auto store_size = std::max(reg_size, pack_size);

            auto aligned_size = (end().align_lower() - it_this) / store_size;
            auto ptr          = &(*it_this);
            for (uint i = 0; i < aligned_size; ++i) {
                for (uint i_reg = 0; i_reg < store_size; i_reg += reg_size) {
                    auto offset = i * store_size + i_reg;
                    auto data_  = expr_traits::template cx_reg<pack_size_expr>(it_expr, offset);
                    auto data   = simd::apply_conj(data_);
                    simd::cxstore<pack_size>(simd::ra_addr<pack_size>(ptr, offset), data);
                }
            }
            it_this += aligned_size * store_size;
            it_expr += aligned_size * store_size;
        }
        while (it_this < end()) {
            *it_this = *it_expr;
            ++it_this;
            ++it_expr;
        }
        return *this;
    };

    ~vector() {
        deallocate();
    };

    void swap(vector& other) noexcept(alloc_traits::propagate_on_container_swap::value ||
                                      alloc_traits::is_always_equal::value) {
        using std::swap;
        if constexpr (alloc_traits::propagate_on_container_swap::value) {
            swap(m_allocator, other.m_allocator);
            swap(m_size, other.m_size);
            swap(m_ptr, other.m_ptr);
        } else if constexpr (alloc_traits::is_always_equal::value) {
            swap(m_size, other.m_size);
            swap(m_ptr, other.m_ptr);
        } else {
            if (m_allocator == other.m_allocator) {
                swap(m_size, other.m_size);
                swap(m_ptr, other.m_ptr);
            } else {
                auto new_this       = real_pointer{};
                auto new_this_size  = other.m_size;
                auto new_other_size = m_size;
                if (other.m_ptr != real_pointer{}) {
                    new_this = alloc_traits::allocate(m_allocator, real_size(other.m_size));
                    std::memcpy(new_this, other.m_ptr, real_size(other.m_size) * sizeof(real_type));
                }
                if (real_size(m_size) != real_size(other.m_size)) {
                    auto new_other = real_pointer{};
                    if (m_ptr != real_pointer{}) {
                        auto new_other = alloc_traits::allocate(other.m_allocator, real_size(m_size));
                        std::memcpy(new_other, m_ptr, real_size(m_size) * sizeof(real_type));
                    }
                    other.deallocate();
                    other.m_ptr = new_other;
                } else {
                    if (m_ptr != real_pointer{}) {
                        std::memcpy(other.m_ptr, m_ptr, real_size(m_size) * sizeof(real_type));
                    }
                }
                deallocate();
                m_ptr        = new_this;
                m_size       = new_this_size;
                other.m_size = new_other_size;
            }
        }
    }

    friend void swap(vector& first, vector& second) noexcept(noexcept(first.swap(second))) {
        first.swap(second);
    }

    [[nodiscard]] auto get_allocator() const noexcept(std::is_nothrow_copy_constructible_v<allocator_type>)
        -> allocator_type {
        return m_allocator;
    }

    [[nodiscard]] auto operator[](size_type idx) noexcept -> reference {
        auto p_idx = packed_idx(idx);
        return reference{m_ptr + p_idx};
    }
    // NOLINTNEXLINE (*const*) proxy reference
    [[nodiscard]] auto operator[](size_type idx) const noexcept -> const_reference {
        auto p_idx = packed_idx(idx);
        return const_reference{m_ptr + p_idx};
    }

    [[nodiscard]] auto at(size_type idx) -> reference {
        if (idx >= size()) {
            throw std::out_of_range(std::string("idx (which is") + std::to_string(idx) +
                                    std::string(") >= vector size (which is ") + std::to_string(size()) +
                                    std::string(")"));
        }
        auto p_idx = packed_idx(idx);
        return {m_ptr + p_idx};
    }
    [[nodiscard]] auto at(size_type idx) const -> const_reference {
        if (idx >= size()) {
            throw std::out_of_range(std::string("idx (which is") + std::to_string(idx) +
                                    std::string(") >= vector size (which is ") + std::to_string(size()) +
                                    std::string(")"));
        }
        auto p_idx = packed_idx(idx);
        return const_reference{m_ptr + p_idx};
    }

    [[nodiscard]] auto data() noexcept -> real_pointer {
        return m_ptr;
    }
    [[nodiscard]] auto data() const noexcept -> const_real_pointer {
        return m_ptr;
    }

    [[nodiscard]] constexpr auto size() const noexcept -> size_type {
        return m_size;
    };
    [[nodiscard]] constexpr auto length() const noexcept -> size_type {
        return size();
    };
    void resize(size_type new_size) {
        if (num_packs(new_size) != num_packs(m_size)) {
            auto* new_ptr = alloc_traits::allocate(m_allocator, real_size(new_size));
            packed_copy(begin(), end(), iterator(new_ptr, 0));
            deallocate();
            m_ptr = new_ptr;
        }
        m_size = new_size;
    }

    [[nodiscard]] auto begin() noexcept -> iterator {
        return iterator(m_ptr, 0);
    }
    [[nodiscard]] auto begin() const noexcept -> const_iterator {
        return const_iterator(m_ptr, 0);
    }
    [[nodiscard]] auto cbegin() const noexcept -> const_iterator {
        return const_iterator(m_ptr, 0);
    }

    [[nodiscard]] auto end() noexcept -> iterator {
        return iterator(m_ptr + packed_idx(size()), size());
    }
    [[nodiscard]] auto end() const noexcept -> const_iterator {
        return const_iterator(m_ptr + packed_idx(size()), size());
    }
    [[nodiscard]] auto cend() const noexcept -> const_iterator {
        return const_iterator(m_ptr + packed_idx(size()), size());
    }

    template<typename Allocator_>
    explicit operator std::vector<std::complex<T>, Allocator_>() const {
        auto it_this      = begin();
        auto aligned_size = end().align_lower() - it_this;

        auto svec = std::vector<std::complex<T>, Allocator_>();
        svec.reserve(m_size);
        svec.resize(aligned_size);
        auto* svec_ptr = reinterpret_cast<T*>(svec.data());

        auto ptr = &(*it_this);
        for (uint i = 0; i < aligned_size; i += pack_size) {
            for (uint i_reg = 0; i_reg < pack_size; i_reg += simd::reg<real_type>::size) {
                auto data = simd::cxload<pack_size>(m_ptr, i + i_reg);
                simd::cxstore<simd::reg<real_type>::size>(simd::ra_addr<1>(svec_ptr, i + i_reg), data);
            }
        }
        it_this += aligned_size;

        while (it_this < end()) {
            svec.push_back(it_this.vale());
            ++it_this;
        }
    }

    template<uZ PackSize_>
    explicit vector(const vector<real_type, PackSize_, allocator_type>& other)
    : m_allocator(alloc_traits::select_on_container_copy_construction(other.m_allocator))
    , m_size(other.m_size) {
        if (m_size == 0) {
            return;
        }
        m_ptr = alloc_traits::allocate(m_allocator, real_size(m_size));
        *this = pcx::subrange(other.begin(), other.end());
    }

    template<typename Allocator_, uZ PackSize_>
    explicit vector(const vector<T, PackSize_, Allocator_>& other, const Allocator& allocator = Allocator{})
    : m_allocator(allocator)
    , m_size(other.m_size) {
        if (m_size == 0) {
            return;
        }
        m_ptr = alloc_traits::allocate(m_allocator, real_size(m_size));
        *this = pcx::subrange(other.begin(), other.end());
    }

private:
    [[no_unique_address]] allocator_type m_allocator{};

    size_type    m_size = 0;
    real_pointer m_ptr{};

    inline void deallocate() noexcept {
        if (m_ptr != real_pointer()) {
            alloc_traits::deallocate(m_allocator, m_ptr, real_size(size()));
            m_ptr  = real_pointer();
            m_size = 0;
        }
    }

    static constexpr auto num_packs(size_type vector_size) -> size_type {
        return (vector_size % PackSize > 0) ? vector_size / PackSize + 1 : vector_size / PackSize;
    }
    static constexpr auto real_size(size_type vector_size) -> size_type {
        return num_packs(vector_size) * PackSize * 2;
    }
    static constexpr auto packed_idx(size_type idx) -> size_type {
        return idx + idx / PackSize * PackSize;
    }
};


template<typename T, bool Const = false, uZ PackSize = pcx::default_pack_size<T>>
class subrange
: public rv::view_base
, detail_::pack_aligned_base<PackSize == 1> {
    template<typename VT, uZ VPackSize, typename>
        requires packed_floating_point<VT, VPackSize>
    friend class vector;

public:
    using real_type       = T;
    using size_type       = uZ;
    using difference_type = std::ptrdiff_t;

    using iterator       = pcx::iterator<T, Const, PackSize>;
    using const_iterator = pcx::iterator<T, true, PackSize>;

    using reference = typename iterator::reference;

    static constexpr size_type pack_size = PackSize;

    subrange() noexcept = default;

    subrange(const iterator& begin, size_type size) noexcept
    : m_begin(begin)
    , m_size(size){};

    subrange(const iterator& begin, const iterator& end) noexcept
    : m_begin(begin)
    , m_size(end - begin){};

    template<typename VAllocator>
    explicit subrange(vector<real_type, pack_size, VAllocator>& vector) noexcept
    : m_begin(vector.begin())
    , m_size(vector.size()){};

    template<typename VAllocator>
    explicit subrange(const vector<real_type, pack_size, VAllocator>& vector) noexcept
        requires(Const)
    : m_begin(vector.begin())
    , m_size(vector.size()){};

    subrange(const subrange&) noexcept = default;
    subrange(subrange&&) noexcept      = default;

    ~subrange() noexcept = default;

    subrange& operator=(subrange&&) noexcept      = default;
    subrange& operator=(const subrange&) noexcept = default;

    void swap(subrange& other) noexcept {
        using std::swap;
        swap(m_begin, other.m_begin);
        swap(m_size, other.m_size);
    }
    friend void swap(subrange& first, subrange& second) noexcept {
        first.swap(second);
    }

    [[nodiscard]] auto begin() const -> iterator {
        return m_begin;
    }
    [[nodiscard]] auto cbegin() const -> const_iterator {
        return m_begin;
    }
    [[nodiscard]] auto end() const -> iterator {
        return m_begin + size();
    }
    [[nodiscard]] auto cend() const -> const_iterator {
        return m_begin + size();
    }

    [[nodiscard]] auto operator[](difference_type idx) const -> reference {
        return *(m_begin + idx);
    }

    [[nodiscard]] auto size() const -> size_type {
        return m_size;
    };
    [[nodiscard]] bool aligned() const {
        return m_begin.aligned();
    }
    [[nodiscard]] bool empty() const {
        return size() == 0;
    }
    // NOLINTNEXTLINE (*explicit*)
    [[nodiscard]] operator bool() const {
        return size() != 0;
    }

    template<typename U>
        requires std::convertible_to<U, std::complex<T>>
    void fill(U value) {
        pcx::fill(begin(), end(), value);
    };

    template<typename R>
        requires(!Const) && (!detail_::vecexpr<R>) && rv::input_range<R> &&
                std::indirectly_copyable<rv::iterator_t<R>, iterator>
    void assign(const R& range) {
        if (range.size() != size()) {
            throw(std::invalid_argument(std::string("source size (which is ")
                                            .append(std::to_string(range.size()))
                                            .append(" is not equal to subrange size (which is ")
                                            .append(std::to_string(size()))
                                            .append(")")));
        }
        rv::copy(range, begin());
    };

    template<typename E>
        requires(!Const) && (detail_::vecexpr<E> || std::derived_from<E, detail_::vecexpr_base>)
    void assign(const E& expression) {
        if (expression.size() != size()) {
            throw(std::invalid_argument(std::string("source size (which is ")
                                            .append(std::to_string(expression.size()))
                                            .append(" is not equal to subrange size (which is ")
                                            .append(std::to_string(size()))
                                            .append(")")));
        }
        assert(size() == expression.size());

        auto it_this       = begin();
        auto it_expr       = expression.begin();
        auto aligned_begin = std::min(it_this.align_upper(), end());
        while (it_this < aligned_begin) {
            *it_this = *it_expr;
            ++it_this;
            ++it_expr;
        }

        using expr_traits             = detail_::expr_traits<E>;
        constexpr auto pack_size_expr = expr_traits::pack_size;

        if (it_this.aligned() && it_expr.aligned()) {
            constexpr auto reg_size   = 32 / sizeof(T);
            constexpr auto store_size = std::max(reg_size, pack_size);

            auto aligned_size = (end().align_lower() - it_this) / store_size;
            auto ptr          = &(*it_this);
            for (uint i = 0; i < aligned_size; ++i) {
                for (uint i_reg = 0; i_reg < store_size; i_reg += reg_size) {
                    auto offset = i * store_size + i_reg;
                    auto data_  = expr_traits::template cx_reg<pack_size_expr>(it_expr, offset);
                    auto data   = simd::apply_conj(data_);
                    simd::cxstore<pack_size>(simd::ra_addr<pack_size>(ptr, offset), data);
                }
            }
            it_this += aligned_size * store_size;
            it_expr += aligned_size * store_size;
        }
        while (it_this < end()) {
            *it_this = *it_expr;
            ++it_this;
            ++it_expr;
        }
    };

private:
    size_type m_size{};
    iterator  m_begin{};
};

}    // namespace pcx
namespace std::ranges {
template<typename T, pcx::uZ PackSize, bool Const>
inline constexpr bool enable_borrowed_range<pcx::subrange<T, Const, PackSize>> = true;
}

#endif
