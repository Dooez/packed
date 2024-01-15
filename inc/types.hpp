#ifndef TYPES_HPP
#define TYPES_HPP

#include <complex>
#include <cstdint>
#include <ranges>
#include <type_traits>

namespace pcx {

using f32 = float;
using f64 = float;

using uZ  = std::size_t;
using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8  = uint8_t;

using iZ  = std::ptrdiff_t;
using i64 = int64_t;
using i32 = int32_t;
using i16 = int16_t;
using i8  = int8_t;

template<uZ Value>
using uZ_constant = std::integral_constant<uZ, Value>;

namespace rv {
using namespace std::ranges;
using namespace rv::views;
}    // namespace rv

template<uZ N>
concept pack_size = N > 0 && (N & (N - 1)) == 0;

template<typename T>
concept floating_point = std::same_as<T, float> || std::same_as<T, double>;

template<typename T, uZ PackSize>
concept packed_floating_point = floating_point<T> && pack_size<PackSize>;

namespace simd {

/**
 * @brief reg::type is an alias for an intrinsic simd vector type
 */
template<typename T>
struct reg;

template<typename T>
using reg_t = typename reg<T>::type;

/**
 * @brief Complex simd vector.
 * Pack size template parameter could be added to streamline some interactons,
 * but it requires possibly quite large refactor. Should consider in the future.
 */
template<typename T, bool Conj = false, uZ PackSize = reg<T>::size>
    requires pack_size<PackSize> && (PackSize <= reg<T>::size)
struct cx_reg {
    reg_t<T>            real;
    reg_t<T>            imag;
    static constexpr uZ size = reg<T>::size;
};

}    // namespace simd

namespace detail_ {
template<bool Always>
struct aligned_base {};
template<>
struct aligned_base<true> {
    static constexpr std::true_type always_aligned{};
};
}    // namespace detail_

template<typename T>
concept always_aligned = std::derived_from<T, detail_::aligned_base<true>>;

template<typename T>
    requires floating_point<T>
constexpr const uZ default_pack_size = 32 / sizeof(T);

template<typename T, std::align_val_t Alignment = std::align_val_t{64}>
class aligned_allocator {
public:
    using value_type      = T;
    using is_always_equal = std::true_type;

    aligned_allocator() noexcept = default;

    template<typename U>
    explicit aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {};

    aligned_allocator(const aligned_allocator&) noexcept = default;
    aligned_allocator(aligned_allocator&&) noexcept      = default;

    ~aligned_allocator() = default;

    aligned_allocator& operator=(const aligned_allocator&) noexcept = default;
    aligned_allocator& operator=(aligned_allocator&&) noexcept      = default;

    [[nodiscard]] auto allocate(uZ n) -> value_type* {
        return reinterpret_cast<value_type*>(::operator new[](n * sizeof(value_type), Alignment));
    }

    void deallocate(value_type* p, uZ) {
        ::operator delete[](reinterpret_cast<void*>(p), Alignment);
    }

    template<typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

private:
};
template<typename T, std::align_val_t Alignment>
bool operator==(const aligned_allocator<T, Alignment>&, const aligned_allocator<T, Alignment>&) noexcept {
    return true;
}

template<typename T, bool Const, uZ PackSize>
class iterator;

template<typename T, bool Const, uZ PackSize>
class cx_ref;

template<typename T, uZ PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize>
class vector;

template<typename T, bool Const, uZ PackSize>
class subrange;

template<uZ PackSize>
constexpr auto pidx(uZ idx) -> uZ {
    return idx + idx / PackSize * PackSize;
}

namespace detail_ {
template<typename T>
struct is_std_complex_floating_point {
    static constexpr bool value = false;
};

template<floating_point F>
struct is_std_complex_floating_point<std::complex<F>> {
    using real_type             = F;
    static constexpr bool value = true;
};

template<typename T>
struct is_pcx_iterator {
    static constexpr bool value = false;
};

template<typename T, bool Const, uZ PackSize>
struct is_pcx_iterator<iterator<T, Const, PackSize>> {
    static constexpr bool value = true;
};
}    // namespace detail_

template<typename V>
struct cx_vector_traits {
    using real_type               = decltype([] {});
    static constexpr uZ pack_size = 0;
};

template<typename R>
    requires rv::contiguous_range<R> && detail_::is_std_complex_floating_point<rv::range_value_t<R>>::value
struct cx_vector_traits<R> {
    using real_type = typename detail_::is_std_complex_floating_point<rv::range_value_t<R>>::real_type;
    static constexpr uZ pack_size = 1;

    static auto re_data(R& vector) -> real_type* {
        return reinterpret_cast<real_type*>(rv::data(vector));
    }
    static auto re_data(const R& vector) -> const real_type* {
        return reinterpret_cast<const real_type*>(rv::data(vector));
    }
    static auto size(const R& vector) {
        return rv::size(vector);
    }
};

template<typename R>
    requires /**/ rv::range<R> && (detail_::is_pcx_iterator<rv::iterator_t<R>>::value)
struct cx_vector_traits<R> {
    using real_type               = typename rv::iterator_t<R>::real_type;
    static constexpr uZ pack_size = rv::iterator_t<R>::pack_size;

    inline static auto re_data(R& vector) {
        auto it = vector.begin();
        if constexpr (!always_aligned<R>) {
            if (!it.aligned()) {
                throw(
                    std::invalid_argument("Packed complex range is not aligned. Packed complex range must be "
                                          "aligned to be accessed as a vector."));
            }
        }
        return &(*it);
    }

    inline static auto re_data(const R& vector) {
        auto it = vector.begin();
        if constexpr (!always_aligned<R>) {
            if (!it.aligned()) {
                throw(
                    std::invalid_argument("Packed complex range is not aligned. Packed complex range must be "
                                          "aligned to be accessed as a vector."));
            }
        }
        return &(*it);
    }

    inline static auto size(const R& vector) {
        return rv::size(vector);
    }
};

template<typename T, typename V>
concept complex_vector_of = std::same_as<T, typename cx_vector_traits<V>::real_type>;

template<typename T, typename R>
concept range_of_complex_vector_of = rv::random_access_range<R> &&    //
                                     complex_vector_of<T, std::remove_pointer_t<rv::range_value_t<R>>>;

}    // namespace pcx
#endif