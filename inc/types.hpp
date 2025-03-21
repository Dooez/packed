#ifndef TYPES_HPP
#define TYPES_HPP

#include <complex>
#include <cstdint>
#include <ranges>
#include <type_traits>

namespace pcxo {

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


template<typename T>
    requires floating_point<T>
constexpr const uZ default_pack_size = 32 / sizeof(T);

template<typename T, bool Const, uZ PackSize>
class iterator;

template<typename T, bool Const, uZ PackSize>
class cx_ref;

template<typename T, uZ PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize>
class vector;

template<typename T, bool Const, uZ PackSize>
class subrange;

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
}    // namespace detail_

/**
 *@brief Used to access raw data of packed complex vectors.
 * Packed complex vector is a range that stores complex floating point values
 * in a contigious memory region, where n real values are followed by 
 * n imaginary values. n is the vector's pack size. 
 *
 * Example of a vector with pack size 2:
 * [re0, re1, im0, im1, re2, re3, im2, im3]
 *
 * A specialization for contigious ranges of std::complex<T> is provided. 
 * cx_vector_traits can be specialized to support arbitraty ranges.
 *
 */
template<typename V>
struct cx_vector_traits {
    using real_type               = decltype([] {});
    static constexpr uZ pack_size = 0;
};

template<typename R>
    requires rv::contiguous_range<R> && detail_::is_std_complex_floating_point<rv::range_value_t<R>>::value
struct cx_vector_traits<R> {
    using real_type        = typename detail_::is_std_complex_floating_point<rv::range_value_t<R>>::real_type;
    using iterator_t       = rv::iterator_t<R>;
    using const_iterator_t = decltype(rv::cbegin(std::declval<R&>()));

    static constexpr uZ   pack_size                 = 1;
    static constexpr bool enable_vector_expressions = false;
    static constexpr bool always_aligned            = true;

    static auto re_data(R& vector) -> real_type* {
        return reinterpret_cast<real_type*>(rv::data(vector));
    }
    static auto re_data(const R& vector) -> const real_type* {
        return reinterpret_cast<const real_type*>(rv::data(vector));
    }
    static auto size(const R& vector) {
        return rv::size(vector);
    }
    static constexpr auto aligned(const R& /*vector*/) {
        return true;
    }
    static constexpr auto aligned(const iterator_t& /*iterator*/) {
        return true;
    }
    static constexpr auto aligned(const const_iterator_t& /*iterator*/) {
        return true;
    }
};

template<typename V>
concept complex_vector = cx_vector_traits<V>::pack_size != 0;

template<typename T, typename V>
concept complex_vector_of = std::same_as<T, typename cx_vector_traits<V>::real_type>;

template<typename T, typename R>
concept range_of_complex_vector_of = rv::random_access_range<R> &&    //
                                     complex_vector_of<T, std::remove_pointer_t<rv::range_value_t<R>>>;

template<typename T>
concept always_aligned = cx_vector_traits<T>::always_aligned;

namespace detail_ {
/**
 *  @brief expr_traits is a struct providing a vector expression
 *  interface for class `T`. 
 *
 *  In the scope of `pcx` library vector expressions require 
 *  methods of accessing packed simd registers and checking
 *  pack alignment.
 *
 *  `expr_traits` specialization for classes satisfying `pcxo::complex_vector`
 *  is provided.
 */
template<typename T>
struct expr_traits {
    static constexpr bool enable_vector_expressions = false;

    /* 
    static constexpr bool always_aligned = ...;    
    static constexpr uZ   pack_size      = ...;

    using real_type        = ...;
    using iterator_t       = ...;
    using const_iterator_t = ...;
    */

    /*
     * Returns complex simd vector starting at `iterator` + `offset`.
     * `iterator` must be aligned.
    template<uZ PackSize, typename Iter>
        requires iterator_of<Iter, R>
    static auto cx_reg(const Iter& iterator, uZ offset);
     */

    /*
     * Returnes whether the `iterator` is aligned
    template<typename Iter>
        requires iterator_of<Iter, R>
    static auto aligned(const Iter& iterator) -> bool {
     */
};

template<typename T>
concept vecexpr = expr_traits<T>::enable_vector_expressions;

}    // namespace detail_

template<detail_::vecexpr E, typename V>
    requires complex_vector_of<typename detail_::expr_traits<E>::real_type, V>
void store(const E& src_expr, V& dest_vec);

}    // namespace pcxo
#endif
