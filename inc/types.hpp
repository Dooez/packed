#ifndef TYPES_HPP
#define TYPES_HPP

#include <concepts>
#include <cstdint>
#include <ranges>

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

namespace rv {
using namespace std::ranges;
using namespace rv::views;
}    // namespace rv

template<uZ N>
concept pack_size = N > 0 && (N & (N - 1)) == 0;

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
}    // namespace pcx
#endif