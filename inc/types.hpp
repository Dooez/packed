
#include <cstdint>

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

namespace simd {

template<typename T>
struct reg;

template<typename T>
using reg_t = typename reg<T>::type;

template<typename T, bool Conj = false>
struct cx_reg {
    reg_t<T>            real;
    reg_t<T>            imag;
    static constexpr uZ size = reg<T>::size;
};

}    // namespace simd
}    // namespace pcx