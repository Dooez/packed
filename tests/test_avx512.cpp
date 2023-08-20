#include "avx512_common.hpp"

#include <array>
#include <complex>
#include <iostream>
#include <vector>

using namespace pcx;

template<uZ pack_from, uZ pack_to, bool detail = false>
auto print(std::vector<std::complex<float>> src) {
    auto dat       = simd::cxload<pack_from>(reinterpret_cast<float*>(src.data()));
    auto [rep_dat] = simd::repack2<pack_to>(dat);
    simd::cxstore<pack_to>(reinterpret_cast<float*>(src.data()), rep_dat);

    std::cout << pack_from << " to " << pack_to << "\n";
    if (detail) {
        for (uZ i = 0; i < 16; ++i) {
            std::cout << src[i];
        }
        std::cout << "\n";
    }
    return src;
}

int main() {
    using std::ranges::equal;
    std::vector<std::complex<float>> sts1(16);
    for (uZ i = 0; i < 16; ++i) {
        sts1[i] = {float(i), 100.F + float(i)};
    }
    auto sts16 = print<1, 16, true>(sts1);
    auto sts8  = print<1, 8, true>(sts1);
    auto sts4  = print<1, 4, true>(sts1);
    auto sts2  = print<1, 2, true>(sts1);

    if (!equal(print<16, 1>(sts16), sts1))
        return 1;
    if (!equal(print<16, 2>(sts16), sts2))
        return 1;
    if (!equal(print<16, 4>(sts16), sts4))
        return 1;
    if (!equal(print<16, 8>(sts16), sts8))
        return 1;

    if (!equal(print<8, 1>(sts8), sts1))
        return 1;
    if (!equal(print<8, 2>(sts8), sts2))
        return 1;
    if (!equal(print<8, 4>(sts8), sts4))
        return 1;
    if (!equal(print<8, 16>(sts8), sts16))
        return 1;

    if (!equal(print<4, 1>(sts4), sts1))
        return 1;
    if (!equal(print<4, 2>(sts4), sts2))
        return 1;
    if (!equal(print<4, 8>(sts4), sts8))
        return 1;
    if (!equal(print<4, 16>(sts4), sts16))
        return 1;

    if (!equal(print<2, 1>(sts2), sts1))
        return 1;
    if (!equal(print<2, 4>(sts2), sts4))
        return 1;
    if (!equal(print<2, 8>(sts2), sts8))
        return 1;
    if (!equal(print<2, 16>(sts2), sts16))
        return 1;

    return 0;
}