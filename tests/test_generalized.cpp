#include "simd_fft_generalized.hpp"

constexpr pcx::uZ reg_size = 8;
/*void              old(std::array<float, 2 * reg_size * 4>& data) {*/
/*    auto dest = std::array<float*, 4>{*/
/*        &data[0],*/
/*        &data[reg_size * 2],*/
/*        &data[reg_size * 4],*/
/*        &data[reg_size * 6],*/
/*    };*/
/*    auto tw = std::array<pcx::simd::cx_reg<float>, 3>{};*/
/*    pcx::detail_::fft::node<4>::perform<float, reg_size, reg_size, false, false>(dest, tw);*/
/*}*/
void new_node(std::array<float, 2 * reg_size * 4>&    data,
              std::array<float, 2 * reg_size * 4>&    source,
              std::array<pcx::simd::cx_reg<float>, 3> tw) {
    auto dest = std::array<float*, 4>{
        &data[0],
        &data[reg_size * 2],
        &data[reg_size * 4],
        &data[reg_size * 6],
    };
    auto src = std::array<const float*, 4>{
        &source[0],
        &source[reg_size * 2],
        &source[reg_size * 4],
        &source[reg_size * 6],
    };
    constexpr auto sett = pcx::detail_::fft::newnode<4, float>::settings{
        .pack_dest = 8,
        .pack_src  = 8,
        .conj_tw   = false,
        .dit       = false,
    };
    pcx::detail_::fft::newnode<4, float>::perform<sett>(dest, src, tw);
    /*pcx::detail_::fft::node<4>::perform<float, sett>(dest, tw);*/

    /*pcx::detail_::fft::node<4>::template perform<float, 8, 8, false, false, false>(dest, tw);*/
}

int main() {
    /*pcx::detail_::fft::newnode<4>::perform();*/
    /*pcx::detail_::fft::node<4>::perform();*/
    return 0;
}
