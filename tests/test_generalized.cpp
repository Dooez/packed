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
void new_node(std::array<float, 2 * reg_size * 4>& data) {
    auto dest = std::array<float*, 4>{
        &data[0],
        &data[reg_size * 2],
        &data[reg_size * 4],
        &data[reg_size * 6],
    };
    constexpr auto sett = pcx::detail_::fft::newnode<4>::settings{
        .pack_dest = 16,
        .pack_src  = 16,
        .conj_tw   = false,
        .dit       = false,
    };
    auto tw = std::array<pcx::simd::cx_reg<float>, 3>{};
    pcx::detail_::fft::newnode<4>::perform<float, sett>(dest, tw);
}

int main() {
    /*pcx::detail_::fft::newnode<4>::perform();*/
    /*pcx::detail_::fft::node<4>::perform();*/
    return 0;
}
