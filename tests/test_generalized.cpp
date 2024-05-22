#include "fft.hpp"
#include "simd_fft_generalized.hpp"


int main() {
    using namespace pcx;
    constexpr uZ NodeSize = 2;
    constexpr uZ reg_size = 8;
    using T               = float;

    auto tw_re = []<uZ... Is>(std::index_sequence<Is...>) {
        return std::array{static_cast<T>(Is)...};
    }(std::make_index_sequence<reg_size>{});
    auto tw_im = []<uZ... Is>(std::index_sequence<Is...>) {
        return std::array{static_cast<T>(Is)...};
    }(std::make_index_sequence<reg_size>{});

    auto x = vector<float>(32);
    auto y = vector<float>(32);

    auto dest_data = []<uZ... Is>(std::index_sequence<Is...>) {
        return std::array{(void(Is), vector<float>(32))...};
    }(std::make_index_sequence<NodeSize>{});

    detail_::fft::node<NodeSize>::template perform<float, 8, 8, false, false>();


    return 0;
}
