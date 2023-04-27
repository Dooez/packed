#include "fxcorr.hpp"

void fill_bark(auto& vector, std::size_t offset) {
    vector[offset + 0] = 1;
    vector[offset + 1] = 1;
    vector[offset + 2] = 1;
    vector[offset + 3] = 1;
    vector[offset + 4] = 1;
    vector[offset + 5] = -1.;
    vector[offset + 6] = -1.;
    vector[offset + 7] = 1;
    vector[offset + 8] = 1;
    vector[offset + 9] = -1.;
    vector[offset + 10] = 1;
    vector[offset + 11] = -1.;
    vector[offset + 12] = 1;
}

int main() {
    auto               size = 128;
    pcx::vector<float> g(size);
    fill_bark(g, 0);

    pcx::vector<float> f(size );
    fill_bark(f, 35);

    auto xcr = pcx::fxcorr_unit(g, 256);
    xcr(f);
    for (auto v: f) {
        std::cout << std::to_string(abs(v.value())) << "\n";
    }

    return 0;
}