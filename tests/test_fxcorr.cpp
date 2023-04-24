#include "fxcorr.hpp"

int main() {
    pcx::vector<float> g(1024);
    //
    auto fconv = pcx::fxcorr_unit(g, 8192);

    return 0;
}