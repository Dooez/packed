#include "fft.hpp"
#include "fxcorr.hpp"
#include "vector_util.hpp"

#include <memory>

int main() {
    pcx::vector<float> g(1024);
    //
    using fft_t   = pcx::fft_unit<float, pcx::fft_output::unsorted, pcx::dynamic_size, 2048>;
    auto fft_unit = std::make_shared<fft_t>(8192);

    auto pseudo_factory = pcx::internal::pseudo_vector_factory<float, std::allocator<float>>(8192);

    auto fxcorr = pcx::fxcorr_unit_shared(g, fft_unit, [&] { return pseudo_factory(); });

    return 0;
}