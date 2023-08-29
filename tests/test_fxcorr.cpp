#include "fft.hpp"
#include "fxcorr.hpp"
#include "simd_fft.hpp"
#include "tests/test_pcx.hpp"
#include "vector_util.hpp"

#include <memory>

template<typename T>
using svector = std::vector<std::complex<T>>;

template<typename T>
auto naive_fxcorr(const svector<T>& signal, const svector<T>& base) -> svector<T> {
    auto len    = signal.size() + base.size() - 1;
    auto l2     = 1U << (pcx::detail_::fft::log2i(len - 1) + 1);
    auto tmp    = svector<T>(l2);
    auto kernel = svector<T>(l2);

    std::ranges::copy(signal, tmp.begin());
    std::ranges::copy(base, kernel.begin());
    auto fft = pcx::fft_unit<T>(l2);
    fft(kernel);
    fft(tmp);

    std::ranges::transform(tmp, kernel, tmp.begin(), [](auto a, auto b) { return a * conj(b); });
    fft.ifft(tmp);
    auto ret = svector<T>(signal.size());
    std::ranges::copy(tmp.begin(), tmp.begin() + signal.size(), ret.begin());
    return ret;
};

void fill_bark(auto& vector, std::size_t offset) {
    vector[offset + 0]  = 1;
    vector[offset + 1]  = 1;
    vector[offset + 2]  = 1;
    vector[offset + 3]  = 1;
    vector[offset + 4]  = 1;
    vector[offset + 5]  = -1.;
    vector[offset + 6]  = -1.;
    vector[offset + 7]  = 1;
    vector[offset + 8]  = 1;
    vector[offset + 9]  = -1.;
    vector[offset + 10] = 1;
    vector[offset + 11] = -1.;
    vector[offset + 12] = 1;
}

int main() {
    auto               size = 128;
    pcx::vector<float> g(size);
    fill_bark(g, 0);

    pcx::vector<float> f(size);
    fill_bark(f, 35);

    using fft_t   = pcx::fft_unit<float, pcx::fft_order::unordered, pcx::aligned_allocator<float>, 2048>;
    auto fft_unit = std::make_shared<fft_t>(8192);

    auto pseudo_factory = pcx::detail_::pseudo_vector_factory<float, std::allocator<float>>(8192);

    auto fxcorr = pcx::fxcorr_unit(g, fft_unit, [&] { return pseudo_factory(); });

    // for (auto v: f) {
    //     std::cout << std::to_string(abs(v.value())) << "\n";
    // }
    auto sg = svector<float>(size);
    auto sf = svector<float>(size);
    fill_bark(sg, 0);
    fill_bark(sf, 0);
    fill_bark(sf, 109);

    for (uint i = 0; i < sf.size(); ++i) {
        sf[i] *= std::complex<float>(0,1);
        f[i] = sf[i];
    }

    auto xcr = pcx::fxcorr_unit(g, size * 2);
    xcr(f);


    auto res = naive_fxcorr(sf, sg);
    for (uint i = 0; i < size; ++i) {
        if (!equal_eps(f[i].value(), res[i], 10000000)) {
            std::cout << i << " " << abs(f[i].value() - res[i]) << " " << res[i] << f[i].value() << "\n";
        }
    }

    // for (auto v: res) {
    //     std::cout << std::to_string(abs(v)) << "\n";
    // }
    return 0;
}