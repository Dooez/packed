#include "fft.hpp"
#include "fxcorr.hpp"
#include "simd_fft.hpp"
#include "tests/test_pcx.hpp"
#include "vector_util.hpp"

#include <cstddef>
#include <cstdlib>
#include <memory>

template<typename T>
using svector = std::vector<std::complex<T>>;

template<typename T>
auto naive_fxcorr(const svector<T>& signal, const svector<T>& base) -> svector<T> {
    auto len    = signal.size() + base.size() - 1;
    auto l2     = 1U << (pcxo::detail_::fft::log2i(len - 1) + 1);
    auto tmp    = svector<T>(l2);
    auto kernel = svector<T>(l2);

    std::ranges::copy(signal, tmp.begin());
    std::ranges::copy(base, kernel.begin());
    auto fft = pcxo::fft_unit<T>(l2);
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
auto next_pow_2(uint64_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}

template<typename T>
int test_xcorr(std::size_t g_size) {
    auto g_pcx = pcxo::vector<T>(g_size);
    auto g_std = std::vector<std::complex<T>>(g_size);

    for (uint i = 0; i < g_size; ++i) {
        g_pcx[i] = std::exp(std::complex<T>(0, std::rand() % 6 - 3));
        g_std[i] = g_pcx[i];
    }
    // using fft_t   = pcxo::fft_unit<T, pcxo::fft_order::unordered, pcxo::aligned_allocator<T>, 2048>;

    auto fft_size = next_pow_2(g_size * 2);

    auto pseudo_factory = pcxo::detail_::pseudo_vector_factory<T, std::allocator<T>>(fft_size);
    auto fxcorr         = pcxo::fxcorr_unit<T>(g_pcx, fft_size);

    auto f_size = fft_size - g_size;
    auto f_pcx  = pcxo::vector<T>(f_size);
    auto f_std  = std::vector<std::complex<T>>(f_size);

    for (uint i = 0; i < f_size; ++i) {
        f_pcx[i] = std::exp(std::complex<T>(0, std::rand() % 6 - 3));
        f_std[i] = f_pcx[i];
    }

    fxcorr(f_pcx);
    auto res = naive_fxcorr(f_std, g_std);

    int ret = 0;
    for (uint i = 0; i < f_size; ++i) {
        if (!equal_eps(f_pcx[i].value(), res[i], 10000000)) {
            std::cout << i << " " << abs(f_pcx[i].value() - res[i]) << " " << res[i] << f_pcx[i].value()
                      << "\n";
            ++ret;
        }
    }
    return ret;
}

int main() {
    auto                size = 128;
    pcxo::vector<float> g(size);
    fill_bark(g, 0);

    pcxo::vector<float> f(size);
    fill_bark(f, 35);

    using fft_t   = pcxo::fft_unit<float, pcxo::fft_order::unordered, pcxo::aligned_allocator<float>, 2048>;
    auto fft_unit = std::make_shared<fft_t>(8192);

    auto pseudo_factory = pcxo::detail_::pseudo_vector_factory<float, std::allocator<float>>(8192);

    auto fxcorr = pcxo::fxcorr_unit(g, fft_unit, [&] { return pseudo_factory(); });

    // for (auto v: f) {
    //     std::cout << std::to_string(abs(v.value())) << "\n";
    // }
    auto sg = svector<float>(size);
    auto sf = svector<float>(size);
    fill_bark(sg, 0);
    fill_bark(sf, 0);
    fill_bark(sf, 109);

    for (uint i = 0; i < sf.size(); ++i) {
        sf[i] *= std::complex<float>(0, 1);
        sf[i] += std::rand() % 7 - 3;
        f[i] = sf[i];
    }

    auto xcr = pcxo::fxcorr_unit(g, size * 2);
    xcr(f);


    auto res = naive_fxcorr(sf, sg);
    for (uint i = 0; i < size; ++i) {
        if (!equal_eps(f[i].value(), res[i], 10000000)) {
            std::cout << i << " " << abs(f[i].value() - res[i]) << " " << res[i] << f[i].value() << "\n";
        }
    }
    return test_xcorr<float>(512);
    // for (auto v: res) {
    //     std::cout << std::to_string(abs(v)) << "\n";
    // }
    return 0;
}
