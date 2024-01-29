
#include "allocators.hpp"
#include "element_access.hpp"
#include "mdstorage.hpp"
#include "meta.hpp"

#include <bits/ranges_base.h>
#include <iostream>

enum class ax1 {
    x = 0,
    y,
    z,
    x_again = 0,
};

enum class ax2 {
    a,
    b,
};

template<typename T>
struct test_ranges {
    static_assert(std::ranges::range<T>);
    static_assert(std::ranges::sized_range<T>);
    static_assert(std::ranges::input_range<T>);
    static_assert(std::ranges::forward_range<T>);
    static_assert(std::ranges::bidirectional_range<T>);
    static_assert(std::ranges::random_access_range<T>);
    static_assert(std::ranges::common_range<T>);
    static_assert(std::ranges::viewable_range<T>);
};

template<pcx::md::layout Layout = pcx::md::layout::left>
auto test_xyz_storage(auto&& storage) {
    using enum ax1;

    auto sx   = storage.template slice<x>(0);
    auto sxy  = sx.template slice<y>(0);
    auto sxyz = sxy.template slice<z>(0);
    auto sy   = storage.template slice<y>(0);
    auto syz  = sy.template slice<z>(0);
    auto syzx = syz.template slice<x>(0);
    auto sz   = storage.template slice<z>(0);
    auto szx  = sz.template slice<x>(0);
    auto szxy = szx.template slice<y>(0);

    (void)test_ranges<decltype(storage)>();
    (void)test_ranges<decltype(sx)>();
    (void)test_ranges<decltype(sxy)>();
    (void)test_ranges<decltype(sy)>();
    (void)test_ranges<decltype(syz)>();
    (void)test_ranges<decltype(sz)>();
    (void)test_ranges<decltype(szx)>();

    static_assert(!pcx::complex_vector_of<float, decltype(storage)>);
    static_assert(!pcx::complex_vector_of<float, decltype(sx)>);
    static_assert(!pcx::complex_vector_of<float, decltype(sy)>);
    static_assert(!pcx::complex_vector_of<float, decltype(sy)>);
    static_assert(!pcx::complex_vector_of<float, decltype(szx)>);
    using enum pcx::md::layout;
    if constexpr (Layout == left) {
        static_assert(!pcx::complex_vector_of<float, decltype(sxy)>);
        static_assert(pcx::complex_vector_of<float, decltype(syz)>);
    } else if constexpr (Layout == right) {
        static_assert(pcx::complex_vector_of<float, decltype(sxy)>);
        static_assert(!pcx::complex_vector_of<float, decltype(syz)>);
    }
}


auto fill_mdstorage(auto&& slice, double exponent) {
    constexpr auto find_max = [](auto&& foo, auto&& slice, double e, double exponent) -> double {
        if constexpr (pcx::detail_::is_pcx_iterator<pcx::rv::iterator_t<decltype(slice)>>::value) {
            return e;
        } else {
            return foo(foo, slice[0], e * exponent, exponent);
        }
    };

    auto cexp = find_max(find_max, slice, 1., exponent);

    constexpr auto fill = [](auto&& f, auto&& slice, double cexp, double exponent, double offset) {
        if constexpr (pcx::detail_::is_pcx_iterator<pcx::rv::iterator_t<decltype(slice)>>::value) {
            pcx::uZ i = 0;
            for (auto v: slice) {
                v = std::complex<float>(offset + i);
                ++i;
            }
        } else {
            for (auto s: slice) {
                f(f, s, cexp / exponent, exponent, offset );
                offset += cexp;
            }
        }
    };

    fill(fill, slice, cexp, exponent, 0);
}

auto check_storage(auto&& storage);

auto print_mdstorage(auto&& slice) {
    if constexpr (pcx::detail_::is_pcx_iterator<pcx::rv::iterator_t<decltype(slice)>>::value) {
        for (auto v: slice) {
            std::cout << std::complex<float>(v) << " ";
        }
    } else {
        for (auto s: slice) {
            print_mdstorage(s);
            std::cout << "\n";
        }
    }
}


int main() {
    using enum ax1;
    static constexpr auto left_basis = pcx::md::left_basis<x, y, z>{3U, 3U, 3U};

    auto static_stoarge_l  = pcx::md::static_stoarge<float, left_basis>{};
    auto dynamic_storage_l = pcx::md::dynamic_storage<float, left_basis>{3U, 3U, 3U};
    test_xyz_storage(static_stoarge_l);
    test_xyz_storage(dynamic_storage_l);
    fill_mdstorage(dynamic_storage_l, 10.);
    print_mdstorage(dynamic_storage_l);

    static constexpr auto right_basis = pcx::md::left_basis<x, y, z>{8U, 16U, 32U};

    auto static_stoarge_r  = pcx::md::static_stoarge<float, right_basis>{};
    auto dynamic_storage_r = pcx::md::dynamic_storage<float, right_basis>{8U, 16U, 32U};
    test_xyz_storage(static_stoarge_l);

    test_xyz_storage(dynamic_storage_l);

    static constexpr auto vector_basis = pcx::md::left_basis<x>{8U};
    using vector_storage_type          = pcx::md::static_stoarge<float, vector_basis>;
    static_assert(pcx::complex_vector_of<float, vector_storage_type>);
    return 0;
}