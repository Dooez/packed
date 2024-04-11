#include "element_access.hpp"
#include "mdstorage.hpp"
#include "types.hpp"

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
    void foo() {
        static_assert(std::ranges::range<T>);
        static_assert(std::ranges::sized_range<T>);
        static_assert(std::ranges::input_range<T>);
        static_assert(std::ranges::forward_range<T>);
        static_assert(std::ranges::bidirectional_range<T>);
        static_assert(std::ranges::random_access_range<T>);
        static_assert(std::ranges::common_range<T>);
        static_assert(std::ranges::viewable_range<T>);
    }
};

template<typename T, pcx::md::layout Layout = pcx::md::layout::left>
auto test_xyz_storage(auto&& storage) {
    using enum ax1;

    auto s    = storage.as_slice();
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
    (void)test_ranges<decltype(s)>();
    (void)test_ranges<decltype(sx)>();
    (void)test_ranges<decltype(sxy)>();
    (void)test_ranges<decltype(sy)>();
    (void)test_ranges<decltype(syz)>();
    (void)test_ranges<decltype(sz)>();
    (void)test_ranges<decltype(szx)>();

    static_assert(!pcx::complex_vector_of<T, decltype(storage)>);
    static_assert(!pcx::complex_vector_of<T, decltype(sx)>);
    static_assert(!pcx::complex_vector_of<T, decltype(sy)>);
    static_assert(!pcx::complex_vector_of<T, decltype(sy)>);
    static_assert(!pcx::complex_vector_of<T, decltype(szx)>);
    using enum pcx::md::layout;
    if constexpr (Layout == left) {
        static_assert(!pcx::complex_vector_of<T, decltype(sxy)>);
        static_assert(pcx::complex_vector_of<T, decltype(syz)>);
    } else if constexpr (Layout == right) {
        static_assert(pcx::complex_vector_of<T, decltype(sxy)>);
        static_assert(!pcx::complex_vector_of<T, decltype(syz)>);
    }
}

template<typename T, pcx::md::layout Layout = pcx::md::layout::left>
auto test_const_xyz_storage(const auto& storage) {
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

    static_assert(!pcx::complex_vector_of<T, decltype(storage)>);
    static_assert(!pcx::complex_vector_of<T, decltype(sx)>);
    static_assert(!pcx::complex_vector_of<T, decltype(sy)>);
    static_assert(!pcx::complex_vector_of<T, decltype(sy)>);
    static_assert(!pcx::complex_vector_of<T, decltype(szx)>);
    using enum pcx::md::layout;
    if constexpr (Layout == left) {
        static_assert(!pcx::complex_vector_of<T, decltype(sxy)>);
        static_assert(pcx::complex_vector_of<T, decltype(syz)>);
    } else if constexpr (Layout == right) {
        static_assert(pcx::complex_vector_of<T, decltype(sxy)>);
        static_assert(!pcx::complex_vector_of<T, decltype(syz)>);
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
                v = static_cast<float>(offset + static_cast<double>(i));
                ++i;
            }
        } else {
            for (auto s: slice) {
                f(f, s, cexp / exponent, exponent, offset);
                offset += cexp;
            }
        }
    };

    fill(fill, slice, cexp, exponent, 0);
}


auto print_mdstorage(auto&& slice) {
    if constexpr (pcx::detail_::is_pcx_iterator<pcx::rv::iterator_t<decltype(slice)>>::value) {
        // std::cout << slice.size() << "\n";
        for (auto v: slice) {
            std::cout << std::complex<float>{v} << " ";
        }
    } else {
        // std::cout << slice.size() << "\n";
        for (auto s: slice) {
            print_mdstorage(s);
            std::cout << "\n";
        }
    }
}

using pcx::uZ;
template<auto Basis>
auto check_storage(auto&& storage) {


};


template<typename T>
int do_tests() {
    using enum ax1;
    constexpr auto           left_basis = pcx::md::left_basis<x, y, z>{5U, 2U, 3U};
    const std::array<T, 128> beging{};
    auto                     static_storage_l = pcx::md::static_stoarge<T, left_basis>{};
    const std::array<T, 128> endg{};
    test_xyz_storage<T>(static_storage_l);
    test_const_xyz_storage<T>(static_storage_l);
    std::cout << "static left 3:2:3 :\n";
    constexpr double ten = 10.;
    fill_mdstorage(static_storage_l.as_slice(), ten);
    print_mdstorage(std::as_const(static_storage_l).as_slice());

    auto dynamic_storage_l = pcx::md::dynamic_storage<T, left_basis>{5U, 8U, 8U};
    test_xyz_storage<T>(dynamic_storage_l);
    std::cout << "dynamic left 5:8:8 :\n";
    fill_mdstorage(dynamic_storage_l, ten);
    print_mdstorage(dynamic_storage_l);
    print_mdstorage(std::as_const(dynamic_storage_l));

    constexpr auto right_basis = pcx::md::right_basis<x, y, z>{3U, 2U, 4U};
    constexpr auto asds        = pcx::md::right_basis<x, y, z>::outer_axis;

    auto static_storage_r  = pcx::md::static_stoarge<T, right_basis>{};
    auto dynamic_storage_r = pcx::md::dynamic_storage<T, right_basis>{8U, 8U, 5U};
    using enum pcx::md::layout;

    test_xyz_storage<T, right>(static_storage_r);
    std::cout << "static right 3|2|4:\n";
    fill_mdstorage(static_storage_r, ten);
    print_mdstorage(static_storage_r);

    test_xyz_storage<T, right>(dynamic_storage_r);
    std::cout << "dynamic right :\n";
    fill_mdstorage(dynamic_storage_r, ten);
    print_mdstorage(dynamic_storage_r);

    constexpr auto short_basis    = pcx::md::left_basis<x>{4U};
    auto           short_static_l = pcx::md::static_stoarge<T, short_basis>{};
    std::cout << "short left 4:\n";
    fill_mdstorage(short_static_l, ten);
    print_mdstorage(short_static_l);
    static_assert(pcx::complex_vector_of<T, decltype(short_static_l)>);
    return 0;
}


int main() {
    std::cout << "float:\n";
    do_tests<float>();
    // std::cout << "double:\n";
    // do_tests<double>();

    return 0;
}
