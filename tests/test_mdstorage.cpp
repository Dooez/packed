
#include "allocators.hpp"
#include "element_access.hpp"
#include "mdstorage.hpp"
#include "meta.hpp"
#include "types.hpp"

#include <bits/ranges_base.h>
#include <cassert>
#include <iostream>
#include <type_traits>

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

template<pcx::md::layout Layout = pcx::md::layout::left>
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
            std::cout << std::complex<float>(v) << " ";
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
namespace pcx::meta {
/**
     * @brief 
     * The following algorithm generates the next permutation lexicographically after a given permutation. It changes the given permutation in-place.
     * - Find the largest index k such that a[k] < a[k + 1]. If no such index exists, the permutation is the last permutation.
     * - Find the largest index l greater than k such that a[k] < a[l].
     * - Swap the value of a[k] with that of a[l].
     * - Reverse the sequence from a[k + 1] up to and including the final element a[n].
     * 
     */
namespace perm {
template<uZ K, uZ A, uZ Aprev, uZ... As>
struct k_reverse_search {
    using type = std::conditional_t<(A > Aprev),    //
                                    uZ_constant<K - 1>,
                                    typename k_reverse_search<K - 1, Aprev, As...>::type>;
};
template<uZ K, uZ A, uZ Aprev>
struct k_reverse_search<K, A, Aprev> {
    using type = std::conditional_t<(A > Aprev), uZ_constant<K - 1>, void>;
};

template<any_value_sequence S>
struct find_k_impl;
template<auto... Vs>
struct find_k_impl<value_sequence<Vs...>> {
    static constexpr uZ max_index = sizeof...(Vs);

    using type = k_reverse_search<max_index, Vs...>::type;
    static_assert(!std::is_void_v<type>);
};
template<any_value_sequence S>
struct find_k {
    static constexpr auto value = find_k_impl<detail_::reverse_value_sequence_impl<S>>::type::value;
};

template<uZ K, uZ I, uZ V, uZ Vprev, uZ... Vs>
struct i_reverse_search {
    using type = std::conditional_t<(V > Vprev),    //
                                    uZ_constant<I>,
                                    typename i_reverse_search<K, I - 1, Vprev, Vs...>::type>;
};
template<uZ K, uZ I, uZ V, uZ Vprev>
struct i_reverse_search<K, I, V, Vprev> {
    using type = std::conditional_t<(V > Vprev),    //
                                    uZ_constant<I>,
                                    void>;
};

template<uZ K, any_value_sequence S>
struct find_i_impl;
template<uZ K, auto... Vs>
struct find_i_impl<K, value_sequence<Vs...>> {
    static constexpr uZ max_index = sizeof...(Vs) - 1;

    using type = i_reverse_search<K, max_index, Vs...>::type;
    static_assert(!std::is_void_v<type>);
};

template<uZ K, any_value_sequence S>
struct find_i {
    static constexpr auto value = find_i_impl<K, detail_::reverse_value_sequence_impl<S>>::type::value;
};

template<any_value_sequence S, uZ I>
struct start_from {};

}    // namespace perm
template<uZ... Is>
struct next_perm {};

}    // namespace pcx::meta
template<auto Basis>
auto check_storage(auto&& storage){


};

int main() {
    using enum ax1;
    constexpr auto         left_basis = pcx::md::left_basis<x, y, z>{3U, 2U, 2U};
    std::array<float, 128> beging{};
    auto                   static_storage_l = pcx::md::static_stoarge<float, left_basis>{};
    std::array<float, 128> endg{};
    test_xyz_storage(static_storage_l);
    test_const_xyz_storage(static_storage_l);
    std::cout << "static l:\n";
    fill_mdstorage(static_storage_l, 10.);
    for (auto v: beging) {
        std::cout << v << " ";
    }
    std::cout << "\n";
    print_mdstorage(static_storage_l);
    print_mdstorage(std::as_const(static_storage_l));
    for (auto v: endg) {
        std::cout << v << " ";
    }
    std::cout << "\n";

    auto dynamic_storage_l = pcx::md::dynamic_storage<float, left_basis>{3U, 2U, 4U};
    test_xyz_storage(dynamic_storage_l);
    std::cout << "dynamic l:\n";
    fill_mdstorage(dynamic_storage_l, 10.);
    print_mdstorage(dynamic_storage_l);
    print_mdstorage(std::as_const(dynamic_storage_l));

    constexpr auto right_basis = pcx::md::right_basis<x, y, z>{1U, 1U, 1U};
    constexpr auto asds        = right_basis.outer_axis;

    auto static_storage_r  = pcx::md::static_stoarge<float, right_basis>{};
    auto dynamic_storage_r = pcx::md::dynamic_storage<float, right_basis>{3U, 2U, 1U};
    using enum pcx::md::layout;
    test_xyz_storage<right>(static_storage_r);
    test_xyz_storage<right>(dynamic_storage_r);
    std::cout << "dynamic r:\n";
    fill_mdstorage(dynamic_storage_r, 10.);
    print_mdstorage(dynamic_storage_r);
    // std::cout << "dynamic r:\n";
    // fill_mdstorage(dynamic_storage_r, 10.);
    // print_mdstorage(dynamic_storage_r);
    // std::cout << "static r:\n";
    // fill_mdstorage(static_storage_r, 10.);
    // print_mdstorage(static_storage_r);

    static constexpr auto vector_basis = pcx::md::left_basis<x>{8U};
    using vector_storage_type          = pcx::md::static_stoarge<float, vector_basis>;
    static_assert(pcx::complex_vector_of<float, vector_storage_type>);
    return 0;
}
