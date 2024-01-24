// #include "mdstorage.hpp"
#include "allocators.hpp"
#include "static_mdstorage.hpp"
#include "types.hpp"

#include <bits/ranges_base.h>

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

int main() {
    //     static_assert(!pcx::equal_values<ax1::x, ax2::a>);
    //     static_assert(!pcx::equal_values<ax1::x, ax1::y>);
    //     static_assert(pcx::equal_values<ax1::x, ax1::x>);
    //     static_assert(pcx::equal_values<ax1::x, ax1::x_again>);
    //     static_assert(pcx::unique_values<ax1::x, ax2::a, ax1::y, ax2::b>);
    //     static_assert(!pcx::unique_values<ax1::x, ax2::a, ax1::y, ax2::a>);
    //     static_assert(!pcx::unique_values<ax1::x, ax2::a, ax1::y, ax1::y>);
    //     static_assert(pcx::unique_values<>);
    //
    //     using enum ax1;
    //     using enum ax2;
    //     using basis_l = pcx::left_first_basis<x, y, z>;
    //     using basis_r = pcx::right_first_basis<x, a, b, y>;
    //     auto i        = basis_l::index<y>;
    //     auto k        = basis_r::index<y>;
    //
    //     auto v = basis_l::axis<2>;
    //     auto l = basis_r::axis<0>;
    //
    //     using nox = basis_l::exclude<x>;
    //     using noy = basis_r::exclude<y>;
    //
    //     static_assert(pcx::md_basis<basis_l>);
    //     static_assert(!pcx::md_basis<ax1>);
    //
    //     using xyz_basis   = pcx::left_first_basis<x, y, z>;
    //     using xyz_storage = pcx::mdstorage<float, xyz_basis>;
    //
    //     auto storage = xyz_storage({32, 16, 8});
    //     auto s_z     = storage.slice<z>(0);
    //     auto s_y     = storage.slice<y>(0);
    //     auto s_x     = storage.slice<x>(0);
    //
    //     auto s_zy = s_z.slice<y>(0);
    //     auto s_yx = s_y.slice<x>(0);
    //     auto s_xz = s_x.slice<z>(0);
    //
    //     using xy_slice = decltype(s_z);
    //     using xz_slice = decltype(s_y);
    //     using yz_slice = decltype(s_x);
    //     using x_slice  = decltype(s_zy);
    //     using y_slice  = decltype(s_yx);
    //     using z_slice  = decltype(s_xz);
    //
    //     (void)test_ranges<xyz_storage>{};
    //     (void)test_ranges<xy_slice>{};
    //     (void)test_ranges<xz_slice>{};
    //     (void)test_ranges<yz_slice>{};
    //     (void)test_ranges<x_slice>{};
    //     (void)test_ranges<y_slice>{};
    //     (void)test_ranges<z_slice>{};
    //
    //     static_assert(std::ranges::output_range<x_slice, std::complex<float>>);
    //     static_assert(std::ranges::output_range<y_slice, std::complex<float>>);
    //     static_assert(std::ranges::output_range<z_slice, std::complex<float>>);
    //
    //     static_assert(pcx::complex_vector_of<float, x_slice>);
    //     static_assert(!pcx::complex_vector_of<float, y_slice>);
    // //     static_assert(!pcx::complex_vector_of<float, z_slice
    //
    using enum ax1;
    static constexpr auto static_basis = pcx::md::static_basis<x, y, z>{8U, 16U, 32U};
    using static_stoarge_type =
        pcx::md::storage<float,    //
                         static_basis,
                         8,
                         16,
                         pcx::md::detail_::static_::storage_base<float, static_basis, 8, 16>>;

    using dynamic_base =
        pcx::md::detail_::dynamic::storage_base<float, static_basis, 8, 16, pcx::aligned_allocator<float>>;
    using dynamic_storage_type = pcx::md::storage<float, static_basis, 8, 16, dynamic_base>;
    auto static_stoarge        = static_stoarge_type{};
    // auto dynamic_storage       = dynamic_storage_type(std::array<pcx::uZ, 3>{8, 16, 32});

    auto sx   = static_stoarge.slice<x>(0);
    auto sxy  = sx.slice<y>(0);
    auto sy   = static_stoarge.slice<y>(0);
    auto syz  = sy.slice<z>(0);
    auto syzx = syz.slice<x>(0);

    static constexpr auto vector_basis = pcx::md::static_basis<x>{8u};
    using vector_storage_type =
        pcx::md::storage<float,
                         vector_basis,
                         8,
                         16,
                         pcx::md::detail_::static_::storage_base<float, vector_basis, 8, 16>>;


    (void)test_ranges<decltype(static_stoarge)>();
    (void)test_ranges<decltype(sx)>();
    (void)test_ranges<decltype(sxy)>();
    (void)test_ranges<vector_storage_type>();

    using sxy_it_t = pcx::rv::iterator_t<decltype(sxy)>;
    using syz_it_t = pcx::rv::iterator_t<decltype(syz)>;


    static_assert(!pcx::complex_vector_of<float, decltype(sxy)>);
    static_assert(pcx::complex_vector_of<float, decltype(syz)>);
    static_assert(pcx::complex_vector_of<float, vector_storage_type>);

    auto ds = pcx::md::d_stoarge<float, static_basis, 8, 16>(pcx::aligned_allocator<float>{}, 3u, 4u, 5u);

    auto ss = pcx::md::static_stoarge<float, static_basis, 8, 16>();

    static_assert(std::allocator_traits<pcx::aligned_allocator<float>>::is_always_equal::value);
    return 0;
}