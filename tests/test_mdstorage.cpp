
#include "allocators.hpp"
#include "mdstorage.hpp"
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
    using enum ax1;
    static constexpr auto static_basis = pcx::md::left_basis<x, y, z>{8U, 16U, 32U};
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

    static constexpr auto vector_basis = pcx::md::left_basis<x>{8u};
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

    auto ds = pcx::md::dynamic_stoarge<float, static_basis, 8, 16>(pcx::aligned_allocator<float>{}, 3u, 4u, 5u);

    auto ss = pcx::md::static_stoarge<float, static_basis, 8, 16>();

    static_assert(std::allocator_traits<pcx::aligned_allocator<float>>::is_always_equal::value);
    return 0;
}