#include "mdvector.hpp"

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


int main() {
    static_assert(!pcx::equal_values<ax1::x, ax2::a>);
    static_assert(!pcx::equal_values<ax1::x, ax1::y>);
    static_assert(pcx::equal_values<ax1::x, ax1::x>);
    static_assert(pcx::equal_values<ax1::x, ax1::x_again>);
    static_assert(pcx::unique_values<ax1::x, ax2::a, ax1::y, ax2::b>);
    static_assert(!pcx::unique_values<ax1::x, ax2::a, ax1::y, ax2::a>);
    static_assert(!pcx::unique_values<ax1::x, ax2::a, ax1::y, ax1::y>);
    static_assert(pcx::unique_values<>);

    using enum ax1;
    using enum ax2;
    using basis_l = pcx::left_first_basis<x, y, z>;
    using basis_r = pcx::right_first_basis<x, a, b, y>;
    auto i        = basis_l::index<y>;
    auto k        = basis_r::index<y>;

    auto v = basis_l::axis<2>;
    auto l = basis_r::axis<0>;

    using nox = basis_l::exclude<x>;
    using noy = basis_r::exclude<y>;

    static_assert(pcx::md_basis<basis_l>);
    static_assert(!pcx::md_basis<ax1>);

    using xyz_basis   = pcx::left_first_basis<x, y, z>;
    using xyz_storage = pcx::mdstorage<float, xyz_basis>;

    auto storage = xyz_storage({32, 16, 8});

    auto s1  = storage.slice<z>(0);
    auto s11 = s1.slice<y>(0);

    using namespace std::ranges;
    // static_assert(range<xyz_storage>);

    return 0;
}