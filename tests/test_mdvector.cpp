#include "mdvector.hpp"

enum class ax1 {
    x = 0,
    y,
    z = 0,
};

enum class ax2 {
    a,
    b,
};


int main() {
    static_assert(!pcx::equal_values<ax1::x, ax2::a>);
    static_assert(!pcx::equal_values<ax1::x, ax1::y>);
    static_assert(pcx::equal_values<ax1::x, ax1::x>);
    static_assert(pcx::equal_values<ax1::x, ax1::z>);
    static_assert(pcx::unique_values<ax1::x, ax2::a, ax1::y, ax2::b>);
    static_assert(!pcx::unique_values<ax1::x, ax2::a, ax1::y, ax2::a>);
    static_assert(!pcx::unique_values<ax1::x, ax2::a, ax1::y, ax1::y>);
    static_assert(pcx::unique_values<>);

    using enum ax1;
    using enum ax2;
    using basis_l = pcx::left_first_basis<x, a, b, y>;
    using basis_r = pcx::right_first_basis<x, a, b, y>;
    auto i        = basis_l::index<y>;
    auto k        = basis_r::index<y>;

    auto v = basis_l::axis<3>;
    auto l = basis_r::axis<0>;

    using nox = basis_l::exclude<x>;
    using noy = basis_r::exclude<y>;

    static_assert(pcx::md_basis<basis_l>);

    static_assert(!pcx::md_basis<ax1>);

    return 0;
}