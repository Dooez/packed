
#include "vector.hpp"
#include "vector_util.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>
// void set_te(packed_cx_vector<double>& v_def_2)
// {
//     set(v_def_2.begin(), v_def_2.end(), 1);
// }

void test_repack(float* data) {
    using namespace pcx::avx;
    auto a   = cxload<8>(data);
    auto [b] = convert<float>::split<1>(a);
    cxstore<8>(data, b);
}

void asm_test_fun(pcx::vector<double>& v1, pcx::vector<double>& v2, std::complex<double> v3) {
    v1 = v1 + v2 * v1;
}
template<typename T>
    requires std::floating_point<T>
int equal_eps(T lhs, T rhs) {
    static constexpr T epsilon = 10 * std::numeric_limits<T>::epsilon;
    if (lhs == rhs) {
        return true;
    }

    T largest = lhs > rhs ? lhs : rhs;
    return abs(lhs - rhs) < largest * epsilon;
}

template<typename T>
    requires std::floating_point<T>
int equal_eps(std::complex<T> lhs, std::complex<T> rhs) {
    static constexpr T epsilon = 10 * std::numeric_limits<T>::epsilon();
    if (lhs == rhs) {
        return true;
    }

    T largest = abs(lhs) > abs(rhs) ? abs(lhs) : abs(rhs);
    return abs(lhs - rhs) < (largest * epsilon);
}

template<typename T, std::size_t PackSize>
    requires pcx::packed_floating_point<T, PackSize>
int test_bin_ops(std::size_t length) {
    auto pcx_lhs = pcx::vector<T, PackSize, pcx::aligned_allocator<T>>(length);
    auto pcx_rhs = pcx::vector<T, PackSize, pcx::aligned_allocator<T>>(length);
    auto pcx_res = pcx::vector<T, PackSize, pcx::aligned_allocator<T>>(length);

    auto std_lhs = std::vector<std::complex<T>>(length);
    auto std_rhs = std::vector<std::complex<T>>(length);

    auto cx_scalar = std::complex<T>(512 * (std::numeric_limits<T>::epsilon() + 1),
                                     -512 * std::numeric_limits<T>::epsilon());

    auto re_scalar = 1024 * (std::numeric_limits<T>::epsilon() - 1);

    for (uint i = 0; i < length; ++i) {
        auto lhs =
            std::complex<T>(i * std::numeric_limits<T>::epsilon(), static_cast<T>(-1) - static_cast<T>(i));
        auto rhs = std::complex<T>(i + 1, i * std::numeric_limits<T>::epsilon());

        pcx_lhs[i] = lhs;
        std_lhs[i] = lhs;
        pcx_rhs[i] = rhs;
        std_rhs[i] = rhs;
    }

    auto cxops = std::make_tuple(    //
        [](auto&& a, auto&& b) { return a + b; },
        [](auto&& a, auto&& b) { return a - b; },
        [](auto&& a, auto&& b) { return a * b; },
        [](auto&& a, auto&& b) { return a / b; },
        [](auto&& a, auto&& b) { return a + conj(b); },
        [](auto&& a, auto&& b) { return a - conj(b); },
        [](auto&& a, auto&& b) { return a * conj(b); },
        [](auto&& a, auto&& b) { return a / conj(b); },
        [](auto&& a, auto&& b) { return conj(a) + b; },
        [](auto&& a, auto&& b) { return conj(a) - b; },
        [](auto&& a, auto&& b) { return conj(a) * b; },
        [](auto&& a, auto&& b) { return conj(a) / b; },
        [](auto&& a, auto&& b) { return conj(a) + conj(b); },
        [](auto&& a, auto&& b) { return conj(a) - conj(b); },
        [](auto&& a, auto&& b) { return conj(a) * conj(b); },
        [](auto&& a, auto&& b) { return conj(a) / conj(b); }
        //
    );


    auto reops = std::make_tuple(    //
        [](auto&& a, auto&& b) { return a + b; },
        [](auto&& a, auto&& b) { return a - b; },
        [](auto&& a, auto&& b) { return a * b; },
        [](auto&& a, auto&& b) { return a / b; }
        //
    );

    auto check_vector = [&]<std::size_t N>(auto&& ops) {
        auto& op = std::get<N>(ops);
        pcx_res  = op(pcx_lhs, pcx_rhs);
        for (uint i = 0; i < length; ++i) {
            auto v   = pcx_res[i].value();
            auto res = op(std_lhs[i], std_rhs[i]);
            if (!equal_eps(res, pcx_res[i].value())) {
                std::cout << "Values not equal after vector•vector operation " << N <<    //
                    " for pack size " << PackSize << ".\n";
                std::cout << "Length: " << length << " Index: " << i <<    //
                    ". Expected value: " << res <<                         //
                    ". Acquired value: " << pcx_res[i].value() << ".\n";
                return 1;
            }
        }
        return 0;
    };

    auto check_subr = [&]<std::size_t N>(auto&& ops) {
        auto& op = std::get<N>(ops);
        pcx::subrange<T, false, pcx::dynamic_size, PackSize>(pcx_res).assign(op(pcx_lhs, pcx_rhs));
        for (uint i = 0; i < length; ++i) {
            auto v   = pcx_res[i].value();
            auto res = op(std_lhs[i], std_rhs[i]);
            if (!equal_eps(res, pcx_res[i].value())) {
                std::cout << "Values not equal after subrange vector•vector operation " << N <<    //
                    " for pack size " << PackSize << ".\n";
                std::cout << "Length: " << length << " Index: " << i <<    //
                    ". Expected value: " << res <<                         //
                    ". Acquired value: " << pcx_res[i].value() << ".\n";
                return 1;
            }
        }
        return 0;
    };
    auto check_cx_scalar = [&]<std::size_t N>(auto&& ops) {
        auto& op = std::get<N>(ops);
        pcx_res  = op(pcx_lhs, cx_scalar);
        for (uint i = 0; i < length; ++i) {
            auto res = op(std_lhs[i], cx_scalar);
            if (!equal_eps(res, pcx_res[i].value())) {
                std::cout << "Values not equal after vector•cx_scalar operation " << N <<    //
                    " for pack size " << PackSize << ".\n";
                std::cout << "Length: " << length << " Index: " << i <<    //
                    ". Expected value: " << res <<                         //
                    ". Acquired value: " << pcx_res[i].value() << ".\n";
                return 1;
            }
        }
        pcx_res = op(cx_scalar, pcx_rhs);
        for (uint i = 0; i < length; ++i) {
            auto res = op(cx_scalar, std_rhs[i]);
            if (!equal_eps(res, pcx_res[i].value())) {
                std::cout << "Values not equal after cx_scalar•vector operation " << N <<    //
                    " for pack size " << PackSize << ".\n";
                std::cout << "Length: " << length << " Index: " << i <<    //
                    ". Expected value: " << res <<                         //
                    ". Acquired value: " << pcx_res[i].value() << ".\n";
                return 1;
            }
        }
        return 0;
    };

    auto check_re_scalar = [&]<std::size_t N>(auto&& ops) {
        auto& op = std::get<N>(ops);
        pcx_res  = op(pcx_lhs, re_scalar);
        for (uint i = 0; i < length; ++i) {
            auto res = op(std_lhs[i], re_scalar);
            if (!equal_eps(res, pcx_res[i].value())) {
                std::cout << "Values not equal after vector•re_scalar operation " << N <<    //
                    " for pack size " << PackSize << ".\n";
                std::cout << "Length: " << length << " Index: " << i <<    //
                    ". Expected value: " << res <<                         //
                    ". Acquired value: " << pcx_res[i].value() << ".\n";
                return 1;
            }
        }
        pcx_res = op(re_scalar, pcx_rhs);
        for (uint i = 0; i < length; ++i) {
            auto res = op(re_scalar, std_rhs[i]);
            if (!equal_eps(res, pcx_res[i].value())) {
                std::cout << "Values not equal after re_scalar•vector operation " << N <<    //
                    " for pack size " << PackSize << ".\n";
                std::cout << "Length: " << length << " Index: " << i <<    //
                    ". Expected value: " << res <<                         //
                    ". Acquired value: " << pcx_res[i].value() << ".\n";
                return 1;
            }
        }
        return 0;
    };

    auto check_cx = [&]<typename... Ops>(const std::tuple<Ops...>& ops) {
        auto check_all_impl = [&]<std::size_t... N>(auto&& ops, std::index_sequence<N...>) {
            return (check_vector.template operator()<N>(ops) + ...) +
                   (check_subr.template operator()<N>(ops) + ...) +
                   (check_cx_scalar.template operator()<N>(ops) + ...);
        };
        return check_all_impl(ops, std::index_sequence_for<Ops...>{});
    };
    auto check_re = [&]<typename... Ops>(const std::tuple<Ops...>& ops) {
        auto check_all_impl = [&]<std::size_t... N>(auto&& ops, std::index_sequence<N...>) {
            return (check_re_scalar.template operator()<N>(ops) + ...);
        };
        return check_all_impl(ops, std::index_sequence_for<Ops...>{});
    };

    return check_cx(cxops) + check_re(reops);
}

template<typename T>
    requires std::floating_point<T>
int test_subrange(std::size_t length) {
    auto vec1 = pcx::vector<T>(length);
    auto vec2 = pcx::vector<T>(length);
    auto vecr = pcx::vector<T>(length);

    auto stdvec1 = std::vector<std::complex<T>>(length);
    auto stdvec2 = std::vector<std::complex<T>>(length);
    auto stdvecr = std::vector<std::complex<T>>(length);

    for (uint i = 1; i < length; ++i) {
        const auto v11 = pcx::subrange(vec1.begin(), i);
        auto       v12 = pcx::subrange(vec1.cbegin() + i, length - i);
        const auto v21 = pcx::subrange(vec2.begin(), i);
        const auto v22 = pcx::subrange(vec2.cbegin() + i, length - i);
        auto       vr1 = pcx::subrange(vecr.begin(), i);
        auto       vr2 = pcx::subrange(vecr.begin() + i, length - i);

        vr1.assign(v11 + v21);
        vr2.assign(v12 + v22);

        // if (!test_add(stdvec1, stdvec2, vecr)) {
        //     return 1;
        // }
    }
    return 0;
}

int main() {
    int res = 0;
    for (uint i = 1; i < 32; ++i) {
        res += test_bin_ops<float, 1>(i);
        res += test_bin_ops<float, 2>(i);
        res += test_bin_ops<float, 4>(i);
        res += test_bin_ops<float, 8>(i);
        res += test_bin_ops<float, 16>(i);
        res += test_bin_ops<float, 32>(i);

        res += test_bin_ops<double, 1>(i);
        res += test_bin_ops<double, 2>(i);
        res += test_bin_ops<double, 4>(i);
        res += test_bin_ops<double, 8>(i);

        // res += test_subrange<float>(i);
        // res += test_subrange<double>(i);
        if (res > 0) {
            std::cout << i << "\n";
            return i;
        }
    }
    return res;
}
