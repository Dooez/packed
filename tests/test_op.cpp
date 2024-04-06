#include "mdstorage.hpp"
#include "simd_common.hpp"
#include "vector.hpp"
#include "vector_arithm.hpp"

#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>

using pcx::uZ;

void test_repack(float* data) {
    using namespace pcx::simd;
    auto a1 = cxload<8>(data);
    auto a2 = cxload<8>(data + 16);
    // auto a3   = cxload<8>(data + 16);
    // auto a4   = cxload<8>(data + 24);
    auto [b, c] = repack<1, 8>(a1, a2);
    cxstore<8>(data, b);
    cxstore<8>(data + 16, c);
    // cxstore<8>(data + 16, d);
    // cxstore<8>(data + 24, e);
}
auto test_multi(pcx::simd::cx_reg<float, false> a1,
                pcx::simd::cx_reg<float, false> a2,
                pcx::simd::cx_reg<float, false> a3,
                pcx::simd::cx_reg<float, false> a4) {
    return pcx::simd::mul_pairs(a1, a2, a3, a4, a2, a3);
}

auto test_non_multi(pcx::simd::cx_reg<float, false> a1,
                    pcx::simd::cx_reg<float, true>  a2,
                    pcx::simd::cx_reg<float, false> a3,
                    pcx::simd::cx_reg<float, false> a4) {
    return std::make_tuple(pcx::simd::mul(a1, a2), pcx::simd::mul(a3, a4), pcx::simd::mul(a2, a3));
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

auto type_names = std::map<std::type_index, std::string>{};
int  reg_type_name(const auto& obj, std::string_view name) {
    auto index = std::type_index{typeid(obj)};
    if (type_names.contains(index)) {
        if (type_names[index] != name)
            return -1;
        return 0;
    }
    type_names[index] = name;
    return 0;
};
auto get_type_name(const auto& obj) -> std::string {
    auto index = std::type_index{typeid(obj)};
    if (type_names.contains(index)) {
        return type_names[index];
    }
    return "<unregistered type " + std::string(typeid(obj).name()) + ">";
};

template<typename T, uZ PackSize>
    requires pcx::packed_floating_point<T, PackSize>
int test_bin_ops(uZ length) {
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

    using namespace pcx;
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
        [](auto&& a, auto&& b) { return conj(a) / conj(b); });

    auto reops = std::make_tuple(    //
        [](auto&& a, auto&& b) { return a + b; },
        [](auto&& a, auto&& b) { return a - b; },
        [](auto&& a, auto&& b) { return a * b; },
        [](auto&& a, auto&& b) { return a / b; }
        //
    );


    auto check_cx = [&](auto&& lhs, auto&& rhs, auto&& ops) {
        return [&]<uZ... I>(std::index_sequence<I...>, auto&& lhs, auto&& rhs, auto& ops) {
            auto single_op = [&](auto&& lhs, auto&& rhs, auto&& op, uZ N) {
                pcx_res = op(lhs, rhs);
                for (uint i = 0; i < length; ++i) {
                    auto v   = pcx_res[i].value();
                    auto res = op(std_lhs[i], std_rhs[i]);
                    if (!equal_eps(res, pcx_res[i].value())) {
                        const auto* lhs_name = typeid(lhs).name();
                        const auto* rhs_name = typeid(rhs).name();
                        std::cout << "Values not equal after " << get_type_name(lhs) << "•"
                                  << get_type_name(rhs) << " operation #" << N << " for pack size "
                                  << PackSize << ".\n";
                        std::cout << "Length: " << length << " Index: " << i <<    //
                            ". Expected value: " << res <<                         //
                            ". Acquired value: " << pcx_res[i].value() << ".\n";
                        return 1;
                    }
                }
                return 0;
            };
            return (single_op(lhs, rhs, std::get<I>(ops), I) + ...);
        }(std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<decltype(ops)>>>{}, lhs, rhs, ops);
    };

    auto check_store_cx = [&](auto&& lhs, auto&& rhs, auto& out, const auto& ops) {
        return [&]<uZ... I>(std::index_sequence<I...>, auto&& lhs, auto&& rhs, auto& out, const auto& ops) {
            auto single_op = [&](auto&& lhs, auto&& rhs, auto& out, const auto& op, uZ N) {
                // vector expression op
                pcx::store(op(lhs, rhs), out);

                for (uZ i = 0; i < length; ++i) {
                    // op with scalar std
                    auto v   = static_cast<std::complex<T>>(out[i]);
                    auto res = op(std_lhs[i], std_rhs[i]);

                    if (!equal_eps(res, v)) {
                        const auto* lhs_name = typeid(lhs).name();
                        const auto* rhs_name = typeid(rhs).name();
                        std::cout << "Values not equal after " << get_type_name(lhs) << "•"
                                  << get_type_name(rhs) << " operation #" << N << " for pack size "
                                  << PackSize << ".\n";
                        std::cout << "Length: " << length << " Index: " << i <<    //
                            ". Expected value: " << res <<                         //
                            ". Acquired value: " << pcx_res[i].value() << ".\n";
                        return 1;
                    }
                }
                return 0;
            };
            return (single_op(lhs, rhs, out, std::get<I>(ops), I) + ...);
        }(std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<decltype(ops)>>>{},
               lhs,
               rhs,
               out,
               ops);
    };

    auto check_subr = [&]<uZ N>(auto&& ops) {
        auto& op = std::get<N>(ops);
        pcx::subrange<T, false, PackSize>(pcx_res).assign(op(pcx_lhs, pcx_rhs));
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
    auto check_cx_scalar = [&]<uZ N>(auto&& ops) {
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

    auto check_re_scalar = [&]<uZ N>(auto&& ops) {
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

    auto check_ch = [&]() {
        auto lhs = pcx::vector<T>(length);
        auto rhs = pcx::vector<T>(length);

        auto lhs2 = pcx::vector<T>(length);
        auto rhs2 = pcx::vector<T>(length);

        check_cx(lhs, rhs, cxops);
        check_cx(lhs, rhs, cxops);
    };
    using subrange = pcx::subrange<T, false, PackSize>;
    auto sub_lhs   = subrange(pcx_lhs);
    auto sub_rhs   = subrange(pcx_rhs);
    auto sub_res   = subrange(pcx_res);


    constexpr auto bas    = pcx::md::right_basis<1, 2, 3>{};
    auto           md_lhs = pcx::md::dynamic_storage<T, bas>{8U, 8U, length};
    auto           md_rhs = pcx::md::dynamic_storage<T, bas>{8U, 8U, length};

    auto slice_lhs = md_lhs.template slice<1>(0).template slice<2>(0);
    auto slice_rhs = md_rhs.template slice<1>(1).template slice<2>(1);

    pcx::rv::copy(pcx_lhs, slice_lhs.begin());
    pcx::rv::copy(pcx_rhs, slice_rhs.begin());

    reg_type_name(pcx_lhs, "pcx::vector<" + get_type_name(T{}) + ">");
    reg_type_name(sub_lhs, "pcx::subrange<" + get_type_name(T{}) + ">");
    reg_type_name(slice_lhs, "pcx::md::slice<" + get_type_name(T{}) + ">");

    auto check_cx_ = [&]<typename... Ops>(const std::tuple<Ops...>& ops) {
        auto check_all_impl = [&]<uZ... N>(auto&& ops, std::index_sequence<N...>) {
            return check_cx(pcx_lhs, pcx_rhs, ops)                     //
                   + check_cx(sub_lhs, sub_rhs, ops)                   //
                   + check_cx(slice_lhs, slice_rhs, ops)               //
                   + check_store_cx(pcx_lhs, pcx_rhs, pcx_res, ops)    //
                   /* (check_vector.template operator()<N>(ops) + ...) + */
                   + (check_subr.template operator()<N>(ops) + ...)    //
                   + (check_cx_scalar.template operator()<N>(ops) + ...);
        };
        return check_all_impl(ops, std::index_sequence_for<Ops...>{});
    };
    auto check_re = [&]<typename... Ops>(const std::tuple<Ops...>& ops) {
        auto check_all_impl = [&]<uZ... N>(auto&& ops, std::index_sequence<N...>) {
            return (check_re_scalar.template operator()<N>(ops) + ...);
        };
        return check_all_impl(ops, std::index_sequence_for<Ops...>{});
    };

    return check_cx_(cxops) + check_re(reops);
}

template<typename T>
    requires std::floating_point<T>
int test_subrange(uZ length) {
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

    auto x  = pcx::vector<double>(128);
    auto y  = pcx::vector<double>(128);
    auto z  = x * y;
    auto ze = pcx::rv::end(z);

    reg_type_name(0.F, "float");
    reg_type_name(0., "double");

    // static_assert(pcx::rv::range<decltype(z)>);

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
            std::cout << "test_op error: " << i << "\n";
            return i;
        }
    }
    std::cout << "test_op finished successfully\n";
    return res;
}
