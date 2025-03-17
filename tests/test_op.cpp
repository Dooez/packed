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

using pcxo::uZ;

void test_repack(float* data) {
    using namespace pcxo::simd;
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
auto test_multi(pcxo::simd::cx_reg<float, false> a1,
                pcxo::simd::cx_reg<float, false> a2,
                pcxo::simd::cx_reg<float, false> a3,
                pcxo::simd::cx_reg<float, false> a4) {
    return pcxo::simd::mul_pairs(a1, a2, a3, a4, a2, a3);
}

auto test_non_multi(pcxo::simd::cx_reg<float, false> a1,
                    pcxo::simd::cx_reg<float, true>  a2,
                    pcxo::simd::cx_reg<float, false> a3,
                    pcxo::simd::cx_reg<float, false> a4) {
    return std::make_tuple(pcxo::simd::mul(a1, a2), pcxo::simd::mul(a3, a4), pcxo::simd::mul(a2, a3));
}

void asm_test_fun(pcxo::vector<double>& v1, pcxo::vector<double>& v2, std::complex<double> v3) {
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
    requires pcxo::packed_floating_point<T, PackSize>
int test_bin_ops(uZ length) {
    auto pcx_lhs = pcxo::vector<T, PackSize, pcxo::aligned_allocator<T>>(length);
    auto pcx_rhs = pcxo::vector<T, PackSize, pcxo::aligned_allocator<T>>(length);
    auto pcx_res = pcxo::vector<T, PackSize, pcxo::aligned_allocator<T>>(length);

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

    constexpr auto cxops = std::make_tuple(    //
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

    auto check_cx_s = [length]<bool Assign>(const auto& lhs,
                                            const auto& rhs,
                                            auto&       out,
                                            const auto& lhs_control,
                                            const auto& rhs_control,
                                            const auto& ops,
                                            std::bool_constant<Assign>) {
        return [&]<uZ... I>(std::index_sequence<I...>) {
            auto single_op = [&](const auto& op, uZ op_id) {
                constexpr auto get_control = []<typename CT>(const CT& control_v, uZ i) {
                    if constexpr (std::same_as<CT, std::complex<T>> or std::same_as<CT, T>) {
                        return control_v;
                    } else if constexpr (std::same_as<CT, std::vector<std::complex<T>>>) {
                        return control_v[i];
                    } else {
                        static_assert(false);
                    }
                };

                if constexpr (!Assign) {
                    store(op(lhs, rhs), out);
                } else {
                    out = op(lhs, rhs);
                }

                for (uZ i = 0; i < length; ++i) {
                    // op with scalar std
                    auto v   = static_cast<std::complex<T>>(out[i]);
                    auto res = op(get_control(lhs_control, i), get_control(rhs_control, i));

                    if (!equal_eps(res, v)) {
                        const auto* lhs_name = typeid(lhs).name();
                        const auto* rhs_name = typeid(rhs).name();
                        std::cout << "Values not equal after " << get_type_name(lhs) << "•"
                                  << get_type_name(rhs) << " operation #" << op_id << " for pack size "
                                  << PackSize << ".\n";
                        std::cout << "Length: " << length << " Index: " << i <<    //
                            ". Expected value: " << res <<                         //
                            ". Acquired value: " << v << ".\n";
                        return 1;
                    }
                }
                return 0;
            };
            return (single_op(std::get<I>(ops), I) + ...);
        }(std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<decltype(ops)>>>{});
    };

    using subrange = pcxo::subrange<T, false, PackSize>;
    auto sub_lhs   = subrange(pcx_lhs);
    auto sub_rhs   = subrange(pcx_rhs);
    auto sub_res   = subrange(pcx_res);


    constexpr auto r_bas       = pcxo::md::right_basis<1, 2, 3>{};
    constexpr auto l_bas       = pcxo::md::left_basis<1, 2, 3>{};
    auto           r_mdstorage = pcxo::md::dynamic_storage<T, r_bas>{8U, 8U, length};
    auto           l_mdstorage = pcxo::md::dynamic_storage<T, l_bas>{8U, 8U, length};

    auto lslice_lhs = r_mdstorage.template slice<1>(0).template slice<2>(0);
    auto lslice_rhs = r_mdstorage.template slice<1>(0).template slice<2>(1);
    auto lslice_res = r_mdstorage.template slice<1>(0).template slice<2>(3);

    auto rslice_lhs = r_mdstorage.template slice<1>(0).template slice<2>(0);
    auto rslice_rhs = r_mdstorage.template slice<1>(0).template slice<2>(1);
    auto rslice_res = r_mdstorage.template slice<1>(0).template slice<2>(3);

    pcxo::rv::copy(pcx_lhs, lslice_lhs.begin());
    pcxo::rv::copy(pcx_rhs, lslice_rhs.begin());
    pcxo::rv::copy(pcx_lhs, rslice_lhs.begin());
    pcxo::rv::copy(pcx_rhs, rslice_rhs.begin());

    reg_type_name(pcx_lhs, "pcxo::vector<" + get_type_name(T{}) + ">");
    reg_type_name(sub_lhs, "pcxo::subrange<" + get_type_name(T{}) + ">");
    reg_type_name(lslice_lhs, "pcxo::md::slice<" + get_type_name(T{}) + ">");

    return check_cx_s(pcx_lhs, pcx_rhs, pcx_res, std_lhs, std_rhs, cxops, std::true_type{})               //
           + check_cx_s(sub_lhs, sub_rhs, sub_res, std_lhs, std_rhs, cxops, std::false_type{})            //
           + check_cx_s(lslice_lhs, lslice_rhs, lslice_res, std_lhs, std_rhs, cxops, std::true_type{})    //
           + check_cx_s(rslice_lhs, rslice_rhs, rslice_res, std_lhs, std_rhs, cxops, std::true_type{})    //
           + check_cx_s(pcx_lhs, cx_scalar, pcx_res, std_lhs, cx_scalar, cxops, std::true_type{})         //
           + check_cx_s(cx_scalar, pcx_rhs, pcx_res, cx_scalar, std_rhs, cxops, std::true_type{})         //
           + check_cx_s(pcx_lhs, re_scalar, pcx_res, std_lhs, re_scalar, reops, std::true_type{})         //
           + check_cx_s(re_scalar, pcx_rhs, pcx_res, re_scalar, std_rhs, reops, std::true_type{})         //
        ;
}

int main() {
    int res = 0;

    auto x  = pcxo::vector<double>(128);
    auto y  = pcxo::vector<double>(128);
    auto z  = x * y;
    auto ze = pcxo::rv::end(z);

    reg_type_name(0.F, "float");
    reg_type_name(0., "double");

    // static_assert(pcxo::rv::range<decltype(z)>);

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
