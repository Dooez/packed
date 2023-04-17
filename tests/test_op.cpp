
#include "vector.hpp"

#include <iostream>
#include <limits>
// void set_te(packed_cx_vector<double>& v_def_2)
// {
//     set(v_def_2.begin(), v_def_2.end(), 1);
// }

void test_repack(float* data)
{
    using namespace pcx::avx;
    auto a   = cxload<8>(data);
    auto [b] = convert<float>::split<1>(a);
    cxstore<8>(data, b);
}

void asm_test_fun(pcx::vector<double>& v1,
                  pcx::vector<double>& v2,
                  std::complex<double> v3)
{
    v1 = v1 + v2 * v1;
}
template<typename T>
    requires std::floating_point<T>
int equal_eps(T lhs, T rhs)
{
    static constexpr T epsilon = 10 * std::numeric_limits<T>::epsilon;
    if (lhs == rhs)
    {
        return true;
    }

    T largest = lhs > rhs ? lhs : rhs;
    return abs(lhs - rhs) < largest * epsilon;
}

template<typename T>
    requires std::floating_point<T>
int equal_eps(std::complex<T> lhs, std::complex<T> rhs)
{
    static constexpr T epsilon = 10 * std::numeric_limits<T>::epsilon();
    if (lhs == rhs)
    {
        return true;
    }

    T largest = abs(lhs) > abs(rhs) ? abs(lhs) : abs(rhs);
    return abs(lhs - rhs) < (largest * epsilon);
}

template<typename T1, typename T2, typename V3>
bool test_add(const std::vector<T1>& val1, const std::vector<T2>& val2, const V3& val3)
{
    for (uint i = 0; i < val1.size(); ++i)
    {
        auto stdr = val1.at(i) + val2.at(i);
        if (!equal_eps(stdr, std::complex<typename V3::real_type>(val3[i])))
        {
            std::cout << "addition failed at " << i << " with values " << stdr << " "
                      << std::complex<typename V3::real_type>(val3[i]) << " "
                      << abs(std::complex<typename V3::real_type>(val3[i]) - stdr)
                      << "\n";
            return false;
        }
    }
    return true;
}
template<typename T1, typename T2, typename V3>
bool test_sub(const std::vector<T1>& val1, const std::vector<T2>& val2, const V3& val3)
{
    for (uint i = 0; i < val1.size(); ++i)
    {
        auto stdr = val1.at(i) - val2.at(i);
        if (!equal_eps(stdr, std::complex<typename V3::real_type>(val3[i])))
        {
            std::cout << "subtraction failed at " << i << " with values" << stdr << "  "
                      << std::complex<typename V3::real_type>(val3[i]) << "\n";
            return false;
        }
    }
    return true;
}
template<typename T1, typename T2, typename V3>
bool test_mul(const std::vector<T1>& val1, const std::vector<T2>& val2, const V3& val3)
{
    for (uint i = 0; i < val1.size(); ++i)
    {
        auto stdr = val1.at(i) * val2.at(i);
        if (!equal_eps(stdr, std::complex<typename V3::real_type>(val3[i])))
        {
            std::cout << "multiplication failed at " << i << " with values" << stdr
                      << "  " << std::complex<typename V3::real_type>(val3[i]) << "\n";
            return false;
        }
    }
    return true;
}
template<typename T1, typename T2, typename V3>
bool test_div(const std::vector<T1>& val1, const std::vector<T2>& val2, const V3& val3)
{
    for (uint i = 0; i < val1.size(); ++i)
    {
        auto stdr = val1.at(i) / val2.at(i);
        if (!equal_eps(stdr, std::complex<typename V3::real_type>(val3[i])))
        {
            std::cout << "division failed at " << i << " with values" << stdr << "  "
                      << std::complex<typename V3::real_type>(val3[i]) << "\n";
            return false;
        }
    }
    return true;
}
template<typename T1, typename T2, typename V3, typename R>
bool test_compound(const std::vector<T1>& val1,
                   const std::vector<T2>& val2,
                   const V3&              val3,
                   R                      rscalar,
                   std::complex<R>        scalar)
{
    for (uint i = 0; i < val1.size(); ++i)
    {
        auto stdr = (rscalar + (scalar * val1.at(i) * rscalar)) +
                    (scalar + (rscalar / val2.at(i) * scalar)) + scalar;
        if (!equal_eps(stdr, std::complex<typename V3::real_type>(val3[i])))
        {
            std::cout << "compound failed at " << i << " with values" << stdr << "  "
                      << std::complex<typename V3::real_type>(val3[i]) << "\n";
            return false;
        }
    }
    return true;
}

template<typename T>
    requires std::floating_point<T>
int test_arithm(std::size_t length)
{
    auto vec1 = pcx::vector<T>(length);
    auto vec2 = pcx::vector<T>(length);
    auto vecr = pcx::vector<T>(length);

    auto stdvec1 = std::vector<std::complex<T>>(length);
    auto stdvec2 = std::vector<std::complex<T>>(length);
    auto stdvecr = std::vector<std::complex<T>>(length);

    for (uint i = 0; i < length; ++i)
    {
        vec1[i]       = i * std::numeric_limits<T>::epsilon();
        vec2[i]       = i + 1;
        stdvec1.at(i) = i * std::numeric_limits<T>::epsilon();
        stdvec2.at(i) = i + 1;
    }


    const T    rval = 13.123 + 13 * std::numeric_limits<T>::epsilon();
    const auto val  = std::complex<T>(rval, rval);

    auto stdrval = std::vector<T>(length, rval);
    auto stdval  = std::vector<std::complex<T>>(length, val);

    vecr = vec1 + vec2;
    if (!test_add(stdvec1, stdvec2, vecr))
    {
        return 1;
    }
    vecr = vec1 - vec2;
    if (!test_sub(stdvec1, stdvec2, vecr))
    {
        return 1;
    }

    vecr = vec1 * vec2;
    if (!test_mul(stdvec1, stdvec2, vecr))
    {
        return 1;
    }
    vecr = vec1 / vec2;
    if (!test_div(stdvec1, stdvec2, vecr))
    {
        return 1;
    }

    vecr = rval + vec2;
    if (!test_add(stdrval, stdvec2, vecr))
    {
        return 1;
    }
    vecr = rval - vec2;
    if (!test_sub(stdrval, stdvec2, vecr))
    {
        return 1;
    }
    vecr = rval * vec2;
    if (!test_mul(stdrval, stdvec2, vecr))
    {
        return 1;
    }
    vecr = rval / vec2;
    if (!test_div(stdrval, stdvec2, vecr))
    {
        return 1;
    }

    vecr = vec1 + rval;
    if (!test_add(stdvec1, stdrval, vecr))
    {
        return 1;
    }
    vecr = vec1 - rval;
    if (!test_sub(stdvec1, stdrval, vecr))
    {
        return 1;
    }
    vecr = vec1 * rval;
    if (!test_mul(stdvec1, stdrval, vecr))
    {
        return 1;
    }
    vecr = vec1 / rval;
    if (!test_div(stdvec1, stdrval, vecr))
    {
        return 1;
    }

    vecr = val + vec2;
    if (!test_add(stdval, stdvec2, vecr))
    {
        return 1;
    }
    vecr = val - vec2;
    if (!test_sub(stdval, stdvec2, vecr))
    {
        return 1;
    }
    vecr = val * vec2;
    if (!test_mul(stdval, stdvec2, vecr))
    {
        return 1;
    }
    vecr = val / vec2;
    if (!test_div(stdval, stdvec2, vecr))
    {
        return 1;
    }

    vecr = vec1 + val;
    if (!test_add(stdvec1, stdval, vecr))
    {
        return 1;
    }
    vecr = vec1 - val;
    if (!test_sub(stdvec1, stdval, vecr))
    {
        return 1;
    }
    vecr = vec1 * val;
    if (!test_mul(stdvec1, stdval, vecr))
    {
        return 1;
    }
    vecr = vec1 / val;
    if (!test_div(stdvec1, stdval, vecr))
    {
        return 1;
    }

    auto cmpnd = (rval + (val * vec1 * rval)) + (val + (rval / vec2 * val)) + val;
    vecr       = cmpnd;
    if (!test_compound(stdvec1, stdvec2, vecr, rval, val))
    {
        return 1;
    }

    return 0;
}

template<typename T>
    requires std::floating_point<T>
int test_subrange(std::size_t length)
{
    auto vec1 = pcx::vector<T>(length);
    auto vec2 = pcx::vector<T>(length);
    auto vecr = pcx::vector<T>(length);

    auto stdvec1 = std::vector<std::complex<T>>(length);
    auto stdvec2 = std::vector<std::complex<T>>(length);
    auto stdvecr = std::vector<std::complex<T>>(length);

    for (uint i = 1; i < length; ++i)
    {
        const auto v11 = pcx::subrange(vec1.begin(), i);
        auto       v12 = pcx::subrange(vec1.cbegin() + i, length - i);
        const auto v21 = pcx::subrange(vec2.begin(), i);
        const auto v22 = pcx::subrange(vec2.cbegin() + i, length - i);
        auto       vr1 = pcx::subrange(vecr.begin(), i);
        auto       vr2 = pcx::subrange(vecr.begin() + i, length - i);

        vr1.assign(v11 + v21);
        vr2.assign(v12 + v22);

        if (!test_add(stdvec1, stdvec2, vecr))
        {
            return 1;
        }
    }
    return 0;
}

int main()
{
    int res = 0;
    for (uint i = 1; i < 64; ++i)
    {
        res += test_arithm<float>(i);
        res += test_arithm<double>(i);
        if (res > 0)
        {
            std::cout << i << "\n";
            return i;
        }
    }
    for (uint i = 1; i < 64; ++i)
    {
        res += test_arithm<float>(i);
        res += test_arithm<double>(i);
        res += test_subrange<float>(i);
        res += test_subrange<double>(i);
        if (res > 0)
        {
            std::cout << i << "\n";
            return i;
        }
    }

    auto N = 8;

    auto vec = std::vector<std::complex<float>>(N);

    for (uint i = 0; i < N; ++i)
    {
        vec[i] = std::complex<float>(i, 100 + i);
    }

    test_repack(reinterpret_cast<float*>(vec.data()));

    for (auto val : vec)
    {
        std::cout << val << " ";
    }
    std::cout << "\n";

    auto t1  = std::make_tuple(1U, "2U", 3U);
    auto t2  = std::make_tuple(1.1, 2.2, 3.3);
    auto tr  = pcx::internal::zip_tuples(t1, t2);
    auto tr2 = pcx::internal::zip_tuples(t1);

    // auto tst_appl =
    //     pcx::internal::apply_for_each([](uint a, double b) { return a + b; }, t1, t2);

    pcx::internal::apply_for_each(
        [](auto a, auto b) { std::cout << a << " " << b << "\n"; },
        t1,
        t2);

    uint  a, b, c = 0;
    uint& ar   = a;
    auto  tref = std::tuple<const uint&, uint&, volatile uint&>{a, b, c};

    auto refzip = pcx::internal::zip_tuples(tref, std::move(t2));

    auto foo = [](uint a, uint b) { return a * 2; };
    // static_assert(std::invocable<decltype(foo), std::string>);
    // static_assert(pcx::internal::appliable<decltype(foo), std::tuple<std::string>> );
    // static_assert(pcx::internal::appliable_for_each<decltype(foo),
    //                                                 std::tuple<int>,
    //                                                 std::tuple<int>>);

    //     bool value = pcx::internal::is_appliable_for_each_<
    //         decltype(foo),
    //         decltype(pcx::internal::zip_tuples(std::declval<std::tuple<int>>(),
    //                                            std::declval<std::tuple<int>>())),
    //         std::make_index_sequence<std::tuple_size_v<decltype(pcx::internal::zip_tuples(
    //             std::declval<std::tuple<int>>(),
    //             std::declval<std::tuple<int>>()))>>>::value;
    //
    //     static_assert(pcx::internal::appliable<decltype(foo), std::tuple<int, int>>);

    auto vec_align_check = pcx::vector<float>(1024);
    std::cout << vec_align_check.data() << "\n";
    return res;
}
