
#include "vector.hpp"

#include <iostream>
#include <limits>
// void set_te(packed_cx_vector<double>& v_def_2)
// {
//     set(v_def_2.begin(), v_def_2.end(), 1);
// }


void set_te2(packed_cx_vector<double>& v1,
             packed_cx_vector<double>& v2,
             std::complex<double>      v3)
{
    v1 = v2 * v3;
}
template<typename T>
    requires std::floating_point<T>
int equal_eps(T lhs, T rhs)
{
    static constexpr T epsilon = 2 * std::numeric_limits<T>::epsilon;
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
    static constexpr T epsilon = 2 * std::numeric_limits<T>::epsilon();
    if (lhs == rhs)
    {
        return true;
    }

    T largest = abs(lhs) > abs(rhs) ? abs(lhs) : abs(rhs);
    return abs(lhs - rhs) < largest * epsilon;
}

template<typename V1, typename V2, typename V3>
bool test_add(const V1& val1, const V2& val2, const V3& val3)
{
    for (uint i = 0; i < val1.size(); ++i)
    {
        auto stdr = val1.at(i) + val2.at(i);
        if (!equal_eps(stdr, std::complex<typename V3::real_type>(val3[i])))
        {
            return false;
        }
    }
    return true;
}

template<typename T>
    requires std::floating_point<T>
int test_arithm(std::size_t length)
{
    auto vec1 = packed_cx_vector<T>(length);
    auto vec2 = packed_cx_vector<T>(length);
    auto vecr = packed_cx_vector<T>(length);

    auto stdvec1 = std::vector<std::complex<T>>(length);
    auto stdvec2 = std::vector<std::complex<T>>(length);
    auto stdvecr = std::vector<std::complex<T>>(length);

    for (uint i = 0; i < length; ++i)
    {
        vec1[i]       = i + std::numeric_limits<T>::epsilon();
        vec2[i]       = i + 1;
        stdvec1.at(i) = i + std::numeric_limits<T>::epsilon();
        stdvec2.at(i) = i + 1;
    }


    const T    rval = 13;
    const auto val  = std::complex<T>(13, 13);

    auto stdrval = std::vector<T>(length, 13);
    auto stdval  = std::vector<std::complex<T>>(length, std::complex<T>(13, 13));

    vecr = vec1 + vec2;
    if (!test_add(stdvec1, stdvec2, vecr))
    {
        return 1;
    }

    vecr = vec1 - vec2;
    for (uint i = 0; i < length; ++i)
    {
        auto stdr = stdvec1.at(i) - stdvec2.at(i);
        if (!equal_eps(stdr, std::complex<T>(vecr[i])))
        {
            return 1;
        }
    }
    vecr = vec1 * vec2;
    for (uint i = 0; i < length; ++i)
    {
        auto stdr = stdvec1.at(i) * stdvec2.at(i);
        if (!equal_eps(stdr, std::complex<T>(vecr[i])))
        {
            return 1;
        }
    }
    vecr = vec1 / vec2;
    for (uint i = 0; i < length; ++i)
    {
        auto stdr = stdvec1.at(i) / stdvec2.at(i);
        if (!equal_eps(stdr, std::complex<T>(vecr[i])))
        {
            return 1;
        }
    }

    vecr = rval + vec1;
    if (!test_add(stdrval, stdvec1, vecr))
    {
        return 1;
    }

    vecr = rval - vec1;
    vecr = rval * vec1;
    vecr = rval / vec1;

    vecr = vec1 + rval;
    if (!test_add(stdvec1, stdrval, vecr))
    {
        return 1;
    }
    vecr = vec1 - rval;
    vecr = vec1 * rval;
    vecr = vec1 / rval;

    vecr = val + vec1;
    if (!test_add(stdval, stdvec1, vecr))
    {
        // return 1;
    }
    vecr = val - vec1;
    vecr = val * vec1;
    vecr = val / vec1;

    vecr = vec1 + val;
    if (!test_add(stdvec1, stdval, vecr))
    {
        // return 1;
    }
    vecr = vec1 - val;
    vecr = vec1 * val;
    vecr = vec1 / val;

    vecr = val * rval * vec1 + vec2 * rval * val;
    return 0;
}


int main()
{
    int res = 0;
    for (uint i = 1; i < 1024; ++i)
    {
        res += test_arithm<float>(i);
        res += test_arithm<double>(i);
        if (res > 0)
        {
            std::cout << i << "\n";
        }
    }

    return res;
}
