#ifndef TEST_PCX_HPP
#define TEST_PCX_HPP
#include <concepts>

template<uint k_epsilon = 10, typename T = double>
    requires std::floating_point<T> bool
equal_eps(T lhs, T rhs)
{
    static constexpr T epsilon = k_epsilon * std::numeric_limits<T>::epsilon();
    if (lhs == rhs)
    {
        return true;
    }

    T largest = abs(lhs) > abs(rhs) ? abs(lhs) : abs(rhs);
    return abs(lhs - rhs) < largest * epsilon;
}

template<uint k_epsilon = 10, typename T = double>
    requires std::floating_point<T> bool
equal_eps(std::complex<T> lhs, std::complex<T> rhs)
{
    static constexpr T epsilon = k_epsilon * std::numeric_limits<T>::epsilon();
    if (lhs == rhs)
    {
        return true;
    }

    T largest = abs(lhs) > abs(rhs) ? abs(lhs) : abs(rhs);
    return abs(lhs - rhs) < (largest * epsilon);
}


#endif