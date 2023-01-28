#include "fft.hpp"
#include "vector.hpp"

#include <assert.h>
#include <iostream>


inline std::size_t n_caught = 0;
#define test_except(test, expression) \
    {                                 \
        if constexpr (noexcept(test)) \
        {                             \
            expression;               \
        } else                        \
        {                             \
            try                       \
            {                         \
                expression;           \
            } catch (...)             \
            {                         \
                ++n_caught;           \
            }                         \
        }                             \
    }

template<typename Vec>
int test_vector(const Vec& vector)
{
    auto size = vector.size();

    auto allocator    = vector.get_allocator();
    using Alloc       = decltype(allocator);
    auto vec_def      = Vec();
    auto vec_size     = Vec(size);
    auto vec_val      = Vec(size, 0);
    auto vec_val_u    = Vec(size, 0U);
    auto vec_val_f    = Vec(size, 0.F);
    auto vec_val_d    = Vec(size, 0.);
    auto vec_val_f_cx = Vec(size, std::complex<float>(0, 0));
    auto vec_cpy      = Vec(vector);
    test_except(Vec(allocator), auto vec_alloc = Vec(allocator));
    test_except(Vec(std::move(vector)), auto vec_tmp = vector;
                auto vec_mov = Vec(std::move(vec_tmp)));
    test_except(Vec(std::move(vector), allocator), auto vec_tmp = vector;
                auto vec_mov = Vec(std::move(vec_tmp), allocator));


    return 0;
}


int main()
{
    packed_cx_vector<float> v_def{};
    packed_cx_vector<float> v_def_2(127);
    packed_cx_vector<float> v_def_3(127);

    test_vector(v_def_2);
    auto val_x = 0;
    for (auto val : v_def_2)
    {
        val = val_x;
        val_x += 1;
    }

    auto test_unit = fft_unit<float, 32>(v_def_2);


    std::complex<float> val   = 1;
    const auto&         conv2 = v_def_2;
    set(v_def_2.begin(), v_def_2.end(), val);
    packed_copy(v_def_2.begin(), v_def_2.end(), v_def_3.begin());

    for (uint i = 0; i < v_def_2.size(); ++i)
    {
        std::cout << v_def_2[i].value() << "\n";
    }
    std::cout << "\n";

    std::cout << "Hello world\n";
    return 0;
}