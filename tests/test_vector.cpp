#include <assert.h>
#include <iostream>
#include <vector.hpp>
template<typename T>
    requires std::random_access_iterator<T>
void asd(){};

int main()
{
    std::vector<float> vovo(12);

    packed_cx_vector<float> v_def{};
    packed_cx_vector<float> v_def_2(127);
    // swap(v_def, v_def_2);
    // v_def.swap(v_def_2);
    packed_cx_vector<float> v_def_3(127);

    asd<packed_cx_vector<float>::iterator>();

    auto val_x = 0;
    for (auto val : v_def_2)
    {
        val = val_x;
        val_x += 1;
    }
//
//
//     for (uint i = 0; i < v_def_3.size(); ++i)
//     {
//         std::cout << v_def_3[i].value() << "\n";
//     }
    // const auto& v_def_2c = v_def_2;
    // auto a1 = v_def_2.begin();
    // auto a2 = v_def_2.end();
    std::complex<float> val = 1;
    set(v_def_2.begin(), v_def_2.end(), val);

    for (uint i = 0; i < v_def_2.size(); ++i)
    {
        std::cout << v_def_2[i].value() << "\n";
    }
    std::cout << "\n";

    std::cout << "Hello world\n";
    return 0;
}