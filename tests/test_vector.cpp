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
    packed_cx_vector<float> v_def_2{};
    swap(v_def, v_def_2);
    v_def.swap(v_def_2);

    asd<packed_cx_vector<float>::iterator>();
    auto it = v_def.begin();
    auto a = it.value();
    auto it2 = vovo.begin();
    auto b = *it2;
    // packed_cx_vector<float> v_int{123};
    std::cout << "Hello world\n";
    return 0;
}