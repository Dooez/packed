#include <iostream>
#include <vector.hpp>

int main()
{
    packed_cx_vector<float> v_def{};
    packed_cx_vector<float> v_def_2{};
    swap(v_def, v_def_2);
    v_def.swap(v_def_2);
    // packed_cx_vector<float> v_int{123};
    std::cout << "Hello world\n";
    return 0;
}