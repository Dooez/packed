#include <assert.h>
#include <iostream>
#include <vector.hpp>
template<typename T>
    requires std::random_access_iterator<T>
void asd(){};

int main()
{
    packed_cx_vector<float> v_def{};
    packed_cx_vector<float> v_def_2{};
    swap(v_def, v_def_2);
    v_def.swap(v_def_2);

    asd<packed_cx_vector<float>::iterator>();
    assert(std::random_access_iterator<packed_cx_vector<float>::iterator>);
    // packed_cx_vector<float> v_int{123};
    std::cout << "Hello world\n";
    return 0;
}