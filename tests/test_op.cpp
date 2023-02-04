
#include <vector.hpp>

// void set_te(packed_cx_vector<double>& v_def_2)
// {
//     set(v_def_2.begin(), v_def_2.end(), 1);
// }


void set_te2(packed_cx_vector<double>& v1,
             packed_cx_vector<double>& v2,
             packed_cx_vector<double>& v3)
{
    v1 = v2 * v3 + v1 * v2;
}

int main()
{
    packed_cx_vector<double> vec1(123);
    packed_cx_vector<double> vec2(123);
    packed_cx_vector<double> vec3(123);

    auto s  = vec1 + 13.0;
    // auto s2 = vec2 + std::complex<double>(13.0, 1);

    vec1 = vec2 + vec3;


    return 0;
}
