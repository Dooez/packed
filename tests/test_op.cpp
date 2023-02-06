
#include <vector.hpp>

// void set_te(packed_cx_vector<double>& v_def_2)
// {
//     set(v_def_2.begin(), v_def_2.end(), 1);
// }


void set_te2(packed_cx_vector<double>& v1,
             packed_cx_vector<double>& v2,
             std::complex<double> v3)
{
    v1 = v2 * v3;
}

int main()
{
    packed_cx_vector<double> vec1(123);
    packed_cx_vector<double> vec2(123);
    packed_cx_vector<double> vec3(123);
    auto vecr = vec1;

    auto cxsca = std::complex<double>(13, 0);

    vecr = vec1 + 13.0 + (vec1 + cxsca);
    vecr = vec1 - 13.0 - (vec1 - cxsca);
    vecr = vec1 * 13.0 * (vec1 * cxsca);
    vecr = vec1 / 13.0 / (vec1 * cxsca);

    vecr = 13.0 + vec1  + (cxsca + vec1);
    vecr = 13.0 - vec1  - (cxsca - vec1);
    vecr = 13.0 * vec1  * (cxsca * vec1);
    vecr = 13.0 / vec1  / (cxsca / vec1);

    vec1 = vec2 + vec3;
    vec1 = vec2 - vec3;

    vec1 = vec2 * vec3;
    vec1 = vec2 / vec3;

    return 0;
}
