#include <assert.h>
#include <iostream>
#include <type_traits>
#include <vector.hpp>

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
bool check_val(const Vec& vector, std::complex<typename Vec::real_type> value)
{
    auto i = 0;
    for (auto val : vector)
    {
        if (abs(val.value() - value) >
            10 * std::numeric_limits<typename Vec::real_type>::epsilon())
        {
            std::cout << i << ": " << value << "\n";
            return false;
        }
        ++i;
    }
    return true;
}

template<typename Vec>
int test_vector(const Vec& vector)
{
    auto size = vector.size();

    auto allocator = vector.get_allocator();
    using Alloc    = decltype(allocator);
    auto vec_def   = Vec();
    auto vec_size  = Vec(size);
    auto vec_val   = Vec(size, 13.1);
    if (!check_val(vec_val, 13.1))
    {
        return 1;
    }
    auto vec_val_u = Vec(size, 11U);
    if (!check_val(vec_val_u, 11U))
    {
        return 1;
    }
    auto vec_val_f = Vec(size, 1131.3F);
    if (!check_val(vec_val_f, 1131.3F))
    {
        return 1;
    }
    auto vec_val_d = Vec(size, 0.3);
    if (!check_val(vec_val_d, .3))
    {
        return 1;
    }
    auto vec_val_f_cx = Vec(size, std::complex<float>(1.3, 0.3));
    if (!check_val(vec_val_f_cx, std::complex<float>(1.3, 0.3)))
    {
        return 1;
    }

    auto vec_cpy = Vec(vector);
    test_except(Vec(allocator), auto vec_alloc = Vec(allocator));
    test_except(Vec(std::move(vector)), auto vec_tmp = vector;
                auto vec_mov = Vec(std::move(vec_tmp)));
    test_except(Vec(std::move(vector), allocator), auto vec_tmp = vector;
                auto vec_mov = Vec(std::move(vec_tmp), allocator));


    return 0;
}

template<typename Vec>
int test_subvector(const Vec& vector)
{
    auto size = vector.size();

    auto allocator = vector.get_allocator();
    using Alloc    = decltype(allocator);
    auto vec_def   = Vec();
    auto vec_size  = Vec(size);
    auto vec_2     = Vec(size, 0.1313);
    auto sub       = vec_size.subrange(0, size);
    sub.fill(0.5533);
    if (!check_val(vec_size, 0.5533))
    {
        return 1;
    }
    using sub_t = std::remove_cvref_t<decltype(sub)>;
    // static_assert(std::indirectly_copyable<std::ranges::iterator_t<std::vector<int>>,
    //                                        std::vector<int>::iterator>);
    // // static_assert(std::indirectly_writable<)
    static_assert(
        std::indirectly_copyable<typename Vec::iterator, typename sub_t::iterator>);
    // std::ranges::copy(sub, vec_2.begin());

    // sub.copy(vec_2);
    return 0;
}


template<std::indirectly_readable T>
using iter_const_reference_t =
    std::common_reference_t<const std::iter_value_t<T>&&, std::iter_reference_t<T>>;

int main()
{
    using test_t = iter_const_reference_t<packed_iterator<float, 32>>;


    packed_cx_vector<float>  v_def{};
    packed_cx_vector<float>  v_def_2(127);
    packed_cx_vector<float>  v_def_3(127);
    packed_cx_vector<double> v_def_d(127);

    auto res = test_vector(v_def_2);
    res += test_subvector(v_def_2);


    if (res > 0)
    {
        return res;
    }

    test_vector(v_def_d);

    auto val_x = 0;
    for (auto val : v_def_2)
    {
        val = val_x;
        val_x += 1;
    }

    std::complex<float> val   = 1;
    const auto&         conv2 = v_def_2;
    fill(v_def_2.begin(), v_def_2.end(), val);
    packed_copy(v_def_2.begin(), v_def_2.end(), v_def_3.begin());
    //
    //     for (uint i = 0; i < v_def_2.size(); ++i)
    //     {
    //         std::cout << v_def_2[i].value() << "\n";
    //     }
    std::cout << "\n";

    std::cout << "Hello world\n";
    return 0;
}