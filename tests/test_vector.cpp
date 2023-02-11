#include <cassert>
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

    using real_type = typename Vec::real_type;

    constexpr auto vals = std::make_tuple(13.1, 11U, 1131.3F, -.3, 1.3);

    auto allocator = vector.get_allocator();
    using Alloc    = decltype(allocator);
    auto vec_def   = Vec();
    auto vec_size  = Vec(size);
    auto vec_val   = Vec(size, std::get<0>(vals));
    if (!check_val(vec_val, std::get<0>(vals)))
    {
        return 1;
    }
    auto vec_val_u = Vec(size, std::get<1>(vals));
    if (!check_val(vec_val_u, std::get<1>(vals)))
    {
        return 1;
    }
    auto vec_val_f = Vec(size, std::get<2>(vals));
    if (!check_val(vec_val_f, std::get<2>(vals)))
    {
        return 1;
    }
    auto vec_val_d = Vec(size, std::get<3>(vals));
    if (!check_val(vec_val_d, std::get<3>(vals)))
    {
        return 1;
    }
    auto cx_val       = std::complex<real_type>(std::get<3>(vals), std::get<4>(vals));
    auto vec_val_f_cx = Vec(size, cx_val);
    if (!check_val(vec_val_f_cx, cx_val))
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
    auto size       = vector.size();
    using real_type = typename Vec::real_type;

    constexpr std::array<real_type, 2> vals{0.5533, 0.1313};

    auto allocator = vector.get_allocator();
    using Alloc    = decltype(allocator);
    auto vec_def   = Vec();
    auto vec_size  = Vec(size);
    auto vec_2     = Vec(size, vals[1]);
    auto sub       = packed_subrange(vec_size.begin(), size);
    sub.fill(vals[0]);
    if (!check_val(vec_size, vals[0]))
    {
        return 1;
    }
    auto sub2 = packed_subrange(vec_2.begin(), size);

    sub.assign(sub * sub2);
    if (!check_val(vec_size, vals[1]))
    {
        return 1;
    }

    return 0;
}

template<typename T>
constexpr void concept_test()
{
    // NOLINTNEXTLINE(*using*)
    using namespace std::ranges;
    constexpr std::size_t pack_size = 8;

    using vector_t         = packed_cx_vector<T>;
    using iterator_t       = packed_iterator<T, pack_size>;
    using cont_iterator_t  = packed_iterator<T, pack_size, true>;
    using subrange_t       = packed_subrange<T, pack_size>;
    using const_subrange_t = packed_subrange<T, pack_size, true>;

    static_assert(!view<vector_t>);
    static_assert(range<vector_t>);
    static_assert(sized_range<vector_t>);
    static_assert(input_range<vector_t>);
    static_assert(output_range<vector_t, std::complex<T>>);
    static_assert(!output_range<const vector_t, std::complex<T>>);
    static_assert(random_access_range<vector_t>);
    static_assert(common_range<vector_t>);

    static_assert(std::random_access_iterator<iterator_t>);
    static_assert(!std::output_iterator<cont_iterator_t, std::complex<T>>);

    static_assert(range<subrange_t>);
    static_assert(sized_range<subrange_t>);
    static_assert(input_range<subrange_t>);
    static_assert(output_range<subrange_t, std::complex<T>>);
    static_assert(random_access_range<subrange_t>);
    static_assert(common_range<subrange_t>);
    static_assert(viewable_range<subrange_t>);

    static_assert(range<const_subrange_t>);
    static_assert(sized_range<const_subrange_t>);
    static_assert(input_range<const_subrange_t>);
    static_assert(!output_range<const_subrange_t, std::complex<T>>);
    static_assert(random_access_range<const_subrange_t>);
    static_assert(common_range<const_subrange_t>);
    static_assert(viewable_range<const_subrange_t>);
    // static_assert(constant_range<const_subrange_t>); c++23
};


int main()
{
    concept_test<float>();
    concept_test<double>();

    int res = 0;
    for (uint i = 0; i < 128; ++i)
    {
        auto v_def = packed_cx_vector<float>(127);

        res += test_vector(v_def);
        res += test_subvector(v_def);
    }

    return res;
}