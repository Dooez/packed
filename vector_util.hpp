#ifndef VECTOR_UTIL_HPP
#define VECTOR_UTIL_HPP

template<std::size_t N>
concept power_of_two = requires { (N & (N - 1)) == 0; };

template<typename T, std::size_t PackSize>
concept packed_floating_point = std::floating_point<T> && power_of_two<PackSize> &&
                                requires { PackSize >= 64 / sizeof(T); };

template<typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize>
class packed_cx_vector;

template<typename T, std::size_t PackSize, typename Allocator, bool Const>
class packed_cx_ref;


template<typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize>
void aligned_copy(packed_cx_ref<T, PackSize, Allocator, false>         dest,
                  packed_cx_ref<T, PackSize, Allocator, true>          source,
                  typename std::allocator_traits<Allocator>::size_type size){


};

template<typename T, std::size_t PackSize, typename Allocator>
    requires packed_floating_point<T, PackSize>
void aligned_copy(typename packed_cx_vector<T, PackSize, Allocator>::const_iterator first,
                  typename packed_cx_vector<T, PackSize, Allocator>::const_iterator last,
                  typename packed_cx_vector<T, PackSize, Allocator>::iterator d_first){


};

#endif