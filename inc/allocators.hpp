#ifndef ALLOCATORS_HPP
#define ALLOCATORS_HPP

#include "types.hpp"

#include <cassert>
#include <type_traits>

namespace pcxo {

template<typename T, std::align_val_t Alignment = std::align_val_t{64}>
class aligned_allocator {
public:
    using value_type      = T;
    using is_always_equal = std::true_type;

    aligned_allocator() noexcept = default;

    template<typename U>
    explicit aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {};

    aligned_allocator(const aligned_allocator&) noexcept = default;
    aligned_allocator(aligned_allocator&&) noexcept      = default;

    ~aligned_allocator() = default;

    aligned_allocator& operator=(const aligned_allocator&) noexcept = default;
    aligned_allocator& operator=(aligned_allocator&&) noexcept      = default;

    [[nodiscard]] auto allocate(uZ n) -> value_type* {
        return reinterpret_cast<value_type*>(::operator new[](n * sizeof(value_type), Alignment));
    }

    void deallocate(value_type* p, uZ) {
        ::operator delete[](reinterpret_cast<void*>(p), Alignment);
    }

    template<typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

private:
};
template<typename T, std::align_val_t Alignment>
bool operator==(const aligned_allocator<T, Alignment>&, const aligned_allocator<T, Alignment>&) noexcept {
    return true;
}

template<typename T>
class null_allocator {
public:
    using value_type      = T;
    using is_always_equal = std::true_type;

    null_allocator() noexcept = default;

    template<typename U>
    explicit null_allocator(const null_allocator<U>&) noexcept {};

    null_allocator(const null_allocator&) noexcept = default;
    null_allocator(null_allocator&&) noexcept      = default;

    ~null_allocator() = default;

    null_allocator& operator=(const null_allocator&) noexcept = default;
    null_allocator& operator=(null_allocator&&) noexcept      = default;

    [[nodiscard]] auto allocate(uZ) -> value_type* {
        return nullptr;
    }

    void deallocate(value_type* p, uZ) {
        assert(p == nullptr);
    }

    template<typename U>
    struct rebind {
        using other = null_allocator<U>;
    };

private:
};
template<typename T>
bool operator==(const null_allocator<T>&, const null_allocator<T>&) noexcept {
    return true;
}

}    // namespace pcx

#endif
