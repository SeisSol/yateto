#ifndef YATETO_INITTOOLS_H_
#define YATETO_INITTOOLS_H_

#include <algorithm>

namespace yateto {
  template<class T>
  constexpr size_t numFamilyMembers() {
    return sizeof(T::Size) / sizeof(T::Size[0]);
  }
  
  template<typename int_t>
  constexpr size_t alignedUpper(int_t size, size_t alignment) {    
    return size + (alignment - size % alignment) % alignment;
  }
  
  template<typename float_t>
  constexpr size_t alignedReals(size_t alignment) {
    return alignment / sizeof(float_t);
  }

  template<class T>
  constexpr size_t computeFamilySize(size_t alignedReals = 1, size_t n = numFamilyMembers<T>()) {
    return n == 0 ? 0 : alignedUpper(T::Size[n-1], alignedReals) + computeFamilySize<T>(alignedReals, n-1);
  }

  template<class T, typename float_t>
  void copyFamilyToMemAndSetPtr(float_t*& mem, float_t* ptrs[], size_t alignment = 1) {
    size_t n = sizeof(T::Size) / sizeof(T::Size[0]);
    for (size_t i = 0; i < n; ++i) {
      ptrs[i] = mem;
      mem = std::copy(T::Values[i], T::Values[i] + T::Size[i], mem);
      mem += (alignedUpper(reinterpret_cast<uintptr_t>(mem), alignment) - reinterpret_cast<uintptr_t>(mem)) / sizeof(float_t);
      assert(reinterpret_cast<uintptr_t>(mem) % alignment == 0);
    }
  }
}

#endif
