#ifndef YATETO_INITTOOLS_H_
#define YATETO_INITTOOLS_H_

#include <algorithm>

namespace yateto {
  template<class T>
  constexpr unsigned numFamilyMembers() {
    return sizeof(T::Size) / sizeof(T::Size[0]);
  }
  template<class T>
  constexpr unsigned computeFamilySize(unsigned n = numFamilyMembers<T>()) {
    return n == 0 ? 0 : T::Size[n-1] + computeFamilySize<T>(n-1);
  }
  
  template<class T, typename float_t>
  void setFamilyPtr(float_t* ptrs[]) {
    
  }

  template<class T, typename float_t>
  void copyFamilyToMemAndSetPtr(float_t*& mem, float_t* ptrs[]) {
    unsigned n = sizeof(T::Size) / sizeof(T::Size[0]);
    for (unsigned i = 0; i < n; ++i) {
      ptrs[i] = mem;
      mem = std::copy(T::Values[i], T::Values[i] + T::Size[i], mem);
    }
  }
}

#endif
