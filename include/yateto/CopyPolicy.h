#ifndef YATETO_COPY_POLICY_H_
#define YATETO_COPY_POLICY_H_

#include <algorithm>

namespace yateto {
  template <typename float_t>
  class SimpleCopyPolicy {
  public:
    float_t* copy(float_t const* first, float_t const* last, float_t*& mem) {
      mem = std::copy(first, last, mem);
      return mem;
    }
  };
}

#endif  // YATETO_COPY_POLICY_H_
