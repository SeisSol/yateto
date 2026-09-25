#ifndef YATETO_COPYPOLICY_H_
#define YATETO_COPYPOLICY_H_

#include <algorithm>

namespace yateto {
template <typename float_t>
class SimpleCopyPolicy {
  public:
  static float_t* copy(const float_t* first, const float_t* last, float_t*& mem) {
    mem = std::copy(first, last, mem);
    return mem;
  }
};
} // namespace yateto

#endif // YATETO_COPYPOLICY_H_
