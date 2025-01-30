#ifndef YATETO_LINEAR_ALLOCATED_H_
#define YATETO_LINEAR_ALLOCATED_H_

#include <cassert>
#include <cstddef>

namespace yateto {
template<typename T>
struct LinearAllocatorT {
public:
  void initialize(T* ptr) {
      isInit = true;
      userSpaceMem = ptr;
  }

  T* allocate(size_t size) {
      assert(isInit && "YATETO: Temporary-Memory manager hasn't been initialized");
      int currentByteCount = byteCount;
      byteCount += size;
      return &userSpaceMem[currentByteCount];
  }

  void free() {
      isInit = false;
      byteCount = 0;
      userSpaceMem = nullptr;
  }

private:
  size_t byteCount{0};
  bool isInit{false};
  T *userSpaceMem{nullptr};
};
} // yateto
#endif // YATETO_LINEAR_ALLOCATED_H_