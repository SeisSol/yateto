#ifndef YATETO_LINEARALLOCATOR_H_
#define YATETO_LINEARALLOCATOR_H_

#include <cassert>
#include <cstddef>

namespace yateto {
template <typename T>
struct LinearAllocatorT {
  public:
  /** Hands the manager the block it is to carve up.
   *
   *  @param ptr the start of the block.
   *  @param capacity how much of it there is. Pass it to have the manager
   *         notice when a consumer asks for more than was reserved; zero
   *         leaves the manager without an end to compare against.
   * */
  void initialize(T* ptr, std::size_t capacity = 0) {
    isInit = true;
    userSpaceMem = ptr;
    byteCapacity = capacity;
  }

  template <typename S>
  void initialize(S* ptr, std::size_t capacity = 0) {
    isInit = true;
    userSpaceMem = reinterpret_cast<T*>(ptr);
    byteCapacity = capacity;
  }

  T* allocate(std::size_t size) {
    assert(isInit && "YATETO: Temporary-Memory manager hasn't been initialized");
    const std::size_t offset = byteCount;
    byteCount += size;
    assert((byteCapacity == 0 || byteCount <= byteCapacity) &&
           "YATETO: the temporary memory block is smaller than the kernel needs");
    return userSpaceMem + offset;
  }

  void free() {
    isInit = false;
    byteCount = 0;
    byteCapacity = 0;
    userSpaceMem = nullptr;
  }

  private:
  std::size_t byteCount{0};
  std::size_t byteCapacity{0};
  bool isInit{false};
  T* userSpaceMem{nullptr};
};
} // namespace yateto
#endif // YATETO_LINEARALLOCATOR_H_
