#ifndef YATETO_MISC_H_
#define YATETO_MISC_H_

#include <cstddef>

namespace yateto {

template <typename KernelType>
auto getMaxTmpMemRequired(KernelType& krnl) {
  return KernelType::TmpMaxMemRequiredInBytes;
}

template <typename KernelType, typename... OtherKernelTypes>
auto getMaxTmpMemRequired(KernelType& krnl, OtherKernelTypes&... otherKrnls) {
  auto currentTmpMem = KernelType::TmpMaxMemRequiredInBytes;
  auto otherTmpMem = getMaxTmpMemRequired(otherKrnls...);
  return (currentTmpMem > otherTmpMem) ? currentTmpMem : otherTmpMem;
}

/// Smaller of two values, of possibly different types.
///
/// std::min and std::max take both arguments by the same type, so a kernel in
/// single precision that compares against a literal does not compile: the
/// literal is a double and there is no overload. The result takes the type of
/// the first argument, which is the type the surrounding expression is in --
/// the second is converted to it, exactly for a literal.
template <typename T, typename U>
constexpr T min(T first, U second) noexcept {
  const auto other = static_cast<T>(second);
  return first < other ? first : other;
}

/// Larger of two values, of possibly different types. See min.
template <typename T, typename U>
constexpr T max(T first, U second) noexcept {
  const auto other = static_cast<T>(second);
  return first > other ? first : other;
}

template <typename Tensor, int Dim>
constexpr size_t dimSize() noexcept {
  return Tensor::Stop[Dim] - Tensor::Start[Dim];
}

template <typename Tensor>
constexpr size_t leadDim() noexcept {
  return dimSize<Tensor, 0>();
}

} // namespace yateto

#endif // YATETO_MISC_H_
