#ifndef YATETO_MISC_H_
#define YATETO_MISC_H_

#include "Marker.h"

#include <cstddef>
#include <type_traits>

namespace yateto {

/** Computes the largest amount of temporary memory any of the given kernels
 * needs.
 *
 * @param kernels the kernels to consider; only their types matter.
 * @return the largest requirement, in bytes.
 * */
template <typename KernelType, typename... OtherKernelTypes>
YATETO_HOSTDEVICE constexpr auto getMaxTmpMemRequired(const KernelType& /*kernel*/,
                                                      const OtherKernelTypes&... /*otherKernels*/) {
  using SizeT = std::common_type_t<decltype(KernelType::TmpMaxMemRequiredInBytes),
                                   decltype(OtherKernelTypes::TmpMaxMemRequiredInBytes)...>;
  SizeT maximum = KernelType::TmpMaxMemRequiredInBytes;
  ((maximum = OtherKernelTypes::TmpMaxMemRequiredInBytes > maximum
                  ? static_cast<SizeT>(OtherKernelTypes::TmpMaxMemRequiredInBytes)
                  : maximum),
   ...);
  return maximum;
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
YATETO_HOSTDEVICE constexpr std::size_t dimSize() noexcept {
  return Tensor::Stop[Dim] - Tensor::Start[Dim];
}

template <typename Tensor>
YATETO_HOSTDEVICE constexpr std::size_t leadDim() noexcept {
  return dimSize<Tensor, 0>();
}

} // namespace yateto

#endif // YATETO_MISC_H_
