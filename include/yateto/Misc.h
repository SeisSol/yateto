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
