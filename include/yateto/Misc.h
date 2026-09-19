#ifndef YATETO_MISC_H_
#define YATETO_MISC_H_

#include <algorithm>
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
constexpr auto getMaxTmpMemRequired(const KernelType& /*kernel*/,
                                    const OtherKernelTypes&... /*otherKernels*/) {
  using SizeT = std::common_type_t<decltype(KernelType::TmpMaxMemRequiredInBytes),
                                   decltype(OtherKernelTypes::TmpMaxMemRequiredInBytes)...>;
  SizeT maximum = KernelType::TmpMaxMemRequiredInBytes;
  ((maximum = std::max<SizeT>(maximum, OtherKernelTypes::TmpMaxMemRequiredInBytes)), ...);
  return maximum;
}

template <typename Tensor, int Dim>
constexpr std::size_t dimSize() noexcept {
  return Tensor::Stop[Dim] - Tensor::Start[Dim];
}

template <typename Tensor>
constexpr std::size_t leadDim() noexcept {
  return dimSize<Tensor, 0>();
}

} // namespace yateto

#endif // YATETO_MISC_H_
