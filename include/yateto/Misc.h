#ifndef YATETO_MISC_H_
#define YATETO_MISC_H_

namespace yateto {

template<typename KernelType>
constexpr auto getMaxTmpMemRequired(KernelType& krnl) {
  return KernelType::TmpMaxMemRequiredInBytes;
}

template<typename KernelType, typename... OtherKernelTypes>
constexpr auto getMaxTmpMemRequired(KernelType& krnl,
                                    OtherKernelTypes&... otherKrnls) {
  constexpr auto currentTmpMem = KernelType::TmpMaxMemRequiredInBytes;
  constexpr auto otherTmpMem = getMaxTmpMemRequired(otherKrnls...);
  return (currentTmpMem > otherTmpMem) ? currentTmpMem : otherTmpMem;
}

template <typename Tensor, int Dim>
constexpr size_t dimSize() noexcept {
  return Tensor::Stop[Dim] - Tensor::Start[Dim];
}

template <typename Tensor>
constexpr size_t leadDim() noexcept {
  return dimSize<Tensor, 0>();
}

} // yateto

#endif // YATETO_MISC_H_