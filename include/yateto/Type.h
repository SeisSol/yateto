#ifndef YATETO_TYPE_H_
#define YATETO_TYPE_H_

#include <cstddef>

// C++23 include
#if __has_include(<stdfloat>)
#include <stdfloat>
#endif

namespace yateto {

// NOTE: the compiler is asked about these types, not <cfloat>. `FLT16_MIN`
//       and `FLT128_MIN` are only declared when `__STDC_WANT_IEC_60559_TYPES_EXT__`
//       is defined *before the first* <cfloat>, and this header is in no
//       position to promise that: including <libxsmm.h> ahead of it pulls
//       <cfloat> in first, the second include is a no-op against its guard,
//       and the tests then fall through to types the compiler may not have --
//       `__fp16` is an ARM spelling and x86-64 GCC has no such thing.
//       `__FLT16_MANT_DIG__` and `__FLT128_MANT_DIG__` are predefined by the
//       compiler and say the same thing regardless of what was included.

#ifdef __STDCPP_FLOAT128_T__
using f128_ty = std::float128_t;
#elif defined(__FLT128_MANT_DIG__)
using f128_ty = _Float128;
#else
using f128_ty = __float128;
#endif
#ifdef __STDCPP_FLOAT16_T__
using f16_ty = std::float16_t;
#elif defined(__FLT16_MANT_DIG__)
using f16_ty = _Float16;
#else
using f16_ty = __fp16;
#endif
#ifdef __STDCPP_BFLOAT16_T__
using bf16_ty = std::bfloat16_t;
#else
using bf16_ty = __bf16;
#endif

} // namespace yateto

#endif // YATETO_TYPE_H_
