#ifndef YATETO_TYPE_H_
#define YATETO_TYPE_H_

#include <cstddef>

// C++23 include
#if __has_include(<stdfloat>)
#include <stdfloat>
#endif

// cf. https://stackoverflow.com/a/70868019
#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <cfloat>

namespace yateto {

#ifdef __STDCPP_FLOAT128_T__
using f128_ty = std::float128_t;
#elif defined(FLT128_MIN)
using f128_ty = _Float128;
#else
using f128_ty = __float128;
#endif
#ifdef __STDCPP_FLOAT16_T__
using f16_ty = std::float16_t;
#elif defined(FLT16_MIN)
using f16_ty = _Float16;
#else
using f16_ty = __fp16;
#endif
#ifdef __STDCPP_BFLOAT16_T__
using bf16_ty = std::bfloat16_t;
#else
using bf16_ty = __bf16;
#endif

} // yateto

#endif // YATETO_TYPE_H_
