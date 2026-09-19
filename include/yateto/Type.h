#ifndef YATETO_TYPE_H_
#define YATETO_TYPE_H_

// C++23 include
#if __has_include(<stdfloat>)
#include <stdfloat>
#endif

// NOTE: the compiler is asked about these types, not <cfloat>. `FLT16_MIN`
//       and `FLT128_MIN` are only declared when `__STDC_WANT_IEC_60559_TYPES_EXT__`
//       is defined *before the first* <cfloat>, and this header is in no
//       position to promise that: including <libxsmm.h> ahead of it pulls
//       <cfloat> in first, the second include is a no-op against its guard,
//       and the tests then fall through to types the compiler may not have.
//       The `__*_MANT_DIG__` macros are predefined by the compiler and say the
//       same thing regardless of what was included.
//
// NOTE: every spelling below is guarded by something that is only true where
//       the spelling works, and there is no last resort: a target without a
//       given format leaves the alias undeclared and its `YATETO_HAS_*` macro
//       at 0, rather than making this header fail to parse. Consumers that can
//       use a format have to ask for it.
//
//       `YATETO_F128_C` renders a literal of whatever `f128_ty` turned out to
//       be. The suffix belongs to the spelling and not to the target: GCC
//       takes `f128` but rejects `q` outside its own dialect, clang takes `q`
//       but not `f128`, and `long double` takes neither.

#if defined(__STDCPP_FLOAT128_T__)
#define YATETO_HAS_F128 1
#define YATETO_F128_C(literal) literal##f128
#elif defined(__FLT128_MANT_DIG__)
#define YATETO_HAS_F128 1
#define YATETO_F128_C(literal) literal##f128
#elif defined(__SIZEOF_FLOAT128__)
#define YATETO_HAS_F128 1
#define YATETO_F128_C(literal) literal##q
#elif defined(__LDBL_MANT_DIG__) && __LDBL_MANT_DIG__ >= 113
#define YATETO_HAS_F128 1
#define YATETO_F128_C(literal) literal##L
#else
#define YATETO_HAS_F128 0
#endif

#if defined(__STDCPP_FLOAT16_T__)
#define YATETO_HAS_F16 1
#elif defined(__FLT16_MANT_DIG__)
#define YATETO_HAS_F16 1
#elif defined(__clang__) || defined(__ARM_FP16_FORMAT_IEEE)
// `__fp16` is an ARM spelling that clang accepts on every target and GCC only
// where the ARM format macros are set; x86-64 GCC has no such thing.
#define YATETO_HAS_F16 1
#else
#define YATETO_HAS_F16 0
#endif

#if defined(__STDCPP_BFLOAT16_T__)
#define YATETO_HAS_BF16 1
#elif defined(__BFLT16_MANT_DIG__)
#define YATETO_HAS_BF16 1
#elif defined(__clang__) && defined(__FLT16_MANT_DIG__)
// clang predefines no macro for `__bf16`. It offers the type on the targets on
// which it also offers `_Float16`, so that is what gets asked; being wrong here
// costs the format, not the build.
#define YATETO_HAS_BF16 1
#else
#define YATETO_HAS_BF16 0
#endif

namespace yateto {

#if defined(__STDCPP_FLOAT128_T__)
using f128_ty = std::float128_t;
#elif defined(__FLT128_MANT_DIG__)
using f128_ty = _Float128;
#elif defined(__SIZEOF_FLOAT128__)
using f128_ty = __float128;
#elif YATETO_HAS_F128
using f128_ty = long double;
#endif

#if defined(__STDCPP_FLOAT16_T__)
using f16_ty = std::float16_t;
#elif defined(__FLT16_MANT_DIG__)
using f16_ty = _Float16;
#elif YATETO_HAS_F16
using f16_ty = __fp16;
#endif

#if defined(__STDCPP_BFLOAT16_T__)
using bf16_ty = std::bfloat16_t;
#elif YATETO_HAS_BF16
using bf16_ty = __bf16;
#endif

} // namespace yateto

#endif // YATETO_TYPE_H_
