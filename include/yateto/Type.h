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

// NOTE: which spelling exists and whether the current pass may use it are
//       two questions. The cascades below answer the first one. The second is
//       answered by the YATETO_HAS_* macros, which a build may define itself
//       -- the detection is only a default, and a conservative one.
//
//       Why that escape hatch: for a CUDA device pass the answer is a
//       function of the compiler rather than of the architecture, and it has
//       moved. Measured against NVPTX with a stand-in for the toolkit
//       headers, emitting code rather than only parsing:
//
//         clang 18   sm_70 .. sm_90a   the declaration is already rejected
//         clang 20   sm_70 .. sm_120   declaration taken, every expression
//                                      rejected by the backend: "target
//                                      'nvptx64-nvidia-cuda' does not
//                                      support it"
//
//       So the default is off for that pass. A toolchain where it works --
//       nvcc is reported to from sm_100, which is not something these tests
//       can confirm -- turns it on by defining YATETO_HAS_F128=1, or the
//       condition below grows an architecture bound.
//
//       AMDGCN needs no exception: gfx900, gfx906, gfx90a, gfx942 and gfx1100
//       all emit code under clang 18 and clang 20, lowered inline without a
//       soft-float runtime.
//
//       The predefined macros are no help on their own here. In a device pass
//       they describe the host: __SIZEOF_FLOAT128__ is set while compiling
//       for NVPTX from an x86-64 host and absent from an aarch64-darwin host,
//       in both compilers, whether or not the device can use the type.

#if defined(__STDCPP_FLOAT128_T__)
#define YATETO_F128_TY std::float128_t
#define YATETO_F128_C(literal) literal##f128
#elif defined(__FLT128_MANT_DIG__)
#define YATETO_F128_TY _Float128
#define YATETO_F128_C(literal) literal##f128
#elif defined(__SIZEOF_FLOAT128__)
#define YATETO_F128_TY __float128
#define YATETO_F128_C(literal) literal##q
#elif defined(__LDBL_MANT_DIG__) && __LDBL_MANT_DIG__ >= 113
#define YATETO_F128_TY long double
#define YATETO_F128_C(literal) literal##L
#endif

#if !defined(YATETO_HAS_F128)
#if defined(YATETO_F128_TY) && !defined(__CUDA_ARCH__)
#define YATETO_HAS_F128 1
#else
#define YATETO_HAS_F128 0
#endif
#endif

#if defined(__STDCPP_FLOAT16_T__)
#define YATETO_F16_TY std::float16_t
#elif defined(__FLT16_MANT_DIG__)
#define YATETO_F16_TY _Float16
#elif defined(__clang__) || defined(__ARM_FP16_FORMAT_IEEE)
// `__fp16` is an ARM spelling that clang accepts on every target and GCC only
// where the ARM format macros are set; x86-64 GCC has no such thing.
#define YATETO_F16_TY __fp16
#endif

#if !defined(YATETO_HAS_F16)
#if defined(YATETO_F16_TY)
#define YATETO_HAS_F16 1
#else
#define YATETO_HAS_F16 0
#endif
#endif

#if defined(__STDCPP_BFLOAT16_T__)
#define YATETO_BF16_TY std::bfloat16_t
#elif defined(__BFLT16_MANT_DIG__)
#define YATETO_BF16_TY __bf16
#elif defined(__clang__) && defined(__FLT16_MANT_DIG__)
// clang predefines no macro for `__bf16`. It offers the type on the targets on
// which it also offers `_Float16`, so that is what gets asked; being wrong here
// costs the format, not the build.
#define YATETO_BF16_TY __bf16
#endif

#if !defined(YATETO_HAS_BF16)
#if defined(YATETO_BF16_TY)
#define YATETO_HAS_BF16 1
#else
#define YATETO_HAS_BF16 0
#endif
#endif

#if YATETO_HAS_F128 && !defined(YATETO_F128_TY)
#error "YATETO_HAS_F128 is set, but this compiler offers no 128 bit floating point type"
#endif
#if YATETO_HAS_F16 && !defined(YATETO_F16_TY)
#error "YATETO_HAS_F16 is set, but this compiler offers no 16 bit floating point type"
#endif
#if YATETO_HAS_BF16 && !defined(YATETO_BF16_TY)
#error "YATETO_HAS_BF16 is set, but this compiler offers no bfloat16 type"
#endif

namespace yateto {

#if YATETO_HAS_F128
using f128_ty = YATETO_F128_TY;
#endif

#if YATETO_HAS_F16
using f16_ty = YATETO_F16_TY;
#endif

#if YATETO_HAS_BF16
using bf16_ty = YATETO_BF16_TY;
#endif

} // namespace yateto

#endif // YATETO_TYPE_H_
