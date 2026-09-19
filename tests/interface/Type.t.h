#include <cxxtest/TestSuite.h>
#include <yateto/Marker.h>
#include <yateto/Type.h>

// Outside CUDA and HIP the markers have to vanish, or every signature that
// carries one stops being valid C++.
YATETO_HOSTDEVICE inline int markersAreEmptyOnAHostBuild() { return 0; }

#ifndef YATETO_HAS_F128
#error "YATETO_HAS_F128 has to be defined on every target, as 0 or as 1"
#endif
#ifndef YATETO_HAS_F16
#error "YATETO_HAS_F16 has to be defined on every target, as 0 or as 1"
#endif
#ifndef YATETO_HAS_BF16
#error "YATETO_HAS_BF16 has to be defined on every target, as 0 or as 1"
#endif

// A format that reports itself as present has to have a spelling behind it,
// including when a build set the answer rather than the detection.
#if YATETO_HAS_F128 && !defined(YATETO_F128_TY)
#error "YATETO_HAS_F128 without YATETO_F128_TY"
#endif
#if YATETO_HAS_F16 && !defined(YATETO_F16_TY)
#error "YATETO_HAS_F16 without YATETO_F16_TY"
#endif
#if YATETO_HAS_BF16 && !defined(YATETO_BF16_TY)
#error "YATETO_HAS_BF16 without YATETO_BF16_TY"
#endif

class TypeTestSuite : public CxxTest::TestSuite {
  public:
  void testFloat128() {
#if YATETO_HAS_F128
    TS_ASSERT(sizeof(yateto::f128_ty) >= sizeof(double));

    // a literal spelled without the suffix would have been rounded to double
    // on the way in, and the two would compare equal
    const yateto::f128_ty precise = YATETO_F128_C(1.00000000000000000001);
    const yateto::f128_ty one = YATETO_F128_C(1.0);
    TS_ASSERT(precise != one);

    // and the arithmetic carries more than a double's worth of mantissa
    TS_ASSERT(one + YATETO_F128_C(1e-20) != one);
    TS_ASSERT_EQUALS(static_cast<double>(one), 1.0);
#else
    TS_SKIP("no 128 bit floating point format on this target");
#endif
  }

  void testFloat16() {
#if YATETO_HAS_F16
    TS_ASSERT_EQUALS(sizeof(yateto::f16_ty), 2);
    const auto exact = static_cast<yateto::f16_ty>(1.5);
    TS_ASSERT_EQUALS(static_cast<double>(exact), 1.5);
    // eleven mantissa bits: 2049 is the first integer that does not survive
    const auto rounded = static_cast<yateto::f16_ty>(2049.0);
    TS_ASSERT_DIFFERS(static_cast<double>(rounded), 2049.0);
#else
    TS_SKIP("no 16 bit floating point format on this target");
#endif
  }

  void testBFloat16() {
#if YATETO_HAS_BF16
    TS_ASSERT_EQUALS(sizeof(yateto::bf16_ty), 2);
    const auto exact = static_cast<yateto::bf16_ty>(1.5);
    TS_ASSERT_EQUALS(static_cast<double>(exact), 1.5);
    // eight mantissa bits, but a float's exponent range
    const auto big = static_cast<yateto::bf16_ty>(1e38);
    TS_ASSERT(static_cast<double>(big) > 1e37);
#else
    TS_SKIP("no bfloat16 format on this target");
#endif
  }
};
