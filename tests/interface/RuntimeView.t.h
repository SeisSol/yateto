#include <cstdint>
#include <cxxtest/TestSuite.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <yateto/RuntimeView.h>

using namespace yateto;

namespace runtime_view_check {
// A 3 x 2 matrix by columns, without padding, in doubles.
constexpr unsigned Shape[] = {3, 2};
constexpr unsigned Start[] = {0, 0};
constexpr unsigned Stop[] = {3, 2};
constexpr unsigned Stride[] = {1, 3};
constexpr TensorDescriptor Plain{
    Datatype::F64, Storage::Dense, 2, Shape, Start, Stop, Stride, nullptr, nullptr, nullptr, 6, 8};

// The same matrix in floats, its columns padded to four rows and aligned to 16 bytes.
constexpr unsigned PaddedStop[] = {4, 2};
constexpr unsigned PaddedStride[] = {1, 4};
constexpr TensorDescriptor Padded{Datatype::F32,
                                  Storage::Dense,
                                  2,
                                  Shape,
                                  Start,
                                  PaddedStop,
                                  PaddedStride,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  8,
                                  16};

// [[1, 0], [0, 2], [3, 0]] by columns: rows 0 and 2 in column 0, row 1 in column 1.
constexpr unsigned RowIndex[] = {0, 2, 1};
constexpr unsigned ColumnPointer[] = {0, 2, 3};
constexpr TensorDescriptor Sparse{Datatype::F64,
                                  Storage::CSC,
                                  2,
                                  Shape,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  RowIndex,
                                  ColumnPointer,
                                  nullptr,
                                  3,
                                  8};

// The same entries as a pattern.
constexpr unsigned Pattern[] = {1, 0, 2, 0, 3, 0};
constexpr TensorDescriptor Patterned{Datatype::F64,
                                     Storage::Pattern,
                                     2,
                                     Shape,
                                     Start,
                                     Stop,
                                     Stride,
                                     nullptr,
                                     nullptr,
                                     Pattern,
                                     3,
                                     8};
} // namespace runtime_view_check

class RuntimeViewTestSuite : public CxxTest::TestSuite {
  private:
  /// Memory aligned to 16 bytes, and an address in it that is not.
  struct Memory {
    alignas(16) double values[16];
    double* aligned() { return values; }
    double* misaligned() { return values + 1; }
  };

  public:
  void testFamilyRunsTheFirstIndexFastest() {
    Family<int, 2, 3> family;
    TS_ASSERT_EQUALS(family(1, 2), 0);
    family(1, 2) = 5;
    TS_ASSERT_EQUALS(family.data[1 + 2 * 2], 5);
    TS_ASSERT_EQUALS((Family<int, 2, 3>::position(1, 2)), 5u);
    TS_ASSERT_EQUALS((Family<int, 2, 3>::Count), 6u);
  }

  void testAViewForWritingIsOneForReading() {
    double value = 1.0;
    const RuntimeView view{&runtime_view_check::Plain, &value};
    const ConstRuntimeView reading = view;
    TS_ASSERT_EQUALS(reading.layout, &runtime_view_check::Plain);
    TS_ASSERT_EQUALS(reading.data, &value);
  }

  void testCopyConvertsAndZeroesThePadding() {
    const double from[] = {1, 2, 3, 4, 5, 6};
    float to[8];
    for (float& value : to) {
      value = -1;
    }
    copy({&runtime_view_check::Plain, from}, {&runtime_view_check::Padded, to});
    const float expected[] = {1, 2, 3, 0, 4, 5, 6, 0};
    for (unsigned i = 0; i < 8; ++i) {
      TS_ASSERT_EQUALS(to[i], expected[i]);
    }

    // And back, where the padding is not read.
    to[3] = 99;
    double back[6]{};
    copy({&runtime_view_check::Padded, to}, {&runtime_view_check::Plain, back});
    for (unsigned i = 0; i < 6; ++i) {
      TS_ASSERT_EQUALS(back[i], from[i]);
    }
  }

  void testCopyByRowsFromByColumns() {
    static constexpr unsigned Shape[] = {2, 3, 2};
    static constexpr unsigned Start[] = {0, 0, 0};
    static constexpr unsigned ByColumns[] = {1, 2, 6};
    static constexpr unsigned ByRows[] = {6, 2, 1};
    const TensorDescriptor columns{Datatype::F64,
                                   Storage::Dense,
                                   3,
                                   Shape,
                                   Start,
                                   Shape,
                                   ByColumns,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   12,
                                   8};
    const TensorDescriptor rows{Datatype::F64,
                                Storage::Dense,
                                3,
                                Shape,
                                Start,
                                Shape,
                                ByRows,
                                nullptr,
                                nullptr,
                                nullptr,
                                12,
                                8};
    double from[12];
    for (unsigned i = 0; i < 12; ++i) {
      from[i] = i;
    }
    double to[12]{};
    copy({&columns, from}, {&rows, to});
    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned j = 0; j < 3; ++j) {
        for (unsigned k = 0; k < 2; ++k) {
          TS_ASSERT_EQUALS(to[6 * i + 2 * j + k], from[i + 2 * j + 6 * k]);
        }
      }
    }
  }

  void testCopyFillsWhatTheSourceDoesNotStoreWithZero() {
    // Only rows 1 and 2 are stored, and the destination stores all of them.
    static constexpr unsigned Start[] = {1, 0};
    static constexpr unsigned Stop[] = {3, 2};
    static constexpr unsigned Stride[] = {1, 2};
    const TensorDescriptor rows{Datatype::F64,
                                Storage::Dense,
                                2,
                                runtime_view_check::Shape,
                                Start,
                                Stop,
                                Stride,
                                nullptr,
                                nullptr,
                                nullptr,
                                4,
                                8};
    const double from[] = {1, 2, 3, 4};
    double to[6];
    for (double& value : to) {
      value = -1;
    }
    copy({&rows, from}, {&runtime_view_check::Plain, to});
    const double expected[] = {0, 1, 2, 0, 3, 4};
    for (unsigned i = 0; i < 6; ++i) {
      TS_ASSERT_EQUALS(to[i], expected[i]);
    }

    // The other way around, what the destination does not store is dropped.
    double back[4]{};
    copy({&runtime_view_check::Plain, expected}, {&rows, back});
    for (unsigned i = 0; i < 4; ++i) {
      TS_ASSERT_EQUALS(back[i], from[i]);
    }
  }

  void testCopyBetweenSparseAndDense() {
    const double sparse[] = {1, 3, 2};
    double dense[6];
    for (double& value : dense) {
      value = -1;
    }
    copy({&runtime_view_check::Sparse, sparse}, {&runtime_view_check::Plain, dense});
    const double expected[] = {1, 0, 3, 0, 2, 0};
    for (unsigned i = 0; i < 6; ++i) {
      TS_ASSERT_EQUALS(dense[i], expected[i]);
    }

    // A value the sparse layout has no place for is lost, as it is zero by the pattern.
    dense[1] = 7;
    double pattern[3]{};
    copy({&runtime_view_check::Plain, dense}, {&runtime_view_check::Patterned, pattern});
    for (unsigned i = 0; i < 3; ++i) {
      TS_ASSERT_EQUALS(pattern[i], sparse[i]);
    }
    double again[3]{};
    copy({&runtime_view_check::Patterned, pattern}, {&runtime_view_check::Sparse, again});
    for (unsigned i = 0; i < 3; ++i) {
      TS_ASSERT_EQUALS(again[i], sparse[i]);
    }
  }

  void testCopyOfATensorWithoutDimensions() {
    const TensorDescriptor single{Datatype::F32,
                                  Storage::Dense,
                                  0,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  1,
                                  4};
    const TensorDescriptor twice{Datatype::I64,
                                 Storage::Dense,
                                 0,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 1,
                                 8};
    const float from = 3.0f;
    std::int64_t to = 0;
    copy({&single, &from}, {&twice, &to});
    TS_ASSERT_EQUALS(to, 3);
  }

  void testCopyThroughSixteenBits() {
    const double from[] = {1, 2, 3, 4, 5, 6};
    double back[6]{};
#if YATETO_HAS_F16
    TensorDescriptor half = runtime_view_check::Plain;
    half.datatype = Datatype::F16;
    f16_ty halves[6];
    copy({&runtime_view_check::Plain, from}, {&half, halves});
    copy({&half, halves}, {&runtime_view_check::Plain, back});
    for (unsigned i = 0; i < 6; ++i) {
      TS_ASSERT_EQUALS(back[i], from[i]);
    }
#endif
#if YATETO_HAS_BF16
    TensorDescriptor brain = runtime_view_check::Plain;
    brain.datatype = Datatype::BF16;
    bf16_ty brains[6];
    copy({&runtime_view_check::Plain, from}, {&brain, brains});
    copy({&brain, brains}, {&runtime_view_check::Plain, back});
    for (unsigned i = 0; i < 6; ++i) {
      TS_ASSERT_EQUALS(back[i], from[i]);
    }
#endif
    static_cast<void>(from);
    static_cast<void>(back);
  }

  void testCopyNeedsTheSameShape() {
    static constexpr unsigned Other[] = {2, 3};
    TensorDescriptor transposed = runtime_view_check::Plain;
    transposed.shape = Other;
    double from[6]{};
    double to[6]{};
    TS_ASSERT_THROWS(copy({&transposed, from}, {&runtime_view_check::Plain, to}),
                     const std::invalid_argument&);
    TS_ASSERT_THROWS(copy({nullptr, from}, {&runtime_view_check::Plain, to}),
                     const std::invalid_argument&);
  }

  void testPassingThroughTakesTheLayoutAndTheAlignment() {
    Memory memory;
    TensorDescriptor aligned = runtime_view_check::Plain;
    aligned.alignment = 16;
    TS_ASSERT(passesThrough({&runtime_view_check::Plain, memory.aligned()}, aligned));
    TS_ASSERT(passesThrough({&aligned, memory.aligned()}, aligned));
    TS_ASSERT(!passesThrough({&runtime_view_check::Plain, memory.misaligned()}, aligned));
    TS_ASSERT(passesThrough({&runtime_view_check::Plain, memory.misaligned()},
                            runtime_view_check::Plain));
    TS_ASSERT(!passesThrough({&runtime_view_check::Sparse, memory.aligned()}, aligned));
  }

  void testAnOperandInItsOwnLayoutIsTheViewItself() {
    Memory memory;
    const Operand<double> operand(ConstRuntimeView{&runtime_view_check::Plain, memory.aligned()},
                                  runtime_view_check::Plain);
    TS_ASSERT_EQUALS(operand.data(), memory.aligned());
    TS_ASSERT(!operand.copied());
  }

  void testAnOperandInAnotherLayoutIsACopyWrittenBack() {
    double values[] = {1, 2, 3, 4, 5, 6};
    Operand<float> operand(RuntimeView{&runtime_view_check::Plain, values},
                           runtime_view_check::Padded);
    TS_ASSERT(operand.copied());
    float* data = operand.data();
    TS_ASSERT_EQUALS(reinterpret_cast<std::uintptr_t>(data) % 16, 0u);
    TS_ASSERT_EQUALS(data[4], 4.0f);
    TS_ASSERT_EQUALS(data[3], 0.0f);
    for (unsigned i = 0; i < 8; ++i) {
      data[i] *= 10;
    }
    TS_ASSERT_EQUALS(values[0], 1.0);
    operand.finish();
    const double expected[] = {10, 20, 30, 40, 50, 60};
    for (unsigned i = 0; i < 6; ++i) {
      TS_ASSERT_EQUALS(values[i], expected[i]);
    }
  }

  void testAnOperandOnlyReadIsNotWrittenBack() {
    double values[] = {1, 2, 3, 4, 5, 6};
    Operand<float> operand(ConstRuntimeView{&runtime_view_check::Plain, values},
                           runtime_view_check::Padded);
    operand.data()[0] = 100;
    operand.finish();
    TS_ASSERT_EQUALS(values[0], 1.0);
  }

  void testAnOperandSaysWhatIsMissing() {
    double values[6]{};
    try {
      const Operand<double> operand(ConstRuntimeView{&runtime_view_check::Plain, nullptr},
                                    runtime_view_check::Plain,
                                    "Q of k");
      TS_FAIL("an operand without values");
    } catch (const std::invalid_argument& error) {
      TS_ASSERT(std::string(error.what()).find("Q of k") != std::string::npos);
    }
    TS_ASSERT_THROWS(
        (Operand<double>(ConstRuntimeView{nullptr, values}, runtime_view_check::Plain)),
        const std::invalid_argument&);
    static constexpr unsigned Other[] = {2, 3};
    TensorDescriptor transposed = runtime_view_check::Plain;
    transposed.shape = Other;
    TS_ASSERT_THROWS(
        (Operand<float>(ConstRuntimeView{&transposed, values}, runtime_view_check::Padded)),
        const std::invalid_argument&);
  }
};
