#include <cxxtest/TestSuite.h>
#include <yateto/Descriptor.h>

using namespace yateto;

// Everything a descriptor answers is a constant expression, so that the answers for a generated
// layout can be checked at compile time where they are known then.
namespace descriptor_check {
// A 3 x 2 matrix in a box padded to four rows.
constexpr unsigned Shape[] = {3, 2};
constexpr unsigned Start[] = {0, 0};
constexpr unsigned Stop[] = {4, 2};
constexpr unsigned Stride[] = {1, 4};
constexpr TensorDescriptor Padded{
    Datatype::F64, Storage::Dense, 2, Shape, Start, Stop, Stride, nullptr, nullptr, nullptr, 8, 32};

constexpr std::ptrdiff_t at(const TensorDescriptor& descriptor, unsigned i, unsigned j) {
  const unsigned index[] = {i, j};
  return offsetOf(descriptor, index);
}

static_assert(at(Padded, 0, 0) == 0, "");
static_assert(at(Padded, 2, 1) == 6, "the first index runs fastest");
static_assert(at(Padded, 3, 1) == 7, "padding is stored like any entry");
static_assert(at(Padded, 4, 0) == -1, "past the box, nothing is stored");
static_assert(sameLayout(Padded, Padded), "");
static_assert(sizeOf(Datatype::F32) == 4 && sizeOf(Datatype::F128) == 16, "");
} // namespace descriptor_check

class DescriptorTestSuite : public CxxTest::TestSuite {
  private:
  static std::ptrdiff_t at(const TensorDescriptor& descriptor, unsigned i, unsigned j) {
    return descriptor_check::at(descriptor, i, j);
  }

  public:
  void testDenseBoxNeedNotStartAtZero() {
    // Only rows 1 and 2 of a 4 x 2 matrix are stored.
    static constexpr unsigned Shape[] = {4, 2};
    static constexpr unsigned Start[] = {1, 0};
    static constexpr unsigned Stop[] = {3, 2};
    static constexpr unsigned Stride[] = {1, 2};
    const TensorDescriptor rows{Datatype::F32,
                                Storage::Dense,
                                2,
                                Shape,
                                Start,
                                Stop,
                                Stride,
                                nullptr,
                                nullptr,
                                nullptr,
                                4,
                                4};
    TS_ASSERT_EQUALS(at(rows, 0, 0), -1);
    TS_ASSERT_EQUALS(at(rows, 1, 0), 0);
    TS_ASSERT_EQUALS(at(rows, 2, 1), 3);
    TS_ASSERT_EQUALS(at(rows, 3, 1), -1);
  }

  void testDenseWithStridesOfItsOwn() {
    // The same matrix by rows rather than by columns.
    static constexpr unsigned Shape[] = {3, 2};
    static constexpr unsigned Start[] = {0, 0};
    static constexpr unsigned Stop[] = {3, 2};
    static constexpr unsigned Stride[] = {2, 1};
    const TensorDescriptor byRows{Datatype::F64,
                                  Storage::Dense,
                                  2,
                                  Shape,
                                  Start,
                                  Stop,
                                  Stride,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  6,
                                  8};
    TS_ASSERT_EQUALS(at(byRows, 1, 0), 2);
    TS_ASSERT_EQUALS(at(byRows, 1, 1), 3);
  }

  void testWithoutDimensionsThereIsOneValue() {
    const TensorDescriptor scalar{Datatype::F64,
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
    TS_ASSERT_EQUALS(offsetOf(scalar, nullptr), 0);
  }

  void testCSCFindsWhatIsStored() {
    // [[1, 0], [0, 2], [3, 0]] by columns: rows 0 and 2 in column 0, row 1 in column 1.
    static constexpr unsigned Shape[] = {3, 2};
    static constexpr unsigned RowIndex[] = {0, 2, 1};
    static constexpr unsigned ColumnPointer[] = {0, 2, 3};
    const TensorDescriptor csc{Datatype::F64,
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
    TS_ASSERT_EQUALS(at(csc, 0, 0), 0);
    TS_ASSERT_EQUALS(at(csc, 2, 0), 1);
    TS_ASSERT_EQUALS(at(csc, 1, 1), 2);
    TS_ASSERT_EQUALS(at(csc, 1, 0), -1);
    TS_ASSERT_EQUALS(at(csc, 0, 2), -1);
  }

  void testPatternCountsFromOne() {
    // The same matrix as a pattern over its shape.
    static constexpr unsigned Shape[] = {3, 2};
    static constexpr unsigned Start[] = {0, 0};
    static constexpr unsigned Stop[] = {3, 2};
    static constexpr unsigned Stride[] = {1, 3};
    static constexpr unsigned Pattern[] = {1, 0, 2, 0, 3, 0};
    const TensorDescriptor pattern{Datatype::F64,
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
    TS_ASSERT_EQUALS(at(pattern, 0, 0), 0);
    TS_ASSERT_EQUALS(at(pattern, 2, 0), 1);
    TS_ASSERT_EQUALS(at(pattern, 1, 1), 2);
    TS_ASSERT_EQUALS(at(pattern, 1, 0), -1);
  }

  void testSameLayoutComparesNumbersNotAddresses() {
    static constexpr unsigned Shape[] = {3, 2};
    static constexpr unsigned Start[] = {0, 0};
    static constexpr unsigned Stop[] = {4, 2};
    static constexpr unsigned Stride[] = {1, 4};
    static constexpr unsigned Unpadded[] = {3, 2};
    static constexpr unsigned UnpaddedStride[] = {1, 3};
    const TensorDescriptor& padded = descriptor_check::Padded;
    const TensorDescriptor copy{Datatype::F64,
                                Storage::Dense,
                                2,
                                Shape,
                                Start,
                                Stop,
                                Stride,
                                nullptr,
                                nullptr,
                                nullptr,
                                8,
                                64};
    TS_ASSERT(sameLayout(padded, copy));

    TensorDescriptor single = copy;
    single.datatype = Datatype::F32;
    TS_ASSERT(!sameLayout(padded, single));

    const TensorDescriptor narrow{Datatype::F64,
                                  Storage::Dense,
                                  2,
                                  Shape,
                                  Start,
                                  Unpadded,
                                  UnpaddedStride,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  6,
                                  8};
    TS_ASSERT(!sameLayout(padded, narrow));
  }

  void testTableFindsByNameAndGroup() {
    using descriptor_check::Padded;
    static constexpr unsigned GroupSize[] = {2, 3};
    // A family of six with four members; (1, 0) and (0, 2) are holes.
    static constexpr const TensorDescriptor* Members[] = {
        &Padded, nullptr, &Padded, &Padded, nullptr, &Padded};
    static constexpr const TensorDescriptor* Single[] = {&Padded};
    static constexpr TensorEntry Entries[] = {
        {"A", 0, nullptr, Single},
        {"F", 2, GroupSize, Members},
        {"nodal::V", 0, nullptr, Single},
    };
    const TensorTable table{Entries, 3};

    TS_ASSERT_EQUALS(table.find("A"), &Entries[0]);
    TS_ASSERT_EQUALS(table.find("nodal::V"), &Entries[2]);
    TS_ASSERT(table.find("B") == nullptr);
    TS_ASSERT(table.find("V") == nullptr);

    TS_ASSERT_EQUALS(table.find("A", {}), &Padded);
    TS_ASSERT_EQUALS(table.find("F", {0, 0}), &Padded);
    TS_ASSERT_EQUALS(table.find("F", {0, 1}), &Padded);
    TS_ASSERT(table.find("F", {1, 0}) == nullptr);
    TS_ASSERT(table.find("F", {2, 0}) == nullptr);
    TS_ASSERT(table.find("F", {0}) == nullptr);
    TS_ASSERT(table.find("A", {0}) == nullptr);
  }

  void testAnEmptyTableFindsNothing() {
    const TensorTable table{nullptr, 0};
    TS_ASSERT(table.find("A") == nullptr);
  }
};
