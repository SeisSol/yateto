#include <cstdint>
#include <cxxtest/TestSuite.h>
#include <vector>
#include <yateto/InitTools.h>
#include <yateto/LinearAllocator.h>
#include <yateto/Misc.h>
#include <yateto/TensorView.h>

using namespace yateto;

// Instantiating a read-only view has to name every member of it, so this is
// where a member that only makes sense on a writable view shows up.
template class yateto::DenseTensorView<3, double, unsigned, true>;
template class yateto::DenseTensorView<0, double, unsigned, true>;
template class yateto::CSCMatrixView<double, unsigned, true>;
template class yateto::PatternTensorView<2, double, unsigned, true>;

// A view is a value made of pointers and indices, and every one of its
// members is a constant expression, so a view over constant data can be
// walked at compile time. These hold as of C++17, which rules out the parts
// of the standard library that only became constexpr in C++20.
namespace constexpr_check {
constexpr double Values[6] = {1., 2., 3., 4., 5., 6.};

constexpr double at(unsigned i, unsigned j) {
  const DenseTensorView<2, double, unsigned, true> view(Values, {2, 3});
  return view(i, j);
}

constexpr unsigned shapeOf(unsigned dim) {
  const DenseTensorView<2, double, unsigned, true> view(Values, {2, 3});
  return view.shape(dim);
}

constexpr double sum() {
  const DenseTensorView<2, double, unsigned, true> view(Values, {2, 3});
  double total = 0.0;
  view.forall([&total](const unsigned* /*entry*/, const double& value) { total += value; });
  return total;
}

static_assert(at(0, 0) == 1.0, "column major: the first index runs fastest");
static_assert(at(1, 0) == 2.0, "");
static_assert(at(0, 1) == 3.0, "");
static_assert(shapeOf(1) == 3, "");
static_assert(sum() == 21.0, "forall is a constant expression too");
} // namespace constexpr_check

class DenseTensorViewTestSuite : public CxxTest::TestSuite {
  private:
  double data_[12];

  public:
  void setUp() {
    for (int i = 0; i < 12; ++i) {
      data_[i] = static_cast<double>(i + 1);
    }
  }

  void testAccess() {
    DenseTensorView<3, double> tensor(data_, {3, 2, 2});
    TS_ASSERT_EQUALS(tensor(0, 0, 0), 1.0);
    TS_ASSERT_EQUALS(tensor(1, 1, 0), 5.0);
    TS_ASSERT_EQUALS(tensor(2, 1, 1), 12.0);
  }

  void testSubtensor() {
    DenseTensorView<3, double> tensor(data_, {3, 2, 2});
    auto sub = tensor.subtensor(1, slice<>(), slice<>());
    TS_ASSERT_EQUALS(sub(0, 0), 2.0);
    TS_ASSERT_EQUALS(sub(1, 0), 5.0);
    TS_ASSERT_EQUALS(sub(0, 1), 8.0);
    TS_ASSERT_EQUALS(sub(1, 1), 11.0);

    auto sub2 = sub.subtensor(1, slice<>());
    TS_ASSERT_EQUALS(sub2(0), 5.0);
    TS_ASSERT_EQUALS(sub2(1), 11.0);

    auto sub3 = tensor.subtensor(slice<>(1, 3), slice<>(), slice<>());
    TS_ASSERT_EQUALS(sub3(0, 0, 0), 2.0);
    TS_ASSERT_EQUALS(sub3(0, 1, 0), 5.0);
    TS_ASSERT_EQUALS(sub3(1, 0, 1), 9.0);
    TS_ASSERT_EQUALS(sub3(1, 1, 1), 12.0);
  }

  void testSetZero() {
    DenseTensorView<3, double> tensor(data_, {3, 2, 2});
    auto sub = tensor.subtensor(1, slice<>(), slice<>());
    sub.setZero();
    for (int i = 0; i < 12; ++i) {
      if ((i - 1) % 3 == 0) {
        TS_ASSERT_EQUALS(data_[i], 0.0);
      } else {
        TS_ASSERT_EQUALS(data_[i], static_cast<double>(i + 1));
      }
    }
  }

  void testForall() {
    DenseTensorView<3, double> tensor(data_, {3, 2, 2});
    double sum = 0.0;
    unsigned visited = 0;
    tensor.forall([&sum, &visited](const unsigned* /*entry*/, double& value) {
      sum += value;
      ++visited;
    });
    TS_ASSERT_EQUALS(visited, 12);
    TS_ASSERT_EQUALS(sum, 78.0);
  }

  void testShape() {
    const DenseTensorView<3, double> tensor(data_, {3, 2, 2});
    TS_ASSERT_EQUALS(tensor.dim(), 3);
    TS_ASSERT_EQUALS(tensor.shape(0), 3);
    TS_ASSERT_EQUALS(tensor.shape(1), 2);
    TS_ASSERT_EQUALS(tensor.shape(2), 2);
  }

  void testForallOnConstView() {
    const DenseTensorView<3, double, unsigned, true> tensor(data_, {3, 2, 2});
    double sum = 0.0;
    tensor.forall([&sum](const unsigned* /*entry*/, const double& value) { sum += value; });
    TS_ASSERT_EQUALS(sum, 78.0);
  }

  void testForallOnSubtensor() {
    DenseTensorView<3, double> tensor(data_, {3, 2, 2});
    auto sub = tensor.subtensor(1, slice<>(), slice<>());
    double sum = 0.0;
    unsigned visited = 0;
    sub.forall([&sum, &visited](const unsigned* /*entry*/, double& value) {
      sum += value;
      ++visited;
    });
    TS_ASSERT_EQUALS(visited, 4);
    TS_ASSERT_EQUALS(sum, 2.0 + 5.0 + 8.0 + 11.0);
  }

  void testZeroDimensional() {
    double source = 3.0;
    double target = 0.0;
    const DenseTensorView<0, double, unsigned, true> from(&source, {}, {}, {});
    DenseTensorView<0, double, unsigned, false> to(&target, {}, {}, {});
    TS_ASSERT_EQUALS(from.size(), 1);
    from.copyToView(to);
    TS_ASSERT_EQUALS(target, 3.0);
    to.setZero();
    TS_ASSERT_EQUALS(target, 0.0);
  }
};

class CSCMatrixViewTestSuite : public CxxTest::TestSuite {
  private:
  // a 3x3 matrix holding (0,0), (2,0) and (1,2)
  double data_[3];
  unsigned rowInd_[3];
  unsigned colPtr_[4];

  public:
  void setUp() {
    data_[0] = 1.0;
    data_[1] = 2.0;
    data_[2] = 3.0;
    rowInd_[0] = 0;
    rowInd_[1] = 2;
    rowInd_[2] = 1;
    colPtr_[0] = 0;
    colPtr_[1] = 2;
    colPtr_[2] = 2;
    colPtr_[3] = 3;
  }

  void testAccess() {
    CSCMatrixView<double, unsigned> matrix(data_, {3, 3}, rowInd_, colPtr_);
    TS_ASSERT_EQUALS(matrix.size(), 3);
    TS_ASSERT(matrix.isInRange(0, 0));
    TS_ASSERT_EQUALS(matrix(0, 0), 1.0);
    TS_ASSERT(matrix.isInRange(2, 0));
    TS_ASSERT_EQUALS(matrix(2, 0), 2.0);
    TS_ASSERT(matrix.isInRange(1, 2));
    TS_ASSERT_EQUALS(matrix(1, 2), 3.0);
    TS_ASSERT(!matrix.isInRange(1, 0));
    TS_ASSERT(!matrix.isInRange(0, 1));
    TS_ASSERT(!matrix.isInRange(0, 2));
  }

  void testForall() {
    CSCMatrixView<double, unsigned> matrix(data_, {3, 3}, rowInd_, colPtr_);
    double sum = 0.0;
    unsigned visited = 0;
    matrix.forall([&sum, &visited](const unsigned* /*entry*/, double& value) {
      sum += value;
      ++visited;
    });
    TS_ASSERT_EQUALS(visited, 3);
    TS_ASSERT_EQUALS(sum, 6.0);
  }

  void testSetZero() {
    CSCMatrixView<double, unsigned> matrix(data_, {3, 3}, rowInd_, colPtr_);
    matrix.setZero();
    TS_ASSERT_EQUALS(data_[0], 0.0);
    TS_ASSERT_EQUALS(data_[1], 0.0);
    TS_ASSERT_EQUALS(data_[2], 0.0);
  }
};

class PatternTensorViewTestSuite : public CxxTest::TestSuite {
  private:
  double data_[6];
  uint32_t pattern_[12];

  public:
  void setUp() {
    for (int i = 0; i < 6; ++i) {
      data_[i] = static_cast<double>(2 * i + 1);
    }
    for (int i = 0; i < 12; ++i) {
      pattern_[i] = (i % 2 == 0) ? (i / 2 + 1) : 0;
    }
  }

  void testBasic() {
    PatternTensorView<3, double, uint32_t> tensor(data_, {3, 2, 2}, pattern_);
    TS_ASSERT_EQUALS(tensor.size(), 6);
  }

  void testSubtensorSize() {
    PatternTensorView<3, double, uint32_t> tensor(data_, {3, 2, 2}, pattern_);
    auto sub = tensor.subtensor(1, slice<>(), slice<>());
    TS_ASSERT_EQUALS(sub.size(), 2);
    auto sub2 = sub.subtensor(1, slice<>());
    TS_ASSERT_EQUALS(sub2.size(), 2);
  }

  void testForall() {
    PatternTensorView<3, double, uint32_t> tensor(data_, {3, 2, 2}, pattern_);
    double sum = 0.0;
    unsigned visited = 0;
    tensor.forall([&sum, &visited](const auto* /*entry*/, double& value) {
      sum += value;
      ++visited;
    });
    TS_ASSERT_EQUALS(visited, 6);
    TS_ASSERT_EQUALS(sum, 36.0);
  }

  void testAccess() {
    PatternTensorView<3, double, uint32_t> tensor(data_, {3, 2, 2}, pattern_);
    TS_ASSERT(tensor.isInRange(0, 0, 0));
    TS_ASSERT_EQUALS(tensor(0, 0, 0), 1.0);
    TS_ASSERT(tensor.isInRange(1, 1, 0));
    TS_ASSERT_EQUALS(tensor(1, 1, 0), 5.0);
    TS_ASSERT(!tensor.isInRange(2, 1, 1));
    // TS_ASSERT_EQUALS(tensor(2,1,1), 0.0);
  }

  void testSubtensor() {
    PatternTensorView<3, double, uint32_t> tensor(data_, {3, 2, 2}, pattern_);
    auto sub = tensor.subtensor(1, slice<>(), slice<>());
    TS_ASSERT(!sub.isInRange(0, 0));
    // TS_ASSERT_EQUALS(sub(0,0), 0.0);
    TS_ASSERT(sub.isInRange(1, 0));
    TS_ASSERT_EQUALS(sub(1, 0), 5.0);
    TS_ASSERT(!sub.isInRange(0, 1));
    // TS_ASSERT_EQUALS(sub(0,1), 0.0);
    TS_ASSERT(sub.isInRange(1, 1));
    TS_ASSERT_EQUALS(sub(1, 1), 11.0);

    auto sub2 = sub.subtensor(1, slice<>());
    TS_ASSERT(sub2.isInRange(0));
    TS_ASSERT_EQUALS(sub2(0), 5.0);
    TS_ASSERT(sub2.isInRange(1));
    TS_ASSERT_EQUALS(sub2(1), 11.0);

    auto sub3 = tensor.subtensor(slice<>(1, 3), slice<>(), slice<>());
    TS_ASSERT(!sub3.isInRange(0, 0, 0));
    // TS_ASSERT_EQUALS(sub3(0,0,0), 0.0);
    TS_ASSERT(sub3.isInRange(0, 1, 0));
    TS_ASSERT_EQUALS(sub3(0, 1, 0), 5.0);
    TS_ASSERT(sub3.isInRange(1, 0, 1));
    TS_ASSERT_EQUALS(sub3(1, 0, 1), 9.0);
    TS_ASSERT(!sub3.isInRange(1, 1, 1));
    // TS_ASSERT_EQUALS(sub3(1,1,1), 0.0);
  }

  void testSetZero() {
    PatternTensorView<3, double, uint32_t> tensor(data_, {3, 2, 2}, pattern_);
    auto sub = tensor.subtensor(1, slice<>(), slice<>());
    sub.setZero();
    for (int i = 0; i < 6; ++i) {
      if ((2 * i - 1) % 3 == 0) {
        TS_ASSERT_EQUALS(data_[i], 0.0);
      } else {
        TS_ASSERT_EQUALS(data_[i], static_cast<double>(2 * i + 1));
      }
    }
  }
};

class InitToolsTestSuite : public CxxTest::TestSuite {
  private:
  struct Family {
    static constexpr std::size_t Size[3] = {3, 8, 5};
  };

  struct SmallKernel {
    static constexpr std::size_t TmpMaxMemRequiredInBytes = 128;
  };

  struct LargeKernel {
    static constexpr std::size_t TmpMaxMemRequiredInBytes = 512;
  };

  public:
  void testAlignedUpper() {
    TS_ASSERT_EQUALS(alignedUpper(0, 4), 0);
    TS_ASSERT_EQUALS(alignedUpper(1, 4), 4);
    TS_ASSERT_EQUALS(alignedUpper(4, 4), 4);
    TS_ASSERT_EQUALS(alignedUpper(5, 4), 8);
    // an alignment of zero leaves the address alone
    TS_ASSERT_EQUALS(alignedUpper(5, 0), 5);
  }

  void testAlignedReals() {
    TS_ASSERT_EQUALS(alignedReals<double>(64), 8);
    TS_ASSERT_EQUALS(alignedReals<float>(64), 16);
  }

  void testFamilySize() {
    TS_ASSERT_EQUALS(numFamilyMembers<Family>(), 3);
    TS_ASSERT_EQUALS(computeFamilySize<Family>(), 16);
    TS_ASSERT_EQUALS(computeFamilySize<Family>(4), 20);
    // the two-argument form gives the offset of the n-th member
    TS_ASSERT_EQUALS(computeFamilySize<Family>(1, 0), 0);
    TS_ASSERT_EQUALS(computeFamilySize<Family>(1, 1), 3);
    TS_ASSERT_EQUALS(computeFamilySize<Family>(1, 2), 11);
  }

  void testMaxTmpMem() {
    constexpr auto maximum = getMaxTmpMemRequired(SmallKernel{}, LargeKernel{});
    static_assert(maximum == 512, "the requirement is known at compile time");
    TS_ASSERT_EQUALS(maximum, 512);
    TS_ASSERT_EQUALS(getMaxTmpMemRequired(SmallKernel{}), 128);
  }

  void testLinearAllocator() {
    std::vector<char> memory(64);
    LinearAllocatorT<char> allocator;
    allocator.initialize(memory.data());
    char* first = allocator.allocate(16);
    char* second = allocator.allocate(8);
    TS_ASSERT_EQUALS(first, memory.data());
    TS_ASSERT_EQUALS(second - first, 16);
    allocator.free();
    allocator.initialize(memory.data());
    TS_ASSERT_EQUALS(allocator.allocate(1), memory.data());
  }

  void testLinearAllocatorWithCapacity() {
    std::vector<char> memory(64);
    LinearAllocatorT<char> allocator;
    allocator.initialize(memory.data(), memory.size());
    TS_ASSERT_EQUALS(allocator.allocate(64), memory.data());
    // a block handed out in full can be handed out again after free()
    allocator.free();
    allocator.initialize(memory.data(), memory.size());
    TS_ASSERT_EQUALS(allocator.allocate(1), memory.data());
  }

  void testLinearAllocatorBeyondIntRange() {
    // the allocator only does address arithmetic, so no memory is touched here
    char* const base = reinterpret_cast<char*>(std::uintptr_t{1} << 16);
    const std::size_t huge = std::size_t{3} << 30;
    LinearAllocatorT<char> allocator;
    allocator.initialize(base);
    allocator.allocate(huge);
    const auto offset = reinterpret_cast<std::uintptr_t>(allocator.allocate(1)) -
                        reinterpret_cast<std::uintptr_t>(base);
    TS_ASSERT_EQUALS(offset, huge);
  }
};
