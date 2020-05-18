#include <cxxtest/TestSuite.h>
#include <yateto/TensorView.h>

using namespace yateto;

class TensorViewTestSuite : public CxxTest::TestSuite
{
private:
  double m_data[12];

public:
  void setUp()
  {
    for (int i = 0; i < 12; ++i) {
      m_data[i] = static_cast<double>(i+1);
    }
  }

  void testAccess()
  {
    DenseTensorView<3, double> tensor(m_data, {3,2,2});
    TS_ASSERT_EQUALS(tensor(0,0,0), 1.0);
    TS_ASSERT_EQUALS(tensor(1,1,0), 5.0);
    TS_ASSERT_EQUALS(tensor(2,1,1), 12.0);
  }

	void testSubtensor()
	{
    DenseTensorView<3, double> tensor(m_data, {3,2,2});
    auto sub = tensor.subtensor(1, slice<>(), slice<>());
    TS_ASSERT_EQUALS(sub(0,0), 2.0);
    TS_ASSERT_EQUALS(sub(1,0), 5.0);
    TS_ASSERT_EQUALS(sub(0,1), 8.0);
    TS_ASSERT_EQUALS(sub(1,1), 11.0);

    auto sub2 = sub.subtensor(1, slice<>());
    TS_ASSERT_EQUALS(sub2(0), 5.0);
    TS_ASSERT_EQUALS(sub2(1), 11.0);

    auto sub3 = tensor.subtensor(slice<>(1,3), slice<>(), slice<>()); 
    TS_ASSERT_EQUALS(sub3(0,0,0), 2.0);
    TS_ASSERT_EQUALS(sub3(0,1,0), 5.0);
    TS_ASSERT_EQUALS(sub3(1,0,1), 9.0);
    TS_ASSERT_EQUALS(sub3(1,1,1), 12.0);
	}

  void testSetZero()
  {
    DenseTensorView<3, double> tensor(m_data, {3,2,2});
    auto sub = tensor.subtensor(1, slice<>(), slice<>());
    sub.setZero();
    for (int i = 0; i < 12; ++i) {
      if ((i-1) % 3 == 0) {
        TS_ASSERT_EQUALS(m_data[i], 0.0);
      } else {
        TS_ASSERT_EQUALS(m_data[i], static_cast<double>(i+1));
      }
    }
  }
};
