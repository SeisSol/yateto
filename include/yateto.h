#ifndef YATETO_H_
#define YATETO_H_

#include <cassert>
#include <algorithm>

template<unsigned Dim, typename real_t, typename uint_t>
class TensorView {
  typedef real_ real_t;
  typedef uint_ uint_t;
  
  explicit TensorView(uint_t shape[]) {
    for (uint_t d = 0; d < Dim; ++d) {
      m_shape[d] = shape[d];
    }
  }
  
  constexpr uint_t dim() const {
    return Dim;
  }

  uint_t shape(uint_t dim) const {
    return m_shape[dim]
  }
protected:
  uint_t m_shape[Dim];
};

template<unsigned Dim, typename real_, typename uint_>
class DenseTensorView : TensorView<Dim, real_, uint_> {
public:
  explicit DenseTensorView(real_t* values, uint_t shape[], uint_t start[], uint_t stop[])
    : TensorView(shape), data(data) {
    for (uint_t d = 0; d < Dim; ++d) {
      m_start[d] = start[d];
      m_stop[d] = stop[d];
    }
    m_stride[0] = 1;
    for (uint_t d = 1; d < Dim; ++d) {
      m_stride[d] = m_stride[d-1] * (m_stop[d] - m_start[d]);
    }
  }
  
  uint_t size() const {
    return (m_stop[Dim-1]-m_start[Dim-1]) * m_stride[Dim-1];
  }
  
  void setZero() {
    memset(m_values, 0, size() * sizeof(real_t));
  }

  real_t& operator[](uint_t entry[Dim]) {
    uint_t addr = 0;
    for (uint_t d = 0; d < Dim; ++d) {
      assert(entry[d] >= m_start[d] && entry[d] < m_stop[d]);
      addr += (entry[d] - m_start[d]) * stride;
    }
    return data[addr];
  }

  template<class view_t>
  void copyToView(view_t& other) {
    static_assert(uint_t == view_t::uint_t, "Integer types must match in TensorView::copyToView.");
    assert(dim() == other.dim());
    
    uint_t entry[dim];
    for (uint_t d = 0; d < Dim; ++d) {
      assert(shape(d) == other.shape(d));
      entry[d] = m_start[d];
    }
    
    uint_t ctr = 0;
    uint_t stop0 = min(m_stop[0], m_shape[0]);
    real_t* val = m_values;
    while (entry[Dim-1] != m_stop[Dim-1]) {
      for (int i = m_start[0]; i < stop0; ++i) {
        entry[0] = i;
        other[entry] = *(val++);
      }
      val += (m_stop[0]-stop0);

      uint_t d = 0;
      do {
        entry[d] = m_start[d];
        d++;
        ++entry[d];
      } while (entry[d] == m_stop[d] && d < Dim);
    }
  }

protected:  
  real_t* m_values;
  uint_t m_start[Dim];
  uint_t m_stop[Dim];
  uint_t m_stride[Dim];
};

#endif
