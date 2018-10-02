#ifndef YATETO_MATRIXVIEW_H_
#define YATETO_MATRIXVIEW_H_

#include <cassert>
#include <algorithm>

namespace yateto {
  template<unsigned Dim, typename real_t, typename uint_t>
  class TensorView {
  public:
    explicit TensorView(std::initializer_list<uint_t> shape) {
      std::copy(shape.begin(), shape.end(), m_shape);
    }

    explicit TensorView(uint_t const shape[]) {
      for (uint_t d = 0; d < Dim; ++d) {
        m_shape[d] = shape[d];
      }
    }
    
    constexpr uint_t dim() const {
      return Dim;
    }

    uint_t shape(uint_t dim) const {
      return m_shape[dim];
    }

  protected:
    uint_t m_shape[Dim];
  };

  template<unsigned Dim, typename real_t, typename uint_t>
  class DenseTensorView : public TensorView<Dim, real_t, uint_t> {
  public:
    explicit DenseTensorView(real_t* values, std::initializer_list<uint_t> shape, std::initializer_list<uint_t> start, std::initializer_list<uint_t> stop)
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values) {
      std::copy(start.begin(), start.end(), m_start);
      std::copy(stop.begin(), stop.end(), m_stop);
      computeStride();
    }

    explicit DenseTensorView(real_t* values, uint_t const shape[], uint_t const start[], uint_t const stop[])
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values) {
      for (uint_t d = 0; d < Dim; ++d) {
        m_start[d] = start[d];
        m_stop[d] = stop[d];
      }
      computeStride();
    }
    
    uint_t size() const {
      return (m_stop[Dim-1]-m_start[Dim-1]) * m_stride[Dim-1];
    }
    
    void setZero() {
      memset(m_values, 0, size() * sizeof(real_t));
    }

    real_t& operator[](uint_t const entry[Dim]) {
      uint_t addr = 0;
      for (uint_t d = 0; d < Dim; ++d) {
        assert(entry[d] >= m_start[d] && entry[d] < m_stop[d]);
        addr += (entry[d] - m_start[d]) * m_stride[d];
      }
      return m_values[addr];
    }

    template<class view_t>
    void copyToView(view_t& other) {
      assert(Dim == other.dim());
      
      uint_t entry[Dim];
      for (uint_t d = 0; d < Dim; ++d) {
        assert(this->shape(d) == other.shape(d));
        entry[d] = m_start[d];
      }
      
      uint_t stop0 = std::min(m_stop[0], this->shape(0));
      real_t* val = m_values;
      while (entry[Dim-1] != m_stop[Dim-1]) {
        for (int i = m_start[0]; i < stop0; ++i) {
          entry[0] = i;
          other[entry] = *(val++);
        }
        val += (m_stop[0]-stop0);

        if (Dim == 1) {
          break;
        }

        uint_t d = 0;
        do {
          entry[d] = m_start[d];
          d++;
          ++entry[d];
        } while (entry[d] == m_stop[d] && d < Dim-1);
      }
    }

    DenseTensorView<Dim, real_t, uint_t> block(uint_t const origin[], uint_t const shape[]) {
      uint_t start[Dim];
      uint_t stop[Dim];
      for (uint_t d = 0; d < Dim; ++d) {
        start[d] = 0;
        stop[d] = shape[d];
        assert(origin[d] + stop[d] >= m_start[d] && origin[d] + stop[d] <= m_stop[d]);
      }

      return DenseTensorView<Dim, real_t, uint_t>(&operator[](origin), shape, start, stop);
    }

  protected:
    void computeStride() {
      m_stride[0] = 1;
      for (uint_t d = 0; d < Dim-1; ++d) {
        m_stride[d+1] = m_stride[d] * (m_stop[d] - m_start[d]);
      }
    }

    real_t* m_values;
    uint_t m_start[Dim];
    uint_t m_stop[Dim];
    uint_t m_stride[Dim];
  };

  template<typename real_t, typename uint_t>
  class CSCMatrixView : public TensorView<2, real_t, uint_t> {
  public:
    explicit CSCMatrixView(real_t* values, std::initializer_list<uint_t> shape, uint_t const* rowInd, uint_t const* colPtr)
      : TensorView<2, real_t, uint_t>(shape), m_values(values), m_rowInd(rowInd), m_colPtr(colPtr) {
    }

    explicit CSCMatrixView(real_t* values, uint_t const shape[], uint_t const* rowInd, uint_t const* colPtr)
      : TensorView<2, real_t, uint_t>(shape), m_values(values), m_rowInd(rowInd), m_colPtr(colPtr) {
    }

    uint_t size() const {
      return m_colPtr[ this->shape(1) ];
    }

    void setZero() {
      memset(m_values, 0, size() * sizeof(real_t));
    }

    real_t& operator[](uint_t entry[2]) {
      assert(entry[1] >= 0 && entry[1] < this->shape(1));
      uint_t addr = m_colPtr[ entry[1] ];
      uint_t stop = m_colPtr[ entry[1]+1 ];
      while (addr < stop) {
        if (m_rowInd[addr] == entry[0]) {
          break;
        }
        ++addr;
      }
      assert(addr != stop);

      return m_values[addr];
    }

    template<class view_t>
    void copyToView(view_t& other) {
      assert(2 == other.dim());
      assert(this->shape(0) == other.shape(0) && this->shape(1) == other.shape(1));

      uint_t entry[2];
      uint_t ncols = this->shape(1);
      for (uint_t col = 0; col < ncols; ++col) {
        entry[1] = col;
        for (uint_t i = m_colPtr[col]; i < m_colPtr[col+1]; ++i) {
          entry[0] = m_rowInd[i];
          other[entry] = m_values[i];
        }
      }
    }

  protected:
    real_t* m_values;
    uint_t const* m_rowInd;
    uint_t const* m_colPtr;
  };
}

#endif
