#ifndef YATETO_MATRIXVIEW_H_
#define YATETO_MATRIXVIEW_H_

#include <cassert>
#include <cstring>
#include <algorithm>
#include <limits>
#include <type_traits>

namespace yateto {
  template<typename uint_t=unsigned>
  class slice {
  public:
    explicit slice(uint_t start = 0, uint_t stop = std::numeric_limits<uint_t>::max())
      : start(start), stop(stop)
        {}

    uint_t start;
    uint_t stop;
  };

  template<typename uint_t, typename... Entry>
  struct count_slices : std::integral_constant<uint_t,0> {};
  template<typename uint_t, typename Head, typename... Tail>
  struct count_slices<uint_t, Head, Tail...> : std::integral_constant<uint_t, ((std::is_same<Head, slice<uint_t>>::value) ? 1 : 0) + count_slices<uint_t, Tail...>::value> {};

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

  template<unsigned Dim, typename real_t, typename uint_t=unsigned>
  class DenseTensorView : public TensorView<Dim, real_t, uint_t> {
  public:
    explicit DenseTensorView(real_t* values, std::initializer_list<uint_t> shape, std::initializer_list<uint_t> start, std::initializer_list<uint_t> stop)
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values) {
      std::copy(start.begin(), start.end(), m_start);
      std::copy(stop.begin(), stop.end(), m_stop);
      computeStride();
    }

    explicit DenseTensorView(real_t* values, std::initializer_list<uint_t> shape)
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values), m_start{} {
      std::copy(shape.begin(), shape.end(), m_stop);
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

    explicit DenseTensorView(real_t* values, uint_t const shape[])
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values), m_start{} {
      for (uint_t d = 0; d < Dim; ++d) {
        m_stop[d] = shape[d];
      }
      computeStride();
    }
 
    explicit DenseTensorView(real_t* values, uint_t const shape[], uint_t const stride[])
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values), m_start{} {
      for (uint_t d = 0; d < Dim; ++d) {
        m_stop[d] = shape[d];
        m_stride[d] = stride[d];
      }
    }

    uint_t size() const {
      return (m_stop[Dim-1]-m_start[Dim-1]) * m_stride[Dim-1];
    }
    
    void setZero() {
      uint_t entry[Dim];
      std::copy(m_start, m_start + Dim, entry);
      while (entry[Dim-1] != m_stop[Dim-1]) {
        auto values = &operator[](entry);
        for (uint_t i = 0.0; i < m_stop[0]-m_start[0]; ++i) {
          values[i*m_stride[0]] = 0.0;
        }
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

    template<typename Head>
    void isInRange(uint_t start[Dim], uint_t stop[Dim], int dim, Head head) {
      assert(static_cast<uint_t>(head) >= start[dim]);
      assert(static_cast<uint_t>(head) < stop[dim]);
    }

    template<typename Head, typename... Tail>
    void isInRange(uint_t start[Dim], uint_t stop[Dim], int dim, Head head, Tail... tail) {
      assert(static_cast<uint_t>(head) >= start[dim]);
      assert(static_cast<uint_t>(head) < stop[dim]);
      isInRange(start, stop, dim+1, tail...);
    }
  
    template<typename... Entry>
    real_t& operator()(Entry... entry) {
      static_assert(sizeof...(entry) == Dim, "Number of arguments to operator() does not match Tensor's dimension.");
      isInRange(m_start, m_stop, 0, entry...);
      return m_values[address(entry...)];
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
        for (uint_t i = m_start[0]; i < stop0; ++i) {
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

    template<typename... Entry>
    auto subtensor(Entry... entry) -> DenseTensorView<count_slices<uint_t, Entry...>::value, real_t, uint_t>  {
      static_assert(sizeof...(entry) == Dim, "Number of arguments to subtensor() does not match Tensor's dimension.");
      constexpr auto nSlices = count_slices<uint_t, Entry...>::value;
      uint_t begin[Dim];
      uint_t size[nSlices];
      uint_t stride[nSlices];
      extractSubtensor(begin, size, stride, entry...);
      DenseTensorView<nSlices, real_t, uint_t> subtensor(&operator[](begin), size, stride);
      return subtensor;
    }

    real_t* data() {
      return m_values;
    }

  protected:
    void computeStride() {
      m_stride[0] = 1;
      for (uint_t d = 0; d < Dim-1; ++d) {
        m_stride[d+1] = m_stride[d] * (m_stop[d] - m_start[d]);
      }
    }

    template<typename Head>
    uint_t address(Head head) {
      assert(static_cast<uint_t>(head) >= m_start[Dim-1] && static_cast<uint_t>(head) < m_stop[Dim-1]);
      return (head - m_start[Dim-1]) * m_stride[Dim-1];
    }

    template<typename Head, typename... Tail>
    uint_t address(Head head, Tail... tail) {
      uint_t const d = (Dim-1) - sizeof...(tail);
      assert(static_cast<uint_t>(head) >= m_start[d] && static_cast<uint_t>(head) < m_stop[d]);
      return (head - m_start[d]) * m_stride[d] + address(tail...);
    }

    template<typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
    void extractDim(uint_t*& begin, uint_t*&, uint_t*&, uint_t dimNo, T entry) {
      assert(static_cast<uint_t>(entry) >= m_start[dimNo] && static_cast<uint_t>(entry) < m_stop[dimNo]);
      *begin++ = entry;
    }

    template<typename T, typename std::enable_if<std::is_same<T, slice<uint_t>>::value, int>::type = 0>
    void extractDim(uint_t*& begin, uint_t*& size, uint_t*& stride, uint_t dimNo, T dim) {
      *begin = std::max(m_start[dimNo], dim.start);
      *size++ = std::min(m_stop[dimNo], dim.stop) - *begin;
      ++begin;
      *stride++ = m_stride[dimNo];
    }

    template<typename Head>
    void extractSubtensor(uint_t* begin, uint_t* size, uint_t* stride, Head head) {
      extractDim<Head>(begin, size, stride, Dim-1, head);
    }

    template<typename Head, typename... Tail>
    void extractSubtensor(uint_t* begin, uint_t* size, uint_t* stride, Head head, Tail... tail) {
      uint_t const d = (Dim-1) - sizeof...(tail);
      extractDim<Head>(begin, size, stride, d, head);
      extractSubtensor(begin, size, stride, tail...);
    }

    real_t* m_values;
    uint_t m_start[Dim];
    uint_t m_stop[Dim];
    uint_t m_stride[Dim];
  };

  template<typename real_t, typename uint_t>
  class DenseTensorView<0,real_t,uint_t> : public TensorView<0, real_t, uint_t> {
  public:
    explicit DenseTensorView(real_t* values, std::initializer_list<uint_t> shape, std::initializer_list<uint_t> start, std::initializer_list<uint_t> stop)
      : TensorView<0, real_t, uint_t>(shape), m_values(values) {
    }

    uint_t size() const {
      return 1;
    }

    void setZero() {
      m_values[0] = 0.0;
    }

    template<class view_t>
    void copyToView(view_t& other) {
      other.m_values[0] = m_values[0];
    }

  protected:
    real_t* m_values;
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

    real_t& operator()(uint_t row, uint_t col) {
      assert(col >= 0 && col < this->shape(1));
      uint_t addr = m_colPtr[ col ];
      uint_t stop = m_colPtr[ col+1 ];
      while (addr < stop) {
        if (m_rowInd[addr] == row) {
          break;
        }
        ++addr;
      }
      assert(addr != stop);

      return m_values[addr];
    }

    real_t& operator[](uint_t entry[2]) {
      return operator()(entry[0], entry[1]);
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
