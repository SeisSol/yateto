#ifndef YATETO_TENSORVIEW_H_
#define YATETO_TENSORVIEW_H_

#include "Marker.h"

#include <cassert>
#include <initializer_list>
#include <limits>
#include <type_traits>

namespace yateto {
namespace detail {
template <typename T>
YATETO_HOSTDEVICE constexpr T smaller(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
YATETO_HOSTDEVICE constexpr T larger(T a, T b) {
  return a < b ? b : a;
}
} // namespace detail

template <typename uint_t = unsigned>
class slice {
  public:
  YATETO_HOSTDEVICE constexpr explicit slice(uint_t start = 0,
                                             uint_t stop = std::numeric_limits<uint_t>::max())
      : start(start), stop(stop) {}

  uint_t start;
  uint_t stop;
};

template <typename uint_t, typename... Entry>
struct count_slices : std::integral_constant<uint_t, 0> {};
template <typename uint_t, typename Head, typename... Tail>
struct count_slices<uint_t, Head, Tail...>
    : std::integral_constant<uint_t,
                             ((std::is_same_v<Head, slice<uint_t>>) ? 1 : 0) +
                                 count_slices<uint_t, Tail...>::value> {};

template <unsigned Dim, typename real_t, typename uint_t>
class TensorView {
  public:
  YATETO_HOSTDEVICE constexpr explicit TensorView(std::initializer_list<uint_t> shape) {
    assert(shape.size() == Dim && "YATETO: the shape does not match the tensor dimension");
    copyInto(shape, m_shape);
  }

  YATETO_HOSTDEVICE constexpr explicit TensorView(const uint_t shape[]) {
    for (uint_t d = 0; d < Dim; ++d) {
      m_shape[d] = shape[d];
    }
  }

  YATETO_HOSTDEVICE static constexpr uint_t dim() { return Dim; }

  YATETO_HOSTDEVICE constexpr uint_t shape(uint_t dim) const {
    assert(dim < Dim && "YATETO: the requested dimension is not part of the tensor");
    return m_shape[dim];
  }

  protected:
  /** Fills a Dim-element array from a list of exactly that length.
   *
   *  Stands in for std::copy, which is not a constant expression before
   *  C++20 and is not available to device code.
   * */
  YATETO_HOSTDEVICE static constexpr void copyInto(std::initializer_list<uint_t> from,
                                                   uint_t (&to)[Dim]) {
    uint_t d = 0;
    for (const uint_t value : from) {
      to[d] = value;
      ++d;
    }
  }

  uint_t m_shape[Dim]{};
};

template <typename real_t, typename uint_t>
class TensorView<0, real_t, uint_t> {
  public:
  YATETO_HOSTDEVICE constexpr explicit TensorView(std::initializer_list<uint_t> shape) {
    assert(shape.size() == 0 && "YATETO: the shape does not match the tensor dimension");
  }

  YATETO_HOSTDEVICE constexpr explicit TensorView(const uint_t /*shape*/[]) {}

  YATETO_HOSTDEVICE static constexpr uint_t dim() { return 0; }

  YATETO_HOSTDEVICE constexpr uint_t shape(uint_t /*dim*/) const { return 0; }
};

template <unsigned Dim, typename real_t, typename uint_t = unsigned, bool Const = false>
class DenseTensorView : public TensorView<Dim, real_t, uint_t> {
  public:
  using data_t = std::conditional_t<Const, const real_t*, real_t*>;
  using dataref_t = std::conditional_t<Const, const real_t&, real_t&>;

  YATETO_HOSTDEVICE constexpr explicit DenseTensorView(data_t values,
                                                       std::initializer_list<uint_t> shape,
                                                       std::initializer_list<uint_t> start,
                                                       std::initializer_list<uint_t> stop)
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values) {
    assert(start.size() == Dim && "YATETO: the start does not match the tensor dimension");
    assert(stop.size() == Dim && "YATETO: the stop does not match the tensor dimension");
    TensorView<Dim, real_t, uint_t>::copyInto(start, m_start);
    TensorView<Dim, real_t, uint_t>::copyInto(stop, m_stop);
    computeStride();
  }

  YATETO_HOSTDEVICE constexpr explicit DenseTensorView(data_t values,
                                                       std::initializer_list<uint_t> shape)
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values), m_start{} {
    assert(shape.size() == Dim && "YATETO: the shape does not match the tensor dimension");
    TensorView<Dim, real_t, uint_t>::copyInto(shape, m_stop);
    computeStride();
  }

  YATETO_HOSTDEVICE constexpr explicit DenseTensorView(data_t values,
                                                       const uint_t shape[],
                                                       const uint_t start[],
                                                       const uint_t stop[])
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values) {
    for (uint_t d = 0; d < Dim; ++d) {
      m_start[d] = start[d];
      m_stop[d] = stop[d];
    }
    computeStride();
  }

  YATETO_HOSTDEVICE constexpr explicit DenseTensorView(data_t values, const uint_t shape[])
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values), m_start{} {
    for (uint_t d = 0; d < Dim; ++d) {
      m_stop[d] = shape[d];
    }
    computeStride();
  }

  YATETO_HOSTDEVICE constexpr explicit DenseTensorView(data_t values,
                                                       const uint_t shape[],
                                                       const uint_t stride[])
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values), m_start{} {
    for (uint_t d = 0; d < Dim; ++d) {
      m_stop[d] = shape[d];
      m_stride[d] = stride[d];
    }
  }

  YATETO_HOSTDEVICE constexpr uint_t size() const {
    return (m_stop[Dim - 1] - m_start[Dim - 1]) * m_stride[Dim - 1];
  }

  template <typename F>
  YATETO_HOSTDEVICE constexpr void forall(F&& function) {
    forallImpl(*this, function);
  }

  template <typename F>
  YATETO_HOSTDEVICE constexpr void forall(F&& function) const {
    forallImpl(*this, function);
  }

  template <bool Writable = !Const, typename = std::enable_if_t<Writable>>
  YATETO_HOSTDEVICE constexpr void setZero() {
    forall([](const uint_t* /*entry*/, real_t& value) { value = real_t{}; });
  }

  protected:
  template <typename Head>
  YATETO_HOSTDEVICE constexpr bool
      isInRange(const uint_t start[Dim], const uint_t stop[Dim], int dim, Head head) const {
    return static_cast<uint_t>(head) >= start[dim] && static_cast<uint_t>(head) < stop[dim];
  }

  template <typename Head, typename... Tail>
  YATETO_HOSTDEVICE constexpr bool isInRange(
      const uint_t start[Dim], const uint_t stop[Dim], int dim, Head head, Tail... tail) const {
    return static_cast<uint_t>(head) >= start[dim] && static_cast<uint_t>(head) < stop[dim] &&
           isInRange(start, stop, dim + 1, tail...);
  }

  public:
  template <typename... Entry>
  YATETO_HOSTDEVICE constexpr bool isInRange(Entry... entry) const {
    static_assert(sizeof...(entry) == Dim,
                  "Number of arguments to isInRange(...) does not match the "
                  "tensor dimension.");
    return isInRange(m_start, m_stop, 0, entry...);
  }

  template <typename... Entry>
  YATETO_HOSTDEVICE constexpr dataref_t operator()(Entry... entry) {
    static_assert(sizeof...(entry) == Dim,
                  "Number of arguments to operator() does not match the tensor "
                  "dimension.");
    assert(isInRange(entry...));
    return m_values[address(entry...)];
  }

  template <typename... Entry>
  YATETO_HOSTDEVICE constexpr const real_t& operator()(Entry... entry) const {
    static_assert(sizeof...(entry) == Dim,
                  "Number of arguments to operator() const does not match the "
                  "tensor dimension.");
    assert(isInRange(entry...));
    return m_values[address(entry...)];
  }

  YATETO_HOSTDEVICE constexpr const real_t& operator[](const uint_t entry[Dim]) const {
    uint_t addr = 0;
    for (uint_t d = 0; d < Dim; ++d) {
      assert(entry[d] >= m_start[d] && entry[d] < m_stop[d]);
      addr += (entry[d] - m_start[d]) * m_stride[d];
    }
    return m_values[addr];
  }

  YATETO_HOSTDEVICE constexpr dataref_t operator[](const uint_t entry[Dim]) {
    uint_t addr = 0;
    for (uint_t d = 0; d < Dim; ++d) {
      assert(entry[d] >= m_start[d] && entry[d] < m_stop[d]);
      addr += (entry[d] - m_start[d]) * m_stride[d];
    }
    return m_values[addr];
  }

  template <class view_t>
  YATETO_HOSTDEVICE constexpr void copyToView(view_t& other) const {
    assert(Dim == other.dim());

    uint_t entry[Dim]{};
    for (uint_t d = 0; d < Dim; ++d) {
      assert(this->shape(d) == other.shape(d));
      entry[d] = m_start[d];
    }

    uint_t stop0 = detail::smaller(m_stop[0], this->shape(0));
    data_t val = m_values;
    while (entry[Dim - 1] != m_stop[Dim - 1]) {
      for (uint_t i = m_start[0]; i < stop0; ++i) {
        entry[0] = i;
        other[entry] = *(val++);
      }
      val += (m_stop[0] - stop0);

      if constexpr (Dim == 1) {
        break;
      } else {

        uint_t d = 0;
        do {
          entry[d] = m_start[d];
          d++;
          ++entry[d];
        } while (entry[d] == m_stop[d] && d < Dim - 1);
      }
    }
  }

  template <typename... Entry>
  YATETO_HOSTDEVICE constexpr auto subtensor(Entry... entry)
      -> DenseTensorView<count_slices<uint_t, Entry...>::value, real_t, uint_t, Const> {
    static_assert(sizeof...(entry) == Dim,
                  "Number of arguments to subtensor() does not match tensor dimension.");
    constexpr auto nSlices = count_slices<uint_t, Entry...>::value;
    uint_t begin[Dim]{};
    uint_t size[nSlices]{};
    uint_t stride[nSlices]{};
    extractSubtensor(begin, size, stride, entry...);
    DenseTensorView<nSlices, real_t, uint_t, Const> subtensor(&operator[](begin), size, stride);
    return subtensor;
  }

  template <typename... Entry>
  YATETO_HOSTDEVICE constexpr auto subtensor(Entry... entry) const
      -> DenseTensorView<count_slices<uint_t, Entry...>::value, real_t, uint_t, true> {
    static_assert(sizeof...(entry) == Dim,
                  "Number of arguments to subtensor() does not match tensor dimension.");
    constexpr auto nSlices = count_slices<uint_t, Entry...>::value;
    uint_t begin[Dim]{};
    uint_t size[nSlices]{};
    uint_t stride[nSlices]{};
    extractSubtensor(begin, size, stride, entry...);
    DenseTensorView<nSlices, real_t, uint_t, true> subtensor(&operator[](begin), size, stride);
    return subtensor;
  }

  YATETO_HOSTDEVICE constexpr data_t data() { return m_values; }

  YATETO_HOSTDEVICE constexpr const real_t* data() const { return m_values; }

  protected:
  /** Visits every entry of the view, innermost dimension first.
   *
   *  @param self the view to walk over; const or non-const.
   *  @param function called with the current index tuple and the entry.
   * */
  template <typename Self, typename F>
  YATETO_HOSTDEVICE constexpr static void forallImpl(Self& self, F&& function) {
    uint_t entry[Dim]{};
    for (uint_t d = 0; d < Dim; ++d) {
      entry[d] = self.m_start[d];
    }
    while (entry[Dim - 1] != self.m_stop[Dim - 1]) {
      auto* values = &self[entry];
      for (uint_t i = 0; i < self.m_stop[0] - self.m_start[0]; ++i) {
        entry[0] = i + self.m_start[0];
        function(entry, values[i * self.m_stride[0]]);
      }
      if constexpr (Dim == 1) {
        break;
      } else {
        uint_t d = 0;
        do {
          entry[d] = self.m_start[d];
          d++;
          ++entry[d];
        } while (entry[d] == self.m_stop[d] && d < Dim - 1);
      }
    }
  }

  YATETO_HOSTDEVICE constexpr void computeStride() {
    m_stride[0] = 1;
    for (uint_t d = 0; d < Dim - 1; ++d) {
      m_stride[d + 1] = m_stride[d] * (m_stop[d] - m_start[d]);
    }
  }

  template <typename Head>
  YATETO_HOSTDEVICE constexpr uint_t address(Head head) const {
    assert(static_cast<uint_t>(head) >= m_start[Dim - 1] &&
           static_cast<uint_t>(head) < m_stop[Dim - 1]);
    return (head - m_start[Dim - 1]) * m_stride[Dim - 1];
  }

  template <typename Head, typename... Tail>
  YATETO_HOSTDEVICE constexpr uint_t address(Head head, Tail... tail) const {
    const uint_t d = (Dim - 1) - sizeof...(tail);
    assert(static_cast<uint_t>(head) >= m_start[d] && static_cast<uint_t>(head) < m_stop[d]);
    return (head - m_start[d]) * m_stride[d] + address(tail...);
  }

  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  YATETO_HOSTDEVICE constexpr void
      extractDim(uint_t*& begin, uint_t*&, uint_t*&, uint_t dimNo, T entry) const {
    assert(static_cast<uint_t>(entry) >= m_start[dimNo] &&
           static_cast<uint_t>(entry) < m_stop[dimNo]);
    *begin++ = entry;
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, slice<uint_t>>, int> = 0>
  YATETO_HOSTDEVICE constexpr void
      extractDim(uint_t*& begin, uint_t*& size, uint_t*& stride, uint_t dimNo, T dim) const {
    *begin = detail::larger(m_start[dimNo], dim.start);
    *size++ = detail::smaller(m_stop[dimNo], dim.stop) - *begin;
    ++begin;
    *stride++ = m_stride[dimNo];
  }

  template <typename Head>
  YATETO_HOSTDEVICE constexpr void
      extractSubtensor(uint_t* begin, uint_t* size, uint_t* stride, Head head) const {
    extractDim<Head>(begin, size, stride, Dim - 1, head);
  }

  template <typename Head, typename... Tail>
  void
      extractSubtensor(uint_t* begin, uint_t* size, uint_t* stride, Head head, Tail... tail) const {
    const uint_t d = (Dim - 1) - sizeof...(tail);
    extractDim<Head>(begin, size, stride, d, head);
    extractSubtensor(begin, size, stride, tail...);
  }

  data_t m_values;
  uint_t m_start[Dim]{};
  uint_t m_stop[Dim]{};
  uint_t m_stride[Dim]{};
};

template <typename real_t, typename uint_t, bool Const>
class DenseTensorView<0, real_t, uint_t, Const> : public TensorView<0, real_t, uint_t> {
  public:
  using data_t = std::conditional_t<Const, const real_t*, real_t*>;
  using dataref_t = std::conditional_t<Const, const real_t&, real_t&>;

  YATETO_HOSTDEVICE constexpr explicit DenseTensorView(data_t values,
                                                       std::initializer_list<uint_t> shape,
                                                       std::initializer_list<uint_t> start,
                                                       std::initializer_list<uint_t> stop)
      : TensorView<0, real_t, uint_t>(shape), m_values(values) {
    assert(start.size() == 0 && "YATETO: the start does not match the tensor dimension");
    assert(stop.size() == 0 && "YATETO: the stop does not match the tensor dimension");
  }

  YATETO_HOSTDEVICE constexpr uint_t size() const { return 1; }

  template <bool Writable = !Const, typename = std::enable_if_t<Writable>>
  YATETO_HOSTDEVICE constexpr void setZero() {
    m_values[0] = real_t{};
  }

  YATETO_HOSTDEVICE constexpr data_t data() { return m_values; }

  YATETO_HOSTDEVICE constexpr const real_t* data() const { return m_values; }

  template <class view_t>
  YATETO_HOSTDEVICE constexpr void copyToView(view_t& other) const {
    assert(0 == other.dim());
    other.data()[0] = m_values[0];
  }

  protected:
  data_t m_values;
};

template <typename real_t, typename uint_t, bool Const = false>
class CSCMatrixView : public TensorView<2, real_t, uint_t> {
  public:
  using data_t = std::conditional_t<Const, const real_t*, real_t*>;
  using dataref_t = std::conditional_t<Const, const real_t&, real_t&>;

  YATETO_HOSTDEVICE constexpr explicit CSCMatrixView(data_t values,
                                                     std::initializer_list<uint_t> shape,
                                                     const uint_t* rowInd,
                                                     const uint_t* colPtr)
      : TensorView<2, real_t, uint_t>(shape), m_values(values), m_rowInd(rowInd), m_colPtr(colPtr) {
  }

  YATETO_HOSTDEVICE constexpr explicit CSCMatrixView(data_t values,
                                                     const uint_t shape[],
                                                     const uint_t* rowInd,
                                                     const uint_t* colPtr)
      : TensorView<2, real_t, uint_t>(shape), m_values(values), m_rowInd(rowInd), m_colPtr(colPtr) {
  }

  YATETO_HOSTDEVICE constexpr uint_t size() const { return m_colPtr[this->shape(1)]; }

  template <bool Writable = !Const, typename = std::enable_if_t<Writable>>
  YATETO_HOSTDEVICE constexpr void setZero() {
    const uint_t stored = size();
    for (uint_t i = 0; i < stored; ++i) {
      m_values[i] = real_t{};
    }
  }

  YATETO_HOSTDEVICE constexpr const real_t& operator()(uint_t row, uint_t col) const {
    const uint_t addr = address(row, col);
    assert(addr != m_colPtr[col + 1]);
    return m_values[addr];
  }

  YATETO_HOSTDEVICE constexpr dataref_t operator()(uint_t row, uint_t col) {
    const uint_t addr = address(row, col);
    assert(addr != m_colPtr[col + 1]);
    return m_values[addr];
  }

  YATETO_HOSTDEVICE constexpr bool isInRange(uint_t row, uint_t col) const {
    return address(row, col) != m_colPtr[col + 1];
  }

  YATETO_HOSTDEVICE constexpr dataref_t operator[](const uint_t entry[2]) {
    return operator()(entry[0], entry[1]);
  }

  YATETO_HOSTDEVICE constexpr const real_t& operator[](const uint_t entry[2]) const {
    return operator()(entry[0], entry[1]);
  }

  template <class view_t>
  YATETO_HOSTDEVICE constexpr void copyToView(view_t& other) {
    assert(2 == other.dim());
    assert(this->shape(0) == other.shape(0) && this->shape(1) == other.shape(1));

    uint_t entry[2];
    uint_t ncols = this->shape(1);
    for (uint_t col = 0; col < ncols; ++col) {
      entry[1] = col;
      for (uint_t i = m_colPtr[col]; i < m_colPtr[col + 1]; ++i) {
        entry[0] = m_rowInd[i];
        other[entry] = m_values[i];
      }
    }
  }

  template <typename F>
  YATETO_HOSTDEVICE constexpr void forall(F&& function) {
    uint_t entry[2];
    uint_t ncols = this->shape(1);
    for (uint_t col = 0; col < ncols; ++col) {
      entry[1] = col;
      for (uint_t i = m_colPtr[col]; i < m_colPtr[col + 1]; ++i) {
        entry[0] = m_rowInd[i];
        function(entry, m_values[i]);
      }
    }
  }

  template <typename F>
  YATETO_HOSTDEVICE constexpr void forall(F&& function) const {
    uint_t entry[2];
    uint_t ncols = this->shape(1);
    for (uint_t col = 0; col < ncols; ++col) {
      entry[1] = col;
      for (uint_t i = m_colPtr[col]; i < m_colPtr[col + 1]; ++i) {
        entry[0] = m_rowInd[i];
        function(entry, m_values[i]);
      }
    }
  }

  protected:
  /** Looks up where an entry is stored.
   *
   *  @param row the row of the entry.
   *  @param col the column of the entry.
   *  @return the offset into the value array, or the end of the column if the
   *          entry is not stored.
   * */
  YATETO_HOSTDEVICE constexpr uint_t address(uint_t row, uint_t col) const {
    assert(col < this->shape(1));
    uint_t addr = m_colPtr[col];
    const uint_t stop = m_colPtr[col + 1];
    while (addr < stop && m_rowInd[addr] != row) {
      ++addr;
    }
    return addr;
  }

  data_t m_values;
  const uint_t* m_rowInd;
  const uint_t* m_colPtr;
};

template <unsigned Dim, typename real_t, typename uint_t, bool Const = false>
class PatternTensorView : public TensorView<Dim, real_t, uint_t> {
  public:
  using data_t = std::conditional_t<Const, const real_t*, real_t*>;
  using dataref_t = std::conditional_t<Const, const real_t&, real_t&>;

  YATETO_HOSTDEVICE constexpr explicit PatternTensorView(data_t values,
                                                         std::initializer_list<uint_t> shape,
                                                         const uint_t* pattern)
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values), m_pattern(pattern, shape) {

    computeSize();
  }

  YATETO_HOSTDEVICE constexpr explicit PatternTensorView(data_t values,
                                                         const uint_t shape[],
                                                         const uint_t* pattern)
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values), m_pattern(pattern, shape) {

    computeSize();
  }

  YATETO_HOSTDEVICE constexpr explicit PatternTensorView(
      data_t values,
      std::initializer_list<uint_t> shape,
      DenseTensorView<Dim, uint_t, uint_t, true> pattern)
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values), m_pattern(pattern) {

    computeSize();
  }

  YATETO_HOSTDEVICE constexpr explicit PatternTensorView(
      data_t values, const uint_t shape[], DenseTensorView<Dim, uint_t, uint_t, true> pattern)
      : TensorView<Dim, real_t, uint_t>(shape), m_values(values), m_pattern(pattern) {

    computeSize();
  }

  YATETO_HOSTDEVICE constexpr uint_t size() const { return m_size; }

  template <bool Writable = !Const, typename = std::enable_if_t<Writable>>
  YATETO_HOSTDEVICE constexpr void setZero() {
    m_pattern.forall([&](const auto& /*index*/, const auto& idxval) {
      if (idxval > 0) {
        m_values[idxval - 1] = 0;
      }
    });
  }

  template <typename... Args>
  YATETO_HOSTDEVICE constexpr const real_t& operator()(Args... index) const {
    static_assert((std::is_integral_v<Args> && ...));
    const auto idx = m_pattern(index...);
    assert(idx > 0 && "YATETO: the requested entry is not part of the tensor pattern");
    return m_values[idx - 1];
  }

  template <typename... Args>
  YATETO_HOSTDEVICE constexpr dataref_t operator()(Args... index) {
    static_assert((std::is_integral_v<Args> && ...));
    const auto idx = m_pattern(index...);
    assert(idx > 0 && "YATETO: the requested entry is not part of the tensor pattern");
    return m_values[idx - 1];
  }

  template <typename... Args>
  YATETO_HOSTDEVICE constexpr bool isInRange(Args... index) const {
    static_assert((std::is_integral_v<Args> && ...));
    const auto idx = m_pattern(index...);
    return idx > 0;
  }

  YATETO_HOSTDEVICE constexpr dataref_t operator[](const uint_t entry[Dim]) {
    const auto idx = m_pattern[entry];
    assert(idx > 0 && "YATETO: the requested entry is not part of the tensor pattern");
    return m_values[idx - 1];
  }

  YATETO_HOSTDEVICE constexpr const real_t& operator[](const uint_t entry[Dim]) const {
    const auto idx = m_pattern[entry];
    assert(idx > 0 && "YATETO: the requested entry is not part of the tensor pattern");
    return m_values[idx - 1];
  }

  template <typename F>
  YATETO_HOSTDEVICE constexpr void forall(F&& function) {
    m_pattern.forall([&](const auto& index, const auto& idxval) {
      if (idxval > 0) {
        function(index, m_values[idxval - 1]);
      }
    });
  }

  template <typename F>
  YATETO_HOSTDEVICE constexpr void forall(F&& function) const {
    m_pattern.forall([&](const auto& index, const auto& idxval) {
      if (idxval > 0) {
        function(index, m_values[idxval - 1]);
      }
    });
  }

  template <class view_t>
  YATETO_HOSTDEVICE constexpr void copyToView(view_t& other) const {
    m_pattern.forall([&](const auto& index, const auto& idxval) {
      if (idxval > 0) {
        other[index] = m_values[idxval - 1];
      } else {
        other[index] = 0;
      }
    });
  }

  template <typename... Entry>
  YATETO_HOSTDEVICE constexpr auto subtensor(Entry... entry) {
    static_assert(sizeof...(entry) == Dim,
                  "Number of arguments to subtensor() does not match tensor dimension.");
    constexpr auto nSlices = count_slices<uint_t, Entry...>::value;
    const auto patternSubtensor = m_pattern.subtensor(entry...);
    uint_t subShape[nSlices > 0 ? nSlices : 1]{};
    for (uint_t d = 0; d < nSlices; ++d) {
      subShape[d] = patternSubtensor.shape(d);
    }
    return PatternTensorView<nSlices, real_t, uint_t, Const>(m_values, subShape, patternSubtensor);
  }

  template <typename... Entry>
  YATETO_HOSTDEVICE constexpr auto subtensor(Entry... entry) const {
    static_assert(sizeof...(entry) == Dim,
                  "Number of arguments to subtensor() does not match tensor dimension.");
    constexpr auto nSlices = count_slices<uint_t, Entry...>::value;
    const auto patternSubtensor = m_pattern.subtensor(entry...);
    uint_t subShape[nSlices > 0 ? nSlices : 1]{};
    for (uint_t d = 0; d < nSlices; ++d) {
      subShape[d] = patternSubtensor.shape(d);
    }
    return PatternTensorView<nSlices, real_t, uint_t, true>(m_values, subShape, patternSubtensor);
  }

  protected:
  YATETO_HOSTDEVICE constexpr void computeSize() {
    m_size = 0;
    m_pattern.forall([&](const auto& /*index*/, const auto& idxval) {
      if (idxval > 0) {
        ++m_size;
      }
    });
  }

  data_t m_values{nullptr};
  DenseTensorView<Dim, uint_t, uint_t, true> m_pattern;
  uint_t m_size{0};
};
} // namespace yateto

#endif // YATETO_TENSORVIEW_H_
