#ifndef YATETO_RUNTIMEVIEW_H_
#define YATETO_RUNTIMEVIEW_H_

#include "Descriptor.h"
#include "Type.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace yateto {

/// Values of a tensor and where in them each entry is: a view whose layout is known at run time.
struct ConstRuntimeView {
  const TensorDescriptor* layout{nullptr};
  const void* data{nullptr};
};

/// A ConstRuntimeView through which the values may be written.
struct RuntimeView {
  const TensorDescriptor* layout{nullptr};
  void* data{nullptr};

  constexpr operator ConstRuntimeView() const noexcept { return {layout, data}; }
};

/// A family of anything -- views, scalars -- indexed like a family of tensors: by one index per
/// dimension, the first one running fastest.
template <typename T, unsigned... Size>
struct Family {
  static_assert(sizeof...(Size) > 0, "a family has at least one index");
  static constexpr unsigned Count = (Size * ...);

  T data[Count]{};

  template <typename... Index>
  constexpr T& operator()(Index... index) noexcept {
    return data[position(static_cast<unsigned>(index)...)];
  }

  template <typename... Index>
  constexpr const T& operator()(Index... index) const noexcept {
    return data[position(static_cast<unsigned>(index)...)];
  }

  /// Where the member at `index` is in `data`.
  template <typename... Index>
  static constexpr unsigned position(Index... index) noexcept {
    static_assert(sizeof...(Index) == sizeof...(Size), "one index per dimension of the family");
    const unsigned sizes[] = {Size...};
    const unsigned indices[] = {static_cast<unsigned>(index)...};
    unsigned result = 0;
    unsigned stride = 1;
    for (unsigned d = 0; d < sizeof...(Size); ++d) {
      assert(indices[d] < sizes[d] && "YATETO: the family has no such member");
      result += indices[d] * stride;
      stride *= sizes[d];
    }
    return result;
  }
};

namespace detail {
template <typename T>
struct Tag {
  using Type = T;
};

/// Calls `function(Tag<T>{})` with the type `datatype` stands for.
template <typename F>
void withType(Datatype datatype, F&& function) {
  switch (datatype) {
  case Datatype::Bool:
    function(Tag<bool>{});
    return;
  case Datatype::I8:
    function(Tag<std::int8_t>{});
    return;
  case Datatype::I16:
    function(Tag<std::int16_t>{});
    return;
  case Datatype::I32:
    function(Tag<std::int32_t>{});
    return;
  case Datatype::I64:
    function(Tag<std::int64_t>{});
    return;
  case Datatype::F32:
    function(Tag<float>{});
    return;
  case Datatype::F64:
    function(Tag<double>{});
    return;
  case Datatype::F16:
#if YATETO_HAS_F16
    function(Tag<f16_ty>{});
    return;
#else
    break;
#endif
  case Datatype::BF16:
#if YATETO_HAS_BF16
    function(Tag<bf16_ty>{});
    return;
#else
    break;
#endif
  case Datatype::F128:
#if YATETO_HAS_F128
    function(Tag<f128_ty>{});
    return;
#else
    break;
#endif
  }
  throw std::invalid_argument("YATETO: this compiler has no type for the values of the view");
}

/// Whether `T` is one of the 16 bit floating point types, which compilers do not all convert
/// between directly.
template <typename T>
constexpr bool isHalf() noexcept {
#if YATETO_HAS_F16
  if constexpr (std::is_same_v<T, f16_ty>) {
    return true;
  }
#endif
#if YATETO_HAS_BF16
  if constexpr (std::is_same_v<T, bf16_ty>) {
    return true;
  }
#endif
  return false;
}

/// `value` as a `To`; a 16 bit floating point value is converted by way of a float, which each of
/// them converts to and from.
template <typename To, typename From>
To convert(From value) noexcept {
  if constexpr (std::is_same_v<To, From>) {
    return value;
  } else if constexpr (isHalf<To>() || isHalf<From>()) {
    return static_cast<To>(static_cast<float>(value));
  } else {
    return static_cast<To>(value);
  }
}

/// How many dimensions a tensor may have for its entries to be walked here.
constexpr unsigned MaxRank = 16;

inline bool inShape(const TensorDescriptor& layout, const unsigned* index) noexcept {
  for (unsigned d = 0; d < layout.rank; ++d) {
    if (index[d] >= layout.shape[d]) {
      return false;
    }
  }
  return true;
}

/// Calls `function(index, offset)` for every value `layout` stores, with the entry it belongs to.
template <typename F>
void forEachStored(const TensorDescriptor& layout, F&& function) {
  if (layout.rank > MaxRank) {
    throw std::invalid_argument("YATETO: a view has more dimensions than can be walked");
  }
  unsigned index[MaxRank + 1]{};
  switch (layout.storage) {
  case Storage::Dense:
  case Storage::Pattern: {
    for (unsigned d = 0; d < layout.rank; ++d) {
      if (layout.start[d] >= layout.stop[d]) {
        return;
      }
      index[d] = layout.start[d];
    }
    if (layout.rank == 0) {
      function(static_cast<const unsigned*>(index), std::ptrdiff_t{0});
      return;
    }
    while (true) {
      std::ptrdiff_t base = 0;
      for (unsigned d = 1; d < layout.rank; ++d) {
        base += static_cast<std::ptrdiff_t>(index[d] - layout.start[d]) * layout.stride[d];
      }
      for (unsigned i = layout.start[0]; i < layout.stop[0]; ++i) {
        index[0] = i;
        const std::ptrdiff_t entry =
            base + static_cast<std::ptrdiff_t>(i - layout.start[0]) * layout.stride[0];
        if (layout.storage == Storage::Dense) {
          function(static_cast<const unsigned*>(index), entry);
        } else if (layout.pattern[entry] > 0) {
          function(static_cast<const unsigned*>(index),
                   static_cast<std::ptrdiff_t>(layout.pattern[entry]) - 1);
        }
      }
      unsigned d = 1;
      while (d < layout.rank) {
        if (++index[d] < layout.stop[d]) {
          break;
        }
        index[d] = layout.start[d];
        ++d;
      }
      if (d >= layout.rank) {
        return;
      }
    }
  }
  case Storage::CSC: {
    for (unsigned column = 0; column < layout.shape[1]; ++column) {
      index[1] = column;
      for (unsigned i = layout.columnPointer[column]; i < layout.columnPointer[column + 1]; ++i) {
        index[0] = layout.rowIndex[i];
        function(static_cast<const unsigned*>(index), static_cast<std::ptrdiff_t>(i));
      }
    }
    return;
  }
  }
}

/// Writes the value at `from`, a `From`, as a `To` to `to`.
template <typename From, typename To>
void convertOne(const void* from, void* to) noexcept {
  *static_cast<To*>(to) = convert<To>(*static_cast<const From*>(from));
}

using Converter = void (*)(const void*, void*);

/// Converts one value of `from` into one of `to`.
///
/// One conversion per value, chosen once per copy, rather than a loop per pair of element types:
/// a copy serves kernels that run rarely, and a hundred loops would take every translation unit
/// that copies longer to compile than its copies ever take to run.
inline Converter converter(Datatype from, Datatype to) {
  Converter result = nullptr;
  withType(from, [&](auto fromTag) {
    withType(to, [&](auto toTag) {
      result = &convertOne<typename decltype(fromTag)::Type, typename decltype(toTag)::Type>;
    });
  });
  return result;
}

inline void copyStored(const void* from,
                       const TensorDescriptor& fromLayout,
                       void* to,
                       const TensorDescriptor& toLayout) {
  const Converter convertValue = converter(fromLayout.datatype, toLayout.datatype);
  const auto fromSize = static_cast<std::ptrdiff_t>(sizeOf(fromLayout.datatype));
  const auto toSize = static_cast<std::ptrdiff_t>(sizeOf(toLayout.datatype));
  const auto* source = static_cast<const unsigned char*>(from);
  auto* target = static_cast<unsigned char*>(to);
  forEachStored(toLayout, [&](const unsigned* index, std::ptrdiff_t offset) {
    unsigned char* value = target + offset * toSize;
    const std::ptrdiff_t at = inShape(toLayout, index) ? offsetOf(fromLayout, index) : -1;
    if (at < 0) {
      // Zero in every element type there is.
      std::memset(value, 0, static_cast<std::size_t>(toSize));
    } else {
      convertValue(source + at * fromSize, value);
    }
  });
}

inline bool sameShape(const TensorDescriptor& a, const TensorDescriptor& b) noexcept {
  bool same = a.rank == b.rank;
  for (unsigned d = 0; same && d < a.rank; ++d) {
    same = a.shape[d] == b.shape[d];
  }
  return same;
}

struct AlignedDelete {
  std::size_t alignment;
  void operator()(void* pointer) const noexcept {
    ::operator delete(pointer, std::align_val_t{alignment});
  }
};
} // namespace detail

/// Copies what `from` holds into what `to` stores.
///
/// Every entry `to` stores gets the value `from` has for it, converted to the element type of
/// `to`, or zero where `from` stores nothing for it -- the tensor is zero wherever its layout
/// stores nothing. An entry of `to` outside the shape is padding and gets zero as well. Both
/// views have to hold a tensor of the same shape.
inline void copy(ConstRuntimeView from, RuntimeView to) {
  if (from.layout == nullptr || to.layout == nullptr) {
    throw std::invalid_argument("YATETO: a view without a layout");
  }
  if (!detail::sameShape(*from.layout, *to.layout)) {
    throw std::invalid_argument("YATETO: the views hold tensors of different shapes");
  }
  detail::copyStored(from.data, *from.layout, to.data, *to.layout);
}

/// Whether a kernel generated for `own` can take the values of `view` as they are: in that
/// layout, and at an address aligned as it asks.
inline bool passesThrough(ConstRuntimeView view, const TensorDescriptor& own) noexcept {
  return (view.layout == &own || sameLayout(*view.layout, own)) &&
         reinterpret_cast<std::uintptr_t>(view.data) % own.alignment == 0;
}

/// An operand of a kernel, in the layout the kernel was generated for.
///
/// It is the values of the view where they are in that layout already, and a copy of them in
/// memory of its own where they are not. A copy of a view the kernel writes starts out with the
/// values of the view, so that what the kernel does not write stays as it was, and `finish`
/// writes it back. `what` names the operand in what is thrown.
template <typename T>
class Operand {
  public:
  Operand(ConstRuntimeView view, const TensorDescriptor& own, const char* what = "an operand")
      : m_view{view.layout, const_cast<void*>(view.data)}, m_own(own), m_writable(false),
        m_what(what) {
    bind();
  }

  Operand(RuntimeView view, const TensorDescriptor& own, const char* what = "an operand")
      : m_view(view), m_own(own), m_writable(true), m_what(what) {
    bind();
  }

  Operand(const Operand&) = delete;
  Operand& operator=(const Operand&) = delete;

  /// What to hand the kernel.
  T* data() const noexcept { return m_data; }

  /// Whether the kernel works on a copy rather than on the values of the view.
  bool copied() const noexcept { return m_copy != nullptr; }

  /// Writes a copy back to the view it was made from.
  void finish() {
    if (m_copy != nullptr && m_writable) {
      copy(ConstRuntimeView{&m_own, m_data}, m_view);
    }
  }

  private:
  [[noreturn]] void fail(const char* why) const {
    throw std::invalid_argument(std::string("YATETO: ") + m_what + why);
  }

  void bind() {
    assert(sizeOf(m_own.datatype) == sizeof(T) && "YATETO: the kernel holds another type");
    if (m_view.data == nullptr) {
      fail(" is read, but not set");
    }
    if (m_view.layout == nullptr) {
      fail(" has values, but no layout to find them by");
    }
    if (passesThrough(m_view, m_own)) {
      m_data = static_cast<T*>(m_view.data);
      return;
    }
    if (!detail::sameShape(*m_view.layout, m_own)) {
      fail(" is of another shape than the kernel reads");
    }
    const std::size_t alignment = m_own.alignment > alignof(T) ? m_own.alignment : alignof(T);
    const std::size_t bytes = (m_own.size > 0 ? m_own.size : 1) * sizeof(T);
    m_copy = std::unique_ptr<void, detail::AlignedDelete>(
        ::operator new(bytes, std::align_val_t{alignment}), detail::AlignedDelete{alignment});
    m_data = static_cast<T*>(m_copy.get());
    copy(m_view, RuntimeView{&m_own, m_data});
  }

  RuntimeView m_view;
  const TensorDescriptor& m_own;
  bool m_writable;
  const char* m_what;
  std::unique_ptr<void, detail::AlignedDelete> m_copy{nullptr, detail::AlignedDelete{1}};
  T* m_data{nullptr};
};

} // namespace yateto

#endif // YATETO_RUNTIMEVIEW_H_
