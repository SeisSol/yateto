#ifndef YATETO_DESCRIPTOR_H_
#define YATETO_DESCRIPTOR_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace yateto {

/// The element type a tensor is stored in.
enum class Datatype : std::uint8_t { Bool, I8, I16, I32, I64, F32, F64, F16, BF16, F128 };

/// Size of one element of `datatype`, in bytes.
constexpr std::size_t sizeOf(Datatype datatype) noexcept {
  switch (datatype) {
  case Datatype::Bool:
  case Datatype::I8:
    return 1;
  case Datatype::I16:
  case Datatype::F16:
  case Datatype::BF16:
    return 2;
  case Datatype::I32:
  case Datatype::F32:
    return 4;
  case Datatype::I64:
  case Datatype::F64:
    return 8;
  case Datatype::F128:
    return 16;
  }
  return 0;
}

/// How the stored values of a tensor are arranged.
enum class Storage : std::uint8_t {
  /// Every entry of a box, at the address its strides give.
  Dense,
  /// A matrix by columns: the row of every stored value, and where each column begins.
  CSC,
  /// Any set of entries: for every entry of a box, its position among the values plus one, or 0.
  Pattern
};

/// Where the values of one tensor are, as data.
///
/// The numbers `init::X::view::create` spells into a view, readable at run time: code that only
/// learns at run time in which layout it holds a tensor can still find every entry of it. All
/// arrays run over the dimensions, the first one fastest.
struct TensorDescriptor {
  Datatype datatype;
  Storage storage;
  unsigned rank;
  /// The logical extent of each dimension; null for a tensor without dimensions.
  const unsigned* shape;
  /// Dense: the box of entries that is stored, [start, stop) per dimension. Pattern: the box the
  /// pattern covers. Either may reach past the shape, into padding.
  const unsigned* start;
  const unsigned* stop;
  /// Dense: the distance between neighbouring values, per dimension. Pattern: the same for the
  /// entries of the pattern.
  const unsigned* stride;
  /// CSC: the row of every value, and where each column begins (one more than there are columns).
  const unsigned* rowIndex;
  const unsigned* columnPointer;
  /// Pattern: per entry of the box, its position among the values plus one, or 0 where there is
  /// no value.
  const unsigned* pattern;
  /// Number of elements the values take, padding included.
  unsigned size;
  /// Alignment, in bytes, the address of the values has to have for the kernels to read them.
  unsigned alignment;
};

namespace detail {
constexpr bool sameArray(const unsigned* a, const unsigned* b, std::size_t count) noexcept {
  if (a == b) {
    return true;
  }
  if (a == nullptr || b == nullptr) {
    return false;
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

constexpr std::size_t product(const unsigned* extent, unsigned rank) noexcept {
  std::size_t result = 1;
  for (unsigned d = 0; d < rank; ++d) {
    result *= extent[d];
  }
  return result;
}
} // namespace detail

/// Whether two descriptors put every value at the same place, in the same element type.
///
/// The alignment is not part of it: it is what the address of the values has to satisfy, not
/// where a value is relative to it.
constexpr bool sameLayout(const TensorDescriptor& a, const TensorDescriptor& b) noexcept {
  if (&a == &b) {
    return true;
  }
  if (a.datatype != b.datatype || a.storage != b.storage || a.rank != b.rank || a.size != b.size ||
      !detail::sameArray(a.shape, b.shape, a.rank)) {
    return false;
  }
  switch (a.storage) {
  case Storage::Dense:
    return detail::sameArray(a.start, b.start, a.rank) &&
           detail::sameArray(a.stop, b.stop, a.rank) &&
           detail::sameArray(a.stride, b.stride, a.rank);
  case Storage::CSC:
    return detail::sameArray(a.columnPointer, b.columnPointer, a.shape[1] + 1) &&
           detail::sameArray(a.rowIndex, b.rowIndex, a.size);
  case Storage::Pattern:
    return detail::sameArray(a.start, b.start, a.rank) &&
           detail::sameArray(a.stop, b.stop, a.rank) &&
           detail::sameArray(a.stride, b.stride, a.rank) &&
           detail::sameArray(a.pattern, b.pattern, detail::product(a.stop, a.rank));
  }
  return false;
}

/// Offset of the value of an entry among the stored values, or -1 where the entry is not stored.
///
/// `index` has one component per dimension. An entry outside the shape but inside a padded box
/// is found as well: padding is stored like anything else, it only holds nothing that counts.
constexpr std::ptrdiff_t offsetOf(const TensorDescriptor& descriptor,
                                  const unsigned* index) noexcept {
  switch (descriptor.storage) {
  case Storage::Dense: {
    std::ptrdiff_t offset = 0;
    for (unsigned d = 0; d < descriptor.rank; ++d) {
      if (index[d] < descriptor.start[d] || index[d] >= descriptor.stop[d]) {
        return -1;
      }
      offset += static_cast<std::ptrdiff_t>(index[d] - descriptor.start[d]) * descriptor.stride[d];
    }
    return offset;
  }
  case Storage::CSC: {
    if (index[1] >= descriptor.shape[1]) {
      return -1;
    }
    for (unsigned i = descriptor.columnPointer[index[1]];
         i < descriptor.columnPointer[index[1] + 1];
         ++i) {
      if (descriptor.rowIndex[i] == index[0]) {
        return i;
      }
    }
    return -1;
  }
  case Storage::Pattern: {
    std::size_t entry = 0;
    for (unsigned d = 0; d < descriptor.rank; ++d) {
      if (index[d] < descriptor.start[d] || index[d] >= descriptor.stop[d]) {
        return -1;
      }
      entry += static_cast<std::size_t>(index[d] - descriptor.start[d]) * descriptor.stride[d];
    }
    return static_cast<std::ptrdiff_t>(descriptor.pattern[entry]) - 1;
  }
  }
  return -1;
}

/// One tensor, or one family of them, of a generated namespace.
struct TensorEntry {
  /// With the namespace it was declared in, as in "nodal::V".
  std::string_view name;
  /// Number of indices of a member of the family; 0 for a tensor that is not part of one.
  unsigned groupRank;
  /// Extent of each index of the family.
  const unsigned* groupSize;
  /// Per member, in the order `tensor::X::index` gives; null where the family has no such member.
  const TensorDescriptor* const* members;

  /// The member at `group`, or null where there is none. A tensor that is not part of a family
  /// is its own member at the empty index.
  constexpr const TensorDescriptor* member(std::initializer_list<unsigned> group) const noexcept {
    if (group.size() != groupRank) {
      return nullptr;
    }
    std::size_t index = 0;
    std::size_t stride = 1;
    unsigned d = 0;
    for (const unsigned component : group) {
      if (component >= groupSize[d]) {
        return nullptr;
      }
      index += component * stride;
      stride *= groupSize[d];
      ++d;
    }
    return members[index];
  }
};

/// Every tensor of a generated namespace, ordered by name.
struct TensorTable {
  const TensorEntry* entries;
  std::size_t count;

  /// The tensor or family named `name`, or null where there is none.
  constexpr const TensorEntry* find(std::string_view name) const noexcept {
    std::size_t lower = 0;
    std::size_t upper = count;
    while (lower < upper) {
      const std::size_t middle = lower + (upper - lower) / 2;
      if (entries[middle].name < name) {
        lower = middle + 1;
      } else {
        upper = middle;
      }
    }
    return lower < count && entries[lower].name == name ? &entries[lower] : nullptr;
  }

  /// The layout of the tensor `name` at `group`, or null where there is no such tensor.
  constexpr const TensorDescriptor* find(std::string_view name,
                                         std::initializer_list<unsigned> group) const noexcept {
    const TensorEntry* entry = find(name);
    return entry == nullptr ? nullptr : entry->member(group);
  }
};

} // namespace yateto

#endif // YATETO_DESCRIPTOR_H_
