from .ast.indices import BoundingBox, Range, Indices
import copy
import itertools
import warnings
import numpy as np
from abc import ABC, abstractmethod

from . import aspp
import sys

class MemoryLayout(ABC):
  def __init__(self, shape):
    self._shape = shape

  def shape(self):
    return self._shape

  @abstractmethod
  def address(self, entry):
    pass

  @abstractmethod
  def subtensorOffset(self, topLeftEntry):
    pass

  @abstractmethod
  def alignedStride(self):
    return False

  @abstractmethod
  def mayVectorizeDim(self, dim):
    pass

  def mayFuse(self, positions):
    return len(positions) == 1

  @classmethod
  @abstractmethod
  def fromSpp(cls, spp, **kwargs):
    pass

  @abstractmethod
  def __contains__(self, entry):
    pass

  @abstractmethod
  def __eq__(self, other):
    pass

  @abstractmethod
  def isCompatible(self, spp):
    pass

  def pack(self, values, fill=0.0):
    """Materialises ``values`` into the flat storage this layout describes.

    ``values`` maps multi-indices to numbers, the way ``Tensor.values()``
    hands them out. The result has ``requiredReals()`` slots; every slot the
    layout does not map an entry of ``values`` onto -- padding, alignment
    gaps, structural zeros -- is set to ``fill``.

    ``fill`` need not be a number. Callers that render the result into source
    text pass the literal they want to see in those slots, so that the choice
    of literal stays with the emitter instead of being fixed here.
    """
    memory = [fill] * self.requiredReals()
    for entry, value in values.items():
      memory[self.address(entry)] = value
    return memory

  def _subShape(self, positions):
    sub = 1
    for p in positions:
      sub *= self._shape[p]
    return sub

  def defuse(self, fusedRange, indices, I):
    positions = indices.positions(I)
    s = self._subShape(positions)
    ranges = dict()
    start = fusedRange.start
    stop = fusedRange.stop-1
    for p in reversed(positions):
      s //= self._shape[p]
      b = start // s
      B = stop // s
      ranges[ indices[p] ] = Range(b, B+1)
      start -= b*s
      stop -= B*s
    return ranges

  def notWrittenAddresses(self, writeBB):
    if writeBB == self._bbox:
      return []

    assert writeBB in self._bbox
    re = [range(r.start, r.stop) for r in self._bbox]
    we = set(itertools.product(*[range(w.start, w.stop) for w in writeBB]))
    # NOTE: iterate the read box, rather than differencing two sets. A set of
    #       tuples enumerates in hash order, which PYTHONHASHSEED varies from
    #       run to run, and these addresses end up as generated code.
    return [self.address(e) for e in itertools.product(*re)
            if e not in we and self.hasValue(e)]

  def relranges(self):
    starts = [0] * len(self._shape)
    ends = list(self._shape)
    return starts, ends

  def sparsityBlockSize(self, dim=0):
    """Largest B for which the sparsity pattern is constant within every
    B-aligned block along `dim`.

    For every block [k*B, (k+1)*B) along `dim`, and every combination of the
    remaining indices, either all entries of the block are stored or none are.
    B == 1 means arbitrary, element-wise sparsity; a dense layout has no
    restriction at all and reports maxsize.

    This is a property of the *data*, computed from the pattern the layout
    actually stores. It must not be derived from the `alignStride` request,
    which is an intent and not a guarantee.
    """
    return sys.maxsize

  def sparsityBlockShape(self):
    """Per-dimension block sizes, as a tuple.

    The all-or-nothing property is separable: if the pattern is complete in
    B_d-aligned blocks along every dimension d independently, then it is also
    complete in every tile (t_0, ..., t_n) with t_d <= B_d. So this tuple
    answers arbitrary tile queries, which a single scalar cannot.
    """
    return tuple(self.sparsityBlockSize(d) for d in range(len(self._shape)))

  def respectsTile(self, tile):
    """Is the pattern all-or-nothing within every `tile`-aligned tile?"""
    shape = self.sparsityBlockShape()
    return all(t <= b for t, b in zip(tile, shape))

class DenseMemoryLayout(MemoryLayout):
  ALIGNMENT_ARCH = None

  @classmethod
  def setAlignmentArch(cls, arch):
    cls.ALIGNMENT_ARCH = arch

  def __init__(self, shape, boundingBox=None, stride=None, alignStride=False):
    super().__init__(shape)

    if boundingBox:
      self._bbox = boundingBox
    else:
      self._bbox = BoundingBox([Range(0, s) for s in self._shape])

    self._range0 = None
    if alignStride:
      self._alignBB()

    if stride:
      self._stride = stride
    else:
      self._computeStride()

  def _computeStride(self):
    stride = [1]
    for i in range(len(self._bbox)-1):
      stride.append(stride[i] * self._bbox[i].size())
    self._stride = tuple(stride)

  def _alignBB(self):
    if len(self._bbox) == 0:
      # a tensor without axes has no column to line up
      return
    if self.ALIGNMENT_ARCH is not None:
      self._range0 = self._bbox[0]
      rnew = Range( self.ALIGNMENT_ARCH.alignedLower(self._range0.start), self.ALIGNMENT_ARCH.alignedUpper(self._range0.stop) )
      self._bbox = BoundingBox([rnew] + self._bbox[1:])
    else:
      warnings.warn('Set architecture with DenseMemoryLayout.setAlignmentArch(arch) if you want to use the align stride feature.', UserWarning)

  def alignedStride(self):
    """Whether the distance between two columns is a multiple of the alignment.

    A tensor without axes has no columns and hence no such distance. That is
    not a promise that happens to be false, it is the absence of one, and the
    answer is the same either way: nothing to rely on.
    """
    if self.ALIGNMENT_ARCH is None or len(self._bbox) == 0:
      return False
    ldOk = self._stride[0] == 1 and (len(self._stride) == 1 or self.ALIGNMENT_ARCH.checkAlignment(self._stride[1]))
    localOk = self.ALIGNMENT_ARCH.checkAlignment(self._bbox[0].stop - self._bbox[0].start)
    return ldOk and localOk

  def mayVectorizeDim(self, dim):
    if self.ALIGNMENT_ARCH is None or dim >= len(self._bbox):
      return False
    return self.ALIGNMENT_ARCH.checkAlignment(self._bbox[dim].size())

  @classmethod
  def fromSpp(cls, spp, alignStride=False, alignOffset=0):
    bbox = BoundingBox.fromSpp(spp)
    shape = tuple(spp.shape)
    if alignStride and alignOffset > 0:
      bbox = BoundingBox([Range(rng.start + alignOffset, rng.stop + alignOffset) if i == 0 else rng for i,rng in enumerate(bbox)])
      shape = tuple(alignOffset + x if i == 0 else x for i, x in enumerate(shape))
      return cls(shape, bbox, alignStride=alignStride).subslice(0, alignOffset, shape[0])
    return cls(shape, bbox, alignStride=alignStride)

  def __contains__(self, entry):
    return entry in self._bbox

  def permuted(self, permutation):
    newShape = tuple([self._shape[p] for p in permutation])

    originalBB = BoundingBox([self._range0] + self._bbox[1:]) if self._range0 else self._bbox
    newBB = BoundingBox([copy.copy(originalBB[p]) for p in permutation])
    return DenseMemoryLayout(newShape, newBB, alignStride=self._range0 is not None)

  def address(self, entry):
    assert entry in self._bbox
    return sum((e - self._bbox[i].start) * self._stride[i] for i, e in enumerate(entry))

  def subtensorOffset(self, topLeftEntry):
    return self.address(topLeftEntry)

  def notWrittenAddresses(self, writeBB):
    if writeBB == self._bbox:
      return []

    assert writeBB in self._bbox
    re = [range(r.start, r.stop) for r in self._bbox]
    we = set(itertools.product(*[range(w.start, w.stop) for w in writeBB]))
    return [self.address(e) for e in itertools.product(*re) if e not in we]

  def stride(self):
    return self._stride

  def stridei(self, dim):
    return self._stride[dim]

  def bbox(self):
    return self._bbox

  def bboxi(self, dim):
    return self._bbox[dim]

  def requiredReals(self):
    if len(self._bbox) == 0:
      return 1
    size = self._bbox[-1].size() * self._stride[-1]
    return size

  def addressString(self, indices, I = None, Z = None, prefix='_', offsets=()):
    if len(self._bbox) == 0:
      return '0'
    if len(offsets) == 0:
      offsets = [0] * len(self._bbox)
    if I is None:
      I = set(indices)
    positions = indices.positions(I)
    a = list()
    for p in positions:
      offset = offsets[p] - self._bbox[p].start
      if offset < 0:
        a.append('{}*({}{}-{})'.format(self._stride[p], prefix, indices[p], -offset))
      elif offset > 0:
        a.append('{}*({}{}+{})'.format(self._stride[p], prefix, indices[p], offset))
      else:
        a.append('{}*{}{}'.format(self._stride[p], prefix, indices[p]))
    return ' + '.join(a)

  def isAlignedAddressString(self, indices, I = None, Z = None):
    if I is None:
      I = set(indices)
    positions = indices.positions(I)
    for p in positions:
      if self.ALIGNMENT_ARCH.checkAlignment(self._stride[p]) == False:
        return False
    return True

  def mayFuse(self, positions):
    return all( [self._stride[j] == self._shape[i]*self._stride[i] for i,j in zip(positions[:-1], positions[1:])] )

  def _subRange(self, positions):
    start = 0
    stop = 0
    s = 1
    for p in positions:
      start += s * self._bbox[p].start
      stop += s * (self._bbox[p].stop-1)
      s *= self._shape[p]
    return Range(start, stop+1)

  def _firstStride(self, positions):
    return self._stride[ positions[0] ]

  def vec(self, indices, I, Z):
    positionsI = indices.positions(I)
    assert self.mayFuse( indices.positions(I) )

    shape = (self._subShape(positionsI),)
    bbox = BoundingBox([self._subRange(positionsI)])
    stride = (self._firstStride(positionsI),)

    return DenseMemoryLayout(shape, bbox, stride)

  def withDummyDimension(self, front=False):
    if front:
      # A 1 x N matrix. The dummy row index never varies, so its stride is
      # never evaluated; 1 keeps the layout well-formed.
      shape = (1,) + self._shape
      bbox = BoundingBox([Range(0,1)] + list(self._bbox))
      stride = (1,) + self._stride
    else:
      shape = self._shape + (1,)
      bbox = BoundingBox(list(self._bbox) + [Range(0,1)])
      stride = self._stride + (self._bbox[-1].size() * self._stride[-1],)
    return DenseMemoryLayout(shape, bbox, stride)

  def unfold(self, indices, I, J, Z):
    positionsI = indices.positions(I)
    positionsJ = indices.positions(J)
    assert self.mayFuse( indices.positions(I) ) and self.mayFuse( indices.positions(J) )

    if positionsI[0] > positionsJ[0]:
      positionsJ, positionsI = positionsI, positionsJ

    shape = (self._subShape(positionsI), self._subShape(positionsJ))
    bbox = BoundingBox([self._subRange(positionsI), self._subRange(positionsJ)])
    stride = (self._firstStride(positionsI), self._firstStride(positionsJ))

    return DenseMemoryLayout(shape, bbox, stride)

  def isCompatible(self, spp):
    return BoundingBox.fromSpp(spp) in self.bbox()

  def subslice(self, index, start, end):
    return MemoryLayoutView(self, index, start, end)

  def __eq__(self, other):
    return self._stride == other._stride and self._bbox == other._bbox and self._stride == other._stride

  def __str__(self):
    return '{}(shape: {}, bounding box: {}, stride: {})'.format(type(self).__name__, self._shape, self._bbox, self._stride)

  def isSparse(self):
    return False

  def hasValue(self, entry):
    assert entry in self._bbox
    return True

  def spp(self):
    raise NotImplementedError()

  def storage(self):
    return self

  def alignmentOffset(self, dim):
    return 0

  def equalStride(self, dim):
    return True

class CSCMemoryLayout(MemoryLayout):
  def __init__(self, spp, alignStride=False):
    super().__init__(spp.shape)

    self.aligned = alignStride
    self._spp = spp
    self._blockSize = None

    if len(self._shape) != 2:
      raise ValueError('CSCMemoryLayout may only be used for matrices.')

    self._bbox = BoundingBox.fromSpp(spp)
    if self.aligned:
      range0 = self._bbox[0]
      rnew = Range( DenseMemoryLayout.ALIGNMENT_ARCH.alignedLower(range0.start), DenseMemoryLayout.ALIGNMENT_ARCH.alignedUpper(range0.stop) )
      self._bbox = BoundingBox([rnew] + self._bbox[1:])

    nonzeros = spp.nonzero()
    nonzeros = sorted(zip(nonzeros[0], nonzeros[1]), key=lambda x: (x[1], x[0]))

    if self.aligned:
      nonzeros_pre = set(nonzeros)
      for nonzero in nonzeros:
        lower = DenseMemoryLayout.ALIGNMENT_ARCH.alignedLower(nonzero[0])
        # no alignedUpper call here: avoid reduction to a single element when on alignment boundaries
        # clamp against the *aligned* bounding box, not against the logical shape:
        # `self._bbox[0]` was rounded up to the next alignment boundary above, and every
        # consumer of an aligned layout relies on each aligned block being either full or
        # empty. Clamping to `self._shape[0]` truncates the last block whenever the row
        # count is not a multiple of the SIMD width.
        upper = min(lower + DenseMemoryLayout.ALIGNMENT_ARCH.alignedReals, self._bbox[0].stop)

        for i in range(lower, upper):
          nonzeros_pre.add((np.int64(i), nonzero[1]))

      nonzeros = list(nonzeros_pre)
      nonzeros = sorted(zip([nonzero[0] for nonzero in nonzeros], [nonzero[1] for nonzero in nonzeros]), key=lambda x: (x[1], x[0]))

    self._rowIndex = np.ndarray(shape=(len(nonzeros),), dtype=int)
    self._colPtr = np.ndarray(shape=(self._shape[1]+1,), dtype=int)

    lastCol = 0
    self._colPtr[0] = 0
    for i,entry in enumerate(nonzeros):
      self._rowIndex[i] = entry[0]
      if entry[1] != lastCol:
        for j in range(lastCol+1, entry[1]+1):
          self._colPtr[ j ] = i
        lastCol = entry[1]
    for j in range(lastCol+1, self._shape[1]+1):
      self._colPtr[j] = len(nonzeros)

  def requiredReals(self):
    return len(self._rowIndex)

  def bbox(self):
    return self._bbox

  def bboxi(self, dim):
    return self._bbox[dim]

  def rowIndex(self):
    return self._rowIndex

  def colPointer(self):
    return self._colPtr

  def isAlignedAddressString(self, indices, I = None, Z = None):
    if I is None:
      I = set(indices)
    positions = indices.positions(I)
    return len(positions) == 0 or (positions[0] == 0 and all(p != 0 for p in positions[1:]))

  def address(self, entry):
    assert entry in self._bbox

    start = self._colPtr[ entry[1] ]
    stop = self._colPtr[ entry[1]+1 ]
    subRowInd = self._rowIndex[start:stop]

    find = np.where(subRowInd == entry[0])[0]
    assert len(find) == 1

    return start + find[0]

  def hasValue(self, entry):
    assert entry in self._bbox

    start = self._colPtr[ entry[1] ]
    stop = self._colPtr[ entry[1]+1 ]
    subRowInd = self._rowIndex[start:stop]

    find = np.where(subRowInd == entry[0])[0]
    return len(find) == 1

  def subtensorOffset(self, topLeftEntry):
    assert topLeftEntry in self._bbox
    assert topLeftEntry[0] <= self._bbox[0].start
    return self._colPtr[ topLeftEntry[1] ]

  def entries(self, rowRange, colRange):
    assert self._bbox[0].start >= rowRange.start
    e = list()
    for col in colRange:
      e.extend([(self._rowIndex[i]-rowRange.start, col-colRange.start) for i in range(self._colPtr[col], self._colPtr[col+1])])
    return e

  def entriesRel(self, *rng):
    entries = self.entries(*rng)
    return list(enumerate(entries))

  def sparsityBlockSize(self, dim=0):
    if dim != 0:
      # CSC only compresses the row dimension
      return 1
    if self._blockSize is None:
      rows = dict()
      for col in range(self._shape[1]):
        rows[col] = set(
          int(self._rowIndex[i]) for i in range(self._colPtr[col], self._colPtr[col+1]))
      extent = self._bbox[0].stop
      B = 1
      candidate = 2
      while candidate <= extent:
        if all(len(r & set(range(b, b+candidate))) in (0, candidate)
               for r in rows.values()
               for b in range(0, extent, candidate)):
          B = candidate
        else:
          break
        candidate *= 2
      self._blockSize = B
    return self._blockSize

  def alignedStride(self):
    return self.aligned

  def mayVectorizeDim(self, dim):
    return dim == 0 and self.aligned

  @classmethod
  def fromSpp(cls, spp, **kwargs):
    return CSCMemoryLayout(spp, **kwargs)

  def __contains__(self, entry):
    return entry in self._bbox

  def isCompatible(self, spp):
    comp = self.fromSpp(spp, alignStride=self.aligned)

    bboxOk = comp._bbox in self._bbox
    sppOk = set(comp.entries(comp._bbox[0], comp._bbox[1])).issubset(set(self.entries(comp._bbox[0], comp._bbox[1])))

    # TODO: also check CSC compatibility?
    # rowIndexOk = np.array_equal(self._rowIndex[:len(comp._rowIndex)], comp._rowIndex)
    # colPtrOk = np.array_equal(self._colPtr[comp._bbox[1].start:comp._bbox[1].stop], comp._colPtr[comp._bbox[1].start:comp._bbox[1].stop])

    return bboxOk and sppOk

  def __eq__(self, other):
    return self._bbox == other._bbox and np.array_equal(self._rowIndex, other._rowIndex) and np.array_equal(self._colPtr, other._colPtr)

  def subslice(self, index, start, end):
    return MemoryLayoutView(self, index, start, end)

  def spp(self):
    return self._spp

  def storage(self):
    return self

  def alignmentOffset(self, dim):
    return 0

  def isSparse(self):
    return True

  def equalStride(self, dim):
    return False


class PatternMemoryLayout(MemoryLayout):
  def __init__(self, spp, alignStride=False, pattern=None):
    super().__init__(spp.shape if spp is not None else pattern.shape)

    if spp is None:
      spp = aspp.general(pattern != 0)

    self.aligned = alignStride

    self._bbox = BoundingBox.fromSpp(spp)
    if alignStride:
      range0 = self._bbox[0]
      rnew = Range( DenseMemoryLayout.ALIGNMENT_ARCH.alignedLower(range0.start), DenseMemoryLayout.ALIGNMENT_ARCH.alignedUpper(range0.stop) )
      self._bbox = BoundingBox([rnew] + self._bbox[1:])

    nonzeros = spp.nonzero()
    nonzeros = sorted(zip(*nonzeros), key=lambda x: x[::-1])

    if alignStride:
      nonzeros_pre = set(nonzeros)
      for nonzero in nonzeros:
        lower = DenseMemoryLayout.ALIGNMENT_ARCH.alignedLower(nonzero[0])
        # no alignedUpper call here: avoid reduction to a single element when on alignment boundaries
        upper = min(lower + DenseMemoryLayout.ALIGNMENT_ARCH.alignedReals, self._bbox[0].stop)

        for i in range(lower, upper):
          nonzeros_pre.add(tuple([np.int64(i)] + list(nonzero[1:])))

      nonzeros = list(nonzeros_pre)
      nonzeros = sorted(zip(*[[nonzero[i] for nonzero in nonzeros] for i in range(len(self._shape))]), key=lambda x: x[::-1])

    # keep everything in F order
    patternShape = (max(self._shape[0], self._bbox[0].stop),) + tuple(self._shape[1:])
    self._pattern = np.zeros(patternShape, dtype=int, order='F')

    for i, nonzero in enumerate(nonzeros):
      self._pattern[tuple(nonzero)] = i + 1 if pattern is None else pattern[tuple(nonzero)]

    self._nonzeros = list(nonzeros)

  def requiredReals(self):
    return len(self._nonzeros)

  def isSparse(self):
    return True

  def bbox(self):
    return self._bbox

  def bboxi(self, dim):
    return self._bbox[dim]

  def hasValue(self, entry):
    return self._pattern[tuple(entry)] > 0

  def address(self, entry):
    assert entry in self._bbox
    assert self._pattern[tuple(entry)] > 0

    return self._pattern[tuple(entry)] - 1

  def subtensorOffset(self, topLeftEntry):
    tle = topLeftEntry
    assert topLeftEntry in self._bbox

    subpat = [self._pattern[ex] for ex in self._nonzeros if
      all(e >= tle[i] for i,e in enumerate(ex))]

    result = subpat[0] - 1 if len(subpat) > 0 else 0

    assert result >= 0

    return result

  def entries(self, *rng):
    return [tuple(e - r.start for e,r in zip(ex, rng)) for ex in self._nonzeros if
      all(e >= r.start and e < r.stop for e,r in zip(ex, rng))]

  def entriesRel(self, *rng):
    offset = self.subtensorOffset(tuple(r.start for r in rng))
    return [(self._pattern[ex] - 1 - offset, tuple(e - r.start for e,r in zip(ex, rng))) for ex in self._nonzeros if
      all(e >= r.start and e < r.stop for e,r in zip(ex, rng))]

  def alignedStride(self):
    return self.aligned

  def mayVectorizeDim(self, dim):
    return dim == 0 and self.aligned

  def pattern(self):
    return self._pattern

  @classmethod
  def fromSpp(cls, spp, **kwargs):
    return PatternMemoryLayout(spp, **kwargs)

  def __contains__(self, entry):
    return entry in self._bbox

  def isCompatible(self, spp):
    comp = self.fromSpp(spp, alignStride=self.aligned)

    bboxOk = comp._bbox in self._bbox
    sppOk = set(comp.entries(*comp._bbox)).issubset(set(self.entries(*comp._bbox)))

    return bboxOk and sppOk

  def vec(self, indices, I, Z):
    positionsI = indices.positions(I)

    # I and Z need to partition perfectly

    selector = [None for _ in range(len(self._shape))]

    for idx, z in Z.items():
      if idx in indices:
        selector[indices.find(idx)] = z
    for p in positionsI:
      selector[p] = slice(None)

    assert all(s is not None for s in selector)

    # positionsI is sorted ascending, hence the sliced array already has
    # the I-axes in the correct relative order; no transpose needed.
    pattern = self._pattern[tuple(selector)].flatten(order='F')

    return PatternMemoryLayout(None, alignStride=self.aligned, pattern=pattern)

  def withDummyDimension(self, front=False):
    pattern = np.expand_dims(self._pattern, 0 if front else -1)
    return PatternMemoryLayout(None, alignStride=self.aligned, pattern=pattern)

  def unfold(self, indices, I, J, Z):
    positionsI = indices.positions(I)
    positionsJ = indices.positions(J)
    # keep positions and values together; Z may contain indices that do
    # not occur in this tensor at all
    fixedZ = [(indices.find(idx), z) for idx, z in Z.items() if idx in indices]

    if positionsI[0] > positionsJ[0]:
      positionsJ, positionsI = positionsI, positionsJ

    positionsZ = [p for p, _ in fixedZ]
    assert sorted(positionsI + positionsJ + positionsZ) == list(range(len(self._shape)))

    # dimmap[destination] = source, which is what np.transpose expects
    dimmap = positionsI + positionsJ + positionsZ
    selector = [slice(None)] * (len(positionsI) + len(positionsJ)) + [z for _, z in fixedZ]

    sizeI = 1
    sizeJ = 1
    for p in positionsI:
      sizeI *= self._pattern.shape[p]
    for p in positionsJ:
      sizeJ *= self._pattern.shape[p]

    pattern = self._pattern.transpose(dimmap)[tuple(selector)].reshape((sizeI, sizeJ), order='F')

    return PatternMemoryLayout(None, alignStride=self.aligned, pattern=pattern)

  def addressString(self, indices, I = None, Z = None, prefix='_', offsets=()):
    # handled differently; via unrolling
    return ''

  def isAlignedAddressString(self, indices, I = None, Z = None):
    # TODO
    return self.aligned

  def mayFuse(self, positions):
    # we can always generate a new pattern
    return True

  def __eq__(self, other):
    if not isinstance(other, PatternMemoryLayout):
      return NotImplemented
    return self._bbox == other._bbox and np.array_equal(self._pattern, other._pattern)

  __hash__ = object.__hash__

  def sparsityBlockSize(self, dim=0):
    nzp = (self._pattern != 0)
    extent = nzp.shape[dim]
    moved = np.moveaxis(nzp, dim, 0).reshape(extent, -1)
    B = 1
    candidate = 2
    while candidate <= extent:
      if extent % candidate != 0:
        break
      blocks = moved.reshape(extent // candidate, candidate, -1)
      counts = blocks.sum(axis=1)
      if bool(np.all((counts == 0) | (counts == candidate))):
        B = candidate
      else:
        break
      candidate *= 2
    return B

  def equalStride(self, dim):
    # every slice along `dim` holds the same number of non-zeros as the fullest one,
    # i.e. the pattern is "rectangular" along that axis
    nzp = (self._pattern != 0)
    return bool(np.all(nzp.sum(axis=dim) == nzp.max(axis=dim) * nzp.shape[dim]))

  def alignmentOffset(self, dim):
    return 0

  def storage(self):
    return self

class AlignedCSCMemoryLayout:
  @classmethod
  def fromSpp(cls, spp, **kwargs):
    return CSCMemoryLayout(spp, alignStride=True)

class AlignedPatternMemoryLayout:
  @classmethod
  def fromSpp(cls, spp, **kwargs):
    return PatternMemoryLayout(spp, alignStride=True)

class MemoryLayoutView(MemoryLayout):
  def isSparse(self):
    return self.base.isSparse()

  def sparsityBlockSize(self, dim=0):
    return self.base.sparsityBlockSize(dim)

  def __init__(self, base, index, start, end):
    super().__init__([base._shape[i] if i != index else end - start for i in range(len(base.shape()))])
    self.base = base
    self.index = index
    self.start = start
    self.end = end

  def relidx(self, index):
    return tuple(index[i] if i != self.index else index[i] + self.start for i in range(len(self._shape)))

  def relbox(self, bbox):
    return BoundingBox([Range(max(bbox[i].start + self.start, self.start), min(bbox[i].stop + self.start, self.end)) if i == self.index else bbox[i] for i in range(len(self._shape))])

  def relspp(self, spp):
    subslice = tuple(slice(self.start, self.end) if i == self.index else slice(None) for i in range(spp.ndim))
    superarray = np.zeros(tuple(self.base.shape()), dtype=bool)
    superarray[subslice] = spp.as_ndarray()
    return aspp.general(superarray)

  def relranges(self):
    starts, ends = self.base.relranges()
    starts[self.index] = max(starts[self.index], self.start)
    ends[self.index] = min(ends[self.index], self.end)
    return starts, ends

  def __contains__(self, bbox):
    return self.base.__contains__(self.relbox(bbox))

  def __eq__(self, other):
    return self.storage() == other.storage() and self.relranges() == other.relranges()

  def address(self, entry):
    return self.base.address(self.relidx(entry))

  def subtensorOffset(self, topLeftEntry):
    return self.base.subtensorOffset(self.relidx(topLeftEntry))

  def alignedStride(self):
    return self.base.alignedStride() and (self.index != 0 or DenseMemoryLayout.ALIGNMENT_ARCH.checkAlignment(self.end - self.start))

  def fromSpp(self):
    # cannot be implemented. Call should result in error.
    raise NotImplementedError()

  def isCompatible(self, spp):
    # only a rough criterion. Can possibly be refined.
    if spp.as_ndarray().shape != tuple(self.shape()):
      return False

    return self.base.isCompatible(self.relspp(spp))

  def mayVectorizeDim(self, dim):
    return self.base.mayVectorizeDim(dim)

  def isAlignedAddressString(self, indices, I = None, Z = None):
    return self.base.isAlignedAddressString(indices, I, Z)

  def addressString(self, indices, I = None, Z = None, prefix='_', offsets=()):
    if len(offsets) == 0:
      offsets = [0] * len(self._shape)
    newOffsets = tuple(offsets[i] if self.index != i else offsets[i] + self.start for i in range(len(self._shape)))
    return self.base.addressString(indices, I, Z, prefix, newOffsets)

  def subslice(self, index, start, end):
    return MemoryLayoutView(self, index, start, end)

  def vec(self, indices, I, Z):
    positions = indices.positions(I)

    if self.index not in positions:
      return self.base.vec(indices, I, Z)

    assert positions[-1] == self.index

    shape = self.base.shape()
    scale = 1
    for p in positions[:-1]:
      scale *= shape[p]

    # the fused result is one-dimensional, so the sliced axis becomes axis 0
    return MemoryLayoutView(self.base.vec(indices, I, Z), 0, self.start * scale, self.end * scale)

  def unfold(self, indices, I, J, Z):
    positionsI = indices.positions(I)
    positionsJ = indices.positions(J)

    if self.index not in positionsI and self.index not in positionsJ:
      return self.base.unfold(indices, I, J, Z)

    newIndex = 0 if self.index in positionsI else 1
    positions = [positionsI, positionsJ][newIndex]
    assert positions[-1] == self.index

    shape = self.base.shape()
    scale = 1
    for p in positions[:-1]:
      scale *= shape[p]

    return MemoryLayoutView(self.base.unfold(indices, I, J, Z), newIndex, self.start * scale, self.end * scale)

  def withDummyDimension(self, front=False):
    index = self.index + 1 if front else self.index
    return MemoryLayoutView(self.base.withDummyDimension(front), index, self.start, self.end)

  def defuse(self, fusedRange, indices, I):
    positions = indices.positions(I)
    if self.index in positions:
      assert positions[-1] == self.index
      size = fusedRange.stop - fusedRange.start
      assert size % (self.end - self.start) == 0
      slicesize = size // (self.end - self.start)

      newFusedRange = Range(slicesize * self.start, slicesize * self.end)
      return self.base.defuse(newFusedRange, indices, I)
    else:
      return self.base.defuse(fusedRange, indices, I)

  def stride(self):
    # pass through
    return self.base.stride()

  def stridei(self, dim):
    # pass through
    return self.base.stridei(dim)

  def notWrittenAddresses(self, writeBB):
    # focus only on the subview
    outside = set(self.base.notWrittenAddresses(self.bbox()))
    return list(set(self.base.notWrittenAddresses(self.relbox(writeBB))) - outside)

  def bbox(self):
    return self.relbox(self.base.bbox())

  def storage(self):
    return self.base.storage()

  def permuted(self, permutation):
    return MemoryLayoutView(self.base.permuted(permutation), permutation[self.index], self.start, self.end)

  def _shiftedRanges(self, rng):
    return [Range(r.start + self.start, r.stop + self.start) if self.index == i else r
            for i, r in enumerate(rng)]

  def entries(self, *rng):
    return self.base.entries(*self._shiftedRanges(rng))

  def entriesRel(self, *rng):
    return self.base.entriesRel(*self._shiftedRanges(rng))

  def mayFuse(self, positions):
    return (self.index not in positions or positions[-1] == self.index) and self.base.mayFuse(positions)

  def __repr__(self):
    return f'MemoryLayoutView(index: {self.index}; range: [{self.start},{self.end}); base: {self.base})'

  def alignmentOffset(self, dim):
    val = self.base.alignmentOffset(dim)
    if self.index == dim:
      newval = val + self.start
      val = newval - DenseMemoryLayout.ALIGNMENT_ARCH.alignedLower(newval)
    return val

  def equalStride(self, dim):
    return self.base.equalStride(dim)
