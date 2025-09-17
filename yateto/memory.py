from .ast.indices import BoundingBox, Range, Indices
import copy
import itertools
import warnings
import numpy as np
from abc import ABC, abstractmethod

from . import aspp

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
    if self.ALIGNMENT_ARCH is not None:
      self._range0 = self._bbox[0]
      rnew = Range( self.ALIGNMENT_ARCH.alignedLower(self._range0.start), self.ALIGNMENT_ARCH.alignedUpper(self._range0.stop) )
      self._bbox = BoundingBox([rnew] + self._bbox[1:])
    else:
      warnings.warn('Set architecture with DenseMemoryLayout.setAlignmentArch(arch) if you want to use the align stride feature.', UserWarning)
  
  def alignedStride(self):
    if self.ALIGNMENT_ARCH is None:
      return False
    ldOk = self._stride[0] == 1 and (len(self._stride) == 1 or self.ALIGNMENT_ARCH.checkAlignment(self._stride[1]))
    localOk = self.ALIGNMENT_ARCH.checkAlignment(self._bbox[0].stop - self._bbox[0].start)
    return ldOk and localOk

  def mayVectorizeDim(self, dim):
    if self.ALIGNMENT_ARCH is None:
      return False
    return self.ALIGNMENT_ARCH.checkAlignment(self._bbox[dim].size())

  @classmethod
  def fromSpp(cls, spp, alignStride=False):
    bbox = BoundingBox.fromSpp(spp)
    return cls(spp.shape, bbox, alignStride=alignStride)

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
    we = [range(w.start, w.stop) for w in writeBB]
    return [self.address(e) for e in set(itertools.product(*re)) - set(itertools.product(*we))]

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
  
  def addressString(self, indices, I = None, prefix='_', offsets=()):
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

  def isAlignedAddressString(self, indices, I = None):
    if I is None:
      I = set(indices)
    positions = indices.positions(I)
    for p in positions:
      if self.ALIGNMENT_ARCH.checkAlignment(self._stride[p]) == False:
        return False
    return True

  def mayFuse(self, positions):
    return all( [self._stride[j] == self._shape[i]*self._stride[i] for i,j in zip(positions[:-1], positions[1:])] )
  
  def _subShape(self, positions):
    sub = 1
    for p in positions:
      sub *= self._shape[p]
    return sub
  
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

  def vec(self, indices, I):
    positionsI = indices.positions(I)
    assert self.mayFuse( indices.positions(I) )

    shape = (self._subShape(positionsI),)
    bbox = BoundingBox([self._subRange(positionsI)])
    stride = (self._firstStride(positionsI),)

    return DenseMemoryLayout(shape, bbox, stride)

  def withDummyDimension(self):
    shape = self._shape + (1,)
    bbox = BoundingBox(list(self._bbox) + [Range(0,1)])
    stride = self._stride + (self._bbox[-1].size() * self._stride[-1],)
    return DenseMemoryLayout(shape, bbox, stride)

  def unfold(self, indices, I, J):
    positionsI = indices.positions(I)
    positionsJ = indices.positions(J)
    assert self.mayFuse( indices.positions(I) ) and self.mayFuse( indices.positions(J) )

    if positionsI[0] > positionsJ[0]:
      positionsJ, positionsI = positionsI, positionsJ

    shape = (self._subShape(positionsI), self._subShape(positionsJ))
    bbox = BoundingBox([self._subRange(positionsI), self._subRange(positionsJ)])
    stride = (self._firstStride(positionsI), self._firstStride(positionsJ))

    return DenseMemoryLayout(shape, bbox, stride)
  
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

  def isCompatible(self, spp):
    return BoundingBox.fromSpp(spp) in self.bbox()
  
  def subslice(self, index, start, end):
    return MemoryLayoutView(self, index, start, end)

  def __eq__(self, other):
    return self._stride == other._stride and self._bbox == other._bbox and self._stride == other._stride

  def __str__(self):
    return '{}(shape: {}, bounding box: {}, stride: {})'.format(type(self).__name__, self._shape, self._bbox, self._stride)
  
  def isCSC(self):
    return False
  
  def spp(self):
    raise NotImplementedError()
    #subslice = tuple(slice(d.start, d.end) if d in self._bbox)
    #superarray = np.zeros(tuple(self._stride), dtype=bool)
    #superarray[subslice] = spp.as_ndarray()
    #return aspp.general(superarray)
  
  def storage(self):
    return self

class CSCMemoryLayout(MemoryLayout):
  def isCSC(self):
    return True

  def __init__(self, spp, alignStride=False):
    super().__init__(spp.shape)

    self.aligned = alignStride
    self._spp = spp
    
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
        upper = lower + DenseMemoryLayout.ALIGNMENT_ARCH.alignedReals

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
  
  def isAlignedAddressString(self, indices, I = None):
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

class AlignedCSCMemoryLayout:
  @classmethod
  def fromSpp(cls, spp, **kwargs):
    return CSCMemoryLayout(spp, alignStride=True)

class MemoryLayoutView(MemoryLayout):
  def isCSC(self):
    return self.base.isCSC()

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

  def __contains__(self, bbox):
    return self.base.__contains__(self.relbox(bbox))
  
  def __eq__(self, other):
    # return np.array_equal(self.spp(), other.spp())
    # TODO: wrong. Check np.array_equal(self.spp(), other.spp()) instead. Once implemented.
    if isinstance(other, MemoryLayoutView):
      return self.base == other.base and self.index == other.index and self.start == other.start and self.end == other.end
    else:
      return self.base == other
  
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
    return self.base.isCompatible(self.relspp(spp))

  def mayVectorizeDim(self, dim):
    return self.base.mayVectorizeDim(dim)
  
  def isAlignedAddressString(self, indices, I = None):
    return self.base.isAlignedAddressString(indices, I)
  
  def addressString(self, indices, I = None, prefix='_', offsets=()):
    if len(offsets) == 0:
      offsets = [0] * len(self._shape)
    newOffsets = tuple(offsets[i] if self.index != i else offsets[i] + self.start for i in range(len(self._shape)))
    return self.base.addressString(indices, I, prefix, newOffsets)
  
  def subslice(self, index, start, end):
    return MemoryLayoutView(self, index, start, end)
  
  def unfold(self, indices, I, J):
    positionsI = indices.positions(I)
    positionsJ = indices.positions(J)

    if self.index not in positionsI and self.index not in positionsJ:
      return self.base.unfold(indices, I, J)

    newIndex = 0 if self.index in positionsI else 1
    positions = [positionsI, positionsJ][newIndex]
    assert positions[-1] == self.index

    shape = self.base.shape()
    scale = 1
    for p in positions[:-1]:
      scale *= shape[p]

    return MemoryLayoutView(self.base.unfold(indices, I, J), newIndex, self.start * scale, self.end * scale)
  
  def withDummyDimension(self):
    return MemoryLayoutView(self.base.withDummyDimension(), self.index, self.start, self.end)

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
  
  def entries(self, rowRange, colRange):
    if self.index == 0:
      return self.base.entries(Range(rowRange.start + self.start, rowRange.stop + self.start), colRange)
    elif self.index == 1:
      return self.base.entries(rowRange, Range(colRange.start + self.start, colRange.stop + self.start))
    else:
      raise NotImplementedError()
  
  def mayFuse(self, positions):
    return (self.index not in positions or positions[-1] == self.index) and self.base.mayFuse(positions)

  def __repr__(self):
    return f'MemoryLayoutView(index: {self.index}; range: [{self.start},{self.end}); base: {self.base})'
