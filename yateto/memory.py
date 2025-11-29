from .ast.indices import BoundingBox, Range
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
    we = [range(w.start, w.stop) for w in writeBB]
    return [self.address(e) for e in set(itertools.product(*re)) - set(itertools.product(*we)) if self.hasValue(e)]

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
    offsetOk = self.ALIGNMENT_ARCH.checkAlignment(self._bbox[0].start)
    ldOk = self._stride[0] == 1 and (len(self._stride) == 1 or self.ALIGNMENT_ARCH.checkAlignment(self._stride[1]))
    return offsetOk and ldOk

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
  
  def addressString(self, indices, I = None, Z = None, prefix='_'):
    if len(self._bbox) == 0:
      return '0'
    if I is None:
      I = set(indices)
    positions = indices.positions(I)
    a = list()
    for p in positions:
      if self._bbox[p].start != 0:
        a.append('{}*({}{}-{})'.format(self._stride[p], prefix, indices[p], self._bbox[p].start))
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

  def withDummyDimension(self):
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

  def __eq__(self, other):
    return self._stride == other._stride and self._bbox == other._bbox and self._stride == other._stride

  def __str__(self):
    return '{}(shape: {}, bounding box: {}, stride: {})'.format(type(self).__name__, self._shape, self._bbox, self._stride)
  
  def isSparse(self):
    return False
  
  def hasValue(self, entry):
    assert entry in self._bbox
    return True

class CSCMemoryLayout(MemoryLayout):
  def __init__(self, spp, alignStride=False):
    super().__init__(spp.shape)

    self.aligned = alignStride
    
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

  def isSparse(self):
    return True


class PatternMemoryLayout(MemoryLayout):
  def __init__(self, spp, alignStride=False, pattern=None):
    super().__init__(spp.shape if spp is not None else pattern.shape)

    if spp is None:
      spp = aspp.general(pattern != 0)

    self.aligned = alignStride

    self._bbox = BoundingBox.fromSpp(spp)
    if self.aligned:
      range0 = self._bbox[0]
      rnew = Range( DenseMemoryLayout.ALIGNMENT_ARCH.alignedLower(range0.start), DenseMemoryLayout.ALIGNMENT_ARCH.alignedUpper(range0.stop) )
      self._bbox = BoundingBox([rnew] + self._bbox[1:])
    
    nonzeros = spp.nonzero()
    nonzeros = sorted(zip(*nonzeros), key=lambda x: x[::-1])

    if self.aligned:
      nonzeros_pre = set(nonzeros)
      for nonzero in nonzeros:
        lower = DenseMemoryLayout.ALIGNMENT_ARCH.alignedLower(nonzero[0])
        # no alignedUpper call here: avoid reduction to a single element when on alignment boundaries
        upper = lower + DenseMemoryLayout.ALIGNMENT_ARCH.alignedReals

        for i in range(lower, upper):
          nonzeros_pre.add(tuple([np.int64(i)] + list(nonzero[1:])))
      
      nonzeros = list(nonzeros_pre)
      nonzeros = sorted(zip(*[[nonzero[i] for nonzero in nonzeros] for i in range(len(self._shape))]), key=lambda x: x[::-1])
    
    self._pattern = np.zeros(self._shape, dtype=int, order='F')

    for i, nonzero in enumerate(nonzeros):
      self._pattern[tuple(nonzero)] = i + 1 if pattern is None else pattern[tuple(nonzero)]

    self._nonzeros = nonzeros

    # TODO: self._next = np.zeros(self._shape, dtype=int, order='F')
    # point to the top-left entry

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
    assert topLeftEntry in self._bbox
    
    return 0

    #assert self._next[tuple(topLeftEntry)] > 0

    #return self._next[tuple(topLeftEntry)] - 1

  def entries(self, *rng):
    return [tuple(e - r.start for e,r in zip(ex, rng)) for ex in self._nonzeros if
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
    positionsZ = indices.positions(Z)

    # I and Z need to partition perfectly

    error = lambda: None
    selector = [error for _ in range(len(self._shape))]

    for p, z in zip(positionsZ, Z.values()):
      selector[p] = z
    for p in positionsI:
      selector[p] = slice(None)
    
    pattern = self._pattern[tuple(selector)].transpose(positionsI).flatten()

    return PatternMemoryLayout(None, alignStride=self.aligned, pattern=pattern)

  def withDummyDimension(self):
    pattern = np.expand_dims(self._pattern, -1)
    return PatternMemoryLayout(None, alignStride=self.aligned, pattern=pattern)

  def unfold(self, indices, I, J, Z):
    positionsI = indices.positions(I)
    positionsJ = indices.positions(J)
    positionsZ = indices.positions(Z)

    if positionsI[0] > positionsJ[0]:
      positionsJ, positionsI = positionsI, positionsJ
    
    error = lambda: None
    selector = [error for _ in range(len(self._shape))]
    dimmap = [error for _ in range(len(self._shape))]

    i = 0
    sizeI = 1
    sizeJ = 1
    for p in positionsI:
      selector[i] = slice(None)
      dimmap[p] = i
      sizeI *= self._pattern.shape[p]
      i += 1
    for p in positionsJ:
      selector[i] = slice(None)
      dimmap[p] = i
      sizeJ *= self._pattern.shape[p]
      i += 1
    for p, z in zip(positionsZ, Z.values()):
      selector[i] = z
      dimmap[p] = i
      i += 1

    pattern = self._pattern.transpose(dimmap)[tuple(selector)].reshape((sizeI, sizeJ))

    return PatternMemoryLayout(None, alignStride=self.aligned, pattern=pattern)
  
  def addressString(self, indices, I = None, Z = None, prefix='_'):
    # handled differently; via unrolling
    return ''

  def isAlignedAddressString(self, indices, I = None, Z = None):
    # TODO
    return self.aligned
  
  def mayFuse(self, positions):
    # we can always generate a new pattern
    return True

  def __eq__(self, other):
    return self._bbox == other._bbox and np.array_equal(self._pattern, other._pattern)

class AlignedCSCMemoryLayout:
  @classmethod
  def fromSpp(cls, spp, **kwargs):
    return CSCMemoryLayout(spp, alignStride=True)

class AlignedPatternMemoryLayout:
  @classmethod
  def fromSpp(cls, spp, **kwargs):
    return PatternMemoryLayout(spp, alignStride=True)
