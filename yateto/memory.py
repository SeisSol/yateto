from .ast.indices import BoundingBox, Range
import copy
import itertools
import warnings
import numpy as np
from abc import ABC, abstractmethod

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
    a = 0
    for i,e in enumerate(entry):
      a += (e-self._bbox[i].start)*self._stride[i]
    return a

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
  
  def addressString(self, indices, I = None, prefix='_'):
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

  def __eq__(self, other):
    return self._stride == other._stride and self._bbox == other._bbox and self._stride == other._stride

  def __str__(self):
    return '{}(shape: {}, bounding box: {}, stride: {})'.format(type(self).__name__, self._shape, self._bbox, self._stride)

class CSCMemoryLayout(MemoryLayout):
  def __init__(self, spp):
    super().__init__(spp.shape)
    
    if len(self._shape) != 2:
      raise ValueError('CSCMemoryLayout may only be used for matrices.')
    
    self._bbox = BoundingBox.fromSpp(spp)
    
    nonzeros = spp.nonzero()
    nonzeros = sorted(zip(nonzeros[0], nonzeros[1]), key=lambda x: (x[1], x[0]))
    
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
    return False

  def mayVectorizeDim(self, dim):
    return False

  @classmethod
  def fromSpp(cls, spp, **kwargs):
    return CSCMemoryLayout(spp)

  def __contains__(self, entry):
    return entry in self._bbox

  def isCompatible(self, spp):
    return self.fromSpp(spp) == self

  def __eq__(self, other):
    return self._bbox == other._bbox and np.array_equal(self._rowIndex, other._rowIndex) and np.array_equal(self._colPtr, other._colPtr)
