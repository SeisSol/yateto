from .ast.indices import BoundingBox, Range
import itertools
import warnings

class MemoryLayout(object):
  def alignedStride(self):
    return False

class DenseMemoryLayout(MemoryLayout):
  ALIGNMENT_ARCH = None

  @classmethod
  def setAlignmentArch(cls, arch):
    cls.ALIGNMENT_ARCH = arch
  
  def __init__(self, shape, boundingBox=None, stride=None, alignStride=False):
    self._shape = shape

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
    
  @classmethod
  def fromSpp(cls, spp, alignStride=False):
    bbox = BoundingBox.fromSpp(spp)
    return cls(spp.shape, bbox, alignStride=alignStride)
  
  def __contains__(self, entry):
    return entry in self._bbox
  
  def permute(self, permutation):
    self._shape = tuple([self._shape[p] for p in permutation])
    
    originalBB = BoundingBox([self._range0] + self._bbox[1:]) if self._range0 else self._bbox
    self._bbox = BoundingBox([originalBB[p] for p in permutation])
    if self._range0:
      self._alignBB()

    self._computeStride()
  
  def address(self, entry):
    assert entry in self._bbox
    a = 0
    for i,e in enumerate(entry):
      a += (e-self._bbox[i].start)*self._stride[i]
    return a
  
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
    
  def shape(self):
    return self._shape
  
  def bbox(self):
    return self._bbox

  def bboxi(self, dim):
    return self._bbox[dim]

  def requiredReals(self):
    size = self._bbox[-1].size() * self._stride[-1]
    return size
  
  def addressString(self, indices, I = None):
    if I is None:
      I = set(indices)
    positions = indices.positions(I)
    a = list()
    for p in positions:
      if self._bbox[p].start != 0:
        a.append('{}*({}-{})'.format(self._stride[p], indices[p], self._bbox[p].start))
      else:
        a.append('{}*{}'.format(self._stride[p], indices[p]))
    return ' + '.join(a)
  
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
  
  def fusedSlice(self, indices, I, J):
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

  def __eq__(self, other):
    return self._stride == other._stride and self._bbox == other._bbox and self._stride == other._stride

  def __str__(self):
    return '{}(shape: {}, bounding box: {}, stride: {})'.format(type(self).__name__, self._shape, self._bbox, self._stride)
    
      
    
    
