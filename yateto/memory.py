from .ast.indices import BoundingBox, Range
import itertools

class MemoryLayout(object):
  pass

class DenseMemoryLayout(MemoryLayout):
  def __init__(self, shape, boundingBox=None, stride=None, aligned=False):
    self._shape = shape

    if boundingBox:
      self._bbox = boundingBox
    else:
      self._bbox = BoundingBox([Range(0, s) for s in self._shape])

    if stride:
      self._stride = stride
    else:
      stride = [1]
      for i in range(len(self._bbox)-1):
        stride.append(stride[i] * self._bbox[i].size())
      self._stride = tuple(stride)

  @classmethod
  def fromSpp(cls, spp):
    bbox = BoundingBox.fromSpp(spp)
    return cls(spp.shape, bbox)
  
  def __contains__(self, entry):
    return entry in self._bbox
  
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
      a.append('{}*({}-{})'.format(self._stride[p], indices[p], self._bbox[p].start))
    return ' + '.join(a)
  
  def mayFuse(self, positions):
    return all( [self._bbox[p].size() == self._shape[p] for p in positions[:-1]] )
  
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

  def __str__(self):
    return '{}(shape: {}, bounding box: {}, stride: {})'.format(type(self).__name__, self._shape, self._bbox, self._stride)
    
      
    
    
