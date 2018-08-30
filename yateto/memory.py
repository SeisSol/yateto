from .ast.indices import BoundingBox, Range

class MemoryLayout(object):
  pass

class DenseMemoryLayout(MemoryLayout):
  def __init__(self, boundingBox, stride=None, aligned=False):
    self._bbox = boundingBox

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
    return cls(bbox)
  
  def __contains__(self, entry):
    return entry in self._bbox
  
  def address(self, entry):
    assert entry in self._bbox
    a = 0
    for i,e in enumerate(entry):
      a += (e-self._bbox[i].start)*self._stride[i]
    return a

  def stride(self):
    return self._stride
  
  def stridei(self, dim):
    return self._stride[dim]
    
  def shape(self):
    return tuple([b.size() for b in self._bbox])
  
  #~ def shapei(self, dim):
    #~ return self._shape[dim]
  #~ 
  def requiredReals(self):
    size = self._bbox[-1].size() * self._stride[-1]
    return size
  
  def offset(self, offset):
    return self.address(offset)
  
  def addressString(self, indices, I = None, offset = None):
    if I is None:
      I = set(indices)
    positions = indices.positions(I)
    a = list()
    if offset:
      o = self.offset(offset)
      if o > 0:
        a.append(o)
    for p in positions:
      a.append('{}*{}'.format(self._stride[p], indices[p]))
    return ' + '.join(a)
  
  def _continuousIndices(self, positions):
    return all( [self._stride[y] == self._bbox[x].size() * self._stride[x] for x,y in zip(positions[:-1], positions[1:])] )

  def _firstStride(self, positions):
    return self._stride[ positions[0] ]
  
  def _subSize(self, positions):
    sub = 1
    for p in positions:
      sub *= self._bbox[p].size()
    return sub
  
  def fusedSlice(self, indices, I, J):
    positionsI = indices.positions(I)
    positionsJ = indices.positions(J)
    assert self._continuousIndices( indices.positions(I) ) and self._continuousIndices( indices.positions(J) )
    
    # TODO: Check slices for non-zero start and stop before stride?
    #~ assert all([r.start == 0 for r in self._bbox])

    stride = (self._firstStride(positionsI), self._firstStride(positionsJ))
    shape = (self._subSize(positionsI), self._subSize(positionsJ))
    if positionsI[0] > positionsJ[0]:
      stride = (stride[1], stride[0])
      shape = (shape[1], shape[0])
    return DenseMemoryLayout(BoundingBox([Range(0, shape[0]), Range(0, shape[1])]), stride)
    
      
    
    
