class MemoryLayout(object):
  pass

class DenseMemoryLayout(MemoryLayout):
  def __init__(self, shape, stride=None, aligned=False):
    self._shape = shape
    if stride == None:
      stride = [1]
      for i in range(len(self._shape)-1):
        stride.append(stride[i] * self._shape[i])
      self._stride = tuple(stride)
    else:
      self._stride = stride
  
  def address(self, entry):
    assert len(entry) <= len(self._shape)
    a = 0
    for i,e in enumerate(entry):
      assert e < self._shape[i]
      a += e*self._stride[i]
    return a

  def stride(self):
    return self._stride
  
  def stridei(self, dim):
    return self._stride[dim]
    
  def shape(self):
    return self._shape
  
  def shapei(self, dim):
    return self._shape[dim]
  
  def size(self):
    size = self._shape[-1]
    for s in self._stride:
      size *= s
    return size
  
  def offset(self, offset):
    assert len(offset) == len(self._stride)
    o = 0
    for i,s in enumerate(self._stride):
      o += offset[i] * s
    return o
  
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
  
  def _positions(self, indices, I):
    positions = sorted([indices.find(i) for i in I])
    return positions
  
  def _continuousIndices(self, positions):
    return all( [y-x == 1 for x,y in zip(positions[:-1], positions[1:])] )

  def _firstStride(self, indices, I):
    positions = indices.positions(I)
    assert self._continuousIndices(positions)
    return self._stride[ self._positions(indices, I)[0] ]
  
  def _subSize(self, indices, I):
    positions = indices.positions(I)
    assert self._continuousIndices(positions)
    size = 1
    for i in positions:
      size *= self._shape[i]
    return size
  
  def slice(self, indices, I, J):
    shape = (self._subSize(indices, I), self._subSize(indices, J))
    stride = (self._firstStride(indices, I), self._firstStride(indices, J))
    if indices.positions(I)[0] > indices.positions(J)[0]:
      stride = (stride[1], stride[0])
      shape = (shape[1], shape[0])
    return DenseMemoryLayout(shape, stride)
    
      
    
    
