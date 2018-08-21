class Indices(object):
  def __init__(self, indexNames = '', shape = ()):
    self._indices = tuple(indexNames)
    self._size = dict()
    
    assert len(self._indices) == len(set(self._indices)), 'Repeated indices are not allowed ({}).'.format(indexNames)
    assert len(self._indices) == len(shape), 'Indices {} do not match tensor shape {}.'.format(str(self), shape)

    self._size = {self._indices[i]: size for i, size in enumerate(shape)}
  
  def tostring(self):
    return ''.join(self._indices)
  
  def firstIndex(self):
    return Indices(self._indices[0], self.subShape(self._indices[0]))

  def shape(self):
    return self.subShape(self._indices)
  
  def subShape(self, indexNames):
    return tuple([self._size[index] for index in indexNames])
  
  def permute(self, indexNames):
    assert set(indexNames) == set(self)
    self._indices = tuple(indexNames)
  
  def __eq__(self, other):
    return other != None and self._indices == other._indices and self._size == other._size
    
  def __ne__(self, other):
    return other == None or self._indices != other._indices or self._size != other._size
  
  def __iter__(self):
    return iter(self._indices)
  
  def __and__(self, other):
    return set(self) & set(other)
  
  def __rand__(self, other):
    return self & other
    
  def __le__(self, other):
    indexNamesContained = set(self._indices) <= set(other._indices)
    return indexNamesContained and all([self._size[index] == other._size[index] for index in self._indices])
  
  def __sub__(self, other):
    indexNames = [index for index in self._indices if index not in other]
    return Indices(indexNames, self.subShape(indexNames))

  def merged(self, other):
    indexNames = self._indices + other._indices
    shape = self.subShape(self._indices) + other.subShape(other._indices)
    return Indices(indexNames, shape)
    
  def sorted(self):
    indexNames = sorted(self._indices)
    return Indices(indexNames, self.subShape(indexNames))
  
  def __str__(self):
    return self.tostring()
    
  def __repr__(self):
    return '({})'.format(','.join(['{}={}'.format(index, self._size[index]) for index in self._indices]))
  
  def size(self):
    return self._size
