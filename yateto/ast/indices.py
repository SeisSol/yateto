import sys
import functools

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
    
  def find(self, index):
    assert len(index) == 1
    return self._indices.index(index)
  
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

@functools.total_ordering
class LoGCost(object):    
  def __init__(self, stride = sys.maxsize, transpose = sys.maxsize, fusedIndices = 0):
    """
    stride (w.r.t. first dimension): 0 = unit stride, 1 non-unit stride (lower is better)
    transpose: Number of required transposes                            (lower is better)
    fusedIndices: Number of tensor indices to be fused in a super-index (higher is better)
    """
    self._stride = stride
    self._transpose = transpose
    self._fusedIndices = fusedIndices
  
  def __lt__(self, other):
    return self._stride < other._stride or \
           (self._stride == other._stride and self._transpose < other._transpose) or \
           (self._stride == other._stride and self._transpose == other._transpose and self._fusedIndices > other._fusedIndices)

  def __eq__(self, other):
    return self._stride == other._stride and self._transpose == other._transpose and self._fusedIndices == other._fusedIndices
  
  def __add__(self, other):
    return LoGCost(self._stride + other._stride, self._transpose + other._transpose, self._fusedIndices + other._fusedIndices)
  
  def __repr__(self):
    return '{{stride: {}, transpose: {}, fused indices: {}}}'.format(self._stride, self._transpose, self._fusedIndices)
