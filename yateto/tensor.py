from .ast.node import IndexedTensor, Indices
from numpy import ndarray, zeros, ones

class Collection(object):
  pass

class Tensor(object):
  def __init__(self, name, shape, matrix=None):
    if not isinstance(shape, tuple):
      raise ValueError('shape must be a tuple')
    
    if any(x < 1 for x in shape):
      raise ValueError('shape must not contain entries smaller than 1')

    self._name = name
    self._shape = shape
    
    if matrix != None:
      if isinstance(matrix, dict):
        self._spp = zeros(shape, dtype=bool)
        for multiIndex, value in matrix.items():
          self._spp[multiIndex] = value
      elif isinstance(matrix, ndarray):
        if matrix.shape != self._shape:
          raise ValueError(name, 'The given Matrix\'s shape must match the shape specification.')
        self._spp = matrix.astype(bool)
      else:
        raise ValueError(name, 'Matrix values must be given as dictionary (e.g. {(1,2,3): 2.0} or as numpy.ndarray.')
    else:
      self._spp = ones(shape, dtype=bool)
    

  def __getitem__(self, indexNames):
    return IndexedTensor(self, indexNames)
  
  def shape(self):
    return self._shape
  
  def name(self):
    return self._name
  
  def spp(self):
    return self._spp
