from .ast.node import IndexedTensor, Indices
from numpy import ndarray, zeros, ones
from .memory import DenseMemoryLayout

class Collection(object):
  pass

class Tensor(object):
  def __init__(self, name, shape, spp=None, memoryLayout=None):
    if not isinstance(shape, tuple):
      raise ValueError('shape must be a tuple')
    
    if any(x < 1 for x in shape):
      raise ValueError('shape must not contain entries smaller than 1')

    self._name = name
    self._shape = shape
    
    if spp is not None:
      if isinstance(spp, dict):
        self._spp = zeros(shape, dtype=bool)
        for multiIndex, value in spp.items():
          self._spp[multiIndex] = value
      elif isinstance(spp, ndarray):
        if spp.shape != self._shape:
          raise ValueError(name, 'The given Matrix\'s shape must match the shape specification.')
        self._spp = spp.astype(bool)
      else:
        raise ValueError(name, 'Matrix values must be given as dictionary (e.g. {(1,2,3): 2.0} or as numpy.ndarray.')
    else:
      self._spp = ones(shape, dtype=bool)
    
    self._memoryLayout = memoryLayout if memoryLayout else DenseMemoryLayout(self._shape)
    

  def __getitem__(self, indexNames):
    return IndexedTensor(self, indexNames)
  
  def shape(self):
    return self._shape
  
  def memoryLayout(self):
    return self._memoryLayout
  
  def name(self):
    return self._name
  
  def spp(self):
    return self._spp
