import re
from .ast.node import Node, IndexedTensor
from numpy import ndarray, zeros, ones, array_equal
from .memory import DenseMemoryLayout

class AbstractType(object):
  @classmethod
  def isValidName(cls, name):
    return re.match(cls.VALID_NAME, name) is not None
  
  def name(self):
    return self._name

class Scalar(AbstractType):
  VALID_NAME = r'^[a-zA-Z]\w*$'
  
  def __init__(self, name):
    if not self.isValidName(name):
      raise ValueError('Scalar name invalid (must match regexp {}): {}'.format(self.VALID_NAME, name))

    self._name = name
  
  def __str__(self):
    return self._name

class Tensor(AbstractType):
  BASE_NAME = r'[a-zA-Z]\w*'
  GROUP_INDEX = r'\[(0|[1-9]\d*)\]'
  VALID_NAME = r'^{}({})?$'.format(BASE_NAME, GROUP_INDEX)
  NUMPY_DEFAULT_ORDER = 'F'

  def __init__(self, name, shape, spp=None, memoryLayoutClass=DenseMemoryLayout, alignStride=False):
    if not isinstance(shape, tuple):
      raise ValueError('shape must be a tuple')
    
    if any(x < 1 for x in shape):
      raise ValueError('shape must not contain entries smaller than 1')
    
    if not self.isValidName(name):
      raise ValueError('Tensor name invalid (must match regexp {}): {}'.format(self.VALID_NAME, name))

    self._name = name
    self._shape = shape
    self._values = None
    
    if spp is not None:
      if isinstance(spp, dict):
        if not isinstance(next(iter(spp.values())), bool):
          self._values = spp
        self._spp = zeros(shape, dtype=bool, order=self.NUMPY_DEFAULT_ORDER)
        for multiIndex, value in spp.items():
          self._spp[multiIndex] = value
      elif isinstance(spp, ndarray):
        self._setSparsityPattern(spp)
      else:
        raise ValueError(name, 'Matrix values must be given as dictionary (e.g. {(1,2,3): 2.0} or as numpy.ndarray.')
    else:
      self._spp = ones(shape, dtype=bool, order=self.NUMPY_DEFAULT_ORDER)
    
    self.setMemoryLayout(memoryLayoutClass, alignStride)

  def setMemoryLayout(self, memoryLayoutClass, alignStride=False):
    self._memoryLayout = memoryLayoutClass.fromSpp(self._spp, alignStride=alignStride)

  def _setSparsityPattern(self, spp):
    if spp.shape != self._shape:
      raise ValueError(name, 'The given Matrix\'s shape must match the shape specification.')
    self._spp = spp.astype(bool, order=self.NUMPY_DEFAULT_ORDER)

  def updateSparsityPattern(self, spp):
    self._setSparsityPattern(spp)
    self.setMemoryLayout(self._memoryLayout.__class__, alignStride=self._memoryLayout.alignedStride())

  def __getitem__(self, indexNames):
    return IndexedTensor(self, indexNames)
  
  def shape(self):
    return self._shape
  
  def memoryLayout(self):
    return self._memoryLayout
  
  def baseName(self):
    return re.match(self.BASE_NAME, self._name).group(0)

  def group(self):
    m = re.search(self.GROUP_INDEX, self._name)
    if m:
      return int(m.group(1))
    return None
  
  def spp(self):
    return self._spp
  
  def values(self):
    return self._values
  
  def __eq__(self, other):
    equal = self._name == other._name
    if equal:
      assert self._shape == other._shape and array_equal(self._spp, other._spp) and self._memoryLayout == other._memoryLayout
    return equal
  
  def __hash__(self):
    return hash(self._name)
  
  def __str__(self):
    return '{}: {}'.format(self._name, self._shape)

class Collection(object):
  def update(self, collection):
    self.__dict__.update(collection.__dict__)

  def __getitem__(self, key):
    return self.__dict__[key]

  def __contains__(self, key):
    return key in self.__dict__
