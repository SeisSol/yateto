import re
from .ast.node import Node, IndexedTensor
from numpy import ndarray, zeros, float64
from .memory import DenseMemoryLayout
from . import aspp

class AbstractType(object):
  @classmethod
  def isValidName(cls, name):
    return re.match(cls.VALID_NAME, name) is not None
  
  def name(self):
    return self._name

class IdentifiedType(AbstractType):
  BASE_NAME = r'[a-zA-Z]\w*'
  GROUP_INDEX = r'(0|[1-9]\d*)'
  GROUP_INDICES = r'\(({0}(,{0})*)\)'.format(GROUP_INDEX)
  VALID_NAME = r'^{}({})?$'.format(BASE_NAME, GROUP_INDICES)

  def __init__(self, name, namespace=None):
    if not self.isValidName(name):
      raise ValueError('Invalid name (must match regexp {}): {}'.format(self.VALID_NAME, name))
    
    self._name = name
    self.namespace = namespace
  
  def __str__(self):
    return self._name

  @classmethod
  def getGroup(cls, name):
    gis = re.search(cls.GROUP_INDICES, name)
    if gis:
      return tuple(int(gi) for gi in re.split(',', gis.group(1)))
    return tuple()

  def group(self):
    return self.getGroup(self._name)
  
  @classmethod
  def getBaseName(cls, name):
    return re.match(cls.BASE_NAME, name).group(0)
  
  def baseName(self):
    return self.getBaseName(self._name)
  
  @classmethod
  def splitBasename(cls, base_name_with_namespace):
    name_parts = base_name_with_namespace.rsplit('::', 1)
    if len(name_parts) > 1:
      prefix = '{}::'.format(name_parts[0])
    else:
      prefix = ''
    base_name = name_parts[-1]
    return prefix, base_name
  
  def prefix(self):
    return '{}::'.format(self.namespace) if self.namespace else ''
  
  def baseNameWithNamespace(self):
    return '{}{}'.format(self.prefix(), self.baseName())

  def nameWithNamespace(self):
    return '{}{}'.format(self.prefix(), self.name())
  
  def __hash__(self):
    return hash(self._name)

class Scalar(IdentifiedType):  
  def __init__(self, name, namespace=None):
    super().__init__(name, namespace=namespace)

class Tensor(IdentifiedType):
  def __init__(self,
               name,
               shape,
               spp=None,
               memoryLayoutClass=DenseMemoryLayout,
               alignStride=False,
               namespace=None):
    super().__init__(name, namespace=namespace)
    if not isinstance(shape, tuple):
      raise ValueError('shape must be a tuple')
    
    if any(x < 1 for x in shape):
      raise ValueError('shape must not contain entries smaller than 1')
    
    if not self.isValidName(name):
      raise ValueError('Tensor name invalid (must match regexp {}): {}'.format(self.VALID_NAME, name))

    self._name = name
    self._shape = shape
    self._values = None

    if namespace is None:
      self.namespace = ''
    else:
      self.namespace = namespace

    if spp is not None:
      if isinstance(spp, dict):
        if not isinstance(next(iter(spp.values())), bool):
          self._values = spp
        npspp = zeros(shape, dtype=bool, order=aspp.general.NUMPY_DEFAULT_ORDER)
        for multiIndex, value in spp.items():
          npspp[multiIndex] = value
        self._spp = aspp.general(npspp)
      elif isinstance(spp, ndarray) or isinstance(spp, aspp.ASpp):
        if isinstance(spp, ndarray):
          if spp.dtype.kind == 'f':
            nonzeros = spp.nonzero()
            self._values = {entry: str(spp[entry]) for entry in zip(*nonzeros)}
        self._setSparsityPattern(spp)
      else:
        raise ValueError(name, 'Matrix values must be given as dictionary (e.g. {(1,2,3): 2.0} or as numpy.ndarray.')
    else:
      self._spp = aspp.dense(shape)
    self._groupSpp = self._spp
    
    self.setMemoryLayout(memoryLayoutClass, alignStride)

  def setMemoryLayout(self, memoryLayoutClass, alignStride=False):
    self._memoryLayout = memoryLayoutClass.fromSpp(self._groupSpp, alignStride=alignStride)

  def _setSparsityPattern(self, spp, setOnlyGroupSpp=False):
    if spp.shape != self._shape:
      raise ValueError(name, 'The given Matrix\'s shape must match the shape specification.')
    spp = aspp.general(spp) if not isinstance(spp, aspp.ASpp) else spp
    if setOnlyGroupSpp == False:
      self._spp = spp
    self._groupSpp = spp

  def setGroupSpp(self, spp):
    self._setSparsityPattern(spp, setOnlyGroupSpp=True)
    self.setMemoryLayout(self._memoryLayout.__class__, alignStride=self._memoryLayout.alignedStride())

  def __getitem__(self, indexNames):
    return IndexedTensor(self, indexNames)
  
  def shape(self):
    return self._shape
  
  def memoryLayout(self):
    return self._memoryLayout
  
  def spp(self, groupSpp=True):
    return self._groupSpp if groupSpp else self._spp
  
  def values(self):
    return self._values

  def values_as_ndarray(self, dtype=float64):
    A = None
    if self._values:
      A = zeros(self._shape, dtype=dtype, order=aspp.general.NUMPY_DEFAULT_ORDER)
      for multiIndex, value in self._values.items():
        A[multiIndex] = value
    return A

  def is_compute_constant(self):
    """Tells whether both values and sparsity pattern were provided.

    The condition indicates that all information about the tensor is known at compiler time. It
    implicitly tells us that the same tensor will be used many DG elements which helps us to
    decide when to generate many-to-one or one-to-many code for batched computations

    Returns:
      bool: true if a tensor contains values. Otherwise false
    """
    return True if self._values else False

  def __eq__(self, other):
    equal = self._name == other._name
    if equal:
      assert self._shape == other._shape and aspp.array_equal(self._spp, other._spp) and self._memoryLayout == other._memoryLayout
    return equal
  
  def __str__(self):
    return '{}: {}'.format(self._name, self._shape)

class Collection(object):
  def update(self, collection):
    self.__dict__.update(collection.__dict__)

  def __getitem__(self, key):
    return self.__dict__[key]
  
  def __setitem__(self, key, value):
    self.__dict__[key] = value

  def __contains__(self, key):
    return key in self.__dict__
  
  @classmethod
  def group(cls, name):
    group = Tensor.getGroup(name)
    return group if len(group) != 1 else group[0]

  def byName(self, name):
    baseName = Tensor.getBaseName(name)
    group = self.group(name)
    return self[baseName][group] if group is not tuple() else self[baseName]

  def containsName(self, name):
    if not Tensor.isValidName(name):
      raise ValueError('Invalid name: {}'.format(name))

    baseName = Tensor.getBaseName(name)
    group = self.group(name)
    return baseName in self and (group is tuple() or group in self[baseName])
