import re
from numpy import ndarray, zeros, float64
from .memory import DenseMemoryLayout
from . import aspp
from enum import Enum

import numpy as np

class Datatype(Enum):
  BOOL = 0
  I8 = 1
  I16 = 2
  I32 = 3
  I64 = 4
  F32 = 5
  F64 = 6
  F16 = 7
  BF16 = 8

  def __str__(self):
    return {
      Datatype.BOOL: 'bool',
      Datatype.I8: 'i8',
      Datatype.I16: 'i16',
      Datatype.I32: 'i32',
      Datatype.I64: 'i64',
      Datatype.F32: 'f32',
      Datatype.F64: 'f64',
      Datatype.F16: 'f16',
      Datatype.BF16: 'bf16',
    }[self]

  def ctype(self):
    return {
      Datatype.BOOL: 'bool',
      Datatype.I8: 'int8_t',
      Datatype.I16: 'int16_t',
      Datatype.I32: 'int32_t',
      Datatype.I64: 'int64_t',
      Datatype.F32: 'float',
      Datatype.F64: 'double',
      Datatype.F16: 'int16_t',
      Datatype.BF16: 'int16_t',
    }[self]
  
  def nptype(self):
    return {
      Datatype.BOOL: np.bool,
      Datatype.I8: np.int8,
      Datatype.I16: np.int16,
      Datatype.I32: np.int32,
      Datatype.I64: np.int64,
      Datatype.F32: np.float32,
      Datatype.F64: np.float64,
      Datatype.F16: np.float16,
      Datatype.BF16: np.float32, # NYI
    }[self]
  
  def size(self):
    # unpacked size
    return {
      Datatype.BOOL: 1,
      Datatype.I8: 1,
      Datatype.I16: 2,
      Datatype.I32: 4,
      Datatype.I64: 8,
      Datatype.F32: 4,
      Datatype.F64: 8,
      Datatype.F16: 2,
      Datatype.BF16: 2,
    }[self]
  
  def literal(self, value):
    # TODO: BF16, F16
    return {
      Datatype.BOOL: 'true' if value else 'false',
      Datatype.I8: f'{int(value)}',
      Datatype.I16: f'{int(value)}',
      Datatype.I32: f'{int(value)}',
      Datatype.I64: f'{int(value)}LL',
      Datatype.F32: f'{float(value):.16}f',
      Datatype.F64: f'{float(value):.16}'
    }[self]

class AddressingMode(Enum):
  DIRECT = 0
  STRIDED = 1
  INDIRECT = 2
  SCALAR = 3

  def pointer_type(self):
    return {
      AddressingMode.DIRECT: '*',
      AddressingMode.STRIDED: '*',
      AddressingMode.INDIRECT: '**',
      AddressingMode.SCALAR: '',
    }[self]

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

  def __init__(self, name, namespace=None, datatype=None):
    if not self.isValidName(name):
      raise ValueError('Invalid name (must match regexp {}): {}'.format(self.VALID_NAME, name))
    
    self._name = name
    self.namespace = namespace

    # datatype == None is treated as datatype == arch.datatype
    self.datatype = datatype

  def getDatatype(self, arch):
    return arch.datatype if self.datatype is None else self.datatype

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
  def __init__(self, name, namespace=None, datatype=None):
    super().__init__(name, namespace=namespace, datatype=datatype)
  
  def __hash__(self):
    return hash(self._name)

class Tensor(IdentifiedType):
  def __init__(self,
               name,
               shape,
               spp=None,
               memoryLayoutClass=DenseMemoryLayout,
               alignStride=False,
               namespace=None,
               datatype=None,
               addressing=None):
    super().__init__(name, namespace=namespace, datatype=datatype)
    if not isinstance(shape, tuple):
      raise ValueError('shape must be a tuple')
    
    if any(x < 1 for x in shape):
      raise ValueError('shape must not contain entries smaller than 1')
    
    if not self.isValidName(name):
      raise ValueError('Tensor name invalid (must match regexp {}): {}'.format(self.VALID_NAME, name))

    self._name = name
    self._shape = shape
    self._values = None

    # default addressing mode. If not given, deduce it
    self.addressing = addressing

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

  def __hash__(self):
    return hash(self._name)

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
    from .ast.node import IndexedTensor
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
