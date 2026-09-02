import re
from numpy import ndarray, zeros, float64
from .memory import DenseMemoryLayout
from . import aspp
from enum import Enum
import math

import numpy as np

class TypeFlavor(Enum):
  """Selects the spelling of a datatype for a given consumer."""
  DEFAULT = 0
  EIGEN = 1

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
  F128 = 9

  def __str__(self):
    return {
      Datatype.BOOL: 'bool',
      Datatype.I8: 'i8',
      Datatype.I16: 'i16',
      Datatype.I32: 'i32',
      Datatype.I64: 'i64',
      Datatype.F32: 'f32',
      Datatype.F64: 'f64',
      Datatype.F128: 'f128',
      Datatype.F16: 'f16',
      Datatype.BF16: 'bf16',
    }[self]

  def ctype(self, flavor=TypeFlavor.DEFAULT):
    if flavor == TypeFlavor.EIGEN:
      # Eigen has its own scalar wrappers for the non-standard FP formats
      eigen = {
        Datatype.F16: 'Eigen::half',
        Datatype.BF16: 'Eigen::bfloat16',
      }
      if self in eigen:
        return eigen[self]
    return {
      Datatype.BOOL: 'bool',
      Datatype.I8: 'int8_t',
      Datatype.I16: 'int16_t',
      Datatype.I32: 'int32_t',
      Datatype.I64: 'int64_t',
      Datatype.F32: 'float',
      Datatype.F64: 'double',
      Datatype.F16: 'yateto::f16_ty',
      Datatype.BF16: 'yateto::bf16_ty',
      Datatype.F128: 'yateto::f128_ty',
    }[self]

  def isFloat(self):
    return self in (Datatype.F16, Datatype.BF16, Datatype.F32, Datatype.F64, Datatype.F128)

  def isInteger(self):
    return self in (Datatype.I8, Datatype.I16, Datatype.I32, Datatype.I64)

  def isBool(self):
    return self == Datatype.BOOL

  def bits(self):
    return 1 if self == Datatype.BOOL else 8 * self.size()

  def limits(self):
    """(lowest, max) representable value; None for the FP types (use infinity there)."""
    if self == Datatype.BOOL:
      return (False, True)
    if self.isInteger():
      return (-2**(self.bits() - 1), 2**(self.bits() - 1) - 1)
    return (None, None)

  def nptype(self):
    # NOTE: np.bool was removed in numpy 1.24, np.float128 does not exist on all
    #       platforms (e.g. macOS/arm64, Windows). Hence the guarded lookups.
    return {
      Datatype.BOOL: np.bool_,
      Datatype.I8: np.int8,
      Datatype.I16: np.int16,
      Datatype.I32: np.int32,
      Datatype.I64: np.int64,
      Datatype.F32: np.float32,
      Datatype.F64: np.float64,
      Datatype.F16: np.float16,
      Datatype.BF16: np.float32, # NYI
      Datatype.F128: getattr(np, 'float128', np.longdouble),
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
      Datatype.F128: 16,
    }[self]

  def safeint(self, value):
    # allow inf/-inf to be treated as int: saturate at the type's own limits
    lo, hi = self.limits()
    if lo is None:
      lo, hi = -2**63, 2**63 - 1
    if value != value: # NaN
      return 0
    return int(max(lo, min(hi, value)))

  def literal(self, value):
    # Non-finite values have no literal spelling in C/C++; route them through
    # <limits> instead. For the integer types they saturate.
    if isinstance(value, float) and not math.isfinite(value):
      ctype = self.ctype()
      if math.isnan(value):
        if self.isFloat():
          return f'std::numeric_limits<{ctype}>::quiet_NaN()'
        return self.literal(0)
      if self.isFloat():
        sign = '-' if value < 0 else ''
        return f'{sign}std::numeric_limits<{ctype}>::infinity()'
      if self.isBool():
        return 'true' if value > 0 else 'false'
      # integers: saturate
      return f'std::numeric_limits<{ctype}>::{"max" if value > 0 else "lowest"}()'

    # (note: the extra lambda mapping is needed to prevent type errors)
    return {
      Datatype.BOOL: lambda value: 'true' if value else 'false',
      Datatype.I8: lambda value: f'static_cast<int8_t>({self.safeint(value)}LL)',
      Datatype.I16: lambda value: f'static_cast<int16_t>({self.safeint(value)}LL)',
      Datatype.I32: lambda value: f'static_cast<int32_t>({self.safeint(value)}LL)',
      Datatype.I64: lambda value: f'static_cast<int64_t>({self.safeint(value)}LL)',
      Datatype.F32: lambda value: f'{float(value):.16}f',
      Datatype.F64: lambda value: f'{float(value):.16}',
      Datatype.F16: lambda value: f'static_cast<yateto::f16_ty>({float(value):.16})',
      Datatype.BF16: lambda value: f'static_cast<yateto::bf16_ty>({float(value):.16})',
      Datatype.F128: lambda value: f'static_cast<yateto::f128_ty>({value!r}q)',
    }[self](value)

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

class Symbol(object):
  def __init__(self, datatype):
    # datatype == None is treated as datatype == arch.datatype
    self.datatype = datatype

  def getDatatype(self, arch):
    return arch.datatype if self.datatype is None else self.datatype

class AbstractType(Symbol):
  def __init__(self, name, datatype):
    super().__init__(datatype)
    self._name = name

  @classmethod
  def isValidName(cls, name):
    return re.match(cls.VALID_NAME, name) is not None

  def name(self):
    return self._name

class IdentifiedType(AbstractType):
  BASE_NAME = r'[a-zA-Z]\w*'
  GROUP_INDEX = r'(0|[1-9]\d*)'
  GROUP_INDICES = rf'\(({GROUP_INDEX}(,{GROUP_INDEX})*)\)'
  VALID_NAME = rf'^{BASE_NAME}({GROUP_INDICES})?$'

  def __init__(self, name, namespace=None, datatype=None):
    super().__init__(name, datatype)
    if not self.isValidName(name):
      raise ValueError(f'Invalid name (must match regexp {self.VALID_NAME}): {name}')

    self._name = name
    self.namespace = namespace

    self.datatype = datatype

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

class Tensor(IdentifiedType):
  def __init__(self,
               name,
               shape,
               spp=None,
               memoryLayoutClass=DenseMemoryLayout,
               alignStride=False,
               namespace=None,
               datatype=None,
               addressing=None,
               temporary=False):
    super().__init__(name, namespace=namespace, datatype=datatype)
    if not isinstance(shape, tuple):
      raise ValueError('shape must be a tuple')

    if any(x < 1 for x in shape):
      raise ValueError('shape must not contain entries smaller than 1')

    if not self.isValidName(name):
      raise ValueError(f'Tensor name invalid (must match regexp {self.VALID_NAME}): {name}')

    self._name = name
    self._shape = shape
    self._values = None

    # default addressing mode. If not given, deduce it
    self.addressing = addressing

    self.temporary = temporary

    if namespace is None:
      self.namespace = ''
    else:
      self.namespace = namespace

    if spp is not None:
      if isinstance(spp, dict):
        if not isinstance(next(iter(spp.values()), False), bool):
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

  def isPassedByValue(self):
    """Whether this tensor is handed over by value rather than by pointer."""
    return self.addressing == AddressingMode.SCALAR

  def __hash__(self):
    # only over what cannot change: the sparsity pattern and the memory layout
    # are set after construction, and hashing them would lose a tensor that is
    # already sitting in a set
    return hash((self._name, self._shape, self.addressing))

  def setMemoryLayout(self, memoryLayoutClass, alignStride=False):
    self._memoryLayout = memoryLayoutClass.fromSpp(self._groupSpp, alignStride=alignStride)

  def _setSparsityPattern(self, spp, setOnlyGroupSpp=False):
    if spp.shape != self._shape:
      raise ValueError(self._name, 'The given Matrix\'s shape must match the shape specification.')
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
    if not isinstance(other, Tensor):
      return NotImplemented
    return self._name == other._name \
       and self._shape == other._shape \
       and self.addressing == other.addressing

  def __str__(self):
    return '{}: {}'.format(self._name, self._shape)

class ScalarExpression:
  """A computation over scalars, evaluated before any kernel call.

  Its leaves are named scalars and numbers, never tensor reads, so it depends on
  nothing the kernel produces and can be hoisted to the top of the kernel.
  """

  def __init__(self, optype, *operands):
    optype.checkArity(len(operands))
    self.optype = optype
    self.operands = operands

  def scalars(self):
    """The named scalars this expression reads."""
    found = set()
    for operand in self.operands:
      if isinstance(operand, ScalarExpression):
        found |= operand.scalars()
      elif isinstance(operand, Tensor):
        found.add(operand)
    return found

  def datatype(self, arch):
    from .ops import promote
    types = []
    for operand in self.operands:
      if isinstance(operand, ScalarExpression):
        types.append(operand.datatype(arch))
      elif isinstance(operand, Tensor):
        types.append(operand.getDatatype(arch))
    return promote(types) if types else arch.datatype

  def ccode(self, arch):
    def spell(operand):
      if isinstance(operand, ScalarExpression):
        return operand.ccode(arch)
      if isinstance(operand, Tensor):
        return operand.name()
      return self.datatype(arch).literal(operand)
    return self.optype.callstr(*[spell(operand) for operand in self.operands])

  def _key(self):
    return (self.optype, tuple(o._key() if isinstance(o, ScalarExpression) else o
                               for o in self.operands))

  def __eq__(self, other):
    return isinstance(other, ScalarExpression) and self._key() == other._key()

  def __hash__(self):
    return hash(self._key())

  def __repr__(self):
    return f'{self.optype}({", ".join(repr(o) for o in self.operands)})'


class Scalar(Tensor):
  """A rank-0 tensor that is handed over by value.

  Everything else about it is a tensor: it has a shape, a memory layout, a
  sparsity pattern, and it is indexed with the empty index. Only the calling
  convention differs, and that is what AddressingMode.SCALAR says.
  """

  def __init__(self, name, namespace=None, datatype=None):
    super().__init__(name, (), namespace=namespace, datatype=datatype,
                     addressing=AddressingMode.SCALAR)

  def __str__(self):
    # a scalar spells itself as its name: it is a value in the generated code,
    # and its shape carries nothing worth printing
    return self.name()

  # Arithmetic between scalars builds a ScalarExpression. Against a tensor node
  # NotImplemented lets Python fall back to the node's reflected operator, which
  # knows how to turn this into a scale factor.
  @staticmethod
  def _combine(optype, left, right):
    from .ast.node import Node
    if isinstance(left, Node) or isinstance(right, Node):
      return NotImplemented
    return derivedScalar(optype, left, right)

  def __mul__(self, other):
    from . import ops
    return self._combine(ops.Mul(), self, other)
  __rmul__ = __mul__

  def __add__(self, other):
    from . import ops
    return self._combine(ops.Add(), self, other)
  __radd__ = __add__

  def __sub__(self, other):
    from . import ops
    negated = self._combine(ops.Mul(), -1.0, other)
    return negated if negated is NotImplemented else self._combine(ops.Add(), self, negated)

  def __rsub__(self, other):
    from . import ops
    return self._combine(ops.Add(), other, -self)

  def __truediv__(self, other):
    from . import ops
    return self._combine(ops.Div(), self, other)

  def __rtruediv__(self, other):
    from . import ops
    return self._combine(ops.Div(), other, self)

  def __neg__(self):
    from . import ops
    return derivedScalar(ops.Mul(), -1.0, self)


class DerivedScalar(Scalar):
  """A scalar computed from other scalars before the kernel runs.

  Temporary, so it is not part of the kernel signature: the caller does not set
  it, the kernel computes it in its prologue.
  """

  # internal name; a user tensor cannot start with an underscore, so these
  # cannot collide with one
  BASE_NAME = r'_s\d+'
  VALID_NAME = r'^_s\d+$'

  _counter = 0

  def __init__(self, expression):
    DerivedScalar._counter += 1
    super().__init__(f'_s{DerivedScalar._counter}')
    self.temporary = True
    self.expression = expression

  def getDatatype(self, arch):
    return self.expression.datatype(arch) if self.datatype is None else self.datatype

  def dependencies(self):
    return self.expression.scalars()


def derivedScalar(optype, *operands):
  """Combine scalars into one derived scalar.

  A derived operand contributes its expression rather than itself, so every
  derived scalar reads only named scalars and numbers -- there is never a chain
  of them to order in the prologue.
  """
  flattened = [o.expression if isinstance(o, DerivedScalar) else o for o in operands]
  expression = ScalarExpression(optype, *flattened)
  # the same expression yields the same scalar, so it is computed once
  return _derivedCache.setdefault(expression, DerivedScalar(expression))

_derivedCache = dict()

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
