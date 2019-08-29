import numpy as np
import numpy.lib
import re
from abc import ABC, abstractmethod

class ASpp(ABC):
  def __init__(self, shape):
    self.shape = shape
    self.size = 1
    for s in shape:
      self.size *= s
    self.ndim = len(shape)

  def identity(self):
    return self

  @abstractmethod
  def count_nonzero(self):
    pass

  @abstractmethod
  def is_dense(self):
    pass

  @abstractmethod
  def nnzbounds(self):
    pass

  @abstractmethod
  def nonzero(self):
    pass

  @abstractmethod
  def copy(self):
    pass

  @abstractmethod
  def reshape(self, shape):
    pass

  @abstractmethod
  def transposed(self, shape):
    pass

  @abstractmethod
  def indexSum(self, sourceIndices, targetIndices):
    pass

  @abstractmethod
  def as_ndarray(self):
    pass

class dense(ASpp):
  def count_nonzero(self):
    return self.size

  def is_dense(self):
    return True

  def nnzbounds(self):
    return [(0, s-1) for s in self.shape]

  def nonzero(self):
    return np.ones(self.shape, dtype=bool, order=general.NUMPY_DEFAULT_ORDER).nonzero()

  def copy(self):
    return type(self)(self.shape)

  def reshape(self, shape):
    rsh = type(self)(shape)
    assert rsh.size == self.size
    return rsh

  def transposed(self, perm):
    return type(self)(tuple(self.shape[p] for p in perm))

  def indexSum(self, sourceIndices, targetIndices):
    return type(self)(tuple(self.shape[sourceIndices.find(targetIndex)] for targetIndex in targetIndices))

  @staticmethod
  def add(a1, a2):
    assert(a1.shape == a2.shape)
    return dense(a1.shape)

  @staticmethod
  def einsum(description, a1, a2):
    p = re.match('(\w*),(\w*)->(\w*)', description)
    if p:
      A = p.group(1)
      B = p.group(2)
      C = p.group(3)
      sz1 = {i: a1.shape[A.find(i)] for i in A}
      sz2 = {i: a2.shape[B.find(i)] for i in B}
      intersect = filter(lambda x: x in sz1, sz2.keys())
      assert all([sz1[i] == sz2[i] for i in intersect])    
      sz1.update(sz2)
      return dense(tuple(sz1[i] for i in C))
    else:
      raise ValueError(description + ' not understood.')

  @staticmethod
  def array_equal(a1, a2):
    return a1.shape == a2.shape

  def as_general(self):
    return general(self.as_ndarray())

  def as_ndarray(self):
    return np.ones(self.shape, dtype=bool, order=general.NUMPY_DEFAULT_ORDER)

class general(ASpp):
  NUMPY_DEFAULT_ORDER = 'F'
  OPTIMIZE_EINSUM = {'optimize': True } if np.lib.NumpyVersion(np.__version__) >= '1.12.0' else {}

  def __init__(self, npspp: np.ndarray):
    super().__init__(npspp.shape)
    if np.ndim(npspp) == 0:
      self.pattern = npspp
    else:
      self.pattern = np.asarray(npspp.astype(bool, order=self.NUMPY_DEFAULT_ORDER, copy=False))

  def count_nonzero(self):
    return np.count_nonzero(self.pattern)

  def is_dense(self):
    return self.count_nonzero() == self.size

  @classmethod
  def sumAxes(cls, spp, cache, axes):
    if len(axes) == 0:
      return spp
    head = axes[0]
    tail = axes[1:]
    if not tail in cache:
      cache[tail] = cls.sumAxes(spp, cache, tail)
    return cache[tail].sum(axis=head)

  def nnzbounds(self):
    n = len(self.shape)
    bounds = list()
    cache = dict()
    for axis in range(n):
      axes = tuple([a for a in range(n) if a != axis])
      reduction = self.sumAxes(self.pattern, cache, axes)
      nonzeros = np.where(reduction)
      assert len(nonzeros) == 1
      m, M = nonzeros[0][[0,-1]]
      bounds.append((m, M))
    return bounds

  def nonzero(self):
    return self.pattern.nonzero()

  def copy(self):
    return type(self)(self.pattern.copy())

  def reshape(self, shape):
    return type(self)(self.pattern.reshape(shape, order=self.NUMPY_DEFAULT_ORDER))

  def transposed(self, perm):
    return type(self)(self.pattern.transpose(perm).copy(order=self.NUMPY_DEFAULT_ORDER))

  def indexSum(self, sourceIndices, targetIndices):
    return general(np.einsum('{}->{}'.format(sourceIndices, targetIndices), self.pattern))

  @staticmethod
  def add(a1, a2):
    return general(np.add(a1.pattern, a2.pattern))

  @staticmethod
  def einsum(description, a1, a2):
    return general(np.einsum(description, a1.pattern, a2.pattern))

  @staticmethod
  def array_equal(a1, a2):
    return np.array_equal(a1.pattern, a2.pattern)

  def as_ndarray(self):
    return self.pattern

_binary_op = {
  (dense, dense): dense,
  (dense, general): general,
  (general, dense): general,
  (general, general): general
}

def dispatch(a1, a2):
  cls = _binary_op[(a1.__class__, a2.__class__)]
  castMethod = 'as_' + cls.__name__
  c1 = getattr(a1, castMethod, a1.identity)
  c2 = getattr(a2, castMethod, a2.identity)
  return cls, c1(), c2()

def add(a1, a2):
  cls, a1, a2 = dispatch(a1, a2)
  return cls.add(a1, a2)

def einsum(description, a1, a2):
  cls, a1, a2 = dispatch(a1, a2)
  return cls.einsum(description, a1, a2)

def array_equal(a1, a2):
  if a1 == None and a2 == None:
    return True
  if isinstance(a1, ASpp) and isinstance(a2, ASpp):
    cls, a1, a2 = dispatch(a1, a2)
    return cls.array_equal(a1, a2)
  return False
