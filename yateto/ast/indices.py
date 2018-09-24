import sys
import functools
import numpy as np

class Indices(object):
  def __init__(self, indexNames = '', shape = ()):
    self._indices = tuple(indexNames)
    self._size = dict()
    
    assert len(self._indices) == len(set(self._indices)), 'Repeated indices are not allowed ({}).'.format(indexNames)
    assert len(self._indices) == len(shape), 'Indices {} do not match tensor shape {}.'.format(str(self), shape)

    self._size = {self._indices[i]: size for i, size in enumerate(shape)}
  
  def tostring(self):
    return ''.join(self._indices)
  
  def extract(self, indexNames):
    return Indices(str(indexNames), self.subShape(indexNames))
  
  def firstIndex(self):
    return self.extract(self._indices[0])

  def shape(self):
    return self.subShape(self._indices)
  
  def subShape(self, indexNames):
    return tuple([self._size[index] for index in indexNames])

  def indexSize(self, index):
    return self._size[index]
  
  def permuted(self, indexNames):
    assert set(indexNames) == set(self)
    return Indices(indexNames, self.subShape(indexNames))
    
  def find(self, index):
    assert len(index) == 1
    return self._indices.index(index)
  
  def positions(self, I):
    return sorted([self.find(i) for i in I])
  
  def __eq__(self, other):
    return other != None and self._indices == other._indices and self._size == other._size
    
  def __ne__(self, other):
    return other == None or self._indices != other._indices or self._size != other._size
  
  def __hash__(self):
    return hash((self._indices, self.shape()))
  
  def __iter__(self):
    return iter(self._indices)
  
  def __getitem__(self, key):
    return self._indices[key]
    
  def __len__(self):
    return len(self._indices)
  
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

class Range(object):
  def __init__(self, start, stop):
    self.start = start
    self.stop = stop
  
  def size(self):
    return self.stop - self.start
  
  def aligned(self, arch):
    return Range(arch.alignedLower(self.start), arch.alignedUpper(self.stop))
  
  def __and__(self, other):
    return Range(max(self.start, other.start), min(self.stop, other.stop))

  def __or__(self, other):
    return Range(min(self.start, other.start), max(self.stop, other.stop))
  
  def __contains__(self, other):
    return self.start <= other.start and self.stop >= other.stop
  
  def __eq__(self, other):
    return self.start == other.start and self.stop == other.stop
  
  def __str__(self):
    return 'Range({}, {})'.format(self.start, self.stop)
  
  def __iter__(self):
    return iter(range(self.start, self.stop))
      
class BoundingBox(object):
  def __init__(self, listOfRanges):
    self._box = listOfRanges
  
  @staticmethod
  def sumAxes(spp, cache, axes):
    if len(axes) == 0:
      return spp
    head = axes[0]
    tail = axes[1:]
    if not tail in cache:
      cache[tail] = BoundingBox.sumAxes(spp, cache, tail)
    return np.sum(cache[tail], axis=head)

  @classmethod
  def fromSpp(cls, spp):
    # dense case
    if np.count_nonzero(spp) == spp.size:
      return cls([Range(0, s) for s in spp.shape])

    n = len(spp.shape)
    ranges = list()
    cache = dict()
    for axis in range(n):
      axes = tuple([a for a in range(n) if a != axis])
      reduction = cls.sumAxes(spp, cache, axes)
      m, M = np.where(reduction)[0][[0,-1]]
      ranges.append(Range(m, M+1))
    return cls(ranges)
  
  def size(self):
    s = 1
    for r in self._box:
      s *= r.size()
    return s
  
  def __contains__(self, entry):
    if len(entry) != len(self):
      return False
    if len(self) == 0:
      return True
    if isinstance(entry[0], Range):
      return all([e in self[i] for i,e in enumerate(entry)])
    return all([e >= self[i].start and e <= self[i].stop for i,e in enumerate(entry)])
  
  def __getitem__(self, key):
    return self._box[key]
  
  def __len__(self):
    return len(self._box)
    
  def __iter__(self):
    return iter(self._box)
  
  def __eq__(self, other):
    return all([s == o for s,o in zip(self,other)])
  
  def __str__(self):
    return '{}({})'.format(type(self).__name__, ', '.join([str(r) for r in self]))

@functools.total_ordering
class LoGCost(object):    
  def __init__(self, stride = sys.maxsize, leftTranspose = sys.maxsize, rightTranspose = sys.maxsize, fusedIndices = 0):
    """
    stride (w.r.t. first dimension): 0 = unit stride, 1 non-unit stride (lower is better)
    transpose: Number of required transposes                            (lower is better)
    fusedIndices: Number of tensor indices to be fused in a super-index (higher is better)
    """
    self._stride = stride
    self._leftTranspose = leftTranspose
    self._rightTranspose = rightTranspose
    self._fusedIndices = fusedIndices
  
  @staticmethod
  def addIdentity():
    return LoGCost(0, 0, 0, 0)
    
  def _totuple(self):
    # minus sign before _fusedIndices as higher is better
    return (self._stride, self._leftTranspose + self._rightTranspose, -self._fusedIndices)
  
  def __lt__(self, other):
    s = self._totuple()
    o = other._totuple()
    if s == o:
      return self._leftTranspose < other._leftTranspose
    return self._totuple() < other._totuple()

  def __eq__(self, other):
    return self._totuple() == other._totuple() and self._leftTranspose == other._leftTranspose
  
  def __add__(self, other):
    return LoGCost(self._stride + other._stride, self._leftTranspose + other._leftTranspose, self._rightTranspose + other._rightTranspose, self._fusedIndices + other._fusedIndices)
  
  def __repr__(self):
    return '{{stride: {}, left transpose: {}, right transpose: {}, fused indices: {}}}'.format(self._stride, self._leftTranspose, self._rightTranspose, self._fusedIndices)
