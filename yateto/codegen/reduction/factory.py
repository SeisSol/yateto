from ..common import *
from .generic import Generic

from ...ops import Operation, CommutativeMonoidMixin

class Description(object):
  def __init__(self, alpha, add: bool, result: IndexedTensorDescription, term: IndexedTensorDescription, optype: Operation):
    self.alpha = alpha
    self.add = add
    self.result = result
    self.term = term
    self.optype = optype

    assert isinstance(optype, CommutativeMonoidMixin), \
      f'{optype} cannot be used as a reduction: it has no neutral element.'

    rA = loopRanges(self.term, self.result.indices)
    rB = loopRanges(self.result, self.result.indices)
    assert testLoopRangesAContainedInB(rA, rB)

    self.loopRanges = rA

    sumIndices = self.term.indices - self.result.indices
    assert len(sumIndices) == 1, \
      f'A Reduction node reduces exactly one index, got {sumIndices}.'
    # keep the plain index name around; str(Indices) is the concatenation of
    # its names, which for a single index is just that name
    self.sumIndex = str(sumIndices)

    self.sumLoopRange = loopRanges(self.term, sumIndices)[self.sumIndex]


def generator(arch, descr, target):
  if target == 'cpu':
    return Generic(arch, descr)
  elif target == 'gpu':
    raise RuntimeError("Reduction operation has not been implemented for GPU-like architectures")
