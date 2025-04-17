from ...memory import CSCMemoryLayout
from ..common import *
from .generic import Generic

from ...ops import Operation

from typing import Union

class Description(object):
  def __init__(self, alpha, add: bool, optype: Operation, result: IndexedTensorDescription, terms: list[IndexedTensorDescription], termTemplate, nodeTermIndices):
    self.alpha = alpha
    self.add = add
    self.result = result
    self.terms = terms
    self.optype = optype
    self.termTemplate = termTemplate
    self.nodeTermIndices = nodeTermIndices

    self.isSparse = [isinstance(term.memoryLayout, CSCMemoryLayout) for term in terms]

    rR = loopRanges(self.result, self.result.indices)

    # TODO: shall we allow boundingboxing?
    if len(terms) == 0:
      self.loopRanges = rR
    else:
      self.loopRanges = loopRanges(self.terms[0], self.result.indices)
      assert testLoopRangesAContainedInB(self.loopRanges, rR)
      for term in self.terms[1:]:
        newRange = loopRanges(term, self.result.indices)
        assert testLoopRangesEqual(newRange, self.loopRanges)
        assert testLoopRangesAContainedInB(newRange, rR)

        self.loopRanges.update(newRange)

def generator(arch, descr, target):
  if target == 'cpu':
    return Generic(arch, descr)
  elif target == 'gpu':
    raise RuntimeError("Elementwise operation has not been implemented for GPU-like architectures. At least not like this.")
