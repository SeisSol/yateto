from ..common import *
from .generic import Generic

from ...ops import Operation

class Description(object):
  def __init__(self, alpha, add: bool, optype: Operation, result: IndexedTensorDescription, terms, termTemplate, nodeTermIndices):
    self.alpha = alpha
    self.add = add
    self.result = result
    self.terms = terms
    self.optype = optype
    self.termTemplate = termTemplate
    self.nodeTermIndices = nodeTermIndices

    # a sparse operand is emitted by unrolling, not by an address expression:
    # its addressString is empty by design
    self.isSparse = [term.memoryLayout.isSparse()
                     for term in list(terms) + [self.result]]

    rR = loopRanges(self.result, self.result.indices)

    # NOTE: operands need not span all of the result's indices -- an operand
    #       that lacks an index is simply broadcast along it. The loop ranges
    #       are therefore driven by the result and only narrowed where an
    #       operand actually has the index.
    self.loopRanges = dict(rR)
    for term in self.terms:
      termRange = loopRanges(term, self.result.indices)
      assert testLoopRangesAContainedInB(termRange, rR), \
        f'Operand {term.name} exceeds the result\'s loop ranges.'
      for index, rng in termRange.items():
        assert self.loopRanges[index] == rng or index not in term.indices, \
          f'Inconsistent loop range for index {index}.'
        self.loopRanges[index] = rng

def generator(arch, descr, target):
  if target == 'cpu':
    return Generic(arch, descr)
  elif target == 'gpu':
    raise RuntimeError("Elementwise operation has not been implemented for GPU-like architectures. At least not like this.")
