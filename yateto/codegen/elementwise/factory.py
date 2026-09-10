from ..common import *
from .generic import Generic, FusedGeneric

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
    if not any(self.isSparse):
      for term in self.terms:
        termRange = loopRanges(term, self.result.indices)
        assert testLoopRangesAContainedInB(termRange, rR), \
          f'Operand {term.name} exceeds the result\'s loop ranges.'
        for index, rng in termRange.items():
          assert self.loopRanges[index] == rng or index not in term.indices, \
            f'Inconsistent loop range for index {index}.'
          self.loopRanges[index] = rng

  def fillTerms(self, args):
    """The operands in their original order, with the templates put back."""
    return [args[index] if template is None else template
            for index, template in zip(self.nodeTermIndices, self.termTemplate)]

class FusedMember(object):
  """One step of a fused nest.

  `terms` holds, per operand, an IndexedTensorDescription to read through, or
  None where the operand is what an earlier step computed -- which step that
  is stands in `step.sources`.
  """

  def __init__(self, step, terms, datatype):
    self.step = step
    self.terms = terms
    self.datatype = datatype


class FusedDescription(object):
  """Several element-wise steps over one index space."""

  def __init__(self, alpha, add, result, members, loopRanges):
    self.alpha = alpha
    self.add = add
    self.result = result
    self.members = members
    self.loopRanges = loopRanges


def generator(arch, descr, target):
  if target == 'cpu':
    return Generic(arch, descr)
  elif target == 'gpu':
    raise RuntimeError("Elementwise operation has not been implemented for GPU-like architectures. At least not like this.")

def fusedGenerator(arch, descr, target):
  if target == 'cpu':
    return FusedGeneric(arch, descr)
  raise RuntimeError('Fused element-wise steps are a CPU thing: on a device the '
                     'loop nest is the kernel launch, and putting several of '
                     'them together is the external generator\'s business.')
