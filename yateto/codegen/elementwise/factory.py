from ..common import *
from .generic import Generic

from ... import aspp
from ...ast.indices import Range
from ...ops import Operation

import numpy as np

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
    #       that lacks an index is simply broadcast along it. Nor need they
    #       span all of an index they do have: a constant with zeros in it has
    #       a narrower pattern than the axis it sits on, and so does anything
    #       computed from one. Where an operand stops, its value is a
    #       structural zero, and the generator spells it as a literal.
    self.termRanges = [loopRanges(term, self.result.indices) for term in self.terms]
    self.loopRanges = dict(rR)
    #       That an index letter carries one extent throughout is settled
    #       during index deduction, so the ranges here differ in their bounds
    #       and not in what they address.
    if not any(self.isSparse):
      for term, termRange in zip(self.terms, self.termRanges):
        assert testLoopRangesAContainedInB(termRange, rR), \
          f'Operand {term.name} exceeds the result\'s loop ranges.'
      for index, rng in rR.items():
        self.loopRanges[index] = self._liveRange(index, rng)

  def _liveRange(self, index, resultRange):
    """The part of `resultRange` over which the result can be non-zero.

    Asked of the operation itself, on one-dimensional patterns built from the
    operands' ranges, so that the answer follows the same rule as the sparsity
    pattern of the whole node: a product is zero wherever one factor is, a sum
    only where every summand is, and a function of zero is not zero at all.
    An operand that lacks the index is broadcast along it and constrains
    nothing.
    """
    patterns = []
    for termRange in self.termRanges:
      rng = termRange.get(index, resultRange)
      covered = np.zeros((resultRange.stop,), dtype=bool)
      covered[rng.start:rng.stop] = True
      patterns.append(aspp.general(covered))
    live = np.flatnonzero(self.optype.sparsityResult(patterns).as_ndarray())
    if live.size == 0:
      return Range(resultRange.start, resultRange.start)
    return Range(max(resultRange.start, int(live[0])),
                 min(resultRange.stop, int(live[-1]) + 1))

  def fillTerms(self, args):
    """The operands in their original order, with the templates put back."""
    return [args[index] if template is None else template
            for index, template in zip(self.nodeTermIndices, self.termTemplate)]

def generator(arch, descr, target):
  if target == 'cpu':
    return Generic(arch, descr)
  elif target == 'gpu':
    raise RuntimeError("Elementwise operation has not been implemented for GPU-like architectures. At least not like this.")
