from ... import ir


def tensorOp(descr):
  """The statement a copy-scale-add description states.

  One operand and a destination cover a plain copy, a transposition and a
  broadcast alike, and which of the three it is can be read off the two index
  tuples. It changes what the statement is called and not how it is lowered,
  but the name is what the levels above this one reason about.
  """
  if len(descr.term.indices) < len(descr.result.indices):
    operation = ir.Broadcast
  elif list(descr.term.indices) != list(descr.result.indices):
    operation = ir.Transpose
  else:
    operation = ir.Copy
  return operation(descr.result, [descr.term], alpha=descr.alpha,
                   add=descr.beta == 1.0, loopRanges=descr.loopRanges,
                   unrolled=descr.term.memoryLayout.isSparse())


class Generic(object):
  """``result <op>= alpha * term``, built as IR and emitted from it."""

  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def lower(self):
    return ir.unroll(tensorOp(self._descr).lower())
