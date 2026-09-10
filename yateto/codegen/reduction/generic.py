from ... import ir


def tensorOp(descr, generator=None):
  """The statement a reduction description states."""
  return ir.Reduction(descr.result, [descr.term], descr.optype, descr.sumIndex,
                      descr.sumLoopRange, alpha=descr.alpha, add=descr.add,
                      loopRanges=descr.loopRanges, generator=generator)


class Generic(object):
  """An operation folding one index of its operand away."""

  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def lower(self):
    return ir.unroll(tensorOp(self._descr).lower())
