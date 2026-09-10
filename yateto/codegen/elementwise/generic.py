from ... import ir


def tensorOp(descr):
  """The statement an element-wise description states."""
  return ir.Elementwise(descr.result, descr.fillTerms(descr.terms), descr.optype,
                        alpha=descr.alpha, add=descr.add,
                        loopRanges=descr.loopRanges,
                        unrolled=any(descr.isSparse))


class Generic(object):
  """One element-wise operation over an index space."""

  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def lower(self):
    return ir.unroll(tensorOp(self._descr).lower())
