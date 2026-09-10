from ... import ir


def tensorOp(descr):
  """The statement an element-wise description states."""
  return ir.Elementwise(descr.result, descr.fillTerms(descr.terms), descr.optype,
                        alpha=descr.alpha, add=descr.add,
                        loopRanges=descr.loopRanges,
                        unrolled=any(descr.isSparse))


def fusedTensorOp(descr):
  """The statement a nest of element-wise steps states."""
  return ir.FusedElementwise(descr.result, descr.members, alpha=descr.alpha,
                             add=descr.add, loopRanges=descr.loopRanges)


class Generic(object):
  """One element-wise operation over an index space."""

  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def generate(self, cpp, routineCache):
    region = ir.unroll(tensorOp(self._descr).lower())
    ir.CppEmitter(cpp).emit(region)
    return ir.countFlops(region)


class FusedGeneric(object):
  """Several element-wise steps emitted into one loop nest.

  A result that does not leave the nest becomes a value of the loop body: the
  buffer it used to occupy disappears, and so does the pass over it.
  """

  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def generate(self, cpp, routineCache):
    region = ir.unroll(fusedTensorOp(self._descr).lower())
    ir.CppEmitter(cpp).emit(region)
    return ir.countFlops(region)
