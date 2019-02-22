from ..common import *

class Generic(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def _mult(self, alpha):
    return '{} * '.format(alpha) if alpha != 1.0 else ''

  def _flop(self, add, alpha):
    flop = 1
    if add:
      flop += 1
    if alpha not in [-1.0, 1.0]:
      flop += 1
    return flop

  def _generateDenseDense(self, cpp):
    d = self._descr

    if not d.add:
      writeBB = boundingBoxFromLoopRanges(d.result.indices, d.loopRanges)
      initializeWithZero(cpp, self._arch, d.result, writeBB)

    class ProductBody(object):
      def __call__(s):
        mult = self._mult(d.alpha)
        cpp( '{}[{}] {} {}{}[{}] * {}[{}];'.format(
            d.result.name, d.result.memoryLayout.addressString(d.result.indices),
            '+=' if d.add else '=',
            mult,
            d.leftTerm.name, d.leftTerm.memoryLayout.addressString(d.leftTerm.indices),
            d.rightTerm.name, d.rightTerm.memoryLayout.addressString(d.rightTerm.indices)
          )
        )
        return self._flop(d.add, d.alpha)

    return forLoops(cpp, d.result.indices, d.loopRanges, ProductBody())

  def _generateSparseDense(self, cpp):
    raise NotImplementedError

  def _generateSparseSparse(self, cpp):
    d = self._descr
    assert d.isACsc and d.isBCsc

    if not d.add:
      initializeWithZero(cpp, self._arch, d.result)

    left = d.result.indices.positions(d.leftTerm.indices, sort=False)
    right = d.result.indices.positions(d.rightTerm.indices, sort=False)

    mult = self._mult(d.alpha)
    flop = self._flop(d.add, d.alpha)

    flops = 0
    nonzeros = d.result.eqspp.nonzero()
    for entry in sorted(zip(*nonzeros), key=lambda x: x[::-1]):
      leftEntry = (entry[ left[0] ], entry[ left[1] ])
      rightEntry = (entry[ right[0] ], entry[ right[1] ])

      cpp( '{}[{}] {} {}{}[{}] * {}[{}];'.format(
          d.result.name, d.result.memoryLayout.address(entry),
          '+=' if d.add else '=',
          mult,
          d.leftTerm.name, d.leftTerm.memoryLayout.address(leftEntry),
          d.rightTerm.name, d.rightTerm.memoryLayout.address(rightEntry)
        )
      )

      flops += flop

    return flops

  def generate(self, cpp, routineCache):
    d = self._descr

    if d.isACsc and d.isBCsc:
      return self._generateSparseSparse(cpp)

    if d.isACsc or d.isBCsc:
      return self._generateSparseDense(cpp)

    return self._generateDenseDense(cpp)
