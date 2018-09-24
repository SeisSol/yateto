from ..common import *

class Generic(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def generate(self, cpp, routineCache):
    d = self._descr
    if not d.add:
      writeBB = boundingBoxFromLoopRanges(d.result.indices, d.loopRanges)
      initializeWithZero(cpp, self._arch, d.result, writeBB)

    class ProductBody(object):
      def __call__(s):
        mult = '{} * '.format(d.alpha) if d.alpha != 1.0 else ''
        cpp( '{}[{}] {} {}{}[{}] * {}[{}];'.format(
            d.result.name, d.result.memoryLayout.addressString(d.result.indices),
            '+=' if d.add else '=',
            mult,
            d.leftTerm.name, d.leftTerm.memoryLayout.addressString(d.leftTerm.indices),
            d.rightTerm.name, d.rightTerm.memoryLayout.addressString(d.rightTerm.indices)
          )
        )
        flop = 1
        if d.add:
          flop += 1
        if d.alpha not in [-1.0, 1.0]:
          flop += 1
        return flop

    return forLoops(cpp, d.result.indices, d.loopRanges, ProductBody())
