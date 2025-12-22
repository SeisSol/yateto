from ..common import *

class Generic(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def _affine(self, add, alpha):
    flops = 1
    scale = f'{alpha} * ' if alpha != 1.0 else ''
    assign = '+=' if add else '='

    if alpha != 1.0: flops += 1
    if add: flops += 1

    return flops, lambda left, right: f'{left} {assign} {scale}{right};'

  def _generateDenseDense(self, cpp):
    d = self._descr

    if not d.add:
      writeBB = boundingBoxFromLoopRanges(d.result.indices, d.loopRanges)
      initializeWithZero(cpp, d.result, writeBB)
    
    flops, assigner = self._affine(d.add, d.alpha)

    class ProductBody(object):
      def __call__(s):
        args = [f'{arg.name}[{arg.memoryLayout.addressString(arg.indices)}]' for arg in d.terms]
        fullArgs = [args[index] if template is None else template for index, template in zip(d.nodeTermIndices, d.termTemplate)]
        opstr = d.optype.callstr(*fullArgs)
        resultstr = f'{d.result.name}[{d.result.memoryLayout.addressString(d.result.indices)}]'
        cpp(assigner(resultstr, opstr))
        return flops
    return forLoops(cpp, d.result.indices, d.loopRanges, ProductBody())

  def generate(self, cpp, routineCache):
    d = self._descr

    return self._generateDenseDense(cpp)
