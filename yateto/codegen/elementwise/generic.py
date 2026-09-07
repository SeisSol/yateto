from ..common import *

class Generic(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def _affine(self, add, alpha, datatype):
    flops = 1
    # NOTE: format the scale factor in the *result's* datatype, so that e.g.
    #       an int32 result does not get multiplied by a double literal
    scale = '' if alpha == 1.0 else f'{scaleFactor(datatype, alpha)} * '
    assign = '+=' if add else '='

    if alpha != 1.0: flops += 1
    if add: flops += 1

    return flops, lambda left, right: f'{left} {assign} {scale}{right};'

  def _generateDenseDense(self, cpp):
    d = self._descr

    if not d.add:
      writeBB = boundingBoxFromLoopRanges(d.result.indices, d.loopRanges)
      initializeWithZero(cpp, d.result, writeBB)

    flops, assigner = self._affine(d.add, d.alpha, d.result.datatype)

    class ElementwiseBody(object):
      def __call__(s):
        args = [f'{arg.name}[{arg.memoryLayout.addressString(arg.indices)}]' for arg in d.terms]
        fullArgs = [args[index] if template is None else template for index, template in zip(d.nodeTermIndices, d.termTemplate)]
        opstr = d.optype.callstr(*fullArgs)
        resultstr = f'{d.result.name}[{d.result.memoryLayout.addressString(d.result.indices)}]'
        cpp(assigner(resultstr, opstr))
        return flops
    return forLoops(cpp, d.result.indices, d.loopRanges, ElementwiseBody())

  def generate(self, cpp, routineCache):
    d = self._descr

    if any(d.isSparse):
      raise NotImplementedError(
        'Element-wise operations on sparse operands are not implemented yet: '
        'they would have to be emitted by unrolling.')

    return self._generateDenseDense(cpp)
