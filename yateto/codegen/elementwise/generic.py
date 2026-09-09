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
        opstr = d.optype.callstr(*d.fillTerms(args))
        resultstr = f'{d.result.name}[{d.result.memoryLayout.addressString(d.result.indices)}]'
        cpp(assigner(resultstr, opstr))
        return flops
    return forLoops(cpp, d.result.indices, d.loopRanges, ElementwiseBody())

  def _generateUnrolled(self, cpp):
    """One statement per non-zero of the result.

    A sparse operand has no address expression to loop over, so the entries are
    written out. An operand that is structurally zero at an entry is spelled as
    a literal zero instead of being read -- it has no address to read from, and
    it lets the compiler fold the operation away where the zero decides it.
    """
    d = self._descr

    if not d.add:
      initializeWithZero(cpp, d.result)

    flops, assigner = self._affine(d.add, d.alpha, d.result.datatype)

    # where each operand's indices sit in the result's, and its pattern
    positions = [d.result.indices.positions(term.indices, sort=False) for term in d.terms]
    patterns = [term.eqspp.as_ndarray() for term in d.terms]

    total = 0
    nonzeros = d.result.eqspp.nonzero()
    for entry in sorted(zip(*nonzeros), key=lambda x: x[::-1]):
      args = []
      for term, position, pattern in zip(d.terms, positions, patterns):
        termEntry = tuple(entry[position] for position in position)
        if pattern[termEntry]:
          args.append(f'{term.name}[{term.memoryLayout.address(termEntry)}]')
        else:
          args.append(term.datatype.literal(0))
      resultstr = f'{d.result.name}[{d.result.memoryLayout.address(entry)}]'
      cpp(assigner(resultstr, d.optype.callstr(*d.fillTerms(args))))
      total += flops

    return total

  def generate(self, cpp, routineCache):
    # a sparse operand or result is addressed entry by entry, so the loops are
    # unrolled; everything dense is looped over
    if any(self._descr.isSparse):
      return self._generateUnrolled(cpp)

    return self._generateDenseDense(cpp)
