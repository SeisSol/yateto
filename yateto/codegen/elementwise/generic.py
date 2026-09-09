from ..common import *

class Generic(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def _affine(self, add, alpha):
    """(flops, assigner) for `result <op>= alpha * value`.

    `alpha` is the expression to scale by, or None where there is nothing to
    scale by -- it has already been formatted, so it cannot be compared to one.
    """
    flops = 1
    assign = '+=' if add else '='

    if alpha is not None: flops += 1
    if add: flops += 1

    # NOTE: the operation is parenthesised before the factor is applied to it.
    #       `*` binds tighter than `&`, `|`, `^`, `+` and every comparison, so
    #       `alpha * a & b` is `(alpha * a) & b` -- the factor would land on
    #       the first operand instead of on the result.
    if alpha is None:
      return flops, lambda left, right: f'{left} {assign} {right};'
    return flops, lambda left, right: f'{left} {assign} {alpha} * ({right});'

  def _generateDenseDense(self, cpp):
    d = self._descr

    if not d.add:
      writeBB = boundingBoxFromLoopRanges(d.result.indices, d.loopRanges)
      initializeWithZero(cpp, d.result, writeBB)

    # NOTE: read the factor in the *result's* datatype, so that e.g. an int32
    #       result does not get multiplied by a double literal
    trivial = d.alpha == 1.0
    with hoisted(cpp, d.result.datatype, d.alpha, '_alpha') as alpha:
      flops, assigner = self._affine(d.add, None if trivial else alpha)

      class ElementwiseBody(object):
        def __call__(s):
          args = [operand(arg) for arg in d.terms]
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

    flops, assigner = self._affine(
        d.add, None if d.alpha == 1.0 else scaleFactor(d.result.datatype, d.alpha))

    # where each operand's indices sit in the result's, and its pattern
    positions = [d.result.indices.positions(term.indices, sort=False) for term in d.terms]
    patterns = [term.eqspp.as_ndarray() for term in d.terms]

    total = 0
    nonzeros = d.result.eqspp.nonzero()
    for entry in sorted(zip(*nonzeros), key=lambda x: x[::-1]):
      args = []
      for term, position, pattern in zip(d.terms, positions, patterns):
        termEntry = tuple(entry[position] for position in position)
        if term.addressing == AddressingMode.SCALAR:
          args.append(term.name)
        elif pattern[termEntry]:
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
