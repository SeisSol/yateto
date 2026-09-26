from ..common import *

from ...ast.indices import Range

import itertools

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

  def _split(self):
    """The loop range cut into blocks along which no operand starts or stops.

    An operand narrower than the range it sits on has no value stored past its
    own end -- reading it there would read a neighbouring entry. Cutting the
    range at the operands' bounds gives blocks over which every operand is
    either readable throughout or structurally zero throughout, and the zero
    goes into the expression as a literal. A single block, which is the usual
    case, reproduces one plain loop nest.
    """
    d = self._descr

    perIndex = []
    for index in d.result.indices:
      whole = d.loopRanges[index]
      cuts = {whole.start, whole.stop}
      for termRange in d.termRanges:
        rng = termRange.get(index)
        if rng is not None:
          cuts.update(cut for cut in (rng.start, rng.stop) if whole.start < cut < whole.stop)
      bounds = sorted(cuts)
      perIndex.append([Range(start, stop) for start, stop in zip(bounds, bounds[1:])])

    for block in itertools.product(*perIndex):
      yield dict(zip(d.result.indices, block))

  def _readable(self, term, termRange, block):
    """Whether `term` has a value stored throughout `block`."""
    return all(block[index] in rng for index, rng in termRange.items())

  def _generateDenseDense(self, cpp):
    d = self._descr

    if any(rng.size() == 0 for rng in d.loopRanges.values()):
      # the operation is zero over the whole result
      if not d.add:
        initializeWithZero(cpp, d.result)
      return 0

    if not d.add:
      writeBB = boundingBoxFromLoopRanges(d.result.indices, d.loopRanges)
      initializeWithZero(cpp, d.result, writeBB)

    # NOTE: read the factor in the *result's* datatype, so that e.g. an int32
    #       result does not get multiplied by a double literal
    trivial = d.alpha == 1.0
    with hoisted(cpp, d.result.datatype, d.alpha, '_alpha') as alpha:
      flops, assigner = self._affine(d.add, None if trivial else alpha)

      class ElementwiseBody(object):
        def __init__(s, args):
          s.args = args

        def __call__(s):
          opstr = d.optype.callstr(*d.fillTerms(s.args))
          resultstr = f'{d.result.name}[{d.result.memoryLayout.addressString(d.result.indices)}]'
          cpp(assigner(resultstr, opstr))
          return flops

      total = 0
      for block in self._split():
        args = [operand(term) if self._readable(term, termRange, block)
                else term.datatype.literal(0)
                for term, termRange in zip(d.terms, d.termRanges)]
        total += forLoops(cpp, d.result.indices, block, ElementwiseBody(args))
      return total

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
        if isImmediate(term):
          # The numbers are the operand. Spelling them out is what the mode
          # was asked for: a zero among them costs nothing, a one is not a
          # multiplication, and the compiler sees both.
          value = immediateValue(term, termEntry) if pattern[termEntry] else 0
          args.append(term.datatype.literal(value))
        elif not hasStorage(term):
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
    if any(self._descr.isSparse) or any(self._descr.isImmediate):
      return self._generateUnrolled(cpp)

    return self._generateDenseDense(cpp)
