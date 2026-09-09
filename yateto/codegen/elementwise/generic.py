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


class FusedGeneric(object):
  """Several element-wise steps emitted into one loop nest.

  A result that does not leave the nest becomes a local of the loop body: the
  buffer it used to occupy disappears, and so does the pass over it.
  """

  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def generate(self, cpp, routineCache):
    d = self._descr

    if not d.add:
      writeBB = boundingBoxFromLoopRanges(d.result.indices, d.loopRanges)
      initializeWithZero(cpp, d.result, writeBB)

    read = lambda term: term if isinstance(term, str) else operand(term)

    def spell(member):
      args = [read(term) for term in member.terms]
      step = member.step
      if step.optype is not None:
        return step.optype.callstr(*step.fillTerms(args))
      # a scaling: the factor written in the type it is applied in
      if step.scalar is None or step.scalar == 1.0:
        return args[0]
      return f'{scaleFactor(member.datatype, step.scalar)} * {args[0]}'

    with hoisted(cpp, d.result.datatype, d.alpha, '_alpha') as alpha:
      scale = '' if d.alpha == 1.0 else f'{alpha} * '
      assign = '+=' if d.add else '='

      def emit(member, write):
        """One step: a value, or an axis walked to make one."""
        step = member.step
        if step.reduction is None:
          write(spell(member))
          return 1
        # a reduction opens an axis of its own inside the nest and accumulates
        # over it; the loop variable is spelled the way forLoops spells one,
        # since the two loops end up in the same scope
        node = step.reduction
        sumIndex = f'{INDEX_PREFIX}{node.sumIndexName()}'
        rng = boundingBoxFromLoopRanges(node.term().indices,
                                        loopRanges(member.terms[0], node.term().indices))[
                node.term().indices.find(node.sumIndexName())]
        acc = f'{member.local or "_acc"}'
        cpp(f'{member.datatype.ctype()} {acc} = {node.optype.neutralLiteral(member.datatype)};')
        with cpp.For(f'int {sumIndex} = {rng.start}; {sumIndex} < {rng.stop}; ++{sumIndex}'):
          cpp(f'{acc} = {node.optype.callstr(acc, operand(member.terms[0]))};')
        if member.local is None:
          write(acc)
        return rng.size()

      class FusedBody(object):
        def __call__(s):
          flops = 0
          for member in d.members[:-1]:
            flops += emit(member, lambda expr, m=member:
                          cpp(f'{m.datatype.ctype()} const {m.local} = {expr};'))
          last = d.members[-1]
          target = operand(d.result)
          flops += emit(last, lambda expr:
                        cpp(f'{target} {assign} {scale}({expr});'))
          if d.alpha != 1.0: flops += 1
          if d.add: flops += 1
          return flops

      return forLoops(cpp, d.result.indices, d.loopRanges, FusedBody())
