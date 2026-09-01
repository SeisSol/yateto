from ..common import *

class Generic(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def generate(self, cpp, routineCache):
    d = self._descr

    if not d.add:
      writeBB = boundingBoxFromLoopRanges(d.result.indices, d.loopRanges)
      initializeWithZero(cpp, d.result, writeBB)

    # forLoops() prefixes its loop variables; the reduction loop is emitted into
    # the same scope and has to use the very same spelling
    sumIndex = f'{INDEX_PREFIX}{d.sumIndex}'
    datatype = d.result.datatype
    accumulator = '_acc'

    class ReductionBody(object):
      def __call__(s):
        target = f'{d.result.name}[{d.result.memoryLayout.addressString(d.result.indices)}]'
        argstr = f'{d.term.name}[{d.term.memoryLayout.addressString(d.term.indices)}]'
        call = d.optype.callstr(accumulator, argstr)

        # the accumulator gets its own scope: a rank-0 reduction has no
        # surrounding loop, and several of them may share one enclosing scope
        with cpp.AnonymousScope():
          cpp(f'{datatype.ctype()} {accumulator} = {d.optype.neutralLiteral(datatype)};')
          with cpp.For(f'int {sumIndex} = {d.sumLoopRange.start}; {sumIndex} < {d.sumLoopRange.stop}; ++{sumIndex}'):
            cpp(f'{accumulator} = {call};')

          # `add` accumulates into the target:
          #   target = target (op) alpha*reduction
          scaled = accumulator if d.alpha == 1.0 else f'{datatype.literal(d.alpha)} * {accumulator}'
          if d.add:
            cpp(f'{target} = {d.optype.callstr(target, scaled)};')
          else:
            cpp(f'{target} = {scaled};')

        flop = d.sumLoopRange.size()
        if d.alpha != 1.0: flop += 1
        if d.add: flop += 1
        return flop

    return forLoops(cpp, d.result.indices, d.loopRanges, ReductionBody())
