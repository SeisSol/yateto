from ..common import *

class Generic(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def _formatTerm(self, alpha, term, entry, datatype=None):
    prefix = ''
    if alpha == 0.0:
      return ''
    if alpha == 1.0:
      prefix = term.name
    else:
      # NOTE: format the scale factor in the result's datatype, so an int32
      #       result is not silently multiplied by a double literal. A factor
      #       generate() has already read into a local arrives as its name and
      #       passes through untouched.
      prefix = f'{scaleFactor(datatype or term.datatype, alpha)} * {term.name}'

    if entry is None:
      return f'{prefix}[{term.memoryLayout.addressString(term.indices)}]'
    else:
      if term.memoryLayout.hasValue(entry):
        return f'{prefix}[{term.memoryLayout.address(entry)}]'
      else:
        # needed for some temporaries
        return self._arch.formatConstant(0.0)

  def generate(self, cpp, routineCache):
    d = self._descr

    if d.beta == 0.0:
      if d.term.memoryLayout.isSparse():
        initializeWithZero(cpp, d.result)
      else:
        writeBB = boundingBoxFromLoopRanges(d.result.indices, d.loopRanges)
        initializeWithZero(cpp, d.result, writeBB)


    class CopyScaleAddBody(object):
      def __init__(self, alpha, resultEntry, termEntry):
        self.alpha = alpha
        self.resultEntry = resultEntry
        self.termEntry = termEntry

      def __call__(s):
        op = '='
        flop = 0
        alpha = s.alpha
        if alpha not in [-1.0, 1.0]:
          flop += 1
        if d.beta == 1.0 and alpha == -1.0:
          alpha = 1.0
          op = '-='
          flop += 1
        elif d.beta == 1.0:
          op = '+='
          flop += 1
        elif d.beta != 0.0:
          raise NotImplementedError
        cpp( f'{self._formatTerm(1.0, d.result, s.resultEntry)} {op} {self._formatTerm(alpha, d.term, s.termEntry, d.result.datatype)};' )

        return flop

    # NOTE: read the factor in the result's datatype, so an int32 result is not
    #       silently multiplied by a double literal. A factor of one or minus
    #       one is folded into the assignment and never reaches a local.
    hoistable = d.alpha if d.alpha not in (0.0, 1.0, -1.0) else float(d.alpha)
    with hoisted(cpp, d.result.datatype, hoistable, '_alpha') as alpha:

      if d.term.memoryLayout.isSparse():

        indexmap = d.result.indices.positions(d.term.indices, sort=False)

        flops = 0
        nonzeros = d.result.eqspp.nonzero()
        for entryR in sorted(zip(*nonzeros), key=lambda x: x[::-1]):
          entry = tuple(entryR[ pos ] for pos in indexmap)
          flops += CopyScaleAddBody(alpha, entryR, entry)()

        return flops

      return forLoops(cpp, d.result.indices, d.loopRanges, CopyScaleAddBody(alpha, None, None))
