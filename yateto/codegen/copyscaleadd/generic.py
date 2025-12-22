from ..common import *

class Generic(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr
  
  def _formatTerm(self, alpha, term, entry):
    prefix = ''
    if alpha == 0.0:
      return ''
    if alpha == 1.0:
      prefix = term.name
    else:
      prefix = f'{alpha} * {term.name}'

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
        initializeWithZero(cpp, self._arch, d.result)
      else:
        writeBB = boundingBoxFromLoopRanges(d.result.indices, d.loopRanges)
        initializeWithZero(cpp, self._arch, d.result, writeBB)
    

    class CopyScaleAddBody(object):
      def __init__(self, resultEntry, termEntry):
        self.resultEntry = resultEntry
        self.termEntry = termEntry

      def __call__(s):
        op = '='
        flop = 0
        alpha = d.alpha
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
        cpp( f'{self._formatTerm(1.0, d.result, s.resultEntry)} {op} {self._formatTerm(alpha, d.term, s.termEntry)};' )

        return flop

    if d.term.memoryLayout.isSparse():

      indexmap = d.result.indices.positions(d.term.indices, sort=False)

      flops = 0
      nonzeros = d.result.eqspp.nonzero()
      for entryR in sorted(zip(*nonzeros), key=lambda x: x[::-1]):
        entry = tuple(entryR[ pos ] for pos in indexmap)
        flops += CopyScaleAddBody(entryR, entry)()
      
      return flops

    else:

      return forLoops(cpp, d.result.indices, d.loopRanges, CopyScaleAddBody(None, None))
