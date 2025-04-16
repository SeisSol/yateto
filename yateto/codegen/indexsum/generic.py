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
    
    sumIndex = d.term.indices - d.result.indices
    assert len(sumIndex) == 1
    class IndexSumBody(object):
      def __call__(s):
        target = '{}[{}]'.format(d.result.name, d.result.memoryLayout.addressString(d.result.indices))
        initialValue = target if d.add else d.result.datatype.literal(0.0)
        cpp(f'{d.result.datatype.ctype()} sum = {initialValue};')
        with cpp.For('int {0} = {1}; {0} < {2}; ++{0}'.format(sumIndex, d.sumLoopRange.start, d.sumLoopRange.stop)):
          cpp( f'sum += {d.term.name}[{d.term.memoryLayout.addressString(d.term.indices)}];' )
        mult = f'{d.alpha} * ' if d.alpha != 1.0 else ''
        cpp( f'{target} = {mult}sum;' )
        
        flop = 1 if d.alpha != 1.0 else 0
        return d.sumLoopRange.size() + flop

    return forLoops(cpp, d.result.indices, d.loopRanges, IndexSumBody())
