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
        target = f'{d.result.name}[{d.result.memoryLayout.addressString(d.result.indices)}]'
        initialValue = target if d.add else d.result.datatype.literal(d.optype.neutral())
        cpp(f'{d.result.datatype.ctype()} acc = {initialValue};')
        with cpp.For(f'int {sumIndex} = {d.sumLoopRange.start}; {sumIndex} < {d.sumLoopRange.stop}; ++{sumIndex}'):
          argstr = f'{d.term.name}[{d.term.memoryLayout.addressString(d.term.indices)}]'
          cpp( f'acc = {d.optype.callstr('acc', argstr)};' )
        mult = f'{d.alpha} * ' if d.alpha != 1.0 else ''
        cpp( f'{target} = {mult}acc;' )
        
        flop = 1 if d.alpha != 1.0 else 0
        return d.sumLoopRange.size() + flop

    return forLoops(cpp, d.result.indices, d.loopRanges, IndexSumBody())
