from ..common import forLoops

class Generic(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def generate(self, cpp):
    d = self._descr
    sumIndex = d.term.indices - d.result.indices
    assert len(sumIndex) == 1
    class IndexSumBody(object):
      def __call__(s):
        target = '{}[{}]'.format(d.result.name, d.result.memoryLayout.addressString(d.result.indices))
        initialValue = target if d.add else self._arch.formatConstant(0.0)
        cpp( '{} sum = {};'.format(self._arch.typename, initialValue) )
        with cpp.For('int {0} = {1}; {0} < {2}; ++{0}'.format(sumIndex, d.sumLoopRange.start, d.sumLoopRange.stop)):
          cpp( 'sum += {}[{}];'.format(d.term.name, d.term.memoryLayout.addressString(d.term.indices)) )
        cpp( '{} = sum;'.format(target) )
    forLoops(cpp, d.result.indices, d.loopRanges, IndexSumBody())
