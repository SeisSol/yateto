from ..common import forLoops

class Generic(object):
  def __init__(self, arch, descr):
    self._descr = descr
  
  def _formatTerm(self, alpha, term):
    prefix = ''
    if alpha == 0.0:
      return None
    if alpha == 1.0:
      prefix = term.name
    else:
      prefix = '{} * {}'.format(alpha, term.name)
    return '{}[{}]'.format(alpha, term.memoryLayout.addressString(term.indices))

  def generate(self, cpp):
    d = self._descr

    class CopyScaleAddBody(object):
      def __call__(s):
        op = '='
        if d.beta == 1.0:
          op = '+='
        elif d.beta != 0.0:
          raise NotImplementedError
        cpp( '{} {} {};'.format(self._formatTerm(1.0, d.result), op, self._formatTerm(d.alpha, d.term)) )

    forLoops(cpp, d.result.indices, d.loopRanges, CopyScaleAddBody())
