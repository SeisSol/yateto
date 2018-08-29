from ..common import forLoops

class Generic(object):
  def __init__(self, arch, descr):
    self._descr = descr
  
  def _formatTerm(self, alpha, name):
    if alpha == 0.0:
      return None
    elif alpha == 1.0:
      return name + '[i]'
    return '{} * {}[i]'.format(alpha, name)

  def generate(self, cpp):
    d = self._descr
    assert d.result.memoryLayout.size() == d.term.memoryLayout.size()
    size = d.result.memoryLayout.size()
    with cpp.For('int i = 0; i < {}; ++i'.format(size)):
      if d.beta == 1.0:
        cpp( '{}[i] += {};'.format(d.result.name, self._formatTerm(d.alpha, d.term.name)) )
      else:
        terms = [self._formatTerm(d.alpha, d.term.name), self._formatTerm(d.beta, d.result.name)]
        cpp( '{}[i] = {};'.format(d.result.name, ' + '.join(term for term in terms if term)) )
