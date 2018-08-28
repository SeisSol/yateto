from ..common import forLoops

class Generic(object):
  def __init__(self, arch, descr):
    self._descr = descr
    
  def generate(self, cpp):
    d = self._descr
    M = d.result.memoryLayout.shape(0)
    N = d.result.memoryLayout.shape(1)
    K = d.leftTerm.memoryLayout.shape(1) if not d.transA else d.leftTerm.memoryLayout.shape(0)
    with cpp.For('int n = 0; n < {0}; ++n'.format(N)):
      with cpp.For('int k = 0; k < {0}; ++k'.format(K)):
        with cpp.For('int m = 0; m < {0}; ++m'.format(M)):
          cpp( '{Cname}[{Cstride0}*m + {Cstride1}*n] = {alpha} * {Aname}[{Astride0}*{Aindex0} + {Astride1}*{Aindex1}] * {Bname}[{Bstride0}*{Bindex0} + {Bstride1}*{Bindex1}] + {beta} * {Cname}[{Cstride0}*m + {Cstride1}*n]];'.format(
              Cname = d.result.name,
              Cstride0 = d.result.memoryLayout.stride(0),
              Cstride1 = d.result.memoryLayout.stride(1),
              Aname = d.leftTerm.name,
              Astride0 = d.leftTerm.memoryLayout.stride(0),
              Aindex0 = 'k' if d.transA else 'm',
              Astride1 = d.leftTerm.memoryLayout.stride(1),
              Aindex1 = 'm' if d.transA else 'k',
              Bname = d.rightTerm.name,
              Bstride0 = d.rightTerm.memoryLayout.stride(0),
              Bindex0 = 'n' if d.transB else 'k',
              Bstride1 = d.rightTerm.memoryLayout.stride(1),
              Bindex1 = 'k' if d.transB else 'n',
              alpha = d.alpha,
              beta = d.beta
            )
          )
