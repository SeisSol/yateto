from ..common import forLoops

class Generic(object):
  def __init__(self, arch, descr):
    self._descr = descr
  
  def _strideOffset(self, term, offset2, transpose):
    stride = term.memoryLayout.stride()
    if transpose:
      stride = stride[::-1]
      offset2 = offset2[::-1]
    offset = term.memoryLayout.offset(offset2)
    return (stride, offset)
    
  def generate(self, cpp):
    d = self._descr
    M, N, K = d.mnk()
    mo, no, ko = d.mnkOffset()
    
    Astride, Aoffset = self._strideOffset(d.leftTerm, (mo, ko), d.transA)
    Bstride, Boffset = self._strideOffset(d.rightTerm, (ko, no), d.transB)
    Cstride, Coffset = self._strideOffset(d.result, (mo, no), False)
    
    with cpp.For('int n = 0; n < {0}; ++n'.format(N)):
      with cpp.For('int k = 0; k < {0}; ++k'.format(K)):
        with cpp.For('int m = 0; m < {0}; ++m'.format(M)):
          cpp( '{Cname}[{Coffset} + {Cstride[0]}*m + {Cstride[1]}*n] = {alpha} * {Aname}[{Aoffset} + {Astride[0]}*m + {Astride[1]}*k] * {Bname}[{Boffset} + {Bstride[0]}*k + {Bstride[1]}*n] + {beta} * {Cname}[{Coffset} + {Cstride[0]}*m + {Cstride[1]}*n];'.format(
              Cname = d.result.name,
              Coffset = Coffset,
              Cstride = Cstride,
              Aname = d.leftTerm.name,
              Aoffset = Aoffset,
              Astride = Astride,
              Bname = d.rightTerm.name,
              Boffset = Boffset,
              Bstride = Bstride,
              alpha = d.alpha,
              beta = d.beta
            )
          )
