from ..common import forLoops

class Generic(object):
  def __init__(self, arch, descr):
    self._descr = descr
  
  def _strideOffset(self, term, offset2, transpose):
    stride = term.memoryLayout.stride()
    if transpose:
      stride = stride[::-1]
      offset2 = offset2[::-1]
    offset = term.memoryLayout.address(offset2)
    return (stride, offset)
    
  def generate(self, cpp, routineCache):
    d = self._descr
    m, n, k = d.mnk()
    
    Astride, Aoffset = self._strideOffset(d.leftTerm, (m.start, k.start), d.transA)
    Bstride, Boffset = self._strideOffset(d.rightTerm, (k.start, n.start), d.transB)
    Cstride, Coffset = self._strideOffset(d.result, (m.start, n.start), False)
    
    CAddr = '{Cname}[{Coffset} + {Cstride[0]}*m + {Cstride[1]}*n]'.format(
			Cname = d.result.name,
			Coffset = Coffset,
			Cstride = Cstride,
    )
    
    with cpp.For('int n = 0; n < {0}; ++n'.format(n.size())):
      if d.beta != 1.0:
        with cpp.For('int m = 0; m < {0}; ++m'.format(m.size())):
          cpp('{} = {}{};'.format(
              CAddr,
              d.beta,
              ' * ' + CAddr if d.beta != 0.0 else ''
            )
          )
      with cpp.For('int k = 0; k < {0}; ++k'.format(k.size())):
        with cpp.For('int m = 0; m < {0}; ++m'.format(m.size())):
          cpp( '{CAddr} += {alpha} * {Aname}[{Aoffset} + {Astride[0]}*m + {Astride[1]}*k] * {Bname}[{Boffset} + {Bstride[0]}*k + {Bstride[1]}*n];'.format(
              CAddr = CAddr,
              Aname = d.leftTerm.name,
              Aoffset = Aoffset,
              Astride = Astride,
              Bname = d.rightTerm.name,
              Boffset = Boffset,
              Bstride = Bstride,
              alpha = d.alpha
            )
          )

    return 2 * m.size() * n.size() * k.size()
