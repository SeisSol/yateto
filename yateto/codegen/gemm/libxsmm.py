class Libxsmm(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr
  
  def generateRoutineName(self, gemm):
    return 'libxsmm_m{M}_n{N}_k{K}_ldA{LDA}_ldB{LDB}_ldC{LDC}_alpha{alpha}_beta{beta}_alignedA{alignedA}_alignedC{alignedC}_{prefetch}'.format(**gemm)
  
  def _pointer(self, term, offset2):
    o = term.memoryLayout.offset(offset2)
    if o > 0:
      return '{} + {}'.format(term.name, o)
    return term.name
    
  def generate(self, cpp):
    d = self._descr
    M, N, K = d.mnk()
    mo, no, ko = d.mnkOffset()
    ldA = d.leftTerm.memoryLayout.stridei(1)
    ldB = d.rightTerm.memoryLayout.stridei(1)
    ldC = d.result.memoryLayout.stridei(1)
    alignedA = self._arch.checkAlignment(ldA)
    alignedC = self._arch.checkAlignment(ldC)
    
    gemm = {
      'M':            M,
      'N':            N,
      'K':            K,
      'LDA':          ldA,
      'LDB':          ldB,
      'LDC':          ldC,
      'alpha':        int(d.alpha),
      'beta':         int(d.beta),
      'alignedA':     int(alignedA),
      'alignedC':     int(alignedC),
      'prefetch':     'pfsigonly'
    }
    
    cpp( '{}({}, {}, {}, NULL, NULL, NULL);'.format(
      self.generateRoutineName(gemm),
      self._pointer(d.leftTerm, (mo, ko)),
      self._pointer(d.rightTerm, (ko, no)),
      self._pointer(d.result, (mo, no))
    ))
