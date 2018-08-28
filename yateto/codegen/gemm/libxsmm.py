class Libxsmm(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr
  
  def generateRoutineName(self, gemm):
    return 'libxsmm_m{M}_n{N}_k{K}_ldA{LDA}_ldB{LDB}_ldC{LDC}_alpha{alpha}_beta{beta}_alignedA{alignedA}_alignedC{alignedC}_{prefetch}'.format(**gemm)
    
  def generate(self, cpp):
    d = self._descr
    M = d.result.memoryLayout.shape(0)
    N = d.result.memoryLayout.shape(1)
    K = d.leftTerm.memoryLayout.shape(1)
    ldA = d.leftTerm.memoryLayout.stride(1)
    ldB = d.rightTerm.memoryLayout.stride(1)
    ldC = d.result.memoryLayout.stride(1)
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
      d.leftTerm.name,
      d.rightTerm.name,
      d.result.name
    ))
