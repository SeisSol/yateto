from ...ast.indices import BoundingBox, Range
from ...memory import CSCMemoryLayout
from ..common import TensorDescription
from .generic import Generic
from .gemmgen import GemmGen

class Description(object):
  def __init__(self, result: TensorDescription, leftTerm: TensorDescription, rightTerm: TensorDescription, transA, transB, alpha, beta, arch, alignedStartA, alignedStartC, prefetchName = None):
    self.result = result
    self.leftTerm = leftTerm
    self.rightTerm = rightTerm
    self.transA = transA
    self.transB = transB
    self.alpha = alpha
    self.beta = beta
    self.prefetchName = prefetchName
    
    self.isACsc = isinstance(self.leftTerm.memoryLayout, CSCMemoryLayout)
    self.isBCsc = isinstance(self.rightTerm.memoryLayout, CSCMemoryLayout)
    
    if self.isACsc and self.isBCsc:
      raise RuntimeError('GEMM: sparse x sparse is currently not supported.')
    
    bbA = BoundingBox.fromSpp(self.leftTerm.eqspp)
    bbB = BoundingBox.fromSpp(self.rightTerm.eqspp)
    bbC = BoundingBox.fromSpp(self.result.eqspp)
    
    kA = 1 if not transA else 0
    kB = 0 if not transB else 1
    
    k = bbA[kA] & bbB[kB]
    m = bbA[1-kA]
    n = bbB[1-kB]

    assert m in bbC[0]
    assert n in bbC[1]

    self.alignedA = alignedStartA and not transA and self.leftTerm.memoryLayout.alignedStride()
    self.alignedC = alignedStartC and self.result.memoryLayout.alignedStride()
    
    if self.alignedA and self.alignedC:
      m = m.aligned(arch)
    else:
      mStartAligned = arch.checkAlignment(m.start)
      self.alignedA = self.alignedA & mStartAligned
      self.alignedC = self.alignedC & mStartAligned
    
    self._mnk = (m, n, k)

  def mnk(self):
    return self._mnk
  
  def setBeta(self, beta):
    self.beta = beta

def generator(arch, descr, gemm_cfg):
  AOk = descr.isACsc or descr.leftTerm.memoryLayout.stridei(0) == 1
  BOk = descr.isBCsc or descr.rightTerm.memoryLayout.stridei(0) == 1
  strideOneC = descr.result.memoryLayout.stridei(0) == 1
  memLayoutOk = AOk and BOk and strideOneC
  if memLayoutOk:
    m, n, k = descr.mnk()
    gemmTool = gemm_cfg.getGemmTool(
      m.size(), n.size(), k.size(),
      descr.isACsc, descr.isBCsc,
      descr.transA, descr.transB,
      descr.alpha, descr.beta)
    if gemmTool:
      return GemmGen(arch, descr, gemmTool)
  return Generic(arch, descr)

