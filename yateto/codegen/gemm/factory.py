from ...ast.indices import BoundingBox, Range
from ...memory import CSCMemoryLayout
from ..common import TensorDescription
from .generic import Generic
from .libxsmm import Libxsmm

class Description(object):
  def __init__(self, result: TensorDescription, leftTerm: TensorDescription, rightTerm: TensorDescription, transA, transB, alpha, beta, arch, prefetchName = None):
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
    
    if (self.isACsc and transA) or (self.isBCsc and transB):
      raise RuntimeError('GEMM: sparse transposition is currently not supported.')
    
    if self.isACsc and self.isBCsc:
      raise RuntimeError('GEMM: sparse x sparse is currently not supported.')
    
    bbA = BoundingBox.fromSpp(self.leftTerm.eqspp)
    bbB = BoundingBox.fromSpp(self.rightTerm.eqspp)
    bbC = BoundingBox.fromSpp(self.result.eqspp)
    
    kA = 1 if not transA else 0
    kB = 0 if not transB else 1
    
    if self.leftTerm.memoryLayout.maySubDimension(kA) and self.rightTerm.memoryLayout.maySubDimension(kB):
      k = bbA[kA] & bbB[kB]
    else:
      k = Range(0, self.rightTerm.memoryLayout.shape()[kB])

    if self.leftTerm.memoryLayout.maySubDimension(1-kA) and self.result.memoryLayout.maySubDimension(0):
      m = bbA[1-kA]
    else:
      m = Range(0, self.leftTerm.memoryLayout.shape()[1-kA])

    if self.rightTerm.memoryLayout.maySubDimension(1-kB) and self.result.memoryLayout.maySubDimension(1):
      n = bbB[1-kB]
    else:
      n = Range(0, self.rightTerm.memoryLayout.shape()[1-kB])

    assert m in bbC[0]
    assert n in bbC[1]

    self.alignedA = not transA and self.leftTerm.memoryLayout.alignedStride()
    self.alignedC = self.result.memoryLayout.alignedStride()
    
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

def generator(arch, descr):
  requiresTranspositions = descr.transA or descr.transB
  simpleAlpha = descr.alpha == 1.0
  simpleBeta = descr.beta in [0.0, 1.0]
  AOk = descr.isACsc or descr.leftTerm.memoryLayout.stridei(0) == 1
  BOk = descr.isBCsc or descr.leftTerm.memoryLayout.stridei(0) == 1
  strideOneC = descr.result.memoryLayout.stridei(0) == 1
  memLayoutOk = AOk and BOk and strideOneC
  if not requiresTranspositions and simpleAlpha and simpleBeta and memLayoutOk:
    return Libxsmm(arch, descr)
  return Generic(arch, descr)

