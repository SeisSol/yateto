from ...ast.indices import BoundingBox
from ..common import TensorDescription
from .generic import Generic
from .libxsmm import Libxsmm

class Description(object):
  def __init__(self, result: TensorDescription, leftTerm: TensorDescription, rightTerm: TensorDescription, transA, transB, alpha, beta, arch):
    self.result = result
    self.leftTerm = leftTerm
    self.rightTerm = rightTerm
    self.transA = transA
    self.transB = transB
    self.alpha = alpha
    self.beta = beta
    
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
  simpleAlpha = descr.alpha in [-1.0, 1.0]
  simpleBeta = descr.beta in [0.0, 1.0]
  strideOneA = descr.leftTerm.memoryLayout.stridei(0) == 1
  strideOneB = descr.rightTerm.memoryLayout.stridei(0) == 1
  strideOneC = descr.result.memoryLayout.stridei(0) == 1
  strideOne = strideOneA and strideOneB and strideOneC
  if not requiresTranspositions and simpleAlpha and simpleBeta and strideOne:
    return Libxsmm(arch, descr)
  return Generic(arch, descr)

