from ..common import TensorDescription
from .generic import Generic
from .libxsmm import Libxsmm

class Description(object):
  def __init__(self, result: TensorDescription, leftTerm: TensorDescription, rightTerm: TensorDescription, transA, transB, alpha, beta):
    self.result = result
    self.leftTerm = leftTerm
    self.rightTerm = rightTerm
    self.transA = transA
    self.transB = transB
    self.alpha = alpha
    self.beta = beta
    

def generator(arch, descr):
  requiresTranspositions = descr.transA or descr.transB
  simpleAlpha = descr.alpha in [-1.0, 1.0]
  simpleBeta = descr.beta in [0.0, 1.0]
  strideOneA = descr.leftTerm.memoryLayout.stride(0) == 1
  strideOneB = descr.rightTerm.memoryLayout.stride(0) == 1
  strideOneC = descr.result.memoryLayout.stride(0) == 1
  strideOne = strideOneA and strideOneB and strideOneC
  if not requiresTranspositions and simpleAlpha and simpleBeta and strideOne:
    return Libxsmm(arch, descr)
  return Generic(arch, descr)

