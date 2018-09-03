from ..common import *
from .generic import Generic

class Description(object):
  def __init__(self, add: bool, result: IndexedTensorDescription, leftTerm: IndexedTensorDescription, rightTerm: IndexedTensorDescription, loopIndices, transA, transB):
    self.add = add
    self.result = result
    self.leftTerm = leftTerm
    self.rightTerm = rightTerm
    self.loopIndices = loopIndices
    self.transA = transA
    self.transB = transB
    
    rA = loopRanges(self.leftTerm, self.loopIndices)
    rB = loopRanges(self.rightTerm, self.loopIndices)
    rC = loopRanges(self.result, self.loopIndices)
    assert testLoopRangesEqual(rA, rB)
    assert testLoopRangesEqual(rA, rC)
    assert testLoopRangesEqual(rB, rC)
    
    rA.update(rB)
    rA.update(rC)

    self.loopRanges = rA

def generator(arch, descr):
  return Generic(arch, descr)

