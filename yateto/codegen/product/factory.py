from ...memory import CSCMemoryLayout
from ..common import *
from .generic import Generic

class Description(object):
  def __init__(self, alpha, add: bool, result: IndexedTensorDescription, leftTerm: IndexedTensorDescription, rightTerm: IndexedTensorDescription):
    self.alpha = alpha
    self.add = add
    self.result = result
    self.leftTerm = leftTerm
    self.rightTerm = rightTerm

    self.isACsc = isinstance(self.leftTerm.memoryLayout, CSCMemoryLayout)
    self.isBCsc = isinstance(self.rightTerm.memoryLayout, CSCMemoryLayout)
    
    rA = loopRanges(self.leftTerm, self.result.indices)
    rB = loopRanges(self.rightTerm, self.result.indices)
    rC = loopRanges(self.result, self.result.indices)
    assert testLoopRangesEqual(rA, rB)
    assert testLoopRangesAContainedInB(rA, rC)
    assert testLoopRangesAContainedInB(rB, rC)
    
    rA.update(rB)

    self.loopRanges = rA    

def generator(arch, descr):
  return Generic(arch, descr)

