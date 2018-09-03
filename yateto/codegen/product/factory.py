from ..common import *
from .generic import Generic

class Description(object):
  def __init__(self, add: bool, result: IndexedTensorDescription, leftTerm: IndexedTensorDescription, rightTerm: IndexedTensorDescription):
    self.add = add
    self.result = result
    self.leftTerm = leftTerm
    self.rightTerm = rightTerm
    
    rA = loopRanges(self.leftTerm, self.result.indices)
    rB = loopRanges(self.rightTerm, self.result.indices)
    rC = loopRanges(self.result, self.result.indices)
    assert testLoopRangesEqual(rA, rB)
    assert testLoopRangesEqual(rA, rC)
    assert testLoopRangesEqual(rB, rC)
    
    rA.update(rB)
    rA.update(rC)

    self.loopRanges = rA    

def generator(arch, descr):
  return Generic(arch, descr)

