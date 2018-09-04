from ..common import *
from .generic import Generic

class Description(object):
  def __init__(self, add: bool, result: IndexedTensorDescription, term: IndexedTensorDescription):
    self.add = add
    self.result = result
    self.term = term
    
    rA = loopRanges(self.result, self.result.indices)
    rB = loopRanges(self.term, self.result.indices)
    print(self.term.memoryLayout.bbox())
    print(self.result.memoryLayout.bbox())
    print(rA['j'], rB['j'])
    assert testLoopRangesAContainedInB(rA, rB)
    
    self.loopRanges = rA
    
    self.sumIndex = self.term.indices - self.result.indices
    assert len(self.sumIndex) == 1

    self.sumLoopRange = loopRanges(self.term, self.sumIndex)[str(self.sumIndex)]
    

def generator(arch, descr):
  return Generic(arch, descr)

