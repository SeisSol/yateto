from ..common import *
from .generic import Generic

class Description(object):
  def __init__(self, alpha, add: bool, result: IndexedTensorDescription, term: IndexedTensorDescription):
    self.alpha = alpha
    self.add = add
    self.result = result
    self.term = term
    
    rA = loopRanges(self.term, self.result.indices)
    rB = loopRanges(self.result, self.result.indices)
    assert testLoopRangesAContainedInB(rA, rB)
    
    self.loopRanges = rA
    
    self.sumIndex = self.term.indices - self.result.indices
    assert len(self.sumIndex) == 1

    self.sumLoopRange = loopRanges(self.term, self.sumIndex)[str(self.sumIndex)]
    

def generator(arch, descr, target):
  if target == 'cpu':
    return Generic(arch, descr)
  elif target == 'gpu':
    raise RuntimeError("IndexSum operation has not been implemented for GPU-like architectures")