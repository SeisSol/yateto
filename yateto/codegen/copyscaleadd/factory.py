from ..common import *
from .generic import Generic
from .csa_gen import CopyScaleAddGenerator

class Description(object):
  def __init__(self, alpha, beta, result: IndexedTensorDescription, term: IndexedTensorDescription):
    self.alpha = alpha
    self.beta = beta
    self.result = result
    self.term = term
    
    assert self.alpha != 0.0, 'copyscaleadd does not support alpha=0.0 at the moment.'
    assert self.beta == 1.0 or self.beta == 0.0, 'copyscaleadd supports only beta=0.0 or beta=1.0 at the moment.'
 
    rA = loopRanges(self.term, self.term.indices)
    rB = loopRanges(self.result, self.result.indices)
    assert testLoopRangesAContainedInB(rA, rB)
    assert self.term.indices <= self.result.indices and self.result.indices <= self.term.indices
    
    self.loopRanges = rA
    

def generator(arch, descr, target):
  return Generic(arch, descr) if target == 'cpu' else CopyScaleAddGenerator(arch, descr)
