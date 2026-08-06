import copy
from ..common import *
from .generic import Generic

class Description(object):
  def __init__(self,
               alpha,
               add: bool,
               result: IndexedTensorDescription,
               leftTerm: IndexedTensorDescription,
               rightTerm: IndexedTensorDescription,
               loopIndices,
               transA,
               transB,
               prefetchName):
    self.alpha = alpha
    self.add = add
    self.result = result
    self.leftTerm = leftTerm
    self.rightTerm = rightTerm
    self.loopIndices = loopIndices
    self.transA = transA
    self.transB = transB
    self.prefetchName = prefetchName

    rA = loopRanges(self.leftTerm, self.loopIndices)
    rB = loopRanges(self.rightTerm, self.loopIndices)
    rC = loopRanges(self.result, self.loopIndices)
    assert testLoopRangesEqual(rA, rB)
    assert testLoopRangesAContainedInB(rA, rC)
    assert testLoopRangesAContainedInB(rB, rC)

    rC.update(rA)
    rC.update(rB)

    self.loopRanges = rC

    self.innerLoopIndices = self.loopIndices - self.result.indices
    self.outerLoopIndices = self.loopIndices - self.innerLoopIndices

    if self.add:
      self.assignLoopRanges = None
      self.addLoopRanges = [copy.deepcopy(self.loopRanges)]
    elif len(self.innerLoopIndices) == 0:
      self.assignLoopRanges = copy.deepcopy(self.loopRanges)
      self.addLoopRanges = None
    else:
      # peel off the very first point of the inner iteration space (that one
      # assigns), and cover the remainder with one box per inner index
      peelOff = [str(i) for i in self.innerLoopIndices]

      self.assignLoopRanges = copy.deepcopy(self.loopRanges)
      for idx in peelOff:
        self.assignLoopRanges[idx].stop = self.loopRanges[idx].start + 1

      self.addLoopRanges = []
      for n, idx in enumerate(peelOff):
        box = copy.deepcopy(self.loopRanges)
        box[idx].start = self.loopRanges[idx].start + 1
        for pinned in peelOff[n+1:]:
          box[pinned].stop = self.loopRanges[pinned].start + 1
        if box[idx].size() > 0:
          self.addLoopRanges.append(box)
      if len(self.addLoopRanges) == 0:
        self.addLoopRanges = None


def generator(arch, descr, target):
  return Generic(arch, descr, target)
