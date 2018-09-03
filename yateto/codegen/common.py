import numpy as np
from ..ast.indices import BoundingBox

class TensorDescription(object):
  def __init__(self, name, memoryLayout, eqspp):
    self.name = name
    self.memoryLayout = memoryLayout
    self.eqspp = eqspp
    BoundingBox(eqspp)
  
  @classmethod
  def fromNode(cls, name, node):
    return cls(name, node.memoryLayout(), node.eqspp())

class IndexedTensorDescription(TensorDescription):
  def __init__(self, name, indices, memoryLayout, eqspp):
    super().__init__(name, memoryLayout, eqspp)
    self.indices = indices

  @classmethod
  def fromNode(cls, name, node):
    return cls(name, node.indices, node.memoryLayout(), node.eqspp())

def forLoops(cpp, indexNames, ranges, body, indexNo=None):
  if indexNo == None:
    indexNo = len(indexNames)-1
  if indexNo < 0:
    body()
  else:
    index = indexNames[indexNo]
    rng = ranges[index]
    with cpp.For('int {0} = {1}; {0} < {2}; ++{0}'.format(index, rng.start, rng.stop)):
      forLoops(cpp, indexNames, ranges, body, indexNo-1)
  
def loopRanges(term: IndexedTensorDescription, loopIndices):
  overlap = set(loopIndices) & set(term.indices)
  return {index: term.memoryLayout.bboxi(term.indices.find(index)) for index in overlap}

def testLoopRangesEqual(A, B):
  overlap = A.keys() & B.keys()
  return all([A[index] == B[index] for index in overlap])
  
def testLoopRangesAContainedInB(A, B):
  overlap = A.keys() & B.keys()
  return all([A[index] in B[index] for index in overlap])

def reduceSpp(spp, sourceIndices, targetIndices):
  return np.einsum('{}->{}'.format(sourceIndices, targetIndices), spp)
