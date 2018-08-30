import numpy as np
from ..ast.indices import BoundingBox

class TensorDescription(object):
  def __init__(self, name, memoryLayout, eqspp):
    self.name = name
    self.memoryLayout = memoryLayout
    self.eqspp = eqspp
    BoundingBox(eqspp)
  
  @classmethod
  def fromNode(self, name, node):
    return cls(name, node.memoryLayout(), node.eqspp())

class IndexedTensorDescription(TensorDescription):
  def __init__(self, name, indices, memoryLayout, eqspp):
    super().__init__(name, memoryLayout, eqspp)
    self.indices = indices

  @classmethod
  def fromNode(cls, name, node):
    return cls(name, node.indices, node.memoryLayout(), node.eqspp())

def forLoops(cpp, indices, indexNo, body):
  if indexNo < 0:
    body()
  else:
    with cpp.For('int {0} = 0; {0} < {1}; ++{0}'.format(indices[indexNo], indices.shape()[indexNo])):
      forLoops(cpp, indices, indexNo-1, body)

def reduceSpp(spp, sourceIndices, targetIndices):
  return np.einsum('{}->{}'.format(sourceIndices, targetIndices), spp)
