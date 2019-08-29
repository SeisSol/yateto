from .. import aspp
from ..ast.indices import BoundingBox
from ..ast.log import splitByDistance

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
  def fromNode(cls, var, node):
    return cls(str(var), node.indices, var.memoryLayout(), node.eqspp())

def forLoops(cpp, indexNames, ranges, body, pragmaSimd=True, prefix='_', indexNo=None):
  flops = 0
  if indexNo == None:
    indexNo = len(indexNames)-1
  if indexNo < 0:
    flops = body()
  else:
    index = indexNames[indexNo]
    rng = ranges[index]
    if pragmaSimd and indexNo == 0:
      cpp('#pragma omp simd')
    with cpp.For('int {3}{0} = {1}; {3}{0} < {2}; ++{3}{0}'.format(index, rng.start, rng.stop, prefix)):
      flops = forLoops(cpp, indexNames, ranges, body, pragmaSimd, prefix, indexNo-1)
    flops = flops * rng.size()
  return flops
  
def loopRanges(term: IndexedTensorDescription, loopIndices):
  overlap = set(loopIndices) & set(term.indices)
  bbox = BoundingBox.fromSpp(term.eqspp)
  return {index: bbox[term.indices.find(index)] for index in overlap}

def testLoopRangesEqual(A, B):
  overlap = A.keys() & B.keys()
  return all([A[index] == B[index] for index in overlap])
  
def testLoopRangesAContainedInB(A, B):
  overlap = A.keys() & B.keys()
  return all([A[index] in B[index] for index in overlap])

def boundingBoxFromLoopRanges(indices, loopRanges):
  return BoundingBox([loopRanges[index] for index in indices])

def reduceSpp(spp, sourceIndices, targetIndices):
  return spp.indexSum(sourceIndices, targetIndices)

def initializeWithZero(cpp, arch, result: TensorDescription, writeBB = None):
  if writeBB:
    addresses = sorted(result.memoryLayout.notWrittenAddresses(writeBB))
    if len(addresses) > 0:
      regions = splitByDistance(addresses)
      for region in regions:
        m, M = min(region), max(region)
        initialAddress = '{} + {}'.format(result.name, m)
        cpp.memset(initialAddress, M-m+1, arch.typename)
  else:
    cpp.memset(result.name, result.memoryLayout.requiredReals(), arch.typename)
