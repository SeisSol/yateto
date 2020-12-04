from .. import aspp
from ..ast.indices import BoundingBox
from ..ast.log import splitByDistance


class TensorDescription(object):
  def __init__(self, name, memoryLayout, eqspp, is_compute_constant=False, is_temporary=False):
    """

    Args:
      name (str): tensor's symbol name
      memoryLayout:
      eqspp:
      is_compute_constant (bool): if true then sparsity patterns and numerical values of tensor
          elements are known at compile time
      is_temporary (bool): if true then the description is for a temporary tensor which
          usually results from a result of an intermediate computation
    """
    self.name = name
    self.memoryLayout = memoryLayout
    self.eqspp = eqspp
    self.is_compute_constant = is_compute_constant
    self.is_temporary = is_temporary
    BoundingBox(eqspp)
  
  @classmethod
  def fromNode(cls, name, node):
    return cls(name, node.memoryLayout(), node.eqspp())

class IndexedTensorDescription(TensorDescription):
  def __init__(self, name, indices, memoryLayout, eqspp, is_compute_constant=False, is_temporary=False):
    super().__init__(name, memoryLayout, eqspp, is_compute_constant, is_temporary)
    self.indices = indices

  @classmethod
  def fromNode(cls, var, node):
    is_const = False
    if hasattr(node, 'tensor'):
      is_const = node.tensor.is_compute_constant()
    return cls(str(var), node.indices, var.memoryLayout(), node.eqspp(), is_const, var.is_temporary)

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


class BatchedOperationsAux:
  NUM_ELEMENTS_NAME = 'numElements'
  EXTRA_OFFSET_NAME = 'extraOffset'
  STREAM_PTR_NAME = 'streamPtr'

  def __init__(self, underlying_data_type):
    self.underlying_data_type = underlying_data_type

  def _get_ptr_type(self, addressing):
    return '**' if addressing == 'pointer_based' else '*'


  def deduce_addresing(self, term):
    if term.is_compute_constant:
      return 'none'
    if term.is_temporary:
      return 'strided'
    else:
      return 'pointer_based'


  def deduce_arg(self, term, as_const=False):
    if term.is_compute_constant or term.is_temporary:
      extra_offset = '0'
    else:
      extra_offset = f'{self.EXTRA_OFFSET_NAME}_{term.name}'

    if as_const:
      addressing = self.deduce_addresing(term)
      ptr = self._get_ptr_type(addressing)
      const_ptr_type = f'const {self.underlying_data_type} {ptr}'
      return f'const_cast<{const_ptr_type}>({term.name}), {extra_offset}'
    else:
      return f'{term.name}, {extra_offset}'
