import numpy as np
from ..memory import DenseMemoryLayout
from .indices import BoundingBox, Indices, LoGCost

class Node(object):
  def __init__(self):
    self.indices = None
    self._children = []
    self._eqspp = None
    self._boundingBox = None
  
  def size(self):
    return self.indices.size()
  
  def shape(self):
    return self.indices.shape()
  
  def nonZeroFlops(self):
    raise NotImplementedError

  def __iter__(self):
    return iter(self._children)
  
  def __getitem__(self, key):
    return self._children[key]
  
  def __len__(self):
    return len(self._children)
  
  def setChildren(self, children):
    self._children = children

  def eqspp(self):
    return self._eqspp
  
  def setEqspp(self, spp):
    self._eqspp = spp

  def boundingBox(self):
    return BoundingBox.fromSpp(self._eqspp)
  
  def memoryLayout(self):
    raise NotImplementedError
  
  def computeMemoryLayout(self):
    pass
  
  def fixedIndexPermutation(self):
    return True
    
  def setIndexPermutation(self, indices):
    if str(indices) != str(self.indices):
      raise NotImplementedError

  def _binOp(self, other, opType):
    if isinstance(self, opType):
      if isinstance(other, opType):
        self._children.extend(other._children)
      else:
        self._children.append(other)
      return self
    elif isinstance(other, opType):
      other._children.insert(0, self)
      return other
    return opType(self, other)

  def __mul__(self, other):
    return self._binOp(other, Einsum)
  
  def __add__(self, other):
    return self._binOp(other, Add)
    
  def __le__(self, other):
    if isinstance(other, IndexedTensor) and not (self == other):
      return Assign(self, Einsum(other))
    return Assign(self, other)

class IndexedTensor(Node):
  def __init__(self, tensor, indexNames):
    super().__init__()
    self.tensor = tensor
    self.indices = Indices(indexNames, self.tensor.shape())
  
  def nonZeroFlops(self):
    return 0
  
  def spp(self):
    return self.tensor.spp()
  
  def name(self):
    return self.tensor.name()
  
  def memoryLayout(self):
    return self.tensor.memoryLayout()

  def __str__(self):
    return '{}[{}]'.format(self.tensor.name(), str(self.indices))

class Op(Node):
  OPTIMIZE_EINSUM = True

  def __init__(self, *args):
    super().__init__()
    self._children = list(args)
    self._memoryLayout = None
  
  def memoryLayout(self):
    return self._memoryLayout

  def computeMemoryLayout(self):
    alignStride = any([child.memoryLayout().alignedStride() for child in self])
    self._memoryLayout = DenseMemoryLayout.fromSpp(self.eqspp(), alignStride=alignStride)
  
  def fixedIndexPermutation(self):
    return False
  
  def setIndexPermutation(self, indices):
    p = tuple([self.indices.find(idx) for idx in indices])
    if self._eqspp is not None:
      self._eqspp = self._eqspp.transpose(p)
    if self._memoryLayout is not None:
      self._memoryLayout.permute(p)
    self.indices.permute(indices)
  
  def __str__(self):
    return '{}[{}]'.format(type(self).__name__, self.indices if self.indices != None else '<not deduced>')
  
  def computeSparsityPattern(self, *spps):
    raise NotImplementedError

class Einsum(Op):
  pass
    
class Add(Op):
  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      spps = [node.eqspp() for node in self]
    spp = spps[0]
    for i in range(1, len(spps)):
      spp = np.add(spp, spps[i])
    return spp
  
  def nonZeroFlops(self):
    nzFlops = 0
    for child in self:
      nzFlops += np.count_nonzero( child.eqspp() )
    return nzFlops - np.count_nonzero( self.eqspp() )

class BinOp(Op):
  def __init__(self, lTerm, rTerm):
    super().__init__(lTerm, rTerm)
  
  def leftTerm(self):
    return self._children[0]
  
  def rightTerm(self):
    return self._children[1]
  
  def setChildren(self, children):
    if len(children) != 2:
      raise ValueError('BinOp node must have exactly 2 children.')
    super().setChildren(children)

class Assign(BinOp):
  def setChildren(self, children):
    if not isinstance(children[0], IndexedTensor):
      raise ValueError('First child of Assign node must be an IndexedTensor: ' + str(children[0]))
    super().setChildren(children)
    
  def nonZeroFlops(self):
    return 0
  
  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      return self.rightTerm().eqspp()
    assert len(spps) == 2
    return spps[1]

def _productContractionLoGSparsityPattern(node, *spps):
  if len(spps) == 0:
    spps = [node.leftTerm().eqspp(), node.rightTerm().eqspp()]
  assert len(spps) == 2
  # dense case
  if all([np.count_nonzero(spps[i]) == spps[i].size for i in range(2)]):
    return np.ones(node.indices.shape(), dtype=bool, order='F')
  # sparse case
  einsumDescription = '{},{}->{}'.format(node.leftTerm().indices.tostring(), node.rightTerm().indices.tostring(), node.indices.tostring())
  return np.einsum(einsumDescription, spps[0], spps[1], optimize=Op.OPTIMIZE_EINSUM)

class Product(BinOp):
  def __init__(self, lTerm, rTerm):
    super().__init__(lTerm, rTerm)
    K = lTerm.indices & rTerm.indices
    assert lTerm.indices.subShape(K) == rTerm.indices.subShape(K)

    self.indices = lTerm.indices.merged(rTerm.indices - K)

    # Cost based on indices
    # self._cost = 1
    # for size in self.indices.shape():
      # self._cost *= size

    # Cost based on sparsity pattern
    self.setEqspp( self.computeSparsityPattern() )
    self._cost = self.nonZeroFlops()

    # Cost based on bounding box
    # self.setBoundingBox( self.computeBoundingBox() )
    # self._cost = self.boundingBox().size()
    
    for child in self._children:
      self._cost += getattr(child, '_cost', 0)
  
  def nonZeroFlops(self):
    return np.count_nonzero( self.eqspp() )
  
  def computeSparsityPattern(self, *spps):
    return _productContractionLoGSparsityPattern(self, *spps)
  
  def computeBoundingBox(self):
    lt = self.leftTerm()
    rt = self.rightTerm()
    lbb = lt.boundingBox()
    rbb = rt.boundingBox()
    lind = set(lt.indices)
    rind = set(rt.indices)
    ranges = list()
    for index in self.indices:
      if index in lind and index in rind:
        lpos = lt.indices.find(index)
        rpos = rt.indices.find(index)
        ranges.append(lbb[lpos] & rbb[rpos])
      elif index in lind:
        ranges.append(lbb[lt.indices.find(index)])
      elif index in rind:
        ranges.append(rbb[rt.indices.find(index)])
      else:
        raise RuntimeError('Not supposed to happen.')
    return BoundingBox(ranges)
  
  def __str__(self):
    return '{} [{}] ({})'.format(type(self).__name__, self.indices, self._cost)

class IndexSum(Op):
  def __init__(self, term, sumIndex):
    super().__init__(term)
    self.indices = term.indices - set([sumIndex])
    self._sumIndex = term.indices.extract(sumIndex)
    
    # Cost based on indices
    # self._cost = term.indices.size()[sumIndex] - 1
    # for size in self.indices.shape():
      # self._cost *= size

    # Cost based on sparsity pattern
    self.setEqspp( self.computeSparsityPattern() )
    self._cost = self.nonZeroFlops()

    # Cost based on bounding box
    # self.setBoundingBox( self.computeBoundingBox() )
    # self._cost = term.boundingBox().size() - self.boundingBox().size()
    
    self._cost += getattr(term, '_cost', 0)
  
  def nonZeroFlops(self):
    return np.count_nonzero( self.term().eqspp() ) - np.count_nonzero( self.eqspp() )
  
  def sumIndex(self):
    return self._sumIndex
  
  def term(self):
    return self._children[0]
  
  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      spps = [self.term().eqspp()]
    assert len(spps) == 1
    einsumDescription = '{}->{}'.format(self.term().indices.tostring(), self.indices.tostring())
    return np.einsum(einsumDescription, spps[0], optimize=Op.OPTIMIZE_EINSUM)

  def computeBoundingBox(self):
    bb = self.term().boundingBox()
    pos = self.term().indices.find(str(self.sumIndex()))
    return BoundingBox([r for i,r in enumerate(bb) if i != pos])

  def __str__(self):
    return '{}_{} [{}] ({})'.format(type(self).__name__, self._sumIndex, self.indices, self._cost)

class Contraction(BinOp):
  def __init__(self, indices, lTerm, rTerm, sumIndices):
    super().__init__(lTerm, rTerm)
    li = lTerm.indices - sumIndices
    lr = rTerm.indices - sumIndices
    self.sumIndices = sumIndices
    self.indices = li.merged(lr)
    self.setIndexPermutation(indices)
    self.setEqspp( self.computeSparsityPattern() )

  def nonZeroFlops(self):
    p = Product(self.leftTerm(), self.rightTerm())
    return 2*p.nonZeroFlops() - np.count_nonzero( self.eqspp() )
  
  def computeSparsityPattern(self, *spps):
    return _productContractionLoGSparsityPattern(self, *spps)
  
  def __str__(self):
    return '{} [{}]'.format(type(self).__name__, self.indices)

class LoopOverGEMM(BinOp):
  def __init__(self, indices, aTerm, bTerm, m, n, k):
    super().__init__(aTerm, bTerm)
    self.indices = indices
    self._m = m
    self._n = n
    self._k = k
    self._transA = aTerm.indices.find(m[0]) > aTerm.indices.find(k[0])
    self._transB = bTerm.indices.find(k[0]) > bTerm.indices.find(n[0])
    self.setEqspp( self.computeSparsityPattern() )

  def nonZeroFlops(self):
    p = Product(self.leftTerm(), self.rightTerm())
    return 2*p.nonZeroFlops() - np.count_nonzero( self.eqspp() )
  
  def computeSparsityPattern(self, *spps):
    return _productContractionLoGSparsityPattern(self, *spps)
  
  def cost(self):
    A = self.leftTerm().indices
    B = self.rightTerm().indices
    AstrideOne = (A.find(self._m[0]) == 0) if not self._transA else (A.find(self._k[0]) == 0)
    BstrideOne = (B.find(self._k[0]) == 0) if not self._transB else (B.find(self._n[0]) == 0)
    cost = LoGCost(int(not AstrideOne) + int(not BstrideOne), int(self._transA), int(self._transB), len(self._m) + len(self._n) + len(self._k))
    return cost
  
  def loopIndices(self):
    i1 = self.indices - (self._m + self._n)
    i2 = (self.leftTerm().indices - (self._m + self._k)) - i1
    i3 = ((self.rightTerm().indices - (self._k + self._n)) - i1) - i2
    return i1.merged(i2).merged(i3)
  
  def transA(self):
    return self._transA

  def transB(self):
    return self._transB

  @staticmethod
  def indexString(name, subset, indices, transpose=False):
    indexStr = ''.join([i if i in subset else ':' for i in indices])
    matrixStr = '{}_{}'.format(name, indexStr)
    return '({})\''.format(matrixStr) if transpose else matrixStr
  
  def __str__(self):
    Astr = self.indexString('A', self._m + self._k, self.leftTerm().indices, self._transA)
    Bstr = self.indexString('B', self._k + self._n, self.rightTerm().indices, self._transB)
    Cstr = self.indexString('C', self._m + self._n, self.indices)
    return '{} [{}]: {} = {} {}'.format(type(self).__name__, self.indices, Cstr, Astr, Bstr)
