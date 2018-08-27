import numpy as np
from .indices import Indices, LoGCost

class Node(object):
  def __init__(self):
    self.indices = None
    self._children = []
    self._eqspp = None
  
  def size(self):
    return self.indices.size()

  def __iter__(self):
    return iter(self._children)
  
  def __getitem__(self, key):
    return self._children[key]
  
  def setChildren(self, children):
    self._children = children

  def eqspp(self):
    return self._eqspp
  
  def setEqspp(self, spp):
    self._eqspp = spp
    
  def setIndexPermutation(self, indices):
    self.indices.permute(indices)

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
    return Assign(self, other)

class IndexedTensor(Node):
  def __init__(self, tensor, indexNames):
    super().__init__()
    self.tensor = tensor
    self.indices = Indices(indexNames, self.tensor.shape())
  
  def spp(self):
    return self.tensor.spp()
  
  def name(self):
    return self.tensor.name()

  def __str__(self):
    return '{}[{}]'.format(self.tensor.name(), str(self.indices))

class Op(Node):
  def __init__(self, *args):
    super().__init__()
    self._children = list(args)
  
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
  
  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      return self.rightTerm().eqspp()
    assert len(spps) == 2
    return spps[1]

def _productContractionLoGSparsityPattern(node, *spps):
  if len(spps) == 0:
    spps = [node.leftTerm().eqspp(), node.rightTerm().eqspp()]
  assert len(spps) == 2
  einsumDescription = '{},{}->{}'.format(node.leftTerm().indices.tostring(), node.rightTerm().indices.tostring(), node.indices.tostring())
  return np.einsum(einsumDescription, spps[0], spps[1])

class Product(BinOp):
  def __init__(self, lTerm, rTerm, target_indices):
    super().__init__(lTerm, rTerm)
    if target_indices.firstIndex() <= rTerm.indices:
      lTerm, rTerm = rTerm, lTerm

    K = lTerm.indices & rTerm.indices
    assert lTerm.indices.subShape(K) == rTerm.indices.subShape(K)

    self.indices = lTerm.indices.merged(rTerm.indices - K)

    self._cost = 1
    for size in self.indices.shape():
      self._cost *= size
    for child in self._children:
      self._cost += getattr(child, '_cost', 0)
  
  def computeSparsityPattern(self, *spps):
    return _productContractionLoGSparsityPattern(self, *spps)
  
  def __str__(self):
    return '{} [{}] ({})'.format(type(self).__name__, self.indices, self._cost)

class IndexSum(Op):
  def __init__(self, term, sumIndex):
    super().__init__(term)
    self.indices = term.indices - set([sumIndex])
    self._sumIndex = term.indices.extract(sumIndex)
    self._cost = term.indices.size()[sumIndex]
    for size in self.indices.shape():
      self._cost *= size
    self._cost += getattr(term, '_cost', 0)
  
  def sumIndex(self):
    return self._sumIndex
  
  def term(self):
    return self._children[0]
  
  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      spps = [self.term().eqspp()]
    assert len(spps) == 1
    einsumDescription = '{}->{}'.format(self.term().indices.tostring(), self.indices.tostring())
    return np.einsum(einsumDescription, spps[0])

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
    self._Atrans = aTerm.indices.find(m[0]) > aTerm.indices.find(k[0])
    self._Btrans = bTerm.indices.find(k[0]) > bTerm.indices.find(n[0])
  
  def computeSparsityPattern(self, *spps):
    return _productContractionLoGSparsityPattern(self, *spps)
  
  def cost(self):
    A = self.leftTerm().indices
    B = self.rightTerm().indices
    AstrideOne = (A.find(self._m[0]) == 0) if not self._Atrans else (A.find(self._k[0]) == 0)
    BstrideOne = (B.find(self._k[0]) == 0) if not self._Btrans else (B.find(self._n[0]) == 0)
    cost = LoGCost(int(not AstrideOne) + int(not BstrideOne), int(self._Atrans), int(self._Btrans), len(self._m) + len(self._n) + len(self._k))
    return cost
  
  def loopIndices(self):
    return self.indices - (self._m + self._n)

  @staticmethod
  def indexString(name, subset, indices, transpose=False):
    indexStr = ''.join([i if i in subset else ':' for i in indices])
    matrixStr = '{}_{}'.format(name, indexStr)
    return '({})\''.format(matrixStr) if transpose else matrixStr
  
  def __str__(self):
    Astr = self.indexString('A', self._m + self._k, self.leftTerm().indices, self._Atrans)
    Bstr = self.indexString('B', self._k + self._n, self.rightTerm().indices, self._Btrans)
    Cstr = self.indexString('C', self._m + self._n, self.indices)
    return '{} [{}]: {} = {} {}'.format(type(self).__name__, self.indices, Cstr, Astr, Bstr)
