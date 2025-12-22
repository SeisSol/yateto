import re
from ..memory import DenseMemoryLayout
from .indices import BoundingBox, Indices, LoGCost
from abc import ABC, abstractmethod
from .. import aspp, ops
import numpy as np

class Node(ABC):
  def __init__(self):
    self.indices = None
    self._children = []
    self._eqspp = None
    self.datatype = None
    self.prefetch = None
  
  def size(self):
    return self.indices.size()
  
  def shape(self):
    return self.indices.shape()
  
  @abstractmethod
  def nonZeroFlops(self):
    pass

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
  
  @abstractmethod
  def memoryLayout(self):
    pass

  def argumentsCompatible(self, layouts):
    return True

  def resultCompatible(self, layout):
    return True

  def fixedIndexPermutation(self):
    return True

  @abstractmethod
  def setIndexPermutation(self, indices, permuteEqspp=True):
    pass

  def permute(self, indices, spp, strict=True):
    perm = tuple(indices.find(idx) for idx in self.indices if idx in indices or strict)
    return spp.transposed(perm)
  
  def reshape(self, indices, spp):
    rshp = [indices.indexSize(idx) if idx in indices else 1 for idx in self.indices]
    return spp.reshape(rshp)
  
  def broadcast(self, indices, spp):
    reshaped = self.reshape(indices, spp)
    bcst = [1 if idx in indices else self.indices.indexSize(idx) for idx in self.indices]
    return reshaped.broadcast(bcst)

  def _checkMultipleScalarMults(self):
    if isinstance(self, ScalarMultiplication):
      raise ValueError('Multiple multiplications with scalars are not allowed. Merge them into a single one.')

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
    if not isinstance(other, Node):
      self._checkMultipleScalarMults()
      return ScalarMultiplication(other, self)
    if isinstance(self, ScalarMultiplication):
      other._checkMultipleScalarMults()
      self.setTerm(self.term() * other)
      return self
    elif isinstance(other, ScalarMultiplication):
      self._checkMultipleScalarMults()
      other.setTerm(self * other.term())
      return other
    return self._binOp(other, Einsum)
  
  def __rmul__(self, other):
    return self.__mul__(other)
  
  def __add__(self, other):
    if not isinstance(other, Node):
      raise ValueError(f'Unsupported operation: Cannot add {self} to {other}.')
    return self._binOp(other, Add)
  
  def __radd__(self, other):
    return self.__add__(other)
  
  def __neg__(self):
    self._checkMultipleScalarMults()
    return ScalarMultiplication(-1.0, self)

  def __sub__(self, other):
    return self._binOp(-other, Add)
    
  def __le__(self, other):
    return Assign(self, other)
  
  def __truediv__(self, other):
    return Elementwise(ops.Div(), self, other)
  
  def __rtruediv__(self, other):
    return Elementwise(ops.Div(), other, self)

  def subslice(self, index, start, end):
    return SliceView(self, index, start, end)
  
  def subselect(self, index, position):
    return SliceView(self, index, position, position + 1)
  
  def viewed(self):
    return self

class SliceView(Node):
  def __init__(self, subnode, index, start, end):
    super().__init__()
    self._children = [subnode]
    self.index = index
    self.start = start
    self.end = end
  
  def name(self):
    return self.term().name()
  
  def viewed(self):
    return self.term().viewed()
  
  def term(self):
    return self[0]
  
  def nonZeroFlops(self):
    return 0
  
  def setIndexPermutation(self, indices, permuteEqspp=True):
    assert str(indices) == str(self.indices)

  def memoryLayout(self):
    return self._memoryLayout
  
  def getMemoryLayout(self, memoryLayout):
    return memoryLayout.subslice(list(self.indices).index(self.index), self.start, self.end)

  def computeMemoryLayout(self):
    self._memoryLayout = self.getMemoryLayout(self.term().memoryLayout())
  
  def computeSparsityPattern(self, *spps):
    assert len(spps) in (0, 1)
    spp = spps[0] if len(spps) == 1 else self.term().eqspp()
    
    if isinstance(spp, aspp.dense):
      nowshape = spp.shape
      subshape = tuple(self.end - self.start if self.indices[i] == self.index else nowshape[i] for i in range(spp.ndim))
      return aspp.dense(subshape)
    else:
      subslice = tuple(slice(self.start, self.end) if self.indices[i] == self.index else slice(None) for i in range(spp.ndim))
      subarray = spp.as_ndarray()[subslice]
      return aspp.general(subarray)
  
  def __str__(self):
    return f'{type(self).__name__}[{self.index}: {self.start}..{self.end}]'

class IndexedTensor(Node):
  def __init__(self, tensor, indexNames):
    super().__init__()
    self.tensor = tensor
    self.indices = Indices(indexNames, self.tensor.shape())
  
  def nonZeroFlops(self):
    return 0
  
  def setIndexPermutation(self, indices, permuteEqspp=True):
    assert str(indices) == str(self.indices)
  
  def spp(self, groupSpp=True):
    return self.tensor.spp(groupSpp)
  
  def name(self):
    return self.tensor.name()
  
  def memoryLayout(self):
    return self.tensor.memoryLayout()

  def __deepcopy__(self, memo):
    it = IndexedTensor(self.tensor, str(self.indices))
    if self._eqspp is not None:
      it._eqspp = self._eqspp.copy()
    return it

  def __str__(self):
    return f'{self.tensor.name()}[{str(self.indices)}]'

class Op(Node):
  def __init__(self, *args):
    super().__init__()
    self._children = list(args)
    self._memoryLayout = None
  
  def memoryLayout(self):
    return self._memoryLayout

  def setMemoryLayout(self, memLayout):
    self._memoryLayout = memLayout

  def computeMemoryLayout(self):
    alignStride = False
    alignOffset = float('inf')

    if self.indices is not None and len(self.indices) > 0:
      for child in self:
        if self.indices[0] in child.indices:
          position = child.indices.find(self.indices[0])
          if child.memoryLayout().mayVectorizeDim(position):
            alignStride = True
            alignOffset = min(alignOffset, child.memoryLayout().alignmentOffset(position))

    # NOTE: the offset is needed for slicing. Since we don't use selector matrices, the EQSPP alignment might be off.

    self._memoryLayout = DenseMemoryLayout.fromSpp(self.eqspp(), alignStride=alignStride, alignOffset=alignOffset)

  def fixedIndexPermutation(self):
    return False

  def setIndexPermutation(self, indices, permuteEqspp=True):
    if str(indices) == str(self.indices):
      return

    p = tuple(self.indices.find(idx) for idx in indices)
    if self._eqspp is not None:
      if permuteEqspp:
        self._eqspp = self._eqspp.transposed(p)
      else:
        self._eqspp = None
    if self._memoryLayout is not None:
      self._memoryLayout = self._memoryLayout.permuted(p)
    self.indices = self.indices.permuted(indices)
  
  def __str__(self):
    return '{}[{}]'.format(type(self).__name__, self.indices if self.indices != None else '<not deduced>')
  
  def computeSparsityPattern(self, *spps):
    raise NotImplementedError

class Einsum(Op):
  def nonZeroFlops(self):
    raise NotImplementedError
    
class Add(Op):
  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      spps = [node.eqspp() for node in self]
    permute_summand = lambda i: self.broadcast(self[i].indices, self.permute(self[i].indices, spps[i], False))
    spp = permute_summand(0)
    for i in range(1, len(spps)):
      add_spp = permute_summand(i)
      spp = aspp.add(spp, add_spp)
    return spp
  
  def nonZeroFlops(self):
    nzFlops = 0
    for child in self:
      nzFlops += child.eqspp().count_nonzero()
    return nzFlops - self.eqspp().count_nonzero()

class UnaryOp(Op):
  def term(self):
    return self._children[0]

class ScalarMultiplication(UnaryOp):
  def __init__(self, scalar, term):
    super().__init__(term)
    self._isConstant = isinstance(scalar, float) or isinstance(scalar, int)
    self._scalar = float(scalar) if self._isConstant else scalar
    self.setTerm(term)

  def fixedIndexPermutation(self):
    return self.term().fixedIndexPermutation()
  
  def setTerm(self, term):
    self._children[0] = term
    if self.fixedIndexPermutation():
      self.indices = self.term().indices
    else:
      self.indices = None

  def name(self):
    return str(self._scalar) if self._isConstant else self._scalar.name()

  def is_constant(self):
    return self._isConstant

  def scalar(self):
    return self._scalar
  
  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      return self.term().eqspp()
    assert len(spps) == 1
    return spps[0]

  def nonZeroFlops(self):
    if self._isConstant and self._scalar in [-1.0, 1.0]:
      return 0
    return self.eqspp().count_nonzero()
  
  def __str__(self):
    return '{}: {}'.format(super().__str__(), str(self._scalar))

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

class Assign(Op):
  def __init__(self, lTerm, rTerm, condition=True):
    if isinstance(condition, Node):
      super().__init__(lTerm, rTerm, condition)
    else:
      super().__init__(lTerm, rTerm)
    
    self._condition = condition
  
  def leftTerm(self):
    return self._children[0]
  
  def rightTerm(self):
    return self._children[1]
  
  def condition(self):
    return self._condition
  
  def setChildren(self, children):
    if not isinstance(children[0].viewed(), IndexedTensor):
      raise ValueError('First child of Assign node must be an IndexedTensor: ' + str(children[0].viewed()))
    super().setChildren(children)
    
  def nonZeroFlops(self):
    return 0
  
  def computeSparsityPattern(self, *spps):
    spp = spps[1] if len(spps) >= 2 else self.rightTerm().eqspp()
    return self.broadcast(self.rightTerm().indices, self.permute(self.rightTerm().indices, spp, False))
  
  def __str__(self):
    selfname = type(self).__name__
    indices = self.indices if self.indices != None else '<not deduced>'
    condition = '' if isinstance(self.condition(), bool) and self.condition() else f' if {self.condition()}'
    return f'{selfname}[{indices}]: {self.leftTerm()} <- {self.rightTerm()}{condition}'

class Permute(UnaryOp):
  # permute a given tensor

  def __init__(self, term, targetIndices):
    super().__init__(term)
    self.indices = targetIndices
    assert term.indices <= self.indices and self.indices <= term.indices

  def nonZeroFlops(self):
    return 0

  def computeSparsityPattern(self, *spps):
    assert len(spps) <= 1
    spp = spps[0] if len(spps) == 1 else self.term().eqspp()
    return self.permute(self.term().indices, spp)
  
  @classmethod
  def subPermute(cls, term, indices):
    subIndexNames = [idx for idx in indices if idx in term.indices]
    subIndices = Indices(subIndexNames, term.indices.subShape(subIndexNames))
    return cls(term, subIndices)

class Broadcast(UnaryOp):
  # broadcast (i.e. copy) a tensor to some extra dimensions
  # needed for an Einstein-sum-conformant accumulator operation

  def __init__(self, term, targetIndices):
    super().__init__(term)
    self.indices = targetIndices
    assert term.indices <= self.indices

  def nonZeroFlops(self):
    return 0

  def computeSparsityPattern(self, *spps):
    assert len(spps) <= 1
    spp = spps[0] if len(spps) == 1 else self.term().eqspp()
    return self.broadcast(self.term().indices, spp)

def _productContractionLoGSparsityPattern(node, *spps):
  if len(spps) == 0:
    spps = (node.leftTerm().eqspp(), node.rightTerm().eqspp())
  assert len(spps) == 2
  einsumDescription = '{},{}->{}'.format(node.leftTerm().indices.tostring(), node.rightTerm().indices.tostring(), node.indices.tostring())
  return aspp.einsum(einsumDescription, spps[0], spps[1])

class Product(BinOp):
  def __init__(self, lTerm, rTerm):
    super().__init__(lTerm, rTerm)
    K = lTerm.indices & rTerm.indices
    assert lTerm.indices.subShape(K) == rTerm.indices.subShape(K)

    self.indices = lTerm.indices.merged(rTerm.indices - K)
  
  def nonZeroFlops(self):
    return self.eqspp().count_nonzero()
  
  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      spps = [node.eqspp() for node in self]
    assert len(spps) == 2
    return _productContractionLoGSparsityPattern(self, *spps)

class IndexSum(UnaryOp):
  def __init__(self, term, sumIndex):
    super().__init__(term)
    self.indices = term.indices - set([sumIndex])
    self._sumIndex = term.indices.extract(sumIndex)
  
  def nonZeroFlops(self):
    return self.term().eqspp().count_nonzero() - self.eqspp().count_nonzero()
  
  def sumIndex(self):
    return self._sumIndex
  
  def computeSparsityPattern(self, *spps):
    assert len(spps) <= 1
    spp = spps[0] if len(spps) == 1 else self.term().eqspp()
    return spp.indexSum(self.term().indices, self.indices)

class Contraction(BinOp):
  def __init__(self, indices, lTerm, rTerm, sumIndices):
    super().__init__(lTerm, rTerm)
    li = lTerm.indices - sumIndices
    lr = (rTerm.indices - sumIndices) - li
    self.indices = li.merged(lr)
    self.sumIndices = sumIndices
    self.setIndexPermutation(indices)

  def nonZeroFlops(self):
    raise NotImplementedError
  
  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      spps = [node.eqspp() for node in self]
    assert len(spps) == 2
    return _productContractionLoGSparsityPattern(self, *spps)

class LoopOverGEMM(BinOp):
  def __init__(self, indices, aTerm, bTerm, m, n, k):
    super().__init__(aTerm, bTerm)
    self.indices = indices
    self._m = m
    self._n = n
    self._k = k
    """ If dim(m) == 0, then A is a vector or scalar. In the case of a vector, the memory layout
        is artificially extended to be a k x 1 matrix. If A is a scalar (dim(n) == 0),
        then the memory layout is a 1 x 1 matrix.
        The same is true for B and dim(n) == 0.

        We have the following four cases:
        dim(m) == 0 and dim(n) == 0: (1 x 1) = (k x 1) * (k x 1)
            => Transpose A, do not transpose B (DOT)
        dim(m) == 0 and dim(n) != 0: (1 x n) = (k x 1) * (k x n or n x k)
            => Transpose A, transpose B if n precedes k (GEMV)
        dim(m) != 0 and dim(n) == 0: (m x 1) = (m x k or k x n) * (n x 1)
            => Transpose A if k precedes m, do not transpose B (GEMV)
        dim(m) != 0 and dim(n) != 0: (m x k) = (m x k or k x n) * (k x n or n x k)
            => Transpose A if k precedes m, transpose B if n precedes k (GEMM)
    """
    self._transA = self.hasDimensionZero(m) or aTerm.indices.find(m[0]) > aTerm.indices.find(k[0])
    self._transB = not self.hasDimensionZero(n) and bTerm.indices.find(k[0]) > bTerm.indices.find(n[0])

  @staticmethod
  def hasDimensionZero(x):
    return len(x) == 0

  def nonZeroFlops(self):
    p = Product(self.leftTerm(), self.rightTerm())
    p.setEqspp( p.computeSparsityPattern() )
    return 2*p.nonZeroFlops() - self.eqspp().count_nonzero()
  
  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      spps = [node.eqspp() for node in self]
    assert len(spps) == 2
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

  def argumentsCompatible(self, layouts):
    super().argumentsCompatible(layouts)
    m = self.leftTerm().indices.positions(self._m)
    k1 = self.leftTerm().indices.positions(self._k)
    k2 = self.rightTerm().indices.positions(self._k)
    n = self.rightTerm().indices.positions(self._n)
    return layouts[0].mayFuse(m) and layouts[0].mayFuse(k1) and layouts[1].mayFuse(k2) and layouts[1].mayFuse(n)

  def resultCompatible(self, layout):
    super().resultCompatible(layout)
    m = self.indices.positions(self._m)
    n = self.indices.positions(self._n)
    return layout.mayFuse(m) and layout.mayFuse(n)

  @staticmethod
  def indexString(name, fused, indices, transpose=False):
    indexStr = str(indices)
    batchedIndices = set(indices)
    for fs in fused:
      if len(fs) > 1:
        indexStr = re.sub(r'([{0}]{{{1},{1}}})'.format(fs, len(fs)), r'(\1)', indexStr)
      batchedIndices = batchedIndices - set(fs)
    if batchedIndices:
      indexStr = re.sub(r'([{}])'.format(''.join(batchedIndices)), r'[\1]', indexStr)
    return '{}{}_{{{}}}'.format(name, '^T' if transpose else '', indexStr)
  
  def __str__(self):
    Astr = self.indexString('A', [self._m, self._k], self.leftTerm().indices, self._transA)
    Bstr = self.indexString('B', [self._k, self._n], self.rightTerm().indices, self._transB)
    Cstr = self.indexString('C', [self._m, self._n], self.indices)
    return '{} [{}]: {} = {} {}'.format(type(self).__name__, self.indices, Cstr, Astr, Bstr)

  def is_pure_gemm(self):
    left_indices = self.leftTerm().indices
    right_indices = self.rightTerm().indices
    if not (len(left_indices) == 2 and len(right_indices) == 2):
      return False

    return True if len(left_indices - right_indices) == 1 else False

class FusedGEMMs(Op):
  def __init__(self):
    super().__init__()

  def add(self, node):
    if isinstance(node, LoopOverGEMM):
      self._children.append(node)
    else:
      raise ValueError(f'expected LoopOverGEMM, received: {type(node)}')

  def get_children(self):
    return self._children

  def get_child(self, index):
    return self._children[index]

  def nonZeroFlops(self):
    nzFlops = 0
    for child in self._children:
      nzFlops += child.nonZeroFlops()
    return nzFlops

  def is_empty(self):
    return len(self._children) == 0

class IfThenElse(Op):
  def __init__(self, condition, yesTerm, noTerm):
    if isinstance(condition, Node):
      super().__init__(yesTerm, noTerm, condition)
    else:
      super().__init__(yesTerm, noTerm)
    
    self._condition = condition
    
  def condition(self):
    return condition
  
  def nonZeroFlops(self):
    return 0
  
  def computeSparsityPattern(self, *spps):
    # TODO: yesTerm OR noTerm
    spp = spps[0] if len(spps) >= 2 else self.term().eqspp()
    return spp
  
  def __str__(self):
    indices = self.indices if self.indices != None else '<not deduced>'
    return f'{type(self).__name__}[{indices}]'

class Elementwise(Op):
  def __init__(self, optype: ops.Operation, *terms):
    nodeTerms = [term for term in terms if isinstance(term, Node)]
    super().__init__(*nodeTerms)

    self.nodeTermIndices = [None] * len(terms)
    self.termTemplate = [None] * len(terms)
    index = 0
    for i, term in enumerate(terms):
      if isinstance(term, Node):
        self.nodeTermIndices[i] = index
        index += 1
      else:
        self.nodeTermIndices[i] = None
        self.termTemplate[i] = term

    self.optype = optype
    self.terms = terms

    self.indices = Indices()
    for nodeTerm in nodeTerms:
      nodeIndices = nodeTerm.indices if nodeTerm.indices is not None else Indices()
      K = self.indices & nodeIndices
      # assert self.indices.subShape(K) == nodeTerm.subShape(K)
      self.indices = self.indices.merged(nodeIndices - K)

  def nonZeroFlops(self):
    return self.eqspp().count_nonzero()
  
  def fillTerms(self, terms):
    assert len(terms) == len(self)
    return [terms[index] if template is None else template for template, index in zip(self.termTemplate, self.nodeTermIndices)]
  
  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      spps = [node.eqspp() for node in self]
    xspp = spps[0]
    return spps[0]
  
  def __str__(self):
    indices = self.indices if self.indices != None else '<not deduced>'
    return f'{type(self).__name__}({self.optype})[{indices}]'

class Reduction(UnaryOp):
  def __init__(self, optype, term, sumIndex):
    # TODO: what if we datatype/field does not match the operation? (w.r.t. the sparsity patterns)
    super().__init__(term)
    self.indices = term.indices - set([sumIndex])
    self._reductionIndex = term.indices.extract(sumIndex)
    self.optype = optype
  
  def nonZeroFlops(self):
    return self.term().eqspp().count_nonzero() - self.eqspp().count_nonzero()
  
  def reductionIndex(self):
    return self._reductionIndex
  
  def reductionIndices(self):
    return [self._reductionIndex]
  
  def computeSparsityPattern(self, *spps):
    assert len(spps) <= 1
    spp = spps[0] if len(spps) == 1 else self.term().eqspp()
    return spp.indexSum(self.term().indices, self.indices)
  
  def __str__(self):
    indices = self.indices if self.indices != None else '<not deduced>'
    return f'{type(self).__name__}({self.optype})[{indices}]'

class Accumulate(Op):
  def __init__(self, optype, *operands):
    super().__init__(*operands)

    self.optype = optype

  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      spps = [node.eqspp() for node in self]
    permute_summand = lambda i: self.permute(self[i].indices, spps[i])
    spp = permute_summand(0)
    for i in range(1, len(spps)):
      add_spp = permute_summand(i)
      spp = aspp.add(spp, add_spp)
    return spp

  def nonZeroFlops(self):
    nzFlops = 0
    for child in self:
      nzFlops += child.eqspp().count_nonzero()
    return nzFlops - self.eqspp().count_nonzero()
