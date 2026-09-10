import re
from copy import deepcopy
from ..memory import DenseMemoryLayout
from .indices import BoundingBox, Indices, LoGCost
from abc import ABC, abstractmethod
from .. import aspp
from ..type import AddressingMode, Tensor, derivedScalar
from .. import ops
import numpy as np

class Node(ABC):
  def __init__(self):
    self.indices = None
    self._children = []
    self._eqspp = None
    self._boundingBox = None
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
    # keyed on the identity of the pattern, so any replacement of _eqspp
    # (setEqspp, setIndexPermutation, deepcopy) drops the cached box; the cache
    # keeps the pattern alive, so no other object can take over its address
    cached = self._boundingBox
    if cached is not None and cached[0] is self._eqspp:
      return cached[1]
    box = BoundingBox.fromSpp(self._eqspp)
    self._boundingBox = (self._eqspp, box)
    return box

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

  @staticmethod
  def _operand(value):
    """A tensor written without indices is the rank-0 operand of that tensor.

    Which is what a scalar is: it has a shape, a layout and a name, and only
    its calling convention sets it apart. Anything else -- a number -- is a
    literal, and Elementwise carries those as templates, since a literal needs
    neither storage nor a name.
    """
    return value[''] if isinstance(value, Tensor) else value

  @staticmethod
  def _scalarOperand(value):
    """Turn a scale factor into an operand.

    A number stays a number, and a named scalar becomes a rank-0 operand.
    """
    if isinstance(value, Node):
      return value
    if isinstance(value, Tensor):
      return value['']
    return float(value)

  @staticmethod
  def _combineFactors(left, right):
    """Multiply two scale factors into one.

    Two numbers collapse right away; anything else becomes a derived scalar,
    computed once in the kernel prologue rather than in a loop.
    """
    if left is None:
      return right
    if right is None:
      return left
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
      return left * right
    return derivedScalar(ops.Mul(), left, right)

  def isScaling(self):
    """Whether this node multiplies a term by a scale factor."""
    return self.scalingOperands() is not None

  def scalingOperands(self):
    return None

  def splitScaling(self):
    """``(factor, term)`` with the scale factor peeled off; factor may be None."""
    scaling = self.scalingOperands()
    return scaling if scaling is not None else (None, self)

  def scaled(self, factor):
    """Multiply by a scale factor, collapsing with one already present."""
    mine, term = self.splitScaling()
    combined = Node._combineFactors(mine, factor)
    return Elementwise(ops.Mul(), Node._scalarOperand(combined), term)

  def _accumulate(self, other, optype):
    """Flatten chains of the same operation into one n-ary Accumulate."""
    matches = lambda node: isinstance(node, Accumulate) and node.optype == optype
    if matches(self):
      if matches(other):
        self._children.extend(other._children)
      else:
        self._children.append(other)
      return self
    elif matches(other):
      other._children.insert(0, self)
      return other
    return Accumulate(optype, self, other)

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
      return self.scaled(other)

    # peel the scale factors off both sides, multiply the tensors, and put the
    # combined factor back on top -- so a product carries at most one factor
    leftFactor, leftTerm = self.splitScaling()
    rightFactor, rightTerm = other.splitScaling()
    product = leftTerm._binOp(rightTerm, Einsum)

    factor = Node._combineFactors(leftFactor, rightFactor)
    return product if factor is None else product.scaled(factor)

  def __rmul__(self, other):
    return self.__mul__(other)

  def __add__(self, other):
    if not isinstance(other, Node):
      raise ValueError(f'Unsupported operation: Cannot add {self} to {other}.')
    return self._accumulate(other, ops.Add())

  def __radd__(self, other):
    return self.__add__(other)

  def __neg__(self):
    return self.scaled(-1.0)

  def __sub__(self, other):
    return self._accumulate(-other, ops.Add())

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

class NAryOp(Node):
  """Mixin for operations whose indices are the merge of their operands'."""

  def deduceIndices(self):
    # Indices are built once and never written to afterwards, so the merge can
    # start from the first child's object instead of a copy of it
    indices = self[0].indices
    for i in range(1, len(self)):
      indices = indices.mergeStrict(self[i].indices)
    if not all(child.indices <= indices for child in self):
      raise ValueError(f'{type(self).__name__}: Indices do not match: ',
                       *[child.indices for child in self])
    self.indices = indices
    return self.indices

  def _deduceIndicesIfPossible(self):
    # the tree is built bottom-up in some places (the contraction search) and
    # top-down in others; deduce eagerly when the children already know theirs.
    # An empty node is legal while a sum is still being assembled.
    if len(self) > 0 and all(child.indices is not None for child in self):
      self.deduceIndices()

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

class UnaryOp(Op):
  def term(self):
    return self._children[0]

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

    self._checkLeftTerm(self._children[0])
    self._condition = condition

  @staticmethod
  def _checkLeftTerm(child):
    lhs = child.viewed()
    if not isinstance(lhs, IndexedTensor):
      raise ValueError('First child of Assign node must be an IndexedTensor: ' + str(lhs))
    if lhs.tensor.isPassedByValue():
      # a by-value operand has no storage to write back into; a rank-0 tensor
      # does, and is the way to compute a scalar result inside a kernel
      raise ValueError(
        f'Cannot assign to "{lhs.name()}": it is passed by value. '
        f'Use a rank-0 tensor if you need to compute the value inside a kernel.')

  def leftTerm(self):
    return self._children[0]

  def rightTerm(self):
    return self._children[1]

  def condition(self):
    return self._condition

  def setChildren(self, children):
    self._checkLeftTerm(children[0])
    super().setChildren(children)

  def nonZeroFlops(self):
    return 0

  def computeSparsityPattern(self, *spps):
    spp = spps[1] if len(spps) >= 2 else self.rightTerm().eqspp()
    return self.broadcast(self.rightTerm().indices, self.permute(self.rightTerm().indices, spp, False))

  def __str__(self):
    selfname = type(self).__name__
    indices = self.indices if self.indices is not None else '<not deduced>'
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
    p = Elementwise(ops.Mul(), self.leftTerm(), self.rightTerm())
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

class FusedElementwise(Op):
  """Several element-wise steps sharing one loop nest.

  A step is either an operation over its operands or a scaling of one operand,
  which is the same thing seen from the control-flow graph, where a scaling is
  an action's factor rather than a node. Each step writes what a later one
  reads, and none of those intermediates leaves the nest, so they become
  locals rather than buffers. The last step produces the result.

  The node carries the steps rather than AST children: a scaling has no node
  of its own to be a child.
  """

  class Step(object):
    def __init__(self, optype=None, termTemplate=None, nodeTermIndices=None,
                 operands=1, scalar=None, add=False):
      self.optype = optype
      self.termTemplate = termTemplate
      self.nodeTermIndices = nodeTermIndices
      self.operands = operands
      self.scalar = scalar
      self.add = add
      # per operand: the index of the step that wrote it, or None
      self.sources = [None] * operands
      # a reduction walks an axis of its own inside the nest
      self.reduction = None
      # the step whose value this one accumulates into, or None where it
      # overwrites -- a step that accumulates reads what it writes, and inside
      # the nest that is a value of the loop body rather than a buffer
      self.accumulateFrom = None

    @classmethod
    def fromElementwise(cls, node, scalar=None):
      return cls(optype=node.optype, termTemplate=node.termTemplate,
                 nodeTermIndices=node.nodeTermIndices, operands=len(node),
                 scalar=scalar)

    @classmethod
    def scaling(cls, scalar, add):
      return cls(operands=1, scalar=scalar, add=add)

    @classmethod
    def fromReduction(cls, node):
      step = cls(optype=node.optype, operands=1)
      step.reduction = node
      return step

    def fillTerms(self, args):
      if self.optype is None:
        return args
      return [args[index] if template is None else template
              for index, template in zip(self.nodeTermIndices, self.termTemplate)]

  def __init__(self, steps, indices):
    super().__init__()
    self.steps = steps
    self.indices = indices

  def stepScalars(self):
    """The factors the steps carry, which are the kernel's to declare."""
    return [step.scalar for step in self.steps if step.scalar is not None]

  def nonZeroFlops(self):
    return self.eqspp().count_nonzero() * len(self.steps)

  def computeSparsityPattern(self, *spps):
    return self.eqspp()

  def __str__(self):
    what = ', '.join(str(step.optype) if step.optype is not None else 'scale'
                     for step in self.steps)
    return f'{type(self).__name__}[{self.indices}]({what})'

class IfThenElse(Op):
  def __init__(self, condition, yesTerm, noTerm):
    if isinstance(condition, Node):
      super().__init__(yesTerm, noTerm, condition)
    else:
      super().__init__(yesTerm, noTerm)

    self._condition = condition

  def yesTerm(self):
    return self._children[0]

  def noTerm(self):
    return self._children[1]

  def condition(self):
    return self._condition

  def nonZeroFlops(self):
    return 0

  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      spps = [child.eqspp() for child in self]
    # either branch may be taken, so over-approximate with their union
    permuted = [self.permute(self[i].indices, spps[i]) for i in range(2)]
    return aspp.add(permuted[0], permuted[1])

  def __str__(self):
    indices = self.indices if self.indices is not None else '<not deduced>'
    return f'{type(self).__name__}[{indices}]'

class Elementwise(NAryOp, Op):
  def __init__(self, optype: ops.Operation, *terms):
    optype.checkArity(len(terms))

    # A tensor handed over without indices is an operand, not a template: it
    # has a name the kernel has to declare and a value the caller sets, and
    # writing it into the expression as if it were a literal leaves the
    # generated code naming something that was never declared.
    terms = tuple(Node._operand(term) for term in terms)

    nodeTerms = [term for term in terms if isinstance(term, Node)]
    if len(nodeTerms) == 0:
      raise ValueError('Elementwise needs at least one tensor-valued operand.')
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
    self._deduceIndicesIfPossible()

    # The indices are deduced by DeduceIndices, which is the first point at
    # which the children's indices are guaranteed to be known.

  @property
  def terms(self):
    """The operands in their original order, derived from the children.

    Kept derived rather than stored: a transformer may replace a child (an
    Einsum becomes a contraction tree, for instance), and a parallel list would
    go stale the moment it does.
    """
    return tuple(self._children[index] if template is None else template
                 for template, index in zip(self.termTemplate, self.nodeTermIndices))

  def nonZeroFlops(self):
    scaling = self.scalingOperands()
    if scaling is not None and scaling[0] in (-1.0, 1.0):
      return 0
    return self.eqspp().count_nonzero()

  def scalingOperands(self):
    """``(factor, term)`` if this is a multiplication by a scale factor.

    The factor is either a number or a by-value rank-0 tensor. Such a product is
    lowered into the scale factor of a single program action rather than into a
    loop, which is what lets it fold into a GEMM's alpha.
    """
    if self.optype != ops.Mul() or len(self.terms) != 2:
      return None
    for i, j in ((0, 1), (1, 0)):
      candidate = self.terms[i]
      if isinstance(candidate, (int, float)):
        return candidate, self.terms[j]
      if isinstance(candidate, IndexedTensor) and candidate.tensor.isPassedByValue():
        return candidate.tensor, self.terms[j]
    return None

  def scaledTerm(self):
    return self.scalingOperands()[1]

  def setScaledTerm(self, term):
    _, old = self.scalingOperands()
    self._children[self._children.index(old)] = term
    self.indices = None
    self._deduceIndicesIfPossible()
    return self

  def fillTerms(self, terms):
    assert len(terms) == len(self)
    return [terms[index] if template is None else template for template, index in zip(self.termTemplate, self.nodeTermIndices)]

  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      spps = [child.eqspp() for child in self]
    # bring every operand into this node's index order and shape first, so the
    # operation only has to combine patterns of equal shape
    aligned = [self.broadcast(self[i].indices, self.permute(self[i].indices, spps[i], strict=False))
               for i in range(len(spps))]
    return self.optype.sparsityResult(aligned)

  def __str__(self):
    indices = self.indices if self.indices is not None else '<not deduced>'
    return f'{type(self).__name__}({self.optype})[{indices}]'

class Reduction(UnaryOp):
  def __init__(self, optype, term, sumIndex):
    super().__init__(term)
    # term.indices may still be None here (e.g. for an Add/Einsum child); in
    # that case DeduceIndices.visit_Reduction computes them later.
    self._sumIndexName = str(sumIndex)
    self._reductionIndex = None
    if term.indices is not None:
      self.deduceIndices()
    self.optype = optype

  def deduceIndices(self):
    term = self.term()
    self.indices = term.indices - set(self._sumIndexName)
    self._reductionIndex = term.indices.extract(self._sumIndexName)
    return self.indices

  def nonZeroFlops(self):
    return self.term().eqspp().count_nonzero() - self.eqspp().count_nonzero()

  def sumIndexName(self):
    """The reduced index, as a plain single-character name."""
    return self._sumIndexName

  def reductionIndex(self):
    return self._reductionIndex

  def reductionIndices(self):
    return [self._reductionIndex]

  def computeSparsityPattern(self, *spps):
    assert len(spps) <= 1
    spp = spps[0] if len(spps) == 1 else self.term().eqspp()
    if not self.optype.preservesZero():
      # folding an all-zero slice need not give zero, so nothing can be ruled out
      return aspp.dense(self.indices.shape())
    # the fold is over slices that are all zero unless one of them is not, which
    # is what the union along the reduced axis says
    return spp.indexSum(self.term().indices, self.indices)

  def __str__(self):
    indices = self.indices if self.indices is not None else '<not deduced>'
    return f'{type(self).__name__}({self.optype})[{indices}]'

class Accumulate(NAryOp, Op):
  def __init__(self, optype, *operands):
    super().__init__(*[Node._operand(operand) for operand in operands])

    self.optype = optype
    self._deduceIndicesIfPossible()

  def computeSparsityPattern(self, *spps):
    if len(spps) == 0:
      spps = [child.eqspp() for child in self]
    aligned = [self.broadcast(self[i].indices, self.permute(self[i].indices, spps[i], strict=False))
               for i in range(len(spps))]
    return self.optype.sparsityResult(aligned)

  def nonZeroFlops(self):
    nzFlops = 0
    for child in self:
      permuted = self.broadcast(child.indices, self.permute(child.indices, child.eqspp(), False))
      nzFlops += permuted.count_nonzero()

    # ignore all first adds against zero (i.e. those in self.eqspp())
    return nzFlops - self.eqspp().count_nonzero()

  def __str__(self):
    indices = self.indices if self.indices is not None else '<not deduced>'
    return f'{type(self).__name__}({self.optype})[{indices}]'
