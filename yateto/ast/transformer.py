import sys
from copy import deepcopy
from typing import Union
from .visitor import Visitor, PrettyPrinter, ComputeSparsityPattern, ComputeIndexSet
from .. import ops
from .node import IndexedTensor, Op, Assign, Einsum, Reduction, Contraction, SliceView, Elementwise
from .indices import Indices
from .log import LoG
from . import opt
from .cost import ShapeCostEstimator
from .. import aspp

# Similar as ast.NodeTransformer
class Transformer(Visitor):
  def generic_visit(self, node, **kwargs):
    newChildren = [self.visit(child, **kwargs) for child in node]
    node.setChildren(newChildren)
    return node

class FoldAccumulate(Transformer):
  """Folds an n-ary accumulation into a chain of binary element-wise steps.

  A sum is left alone: it is lowered into a chain of accumulating stores, which
  is what lets a GEMM write into the result with beta = 1 rather than into a
  temporary. Every other operation has no such lowering and becomes a fold.
  """

  def visit_Accumulate(self, node):
    self.generic_visit(node)
    if node.optype == ops.Add():
      return node
    folded = node[0]
    for i in range(1, len(node)):
      folded = Elementwise(node.optype, folded, node[i])
    return folded

class DeduceIndices(Transformer):
  def __init__(self, targetIndices: Union[str, Indices] = None):
    self._targetIndices = targetIndices
    self._indexSetVisitor = ComputeIndexSet()

  def visit(self, node, bound=None):
    forceIndices = bound is None and self._targetIndices is not None
    if bound is None:
      bound = set(self._targetIndices) if self._targetIndices is not None else set()
    node = super().visit(node, bound=bound)

    if forceIndices:
      oldIndices = node.indices
      if isinstance(self._targetIndices, str):
        node.indices = node.indices.permuted(self._targetIndices)
      elif isinstance(self._targetIndices, Indices):
        node.indices = self._targetIndices
      else:
        raise ValueError(f'Target indices type ({self._targetIndices.__class__.__name__}) is not supported.')
      if not (node.indices <= oldIndices and oldIndices <= node.indices):
        raise ValueError(f'Target index dimensions do not match: {node.indices.__repr__()} != {oldIndices.__repr__()}')

    return node

  def visit_IndexedTensor(self, node, bound):
    if set(node.indices) > bound:
      free = node.indices - bound
      raise ValueError(f'The indices {free.__repr__()} are not bound in {node}.')
    return node

  def visit_Einsum(self, node, bound):
    # Computes pairwise intersection of the children's indices
    indexUnion = set()
    contractions = set()
    for child in node:
      childIndexUnion = self._indexSetVisitor.visit(child)
      contractions = contractions | (indexUnion & childIndexUnion)
      indexUnion = indexUnion | childIndexUnion

    contractions = contractions - bound

    node = self.generic_visit(node, bound=bound | contractions)

    # Check if index sizes match
    g = Indices()
    for child in node:
      overlap = g & child.indices
      if any(g.size()[index] != child.size()[index] for index in overlap):
        PrettyPrinter().visit(node)
        raise ValueError('Einsum: Index dimensions do not match: ', g, child.indices, str(child))
      g = g.merged(child.indices - overlap)

    deduced = g - contractions
    node.indices = deduced.sorted()
    return node

  def visit_Elementwise(self, node, bound):
    for child in node:
      self.visit(child, bound)
    node.deduceIndices()
    return node

  def visit_Accumulate(self, node, bound):
    for child in node:
      self.visit(child, bound)
    node.deduceIndices()
    return node

  def visit_Reduction(self, node, bound):
    # `bound` holds plain index names; sumIndexName() is one of those, whereas
    # reductionIndices() yields Indices objects
    subbound = bound | set(node.sumIndexName())
    self.visit(node.term(), subbound)
    node.deduceIndices()
    return node

  def visit_SliceView(self, node, bound):
    self.visit(node.term(), bound)
    node.indices = Indices(node.term().indices, [shape if index != node.index else (node.end - node.start) for index, shape in zip(node.term().indices, node.term().shape())])
    return node

  def visit_Assign(self, node, bound):
    lhs = node[0]
    rhs = node[1]

    lhsTensor = lhs.viewed()
    if not isinstance(lhsTensor, IndexedTensor):
      raise ValueError('Assign: Left-hand side must be of type IndexedTensor')

    # we cannot get the indices of a view directly; so we need to start at the viewed lhs tensor
    # (for tensors, we know their final indices before this transform)

    self.visit(lhs, bound=set(lhsTensor.indices))

    # now use the restricted (viewed) index ranges onto the rhs AST

    self.visit(rhs, bound=set(lhs.indices))

    node.indices = lhs.indices
    if not (rhs.indices <= lhs.indices):
      raise ValueError(f'Index dimensions do not match: {lhs.indices.__repr__()} != {rhs.indices.__repr__()}')

    return node

### Optimal binary tree

class StrengthReduction(Transformer):
  def __init__(self, costEstimator):
    self._costEstimator = costEstimator

  def visit_Einsum(self, node):
    self.generic_visit(node)
    minTree = opt.strengthReduction(list(node), node.indices, self._costEstimator())
    minTree.setIndexPermutation(node.indices)
    return minTree

class FindContractions(Transformer):
  """Folds a sum of products into a single Contraction.

  Only the (*, +) ring is folded: that is the one the GEMM backends implement.
  A reduction over any other operation -- a boolean semiring, say -- stays a
  Reduction over an Elementwise and is generated as loops, which keeps a
  non-arithmetic ring from being handed to BLAS.
  """

  @staticmethod
  def isSummation(node):
    return isinstance(node, Reduction) and node.optype == ops.Add()

  @staticmethod
  def isProduct(node):
    return isinstance(node, Elementwise) and node.optype == ops.Mul() and len(node) == 2

  def visit_Reduction(self, node):
    if not self.isSummation(node):
      return self.generic_visit(node)

    sumIndices = set(node.sumIndexName())
    child = node.term()
    while self.isSummation(child):
      sumIndices = sumIndices.union(child.sumIndexName())
      child = child.term()
    if self.isProduct(child):
      return Contraction(node.indices, self.visit(child[0]), self.visit(child[1]), sumIndices)
    return node

class SelectIndexPermutations(Transformer):
  def __init__(self, permutationVariants):
    self._permutationVariants = permutationVariants

  def generic_visit(self, node):
    variant = self._permutationVariants[node][str(node.indices)]
    choice = iter(variant._choices)
    for child in node:
      child.setIndexPermutation(next(choice))
    super().generic_visit(node)
    return node

class AssignPrefetch(Transformer):
  def __init__(self, prefetchCapabilities, prefetchTensors):
    self._assigned = set()
    self._bestMatch = dict()
    for tensor in prefetchTensors:
      tsize = tensor.memoryLayout().requiredReals()
      minDelta = sys.maxsize
      match = None
      for node, size in prefetchCapabilities.items():
        delta = abs(size - tsize)
        if delta < minDelta:
          minDelta = delta
          match = node
      self._bestMatch[node] = tensor
      del prefetchCapabilities[node]

  def generic_visit(self, node):
    if node in self._bestMatch:
      node.prefetch = self._bestMatch[node]
      self._assigned |= {self._bestMatch[node]}
    super().generic_visit(node)
    return node

  def assigned(self):
    return self._assigned

class ImplementContractions(Transformer):
  def visit_Contraction(self, node):
    self.generic_visit(node)
    newNode = LoG(node)
    newNode.setEqspp( node.eqspp() )
    newNode.computeMemoryLayout()
    return newNode

class EquivalentSparsityPattern(Transformer):
  def __init__(self, groupSpp=True):
    self._groupSpp = groupSpp

  def visit_IndexedTensor(self, node):
    node.setEqspp(node.spp(self._groupSpp).copy())
    return node

  def visit_Assign(self, node):
    self.generic_visit(node)
    node.setEqspp( node.computeSparsityPattern() )
    return node

  def visit_Elementwise(self, node):
    self.generic_visit(node)
    node.setEqspp( node.computeSparsityPattern() )
    return node

  def visit_Reduction(self, node):
    self.generic_visit(node)
    node.setEqspp( node.computeSparsityPattern() )
    return node

  def visit_Accumulate(self, node):
    self.generic_visit(node)
    node.setEqspp( node.computeSparsityPattern() )
    return node

  def visit_IfThenElse(self, node):
    self.generic_visit(node)
    node.setEqspp( node.computeSparsityPattern() )
    return node

  def getEqspp(self, terms, targetIndices):
    # Shortcut if all terms have dense eqspps
    if all(term.eqspp().is_dense() for term in terms):
      return aspp.dense(targetIndices.shape())

    minTree = opt.strengthReduction(terms, targetIndices, ShapeCostEstimator())
    if isinstance(minTree, IndexedTensor):
      return minTree.eqspp()
    minTree.setIndexPermutation(targetIndices)
    minTree = FindContractions().visit(minTree)
    return ComputeSparsityPattern(True).visit(minTree)

  def visit_Einsum(self, node):
    self.generic_visit(node)
    terms = list(node)
    node.setEqspp( self.getEqspp(terms, node.indices) )

    for child in node:
      child.setEqspp( self.getEqspp(terms, child.indices) )

    # TODO: Backtracking of equivalent sparsity pattern to children?

    return node

  def visit_SliceView(self, node):
    self.generic_visit(node)
    node.setEqspp(node.computeSparsityPattern())
    return node

class SetSparsityPattern(Transformer):
  def generic_visit(self, node):
    super().generic_visit(node)
    node.setEqspp( node.computeSparsityPattern() )
    return node

  def visit_IndexedTensor(self, node):
    return node

class ComputeMemoryLayout(Transformer):
  def generic_visit(self, node):
    super().generic_visit(node)
    node.setEqspp( node.computeSparsityPattern() )
    node.computeMemoryLayout()
    return node

  def visit_IndexedTensor(self, node):
    return node

class SetDatatype(Transformer):
  """Propagates datatypes bottom-up through the AST.

  `arch` is only needed on the first pass (before the tensors' datatypes have
  been resolved); afterwards the IndexedTensor nodes already carry their type.
  """

  def __init__(self, arch=None):
    self.arch = arch

  def _childTypes(self, node):
    return [child.datatype for child in node]

  def generic_visit(self, node):
    super().generic_visit(node)
    assert len(node) > 0, f'Cannot deduce a datatype for the childless node {node}.'
    types = self._childTypes(node)
    assert all(t == types[0] for t in types), \
      f'Mismatching operand datatypes in {node}: {[str(t) for t in types]}'
    node.datatype = types[0]
    return node

  def visit_IndexedTensor(self, node):
    super().generic_visit(node)
    if self.arch is not None:
      node.datatype = node.tensor.getDatatype(self.arch)
    return node

  def visit_Elementwise(self, node):
    super().generic_visit(node)
    node.datatype = node.optype.datatypeResult(self._childTypes(node))
    return node

  def visit_Reduction(self, node):
    super().generic_visit(node)
    node.datatype = node.optype.datatypeResult(self._childTypes(node))
    return node

  def visit_Accumulate(self, node):
    super().generic_visit(node)
    node.datatype = node.optype.datatypeResult(self._childTypes(node))
    return node

  def visit_IfThenElse(self, node):
    super().generic_visit(node)
    assert node[0].datatype == node[1].datatype, \
      f'Both branches of {node} must have the same datatype.'
    node.datatype = node[0].datatype
    return node

  def visit_Assign(self, node):
    super().generic_visit(node)
    node.datatype = node[0].datatype
    return node

# backwards-compatible aliases (SetDatatype1/2 only differed in `arch`)
SetDatatype1 = SetDatatype

def SetDatatype2():
  return SetDatatype()
