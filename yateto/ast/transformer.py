import sys
from copy import deepcopy
from typing import Union
from .visitor import Visitor, PrettyPrinter, ComputeSparsityPattern, ComputeIndexSet
from .node import IndexedTensor, Op, Assign, Einsum, Add, Product, IndexSum, Contraction, ScalarMultiplication
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
        raise ValueError('Target indices type ({}) is not supported.'.format(self._targetIndices.__class__.__name__))
      if not (node.indices <= oldIndices and oldIndices <= node.indices):
        raise ValueError('Target index dimensions do not match: {} != {}'.format(node.indices.__repr__(), oldIndices.__repr__()))

    return node

  def visit_IndexedTensor(self, node, bound):
    if set(node.indices) > bound:
      free = node.indices - bound
      raise ValueError('The indices {} are not bound in {}.'.format(free.__repr__(), node))
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
      if any([g.size()[index] != child.size()[index] for index in overlap]):
        PrettyPrinter().visit(node)
        raise ValueError('Einsum: Index dimensions do not match: ', g, child.indices, str(child))
      g = g.merged(child.indices - overlap)

    deduced = g - contractions
    node.indices = deduced.sorted()
    return node
  
  def visit_Add(self, node, bound):
    for child in node:
      self.visit(child, bound)

    ok = all([node[0].indices <= child.indices and child.indices <= node[0].indices for child in node])
    if not ok:
      raise ValueError('Add: Indices do not match: ', *[child.indices for child in node])

    node.indices = deepcopy(node[0].indices)
    return node

  def visit_ScalarMultiplication(self, node, bound):
    self.visit(node.term(), bound)
    node.indices = deepcopy(node.term().indices)
    return node

  def visit_Assign(self, node, bound):
    lhs = node[0]
    rhs = node[1]
    
    if not isinstance(lhs, IndexedTensor):
      raise ValueError('Assign: Left-hand side must be of type IndexedTensor')

    self.visit(rhs, bound=set(lhs.indices))

    node.indices = lhs.indices
    if not (lhs.indices <= rhs.indices and lhs.indices <= rhs.indices):
      raise ValueError('Index dimensions do not match: {} != {}'.format(lhs.indices.__repr__(), rhs.indices.__repr__()))

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
  def visit_IndexSum(self, node):
    sumIndices = set(node.sumIndex())
    child = node.term()
    while isinstance(child, IndexSum):
      sumIndices = sumIndices.union(child.sumIndex())
      child = child.term()
    if isinstance(child, Product):
      newNode = Contraction(node.indices, self.visit(child.leftTerm()), self.visit(child.rightTerm()), sumIndices)
      return newNode
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

  def visit_Add(self, node):
    self.generic_visit(node)
    node.setEqspp( node.computeSparsityPattern() )
    return node

  def visit_ScalarMultiplication(self, node):
    self.generic_visit(node)
    node.setEqspp(node.term().eqspp())
    return node
  
  def visit_Assign(self, node):
    self.generic_visit(node)
    node.setEqspp( node.computeSparsityPattern() )
    return node
  
  def getEqspp(self, terms, targetIndices):
    # Shortcut if all terms have dense eqspps
    if all([term.eqspp().is_dense() for term in terms]):
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
