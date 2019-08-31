import sys
from copy import deepcopy
from typing import Union
from .visitor import Visitor, PrettyPrinter, ComputeSparsityPattern
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
    self._targetIndices = None
    if isinstance(targetIndices, str):
      self._targetIndices = lambda indices: indices.permuted(targetIndices)
    elif isinstance(targetIndices, Indices):
      self._targetIndices = lambda indices: targetIndices
  
  def visit(self, node, root=True):
    if self._targetIndices and root:
      if not (isinstance(node, Einsum) or isinstance(node, Add) or isinstance(node, ScalarMultiplication)):
        raise ValueError('Setting target indices in DeduceIndices is only allowed if the root node is of type Add, Einsum, or ScalarMultiplication.')
    return super().visit(node, root=root)

  def visit_Einsum(self, node, root):
    self.generic_visit(node, root=False)

    g = Indices()
    contractions = set()
    for child in node:
      overlap = g & child.indices
      if any([g.size()[index] != child.size()[index] for index in overlap]):
        PrettyPrinter().visit(node)
        raise ValueError('Einsum: Index dimensions do not match: ', g, child.indices, str(child))
      g = g.merged(child.indices - overlap)
      contractions.update(overlap)

    deduced = g - contractions
    if self._targetIndices and root:
      node.indices = self._targetIndices(deduced)
    if node.indices == None:
      node.indices = deduced.sorted()
    elif not node.indices <= g:
      raise ValueError('Einsum: Indices are not contained in deduced indices or sizes do not match. [{} not contained in {}]'.format(node.indices.__repr__(), deduced.__repr__()))
    return node
  
  def visit_Add(self, node, root):
    if self._targetIndices and root:
      node.indices = self._targetIndices(node[0].indices)

    for child in node:
      self._setSingleChildIndices(node, child, True)
      self.visit(child, root=False)

    ok = all([node[0].indices <= child.indices and child.indices <= node[0].indices for child in node])
    if not ok:
      raise ValueError('Add: Indices do not match: ', *[child.indices for child in node])

    if node.indices == None:
      node.indices = deepcopy(node[0].indices)
    if node.indices == None:
      node.indices = node[0].indices
    elif not (node.indices <= node[0].indices and node[0].indices <= node.indices):
      raise ValueError('Add: {} is not a equal to {}'.format(node.indices.__repr__(), node[0].indices.__repr__()))
    return node
  
  def _setSingleChildIndices(self, node, term, allowPermutation):
    if term.indices != node.indices:
      mayNotPermute = not allowPermutation and term.fixedIndexPermutation()
      if term.indices is None:
        term.indices = node.indices
      elif mayNotPermute or not (term.indices <= node.indices and node.indices <= term.indices):
        raise ValueError('Index dimensions do not match: {} != {}'.format(node.indices.__repr__(), term.indices.__repr__()))
  
  def visit_ScalarMultiplication(self, node, root):
    if node.indices is not None:
      self._setSingleChildIndices(node, node.term(), False)
    self.visit(node.term(), root)
    if node.indices is None:
      node.indices = node.term().indices
    return node

  def visit_Assign(self, node, root):
    lhs = node[0]
    rhs = node[1]
    
    if not isinstance(lhs, IndexedTensor):
      raise ValueError('Assign: Left-hand side must be of type IndexedTensor')

    node.indices = lhs.indices

    self._setSingleChildIndices(node, rhs, True)
    self.visit(rhs, root=False)

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
