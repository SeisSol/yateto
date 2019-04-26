import sys
from .visitor import Visitor, PrettyPrinter, ComputeSparsityPattern
from .node import IndexedTensor, Op, Assign, Einsum, Add, Product, IndexSum, Contraction
from .indices import Indices
from .log import LoG
from . import opt
from .cost import ShapeCostEstimator
from .. import aspp

# Similar as ast.NodeTransformer
class Transformer(Visitor): 
  def generic_visit(self, node):
    newChildren = [self.visit(child) for child in node]
    node.setChildren(newChildren)
    return node

class DeduceIndices(Transformer):
  def __init__(self, targetIndices=None):
    self._targetIndices = targetIndices
  
  def visit(self, node):
    if self._targetIndices:
      if isinstance(node, Einsum) or isinstance(node, Add):
        node.indices = self._targetIndices
      else:
        raise ValueError('Setting target indices in DeduceIndices is only allowed if the root node is of type Add or Einsum.')
      self._targetIndices = None
    return super().visit(node)

  def visit_Einsum(self, node):
    self.generic_visit(node)

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
    if node.indices == None:
      node.indices = deduced.sorted()
    elif not node.indices <= g:
      raise ValueError('Einsum: Indices are not contained in deduced indices or sizes do not match. [{} not contained in {}]'.format(node.indices.__repr__(), deduced.__repr__()))
    return node
  
  def visit_Add(self, node):
    if node.indices == None:
      for child in node:
        if child.fixedIndexPermutation():
          node.indices = child.indices
          break

    for child in node:
      if not child.fixedIndexPermutation():
        child.indices = node.indices
      self.visit(child)

    ok = all([node[0].indices == child.indices for child in node])
    if not ok:
      raise ValueError('Add: Indices do not match: ', *[child.indices for child in node])

    if node.indices == None:
      node.indices = node[0].indices
    elif node.indices != node[0].indices:
      raise ValueError('Add: {} is not a equal to {}'.format(node.indices.__repr__(), node[0].indices.__repr__()))
    return node
  
  def _setSingleChildIndices(self, node, term):
    if term.indices != node.indices:
      if term.indices is None or (not term.fixedIndexPermutation() and term.indices <= node.indices and node.indices <= term.indices):
        term.indices = node.indices
      else:
        raise ValueError('Index dimensions do not match: {} != {}'.format(node.indices.__repr__(), term.indices.__repr__()))
  
  def visit_ScalarMultiplication(self, node):
    if node.indices is not None:
      self._setSingleChildIndices(node, node.term())
    self.visit(node.term())
    if node.indices is None:
      node.indices = node.term().indices
    return node

  def visit_Assign(self, node):
    lhs = node[0]
    rhs = node[1]
    
    if not isinstance(lhs, IndexedTensor):
      raise ValueError('Assign: Left-hand side must be of type IndexedTensor')

    node.indices = lhs.indices

    self._setSingleChildIndices(node, rhs)
    self.visit(rhs)
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
    assert isinstance(node[1].eqspp(), aspp.ASpp)
    node.setEqspp(node[1].eqspp())
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
