from numpy import ndarray, zeros, einsum
from .visitor import Visitor, PrettyPrinter, ComputeSparsityPattern
from .node import IndexedTensor, Op, Assign, Einsum, Add, Product, IndexSum, Contraction
from .indices import Indices
from .log import LoG
from . import opt
from .cost import ShapeCostEstimator

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
    elif not node.indices <= deduced:
      raise ValueError('Einsum: Indices are not contained in deduced indices or sizes do not match. [{} not contained in {}]'.format(node.indices.__repr__(), deduced.__repr__()))
    return node
  
  def visit_Add(self, node):
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

  def visit_Assign(self, node):
    lhs = node[0]
    rhs = node[1]
    
    if not isinstance(lhs, IndexedTensor):
      raise ValueError('Assign: Left-hand side must be of type IndexedTensor')

    node.indices = lhs.indices

    if isinstance(rhs, Op):
      rhs.indices = node.indices
    elif rhs.indices != node.indices:
      raise ValueError('Assign: Index dimensions do not match: {} != {}'.format(node.indices.__repr__(), rhs.indices.__repr__()))
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

class ImplementContractions(Transformer):
  def visit_Contraction(self, node):
    self.generic_visit(node)
    newNode = LoG(node)
    newNode.setEqspp( node.eqspp() )
    newNode.computeMemoryLayout()
    return newNode

class EquivalentSparsityPattern(Transformer):
  def visit_IndexedTensor(self, node):
    node.setEqspp(node.spp().copy())
    return node

  def visit_Add(self, node):
    self.generic_visit(node)
    
    spp = zeros(node.indices.shape(), dtype=bool)
    for child in node:
      assert isinstance(child.eqspp(), ndarray)
      spp += child.eqspp()
    node.setEqspp(spp)
    return node
  
  def visit_Assign(self, node):
    self.generic_visit(node)
    assert isinstance(node[1].eqspp(), ndarray)
    node.setEqspp(node[1].eqspp())
    return node
  
  def getEqspp(self, terms, targetIndices):
    minTree = opt.strengthReduction(terms, targetIndices, ShapeCostEstimator())
    if isinstance(minTree, IndexedTensor):
      return minTree.eqspp()
    minTree.setIndexPermutation(targetIndices)
    minTree = FindContractions().visit(minTree)
    return ComputeSparsityPattern().visit(minTree)
  
  def visit_Einsum(self, node):
    self.generic_visit(node)
    terms = list(node)
    node.setEqspp( self.getEqspp(terms, node.indices) )
    
    for child in node:
      child.setEqspp( self.getEqspp(terms, child.indices) )
      
    # TODO: Backtracking of equivalent sparsity pattern to children?

    return node

class ComputeMemoryLayout(Transformer):
  def generic_visit(self, node):
    super().generic_visit(node)
    node.setEqspp( node.computeSparsityPattern() )
    node.computeMemoryLayout()
    return node
  
  def visit_IndexedTensor(self, node):
    return node
