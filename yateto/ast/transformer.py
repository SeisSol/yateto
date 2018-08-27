import itertools
from numpy import ndarray, zeros, einsum
from .visitor import Visitor, PrettyPrinter, ComputeSparsityPattern
from .node import IndexedTensor, Op, Assign, Einsum, Add, Product, IndexSum, Contraction
from .indices import Indices, LoGCost
from .log import LoG
from . import opt

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
      if isinstance(child, Op):
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
  def visit_Einsum(self, node):
    self.generic_visit(node)
    minTree = opt.strengthReduction(list(node), node.indices)
    minTree.setIndexPermutation(node.indices)
    return minTree

class FindContractions(Transformer):
  def visit_IndexSum(self, node):
    sumIndices = set([node.sumIndex()])
    child = node.term()
    while isinstance(child, IndexSum):
      sumIndices = sumIndices.union(child.sumIndex())
      child = child.term()
    if isinstance(child, Product):
      newNode = Contraction(node.indices, self.visit(child.leftTerm()), self.visit(child.rightTerm()), sumIndices)
      return newNode
    return node

class FindIndexPermutations(Transformer):
  class Variant(object):
    def __init__(self, cost, choices):
      self._cost = cost
      self._choices = choices
    
  def generic_visit(self, node):
    super().generic_visit(node)
    choices = list()
    for child in node:
      choices.append( str(child.indices) )
    variants = {str(node.indices): self.Variant(LoGCost.addIdentity(), choices)}
    setattr(node, '_findIndexPermutationsVariants', variants) 
    return node
  
  def allPermutations(self, node):
    super().generic_visit(node)
    choices = list()
    for child in node:
      choices.append( str(child.indices) )
    iterator = itertools.permutations(node.indices)
    variants = {''.join(Cs): self.Variant(LoGCost.addIdentity(), choices) for Cs in iterator}
    setattr(node, '_findIndexPermutationsVariants', variants) 
    return node
  
  def visit_Product(self, node):
    return self.allPermutations(node)
    
  def visit_IndexSum(self, node):
    return self.allPermutations(node)

  def visit_Contraction(self, node):
    node.setChildren([self.visit(node.leftTerm()), self.visit(node.rightTerm())])
    
    variants = dict()
    iterator = itertools.permutations(node.indices)
    for Cs in iterator:
      C = ''.join(Cs)
      minCost = LoGCost()
      minAind = None
      minBind = None
      lV = node.leftTerm()._findIndexPermutationsVariants
      rV = node.rightTerm()._findIndexPermutationsVariants
      for Aind in sorted(lV):
        for Bind in sorted(rV):
          log = LoG(node, Aind, Bind, C)
          cost = log.cost() + lV[Aind]._cost + rV[Bind]._cost
          if cost < minCost:
            minCost = cost
            minAind = Aind
            minBind = Bind
      if minAind is not None and minBind is not None:
        variants[C] = self.Variant(minCost, [minAind, minBind])
    assert variants, 'Could not find implementation for Contraction. (Note: Matrix-Vector multiplication currently not supported.)'
    setattr(node, '_findIndexPermutationsVariants', variants)
    return node

class SelectIndexPermutations(Transformer):
  def generic_visit(self, node):
    variant = node._findIndexPermutationsVariants[str(node.indices)]
    choice = iter(variant._choices)
    for child in node:
      child.setIndexPermutation(next(choice))
    super().generic_visit(node)
    return node

class ImplementContractions(Transformer):
  def visit_Contraction(self, node):
    self.generic_visit(node)
    newNode = LoG(node)
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
    minTree = opt.strengthReduction(terms, targetIndices)
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

class ComputeAndSetSparsityPattern(Transformer):
  def generic_visit(self, node):
    super().generic_visit(node)
    node.setEqspp( node.computeSparsityPattern() )
    return node
  
  def visit_IndexedTensor(self, node):
    return node
