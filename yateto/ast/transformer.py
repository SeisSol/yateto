import itertools
from functools import singledispatch
from numpy import ndarray, zeros, einsum
from .visitor import Visitor, PrettyPrinter, ComputeSparsityPattern
from .node import IndexedTensor, Op, Assign, Einsum, Add, Product, IndexSum, Contraction
from .indices import Indices
from .log import LoG, Cost
from . import opt

# Similar as ast.NodeTransformer
class Transformer(Visitor): 
  def generic_visit(self, node):
    newChildren = [self.visit(child) for child in node]
    node.setChildren(newChildren)
    return node

class DeduceIndices(Transformer):    
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

### Equivalent sparsity patterns

@singledispatch
def equivalentSparsityPattern(node):
  pass

@equivalentSparsityPattern.register(IndexedTensor)  
def _(node):
  node.setEqspp(node.spp().copy())

@equivalentSparsityPattern.register(Einsum)
def _(node):
  for child in node:
    equivalentSparsityPattern(child)

  spps = [child.eqspp() for child in node]
  indices = ','.join([child.indices.tostring() for child in node])
  node.setEqspp( einsum('{}->{}'.format(indices, node.indices.tostring()), *spps, optimize=True) )
  
  for child in node:
    child.setEqspp( einsum('{}->{}'.format(indices, child.indices.tostring()), *spps, optimize=True) )
  
  # TODO: Backtracking of equivalent sparsity pattern to children?
    

@equivalentSparsityPattern.register(Add)
def _(node):
  for child in node:
    equivalentSparsityPattern(child)
  
  spp = zeros(node.indices.shape(), dtype=bool)
  for child in node:
    spp += child.eqspp()
  node.setEqspp(spp)

@equivalentSparsityPattern.register(Assign)
def _(node):
  for child in node:
    equivalentSparsityPattern(child)
  node.setEqspp(child[1].eqspp())

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

@singledispatch
def findPermutations(node, top=True):
  node._cost = {str(node.indices): (Cost(0, 0), None, None)}

@findPermutations.register(Op)
def _(node, top=True):
  for child in node:
    findPermutations(child)
  node._cost = {str(node.indices): (Cost(0, 0), None, None)}

@findPermutations.register(Contraction)
def _(node, top=True):
  findPermutations(node.leftTerm(), False)
  findPermutations(node.rightTerm(), False)
  leftVariants = node.leftTerm()._cost
  rightVariants = node.rightTerm()._cost
  
  ATFree = isinstance(node.leftTerm(), IndexedTensor)
  BTFree = isinstance(node.rightTerm(), IndexedTensor)
    
  node._cost = dict()
  iterator = [list(node.indices)] if top else itertools.permutations(node.indices)
  for Cs in iterator:
    C = ''.join(Cs)
    minCost = Cost()
    minA = None
    minB = None
    for A, Acost in leftVariants.items():
      for B, Bcost in rightVariants.items():
        cost = LoG(A, B, C, ATFree, BTFree) + Acost[0] + Bcost[0]
        if cost < minCost:
          minCost = cost
          minA = A
          minB = B
    node._cost[C] = (minCost, minA, minB)

  if top:
    selectPermutation(node, ''.join(iterator[0]))

@singledispatch
def selectPermutation(node, perm):
  pass

@selectPermutation.register(Contraction)
def _(node, perm):
  node.setIndexPermutation(perm)
  selectPermutation(node.leftTerm(),  node._cost[perm][1])
  selectPermutation(node.rightTerm(), node._cost[perm][2])

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
