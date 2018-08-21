import sys
import itertools
from functools import singledispatch
from numpy import zeros, einsum
from .node import IndexedTensor, Op, Assign, Einsum, Add, Product, IndexSum, Contraction
from .indices import Indices
from .log import LoG, Cost

### Simplify

@singledispatch
def simplify(node):
  pass

@simplify.register(Op)
def _(node):
  newChildren = []
  for child in node:
    simplify(child)
    if isinstance(child, type(node)):
      newChildren.extend(list(child))
    else:
      newChildren.append(child)
  node.setChildren(newChildren)

@singledispatch
def evaluate(node):
  pass

### Evaluate
  
@evaluate.register(Op)
def _(node):
  raise NotImplementedError()

@evaluate.register(Einsum)
def _(node):
  for child in node:
    evaluate(child)
  
  g = Indices()
  contractions = set()
  for child in node:
    overlap = g & child.indices
    if any([g.size()[index] != child.size()[index] for index in overlap]):
      pprint(node)
      raise ValueError('Einsum: Index dimensions do not match: ', g, child.indices, str(child))
    g = g.merged(child.indices - overlap)
    contractions.update(overlap)

  deduced = g - contractions
  if node.indices == None:
    node.indices = deduced.sorted()
  elif not node.indices <= deduced:
    raise ValueError('Einsum: Indices are not contained in deduced indices or sizes do not match. [{} not contained in {}]'.format(node.indices.__repr__(), deduced.__repr__()))

@evaluate.register(Add)
def _(node):
  for child in node:
    if isinstance(child, Op):
      child.indices = node.indices
    evaluate(child)

  ok = all([node[0].indices == child.indices for child in node])
  if not ok:
    raise ValueError('Add: Indices do not match: ', *[child.indices for child in node])

  if node.indices == None:
    node.indices = node[0].indices
  elif node.indices != node[0].indices:
    raise ValueError('Add: {} is not a equal to {}'.format(node.indices.__repr__(), node[0].indices.__repr__()))

@evaluate.register(Assign)
def _(node):
  lhs = node[0]
  rhs = node[1]
  
  if not isinstance(lhs, IndexedTensor):
    raise ValueError('Assign: Left-hand side must be of type IndexedTensor')

  node.indices = lhs.indices

  if isinstance(rhs, Op):
    rhs.indices = node.indices
  elif rhs.indices != node.indices:
    raise ValueError('Assign: Index dimensions do not match: {} != {}'.format(node.indices.__repr__(), rhs.indices.__repr__()))
  evaluate(rhs)

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

@singledispatch
def strengthReduction(node):
  return node

@strengthReduction.register(Op)
def _(node):
  newChildren = [strengthReduction(child) for child in node]
  node.setChildren(newChildren)
  return node

def opMin(terms, target_indices, split = 0):
  n = len(terms)
  
  indexList = [index for term in terms for index in term.indices]
  uniqueIndices = set(indexList)
  summationIndices = set([index for index in uniqueIndices if indexList.count(index) == 1]) - set(target_indices)
  
  while len(summationIndices) != 0:
    for i in range(split,n):
      intersection = summationIndices & terms[i].indices
      if len(intersection) > 0:
        index = next(iter(intersection))
        addTerm = IndexSum(terms[i], index)
        selection = set(range(n)) - set([i])
        terms = [terms[i] for i in selection] + [addTerm]
        split = i
        summationIndices -= set([index])

  possibilites = list()
  if n == 1:
    return terms[0]
  else:
    for i in range(n):
      for j in range(max(i+1,split),n):
        mulTerm = Product(terms[i], terms[j], target_indices)
        selection = set(range(n)) - set([i,j])
        tree = opMin([terms[i] for i in selection] + [mulTerm], target_indices, j-1)
        possibilites.append(tree)
  best = None
  minCost = sys.maxsize
  for p in possibilites:
    if p._cost < minCost:
      best = p
      minCost = p._cost
  return best

@strengthReduction.register(Einsum)
def _(node):
  newChildren = [strengthReduction(child) for child in node]
  minTree = opMin(newChildren, node.indices)
  minTree.setIndexPermutation(node.indices)
  return minTree
  
@singledispatch
def findContractions(node):
  return node

@findContractions.register(Op)
def _(node):
  newChildren = [findContractions(child) for child in node]
  node.setChildren(newChildren)
  return node

@findContractions.register(IndexSum)
def _(node):
  sumIndices = set([node.sumIndex()])
  child = node.term()
  while isinstance(child, IndexSum):
    sumIndices = sumIndices.union(child.sumIndex())
    child = child.term()
  if isinstance(child, Product):
    newNode = Contraction(node.indices, findContractions(child.leftTerm()), findContractions(child.rightTerm()), sumIndices)
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
  
