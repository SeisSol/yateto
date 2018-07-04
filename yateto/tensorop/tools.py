from functools import singledispatch
import sys

class Term(object):
  def __init__(self, name, indices):
    self._name = name
    self._indices = set(indices)
    self._cost = 0

class MulTerm(Term):
  def __init__(self, lTerm, rTerm, index_dimension):
    super().__init__('f_m', lTerm._indices | rTerm._indices)
    self._lTerm = lTerm
    self._rTerm = rTerm
    self._cost = 1
    for index in self._indices:
      self._cost *= index_dimension[index]
    self._cost += lTerm._cost + rTerm._cost
    
  
class AddTerm(Term):
  def __init__(self, term, sumIndex, index_dimension):
    super().__init__('f_a', term._indices - set([sumIndex]))
    self._term = term
    self._sumIndex = sumIndex
    self._cost = index_dimension[sumIndex]
    for index in self._indices:
      self._cost *= index_dimension[index]
    self._cost += term._cost

@singledispatch
def pprint(node, indent=''):
  print(indent + node._name)

@pprint.register(MulTerm)
def _(node, indent=''):
  print(indent + 'x: {}'.format(node._cost))
  pprint(node._lTerm, indent + '  ')
  pprint(node._rTerm, indent + '  ')

@pprint.register(AddTerm)
def _(node, indent=''):
  print(indent + 'sum_{}: {}'.format(node._sumIndex, node._cost))
  pprint(node._term, indent + '  ')

def opMin(terms, index_dimension, target_indices, split = 0):
  n = len(terms)
  
  indexList = [index for term in terms for index in term._indices]
  uniqueIndices = set(indexList)
  summationIndices = set([index for index in uniqueIndices if indexList.count(index) == 1]) - target_indices
  
  while len(summationIndices) != 0:
    for i in range(split,n):
      intersection = summationIndices & terms[i]._indices
      if len(intersection) > 0:
        index = next(iter(intersection))
        addTerm = AddTerm(terms[i], index, index_dimension)
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
        mulTerm = MulTerm(terms[i], terms[j], index_dimension)
        selection = set(range(n)) - set([i,j])
        tree = opMin([terms[i] for i in selection] + [mulTerm], index_dimension, target_indices, j-1)
        possibilites.append(tree)
  best = None
  minCost = sys.maxsize
  for p in possibilites:
    if p._cost < minCost:
      best = p
      minCost = p._cost
  return best

def optimalBinaryTree(contract):
  terms = [Term(str(child), child.indices) for child in contract]
  index_dimension = dict()
  for child in contract:
    index_dimension.update(child.indices.size())
  target_indices = set(contract.indices)
  pprint( opMin(terms, index_dimension, target_indices) )
