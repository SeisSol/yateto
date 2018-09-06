import sys
from .node import IndexSum, Product

def strengthReduction(terms, target_indices, split = 0):
  n = len(terms)
  
  indexList = [index for term in terms for index in term.indices]
  uniqueIndices = set(indexList)
  summationIndices = set([index for index in uniqueIndices if indexList.count(index) == 1]) - set(target_indices)
  
  while len(summationIndices) != 0:
    i = split
    while i < n:
      intersection = summationIndices & terms[i].indices
      if len(intersection) > 0:
        index = next(iter(intersection))
        addTerm = IndexSum(terms[i], index)
        selection = set(range(n)) - set([i])
        terms = [terms[i] for i in selection] + [addTerm]
        summationIndices -= set([index])
      else:
        i = i + 1

  possibilites = list()
  if n == 1:
    return terms[0]
  else:
    for i in range(n):
      for j in range(max(i+1,split),n):
        mulTerm = Product(terms[i], terms[j])
        selection = set(range(n)) - set([i,j])
        tree = strengthReduction([terms[i] for i in selection] + [mulTerm], target_indices, j-1)
        possibilites.append(tree)
  best = None
  minCost = sys.maxsize
  for p in possibilites:
    if p._cost < minCost:
      best = p
      minCost = p._cost
  return best
