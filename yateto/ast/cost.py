from .indices import BoundingBox
from abc import ABC, abstractmethod

class CostEstimator(ABC):
  def estimate(self, node):
    childCost = 0
    for child in node:
      childCost = childCost + self.estimate(child)
    method = 'estimate_' + node.__class__.__name__
    estimator = getattr(self, method, self.generic_estimate)
    return childCost + estimator(node)

  @abstractmethod
  def generic_estimate(self, node):
    pass

class ShapeCostEstimator(CostEstimator):
  def generic_estimate(self, node):
    return 0

  def estimate_Product(self, node):
    cost = 1
    for size in node.shape():
      cost *= size
    return cost
  
  def estimate_IndexSum(self, node):
    cost = node.sumIndex().shape()[0] - 1
    for size in node.indices.shape():
      cost *= size
    return cost

class CachedCostEstimator(CostEstimator):
  def __init__(self):
    self._cost = dict()
  
  def estimate(self, node):
    if node in self._cost:
      return self._cost[node]
    cost = super().estimate(node)
    self._cost[node] = cost
    return cost

class BoundingBoxCostEstimator(CachedCostEstimator):
  def __init__(self):
    super().__init__()
    self._cache = dict()

  def generic_estimate(self, node):
    self._cache[node] = node.boundingBox()
    return 0

  def estimate_Product(self, node):
    lbb = self._cache[node.leftTerm()]
    rbb = self._cache[node.rightTerm()]
    lind = node.leftTerm().indices
    rind = node.rightTerm().indices
    ranges = list()
    for index in node.indices:
      if index in lind and index in rind:
        lpos = lind.find(index)
        rpos = rind.find(index)
        ranges.append(lbb[lpos] & rbb[rpos])
      elif index in lind:
        ranges.append(lbb[lind.find(index)])
      elif index in rind:
        ranges.append(rbb[rind.find(index)])
      else:
        raise RuntimeError('Not supposed to happen.')
    bb = BoundingBox(ranges)
    self._cache[node] = bb

    return bb.size()
  
  def estimate_IndexSum(self, node):
    tbb = self._cache[node.term()]
    pos = node.term().indices.find(str(node.sumIndex()))
    bb = BoundingBox([r for i,r in enumerate(tbb) if i != pos])
    self._cache[node] = bb
    return tbb.size() - bb.size()

class ExactCost(CachedCostEstimator):
  def __init__(self):
    super().__init__()
    self._cache = dict()

  def generic_estimate(self, node):
    self._cache[node] = node.eqspp()
    return 0
  
  def estimate_Product(self, node):
    spp = node.computeSparsityPattern(self._cache[node.leftTerm()], self._cache[node.rightTerm()])
    self._cache[node] = spp
    return spp.count_nonzero()
  
  def estimate_IndexSum(self, node):
    termSpp = self._cache[node.term()]
    spp = node.computeSparsityPattern(termSpp)
    self._cache[node] = spp    
    return termSpp.count_nonzero() - spp.count_nonzero()
