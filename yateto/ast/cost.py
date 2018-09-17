from numpy import count_nonzero

class CostEstimator(object):
  def estimate(self, node):
    method = 'estimate_' + node.__class__.__name__
    estimator = getattr(self, method, self.generic_estimate)
    return estimator(node)
  
  def generic_estimate(self, node):
    raise NotImplementedError

class ShapeCostEstimator(CostEstimator):    
  def estimate_Add(self, node):
    return 0
    
  def estimate_IndexedTensor(self, node):
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

class ExactCost(CachedCostEstimator):
  def __init__(self):
    super().__init__()
    self._cache = dict()

  def estimate_Add(self, node):
    self._cache[node] = node.eqspp()
    return 0
    
  def estimate_IndexedTensor(self, node):
    self._cache[node] = node.eqspp()
    return 0
  
  def estimate_Product(self, node):
    childCost = self.estimate(node.leftTerm()) + self.estimate(node.rightTerm())
    spp = node.computeSparsityPattern(self._cache[node.leftTerm()], self._cache[node.rightTerm()])
    self._cache[node] = spp
    return childCost + count_nonzero( spp )
  
  def estimate_IndexSum(self, node):
    childCost = self.estimate(node.term())
    termSpp = self._cache[node.term()]
    spp = node.computeSparsityPattern(termSpp)
    self._cache[node] = spp    
    return childCost + count_nonzero( termSpp ) - count_nonzero( spp )
