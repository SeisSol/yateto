from .indices import BoundingBox
from .node import IndexSum
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


class FusedGemmsBoundingBoxCostEstimator(BoundingBoxCostEstimator):
  """Estimates num. of hardware flops for a tensor operation per GPU thread.
  Therefore, results of BoundingBoxCostEstimator are divided by a size
  of the first dimension of lhs because this dimension is fully parallelized.
  Note, the estimator includes GPU caching. This estimator is relevant to
  fused gemms kernels.
  """
  def __init__(self):
    super().__init__()
    self._lead_dim = 0
    self._loaded_to_gpu_cache = {}
  
  def generic_estimate(self, node):
    result = super().generic_estimate(node)
    self._loaded_to_gpu_cache[node] = set()
    return result

  def _get_terms(self, node):
    left_indices = node.leftTerm().indices
    right_indices = node.rightTerm().indices
    common_indices = left_indices & right_indices

    if left_indices[0] in common_indices:
      # swap terms becase we do not allow
      # tensor product along the leading dimension.
      # In other words, LoG will try to swap terms
      # in the future
      return node.rightTerm(), node.leftTerm()
    else:
      return node.leftTerm(), node.rightTerm()

  def estimate_Product(self, node):
    cost = super().estimate_Product(node)
    left_term, right_term = self._get_terms(node)

    bb = self._cache[left_term]
    cost /= bb[self._lead_dim].size()

    # take the union of all cached nodes
    self._loaded_to_gpu_cache[node] = self._loaded_to_gpu_cache[left_term].union(self._loaded_to_gpu_cache[right_term])

    extra_cost = 0
    if not right_term in self._loaded_to_gpu_cache[node]:
      self._loaded_to_gpu_cache[node].add(right_term)
      rbb = self._cache[right_term]
      extra_cost += rbb.size()

    if node.indices[self._lead_dim] != left_term.indices[self._lead_dim]:
      if not node.leftTerm in self._loaded_to_gpu_cache[node]:
        self._loaded_to_gpu_cache[node].add(left_term)
        lbb = self._cache[left_term]
        extra_cost += lbb.size()
    return cost + extra_cost

  def estimate_IndexSum(self, node):
    cost = super().estimate_IndexSum(node)

    # Note: we cannot derive the dimension along which we
    # are going to apply parallelization directly from
    # the IndexSum. Therefore we need to find a next Product
    # term and look at the left term
    child = node.term()
    while isinstance(child, IndexSum):
      child = child.term()

    left_term, _ = self._get_terms(child)
    bb = self._cache[left_term]

    # we will have visited node.term() as well at this point
    # (but we need to add ourselves as well)
    self._loaded_to_gpu_cache[node] = set(self._loaded_to_gpu_cache[node.term()])
    self._loaded_to_gpu_cache[node].add(node)

    return cost / bb[self._lead_dim].size()


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
