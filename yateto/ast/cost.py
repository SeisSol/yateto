from .indices import BoundingBox
from .. import ops
from .node import Reduction
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

def isProduct(node):
  """A binary multiplication -- the shape a contraction is built from."""
  return node.optype == ops.Mul() and len(node) == 2

def isSummation(node):
  return node.optype == ops.Add()

class ShapeCostEstimator(CostEstimator):
  def generic_estimate(self, node):
    return 0

  def estimate_Elementwise(self, node):
    if not isProduct(node):
      return self.generic_estimate(node)
    cost = 1
    for size in node.shape():
      cost *= size
    return cost

  def estimate_Reduction(self, node):
    if not isSummation(node):
      return self.generic_estimate(node)
    cost = node.reductionIndex().shape()[0] - 1
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

  def estimate_Elementwise(self, node):
    if not isProduct(node):
      return self.generic_estimate(node)
    lbb = self._cache[node[0]]
    rbb = self._cache[node[1]]
    lind = node[0].indices
    rind = node[1].indices
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

  def estimate_Reduction(self, node):
    if not isSummation(node):
      return self.generic_estimate(node)
    tbb = self._cache[node.term()]
    pos = node.term().indices.find(node.sumIndexName())
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
    left_indices = node[0].indices
    right_indices = node[1].indices
    common_indices = left_indices & right_indices

    if len(left_indices) == 0 or left_indices[0] in common_indices:
      # swap terms becase we do not allow
      # tensor product along the leading dimension.
      # In other words, LoG will try to swap terms
      # in the future
      return node[1], node[0]
    else:
      return node[0], node[1]

  def estimate_Elementwise(self, node):
    if not isProduct(node):
      return super().estimate_Elementwise(node)
    cost = super().estimate_Elementwise(node)
    left_term, right_term = self._get_terms(node)

    # NOTE: the case rank-0 tensor product rank-0 tensor is currently ill-supported here,
    # (we only save against _one_ rank-0 tensor in self._get_terms)

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
      if left_term not in self._loaded_to_gpu_cache[node]:
        self._loaded_to_gpu_cache[node].add(left_term)
        lbb = self._cache[left_term]
        extra_cost += lbb.size()
    return cost + extra_cost

  def estimate_Reduction(self, node):
    if not isSummation(node):
      return super().estimate_Reduction(node)
    cost = super().estimate_Reduction(node)

    # Note: we cannot derive the dimension along which we
    # are going to apply parallelization directly from
    # the reduction. Therefore we need to find the next product
    # term and look at the left term
    child = node.term()
    while isinstance(child, Reduction) and isSummation(child):
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

  def estimate_Elementwise(self, node):
    if not isProduct(node):
      return self.generic_estimate(node)
    spp = node.computeSparsityPattern(self._cache[node[0]], self._cache[node[1]])
    self._cache[node] = spp
    return spp.count_nonzero()

  def estimate_Reduction(self, node):
    if not isSummation(node):
      return self.generic_estimate(node)
    termSpp = self._cache[node.term()]
    spp = node.computeSparsityPattern(termSpp)
    self._cache[node] = spp
    return termSpp.count_nonzero() - spp.count_nonzero()
