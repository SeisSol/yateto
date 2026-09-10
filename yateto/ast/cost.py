from .indices import BoundingBox
from .. import ops
from .node import Reduction
from abc import ABC, abstractmethod
from fractions import Fraction


class CostEstimator(ABC):
  """Estimates what a node costs, and tells the contraction search how far it
  can go without building trees.

  Besides `estimate`, an estimator declares two properties that the search in
  ast/opt.py acts on:

  * `searchModel` -- if the cost of a tree follows from the per-index ranges of
    its leaves, it hands those ranges back and the search runs on an analytic
    model: integers and bitmasks, with no AST node allocated before the winner
    is known. None selects the enumeration instead.
  * `planSignature` -- a hashable description of everything the estimator reads
    beyond the index names and shapes of the leaves. It lets the search reuse a
    contraction order across kernels, and is independent of `searchModel`: an
    estimator whose verdict is reproducible from the signature can be cached
    even when its cost does not decompose over the tree. None disables caching.
  """

  def searchModel(self, terms):
    return None

  def planSignature(self, terms):
    return None

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
  """A binary multiplication -- the shape a contraction is built from.

  Total over nodes: an operand of a contraction may be any node, and only
  those carrying an operation can be one of these two shapes.
  """
  return getattr(node, 'optype', None) == ops.Mul() and len(node) == 2

def isSummation(node):
  return getattr(node, 'optype', None) == ops.Add()

class ShapeCostEstimator(CostEstimator):
  def searchModel(self, terms):
    return [term.indices.ranges() for term in terms]

  def planSignature(self, terms):
    # the index names and shapes are the whole of it, and the search keys on
    # those already
    return (type(self).__name__,)

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
  def searchModel(self, terms):
    return [[(r.start, r.stop) for r in term.boundingBox()] for term in terms]

  def planSignature(self, terms):
    return (type(self).__name__,
            tuple(tuple((r.start, r.stop) for r in term.boundingBox())
                  for term in terms))

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

  # The cost of a subtree is not settled by its leaves alone: both
  # estimate_Elementwise and estimate_Reduction divide by the extent of the
  # leading dimension of the *left* operand, which depends on the shape of the
  # subtree. Optimal substructure fails and a subset DP would hand back more
  # expensive trees, so the analytic search does not apply. The inherited plan
  # signature does: the cost still only reads the index names and bounding
  # boxes of the leaves, so the same leaves yield the same contraction order.
  def searchModel(self, terms):
    return None

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

  def _perThread(self, cost, term):
    """Divide by what the leading dimension of `term` parallelises over.

    Exact rather than floating point: the quotient is not always whole -- the
    leading index of the left operand may itself be contracted, and then the
    node's range for it is narrower than the operand's -- and comparing floats
    would let the last bit of a rounding decide a contraction order.

    A rank-0 term has no leading dimension -- which happens when both operands
    of a product are rank-0, since _get_terms can only move one of them out of
    the way -- and a structurally zero one nothing to spread over threads; in
    both cases the cost stands as it is.
    """
    bb = self._cache[term]
    if len(bb) == 0:
      return cost
    threads = bb[self._lead_dim].size()
    return Fraction(cost, threads) if threads > 0 else cost

  def _leadIndex(self, node):
    return node.indices[self._lead_dim] if len(node.indices) > 0 else None

  def estimate_Elementwise(self, node):
    if not isProduct(node):
      return super().estimate_Elementwise(node)
    cost = super().estimate_Elementwise(node)
    left_term, right_term = self._get_terms(node)

    cost = self._perThread(cost, left_term)

    # take the union of all cached nodes
    self._loaded_to_gpu_cache[node] = self._loaded_to_gpu_cache[left_term].union(self._loaded_to_gpu_cache[right_term])

    extra_cost = 0
    if not right_term in self._loaded_to_gpu_cache[node]:
      self._loaded_to_gpu_cache[node].add(right_term)
      rbb = self._cache[right_term]
      extra_cost += rbb.size()

    if self._leadIndex(node) != self._leadIndex(left_term):
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

    # a reduction may also sit straight on a term nothing was contracted into,
    # and then that term is what the leading dimension parallelises over
    left_term = self._get_terms(child)[0] if isProduct(child) else child

    # we will have visited node.term() as well at this point
    # (but we need to add ourselves as well)
    self._loaded_to_gpu_cache[node] = set(self._loaded_to_gpu_cache[node.term()])
    self._loaded_to_gpu_cache[node].add(node)

    return self._perThread(cost, left_term)


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
