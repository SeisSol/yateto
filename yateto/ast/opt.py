import sys
import warnings
from .. import ops
from .node import Reduction, Elementwise

###############################################################################
# Contraction plans
#
# A plan names a binary tree by leaf position and summation index:
#
#   ('leaf', k) | ('prod', plan, plan) | ('sum', plan, indexName)
#
# It holds no AST node, so replaying a plan onto a term list allocates fresh
# interior nodes and can never alias a tree built earlier.  That is what makes
# the plan cache safe: the AST is mutable (setIndexPermutation, setEqspp,
# setMemoryLayout, ...), so caching trees would let one kernel write into
# another's.
###############################################################################

def _buildPlan(plan, terms):
  kind = plan[0]
  if kind == 'leaf':
    return terms[plan[1]]
  if kind == 'prod':
    return Elementwise(ops.Mul(), _buildPlan(plan[1], terms), _buildPlan(plan[2], terms))
  return Reduction(ops.Add(), _buildPlan(plan[1], terms), plan[2])


# A plan is a few dozen bytes, and a kernel set produces a few hundred distinct
# ones; the cache is capped anyway so that a generator run has a bound on it.
_PLAN_CACHE_SIZE = 4096
_planCache = dict()

def _planKey(terms, target_indices, cost_estimator, split):
  """Everything the estimator's verdict depends on, or None if not cacheable."""
  signature = cost_estimator.planSignature(terms)
  if signature is None:
    return None
  if len(set(id(term) for term in terms)) != len(terms):
    # one node object in two positions: replaying a plan would put the same
    # object into two places of one tree, and the AST is mutable
    return None
  return (tuple((str(term.indices), term.indices.shape()) for term in terms),
          tuple(sorted(set(target_indices))), split, signature)

def _cachePlan(key, plan):
  if len(_planCache) >= _PLAN_CACHE_SIZE:
    _planCache.pop(next(iter(_planCache)))
  _planCache[key] = plan


###############################################################################
# The network the analytic searches run on
#
# ShapeCostEstimator and BoundingBoxCostEstimator are pure functions of the
# per-index ranges of the leaves:
#
#   product : cost = extent over the union of the operands' external indices
#   sum     : cost = (extent before) - (extent after)
#
# ShapeCostEstimator is the case in which every leaf range is the full extent
# of its index.  Modelling both like this lets the search run on integers and
# bitmasks: not a single AST node is allocated before the winner is known.
###############################################################################

class _Network:
  """Leaves as index bitmasks plus a range per index.

  An index is *external* to a set of leaves if it is a target index or occurs
  outside that set as well; everything else is summed away the moment the set
  is formed, which is what the enumeration in _exhaustivePlan does too.
  """

  def __init__(self, terms, target_indices, leafRanges):
    names = []
    for term in terms:
      for index in term.indices:
        if index not in names:
          names.append(index)
    position = {index: p for p, index in enumerate(names)}

    self.names = names
    self.nIndices = len(names)
    self.n = len(terms)
    self.full = (1 << self.n) - 1

    self.targetMask = 0
    for index in target_indices:
      if index in position:
        self.targetMask |= 1 << position[index]

    self.leafMask = []
    self.leafLo = []
    self.leafHi = []
    for term, ranges in zip(terms, leafRanges):
      mask = 0
      lo = [0] * self.nIndices
      hi = [0] * self.nIndices
      for index, (start, stop) in zip(term.indices, ranges):
        p = position[index]
        mask |= 1 << p
        lo[p] = start
        hi[p] = stop
      self.leafMask.append(mask)
      self.leafLo.append(lo)
      self.leafHi.append(hi)

  def summations(self, fromMask, toMask):
    """The indices dropped between the two masks, in a reproducible order."""
    return sorted(self.names[p] for p in range(self.nIndices)
                  if (fromMask >> p & 1) and not (toMask >> p & 1))


def _optimalPlan(net):
  """The cheapest tree, by a dynamic program over the subsets of the leaves.

    cost(S) = min over bipartitions S = L | R of
                cost(L) + cost(R) + size + (size - extent(S))

  with `size` the extent over the union of the operands' external indices, i.e.
  the size of the product node, and extent(S) what is left of it after the
  summations that sit on top.
  """
  n, nIndices = net.n, net.nIndices
  count = 1 << n
  leafMask, leafLo, leafHi = net.leafMask, net.leafLo, net.leafHi

  # ranges are held flat, at subset * nIndices + index position: at the search
  # limit that is two lists instead of 2 * 16384 small ones
  mask = [0] * count
  lo = [0] * (count * nIndices)
  hi = [0] * (count * nIndices)
  for k in range(n):
    subset = 1 << k
    base = subset * nIndices
    mask[subset] = leafMask[k]
    lo[base:base + nIndices] = leafLo[k]
    hi[base:base + nIndices] = leafHi[k]

  for subset in range(3, count):
    if subset & (subset - 1) == 0:
      continue
    low = subset & -subset
    rest = subset ^ low
    k = low.bit_length() - 1
    here, there = subset * nIndices, rest * nIndices
    lo[here:here + nIndices] = lo[there:there + nIndices]
    hi[here:here + nIndices] = hi[there:there + nIndices]
    kLo, kHi = leafLo[k], leafHi[k]
    shared = mask[rest] & leafMask[k]
    bits = leafMask[k]
    while bits:
      bit = bits & -bits
      bits ^= bit
      p = bit.bit_length() - 1
      if shared & bit:
        # an index carried by both sides is restricted to the overlap of the
        # two ranges, as Range.__and__ does
        if kLo[p] > lo[here + p]:
          lo[here + p] = kLo[p]
        if kHi[p] < hi[here + p]:
          hi[here + p] = kHi[p]
      else:
        lo[here + p] = kLo[p]
        hi[here + p] = kHi[p]
    mask[subset] = mask[rest] | leafMask[k]

  full, targetMask = net.full, net.targetMask
  external = [0] * count
  for subset in range(1, count):
    external[subset] = mask[subset] & (targetMask | mask[full ^ subset])

  def extent(indexMask, subset):
    size = 1
    base = subset * nIndices
    while indexMask:
      bit = indexMask & -indexMask
      indexMask ^= bit
      p = base + bit.bit_length() - 1
      size *= hi[p] - lo[p]
    return size

  cost = [0] * count
  # the largest intermediate a subtree has to hold: part of the tie-break, and
  # what the kernel's scratch buffer ends up being
  peak = [0] * count
  choice = [0] * count
  for k in range(n):
    subset = 1 << k
    cost[subset] = extent(mask[subset], subset) - extent(external[subset], subset)
    # a leaf is an operand the caller passes, not something the kernel holds

  def tieBreak(left, subset):
    """How to choose between two trees of the same cost.

    An index both operands carry and the result keeps is not a dimension of the
    contraction, it is a loop around it: it shrinks the GEMM the pair becomes
    and the vectoriser is left with what is inside. Its extent therefore comes
    first, and the size of the larger operand second.
    """
    right = subset ^ left
    batch = external[left] & external[right] & external[subset]
    return (extent(batch, subset),
            peak[left] if peak[left] > peak[right] else peak[right])

  for subset in range(3, count):
    if subset & (subset - 1) == 0:
      continue
    base = extent(external[subset], subset)
    best = sys.maxsize
    bestLeft = 0
    bestKey = None
    low = subset & -subset
    rest = subset ^ low
    # the submasks of `rest`, with `low` pinned into the left half: that way
    # every bipartition is seen once, in 3^n / 2 steps rather than 3^n
    sub = rest
    while True:
      if sub != rest:
        left = sub | low
        right = subset ^ left
        # an outer product (external[left] & external[right] == 0) stays in the
        # search: for small operands it is occasionally the cheaper tree
        size = extent(external[left] | external[right], subset)
        candidate = cost[left] + cost[right] + size + (size - base)
        if candidate < best:
          best = candidate
          bestLeft = left
          bestKey = None
        elif candidate == best:
          # only a tie pays for the key, which is why it is not part of the cost
          if bestKey is None:
            bestKey = tieBreak(bestLeft, subset)
          key = tieBreak(left, subset)
          if key < bestKey:
            bestLeft = left
            bestKey = key
      if sub == 0:
        break
      sub = (sub - 1) & rest
    cost[subset] = best
    choice[subset] = bestLeft
    held = peak[bestLeft] if peak[bestLeft] > peak[subset ^ bestLeft] else peak[subset ^ bestLeft]
    peak[subset] = held if held > base else base

  def build(subset):
    if subset & (subset - 1) == 0:
      plan = ('leaf', subset.bit_length() - 1)
      fromMask = mask[subset]
    else:
      left = choice[subset]
      right = subset ^ left
      plan = ('prod', build(left), build(right))
      fromMask = external[left] | external[right]
    for index in net.summations(fromMask, external[subset]):
      plan = ('sum', plan, index)
    return plan

  return build(full)


# The subset DP is Theta(3^n) in time: about 3 s at n = 14 and 30 s at n = 16.
_DP_LIMIT = 14

def _greedyPlan(net):
  """Repeatedly contract the cheapest pair.

  O(n^3) and never optimal, but it keeps a kernel with more terms than the
  exact search can take from stalling a build for hours.
  """
  def extent(indexMask, lo, hi):
    size = 1
    while indexMask:
      bit = indexMask & -indexMask
      indexMask ^= bit
      p = bit.bit_length() - 1
      size *= hi[p] - lo[p]
    return size

  def merged(a, b):
    lo = list(a[1])
    hi = list(a[2])
    bits = b[0]
    while bits:
      bit = bits & -bits
      bits ^= bit
      p = bit.bit_length() - 1
      if a[0] & bit:
        if b[1][p] > lo[p]:
          lo[p] = b[1][p]
        if b[2][p] < hi[p]:
          hi[p] = b[2][p]
      else:
        lo[p] = b[1][p]
        hi[p] = b[2][p]
    return a[0] | b[0], lo, hi

  def others(groups, *skip):
    mask = 0
    for k, group in enumerate(groups):
      if k not in skip:
        mask |= group[0]
    return mask

  def sumDown(mask, ext, plan):
    for index in net.summations(mask, ext):
      plan = ('sum', plan, index)
    return plan

  groups = [[net.leafMask[k], net.leafLo[k], net.leafHi[k], ('leaf', k)]
            for k in range(net.n)]

  # an index nothing else carries is summed away before anything is contracted,
  # so from here on a group's mask is its external index set
  for i, group in enumerate(groups):
    ext = group[0] & (net.targetMask | others(groups, i))
    group[3] = sumDown(group[0], ext, group[3])
    group[0] = ext

  while len(groups) > 1:
    best = None
    for i in range(len(groups)):
      for j in range(i + 1, len(groups)):
        mask, lo, hi = merged(groups[i], groups[j])
        ext = mask & (net.targetMask | others(groups, i, j))
        size = extent(mask, lo, hi)
        cost = size + (size - extent(ext, lo, hi))
        if best is None or cost < best[0]:
          best = (cost, i, j, mask, lo, hi, ext)

    _, i, j, mask, lo, hi, ext = best
    plan = sumDown(mask, ext, ('prod', groups[i][3], groups[j][3]))
    groups = [group for k, group in enumerate(groups) if k != i and k != j]
    groups.append([ext, lo, hi, plan])

  return groups[0][3]


###############################################################################
# Exhaustive search
#
# For an estimator whose cost is not a function of the leaves' ranges alone --
# FusedGemmsBoundingBoxCostEstimator divides by the extent of the leading
# dimension of the left operand, which depends on how the subtree was built --
# the tree comes out of the full enumeration.
#
# It runs on plans as well, and every distinct subtree is built exactly once:
# the enumeration reaches the same subtree along many branches, where it would
# otherwise allocate a fresh node -- and a fresh miss in the estimator's cache
# -- each time.
###############################################################################

class _Pool:
  """The AST nodes of one search, one node per distinct subtree."""

  def __init__(self, terms, cost_estimator):
    self._terms = terms
    self._estimator = cost_estimator
    self._nodes = dict()
    self._costs = dict()
    self._indices = dict()

  def node(self, plan):
    node = self._nodes.get(plan)
    if node is None:
      kind = plan[0]
      if kind == 'leaf':
        node = self._terms[plan[1]]
      elif kind == 'prod':
        node = Elementwise(ops.Mul(), self.node(plan[1]), self.node(plan[2]))
      else:
        node = Reduction(ops.Add(), self.node(plan[1]), plan[2])
      self._nodes[plan] = node
    return node

  def indices(self, plan):
    indices = self._indices.get(plan)
    if indices is None:
      indices = frozenset(self.node(plan).indices)
      self._indices[plan] = indices
    return indices

  def cost(self, plan):
    cost = self._costs.get(plan)
    if cost is None:
      cost = self._estimator.estimate(self.node(plan))
      self._costs[plan] = cost
    return cost


def _exhaustivePlan(pool, plans, targetNames, split):
  n = len(plans)
  indices = pool.indices

  counts = dict()
  for plan in plans:
    for index in indices(plan):
      counts[index] = counts.get(index, 0) + 1
  summationIndices = set(index for index, count in counts.items()
                         if count == 1) - targetNames

  while summationIndices:
    i = split
    while i < n:
      intersection = summationIndices.intersection(indices(plans[i]))
      if intersection:
        # the smallest name, so that the tree does not depend on the order a
        # set of strings happens to enumerate in
        index = min(intersection)
        plans = plans[:i] + plans[i+1:] + (('sum', plans[i], index),)
        summationIndices.discard(index)
      else:
        i = i + 1

  if n == 1:
    return plans[0], pool.cost(plans[0])

  # NOTE: memoising on (plans, split) buys nothing -- `split` already keeps two
  # independent pairs from being contracted in both orders, so no list of plans
  # is ever reached twice. Measured at zero hits over the example kernels.
  cost = pool.cost
  best = None
  minCost = sys.maxsize
  for i in range(n):
    for j in range(max(i+1, split), n):
      product = ('prod', plans[i], plans[j])
      if best is None or cost(product) < minCost:
        rest = plans[:i] + plans[i+1:j] + plans[j+1:] + (product,)
        plan, planCost = _exhaustivePlan(pool, rest, targetNames, j-1)
        if best is None or planCost < minCost:
          best = plan
          minCost = planCost

  return best, minCost


def strengthReduction(terms, target_indices, cost_estimator, split = 0):
  terms = list(terms)
  if len(terms) == 0:
    return None

  # look the plan up before asking for a search model: on a hit the model is
  # never needed, and building it walks the bounding box of every leaf
  key = _planKey(terms, target_indices, cost_estimator, split)
  if key is not None:
    plan = _planCache.get(key)
    if plan is not None:
      return _buildPlan(plan, terms)

  ranges = cost_estimator.searchModel(terms)
  if ranges is None:
    pool = _Pool(terms, cost_estimator)
    leaves = tuple(('leaf', k) for k in range(len(terms)))
    plan, _ = _exhaustivePlan(pool, leaves, set(target_indices), split)
    # the pool already holds these nodes, and a plan uses every leaf once, so
    # the tree cannot alias itself
    tree = pool.node(plan)
  else:
    net = _Network(terms, target_indices, ranges)
    if len(terms) <= _DP_LIMIT:
      plan = _optimalPlan(net)
    else:
      warnings.warn(
        'Contracting {} terms exceeds the exact search limit of {}; falling '
        'back to a greedy contraction order, which may be considerably more '
        'expensive than the optimum.'.format(len(terms), _DP_LIMIT),
        stacklevel=2)
      plan = _greedyPlan(net)
    tree = _buildPlan(plan, terms)

  if key is not None:
    _cachePlan(key, plan)
  return tree
