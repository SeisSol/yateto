"""
Tests for the two searches behind ``yateto.ast.opt.strengthReduction``.

Which one runs is decided by the cost estimator:

* an estimator that hands back a ``searchModel`` is solved by a dynamic
  program over the subsets of the leaves, on integers alone;
* one that does not -- ``FusedGemmsBoundingBoxCostEstimator``, whose cost
  reads the shape of a subtree and not just its leaves -- is solved by the
  enumeration.

Both are required to return a tree that computes the contraction, and both
are required to return the *cheapest* such tree.  The reference for "cheapest"
here is a brute-force enumeration of every binary tree, written out in this
file so that it shares nothing with the code under test.
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest

from yateto import Tensor
from yateto import ops
from yateto.ast import opt
from yateto.ast.cost import (
    BoundingBoxCostEstimator,
    FusedGemmsBoundingBoxCostEstimator,
    ShapeCostEstimator,
)
from yateto.ast.node import Elementwise, IndexedTensor, Reduction
from yateto.ast.transformer import DeduceIndices, EquivalentSparsityPattern


def _network(termIndices, target, sizes, spp=None):
    """The Einsum of ``termIndices`` with its indices deduced and eqspp set."""
    node = None
    for k, indices in enumerate(termIndices):
        shape = tuple(sizes[c] for c in indices)
        pattern = None if spp is None else spp[k]
        tensor = Tensor('T{}'.format(k), shape, spp=pattern)[indices]
        node = tensor if node is None else node * tensor
    node = DeduceIndices(target).visit(node)
    return EquivalentSparsityPattern().visit(node)


def _bruteForce(terms, target, Estimator):
    """The cheapest tree, by trying every one of them.

    Every ordered sequence of pair contractions is built, with the indices that
    have become private summed away after each step -- which is what any
    contraction order does.  Exponential, hence only used on small inputs.
    """
    def free(nodes):
        counts = {}
        for node in nodes:
            for index in node.indices:
                counts[index] = counts.get(index, 0) + 1
        return counts

    def reduce(node, nodes, counts):
        for index in sorted(node.indices):
            if index not in target and counts[index] == 1:
                node = Reduction(ops.Add(), node, index)
        return node

    best = [None, None]

    def recurse(nodes):
        if len(nodes) == 1:
            cost = Estimator().estimate(copy.deepcopy(nodes[0]))
            if best[0] is None or cost < best[0]:
                best[0], best[1] = cost, nodes[0]
            return
        for i, j in itertools.combinations(range(len(nodes)), 2):
            product = Elementwise(ops.Mul(), nodes[i], nodes[j])
            rest = [n for k, n in enumerate(nodes) if k != i and k != j]
            counts = free(rest + [product])
            recurse(rest + [reduce(product, rest, counts)])

    counts = free(terms)
    recurse([reduce(t, terms, counts) for t in terms])
    return best[0]


def _cost(tree, Estimator):
    return Estimator().estimate(copy.deepcopy(tree))


def _evaluate(node, values):
    """(array, index string) of the subtree, via numpy."""
    if isinstance(node, IndexedTensor):
        return values[node.tensor.name()], str(node.indices)
    if isinstance(node, Elementwise):
        a, ai = _evaluate(node[0], values)
        b, bi = _evaluate(node[1], values)
        out = str(node.indices)
        return np.einsum('{},{}->{}'.format(ai, bi, out), a, b), out
    if isinstance(node, Reduction):
        a, ai = _evaluate(node.term(), values)
        out = str(node.indices)
        return np.einsum('{}->{}'.format(ai, out), a), out
    raise AssertionError(type(node).__name__)


NETWORKS = [
    # (terms, target, sizes)
    (['ij', 'jk'], 'ik', dict(i=4, j=6, k=5)),
    (['ij', 'jk', 'kl'], 'il', dict(i=4, j=6, k=5, l=7)),
    (['lk', 'slq', 'qp'], 'skp', dict(l=8, k=8, s=3, q=5, p=5)),
    (['xl', 'li', 'ym', 'mj'], 'ijxy', dict(x=6, l=3, i=6, y=6, m=3, j=6)),
    (['abc', 'cd', 'de'], 'abe', dict(a=3, b=4, c=5, d=6, e=2)),
    (['ij', 'ik', 'il'], 'jkl', dict(i=8, j=3, k=3, l=3)),
]

ESTIMATORS = [ShapeCostEstimator, BoundingBoxCostEstimator,
              FusedGemmsBoundingBoxCostEstimator]


@pytest.fixture(autouse=True)
def _emptyPlanCache():
    opt._planCache.clear()
    yield
    opt._planCache.clear()


@pytest.mark.parametrize('terms,target,sizes', NETWORKS)
@pytest.mark.parametrize('Estimator', ESTIMATORS)
def test_search_finds_the_cheapest_tree(terms, target, sizes, Estimator):
    node = _network(terms, target, sizes)
    operands = list(node)
    tree = opt.strengthReduction([copy.deepcopy(t) for t in operands],
                                 node.indices, Estimator())
    optimum = _bruteForce([copy.deepcopy(t) for t in operands], set(target), Estimator)
    assert _cost(tree, Estimator) == pytest.approx(optimum)


@pytest.mark.parametrize('terms,target,sizes', NETWORKS)
@pytest.mark.parametrize('Estimator', ESTIMATORS)
def test_tree_computes_the_contraction(terms, target, sizes, Estimator):
    node = _network(terms, target, sizes)
    operands = list(node)
    values = {t.tensor.name(): np.random.default_rng(k).standard_normal(t.tensor.shape())
              for k, t in enumerate(operands)}
    tree = opt.strengthReduction([copy.deepcopy(t) for t in operands],
                                 node.indices, Estimator())
    got, indices = _evaluate(tree, values)
    assert sorted(indices) == sorted(str(node.indices))
    spec = '{}->{}'.format(','.join(str(t.indices) for t in operands), indices)
    expected = np.einsum(spec, *[values[t.tensor.name()] for t in operands])
    assert np.allclose(got, expected)


@pytest.mark.parametrize('Estimator', ESTIMATORS)
def test_a_sparse_network_is_searched_on_its_bounding_boxes(Estimator):
    sizes = dict(i=6, j=6, k=6, l=6)
    spp = [np.zeros((6, 6), dtype=bool) for _ in range(3)]
    spp[0][:2, :] = True
    spp[1][:, 3:] = True
    spp[2][:] = True
    node = _network(['ij', 'jk', 'kl'], 'il', sizes, spp=spp)
    operands = list(node)
    tree = opt.strengthReduction([copy.deepcopy(t) for t in operands],
                                 node.indices, Estimator())
    optimum = _bruteForce([copy.deepcopy(t) for t in operands], set('il'), Estimator)
    assert _cost(tree, Estimator) == pytest.approx(optimum)


@pytest.mark.parametrize('terms,target,sizes', NETWORKS)
@pytest.mark.parametrize('Estimator', ESTIMATORS)
def test_a_replayed_plan_is_a_fresh_tree(terms, target, sizes, Estimator):
    """A cached plan must be rebuilt onto the terms it is replayed for.

    Handing back the tree itself would let one kernel write into another's
    through setIndexPermutation.
    """
    node = _network(terms, target, sizes)
    operands = list(node)
    first = opt.strengthReduction([copy.deepcopy(t) for t in operands],
                                  node.indices, Estimator())
    second = opt.strengthReduction([copy.deepcopy(t) for t in operands],
                                   node.indices, Estimator())

    def nodes(tree):
        yield tree
        for child in tree:
            yield from nodes(child)

    assert first is not second
    assert not (set(id(n) for n in nodes(first)) & set(id(n) for n in nodes(second)))
    assert _cost(first, Estimator) == pytest.approx(_cost(second, Estimator))


def test_the_plan_cache_separates_estimators():
    """Two estimators may want different trees for the same terms."""
    sizes = dict(i=8, j=2, k=8, l=2)
    node = _network(['ij', 'jk', 'kl'], 'il', sizes)
    operands = list(node)
    trees = {}
    for Estimator in ESTIMATORS:
        tree = opt.strengthReduction([copy.deepcopy(t) for t in operands],
                                     node.indices, Estimator())
        trees[Estimator] = _cost(tree, Estimator)
    for Estimator, cost in trees.items():
        again = opt.strengthReduction([copy.deepcopy(t) for t in operands],
                                      node.indices, Estimator())
        assert _cost(again, Estimator) == pytest.approx(cost)


def test_a_term_used_twice_is_not_cached():
    """The same node object in two positions must not go through the cache."""
    tensor = Tensor('A', (4, 4))
    shared = tensor['ij']
    key = opt._planKey([shared, shared], 'ij', ShapeCostEstimator(), 0)
    assert key is None


def test_the_plan_cache_stays_bounded():
    sizes = dict(i=3, j=3, k=3)
    node = _network(['ij', 'jk'], 'ik', sizes)
    operands = list(node)
    for extra in range(opt._PLAN_CACHE_SIZE + 16):
        opt._cachePlan(('filler', extra), ('leaf', 0))
    assert len(opt._planCache) <= opt._PLAN_CACHE_SIZE
    tree = opt.strengthReduction([copy.deepcopy(t) for t in operands],
                                 node.indices, ShapeCostEstimator())
    assert tree is not None


def test_the_greedy_fallback_returns_a_usable_tree():
    """Beyond the exact limit the search warns and falls back."""
    sizes = {c: 2 for c in 'abcdefghijklmnop'}
    terms = ['ab', 'bc', 'cd', 'de', 'ef', 'fg', 'gh', 'hi',
             'ij', 'jk', 'kl', 'lm', 'mn', 'no', 'op']
    node = _network(terms, 'ap', sizes)
    operands = list(node)
    assert len(operands) > opt._DP_LIMIT
    values = {t.tensor.name(): np.random.default_rng(k).standard_normal(t.tensor.shape())
              for k, t in enumerate(operands)}
    with pytest.warns(UserWarning, match='greedy'):
        tree = opt.strengthReduction([copy.deepcopy(t) for t in operands],
                                     node.indices, ShapeCostEstimator())
    got, indices = _evaluate(tree, values)
    spec = '{}->{}'.format(','.join(str(t.indices) for t in operands), indices)
    expected = np.einsum(spec, *[values[t.tensor.name()] for t in operands])
    assert np.allclose(got, expected)


def test_no_search_model_means_the_enumeration():
    assert FusedGemmsBoundingBoxCostEstimator().searchModel([]) is None
    assert ShapeCostEstimator().searchModel([]) == []


def test_the_fused_estimator_still_shares_a_plan_signature():
    """It cannot use the analytic search, but its verdict is reproducible."""
    node = _network(['ij', 'jk'], 'ik', dict(i=4, j=5, k=6))
    terms = list(node)
    assert FusedGemmsBoundingBoxCostEstimator().planSignature(terms) is not None


def test_a_reduction_over_a_plain_term_is_estimated():
    """The fused estimator also sees reductions with no product underneath.

    They arise whenever the target indices are narrower than the contraction's
    own, as when an equivalent sparsity pattern is computed for one operand.
    """
    node = _network(['ij', 'jk', 'kl'], 'il', dict(i=4, j=5, k=3, l=6))
    operands = list(node)
    estimator = FusedGemmsBoundingBoxCostEstimator()
    tree = opt.strengthReduction([copy.deepcopy(t) for t in operands],
                                 operands[0].indices, estimator)
    assert tree is not None


def test_a_rank_zero_product_is_estimated():
    """Both operands rank-0: there is no leading dimension to divide by."""
    node = _network(['', ''], '', {})
    operands = list(node)
    tree = opt.strengthReduction([copy.deepcopy(t) for t in operands],
                                 node.indices, FusedGemmsBoundingBoxCostEstimator())
    assert tree is not None
