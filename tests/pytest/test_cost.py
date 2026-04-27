"""
Tests for ``yateto.ast.cost`` - the cost model that drives strength
reduction.

There are three concrete estimators:

* ``ShapeCostEstimator``       - counts flops using the declared shape
                                 (dense upper bound).
* ``BoundingBoxCostEstimator`` - counts flops using the nonzero
                                 bounding box (caches per node).
* ``ExactCost``                - counts flops using the actual
                                 equivalent sparsity pattern (most
                                 accurate, also most expensive).

All estimators subclass ``CostEstimator`` and implement the
``estimate_<NodeClass>`` dispatch pattern.  Users of the API typically
hand a *class* (not an instance) to the generator, which then
instantiates a fresh estimator per AST.
"""
from __future__ import annotations

import numpy as np
import pytest

from yateto import Tensor
from yateto.ast.cost import (
    BoundingBoxCostEstimator,
    CachedCostEstimator,
    CostEstimator,
    ExactCost,
    ShapeCostEstimator,
)
from yateto.ast.node import IndexedTensor, IndexSum, Product
from yateto.ast.transformer import (
    DeduceIndices,
    EquivalentSparsityPattern,
    FindContractions,
    SetSparsityPattern,
    StrengthReduction,
)


def _lower_to_product_tree(kernel, estimator_cls=BoundingBoxCostEstimator):
    """Helper: run the minimal set of passes so Product / IndexSum nodes
    exist and eqspps are set.
    """
    kernel = DeduceIndices().visit(kernel)
    kernel = EquivalentSparsityPattern().visit(kernel)
    kernel = StrengthReduction(estimator_cls).visit(kernel)
    kernel = SetSparsityPattern().visit(kernel)
    return kernel


# ---------------------------------------------------------------------------
# ShapeCostEstimator
# ---------------------------------------------------------------------------


class TestShapeCostEstimator:
    def test_generic_node_is_free(self):
        # Leaves and unopinionated nodes contribute nothing.
        it = IndexedTensor(Tensor("A", (3, 4)), "ij")
        assert ShapeCostEstimator().generic_estimate(it) == 0

    def test_matmul_cost_is_ijk(self, square_tensors):
        # For an 8x8 matmul built of a Product node (shape i,j,k) + an
        # IndexSum over k, the cost is shape-based:
        #   Product: 8 * 8 * 8        = 512
        #   IndexSum: (k-1) * i * j   = 7 * 8 * 8 = 448
        # total = 960.  Same as ComputeOptimalFlopCount's answer above
        # (the cost model agrees with the flop counter for dense ops).
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = _lower_to_product_tree(kernel)
        cost = ShapeCostEstimator().estimate(kernel)
        assert cost == 960

    def test_cost_scales_with_shape(self):
        # Doubling a dimension doubles the estimated cost for a matmul.
        def cost_for(N):
            A = Tensor(f"A{N}", (N, N))
            B = Tensor(f"B{N}", (N, N))
            C = Tensor(f"C{N}", (N, N))
            kernel = C["ij"] <= A["ik"] * B["kj"]
            kernel = _lower_to_product_tree(kernel)
            return ShapeCostEstimator().estimate(kernel)
        c4 = cost_for(4)
        c8 = cost_for(8)
        # Doubling N roughly eight-folds the matmul cost (N^3 term).
        assert c8 / c4 >= 7


# ---------------------------------------------------------------------------
# BoundingBoxCostEstimator
# ---------------------------------------------------------------------------


class TestBoundingBoxCostEstimator:
    def test_dense_matches_shape_cost(self, square_tensors):
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = _lower_to_product_tree(kernel)
        shape = ShapeCostEstimator().estimate(kernel)
        bbox = BoundingBoxCostEstimator().estimate(kernel)
        # On a fully dense kernel, bbox-based and shape-based estimators
        # agree (bbox == full shape in that case).
        assert bbox == shape

    def test_sparse_reduces_cost(self):
        # Diagonal A * dense B: the bounding box of the diagonal is still
        # the full (i,k)-plane (the diagonal spans from (0,0) to (N,N)),
        # so the *bounding-box* estimator may not detect the saving.
        # ``ExactCost`` does (see below).  This is a useful distinction
        # to lock in with the tests.
        N = 4
        diag = np.eye(N, dtype=bool)
        A = Tensor("A", (N, N), spp=diag)
        B = Tensor("B", (N, N))
        C = Tensor("C", (N, N))
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = _lower_to_product_tree(kernel, estimator_cls=ExactCost)

        bb_cost = BoundingBoxCostEstimator().estimate(kernel)
        exact_cost = ExactCost().estimate(kernel)
        # The exact cost must be strictly lower than the bounding-box
        # upper bound.
        assert exact_cost < bb_cost

    def test_caches_per_node(self, square_tensors):
        # BoundingBoxCostEstimator inherits the cache of CachedCostEstimator.
        # Estimating the same node twice must return the same value
        # without re-doing work.
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = _lower_to_product_tree(C["ij"] <= A["ik"] * B["kj"])
        est = BoundingBoxCostEstimator()
        first = est.estimate(kernel)
        second = est.estimate(kernel)
        assert first == second


# ---------------------------------------------------------------------------
# ExactCost
# ---------------------------------------------------------------------------


class TestExactCost:
    def test_diagonal_times_diagonal_is_cheap(self):
        # D * D = D (diagonal).  A 4x4 matmul of two diagonals needs
        # only N (not N^3) multiply-adds.
        N = 4
        diag = np.eye(N, dtype=bool)
        A = Tensor("A", (N, N), spp=diag)
        B = Tensor("B", (N, N), spp=diag)
        C = Tensor("C", (N, N))
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = _lower_to_product_tree(kernel, estimator_cls=ExactCost)
        cost = ExactCost().estimate(kernel)
        # A dense 4x4 matmul costs 112 in our accounting (2*N^3 - N^2 = 128-16=112).
        # Diagonal-of-diagonal must be much cheaper - exactly N products
        # (the contraction collapses to elementwise pairing).
        # Thus, we obtain (FMA - add to zero): 2*N - N = 8 - 4 = 4.
        assert cost == 4


# ---------------------------------------------------------------------------
# Abstract base behaviour
# ---------------------------------------------------------------------------


class TestAbstractBase:
    def test_CostEstimator_requires_generic_estimate(self):
        # Subclasses MUST implement generic_estimate - it's abstract.
        class Incomplete(CostEstimator):
            pass  # no generic_estimate
        with pytest.raises(TypeError):
            Incomplete()

    def test_cached_estimator_is_abstract_too(self):
        class Incomplete(CachedCostEstimator):
            pass
        with pytest.raises(TypeError):
            Incomplete()
