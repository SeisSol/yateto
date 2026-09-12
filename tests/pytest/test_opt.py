"""
Direct tests for ``yateto.ast.opt.strengthReduction``.

``strengthReduction`` is Yateto's well-pruned exhaustive search for the
optimal pairing of tensor operands in a multi-way contraction (cf.
Lam et al., 1997 - referenced in the paper).  Given a list of terms
and target indices, it returns an AST built from nested
``Elementwise`` / ``Reduction`` nodes whose total cost (per the supplied
cost estimator) is minimal.

The transformer ``ast.transformer.StrengthReduction`` wraps this into a
visitor pass.  We already exercise that end-to-end in
``test_ast_transformer.py``; here we poke the lower-level function
directly to lock in specific decisions (which pair to contract first,
what shape the tree should have).
"""
from __future__ import annotations

import pytest

from yateto import ops
from yateto import Tensor
from yateto.ast.cost import ShapeCostEstimator
from yateto.ast.indices import Indices
from yateto.ast.node import IndexedTensor, Reduction, Elementwise
from yateto.ast.opt import strengthReduction


def _it(name, shape, letters):
    """Shortcut: an IndexedTensor for ``Tensor(name, shape)[letters]``."""
    return IndexedTensor(Tensor(name, shape), letters)


def _has(cls, node):
    """Recursive instance-of check."""
    if isinstance(node, cls):
        return True
    return any(_has(cls, child) for child in node)


def _count(cls, node):
    """Recursive count of nodes of ``cls``."""
    n = 1 if isinstance(node, cls) else 0
    for c in node:
        n += _count(cls, c)
    return n


# ---------------------------------------------------------------------------
# Trivial degenerate cases
# ---------------------------------------------------------------------------


class TestDegenerate:
    def test_single_term_is_returned_as_is(self):
        # With one operand there is nothing to pair, so the returned AST
        # is just the single term itself.
        a = _it("A", (3, 4), "ij")
        tree = strengthReduction([a], Indices("ij", (3, 4)), ShapeCostEstimator())
        assert tree is a

    def test_single_term_with_summation_wraps_in_indexsum(self):
        # Summing the only term over index ``j`` (not in target) must
        # emit an Reduction even though there's no Elementwise.
        a = _it("A", (3, 4), "ij")
        tree = strengthReduction([a], Indices("i", (3,)), ShapeCostEstimator())
        assert isinstance(tree, Reduction)
        # The inner node is the original operand.
        assert tree.term() is a


# ---------------------------------------------------------------------------
# Classical matmul (two operands)
# ---------------------------------------------------------------------------


class TestPairwiseMatmul:
    def test_matmul_structure(self):
        # A @ B, summing over k.  Expected tree shape:
        #   Reduction(ops.Add(), Elementwise(ops.Mul(), A, B), k)
        A = _it("A", (3, 4), "ik")
        B = _it("B", (4, 5), "kj")
        tree = strengthReduction([A, B], Indices("ij", (3, 5)),
                                 ShapeCostEstimator())

        assert isinstance(tree, Reduction)
        assert isinstance(tree.term(), Elementwise)
        # Both leaves are preserved.
        assert {c.tensor.name() for c in tree.term()} == {"A", "B"}


# ---------------------------------------------------------------------------
# Three-way contraction - the estimator decides the pairing
# ---------------------------------------------------------------------------


class TestThreeWay:
    def test_picks_cheapest_pair_first(self):
        # Three matrices:
        #   A: I x J  (2x3)
        #   B: J x K  (3x100)
        #   C: K x L  (100x2)
        # target: I x L
        #
        # The shape-cost estimator charges ``product(shape)`` per
        # Elementwise node.  Pairing (B, C) first gives a J x L intermediate
        # (3*100*2 = 600 weight); pairing (A, B) first gives a I x K
        # intermediate (2*3*100 = 600 weight).  Both are equally cheap
        # for the first Elementwise, but the next Elementwise matters:
        #
        #   ((AB) C) -> I x K * K x L -> 2*100*2 = 400
        #   (A (BC)) -> I x J * J x L -> 2*3*2   = 12
        #
        # So the second pairing is strictly cheaper.  The algorithm
        # must prefer it.
        A = _it("A", (2, 3), "ij")
        B = _it("B", (3, 100), "jk")
        C = _it("C", (100, 2), "kl")
        tree = strengthReduction([A, B, C], Indices("il", (2, 2)),
                                 ShapeCostEstimator())
        # The outer Elementwise should have A on one side and a BC sub-tree
        # (possibly wrapped in an Reduction) on the other.
        # Drill in until we find a Elementwise of Elementwises (i.e. a Elementwise
        # whose argument is itself a Elementwise-based subtree).
        products = []
        def collect(node):
            if isinstance(node, Elementwise):
                products.append(node)
            for c in node:
                collect(c)
        collect(tree)
        assert products, "No Elementwise node found in the strength-reduced tree"

        # The cheaper pairing keeps the A leaf paired *last* with the
        # (BC) intermediate.  Equivalently: A is not paired together
        # with B directly.  Check by looking at the innermost Elementwise
        # - which should be BC, not AB.
        innermost = min(products, key=lambda p: _count(Elementwise, p))
        inner_names = {c.tensor.name() for c in innermost
                       if isinstance(c, IndexedTensor)}
        # The innermost product is between leaves; neither side is A.
        assert "A" not in inner_names, (
            f"Strength reducer picked the more expensive pairing: "
            f"innermost Elementwise is {inner_names}"
        )


# ---------------------------------------------------------------------------
# Sum-only case (e.g. trace)
# ---------------------------------------------------------------------------


class TestReductionOnly:
    def test_trace_lowers_to_nested_indexsum(self):
        # trace(A) = sum_i A[i,i].  This isn't really a valid Yateto
        # input (index letters must be distinct in one tensor), but a
        # scalar = sum_ij A[i] * B[j] is:
        A = _it("A", (3,), "i")
        B = _it("B", (4,), "j")
        tree = strengthReduction([A, B], Indices("", ()),
                                 ShapeCostEstimator())
        # Result has the two reduction axes lifted out as Reduction.
        assert _has(Reduction, tree)
        # And exactly one Elementwise.
        assert _count(Elementwise, tree) == 1

    def test_sum_indices_are_all_eliminated(self):
        # Any target-free index of A must be summed out before we exit.
        A = _it("A", (3, 4), "ij")
        # target = "i" -> j is a sum index
        tree = strengthReduction([A], Indices("i", (3,)),
                                 ShapeCostEstimator())
        assert isinstance(tree, Reduction)
        # The sum index should be j.
        assert str(tree.reductionIndex()) == "j"
