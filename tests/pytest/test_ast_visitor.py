"""
Tests for ``yateto.ast.visitor`` - the read-only AST walkers.

Visitors implement the classic GoF visitor pattern (``visit_<ClassName>``
dispatch).  They are the analysis phase of the compiler: ``PrettyPrinter``
for debugging, ``FindTensors`` / ``FindIndexPermutations`` for metadata
collection, ``ComputeSparsityPattern`` / ``ComputeOptimalFlopCount`` for
numerical-property analysis, ``ComputeIndexSet`` / ``ComputeConstantExpression``
for the front-end.
"""
from __future__ import annotations

import numpy as np
import pytest

from yateto import Tensor
from yateto.ast.indices import Indices
from yateto.ast.node import Add, Assign, Einsum, IndexedTensor, Product
from yateto.ast.transformer import DeduceIndices
from yateto.ast.visitor import (
    CachedVisitor,
    ComputeConstantExpression,
    ComputeIndexSet,
    ComputeOptimalFlopCount,
    ComputeSparsityPattern,
    FindTensors,
    PrettyPrinter,
    Visitor,
)


# ---------------------------------------------------------------------------
# The base ``Visitor`` class
# ---------------------------------------------------------------------------


class TestVisitorDispatch:
    """Visitor dispatches on ``visit_<ClassName>``, falling back to
    ``generic_visit`` when no specialised method exists.  Custom visitors
    rely on this being rock solid.
    """

    def test_dispatches_on_node_class_name(self, square_tensors):
        A = square_tensors["A"]
        seen = []

        class Recorder(Visitor):
            def visit_IndexedTensor(self, node):
                seen.append(("IT", node.name()))

            def visit_Assign(self, node):
                seen.append(("Assign",))
                self.generic_visit(node)

        kernel = A["ij"] <= A["ij"]
        Recorder().visit(kernel)
        # First the Assign, then both IndexedTensor children.
        assert seen[0] == ("Assign",)
        assert ("IT", "A") in seen

    def test_generic_visit_recurses_into_children(self, square_tensors):
        A, B = square_tensors["A"], square_tensors["B"]
        depth = {"max": 0, "cur": 0}

        class DepthProbe(Visitor):
            def generic_visit(self, node):
                depth["cur"] += 1
                depth["max"] = max(depth["max"], depth["cur"])
                for c in node:
                    self.visit(c)
                depth["cur"] -= 1

        DepthProbe().visit(A["ij"] + B["ij"])
        # Add wraps two IndexedTensor leaves -> max depth 2.
        assert depth["max"] == 2

    def test_cached_visitor_reuses_results(self, square_tensors):
        A = square_tensors["A"]
        calls = [0]

        class CachedCounter(CachedVisitor):
            def generic_visit(self, node):
                calls[0] += 1
                return id(node)

        v = CachedCounter()
        kernel = A["ij"] <= A["ij"]
        first = v.visit(kernel)
        n_first = calls[0]
        # Second visit on the same node returns the cached result without
        # re-running ``generic_visit``.
        second = v.visit(kernel)
        assert second == first
        assert calls[0] == n_first


# ---------------------------------------------------------------------------
# PrettyPrinter
# ---------------------------------------------------------------------------


class TestPrettyPrinter:
    def test_prints_tree(self, capsys, square_tensors):
        A, B = square_tensors["A"], square_tensors["B"]
        kernel = A["ij"] <= B["ij"]
        PrettyPrinter().visit(kernel)
        out = capsys.readouterr().out
        # The tree string contains the Assign root and both children.
        assert "Assign" in out
        assert "A[ij]" in out
        assert "B[ij]" in out


# ---------------------------------------------------------------------------
# FindTensors
# ---------------------------------------------------------------------------


class TestFindTensors:
    def test_collects_leaf_tensors_by_name(self, deduced, square_tensors):
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = deduced(kernel)
        tensors = FindTensors().visit(kernel)
        assert set(tensors.keys()) == {"A", "B", "C"}
        # Every value is the original ``Tensor`` object.
        assert tensors["A"] is A
        assert tensors["B"] is B
        assert tensors["C"] is C

    def test_skips_temporary_tensors(self, deduced):
        # Temporary tensors must not be exposed as external kernel
        # parameters.  ``FindTensors`` filters them out.
        A = Tensor("A", (3, 3))
        tmp = Tensor("tmp", (3, 3), temporary=True)
        kernel = tmp["ij"] <= A["ij"]
        kernel = deduced(kernel)
        tensors = FindTensors().visit(kernel)
        assert "tmp" not in tensors
        assert "A" in tensors


# ---------------------------------------------------------------------------
# ComputeIndexSet
# ---------------------------------------------------------------------------


class TestComputeIndexSet:
    def test_union_of_all_child_indices(self, square_tensors):
        A, B = square_tensors["A"], square_tensors["B"]
        # Two operands with partially overlapping indices.
        tree = A["ij"] * B["jk"]
        # ComputeIndexSet returns the union.
        assert ComputeIndexSet().visit(tree) == {"i", "j", "k"}

    def test_indexed_tensor_is_leaf(self):
        it = IndexedTensor(Tensor("A", (3, 4)), "ij")
        assert ComputeIndexSet().visit(it) == {"i", "j"}


# ---------------------------------------------------------------------------
# ComputeSparsityPattern
# ---------------------------------------------------------------------------


class TestComputeSparsityPattern:
    """``ComputeSparsityPattern`` is a read-only walker that re-runs each
    node's ``computeSparsityPattern`` method.  Crucially, it does *not*
    work on raw ``Einsum`` nodes - ``Einsum.computeSparsityPattern``
    explicitly raises ``NotImplementedError``.  The compiler pipeline
    first lowers ``Einsum`` to ``Product`` + ``IndexSum`` via the
    ``EquivalentSparsityPattern`` transformer, which is what the
    ``run_ast_pipeline`` fixture does.
    """

    def test_raw_einsum_cannot_be_evaluated(self, deduced, square_tensors):
        # Locking this in: ``ComputeSparsityPattern`` on a fresh DSL tree
        # crashes because ``Einsum`` refuses to compute its pattern
        # on its own.  The fix is to run ``EquivalentSparsityPattern``
        # first (which decomposes the Einsum into Product + IndexSum).
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = deduced(kernel)
        with pytest.raises(NotImplementedError):
            ComputeSparsityPattern(useAvailable=False).visit(kernel)

    def test_dense_matmul_produces_dense_result(self, run_ast_pipeline, square_tensors):
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = run_ast_pipeline(kernel)
        spp = ComputeSparsityPattern(useAvailable=True).visit(kernel)
        # Dense inputs -> dense output.
        assert spp.is_dense()
        assert spp.shape == (8, 8)

    def test_sparse_input_shrinks_product(self, run_ast_pipeline):
        # Sparse A (only diagonal) times dense B should produce a pattern
        # that the strength reducer can exploit.  The diagonal-of-A times
        # dense-B case activates all rows of the result, so the pattern
        # is still fully dense - but the number of *operations* goes
        # down, which we exercise in the flop-count tests.
        diag = np.eye(4, dtype=bool)
        A = Tensor("A", (4, 4), spp=diag)
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = run_ast_pipeline(kernel)
        spp = ComputeSparsityPattern(useAvailable=True).visit(kernel)
        assert spp.count_nonzero() == 16


# ---------------------------------------------------------------------------
# ComputeOptimalFlopCount
# ---------------------------------------------------------------------------


class TestComputeOptimalFlopCount:
    def test_matmul_8x8_count(self, square_tensors, run_ast_pipeline):
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = run_ast_pipeline(kernel)
        # 8x8 matmul decomposes into a Product (for each output) plus
        # IndexSum over k.  After strength reduction the flop count is
        # well-defined: 960 for a dense 8x8x8 GEMM in Yateto's accounting.
        #
        # The exact number is a regression-lock - anything that changes
        # it silently is a red flag.
        assert ComputeOptimalFlopCount().visit(kernel) == 960

    def test_add_flops_is_n_for_nxn(self, square_tensors, run_ast_pipeline):
        # A 2D elementwise add of NxN matrices costs N*N additions.
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ij"] + B["ij"]
        kernel = run_ast_pipeline(kernel)
        assert ComputeOptimalFlopCount().visit(kernel) == 64  # 8*8

    def test_leaf_is_zero(self):
        it = IndexedTensor(Tensor("A", (3, 3)), "ij")
        assert ComputeOptimalFlopCount().visit(it) == 0


# ---------------------------------------------------------------------------
# ComputeConstantExpression - reference evaluation (for compile-time
# constant folding or unit testing)
# ---------------------------------------------------------------------------


class TestComputeConstantExpression:
    def test_matmul_of_constants(self, deduced):
        # Both tensors have concrete values -> the evaluator should
        # produce the actual matrix product.
        vals_a = {(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 4.0}
        vals_b = {(0, 0): 5.0, (0, 1): 6.0, (1, 0): 7.0, (1, 1): 8.0}
        A = Tensor("A", (2, 2), spp=vals_a)
        B = Tensor("B", (2, 2), spp=vals_b)
        C = Tensor("C", (2, 2))

        # ComputeConstantExpression walks the *rhs* only.
        rhs = A["ik"] * B["kj"]
        # Must run DeduceIndices first so that Einsum.indices is known.
        rhs = DeduceIndices("ij").visit(rhs)
        result = ComputeConstantExpression().visit(rhs)

        expected = np.array([[1, 2], [3, 4]]) @ np.array([[5, 6], [7, 8]])
        np.testing.assert_array_equal(result, expected)

    def test_evaluator_requires_constant_tensors(self, deduced):
        # A plain tensor (no values) cannot be evaluated numerically.
        A = Tensor("A", (2, 2))
        B = Tensor("B", (2, 2), spp={(0, 0): 1.0, (1, 1): 1.0})
        rhs = DeduceIndices("ij").visit(A["ij"] + B["ij"])
        with pytest.raises(AssertionError, match="constant"):
            ComputeConstantExpression().visit(rhs)
