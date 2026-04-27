"""
Tests for ``yateto.ast.transformer`` - the AST rewriting passes.

Transformers are the compiler's middle-end.  Each one takes an AST,
walks it, possibly substitutes nodes with different ones, and returns
the new root.  They must run in a prescribed order - see
``generator.py::Kernel.prepareUntilCodeGen`` - and many of them only
make sense once the preceding ones have run.

This file checks each pass in isolation (what invariant does it
establish? what does it assume?) plus a few multi-step compositions
that exercise the real ordering.
"""
from __future__ import annotations

import numpy as np
import pytest

from yateto import Tensor
from yateto.ast.cost import BoundingBoxCostEstimator, ShapeCostEstimator
from yateto.ast.indices import Indices
from yateto.ast.node import (
    Add,
    Assign,
    Contraction,
    Einsum,
    IndexedTensor,
    IndexSum,
    Product,
    ScalarMultiplication,
)
from yateto.ast.transformer import (
    ComputeMemoryLayout,
    DeduceIndices,
    EquivalentSparsityPattern,
    FindContractions,
    ImplementContractions,
    SetSparsityPattern,
    StrengthReduction,
)
from yateto.ast.visitor import FindIndexPermutations
from yateto.ast.transformer import SelectIndexPermutations


# ---------------------------------------------------------------------------
# DeduceIndices - the mandatory first pass
# ---------------------------------------------------------------------------


class TestDeduceIndices:
    def test_matmul_indices_are_deduced(self, square_tensors):
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ik"] * B["kj"]
        # Before the pass, non-leaf nodes have ``indices = None``.
        assert kernel.indices is None
        assert kernel.rightTerm().indices is None

        kernel = DeduceIndices().visit(kernel)
        # Afterwards, indices are set all the way down.
        assert str(kernel.indices) == "ij"
        assert str(kernel.rightTerm().indices) == "ij"

    def test_add_requires_same_sizes(self):
        # Two addends sharing letters but different sizes -> error.
        A = Tensor("A", (3, 4))
        B = Tensor("B", (3, 5))  # j has size 5 vs A's j=4
        C = Tensor("C", (3, 4))
        with pytest.raises(AssertionError):
            # mergeStrict inside DeduceIndices.visit_Add catches this.
            DeduceIndices().visit(C["ij"] <= A["ij"] + B["ij"])

    def test_lhs_rhs_index_mismatch_is_flagged(self, square_tensors):
        # LHS asks for an index that doesn't appear on RHS.
        A, B = square_tensors["A"], square_tensors["B"]
        kernel = A["ij"] <= B["jk"]
        # "j" is unbound on the rhs in the context of lhs "ij" - there
        # is no "i" on the rhs, so DeduceIndices must reject the kernel.
        with pytest.raises(ValueError):
            DeduceIndices().visit(kernel)

    def test_unbound_indices_on_lhs_rejected(self, square_tensors):
        # A free index on the rhs that doesn't appear on the lhs is a
        # contraction if the tree is an Einsum - otherwise it's an error.
        A = Tensor("A", (3,))
        B = Tensor("B", (3,))
        C = Tensor("C", ())  # scalar lhs
        # C[''] = A['i'] * B['i']  -- this is a dot product, indices must
        # be bound, so DeduceIndices should accept it.
        kernel = C[""] <= A["i"] * B["i"]
        kernel = DeduceIndices().visit(kernel)
        assert str(kernel.indices) == ""

    def test_target_indices_force_permutation(self, square_tensors):
        # Passing an explicit target permutes the root.
        A, B = square_tensors["A"], square_tensors["B"]
        kernel = A["ji"] <= B["ij"]
        kernel = DeduceIndices("ji").visit(kernel)
        assert str(kernel.indices) == "ji"


# ---------------------------------------------------------------------------
# EquivalentSparsityPattern - computes every node's eqspp
# ---------------------------------------------------------------------------


class TestEquivalentSparsityPattern:
    def test_sets_eqspp_on_every_node(self, deduced, square_tensors):
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = deduced(kernel)
        # Before the pass, eqspp is None.
        assert kernel.eqspp() is None
        kernel = EquivalentSparsityPattern().visit(kernel)
        # After the pass, every node - including the leaves - has an eqspp.
        assert kernel.eqspp() is not None
        assert kernel.rightTerm().eqspp() is not None
        for child in kernel.rightTerm():
            assert child.eqspp() is not None

    def test_dense_matmul_eqspp_is_dense(self, deduced, square_tensors):
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = deduced(kernel)
        kernel = EquivalentSparsityPattern().visit(kernel)
        assert kernel.eqspp().is_dense()

    def test_sparse_matmul_shrinks_eqspp(self, deduced):
        # Matmul of two diagonal matrices has a diagonal pattern.
        diag = np.eye(4, dtype=bool)
        A = Tensor("A", (4, 4), spp=diag)
        B = Tensor("B", (4, 4), spp=diag)
        C = Tensor("C", (4, 4))
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = deduced(kernel)
        kernel = EquivalentSparsityPattern().visit(kernel)
        # D * D = D (diagonal) -> 4 nonzeros, not 16.
        assert kernel.eqspp().count_nonzero() == 4


# ---------------------------------------------------------------------------
# StrengthReduction - lowers Einsum to Product(...) / IndexSum(...)
# ---------------------------------------------------------------------------


class TestStrengthReduction:
    def test_einsum_disappears(self, deduced, square_tensors):
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = deduced(kernel)
        kernel = EquivalentSparsityPattern().visit(kernel)
        kernel = StrengthReduction(BoundingBoxCostEstimator).visit(kernel)

        # Einsum is gone; the rhs is now a Product-under-IndexSum tree.
        def has(cls, node):
            if isinstance(node, cls):
                return True
            return any(has(cls, c) for c in node)

        assert not has(Einsum, kernel)
        assert has(Product, kernel)
        assert has(IndexSum, kernel)

    def test_costEstimator_is_a_class_not_an_instance(self, deduced, square_tensors):
        # ``StrengthReduction`` calls ``self._costEstimator()`` internally
        # - i.e. it instantiates a fresh estimator per Einsum node.  This
        # is a silent trap when users pass an already-constructed one.
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = deduced(C["ij"] <= A["ik"] * B["kj"])
        kernel = EquivalentSparsityPattern().visit(kernel)
        # Passing an instance fails with TypeError (not callable).
        with pytest.raises(TypeError):
            StrengthReduction(BoundingBoxCostEstimator()).visit(kernel)

    def test_ternary_product_is_strength_reduced(self, deduced):
        # a * b * c should be ordered by the cost estimator, not flattened
        # into a single big product.  The exact order depends on sizes.
        A = Tensor("A", (2, 3))
        B = Tensor("B", (3, 4))
        C = Tensor("C", (4, 5))
        D = Tensor("D", (2, 5))
        kernel = D["il"] <= A["ij"] * B["jk"] * C["kl"]
        kernel = deduced(kernel)
        kernel = EquivalentSparsityPattern().visit(kernel)
        kernel = StrengthReduction(BoundingBoxCostEstimator).visit(kernel)
        # The result should be a binary tree of Products + IndexSums
        # (i.e. Einsum has been fully split into pairwise GEMMs).
        from yateto.ast.node import Einsum as _E
        def has_einsum(node):
            if isinstance(node, _E):
                return True
            return any(has_einsum(c) for c in node)
        assert not has_einsum(kernel)


# ---------------------------------------------------------------------------
# FindContractions - fuses Product+IndexSum into a single Contraction node
# ---------------------------------------------------------------------------


class TestFindContractions:
    def test_matmul_becomes_contraction(self, deduced, square_tensors):
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = deduced(kernel)
        kernel = EquivalentSparsityPattern().visit(kernel)
        kernel = StrengthReduction(BoundingBoxCostEstimator).visit(kernel)
        kernel = FindContractions().visit(kernel)

        # The rhs should now be a single Contraction node with sumIndices={"k"}.
        rhs = kernel.rightTerm()
        assert isinstance(rhs, Contraction)
        assert rhs.sumIndices == {"k"}
        assert set(rhs.indices) == {"i", "j"}

    def test_dot_product_becomes_contraction(self, deduced):
        # Vector dot-product: scalar = sum_i A[i] * B[i]
        A = Tensor("A", (5,))
        B = Tensor("B", (5,))
        C = Tensor("C", ())
        kernel = C[""] <= A["i"] * B["i"]
        kernel = deduced(kernel)
        kernel = EquivalentSparsityPattern().visit(kernel)
        kernel = StrengthReduction(BoundingBoxCostEstimator).visit(kernel)
        kernel = FindContractions().visit(kernel)
        rhs = kernel.rightTerm()
        assert isinstance(rhs, Contraction)
        assert rhs.sumIndices == {"i"}


# ---------------------------------------------------------------------------
# SetSparsityPattern
# ---------------------------------------------------------------------------


class TestSetSparsityPattern:
    def test_populates_eqspp_bottom_up(self, deduced, square_tensors):
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = deduced(kernel)
        kernel = EquivalentSparsityPattern().visit(kernel)
        kernel = StrengthReduction(BoundingBoxCostEstimator).visit(kernel)
        # SetSparsityPattern re-computes eqspps using the concrete tree
        # (rather than equivalent patterns).  After this, nonZeroFlops
        # must not crash.
        kernel = SetSparsityPattern().visit(kernel)
        assert kernel.eqspp() is not None

    def test_indexed_tensor_eqspp_is_preserved(self, square_tensors):
        # SetSparsityPattern treats IndexedTensor as a no-op and does not
        # overwrite any existing eqspp.
        A = Tensor("A", (3, 3))
        it = IndexedTensor(A, "ij")
        from yateto.aspp import dense
        it.setEqspp(dense((3, 3)))
        SetSparsityPattern().visit_IndexedTensor(it)
        assert it.eqspp() is not None


# ---------------------------------------------------------------------------
# ComputeMemoryLayout
# ---------------------------------------------------------------------------


class TestComputeMemoryLayout:
    def test_assigns_memory_layouts(self, deduced, square_tensors, arch):
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = deduced(kernel)
        kernel = EquivalentSparsityPattern().visit(kernel)
        kernel = StrengthReduction(BoundingBoxCostEstimator).visit(kernel)
        kernel = FindContractions().visit(kernel)
        kernel = ComputeMemoryLayout().visit(kernel)
        # Every op node now has a memory layout.
        assert kernel.rightTerm().memoryLayout() is not None


# ---------------------------------------------------------------------------
# SelectIndexPermutations + ImplementContractions -> LoopOverGEMM
# ---------------------------------------------------------------------------


class TestLowerToLoG:
    def test_contraction_becomes_log(self, deduced, square_tensors, arch):
        from yateto.ast.node import LoopOverGEMM
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        kernel = C["ij"] <= A["ik"] * B["kj"]
        kernel = deduced(kernel)
        kernel = EquivalentSparsityPattern().visit(kernel)
        kernel = StrengthReduction(BoundingBoxCostEstimator).visit(kernel)
        kernel = FindContractions().visit(kernel)
        kernel = ComputeMemoryLayout().visit(kernel)
        variants = FindIndexPermutations().visit(kernel)
        kernel = SelectIndexPermutations(variants).visit(kernel)
        kernel = ImplementContractions().visit(kernel)

        # The Contraction node is replaced with a LoopOverGEMM node.
        rhs = kernel.rightTerm()
        assert isinstance(rhs, LoopOverGEMM)
        # The LoG knows which m/n/k dimensions it covers.
        assert rhs.is_pure_gemm()
