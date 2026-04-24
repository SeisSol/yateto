"""
Tests for ``yateto.ast.node`` - the core AST.

The DSL turns Python expressions into a tree of ``Node`` subclasses using
operator overloading:

    C['ij'] <= A['ik'] * B['kj']

becomes

    Assign(
        IndexedTensor(C, "ij"),
        Einsum(
            IndexedTensor(A, "ik"),
            IndexedTensor(B, "kj"),
        ),
    )

This module checks that:

* the DSL really produces the expected tree shape,
* the tree's invariants (no nested ``ScalarMultiplication``, ``Assign`` lhs
  must be an ``IndexedTensor``, associative operators absorb their peers, ...)
  are enforced,
* the per-node sparsity-pattern / flop-count helpers are correct,
* the specialised nodes used by the middle-end (``Product``, ``IndexSum``,
  ``Contraction``, ``LoopOverGEMM``, ``FusedGEMMs``, ``SliceView``,
  ``Permute``, ``Broadcast``) behave as advertised.
"""
from __future__ import annotations

import pytest

from yateto import Tensor
from yateto.ast.indices import Indices
from yateto.ast.node import (
    Add,
    Assign,
    BinOp,
    Broadcast,
    Contraction,
    Einsum,
    FusedGEMMs,
    IndexedTensor,
    IndexSum,
    LoopOverGEMM,
    Op,
    Permute,
    Product,
    ScalarMultiplication,
    SliceView,
    UnaryOp,
)
from yateto.memory import DenseMemoryLayout
from yateto import aspp


# ---------------------------------------------------------------------------
# IndexedTensor - leaves of the tree
# ---------------------------------------------------------------------------


class TestIndexedTensor:
    def test_construction_via_getitem(self):
        A = Tensor("A", (3, 4))
        it = A["ij"]
        assert isinstance(it, IndexedTensor)
        assert it.tensor is A
        assert str(it.indices) == "ij"
        assert it.indices.shape() == (3, 4)

    def test_index_arity_must_match_tensor_rank(self):
        A = Tensor("A", (3, 4))
        with pytest.raises(AssertionError):
            A["ijk"]  # tensor is rank 2

    def test_name_delegates_to_tensor(self):
        A = Tensor("A", (2, 2))
        assert A["ij"].name() == "A"

    def test_nonZeroFlops_is_zero(self):
        # Reading a leaf costs nothing.
        assert Tensor("A", (2, 2))["ij"].nonZeroFlops() == 0


# ---------------------------------------------------------------------------
# Einsum - via ``*``
# ---------------------------------------------------------------------------


class TestEinsumBuilding:
    def test_mul_creates_einsum(self, square_tensors):
        A, B = square_tensors["A"], square_tensors["B"]
        expr = A["ik"] * B["kj"]
        assert isinstance(expr, Einsum)
        assert len(expr) == 2

    def test_mul_is_left_associative_and_flattens(self, square_tensors):
        # ``a * b * c`` should produce a single ``Einsum(a, b, c)`` node,
        # not ``Einsum(Einsum(a,b), c)``.  This is what ``_binOp`` guarantees.
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        expr = A["ij"] * B["jk"] * C["kl"]
        assert isinstance(expr, Einsum)
        assert len(expr) == 3

    def test_mul_flattens_right_side_too(self, square_tensors):
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        # (A) * (B * C) should also flatten.
        right = B["jk"] * C["kl"]
        expr = A["ij"] * right
        assert isinstance(expr, Einsum)
        assert len(expr) == 3


# ---------------------------------------------------------------------------
# ScalarMultiplication - via ``*`` with a float/int
# ---------------------------------------------------------------------------


class TestScalarMultiplication:
    def test_lhs_scalar(self, square_tensors):
        A = square_tensors["A"]
        expr = 2.0 * A["ij"]
        assert isinstance(expr, ScalarMultiplication)
        assert expr.is_constant()
        assert expr.scalar() == 2.0

    def test_rhs_scalar(self, square_tensors):
        A = square_tensors["A"]
        expr = A["ij"] * 2.0
        assert isinstance(expr, ScalarMultiplication)
        assert expr.scalar() == 2.0

    def test_negation(self, square_tensors):
        A = square_tensors["A"]
        expr = -A["ij"]
        assert isinstance(expr, ScalarMultiplication)
        assert expr.scalar() == -1.0

    def test_nested_scalar_mul_rejected(self, square_tensors):
        # ``k1 * (k2 * A)`` is disallowed by design - the user must
        # pre-fold scalars into a single coefficient.  This keeps the AST
        # unambiguous and the code generator simple.
        A = square_tensors["A"]
        with pytest.raises(ValueError, match="Multiple multiplications"):
            2.0 * (3.0 * A["ij"])

    def test_scalar_times_einsum_preserves_einsum_child(self, square_tensors):
        A, B = square_tensors["A"], square_tensors["B"]
        expr = 2.0 * (A["ik"] * B["kj"])
        assert isinstance(expr, ScalarMultiplication)
        # The term inside is an Einsum, not a ScalarMultiplication.
        assert isinstance(expr.term(), Einsum)

    def test_nonZeroFlops_is_zero_for_pm_one(self, square_tensors, run_ast_pipeline):
        A = square_tensors["A"]
        B = square_tensors["B"]
        expr = Assign(A["ij"], -B["ij"])
        ast = run_ast_pipeline(expr)
        # Find the scalar-mul child (the rhs) and check its flops.
        rhs = ast.rightTerm()
        # ``-1.0`` is a free sign flip.
        assert isinstance(rhs, ScalarMultiplication)
        assert rhs.nonZeroFlops() == 0


# ---------------------------------------------------------------------------
# Add - via ``+``
# ---------------------------------------------------------------------------


class TestAddBuilding:
    def test_add_creates_add_node(self, square_tensors):
        A, B = square_tensors["A"], square_tensors["B"]
        expr = A["ij"] + B["ij"]
        assert isinstance(expr, Add)

    def test_add_flattens(self, square_tensors):
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        expr = A["ij"] + B["ij"] + C["ij"]
        assert isinstance(expr, Add)
        assert len(expr) == 3

    def test_sub_via_neg(self, square_tensors):
        A, B = square_tensors["A"], square_tensors["B"]
        expr = A["ij"] - B["ij"]
        # ``a - b`` == ``a + (-b)``, i.e. an Add with a ScalarMul(-1) child.
        assert isinstance(expr, Add)
        assert isinstance(expr[1], ScalarMultiplication)
        assert expr[1].scalar() == -1.0

    def test_add_with_non_node_raises(self, square_tensors):
        A = square_tensors["A"]
        with pytest.raises(ValueError, match="Cannot add"):
            A["ij"] + 5


# ---------------------------------------------------------------------------
# Assign - via ``<=``
# ---------------------------------------------------------------------------


class TestAssign:
    def test_assign_builds_kernel(self, square_tensors):
        A, B = square_tensors["A"], square_tensors["B"]
        kernel = A["ij"] <= B["ij"]
        assert isinstance(kernel, Assign)
        assert len(kernel) == 2

    def test_assign_lhs_must_be_indexed_tensor(self, square_tensors):
        A, B, C = square_tensors["A"], square_tensors["B"], square_tensors["C"]
        # The invariant ("first child of Assign must be an IndexedTensor")
        # is enforced inside ``Assign.setChildren`` - i.e. when a later
        # transformer pass rewrites the tree - not in the constructor.
        # The DSL's ``__le__`` calls the constructor directly, so this
        # expression is *accepted* at build time; the check fires only
        # when a transformer tries to re-install children.
        bad = (A["ij"] * B["ij"]) <= C["ij"]
        assert isinstance(bad, Assign)
        with pytest.raises(ValueError, match="must be an IndexedTensor"):
            bad.setChildren([A["ij"] * B["ij"], C["ij"]])

    def test_assign_flops_are_zero(self, square_tensors):
        A, B = square_tensors["A"], square_tensors["B"]
        kernel = A["ij"] <= B["ij"]
        assert kernel.nonZeroFlops() == 0


# ---------------------------------------------------------------------------
# SliceView - ``A['ij'].subslice(...)`` / ``.subselect(...)``
# ---------------------------------------------------------------------------


class TestSliceView:
    def test_subslice_is_a_sliceview(self, square_tensors):
        A = square_tensors["A"]
        sv = A["ij"].subslice("i", 1, 4)
        assert isinstance(sv, SliceView)
        assert sv.index == "i"
        assert sv.start == 1
        assert sv.end == 4

    def test_subselect_is_single_index_slice(self, square_tensors):
        A = square_tensors["A"]
        sv = A["ij"].subselect("i", 2)
        assert isinstance(sv, SliceView)
        # subselect(i, 2) should cover [2, 3)
        assert sv.start == 2
        assert sv.end == 3

    def test_name_delegates_through_the_view(self, square_tensors):
        A = square_tensors["A"]
        assert A["ij"].subslice("i", 0, 3).name() == "A"

    def test_viewed_unwraps_to_indexed_tensor(self, square_tensors):
        A = square_tensors["A"]
        sv = A["ij"].subslice("i", 0, 3)
        inner = sv.viewed()
        assert isinstance(inner, IndexedTensor)
        assert inner.tensor is A


# ---------------------------------------------------------------------------
# Product / IndexSum / Contraction - the "lowered" Einsum
# ---------------------------------------------------------------------------


class TestLoweredNodes:
    """After ``FindContractions``, ``Einsum`` is decomposed into
    ``Product`` + ``IndexSum`` (or ``Contraction`` for binary cases).
    The tests below construct them directly to pin down their contracts.
    """

    def test_product_merges_indices(self):
        a = IndexedTensor(Tensor("A", (3, 4)), "ij")
        b = IndexedTensor(Tensor("B", (4, 5)), "jk")
        prod = Product(a, b)
        # Product keeps every dimension, including the shared "j".
        assert set(prod.indices) == {"i", "j", "k"}

    def test_product_rejects_mismatching_shared_dim(self):
        a = IndexedTensor(Tensor("A", (3, 4)), "ij")
        b = IndexedTensor(Tensor("B", (9, 5)), "jk")  # j=9 vs j=4
        with pytest.raises(AssertionError):
            Product(a, b)

    def test_indexsum_drops_one_index(self):
        a = IndexedTensor(Tensor("A", (3, 4)), "ij")
        s = IndexSum(a, "j")
        assert str(s.indices) == "i"
        # The stored sumIndex knows its size.
        assert s.sumIndex().indexSize("j") == 4

    def test_contraction_matmul(self):
        a = IndexedTensor(Tensor("A", (3, 4)), "ij")
        b = IndexedTensor(Tensor("B", (4, 5)), "jk")
        c = Contraction(
            indices=Indices("ik", (3, 5)),
            lTerm=a,
            rTerm=b,
            sumIndices={"j"},
        )
        assert set(c.indices) == {"i", "k"}
        assert c.sumIndices == {"j"}


# ---------------------------------------------------------------------------
# LoopOverGEMM - the node the Codegen actually emits
# ---------------------------------------------------------------------------


class TestLoopOverGEMM:
    def _simple_gemm(self):
        a = IndexedTensor(Tensor("A", (3, 4)), "ij")
        b = IndexedTensor(Tensor("B", (4, 5)), "jk")
        indices = Indices("ik", (3, 5))
        return LoopOverGEMM(
            indices=indices,
            aTerm=a,
            bTerm=b,
            m=Indices("i", (3,)),
            n=Indices("k", (5,)),
            k=Indices("j", (4,)),
        )

    def test_pure_gemm_detection(self):
        log = self._simple_gemm()
        assert log.is_pure_gemm()

    def test_non_pure_gemm_has_extra_dim(self):
        # An outer-indexed third dimension breaks the ``is_pure_gemm`` test.
        a = IndexedTensor(Tensor("A", (3, 4, 2)), "ijl")
        b = IndexedTensor(Tensor("B", (4, 5)), "jk")
        indices = Indices("ikl", (3, 5, 2))
        log = LoopOverGEMM(
            indices=indices,
            aTerm=a,
            bTerm=b,
            m=Indices("i", (3,)),
            n=Indices("k", (5,)),
            k=Indices("j", (4,)),
        )
        assert not log.is_pure_gemm()

    def test_trans_flags_for_simple_gemm(self):
        # Layout: A is (i,j), B is (j,k), result is (i,k).  Both operands
        # have the GEMM-friendly order, so no transposes are needed.
        log = self._simple_gemm()
        assert log.transA() is False
        assert log.transB() is False

    def test_trans_flag_when_k_precedes_m(self):
        # A uses ``ji`` instead of ``ij`` - k-index precedes m-index, so
        # LoG must request a transpose on A.
        a = IndexedTensor(Tensor("A", (4, 3)), "ji")
        b = IndexedTensor(Tensor("B", (4, 5)), "jk")
        indices = Indices("ik", (3, 5))
        log = LoopOverGEMM(
            indices=indices,
            aTerm=a,
            bTerm=b,
            m=Indices("i", (3,)),
            n=Indices("k", (5,)),
            k=Indices("j", (4,)),
        )
        assert log.transA() is True
        assert log.transB() is False


# ---------------------------------------------------------------------------
# FusedGEMMs - list-like container of LoGs
# ---------------------------------------------------------------------------


class TestFusedGEMMs:
    def _log(self):
        a = IndexedTensor(Tensor("A", (3, 4)), "ij")
        b = IndexedTensor(Tensor("B", (4, 5)), "jk")
        return LoopOverGEMM(
            indices=Indices("ik", (3, 5)),
            aTerm=a, bTerm=b,
            m=Indices("i", (3,)),
            n=Indices("k", (5,)),
            k=Indices("j", (4,)),
        )

    def test_is_empty_on_construction(self):
        fg = FusedGEMMs()
        assert fg.is_empty()

    def test_add_accepts_only_log(self):
        fg = FusedGEMMs()
        fg.add(self._log())
        assert not fg.is_empty()
        assert len(fg.get_children()) == 1
        # Non-LoG child -> rejected.
        with pytest.raises(ValueError, match="expected LoopOverGEMM"):
            fg.add(IndexedTensor(Tensor("x", (2, 2)), "ij"))


# ---------------------------------------------------------------------------
# Permute / Broadcast - required at LoG time to make indices line up
# ---------------------------------------------------------------------------


class TestPermuteBroadcast:
    def test_permute_requires_same_index_set(self):
        a = IndexedTensor(Tensor("A", (3, 4)), "ij")
        # Permute cannot introduce or remove indices; it must be a pure reorder.
        good = Permute(a, Indices("ji", (4, 3)))
        assert set(good.indices) == {"i", "j"}

    def test_broadcast_adds_indices(self):
        a = IndexedTensor(Tensor("A", (3,)), "i")
        bcst = Broadcast(a, Indices("ij", (3, 4)))
        assert set(bcst.indices) == {"i", "j"}

    def test_permute_nonZeroFlops_is_zero(self):
        a = IndexedTensor(Tensor("A", (3, 4)), "ij")
        p = Permute(a, Indices("ji", (4, 3)))
        # Data-movement-only ops are free in the flop accounting.
        assert p.nonZeroFlops() == 0


# ---------------------------------------------------------------------------
# Common node invariants via ABCs
# ---------------------------------------------------------------------------


class TestNodeAbstractInvariants:
    def test_unaryop_term_is_first_child(self):
        a = IndexedTensor(Tensor("A", (3, 4)), "ij")
        s = IndexSum(a, "j")  # a UnaryOp
        assert s.term() is s[0]
        assert isinstance(s, UnaryOp)

    def test_binop_left_and_right_term(self):
        a = IndexedTensor(Tensor("A", (3, 4)), "ij")
        b = IndexedTensor(Tensor("B", (4, 5)), "jk")
        p = Product(a, b)  # BinOp
        assert p.leftTerm() is p[0]
        assert p.rightTerm() is p[1]
        assert isinstance(p, BinOp)

    def test_op_is_iterable_over_children(self, square_tensors):
        A, B = square_tensors["A"], square_tensors["B"]
        expr = A["ij"] + B["ij"]
        assert list(expr) == [expr[0], expr[1]]
