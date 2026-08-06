"""
Tests for correct
``Add.nonZeroFlops`` accounting under index permutation and
broadcasting.

The previous bug
----------------
``Add.nonZeroFlops`` summed the *raw* ``count_nonzero`` of each
summand's equivalent sparsity pattern:

.. code-block:: python

    for child in self:
        nzFlops += child.eqspp().count_nonzero()
    return nzFlops - self.eqspp().count_nonzero()

That is wrong whenever a summand has a smaller index set than the Add
node itself.  In ``B[ij] <= A[i] + B[ij]`` the summand ``A[i]`` is
broadcast over ``j`` at runtime, so each of its ``N`` non-zeros
participates in ``M`` additions.  The pre-fix code only counts ``N``
operations for ``A`` instead of ``N*M``, so the reported flop count is
off by a factor that grows with the broadcast size.

The fix
-------
Permute and broadcast each summand to the Add's index set before
counting non-zeros:

.. code-block:: python

    for child in self:
        permuted = self.broadcast(child.indices,
                                  self.permute(child.indices, child.eqspp(), False))
        nzFlops += permuted.count_nonzero()

After the fix, ``A[i] + B[ij]`` correctly reports ``N*M`` adds
(everything in the result, minus the ``N*M`` "first adds against zero"
that initialise the result, which subtraction the old code already
got right).

The tests below
---------------
* ``TestAddNonZeroFlopsBroadcast`` pins down the correct behavior
  after the fix.
* ``TestAddNonZeroFlopsRegression`` verifies that *non*-broadcasting
  cases (which were correct pre-fix) keep returning the same
  numbers.  This guards against the fix accidentally changing the
  answer for plain element-wise additions.
* ``TestAddNonZeroFlopsSparse`` exercises the fix on sparse summands
  with permutation, where the count_nonzero is invariant under
  transpose but the path through ``self.permute`` / ``self.broadcast``
  is not - so any off-by-one in the new code path would surface.
"""
from __future__ import annotations

import numpy as np
import pytest

from yateto import Tensor
from yateto.ast.cost import BoundingBoxCostEstimator
from yateto.ast.transformer import (
    DeduceIndices,
    EquivalentSparsityPattern,
    SetSparsityPattern,
    StrengthReduction,
)
from yateto.ast.visitor import ComputeOptimalFlopCount


def _run(kernel):
    kernel = DeduceIndices().visit(kernel)
    kernel = EquivalentSparsityPattern().visit(kernel)
    kernel = StrengthReduction(BoundingBoxCostEstimator).visit(kernel)
    kernel = SetSparsityPattern().visit(kernel)
    return kernel


# ---------------------------------------------------------------------------
# The actual bug: broadcasting summand
# ---------------------------------------------------------------------------


class TestAddNonZeroFlopsBroadcast:
    """The flagship reproducer: a vector summand broadcast over a
    matrix.
    """

    @pytest.mark.parametrize("N,M", [(4, 3), (8, 8), (5, 11), (1, 7)])
    def test_vector_plus_matrix_counts_broadcast_adds(self, N, M):
        # B[ij] = A[i] + B[ij]  -- A is broadcast over j.
        # Adding two dense quantities into a dense result of N*M
        # entries means N*M additions overall.  Of those, N*M are
        # "first adds against zero" that initialise the destination
        # and don't count.  So:
        #
        #   summand_count = N*M (A bcst) + N*M (B) = 2*N*M
        #   minus initialiser = N*M
        #   -> N*M adds
        #
        A = Tensor("A", (N,))
        B = Tensor("B", (N, M))
        kernel = _run(B["ij"] <= A["i"] + B["ij"])
        assert ComputeOptimalFlopCount().visit(kernel) == N * M

    def test_scalar_plus_matrix_counts_broadcast(self):
        # A scalar summand should be broadcast over both i and j.
        N, M = 4, 5
        A = Tensor("A", ())
        B = Tensor("B", (N, M))
        kernel = _run(B["ij"] <= A[""] + B["ij"])
        assert ComputeOptimalFlopCount().visit(kernel) == N * M

    def test_three_way_with_broadcast(self):
        # C[ij] = A[i] + B[ij] + D[j] - two broadcast summands and one
        # full-rank summand.  Each broadcast summand should contribute
        # N*M, so the total non-zero adds is
        #     3*N*M (sum of broadcasted summands) - N*M (init)  = 2*N*M.
        N, M = 3, 4
        A = Tensor("A", (N,))
        B = Tensor("B", (N, M))
        D = Tensor("D", (M,))
        C = Tensor("C", (N, M))
        kernel = _run(C["ij"] <= A["i"] + B["ij"] + D["j"])
        assert ComputeOptimalFlopCount().visit(kernel) == 2 * N * M


# ---------------------------------------------------------------------------
# Non-broadcasting cases: regression - the fix must not change them
# ---------------------------------------------------------------------------


class TestAddNonZeroFlopsRegression:
    """When all summands have the *same* index set as the Add node, the
    fix should be a no-op: the broadcast collapses to the identity, and
    the permute is a possibly-non-trivial transpose that count_nonzero
    is invariant under.  These tests guard against the fix accidentally
    changing the count in these cases.
    """

    @pytest.mark.parametrize("N", [4, 8, 16])
    def test_dense_elementwise_add_two_summands(self, N):
        # C[ij] = A[ij] + B[ij].  Result is dense, so:
        #   2*N*N (sum of count_nonzero) - N*N (init) = N*N adds.
        A = Tensor("A", (N, N))
        B = Tensor("B", (N, N))
        C = Tensor("C", (N, N))
        kernel = _run(C["ij"] <= A["ij"] + B["ij"])
        assert ComputeOptimalFlopCount().visit(kernel) == N * N

    def test_dense_elementwise_add_three_summands(self):
        # 3*N*N - N*N = 2*N*N.
        N = 5
        A = Tensor("A", (N, N))
        B = Tensor("B", (N, N))
        C = Tensor("C", (N, N))
        D = Tensor("D", (N, N))
        kernel = _run(D["ij"] <= A["ij"] + B["ij"] + C["ij"])
        assert ComputeOptimalFlopCount().visit(kernel) == 2 * N * N

    def test_pure_assign_is_zero_flops(self):
        # ``B = A`` is data movement, not arithmetic.
        N = 6
        A = Tensor("A", (N, N))
        B = Tensor("B", (N, N))
        kernel = _run(B["ij"] <= A["ij"])
        assert ComputeOptimalFlopCount().visit(kernel) == 0

    def test_subtraction_via_neg(self):
        # ``A - B`` lowers to ``Add(A, -B)`` (a ScalarMultiplication
        # with factor -1.0).  The flop count is still N*N additions.
        N = 4
        A = Tensor("A", (N, N))
        B = Tensor("B", (N, N))
        C = Tensor("C", (N, N))
        kernel = _run(C["ij"] <= A["ij"] - B["ij"])
        assert ComputeOptimalFlopCount().visit(kernel) == N * N


# ---------------------------------------------------------------------------
# Sparse summands - the new path uses permute/broadcast on the eqspp,
# so any off-by-one in that pipeline shows up here.
# ---------------------------------------------------------------------------


class TestAddNonZeroFlopsSparse:
    """``count_nonzero`` is invariant under transposition, so a permuted
    sparse summand must yield the same flop count as the un-permuted
    one.  But the new code path goes through ``self.permute`` ->
    ``self.broadcast`` ; if either step mishandles the sparsity pattern,
    these tests catch it.
    """

    def test_sparse_plus_sparse_same_layout(self):
        # Both summands are diagonal NxN: 2*N - N (init) = N adds.
        N = 6
        diag = np.eye(N, dtype=bool)
        A = Tensor("A", (N, N), spp=diag)
        B = Tensor("B", (N, N), spp=diag)
        C = Tensor("C", (N, N))
        kernel = _run(C["ij"] <= A["ij"] + B["ij"])
        assert ComputeOptimalFlopCount().visit(kernel) == N

    def test_sparse_plus_sparse_with_transposed_summand(self):
        # B[ij] gets summed against A[ji].  Both are diagonals, and the
        # diagonal of the transpose is the same set, so the result is
        # also the diagonal -> N adds, just like the non-transposed case.
        N = 5
        diag = np.eye(N, dtype=bool)
        A = Tensor("A", (N, N), spp=diag)
        B = Tensor("B", (N, N), spp=diag)
        C = Tensor("C", (N, N))
        kernel = _run(C["ij"] <= A["ij"] + B["ji"])
        # If permute-on-eqspp ever silently corrupts the result, this
        # number changes.  N is the regression-locked correct count.
        assert ComputeOptimalFlopCount().visit(kernel) == N

    def test_sparse_vector_broadcast_into_dense_matrix(self):
        # A is a sparse 1-d tensor (only entries 0 and 2 set), to be
        # broadcast over j into a 4x3 dense result B.
        N, M = 4, 3
        spp = np.zeros((N,), dtype=bool)
        spp[[0, 2]] = True
        A = Tensor("A", (N,), spp=spp)
        B = Tensor("B", (N, M))
        kernel = _run(B["ij"] <= A["i"] + B["ij"])
        # After broadcast, A's pattern covers rows {0, 2} entirely,
        # i.e. 2*M = 6 non-zeros.  Plus B's N*M = 12.  Minus the
        # result's N*M = 12 (dense) initialiser.  -> 6 adds.
        assert ComputeOptimalFlopCount().visit(kernel) == 2 * M
