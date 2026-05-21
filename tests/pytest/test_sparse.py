"""
Tests for native
support for sparse-storage memory layouts on tensors.

How sparsity is encoded
-----------------------
Two concrete ``MemoryLayout`` subclasses
ship in ``yateto.memory``:

* ``CSCMemoryLayout``       - 2-D compressed-sparse-column storage
  (``rowIndex`` + ``colPointer`` arrays + a contiguous values vector).
* ``PatternMemoryLayout``   - n-D positional layout backed by a
  per-entry index pattern.  Suitable for arbitrary-rank tensors.

Both expose ``isSparse() == True`` and a ``requiredReals()`` that is
the actual non-zero count, not the bounding-box volume.  ``Tensor``
uses a ``memoryLayoutClass=...`` parameter so users can opt into the
sparse layouts:

.. code-block:: python

    P = Tensor("P", (4, 4),
               spp=np.eye(4, dtype=bool),
               memoryLayoutClass=PatternMemoryLayout)

What this file checks
---------------------
* The layout classes are constructible and obey the
  ``MemoryLayout`` ABC contract.
* The constraints that the constructors document (CSC: rank 2 only;
  ``requiredReals`` = nnz; ``isSparse`` is True; addresses are valid
  only inside the bounding box) all hold.
* Specific numerical addresses for canonical patterns (a diagonal,
  a "checkerboard", a single column).
* The ``Tensor`` API plumbs ``memoryLayoutClass`` correctly and
  accepts both ``CSCMemoryLayout`` and ``PatternMemoryLayout``.
* End-to-end pipeline smoke: every kernel from ``tests/code-gen/sparsity.py``
  prepares without error.
"""
from __future__ import annotations

import os
import re
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helper: a canonical 4x4 diagonal pattern, used in many tests below
# ---------------------------------------------------------------------------


def _diag(n: int) -> np.ndarray:
    return np.eye(n, dtype=bool)


def _checkerboard(shape):
    """A 0/1 chequerboard pattern of arbitrary rank, like the one in
    ``tests/code-gen/sparsity.py``.
    """
    return np.indices(tuple(shape)).sum(axis=0) % 2


# ---------------------------------------------------------------------------
# CSCMemoryLayout
# ---------------------------------------------------------------------------


class TestCSCMemoryLayout:
    """Compressed-sparse-column storage.  Two-dimensional only."""

    def test_construction_from_diagonal(self):
        from yateto.memory import CSCMemoryLayout
        from yateto.aspp import general

        N = 4
        ml = CSCMemoryLayout(general(_diag(N)))
        assert ml.shape() == (N, N)
        # Only N non-zeros stored.
        assert ml.requiredReals() == N
        assert ml.isSparse() is True

    def test_rejects_non_2d(self):
        from yateto.memory import CSCMemoryLayout
        from yateto.aspp import general
        # CSC is matrix-only.  Anything but rank-2 must be rejected.
        pat = np.ones((3, 3, 3), dtype=bool)
        with pytest.raises(ValueError, match="matrices"):
            CSCMemoryLayout(general(pat))

    def test_requiredReals_matches_nnz(self):
        from yateto.memory import CSCMemoryLayout
        from yateto.aspp import general
        pat = np.zeros((4, 4), dtype=bool)
        # A handful of irregular non-zeros.
        for i, j in [(0, 0), (1, 0), (1, 2), (3, 3)]:
            pat[i, j] = True
        ml = CSCMemoryLayout(general(pat))
        assert ml.requiredReals() == 4

    def test_address_valid_for_each_nonzero(self):
        from yateto.memory import CSCMemoryLayout
        from yateto.aspp import general

        # Diagonal: each (i, i) must map to a unique address in [0, N).
        N = 4
        ml = CSCMemoryLayout(general(_diag(N)))
        addresses = {ml.address((i, i)) for i in range(N)}
        assert addresses == {0, 1, 2, 3}

    def test_hasValue_distinguishes_zeros_and_nonzeros(self):
        from yateto.memory import CSCMemoryLayout
        from yateto.aspp import general
        N = 4
        ml = CSCMemoryLayout(general(_diag(N)))
        # Diagonal entries are stored.
        for i in range(N):
            assert ml.hasValue((i, i))
        # Off-diagonal entries are not.
        for i in range(N):
            for j in range(N):
                if i != j:
                    assert not ml.hasValue((i, j))

    def test_address_outside_bbox_is_caught(self):
        from yateto.memory import CSCMemoryLayout
        from yateto.aspp import general
        N = 4
        ml = CSCMemoryLayout(general(_diag(N)))
        # Off the matrix entirely -> assert fires.
        with pytest.raises(AssertionError):
            ml.address((N, 0))

    def test_colPointer_layout(self):
        from yateto.memory import CSCMemoryLayout
        from yateto.aspp import general
        # A 4x4 with two non-zeros in column 0, none in column 1, three
        # in column 2, one in column 3 - the canonical CSC textbook
        # example.
        pat = np.zeros((4, 4), dtype=bool)
        pat[0, 0] = pat[2, 0] = True
        pat[1, 2] = pat[2, 2] = pat[3, 2] = True
        pat[0, 3] = True
        ml = CSCMemoryLayout(general(pat))
        cp = ml.colPointer()
        # colPointer[c] = number of non-zeros in columns [0, c).
        assert list(cp) == [0, 2, 2, 5, 6]

    def test_rowIndex_layout(self):
        from yateto.memory import CSCMemoryLayout
        from yateto.aspp import general
        # Row indices appear in column-major non-zero order.
        pat = np.zeros((3, 3), dtype=bool)
        pat[0, 0] = True
        pat[2, 0] = True
        pat[1, 2] = True
        ml = CSCMemoryLayout(general(pat))
        # First column: rows [0, 2]; second column: empty; third: [1].
        assert list(ml.rowIndex()) == [0, 2, 1]

    def test_alignedStride_default_false(self):
        from yateto.memory import CSCMemoryLayout
        from yateto.aspp import general
        ml = CSCMemoryLayout(general(_diag(4)))
        assert ml.alignedStride() is False


# ---------------------------------------------------------------------------
# PatternMemoryLayout
# ---------------------------------------------------------------------------


class TestPatternMemoryLayout:
    """N-dimensional positional pattern storage."""

    @pytest.mark.parametrize("rank", [1, 2, 3, 4])
    def test_construction_for_arbitrary_rank(self, rank):
        from yateto.memory import PatternMemoryLayout
        from yateto.aspp import general
        # A dense pattern of the given rank.
        shape = tuple([2] * rank)
        ml = PatternMemoryLayout(general(np.ones(shape, dtype=bool)))
        assert ml.shape() == shape
        assert ml.isSparse() is True

    def test_requiredReals_is_nnz(self):
        from yateto.memory import PatternMemoryLayout
        from yateto.aspp import general
        # A 4x4 chequerboard has half the entries set.
        N = 4
        pat = _checkerboard((N, N)).astype(bool)
        ml = PatternMemoryLayout(general(pat))
        assert ml.requiredReals() == N * N // 2

    def test_diagonal_address_uniqueness(self):
        from yateto.memory import PatternMemoryLayout
        from yateto.aspp import general
        N = 4
        ml = PatternMemoryLayout(general(_diag(N)))
        # All N diagonal addresses are distinct and in [0, N).
        addrs = [ml.address((i, i)) for i in range(N)]
        assert sorted(addrs) == list(range(N))

    def test_hasValue_matches_pattern(self):
        from yateto.memory import PatternMemoryLayout
        from yateto.aspp import general
        N = 4
        pat = _checkerboard((N, N)).astype(bool)
        ml = PatternMemoryLayout(general(pat))
        for i in range(N):
            for j in range(N):
                assert ml.hasValue((i, j)) == bool(pat[i, j])

    def test_3d_dense_pattern(self):
        from yateto.memory import PatternMemoryLayout
        from yateto.aspp import general
        # All entries set: requiredReals == prod(shape).
        shape = (2, 3, 4)
        ml = PatternMemoryLayout(general(np.ones(shape, dtype=bool)))
        assert ml.requiredReals() == 2 * 3 * 4

    def test_pattern_attribute_available(self):
        from yateto.memory import PatternMemoryLayout
        from yateto.aspp import general
        ml = PatternMemoryLayout(general(_diag(3)))
        # Each non-zero gets a positive identifier; zero entries stay 0.
        pat = ml.pattern()
        assert pat.shape == (3, 3)
        # Diagonal entries are positive (1..N), off-diagonals are 0.
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert pat[i, j] > 0
                else:
                    assert pat[i, j] == 0

    def test_alignedStride_and_mayVectorizeDim(self):
        from yateto.memory import PatternMemoryLayout
        from yateto.aspp import general
        ml = PatternMemoryLayout(general(_diag(4)))
        # Without alignment, neither flag fires.
        assert ml.alignedStride() is False
        assert ml.mayVectorizeDim(0) is False

    def test_mayFuse_always_true_for_pattern_layout(self):
        from yateto.memory import PatternMemoryLayout
        from yateto.aspp import general
        # Per the docstring inside ``mayFuse``: PatternMemoryLayout can
        # always synthesise a new pattern, so any subset of dims fuses.
        ml = PatternMemoryLayout(general(np.ones((2, 3, 4), dtype=bool)))
        assert ml.mayFuse([0, 1])
        assert ml.mayFuse([0, 2])
        assert ml.mayFuse([1, 2])


# ---------------------------------------------------------------------------
# fromSpp class methods
# ---------------------------------------------------------------------------


class TestFromSpp:
    def test_csc_fromSpp_returns_cscmemorylayout(self):
        from yateto.memory import CSCMemoryLayout
        from yateto.aspp import general
        ml = CSCMemoryLayout.fromSpp(general(_diag(3)))
        assert isinstance(ml, CSCMemoryLayout)

    def test_pattern_fromSpp_returns_patternmemorylayout(self):
        from yateto.memory import PatternMemoryLayout
        from yateto.aspp import general
        ml = PatternMemoryLayout.fromSpp(general(_diag(3)))
        assert isinstance(ml, PatternMemoryLayout)

    def test_fromSpp_passes_alignStride(self):
        # Just make sure ``alignStride`` plumbing compiles - no value
        # check, since with no global alignment arch set the flag has
        # no observable effect.
        from yateto.memory import PatternMemoryLayout
        from yateto.aspp import general
        ml = PatternMemoryLayout.fromSpp(general(_diag(3)), alignStride=False)
        assert ml.alignedStride() is False


# ---------------------------------------------------------------------------
# Tensor integration
# ---------------------------------------------------------------------------


class TestTensorMemoryLayoutClass:
    """The ``memoryLayoutClass=...`` parameter on ``Tensor`` is the
    user-facing entry point.  It must instantiate the requested layout
    class from the given sparsity pattern.
    """

    def test_default_is_dense(self):
        from yateto import Tensor
        from yateto.memory import DenseMemoryLayout
        T = Tensor("T", (3, 3))
        assert isinstance(T.memoryLayout(), DenseMemoryLayout)

    def test_pattern_layout(self):
        from yateto import Tensor
        from yateto.memory import PatternMemoryLayout
        N = 4
        T = Tensor("T", (N, N), spp=_diag(N),
                   memoryLayoutClass=PatternMemoryLayout)
        assert isinstance(T.memoryLayout(), PatternMemoryLayout)
        assert T.memoryLayout().requiredReals() == N

    def test_csc_layout(self):
        from yateto import Tensor
        from yateto.memory import CSCMemoryLayout
        N = 4
        T = Tensor("T", (N, N), spp=_diag(N),
                   memoryLayoutClass=CSCMemoryLayout)
        assert isinstance(T.memoryLayout(), CSCMemoryLayout)
        assert T.memoryLayout().requiredReals() == N

    def test_setMemoryLayout_can_switch_layouts(self):
        from yateto import Tensor
        from yateto.memory import (CSCMemoryLayout, DenseMemoryLayout,
                                   PatternMemoryLayout)
        N = 4
        T = Tensor("T", (N, N), spp=_diag(N))
        # Default is dense - swap to pattern.
        assert isinstance(T.memoryLayout(), DenseMemoryLayout)
        T.setMemoryLayout(PatternMemoryLayout)
        assert isinstance(T.memoryLayout(), PatternMemoryLayout)
        T.setMemoryLayout(CSCMemoryLayout)
        assert isinstance(T.memoryLayout(), CSCMemoryLayout)

    def test_pattern_layout_for_higher_rank(self):
        from yateto import Tensor
        from yateto.memory import PatternMemoryLayout
        # CSC can't, but Pattern can handle rank > 2.
        shape = (3, 3, 3)
        pat = _checkerboard(shape).astype(bool)
        T = Tensor("T", shape, spp=pat, memoryLayoutClass=PatternMemoryLayout)
        assert T.memoryLayout().requiredReals() == int(pat.sum())


# ---------------------------------------------------------------------------
# End-to-end: the sparsity.py example script
# ---------------------------------------------------------------------------


SPARSITY_SCRIPT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "code-gen", "sparsity.py"))


def _load_sparsity():
    """Import ``tests/code-gen/sparsity.py`` as a module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_sparsity_example",
                                                  SPARSITY_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not os.path.isfile(SPARSITY_SCRIPT),
                    reason="tests/code-gen/sparsity.py not present")
class TestSparsityExample:
    """The example script registers 13 kernels covering the full range
    of sparse-tensor expressions: outer product with broadcasting,
    sparse * dense matmul, sparse * sparse matmul, indexed
    contractions of rank-3 / rank-6 tensors, etc.

    Pushing all of them through ``prepareUntilUnitTest`` is a strong
    end-to-end check on the Python side - and one that's free to run
    without C++.
    """

    def test_loads_expected_number_of_kernels(self, arch):
        from yateto import Generator
        mod = _load_sparsity()
        g = Generator(arch)
        mod.add(g)
        # The script is meant to ship 13 kernels.  If the count
        # changes it's a deliberate edit and this number should be
        # updated to match.
        assert len(g.kernels()) == 13

    def test_every_kernel_prepares(self, arch):
        from yateto import Generator
        from yateto.generator import Kernel
        import inspect

        mod = _load_sparsity()
        g = Generator(arch)
        mod.add(g)

        sig = inspect.signature(Kernel.prepareUntilUnitTest)
        for kernel in g.kernels():
            if "arch" in sig.parameters:
                kernel.prepareUntilUnitTest(arch)
            else:
                kernel.prepareUntilUnitTest()
            assert kernel.cfg is not None
