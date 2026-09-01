"""
Tests for ``yateto.aspp`` - abstract sparsity patterns.

Yateto propagates a sparsity pattern through every AST node, uses it to
perform strength reduction, and finally feeds it to the back-end so dense
GEMM calls can be specialised for zero-filled rows/columns.  Two concrete
implementations exist:

* ``dense``    - a lightweight, shape-only representation
* ``general``  - a numpy-backed bit pattern

Mixed operations dispatch between them.
"""
from __future__ import annotations

import numpy as np
import pytest

from yateto import aspp


# ---------------------------------------------------------------------------
# dense
# ---------------------------------------------------------------------------


class TestDense:
    def test_count_nonzero_is_size(self):
        d = aspp.dense((3, 4))
        assert d.count_nonzero() == 12
        assert d.size == 12

    def test_is_dense(self):
        assert aspp.dense((3, 4)).is_dense()

    def test_shape(self):
        d = aspp.dense((2, 3, 5))
        assert d.shape == (2, 3, 5)
        assert d.ndim == 3

    def test_reshape(self):
        d = aspp.dense((2, 3)).reshape((6,))
        assert d.shape == (6,)

    def test_reshape_checks_size(self):
        with pytest.raises(AssertionError, match="Size mismatch"):
            aspp.dense((2, 3)).reshape((4,))

    def test_transpose(self):
        d = aspp.dense((2, 3)).transposed((1, 0))
        assert d.shape == (3, 2)

    def test_broadcast(self):
        d = aspp.dense((2, 3)).broadcast((3, 2))
        # broadcast multiplies each dim by the factor
        assert d.shape == (6, 6)

    def test_indexSum_drops_axes(self):
        from yateto.ast.indices import Indices
        src = Indices("ijk", (2, 3, 4))
        tgt = Indices("ik", (2, 4))
        d = aspp.dense((2, 3, 4)).indexSum(src, tgt, ())
        assert d.shape == (2, 4)

    def test_add_same_shape(self):
        result = aspp.dense.add(aspp.dense((2, 3)), aspp.dense((2, 3)))
        assert result.shape == (2, 3)
        assert result.is_dense()

    def test_add_shape_mismatch_asserts(self):
        with pytest.raises(AssertionError):
            aspp.dense.add(aspp.dense((2, 3)), aspp.dense((3, 2)))

    def test_einsum_shape_inference(self):
        # Classic matmul: (i,j) * (j,k) -> (i,k)
        result = aspp.dense.einsum("ij,jk->ik", aspp.dense((3, 4)), aspp.dense((4, 5)))
        assert result.shape == (3, 5)

    def test_einsum_rejects_bad_description(self):
        with pytest.raises(ValueError, match="not understood"):
            aspp.dense.einsum("bogus", aspp.dense((2,)), aspp.dense((2,)))

    def test_as_ndarray_is_all_ones(self):
        arr = aspp.dense((2, 3)).as_ndarray()
        assert arr.shape == (2, 3)
        assert arr.dtype == bool
        assert arr.all()


# ---------------------------------------------------------------------------
# general
# ---------------------------------------------------------------------------


class TestGeneral:
    def test_basic_count(self):
        pattern = np.array([[1, 0], [0, 1]], dtype=bool)
        g = aspp.general(pattern)
        assert g.count_nonzero() == 2
        assert g.shape == (2, 2)

    def test_is_dense_only_if_fully_filled(self):
        g_full = aspp.general(np.ones((2, 2), dtype=bool))
        assert g_full.is_dense()
        g_sparse = aspp.general(np.eye(2, dtype=bool))
        assert not g_sparse.is_dense()

    def test_transpose(self):
        pattern = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
        g = aspp.general(pattern).transposed((1, 0))
        # Transposing a 2x3 pattern should produce a 3x2 pattern that is
        # element-wise consistent with np.transpose.
        assert np.array_equal(g.as_ndarray(), pattern.T)

    def test_einsum_matches_numpy(self):
        A = np.array([[1, 0], [1, 1]], dtype=bool)
        B = np.array([[1, 1], [0, 1]], dtype=bool)
        g = aspp.general.einsum("ij,jk->ik", aspp.general(A), aspp.general(B))
        # Note: numpy einsum on bool does a logical OR / AND, but here we
        # compare against the cast-to-bool of a real matmul.
        expected = (A.astype(int) @ B.astype(int)) > 0
        assert np.array_equal(g.as_ndarray(), expected)

    def test_nonzero(self):
        pattern = np.array([[1, 0], [0, 1]], dtype=bool)
        nz = aspp.general(pattern).nonzero()
        # numpy's ``.nonzero()`` returns a tuple of arrays
        assert len(nz) == 2
        np.testing.assert_array_equal(nz[0], [0, 1])
        np.testing.assert_array_equal(nz[1], [0, 1])

    def test_nnzbounds_returns_inclusive_bounds_per_axis(self):
        pattern = np.zeros((5, 5), dtype=bool)
        pattern[1:4, 2:5] = True
        g = aspp.general(pattern)
        bounds = g.nnzbounds()
        # Per-axis (min, max) inclusive bounds of nonzero entries.
        assert bounds == [(1, 3), (2, 4)]

    def test_copy_is_independent(self):
        pattern = np.eye(3, dtype=bool)
        g = aspp.general(pattern)
        h = g.copy()
        # Mutating the source must not affect the copy.
        g.pattern[0, 0] = False
        assert h.count_nonzero() == 3


# ---------------------------------------------------------------------------
# Cross-class dispatch (dense/general mixed)
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_add_dense_and_general_promotes_to_general(self):
        d = aspp.dense((2, 2))
        g = aspp.general(np.eye(2, dtype=bool))
        result = aspp.add(d, g)
        assert isinstance(result, aspp.general)
        # dense contributes "all ones", so the result is all ones
        assert result.count_nonzero() == 4

    def test_add_two_dense_stays_dense(self):
        result = aspp.add(aspp.dense((2, 3)), aspp.dense((2, 3)))
        assert isinstance(result, aspp.dense)

    def test_add_two_general(self):
        a = aspp.general(np.array([[1, 0], [0, 0]], dtype=bool))
        b = aspp.general(np.array([[0, 1], [0, 0]], dtype=bool))
        result = aspp.add(a, b)
        assert isinstance(result, aspp.general)
        assert result.count_nonzero() == 2

    def test_einsum_dispatches(self):
        # Mixed types in an einsum must not crash - they route through
        # ``dispatch`` which converts dense -> general on demand.
        d = aspp.dense((3, 4))
        g = aspp.general(np.ones((4, 5), dtype=bool))
        result = aspp.einsum("ij,jk->ik", d, g)
        assert result.shape == (3, 5)

    def test_array_equal_across_classes(self):
        d = aspp.dense((2, 3))
        g = aspp.general(np.ones((2, 3), dtype=bool))
        assert aspp.array_equal(d, g)
        # Different shape -> not equal.
        assert not aspp.array_equal(d, aspp.dense((3, 2)))

    def test_array_equal_handles_none(self):
        # Yateto occasionally compares ``None`` equivalents.
        assert aspp.array_equal(None, None)
        assert not aspp.array_equal(None, aspp.dense((2, 2)))
