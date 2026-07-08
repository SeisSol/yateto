"""
Tests for ``yateto.memory`` - the dense memory layout backend.

``DenseMemoryLayout`` is Yateto's representation of how a tensor is laid
out in C++ memory: shape, bounding box (the "interesting" sub-rectangle
inside the full shape, e.g. the non-zero rows of a sparse matrix),
column-major strides, and optional leading-dimension alignment for
vectorisation.

The layout drives:

* address computation inside generated C++ code
* which dimensions may be fused into a single GEMM "m" / "n" / "k"
* whether a tensor can be vectorised along a given axis
* how permutations propagate through strength-reduced AST nodes
"""
from __future__ import annotations

import pytest

from yateto.ast.indices import BoundingBox, Indices, Range
from yateto.aspp import dense as aspp_dense, general as aspp_general
from yateto.memory import DenseMemoryLayout

import numpy as np


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_bbox_is_full_shape(self):
        ml = DenseMemoryLayout((3, 4))
        assert ml.shape() == (3, 4)
        assert ml.bbox() == BoundingBox([Range(0, 3), Range(0, 4)])

    def test_explicit_bbox(self):
        bb = BoundingBox([Range(1, 3), Range(0, 2)])
        ml = DenseMemoryLayout((4, 4), boundingBox=bb)
        # Strides are computed from the bbox, not the full shape: only
        # the non-zero sub-rectangle of an optimised sparse tensor needs
        # storage.
        assert ml.stride()[0] == 1
        assert ml.stride()[1] == bb[0].size()

    def test_from_dense_spp(self):
        ml = DenseMemoryLayout.fromSpp(aspp_dense((5, 6)))
        assert ml.shape() == (5, 6)
        # Fully dense -> bounding box equals full shape.
        assert ml.bbox() == BoundingBox([Range(0, 5), Range(0, 6)])

    def test_from_sparse_spp_shrinks_bbox(self):
        # Only a small sub-block is nonzero; the layout should contract
        # its bounding box accordingly.
        arr = np.zeros((5, 6), dtype=bool)
        arr[1:3, 2:5] = True
        ml = DenseMemoryLayout.fromSpp(aspp_general(arr))
        assert ml.shape() == (5, 6)
        assert ml.bbox()[0] == Range(1, 3)
        assert ml.bbox()[1] == Range(2, 5)


# ---------------------------------------------------------------------------
# Strides and addresses
# ---------------------------------------------------------------------------


class TestStridesAndAddresses:
    def test_strides_are_column_major(self):
        # Yateto uses column-major (Fortran) order so that the leading
        # dimension is unit-stride, matching how GEMMs expect matrices.
        ml = DenseMemoryLayout((3, 4, 5))
        assert ml.stride() == (1, 3, 12)

    def test_address_linear(self):
        ml = DenseMemoryLayout((3, 4))
        # Column-major: linear = i + 3*j
        assert ml.address((0, 0)) == 0
        assert ml.address((1, 0)) == 1
        assert ml.address((0, 1)) == 3
        assert ml.address((2, 3)) == 2 + 3 * 3

    def test_address_with_bbox_offset(self):
        # When the bbox doesn't start at zero, the address is relative to
        # the bbox start (subtracts bbox.start per dim).
        bb = BoundingBox([Range(1, 4), Range(2, 6)])
        ml = DenseMemoryLayout((4, 6), boundingBox=bb)
        assert ml.address((1, 2)) == 0
        # (1,2) is the bbox origin - linear address 0.
        assert ml.address((2, 2)) == 1  # unit stride along first dim
        assert ml.address((1, 3)) == ml.stride()[1]

    def test_address_rejects_out_of_bbox(self):
        bb = BoundingBox([Range(1, 4), Range(2, 6)])
        ml = DenseMemoryLayout((4, 6), boundingBox=bb)
        with pytest.raises(AssertionError):
            ml.address((0, 0))

    def test_required_reals(self):
        # ``requiredReals`` is the allocation size: stride[-1] * bbox[-1].size()
        ml = DenseMemoryLayout((3, 4))
        assert ml.requiredReals() == 12
        # Sparse variant with smaller bbox allocates less.
        bb = BoundingBox([Range(0, 2), Range(0, 3)])
        ml = DenseMemoryLayout((3, 4), boundingBox=bb)
        assert ml.requiredReals() == 6


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_alignment_false_without_arch(self):
        # Without an architecture registered globally, the layout cannot
        # claim aligned-stride and ``mayVectorizeDim`` returns False.
        ml = DenseMemoryLayout((5, 3))
        assert ml.alignedStride() is False
        assert ml.mayVectorizeDim(0) is False

    def test_aligned_stride_with_arch(self, arch):
        # ``dhsw`` has 32B alignment -> 4 aligned doubles.  A 5-entry
        # leading dim gets extended to 8 when alignStride=True.
        ml = DenseMemoryLayout((5, 3), alignStride=True)
        assert ml.bbox()[0].size() == 8  # 5 -> aligned up to 8
        assert ml.alignedStride() is True

    def test_may_vectorize_when_dim_is_aligned(self, arch):
        ml = DenseMemoryLayout((8, 3))  # 8 is a multiple of 4
        assert ml.mayVectorizeDim(0) is True

    def test_may_not_vectorize_when_dim_misaligned(self, arch):
        ml = DenseMemoryLayout((7, 3))
        assert ml.mayVectorizeDim(0) is False


# ---------------------------------------------------------------------------
# mayFuse - contiguity predicate for LoopOverGEMM
# ---------------------------------------------------------------------------


class TestFusion:
    def test_single_position_always_fuses(self):
        ml = DenseMemoryLayout((3, 4, 5))
        assert ml.mayFuse([1]) is True

    def test_consecutive_indices_fuse_in_column_major(self):
        # In column-major layout, stride[i+1] == shape[i] * stride[i].
        # So consecutive positions fuse - this is what lets Yateto fold
        # multi-index GEMM operands into a single "m" dimension.
        ml = DenseMemoryLayout((3, 4, 5))
        assert ml.mayFuse([0, 1]) is True
        assert ml.mayFuse([1, 2]) is True
        assert ml.mayFuse([0, 1, 2]) is True

    def test_non_consecutive_does_not_fuse(self):
        ml = DenseMemoryLayout((3, 4, 5))
        # Skipping the middle index breaks contiguity.
        assert ml.mayFuse([0, 2]) is False


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------


class TestPermutation:
    def test_permute_reorders_shape_and_bbox(self):
        bb = BoundingBox([Range(1, 3), Range(0, 4), Range(0, 5)])
        ml = DenseMemoryLayout((3, 4, 5), boundingBox=bb)
        pm = ml.permuted((2, 0, 1))
        # Shape and bbox must follow the permutation.
        assert pm.shape() == (5, 3, 4)
        assert pm.bbox()[0] == Range(0, 5)
        assert pm.bbox()[1] == Range(1, 3)
        assert pm.bbox()[2] == Range(0, 4)


# ---------------------------------------------------------------------------
# Address string formatting (used to emit C++)
# ---------------------------------------------------------------------------


class TestAddressString:
    def test_address_string_simple(self):
        ml = DenseMemoryLayout((3, 4))
        idx = Indices("ij", (3, 4))
        s = ml.addressString(idx)
        # Should mention both index letters with their strides.
        assert "_i" in s
        assert "_j" in s

    def test_scalar_address_string(self):
        ml = DenseMemoryLayout(())  # 0-D tensor
        idx = Indices("", ())
        assert ml.addressString(idx) == "0"


# ---------------------------------------------------------------------------
# Global arch state isolation (via the fixture)
# ---------------------------------------------------------------------------


class TestAlignmentArchIsolation:
    def test_fixture_resets_arch(self, arch):
        assert DenseMemoryLayout.ALIGNMENT_ARCH is not None

    def test_fresh_test_starts_without_arch(self):
        # This test uses no ``arch`` fixture and must see a reset global.
        # (If this fails, the ``arch`` fixture's teardown is broken.)
        assert DenseMemoryLayout.ALIGNMENT_ARCH is None
