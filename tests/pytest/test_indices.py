"""
Tests for ``yateto.ast.indices`` - ``Indices``, ``Range``, ``BoundingBox``,
``LoGCost``.

These are the type-theoretic bookkeeping objects of the Einstein-notation
DSL.  Nearly every AST transformer and every cost estimator touches them,
so bugs here tend to surface as confusing downstream failures.  Worth
nailing down with direct unit tests.
"""
from __future__ import annotations

import pytest

from yateto.ast.indices import Indices, Range, BoundingBox, LoGCost


# ---------------------------------------------------------------------------
# Indices construction and basic invariants
# ---------------------------------------------------------------------------


class TestIndicesConstruction:
    def test_basic(self):
        idx = Indices("ij", (4, 5))
        assert str(idx) == "ij"
        assert len(idx) == 2
        assert idx.shape() == (4, 5)
        assert idx.indexSize("i") == 4
        assert idx.indexSize("j") == 5

    def test_empty_indices_for_scalar(self):
        idx = Indices("", ())
        assert str(idx) == ""
        assert len(idx) == 0
        assert idx.shape() == ()

    def test_default_constructor_is_empty(self):
        assert len(Indices()) == 0

    def test_repeated_index_names_rejected(self):
        with pytest.raises(AssertionError, match="Repeated indices"):
            Indices("ii", (4, 4))

    def test_shape_length_mismatch_rejected(self):
        with pytest.raises(AssertionError, match="do not match tensor shape"):
            Indices("ij", (4, 5, 6))
        with pytest.raises(AssertionError, match="do not match tensor shape"):
            Indices("ijk", (4, 5))


# ---------------------------------------------------------------------------
# Set-like operations
# ---------------------------------------------------------------------------


class TestIndicesSetOps:
    def test_intersection_returns_raw_set(self):
        a = Indices("ij", (3, 4))
        b = Indices("jk", (4, 5))
        # Intersection is a plain ``set``, not an ``Indices`` object.
        # This is how ``Einsum`` / ``Product`` identify contraction indices.
        assert a & b == {"j"}
        # Commutative
        assert b & a == {"j"}

    def test_difference_returns_indices(self):
        a = Indices("ijk", (3, 4, 5))
        b = Indices("jk", (4, 5))
        diff = a - b
        assert isinstance(diff, Indices)
        assert str(diff) == "i"
        assert diff.shape() == (3,)

    def test_difference_preserves_order(self):
        # Important - strength reduction + LoG rely on this.
        a = Indices("abcd", (1, 2, 3, 4))
        assert str(a - Indices("bd", (2, 4))) == "ac"

    def test_le_is_subset_with_matching_sizes(self):
        a = Indices("ij", (3, 4))
        big = Indices("ijk", (3, 4, 5))
        assert a <= big
        assert not (big <= a)

    def test_le_rejects_mismatched_sizes(self):
        a = Indices("ij", (3, 4))
        bad = Indices("ij", (3, 5))  # same letters, different shape
        assert not (a <= bad)

    def test_contains(self):
        idx = Indices("ij", (3, 4))
        assert "i" in idx
        assert "j" in idx
        assert "k" not in idx


# ---------------------------------------------------------------------------
# Merging / permuting
# ---------------------------------------------------------------------------


class TestIndicesMerge:
    def test_merged_concatenates(self):
        a = Indices("ij", (3, 4))
        b = Indices("kl", (5, 6))
        merged = a.merged(b)
        assert str(merged) == "ijkl"
        assert merged.shape() == (3, 4, 5, 6)

    def test_merged_allows_duplicate_names_without_check(self):
        # ``merged`` is the naive concat; ``mergeStrict`` is the checked
        # variant that deduplicates.  Exercise both contracts.
        a = Indices("ij", (3, 4))
        # Duplicating "i" via ``merged`` triggers the "Repeated indices"
        # assert in ``Indices.__init__``.
        with pytest.raises(AssertionError):
            a.merged(a)

    def test_merge_strict_dedupes(self):
        a = Indices("ij", (3, 4))
        b = Indices("jk", (4, 5))
        merged = a.mergeStrict(b)
        assert str(merged) == "ijk"
        assert merged.shape() == (3, 4, 5)

    def test_merge_strict_rejects_incompatible_shape(self):
        a = Indices("ij", (3, 4))
        b = Indices("jk", (9, 5))  # j has different size
        with pytest.raises(AssertionError, match="Index merge failed"):
            a.mergeStrict(b)

    def test_permuted_reorders(self):
        idx = Indices("ijk", (3, 4, 5))
        p = idx.permuted("kij")
        assert str(p) == "kij"
        assert p.shape() == (5, 3, 4)
        # Sizes must stay attached to the correct letters after permutation.
        assert p.indexSize("i") == 3
        assert p.indexSize("j") == 4
        assert p.indexSize("k") == 5

    def test_sorted(self):
        idx = Indices("cab", (1, 2, 3))
        s = idx.sorted()
        assert str(s) == "abc"
        assert s.indexSize("a") == 2


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


class TestIndicesPositions:
    def test_find(self):
        idx = Indices("ijk", (3, 4, 5))
        assert idx.find("i") == 0
        assert idx.find("j") == 1
        assert idx.find("k") == 2

    def test_positions_sorted_by_default(self):
        idx = Indices("ijk", (3, 4, 5))
        # Positions are returned sorted by default - this is important for
        # LoG (m/n/k indices must be a contiguous range to be fuseable).
        assert idx.positions("kj") == [1, 2]
        assert idx.positions(["j", "i"]) == [0, 1]

    def test_positions_unsorted(self):
        idx = Indices("ijk", (3, 4, 5))
        assert idx.positions("kj", sort=False) == [2, 1]


# ---------------------------------------------------------------------------
# Hash / equality
# ---------------------------------------------------------------------------


class TestIndicesEquality:
    def test_hashable(self):
        # Indices are used as dict keys in the generator.
        a = Indices("ij", (3, 4))
        b = Indices("ij", (3, 4))
        assert hash(a) == hash(b)
        d = {a: 1}
        assert d[b] == 1

    def test_eq_considers_both_names_and_shape(self):
        assert Indices("ij", (3, 4)) == Indices("ij", (3, 4))
        assert Indices("ij", (3, 4)) != Indices("ji", (3, 4))
        assert Indices("ij", (3, 4)) != Indices("ij", (3, 5))


# ---------------------------------------------------------------------------
# Range
# ---------------------------------------------------------------------------


class TestRange:
    def test_size(self):
        assert Range(3, 8).size() == 5
        assert Range(0, 0).size() == 0

    def test_intersection(self):
        r = Range(2, 8) & Range(5, 10)
        assert r.start == 5
        assert r.stop == 8

    def test_union(self):
        r = Range(2, 4) | Range(6, 10)
        assert r.start == 2
        assert r.stop == 10

    def test_contains(self):
        outer = Range(0, 10)
        assert Range(2, 5) in outer
        assert Range(5, 10) in outer
        assert Range(0, 11) not in outer

    def test_iter_enumerates_range(self):
        assert list(Range(2, 5)) == [2, 3, 4]

    def test_eq(self):
        assert Range(1, 5) == Range(1, 5)
        assert Range(1, 5) != Range(1, 6)


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------


class TestBoundingBox:
    def test_length_and_iter(self):
        bb = BoundingBox([Range(0, 3), Range(0, 4)])
        assert len(bb) == 2
        assert [r.size() for r in bb] == [3, 4]

    def test_size_is_volume(self):
        assert BoundingBox([Range(0, 3), Range(0, 4)]).size() == 12
        # Empty box has size 1 (the empty product) - needed by scalar ops.
        assert BoundingBox([]).size() == 1

    def test_contains_point(self):
        bb = BoundingBox([Range(0, 3), Range(0, 4)])
        assert (1, 2) in bb
        # Wrong arity is a rejection, not a crash
        assert (1, 2, 3) not in bb

    def test_fromSpp_dense(self):
        # Build from a dense sparsity pattern - each dimension's range
        # spans the whole shape (with an off-by-one because ``nnzbounds``
        # returns inclusive bounds).
        from yateto.aspp import dense
        bb = BoundingBox.fromSpp(dense((3, 4)))
        assert len(bb) == 2
        assert bb[0].start == 0 and bb[0].stop == 3
        assert bb[1].start == 0 and bb[1].stop == 4


# ---------------------------------------------------------------------------
# LoGCost - the tuple cost model used for GEMM variant selection
# ---------------------------------------------------------------------------


class TestLoGCost:
    def test_identity_is_zero_cost(self):
        c = LoGCost.addIdentity()
        assert c == LoGCost(0, 0, 0, 0)

    def test_addition_is_componentwise(self):
        a = LoGCost(1, 2, 3, 4)
        b = LoGCost(10, 20, 30, 40)
        c = a + b
        # We can't access internals, but equality with the expected sum works
        assert c == LoGCost(11, 22, 33, 44)

    def test_lower_stride_is_cheaper(self):
        # Unit-stride (0) beats non-unit (1)
        assert LoGCost(0, 0, 0, 0) < LoGCost(1, 0, 0, 0)

    def test_more_fused_indices_is_cheaper(self):
        # fused_indices contributes negatively to the cost tuple, i.e. more
        # fused indices is better.
        assert LoGCost(0, 0, 0, 2) < LoGCost(0, 0, 0, 1)

    def test_fewer_transposes_is_cheaper(self):
        assert LoGCost(0, 0, 0, 0) < LoGCost(0, 1, 0, 0)
        assert LoGCost(0, 1, 0, 0) < LoGCost(0, 2, 0, 0)

    def test_tiebreak_prefers_fewer_left_transposes(self):
        # When the summed-transposes match, the comparator falls back to
        # ``_leftTranspose``.  A > B iff A.leftTranspose > B.leftTranspose.
        a = LoGCost(0, 0, 1, 0)
        b = LoGCost(0, 1, 0, 0)
        assert a < b
