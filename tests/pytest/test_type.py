"""
Tests for ``yateto.type`` - the frontend types users interact with.

This is the DSL's **surface** layer: ``Tensor``, ``Scalar``, and ``Collection``
are the objects that the Python operator-overloading machinery wraps into an
AST.  Name validation, grouping, sparsity storage and the shape invariants
belong to this layer - not to any of the downstream passes.
"""
from __future__ import annotations

import numpy as np
import pytest

from yateto import Tensor, Scalar
from yateto.type import Collection, IdentifiedType


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------


class TestTensorNames:
    """Tensor names follow a strict regexp: ``<base>[(g1,g2,...)]``."""

    @pytest.mark.parametrize(
        "name",
        [
            "A",
            "AB",
            "foo",
            "A0",
            "A_1",
            "A(0)",
            "A(1)",
            "A(0,1)",
            "A(10)",
            "A(0,1,2,3)",
        ],
    )
    def test_valid_names_accepted(self, name):
        Tensor(name, (2, 2))  # must not raise

    @pytest.mark.parametrize(
        "name",
        [
            "",          # empty
            "0A",        # leading digit
            "_A",        # leading underscore
            "A(01)",     # leading zero in group index (regex disallows)
            "A()",       # empty group
            "A(,)",      # empty group index
            "A(-1)",     # negative group
            "A-B",       # invalid char
            "A(1,)",     # trailing comma
        ],
    )
    def test_invalid_names_rejected(self, name):
        with pytest.raises(ValueError):
            Tensor(name, (2, 2))

    def test_group_extraction(self):
        assert Tensor("A", (2, 2)).group() == ()
        assert Tensor("A(3)", (2, 2)).group() == (3,)
        assert Tensor("A(1,2,3)", (2, 2)).group() == (1, 2, 3)

    def test_base_name_extraction(self):
        assert Tensor("foo(1,2)", (2, 2)).baseName() == "foo"
        assert Tensor("foo", (2, 2)).baseName() == "foo"
        assert Tensor("foo42", (2, 2)).baseName() == "foo42"

    def test_is_valid_name_classmethod(self):
        assert Tensor.isValidName("A(0)")
        assert not Tensor.isValidName("0A")


# ---------------------------------------------------------------------------
# Shape invariants
# ---------------------------------------------------------------------------


class TestTensorShape:
    def test_shape_must_be_tuple(self):
        with pytest.raises(ValueError, match="shape must be a tuple"):
            Tensor("A", [2, 2])  # list, not tuple
        with pytest.raises(ValueError, match="shape must be a tuple"):
            Tensor("A", 42)  # int

    def test_shape_entries_must_be_positive(self):
        with pytest.raises(ValueError, match="smaller than 1"):
            Tensor("A", (0, 2))
        with pytest.raises(ValueError, match="smaller than 1"):
            Tensor("A", (2, -1))

    def test_shape_is_stored_as_tuple(self):
        t = Tensor("A", (3, 4, 5))
        assert t.shape() == (3, 4, 5)
        assert isinstance(t.shape(), tuple)

    def test_zero_dimensional_tensor_is_allowed(self):
        # A scalar tensor: shape == ().  Used by reductions.
        t = Tensor("A", ())
        assert t.shape() == ()


# ---------------------------------------------------------------------------
# Namespacing
# ---------------------------------------------------------------------------


class TestTensorNamespace:
    def test_default_namespace_is_empty_string(self):
        t = Tensor("A", (2, 2))
        assert t.namespace == ""
        assert t.prefix() == ""
        assert t.nameWithNamespace() == "A"

    def test_explicit_namespace_is_used_as_cpp_prefix(self):
        t = Tensor("A", (2, 2), namespace="foo")
        assert t.namespace == "foo"
        assert t.prefix() == "foo::"
        assert t.nameWithNamespace() == "foo::A"
        assert t.baseNameWithNamespace() == "foo::A"

    def test_split_base_name_with_namespace(self):
        # classmethod that undoes the ``foo::bar`` encoding
        prefix, base = IdentifiedType.splitBasename("foo::bar")
        assert prefix == "foo::"
        assert base == "bar"
        prefix, base = IdentifiedType.splitBasename("bar")
        assert prefix == ""
        assert base == "bar"


# ---------------------------------------------------------------------------
# Sparsity patterns
# ---------------------------------------------------------------------------


class TestTensorSparsity:
    def test_dense_default(self):
        t = Tensor("A", (3, 3))
        # Without an explicit spp, the tensor is dense.
        assert t.spp().is_dense()
        # Dense sparsity reports count_nonzero == total size.
        assert t.spp().count_nonzero() == 9

    def test_dict_sparsity_with_bool_values(self):
        spp = {(0, 0): True, (1, 1): True, (2, 2): True}
        t = Tensor("A", (3, 3), spp=spp)
        assert t.spp().count_nonzero() == 3
        # No numerical values stored, as bool dict means pattern only
        assert t.values() is None

    def test_dict_sparsity_with_float_values(self):
        # Providing floats means Yateto also records the literal values.
        spp = {(0, 0): 1.0, (1, 1): 2.0, (2, 2): 3.0}
        t = Tensor("A", (3, 3), spp=spp)
        assert t.spp().count_nonzero() == 3
        assert t.values() == {(0, 0): 1.0, (1, 1): 2.0, (2, 2): 3.0}
        assert t.is_compute_constant() is True

    def test_ndarray_sparsity(self):
        arr = np.eye(4, dtype=bool)
        t = Tensor("A", (4, 4), spp=arr)
        assert t.spp().count_nonzero() == 4

    def test_float_ndarray_keeps_values(self):
        arr = np.eye(3)  # float64 identity
        t = Tensor("A", (3, 3), spp=arr)
        # Yateto extracts nonzero float values as strings (for codegen).
        assert t.values() is not None
        assert len(t.values()) == 3

    def test_values_as_ndarray_roundtrip(self):
        spp = {(0, 0): 1.5, (1, 2): -2.0}
        t = Tensor("A", (3, 3), spp=spp)
        arr = t.values_as_ndarray()
        assert arr.shape == (3, 3)
        assert arr[0, 0] == pytest.approx(1.5)
        assert arr[1, 2] == pytest.approx(-2.0)
        # All other entries remain zero.
        arr[0, 0] = 0
        arr[1, 2] = 0
        assert np.all(arr == 0)

    def test_bad_sparsity_shape_raises(self):
        wrong = np.ones((2, 2), dtype=bool)
        with pytest.raises(Exception):
            Tensor("A", (3, 3), spp=wrong)

    def test_is_compute_constant_false_for_plain_tensor(self):
        assert Tensor("A", (3, 3)).is_compute_constant() is False


# ---------------------------------------------------------------------------
# Tensor identity / hashing / equality
# ---------------------------------------------------------------------------


class TestTensorIdentity:
    def test_hash_is_name_based(self):
        # The hash is built from the tensor name, so two tensors with the
        # same name can be put in a set even if they live in different
        # scopes.  This is what the codegen relies on.
        assert hash(Tensor("A", (2, 2))) == hash(Tensor("A", (4, 4)))

    def test_equality_by_name(self):
        t1 = Tensor("A", (2, 2))
        t2 = Tensor("A", (2, 2))
        assert t1 == t2

    def test_equality_across_shapes_asserts(self):
        # Yateto's ``__eq__`` asserts same shape/layout when names match -
        # i.e. two tensors that share a name but differ structurally are
        # detected as a bug in the user's code, not silently un-equal.
        t1 = Tensor("A", (2, 2))
        t2 = Tensor("A", (3, 3))
        with pytest.raises(AssertionError):
            t1 == t2

    def test_inequality_by_name(self):
        assert (Tensor("A", (2, 2)) == Tensor("B", (2, 2))) is False


# ---------------------------------------------------------------------------
# Memory layout attached to tensor
# ---------------------------------------------------------------------------


class TestTensorMemoryLayout:
    def test_memory_layout_is_set(self):
        t = Tensor("A", (3, 4))
        ml = t.memoryLayout()
        assert ml is not None
        assert ml.shape() == (3, 4)

    def test_aligned_stride_requires_arch(self, arch):
        # With the arch fixture (sets DenseMemoryLayout's global alignment),
        # we can request an aligned leading dimension.
        t = Tensor("A", (5, 3), alignStride=True)
        assert t.memoryLayout().alignedStride() is True


# ---------------------------------------------------------------------------
# Scalar
# ---------------------------------------------------------------------------


class TestScalar:
    def test_basic(self):
        s = Scalar("alpha")
        assert s.name() == "alpha"
        assert s.baseName() == "alpha"

    def test_scalar_group(self):
        s = Scalar("alpha(2)")
        assert s.group() == (2,)

    def test_invalid_scalar_name(self):
        with pytest.raises(ValueError):
            Scalar("0alpha")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class TestCollection:
    def test_set_and_get(self):
        c = Collection()
        c["foo"] = Tensor("foo", (2, 2))
        assert "foo" in c
        assert c["foo"].name() == "foo"

    def test_containsName(self):
        c = Collection()
        c["A"] = Tensor("A", (2, 2))
        assert c.containsName("A")
        assert not c.containsName("B")

    def test_containsName_rejects_invalid_names(self):
        c = Collection()
        with pytest.raises(ValueError):
            c.containsName("0bad")

    def test_group_classmethod(self):
        # Collection.group returns a single int for 1-tuple groups, a
        # tuple otherwise, and () for non-grouped names.  This mirrors how
        # ``Generator.add`` dispatches ``A(0)`` vs ``A(0,1)``.
        assert Collection.group("A") == ()
        assert Collection.group("A(3)") == 3
        assert Collection.group("A(1,2)") == (1, 2)

    def test_update_merges_collections(self):
        a = Collection()
        a["A"] = Tensor("A", (2, 2))
        b = Collection()
        b["B"] = Tensor("B", (3, 3))
        a.update(b)
        assert "A" in a and "B" in a
