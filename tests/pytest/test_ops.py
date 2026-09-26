"""The operation layer: datatypes, neutral elements and sparsity propagation."""

import math

import numpy as np
import pytest

from yateto import aspp, ops
from yateto.type import Datatype, TypeFlavor

FLOATS = [Datatype.F16, Datatype.BF16, Datatype.F32, Datatype.F64, Datatype.F128]
INTS = [Datatype.I8, Datatype.I16, Datatype.I32, Datatype.I64]
ALL = [Datatype.BOOL] + INTS + FLOATS


class TestDatatype:
    @pytest.mark.parametrize('datatype', ALL)
    def test_literal_of_zero_is_emittable(self, datatype):
        assert datatype.literal(0)

    @pytest.mark.parametrize('datatype', ALL)
    def test_infinities_do_not_leak_python_spellings(self, datatype):
        for value in (float('inf'), -float('inf'), float('nan')):
            literal = datatype.literal(value)
            assert 'inf' not in literal.replace('infinity', '')
            assert 'nan' not in literal.replace('quiet_NaN', '')

    @pytest.mark.parametrize('datatype', FLOATS)
    def test_float_infinity_uses_limits(self, datatype):
        assert 'infinity()' in datatype.literal(float('inf'))
        assert datatype.literal(-float('inf')).startswith('-')

    @pytest.mark.parametrize('datatype', INTS)
    def test_integer_infinity_saturates(self, datatype):
        ctype = datatype.ctype()
        assert datatype.literal(float('inf')) == f'std::numeric_limits<{ctype}>::max()'
        assert datatype.literal(-float('inf')) == f'std::numeric_limits<{ctype}>::lowest()'

    @pytest.mark.parametrize('datatype', INTS)
    def test_integer_extremes_are_emitted_as_values(self, datatype):
        lo, hi = datatype.limits()
        assert str(hi) in datatype.literal(hi)
        assert str(lo) in datatype.literal(lo)

    @pytest.mark.parametrize('datatype', INTS)
    def test_safeint_stays_in_range(self, datatype):
        lo, hi = datatype.limits()
        for value in (float('inf'), -float('inf'), 10 ** 30, -(10 ** 30), 0, 1):
            assert lo <= datatype.safeint(value) <= hi

    @pytest.mark.parametrize('datatype', ALL)
    def test_nptype_exists(self, datatype):
        assert np.dtype(datatype.nptype()) is not None

    @pytest.mark.parametrize('datatype', ALL)
    def test_classification_is_exclusive(self, datatype):
        assert sum([datatype.isFloat(), datatype.isInteger(), datatype.isBool()]) == 1

    def test_eigen_flavor_falls_back_to_the_default(self):
        assert Datatype.F64.ctype(TypeFlavor.EIGEN) == Datatype.F64.ctype()

    def test_eigen_flavor_wraps_the_non_standard_formats(self):
        assert Datatype.F16.ctype(TypeFlavor.EIGEN) != Datatype.F16.ctype()


class TestOperationIdentity:
    def test_operations_compare_by_kind(self):
        assert ops.Add() == ops.Add()
        assert ops.Add() != ops.Mul()

    def test_a_cast_carries_its_target(self):
        assert ops.Typecast(Datatype.F32) != ops.Typecast(Datatype.F64)
        assert ops.Typecast(Datatype.F32) == ops.Typecast(Datatype.F32)

    def test_operations_are_hashable(self):
        assert len({ops.Add(), ops.Add(), ops.Mul()}) == 2
        assert len({ops.Typecast(Datatype.F32), ops.Typecast(Datatype.F64)}) == 2

    @pytest.mark.parametrize('op,arity', [
        (ops.Sqrt(), 1), (ops.Add(), 2), (ops.Min(), 2), (ops.Ternary(), 3),
    ])
    def test_arity_is_checked(self, op, arity):
        op.checkArity(arity)
        with pytest.raises(ValueError):
            op.checkArity(arity + 1)


class TestNeutralElements:
    @pytest.mark.parametrize('datatype', INTS + FLOATS)
    def test_add_is_zero(self, datatype):
        assert ops.Add().neutral(datatype) == 0

    @pytest.mark.parametrize('datatype', INTS + FLOATS)
    def test_mul_is_one(self, datatype):
        assert ops.Mul().neutral(datatype) == 1

    @pytest.mark.parametrize('datatype', INTS)
    def test_bitwise_and_is_all_ones_over_integers(self, datatype):
        # 1 would leave every bit but the lowest cleared
        assert ops.And().neutral(datatype) == -1

    def test_bitwise_and_is_true_over_bool(self):
        assert ops.And().neutral(Datatype.BOOL) is True

    @pytest.mark.parametrize('datatype', INTS + [Datatype.BOOL])
    def test_or_and_xor_are_zero(self, datatype):
        assert not ops.Or().neutral(datatype)
        assert not ops.Xor().neutral(datatype)

    @pytest.mark.parametrize('datatype', INTS)
    def test_min_max_use_the_integer_extremes(self, datatype):
        lo, hi = datatype.limits()
        assert ops.Min().neutral(datatype) == hi
        assert ops.Max().neutral(datatype) == lo

    @pytest.mark.parametrize('datatype', FLOATS)
    def test_min_max_use_infinity_for_floats(self, datatype):
        assert math.isinf(ops.Min().neutral(datatype))
        assert math.isinf(ops.Max().neutral(datatype))

    @pytest.mark.parametrize('op', [ops.Add(), ops.Mul(), ops.And(), ops.Or(),
                                    ops.Xor(), ops.Min(), ops.Max()])
    @pytest.mark.parametrize('datatype', [Datatype.BOOL, Datatype.I32, Datatype.F64])
    def test_neutral_literal_is_emittable(self, op, datatype):
        assert op.neutralLiteral(datatype)

    @pytest.mark.parametrize('op', [ops.Add(), ops.Mul(), ops.Min(), ops.Max()])
    def test_neutral_acts_as_a_unit(self, op):
        values = np.array([3.0, -1.0, 7.5])
        neutral = np.full_like(values, op.neutral(Datatype.F64))
        assert np.allclose(op.call(values, neutral), values)


class TestDatatypeResult:
    def test_comparisons_yield_bool(self):
        for op in (ops.CmpEq(), ops.CmpNe(), ops.CmpLt(), ops.CmpGe()):
            assert op.datatypeResult([Datatype.F64, Datatype.F64]) == Datatype.BOOL

    def test_a_cast_yields_its_target(self):
        assert ops.Typecast(Datatype.I32).datatypeResult([Datatype.F64]) == Datatype.I32

    def test_arithmetic_promotes_to_the_wider_type(self):
        assert ops.Add().datatypeResult([Datatype.I32, Datatype.F64]) == Datatype.F64
        assert ops.Add().datatypeResult([Datatype.I16, Datatype.I64]) == Datatype.I64

    def test_logical_not_yields_bool(self):
        assert ops.LogicalNot().datatypeResult([Datatype.I32]) == Datatype.BOOL

    def test_bitwise_not_rejects_bool(self):
        # ~x on a bool is never false; LogicalNot is the operation wanted there
        with pytest.raises(AssertionError):
            ops.Not().datatypeResult([Datatype.BOOL])


class TestSparsityPropagation:
    """A pattern may be over-approximated but never under-approximated:
    claiming a zero that is non-zero drops the value during code generation."""

    @staticmethod
    def pattern(*rows):
        return aspp.general(np.array(rows, dtype=bool))

    def test_zero_preserving_unary_keeps_the_pattern(self):
        spp = self.pattern([True, False], [False, True])
        for op in (ops.Sqrt(), ops.Sin(), ops.Tanh(), ops.Abs()):
            assert op.sparsityResult([spp]).count_nonzero() == 2

    def test_non_zero_preserving_unary_is_dense(self):
        spp = self.pattern([True, False], [False, False])
        for op in (ops.Cos(), ops.Exp(), ops.Cosh()):
            assert op.sparsityResult([spp]).is_dense()

    def test_addition_unions(self):
        left = self.pattern([True, False], [False, False])
        right = self.pattern([False, True], [False, False])
        assert ops.Add().sparsityResult([left, right]).count_nonzero() == 2

    def test_multiplication_intersects(self):
        left = self.pattern([True, True], [False, False])
        right = self.pattern([True, False], [False, False])
        assert ops.Mul().sparsityResult([left, right]).count_nonzero() == 1

    def test_comparisons_are_dense(self):
        spp = self.pattern([True, False], [False, False])
        assert ops.CmpGe().sparsityResult([spp, spp]).is_dense()

    def test_division_follows_the_numerator(self):
        num = self.pattern([True, False], [False, False])
        den = self.pattern([True, True], [True, True])
        assert ops.Div().sparsityResult([num, den]).count_nonzero() == 1

    def test_ternary_ignores_the_condition(self):
        yes = self.pattern([True, False], [False, False])
        no = self.pattern([False, True], [False, False])
        cond = self.pattern([True, True], [True, True])
        assert ops.Ternary().sparsityResult([yes, no, cond]).count_nonzero() == 2


class TestNumericSemantics:
    """The Python evaluation has to agree with what the emitted C++ computes."""

    def test_min_max_work_elementwise(self):
        a = np.array([1.0, 5.0]); b = np.array([3.0, 2.0])
        assert np.array_equal(ops.Min().call(a, b), [1.0, 2.0])
        assert np.array_equal(ops.Max().call(a, b), [3.0, 5.0])

    def test_inverse_trigonometry_is_available(self):
        values = np.array([0.0, 0.5])
        for op in (ops.Asin(), ops.Atan(), ops.Asinh(), ops.Atanh()):
            assert op.call(values).shape == values.shape

    def test_cast_changes_the_dtype(self):
        result = ops.Typecast(Datatype.I32).call(np.array([1.7, -2.3]))
        assert result.dtype == np.int32
        assert np.array_equal(result, [1, -2])

    def test_xor_exists_and_is_involutive(self):
        a = np.array([0b1010, 0b0101])
        assert np.array_equal(ops.Xor().call(a, a), [0, 0])


class TestSparsityHelpers:
    def test_multiply_intersects(self):
        left = aspp.general(np.array([[True, True], [False, False]]))
        right = aspp.general(np.array([[True, False], [True, False]]))
        assert aspp.multiply(left, right).count_nonzero() == 1

    def test_nonzero_indices_on_a_matrix(self):
        spp = aspp.general(np.array([[True, False], [False, True]]))
        assert sorted(aspp.nonzeroIndices(spp)) == [(0, 0), (1, 1)]

    def test_nonzero_indices_on_a_rank_zero_pattern(self):
        # numpy refuses nonzero() on 0-d arrays, but rank-0 tensors have a
        # pattern too -- every condition variable is one
        assert aspp.nonzeroIndices(aspp.dense(())) == [()]
        assert aspp.nonzeroIndices(aspp.general(np.array(False))) == []
