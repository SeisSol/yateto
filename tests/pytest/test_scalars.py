"""Scalars and rank-0 tensors share one interface.

What separates them is the calling convention -- a scalar is handed over by
value, a tensor by pointer -- and nothing else.
"""

import os
import re
import tempfile

import numpy as np
import pytest

from yateto import Generator, GeneratorCollection, Tensor, ops
from yateto.arch import useArchitectureIdentifiedBy
from yateto.ast.cost import BoundingBoxCostEstimator
from yateto.ast.node import Elementwise, IndexedTensor
from yateto.ast.visitor import FindTensors
from yateto.generator import Kernel
from yateto.type import AddressingMode, Datatype, DerivedScalar, Scalar

import yateto.functions as yf

N = 6


@pytest.fixture
def arch():
    return useArchitectureIdentifiedBy('dhsw')


@pytest.fixture
def quantities():
    return {
        'scalar': Scalar('alpha'),
        'rank0': Tensor('s', ()),
        'matrix': Tensor('A', (N, N)),
    }


class TestRankZeroInterface:
    @pytest.mark.parametrize('key', ['scalar', 'rank0'])
    def test_shape_is_empty(self, quantities, key):
        assert quantities[key].shape() == ()

    @pytest.mark.parametrize('key', ['scalar', 'rank0'])
    def test_one_entry_of_storage(self, quantities, key):
        assert quantities[key].memoryLayout().requiredReals() == 1

    @pytest.mark.parametrize('key', ['scalar', 'rank0'])
    def test_sparsity_pattern_is_rank_zero(self, quantities, key):
        assert quantities[key].spp().shape == ()

    @pytest.mark.parametrize('key', ['scalar', 'rank0'])
    def test_indexing_with_the_empty_index(self, quantities, key):
        assert isinstance(quantities[key][''], IndexedTensor)

    def test_indices_are_rejected(self, quantities):
        with pytest.raises((ValueError, AssertionError)):
            quantities['scalar']['ij']

    def test_a_scalar_is_a_tensor(self, quantities):
        assert isinstance(quantities['scalar'], Tensor)

    def test_a_scalar_matches_a_rank_zero_tensor(self, quantities):
        scalar, rank0 = quantities['scalar'], quantities['rank0']
        assert scalar.shape() == rank0.shape()
        assert scalar.memoryLayout().requiredReals() == rank0.memoryLayout().requiredReals()


class TestCallingConvention:
    def test_a_scalar_is_passed_by_value(self, quantities):
        assert quantities['scalar'].addressing == AddressingMode.SCALAR
        assert quantities['scalar'].isPassedByValue()

    def test_a_tensor_is_passed_by_pointer(self, quantities):
        assert quantities['rank0'].addressing != AddressingMode.SCALAR
        assert not quantities['rank0'].isPassedByValue()
        assert not quantities['matrix'].isPassedByValue()

    def test_a_scalar_and_a_rank_zero_tensor_are_different(self, quantities):
        # same name would collide in the signature, but they are not the same
        # thing: one is passed by value, the other by pointer
        assert Scalar('x') != Tensor('x', ())

    def test_a_scalar_is_not_a_tensor_argument(self, quantities):
        scalar, matrix = quantities['scalar'], quantities['matrix']
        result = Tensor('C', (N, N))
        found = FindTensors().visit(result['ij'] <= scalar * matrix['ij'])
        assert 'alpha' not in found
        assert 'A' in found and 'C' in found

    def test_a_rank_zero_tensor_is_a_tensor_argument(self, quantities):
        rank0, matrix = quantities['rank0'], quantities['matrix']
        result = Tensor('C', (N, N))
        found = FindTensors().visit(result['ij'] <= yf.mul(rank0[''], matrix['ij']))
        assert 's' in found


class TestScalingLowering:
    """A by-value rank-0 operand becomes a scale factor, not a loop."""

    @staticmethod
    def emit(arch, statements, gemm_cfg=None):
        generator = Generator(arch)
        for i, statement in enumerate(statements):
            generator.add(f'k{i}', statement)
        with tempfile.TemporaryDirectory() as out:
            generator.generate(out, gemm_cfg=gemm_cfg or GeneratorCollection([]))
            return (open(os.path.join(out, 'kernel.cpp')).read(),
                    open(os.path.join(out, 'kernel.h')).read())

    def test_a_named_scalar_is_declared_by_value(self, arch, quantities):
        A, C = quantities['matrix'], Tensor('C', (N, N))
        _, header = self.emit(arch, [C['ij'] <= Scalar('alpha') * A['ij']])
        assert re.search(r'double alpha\b', header)
        assert not re.search(r'double\s*\*\s*alpha', header)

    def test_a_rank_zero_tensor_is_declared_by_pointer(self, arch, quantities):
        A, C, s = quantities['matrix'], Tensor('C', (N, N)), quantities['rank0']
        _, header = self.emit(arch, [C['ij'] <= yf.mul(s[''], A['ij'])])
        assert re.search(r'double const\s*\*\s*s', header)

    def test_a_scaling_does_not_get_its_own_loop(self, arch, quantities):
        A, C = quantities['matrix'], Tensor('C', (N, N))
        code, _ = self.emit(arch, [C['ij'] <= 2.0 * A['ij']])
        body = code[code.index('k0::execute'):]
        body = body[:body.index('\n  }\n')]
        assert body.count('for (') == 2, 'one loop nest over i and j, no extra pass'

    def test_a_rank_zero_operand_does_get_a_loop(self, arch, quantities):
        A, C, s = quantities['matrix'], Tensor('C', (N, N)), quantities['rank0']
        code, _ = self.emit(arch, [C['ij'] <= yf.mul(s[''], A['ij'])])
        assert 's[0]' in code

    def test_an_integer_scaling_keeps_the_integer_type(self, arch):
        AI = Tensor('AI', (N, N), datatype=Datatype.I32)
        BI = Tensor('BI', (N, N), datatype=Datatype.I32)
        code, _ = self.emit(arch, [AI['ij'] <= -BI['ij']])
        body = code[code.index('k0::execute'):]
        body = body[:body.index('\n  }\n')]
        assert '-1.0' not in body
        assert 'int32_t' in body

    def test_the_scale_factor_reaches_the_program_action(self, arch, quantities):
        A, C = quantities['matrix'], Tensor('C', (N, N))
        kernel = Kernel('k', C['ij'] <= 2.0 * A['ij'])
        kernel.prepareUntilUnitTest(arch)
        kernel.prepareUntilCodeGen(BoundingBoxCostEstimator)
        scalars = [pp.action.scalar for pp in kernel.cfg if pp.action is not None]
        assert 2.0 in scalars

    def test_a_scalar_cannot_be_written(self, quantities):
        scalar, rank0 = quantities['scalar'], quantities['rank0']
        with pytest.raises(ValueError, match='passed by value'):
            scalar[''] <= rank0['']

    def test_a_rank_zero_tensor_can_be_written(self, quantities):
        rank0, matrix = quantities['rank0'], quantities['matrix']
        assert (rank0[''] <= yf.sum(matrix['ij'], 'ij')) is not None

    def test_nested_numeric_scalings_collapse(self, quantities):
        A = quantities['matrix']
        expr = 2.0 * (3.0 * A['ij'])
        factor, _ = expr.scalingOperands()
        assert factor == 6.0

    def test_nested_named_scalings_become_one_derived_scalar(self, quantities):
        A = quantities['matrix']
        alpha, beta = Scalar('alpha'), Scalar('beta')
        expr = alpha * (beta * A['ij'])
        factor, _ = expr.scalingOperands()
        assert isinstance(factor, DerivedScalar)
        assert {s.name() for s in factor.dependencies()} == {'alpha', 'beta'}


class TestScalingSemantics:
    def test_a_sign_flip_is_free(self, arch, quantities):
        A, C = quantities['matrix'], Tensor('C', (N, N))
        kernel = Kernel('k', C['ij'] <= -A['ij'])
        kernel.prepareUntilUnitTest(arch)
        kernel.prepareUntilCodeGen(BoundingBoxCostEstimator)
        assert kernel.nonZeroFlops == 0

    def test_a_general_factor_is_not_free(self, arch, quantities):
        A, C = quantities['matrix'], Tensor('C', (N, N))
        kernel = Kernel('k', C['ij'] <= 2.0 * A['ij'])
        kernel.prepareUntilUnitTest(arch)
        kernel.prepareUntilCodeGen(BoundingBoxCostEstimator)
        assert kernel.nonZeroFlops > 0

    def test_scaling_a_product_keeps_the_product(self, quantities):
        A, B = quantities['matrix'], Tensor('B', (N, N))
        expr = 2.0 * (A['ik'] * B['kj'])
        symbol, term = expr.scalingOperands()
        assert symbol == 2.0
        assert term is not None

    def test_the_operands_follow_a_replaced_child(self, quantities):
        A, B = quantities['matrix'], Tensor('B', (N, N))
        expr = 2.0 * A['ij']
        replacement = B['ij']
        expr.setScaledTerm(replacement)
        # terms is derived from the children, so it must show the new one
        assert replacement in expr.terms
        assert expr.scaledTerm() is replacement


class TestScalarOperand:
    """A scalar handed to an operation directly, rather than through `*`."""

    @staticmethod
    def _emit(statements):
        import pathlib
        import tempfile
        from yateto import Generator, useArchitectureIdentifiedBy
        from yateto.gemm_configuration import GeneratorCollection
        arch = useArchitectureIdentifiedBy('dhsw')
        generator = Generator(arch)
        for i, statement in enumerate(statements):
            generator.add(f'k{i}', statement)
        with tempfile.TemporaryDirectory() as out:
            generator.generate(out, gemm_cfg=GeneratorCollection([]))
            return ((pathlib.Path(out) / 'kernel.cpp').read_text(),
                    (pathlib.Path(out) / 'kernel.h').read_text())

    def test_a_scalar_operand_reaches_the_signature(self):
        """It used to be carried as a template, i.e. written into the
        expression as if it were a literal, and the kernel then named
        something it never declared."""
        import yateto.functions as yf
        A = Tensor('A', (8, 8))
        C = Tensor('C', (8, 8))
        alpha = Scalar('alpha')
        code, header = self._emit([C['ij'] <= yf.mul(alpha, A['ij'])])
        assert 'double alpha' in header
        assert 'alpha' in code

    def test_it_agrees_with_the_operator(self):
        import yateto.functions as yf
        A = Tensor('A', (8, 8))
        C = Tensor('C', (8, 8))
        alpha = Scalar('alpha')
        viaOperator, _ = self._emit([C['ij'] <= alpha * A['ij']])
        viaFunction, _ = self._emit([C['ij'] <= yf.mul(alpha, A['ij'])])
        assert viaOperator == viaFunction

    def test_a_scalar_summand_reaches_the_signature(self):
        """Not only a factor: any operation may take one.

        NOTE: a scalar that is not a factor arrives as a pointer rather than by
        value, because only the scaling path routes it through ScalarsSet. The
        kernel is consistent -- it declares the name and reads `beta[0]` -- but
        that is not what Scalar promises.
        """
        import yateto.functions as yf
        A = Tensor('A', (8, 8))
        C = Tensor('C', (8, 8))
        beta = Scalar('beta')
        code, header = self._emit([C['ij'] <= yf.add(beta, A['ij'])])
        assert 'beta' in header
        assert 'beta' in code

    def test_a_literal_stays_a_literal(self):
        """A number needs neither storage nor a name."""
        import yateto.functions as yf
        A = Tensor('A', (8, 8))
        C = Tensor('C', (8, 8))
        code, header = self._emit([C['ij'] <= yf.maximum(2.5, A['ij'])])
        assert '2.5' in code


class TestScaledOperations:
    """A factor applies to the operation's result, not to its first operand."""

    @staticmethod
    def _emit(statements):
        import pathlib
        import tempfile
        from yateto import Generator, useArchitectureIdentifiedBy
        from yateto.gemm_configuration import GeneratorCollection
        generator = Generator(useArchitectureIdentifiedBy('dhsw'))
        for i, statement in enumerate(statements):
            generator.add(f'k{i}', statement)
        with tempfile.TemporaryDirectory() as out:
            generator.generate(out, gemm_cfg=GeneratorCollection([]))
            return (pathlib.Path(out) / 'kernel.cpp').read_text()

    def test_a_scaled_operation_is_parenthesised(self):
        """`*` binds tighter than `+`, `&`, `|`, `^` and every comparison."""
        import yateto.functions as yf
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        C = Tensor('C', (N, N))
        code = self._emit([C['ij'] <= 2.0 * yf.add(A['ij'], B['ij'])])
        entry = lambda name: f'{name}[1*_i + {N}*_j]'
        assert f"2.0 * ({entry('A')} + {entry('B')})" in code

    def test_a_boolean_result_is_not_scaled(self):
        """Every non-zero factor is the same boolean, so it would be lost."""
        import pytest as _pytest
        import yateto.functions as yf
        A = Tensor('A', (N, N), datatype=Datatype.BOOL)
        S = Tensor('S', (N, N), datatype=Datatype.BOOL)
        with _pytest.raises(ValueError, match='boolean'):
            self._emit([S['ij'] <= 2.0 * yf.logical_and(A['ij'], A['ij'])])

    def test_an_integer_result_is_not_scaled_by_a_fraction(self):
        import pytest as _pytest
        AI = Tensor('AI', (N, N), datatype=Datatype.I32)
        CI = Tensor('CI', (N, N), datatype=Datatype.I32)
        with _pytest.raises(ValueError, match='truncate'):
            self._emit([CI['ij'] <= 2.5 * AI['ij']])

    def test_a_whole_factor_on_an_integer_result_is_fine(self):
        AI = Tensor('AI', (N, N), datatype=Datatype.I32)
        CI = Tensor('CI', (N, N), datatype=Datatype.I32)
        code = self._emit([CI['ij'] <= 2.0 * AI['ij']])
        body = code[code.index('k0::execute'):]
        body = body[:body.index('\n  }\n')]
        assert 'int32_t>(2LL)' in body
        assert '2.0' not in body


class TestByValueOperand:
    """A by-value operand stays by value wherever it appears."""

    @staticmethod
    def _emit(statements):
        import pathlib
        import tempfile
        from yateto import Generator, useArchitectureIdentifiedBy
        from yateto.gemm_configuration import GeneratorCollection
        generator = Generator(useArchitectureIdentifiedBy('dhsw'))
        for i, statement in enumerate(statements):
            generator.add(f'k{i}', statement)
        with tempfile.TemporaryDirectory() as out:
            generator.generate(out, gemm_cfg=GeneratorCollection([]))
            return ((pathlib.Path(out) / 'kernel.cpp').read_text(),
                    (pathlib.Path(out) / 'kernel.h').read_text())

    def test_an_operand_is_read_by_name(self):
        import yateto.functions as yf
        A = Tensor('A', (N, N))
        C = Tensor('C', (N, N))
        beta = Scalar('beta')
        code, header = self._emit([C['ij'] <= yf.add(beta, A['ij'])])
        assert 'double beta' in header
        assert 'beta[0]' not in code

    def test_one_scalar_used_two_ways_is_declared_once(self):
        """As a factor it went through ScalarsSet and as an operand through the
        variables, so the kernel declared the name twice with two types."""
        import yateto.functions as yf
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        C = Tensor('C', (N, N))
        beta = Scalar('beta')
        # one kernel, two statements: the same name reached both collections
        code, header = self._emit([[C['ij'] <= beta * A['ij'],
                                    C['ij'] <= yf.add(beta, B['ij'])]])
        assert header.count('double beta') == 1
        assert 'double const* beta' not in header
        assert 'beta[0]' not in code
