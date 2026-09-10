"""Scalar arithmetic and the prologue it is computed in.

Anything that reads only scalars is hoisted to the top of the kernel, ahead of
every statement and every external routine call.
"""

import os
import re
import tempfile

import pytest

from yateto import Generator, GeneratorCollection, Tensor, ops
from yateto.arch import useArchitectureIdentifiedBy
from yateto.ast.cost import BoundingBoxCostEstimator
from yateto.controlflow.visitor import DerivedScalarsList, ScalarsSet
from yateto.generator import Kernel
from yateto.type import Datatype, DerivedScalar, Scalar, ScalarExpression

import yateto.functions as yf

N = 6


@pytest.fixture
def arch():
    return useArchitectureIdentifiedBy('dhsw')


@pytest.fixture
def q():
    return {
        'A': Tensor('A', (N, N)),
        'B': Tensor('B', (N, N)),
        'C': Tensor('C', (N, N)),
        'alpha': Scalar('alpha'),
        'beta': Scalar('beta'),
        'gamma': Scalar('gamma'),
    }


def factorOf(expr):
    factor, _ = expr.scalingOperands()
    return factor


class TestScalarExpressions:
    def test_multiplication(self, q):
        assert isinstance(q['alpha'] * q['beta'], DerivedScalar)

    @pytest.mark.parametrize('build', [
        lambda a, b: a * b,
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a / b,
        lambda a, b: 2.0 * a,
        lambda a, b: 2.0 + a,
        lambda a, b: 2.0 - a,
        lambda a, b: 2.0 / a,
        lambda a, b: -a,
    ])
    def test_every_operator_yields_a_derived_scalar(self, q, build):
        assert isinstance(build(q['alpha'], q['beta']), DerivedScalar)

    def test_an_expression_reads_its_scalars(self, q):
        expr = (q['alpha'] * q['beta']) / q['gamma']
        assert {s.name() for s in expr.dependencies()} == {'alpha', 'beta', 'gamma'}

    def test_a_literal_is_not_a_dependency(self, q):
        assert {s.name() for s in (q['alpha'] * 2.0).dependencies()} == {'alpha'}

    def test_expressions_flatten(self, q):
        # combining a derived operand takes its expression, so there is never a
        # chain of derived scalars to order in the prologue
        expr = q['alpha'] * q['beta'] * q['gamma']
        assert all(not isinstance(o, DerivedScalar) for o in expr.expression.operands)

    def test_identical_expressions_share_one_scalar(self, q):
        assert (q['alpha'] * q['beta']) is (q['alpha'] * q['beta'])

    def test_different_expressions_do_not(self, q):
        assert (q['alpha'] * q['beta']) is not (q['alpha'] + q['beta'])

    def test_a_derived_scalar_is_temporary(self, q):
        assert (q['alpha'] * q['beta']).temporary

    def test_the_datatype_is_promoted(self, arch):
        single = Scalar('s', datatype=Datatype.F32)
        double = Scalar('d', datatype=Datatype.F64)
        assert (single * double).getDatatype(arch) == Datatype.F64

    def test_the_datatype_falls_back_to_the_architecture(self, arch, q):
        assert (q['alpha'] * q['beta']).getDatatype(arch) == arch.datatype

    def test_a_scalar_expression_is_not_a_node(self, q):
        # it is a host-side computation, not part of the tensor tree
        from yateto.ast.node import Node
        assert not isinstance(q['alpha'] * q['beta'], Node)


class TestScalingCollapse:
    def test_two_numbers_collapse_at_build_time(self, q):
        assert factorOf(2.0 * (3.0 * q['A']['ij'])) == 6.0

    def test_a_number_and_a_name_become_one_derived_scalar(self, q):
        assert isinstance(factorOf(2.0 * (q['alpha'] * q['A']['ij'])), DerivedScalar)

    def test_two_names_become_one_derived_scalar(self, q):
        factor = factorOf(q['alpha'] * (q['beta'] * q['A']['ij']))
        assert {s.name() for s in factor.dependencies()} == {'alpha', 'beta'}

    def test_factors_on_both_sides_of_a_product_combine(self, q):
        expr = (q['alpha'] * q['A']['ik']) * (q['beta'] * q['B']['kj'])
        assert {s.name() for s in factorOf(expr).dependencies()} == {'alpha', 'beta'}

    def test_negating_a_scaled_term_folds_into_the_factor(self, q):
        assert factorOf(-(2.0 * q['A']['ij'])) == -2.0

    def test_a_term_carries_at_most_one_factor(self, q):
        expr = q['alpha'] * (q['beta'] * (2.0 * q['A']['ij']))
        assert not expr.scaledTerm().isScaling()

    def test_an_unscaled_product_has_no_factor(self, q):
        assert q['A']['ik'].splitScaling()[0] is None
        assert (q['A']['ik'] * q['B']['kj']).splitScaling()[0] is None


class TestPrologue:
    @staticmethod
    def lower(arch, statement):
        kernel = Kernel('k', statement)
        kernel.prepareUntilUnitTest(arch)
        kernel.prepareUntilCodeGen(BoundingBoxCostEstimator)
        return kernel

    @staticmethod
    def emit(arch, statements):
        generator = Generator(arch)
        for i, statement in enumerate(statements):
            generator.add(f'k{i}', statement)
        with tempfile.TemporaryDirectory() as out:
            generator.generate(out, gemm_cfg=GeneratorCollection([]))
            return (open(os.path.join(out, 'kernel.cpp')).read(),
                    open(os.path.join(out, 'kernel.h')).read())

    def test_a_derived_scalar_is_listed_for_the_prologue(self, arch, q):
        kernel = self.lower(arch, q['C']['ij'] <= (q['alpha'] * q['beta']) * q['A']['ij'])
        assert len(DerivedScalarsList().visit(kernel.cfg)) == 1

    def test_the_signature_holds_the_named_scalars(self, arch, q):
        kernel = self.lower(arch, q['C']['ij'] <= (q['alpha'] * q['beta']) * q['A']['ij'])
        names = {s.name() for s in ScalarsSet().visit(kernel.cfg)}
        assert names == {'alpha', 'beta'}

    def test_the_prologue_precedes_every_statement(self, arch, q):
        code, _ = self.emit(arch, [q['C']['ij'] <= (q['alpha'] * q['beta']) * q['A']['ij']])
        body = code[code.index('k0::execute'):]
        body = body[:body.index('\n  }\n')]
        assert re.search(r'const _s\d+ =', body)
        assert body.index('const _s') < body.index('for (')

    def test_the_prologue_declares_the_promoted_type(self, arch, q):
        code, _ = self.emit(arch, [q['C']['ij'] <= (q['alpha'] / q['beta']) * q['A']['ij']])
        assert re.search(r'double const _s\d+ = \(alpha\) / \(beta\);', code)

    def test_the_derived_scalar_is_not_a_kernel_argument(self, arch, q):
        _, header = self.emit(arch, [q['C']['ij'] <= (q['alpha'] * q['beta']) * q['A']['ij']])
        assert 'alpha' in header and 'beta' in header
        assert not re.search(r'_s\d+ = std::numeric_limits', header)

    def test_one_scalar_serves_several_statements(self, arch, q):
        code, _ = self.emit(arch, [[
            q['C']['ij'] <= (q['alpha'] * q['beta']) * q['A']['ij'],
            q['C']['ij'] <= q['C']['ij'] + (q['alpha'] * q['beta']) * q['B']['ij'],
        ]])
        body = code[code.index('k0::execute'):]
        body = body[:body.index('\n  }\n')]
        assert len(re.findall(r'const _s\d+ =', body)) == 1

    def test_a_guarded_factor_is_still_hoisted(self, arch, q):
        flag = Tensor('flag', (), datatype=Datatype.BOOL)
        code, _ = self.emit(arch, [
            yf.assignIf(flag[''], q['C']['ij'], (q['alpha'] * q['beta']) * q['A']['ij'])])
        body = code[code.index('k0::execute'):]
        body = body[:body.index('\n  }\n')]
        # it reads no kernel result, so it is computed once rather than per branch
        assert body.index('const _s') < body.index('if (')

    def test_a_purely_numeric_factor_needs_no_prologue(self, arch, q):
        code, _ = self.emit(arch, [q['C']['ij'] <= 2.0 * (3.0 * q['A']['ij'])])
        assert not re.search(r'const _s\d+ =', code)
        assert '6.0' in code
