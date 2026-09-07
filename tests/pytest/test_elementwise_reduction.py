"""Elementwise and Reduction: index deduction, broadcasting, and agreement
between the Python evaluation and the generated C++."""

import os
import re
import tempfile

import numpy as np
import pytest

from yateto import Generator, GeneratorCollection, Tensor
from yateto.arch import useArchitectureIdentifiedBy
from yateto.ast.cost import BoundingBoxCostEstimator
from yateto.ast.transformer import DeduceIndices, EquivalentSparsityPattern
from yateto.ast.visitor import ComputeConstantExpression
from yateto.generator import Kernel
from yateto.type import Datatype

import yateto.functions as yf

N = 6


@pytest.fixture
def arch():
    return useArchitectureIdentifiedBy('dhsw')


def constants(**arrays):
    return {name: Tensor(name, values.shape, values) for name, values in arrays.items()}


def deduce(expression, outIndices, outShape):
    """Deduce indices the way a kernel does: through an assignment."""
    result = Tensor('result', outShape)
    assignment = result[outIndices] <= expression
    DeduceIndices().visit(assignment)
    return assignment


def deduced(expression, outIndices, outShape):
    return deduce(expression, outIndices, outShape).rightTerm()


def evaluate(expression, outIndices, outShape):
    assignment = deduce(expression, outIndices, outShape)
    EquivalentSparsityPattern().visit(assignment)
    return ComputeConstantExpression().visit(assignment.rightTerm())


class TestIndexDeduction:
    def test_matching_indices_are_kept(self):
        A = Tensor('A', (N, N)); B = Tensor('B', (N, N))
        node = deduced(yf.maximum(A['ij'], B['ij']), 'ij', (N, N))
        assert str(node.indices) == 'ij'

    def test_operands_may_be_permuted(self):
        A = Tensor('A', (N, N)); B = Tensor('B', (N, N))
        node = deduced(yf.maximum(A['ij'], B['ji']), 'ij', (N, N))
        assert set(str(node.indices)) == {'i', 'j'}

    def test_a_shorter_operand_broadcasts(self):
        A = Tensor('A', (N, N)); v = Tensor('v', (N,))
        node = deduced(yf.mul(A['ij'], v['j']), 'ij', (N, N))
        assert set(str(node.indices)) == {'i', 'j'}

    def test_mismatching_index_sizes_are_rejected(self):
        A = Tensor('A', (N, N)); B = Tensor('B', (N, N + 1))
        with pytest.raises((ValueError, AssertionError)):
            deduced(yf.maximum(A['ij'], B['ij']), 'ij', (N, N))

    def test_a_reduction_drops_its_index(self):
        A = Tensor('A', (N, N))
        node = deduced(yf.sum(A['ij'], 'j'), 'i', (N,))
        assert str(node.indices) == 'i'

    def test_reducing_every_index_gives_a_scalar(self):
        A = Tensor('A', (N, N))
        node = deduced(yf.sum(A['ij'], 'ij'), '', ())
        assert len(node.indices) == 0

    def test_a_reduction_over_a_composite_term(self):
        A = Tensor('A', (N, N)); v = Tensor('v', (N,))
        node = deduced(yf.sum(yf.mul(A['ij'], v['j']), 'j'), 'i', (N,))
        assert str(node.indices) == 'i'

    def test_arity_is_enforced(self):
        A = Tensor('A', (N, N))
        from yateto.ast.node import Elementwise
        from yateto import ops
        with pytest.raises(ValueError):
            Elementwise(ops.Sqrt(), A['ij'], A['ij'])


class TestConstantEvaluation:
    """The Python evaluation is the reference the generated code is tested against."""

    def test_unary(self):
        values = np.abs(np.random.default_rng(0).random((N, N))) + 0.5
        t = constants(A=values)
        assert np.allclose(evaluate(yf.sqrt(t['A']['ij']), 'ij', (N, N)), np.sqrt(values))

    def test_binary(self):
        rng = np.random.default_rng(1)
        a, b = rng.random((N, N)), rng.random((N, N))
        t = constants(A=a, B=b)
        assert np.allclose(evaluate(yf.minimum(t['A']['ij'], t['B']['ij']), 'ij', (N, N)),
                           np.minimum(a, b))

    def test_a_permuted_operand_is_aligned(self):
        rng = np.random.default_rng(2)
        a, b = rng.random((N, N)), rng.random((N, N))
        t = constants(A=a, B=b)
        assert np.allclose(evaluate(yf.minimum(t['A']['ij'], t['B']['ji']), 'ij', (N, N)),
                           np.minimum(a, b.T))

    def test_reduction_sum(self):
        values = np.random.default_rng(3).random((N, N))
        t = constants(A=values)
        assert np.allclose(evaluate(yf.sum(t['A']['ij'], 'j'), 'i', (N,)), values.sum(axis=1))

    def test_reduction_min(self):
        values = np.random.default_rng(4).random((N, N))
        t = constants(A=values)
        assert np.allclose(evaluate(yf.min(t['A']['ij'], 'j'), 'i', (N,)), values.min(axis=1))

    def test_nested_reduction(self):
        values = np.random.default_rng(5).random((N, N))
        t = constants(A=values)
        assert np.allclose(evaluate(yf.sum(t['A']['ij'], 'ij'), '', ()), values.sum())


class TestGeneratedCode:
    @staticmethod
    def emit(arch, statements):
        generator = Generator(arch)
        for i, statement in enumerate(statements):
            generator.add(f'k{i}', statement)
        with tempfile.TemporaryDirectory() as out:
            generator.generate(out, gemm_cfg=GeneratorCollection([]))
            return open(os.path.join(out, 'kernel.cpp')).read()

    def test_every_loop_variable_used_is_declared(self, arch):
        A = Tensor('A', (N, N)); v = Tensor('v', (N,)); out = Tensor('out', (N, N))
        scalar = Tensor('scalar', ())
        code = self.emit(arch, [
            out['ij'] <= yf.mul(A['ij'], v['j']),
            scalar[''] <= yf.sum(A['ij'], 'ij'),
            scalar[''] <= yf.min(A['ij'], 'ij'),
        ])
        declared = set(re.findall(r'for \(int (\w+) =', code))
        used = set(re.findall(r'\[[^\]]*?(\b_[a-z]\b)', code))
        assert used <= declared, f'undeclared: {used - declared}'

    def test_several_reductions_do_not_clash(self, arch):
        A = Tensor('A', (N, N)); s1 = Tensor('s1', ()); s2 = Tensor('s2', ())
        # two rank-0 reductions share one enclosing scope
        code = self.emit(arch, [[
            s1[''] <= yf.sum(A['ij'], 'ij'),
            s2[''] <= yf.max(A['ij'], 'ij'),
        ]])
        body = code[code.index('k0::execute'):]
        body = body[:body.index('\n  }\n')]
        for line in body.split('\n'):
            assert not re.match(r'^\s{4}\w+ _acc =', line), \
                'an accumulator escaped into the shared scope'

    def test_a_broadcast_operand_uses_its_own_indices(self, arch):
        A = Tensor('A', (N, N)); v = Tensor('v', (N,)); out = Tensor('out', (N, N))
        code = self.emit(arch, [out['ij'] <= yf.mul(A['ij'], v['j'])])
        assert re.search(r'v\[1\*_[a-z]\]', code), 'the vector should be indexed once'

    def test_integer_kernels_do_not_get_floating_point_literals(self, arch):
        AI = Tensor('AI', (N, N), datatype=Datatype.I32)
        BI = Tensor('BI', (N, N), datatype=Datatype.I32)
        code = self.emit(arch, [AI['ij'] <= -BI['ij']])
        body = code[code.index('k0::execute'):]
        body = body[:body.index('\n  }\n')]
        assert '-1.0 *' not in body

    def test_a_bitwise_and_reduction_starts_from_all_ones(self, arch):
        AI = Tensor('AI', (N, N), datatype=Datatype.I32)
        s = Tensor('s', (), datatype=Datatype.I32)
        code = self.emit(arch, [s[''] <= yf.all(AI['ij'], 'ij')])
        # 1 would clear every bit but the lowest
        assert re.search(r'_acc = static_cast<int32_t>\(-1LL\)', code)

    def test_a_float_min_reduction_starts_from_infinity(self, arch):
        A = Tensor('A', (N, N)); s = Tensor('s', ())
        code = self.emit(arch, [s[''] <= yf.min(A['ij'], 'ij')])
        assert 'std::numeric_limits<double>::infinity()' in code


class TestFlopCounts:
    def test_a_reduction_counts_its_additions(self, arch):
        A = Tensor('A', (N, N)); s = Tensor('s', ())
        kernel = Kernel('k', s[''] <= yf.sum(A['ij'], 'ij'))
        kernel.prepareUntilUnitTest(arch)
        kernel.prepareUntilCodeGen(BoundingBoxCostEstimator, enableFusedGemm=False)
        assert kernel.nonZeroFlops > 0

    def test_an_elementwise_counts_its_entries(self, arch):
        A = Tensor('A', (N, N)); out = Tensor('out', (N, N))
        kernel = Kernel('k', out['ij'] <= yf.sqrt(A['ij']))
        kernel.prepareUntilUnitTest(arch)
        kernel.prepareUntilCodeGen(BoundingBoxCostEstimator, enableFusedGemm=False)
        assert kernel.nonZeroFlops == N * N
