"""Contraction and the ring it runs on.

A reduction over a product is a contraction only on the (*, +) ring -- the one
the GEMM backends implement. Every other ring keeps its structure and is
generated as loops.
"""

import re
import tempfile

import numpy as np
import pytest

from yateto import Generator, GeneratorCollection, Tensor, ops
from yateto.arch import useArchitectureIdentifiedBy
from yateto.ast.cost import BoundingBoxCostEstimator, isProduct, isSummation
from yateto.ast.node import Contraction, Elementwise, LoopOverGEMM, Reduction
from yateto.ast.transformer import DeduceIndices, FindContractions, StrengthReduction
from yateto.ast.visitor import PrettyPrinter
from yateto.generator import Kernel
from yateto.type import Datatype, Scalar

import yateto.functions as yf

N = 6


@pytest.fixture
def arch():
    return useArchitectureIdentifiedBy('dhsw')


@pytest.fixture
def tensors():
    return {
        'A': Tensor('A', (N, N)),
        'B': Tensor('B', (N, N)),
        'C': Tensor('C', (N, N)),
        'AB': Tensor('AB', (N, N), datatype=Datatype.BOOL),
        'BB': Tensor('BB', (N, N), datatype=Datatype.BOOL),
        'CB': Tensor('CB', (N, N), datatype=Datatype.BOOL),
    }


def lower(arch, statement):
    kernel = Kernel('k', statement)
    kernel.prepareUntilUnitTest(arch)
    kernel.prepareUntilCodeGen(BoundingBoxCostEstimator, enableFusedGemm=False)
    return kernel


def nodesOfType(node, kind, found=None):
    found = [] if found is None else found
    if isinstance(node, kind):
        found.append(node)
    for child in node:
        nodesOfType(child, kind, found)
    return found


class TestContractionDetection:
    def test_the_arithmetic_ring_becomes_a_gemm(self, arch, tensors):
        t = tensors
        kernel = lower(arch, t['C']['ij'] <= t['A']['ik'] * t['B']['kj'])
        assert nodesOfType(kernel.ast, LoopOverGEMM)

    def test_or_over_and_stays_a_reduction(self, arch, tensors):
        t = tensors
        kernel = lower(arch, t['CB']['ij'] <= yf.any(
            yf.bitwise_and(t['AB']['ik'], t['BB']['kj']), 'k'))
        assert not nodesOfType(kernel.ast, LoopOverGEMM)
        assert nodesOfType(kernel.ast, Reduction)

    def test_and_over_or_stays_a_reduction(self, arch, tensors):
        t = tensors
        kernel = lower(arch, t['CB']['ij'] <= yf.all(
            yf.bitwise_or(t['AB']['ik'], t['BB']['kj']), 'k'))
        assert not nodesOfType(kernel.ast, LoopOverGEMM)

    def test_the_tropical_semiring_stays_a_reduction(self, arch, tensors):
        t = tensors
        kernel = lower(arch, t['C']['ij'] <= yf.min(
            yf.add(t['A']['ik'], t['B']['kj']), 'k'))
        assert not nodesOfType(kernel.ast, LoopOverGEMM)

    def test_max_over_times_stays_a_reduction(self, arch, tensors):
        t = tensors
        kernel = lower(arch, t['C']['ij'] <= yf.max(
            yf.mul(t['A']['ik'], t['B']['kj']), 'k'))
        assert not nodesOfType(kernel.ast, LoopOverGEMM)

    def test_a_sum_over_a_non_product_is_not_a_contraction(self, arch, tensors):
        t = tensors
        kernel = lower(arch, t['C']['ij'] <= yf.sum(
            yf.maximum(t['A']['ik'], t['B']['kj']), 'k'))
        assert not nodesOfType(kernel.ast, LoopOverGEMM)

    def test_a_product_without_a_reduction_is_not_a_contraction(self, arch, tensors):
        t = tensors
        kernel = lower(arch, t['C']['ij'] <= yf.mul(t['A']['ij'], t['B']['ij']))
        assert not nodesOfType(kernel.ast, LoopOverGEMM)

    def test_an_integer_contraction_is_still_a_contraction(self, arch):
        AI = Tensor('AI', (N, N), datatype=Datatype.I32)
        BI = Tensor('BI', (N, N), datatype=Datatype.I32)
        CI = Tensor('CI', (N, N), datatype=Datatype.I32)
        # whether BLAS can run it is a separate question, decided per tool
        kernel = lower(arch, CI['ij'] <= AI['ik'] * BI['kj'])
        assert nodesOfType(kernel.ast, LoopOverGEMM)


class TestRingPredicates:
    def test_a_binary_multiplication_is_a_product(self, tensors):
        t = tensors
        assert isProduct(Elementwise(ops.Mul(), t['A']['ik'], t['B']['kj']))

    def test_multiplication_is_binary(self, tensors):
        t = tensors
        # a contraction is binary; three factors are contracted pairwise, and
        # the operation refuses a third operand outright
        with pytest.raises(ValueError):
            Elementwise(ops.Mul(), t['A']['ij'], t['B']['ij'], t['C']['ij'])

    def test_another_operation_is_not_a_product(self, tensors):
        t = tensors
        assert not isProduct(Elementwise(ops.Max(), t['A']['ij'], t['B']['ij']))

    def test_a_summation_is_recognised(self, tensors):
        t = tensors
        assert isSummation(Reduction(ops.Add(), t['A']['ij'], 'j'))
        assert not isSummation(Reduction(ops.Min(), t['A']['ij'], 'j'))


class TestEagerIndexDeduction:
    """The contraction search builds bottom-up and reads indices off fresh nodes."""

    def test_a_product_knows_its_indices_immediately(self, tensors):
        t = tensors
        node = Elementwise(ops.Mul(), t['A']['ik'], t['B']['kj'])
        assert set(str(node.indices)) == {'i', 'j', 'k'}

    def test_a_reduction_knows_its_indices_immediately(self, tensors):
        t = tensors
        product = Elementwise(ops.Mul(), t['A']['ik'], t['B']['kj'])
        node = Reduction(ops.Add(), product, 'k')
        assert set(str(node.indices)) == {'i', 'j'}

    def test_an_accumulation_knows_its_indices_immediately(self, tensors):
        t = tensors
        node = t['A']['ij'] + t['B']['ij']
        assert str(node.indices) == 'ij'

    def test_deduction_is_deferred_when_a_child_is_unknown(self, tensors):
        t = tensors
        # an Einsum has no indices until DeduceIndices runs
        inner = t['A']['ik'] * t['B']['kj']
        assert inner.indices is None
        outer = Elementwise(ops.Mul(), inner, t['C']['ij'])
        assert outer.indices is None

    def test_mismatching_index_sizes_are_rejected_eagerly(self):
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N + 1))
        with pytest.raises((ValueError, AssertionError)):
            Elementwise(ops.Mul(), A['ij'], B['ij'])

    def test_an_empty_accumulation_is_allowed(self):
        from yateto.ast.node import Accumulate
        # a one-element sum is assembled as Accumulate() + term
        node = Accumulate(ops.Add())
        assert node.indices is None


class TestGeneratedRings:
    @staticmethod
    def emit(arch, statements):
        import os
        generator = Generator(arch)
        for i, statement in enumerate(statements):
            generator.add(f'k{i}', statement)
        with tempfile.TemporaryDirectory() as out:
            generator.generate(out, gemm_cfg=GeneratorCollection([]))
            return open(os.path.join(out, 'kernel.cpp')).read()

    def test_the_boolean_semiring_uses_the_right_neutral_element(self, arch, tensors):
        t = tensors
        code = self.emit(arch, [t['CB']['ij'] <= yf.any(
            yf.bitwise_and(t['AB']['ik'], t['BB']['kj']), 'k')])
        assert '_acc = false' in code
        assert '|' in code and '&' in code

    def test_the_tropical_semiring_starts_from_infinity(self, arch, tensors):
        t = tensors
        code = self.emit(arch, [t['C']['ij'] <= yf.min(
            yf.add(t['A']['ik'], t['B']['kj']), 'k')])
        assert 'std::numeric_limits<double>::infinity()' in code
        assert 'std::min' in code

    def test_a_scale_factor_folds_into_the_contraction(self, arch, tensors):
        t = tensors
        alpha = Scalar('alpha')
        code = self.emit(arch, [t['C']['ij'] <= alpha * t['A']['ik'] * t['B']['kj']])
        body = code[code.index('k0::execute'):]
        body = body[:body.index('\n  }\n')]
        # one loop nest, with alpha inside it -- not a separate scaling pass
        assert 'alpha *' in body
        assert body.count('for (int n') == 1

    def test_every_loop_variable_used_is_declared(self, arch, tensors):
        t = tensors
        code = self.emit(arch, [
            t['C']['ij'] <= yf.min(yf.add(t['A']['ik'], t['B']['kj']), 'k'),
            t['CB']['ij'] <= yf.all(yf.bitwise_or(t['AB']['ik'], t['BB']['kj']), 'k'),
        ])
        declared = set(re.findall(r'for \(int (\w+) =', code))
        used = set(re.findall(r'\[[^\]]*?(\b_[a-z]\b)', code))
        assert used <= declared, f'undeclared: {used - declared}'
