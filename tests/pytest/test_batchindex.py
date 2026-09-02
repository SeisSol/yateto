"""
Tests for batch (diagonal) indices in loop-over-GEMM.

A *batch index* occurs in the result **and** in both operands of a
contraction, e.g.::

    R['sIJ'] <= M['ij'] * Q['siI'] * Q['sjJ']

Here ``s`` is not a GEMM dimension at all -- it selects one independent
problem per value. This is the shape SeisSol's fused simulations produce,
where every tensor carries a leading simulation index.

``yateto.ast.log`` recognises such indices as ``Icommon``, removes them from
``m``/``n``/``k`` and lets ``LoopOverGEMM.loopIndices`` turn them into loops.
The code generator then has to build matching two-dimensional views of the
operands and the result. The tricky part is that removing the batch index can
leave the ``m`` dimension empty, which turns the GEMM into ``(1 x n)``: for the
operands that is absorbed by ``transA``/``transB``, but the result has no
``transC`` and needs its dummy dimension in front.
"""

import numpy as np
import pytest

from yateto import Tensor
from yateto.ast.node import Assign
from yateto.memory import DenseMemoryLayout, Range


class TestWithDummyDimension:
    """DenseMemoryLayout.withDummyDimension(front=...)"""

    def test_appended_by_default(self):
        ml = DenseMemoryLayout((8,))
        dummy = ml.withDummyDimension()
        assert dummy.shape() == (8, 1)

    def test_prepended_on_request(self):
        ml = DenseMemoryLayout((8,))
        dummy = ml.withDummyDimension(front=True)
        assert dummy.shape() == (1, 8)

    def test_prepended_bounding_box(self):
        ml = DenseMemoryLayout((8,))
        dummy = ml.withDummyDimension(front=True)
        assert dummy.bbox()[0] == Range(0, 1)
        assert dummy.bbox()[1] == Range(0, 8)

    def test_prepended_addressing_is_the_identity_on_the_vector(self):
        """A 1 x N view must address element j at offset j."""
        ml = DenseMemoryLayout((8,))
        dummy = ml.withDummyDimension(front=True)
        for j in range(8):
            assert dummy.address((0, j)) == ml.address((j,))


class TestBatchIndexPlanning:
    """ast/log.py must keep batch indices out of m/n/k."""

    @staticmethod
    def _kernelAst(tmp_path, equationFactory):
        from yateto import Generator, useArchitectureIdentifiedBy
        from yateto.gemm_configuration import GeneratorCollection

        g = Generator(useArchitectureIdentifiedBy('dhsw'))
        equationFactory(g)
        g.generate(str(tmp_path), gemm_cfg=GeneratorCollection([]))
        return [k.ast for k in g.kernels()]

    @staticmethod
    def _collectLogs(node, out=None):
        from yateto.ast.node import LoopOverGEMM

        if out is None:
            out = []
        if isinstance(node, LoopOverGEMM):
            out.append(node)
        for child in node:
            TestBatchIndexPlanning._collectLogs(child, out)
        return out

    @staticmethod
    def _isBatched(log, index):
        """An index is a batch index iff it occurs in the result *and* in both
        operands. An index shared by only two of the three is an ordinary
        m/n/k dimension."""
        return all(index in str(t) for t in (log.indices,
                                             log.leftTerm().indices,
                                             log.rightTerm().indices))

    def test_batch_index_becomes_a_loop_index(self, tmp_path):
        S, N, P = 8, 20, 9

        def build(g):
            Q = Tensor('Q', (S, N, P))
            M = Tensor('M', (N, N))
            R = Tensor('R', (S, P, P))
            g.add('batchedGram', R['sIJ'] <= M['ij'] * Q['siI'] * Q['sjJ'])

        asts = self._kernelAst(tmp_path, build)
        logs = [log for ast in asts for log in self._collectLogs(ast)]
        assert logs, 'expected at least one LoopOverGEMM'

        batched = [log for log in logs if self._isBatched(log, 's')]
        assert batched, 'expected a LoopOverGEMM with a batch index'
        for log in batched:
            assert 's' in str(log.loopIndices()), (
                'the batch index must be a loop index, not a GEMM dimension'
            )

    def test_two_batch_indices(self, tmp_path):
        S, T, N, P = 4, 3, 12, 6

        def build(g):
            Q = Tensor('Q', (S, T, N, P))
            M = Tensor('M', (N, N))
            R = Tensor('R', (S, T, P, P))
            g.add('batchedGram2', R['stIJ'] <= M['ij'] * Q['stiI'] * Q['stjJ'])

        asts = self._kernelAst(tmp_path, build)
        logs = [log for ast in asts for log in self._collectLogs(ast)]
        batched = [log for log in logs
                   if self._isBatched(log, 's') and self._isBatched(log, 't')]
        assert batched, 'expected a LoopOverGEMM with two batch indices'
        for log in batched:
            loop = str(log.loopIndices())
            assert 's' in loop and 't' in loop


class TestBatchIndexCodegen:
    """End-to-end: the generator must not raise on batch indices."""

    @pytest.mark.parametrize('multipleSimulations', [1, 8])
    def test_batched_gram_generates(self, tmp_path, multipleSimulations):
        from yateto import Generator, useArchitectureIdentifiedBy
        from yateto.gemm_configuration import GeneratorCollection

        S, N, P = multipleSimulations, 20, 9
        arch = useArchitectureIdentifiedBy('dhsw')
        g = Generator(arch)

        Q = Tensor('Q', (S, N, P))
        M = Tensor('M', (N, N))
        R = Tensor('R', (S, P, P))
        g.add('batchedGram', R['sIJ'] <= M['ij'] * Q['siI'] * Q['sjJ'])

        g.generate(str(tmp_path), gemm_cfg=GeneratorCollection([]))
        assert (tmp_path / 'kernel.cpp').exists()

    def test_batched_gram_is_correct(self, tmp_path):
        """Compare the generated flop count against the analytic one."""
        from yateto import Generator, useArchitectureIdentifiedBy
        from yateto.gemm_configuration import GeneratorCollection

        S, N, P = 8, 20, 9
        arch = useArchitectureIdentifiedBy('dhsw')
        g = Generator(arch)

        Q = Tensor('Q', (S, N, P))
        M = Tensor('M', (N, N))
        R = Tensor('R', (S, P, P))
        g.add('batchedGram', R['sIJ'] <= M['ij'] * Q['siI'] * Q['sjJ'])
        g.generate(str(tmp_path), gemm_cfg=GeneratorCollection([]))

        # M^T Q  (S*P GEMMs of N x N times N x 1) plus the contraction with Q
        # (S*P GEMMs of 1 x N times N x P); 2 flops per multiply-add.
        expected = 2 * S * P * N * N + 2 * S * P * N * P
        kernel = next(k for k in g.kernels() if k.name == 'batchedGram')
        assert kernel.nonZeroFlops <= expected
