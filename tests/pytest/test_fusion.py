"""
Tests for ``FindFusedElementwise``, the pass that puts adjacent element-wise
steps into one loop nest.

The pass is deliberately strict, so most of what is worth testing is what it
declines to do. A step that does not walk exactly the index space the group
walks, or does not walk it under the same guard, or leaves something behind
that is read outside the group, ends the group -- and each of those is a way
to generate code that reads outside what an operand stores, or that runs a
statement where it should not.
"""
from __future__ import annotations

import io
import contextlib
import pathlib
import tempfile

import numpy as np
import pytest

import yateto.functions as yf
from yateto import Datatype, Generator, Scalar, Tensor, useArchitectureIdentifiedBy
from yateto.gemm_configuration import GeneratorCollection
from yateto.memory import CSCMemoryLayout

N = 6


def emit(statements, name='k'):
    """The generated kernel body for one kernel made of `statements`."""
    generator = Generator(useArchitectureIdentifiedBy('dhsw'))
    generator.add(name, statements)
    with tempfile.TemporaryDirectory() as out:
        with contextlib.redirect_stdout(io.StringIO()):
            generator.generate(out, gemm_cfg=GeneratorCollection([]))
        code = (pathlib.Path(out) / 'kernel.cpp').read_text()
    body = code[code.index(f'{name}::execute'):]
    return body[:body.index('\n  }\n')]


def steps(body):
    """How many intermediates the nest keeps in registers."""
    return body.count('_fused')


def nests(body):
    return body.count('#pragma omp simd')


class TestFuses:
    def test_a_chain_over_one_index_space_becomes_one_nest(self):
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= yf.maximum(yf.sqrt(yf.add(A['ij'], B['ij'])), B['ij'])])
        assert nests(body) == 1
        assert steps(body) > 0

    def test_a_scaling_is_a_step_like_any_other(self):
        """It is an element-wise multiplication seen from the control-flow
        graph, where it is an action's factor rather than a node. Leaving it
        out breaks a group wherever a kernel scales something mid-chain."""
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        C = Tensor('C', (N, N))
        alpha = Scalar('alpha')
        body = emit([C['ij'] <= yf.add(alpha * A['ij'], yf.sqrt(B['ij']))])
        assert nests(body) == 1

    def test_an_intermediate_read_late_is_still_kept_in_a_register(self):
        """A chain is a tree, not a line: the first thing computed is often
        read by the last thing done. A group that stopped at the first result
        with a later reader would never grow past it."""
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= yf.where(yf.greater(yf.sqrt(B['ij']), B['ij']),
                                         yf.mul(A['ij'], B['ij']),
                                         A['ij'])])
        assert nests(body) == 1
        assert 'int8_t _buffer' not in body


class TestDeclines:
    def test_a_different_index_space_ends_the_group(self):
        """Outer products grow the space step by step, so no two of them share
        a nest."""
        a = [Tensor(chr(ord('a') + k), (N,)) for k in range(4)]
        T = Tensor('T', (N,) * 4)
        body = emit([T['abcd'] <= a[0]['a'] * a[1]['b'] * a[2]['c'] * a[3]['d']])
        assert steps(body) == 0

    def test_a_broadcast_operand_ends_the_group(self):
        """It is addressed over fewer axes than the nest walks."""
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        v = Tensor('v', (N,))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= yf.add(yf.mul(A['ij'], v['i']), B['ij'])])
        assert steps(body) == 0

    def test_a_sparse_operand_ends_the_group(self):
        """A sparse operand is written out entry by entry, not looped over."""
        pattern = np.zeros((N, N), dtype=bool)
        pattern[0, 0] = pattern[1, 1] = pattern[3, 2] = True
        P = Tensor('P', (N, N), spp=pattern)
        P.setMemoryLayout(CSCMemoryLayout)
        Q = Tensor('Q', (N, N), spp=pattern)
        Q.setMemoryLayout(CSCMemoryLayout)
        R = Tensor('R', (N, N))
        body = emit([R['ij'] <= yf.add(yf.maximum(P['ij'], Q['ij']), P['ij'])])
        assert steps(body) == 0

    def test_a_different_guard_ends_the_group(self):
        """Two statements that do not run together may not share a nest."""
        A = Tensor('A', (N, N))
        C = Tensor('C', (N, N))
        flag = Tensor('flag', (), datatype=Datatype.BOOL)
        body = emit([yf.assignIf(flag[''], C['ij'], yf.sqrt(A['ij'])),
                     C['ij'] <= yf.add(A['ij'], A['ij'])])
        assert steps(body) == 0

    def test_operand_ranges_have_to_agree_before_the_pass_is_reached(self):
        """Every operand is read over the nest's range, so it has to have that
        range. The front end already refuses a mismatch, which is what the
        pass's own check on the operands' boxes rests on."""
        wide = np.ones((N, N), dtype=bool)
        narrow = np.zeros((N, N), dtype=bool)
        narrow[1:3, 1:3] = True
        A = Tensor('A', (N, N), spp=wide)
        B = Tensor('B', (N, N), spp=narrow)
        C = Tensor('C', (N, N), spp=wide)
        with pytest.raises(AssertionError, match='Inconsistent loop range'):
            emit([C['ij'] <= yf.add(yf.maximum(A['ij'], A['ij']), B['ij'])])


class TestStaysCorrect:
    def test_the_factor_of_a_fused_scaling_reaches_the_signature(self):
        """It moves out of the action and into the step, and the kernel's
        scalars are collected from the action."""
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        C = Tensor('C', (N, N))
        mean = Scalar('mean')
        body = emit([C['ij'] <= yf.add(mean * A['ij'], yf.sqrt(B['ij']))])
        assert 'mean' in body
