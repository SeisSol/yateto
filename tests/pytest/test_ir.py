"""
Tests for the intermediate representation and for the copy-scale-add backend,
which is generated from it.

Two things are worth pinning down. The first is that an address is an affine
expression and behaves like one: pinning an index to a value has to fold it
away, or the unrolled path has no addresses at all. The second is what comes
out the other end, since the backend's whole job is the C++ it writes.
"""
from __future__ import annotations

import contextlib
import io
import pathlib
import tempfile

import numpy as np
import pytest

import yateto.functions as yf
from yateto import Generator, Scalar, Tensor, ops, useArchitectureIdentifiedBy
from yateto.ast.node import Reduction as ReductionNode
from yateto import aspp, ir
from yateto.codegen.code import Cpp
from yateto.codegen.common import IndexedTensorDescription
from yateto.codegen.copyscaleadd.factory import Description
from yateto.codegen.copyscaleadd.generic import tensorOp
from yateto.codegen.lowering import lower
from yateto.gemm_configuration import GeneratorCollection
from yateto.ast.indices import Indices, Range
from yateto.memory import CSCMemoryLayout, DenseMemoryLayout
from yateto.type import Datatype

N = 4


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


def description(name, indices, shape, layout=None, datatype=Datatype.F64, spp=None):
    pattern = aspp.general(np.ones(shape, dtype=bool) if spp is None else spp)
    return IndexedTensorDescription(
        name, Indices(indices, shape),
        layout if layout is not None else DenseMemoryLayout(shape),
        pattern, datatype=datatype)


def lowered(alpha, beta, result, term):
    """The region a copy-scale-add becomes, after unrolling."""
    descr = Description(alpha=alpha, beta=beta, result=result, term=term)
    return ir.unroll(tensorOp(descr).lower())


def code(region):
    out = io.StringIO()
    with Cpp(out) as cpp:
        ir.CppEmitter(cpp).emit(region)
        return out.getvalue()


class TestAffine:
    def test_terms_over_the_same_index_are_added_up(self):
        i = ir.Index('i')
        expression = 2 * ir.Affine.of(i) + 3 * ir.Affine.of(i)
        assert expression.coefficient(i) == 5
        assert expression.ccode() == '5*_i'

    def test_a_coefficient_that_cancels_leaves_no_term(self):
        i = ir.Index('i')
        expression = ir.Affine.of(i) - ir.Affine.of(i)
        assert expression.isConstant()
        assert expression.constant() == 0

    def test_pinning_an_index_folds_it_into_the_constant(self):
        i, j = ir.Index('i'), ir.Index('j')
        expression = 1 * ir.Affine.of(i) + 8 * ir.Affine.of(j) + 3
        assert not expression.isConstant()
        pinned = expression.substituted({i: 2, j: 1})
        assert pinned.isConstant()
        assert pinned.constant() == 13

    def test_a_negative_constant_is_written_as_a_subtraction(self):
        i = ir.Index('i')
        assert (ir.Affine.of(i) - 3).ccode() == '1*_i - 3'


class TestAddress:
    def test_a_dense_layout_addresses_by_stride(self):
        layout = DenseMemoryLayout((4, 6))
        i, j = ir.Index('i'), ir.Index('j')
        assert ir.address(layout, [i, j]).ccode() == '1*_i + 4*_j'

    def test_a_bounding_box_shifts_the_address(self):
        spp = np.zeros((4, 6), dtype=bool)
        spp[1:3, 2:4] = True
        layout = DenseMemoryLayout.fromSpp(aspp.general(spp))
        i, j = ir.Index('i'), ir.Index('j')
        pinned = ir.address(layout, [ir.Affine(1), ir.Affine(2)])
        assert pinned.isConstant()
        assert pinned.constant() == layout.address((1, 2))
        assert not ir.address(layout, [i, j]).isConstant()

    def test_a_sparse_layout_has_no_address_for_an_index(self):
        spp = np.eye(4, dtype=bool)
        layout = CSCMemoryLayout(aspp.general(spp))
        assert ir.address(layout, [ir.Affine(2), ir.Affine(2)]).isConstant()
        with pytest.raises(ValueError, match='Unroll'):
            ir.address(layout, [ir.Index('i'), ir.Index('j')])


class TestLowering:
    def test_a_dense_copy_becomes_one_loop_per_index(self):
        result = description('C', 'ij', (N, N))
        term = description('A', 'ij', (N, N))
        region = lowered(1.0, 0.0, result, term)
        loops = [op for op in region.walk() if isinstance(op, ir.Loop)]
        assert len(loops) == 2
        assert [op for op in region.walk() if isinstance(op, ir.Store)]

    def test_a_sparse_operand_states_its_entries_and_is_unrolled(self):
        spp = np.eye(N, dtype=bool)
        result = description('C', 'ij', (N, N))
        term = description('S', 'ij', (N, N),
                           layout=CSCMemoryLayout(aspp.general(spp)), spp=spp)
        descr = Description(alpha=1.0, beta=0.0, result=result, term=term)
        region = tensorOp(descr).lower()

        entryLoops = [op for op in region.walk()
                      if isinstance(op, ir.Loop) and op.isUnrollable()]
        assert len(entryLoops) == 1

        ir.unroll(region)
        assert not [op for op in region.walk() if isinstance(op, ir.Loop)]
        stores = [op for op in region.walk() if isinstance(op, ir.Store)]
        assert len(stores) == int(result.eqspp.count_nonzero())
        assert all(coord.isConstant() for store in stores for coord in store.coords)

    def test_an_entry_the_operand_does_not_store_becomes_a_zero(self):
        spp = np.eye(N, dtype=bool)
        result = description('C', 'ij', (N, N))
        term = description('S', 'ij', (N, N),
                           layout=CSCMemoryLayout(aspp.general(spp)), spp=spp)
        region = lowered(1.0, 0.0, result, term)

        loads = [op for op in region.walk() if isinstance(op, ir.Load)]
        assert len(loads) == int(spp.sum())
        # the entries off the diagonal are written, but nothing is read for them
        assert code(region).count('S[') == int(spp.sum())
        assert code(region).count('= 0.0;') == N * N - int(spp.sum())


class TestFlops:
    def test_a_plain_copy_does_no_arithmetic(self):
        result = description('C', 'ij', (N, N))
        term = description('A', 'ij', (N, N))
        assert ir.countFlops(lowered(1.0, 0.0, result, term)) == 0

    def test_a_factor_is_one_operation_per_entry(self):
        result = description('C', 'ij', (N, N))
        term = description('A', 'ij', (N, N))
        assert ir.countFlops(lowered(2.0, 0.0, result, term)) == N * N

    def test_an_accumulation_is_one_operation_per_entry(self):
        result = description('C', 'ij', (N, N))
        term = description('A', 'ij', (N, N))
        assert ir.countFlops(lowered(1.0, 1.0, result, term)) == N * N

    def test_a_subtraction_is_one_operation_and_not_two(self):
        result = description('C', 'ij', (N, N))
        term = description('A', 'ij', (N, N))
        assert ir.countFlops(lowered(-1.0, 1.0, result, term)) == N * N


class TestElementwise:
    def test_an_operand_that_binds_tightest_is_not_bracketed(self):
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= yf.add(A['ij'], B['ij'])])
        assert f'A[1*_i + {N}*_j] + B[1*_i + {N}*_j]' in body

    def test_a_factor_brackets_the_operation_it_scales(self):
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= 2.0 * yf.mul(A['ij'], B['ij'])])
        assert f'2.0 * (A[1*_i + {N}*_j] * B[1*_i + {N}*_j])' in body

    def test_accumulating_a_negated_operation_is_a_subtraction(self):
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= C['ij'] - yf.mul(A['ij'], B['ij'])])
        assert '-=' in body
        assert '-1.0' not in body

    def test_a_nest_computes_what_it_needs_where_it_needs_it(self):
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        D = Tensor('D', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= yf.maximum(yf.add(A['ij'], B['ij']), D['ij'])])
        assert body.count('#pragma omp simd') == 1
        assert '_tmp' not in body
        assert 'std::max((A[' in body

    def test_a_nest_reads_no_buffer_for_its_intermediates(self):
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        D = Tensor('D', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= yf.mul(yf.add(A['ij'], B['ij']),
                                       yf.sqrt(yf.mul(D['ij'], A['ij'])))])
        assert body.count('#pragma omp simd') == 1
        assert '_tmp' not in body

    def test_a_factor_the_statement_does_not_state_is_no_factor(self):
        from yateto.ir.lower import _factor
        assert _factor(None, Datatype.F64) is None
        assert _factor(1.0, Datatype.F64) is None

    def test_a_step_of_a_nest_keeps_its_factor(self):
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        D = Tensor('D', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= yf.maximum(2.0 * yf.add(A['ij'], B['ij']), D['ij'])])
        assert f'2.0 * (A[1*_i + {N}*_j] + B[1*_i + {N}*_j])' in body

    def test_a_named_factor_on_a_step_is_read_outside_the_nest(self):
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        D = Tensor('D', (N, N))
        C = Tensor('C', (N, N))
        s = Scalar('s')
        body = emit([C['ij'] <= yf.maximum(s * yf.add(A['ij'], B['ij']), D['ij'])])
        assert body.index('double const _alpha = s;') < body.index('for (')
        assert body.count('_alpha *') == 1

    def test_a_sum_that_feeds_the_nest_stays_in_it(self):
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        D = Tensor('D', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= (A['ij'] + B['ij']) * D['ij']])
        assert body.count('#pragma omp simd') == 1
        assert '_tmp' not in body
        assert f'A[1*_a + {N}*_b] + B[1*_a + {N}*_b]' in body

    def test_a_longer_sum_stays_in_the_nest_too(self):
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        D = Tensor('D', (N, N))
        E = Tensor('E', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= (A['ij'] + B['ij'] + E['ij']) * D['ij']])
        assert body.count('#pragma omp simd') == 1
        assert '_tmp' not in body

    def test_a_sum_written_straight_to_the_result_shares_one_nest(self):
        # The destination is a buffer the caller sees, so every step of the
        # chain reaches it -- but they reach it from one pass over the index
        # space rather than from one pass each.
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        D = Tensor('D', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= A['ij'] + B['ij'] + D['ij']])
        assert body.count('#pragma omp simd') == 1
        assert body.count('C[1*_a + 4*_b]') == 3

    def test_a_sparse_operand_reads_a_zero_where_it_has_no_entry(self):
        left = np.zeros((N, N), dtype=bool)
        left[0, 0] = left[1, 1] = True
        right = np.zeros((N, N), dtype=bool)
        right[0, 0] = right[2, 1] = True
        A = Tensor('A', (N, N), spp=left, memoryLayoutClass=CSCMemoryLayout)
        B = Tensor('B', (N, N), spp=right, memoryLayoutClass=CSCMemoryLayout)
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= yf.add(A['ij'], B['ij'])])
        assert 'for (' not in body
        assert 'C[0] = A[0] + B[0];' in body
        assert '+ 0.0;' in body

    def test_a_named_factor_is_read_once_however_the_nest_runs(self):
        spp = np.eye(N, dtype=bool)
        A = Tensor('A', (N, N), spp=spp, memoryLayoutClass=CSCMemoryLayout)
        C = Tensor('C', (N, N))
        s = Scalar('s')
        body = emit([C['ij'] <= s * yf.sqrt(A['ij'])])
        assert body.count('double const _alpha = s;') == 1
        assert body.count('s *') == 0


def rowFold(optype):
    """Folding the second index of a matrix away, and where it lands."""
    A = Tensor('A', (N, N))
    u = Tensor('u', (N,))
    return u, ReductionNode(optype, A['ij'], 'j')


class TestReduction:
    def test_a_fold_starts_from_the_neutral_element(self):
        u, reduction = rowFold(ops.Max())
        body = emit([u['i'] <= reduction])
        assert 'double _acc = -std::numeric_limits<double>::infinity();' in body
        assert '_acc = std::max(_acc, A[' in body

    def test_a_fold_combines_once_per_step(self):
        u, reduction = rowFold(ops.Add())
        body = emit([u['i'] <= reduction])
        assert f'_acc += A[1*_i + {N}*_j];' in body
        assert 'u[1*_i] = _acc;' in body

    def test_the_running_value_is_not_const(self):
        u, reduction = rowFold(ops.Add())
        assert 'double const _acc' not in emit([u['i'] <= reduction])

    def test_a_destination_without_axes_gets_a_scope_all_the_same(self):
        w = Tensor('w', (N,))
        z = Tensor('z', ())
        body = emit([z[''] <= ReductionNode(ops.Add(), w['i'], 'i')])
        assert 'double _acc = 0.0;' in body
        assert 'z[0] = _acc;' in body

    def test_accumulating_uses_the_operation_that_was_folded(self):
        u, reduction = rowFold(ops.Add())
        assert 'u[1*_i] += _acc;' in emit([u['i'] <= u['i'] + reduction])

    def test_the_flops_count_one_combination_per_step(self):
        from yateto.codegen.reduction.factory import Description
        from yateto.codegen.reduction.generic import tensorOp
        descr = Description(alpha=1.0, add=False,
                            result=description('u', 'i', (N,)),
                            term=description('A', 'ij', (N, N)),
                            optype=ops.Add())
        assert ir.countFlops(tensorOp(descr).lower()) == N * N


class TestContraction:
    def test_a_loop_over_products_moves_a_pointer_per_operand(self):
        A = Tensor('A', (N, N))
        T = Tensor('T', (N, N, 3))
        U = Tensor('U', (N, N, 3))
        body = emit([U['ijl'] <= T['ikl'] * A['kj']])
        assert f'double const* _A = T + {N * N}*_l;' in body
        assert 'double const* _B = A;' in body
        assert f'double * _C = U + {N * N}*_l;' in body

    def test_a_pointer_that_moves_along_nothing_carries_no_offset(self):
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= A['ik'] * B['kj']])
        assert '_A = A +' not in body
        assert '+ 0]' not in body.split('=')[0]

    def test_the_flops_of_a_call_come_from_the_callee(self):
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        C = Tensor('C', (N, N))
        with tempfile.TemporaryDirectory() as out:
            generator = Generator(useArchitectureIdentifiedBy('dhsw'))
            generator.add('k', [C['ij'] <= A['ik'] * B['kj']])
            with contextlib.redirect_stdout(io.StringIO()):
                generator.generate(out, gemm_cfg=GeneratorCollection([]))
            header = (pathlib.Path(out) / 'kernel.h').read_text()
        # the generic product assigns once and multiplies and adds per step
        assert f'HardwareFlops = {2 * N * N * N}' in header

    def test_a_call_states_no_arithmetic_before_it_is_written(self):
        from yateto.ir.ops import Call
        region = ir.Region([Call(lambda cpp, cache: 7)])
        with pytest.raises(AssertionError, match='after emitting'):
            ir.countFlops(region)


class TestKernelRegion:
    def test_the_storage_statements_share_is_settled_before_they_run(self):
        A = Tensor('A', (N, N))
        B = Tensor('B', (N, N))
        D = Tensor('D', (N, N))
        C = Tensor('C', (N, N))
        # two statements with a temporary between them: both reach the same
        # region, and what stands next to what is a question it can answer
        body = emit([C['ij'] <= A['ik'] * B['kj'] * D['ij']])
        assert '_tmp0 = reinterpret_cast' in body
        assert body.index('reinterpret_cast') < body.index('for (')

    def test_a_guarded_statement_is_written_inside_its_condition(self):
        import yateto.functions as functions
        A = Tensor('A', (N, N))
        C = Tensor('C', (N, N))
        flag = Tensor('flag', (), addressing='scalar')
        body = emit([functions.assignIf(flag[''], C['ij'], A['ij'])])
        assert body.count('if (') == 1
        assert body.index('if (') < body.index('C[')

    def test_a_statement_whose_guard_never_holds_is_not_written(self):
        A = Tensor('A', (N, N))
        C = Tensor('C', (N, N))
        from yateto.codegen.factory import OptimizedKernelFactory
        from yateto.controlflow.graph import Guard
        factory = OptimizedKernelFactory(None, useArchitectureIdentifiedBy('dhsw'), 'cpu')
        region = factory._conditional(Guard.never(), lambda: 1 / 0)
        assert len(region) == 0
        assert ir.countFlops(region) == 0


class TestFusion:
    def _loop(self, index, buffers, coord=None):
        """A nest over `index` storing into each buffer in turn."""
        body = ir.Region()
        builder = ir.Builder(body)
        for target, source in buffers:
            value = builder.add(ir.Load(source, [coord or index]))
            builder.add(ir.Store(target, [coord or index], value))
        return ir.Loop([index], Range(0, N), body)

    def _buffer(self, name):
        return ir.Buffer(name, Datatype.F64, DenseMemoryLayout((N,)))

    def test_nests_over_one_space_become_one(self):
        index = ir.Index('i')
        A, B, C = (self._buffer(name) for name in 'ABC')
        region = ir.Region([self._loop(index, [(C, A)]),
                            self._loop(ir.Index('i'), [(C, B)])])
        ir.fuseLoops(region)
        loops = [op for op in region.ops if isinstance(op, ir.Loop)]
        assert len(loops) == 1
        assert len([op for op in loops[0].region.walk()
                    if isinstance(op, ir.Store)]) == 2

    def test_nests_over_different_spaces_are_left_alone(self):
        A, B, C = (self._buffer(name) for name in 'ABC')
        first = self._loop(ir.Index('i'), [(C, A)])
        second = self._loop(ir.Index('i'), [(C, B)])
        second.domain = Range(0, N - 1)
        region = ir.Region([first, second])
        ir.fuseLoops(region)
        assert len([op for op in region.ops if isinstance(op, ir.Loop)]) == 2

    def test_a_nest_that_hands_work_over_is_left_alone(self):
        A, B, C = (self._buffer(name) for name in 'ABC')
        called = ir.Loop([ir.Index('i')], Range(0, N),
                         ir.Region([ir.Call(lambda cpp, cache: 0)]))
        region = ir.Region([called, self._loop(ir.Index('i'), [(C, B)])])
        ir.fuseLoops(region)
        assert len([op for op in region.ops if isinstance(op, ir.Loop)]) == 2

    def test_a_zeroing_between_two_nests_moves_in_front_of_them(self):
        index = ir.Index('i')
        A, B, C = (self._buffer(name) for name in 'ABC')
        D = self._buffer('D')
        region = ir.Region([self._loop(index, [(C, A)]),
                            ir.Memset(D, 0, N),
                            self._loop(ir.Index('i'), [(D, B)])])
        ir.fuseLoops(region)
        assert isinstance(region.ops[0], ir.Memset)
        assert len([op for op in region.ops if isinstance(op, ir.Loop)]) == 1

    def test_a_zeroing_behind_the_group_is_emitted_once(self):
        """It stays behind the merged nest and is not looked at a second
        time, which would zero the buffer twice."""
        index = ir.Index('i')
        A, B, C = (self._buffer(name) for name in 'ABC')
        D = self._buffer('D')
        region = ir.Region([self._loop(index, [(C, A)]),
                            self._loop(ir.Index('i'), [(C, B)]),
                            ir.Memset(D, 0, N)])
        ir.fuseLoops(region)
        assert len([op for op in region.ops if isinstance(op, ir.Memset)]) == 1
        assert len([op for op in region.ops if isinstance(op, ir.Loop)]) == 1

    def test_a_zeroing_of_what_the_group_wrote_ends_the_group(self):
        index = ir.Index('i')
        A, B, C = (self._buffer(name) for name in 'ABC')
        region = ir.Region([self._loop(index, [(C, A)]),
                            ir.Memset(C, 0, N),
                            self._loop(ir.Index('i'), [(C, B)])])
        ir.fuseLoops(region)
        assert isinstance(region.ops[0], ir.Loop)
        assert isinstance(region.ops[1], ir.Memset)
        assert len([op for op in region.ops if isinstance(op, ir.Loop)]) == 2


class TestStorage:
    def _buffer(self, name, temporary=True):
        return ir.Buffer(name, Datatype.F64, DenseMemoryLayout((N,)),
                         temporary=temporary)

    def _statement(self, index, target, source):
        body = ir.Region()
        builder = ir.Builder(body)
        builder.add(ir.Store(target, [index],
                             builder.add(ir.Load(source, [index]))))
        return ir.Loop([index], Range(0, N), body)

    def test_temporaries_whose_lives_do_not_overlap_share(self):
        A, C = self._buffer('A', False), self._buffer('C', False)
        one, two = self._buffer('_t1'), self._buffer('_t2')
        region = ir.Region([self._statement(ir.Index('i'), one, A),
                            self._statement(ir.Index('i'), C, one),
                            self._statement(ir.Index('i'), two, A),
                            self._statement(ir.Index('i'), C, two)])
        shares, sizes = ir.assign(region)
        assert shares['_t1'] == shares['_t2']
        assert len(sizes) == 1

    def test_temporaries_that_overlap_do_not(self):
        A = self._buffer('A', False)
        one, two = self._buffer('_t1'), self._buffer('_t2')
        region = ir.Region([self._statement(ir.Index('i'), one, A),
                            self._statement(ir.Index('i'), two, one),
                            self._statement(ir.Index('i'), one, two)])
        shares, sizes = ir.assign(region)
        assert shares['_t1'] != shares['_t2']
        assert len(sizes) == 2

    def test_what_the_caller_sees_gets_no_share(self):
        A, C = self._buffer('A', False), self._buffer('C', False)
        region = ir.Region([self._statement(ir.Index('i'), C, A)])
        shares, sizes = ir.assign(region)
        assert shares == {} and sizes == {}

    def test_a_statement_that_does_not_say_stops_the_sharing(self):
        A, C = self._buffer('A', False), self._buffer('C', False)
        one, two = self._buffer('_t1'), self._buffer('_t2')
        region = ir.Region([self._statement(ir.Index('i'), one, A),
                            ir.Call(lambda cpp, cache: 0),
                            self._statement(ir.Index('i'), two, A)])
        shares, sizes = ir.assign(region)
        assert shares['_t1'] != shares['_t2']


class TestEmission:
    def test_a_copy_carries_no_factor(self):
        A = Tensor('A', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= A['ij']])
        assert '1.0 *' not in body
        assert 'C[1*_a + 4*_b] = A[1*_a + 4*_b];' in body

    def test_a_factor_is_written_in_the_result_datatype(self):
        A = Tensor('A', (N, N))
        C = Tensor('C', (N, N))
        assert '2.5 * A[' in emit([C['ij'] <= 2.5 * A['ij']])

    def test_a_named_factor_is_read_once_into_a_local(self):
        A = Tensor('A', (N, N))
        C = Tensor('C', (N, N))
        s = Scalar('s')
        body = emit([C['ij'] <= s * A['ij']])
        assert 'double const _alpha = s;' in body
        assert body.count('_alpha * A[') == 1

    def test_accumulating_a_negated_operand_is_a_subtraction(self):
        A = Tensor('A', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= C['ij'] - A['ij']])
        assert '-= A[' in body
        assert '-1.0' not in body

    def test_a_transposed_operand_addresses_the_other_order(self):
        A = Tensor('A', (N, N))
        C = Tensor('C', (N, N))
        assert 'C[1*_i + 4*_j] = A[1*_j + 4*_i];' in emit([C['ij'] <= A['ji']])

    def test_a_broadcast_operand_does_not_address_the_missing_index(self):
        v = Tensor('v', (N,))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= v['i']])
        assert 'v[1*_i]' in body
        assert 'v[1*_i +' not in body

    def test_a_destination_without_axes_is_zeroed_whole(self):
        a = Tensor('a', ())
        b = Tensor('b', ())
        body = emit([a[''] <= b['']])
        assert 'memset(a, 0, 1 * sizeof(double));' in body
        assert 'a[0] = b[0];' in body

    def test_a_nest_is_one_iteration_space(self):
        A = Tensor('A', (N, N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= A['ij']])
        assert body.count('#pragma omp simd') == 1
        assert '#pragma omp simd collapse(2)' in body

    def test_a_single_loop_asks_for_no_collapse(self):
        u = Tensor('u', (N,))
        v = Tensor('v', (N,))
        body = emit([u['i'] <= v['i']])
        assert '#pragma omp simd\n' in body
        assert 'collapse' not in body

    def test_a_sparse_operand_is_addressed_by_number(self):
        spp = np.eye(N, dtype=bool)
        S = Tensor('S', (N, N), spp=spp, memoryLayoutClass=CSCMemoryLayout)
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= S['ij']])
        assert 'for (' not in body
        assert 'C[0] = S[0];' in body


class TestViews:
    """A view starts somewhere inside the buffer it views.

    Two of them into one buffer therefore name the same coordinate and mean
    two different entries, which is what an access has to be told apart by.
    Getting it wrong is silent: the code compiles, the numbers are wrong.
    """

    def test_two_slices_of_one_tensor_are_read_where_each_starts(self):
        A = Tensor('A', (N, 2 * N))
        C = Tensor('C', (N, N))
        body = emit([C['ij'] <= (A['ij']).subslice('j', N, 2 * N)
                     * (A['ij']).subslice('j', 0, N)])
        assert 'A[1*_i + 4*_j + 16]' in body
        assert 'A[1*_i + 4*_j]' in body

    def test_a_load_of_what_a_slice_wrote_is_not_that_value(self):
        base = ir.Buffer('A', Datatype.F64, DenseMemoryLayout((N, N)))
        view = ir.Buffer('A', Datatype.F64,
                         DenseMemoryLayout((N, N)).subslice(1, 1, N))
        index = ir.Index('i')
        body = ir.Region()
        builder = ir.Builder(body)
        builder.add(ir.Store(view, [index, ir.Affine(0)], ir.Const(1.0, Datatype.F64)))
        read = builder.add(ir.Load(base, [index, ir.Affine(0)]))
        builder.add(ir.Store(base, [index, ir.Affine(1)], read))
        region = ir.Region([ir.Loop([index], Range(0, N), body)])
        ir.scalarize(region)
        assert len([op for op in region.walk() if isinstance(op, ir.Load)]) == 1


class TestStatementLowering:
    """A statement stands in the region as the statement it is.

    What writes it is settled by a pass afterwards, which is what lets
    anything in between read a contraction as a contraction rather than as a
    nest around somebody's matrix product.
    """

    class _Lowers:
        def __init__(self, region):
            self._region = region
            self.handed = None

        def lower(self, *arguments):
            self.handed = arguments
            return self._region

    class _Writes:
        def __init__(self):
            self.handed = None

        def generate(self, cpp, routineCache, *arguments):
            self.handed = (cpp, routineCache) + arguments
            return 17

    def _statement(self, generator):
        return ir.Copy(description('C', 'ij', (N, N)),
                       [description('A', 'ij', (N, N))], generator=generator)

    def test_a_backend_that_lowers_stands_in_the_statement_s_place(self):
        loop = ir.Loop([ir.Index('i')], Range(0, N))
        backend = self._Lowers(ir.Region([loop]))
        region = ir.Region([self._statement(backend)])
        lower(region, None)
        assert region.ops == [loop]

    def test_a_backend_that_writes_itself_becomes_a_call_stating_what_it_reaches(self):
        backend = self._Writes()
        region = ir.Region([self._statement(backend)])
        lower(region, None)
        call, = region.ops
        assert isinstance(call, ir.Call)
        assert {buffer.name for buffer in call.reads} == {'A', 'C'}
        assert [buffer.name for buffer in call.writes] == ['C']

    def test_the_call_is_written_by_whoever_emits_the_region(self):
        backend = self._Writes()
        region = ir.Region([self._statement(backend)])
        lower(region, None)
        cpp, cache = object(), object()
        assert region.ops[0].generate(cpp, cache) == 17
        assert backend.handed == (cpp, cache)

    def test_a_contraction_is_the_one_statement_asked_with_the_configuration(self):
        gemm_cfg = object()
        contraction = self._Lowers(ir.Region())
        elementwise = self._Lowers(ir.Region())
        lower(ir.Region([
            ir.LoopOverGEMM(description('C', 'ij', (N, N)),
                            [description('A', 'ik', (N, N)),
                             description('B', 'kj', (N, N))],
                            generator=contraction),
            self._statement(elementwise)]), gemm_cfg)
        assert contraction.handed == (gemm_cfg,)
        assert elementwise.handed == ()

    def test_a_guarded_statement_is_lowered_where_it_stands(self):
        loop = ir.Loop([ir.Index('i')], Range(0, N))
        guarded = ir.If('flag', ir.Region([self._statement(self._Lowers(ir.Region([loop])))]))
        region = ir.Region([guarded])
        lower(region, None)
        assert guarded.region.ops == [loop]


class TestChains:
    """Adjacent matrix products a backend takes as one chain.

    Whether it may is a question about what stands next to what, which is what
    the region answers. The backends themselves cannot be exercised here --
    neither chainforge nor the tiny tensor compiler is installed -- so what is
    pinned down is which statements are put together and which are not.
    """

    class _Chains:
        """A configuration with a backend that takes chains."""

        def __init__(self):
            self.gemmTools = []
            self.chained = []

        def __call__(self, arch, descr, gemm_cfg, target, attrs=None):
            self.chained.append(list(descr))
            return self

    def _gemm(self, name, left='A', right='B', indices=('ij', 'ik', 'kj')):
        result, a, b = indices
        return ir.LoopOverGEMM(
            description(name, result, (N,) * len(result)),
            [description(left, a, (N,) * len(a)),
             description(right, b, (N,) * len(b))])

    @pytest.fixture
    def fuse(self, monkeypatch):
        """`fuseChains`, with a backend that records the chains it is given."""
        from yateto.codegen.fused_gemms import chain

        def run(region, backend, available=True):
            monkeypatch.setattr(chain, 'available',
                                lambda gemm_cfg, target: available)
            monkeypatch.setattr(chain, 'generator', backend)
            return chain.fuseChains(region, None, backend, 'gpu')

        return run

    def test_adjacent_products_become_one_statement(self, fuse):
        backend = self._Chains()
        region = ir.Region([self._gemm('T'), self._gemm('C', left='T')])
        fuse(region, backend)
        chained, = region.ops
        assert isinstance(chained, ir.FusedGEMMs)
        assert len(chained.statements) == 2
        assert chained.result.name == 'C'

    def test_what_the_chain_reaches_is_what_its_members_reach(self, fuse):
        backend = self._Chains()
        region = ir.Region([self._gemm('T'), self._gemm('C', left='T')])
        fuse(region, backend)
        reached, written = region.ops[0].touched()
        assert {term.name for term in reached} == {'A', 'B', 'T', 'C'}
        assert [term.name for term in written] == ['T', 'C']

    def test_a_statement_between_them_ends_the_chain(self, fuse):
        backend = self._Chains()
        between = ir.Copy(description('D', 'ij', (N, N)),
                          [description('A', 'ij', (N, N))])
        region = ir.Region([self._gemm('T'), between, self._gemm('C', left='T')])
        fuse(region, backend)
        assert [type(op) for op in region.ops] == \
            [ir.FusedGEMMs, ir.Copy, ir.FusedGEMMs]

    def test_a_contraction_that_is_not_a_matrix_product_is_left_alone(self, fuse):
        backend = self._Chains()
        looped = self._gemm('C', indices=('ijl', 'ikl', 'kjl'))
        assert not looped.isPureGEMM()
        region = ir.Region([looped])
        fuse(region, backend)
        assert region.ops == [looped]

    def test_nothing_is_chained_where_no_backend_takes_chains(self, fuse):
        backend = self._Chains()
        first, second = self._gemm('T'), self._gemm('C', left='T')
        region = ir.Region([first, second])
        fuse(region, backend, available=False)
        assert region.ops == [first, second]

    def test_a_guarded_statement_is_chained_only_with_its_own_guard(self, fuse):
        """Each guard is a region of its own, so a chain never spans two."""
        backend = self._Chains()
        guarded = ir.If('flag', ir.Region([self._gemm('T')]))
        region = ir.Region([guarded, self._gemm('C', left='T')])
        fuse(region, backend)
        assert len(guarded.region.ops) == 1
        assert isinstance(guarded.region.ops[0], ir.FusedGEMMs)
        assert len(guarded.region.ops[0].statements) == 1
        assert len(region.ops[1].statements) == 1
