"""Conditional execution end to end: guards on the control flow graph and in
the generated code."""

import contextlib
import io
import re

import pytest

from yateto import Tensor
from yateto.arch import useArchitectureIdentifiedBy
from yateto.ast.cost import BoundingBoxCostEstimator
from yateto.controlflow.graph import Guard
from yateto.controlflow.transformer import liveness
from yateto.generator import Kernel
from yateto.type import Datatype

import yateto.functions as yf

N = 8


@pytest.fixture
def arch():
    return useArchitectureIdentifiedBy('dhsw')


@pytest.fixture
def tensors():
    return {
        'S': Tensor('S', (N, N)),
        'Y': Tensor('Y', (N, N)),
        'Z': Tensor('Z', (N, N)),
        'o1': Tensor('o1', (N, N)),
        'o2': Tensor('o2', (N, N)),
        'flag': Tensor('flag', (), datatype=Datatype.BOOL),
        'other': Tensor('other', (), datatype=Datatype.BOOL),
        'inner': Tensor('inner', (), datatype=Datatype.BOOL),
    }


def build(arch, statements):
    kernel = Kernel('k', statements)
    kernel.prepareUntilUnitTest(arch)
    kernel.prepareUntilCodeGen(BoundingBoxCostEstimator)
    return kernel


def guards(kernel):
    return [action.getGuard() for action in kernel.cfg]


class TestGuardPropagation:
    def test_an_unconditional_kernel_has_no_guards(self, arch, tensors):
        t = tensors
        kernel = build(arch, [t['o1']['ij'] <= yf.sqrt(t['S']['ij'])])
        assert all(g.isAlways() for g in guards(kernel))

    def test_the_whole_right_hand_side_is_guarded(self, arch, tensors):
        t = tensors
        # the intermediates of a conditional assignment must not run
        # unconditionally -- only the condition itself may
        kernel = build(arch, [
            yf.assignIf(t['flag'][''], t['o1']['ij'],
                        t['S']['ik'] * t['Z']['kj'] + t['Y']['ij']),
        ])
        assert all(not g.isAlways() for g in guards(kernel))

    def test_an_unguarded_statement_keeps_its_freedom(self, arch, tensors):
        t = tensors
        kernel = build(arch, [
            yf.assignIf(t['flag'][''], t['o1']['ij'], yf.sqrt(t['S']['ij'])),
            t['o2']['ij'] <= yf.sqrt(t['Y']['ij']),
        ])
        assert any(g.isAlways() for g in guards(kernel))

    def test_distinct_conditions_do_not_bleed_into_each_other(self, arch, tensors):
        t = tensors
        kernel = build(arch, [
            yf.assignIf(t['flag'][''], t['o1']['ij'], yf.sqrt(t['S']['ij'])),
            yf.assignIf(t['other'][''], t['o2']['ij'], yf.sqrt(t['Z']['ij'])),
        ])
        names = {frozenset(str(v) for v in g.variables()) for g in guards(kernel)}
        assert frozenset({'flag', 'other'}) not in names

    def test_a_nested_condition_carries_its_definition_guard(self, arch, tensors):
        t = tensors
        # `inner` only has a value where `flag` holds, so reading it elsewhere
        # would read whatever the buffer happened to contain
        kernel = build(arch, [
            yf.assignIf(t['flag'][''], t['inner'][''],
                        yf.all(yf.greater(t['Z']['ij'], t['Y']['ij']), 'ij')),
            yf.assignIf(t['inner'][''], t['o1']['ij'], yf.sqrt(t['Z']['ij'])),
        ])
        final = guards(kernel)[-1]
        assert {str(v) for v in final.variables()} == {'flag', 'inner'}


class TestConditionVersioning:
    def test_rewriting_a_condition_starts_a_new_literal(self, arch, tensors):
        t = tensors
        kernel = build(arch, [
            t['flag'][''] <= yf.any(yf.greater(t['S']['ij'], t['Y']['ij']), 'ij'),
            yf.assignIf(t['flag'][''], t['o1']['ij'], yf.sqrt(t['S']['ij'])),
            t['flag'][''] <= yf.any(yf.greater(t['S']['ij'], t['Z']['ij']), 'ij'),
            yf.assignIf(t['flag'][''], t['o2']['ij'], yf.sqrt(t['Z']['ij'])),
        ])
        conditional = [g for g in guards(kernel) if not g.isAlways()]
        versions = {version for g in conditional for _, version, _ in g.literals()}
        assert len(versions) == 2

    def test_the_two_versions_do_not_imply_each_other(self, arch, tensors):
        t = tensors
        kernel = build(arch, [
            t['flag'][''] <= yf.any(yf.greater(t['S']['ij'], t['Y']['ij']), 'ij'),
            yf.assignIf(t['flag'][''], t['o1']['ij'], yf.sqrt(t['S']['ij'])),
            t['flag'][''] <= yf.any(yf.greater(t['S']['ij'], t['Z']['ij']), 'ij'),
            yf.assignIf(t['flag'][''], t['o2']['ij'], yf.sqrt(t['Z']['ij'])),
        ])
        conditional = [g for g in guards(kernel) if not g.isAlways()]
        first, last = conditional[0], conditional[-1]
        assert not last.implies(first)
        assert not first.implies(last)


class TestConditionLiveness:
    def test_a_condition_variable_is_live_where_it_is_read(self, arch, tensors):
        t = tensors
        kernel = build(arch, [
            t['flag'][''] <= yf.any(yf.greater(t['S']['ij'], t['Y']['ij']), 'ij'),
            yf.assignIf(t['flag'][''], t['o1']['ij'], yf.sqrt(t['S']['ij'])),
        ])
        live = liveness(kernel.cfg)
        guarded = [position for position, action in enumerate(kernel.cfg)
                   if not action.getGuard().isAlways()]
        assert guarded
        for position in guarded:
            assert 'flag' in {str(v) for v in live[position].variables()}

    def test_a_condition_variable_counts_as_a_use(self, arch, tensors):
        t = tensors
        kernel = build(arch, [
            yf.assignIf(t['flag'][''], t['o1']['ij'], yf.sqrt(t['S']['ij'])),
        ])
        guarded = [action for action in kernel.cfg
                   if not action.getGuard().isAlways()]
        for action in guarded:
            assert action.guardVariables() <= action.allVariables()
            assert 'flag' in {str(v) for v in action.allVariables()}

    def test_a_condition_variable_is_live_regardless_of_the_outcome(self, arch, tensors):
        t = tensors
        kernel = build(arch, [
            yf.assignIf(t['flag'][''], t['o1']['ij'], yf.sqrt(t['S']['ij'])),
        ])
        live = liveness(kernel.cfg)
        for position, action in enumerate(kernel.cfg):
            if action.getGuard().isAlways():
                continue
            for var in action.guardVariables():
                assert live[position].guardOf(var).isAlways()


class TestEmittedCode:
    """The guards have to survive all the way into the generated C++."""

    @staticmethod
    def emit(arch, statements, header=False):
        from yateto import Generator, GeneratorCollection
        import tempfile, os
        generator = Generator(arch)
        for i, statement in enumerate(statements):
            generator.add(f'k{i}', statement)
        with tempfile.TemporaryDirectory() as out:
            with contextlib.redirect_stdout(io.StringIO()):
                generator.generate(out, gemm_cfg=GeneratorCollection([]))
            name = 'kernel.h' if header else 'kernel.cpp'
            return open(os.path.join(out, name)).read()

    def test_a_guard_becomes_an_if(self, arch, tensors):
        t = tensors
        code = self.emit(arch, [
            yf.assignIf(t['flag'][''], t['o1']['ij'], yf.sqrt(t['S']['ij'])),
        ])
        assert 'if (' in code
        assert 'flag[0]' in code

    def test_every_statement_of_a_guarded_assignment_is_inside_an_if(self, arch, tensors):
        t = tensors
        code = self.emit(arch, [[
            yf.assignIf(t['flag'][''], t['o1']['ij'],
                        t['o1']['ij'] + t['S']['ik'] * t['Z']['kj'] + t['Y']['ij']),
        ]])
        body = code[code.index('k0::execute'):]
        body = body[:body.index('\n  }')]
        # no assignment to a tensor may sit at the outermost level of the body
        for line in body.split('\n'):
            if re.search(r'^\s{4}(o1|o2)\[', line):
                pytest.fail(f'unguarded store outside the if: {line.strip()}')

    def test_loop_variables_are_declared(self, arch, tensors):
        t = tensors
        code = self.emit(arch, [
            t['flag'][''] <= yf.any(yf.greater(t['S']['ij'], t['Y']['ij']), 'ij'),
            t['o1']['ij'] <= yf.sqrt(t['S']['ij']),
        ])
        declared = set(re.findall(r'for \(int (\w+) =', code))
        used = set(re.findall(r'\[[^\]]*?(\b_[a-z]\b)', code))
        assert used <= declared, f'undeclared loop variables: {used - declared}'

    def test_no_python_float_spellings_reach_the_output(self, arch, tensors):
        t = tensors
        code = self.emit(arch, [
            t['flag'][''] <= yf.any(yf.greater(t['S']['ij'], t['Y']['ij']), 'ij'),
            t['o1']['ij'] <= yf.sqrt(t['S']['ij']),
        ])
        assert not re.search(r'=\s*-?inf\b', code)
        assert not re.search(r'=\s*-?nan\b', code)


class TestUnreachable:
    """A statement whose guard can never hold is generated nowhere.

    It is written as such -- a plain false rather than something the kernel
    reads -- so it is either what was asked for or a condition that came out
    contradictory. Either way it is worth saying, because the operands no
    other statement names leave the kernel's interface with it.
    """

    @staticmethod
    def note(arch, statements):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            build(arch, statements)
        return [line for line in printed.getvalue().splitlines()
                if line.startswith('Note:')]

    def test_nothing_is_said_where_everything_can_run(self, arch, tensors):
        t = tensors
        assert self.note(arch, [t['o1']['ij'] <= t['S']['ij']]) == []

    def test_what_can_never_run_is_said_once(self, arch, tensors):
        t = tensors
        note, = self.note(arch, [t['o1']['ij'] <= t['S']['ij'],
                                 yf.assignIf(False, t['o2']['ij'], t['Y']['ij'])])
        assert '1 statement' in note and '"k"' in note

    def test_the_tensors_that_leave_with_it_are_named(self, arch, tensors):
        t = tensors
        note, = self.note(arch, [t['o1']['ij'] <= t['S']['ij'],
                                 yf.assignIf(False, t['o2']['ij'], t['Y']['ij'])])
        assert 'Y' in note and 'o2' in note

    def test_a_tensor_another_statement_names_stays(self, arch, tensors):
        t = tensors
        note, = self.note(arch, [t['o1']['ij'] <= t['S']['ij'],
                                 yf.assignIf(False, t['o2']['ij'], t['S']['ij'])])
        assert 'S' not in note.split('with them')[-1]


class TestUnreachableInterface:
    """What a kernel asks its caller for is what its statements name.

    A statement that can never run stands nowhere, so it asks for nothing --
    and a tensor no other statement names is not a member the caller has to
    set, nor bytes the kernel is counted as moving.
    """

    @staticmethod
    def struct(arch, statements):
        header = TestEmittedCode.emit(arch, statements, header=True)
        body = header[header.index('struct k0'):]
        # the members stand between the counters and the entry point
        return body[:body.index('void execute')]

    def test_a_tensor_only_an_unreachable_statement_names_is_no_member(self, arch, tensors):
        t = tensors
        members = self.struct(arch, [[t['o1']['ij'] <= t['S']['ij'],
                                      yf.assignIf(False, t['o2']['ij'], t['Y']['ij'])]])
        assert 'o1' in members and 'S' in members
        assert 'o2' not in members and 'Y' not in members

    def test_a_tensor_a_reachable_statement_names_stays_a_member(self, arch, tensors):
        t = tensors
        members = self.struct(arch, [[t['o1']['ij'] <= t['S']['ij'],
                                      yf.assignIf(False, t['o2']['ij'], t['S']['ij'])]])
        assert 'S' in members
