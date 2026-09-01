"""The external generator interface: what an exporter such as TensorForge sees.

An exporter is registered under a target name and replaces the built-in factory
for that target.
"""

import os
import tempfile

import pytest

from yateto import Generator, GeneratorCollection, Tensor
from yateto.arch import useArchitectureIdentifiedBy
from yateto.codegen.factory import ExportGenerator
from yateto.type import Datatype

import yateto.functions as yf

N = 4


class Collector(ExportGenerator):
    """Records the descriptors instead of emitting anything."""

    def __init__(self, arch):
        super().__init__(arch)
        self.operations = []
        self.tensors = []

    def add_operation(self, description):
        self.operations.append(description)
        return 0

    def add_tensor(self, description):
        self.tensors.append(description)

    def generate(self, cpp, cache):
        pass


def export(statements, target='gpu'):
    """Generate with an exporter installed for `target` and return it."""
    arch = useArchitectureIdentifiedBy('dhsw', 'dsm_86', 'cuda')
    collector = {}

    def make(a):
        collector['it'] = Collector(a)
        return collector['it']

    generator = Generator(arch)
    for i, statement in enumerate(statements):
        generator.add(f'k{i}', statement, target=target)
    with tempfile.TemporaryDirectory() as out:
        generator.generate(out, gemm_cfg=GeneratorCollection([]),
                           routine_exporters={target: make})
    return collector['it']


@pytest.fixture
def tensors():
    return {
        'A': Tensor('A', (N, N)),
        'B': Tensor('B', (N, N)),
        'out': Tensor('out', (N, N)),
        'scalar': Tensor('scalar', ()),
        'flag': Tensor('flag', (), datatype=Datatype.BOOL),
        'other': Tensor('other', (), datatype=Datatype.BOOL),
    }


class TestExporterRegistration:
    def test_an_exporter_replaces_the_target_factory(self, tensors):
        t = tensors
        collector = export([t['out']['ij'] <= yf.sqrt(t['A']['ij'])])
        assert collector.operations, 'the exporter received nothing'

    def test_elementwise_is_exported_with_its_operation(self, tensors):
        t = tensors
        collector = export([t['out']['ij'] <= yf.sqrt(t['A']['ij'])])
        kinds = {op['type'] for op in collector.operations}
        assert 'elementwise' in kinds
        assert 'Sqrt' in {op.get('optype') for op in collector.operations}

    def test_a_gemm_is_exported_as_multilinear(self, tensors):
        t = tensors
        collector = export([t['out']['ij'] <= t['A']['ik'] * t['B']['kj']])
        assert 'multilinear' in {op['type'] for op in collector.operations}

    def test_a_reduction_is_exported_with_its_operation(self, tensors):
        t = tensors
        collector = export([t['scalar'][''] <= yf.sum(t['A']['ij'], 'ij')])
        reductions = [op for op in collector.operations if op['type'] == 'reduction']
        assert reductions
        assert all(op['optype'] == 'Add' for op in reductions)


class TestRankZeroTensors:
    """Condition variables and scalar reduction results are rank-0."""

    def test_a_rank_zero_result_is_exported(self, tensors):
        t = tensors
        collector = export([t['scalar'][''] <= yf.sum(t['A']['ij'], 'ij')])
        shapes = {d['name']: d['storage']['shape'] for d in collector.tensors}
        assert shapes['scalar'] == []

    def test_a_rank_zero_condition_is_exported(self, tensors):
        t = tensors
        collector = export([yf.assignIf(t['flag'][''], t['out']['ij'],
                                        yf.sqrt(t['A']['ij']))])
        assert 'flag' in {d['name'] for d in collector.tensors}

    def test_addressing_and_datatype_reach_the_exporter(self, tensors):
        t = tensors
        collector = export([yf.assignIf(t['flag'][''], t['out']['ij'],
                                        yf.sqrt(t['A']['ij']))])
        flag = next(d for d in collector.tensors if d['name'] == 'flag')
        assert flag['datatype'] == 'bool'
        assert flag['addressing']


class TestExportedGuards:
    def test_an_unguarded_operation_has_an_empty_guard(self, tensors):
        t = tensors
        collector = export([t['out']['ij'] <= yf.sqrt(t['A']['ij'])])
        assert all(op['condition'] == [] for op in collector.operations)

    def test_a_guard_exports_as_a_flat_literal_list(self, tensors):
        t = tensors
        collector = export([yf.assignIf(t['flag'][''], t['out']['ij'],
                                        yf.sqrt(t['A']['ij']))])
        guarded = [op for op in collector.operations if op['condition']]
        assert guarded
        for op in guarded:
            for literal in op['condition']:
                assert set(literal) == {'tensor', 'version', 'negated'}
                assert literal['negated'] is False

    def test_the_guard_names_the_condition_tensor(self, tensors):
        t = tensors
        collector = export([yf.assignIf(t['flag'][''], t['out']['ij'],
                                        yf.sqrt(t['A']['ij']))])
        named = {literal['tensor']['name']
                 for op in collector.operations for literal in op['condition']}
        assert named == {'flag'}

    def test_nested_conditions_export_as_a_conjunction(self, tensors):
        t = tensors
        collector = export([[
            yf.assignIf(t['flag'][''], t['other'][''],
                        yf.any(yf.greater(t['A']['ij'], t['B']['ij']), 'ij')),
            yf.assignIf(t['other'][''], t['out']['ij'], yf.sqrt(t['A']['ij'])),
        ]])
        widest = max((op['condition'] for op in collector.operations), key=len)
        assert {literal['tensor']['name'] for literal in widest} == {'flag', 'other'}

    def test_a_rewritten_condition_gets_a_new_version(self, tensors):
        t = tensors
        collector = export([[
            t['flag'][''] <= yf.any(yf.greater(t['A']['ij'], t['B']['ij']), 'ij'),
            yf.assignIf(t['flag'][''], t['out']['ij'], yf.sqrt(t['A']['ij'])),
            t['flag'][''] <= yf.any(yf.greater(t['B']['ij'], t['A']['ij']), 'ij'),
            yf.assignIf(t['flag'][''], t['out']['ij'], yf.sqrt(t['B']['ij'])),
        ]])
        versions = {literal['version']
                    for op in collector.operations for literal in op['condition']}
        assert len(versions) == 2, 'the two values of `flag` must be distinguishable'


class TestExportedTensors:
    def test_every_operand_is_registered_as_a_tensor(self, tensors):
        t = tensors
        collector = export([t['out']['ij'] <= yf.sqrt(t['A']['ij'])])
        registered = {d['name'] for d in collector.tensors}
        for op in collector.operations:
            for arg in op['args']:
                assert arg['name'] in registered
            assert op['result']['name'] in registered

    def test_operands_carry_indices(self, tensors):
        t = tensors
        collector = export([t['out']['ij'] <= yf.sqrt(t['A']['ij'])])
        elementwise = next(op for op in collector.operations
                           if op['type'] == 'elementwise')
        assert all('indices' in arg for arg in elementwise['args'])

    def test_a_temporary_is_flagged(self, tensors):
        t = tensors
        collector = export([t['scalar'][''] <= yf.sum(t['A']['ij'], 'ij')])
        flags = {d['name']: d['flags']['temporary'] for d in collector.tensors}
        assert any(flags.values()), 'the intermediate reduction result is temporary'
        assert flags['A'] is False
