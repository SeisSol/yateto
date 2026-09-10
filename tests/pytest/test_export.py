"""The external generator interface: what an exporter such as TensorForge sees.

An exporter is registered under a target name and replaces the built-in factory
for that target.
"""

import json
import os
import pathlib
import tempfile

import pytest

from yateto import Generator, GeneratorCollection, Tensor
from yateto.arch import useArchitectureIdentifiedBy
from yateto import ir
from yateto.codegen.factory import ExportedStatement, ExportFactory, ExportGenerator
from yateto.codegen.lowering import lower
from yateto.guard import Guard
from yateto.type import Datatype

import yateto.functions as yf

N = 4


class Collector(ExportGenerator):
    """Records the description instead of emitting anything."""

    def __init__(self, arch, attrs=None):
        super().__init__(arch, attrs)
        self.kernel = None
        self.operations = []
        self.tensors = []

    def add_kernel(self, description):
        self.kernel = description
        self.operations = description["operations"]
        self.tensors = description["tensors"]

    def generate(self, cpp, cache):
        pass


def export(statements, target='gpu'):
    """Generate with an exporter installed for `target` and return it."""
    arch = useArchitectureIdentifiedBy('dhsw', 'dsm_86', 'cuda')
    collector = {}

    def make(a, attrs=None):
        collector['it'] = Collector(a, attrs)
        return collector['it']

    generator = Generator(arch)
    for i, statement in enumerate(statements):
        generator.add(f'k{i}', statement, target=target)
    with tempfile.TemporaryDirectory() as out:
        generator.generate(out, gemm_cfg=GeneratorCollection([]),
                           routine_exporters={target: make})
    return collector['it']


def emitted(statements, target='gpu'):
    """The generated kernel source for statements handed to an exporter."""
    arch = useArchitectureIdentifiedBy('dhsw', 'dsm_86', 'cuda')

    def make(a, attrs=None):
        return Collector(a, attrs)

    generator = Generator(arch)
    for i, statement in enumerate(statements):
        generator.add(f'k{i}', statement, target=target)
    with tempfile.TemporaryDirectory() as out:
        generator.generate(out, gemm_cfg=GeneratorCollection([]),
                           routine_exporters={target: make})
        return (pathlib.Path(out) / 'kernel.cpp').read_text()


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

    def test_an_operation_that_can_never_run_is_not_exported(self, tensors):
        """A guard that is a contradiction produces no operation at all.

        `None` and `[]` are both falsy, so an exporter cannot tell "never" from
        "always" by reading the field; the C++ factory emits nothing for such
        an action either.
        """
        from yateto.codegen.factory import ExportFactory
        from yateto.controlflow.graph import Guard

        collector = export([tensors['out']['ij'] <= tensors['A']['ij']])
        assert collector.operations
        for op in collector.operations:
            assert op['condition'] is not None

        factory = ExportFactory.__new__(ExportFactory)
        factory.operations = []
        assert factory._handleCondition(Guard.never()) is None
        factory._emit({'condition': factory._handleCondition(Guard.never())},
                      Guard.never(), None)
        assert factory.operations == []
        factory._emit({'condition': factory._handleCondition(Guard.always())},
                      Guard.always(), None)
        assert len(factory.operations) == 1

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


class TestExportedScalars:
    """A derived scalar reaches the external generator as a named operand; the
    prologue that computes it runs before the routine call."""

    def test_a_named_scalar_is_exported_by_name(self, tensors):
        t = tensors
        from yateto.type import Scalar
        collector = export([t['out']['ij'] <= Scalar('alpha') * t['A']['ij']])
        factors = {op['linear']['alpha']['name'] for op in collector.operations}
        assert 'alpha' in factors

    def test_a_derived_scalar_is_exported_by_name(self, tensors):
        t = tensors
        from yateto.type import Scalar
        alpha, beta = Scalar('alpha'), Scalar('beta')
        collector = export([t['out']['ij'] <= (alpha * beta) * t['A']['ij']])
        factors = {op['linear']['alpha']['name'] for op in collector.operations}
        assert any(name.startswith('_s') for name in factors), factors
        # the expression itself stays on the host; the generator sees one value
        assert 'alpha' not in factors and 'beta' not in factors

    def test_a_numeric_factor_is_exported_with_its_value(self, tensors):
        t = tensors
        collector = export([t['out']['ij'] <= 2.0 * t['A']['ij']])
        # the factor is referenced by name; the value sits in its descriptor
        factors = {op['linear']['alpha']['name'] for op in collector.operations}
        assert any(name.startswith('_scalar') for name in factors), factors
        values = {d['name']: d.get('values') for d in collector.tensors}
        assert any(v == {'kind': 'entries', 'data': [[[], 2.0]]}
                   for v in values.values()), values

    def test_a_factor_is_stated_once(self, tensors):
        """Never as an operand as well as alpha.

        Compared by value, not by name: listing it twice used to mint a second
        scalar tensor with the same value, which an exporter reading both the
        operands and alpha applies twice all the same.
        """
        t = tensors
        for statement in (t['out']['ij'] <= 2.0 * t['A']['ij'],
                          t['out']['ij'] <= 2.0 * t['A']['ik'] * t['B']['kj']):
            collector = export([statement])
            descriptors = {d['name']: d for d in collector.tensors}
            for op in collector.operations:
                factor = descriptors[op['linear']['alpha']['name']]
                for arg in op['args']:
                    operand = descriptors[arg['name']]
                    assert operand['values'] is None \
                        or operand['values'] != factor['values'], op


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


class TestTheDescriptionIsData:
    """A kernel arrives as one object, and that object is data.

    It is written out and read back by the host-side tooling, so anything in
    it that only Python understands -- an `Indices`, a tuple used as a dict
    key -- is a field that tooling cannot carry.
    """

    def test_a_kernel_arrives_as_one_description(self, tensors):
        A, B, out = tensors['A'], tensors['B'], tensors['out']
        collector = export([out['ij'] <= A['ik'] * B['kj']])
        assert collector.kernel is not None
        assert set(collector.kernel) == {'version', 'tensors', 'operations'}
        assert collector.kernel['version'] == ExportGenerator.INTERFACE_VERSION

    def test_the_description_survives_a_round_trip_through_json(self, tensors):
        A, B, out = tensors['A'], tensors['B'], tensors['out']
        collector = export([
            out['ij'] <= A['ik'] * B['kj'],
            out['ij'] <= 2.0 * A['ij'] + B['ij'],
        ])
        assert json.loads(json.dumps(collector.kernel)) == collector.kernel

    def test_an_index_is_a_name(self, tensors):
        A, B, out = tensors['A'], tensors['B'], tensors['out']
        collector = export([out['ij'] <= A['ik'] * B['kj']])
        for operation in collector.operations:
            for ref in [operation['result']] + operation['args']:
                assert all(isinstance(index, str) for index in ref['indices'])

    def test_every_tensor_an_operation_names_is_in_the_description(self, tensors):
        A, B, out = tensors['A'], tensors['B'], tensors['out']
        collector = export([out['ij'] <= A['ik'] * B['kj']])
        known = {tensor['name'] for tensor in collector.tensors}
        for operation in collector.operations:
            for ref in [operation['result']] + operation['args']:
                assert ref['name'] in known


class TestExportedRegion:
    """A statement the exporter takes away still stands in the kernel's region.

    That is what lets the kernel be asked what it touches: the exporter writes
    the statement into its own output, but which tensors the statement names,
    and under which guard it runs, are questions about this kernel.
    """

    @staticmethod
    def _factory():
        factory = ExportFactory.__new__(ExportFactory)
        factory.operations = []
        factory._target = 'gpu'
        return factory

    def test_a_statement_stands_in_the_region(self):
        statement = ir.TensorOp(None, [])
        region = self._factory()._emit({'condition': []}, Guard.always(), statement)
        assert list(region) == [statement]

    def test_a_guarded_statement_stands_under_its_guard(self):
        statement = ir.TensorOp(None, [])
        guard = Guard.literal('flag')
        region = self._factory()._emit({'condition': [{}]}, guard, statement)
        guarded, = region
        assert isinstance(guarded, ir.If)
        assert guarded.condition == guard
        assert list(guarded.region) == [statement]

    def test_one_that_can_never_run_stands_nowhere(self):
        statement = ir.TensorOp(None, [])
        region = self._factory()._emit({'condition': None}, Guard.never(), statement)
        assert len(region) == 0

    def test_what_the_exporter_takes_away_leaves_nothing_behind(self):
        region = ir.Region([ir.TensorOp(None, [], generator=ExportedStatement)])
        lower(region, None)
        assert len(region) == 0

    def test_a_guard_around_nothing_is_not_written(self, tensors):
        """The statement is gone by the time the region is emitted, so what is
        left is a test with nothing behind it, and nothing to test for."""
        source = emitted([yf.assignIf(tensors['flag'][''], tensors['out']['ij'],
                                      yf.sqrt(tensors['A']['ij']))])
        assert 'if (' not in source
