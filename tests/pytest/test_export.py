"""The external generator interface: what an exporter such as TensorForge sees.

An exporter is registered under a target name and replaces the built-in factory
for that target.
"""

import json
import os
import re
import tempfile

import numpy as np
import pytest

from yateto import Generator, GeneratorCollection, Tensor, simpleParameterSpace
from yateto.arch import useArchitectureIdentifiedBy
from yateto.codegen.factory import ExportGenerator
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
        factory._emit({'condition': factory._handleCondition(Guard.never())})
        assert factory.operations == []
        factory._emit({'condition': factory._handleCondition(Guard.always())})
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


class TestConstantValues:
    """A tensor whose data is known states it in the description."""

    @staticmethod
    def _constant():
        data = np.zeros((N, N))
        data[0, 0] = 0.5
        data[1, 2] = -1.0
        return Tensor('C', (N, N), spp=data)

    def _exported(self, tensors):
        C = self._constant()
        collector = export([tensors['out']['ij'] <= C['ik'] * tensors['B']['kj']])
        return next(d for d in collector.tensors if d['name'] == 'C')

    def test_the_values_reach_the_exporter(self, tensors):
        assert self._exported(tensors)['values'] is not None

    def test_a_value_comes_with_the_entry_it_belongs_to(self, tensors):
        values = self._exported(tensors)['values']
        assert values['kind'] == 'entries'
        assert values['data'] == [[[0, 0], 0.5], [[1, 2], -1.0]]

    def test_a_tensor_without_values_states_none(self, tensors):
        C = self._constant()
        collector = export([tensors['out']['ij'] <= C['ik'] * tensors['B']['kj']])
        B = next(d for d in collector.tensors if d['name'] == 'B')
        assert B['values'] is None

    def test_the_values_survive_a_json_round_trip(self, tensors):
        C = self._constant()
        collector = export([tensors['out']['ij'] <= C['ik'] * tensors['B']['kj']])
        assert json.loads(json.dumps(collector.kernel)) == collector.kernel


class TestConflictingOccurrences:
    """Two occurrences of one name that describe two different tensors.

    A real ambiguity, and the exporter cannot resolve it: it declares a name
    once. The message has to say which name and which field, because finding
    that out is otherwise most of the work.
    """

    @pytest.fixture
    def factory(self):
        from yateto.codegen.factory import ExportFactory
        factory = ExportFactory.__new__(ExportFactory)
        factory.tensors = {}
        return factory

    @staticmethod
    def description(datatype='f64', sizes=(4, 4)):
        return {'name': 'damageGrowing', 'datatype': datatype, 'flags': {'constant': False},
                'storage': {'shape': [4, 4], 'type': 'bbox', 'sizes': list(sizes)}}

    def test_the_same_description_twice_is_fine(self, factory):
        factory._handleTensor(self.description(), ['i', 'j'])
        factory._handleTensor(self.description(), ['i', 'j'])

    def test_the_message_names_the_tensor_and_the_field(self, factory):
        factory._handleTensor(self.description(datatype='f64'), ['i', 'j'])
        with pytest.raises(ValueError) as raised:
            factory._handleTensor(self.description(datatype='bool'), ['i', 'j'])
        message = str(raised.value)
        assert "'damageGrowing'" in message
        assert "datatype: 'bool' here, 'f64' before" in message
        assert 'storage' not in message

    def test_a_nested_field_is_named_by_its_path(self, factory):
        factory._handleTensor(self.description(sizes=(4, 4)), ['i', 'j'])
        with pytest.raises(ValueError, match=r'storage\.sizes: \[4, 2\] here, \[4, 4\] before'):
            factory._handleTensor(self.description(sizes=(4, 2)), ['i', 'j'])


class TestAlignmentIsTheStorages:
    """`alignment` is stated for the tensor, so it is the storage's promise.

    Asked of the occurrence, a slice along the leading axis of a width that is
    not a multiple of the vector length promised nothing while the whole
    tensor promised its alignment -- two descriptions of one name, and the
    exporter refused the kernel. The shift of a slice arrives with its
    reference; what it means for the alignment is the far side's to derive.
    """

    @pytest.fixture
    def aligned(self):
        from yateto.memory import MemoryLayout
        arch = useArchitectureIdentifiedBy('dhsw', 'dsm_86', 'cuda')
        previous = MemoryLayout.DEFAULT_ALIGNMENT_ARCH
        MemoryLayout.setAlignmentArch(arch)
        yield arch
        MemoryLayout.DEFAULT_ALIGNMENT_ARCH = previous

    def test_a_slice_and_the_whole_describe_one_tensor(self, aligned):
        X = Tensor('X', (N, N), alignStride=True)
        out = Tensor('out', (N, N))
        part = Tensor('part', (N - 1, N))
        collector = export([[out['ij'] <= X['ij'],
                             part['ij'] <= X['ij'].subslice('i', 1, N)]])
        described = [t for t in collector.tensors if t['name'] == 'X']
        assert len(described) == 1
        assert described[0]['alignment'] == aligned.alignment


class TestExportedAlignment:
    """What a tensor promises about the address of a column, in bytes.

    Read from the layout, so that a tensor laid out for one target keeps its
    promise while a kernel is generated for another.
    """

    def _tensorsOf(self, tensor):
        exporter = export([tensor['out']['ij'] <= tensor['A']['ik'] * tensor['B']['kj']])
        return {t['name']: t for t in exporter.tensors}

    def test_an_aligned_tensor_reports_its_own_architecture(self):
        from yateto.arch import getArchitectureIdentifiedBy
        from yateto.memory import DenseMemoryLayout

        wide = getArchitectureIdentifiedBy('dskx')
        out = Tensor('out', (N, N))
        out.setMemoryLayout(DenseMemoryLayout, alignStride=True, alignmentArch=wide)

        described = self._tensorsOf({
            'A': Tensor('A', (N, N)), 'B': Tensor('B', (N, N)), 'out': out})

        assert described['out']['alignment'] == wide.alignment

    def test_an_unaligned_tensor_promises_nothing(self):
        described = self._tensorsOf({
            'A': Tensor('A', (N, N)), 'B': Tensor('B', (N, N)),
            'out': Tensor('out', (N, N))})

        assert described['out']['alignment'] == 0

    def test_the_promise_does_not_follow_the_run(self):
        """Two tensors, two architectures, one generator run."""
        from yateto.arch import getArchitectureIdentifiedBy
        from yateto.memory import DenseMemoryLayout

        narrow = getArchitectureIdentifiedBy('dhsw')
        wide = getArchitectureIdentifiedBy('dskx')
        a = Tensor('A', (N, N))
        a.setMemoryLayout(DenseMemoryLayout, alignStride=True, alignmentArch=narrow)
        out = Tensor('out', (N, N))
        out.setMemoryLayout(DenseMemoryLayout, alignStride=True, alignmentArch=wide)

        described = self._tensorsOf({'A': a, 'B': Tensor('B', (N, N)), 'out': out})

        assert described['A']['alignment'] == narrow.alignment
        assert described['out']['alignment'] == wide.alignment
        assert narrow.alignment != wide.alignment


class Offerer(Collector):
    """An exporter that asks for its constants in an order of its own."""

    offerings = {}

    def layout_offerings(self):
        return dict(self.offerings)


def exportOffering(statements, offerings, target='gpu'):
    arch = useArchitectureIdentifiedBy('dhsw', 'dsm_86', 'cuda')
    collector = {}

    def make(a, attrs=None):
        it = Offerer(a, attrs)
        it.offerings = offerings
        collector['it'] = it
        return it

    generator = Generator(arch)
    for i, statement in enumerate(statements):
        generator.add(f'k{i}', statement, target=target)
    out = tempfile.mkdtemp()
    generator.generate(out, gemm_cfg=GeneratorCollection([]),
                       routine_exporters={target: make})
    return out


class TestLayoutOfferings:
    """An exporter may ask for a constant in an arrangement of its own."""

    @staticmethod
    def _tensors():
        import numpy as np

        values = np.zeros((N, N))
        for i in range(N):
            values[i, (i + 1) % N] = float(i + 1)
        return {'A': Tensor('A', (N, N), values), 'B': Tensor('B', (N, N)),
                'out': Tensor('out', (N, N))}

    def _generate(self, offerings):
        t = self._tensors()
        out = exportOffering([t['out']['ij'] <= t['A']['ik'] * t['B']['kj']], offerings)
        return (open(os.path.join(out, 'pool.h')).read(),
                open(os.path.join(out, 'pool.cpp')).read(),
                open(os.path.join(out, 'kernel.h')).read())

    def test_without_an_offering_nothing_changes(self):
        pool_h, pool_cpp, kernel_h = self._generate({})

        assert pool_cpp.count('= {') >= 1
        assert 'A = pool.A_' in kernel_h

    def test_an_offered_order_reaches_the_image(self):
        _, plain, _ = self._generate({})
        _, offered, _ = self._generate({'A': {'order': [1, 0]}})

        assert plain != offered

    def test_the_kernel_binds_the_arrangement_it_asked_for(self):
        _, _, plain = self._generate({})
        _, _, offered = self._generate({'A': {'order': [1, 0]}})

        assert 'A = pool.A_' in offered
        assert plain != offered

    def test_an_offering_names_the_member_it_is_about(self):
        """One member rearranged; the others stay as they were."""
        import numpy as np

        values = [np.zeros((N, N)) for _ in range(2)]
        for k, v in enumerate(values):
            for i in range(N):
                v[i, (i + k + 1) % N] = float(i + 1)
        F = {k: Tensor('F({})'.format(k), (N, N), v) for k, v in enumerate(values)}
        B = Tensor('B', (N, N))
        out = Tensor('out', (N, N))
        plain = exportOffering([out['ij'] <= F[0]['ik'] * B['kj'],
                                out['ij'] <= F[1]['ik'] * B['kj']], {})
        offered = exportOffering([out['ij'] <= F[0]['ik'] * B['kj'],
                                  out['ij'] <= F[1]['ik'] * B['kj']],
                                 {'F(0)': {'order': [1, 0]}})

        plain_h = open(os.path.join(plain, 'pool.h')).read()
        offered_h = open(os.path.join(offered, 'pool.h')).read()
        # One member for the family in both, and a different one once asked
        assert len(re.findall(r'Container<double const\*> F_\w+', plain_h)) == 1
        assert len(re.findall(r'Container<double const\*> F_\w+', offered_h)) == 1
        assert plain_h != offered_h

    def test_an_offering_for_a_member_of_it_that_is_not_read_is_refused(self):
        import numpy as np

        values = np.zeros((N, N))
        for i in range(N):
            values[i, (i + 1) % N] = float(i + 1)
        F = Tensor('F(0)', (N, N), values)
        B = Tensor('B', (N, N))
        out = Tensor('out', (N, N))
        with pytest.raises(ValueError, match=r'F\(3\)'):
            exportOffering([out['ij'] <= F['ik'] * B['kj']],
                           {'F(3)': {'order': [1, 0]}})

    def test_an_unknown_field_is_refused_by_name(self):
        with pytest.raises(ValueError, match='storage_parts'):
            self._generate({'A': {'storage_parts': 2}})

    def test_an_offering_for_something_unread_is_refused(self):
        with pytest.raises(ValueError, match='nosuch'):
            self._generate({'nosuch': {'order': [1, 0]}})

    def test_an_offering_for_a_writable_operand_is_refused(self):
        with pytest.raises(ValueError, match='out'):
            self._generate({'out': {'order': [1, 0]}})


class TestPreparedConstants:
    """An exporter may hand back numbers of its own, not just an order.

    Where one element is split across several scalars -- an emulated
    precision, a matrix instruction's fragments -- the split is the
    generator's arithmetic. It computes the numbers; the pool stores them.
    """

    @staticmethod
    def _tensors():
        import numpy as np

        values = np.zeros((N, N))
        for i in range(N):
            values[i, (i + 1) % N] = float(i + 1)
        return {'A': Tensor('A', (N, N), values), 'B': Tensor('B', (N, N)),
                'out': Tensor('out', (N, N))}

    def _generate(self, offerings):
        t = self._tensors()
        out = exportOffering([t['out']['ij'] <= t['A']['ik'] * t['B']['kj']], offerings)
        return (open(os.path.join(out, 'pool.h')).read(),
                open(os.path.join(out, 'pool.cpp')).read())

    def test_the_numbers_are_stored_as_they_came(self):
        prepared = [float(i) for i in range(2 * N * N)]

        _, pool_cpp = self._generate({'A': {'data': prepared, 'parts': 2,
                                            'planar': True}})

        assert '31' in pool_cpp
        assert pool_cpp.count('{') >= 1

    def test_the_entry_is_as_long_as_the_image(self):
        prepared = [float(i) for i in range(2 * N * N)]

        pool_h, _ = self._generate({'A': {'data': prepared, 'parts': 2}})

        assert '[{}]'.format(2 * N * N) in pool_h

    def test_the_element_type_is_still_the_tensor_s(self):
        prepared = [float(i) for i in range(2 * N * N)]

        pool_h, _ = self._generate({'A': {'data': prepared, 'parts': 2}})

        assert 'double const A_' in pool_h

    def test_a_shape_without_numbers_is_refused(self):
        with pytest.raises(ValueError, match='without the'):
            self._generate({'A': {'parts': 2}})

    def test_numbers_and_an_order_together_are_refused(self):
        with pytest.raises(ValueError, match='both'):
            self._generate({'A': {'data': [1.0, 2.0], 'order': [1, 0]}})

    def test_members_prepared_to_different_lengths_are_refused(self):
        with pytest.raises(ValueError, match='one length'):
            self._generate({'A': {'data': {0: [1.0, 2.0], 1: [1.0]}}})

    def test_numbers_that_do_not_divide_into_parts_are_refused(self):
        with pytest.raises(ValueError, match='do not divide'):
            self._generate({'A': {'data': [1.0, 2.0, 3.0], 'parts': 2}})


class Counter(Collector):
    """An exporter that says what kind of arithmetic it issued."""

    report = {}

    def flop_report(self):
        return dict(self.report)


def exportCounting(statements, report, target='gpu'):
    arch = useArchitectureIdentifiedBy('dhsw', 'dsm_86', 'cuda')

    def make(a, attrs=None):
        it = Counter(a, attrs)
        it.report = report
        return it

    generator = Generator(arch)
    for i, statement in enumerate(statements):
        generator.add(f'k{i}', statement, target=target)
    out = tempfile.mkdtemp()
    generator.generate(out, gemm_cfg=GeneratorCollection([]),
                       routine_exporters={target: make})
    return open(os.path.join(out, 'kernel.h')).read()


class TestReportedFlops:
    """What a generator issued, in the currency it issued it in."""

    @staticmethod
    def _tensors():
        return {'A': Tensor('A', (N, N)), 'B': Tensor('B', (N, N)),
                'out': Tensor('out', (N, N))}

    def _kernelH(self, report):
        t = self._tensors()
        return exportCounting([t['out']['ij'] <= t['A']['ik'] * t['B']['kj']], report)

    def test_without_a_report_the_count_stays_zero(self):
        kernel_h = self._kernelH({})

        assert 'HardwareFlops' in kernel_h
        assert 'HardwareFlops:' not in kernel_h

    def test_a_reported_count_reaches_the_total(self):
        kernel_h = self._kernelH({'mma:tf32': 300})

        assert 'HardwareFlops' in kernel_h
        assert '300' in kernel_h

    def test_the_kinds_are_stated_beside_the_total(self):
        kernel_h = self._kernelH({'mma:tf32': 300, 'fma:f32': 12})

        assert '300 mma:tf32' in kernel_h
        assert '12 fma:f32' in kernel_h

    def test_a_total_of_one_kind_needs_no_explaining(self):
        kernel_h = self._kernelH({'plain': 44})

        assert '44' in kernel_h
        assert 'HardwareFlops:' not in kernel_h


class TestFlopCount:
    """The count itself: it adds like a number and keeps its kinds."""

    def test_it_adds_to_a_plain_number(self):
        from yateto.codegen.flops import FlopCount

        assert int(0 + FlopCount(12) + 30) == 42

    def test_kinds_stay_apart_while_the_total_adds_up(self):
        from yateto.codegen.flops import FlopCount

        counted = FlopCount({'a': 2}) + FlopCount({'b': 3}) + FlopCount({'a': 1})

        assert counted.kinds() == {'a': 3, 'b': 3}
        assert int(counted) == 6

    def test_it_spells_itself_as_its_total(self):
        from yateto.codegen.flops import FlopCount

        assert '{}'.format(FlopCount({'a': 2, 'b': 3})) == '5'

    def test_nothing_counted_is_falsy(self):
        from yateto.codegen.flops import FlopCount

        assert not FlopCount()
        assert FlopCount() == 0


class ReadOfferer(Collector):
    """Asks for an order for every constant it was described that `wants`.

    Like a real exporter, it speaks only about what the kernel in hand reads,
    so the variants of one family each speak for their own members.
    """

    wants = staticmethod(lambda name, described: False)

    def layout_offerings(self):
        described = [t['name'] for t in self.tensors]
        return {t['name']: {'order': [1, 0]} for t in self.tensors
                if t['flags']['constant'] and self.wants(t['name'], described)}


def exportFamily(build, wants):
    arch = useArchitectureIdentifiedBy('dhsw', 'dsm_86', 'cuda')

    def make(a, attrs=None):
        it = ReadOfferer(a, attrs)
        it.wants = wants
        return it

    generator = Generator(arch)
    build(generator)
    out = tempfile.mkdtemp()
    generator.generate(out, gemm_cfg=GeneratorCollection([]),
                       routine_exporters={'gpu': make})
    return out


class TestOfferingsAcrossVariants:
    """The variants of one family share its members, so what they ask for has
    to fit into one arrangement: each speaks for the members it reads."""

    @staticmethod
    def constants(name, count):
        import numpy as np

        tensors = []
        for k in range(count):
            values = np.zeros((N, N))
            for i in range(N):
                values[i, (i + k + 1) % N] = float(i + 1)
            tensors.append(Tensor('{}({})'.format(name, k), (N, N), values))
        return tensors

    def test_variants_rearranging_different_members_share_one_arrangement(self):
        F = self.constants('F', 2)
        B = Tensor('B', (N, N))
        out = Tensor('out', (N, N))
        build = lambda g: g.addFamily('fam', simpleParameterSpace(2),
                                      lambda i: out['ij'] <= F[i]['ik'] * B['kj'], target='gpu')
        plain = exportFamily(build, lambda name, described: False)
        offered = exportFamily(build, lambda name, described: name.startswith('F('))

        plain_h = open(os.path.join(plain, 'pool.h')).read()
        offered_h = open(os.path.join(offered, 'pool.h')).read()
        # one member for the family, bound by the family, and a different
        # arrangement than without the offerings -- not one per variant
        assert len(re.findall(r'Container<double const\*> F_\w+', offered_h)) == 1
        assert offered_h != plain_h
        member = re.search(r'Container<double const\*> (F_\w+)', offered_h).group(1)
        assert 'F = pool.{};'.format(member) in open(os.path.join(offered, 'kernel.h')).read()

    def test_variants_reading_one_member_differently_are_refused_by_name(self):
        F = self.constants('F', 1)
        G = self.constants('G', 2)
        out = Tensor('out', (N, N))
        build = lambda g: g.addFamily('fam', simpleParameterSpace(2),
                                      lambda i: out['ij'] <= F[0]['ik'] * G[i]['kj'], target='gpu')
        # only the variant that reads G(0) asks for F(0) rearranged
        with pytest.raises(ValueError, match=r'Variants 0 and 1 of fam read F\(0\)'):
            exportFamily(build, lambda name, described: name == 'F(0)' and 'G(0)' in described)

    def test_an_offering_names_a_namespaced_tensor_as_it_was_described(self):
        import numpy as np

        values = np.zeros((N, N))
        for i in range(N):
            values[i, (i + 1) % N] = float(i + 1)
        A = Tensor('A', (N, N), values, namespace='ns')
        B = Tensor('B', (N, N))
        out = Tensor('out', (N, N))
        plain = exportFamily(lambda g: g.add('k', out['ij'] <= A['ik'] * B['kj'], target='gpu'),
                             lambda name, described: False)
        offered = exportFamily(lambda g: g.add('k', out['ij'] <= A['ik'] * B['kj'], target='gpu'),
                               lambda name, described: name == 'A')
        assert open(os.path.join(plain, 'pool.h')).read() \
            != open(os.path.join(offered, 'pool.h')).read()
