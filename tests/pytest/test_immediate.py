"""Operands whose data is part of the generated code.

`AddressingMode.IMMEDIATE` states that the generator supplies the numbers.
Nothing is passed for such an operand, nothing is bound for it, and the
generators that read their operands from memory say so rather than emitting
an address for something that has none.
"""

import numpy as np
import pytest

from yateto import Generator, GeneratorCollection, Tensor
from yateto.arch import useArchitectureIdentifiedBy
from yateto.codegen.factory import ExportGenerator
from yateto.type import AddressingMode

N = 4


def values():
    data = np.zeros((N, N))
    data[0, 0] = 0.5
    data[1, 2] = -1.0
    return data


@pytest.fixture
def immediate():
    return Tensor('C', (N, N), spp=values(), addressing=AddressingMode.IMMEDIATE)


@pytest.fixture
def operands():
    return {'B': Tensor('B', (N, N)), 'out': Tensor('out', (N, N))}


class Exporter(ExportGenerator):
    """Accepts everything and records the description."""

    seen = None

    def add_kernel(self, description):
        Exporter.seen = description

    def generate(self, cpp, cache):
        cpp('// external kernel')


def generate(tmp_path, statements, target='cpu'):
    """Generate into `tmp_path` and return the emitted files as text."""
    arch = useArchitectureIdentifiedBy('dhsw', 'dsm_86', 'cuda')
    generator = Generator(arch)
    for i, statement in enumerate(statements):
        generator.add(f'k{i}', statement, target=target)
    exporters = ({target: lambda a, attrs=None: Exporter(a, attrs)}
                 if target != 'cpu' else {})
    generator.generate(str(tmp_path), gemm_cfg=GeneratorCollection([]),
                       routine_exporters=exporters)
    return {path.name: path.read_text() for path in tmp_path.iterdir()}


class TestPrecondition:
    def test_a_tensor_without_values_cannot_be_immediate(self):
        with pytest.raises(ValueError, match='carries no values'):
            Tensor('D', (N, N), addressing=AddressingMode.IMMEDIATE)

    def test_the_mode_is_not_an_argument_and_has_no_storage(self, immediate):
        assert not immediate.isPassedAsArgument()
        assert not immediate.hasStorage()
        assert not immediate.isPassedByValue()

    def test_it_has_no_pointer_type(self):
        with pytest.raises(ValueError, match='not passed'):
            AddressingMode.IMMEDIATE.pointer_type()

    def test_it_cannot_be_assigned_to(self, immediate, operands):
        with pytest.raises(ValueError, match='generated code'):
            immediate['ij'] <= operands['B']['ij']


class TestRefusal:
    def test_a_generator_reading_from_memory_refuses_it(self, tmp_path,
                                                        immediate, operands):
        with pytest.raises(NotImplementedError, match='reads its operands'):
            generate(tmp_path, [operands['out']['ij']
                                <= immediate['ik'] * operands['B']['kj']])

    def test_the_message_names_the_tensor_and_the_operation(self, tmp_path,
                                                            immediate, operands):
        with pytest.raises(NotImplementedError) as raised:
            generate(tmp_path, [operands['out']['ij']
                                <= immediate['ik'] * operands['B']['kj']])
        assert 'C' in str(raised.value)
        assert 'LoopOverGEMM' in str(raised.value)


class TestDescription:
    @pytest.fixture(autouse=True)
    def exported(self, tmp_path, immediate, operands):
        generate(tmp_path, [operands['out']['ij']
                            <= immediate['ik'] * operands['B']['kj']],
                 target='gpu')
        self.tensors = {t['name']: t for t in Exporter.seen['tensors']}

    def test_an_immediate_operand_has_no_address_formula(self):
        assert self.tensors['C']['addressing'] is None

    def test_it_states_where_its_data_is(self):
        assert self.tensors['C']['residence'] == 'code'
        assert self.tensors['B']['residence'] == 'memory'

    def test_it_carries_the_values_the_exporter_has_to_write(self):
        assert self.tensors['C']['values']['data'] == [[[0, 0], 0.5],
                                                       [[1, 2], -1.0]]

    def test_a_scale_factor_states_that_it_is_an_argument(self, tmp_path,
                                                          operands):
        generate(tmp_path, [operands['out']['ij'] <= 2.0 * operands['B']['ij']],
                 target='gpu')
        residences = {t['residence'] for t in Exporter.seen['tensors']
                      if t['storage']['shape'] == []}
        assert residences == {'argument'}


class TestEmittedCode:
    @pytest.fixture(autouse=True)
    def emitted(self, tmp_path, immediate, operands):
        inMemory = Tensor('M', (N, N), spp=values())
        self.files = generate(
            tmp_path,
            [operands['out']['ij'] <= immediate['ik'] * operands['B']['kj'],
             operands['out']['ij'] <= inMemory['ik'] * operands['B']['kj']],
            target='gpu')

    def test_the_kernel_has_no_member_for_it(self):
        struct = self.files['kernel.h'].split('struct k0')[1].split('struct')[0]
        assert 'B' in struct
        assert '* C' not in struct and '** C' not in struct

    def test_nothing_is_bound_for_it(self):
        assert 'C = pool.' not in self.files['pool.cpp'] + self.files['init.cpp']
        assert 'M = pool.' in self.files['kernel.h']

    def test_the_pool_holds_only_what_is_read_from_memory(self):
        assert 'M_' in self.files['pool.h']
        assert 'C_' not in self.files['pool.h']

    def test_its_values_are_still_stated_for_the_host(self):
        assert 'init::C::Values[] = {0.5' in self.files['init.cpp']

    def test_the_test_does_not_assign_a_member_that_is_absent(self):
        assert 'krnl.C' not in self.files['KernelTest.t.h']
        assert 'krnl.M' in self.files['KernelTest.t.h']

    def test_the_reference_computes_with_the_same_numbers(self):
        """A filling pattern would put a different operand on each side."""
        assert 'double C[6]  = {0.5, 0.0, 0.0, 0.0, 0.0, -1.0}' \
               in self.files['KernelTest.t.h']
