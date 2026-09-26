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


class TestFallback:
    """A generator that cannot take the numbers gets the tensor from memory.

    The mode is a request: the occurrence whose generator cannot take it reads
    the pool entry instead, and the generation says so. Here that is a GEMM
    the immediate operand is looped over -- a different matrix in every
    iteration -- while an element-wise statement elsewhere reads the same
    tensor with its numbers in the code.
    """

    @pytest.fixture(autouse=True)
    def emitted(self, tmp_path, capsys):
        data = np.zeros((N, N, 2))
        data[0, 0, 0] = 0.5
        data[1, 2, 1] = -1.0
        C = Tensor('C', (N, N, 2), spp=data, addressing=AddressingMode.IMMEDIATE)
        A = Tensor('A', (N, N, 2))
        B = Tensor('B', (N, N))
        out = Tensor('out', (N, N, 2))
        self.files = generate(tmp_path, [out['ijl'] <= C['ikl'] * B['kj'],
                                         out['ijl'] <= C['ijl'] * A['ijl']])
        self.printed = capsys.readouterr().out

    def struct(self, name):
        return self.files['kernel.h'].split(f'struct {name}')[1].split('struct')[0]

    def test_the_gemm_reads_it_from_memory(self):
        body = self.files['kernel.cpp'].split('k0::execute')[1].split('execute')[0]
        assert 'C != nullptr' in body

    def test_that_kernel_declares_a_member_and_binds_it(self):
        struct = self.struct('k0')
        assert 'C' in struct
        assert 'C = pool.' in struct

    def test_the_pool_holds_it(self):
        assert 'C_' in self.files['pool.h']

    def test_the_element_wise_kernel_still_writes_the_numbers(self):
        struct = self.struct('k1')
        assert '* C' not in struct and '** C' not in struct
        body = self.files['kernel.cpp'].split('k1::execute')[1]
        assert '(0.5) * (A[0])' in body
        assert 'C[' not in body

    def test_the_generation_says_so(self):
        assert 'Note: C is read from memory in k0; LoopOverGEMM cannot take it ' \
               'as immediate.' in self.printed
        assert 'k1' not in self.printed.split('Note:')[1]

    def test_the_unit_test_binds_the_pool_before_its_own_buffers(self):
        test = self.files['KernelTest.t.h'].split('void testk0')[1].split('void test')[0]
        assert test.index('krnl.bindGlobals(Pool::host());') < test.index('krnl.B = B;')

    def test_a_kernel_without_immediates_binds_nothing_in_its_test(self, tmp_path):
        B = Tensor('B', (N, N))
        out = Tensor('out', (N, N))
        (tmp_path / 'plain').mkdir()
        files = generate(tmp_path / 'plain', [out['ij'] <= 2.0 * B['ij']])
        assert 'bindGlobals' not in files['KernelTest.t.h']


class TestHostGemm:
    """A GEMM on the host writes the numbers into its loops.

    One loop per column of the result (per row, for a left operand), holding
    that column's entries as a sum: the loop over k and the operand are gone,
    a zero costs nothing, a one is not a multiplication and a minus one is a
    sign. A selector is therefore the column it selects.
    """

    def body(self, tmp_path, statement, capsys=None):
        files = generate(tmp_path, [statement])
        body = files['kernel.cpp'].split('k0::execute')[1]
        return files, body

    def test_a_right_operand_is_one_loop_per_column(self, tmp_path, immediate,
                                                   operands, capsys):
        files, body = self.body(tmp_path, operands['out']['ij']
                                <= operands['B']['ik'] * immediate['kj'])
        assert '= 0.5 * B[0 + 1*m + 4*0];' in body
        assert '= -B[0 + 1*m + 4*1];' in body
        assert 'Note:' not in capsys.readouterr().out
        struct = files['kernel.h'].split('struct k0')[1].split('struct')[0]
        assert '* C' not in struct and '** C' not in struct

    def test_a_left_operand_is_one_loop_per_row(self, tmp_path, immediate, operands):
        _, body = self.body(tmp_path, operands['out']['ij']
                            <= immediate['ik'] * operands['B']['kj'])
        assert 'out[0 + 1*0 + 4*n] = 0.5 * B[0 + 1*0 + 4*n];' in body
        assert 'out[0 + 1*1 + 4*n] = -B[0 + 1*2 + 4*n];' in body

    def test_a_selector_is_the_entry_it_selects(self, tmp_path):
        unit = np.zeros(13)
        unit[5] = 1.0
        pick = Tensor('pick', (13,), spp=unit, addressing=AddressingMode.IMMEDIATE)
        p = Tensor('p', (13,))
        s = Tensor('s', ())
        _, body = self.body(tmp_path, s[''] <= p['z'] * pick['z'])
        assert '] = p[5 + 13*m + 1*0];' in body
        assert '*' not in body.split('] = p[')[1].split(';')[0].replace('13*m', '').replace('1*0', '')

    def test_the_scale_factor_is_folded_into_the_numbers(self, tmp_path, immediate,
                                                        operands):
        _, body = self.body(tmp_path, operands['out']['ij']
                            <= 2.0 * operands['B']['ik'] * immediate['kj'])
        # 2 * 0.5 is one, and 2 * -1 a factor of two with a sign
        assert '= B[0 + 1*m + 4*0];' in body
        assert '= -2.0 * B[0 + 1*m + 4*1];' in body

    def test_nothing_is_read_for_it(self, tmp_path, immediate, operands):
        _, body = self.body(tmp_path, operands['out']['ij']
                            <= operands['B']['ik'] * immediate['kj'])
        assert 'C[' not in body and 'C != nullptr' not in body


class TestInMemoryTwin:
    def test_it_is_the_same_tensor_read_through_an_address(self, immediate):
        twin = immediate.inMemory()
        assert twin.name() == immediate.name()
        assert twin.values() is immediate.values()
        assert twin.memoryLayout() is immediate.memoryLayout()
        assert twin.isPassedAsArgument() and twin.hasStorage()


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


class TestMaterialization:
    """The element-wise generator writes the numbers where it reads them."""

    def emitted(self, tmp_path, statements):
        return generate(tmp_path, statements)['kernel.cpp']

    def test_the_values_are_written_into_the_kernel(self, tmp_path, immediate):
        A = Tensor('A', (N, N))
        out = Tensor('out', (N, N))
        body = self.emitted(tmp_path, [out['ij'] <= immediate['ij'] * A['ij']])
        assert '(0.5) * (A[0])' in body
        assert '(-1.0) * (A[9])' in body

    def test_nothing_is_read_for_it(self, tmp_path, immediate):
        A = Tensor('A', (N, N))
        out = Tensor('out', (N, N))
        body = self.emitted(tmp_path, [out['ij'] <= immediate['ij'] * A['ij']])
        assert 'C[' not in body
        assert 'C != nullptr' not in body

    def test_an_entry_the_pattern_excludes_costs_nothing(self, tmp_path,
                                                         immediate):
        """Two non-zeros out of sixteen entries, so two statements."""
        A = Tensor('A', (N, N))
        out = Tensor('out', (N, N))
        body = self.emitted(tmp_path, [out['ij'] <= immediate['ij'] * A['ij']])
        assert body.count('out[') == 2  # one statement per non-zero


class TestSlicedOperand:
    def test_the_value_is_looked_up_in_the_tensor_s_own_index_space(self):
        """An entry is in the operand's space, the values in the tensor's.

        A slicing operand differs between the two by the shift it imposes,
        and the view is what knows it.
        """
        from yateto.codegen.common import immediateValue
        from yateto.memory import DenseMemoryLayout, MemoryLayoutView

        tensor = Tensor('S', (N, N), spp=values(),
                        addressing=AddressingMode.IMMEDIATE)
        base = DenseMemoryLayout.fromSpp(tensor.spp())

        class Term:
            addressing = AddressingMode.IMMEDIATE
            values = tensor.values()
            memoryLayout = base

        assert immediateValue(Term, (1, 2)) == '-1.0'

        # the same number, now reached through a slice that starts at 2
        Term.memoryLayout = MemoryLayoutView(base, 1, 2, N)
        assert immediateValue(Term, (1, 0)) == '-1.0'
        assert immediateValue(Term, (1, 2)) == 0


class TestSlicedResult:
    """A result cut out of a larger tensor owns its window and nothing else.

    An immediate operand sends the element-wise generator down the unrolled
    path, which clears its result before it writes the non-zeros. A result
    that is a view has no size of its own to clear by -- and clearing the
    tensor behind it would take the neighbouring columns with it.
    """

    def test_only_the_window_is_cleared(self, tmp_path):
        unit = np.zeros(N)
        unit[0] = 1.0
        c = Tensor('c', (N,), spp=unit, addressing=AddressingMode.IMMEDIATE)
        v = Tensor('v', (2,))
        out = Tensor('out', (N, N))
        code = generate(tmp_path, [out['kc'].subslice('c', 1, 3) <= c['k'] * v['c']])
        body = code['kernel.cpp'].split('k0::execute')[1]
        # columns 1 and 2 of a 4x4 column-major block are addresses 4..11
        assert 'memset(out + 4, 0, 8 * sizeof(double));' in body
        assert 'memset(out, ' not in body
        assert 'out[4] = (1.0) * (v[0]);' in body
        assert 'out[8] = (1.0) * (v[1]);' in body
