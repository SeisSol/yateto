"""Addressing modes with parameters, and the indexed modes.

An element of the batch used to be reached in one of two ways: at the element
size from the last one (`STRIDED`), or through a pointer of its own
(`INDIRECT`). `ElementAddressing` gives a mode the numbers it left at their
defaults -- a stride and an offset, so that a member of a struct can be
strided -- and two modes reach an element through the launch's index list:
`INDEXED` (`base + index[n] * stride + offset`) and `INDEXED_POINTER`
(`vars[index[n] * slots + slot] + offset`).

What these tests pin: the API (validation, the bare mode for the defaults,
equality that never mistakes a parameterized mode for a bare one), what an
exporter is told (`binding`, the formulas, interface version 8), the kernel
struct's members (the index list, a stride or a slot set at run time, the
pointer types), the generators that cannot read the modes refusing them, and
that a kernel stating none of it is generated exactly as before.
"""
from __future__ import annotations

import copy
import json
import pathlib
import pickle
import shutil
import subprocess
import sys
import types
from io import StringIO

import numpy as np
import pytest

from yateto import Generator, Tensor, simpleParameterSpace
from yateto.arch import useArchitectureIdentifiedBy
from yateto.ast.cost import BoundingBoxCostEstimator
from yateto.codegen.cache import RoutineCache
from yateto.codegen.code import Cpp
from yateto.codegen.common import (BatchedOperationsAux, KernelAttributes,
                                   refuseUnreadableAddressing)
from yateto.codegen.datacache import DataCache
from yateto.codegen.factory import ExportFactory, ExportGenerator
from yateto.codegen.visitor import OptimizedKernelGenerator
from yateto.gemm_configuration import GeneratorCollection, tinytc
from yateto.generator import Kernel
from yateto.memory import MemoryLayout
from yateto.type import (RUNTIME, AddressingMode, ElementAddressing,
                         elementAddressing, indexed, indexedPointer, indirect,
                         strided)

N = 4


@pytest.fixture(autouse=True)
def _alignment():
    yield
    # layouts built for a device align against its architecture
    MemoryLayout.DEFAULT_ALIGNMENT_ARCH = None


# ---------------------------------------------------------------------------
# The modes and their parameters
# ---------------------------------------------------------------------------


class TestModes:
    @pytest.mark.parametrize('mode, stars, spelled', [
        (AddressingMode.INDEXED, '*', 'indexed'),
        (AddressingMode.INDEXED_POINTER, '**', 'indexed_pointer'),
    ])
    def test_the_indexed_modes_are_memory(self, mode, stars, spelled):
        assert mode.hasStorage() and mode.isPassedAsArgument()
        assert not mode.isPassedByValue()
        assert mode.isIndexed()
        assert mode.pointer_type() == stars
        assert str(mode) == spelled

    def test_the_others_are_not_indexed(self):
        for mode in (AddressingMode.DIRECT, AddressingMode.STRIDED,
                     AddressingMode.INDIRECT, AddressingMode.SCALAR,
                     AddressingMode.IMMEDIATE):
            assert not mode.isIndexed()

    def test_a_bare_mode_is_its_own_mode(self):
        assert AddressingMode.STRIDED.mode is AddressingMode.STRIDED


class TestElementAddressing:
    def test_the_defaults_are_the_bare_mode(self):
        assert strided() is AddressingMode.STRIDED
        assert strided(stride=None, offset=0) is AddressingMode.STRIDED
        assert indexed() is AddressingMode.INDEXED
        assert indexedPointer() is AddressingMode.INDEXED_POINTER
        assert indirect() is AddressingMode.INDIRECT

    def test_parameters_make_a_parameterized_mode(self):
        addressing = indexed(stride=RUNTIME, offset=12)
        assert isinstance(addressing, ElementAddressing)
        assert addressing.mode is AddressingMode.INDEXED
        assert (addressing.stride, addressing.offset, addressing.slots, addressing.slot) == \
            (RUNTIME, 12, 1, 0)
        assert addressing.hasStorage() and addressing.isIndexed()
        assert addressing.pointer_type() == '*'
        assert indexedPointer(slots=4, slot=2).pointer_type() == '**'

    def test_it_is_never_equal_to_a_bare_mode(self):
        """A site comparing against `STRIDED` assumes the default stride, and a
        stride of its own is exactly what it must not match."""
        addressing = strided(stride=96)
        assert addressing != AddressingMode.STRIDED
        assert AddressingMode.STRIDED != addressing
        assert addressing not in (AddressingMode.STRIDED, AddressingMode.INDIRECT)
        assert addressing not in {AddressingMode.STRIDED: 'strided'}

    def test_equality_and_hash_are_by_value(self):
        assert indexed(stride=RUNTIME) == indexed(stride=RUNTIME)
        assert hash(indexed(stride=RUNTIME)) == hash(indexed(stride=RUNTIME))
        assert indexed(stride=8) != indexed(stride=9)
        assert indexed(stride=8) != strided(stride=8)

    def test_it_survives_copies_and_pickling(self):
        addressing = indexedPointer(slots=4, slot=RUNTIME, offset=3)
        for copied in (copy.copy(addressing), copy.deepcopy(addressing),
                       pickle.loads(pickle.dumps(addressing))):
            assert copied == addressing
            assert copied.slot is RUNTIME
        assert copy.deepcopy(RUNTIME) is RUNTIME
        assert pickle.loads(pickle.dumps(RUNTIME)) is RUNTIME

    def test_it_spells_what_differs_from_the_default(self):
        assert str(indexed(stride=RUNTIME)) == 'indexed(stride=runtime)'
        assert str(strided(stride=40, offset=8)) == 'strided(stride=40, offset=8)'
        assert str(indexedPointer(slots=4, slot=1)) == 'indexed_pointer(slots=4, slot=1)'
        assert str(RUNTIME) == 'runtime'

    def test_numbers_are_plain_integers(self):
        addressing = strided(stride=np.int64(12), offset=np.int32(4))
        assert type(addressing.stride) is int and type(addressing.offset) is int

    @pytest.mark.parametrize('make, match', [
        (lambda: strided(stride=0), 'at least 1'),
        (lambda: strided(stride=True), 'integer'),
        (lambda: strided(stride=1.5), 'integer'),
        (lambda: indexed(offset=-1), 'at least 0'),
        (lambda: indexed(offset=RUNTIME), 'cannot be bound at run time'),
        (lambda: indexedPointer(slots=0), 'at least 1'),
        (lambda: indexedPointer(slots=2, slot=2), 'slot 2 of 2'),
        (lambda: ElementAddressing(AddressingMode.INDIRECT, stride=8), 'stride means nothing'),
        (lambda: ElementAddressing(AddressingMode.STRIDED, slots=2), 'slots mean nothing'),
        (lambda: ElementAddressing(AddressingMode.INDEXED, slot=RUNTIME), 'slots mean nothing'),
        (lambda: ElementAddressing(AddressingMode.DIRECT), 'takes no parameters'),
        (lambda: ElementAddressing(AddressingMode.SCALAR), 'takes no parameters'),
    ])
    def test_what_does_not_make_sense_is_refused(self, make, match):
        with pytest.raises(ValueError, match=match):
            make()

    def test_the_parameters_of_either_form(self):
        assert elementAddressing(None) is None
        assert elementAddressing(AddressingMode.DIRECT) is None
        assert elementAddressing(AddressingMode.SCALAR) is None
        assert elementAddressing(AddressingMode.STRIDED) == \
            ElementAddressing(AddressingMode.STRIDED)
        addressing = indexed(stride=7)
        assert elementAddressing(addressing) is addressing


class TestTensorAddressing:
    def test_a_tensor_stores_the_defaults_as_the_bare_mode(self):
        t = Tensor('Q', (N, N), addressing=ElementAddressing(AddressingMode.INDEXED))
        assert t.addressing is AddressingMode.INDEXED
        t.addressing = ElementAddressing(AddressingMode.STRIDED, stride=None)
        assert t.addressing is AddressingMode.STRIDED

    def test_the_parameters_are_part_of_the_identity(self):
        assert Tensor('Q', (N, N), addressing=strided(stride=40)) != \
            Tensor('Q', (N, N), addressing=AddressingMode.STRIDED)
        assert Tensor('Q', (N, N), addressing=strided(stride=40)) == \
            Tensor('Q', (N, N), addressing=strided(stride=40))

    def test_anything_else_is_refused(self):
        with pytest.raises(ValueError, match='Q: the addressing'):
            Tensor('Q', (N,), addressing='indexed')

    def test_an_indexed_tensor_has_storage(self):
        assert Tensor('Q', (N,), addressing=indexed(stride=RUNTIME)).hasStorage()


# ---------------------------------------------------------------------------
# What an exporter is told
# ---------------------------------------------------------------------------


class Collector(ExportGenerator):
    def __init__(self, arch, attrs=None):
        super().__init__(arch, attrs)
        self.kernel = None

    def add_kernel(self, description):
        self.kernel = description

    def generate(self, cpp, cache):
        pass


def device():
    return useArchitectureIdentifiedBy('dhsw', 'dsm_86', 'cuda')


def exportAll(build, tmp_path, target='gpu'):
    """Generate `build(generator)` through a Collector; the descriptions, by
    kernel, and the files written."""
    collected = []

    def make(arch, attrs=None):
        collected.append(Collector(arch, attrs))
        return collected[-1]

    generator = Generator(device())
    build(generator)
    generator.generate(str(tmp_path), gemm_cfg=GeneratorCollection([]),
                       routine_exporters={target: make})
    files = {p.name: p.read_text() for p in tmp_path.iterdir() if p.is_file()}
    return [c.kernel for c in collected], files


def tensorsOf(description):
    return {t['name']: t for t in description['tensors']}


def mixed(g):
    """One kernel over every mode, written through a strided destination."""
    Q = Tensor('Q', (N, N), addressing=indexed())
    star = [Tensor(f'star({i})', (N, N), addressing=indexed(stride=RUNTIME, offset=16 * i))
            for i in range(3)]
    nb = Tensor('nb', (N, N), addressing=indexedPointer(slots=4, slot=RUNTIME, offset=2))
    out = Tensor('out', (N, N), addressing=strided(stride=40))
    A = Tensor('A', (N, N))
    g.add('k', out['ij'] <= Q['ik'] * star[1]['kj'] + nb['ij'] + A['ij'], target='gpu')


class TestExport:
    def test_the_version_is_eight(self):
        assert ExportGenerator.INTERFACE_VERSION == 8

    def test_a_kernel_states_the_version(self, tmp_path):
        [description], _ = exportAll(mixed, tmp_path)
        assert description['version'] == 8

    def test_every_mode_has_its_formula_and_binding(self, tmp_path):
        [description], _ = exportAll(mixed, tmp_path)
        t = tensorsOf(description)
        assert t['Q']['addressing'] == 'i*N+o&'
        assert t['Q']['binding'] == {'stride': None, 'offset': 0, 'slots': 1, 'slot': 0}
        assert t['star(1)']['addressing'] == 'i*N+o&'
        assert t['star(1)']['binding'] == {'stride': 'runtime', 'offset': 16, 'slots': 1, 'slot': 0}
        assert t['nb']['addressing'] == 'i*K+k&+o&'
        assert t['nb']['binding'] == {'stride': None, 'offset': 2, 'slots': 4, 'slot': 'runtime'}
        assert t['out']['addressing'] == 'n*N+o&'
        assert t['out']['binding'] == {'stride': 40, 'offset': 0, 'slots': 1, 'slot': 0}
        # a tensor that states nothing is deduced, and its binding is the default
        assert t['A']['addressing'] == 'n&+o&'
        assert t['A']['binding'] == {'stride': None, 'offset': 0, 'slots': 1, 'slot': 0}

    def test_a_tensor_whose_formula_names_no_element_has_no_binding(self, tmp_path):
        def build(g):
            K = Tensor('K', (N, N), np.ones((N, N)))
            A = Tensor('A', (N, N))
            C = Tensor('C', (N, N))
            g.add('k', C['ij'] <= 2.0 * A['ik'] * K['kj'], target='gpu')

        [description], _ = exportAll(build, tmp_path)
        for t in description['tensors']:
            if t['addressing'] in ('&', '', None):
                assert t['binding'] is None, t['name']
            else:
                assert t['binding'] is not None, t['name']
        assert any(t['addressing'] == '&' for t in description['tensors'])
        assert any(t['addressing'] == '' for t in description['tensors'])

    def test_a_temporary_has_the_default_binding(self, tmp_path):
        def build(g):
            A, B, C, D = (Tensor(n, (N, N)) for n in 'ABCD')
            g.add('k', D['ij'] <= A['ik'] * B['kl'] * C['lj'], target='gpu')

        [description], _ = exportAll(build, tmp_path)
        temporaries = [t for t in description['tensors'] if t['flags']['temporary']]
        assert temporaries
        for t in temporaries:
            assert t['addressing'] == 'n*N+o&'
            assert t['binding'] == {'stride': None, 'offset': 0, 'slots': 1, 'slot': 0}

    def test_the_description_is_still_data(self, tmp_path):
        [description], _ = exportAll(mixed, tmp_path)
        assert json.loads(json.dumps(description)) == description

    def test_a_written_tensor_that_states_its_mode_is_described_once(self, tmp_path):
        """A result was described from the operation computing it, which knows
        no tensor, and left to be deduced: two descriptions of one name, which
        the export refused."""
        def build(g):
            Q = Tensor('Q', (N, N), addressing=indexed())
            A = Tensor('A', (N, N))
            g.add('k', Q['ij'] <= Q['ik'] * A['kj'] + A['ij'], target='gpu')

        [description], _ = exportAll(build, tmp_path)
        assert tensorsOf(description)['Q']['addressing'] == 'i*N+o&'

    def test_an_exporter_speaking_version_seven_is_refused(self):
        class Seven:
            INTERFACE_VERSION = 7

            def __init__(self, arch, attrs=None):
                pass

        with pytest.raises(RuntimeError, match='interface version 7'):
            ExportFactory.makeFactory(Seven)(Cpp(StringIO()), device(), 'gpu',
                                             KernelAttributes())


# ---------------------------------------------------------------------------
# The kernel struct
# ---------------------------------------------------------------------------


def struct(header, name):
    return header.split(f'struct {name} {{')[1].split('\n    };')[0]


def execute(source, name='k', index=''):
    return source.split(f'void kernel::{name}::execute{index}() {{')[1].split('\n    }')[0]


class TestKernelStruct:
    def test_the_members_a_device_kernel_reaches_its_operands_by(self, tmp_path):
        _, files = exportAll(mixed, tmp_path)
        k = struct(files['kernel.h'], 'k')
        assert 'unsigned const* elementIndices = nullptr;' in k
        assert 'tensor::star::Container<std::size_t> stride_star;' in k
        assert 'unsigned slot_nb{};' in k
        # pointers by mode: the element's data, or a table of pointers to it
        assert 'double const* Q{nullptr};' in k
        assert 'tensor::star::Container<double const*> star;' in k
        assert 'double const** nb{nullptr};' in k
        assert 'double* out{nullptr};' in k
        assert 'double const** A{nullptr};' in k
        # the offsets are there as they always were
        assert 'int extraOffset_Q{};' in k and 'int extraOffset_nb{};' in k

    def test_execute_asserts_they_are_set(self, tmp_path):
        _, files = exportAll(mixed, tmp_path)
        body = execute(files['kernel.cpp'])
        assert 'assert(elementIndices != nullptr);' in body
        assert 'assert(stride_star(1) != 0);' in body
        assert 'assert(slot_nb < 4);' in body

    def test_none_of_it_where_nothing_asks(self, tmp_path):
        def build(g):
            A, B, C = (Tensor(n, (N, N)) for n in 'ABC')
            g.add('k', C['ij'] <= A['ik'] * B['kj'], target='gpu')

        _, files = exportAll(build, tmp_path)
        k = struct(files['kernel.h'], 'k')
        for member in ('elementIndices', 'stride_', 'slot_'):
            assert member not in k
            assert member not in files['kernel.cpp']
        assert 'double const** A{nullptr};' in k and 'double** C{nullptr};' in k

    def test_a_stride_known_when_generating_needs_no_member(self, tmp_path):
        def build(g):
            Q = Tensor('Q', (N, N), addressing=indexed(stride=40, offset=8))
            C = Tensor('C', (N, N), addressing=strided(stride=24))
            g.add('k', C['ij'] <= Q['ij'], target='gpu')

        _, files = exportAll(build, tmp_path)
        k = struct(files['kernel.h'], 'k')
        assert 'elementIndices' in k
        assert 'stride_' not in k and 'slot_' not in k

    def test_a_host_kernel_is_handed_one_element_whatever_the_mode(self, tmp_path):
        """On the host the caller hands over the element's own data, as it does
        for a strided or an indirect operand: no list, no stride, no slot."""
        def build(g):
            Q = Tensor('Q', (N, N), addressing=indexed(stride=RUNTIME))
            nb = Tensor('nb', (N, N), addressing=indexedPointer(slots=4, slot=RUNTIME))
            C = Tensor('C', (N, N))
            g.add('k', C['ij'] <= Q['ik'] * nb['kj'])

        g = Generator(useArchitectureIdentifiedBy('dhsw'))
        build(g)
        g.generate(str(tmp_path), gemm_cfg=GeneratorCollection([]))
        k = struct((tmp_path / 'kernel.h').read_text(), 'k')
        for member in ('elementIndices', 'stride_', 'slot_', 'numElements'):
            assert member not in k
        assert 'double const* Q{nullptr};' in k and 'double const* nb{nullptr};' in k

    def test_a_family_shares_the_list_and_asserts_where_it_is_read(self, tmp_path):
        def build(g):
            Q = [Tensor(f'Q({i})', (N, N), addressing=indexed(stride=RUNTIME)) for i in range(2)]
            P = Tensor('P', (N, N))
            C = Tensor('C', (N, N))
            g.addFamily('fam', simpleParameterSpace(2),
                        lambda i: C['ij'] <= (Q[i] if i == 0 else P)['ij'], target='gpu')

        _, files = exportAll(build, tmp_path)
        fam = struct(files['kernel.h'], 'fam')
        assert fam.count('unsigned const* elementIndices = nullptr;') == 1
        assert 'tensor::Q::Container<std::size_t> stride_Q;' in fam
        assert 'assert(elementIndices != nullptr);' in execute(files['kernel.cpp'], 'fam', '0')
        assert 'assert(stride_Q(0) != 0);' in execute(files['kernel.cpp'], 'fam', '0')
        assert 'elementIndices' not in execute(files['kernel.cpp'], 'fam', '1')

    def test_the_members_of_a_family_agree_on_the_mode(self, tmp_path):
        def build(g):
            Q = [Tensor('Q(0)', (N, N), addressing=indexed()),
                 Tensor('Q(1)', (N, N))]
            C = Tensor('C', (N, N))
            g.add('k', C['ij'] <= Q[0]['ij'] + Q[1]['ij'], target='gpu')

        with pytest.raises(ValueError, match='members of Q are addressed in different modes'):
            exportAll(build, tmp_path)

    def test_a_tensor_named_like_the_list_is_refused(self, tmp_path):
        def build(g):
            Q = Tensor('Q', (N, N), addressing=indexed())
            clash = Tensor('elementIndices', (N, N))
            g.add('k', clash['ij'] <= Q['ij'], target='gpu')

        with pytest.raises(ValueError, match='member elementIndices'):
            exportAll(build, tmp_path)


class TestUnitTests:
    """The device test of a kernel hands it what its members ask for."""

    def test_the_test_binds_the_list_the_stride_the_slot_and_the_offset(self, tmp_path):
        _, files = exportAll(mixed, tmp_path)
        test = files['test-kernel.cpp']
        assert 'krnl.elementIndices = _dev_elementIndices;' in test
        assert 'krnl.stride_star(1) = 16;' in test
        assert 'krnl.slot_nb = 0;' in test
        # the element's data, not a table of pointers to it
        assert 'krnl.Q = const_cast<double const*>(_dev_Q);' in test
        assert 'krnl.out = _dev_out;' in test
        assert 'krnl.nb = const_cast<double const**>(_dev_ptr_nb);' in test
        # room for the offset, and the data behind it
        assert 'cudaMalloc(&_dev_star_1, sizeof(star_1) + 16 * sizeof(double));' in test
        assert 'cudaMemcpyAsync(_dev_star_1 + 16, star_1, sizeof(star_1)' in test
        # a pointer for every slot
        assert 'double* _ptrs_nb[4] = {_dev_nb, _dev_nb, _dev_nb, _dev_nb};' in test
        assert 'cudaFree(_dev_elementIndices);' in test

    @pytest.mark.skipif(shutil.which('nvcc') is None, reason='needs nvcc')
    def test_it_compiles_as_cuda(self, tmp_path):
        gen = tmp_path / 'gen'
        gen.mkdir()
        exportAll(mixed, gen)
        include = pathlib.Path(__file__).resolve().parents[2] / 'include'
        doctest = [p for p in (pathlib.Path('/usr/include/doctest'), pathlib.Path('/usr/include'))
                   if (p / 'doctest.h').is_file()]
        sources = ['kernel.cpp'] + (['test-kernel.cpp'] if doctest else [])
        for name in sources:
            subprocess.run(['nvcc', '-std=c++17', '-x', 'cu', '-c', str(gen / name),
                            '-o', str(tmp_path / f'{name}.o'), f'-I{include}', f'-I{gen}',
                            *[f'-I{p}' for p in doctest[:1]]],
                           check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# The generators that cannot read the modes refuse them
# ---------------------------------------------------------------------------


class Term:
    def __init__(self, addressing=None, name='X', is_compute_constant=False, is_temporary=False):
        self.addressing = addressing
        self.name = name
        self.is_compute_constant = is_compute_constant
        self.is_temporary = is_temporary


UNREADABLE = [indexed(), indexedPointer(), indexed(stride=RUNTIME),
              strided(stride=40), strided(offset=8), indirect(offset=8),
              indexedPointer(slots=4, slot=1)]


class TestRefusals:
    @pytest.mark.parametrize('addressing', UNREADABLE, ids=str)
    def test_gemmforge_and_chainforge(self, addressing):
        with pytest.raises(ValueError, match=r'cannot read an operand addressed as .* \(Q\)'):
            BatchedOperationsAux.forge_addressing(Term(addressing, name='Q'))

    @pytest.mark.parametrize('addressing', UNREADABLE, ids=str)
    def test_tinytc(self, addressing):
        with pytest.raises(NotImplementedError, match='tinytc cannot read Q'):
            BatchedOperationsAux.classicAddressing(Term(addressing, name='Q'), 'tinytc')
        with pytest.raises(NotImplementedError, match='tinytc cannot read Q'):
            refuseUnreadableAddressing('Q', addressing, 'tinytc')

    @pytest.mark.parametrize('addressing', [None, AddressingMode.DIRECT, AddressingMode.STRIDED,
                                            AddressingMode.INDIRECT])
    def test_the_modes_they_read_pass(self, addressing):
        refuseUnreadableAddressing('Q', addressing, 'tinytc')
        assert BatchedOperationsAux.classicAddressing(Term(addressing), 'tinytc') == \
            BatchedOperationsAux.deduce_addresing(Term(addressing))

    @pytest.mark.parametrize('fused', [False, True])
    @pytest.mark.parametrize('addressing', [indexed(), strided(stride=80),
                                            indexedPointer(slots=4, slot=1)], ids=str)
    def test_a_tinytc_kernel_names_the_operand(self, fused, addressing):
        """Through the whole path: the GEMM and the fused-GEMM generators of a
        device kernel yateto writes itself."""
        arch = useArchitectureIdentifiedBy('dhsw', 'dpvc', 'oneapi')
        A = Tensor('A', (8, 8), addressing=addressing)
        B, C = Tensor('B', (8, 8)), Tensor('C', (8, 8))
        kernel = Kernel('k', C['ij'] <= A['ik'] * B['kj'], target='gpu')
        kernel.prepareUntilUnitTest(arch)
        kernel.prepareUntilCodeGen(BoundingBoxCostEstimator, fused, True)
        with pytest.raises(NotImplementedError, match='tinytc cannot read A'):
            OptimizedKernelGenerator(arch, RoutineCache(), DataCache(), {}).generateKernelOutline(
                kernel.nonZeroFlops, kernel.cfg, GeneratorCollection([tinytc(arch)]), 'gpu')

    @pytest.mark.parametrize('addressing', [indexed(), strided(stride=80)], ids=str)
    def test_a_gemmforge_kernel_names_the_operand(self, addressing, monkeypatch):
        """gemmforge is not installed here, so a stand-in is: the refusal comes
        before anything is asked of it."""
        from yateto.codegen.gemm import gemmgen
        from yateto.gemm_configuration import GemmForge
        fake = types.ModuleType('gemmforge')
        fake.YatetoInterface = types.SimpleNamespace(
            produce_dense_matrix=lambda *args, **kwargs: None)
        monkeypatch.setitem(sys.modules, 'gemmforge', fake)
        monkeypatch.setattr(gemmgen, 'gf_spec', True)
        arch = device()
        A = Tensor('A', (8, 8), addressing=addressing)
        B, C = Tensor('B', (8, 8)), Tensor('C', (8, 8))
        kernel = Kernel('k', C['ij'] <= A['ik'] * B['kj'], target='gpu')
        kernel.prepareUntilUnitTest(arch)
        kernel.prepareUntilCodeGen(BoundingBoxCostEstimator, False, True)
        with pytest.raises(ValueError, match=r'cannot read an operand addressed as .* \(A\)'):
            OptimizedKernelGenerator(arch, RoutineCache(), DataCache(), {}).generateKernelOutline(
                kernel.nonZeroFlops, kernel.cfg, GeneratorCollection([GemmForge(arch)]), 'gpu')
